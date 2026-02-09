#![forbid(unsafe_code)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub(crate) mod dither;
pub mod error;
pub(crate) mod histogram;
pub(crate) mod masking;
pub(crate) mod median_cut;
pub(crate) mod oklab;
pub(crate) mod palette;
pub(crate) mod remap;

pub use error::QuantizeError;
pub use imgref::{Img, ImgRef, ImgVec};
pub use rgb::{RGB, RGBA};

// Re-export internal helpers used by tests and benchmarking examples.
// Not part of the public API — may change without notice.
#[doc(hidden)]
pub mod _internals {
    pub use crate::dither::DitherMode;
    pub use crate::masking::compute_masking_weights;
    pub use crate::oklab::srgb_to_oklab;
    pub use crate::palette::index_delta_score;
    pub use crate::remap::{RunPriority, average_run_length};
}

// Internal modules re-exposed for profiling/debugging examples.
// Gated behind the `_dev` feature — not part of the public API.
#[cfg(feature = "_dev")]
#[doc(hidden)]
pub mod _dev {
    pub use crate::dither;
    pub use crate::histogram;
    pub use crate::masking;
    pub use crate::median_cut;
    pub use crate::oklab;
    pub use crate::palette;
    pub use crate::remap;
}

use alloc::vec::Vec;

/// Quality preset — controls k-means iterations, AQ masking, and Viterbi optimization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Quality {
    /// Fast mode — no masking or k-means refinement. Roughly 30ms per 512x512 image.
    Fast,
    /// Balanced — AQ masking + 2 k-means iterations + greedy run extension.
    Balanced,
    /// Best quality — AQ masking + 8 k-means iterations + Viterbi DP.
    Best,
}

impl Default for Quality {
    fn default() -> Self {
        Self::Best
    }
}

/// Target output format — controls palette sorting, dither strength, and compression tuning.
///
/// Different image formats have fundamentally different compression algorithms,
/// so the optimal palette ordering and dithering strategy varies per format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum OutputFormat {
    /// GIF: LZW compression, binary transparency only.
    /// Uses delta-minimize sort + post-remap frequency reorder.
    Gif,
    /// PNG: Deflate + scanline filters, per-index alpha via tRNS.
    /// Uses luminance sort for spatial locality.
    Png,
    /// WebP VP8L: Delta palette encoding + spatial prediction.
    /// Uses delta-minimize sort. Full RGBA palette.
    WebpLossless,
}

/// Internal tuning parameters derived from OutputFormat + user overrides.
#[derive(Debug, Clone)]
pub(crate) struct QuantizeTuning {
    pub(crate) dither_strength: f32,
    pub(crate) sort_strategy: palette::PaletteSortStrategy,
    pub(crate) gif_frequency_reorder: bool,
    pub(crate) alpha_mode: AlphaMode,
    pub(crate) viterbi_lambda_scale: f32,
}

/// How to handle alpha in quantization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AlphaMode {
    /// Binary: alpha==0 → transparent index, else opaque.
    Binary,
    /// Full alpha quantization: alpha is a quantizable dimension.
    Full,
}

impl QuantizeTuning {
    pub(crate) fn from_config(config: &QuantizeConfig) -> Self {
        // Per-format tuning:
        //   dither_strength: lower = fewer broken runs, better compression
        //   sort_strategy: Luminance for PNG (sub filter), DeltaMinimize for others
        //   viterbi_lambda_scale: multiplier on the run-extension lambda
        //     GIF/WebP benefit heavily from longer runs (LZW/entropy coding)
        let (default_dither, sort, gif_reorder, alpha, lambda_scale) = match config.output_format {
            OutputFormat::Gif => (
                0.35,
                palette::PaletteSortStrategy::DeltaMinimize,
                true,
                AlphaMode::Binary,
                3.0, // GIF's LZW rewards long runs heavily
            ),
            OutputFormat::Png => (
                0.3,
                palette::PaletteSortStrategy::Luminance,
                false,
                AlphaMode::Full,
                1.0,
            ),
            OutputFormat::WebpLossless => (
                0.4,
                palette::PaletteSortStrategy::DeltaMinimize,
                false,
                AlphaMode::Full,
                2.0, // WebP entropy coding also benefits from runs
            ),
        };

        Self {
            dither_strength: config.dither_strength.unwrap_or(default_dither),
            sort_strategy: sort,
            gif_frequency_reorder: gif_reorder,
            alpha_mode: alpha,
            viterbi_lambda_scale: lambda_scale,
        }
    }
}

/// Configuration for palette quantization.
///
/// Create with [`QuantizeConfig::new`], then optionally set quality and max colors.
///
/// # Example
///
/// ```
/// use zenquant::{QuantizeConfig, OutputFormat, Quality};
///
/// let config = QuantizeConfig::new(OutputFormat::Png)
///     .quality(Quality::Best)
///     .max_colors(128);
/// ```
#[derive(Debug, Clone)]
pub struct QuantizeConfig {
    max_colors: u32,
    quality: Quality,
    output_format: OutputFormat,
    // Internal knobs — not part of the public API.
    run_priority: remap::RunPriority,
    dither_mode: dither::DitherMode,
    dither_strength: Option<f32>,
    viterbi_lambda: Option<f32>,
}

impl QuantizeConfig {
    /// Create a new config for the given output format.
    ///
    /// Defaults: 256 colors, [`Quality::Best`], adaptive dithering.
    pub fn new(format: OutputFormat) -> Self {
        Self {
            max_colors: 256,
            quality: Quality::Best,
            output_format: format,
            run_priority: remap::RunPriority::Balanced,
            dither_mode: dither::DitherMode::Adaptive,
            dither_strength: None,
            viterbi_lambda: None,
        }
    }

    /// Maximum palette colors (2–256). Default: 256.
    ///
    /// GIF, PNG, and WebP lossless all support up to 256 palette entries.
    pub fn max_colors(mut self, n: u32) -> Self {
        self.max_colors = n;
        self
    }

    /// Quality preset. Default: [`Quality::Best`].
    pub fn quality(mut self, q: Quality) -> Self {
        self.quality = q;
        self
    }

    // --- Hidden expert methods (not public API) ---

    /// Override dither mode. Not part of the public API.
    #[doc(hidden)]
    pub fn _no_dither(mut self) -> Self {
        self.dither_mode = dither::DitherMode::None;
        self
    }

    /// Set run priority to Quality (no run bias). Not part of the public API.
    #[doc(hidden)]
    pub fn _run_priority_quality(mut self) -> Self {
        self.run_priority = remap::RunPriority::Quality;
        self
    }

    /// Set run priority to Compression (aggressive runs). Not part of the public API.
    #[doc(hidden)]
    pub fn _run_priority_compression(mut self) -> Self {
        self.run_priority = remap::RunPriority::Compression;
        self
    }

    /// Override dither strength (0.0–1.0). Not part of the public API.
    #[doc(hidden)]
    pub fn _dither_strength(mut self, strength: f32) -> Self {
        self.dither_strength = Some(strength);
        self
    }

    /// Override Viterbi lambda. Not part of the public API.
    #[doc(hidden)]
    pub fn _viterbi_lambda(mut self, lambda: f32) -> Self {
        self.viterbi_lambda = Some(lambda);
        self
    }
}

/// Result of palette quantization.
///
/// Contains an optimized palette and per-pixel indices into that palette.
/// Use [`palette()`](Self::palette) for RGB or [`palette_rgba()`](Self::palette_rgba) for RGBA,
/// and [`indices()`](Self::indices) for the index map.
#[derive(Debug)]
pub struct QuantizeResult {
    palette: palette::Palette,
    indices: Vec<u8>,
}

impl QuantizeResult {
    /// sRGB palette entries, sorted for the target output format.
    pub fn palette(&self) -> &[[u8; 3]] {
        self.palette.entries()
    }

    /// Palette index for each pixel, in row-major order: `pixel = y * width + x`.
    pub fn indices(&self) -> &[u8] {
        &self.indices
    }

    /// Get the transparent palette index, if any.
    pub fn transparent_index(&self) -> Option<u8> {
        self.palette.transparent_index()
    }

    /// Number of colors in the palette.
    pub fn palette_len(&self) -> usize {
        self.palette.len()
    }

    /// Get RGBA palette entries. Each entry has alpha: 255 for opaque,
    /// 0 for the transparent index, or the quantized alpha value.
    pub fn palette_rgba(&self) -> &[[u8; 4]] {
        self.palette.entries_rgba()
    }

    /// Get the alpha table suitable for a PNG tRNS chunk.
    ///
    /// Returns alpha values for each palette index, truncated at the last
    /// non-255 value. Returns `None` if all entries are fully opaque (no tRNS needed).
    pub fn alpha_table(&self) -> Option<Vec<u8>> {
        let rgba = self.palette.entries_rgba();
        let alphas: Vec<u8> = rgba.iter().map(|e| e[3]).collect();

        // Find last non-255 alpha
        let last_non_opaque = alphas.iter().rposition(|&a| a != 255);
        last_non_opaque.map(|pos| alphas[..=pos].to_vec())
    }

    /// Remap an RGB image against this result's palette.
    ///
    /// Skips palette construction — uses the existing palette and applies
    /// dithering + run optimization from `config`. The palette order is
    /// preserved (no frequency reorder), making this suitable for animation
    /// frames that share a palette.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use zenquant::{QuantizeConfig, OutputFormat};
    ///
    /// # let combined: Vec<rgb::RGB<u8>> = vec![];
    /// # let frame: Vec<rgb::RGB<u8>> = vec![];
    /// # let (w, h) = (64, 64);
    /// // Build a shared palette from a representative sample
    /// let config = QuantizeConfig::new(OutputFormat::Png);
    /// let shared = zenquant::quantize(&combined, w * 2, h, &config).unwrap();
    ///
    /// // Remap each frame against the shared palette
    /// let frame_result = shared.remap(&frame, w, h, &config).unwrap();
    /// ```
    pub fn remap(
        &self,
        pixels: &[rgb::RGB<u8>],
        width: usize,
        height: usize,
        config: &QuantizeConfig,
    ) -> Result<QuantizeResult, QuantizeError> {
        remap_rgb_impl(&self.palette, pixels, width, height, config)
    }

    /// Remap an RGBA image against this result's palette.
    ///
    /// Skips palette construction — uses the existing palette and applies
    /// dithering + run optimization from `config`. The palette order is
    /// preserved (no frequency reorder), making this suitable for GIF
    /// animation frames that share a global color table.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use zenquant::{QuantizeConfig, OutputFormat};
    ///
    /// # let combined: Vec<rgb::RGBA<u8>> = vec![];
    /// # let frames: Vec<Vec<rgb::RGBA<u8>>> = vec![];
    /// # let (w, h) = (64usize, 64usize);
    /// // Build a shared palette from sampled frames
    /// let config = QuantizeConfig::new(OutputFormat::Gif);
    /// let shared = zenquant::quantize_rgba(&combined, w * 4, h, &config).unwrap();
    ///
    /// // Remap each frame
    /// for frame in &frames {
    ///     let result = shared.remap_rgba(frame, w, h, &config).unwrap();
    ///     // result.palette() is identical across all frames
    /// }
    /// ```
    pub fn remap_rgba(
        &self,
        pixels: &[rgb::RGBA<u8>],
        width: usize,
        height: usize,
        config: &QuantizeConfig,
    ) -> Result<QuantizeResult, QuantizeError> {
        remap_rgba_impl(&self.palette, pixels, width, height, config)
    }
}

/// Quantize an RGB image to an indexed palette.
///
/// Returns a [`QuantizeResult`] with palette entries and per-pixel indices.
/// Indices are in row-major order: `index = y * width + x`.
///
/// # Example
///
/// ```no_run
/// use zenquant::{QuantizeConfig, OutputFormat};
///
/// # let pixels: Vec<rgb::RGB<u8>> = vec![];
/// # let (width, height) = (64, 64);
/// let config = QuantizeConfig::new(OutputFormat::Png);
/// let result = zenquant::quantize(&pixels, width, height, &config).unwrap();
///
/// let palette = result.palette();   // &[[u8; 3]] — sRGB palette
/// let indices = result.indices();   // &[u8] — row-major palette indices
/// ```
pub fn quantize(
    pixels: &[rgb::RGB<u8>],
    width: usize,
    height: usize,
    config: &QuantizeConfig,
) -> Result<QuantizeResult, QuantizeError> {
    validate_inputs(pixels.len(), width, height, config)?;

    let tuning = QuantizeTuning::from_config(config);
    let max_colors = config.max_colors as usize;

    // Fast path: image already has ≤max_colors unique colors
    if let Some(exact_colors) = histogram::detect_exact_palette(pixels, max_colors) {
        let centroids: Vec<oklab::OKLab> = exact_colors
            .iter()
            .map(|c| oklab::srgb_to_oklab(c.r, c.g, c.b))
            .collect();
        let pal = palette::Palette::from_centroids_sorted(centroids, false, tuning.sort_strategy);
        let mut indices = dither::simple_remap(pixels, &pal);
        let pal = if tuning.gif_frequency_reorder {
            palette::reorder_by_frequency(&pal, &mut indices)
        } else {
            pal
        };
        return Ok(QuantizeResult {
            palette: pal,
            indices,
        });
    }

    // Pipeline tiers based on quality:
    //   Fast:     no masking, no k-means, no Viterbi
    //   Balanced: masking + light k-means (2 iters) + run extension
    //   Best:     masking + full k-means (8 iters) + Viterbi DP
    let use_masking = matches!(config.quality, Quality::Balanced | Quality::Best);
    let kmeans_iters: usize = match config.quality {
        Quality::Best => 8,
        Quality::Balanced => 2,
        Quality::Fast => 0,
    };

    // 1. Compute AQ masking weights (skip for fast mode — uniform weights)
    let weights = if use_masking {
        masking::compute_masking_weights(pixels, width, height)
    } else {
        vec![1.0f32; pixels.len()]
    };

    // 2. Build weighted histogram
    let hist = histogram::build_histogram(pixels, &weights);

    // 3. Median cut with histogram-level k-means refinement
    let mut centroids = median_cut::median_cut(hist, max_colors, kmeans_iters > 0);

    // 3b. Pixel-level k-means refinement.
    if kmeans_iters > 0 {
        if pixels.len() <= 500_000 {
            centroids =
                median_cut::refine_against_pixels(centroids, pixels, &weights, kmeans_iters);
        } else {
            let (sub_pixels, sub_weights) = subsample_pixels(pixels, &weights, width);
            centroids = median_cut::refine_against_pixels(
                centroids,
                &sub_pixels,
                &sub_weights,
                kmeans_iters,
            );
        }
    }

    // 4. Build palette with format-specific sort
    let mut pal = palette::Palette::from_centroids_sorted(centroids, false, tuning.sort_strategy);
    // Build sRGB nearest-neighbor cache — always pays for itself
    // (32K cells vs 262K+ pixels, each saving 256 distance checks)
    pal.build_nn_cache();

    // 5. Dither / remap
    // Keep greedy run-bias during dithering even when Viterbi will run.
    // The greedy bias shapes error diffusion favorably, and Viterbi can
    // further optimize the index sequence post-hoc.
    let mut indices = dither::dither_image(
        pixels,
        width,
        height,
        &weights,
        &pal,
        config.dither_mode,
        config.run_priority,
        tuning.dither_strength,
    );

    // 5b. Run optimization
    //   Best:     full Viterbi DP (optimal run extension, ~26ms)
    //   Balanced: fast run-extend post-pass (greedy bidirectional, ~1ms)
    //   Fast:     none (dither-level greedy run-bias only)
    let use_viterbi = matches!(config.quality, Quality::Best);
    let run_lambda = if use_masking {
        config.viterbi_lambda.unwrap_or(match config.run_priority {
            remap::RunPriority::Quality => 0.0,
            remap::RunPriority::Balanced => 0.01,
            remap::RunPriority::Compression => 0.02,
        }) * tuning.viterbi_lambda_scale
    } else {
        config.viterbi_lambda.unwrap_or(0.0)
    };
    if run_lambda > 0.0 {
        if use_viterbi {
            remap::viterbi_refine(
                pixels,
                width,
                height,
                &weights,
                &pal,
                &mut indices,
                run_lambda,
            );
        } else {
            remap::run_extend_refine(
                pixels,
                width,
                height,
                &weights,
                &pal,
                &mut indices,
                run_lambda,
            );
        }
    }

    // 6. GIF frequency reorder (post-dither)
    let pal = if tuning.gif_frequency_reorder {
        palette::reorder_by_frequency(&pal, &mut indices)
    } else {
        pal
    };

    Ok(QuantizeResult {
        palette: pal,
        indices,
    })
}

/// Quantize an RGBA image to an indexed palette.
///
/// Transparent pixels (alpha == 0) get a dedicated transparent palette index,
/// accessible via [`QuantizeResult::transparent_index()`]. For formats with
/// per-index alpha (PNG tRNS, WebP), use [`QuantizeResult::palette_rgba()`]
/// and [`QuantizeResult::alpha_table()`].
///
/// # Example
///
/// ```no_run
/// use zenquant::{QuantizeConfig, OutputFormat};
///
/// # let pixels: Vec<rgb::RGBA<u8>> = vec![];
/// # let (width, height) = (64, 64);
/// let config = QuantizeConfig::new(OutputFormat::Gif);
/// let result = zenquant::quantize_rgba(&pixels, width, height, &config).unwrap();
///
/// let palette = result.palette();            // &[[u8; 3]]
/// let indices = result.indices();            // &[u8]
/// let transparent = result.transparent_index(); // Option<u8>
/// ```
pub fn quantize_rgba(
    pixels: &[rgb::RGBA<u8>],
    width: usize,
    height: usize,
    config: &QuantizeConfig,
) -> Result<QuantizeResult, QuantizeError> {
    validate_inputs(pixels.len(), width, height, config)?;

    let tuning = QuantizeTuning::from_config(config);
    let max_colors = config.max_colors as usize;

    // Fast path: image already has ≤max_colors unique colors
    if let Some((exact_colors, has_transparent)) =
        histogram::detect_exact_palette_rgba(pixels, max_colors)
    {
        let centroids: Vec<oklab::OKLab> = exact_colors
            .iter()
            .map(|c| oklab::srgb_to_oklab(c.r, c.g, c.b))
            .collect();
        let pal = palette::Palette::from_centroids_sorted(
            centroids,
            has_transparent,
            tuning.sort_strategy,
        );
        let transparent_idx = pal.transparent_index().unwrap_or(0);
        let mut indices = dither::simple_remap_rgba(pixels, &pal, transparent_idx);
        let pal = if tuning.gif_frequency_reorder {
            palette::reorder_by_frequency(&pal, &mut indices)
        } else {
            pal
        };
        return Ok(QuantizeResult {
            palette: pal,
            indices,
        });
    }

    let use_masking = matches!(config.quality, Quality::Balanced | Quality::Best);
    let use_viterbi = matches!(config.quality, Quality::Best);
    let kmeans_iters: usize = match config.quality {
        Quality::Best => 8,
        Quality::Balanced => 2,
        Quality::Fast => 0,
    };
    let weights = if use_masking {
        masking::compute_masking_weights_rgba(pixels, width, height)
    } else {
        vec![1.0f32; pixels.len()]
    };

    let (pal, mut indices) = if tuning.alpha_mode == AlphaMode::Full {
        // Full alpha quantization: 4D OKLabA pipeline
        let (hist, has_transparent) = histogram::build_histogram_alpha(pixels, &weights);
        let opaque_colors = if has_transparent {
            max_colors.saturating_sub(1)
        } else {
            max_colors
        };
        let mut centroids = median_cut::median_cut_alpha(hist, opaque_colors, kmeans_iters > 0);

        if kmeans_iters > 0 {
            if pixels.len() <= 500_000 {
                centroids = median_cut::refine_against_pixels_alpha(
                    centroids,
                    pixels,
                    &weights,
                    kmeans_iters,
                );
            } else {
                let (sub_pixels, sub_weights) = subsample_pixels_rgba(pixels, &weights, width);
                centroids = median_cut::refine_against_pixels_alpha(
                    centroids,
                    &sub_pixels,
                    &sub_weights,
                    kmeans_iters,
                );
            }
        }

        let pal = palette::Palette::from_centroids_alpha(
            centroids,
            has_transparent,
            tuning.sort_strategy,
        );

        let viterbi_lambda = if use_masking {
            config.viterbi_lambda.unwrap_or(match config.run_priority {
                remap::RunPriority::Quality => 0.0,
                remap::RunPriority::Balanced => 0.01,
                remap::RunPriority::Compression => 0.02,
            }) * tuning.viterbi_lambda_scale
        } else {
            config.viterbi_lambda.unwrap_or(0.0)
        };
        let mut indices = dither::dither_image_rgba_alpha(
            pixels,
            width,
            height,
            &weights,
            &pal,
            config.dither_mode,
            config.run_priority,
            tuning.dither_strength,
        );

        if viterbi_lambda > 0.0 {
            if use_viterbi {
                remap::viterbi_refine_rgba(
                    pixels,
                    width,
                    height,
                    &weights,
                    &pal,
                    &mut indices,
                    viterbi_lambda,
                );
            } else {
                remap::run_extend_refine_rgba(
                    pixels,
                    width,
                    height,
                    &weights,
                    &pal,
                    &mut indices,
                    viterbi_lambda,
                );
            }
        }

        (pal, indices)
    } else {
        // Binary transparency: opaque pipeline with transparent index
        let (hist, has_transparent) = histogram::build_histogram_rgba(pixels, &weights);
        let opaque_colors = if has_transparent {
            max_colors.saturating_sub(1)
        } else {
            max_colors
        };
        let mut centroids = median_cut::median_cut(hist, opaque_colors, kmeans_iters > 0);

        if kmeans_iters > 0 {
            if pixels.len() <= 500_000 {
                centroids = median_cut::refine_against_pixels_rgba(
                    centroids,
                    pixels,
                    &weights,
                    kmeans_iters,
                );
            } else {
                let (sub_pixels, sub_weights) = subsample_pixels_rgba(pixels, &weights, width);
                centroids = median_cut::refine_against_pixels_rgba(
                    centroids,
                    &sub_pixels,
                    &sub_weights,
                    kmeans_iters,
                );
            }
        }

        let mut pal = palette::Palette::from_centroids_sorted(
            centroids,
            has_transparent,
            tuning.sort_strategy,
        );
        pal.build_nn_cache();

        let viterbi_lambda = if use_masking {
            config.viterbi_lambda.unwrap_or(match config.run_priority {
                remap::RunPriority::Quality => 0.0,
                remap::RunPriority::Balanced => 0.01,
                remap::RunPriority::Compression => 0.02,
            }) * tuning.viterbi_lambda_scale
        } else {
            config.viterbi_lambda.unwrap_or(0.0)
        };
        let mut indices = dither::dither_image_rgba(
            pixels,
            width,
            height,
            &weights,
            &pal,
            config.dither_mode,
            config.run_priority,
            tuning.dither_strength,
        );

        if viterbi_lambda > 0.0 {
            if use_viterbi {
                remap::viterbi_refine_rgba(
                    pixels,
                    width,
                    height,
                    &weights,
                    &pal,
                    &mut indices,
                    viterbi_lambda,
                );
            } else {
                remap::run_extend_refine_rgba(
                    pixels,
                    width,
                    height,
                    &weights,
                    &pal,
                    &mut indices,
                    viterbi_lambda,
                );
            }
        }

        (pal, indices)
    };

    // GIF frequency reorder (post-dither)
    let pal = if tuning.gif_frequency_reorder {
        palette::reorder_by_frequency(&pal, &mut indices)
    } else {
        pal
    };

    Ok(QuantizeResult {
        palette: pal,
        indices,
    })
}

/// Build a shared palette from multiple RGB frames.
///
/// Each frame can have different dimensions. Returns a [`QuantizeResult`] whose
/// palette is optimized across all frames. Call [`QuantizeResult::remap()`] on
/// each frame to produce per-frame indices against the shared palette.
///
/// The returned result has empty indices — only the palette is meaningful.
///
/// # Example
///
/// ```no_run
/// use zenquant::{QuantizeConfig, OutputFormat, ImgRef};
///
/// # let buf1 = vec![rgb::RGB::new(0u8,0,0); 64];
/// # let buf2 = vec![rgb::RGB::new(0u8,0,0); 64];
/// # let frame1 = ImgRef::new(&buf1, 8, 8);
/// # let frame2 = ImgRef::new(&buf2, 8, 8);
/// let config = QuantizeConfig::new(OutputFormat::Png);
/// let shared = zenquant::build_palette(&[frame1, frame2], &config).unwrap();
///
/// // Remap each frame against the shared palette
/// for frame in &[frame1, frame2] {
///     let pixels: Vec<_> = frame.pixels().collect();
///     let result = shared.remap(&pixels, frame.width(), frame.height(), &config).unwrap();
/// }
/// ```
pub fn build_palette(
    frames: &[ImgRef<'_, rgb::RGB<u8>>],
    config: &QuantizeConfig,
) -> Result<QuantizeResult, QuantizeError> {
    if frames.is_empty() {
        return Err(QuantizeError::ZeroDimension);
    }
    for frame in frames {
        if frame.width() == 0 || frame.height() == 0 {
            return Err(QuantizeError::ZeroDimension);
        }
    }
    if config.max_colors < 2 || config.max_colors > 256 {
        return Err(QuantizeError::InvalidMaxColors(config.max_colors));
    }

    let tuning = QuantizeTuning::from_config(config);
    let max_colors = config.max_colors as usize;

    // Fast path: all frames combined have ≤max_colors unique colors
    let all_exact = detect_exact_palette_multi_rgb(frames, max_colors);
    if let Some(exact_colors) = all_exact {
        let centroids: Vec<oklab::OKLab> = exact_colors
            .iter()
            .map(|c| oklab::srgb_to_oklab(c.r, c.g, c.b))
            .collect();
        let mut pal =
            palette::Palette::from_centroids_sorted(centroids, false, tuning.sort_strategy);
        pal.build_nn_cache();
        return Ok(QuantizeResult {
            palette: pal,
            indices: Vec::new(),
        });
    }

    let use_masking = matches!(config.quality, Quality::Balanced | Quality::Best);
    let kmeans_iters: usize = match config.quality {
        Quality::Best => 8,
        Quality::Balanced => 2,
        Quality::Fast => 0,
    };

    // Build merged histogram from all frames
    let mut merged_hist: Vec<(oklab::OKLab, f32)> = Vec::new();
    let mut all_pixels: Vec<rgb::RGB<u8>> = Vec::new();
    let mut all_weights: Vec<f32> = Vec::new();

    for frame in frames {
        let pixels: Vec<rgb::RGB<u8>> = frame.pixels().collect();
        let w = frame.width();
        let h = frame.height();

        let weights = if use_masking {
            masking::compute_masking_weights(&pixels, w, h)
        } else {
            vec![1.0f32; pixels.len()]
        };

        let hist = histogram::build_histogram(&pixels, &weights);
        merged_hist.extend_from_slice(&hist);

        if kmeans_iters > 0 {
            all_pixels.extend_from_slice(&pixels);
            all_weights.extend_from_slice(&weights);
        }
    }

    // Median cut on merged histogram
    let mut centroids = median_cut::median_cut(merged_hist, max_colors, kmeans_iters > 0);

    // K-means refinement against subsampled pixels from all frames
    if kmeans_iters > 0 {
        let total = all_pixels.len();
        if total <= 500_000 {
            centroids = median_cut::refine_against_pixels(
                centroids,
                &all_pixels,
                &all_weights,
                kmeans_iters,
            );
        } else {
            // Subsample uniformly across all frames
            let step = (total / 250_000).max(1);
            let sub_pixels: Vec<rgb::RGB<u8>> = all_pixels.iter().step_by(step).copied().collect();
            let sub_weights: Vec<f32> = all_weights.iter().step_by(step).copied().collect();
            centroids = median_cut::refine_against_pixels(
                centroids,
                &sub_pixels,
                &sub_weights,
                kmeans_iters,
            );
        }
    }

    let mut pal = palette::Palette::from_centroids_sorted(centroids, false, tuning.sort_strategy);
    pal.build_nn_cache();

    Ok(QuantizeResult {
        palette: pal,
        indices: Vec::new(),
    })
}

/// Build a shared palette from multiple RGBA frames.
///
/// Each frame can have different dimensions. Returns a [`QuantizeResult`] whose
/// palette is optimized across all frames. Call [`QuantizeResult::remap_rgba()`]
/// on each frame to produce per-frame indices against the shared palette.
///
/// The returned result has empty indices — only the palette is meaningful.
///
/// # Example
///
/// ```no_run
/// use zenquant::{QuantizeConfig, OutputFormat, ImgRef};
///
/// # let buf1 = vec![rgb::RGBA::new(0u8,0,0,255); 64];
/// # let buf2 = vec![rgb::RGBA::new(0u8,0,0,255); 48];
/// # let frame1 = ImgRef::new(&buf1, 8, 8);
/// # let frame2 = ImgRef::new(&buf2, 8, 6);
/// let config = QuantizeConfig::new(OutputFormat::Gif);
/// let shared = zenquant::build_palette_rgba(&[frame1, frame2], &config).unwrap();
///
/// // Remap each frame against the shared palette
/// for frame in &[frame1, frame2] {
///     let pixels: Vec<_> = frame.pixels().collect();
///     let result = shared.remap_rgba(&pixels, frame.width(), frame.height(), &config).unwrap();
/// }
/// ```
pub fn build_palette_rgba(
    frames: &[ImgRef<'_, rgb::RGBA<u8>>],
    config: &QuantizeConfig,
) -> Result<QuantizeResult, QuantizeError> {
    if frames.is_empty() {
        return Err(QuantizeError::ZeroDimension);
    }
    for frame in frames {
        if frame.width() == 0 || frame.height() == 0 {
            return Err(QuantizeError::ZeroDimension);
        }
    }
    if config.max_colors < 2 || config.max_colors > 256 {
        return Err(QuantizeError::InvalidMaxColors(config.max_colors));
    }

    let tuning = QuantizeTuning::from_config(config);
    let max_colors = config.max_colors as usize;

    // Fast path: all frames combined have ≤max_colors unique colors
    let all_exact = detect_exact_palette_multi_rgba(frames, max_colors);
    if let Some((exact_colors, has_transparent)) = all_exact {
        if tuning.alpha_mode == AlphaMode::Full {
            let centroids: Vec<oklab::OKLabA> = exact_colors
                .iter()
                .map(|c| oklab::srgb_to_oklab(c.r, c.g, c.b).with_alpha(c.a as f32 / 255.0))
                .collect();
            let mut pal =
                palette::Palette::from_centroids_alpha(centroids, false, tuning.sort_strategy);
            pal.build_nn_cache();
            return Ok(QuantizeResult {
                palette: pal,
                indices: Vec::new(),
            });
        } else {
            let centroids: Vec<oklab::OKLab> = exact_colors
                .iter()
                .map(|c| oklab::srgb_to_oklab(c.r, c.g, c.b))
                .collect();
            let mut pal = palette::Palette::from_centroids_sorted(
                centroids,
                has_transparent,
                tuning.sort_strategy,
            );
            pal.build_nn_cache();
            return Ok(QuantizeResult {
                palette: pal,
                indices: Vec::new(),
            });
        }
    }

    let use_masking = matches!(config.quality, Quality::Balanced | Quality::Best);
    let kmeans_iters: usize = match config.quality {
        Quality::Best => 8,
        Quality::Balanced => 2,
        Quality::Fast => 0,
    };

    let mut all_pixels: Vec<rgb::RGBA<u8>> = Vec::new();
    let mut all_weights: Vec<f32> = Vec::new();

    // Collect pixels and weights from all frames (masking is per-frame/spatial)
    for frame in frames {
        let pixels: Vec<rgb::RGBA<u8>> = frame.pixels().collect();
        let w = frame.width();
        let h = frame.height();

        let weights = if use_masking {
            masking::compute_masking_weights_rgba(&pixels, w, h)
        } else {
            vec![1.0f32; pixels.len()]
        };

        all_pixels.extend_from_slice(&pixels);
        all_weights.extend_from_slice(&weights);
    }

    let pal = if tuning.alpha_mode == AlphaMode::Full {
        // Full alpha: 4D OKLabA histogram and median cut
        let (merged_hist, has_transparent) =
            histogram::build_histogram_alpha(&all_pixels, &all_weights);
        let _ = has_transparent; // transparency handled by alpha channel in palette entries

        let mut centroids =
            median_cut::median_cut_alpha(merged_hist, max_colors, kmeans_iters > 0);

        if kmeans_iters > 0 {
            let total = all_pixels.len();
            if total <= 500_000 {
                centroids = median_cut::refine_against_pixels_alpha(
                    centroids,
                    &all_pixels,
                    &all_weights,
                    kmeans_iters,
                );
            } else {
                let step = (total / 250_000).max(1);
                let sub_pixels: Vec<rgb::RGBA<u8>> =
                    all_pixels.iter().step_by(step).copied().collect();
                let sub_weights: Vec<f32> =
                    all_weights.iter().step_by(step).copied().collect();
                centroids = median_cut::refine_against_pixels_alpha(
                    centroids,
                    &sub_pixels,
                    &sub_weights,
                    kmeans_iters,
                );
            }
        }

        let mut p =
            palette::Palette::from_centroids_alpha(centroids, false, tuning.sort_strategy);
        p.build_nn_cache();
        p
    } else {
        // Binary alpha: 3D OKLab histogram, transparent pixels excluded
        let (merged_hist, has_transparent) =
            histogram::build_histogram_rgba(&all_pixels, &all_weights);

        let opaque_colors = if has_transparent {
            max_colors.saturating_sub(1)
        } else {
            max_colors
        };

        let mut centroids =
            median_cut::median_cut(merged_hist, opaque_colors, kmeans_iters > 0);

        if kmeans_iters > 0 {
            let total = all_pixels.len();
            if total <= 500_000 {
                centroids = median_cut::refine_against_pixels_rgba(
                    centroids,
                    &all_pixels,
                    &all_weights,
                    kmeans_iters,
                );
            } else {
                let step = (total / 250_000).max(1);
                let sub_pixels: Vec<rgb::RGBA<u8>> =
                    all_pixels.iter().step_by(step).copied().collect();
                let sub_weights: Vec<f32> =
                    all_weights.iter().step_by(step).copied().collect();
                centroids = median_cut::refine_against_pixels_rgba(
                    centroids,
                    &sub_pixels,
                    &sub_weights,
                    kmeans_iters,
                );
            }
        }

        let mut p = palette::Palette::from_centroids_sorted(
            centroids,
            has_transparent,
            tuning.sort_strategy,
        );
        p.build_nn_cache();
        p
    };

    Ok(QuantizeResult {
        palette: pal,
        indices: Vec::new(),
    })
}

/// Internal: remap RGB pixels against an existing palette.
fn remap_rgb_impl(
    source_palette: &palette::Palette,
    pixels: &[rgb::RGB<u8>],
    width: usize,
    height: usize,
    config: &QuantizeConfig,
) -> Result<QuantizeResult, QuantizeError> {
    if width == 0 || height == 0 {
        return Err(QuantizeError::ZeroDimension);
    }
    if pixels.len() != width * height {
        return Err(QuantizeError::DimensionMismatch {
            len: pixels.len(),
            width,
            height,
        });
    }

    let tuning = QuantizeTuning::from_config(config);

    let mut pal = source_palette.clone();
    if !pal.has_nn_cache() {
        pal.build_nn_cache();
    }

    let use_masking = matches!(config.quality, Quality::Balanced | Quality::Best);
    let use_viterbi = matches!(config.quality, Quality::Best);

    let weights = if use_masking {
        masking::compute_masking_weights(pixels, width, height)
    } else {
        vec![1.0f32; pixels.len()]
    };

    let mut indices = dither::dither_image(
        pixels,
        width,
        height,
        &weights,
        &pal,
        config.dither_mode,
        config.run_priority,
        tuning.dither_strength,
    );

    let run_lambda = if use_masking {
        config.viterbi_lambda.unwrap_or(match config.run_priority {
            remap::RunPriority::Quality => 0.0,
            remap::RunPriority::Balanced => 0.01,
            remap::RunPriority::Compression => 0.02,
        }) * tuning.viterbi_lambda_scale
    } else {
        config.viterbi_lambda.unwrap_or(0.0)
    };
    if run_lambda > 0.0 {
        if use_viterbi {
            remap::viterbi_refine(pixels, width, height, &weights, &pal, &mut indices, run_lambda);
        } else {
            remap::run_extend_refine(
                pixels, width, height, &weights, &pal, &mut indices, run_lambda,
            );
        }
    }

    // No frequency reorder — palette order must be stable for shared-palette use.
    Ok(QuantizeResult {
        palette: pal,
        indices,
    })
}

/// Internal: remap RGBA pixels against an existing palette.
fn remap_rgba_impl(
    source_palette: &palette::Palette,
    pixels: &[rgb::RGBA<u8>],
    width: usize,
    height: usize,
    config: &QuantizeConfig,
) -> Result<QuantizeResult, QuantizeError> {
    if width == 0 || height == 0 {
        return Err(QuantizeError::ZeroDimension);
    }
    if pixels.len() != width * height {
        return Err(QuantizeError::DimensionMismatch {
            len: pixels.len(),
            width,
            height,
        });
    }

    let tuning = QuantizeTuning::from_config(config);

    let mut pal = source_palette.clone();
    if !pal.has_nn_cache() {
        pal.build_nn_cache();
    }

    let use_masking = matches!(config.quality, Quality::Balanced | Quality::Best);
    let use_viterbi = matches!(config.quality, Quality::Best);

    let weights = if use_masking {
        masking::compute_masking_weights_rgba(pixels, width, height)
    } else {
        vec![1.0f32; pixels.len()]
    };

    // Detect alpha mode from the palette: if any entry has alpha between 1-254,
    // the palette was built with full alpha quantization.
    let has_full_alpha = pal.entries_rgba().iter().any(|e| e[3] > 0 && e[3] < 255);

    let mut indices = if has_full_alpha {
        dither::dither_image_rgba_alpha(
            pixels,
            width,
            height,
            &weights,
            &pal,
            config.dither_mode,
            config.run_priority,
            tuning.dither_strength,
        )
    } else {
        dither::dither_image_rgba(
            pixels,
            width,
            height,
            &weights,
            &pal,
            config.dither_mode,
            config.run_priority,
            tuning.dither_strength,
        )
    };

    let run_lambda = if use_masking {
        config.viterbi_lambda.unwrap_or(match config.run_priority {
            remap::RunPriority::Quality => 0.0,
            remap::RunPriority::Balanced => 0.01,
            remap::RunPriority::Compression => 0.02,
        }) * tuning.viterbi_lambda_scale
    } else {
        config.viterbi_lambda.unwrap_or(0.0)
    };
    if run_lambda > 0.0 {
        if use_viterbi {
            remap::viterbi_refine_rgba(
                pixels, width, height, &weights, &pal, &mut indices, run_lambda,
            );
        } else {
            remap::run_extend_refine_rgba(
                pixels, width, height, &weights, &pal, &mut indices, run_lambda,
            );
        }
    }

    // No frequency reorder — palette order must be stable for shared-palette use.
    Ok(QuantizeResult {
        palette: pal,
        indices,
    })
}

/// Subsample RGB pixels for k-means refinement on large images.
/// Takes every Nth pixel (with corresponding weight) to produce ~250K samples.
fn subsample_pixels(
    pixels: &[rgb::RGB<u8>],
    weights: &[f32],
    width: usize,
) -> (Vec<rgb::RGB<u8>>, Vec<f32>) {
    const TARGET: usize = 250_000;
    let step = (pixels.len() / TARGET).max(1);
    let height = pixels.len() / width;

    let mut sub_pixels = Vec::with_capacity(TARGET + width);
    let mut sub_weights = Vec::with_capacity(TARGET + width);

    // Sample evenly across rows to maintain spatial distribution
    for y in 0..height {
        let row_start = y * width;
        let mut x = (y * 3) % step; // offset per row to avoid column aliasing
        while x < width {
            let idx = row_start + x;
            sub_pixels.push(pixels[idx]);
            sub_weights.push(weights[idx]);
            x += step;
        }
    }

    (sub_pixels, sub_weights)
}

/// Subsample RGBA pixels for k-means refinement on large images.
fn subsample_pixels_rgba(
    pixels: &[rgb::RGBA<u8>],
    weights: &[f32],
    width: usize,
) -> (Vec<rgb::RGBA<u8>>, Vec<f32>) {
    const TARGET: usize = 250_000;
    let step = (pixels.len() / TARGET).max(1);
    let height = pixels.len() / width;

    let mut sub_pixels = Vec::with_capacity(TARGET + width);
    let mut sub_weights = Vec::with_capacity(TARGET + width);

    for y in 0..height {
        let row_start = y * width;
        let mut x = (y * 3) % step;
        while x < width {
            let idx = row_start + x;
            sub_pixels.push(pixels[idx]);
            sub_weights.push(weights[idx]);
            x += step;
        }
    }

    (sub_pixels, sub_weights)
}

/// Detect if all RGB frames combined have ≤max_colors unique colors.
fn detect_exact_palette_multi_rgb(
    frames: &[ImgRef<'_, rgb::RGB<u8>>],
    max_colors: usize,
) -> Option<Vec<rgb::RGB<u8>>> {
    let mut seen = alloc::collections::BTreeSet::new();
    for frame in frames {
        for p in frame.pixels() {
            let key = (p.r as u32) << 16 | (p.g as u32) << 8 | p.b as u32;
            seen.insert(key);
            if seen.len() > max_colors {
                return None;
            }
        }
    }
    Some(
        seen.into_iter()
            .map(|k| rgb::RGB {
                r: (k >> 16) as u8,
                g: (k >> 8) as u8,
                b: k as u8,
            })
            .collect(),
    )
}

/// Detect if all RGBA frames combined have ≤max_colors unique opaque colors.
/// Returns the palette and whether any fully-transparent pixels exist.
fn detect_exact_palette_multi_rgba(
    frames: &[ImgRef<'_, rgb::RGBA<u8>>],
    max_colors: usize,
) -> Option<(Vec<rgb::RGBA<u8>>, bool)> {
    let mut seen = alloc::collections::BTreeSet::new();
    let mut has_transparent = false;
    for frame in frames {
        for p in frame.pixels() {
            if p.a == 0 {
                has_transparent = true;
                continue;
            }
            let key = (p.r as u32) << 24 | (p.g as u32) << 16 | (p.b as u32) << 8 | p.a as u32;
            seen.insert(key);
            if seen.len() > max_colors {
                return None;
            }
        }
    }
    let colors = seen
        .into_iter()
        .map(|k| rgb::RGBA {
            r: (k >> 24) as u8,
            g: (k >> 16) as u8,
            b: (k >> 8) as u8,
            a: k as u8,
        })
        .collect();
    Some((colors, has_transparent))
}

fn validate_inputs(
    pixel_count: usize,
    width: usize,
    height: usize,
    config: &QuantizeConfig,
) -> Result<(), QuantizeError> {
    if width == 0 || height == 0 {
        return Err(QuantizeError::ZeroDimension);
    }
    if pixel_count != width * height {
        return Err(QuantizeError::DimensionMismatch {
            len: pixel_count,
            width,
            height,
        });
    }
    if config.max_colors < 2 || config.max_colors > 256 {
        return Err(QuantizeError::InvalidMaxColors(config.max_colors));
    }
    Ok(())
}
