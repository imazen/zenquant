//! AQ-informed color quantization for indexed image formats.
//!
//! zenquant reduces truecolor images to 256-color palettes using perceptual
//! masking (butteraugli-inspired AQ weights), OKLab color space, and optional
//! Viterbi DP for run-length–friendly index ordering.
//!
//! # Quick start
//!
//! ```
//! use zenquant::{QuantizeConfig, OutputFormat};
//!
//! # let pixels = vec![rgb::RGB::new(128u8, 64, 32); 64];
//! let config = QuantizeConfig::new(OutputFormat::Png);
//! let result = zenquant::quantize(&pixels, 8, 8, &config).unwrap();
//!
//! let palette = result.palette();   // &[[u8; 3]]
//! let indices = result.indices();   // &[u8]
//! ```
//!
//! # Features
//!
//! - **Perceptual masking**: concentrates palette entries where human vision
//!   is most sensitive (smooth gradients, skin tones) rather than wasting
//!   entries on noisy textures.
//! - **OKLab color space**: all clustering happens in a perceptually uniform
//!   space, so "nearest color" actually looks nearest.
//! - **Format-aware tuning**: palette sorting and dither strength are
//!   optimized per output format (GIF, PNG, WebP lossless).
//! - **Shared palettes**: [`build_palette`] and [`build_palette_rgba`] build
//!   a single palette from multiple frames, then [`QuantizeResult::remap`]
//!   maps each frame against it.
//! - **`no_std` + `alloc`**: works in WASM and embedded contexts.

#![deny(unsafe_code)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

// When _dev is enabled, expose internal modules as pub for profiling examples.
// Otherwise keep them pub(crate).
macro_rules! dev_modules {
    ($($mod:ident),* $(,)?) => {
        $(
            #[cfg(feature = "_dev")]
            pub mod $mod;
            #[cfg(not(feature = "_dev"))]
            pub(crate) mod $mod;
        )*
    };
}
pub(crate) mod blue_noise;
dev_modules!(
    dither, histogram, masking, median_cut, metric, oklab, palette, remap
);
pub mod error;
#[cfg(feature = "joint")]
pub(crate) mod joint;
#[cfg(feature = "joint")]
mod joint_predict;
#[cfg(feature = "simd")]
pub(crate) mod simd;
#[cfg(not(feature = "simd"))]
pub(crate) mod simd {
    //! Scalar fallback when `simd` feature is disabled.
    extern crate alloc;
    use crate::oklab::{OKLab, srgb_to_oklab};
    use alloc::vec::Vec;

    pub(crate) fn batch_srgb_to_oklab(pixels: &[rgb::RGB<u8>], out: &mut [[f32; 3]]) {
        for (px, o) in pixels.iter().zip(out.iter_mut()) {
            let lab = srgb_to_oklab(px.r, px.g, px.b);
            *o = [lab.l, lab.a, lab.b];
        }
    }

    pub(crate) fn batch_srgb_to_oklab_vec(pixels: &[rgb::RGB<u8>]) -> Vec<OKLab> {
        pixels
            .iter()
            .map(|p| srgb_to_oklab(p.r, p.g, p.b))
            .collect()
    }

    #[derive(Debug, Clone)]
    pub(crate) struct PaletteSimd {
        entries: Vec<OKLab>,
        start: usize,
    }

    impl PaletteSimd {
        pub(crate) fn empty() -> Self {
            Self {
                entries: Vec::new(),
                start: 0,
            }
        }

        pub(crate) fn from_palette(palette: &crate::palette::Palette) -> Self {
            let start = if palette.transparent_index().is_some() {
                1
            } else {
                0
            };
            Self {
                entries: palette.entries_oklab().to_vec(),
                start,
            }
        }

        pub(crate) fn from_oklab_slice(entries: &[OKLab], start: usize) -> Self {
            Self {
                entries: entries.to_vec(),
                start,
            }
        }

        pub(crate) fn nearest(&self, color: OKLab) -> u8 {
            let mut best_idx = self.start;
            let mut best_dist = f32::MAX;
            for i in self.start..self.entries.len() {
                let d = color.distance_sq(self.entries[i]);
                if d < best_dist {
                    best_dist = d;
                    best_idx = i;
                }
            }
            best_idx as u8
        }
    }
}

pub use error::QuantizeError;
pub use imgref::{Img, ImgRef, ImgVec};
pub use rgb::{RGB, RGBA};

// Re-export internal helpers used by tests and benchmarking examples.
// Not part of the public API — may change without notice.
#[doc(hidden)]
pub mod _internals {
    pub use crate::dither::DitherMode;
    pub use crate::masking::compute_masking_weights;
    pub use crate::metric::{MpeResult, compute_mpe, compute_mpe_rgba};
    pub use crate::oklab::srgb_to_oklab;
    pub use crate::palette::index_delta_score;
    pub use crate::remap::{RunPriority, average_run_length};
}

// When _dev is enabled, re-export internal modules via _dev for backwards compat
// with profiling examples that use `zenquant::_dev::module`.
#[cfg(feature = "_dev")]
#[doc(hidden)]
pub mod _dev {
    pub use crate::{dither, histogram, masking, median_cut, metric, oklab, palette, remap};
}

use alloc::vec::Vec;

/// Quality preset — controls k-means iterations, AQ masking, and Viterbi optimization.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Quality {
    /// Fast mode — no masking, histogram-only k-means. Roughly 25ms per 512x512 image.
    Fast,
    /// Balanced — AQ masking + 2 k-means iterations + greedy run extension.
    Balanced,
    /// Best quality — AQ masking + 8 k-means iterations + Viterbi DP.
    #[default]
    Best,
}

/// Target output format — controls palette sorting, dither strength, and compression tuning.
///
/// Different image formats have fundamentally different compression algorithms,
/// so the optimal palette ordering and dithering strategy varies per format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum OutputFormat {
    /// GIF: LZW compression, binary transparency only.
    /// Uses delta-minimize sort + post-remap frequency reorder.
    Gif,
    /// PNG: Deflate + scanline filters, per-index alpha via tRNS.
    /// Uses luminance sort for spatial locality.
    Png,
    /// PNG with joint deflate+quantization optimization.
    ///
    /// Same tuning as [`Png`](OutputFormat::Png) (luminance sort, full alpha),
    /// but runs a post-processing pass that jointly selects palette indices and
    /// PNG filter types per scanline to minimize deflate-compressed size while
    /// keeping every pixel within its perceptual tolerance budget.
    ///
    /// The optimized indices are returned in the normal [`QuantizeResult::indices()`];
    /// downstream encoders compress them through their standard pipeline.
    ///
    /// Requires the `joint` feature.
    PngJoint,
    /// PNG optimized for minimum file size. Uses position-deterministic
    /// blue noise dithering at very low strength, aggressive run extension,
    /// and joint deflate+quantization optimization.
    ///
    /// Requires the `joint` feature.
    PngMinSize,
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
                0.5,
                palette::PaletteSortStrategy::Luminance,
                false,
                AlphaMode::Full,
                1.0,
            ),
            OutputFormat::PngJoint => (
                0.3, // lower dither — joint exploits compressible patterns better
                palette::PaletteSortStrategy::Luminance,
                false,
                AlphaMode::Full,
                1.0,
            ),
            OutputFormat::PngMinSize => (
                0.1, // minimal dither — just enough to break banding
                palette::PaletteSortStrategy::Luminance,
                false,
                AlphaMode::Full,
                2.5, // aggressive run extension for deflate
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
///     .with_quality(Quality::Best)
///     .with_max_colors(128);
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
    compute_metric: bool,
    target_ssim2: Option<f32>,
    min_ssim2: Option<f32>,
    /// Deflate effort for the joint evaluation pass (1–22). Default: 10.
    /// (Retained for API compatibility; the vendored predictor ignores this.)
    joint_deflate_effort: u32,
    /// Base OKLab distance tolerance for joint candidate selection. Default: 0.015.
    joint_tolerance: f32,
    /// Maximum pixels sampled per k-means iteration. Larger images are
    /// stride-subsampled internally. Default: 131072.
    kmeans_sample_cap: usize,
}

impl QuantizeConfig {
    /// Create a new config for the given output format.
    ///
    /// Defaults: 256 colors, [`Quality::Best`], adaptive dithering.
    #[must_use]
    pub fn new(format: OutputFormat) -> Self {
        Self {
            max_colors: 256,
            quality: Quality::Best,
            output_format: format,
            run_priority: match format {
                OutputFormat::PngMinSize => remap::RunPriority::Compression,
                _ => remap::RunPriority::Balanced,
            },
            dither_mode: match format {
                OutputFormat::PngMinSize => dither::DitherMode::BlueNoise,
                _ => dither::DitherMode::Adaptive,
            },
            dither_strength: None,
            viterbi_lambda: None,
            compute_metric: false,
            target_ssim2: None,
            min_ssim2: None,
            joint_deflate_effort: 10,
            joint_tolerance: 0.01,
            kmeans_sample_cap: 131_072,
        }
    }

    /// Maximum palette colors (2–256). Default: 256.
    ///
    /// GIF, PNG, and WebP lossless all support up to 256 palette entries.
    #[must_use]
    pub fn with_max_colors(mut self, n: u32) -> Self {
        self.max_colors = n;
        self
    }

    /// Quality preset. Default: [`Quality::Best`].
    #[must_use]
    pub fn with_quality(mut self, q: Quality) -> Self {
        self.quality = q;
        self
    }

    /// Compute MPE quality metric during quantization.
    ///
    /// When enabled, the result includes per-block and global quality scores
    /// accessible via [`QuantizeResult::mpe_result()`] and [`QuantizeResult::mpe_score()`].
    /// Adds ~8 FLOPs/pixel overhead during dithering. Default: `false`.
    #[must_use]
    pub fn with_compute_quality_metric(mut self, enable: bool) -> Self {
        self.compute_metric = enable;
        self
    }

    /// Set target SSIM2 quality level (0–100, higher = better).
    ///
    /// When set, auto-tunes compression knobs (quality preset, dither strength,
    /// run priority) to maximize compression while staying above this level.
    /// Color count is NOT adjusted. Implicitly enables metric computation.
    #[must_use]
    pub fn with_target_ssim2(mut self, score: f32) -> Self {
        self.target_ssim2 = Some(score);
        self
    }

    /// Set minimum acceptable SSIM2 quality level (0–100, higher = better).
    ///
    /// Returns [`QuantizeError::QualityNotMet`] if the result falls below this.
    /// Implicitly enables metric computation.
    #[must_use]
    pub fn with_min_ssim2(mut self, score: f32) -> Self {
        self.min_ssim2 = Some(score);
        self
    }

    /// Maximum pixels sampled per k-means iteration. Default: 131072.
    ///
    /// Images with more pixels than this are stride-subsampled internally
    /// during pixel-level k-means refinement. The stride rotates each
    /// iteration to cover different pixels. Set to 0 to disable subsampling.
    #[must_use]
    pub fn with_kmeans_sample_cap(mut self, cap: usize) -> Self {
        self.kmeans_sample_cap = cap;
        self
    }

    // --- Hidden expert methods (not public API) ---

    /// Override dither mode. Not part of the public API.
    #[doc(hidden)]
    #[must_use]
    pub fn _with_no_dither(mut self) -> Self {
        self.dither_mode = dither::DitherMode::None;
        self
    }

    /// Use adaptive Floyd-Steinberg dithering (the default for most formats).
    #[doc(hidden)]
    #[must_use]
    pub fn _with_adaptive_dither(mut self) -> Self {
        self.dither_mode = dither::DitherMode::Adaptive;
        self
    }

    /// Set run priority to Quality (no run bias). Not part of the public API.
    #[doc(hidden)]
    #[must_use]
    pub fn _with_run_priority_quality(mut self) -> Self {
        self.run_priority = remap::RunPriority::Quality;
        self
    }

    /// Set run priority to Compression (aggressive runs). Not part of the public API.
    #[doc(hidden)]
    #[must_use]
    pub fn _with_run_priority_compression(mut self) -> Self {
        self.run_priority = remap::RunPriority::Compression;
        self
    }

    /// Override dither strength (0.0–1.0). Not part of the public API.
    #[doc(hidden)]
    #[must_use]
    pub fn _with_dither_strength(mut self, strength: f32) -> Self {
        self.dither_strength = Some(strength);
        self
    }

    /// Override Viterbi lambda. Not part of the public API.
    #[doc(hidden)]
    #[must_use]
    pub fn _with_viterbi_lambda(mut self, lambda: f32) -> Self {
        self.viterbi_lambda = Some(lambda);
        self
    }

    /// Use blue noise dithering (position-deterministic, zero flicker).
    #[doc(hidden)]
    #[must_use]
    pub fn _with_blue_noise_dither(mut self) -> Self {
        self.dither_mode = dither::DitherMode::BlueNoise;
        self
    }

    /// Use Sierra Lite dithering (lighter error diffusion, less temporal cascade).
    #[doc(hidden)]
    #[must_use]
    pub fn _with_sierra_lite_dither(mut self) -> Self {
        self.dither_mode = dither::DitherMode::SierraLite;
        self
    }

    /// Use linear dithering (unidirectional Floyd-Steinberg, no serpentine,
    /// no edge-aware dither map). Creates row-coherent patterns ideal for
    /// PNG compression. Best at low strength (0.1–0.3).
    #[doc(hidden)]
    #[must_use]
    pub fn _with_linear_dither(mut self) -> Self {
        self.dither_mode = dither::DitherMode::Linear;
        self
    }

    /// Override joint deflate effort (clamped to 1–22). Not part of the public API.
    /// (Retained for API compatibility; the vendored predictor ignores this.)
    #[doc(hidden)]
    #[must_use]
    pub fn _with_joint_deflate_effort(mut self, effort: u32) -> Self {
        self.joint_deflate_effort = effort.clamp(1, 22);
        self
    }

    /// Override joint base OKLab distance tolerance. Not part of the public API.
    #[doc(hidden)]
    #[must_use]
    pub fn _with_joint_tolerance(mut self, tolerance: f32) -> Self {
        self.joint_tolerance = tolerance;
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
    mpe_result: Option<metric::MpeResult>,
}

impl QuantizeResult {
    /// sRGB palette entries, sorted for the target output format.
    #[must_use]
    pub fn palette(&self) -> &[[u8; 3]] {
        self.palette.entries()
    }

    /// Palette index for each pixel, in row-major order: `pixel = y * width + x`.
    #[must_use]
    pub fn indices(&self) -> &[u8] {
        &self.indices
    }

    /// Get the transparent palette index, if any.
    #[must_use]
    pub fn transparent_index(&self) -> Option<u8> {
        self.palette.transparent_index()
    }

    /// Number of colors in the palette.
    #[must_use]
    pub fn palette_len(&self) -> usize {
        self.palette.len()
    }

    /// Get RGBA palette entries. Each entry has alpha: 255 for opaque,
    #[must_use]
    /// 0 for the transparent index, or the quantized alpha value.
    pub fn palette_rgba(&self) -> &[[u8; 4]] {
        self.palette.entries_rgba()
    }

    /// Get the alpha table suitable for a PNG tRNS chunk.
    #[must_use]
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

    /// Per-block and global MPE quality metric, if computed.
    #[must_use]
    ///
    /// Returns `Some` when [`QuantizeConfig::with_compute_quality_metric`] was used.
    pub fn mpe_result(&self) -> Option<&metric::MpeResult> {
        self.mpe_result.as_ref()
    }

    /// Global MPE quality score (lower is better), if computed.
    #[must_use]
    ///
    /// Convenience accessor — equivalent to `self.mpe_result().map(|r| r.score)`.
    pub fn mpe_score(&self) -> Option<f32> {
        self.mpe_result.as_ref().map(|r| r.score)
    }

    /// Estimated SSIMULACRA2 score (100 = identical), if metric was computed.
    #[must_use]
    pub fn ssimulacra2_estimate(&self) -> Option<f32> {
        self.mpe_result.as_ref().map(|r| r.ssimulacra2_estimate)
    }

    /// Estimated butteraugli distance, if metric was computed.
    #[must_use]
    pub fn butteraugli_estimate(&self) -> Option<f32> {
        self.mpe_result.as_ref().map(|r| r.butteraugli_estimate)
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
        remap_rgb_impl(&self.palette, pixels, width, height, config, None)
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
        remap_rgba_impl(&self.palette, pixels, width, height, config, None)
    }

    /// Remap with temporal clamping. Pixels whose undithered nearest palette
    /// match matches `prev_indices[i]` retain that index, preventing flicker.
    ///
    /// Both `pixels` and `prev_indices` must be full-frame buffers
    /// (`width × height` elements). Subframe cropping and disposal are the
    /// caller's responsibility — composite subframes onto the full canvas
    /// before calling this. Blue noise dithering relies on frame-absolute
    /// pixel coordinates for position-deterministic patterns; passing a
    /// cropped subregion would shift the noise tile and break temporal
    /// stability.
    #[doc(hidden)]
    pub fn remap_with_prev(
        &self,
        pixels: &[rgb::RGB<u8>],
        width: usize,
        height: usize,
        config: &QuantizeConfig,
        prev_indices: &[u8],
    ) -> Result<QuantizeResult, QuantizeError> {
        remap_rgb_impl(
            &self.palette,
            pixels,
            width,
            height,
            config,
            Some(prev_indices),
        )
    }

    /// Remap RGBA with temporal clamping. See [`remap_with_prev`](Self::remap_with_prev)
    /// for full-frame buffer requirements.
    #[doc(hidden)]
    pub fn remap_rgba_with_prev(
        &self,
        pixels: &[rgb::RGBA<u8>],
        width: usize,
        height: usize,
        config: &QuantizeConfig,
        prev_indices: &[u8],
    ) -> Result<QuantizeResult, QuantizeError> {
        remap_rgba_impl(
            &self.palette,
            pixels,
            width,
            height,
            config,
            Some(prev_indices),
        )
    }
}

/// Compression tier: combination of knobs that trade quality for compression.
///
/// Higher tiers = more aggressive compression, lower quality.
/// Calibrated thresholds based on MPE → SSIM2 lookup table measurements.
#[derive(Debug, Clone, Copy)]
struct CompressionTier {
    quality: Quality,
    run_priority: remap::RunPriority,
    /// Multiplied against format-default dither strength.
    dither_strength_mult: f32,
}

const COMPRESSION_TIERS: [CompressionTier; 5] = [
    // Tier 0: Maximum quality, no run optimization
    CompressionTier {
        quality: Quality::Best,
        run_priority: remap::RunPriority::Quality,
        dither_strength_mult: 1.0,
    },
    // Tier 1: Default Best settings
    CompressionTier {
        quality: Quality::Best,
        run_priority: remap::RunPriority::Balanced,
        dither_strength_mult: 1.0,
    },
    // Tier 2: Aggressive run optimization
    CompressionTier {
        quality: Quality::Best,
        run_priority: remap::RunPriority::Compression,
        dither_strength_mult: 1.0,
    },
    // Tier 3: Balanced quality + aggressive runs + reduced dither
    CompressionTier {
        quality: Quality::Balanced,
        run_priority: remap::RunPriority::Compression,
        dither_strength_mult: 0.8,
    },
    // Tier 4: Maximum compression
    CompressionTier {
        quality: Quality::Fast,
        run_priority: remap::RunPriority::Compression,
        dither_strength_mult: 0.6,
    },
];

/// Select a compression tier based on target SSIM2 score.
///
/// Uses conservative thresholds with safety margin so the resulting
/// quality stays above the target. Each tier step costs roughly 3–8
/// SSIM2 points based on calibration data.
fn select_compression_tier(target_ssim2: f32) -> &'static CompressionTier {
    // Conservative thresholds — tier is selected only if target allows
    // enough headroom. Based on MPE→SSIM2 calibration:
    //   Tier 0→1 costs ~2–4 SSIM2 points (run bias only)
    //   Tier 1→2 costs ~3–5 SSIM2 points (aggressive runs)
    //   Tier 2→3 costs ~5–8 SSIM2 points (fewer k-means iters)
    //   Tier 3→4 costs ~8–12 SSIM2 points (no masking)
    if target_ssim2 > 90.0 {
        &COMPRESSION_TIERS[0] // Can't afford any quality loss
    } else if target_ssim2 > 82.0 {
        &COMPRESSION_TIERS[1] // Default balanced runs
    } else if target_ssim2 > 74.0 {
        &COMPRESSION_TIERS[2] // Aggressive runs
    } else if target_ssim2 > 60.0 {
        &COMPRESSION_TIERS[3] // Balanced quality + compression
    } else {
        &COMPRESSION_TIERS[4] // Maximum compression
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

    // When target_ssim2 or min_ssim2 is set, force metric computation
    let needs_metric =
        config.compute_metric || config.target_ssim2.is_some() || config.min_ssim2.is_some();

    // Apply compression tier overrides when target_ssim2 is set
    let (effective_quality, effective_run_priority, effective_dither_strength) =
        if let Some(target) = config.target_ssim2 {
            let tier = select_compression_tier(target);
            (
                tier.quality,
                tier.run_priority,
                config
                    .dither_strength
                    .map(|s| s * tier.dither_strength_mult),
            )
        } else {
            (config.quality, config.run_priority, config.dither_strength)
        };

    // Build tuning from effective settings
    let effective_config = QuantizeConfig {
        quality: effective_quality,
        run_priority: effective_run_priority,
        dither_strength: effective_dither_strength,
        compute_metric: needs_metric,
        ..config.clone()
    };
    let tuning = QuantizeTuning::from_config(&effective_config);
    let max_colors = config.max_colors as usize;

    // Fast path: image already has ≤max_colors unique colors
    if let Some(exact_colors) = histogram::detect_exact_palette(pixels, max_colors) {
        let centroids = simd::batch_srgb_to_oklab_vec(&exact_colors);
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
            mpe_result: None,
        });
    }

    // Pipeline tiers based on quality:
    //   Fast:     no masking, histogram-only k-means, no Viterbi
    //   Balanced: masking + light pixel k-means (2 iters) + run extension
    //   Best:     masking + full pixel k-means (8 iters) + Viterbi DP
    let use_masking = matches!(effective_quality, Quality::Balanced | Quality::Best);
    let kmeans_iters: usize = match effective_quality {
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

    // 3. Median cut with histogram-level k-means refinement (always enabled)
    let mut centroids = median_cut::median_cut(hist, max_colors, true);

    // 3b. Pixel-level k-means refinement (skip for Fast — histogram refinement suffices).
    if kmeans_iters > 0 {
        centroids = median_cut::refine_against_pixels(
            centroids,
            pixels,
            &weights,
            kmeans_iters,
            config.kmeans_sample_cap,
        );
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
    let mut mpe_acc = if needs_metric {
        Some(metric::MpeAccumulator::new(width, height))
    } else {
        None
    };
    let dither_params = dither::DitherParams {
        width,
        height,
        weights: &weights,
        palette: &pal,
        mode: effective_config.dither_mode,
        run_priority: effective_run_priority,
        dither_strength: tuning.dither_strength,
        prev_indices: None,
    };
    let mut indices = dither::dither_image(pixels, &dither_params, mpe_acc.as_mut());

    // 5b. Run optimization
    //   Best:     full Viterbi DP (optimal run extension, ~26ms)
    //   Balanced: fast run-extend post-pass (greedy bidirectional, ~1ms)
    //   Fast:     none (dither-level greedy run-bias only)
    let use_viterbi = matches!(effective_quality, Quality::Best);
    let run_lambda = if use_masking {
        config
            .viterbi_lambda
            .unwrap_or(match effective_run_priority {
                remap::RunPriority::Quality => 0.0,
                remap::RunPriority::Balanced => 0.01,
                remap::RunPriority::Compression => 0.02,
            })
            * tuning.viterbi_lambda_scale
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

    let mpe_result = mpe_acc.map(|acc| acc.finalize());

    // Check min_ssim2 quality floor
    if let Some(min) = config.min_ssim2 {
        let achieved = mpe_result
            .as_ref()
            .map(|r| r.ssimulacra2_estimate)
            .unwrap_or(100.0);
        if achieved < min {
            return Err(QuantizeError::QualityNotMet {
                min_ssim2: min,
                achieved_ssim2: achieved,
            });
        }
    }

    // 7. Joint deflate+quantization optimization
    #[cfg(feature = "joint")]
    let indices = if matches!(
        config.output_format,
        OutputFormat::PngJoint | OutputFormat::PngMinSize
    ) {
        joint::optimize_rgb(
            pixels,
            width,
            height,
            &weights,
            &pal,
            &indices,
            config.joint_deflate_effort,
            config.joint_tolerance,
        )
    } else {
        indices
    };

    Ok(QuantizeResult {
        palette: pal,
        indices,
        mpe_result,
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

    // When target_ssim2 or min_ssim2 is set, force metric computation
    let needs_metric =
        config.compute_metric || config.target_ssim2.is_some() || config.min_ssim2.is_some();

    // Apply compression tier overrides when target_ssim2 is set
    let (effective_quality, effective_run_priority, effective_dither_strength) =
        if let Some(target) = config.target_ssim2 {
            let tier = select_compression_tier(target);
            (
                tier.quality,
                tier.run_priority,
                config
                    .dither_strength
                    .map(|s| s * tier.dither_strength_mult),
            )
        } else {
            (config.quality, config.run_priority, config.dither_strength)
        };

    let effective_config = QuantizeConfig {
        quality: effective_quality,
        run_priority: effective_run_priority,
        dither_strength: effective_dither_strength,
        compute_metric: needs_metric,
        ..config.clone()
    };
    let tuning = QuantizeTuning::from_config(&effective_config);
    let max_colors = config.max_colors as usize;

    // Fast path: image already has ≤max_colors unique colors
    if let Some((exact_colors, has_transparent)) =
        histogram::detect_exact_palette_rgba(pixels, max_colors)
    {
        let rgb_colors: Vec<rgb::RGB<u8>> = exact_colors
            .iter()
            .map(|c| rgb::RGB::new(c.r, c.g, c.b))
            .collect();
        let centroids = simd::batch_srgb_to_oklab_vec(&rgb_colors);
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
            mpe_result: None,
        });
    }

    let use_masking = matches!(effective_quality, Quality::Balanced | Quality::Best);
    let use_viterbi = matches!(effective_quality, Quality::Best);
    let kmeans_iters: usize = match effective_quality {
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
        let mut centroids = median_cut::median_cut_alpha(hist, opaque_colors, true);

        if kmeans_iters > 0 {
            centroids = median_cut::refine_against_pixels_alpha(
                centroids,
                pixels,
                &weights,
                kmeans_iters,
                config.kmeans_sample_cap,
            );
        }

        let pal = palette::Palette::from_centroids_alpha(
            centroids,
            has_transparent,
            tuning.sort_strategy,
        );

        let viterbi_lambda = if use_masking {
            config
                .viterbi_lambda
                .unwrap_or(match effective_run_priority {
                    remap::RunPriority::Quality => 0.0,
                    remap::RunPriority::Balanced => 0.01,
                    remap::RunPriority::Compression => 0.02,
                })
                * tuning.viterbi_lambda_scale
        } else {
            config.viterbi_lambda.unwrap_or(0.0)
        };
        let dither_params = dither::DitherParams {
            width,
            height,
            weights: &weights,
            palette: &pal,
            mode: effective_config.dither_mode,
            run_priority: effective_run_priority,
            dither_strength: tuning.dither_strength,
            prev_indices: None,
        };
        let mut indices = dither::dither_image_rgba_alpha(pixels, &dither_params, None);

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
        let mut centroids = median_cut::median_cut(hist, opaque_colors, true);

        if kmeans_iters > 0 {
            centroids = median_cut::refine_against_pixels_rgba(
                centroids,
                pixels,
                &weights,
                kmeans_iters,
                config.kmeans_sample_cap,
            );
        }

        let mut pal = palette::Palette::from_centroids_sorted(
            centroids,
            has_transparent,
            tuning.sort_strategy,
        );
        pal.build_nn_cache();

        let viterbi_lambda = if use_masking {
            config
                .viterbi_lambda
                .unwrap_or(match effective_run_priority {
                    remap::RunPriority::Quality => 0.0,
                    remap::RunPriority::Balanced => 0.01,
                    remap::RunPriority::Compression => 0.02,
                })
                * tuning.viterbi_lambda_scale
        } else {
            config.viterbi_lambda.unwrap_or(0.0)
        };
        let dither_params = dither::DitherParams {
            width,
            height,
            weights: &weights,
            palette: &pal,
            mode: effective_config.dither_mode,
            run_priority: effective_run_priority,
            dither_strength: tuning.dither_strength,
            prev_indices: None,
        };
        let mut indices = dither::dither_image_rgba(pixels, &dither_params, None);

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

    // Check min_ssim2 quality floor
    // Note: RGBA path doesn't currently compute inline MPE (mpe_result is None).
    // For min_ssim2 checking, we'd need standalone MPE computation here.
    // For now, compute standalone MPE when min_ssim2 is set.
    let mpe_result = if needs_metric {
        Some(metric::compute_mpe_rgba(
            pixels,
            pal.entries_rgba(),
            &indices,
            width,
            height,
            None,
        ))
    } else {
        None
    };

    if let Some(min) = config.min_ssim2 {
        let achieved = mpe_result
            .as_ref()
            .map(|r| r.ssimulacra2_estimate)
            .unwrap_or(100.0);
        if achieved < min {
            return Err(QuantizeError::QualityNotMet {
                min_ssim2: min,
                achieved_ssim2: achieved,
            });
        }
    }

    // Joint deflate+quantization optimization
    #[cfg(feature = "joint")]
    let indices = if matches!(
        config.output_format,
        OutputFormat::PngJoint | OutputFormat::PngMinSize
    ) {
        joint::optimize_rgba(
            pixels,
            width,
            height,
            &weights,
            &pal,
            &indices,
            config.joint_deflate_effort,
            config.joint_tolerance,
        )
    } else {
        indices
    };

    Ok(QuantizeResult {
        palette: pal,
        indices,
        mpe_result,
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
        let centroids = simd::batch_srgb_to_oklab_vec(&exact_colors);
        let mut pal =
            palette::Palette::from_centroids_sorted(centroids, false, tuning.sort_strategy);
        pal.build_nn_cache();
        return Ok(QuantizeResult {
            palette: pal,
            indices: Vec::new(),
            mpe_result: None,
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
    let mut centroids = median_cut::median_cut(merged_hist, max_colors, true);

    // K-means refinement against all pixels (internally subsampled)
    if kmeans_iters > 0 {
        centroids = median_cut::refine_against_pixels(
            centroids,
            &all_pixels,
            &all_weights,
            kmeans_iters,
            config.kmeans_sample_cap,
        );
    }

    let mut pal = palette::Palette::from_centroids_sorted(centroids, false, tuning.sort_strategy);
    pal.build_nn_cache();

    Ok(QuantizeResult {
        palette: pal,
        indices: Vec::new(),
        mpe_result: None,
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
            let rgb_colors: Vec<rgb::RGB<u8>> = exact_colors
                .iter()
                .map(|c| rgb::RGB::new(c.r, c.g, c.b))
                .collect();
            let labs = simd::batch_srgb_to_oklab_vec(&rgb_colors);
            let centroids: Vec<oklab::OKLabA> = labs
                .into_iter()
                .zip(exact_colors.iter())
                .map(|(lab, c)| lab.with_alpha(c.a as f32 / 255.0))
                .collect();
            let mut pal =
                palette::Palette::from_centroids_alpha(centroids, false, tuning.sort_strategy);
            pal.build_nn_cache();
            return Ok(QuantizeResult {
                palette: pal,
                indices: Vec::new(),
                mpe_result: None,
            });
        } else {
            let rgb_colors: Vec<rgb::RGB<u8>> = exact_colors
                .iter()
                .map(|c| rgb::RGB::new(c.r, c.g, c.b))
                .collect();
            let centroids = simd::batch_srgb_to_oklab_vec(&rgb_colors);
            let mut pal = palette::Palette::from_centroids_sorted(
                centroids,
                has_transparent,
                tuning.sort_strategy,
            );
            pal.build_nn_cache();
            return Ok(QuantizeResult {
                palette: pal,
                indices: Vec::new(),
                mpe_result: None,
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

        let mut centroids = median_cut::median_cut_alpha(merged_hist, max_colors, true);

        if kmeans_iters > 0 {
            centroids = median_cut::refine_against_pixels_alpha(
                centroids,
                &all_pixels,
                &all_weights,
                kmeans_iters,
                config.kmeans_sample_cap,
            );
        }

        let mut p = palette::Palette::from_centroids_alpha(centroids, false, tuning.sort_strategy);
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

        let mut centroids = median_cut::median_cut(merged_hist, opaque_colors, true);

        if kmeans_iters > 0 {
            centroids = median_cut::refine_against_pixels_rgba(
                centroids,
                &all_pixels,
                &all_weights,
                kmeans_iters,
                config.kmeans_sample_cap,
            );
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
        mpe_result: None,
    })
}

/// Internal: remap RGB pixels against an existing palette.
fn remap_rgb_impl(
    source_palette: &palette::Palette,
    pixels: &[rgb::RGB<u8>],
    width: usize,
    height: usize,
    config: &QuantizeConfig,
    prev_indices: Option<&[u8]>,
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

    // When target_ssim2 or min_ssim2 is set, force metric computation
    let needs_metric =
        config.compute_metric || config.target_ssim2.is_some() || config.min_ssim2.is_some();

    // Apply compression tier overrides when target_ssim2 is set
    let (effective_quality, effective_run_priority, effective_dither_strength) =
        if let Some(target) = config.target_ssim2 {
            let tier = select_compression_tier(target);
            (
                tier.quality,
                tier.run_priority,
                config
                    .dither_strength
                    .map(|s| s * tier.dither_strength_mult),
            )
        } else {
            (config.quality, config.run_priority, config.dither_strength)
        };

    let effective_config = QuantizeConfig {
        quality: effective_quality,
        run_priority: effective_run_priority,
        dither_strength: effective_dither_strength,
        compute_metric: needs_metric,
        ..config.clone()
    };
    let tuning = QuantizeTuning::from_config(&effective_config);

    let mut pal = source_palette.clone();
    if !pal.has_nn_cache() {
        pal.build_nn_cache();
    }

    let use_masking = matches!(effective_quality, Quality::Balanced | Quality::Best);
    let use_viterbi = matches!(effective_quality, Quality::Best);

    let weights = if use_masking {
        masking::compute_masking_weights(pixels, width, height)
    } else {
        vec![1.0f32; pixels.len()]
    };

    let dither_params = dither::DitherParams {
        width,
        height,
        weights: &weights,
        palette: &pal,
        mode: config.dither_mode,
        run_priority: effective_run_priority,
        dither_strength: tuning.dither_strength,
        prev_indices,
    };
    let mut indices = dither::dither_image(pixels, &dither_params, None);

    let run_lambda = if use_masking {
        config
            .viterbi_lambda
            .unwrap_or(match effective_run_priority {
                remap::RunPriority::Quality => 0.0,
                remap::RunPriority::Balanced => 0.01,
                remap::RunPriority::Compression => 0.02,
            })
            * tuning.viterbi_lambda_scale
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

    // Compute MPE quality metric if requested
    let mpe_result = if needs_metric {
        let w = if use_masking {
            Some(&weights[..])
        } else {
            None
        };
        Some(metric::compute_mpe(
            pixels,
            pal.entries(),
            &indices,
            width,
            height,
            w,
        ))
    } else {
        None
    };

    // Check min_ssim2 quality floor
    if let Some(min) = config.min_ssim2 {
        let achieved = mpe_result
            .as_ref()
            .map(|r| r.ssimulacra2_estimate)
            .unwrap_or(100.0);
        if achieved < min {
            return Err(QuantizeError::QualityNotMet {
                min_ssim2: min,
                achieved_ssim2: achieved,
            });
        }
    }

    // No frequency reorder — palette order must be stable for shared-palette use.
    Ok(QuantizeResult {
        palette: pal,
        indices,
        mpe_result,
    })
}

/// Internal: remap RGBA pixels against an existing palette.
fn remap_rgba_impl(
    source_palette: &palette::Palette,
    pixels: &[rgb::RGBA<u8>],
    width: usize,
    height: usize,
    config: &QuantizeConfig,
    prev_indices: Option<&[u8]>,
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

    // When target_ssim2 or min_ssim2 is set, force metric computation
    let needs_metric =
        config.compute_metric || config.target_ssim2.is_some() || config.min_ssim2.is_some();

    // Apply compression tier overrides when target_ssim2 is set
    let (effective_quality, effective_run_priority, effective_dither_strength) =
        if let Some(target) = config.target_ssim2 {
            let tier = select_compression_tier(target);
            (
                tier.quality,
                tier.run_priority,
                config
                    .dither_strength
                    .map(|s| s * tier.dither_strength_mult),
            )
        } else {
            (config.quality, config.run_priority, config.dither_strength)
        };

    let effective_config = QuantizeConfig {
        quality: effective_quality,
        run_priority: effective_run_priority,
        dither_strength: effective_dither_strength,
        compute_metric: needs_metric,
        ..config.clone()
    };
    let tuning = QuantizeTuning::from_config(&effective_config);

    let mut pal = source_palette.clone();
    if !pal.has_nn_cache() {
        pal.build_nn_cache();
    }

    let use_masking = matches!(effective_quality, Quality::Balanced | Quality::Best);
    let use_viterbi = matches!(effective_quality, Quality::Best);

    let weights = if use_masking {
        masking::compute_masking_weights_rgba(pixels, width, height)
    } else {
        vec![1.0f32; pixels.len()]
    };

    // Detect alpha mode from the palette: if any entry has alpha between 1-254,
    // the palette was built with full alpha quantization.
    let has_full_alpha = pal.entries_rgba().iter().any(|e| e[3] > 0 && e[3] < 255);

    let dither_params = dither::DitherParams {
        width,
        height,
        weights: &weights,
        palette: &pal,
        mode: config.dither_mode,
        run_priority: effective_run_priority,
        dither_strength: tuning.dither_strength,
        prev_indices,
    };
    let mut indices = if has_full_alpha {
        dither::dither_image_rgba_alpha(pixels, &dither_params, None)
    } else {
        dither::dither_image_rgba(pixels, &dither_params, None)
    };

    let run_lambda = if use_masking {
        config
            .viterbi_lambda
            .unwrap_or(match effective_run_priority {
                remap::RunPriority::Quality => 0.0,
                remap::RunPriority::Balanced => 0.01,
                remap::RunPriority::Compression => 0.02,
            })
            * tuning.viterbi_lambda_scale
    } else {
        config.viterbi_lambda.unwrap_or(0.0)
    };
    if run_lambda > 0.0 {
        if use_viterbi {
            remap::viterbi_refine_rgba(
                pixels,
                width,
                height,
                &weights,
                &pal,
                &mut indices,
                run_lambda,
            );
        } else {
            remap::run_extend_refine_rgba(
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

    // Compute MPE quality metric if requested
    let mpe_result = if needs_metric {
        Some(metric::compute_mpe_rgba(
            pixels,
            pal.entries_rgba(),
            &indices,
            width,
            height,
            None,
        ))
    } else {
        None
    };

    // Check min_ssim2 quality floor
    if let Some(min) = config.min_ssim2 {
        let achieved = mpe_result
            .as_ref()
            .map(|r| r.ssimulacra2_estimate)
            .unwrap_or(100.0);
        if achieved < min {
            return Err(QuantizeError::QualityNotMet {
                min_ssim2: min,
                achieved_ssim2: achieved,
            });
        }
    }

    // No frequency reorder — palette order must be stable for shared-palette use.
    Ok(QuantizeResult {
        palette: pal,
        indices,
        mpe_result,
    })
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
