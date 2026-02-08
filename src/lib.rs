#![forbid(unsafe_code)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod dither;
pub mod error;
pub mod histogram;
pub mod masking;
pub mod median_cut;
pub mod oklab;
pub mod palette;
pub mod remap;

pub use dither::DitherMode;
pub use error::QuantizeError;
pub use remap::RunPriority;

use alloc::vec::Vec;

/// Target output format — controls palette sorting, dither strength, and compression tuning.
///
/// Different image formats have fundamentally different compression algorithms,
/// so the optimal palette ordering and dithering strategy varies per format.
/// Setting the output format applies sensible defaults that can be overridden
/// via subsequent builder calls.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    /// No format-specific optimization. Full alpha quantization if RGBA.
    Generic,
    /// GIF: LZW compression, binary transparency only.
    /// Uses delta-minimize sort + post-remap frequency reorder.
    Gif,
    /// PNG: Deflate + scanline filters, per-index alpha via tRNS.
    /// Uses luminance sort for spatial locality.
    Png,
    /// WebP VP8L: Delta palette encoding + spatial prediction.
    /// Uses delta-minimize sort. Full RGBA palette.
    WebpLossless,
    /// JPEG XL modular: Squeeze wavelet + meta-adaptive entropy.
    /// Uses delta-minimize sort. Full RGBA palette.
    JxlModular,
}

impl Default for OutputFormat {
    fn default() -> Self {
        Self::Generic
    }
}

/// Internal tuning parameters derived from OutputFormat + user overrides.
#[derive(Debug, Clone)]
pub(crate) struct QuantizeTuning {
    pub(crate) dither_strength: f32,
    pub(crate) sort_strategy: palette::PaletteSortStrategy,
    pub(crate) gif_frequency_reorder: bool,
    pub(crate) alpha_mode: AlphaMode,
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
        let (default_dither, sort, gif_reorder, alpha) = match config.output_format {
            OutputFormat::Generic => (
                0.5,
                palette::PaletteSortStrategy::DeltaMinimize,
                false,
                AlphaMode::Full,
            ),
            OutputFormat::Gif => (
                0.5,
                palette::PaletteSortStrategy::DeltaMinimize,
                true,
                AlphaMode::Binary,
            ),
            OutputFormat::Png => (
                0.3,
                palette::PaletteSortStrategy::Luminance,
                false,
                AlphaMode::Full,
            ),
            OutputFormat::WebpLossless => (
                0.5,
                palette::PaletteSortStrategy::DeltaMinimize,
                false,
                AlphaMode::Full,
            ),
            OutputFormat::JxlModular => (
                0.4,
                palette::PaletteSortStrategy::DeltaMinimize,
                false,
                AlphaMode::Full,
            ),
        };

        Self {
            dither_strength: config.dither_strength.unwrap_or(default_dither),
            sort_strategy: sort,
            gif_frequency_reorder: gif_reorder,
            alpha_mode: alpha,
        }
    }
}

/// Configuration for palette quantization.
#[derive(Debug, Clone)]
pub struct QuantizeConfig {
    /// Maximum number of palette colors (2..=256).
    pub max_colors: u32,
    /// Quality parameter (0..=100). Higher = more accurate, slower.
    /// Controls k-means refinement and AQ damping.
    pub quality: u32,
    /// Run extension priority.
    pub run_priority: RunPriority,
    /// Dithering mode.
    pub dither: DitherMode,
    /// Target output format — controls sorting, dither strength, and alpha handling.
    pub output_format: OutputFormat,
    /// Override dither strength (0.0–1.0). If None, uses format-specific default.
    pub dither_strength: Option<f32>,
}

impl Default for QuantizeConfig {
    fn default() -> Self {
        Self {
            max_colors: 256,
            quality: 85,
            run_priority: RunPriority::Balanced,
            dither: DitherMode::Adaptive,
            output_format: OutputFormat::Generic,
            dither_strength: None,
        }
    }
}

impl QuantizeConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn max_colors(mut self, n: u32) -> Self {
        self.max_colors = n;
        self
    }

    pub fn quality(mut self, q: u32) -> Self {
        self.quality = q;
        self
    }

    pub fn run_priority(mut self, rp: RunPriority) -> Self {
        self.run_priority = rp;
        self
    }

    pub fn dither(mut self, mode: DitherMode) -> Self {
        self.dither = mode;
        self
    }

    pub fn output_format(mut self, format: OutputFormat) -> Self {
        self.output_format = format;
        self
    }

    pub fn dither_strength(mut self, strength: f32) -> Self {
        self.dither_strength = Some(strength);
        self
    }
}

/// Quantization result.
#[derive(Debug)]
pub struct QuantizeResult {
    palette: palette::Palette,
    indices: Vec<u8>,
}

impl QuantizeResult {
    /// Get the sRGB palette entries, delta-sorted.
    pub fn palette(&self) -> &[[u8; 3]] {
        self.palette.entries()
    }

    /// Get the palette index for each pixel.
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
}

/// Quantize an RGB image to a palette.
pub fn quantize(
    pixels: &[rgb::RGB<u8>],
    width: usize,
    height: usize,
    config: &QuantizeConfig,
) -> Result<QuantizeResult, QuantizeError> {
    validate_inputs(pixels.len(), width, height, config)?;

    let tuning = QuantizeTuning::from_config(config);

    // 1. Compute AQ masking weights
    let weights = masking::compute_masking_weights(pixels, width, height);

    // 2. Build weighted histogram
    let hist = histogram::build_histogram(pixels, &weights);

    // 3. Median cut with histogram-level k-means refinement
    let refine = config.quality >= 50;
    let max_colors = config.max_colors as usize;
    let mut centroids = median_cut::median_cut(hist, max_colors, refine);

    // 3b. Pixel-level k-means refinement.
    if refine {
        if pixels.len() <= 500_000 {
            centroids = median_cut::refine_against_pixels(centroids, pixels, &weights, 8);
        } else {
            let (sub_pixels, sub_weights) = subsample_pixels(pixels, &weights, width);
            centroids = median_cut::refine_against_pixels(centroids, &sub_pixels, &sub_weights, 8);
        }
    }

    // 4. Build palette with format-specific sort
    let pal = palette::Palette::from_centroids_sorted(centroids, false, tuning.sort_strategy);

    // 5. Dither / remap
    let indices = dither::dither_image(
        pixels,
        width,
        height,
        &weights,
        &pal,
        config.dither,
        config.run_priority,
        tuning.dither_strength,
    );

    Ok(QuantizeResult {
        palette: pal,
        indices,
    })
}

/// Quantize an RGBA image to a palette.
/// Fully transparent pixels (alpha == 0) are assigned a dedicated transparent index.
pub fn quantize_rgba(
    pixels: &[rgb::RGBA<u8>],
    width: usize,
    height: usize,
    config: &QuantizeConfig,
) -> Result<QuantizeResult, QuantizeError> {
    validate_inputs(pixels.len(), width, height, config)?;

    let tuning = QuantizeTuning::from_config(config);

    let weights = masking::compute_masking_weights_rgba(pixels, width, height);

    let (hist, has_transparent) = histogram::build_histogram_rgba(pixels, &weights);

    let refine = config.quality >= 50;
    // Reserve one slot for transparency if needed
    let max_colors = if has_transparent {
        (config.max_colors as usize).saturating_sub(1)
    } else {
        config.max_colors as usize
    };
    let mut centroids = median_cut::median_cut(hist, max_colors, refine);

    if refine {
        if pixels.len() <= 500_000 {
            centroids = median_cut::refine_against_pixels_rgba(centroids, pixels, &weights, 8);
        } else {
            let (sub_pixels, sub_weights) = subsample_pixels_rgba(pixels, &weights, width);
            centroids =
                median_cut::refine_against_pixels_rgba(centroids, &sub_pixels, &sub_weights, 8);
        }
    }

    let pal =
        palette::Palette::from_centroids_sorted(centroids, has_transparent, tuning.sort_strategy);

    let indices = dither::dither_image_rgba(
        pixels,
        width,
        height,
        &weights,
        &pal,
        config.dither,
        config.run_priority,
        tuning.dither_strength,
    );

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
    if config.quality > 100 {
        return Err(QuantizeError::InvalidQuality(config.quality));
    }
    Ok(())
}
