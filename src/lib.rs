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
}

impl Default for QuantizeConfig {
    fn default() -> Self {
        Self {
            max_colors: 256,
            quality: 85,
            run_priority: RunPriority::Balanced,
            dither: DitherMode::Adaptive,
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
}

/// Quantize an RGB image to a palette.
pub fn quantize(
    pixels: &[rgb::RGB<u8>],
    width: usize,
    height: usize,
    config: &QuantizeConfig,
) -> Result<QuantizeResult, QuantizeError> {
    validate_inputs(pixels.len(), width, height, config)?;

    // 1. Compute AQ masking weights
    let weights = masking::compute_masking_weights(pixels, width, height);

    // 2. Build weighted histogram
    let hist = histogram::build_histogram(pixels, &weights);

    // 3. Median cut with histogram-level k-means refinement
    let refine = config.quality >= 50;
    let max_colors = config.max_colors as usize;
    let mut centroids = median_cut::median_cut(hist, max_colors, refine);

    // 3b. Pixel-level k-means refinement.
    // For small images: refine against full pixel data (high precision).
    // For large images: subsample pixels for refinement (avoids O(n*k*iter) blowup).
    if refine {
        if pixels.len() <= 500_000 {
            centroids =
                median_cut::refine_against_pixels(centroids, pixels, &weights, 8);
        } else {
            let (sub_pixels, sub_weights) = subsample_pixels(pixels, &weights, width);
            centroids =
                median_cut::refine_against_pixels(centroids, &sub_pixels, &sub_weights, 8);
        }
    }

    // 4. Build palette with delta-minimizing sort
    let pal = palette::Palette::from_centroids(centroids, false);

    // 5. Dither / remap
    let indices = dither::dither_image(
        pixels,
        width,
        height,
        &weights,
        &pal,
        config.dither,
        config.run_priority,
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
            centroids =
                median_cut::refine_against_pixels_rgba(centroids, pixels, &weights, 8);
        } else {
            let (sub_pixels, sub_weights) = subsample_pixels_rgba(pixels, &weights, width);
            centroids =
                median_cut::refine_against_pixels_rgba(centroids, &sub_pixels, &sub_weights, 8);
        }
    }

    let pal = palette::Palette::from_centroids(centroids, has_transparent);

    let indices = dither::dither_image_rgba(
        pixels,
        width,
        height,
        &weights,
        &pal,
        config.dither,
        config.run_priority,
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
