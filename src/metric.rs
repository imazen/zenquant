//! Masked Perceptual Error (MPE) — fast perceptual quality metric.
//!
//! Combines luminance-biased pixel error with 4×4 block banding detection
//! and Minkowski-4 pooling that emphasizes worst-case blocks.
//!
//! Designed for inline accumulation during dithering (8 extra FLOPs/pixel)
//! or standalone comparison of any original/quantized image pair.

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

use crate::masking;
use crate::oklab::{OKLab, srgb_to_oklab};

/// Luminance weight — L channel errors are 2× more visible than chroma.
const W_L: f32 = 2.0;

/// Structure penalty weight — banding/noise penalty contribution relative to RMSE.
const LAMBDA: f32 = 0.5;

/// Minimum variance to consider a block as having meaningful structure.
const EPSILON: f32 = 1e-6;

/// Block size for structure analysis.
const BLOCK_SIZE: usize = 4;

/// Per-block and global perceptual quality score.
#[derive(Debug, Clone)]
pub struct MpeResult {
    /// Global Minkowski-4 pooled score (lower is better, 0 = identical).
    pub score: f32,
    /// Per-block scores in row-major order.
    pub block_scores: Vec<f32>,
    /// Number of block columns.
    pub block_cols: usize,
    /// Number of block rows.
    pub block_rows: usize,
    /// Estimated butteraugli distance (calibrated mapping).
    pub butteraugli_estimate: f32,
}

/// Accumulator for inline metric computation during dithering.
///
/// Tracks per-block statistics with 5 accumulators per block.
/// Call [`accumulate`](Self::accumulate) once per pixel during dithering,
/// then [`finalize`](Self::finalize) to produce the result.
pub struct MpeAccumulator {
    block_cols: usize,
    block_rows: usize,
    width: usize,
    /// Per-block: sum of weighted pixel errors.
    error_sum: Vec<f32>,
    /// Per-block: sum of original L values.
    orig_l_sum: Vec<f32>,
    /// Per-block: sum of original L² values.
    orig_l2_sum: Vec<f32>,
    /// Per-block: sum of quantized L values.
    quant_l_sum: Vec<f32>,
    /// Per-block: sum of quantized L² values.
    quant_l2_sum: Vec<f32>,
    /// Per-block: sum of masking weights (for averaging).
    weight_sum: Vec<f32>,
    /// Per-block: pixel count.
    pixel_count: Vec<u32>,
}

impl MpeAccumulator {
    /// Create a new accumulator for an image of the given dimensions.
    pub fn new(width: usize, height: usize) -> Self {
        let block_cols = width.div_ceil(BLOCK_SIZE);
        let block_rows = height.div_ceil(BLOCK_SIZE);
        let n = block_cols * block_rows;
        Self {
            block_cols,
            block_rows,
            width,
            error_sum: vec![0.0; n],
            orig_l_sum: vec![0.0; n],
            orig_l2_sum: vec![0.0; n],
            quant_l_sum: vec![0.0; n],
            quant_l2_sum: vec![0.0; n],
            weight_sum: vec![0.0; n],
            pixel_count: vec![0; n],
        }
    }

    /// Accumulate one pixel's contribution. Called during dithering.
    ///
    /// `pixel_idx` is the linear pixel index (y * width + x).
    #[inline]
    pub fn accumulate(
        &mut self,
        pixel_idx: usize,
        orig_lab: OKLab,
        chosen_lab: OKLab,
        weight: f32,
    ) {
        let x = pixel_idx % self.width;
        let y = pixel_idx / self.width;
        let bx = x / BLOCK_SIZE;
        let by = y / BLOCK_SIZE;
        let bi = by * self.block_cols + bx;

        let dl = orig_lab.l - chosen_lab.l;
        let da = orig_lab.a - chosen_lab.a;
        let db = orig_lab.b - chosen_lab.b;
        let pixel_error = (W_L * dl * dl + da * da + db * db) * weight;

        self.error_sum[bi] += pixel_error;
        self.orig_l_sum[bi] += orig_lab.l;
        self.orig_l2_sum[bi] += orig_lab.l * orig_lab.l;
        self.quant_l_sum[bi] += chosen_lab.l;
        self.quant_l2_sum[bi] += chosen_lab.l * chosen_lab.l;
        self.weight_sum[bi] += weight;
        self.pixel_count[bi] += 1;
    }

    /// Finalize: compute per-block RMSE scores and global Minkowski-4 pooled score.
    pub fn finalize(self) -> MpeResult {
        let n = self.block_cols * self.block_rows;
        let mut block_scores = Vec::with_capacity(n);

        for bi in 0..n {
            let count = self.pixel_count[bi] as f32;
            if count < 1.0 {
                block_scores.push(0.0);
                continue;
            }

            // Block RMSE — sqrt of mean squared error gives scores in a useful
            // range (~0.01–0.5) instead of MSE's tiny values (~0.0001–0.01).
            let rmse = (self.error_sum[bi] / count).sqrt();

            // Variance of original L: E[L²] - E[L]²
            let orig_l_mean = self.orig_l_sum[bi] / count;
            let orig_l_var =
                ((self.orig_l2_sum[bi] / count) - orig_l_mean * orig_l_mean).max(0.0);

            // Variance of quantized L
            let quant_l_mean = self.quant_l_sum[bi] / count;
            let quant_l_var =
                ((self.quant_l2_sum[bi] / count) - quant_l_mean * quant_l_mean).max(0.0);

            // Structure penalty: detects both banding (variance collapse) and
            // excessive dither noise (variance inflation). Measures how far the
            // variance ratio deviates from 1.0.
            let structure_penalty = if orig_l_var > EPSILON {
                let ratio = quant_l_var / orig_l_var;
                // |log(ratio)| penalizes both under and over. Clamped to [0, 2].
                let log_ratio = ratio.ln().abs();
                log_ratio.min(2.0)
            } else if quant_l_var > EPSILON {
                // Original was flat but quantized has noise → dither artifact
                1.0
            } else {
                0.0
            };

            let mean_weight = self.weight_sum[bi] / count;
            let score = rmse + LAMBDA * structure_penalty * mean_weight * rmse;
            block_scores.push(score);
        }

        // Minkowski-4 pooling: emphasizes worst-case blocks.
        let global = minkowski8_pool(&block_scores);
        let butteraugli_estimate = mpe_to_butteraugli(global);

        MpeResult {
            score: global,
            block_scores,
            block_cols: self.block_cols,
            block_rows: self.block_rows,
            butteraugli_estimate,
        }
    }
}

/// Compute MPE from original RGB pixels and a quantized result.
///
/// This is the standalone variant — works with output from any quantizer.
/// For inline accumulation during zenquant dithering, use the config flag instead.
///
/// When `weights` is `None`, computes AQ masking weights from the original image
/// (edge/texture detection, same as zenquant's quantization pipeline). This gives
/// perceptually-weighted scores where errors in textured regions are masked.
/// Pass explicit weights to override.
pub fn compute_mpe(
    pixels: &[rgb::RGB<u8>],
    palette: &[[u8; 3]],
    indices: &[u8],
    width: usize,
    height: usize,
    weights: Option<&[f32]>,
) -> MpeResult {
    debug_assert_eq!(pixels.len(), width * height);
    debug_assert_eq!(indices.len(), width * height);

    // Auto-compute masking weights when none provided
    let auto_weights;
    let w_slice: &[f32] = match weights {
        Some(ws) => ws,
        None => {
            auto_weights = masking::compute_masking_weights(pixels, width, height);
            &auto_weights
        }
    };

    let mut acc = MpeAccumulator::new(width, height);

    for (i, (pixel, &idx)) in pixels.iter().zip(indices.iter()).enumerate() {
        let orig = srgb_to_oklab(pixel.r, pixel.g, pixel.b);
        let p = palette[idx as usize];
        let quant = srgb_to_oklab(p[0], p[1], p[2]);
        acc.accumulate(i, orig, quant, w_slice[i]);
    }

    acc.finalize()
}

/// Compute MPE from original RGBA pixels and a quantized result.
///
/// Transparent pixels (alpha == 0) are skipped.
pub fn compute_mpe_rgba(
    pixels: &[rgb::RGBA<u8>],
    palette: &[[u8; 4]],
    indices: &[u8],
    width: usize,
    height: usize,
    weights: Option<&[f32]>,
) -> MpeResult {
    debug_assert_eq!(pixels.len(), width * height);
    debug_assert_eq!(indices.len(), width * height);

    let default_weight = 1.0f32;
    let mut acc = MpeAccumulator::new(width, height);

    for (i, (pixel, &idx)) in pixels.iter().zip(indices.iter()).enumerate() {
        if pixel.a == 0 {
            continue;
        }
        let orig = srgb_to_oklab(pixel.r, pixel.g, pixel.b);
        let p = palette[idx as usize];
        let quant = srgb_to_oklab(p[0], p[1], p[2]);
        let w = weights.map_or(default_weight, |ws| ws[i]);
        acc.accumulate(i, orig, quant, w);
    }

    acc.finalize()
}

/// Minkowski-8 pooling: `(mean(x⁸))^(1/8)`.
///
/// Higher exponent (8 vs 4) better matches butteraugli's worst-case behavior.
/// Mean is over ALL blocks (including zeros), so images with few bad blocks
/// get pulled down rather than having zeros excluded from the denominator.
fn minkowski8_pool(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sum = 0.0f64;
    for &v in values {
        let v64 = v as f64;
        let v2 = v64 * v64;
        let v4 = v2 * v2;
        sum += v4 * v4;
    }
    let mean = sum / values.len() as f64;
    mean.powf(0.125) as f32
}

/// Map MPE score to estimated butteraugli distance.
///
/// Calibrated from CID22 (209 photos), CLIC 2025 (62 images), and gb82-sc
/// (11 screenshots) across 8–256 color quantization levels (N=1677).
/// Power-law fit: BA ≈ 111 × MPE^0.95 (R²=0.70 log-log, Spearman 0.83).
fn mpe_to_butteraugli(mpe: f32) -> f32 {
    111.0 * mpe.powf(0.948)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_flat_image(r: u8, g: u8, b: u8, n: usize) -> Vec<rgb::RGB<u8>> {
        vec![rgb::RGB { r, g, b }; n]
    }

    fn make_gradient(width: usize, height: usize) -> Vec<rgb::RGB<u8>> {
        let mut pixels = Vec::with_capacity(width * height);
        for y in 0..height {
            for x in 0..width {
                let v = ((y * width + x) * 255 / (width * height).max(1)) as u8;
                pixels.push(rgb::RGB { r: v, g: v, b: v });
            }
        }
        pixels
    }

    #[test]
    fn identical_image_scores_zero() {
        let pixels = make_flat_image(128, 64, 32, 16);
        let palette = [[128, 64, 32]];
        let indices = vec![0u8; 16];

        let result = compute_mpe(&pixels, &palette, &indices, 4, 4, None);
        assert_eq!(result.score, 0.0, "identical image should score 0");
        assert!(
            result.block_scores.iter().all(|&s| s == 0.0),
            "all blocks should be 0"
        );
    }

    #[test]
    fn different_image_scores_positive() {
        let pixels = make_flat_image(0, 0, 0, 16);
        let palette = [[255, 255, 255]];
        let indices = vec![0u8; 16];

        let result = compute_mpe(&pixels, &palette, &indices, 4, 4, None);
        assert!(result.score > 0.0, "different image should score > 0");
    }

    #[test]
    fn banding_detection() {
        // Original: smooth gradient across a 4×4 block
        let pixels = make_gradient(4, 4);

        // "Good" quantization: 4 distinct levels
        let palette_good = [[0, 0, 0], [85, 85, 85], [170, 170, 170], [255, 255, 255]];
        let indices_good: Vec<u8> = (0..16)
            .map(|i| {
                let v = (i * 255 / 15) as u8;
                if v < 64 {
                    0
                } else if v < 128 {
                    1
                } else if v < 192 {
                    2
                } else {
                    3
                }
            })
            .collect();

        // "Bad" quantization: everything mapped to one color (total banding)
        let palette_bad = [[128, 128, 128]];
        let indices_bad = vec![0u8; 16];

        let result_good = compute_mpe(&pixels, &palette_good, &indices_good, 4, 4, None);
        let result_bad = compute_mpe(&pixels, &palette_bad, &indices_bad, 4, 4, None);

        assert!(
            result_bad.score > result_good.score,
            "flat quantization should score worse than multi-level: bad={}, good={}",
            result_bad.score,
            result_good.score
        );
    }

    #[test]
    fn masking_weights_modulate_score() {
        let pixels = make_flat_image(0, 0, 0, 16);
        let palette = [[64, 64, 64]];
        let indices = vec![0u8; 16];

        let high_weights = vec![1.0f32; 16];
        let low_weights = vec![0.1f32; 16];

        let result_high = compute_mpe(&pixels, &palette, &indices, 4, 4, Some(&high_weights));
        let result_low = compute_mpe(&pixels, &palette, &indices, 4, 4, Some(&low_weights));

        assert!(
            result_high.score > result_low.score,
            "high masking weight should produce higher score: high={}, low={}",
            result_high.score,
            result_low.score
        );
    }

    #[test]
    fn block_dimensions_correct() {
        // 8×8 image → 2×2 blocks
        let pixels = make_flat_image(128, 128, 128, 64);
        let palette = [[128, 128, 128]];
        let indices = vec![0u8; 64];

        let result = compute_mpe(&pixels, &palette, &indices, 8, 8, None);
        assert_eq!(result.block_cols, 2);
        assert_eq!(result.block_rows, 2);
        assert_eq!(result.block_scores.len(), 4);
    }

    #[test]
    fn non_aligned_dimensions() {
        // 5×5 image → 2×2 blocks (ceil(5/4) = 2)
        let pixels = make_flat_image(100, 100, 100, 25);
        let palette = [[100, 100, 100]];
        let indices = vec![0u8; 25];

        let result = compute_mpe(&pixels, &palette, &indices, 5, 5, None);
        assert_eq!(result.block_cols, 2);
        assert_eq!(result.block_rows, 2);
        assert_eq!(result.block_scores.len(), 4);
        assert_eq!(result.score, 0.0);
    }

    #[test]
    fn single_pixel() {
        let pixels = vec![rgb::RGB {
            r: 100,
            g: 50,
            b: 25,
        }];
        let palette = [[100, 50, 25]];
        let indices = vec![0u8];

        let result = compute_mpe(&pixels, &palette, &indices, 1, 1, None);
        assert_eq!(result.block_cols, 1);
        assert_eq!(result.block_rows, 1);
        assert_eq!(result.score, 0.0);
    }

    #[test]
    fn rgba_transparent_pixels_skipped() {
        let pixels = vec![
            rgb::RGBA {
                r: 0,
                g: 0,
                b: 0,
                a: 0,
            };
            16
        ];
        let palette = [[255, 255, 255, 255]];
        let indices = vec![0u8; 16];

        let result = compute_mpe_rgba(&pixels, &palette, &indices, 4, 4, None);
        // All transparent → no accumulation → score 0
        assert_eq!(result.score, 0.0);
    }

    #[test]
    fn accumulator_matches_standalone() {
        let pixels = make_gradient(8, 8);
        let palette = [[0, 0, 0], [128, 128, 128], [255, 255, 255]];
        let indices: Vec<u8> = pixels
            .iter()
            .map(|p| {
                if p.r < 85 {
                    0
                } else if p.r < 170 {
                    1
                } else {
                    2
                }
            })
            .collect();
        let weights = vec![0.8f32; 64];

        // Standalone
        let standalone = compute_mpe(&pixels, &palette, &indices, 8, 8, Some(&weights));

        // Accumulator
        let mut acc = MpeAccumulator::new(8, 8);
        for (i, (pixel, &idx)) in pixels.iter().zip(indices.iter()).enumerate() {
            let orig = srgb_to_oklab(pixel.r, pixel.g, pixel.b);
            let p = palette[idx as usize];
            let quant = srgb_to_oklab(p[0], p[1], p[2]);
            acc.accumulate(i, orig, quant, weights[i]);
        }
        let from_acc = acc.finalize();

        assert!(
            (standalone.score - from_acc.score).abs() < 1e-6,
            "standalone={} vs accumulator={}",
            standalone.score,
            from_acc.score
        );
        assert_eq!(standalone.block_scores.len(), from_acc.block_scores.len());
        for (i, (a, b)) in standalone
            .block_scores
            .iter()
            .zip(from_acc.block_scores.iter())
            .enumerate()
        {
            assert!(
                (a - b).abs() < 1e-6,
                "block {i}: standalone={a} vs accumulator={b}"
            );
        }
    }

    #[test]
    fn butteraugli_estimate_finite() {
        let pixels = make_gradient(8, 8);
        let palette = [[0, 0, 0], [255, 255, 255]];
        let indices: Vec<u8> = pixels.iter().map(|p| if p.r < 128 { 0 } else { 1 }).collect();

        let result = compute_mpe(&pixels, &palette, &indices, 8, 8, None);
        assert!(result.butteraugli_estimate.is_finite());
        assert!(result.butteraugli_estimate >= 0.0);
    }

    #[test]
    fn minkowski8_empty() {
        assert_eq!(minkowski8_pool(&[]), 0.0);
    }

    #[test]
    fn minkowski8_uniform() {
        // All same value → result should equal that value.
        let values = vec![0.5f32; 10];
        let result = minkowski8_pool(&values);
        assert!((result - 0.5).abs() < 1e-5, "got {result}");
    }

    #[test]
    fn minkowski8_emphasizes_outliers() {
        // One outlier among zeros: result should be > simple mean but < outlier value.
        let mut values = vec![0.0f32; 9];
        values.push(1.0);
        let mink = minkowski8_pool(&values);
        // mean(x^8) = 1/10, (1/10)^(1/8) ≈ 0.749
        let expected = (1.0f64 / 10.0).powf(0.125) as f32;
        assert!(
            (mink - expected).abs() < 1e-5,
            "expected {expected}, got {mink}"
        );
    }
}
