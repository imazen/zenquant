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
    /// Estimated butteraugli distance (calibrated lookup table).
    pub butteraugli_estimate: f32,
    /// Estimated SSIMULACRA2 score (calibrated lookup table, 100 = identical).
    pub ssimulacra2_estimate: f32,
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
        let ssimulacra2_estimate = mpe_to_ssimulacra2(global);

        MpeResult {
            score: global,
            block_scores,
            block_cols: self.block_cols,
            block_rows: self.block_rows,
            butteraugli_estimate,
            ssimulacra2_estimate,
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
/// Composites both original and quantized against a checkerboard pattern
/// so that transparency differences (especially at edges) show up as
/// visible color errors rather than being silently skipped.
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
        let x = i % width;
        let y = i / width;

        // Composite both original and quantized against checkerboard
        let bg = checkerboard_color(x, y);
        let orig = composite_over_bg(pixel.r, pixel.g, pixel.b, pixel.a, bg);
        let p = palette[idx as usize];
        let quant = composite_over_bg(p[0], p[1], p[2], p[3], bg);

        let w = weights.map_or(default_weight, |ws| ws[i]);
        acc.accumulate(i, orig, quant, w);
    }

    acc.finalize()
}

/// Get checkerboard background color for a pixel position.
///
/// 8×8 pixel squares alternating between white (255) and black (0).
/// Full-contrast B/W maximizes sensitivity to alpha errors: dark foreground
/// shows up against white squares, light foreground against black. Measured
/// at 92% of the theoretical optimum (two-pass max) across 9 synthetic
/// transparency scenarios.
#[inline]
fn checkerboard_color(x: usize, y: usize) -> u8 {
    if ((x >> 3) ^ (y >> 3)) & 1 == 0 {
        255 // white square
    } else {
        0 // black square
    }
}

/// Composite an RGBA pixel over a solid background using standard alpha blending.
///
/// Returns the composited color in OKLab space.
#[inline]
fn composite_over_bg(r: u8, g: u8, b: u8, a: u8, bg: u8) -> OKLab {
    let alpha = a as f32 / 255.0;
    let inv_alpha = 1.0 - alpha;
    let bg_f = bg as f32;

    let cr = (r as f32 * alpha + bg_f * inv_alpha) as u8;
    let cg = (g as f32 * alpha + bg_f * inv_alpha) as u8;
    let cb = (b as f32 * alpha + bg_f * inv_alpha) as u8;

    srgb_to_oklab(cr, cg, cb)
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

/// Lookup table: MPE (mink8) → estimated butteraugli distance.
///
/// Calibrated from 1992 data points across CID22, CLIC 2025, gb82-sc,
/// and KADID-10k corpuses at 8–256 color quantization levels.
/// Median butteraugli per 0.01-wide MPE bin, monotonicity enforced.
///
/// For reference, JPEG quality equivalences (median across same corpuses, N=3320):
///
/// | JPEG q | butteraugli | SSIMULACRA2 | ≈ MPE  |
/// |--------|-------------|-------------|--------|
/// |     95 |        1.18 |        91.1 | 0.0077 |
/// |     90 |        1.80 |        87.5 | 0.0117 |
/// |     85 |        2.26 |        84.2 | 0.0146 |
/// |     80 |        2.62 |        81.5 | 0.0174 |
/// |     75 |        2.96 |        78.9 | 0.0201 |
/// |     70 |        3.20 |        76.5 | 0.0221 |
/// |     60 |        3.57 |        72.1 | 0.0250 |
/// |     50 |        3.90 |        68.6 | 0.0276 |
/// |     40 |        4.26 |        63.5 | 0.0304 |
/// |     30 |        4.72 |        55.8 | 0.0339 |
const MPE_BA_TABLE: [(f32, f32); 24] = [
    (0.0, 0.0),
    (0.005, 0.76),
    (0.015, 2.32),
    (0.025, 3.57),
    (0.035, 4.85),
    (0.045, 5.99),
    (0.055, 7.34),
    (0.065, 8.62),
    (0.075, 9.73),
    (0.085, 11.59),
    (0.095, 12.83),
    (0.105, 13.64),
    (0.115, 16.65),
    (0.125, 16.65),
    (0.135, 18.80),
    (0.145, 18.96),
    (0.155, 19.98),
    (0.165, 22.59),
    (0.175, 22.77),
    (0.185, 26.88),
    (0.195, 26.88),
    (0.205, 27.22),
    (0.215, 27.22),
    (0.225, 28.42),
];

/// Lookup table: MPE (mink8) → estimated SSIMULACRA2 score.
///
/// Same calibration dataset as `MPE_BA_TABLE`.
/// Median SSIMULACRA2 per bin, monotonicity enforced (decreasing).
const MPE_SSIM2_TABLE: [(f32, f32); 24] = [
    (0.0, 100.0),
    (0.005, 93.54),
    (0.015, 84.18),
    (0.025, 78.64),
    (0.035, 73.18),
    (0.045, 67.70),
    (0.055, 61.83),
    (0.065, 56.93),
    (0.075, 51.84),
    (0.085, 45.49),
    (0.095, 36.74),
    (0.105, 29.34),
    (0.115, 18.84),
    (0.125, 18.84),
    (0.135, 11.19),
    (0.145, 7.61),
    (0.155, -1.61),
    (0.165, -1.61),
    (0.175, -12.30),
    (0.185, -12.30),
    (0.195, -16.69),
    (0.205, -16.86),
    (0.215, -19.89),
    (0.225, -19.89),
];

/// Piecewise linear interpolation on a sorted lookup table.
///
/// Binary-searches for the bracketing interval, then linearly interpolates.
/// Clamps to endpoint values for out-of-range inputs.
fn interpolate(table: &[(f32, f32)], x: f32) -> f32 {
    debug_assert!(!table.is_empty());

    // Clamp to endpoints
    if x <= table[0].0 {
        return table[0].1;
    }
    let last = table.len() - 1;
    if x >= table[last].0 {
        return table[last].1;
    }

    // Binary search for the interval [table[i].0, table[i+1].0) containing x
    let i = match table.binary_search_by(|entry| entry.0.partial_cmp(&x).unwrap()) {
        Ok(exact) => return table[exact].1,
        Err(pos) => pos - 1, // pos is where x would be inserted; i = pos-1
    };

    let (x0, y0) = table[i];
    let (x1, y1) = table[i + 1];
    let t = (x - x0) / (x1 - x0);
    y0 + t * (y1 - y0)
}

/// Map MPE score to estimated butteraugli distance.
fn mpe_to_butteraugli(mpe: f32) -> f32 {
    interpolate(&MPE_BA_TABLE, mpe)
}

/// Map MPE score to estimated SSIMULACRA2 score.
fn mpe_to_ssimulacra2(mpe: f32) -> f32 {
    interpolate(&MPE_SSIM2_TABLE, mpe)
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
    fn rgba_transparent_matching_scores_zero() {
        // When both original and palette are fully transparent, compositing
        // over checkerboard gives identical results → score 0.
        let pixels = vec![
            rgb::RGBA {
                r: 0,
                g: 0,
                b: 0,
                a: 0,
            };
            16
        ];
        let palette = [[0, 0, 0, 0]]; // also transparent
        let indices = vec![0u8; 16];

        let result = compute_mpe_rgba(&pixels, &palette, &indices, 4, 4, None);
        assert_eq!(result.score, 0.0);
    }

    #[test]
    fn rgba_transparency_mismatch_detected() {
        // Original is fully transparent but quantized is opaque red →
        // compositing over checkerboard should produce a large error.
        // (Uses red, not white, because a 4×4 image fits in one white
        // checker cell — opaque white would blend identically.)
        let pixels = vec![
            rgb::RGBA {
                r: 0,
                g: 0,
                b: 0,
                a: 0,
            };
            16
        ];
        let palette = [[255, 0, 0, 255]]; // opaque red
        let indices = vec![0u8; 16];

        let result = compute_mpe_rgba(&pixels, &palette, &indices, 4, 4, None);
        assert!(
            result.score > 0.0,
            "transparency mismatch should be detected, got score={}",
            result.score
        );
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
        assert!(result.ssimulacra2_estimate.is_finite());
        assert!(result.ssimulacra2_estimate < 100.0);
    }

    #[test]
    fn identical_image_estimates() {
        let pixels = make_flat_image(128, 64, 32, 16);
        let palette = [[128, 64, 32]];
        let indices = vec![0u8; 16];

        let result = compute_mpe(&pixels, &palette, &indices, 4, 4, None);
        assert_eq!(result.butteraugli_estimate, 0.0, "identical → BA=0");
        assert_eq!(result.ssimulacra2_estimate, 100.0, "identical → SS2=100");
    }

    #[test]
    fn different_image_estimates() {
        let pixels = make_flat_image(0, 0, 0, 16);
        let palette = [[255, 255, 255]];
        let indices = vec![0u8; 16];

        let result = compute_mpe(&pixels, &palette, &indices, 4, 4, None);
        assert!(
            result.butteraugli_estimate > 0.0,
            "different → BA > 0: {}",
            result.butteraugli_estimate
        );
        assert!(
            result.ssimulacra2_estimate < 100.0,
            "different → SS2 < 100: {}",
            result.ssimulacra2_estimate
        );
    }

    #[test]
    fn interpolate_endpoints() {
        let table = [(0.0f32, 0.0f32), (1.0, 10.0), (2.0, 20.0)];
        // Exact endpoints
        assert_eq!(interpolate(&table, 0.0), 0.0);
        assert_eq!(interpolate(&table, 2.0), 20.0);
        // Below range → clamp to first
        assert_eq!(interpolate(&table, -1.0), 0.0);
        // Above range → clamp to last
        assert_eq!(interpolate(&table, 5.0), 20.0);
    }

    #[test]
    fn interpolate_midpoint() {
        let table = [(0.0f32, 0.0f32), (1.0, 10.0), (2.0, 20.0)];
        let mid = interpolate(&table, 0.5);
        assert!((mid - 5.0).abs() < 1e-5, "expected 5.0, got {mid}");

        let mid2 = interpolate(&table, 1.5);
        assert!((mid2 - 15.0).abs() < 1e-5, "expected 15.0, got {mid2}");
    }

    #[test]
    fn interpolate_exact_table_entry() {
        let table = [(0.0f32, 0.0f32), (0.5, 7.0), (1.0, 10.0)];
        assert_eq!(interpolate(&table, 0.5), 7.0);
    }

    #[test]
    fn ba_table_monotonic() {
        for i in 1..MPE_BA_TABLE.len() {
            assert!(
                MPE_BA_TABLE[i].0 > MPE_BA_TABLE[i - 1].0,
                "MPE_BA_TABLE x not strictly increasing at index {i}"
            );
            assert!(
                MPE_BA_TABLE[i].1 >= MPE_BA_TABLE[i - 1].1,
                "MPE_BA_TABLE y not monotonically increasing at index {i}: {} < {}",
                MPE_BA_TABLE[i].1,
                MPE_BA_TABLE[i - 1].1
            );
        }
    }

    #[test]
    fn ssim2_table_monotonic() {
        for i in 1..MPE_SSIM2_TABLE.len() {
            assert!(
                MPE_SSIM2_TABLE[i].0 > MPE_SSIM2_TABLE[i - 1].0,
                "MPE_SSIM2_TABLE x not strictly increasing at index {i}"
            );
            assert!(
                MPE_SSIM2_TABLE[i].1 <= MPE_SSIM2_TABLE[i - 1].1,
                "MPE_SSIM2_TABLE y not monotonically decreasing at index {i}: {} > {}",
                MPE_SSIM2_TABLE[i].1,
                MPE_SSIM2_TABLE[i - 1].1
            );
        }
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
