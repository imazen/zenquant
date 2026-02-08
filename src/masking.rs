extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

use crate::oklab::srgb_to_oklab;

/// Compute per-pixel AQ masking weights from an RGB image.
///
/// Returns weights in [0.1, 1.0] where:
/// - High weight (≈1.0) = smooth region → protect quality, allocate palette entries
/// - Low weight (≈0.1) = textured region → error is masked, allow more quantization
pub fn compute_masking_weights(pixels: &[rgb::RGB<u8>], width: usize, height: usize) -> Vec<f32> {
    let luminance = extract_luminance(pixels);
    let contrast = compute_local_contrast(&luminance, width, height);
    let block_w = width.div_ceil(4);
    let block_h = height.div_ceil(4);
    let block_masking = erode_to_blocks(&contrast, width, height, block_w, block_h);
    let per_pixel = upscale_bilinear(&block_masking, block_w, block_h, width, height);
    masking_to_weights(&per_pixel)
}

/// Same as above but for RGBA input.
pub fn compute_masking_weights_rgba(
    pixels: &[rgb::RGBA<u8>],
    width: usize,
    height: usize,
) -> Vec<f32> {
    let luminance = extract_luminance_rgba(pixels);
    let contrast = compute_local_contrast(&luminance, width, height);
    let block_w = width.div_ceil(4);
    let block_h = height.div_ceil(4);
    let block_masking = erode_to_blocks(&contrast, width, height, block_w, block_h);
    let per_pixel = upscale_bilinear(&block_masking, block_w, block_h, width, height);
    masking_to_weights(&per_pixel)
}

/// Extract OKLab L (lightness) channel from RGB pixels.
fn extract_luminance(pixels: &[rgb::RGB<u8>]) -> Vec<f32> {
    pixels
        .iter()
        .map(|p| srgb_to_oklab(p.r, p.g, p.b).l)
        .collect()
}

fn extract_luminance_rgba(pixels: &[rgb::RGBA<u8>]) -> Vec<f32> {
    pixels
        .iter()
        .map(|p| srgb_to_oklab(p.r, p.g, p.b).l)
        .collect()
}

/// Compute local contrast: (L - avg_4_neighbors)², clamped to [0, 0.2].
fn compute_local_contrast(luminance: &[f32], width: usize, height: usize) -> Vec<f32> {
    let mut contrast = vec![0.0f32; luminance.len()];

    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            let center = luminance[idx];

            let mut sum = 0.0f32;
            let mut count = 0u32;

            if x > 0 {
                sum += luminance[idx - 1];
                count += 1;
            }
            if x + 1 < width {
                sum += luminance[idx + 1];
                count += 1;
            }
            if y > 0 {
                sum += luminance[idx - width];
                count += 1;
            }
            if y + 1 < height {
                sum += luminance[idx + width];
                count += 1;
            }

            let avg = if count > 0 {
                sum / count as f32
            } else {
                center
            };
            let diff = center - avg;
            contrast[idx] = (diff * diff).min(0.2);
        }
    }

    contrast
}

/// Min-biased erosion: for each block, take the weighted average of the 4 smallest contrast values.
/// Weights: [0.40, 0.25, 0.20, 0.15] — heavier min-bias for banding protection.
fn erode_to_blocks(
    contrast: &[f32],
    width: usize,
    height: usize,
    block_w: usize,
    block_h: usize,
) -> Vec<f32> {
    const WEIGHTS: [f32; 4] = [0.40, 0.25, 0.20, 0.15];

    let mut blocks = vec![0.0f32; block_w * block_h];

    for by in 0..block_h {
        for bx in 0..block_w {
            // Gather all contrast values in this 4x4 block
            let mut values = Vec::new();
            let y_start = by * 4;
            let x_start = bx * 4;
            let y_end = (y_start + 4).min(height);
            let x_end = (x_start + 4).min(width);

            for y in y_start..y_end {
                for x in x_start..x_end {
                    values.push(contrast[y * width + x]);
                }
            }

            if values.is_empty() {
                continue;
            }

            // Sort ascending to find smallest values
            values.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));

            // Weighted average of up to 4 smallest values
            let n = values.len().min(4);
            let mut weighted_sum = 0.0f32;
            let mut weight_sum = 0.0f32;
            for i in 0..n {
                weighted_sum += values[i] * WEIGHTS[i];
                weight_sum += WEIGHTS[i];
            }

            blocks[by * block_w + bx] = weighted_sum / weight_sum;
        }
    }

    blocks
}

/// Bilinear upscale from block grid to per-pixel resolution.
fn upscale_bilinear(
    blocks: &[f32],
    block_w: usize,
    block_h: usize,
    width: usize,
    height: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; width * height];

    for y in 0..height {
        for x in 0..width {
            // Map pixel center to block grid coordinates
            // Block centers are at (bx * 4 + 2, by * 4 + 2)
            let bx_f = (x as f32 - 2.0) / 4.0;
            let by_f = (y as f32 - 2.0) / 4.0;

            let bx0 = (bx_f.floor() as isize).max(0) as usize;
            let by0 = (by_f.floor() as isize).max(0) as usize;
            let bx1 = (bx0 + 1).min(block_w - 1);
            let by1 = (by0 + 1).min(block_h - 1);

            let fx = (bx_f - bx0 as f32).clamp(0.0, 1.0);
            let fy = (by_f - by0 as f32).clamp(0.0, 1.0);

            let v00 = blocks[by0 * block_w + bx0];
            let v10 = blocks[by0 * block_w + bx1];
            let v01 = blocks[by1 * block_w + bx0];
            let v11 = blocks[by1 * block_w + bx1];

            let top = v00 * (1.0 - fx) + v10 * fx;
            let bot = v01 * (1.0 - fx) + v11 * fx;
            output[y * width + x] = top * (1.0 - fy) + bot * fy;
        }
    }

    output
}

/// Convert masking values to weights.
/// Low contrast (smooth) → high weight, high contrast (texture) → low weight.
/// K is tuned so typical images have mean weight ~0.5.
fn masking_to_weights(masking: &[f32]) -> Vec<f32> {
    // K=4.0 provides softer masking than K=8.0 — less aggressive suppression
    // of dithering in moderately textured regions, better overall quality.
    const K: f32 = 4.0;

    masking
        .iter()
        .map(|&m| {
            let w = 0.1 + 0.9 / (1.0 + K * m.sqrt());
            w.clamp(0.1, 1.0)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat_image(r: u8, g: u8, b: u8, width: usize, height: usize) -> Vec<rgb::RGB<u8>> {
        vec![rgb::RGB { r, g, b }; width * height]
    }

    #[test]
    fn flat_image_high_weights() {
        let pixels = flat_image(128, 128, 128, 16, 16);
        let weights = compute_masking_weights(&pixels, 16, 16);
        assert_eq!(weights.len(), 256);
        // All pixels identical → zero contrast → weight should be ~1.0
        for &w in &weights {
            assert!(w > 0.95, "expected high weight for flat image, got {w}");
        }
    }

    #[test]
    fn checkerboard_low_weights() {
        let mut pixels = Vec::with_capacity(16 * 16);
        for y in 0..16 {
            for x in 0..16 {
                if (x + y) % 2 == 0 {
                    pixels.push(rgb::RGB { r: 0, g: 0, b: 0 });
                } else {
                    pixels.push(rgb::RGB {
                        r: 255,
                        g: 255,
                        b: 255,
                    });
                }
            }
        }
        let weights = compute_masking_weights(&pixels, 16, 16);
        assert_eq!(weights.len(), 256);
        let mean: f32 = weights.iter().sum::<f32>() / weights.len() as f32;
        // High contrast checkerboard → low mean weight
        assert!(
            mean < 0.5,
            "expected low mean weight for checkerboard, got {mean}"
        );
    }

    #[test]
    fn weights_in_valid_range() {
        let mut pixels = Vec::with_capacity(32 * 32);
        for i in 0..(32 * 32) {
            let v = (i % 256) as u8;
            pixels.push(rgb::RGB { r: v, g: v, b: v });
        }
        let weights = compute_masking_weights(&pixels, 32, 32);
        for &w in &weights {
            assert!(
                (0.1..=1.0).contains(&w),
                "weight {w} out of range [0.1, 1.0]"
            );
        }
    }
}
