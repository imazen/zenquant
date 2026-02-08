extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

use crate::oklab::srgb_to_oklab;
use crate::palette::Palette;

/// How aggressively to prefer extending runs over quality.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RunPriority {
    /// Pure quality — no run bias.
    Quality,
    /// Balance quality and run length.
    Balanced,
    /// Aggressive run extension for better compression.
    Compression,
}

impl RunPriority {
    /// Bias factor: 0.0 = quality only, higher = prefer runs.
    pub fn bias(self) -> f32 {
        match self {
            Self::Quality => 0.0,
            Self::Balanced => 0.3,
            Self::Compression => 0.7,
        }
    }
}

/// Map RGB pixels to palette indices with run-biased selection.
///
/// The AQ weights modulate the run-extension threshold:
/// - Smooth regions (high weight): lower threshold → prefer quality
/// - Textured regions (low weight): higher threshold → prefer runs
pub fn remap_pixels(
    pixels: &[rgb::RGB<u8>],
    weights: &[f32],
    palette: &Palette,
    run_priority: RunPriority,
) -> Vec<u8> {
    let bias = run_priority.bias();
    let k = if bias > 0.0 { 4 } else { 1 };

    let mut indices = vec![0u8; pixels.len()];
    let mut prev_index: Option<u8> = None;

    for (i, pixel) in pixels.iter().enumerate() {
        let lab = srgb_to_oklab(pixel.r, pixel.g, pixel.b);

        if bias == 0.0 {
            // Pure quality mode — just find nearest
            indices[i] = palette.nearest(lab);
        } else {
            let candidates = palette.k_nearest(lab, k);
            let best = candidates[0];
            let best_dist = palette.distance_sq(lab, best);

            // AQ-modulated threshold: smooth (high weight) → low threshold,
            // textured (low weight) → high threshold
            let w = weights[i];
            let max_acceptable = best_dist * 2.0; // base error budget
            let threshold = bias * max_acceptable * (1.1 - w);

            if let Some(prev) = prev_index {
                if candidates.contains(&prev) {
                    let prev_dist = palette.distance_sq(lab, prev);
                    if prev_dist < best_dist + threshold {
                        indices[i] = prev;
                        prev_index = Some(prev);
                        continue;
                    }
                }
            }

            indices[i] = best;
        }

        prev_index = Some(indices[i]);
    }

    indices
}

/// Map RGBA pixels to palette indices. Transparent pixels get the transparent index.
pub fn remap_pixels_rgba(
    pixels: &[rgb::RGBA<u8>],
    weights: &[f32],
    palette: &Palette,
    run_priority: RunPriority,
) -> Vec<u8> {
    let bias = run_priority.bias();
    let k = if bias > 0.0 { 4 } else { 1 };
    let transparent_idx = palette.transparent_index().unwrap_or(0);

    let mut indices = vec![0u8; pixels.len()];
    let mut prev_index: Option<u8> = None;

    for (i, pixel) in pixels.iter().enumerate() {
        if pixel.a == 0 {
            indices[i] = transparent_idx;
            prev_index = Some(transparent_idx);
            continue;
        }

        let lab = srgb_to_oklab(pixel.r, pixel.g, pixel.b);

        if bias == 0.0 {
            indices[i] = palette.nearest(lab);
        } else {
            let candidates = palette.k_nearest(lab, k);
            let best = candidates[0];
            let best_dist = palette.distance_sq(lab, best);

            let w = weights[i];
            let max_acceptable = best_dist * 2.0;
            let threshold = bias * max_acceptable * (1.1 - w);

            if let Some(prev) = prev_index {
                if prev != transparent_idx && candidates.contains(&prev) {
                    let prev_dist = palette.distance_sq(lab, prev);
                    if prev_dist < best_dist + threshold {
                        indices[i] = prev;
                        prev_index = Some(prev);
                        continue;
                    }
                }
            }

            indices[i] = best;
        }

        prev_index = Some(indices[i]);
    }

    indices
}

/// Count the number of runs in an index stream.
pub fn count_runs(indices: &[u8]) -> usize {
    if indices.is_empty() {
        return 0;
    }
    let mut runs = 1;
    for i in 1..indices.len() {
        if indices[i] != indices[i - 1] {
            runs += 1;
        }
    }
    runs
}

/// Average run length.
pub fn average_run_length(indices: &[u8]) -> f32 {
    if indices.is_empty() {
        return 0.0;
    }
    indices.len() as f32 / count_runs(indices) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_palette() -> Palette {
        use crate::oklab::srgb_to_oklab;
        let centroids = vec![
            srgb_to_oklab(0, 0, 0),
            srgb_to_oklab(85, 85, 85),
            srgb_to_oklab(170, 170, 170),
            srgb_to_oklab(255, 255, 255),
        ];
        Palette::from_centroids(centroids, false)
    }

    #[test]
    fn quality_mode_finds_nearest() {
        let palette = make_test_palette();
        let pixels = vec![
            rgb::RGB { r: 0, g: 0, b: 0 },
            rgb::RGB {
                r: 255,
                g: 255,
                b: 255,
            },
        ];
        let weights = vec![1.0; 2];
        let indices = remap_pixels(&pixels, &weights, &palette, RunPriority::Quality);
        assert_eq!(indices.len(), 2);
        // Black and white should map to different indices
        assert_ne!(indices[0], indices[1]);
    }

    #[test]
    fn compression_mode_extends_runs() {
        let palette = make_test_palette();
        // Sequence of very similar colors that straddle two palette entries
        let mut pixels = Vec::new();
        for i in 0..20 {
            // Oscillate between 84 and 86 — straddles the 85/170 boundary area
            let v = if i % 2 == 0 { 84 } else { 86 };
            pixels.push(rgb::RGB { r: v, g: v, b: v });
        }
        let weights = vec![0.2; 20]; // low weight = textured

        let quality_indices = remap_pixels(&pixels, &weights, &palette, RunPriority::Quality);
        let compression_indices =
            remap_pixels(&pixels, &weights, &palette, RunPriority::Compression);

        let quality_runs = count_runs(&quality_indices);
        let compression_runs = count_runs(&compression_indices);

        // Compression mode should produce fewer runs (longer runs)
        assert!(
            compression_runs <= quality_runs,
            "compression mode should have ≤ runs: quality={quality_runs}, compression={compression_runs}"
        );
    }

    #[test]
    fn count_runs_basic() {
        assert_eq!(count_runs(&[]), 0);
        assert_eq!(count_runs(&[1]), 1);
        assert_eq!(count_runs(&[1, 1, 1]), 1);
        assert_eq!(count_runs(&[1, 2, 3]), 3);
        assert_eq!(count_runs(&[1, 1, 2, 2, 3, 3]), 3);
    }
}
