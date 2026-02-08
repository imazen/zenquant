extern crate alloc;
use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use crate::oklab::{srgb_to_oklab, OKLab};

/// A histogram entry: accumulated color, weight, and count for a unique RGB color.
#[derive(Debug, Clone)]
pub struct HistEntry {
    /// OKLab value for this unique color
    pub lab: OKLab,
    /// Total AQ weight accumulated
    pub weight: f32,
    /// Number of pixels with this exact color
    pub count: u32,
}

/// Build a weighted color histogram from RGB pixels and per-pixel AQ weights.
///
/// Each unique sRGB color becomes one histogram entry — no quantization loss.
/// Returns (OKLab centroid, accumulated weight) pairs for median cut.
pub fn build_histogram(pixels: &[rgb::RGB<u8>], weights: &[f32]) -> Vec<(OKLab, f32)> {
    assert_eq!(pixels.len(), weights.len());

    // Key on exact RGB triple — no information loss.
    // Pack into u32: (r << 16) | (g << 8) | b
    let mut buckets: BTreeMap<u32, HistEntry> = BTreeMap::new();

    for (pixel, &weight) in pixels.iter().zip(weights.iter()) {
        let key = (pixel.r as u32) << 16 | (pixel.g as u32) << 8 | pixel.b as u32;

        buckets
            .entry(key)
            .and_modify(|e| {
                e.weight += weight;
                e.count += 1;
            })
            .or_insert_with(|| HistEntry {
                lab: srgb_to_oklab(pixel.r, pixel.g, pixel.b),
                weight,
                count: 1,
            });
    }

    buckets
        .into_values()
        .map(|e| (e.lab, e.weight))
        .collect()
}

/// Build a weighted color histogram from RGBA pixels and per-pixel AQ weights.
/// Fully transparent pixels (alpha == 0) are skipped.
pub fn build_histogram_rgba(
    pixels: &[rgb::RGBA<u8>],
    weights: &[f32],
) -> (Vec<(OKLab, f32)>, bool) {
    assert_eq!(pixels.len(), weights.len());

    let mut buckets: BTreeMap<u32, HistEntry> = BTreeMap::new();
    let mut has_transparent = false;

    for (pixel, &weight) in pixels.iter().zip(weights.iter()) {
        if pixel.a == 0 {
            has_transparent = true;
            continue;
        }

        let key = (pixel.r as u32) << 16 | (pixel.g as u32) << 8 | pixel.b as u32;

        buckets
            .entry(key)
            .and_modify(|e| {
                e.weight += weight;
                e.count += 1;
            })
            .or_insert_with(|| HistEntry {
                lab: srgb_to_oklab(pixel.r, pixel.g, pixel.b),
                weight,
                count: 1,
            });
    }

    let entries: Vec<(OKLab, f32)> = buckets
        .into_values()
        .map(|e| (e.lab, e.weight))
        .collect();

    (entries, has_transparent)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_color_one_bucket() {
        let pixels = vec![
            rgb::RGB {
                r: 128,
                g: 128,
                b: 128
            };
            100
        ];
        let weights = vec![1.0; 100];
        let hist = build_histogram(&pixels, &weights);
        assert_eq!(hist.len(), 1);
        assert!((hist[0].1 - 100.0).abs() < 0.01);
    }

    #[test]
    fn weights_accumulate() {
        let pixels = vec![
            rgb::RGB {
                r: 128,
                g: 128,
                b: 128
            };
            10
        ];
        let weights = vec![0.5; 10];
        let hist = build_histogram(&pixels, &weights);
        assert_eq!(hist.len(), 1);
        assert!((hist[0].1 - 5.0).abs() < 0.01);
    }

    #[test]
    fn distinct_colors_separate_buckets() {
        let pixels = vec![
            rgb::RGB { r: 0, g: 0, b: 0 },
            rgb::RGB {
                r: 255,
                g: 255,
                b: 255,
            },
        ];
        let weights = vec![1.0; 2];
        let hist = build_histogram(&pixels, &weights);
        assert_eq!(hist.len(), 2);
    }

    #[test]
    fn similar_colors_stay_separate() {
        // Adjacent RGB values should NOT be merged (unlike old 4-bit quantization)
        let pixels = vec![
            rgb::RGB {
                r: 128,
                g: 128,
                b: 128,
            },
            rgb::RGB {
                r: 129,
                g: 128,
                b: 128,
            },
        ];
        let weights = vec![1.0; 2];
        let hist = build_histogram(&pixels, &weights);
        assert_eq!(hist.len(), 2, "similar colors must stay in separate buckets");
    }

    #[test]
    fn rgba_skips_transparent() {
        let pixels = vec![
            rgb::RGBA {
                r: 128,
                g: 128,
                b: 128,
                a: 255,
            },
            rgb::RGBA {
                r: 0,
                g: 0,
                b: 0,
                a: 0,
            },
        ];
        let weights = vec![1.0; 2];
        let (hist, has_transparent) = build_histogram_rgba(&pixels, &weights);
        assert!(has_transparent);
        assert_eq!(hist.len(), 1);
    }
}
