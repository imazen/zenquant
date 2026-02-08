extern crate alloc;
use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use crate::oklab::{srgb_to_oklab, OKLab};

/// A histogram entry: accumulated color, weight, and count for a quantized bucket.
#[derive(Debug, Clone)]
pub struct HistEntry {
    /// Weighted sum of OKLab L values
    pub l_sum: f64,
    /// Weighted sum of OKLab a values
    pub a_sum: f64,
    /// Weighted sum of OKLab b values
    pub b_sum: f64,
    /// Total AQ weight accumulated
    pub weight: f64,
    /// Number of pixels in this bucket
    pub count: u32,
}

impl HistEntry {
    /// Compute the weighted centroid of this bucket.
    pub fn centroid(&self) -> OKLab {
        if self.weight < 1e-10 {
            return OKLab::new(0.0, 0.0, 0.0);
        }
        OKLab::new(
            (self.l_sum / self.weight) as f32,
            (self.a_sum / self.weight) as f32,
            (self.b_sum / self.weight) as f32,
        )
    }
}

/// Quantize an OKLab value to an 18-bit bucket key (6 bits per channel).
///
/// 6 bits per channel → 262,144 possible buckets. This is fine-grained enough
/// to separate visually distinct colors while keeping the histogram manageable
/// for median cut + k-means (typically a few thousand active entries).
fn quantize_key(lab: OKLab) -> u32 {
    // L is roughly [0, 1], a and b are roughly [-0.4, 0.4]
    let l_bin = ((lab.l * 63.0).round() as u32).min(63);
    let a_bin = (((lab.a + 0.4) * (63.0 / 0.8)).round() as u32).min(63);
    let b_bin = (((lab.b + 0.4) * (63.0 / 0.8)).round() as u32).min(63);
    (l_bin << 12) | (a_bin << 6) | b_bin
}

/// Build a weighted color histogram from RGB pixels and per-pixel AQ weights.
///
/// Colors are quantized to 6-bit-per-channel OKLab buckets (262K possible bins,
/// typically a few thousand active). Weighted centroids are computed in f64 for
/// accumulation stability.
pub fn build_histogram(pixels: &[rgb::RGB<u8>], weights: &[f32]) -> Vec<(OKLab, f32)> {
    assert_eq!(pixels.len(), weights.len());

    let mut buckets: BTreeMap<u32, HistEntry> = BTreeMap::new();

    for (pixel, &weight) in pixels.iter().zip(weights.iter()) {
        let lab = srgb_to_oklab(pixel.r, pixel.g, pixel.b);
        let key = quantize_key(lab);
        let w64 = weight as f64;

        buckets
            .entry(key)
            .and_modify(|e| {
                e.l_sum += lab.l as f64 * w64;
                e.a_sum += lab.a as f64 * w64;
                e.b_sum += lab.b as f64 * w64;
                e.weight += w64;
                e.count += 1;
            })
            .or_insert_with(|| HistEntry {
                l_sum: lab.l as f64 * w64,
                a_sum: lab.a as f64 * w64,
                b_sum: lab.b as f64 * w64,
                weight: w64,
                count: 1,
            });
    }

    buckets
        .into_values()
        .map(|e| (e.centroid(), e.weight as f32))
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

        let lab = srgb_to_oklab(pixel.r, pixel.g, pixel.b);
        let key = quantize_key(lab);
        let w64 = weight as f64;

        buckets
            .entry(key)
            .and_modify(|e| {
                e.l_sum += lab.l as f64 * w64;
                e.a_sum += lab.a as f64 * w64;
                e.b_sum += lab.b as f64 * w64;
                e.weight += w64;
                e.count += 1;
            })
            .or_insert_with(|| HistEntry {
                l_sum: lab.l as f64 * w64,
                a_sum: lab.a as f64 * w64,
                b_sum: lab.b as f64 * w64,
                weight: w64,
                count: 1,
            });
    }

    let entries: Vec<(OKLab, f32)> = buckets
        .into_values()
        .map(|e| (e.centroid(), e.weight as f32))
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

    #[test]
    fn centroid_precision() {
        // Many pixels of the same color — centroid should be close to the input
        let lab = srgb_to_oklab(100, 150, 200);
        let pixels = vec![rgb::RGB { r: 100, g: 150, b: 200 }; 10000];
        let weights = vec![1.0; 10000];
        let hist = build_histogram(&pixels, &weights);
        assert_eq!(hist.len(), 1);
        let centroid = hist[0].0;
        assert!(
            (centroid.l - lab.l).abs() < 0.01,
            "L mismatch: {} vs {}",
            centroid.l,
            lab.l
        );
        assert!(
            (centroid.a - lab.a).abs() < 0.01,
            "a mismatch: {} vs {}",
            centroid.a,
            lab.a
        );
        assert!(
            (centroid.b - lab.b).abs() < 0.01,
            "b mismatch: {} vs {}",
            centroid.b,
            lab.b
        );
    }
}
