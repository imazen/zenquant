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

/// Quantize an OKLab value to a bucket key at the given bit depth (4, 5, or 6 bits per channel).
fn quantize_key(lab: OKLab, bits: u32) -> u32 {
    let max_val = (1u32 << bits) - 1;
    let scale = max_val as f32;
    let l_bin = ((lab.l * scale).round() as u32).min(max_val);
    let a_bin = (((lab.a + 0.4) * (scale / 0.8)).round() as u32).min(max_val);
    let b_bin = (((lab.b + 0.4) * (scale / 0.8)).round() as u32).min(max_val);
    (l_bin << (bits * 2)) | (a_bin << bits) | b_bin
}

/// Build a weighted color histogram from RGB pixels and per-pixel AQ weights.
///
/// Uses adaptive bit depth: 5-bit for small images (more color precision), 4-bit
/// for large images (avoids dominant-color fragmentation on screenshots).
/// Weighted centroids are computed in f64 for accumulation stability.
pub fn build_histogram(pixels: &[rgb::RGB<u8>], weights: &[f32]) -> Vec<(OKLab, f32)> {
    assert_eq!(pixels.len(), weights.len());

    let labs: Vec<OKLab> = pixels
        .iter()
        .map(|p| srgb_to_oklab(p.r, p.g, p.b))
        .collect();

    let bits = if pixels.len() <= 500_000 { 6 } else { 4 };
    build_hist_at_depth(&labs, weights, bits)
}

fn build_hist_at_depth(labs: &[OKLab], weights: &[f32], bits: u32) -> Vec<(OKLab, f32)> {
    let mut buckets: BTreeMap<u32, HistEntry> = BTreeMap::new();

    for (lab, &weight) in labs.iter().zip(weights.iter()) {
        let key = quantize_key(*lab, bits);
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

    let mut has_transparent = false;
    let mut labs = Vec::with_capacity(pixels.len());
    let mut opaque_weights = Vec::with_capacity(pixels.len());

    for (pixel, &weight) in pixels.iter().zip(weights.iter()) {
        if pixel.a == 0 {
            has_transparent = true;
            continue;
        }
        labs.push(srgb_to_oklab(pixel.r, pixel.g, pixel.b));
        opaque_weights.push(weight);
    }

    let bits = if pixels.len() <= 500_000 { 5 } else { 4 };
    let entries = build_hist_at_depth(&labs, &opaque_weights, bits);
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
