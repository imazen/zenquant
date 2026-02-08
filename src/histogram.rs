extern crate alloc;
use alloc::vec::Vec;

use crate::oklab::{OKLab, srgb_to_oklab};

/// A histogram entry: accumulated color, weight, and count for a quantized bucket.
#[derive(Debug, Clone)]
pub struct HistEntry {
    /// Sum of OKLab L values (weighted)
    pub l_sum: f32,
    /// Sum of OKLab a values (weighted)
    pub a_sum: f32,
    /// Sum of OKLab b values (weighted)
    pub b_sum: f32,
    /// Total AQ weight accumulated
    pub weight: f32,
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
            self.l_sum / self.weight,
            self.a_sum / self.weight,
            self.b_sum / self.weight,
        )
    }
}

/// Quantize an OKLab value to a 12-bit bucket key (4 bits per channel).
fn quantize_key(lab: OKLab) -> u16 {
    // L is roughly [0, 1], a and b are roughly [-0.4, 0.4]
    let l_bin = ((lab.l * 15.0).round() as u16).min(15);
    let a_bin = (((lab.a + 0.4) * (15.0 / 0.8)).round() as u16).min(15);
    let b_bin = (((lab.b + 0.4) * (15.0 / 0.8)).round() as u16).min(15);
    (l_bin << 8) | (a_bin << 4) | b_bin
}

/// Build a weighted color histogram from RGB pixels and per-pixel AQ weights.
pub fn build_histogram(pixels: &[rgb::RGB<u8>], weights: &[f32]) -> Vec<(OKLab, f32)> {
    assert_eq!(pixels.len(), weights.len());

    // Use a simple array for 4096 buckets (12-bit keys)
    let mut buckets: Vec<Option<HistEntry>> = (0..4096).map(|_| None).collect();

    for (pixel, &weight) in pixels.iter().zip(weights.iter()) {
        let lab = srgb_to_oklab(pixel.r, pixel.g, pixel.b);
        let key = quantize_key(lab) as usize;

        match &mut buckets[key] {
            Some(entry) => {
                entry.l_sum += lab.l * weight;
                entry.a_sum += lab.a * weight;
                entry.b_sum += lab.b * weight;
                entry.weight += weight;
                entry.count += 1;
            }
            slot @ None => {
                *slot = Some(HistEntry {
                    l_sum: lab.l * weight,
                    a_sum: lab.a * weight,
                    b_sum: lab.b * weight,
                    weight,
                    count: 1,
                });
            }
        }
    }

    buckets
        .into_iter()
        .flatten()
        .map(|e| (e.centroid(), e.weight))
        .collect()
}

/// Build a weighted color histogram from RGBA pixels and per-pixel AQ weights.
/// Fully transparent pixels (alpha == 0) are skipped.
pub fn build_histogram_rgba(
    pixels: &[rgb::RGBA<u8>],
    weights: &[f32],
) -> (Vec<(OKLab, f32)>, bool) {
    assert_eq!(pixels.len(), weights.len());

    let mut buckets: Vec<Option<HistEntry>> = (0..4096).map(|_| None).collect();
    let mut has_transparent = false;

    for (pixel, &weight) in pixels.iter().zip(weights.iter()) {
        if pixel.a == 0 {
            has_transparent = true;
            continue;
        }

        let lab = srgb_to_oklab(pixel.r, pixel.g, pixel.b);
        let key = quantize_key(lab) as usize;

        match &mut buckets[key] {
            Some(entry) => {
                entry.l_sum += lab.l * weight;
                entry.a_sum += lab.a * weight;
                entry.b_sum += lab.b * weight;
                entry.weight += weight;
                entry.count += 1;
            }
            slot @ None => {
                *slot = Some(HistEntry {
                    l_sum: lab.l * weight,
                    a_sum: lab.a * weight,
                    b_sum: lab.b * weight,
                    weight,
                    count: 1,
                });
            }
        }
    }

    let entries: Vec<(OKLab, f32)> = buckets
        .into_iter()
        .flatten()
        .map(|e| (e.centroid(), e.weight))
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
        assert!(
            hist.len() >= 2,
            "expected at least 2 buckets, got {}",
            hist.len()
        );
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
