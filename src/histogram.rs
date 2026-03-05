extern crate alloc;
use alloc::collections::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;

use crate::oklab::{OKLab, OKLabA, srgb_to_oklab};

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
/// Uses adaptive bit depth: 6-bit for small images (more color precision), 5-bit
/// for large images (avoids dominant-color fragmentation on screenshots).
/// Weighted centroids are computed in f64 for accumulation stability.
///
/// For images with many duplicate colors (unique < total/4), deduplicates pixels
/// first to reduce the number of sRGB→OKLab conversions.
pub fn build_histogram(pixels: &[rgb::RGB<u8>], weights: &[f32]) -> Vec<(OKLab, f32)> {
    assert_eq!(pixels.len(), weights.len());

    // Try pixel deduplication for large images with many duplicates
    if pixels.len() >= 65_536
        && let Some(result) = build_histogram_dedup_rgb(pixels, weights)
    {
        return result;
    }

    let labs: Vec<OKLab> = crate::simd::batch_srgb_to_oklab_vec(pixels);

    let bits = if pixels.len() <= 500_000 { 6 } else { 5 };
    build_hist_at_depth(&labs, weights, bits)
}

/// Build histogram from pre-computed OKLab values.
///
/// Skips sRGB→OKLab conversion entirely — uses the provided labs directly.
/// Does not attempt pixel deduplication (labs are already computed).
///
/// When `min_entries > 0`, adaptively increases bit depth (up to 7) to ensure
/// the histogram has at least `min_entries` buckets. This prevents palette
/// underutilization on large images where 5-bit bucketing is too coarse.
///
/// Returns `(histogram, was_bumped)` where `was_bumped` is true if the bit depth
/// was increased beyond the default to meet `min_entries`.
///
/// Only bumps when the coarse histogram is at least 75% of `min_entries`,
/// indicating genuine color diversity lost to bucketing precision. Below that,
/// the image simply doesn't need that many palette entries.
pub fn build_histogram_from_labs(labs: &[crate::oklab::OKLab], weights: &[f32], min_entries: usize) -> (Vec<(crate::oklab::OKLab, f32)>, bool) {
    assert_eq!(labs.len(), weights.len());
    let start_bits = if labs.len() <= 500_000 { 6 } else { 5 };
    let hist = build_hist_at_depth(labs, weights, start_bits);

    if min_entries == 0 || hist.len() >= min_entries || start_bits >= 7 {
        return (hist, false);
    }

    // Only bump if the coarse histogram is at least 75% full — below that,
    // the image genuinely has fewer meaningful color clusters than max_colors.
    if hist.len() * 4 < min_entries * 3 {
        return (hist, false);
    }

    // Try finer bucketing to get at least min_entries
    for bits in (start_bits + 1)..=7 {
        let finer = build_hist_at_depth(labs, weights, bits);
        if finer.len() >= min_entries || bits == 7 {
            return (finer, true);
        }
    }

    (hist, false)
}

/// Attempt RGB pixel deduplication. Returns Some if unique < total/4.
fn build_histogram_dedup_rgb(
    pixels: &[rgb::RGB<u8>],
    weights: &[f32],
) -> Option<Vec<(OKLab, f32)>> {
    // 2MB bitvec: one bit per RGB triplet (2^24 = 16M possible RGB values)
    const BITVEC_SIZE: usize = 1 << 24; // 16,777,216 bits
    let mut seen = vec![0u8; BITVEC_SIZE / 8]; // 2MB
    let mut unique_count = 0usize;

    for p in pixels {
        let key = ((p.r as usize) << 16) | ((p.g as usize) << 8) | p.b as usize;
        let byte_idx = key >> 3;
        let bit_idx = key & 7;
        if seen[byte_idx] & (1u8 << bit_idx) == 0 {
            seen[byte_idx] |= 1u8 << bit_idx;
            unique_count += 1;
        }
    }

    // Only deduplicate if unique colors are < 1/4 of total pixels
    if unique_count >= pixels.len() / 4 {
        return None;
    }

    // Aggregate weights by exact RGB value
    let mut weight_map: BTreeMap<u32, f32> = BTreeMap::new();
    for (p, &w) in pixels.iter().zip(weights.iter()) {
        let key = ((p.r as u32) << 16) | ((p.g as u32) << 8) | p.b as u32;
        *weight_map.entry(key).or_default() += w;
    }

    // Convert only unique colors to OKLab via SIMD batch
    let unique_pixels: Vec<rgb::RGB<u8>> = weight_map
        .keys()
        .map(|&k| rgb::RGB {
            r: (k >> 16) as u8,
            g: (k >> 8) as u8,
            b: k as u8,
        })
        .collect();
    let unique_weights: Vec<f32> = weight_map.into_values().collect();

    let labs = crate::simd::batch_srgb_to_oklab_vec(&unique_pixels);
    let bits = if pixels.len() <= 500_000 { 6 } else { 5 };
    Some(build_hist_at_depth(&labs, &unique_weights, bits))
}

pub(crate) fn build_hist_at_depth(labs: &[OKLab], weights: &[f32], bits: u32) -> Vec<(OKLab, f32)> {
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

    // Collect opaque pixels and their weights
    let mut opaque_pixels: Vec<rgb::RGB<u8>> = Vec::with_capacity(pixels.len());
    let mut opaque_weights: Vec<f32> = Vec::with_capacity(pixels.len());

    for (pixel, &weight) in pixels.iter().zip(weights.iter()) {
        if pixel.a == 0 {
            has_transparent = true;
            continue;
        }
        opaque_pixels.push(rgb::RGB {
            r: pixel.r,
            g: pixel.g,
            b: pixel.b,
        });
        opaque_weights.push(weight);
    }

    // Delegate to the RGB build (which handles dedup internally)
    let entries = build_histogram(&opaque_pixels, &opaque_weights);
    (entries, has_transparent)
}

/// Build a weighted histogram from RGBA pixels with alpha as a quantizable dimension.
///
/// Alpha is quantized to 6 bits (64 levels) for bucketing. Fully transparent pixels
/// (alpha == 0) are excluded and flagged separately.
/// Returns histogram entries as (OKLabA, weight) pairs.
///
/// For images with many duplicate RGBA values (unique < total/4), deduplicates
/// pixels first to reduce sRGB→OKLab conversions.
pub(crate) fn build_histogram_alpha(
    pixels: &[rgb::RGBA<u8>],
    weights: &[f32],
) -> (Vec<(OKLabA, f32)>, bool) {
    assert_eq!(pixels.len(), weights.len());

    // Try RGBA dedup for large images
    if pixels.len() >= 65_536
        && let Some(result) = build_histogram_alpha_dedup(pixels, weights)
    {
        return result;
    }

    build_histogram_alpha_direct(pixels, weights)
}

fn build_histogram_alpha_direct(
    pixels: &[rgb::RGBA<u8>],
    weights: &[f32],
) -> (Vec<(OKLabA, f32)>, bool) {
    let bits: u32 = 5;
    let alpha_bits: u32 = 6; // 64 alpha levels
    let alpha_max = (1u32 << alpha_bits) - 1;
    let alpha_scale = alpha_max as f32;

    let mut has_transparent = false;
    let mut buckets: BTreeMap<u64, AlphaHistEntry> = BTreeMap::new();

    for (pixel, &weight) in pixels.iter().zip(weights.iter()) {
        if pixel.a == 0 {
            has_transparent = true;
            continue;
        }

        let lab = srgb_to_oklab(pixel.r, pixel.g, pixel.b);
        let alpha_f = pixel.a as f32 / 255.0;
        let color_key = quantize_key(lab, bits);
        let alpha_bin = ((alpha_f * alpha_scale).round() as u32).min(alpha_max);
        let key = (color_key as u64) << alpha_bits | alpha_bin as u64;

        let w64 = weight as f64;
        buckets
            .entry(key)
            .and_modify(|e| {
                e.l_sum += lab.l as f64 * w64;
                e.a_sum += lab.a as f64 * w64;
                e.b_sum += lab.b as f64 * w64;
                e.alpha_sum += alpha_f as f64 * w64;
                e.weight += w64;
            })
            .or_insert_with(|| AlphaHistEntry {
                l_sum: lab.l as f64 * w64,
                a_sum: lab.a as f64 * w64,
                b_sum: lab.b as f64 * w64,
                alpha_sum: alpha_f as f64 * w64,
                weight: w64,
            });
    }

    let entries = buckets
        .into_values()
        .map(|e| {
            if e.weight < 1e-10 {
                (OKLabA::new(0.0, 0.0, 0.0, 0.0), 0.0)
            } else {
                let laba = OKLabA::new(
                    (e.l_sum / e.weight) as f32,
                    (e.a_sum / e.weight) as f32,
                    (e.b_sum / e.weight) as f32,
                    (e.alpha_sum / e.weight) as f32,
                );
                (laba, e.weight as f32)
            }
        })
        .collect();

    (entries, has_transparent)
}

/// RGBA dedup: aggregate weights by exact RGBA value using BTreeMap.
fn build_histogram_alpha_dedup(
    pixels: &[rgb::RGBA<u8>],
    weights: &[f32],
) -> Option<(Vec<(OKLabA, f32)>, bool)> {
    let mut has_transparent = false;

    // Count unique RGBA values and aggregate weights in one pass
    let mut weight_map: BTreeMap<u32, f32> = BTreeMap::new();
    for (p, &w) in pixels.iter().zip(weights.iter()) {
        if p.a == 0 {
            has_transparent = true;
            continue;
        }
        let key = ((p.r as u32) << 24) | ((p.g as u32) << 16) | ((p.b as u32) << 8) | p.a as u32;
        *weight_map.entry(key).or_default() += w;
    }

    // Only proceed with dedup if unique colors are < 1/4 of total opaque pixels
    let opaque_count = pixels.len() - if has_transparent { 1 } else { 0 };
    if weight_map.len() >= opaque_count / 4 {
        return None;
    }

    // Build histogram from deduplicated entries
    let bits: u32 = 5;
    let alpha_bits: u32 = 6;
    let alpha_max = (1u32 << alpha_bits) - 1;
    let alpha_scale = alpha_max as f32;

    let mut buckets: BTreeMap<u64, AlphaHistEntry> = BTreeMap::new();

    for (&rgba_key, &weight) in &weight_map {
        let r = (rgba_key >> 24) as u8;
        let g = (rgba_key >> 16) as u8;
        let b = (rgba_key >> 8) as u8;
        let a = rgba_key as u8;

        let lab = srgb_to_oklab(r, g, b);
        let alpha_f = a as f32 / 255.0;
        let color_key = quantize_key(lab, bits);
        let alpha_bin = ((alpha_f * alpha_scale).round() as u32).min(alpha_max);
        let key = (color_key as u64) << alpha_bits | alpha_bin as u64;

        let w64 = weight as f64;
        buckets
            .entry(key)
            .and_modify(|e| {
                e.l_sum += lab.l as f64 * w64;
                e.a_sum += lab.a as f64 * w64;
                e.b_sum += lab.b as f64 * w64;
                e.alpha_sum += alpha_f as f64 * w64;
                e.weight += w64;
            })
            .or_insert_with(|| AlphaHistEntry {
                l_sum: lab.l as f64 * w64,
                a_sum: lab.a as f64 * w64,
                b_sum: lab.b as f64 * w64,
                alpha_sum: alpha_f as f64 * w64,
                weight: w64,
            });
    }

    let entries = buckets
        .into_values()
        .map(|e| {
            if e.weight < 1e-10 {
                (OKLabA::new(0.0, 0.0, 0.0, 0.0), 0.0)
            } else {
                let laba = OKLabA::new(
                    (e.l_sum / e.weight) as f32,
                    (e.a_sum / e.weight) as f32,
                    (e.b_sum / e.weight) as f32,
                    (e.alpha_sum / e.weight) as f32,
                );
                (laba, e.weight as f32)
            }
        })
        .collect();

    Some((entries, has_transparent))
}

/// Histogram entry with alpha accumulation.
#[derive(Debug, Clone)]
struct AlphaHistEntry {
    l_sum: f64,
    a_sum: f64,
    b_sum: f64,
    alpha_sum: f64,
    weight: f64,
}

/// Detect if an RGB image uses at most `max_colors` unique colors.
/// Returns the exact palette if so, `None` if more colors exist.
/// Uses early exit — scans until `max_colors + 1` unique colors found.
pub(crate) fn detect_exact_palette(
    pixels: &[rgb::RGB<u8>],
    max_colors: usize,
) -> Option<Vec<rgb::RGB<u8>>> {
    let mut seen = alloc::collections::BTreeSet::new();
    for p in pixels {
        let key = (p.r as u32) << 16 | (p.g as u32) << 8 | p.b as u32;
        seen.insert(key);
        if seen.len() > max_colors {
            return None;
        }
    }
    Some(
        seen.into_iter()
            .map(|k| rgb::RGB {
                r: (k >> 16) as u8,
                g: (k >> 8) as u8,
                b: k as u8,
            })
            .collect(),
    )
}

/// Detect if an RGBA image uses at most `max_colors` unique colors (including alpha).
/// Returns the exact palette and whether any fully-transparent pixels exist.
pub(crate) fn detect_exact_palette_rgba(
    pixels: &[rgb::RGBA<u8>],
    max_colors: usize,
) -> Option<(Vec<rgb::RGBA<u8>>, bool)> {
    let mut seen = alloc::collections::BTreeSet::new();
    let mut has_transparent = false;
    for p in pixels {
        if p.a == 0 {
            has_transparent = true;
            continue; // transparent pixels don't count toward palette
        }
        let key = (p.r as u32) << 24 | (p.g as u32) << 16 | (p.b as u32) << 8 | p.a as u32;
        seen.insert(key);
        if seen.len() > max_colors {
            return None;
        }
    }
    let colors = seen
        .into_iter()
        .map(|k| rgb::RGBA {
            r: (k >> 24) as u8,
            g: (k >> 16) as u8,
            b: (k >> 8) as u8,
            a: k as u8,
        })
        .collect();
    Some((colors, has_transparent))
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
        let pixels = vec![
            rgb::RGB {
                r: 100,
                g: 150,
                b: 200
            };
            10000
        ];
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
