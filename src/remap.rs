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

            if let Some(prev) = prev_index.filter(|p| candidates.contains(p)) {
                let prev_dist = palette.distance_sq(lab, prev);
                if prev_dist < best_dist + threshold {
                    indices[i] = prev;
                    prev_index = Some(prev);
                    continue;
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

            if let Some(prev) =
                prev_index.filter(|&p| p != transparent_idx && candidates.contains(&p))
            {
                let prev_dist = palette.distance_sq(lab, prev);
                if prev_dist < best_dist + threshold {
                    indices[i] = prev;
                    prev_index = Some(prev);
                    continue;
                }
            }

            indices[i] = best;
        }

        prev_index = Some(indices[i]);
    }

    indices
}

/// Per-scanline Viterbi DP optimization of palette indices.
///
/// After dithering produces initial indices, this post-pass finds the globally
/// optimal index sequence per row that balances quality (color accuracy) and
/// compression (run-length extension). This is analogous to trellis quantization
/// in video/JPEG codecs.
///
/// `lambda` controls the quality-vs-compression tradeoff:
/// - 0.0: pure quality (no run extension)
/// - 0.003: balanced (modest run extension in textured areas)
/// - 0.01: aggressive compression (long runs where masked by texture)
pub fn viterbi_refine(
    pixels: &[rgb::RGB<u8>],
    width: usize,
    height: usize,
    weights: &[f32],
    palette: &Palette,
    indices: &mut [u8],
    lambda: f32,
) {
    if width <= 1 || lambda <= 0.0 {
        return;
    }

    const K: usize = 4;

    // Pre-allocate buffers reused across scanlines
    let mut bufs = ViterbiBufs::new(width);

    for y in 0..height {
        let row_start = y * width;
        let row = &pixels[row_start..row_start + width];
        let row_weights = &weights[row_start..row_start + width];
        let row_indices = &mut indices[row_start..row_start + width];

        viterbi_scanline(row, row_weights, palette, row_indices, lambda, K, &mut bufs);
    }
}

/// Per-scanline Viterbi DP for RGBA images. Transparent pixels are skipped.
pub fn viterbi_refine_rgba(
    pixels: &[rgb::RGBA<u8>],
    width: usize,
    height: usize,
    weights: &[f32],
    palette: &Palette,
    indices: &mut [u8],
    lambda: f32,
) {
    if width <= 1 || lambda <= 0.0 {
        return;
    }

    const K: usize = 4;
    let transparent_idx = palette.transparent_index().unwrap_or(0);

    // Pre-allocate buffers reused across scanlines/segments
    let mut bufs = ViterbiBufs::new(width);

    for y in 0..height {
        let row_start = y * width;

        // Find contiguous opaque segments within the row and run Viterbi on each
        let mut seg_start = None;
        for x in 0..=width {
            let is_transparent = x < width && pixels[row_start + x].a == 0;
            let past_end = x == width;

            if past_end || is_transparent {
                // End of an opaque segment
                if let Some(start) = seg_start {
                    let seg_len = x - start;
                    if seg_len > 1 {
                        // Build RGB view for the opaque segment
                        let seg_pixels: Vec<rgb::RGB<u8>> = (start..x)
                            .map(|sx| {
                                let p = &pixels[row_start + sx];
                                rgb::RGB {
                                    r: p.r,
                                    g: p.g,
                                    b: p.b,
                                }
                            })
                            .collect();
                        let seg_weights = &weights[row_start + start..row_start + x];
                        let seg_indices = &mut indices[row_start + start..row_start + x];

                        bufs.ensure_capacity(seg_len);
                        viterbi_scanline(
                            &seg_pixels,
                            seg_weights,
                            palette,
                            seg_indices,
                            lambda,
                            K,
                            &mut bufs,
                        );
                    }
                    seg_start = None;
                }

                if is_transparent {
                    indices[row_start + x] = transparent_idx;
                }
            } else if seg_start.is_none() {
                seg_start = Some(x);
            }
        }
    }
}

/// Pre-allocated buffers for Viterbi scanline processing.
/// Reused across scanlines to avoid per-scanline allocation.
struct ViterbiBufs {
    candidates: Vec<[u8; 5]>,
    cand_counts: Vec<u8>,
    quality_costs: Vec<[f32; 5]>,
    backptrs: Vec<[u8; 5]>,
}

impl ViterbiBufs {
    fn new(capacity: usize) -> Self {
        Self {
            candidates: vec![[0u8; 5]; capacity],
            cand_counts: vec![0; capacity],
            quality_costs: vec![[0.0f32; 5]; capacity],
            backptrs: vec![[0u8; 5]; capacity],
        }
    }

    fn ensure_capacity(&mut self, n: usize) {
        if self.candidates.len() < n {
            self.candidates.resize(n, [0u8; 5]);
            self.cand_counts.resize(n, 0);
            self.quality_costs.resize(n, [0.0f32; 5]);
            self.backptrs.resize(n, [0u8; 5]);
        }
    }
}

/// Core Viterbi DP for a single scanline (or opaque segment).
///
/// For each pixel, we consider K candidate palette indices. The DP finds the
/// sequence of candidates that minimizes total cost = sum of quality costs +
/// transition costs.
///
/// Quality cost uses squared distance, which gives Viterbi a strong bias
/// toward preserving the original dithered choice (it's always a candidate
/// and usually the cheapest). Only when extending a run saves significant
/// transition cost will Viterbi override the dithered index.
fn viterbi_scanline(
    pixels: &[rgb::RGB<u8>],
    weights: &[f32],
    palette: &Palette,
    indices: &mut [u8],
    lambda: f32,
    k: usize,
    bufs: &mut ViterbiBufs,
) {
    let n = pixels.len();
    if n == 0 {
        return;
    }

    bufs.ensure_capacity(n);
    let max_cands = k + 1;
    let candidates = &mut bufs.candidates[..n];
    let cand_counts = &mut bufs.cand_counts[..n];
    let quality_costs = &mut bufs.quality_costs[..n];

    let mut knn_buf = [0u8; 4]; // K=4

    let has_cache = palette.has_nn_cache();

    for x in 0..n {
        let lab = srgb_to_oklab(pixels[x].r, pixels[x].g, pixels[x].b);
        let found = if has_cache {
            let seed = palette.nearest_cached(pixels[x].r, pixels[x].g, pixels[x].b);
            palette.k_nearest_seeded(lab, seed, &mut knn_buf[..k])
        } else {
            palette.k_nearest_into(lab, &mut knn_buf[..k])
        };

        // Copy candidates to fixed array
        let mut count = 0usize;
        let current_idx = indices[x];
        let mut dith_pos = None;

        for &k in &knn_buf[..found] {
            candidates[x][count] = k;
            if k == current_idx {
                dith_pos = Some(count);
            }
            count += 1;
        }

        // Ensure the current dithered index is always a candidate
        if dith_pos.is_none() {
            if count >= max_cands {
                count -= 1; // drop worst
            }
            candidates[x][count] = current_idx;
            dith_pos = Some(count);
            count += 1;
        }

        let dith_pos = dith_pos.unwrap();
        cand_counts[x] = count as u8;

        // Compute quality costs
        for j in 0..count {
            quality_costs[x][j] = if j == dith_pos {
                // Zero cost for keeping the dithered choice
                0.0
            } else {
                weights[x] * palette.distance_sq(lab, candidates[x][j])
            };
        }
    }

    // DP tables — fixed-size arrays, no per-pixel allocation
    // dp[j] = minimum total cost to reach candidate j at the current pixel
    // backptrs[x][j] = which candidate index (into candidates[x-1]) was the predecessor
    let k0 = cand_counts[0] as usize;
    let mut dp = [f32::MAX; 5];
    dp[..k0].copy_from_slice(&quality_costs[0][..k0]);
    let backptrs = &mut bufs.backptrs[..n];

    // Forward pass
    for x in 1..n {
        let k_cur = cand_counts[x] as usize;
        let k_prev = cand_counts[x - 1] as usize;
        let mut new_dp = [f32::MAX; 5];
        let mut bp = [0u8; 5];

        let w = weights[x];
        // Transition cost: lambda * (1 - weight) when indices differ.
        // Smooth regions (high weight) → low transition cost → keep dither patterns.
        // Textured regions (low weight) → high transition cost → favor runs.
        let trans_cost = lambda * (1.0 - w);

        for j in 0..k_cur {
            let q_cost = quality_costs[x][j];
            let cand_j = candidates[x][j];

            for i in 0..k_prev {
                let transition = if candidates[x - 1][i] == cand_j {
                    0.0
                } else {
                    trans_cost
                };
                let total = dp[i] + q_cost + transition;
                if total < new_dp[j] {
                    new_dp[j] = total;
                    bp[j] = i as u8;
                }
            }
        }

        dp = new_dp;
        backptrs[x] = bp;
    }

    // Traceback: find the best final state
    let k_last = cand_counts[n - 1] as usize;
    let mut best_j = 0;
    let mut best_cost = f32::MAX;
    for (j, &cost) in dp[..k_last].iter().enumerate() {
        if cost < best_cost {
            best_cost = cost;
            best_j = j;
        }
    }

    // Walk backwards
    indices[n - 1] = candidates[n - 1][best_j];
    let mut j = best_j;
    for x in (1..n).rev() {
        j = backptrs[x][j] as usize;
        indices[x - 1] = candidates[x - 1][j];
    }
}

/// Lightweight run-extension post-pass. O(n) per scanline.
///
/// Extends runs where the quality cost is within threshold. Much cheaper than
/// full Viterbi (no k_nearest search, no DP) but greedy — can't plan ahead.
/// Good for balanced mode where we want SOME compression benefit quickly.
pub fn run_extend_refine(
    pixels: &[rgb::RGB<u8>],
    width: usize,
    height: usize,
    weights: &[f32],
    palette: &Palette,
    indices: &mut [u8],
    lambda: f32,
) {
    if width <= 1 || lambda <= 0.0 {
        return;
    }

    for y in 0..height {
        let row_start = y * width;
        // Forward pass: extend runs rightward where quality cost is acceptable.
        // Only extend in textured regions (low weight) where changes are masked.
        for x in 1..width {
            let i = row_start + x;
            if indices[i] != indices[i - 1] {
                let w = weights[i];
                // Only consider run extension in textured regions
                if w > 0.7 {
                    continue;
                }
                let lab = srgb_to_oklab(pixels[i].r, pixels[i].g, pixels[i].b);
                let curr_dist = palette.distance_sq(lab, indices[i]);
                let prev_dist = palette.distance_sq(lab, indices[i - 1]);
                let threshold = lambda * (1.0 - w);
                if prev_dist <= curr_dist + threshold {
                    indices[i] = indices[i - 1];
                }
            }
        }
    }
}

/// Lightweight run-extension post-pass for RGBA images. Transparent pixels skipped.
pub fn run_extend_refine_rgba(
    pixels: &[rgb::RGBA<u8>],
    width: usize,
    height: usize,
    weights: &[f32],
    palette: &Palette,
    indices: &mut [u8],
    lambda: f32,
) {
    if width <= 1 || lambda <= 0.0 {
        return;
    }

    for y in 0..height {
        let row_start = y * width;
        for x in 1..width {
            let i = row_start + x;
            if pixels[i].a == 0 || pixels[i - 1].a == 0 {
                continue;
            }
            if indices[i] != indices[i - 1] {
                let w = weights[i];
                if w > 0.7 {
                    continue;
                }
                let lab = srgb_to_oklab(pixels[i].r, pixels[i].g, pixels[i].b);
                let curr_dist = palette.distance_sq(lab, indices[i]);
                let prev_dist = palette.distance_sq(lab, indices[i - 1]);
                let threshold = lambda * (1.0 - w);
                if prev_dist <= curr_dist + threshold {
                    indices[i] = indices[i - 1];
                }
            }
        }
    }
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
