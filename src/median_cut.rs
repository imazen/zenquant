extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

use crate::oklab::{OKLab, OKLabA};

// Old median cut kept for comparison tests only.
#[cfg(test)]
/// A box of color entries for median cut subdivision.
#[derive(Debug, Clone)]
struct ColorBox {
    entries: Vec<(OKLab, f32)>, // (color, accumulated_weight)
}

#[cfg(test)]
impl ColorBox {
    fn new(entries: Vec<(OKLab, f32)>) -> Self {
        Self { entries }
    }

    fn total_weight(&self) -> f32 {
        self.entries.iter().map(|(_, w)| w).sum()
    }

    /// Compute the range (max - min) along each OKLab axis.
    fn ranges(&self) -> (f32, f32, f32) {
        let mut l_min = f32::MAX;
        let mut l_max = f32::MIN;
        let mut a_min = f32::MAX;
        let mut a_max = f32::MIN;
        let mut b_min = f32::MAX;
        let mut b_max = f32::MIN;

        for (lab, _) in &self.entries {
            l_min = l_min.min(lab.l);
            l_max = l_max.max(lab.l);
            a_min = a_min.min(lab.a);
            a_max = a_max.max(lab.a);
            b_min = b_min.min(lab.b);
            b_max = b_max.max(lab.b);
        }

        (l_max - l_min, a_max - a_min, b_max - b_min)
    }

    /// Volume of this box (product of ranges).
    fn volume(&self) -> f32 {
        let (rl, ra, rb) = self.ranges();
        rl * ra * rb
    }

    /// Split priority: larger weighted boxes with more color variation split first.
    fn priority(&self) -> f32 {
        self.total_weight() * self.volume()
    }

    /// Weighted centroid of all entries.
    fn centroid(&self) -> OKLab {
        let mut l_sum = 0.0f32;
        let mut a_sum = 0.0f32;
        let mut b_sum = 0.0f32;
        let mut w_sum = 0.0f32;

        for (lab, w) in &self.entries {
            l_sum += lab.l * w;
            a_sum += lab.a * w;
            b_sum += lab.b * w;
            w_sum += w;
        }

        if w_sum < 1e-10 {
            return OKLab::new(0.0, 0.0, 0.0);
        }

        OKLab::new(l_sum / w_sum, a_sum / w_sum, b_sum / w_sum)
    }

    /// Split this box along the axis with the largest range at the weighted median.
    fn split(mut self) -> (ColorBox, ColorBox) {
        let (rl, ra, rb) = self.ranges();

        // Choose split axis
        let axis = if rl >= ra && rl >= rb {
            0 // L
        } else if ra >= rb {
            1 // a
        } else {
            2 // b
        };

        // Sort by chosen axis
        self.entries.sort_unstable_by(|a, b| {
            let va = match axis {
                0 => a.0.l,
                1 => a.0.a,
                _ => a.0.b,
            };
            let vb = match axis {
                0 => b.0.l,
                1 => b.0.a,
                _ => b.0.b,
            };
            va.partial_cmp(&vb).unwrap_or(core::cmp::Ordering::Equal)
        });

        // Find weighted median split point
        let half_weight = self.total_weight() / 2.0;
        let mut accumulated = 0.0f32;
        let mut split_idx = 1; // At least one entry per side

        for (i, (_, w)) in self.entries.iter().enumerate() {
            accumulated += w;
            if accumulated >= half_weight && i + 1 < self.entries.len() {
                split_idx = i + 1;
                break;
            }
        }

        // Ensure at least one entry per side
        split_idx = split_idx.max(1).min(self.entries.len() - 1);

        let right = self.entries.split_off(split_idx);
        (ColorBox::new(self.entries), ColorBox::new(right))
    }
}

#[cfg(test)]
/// Perform weighted median cut quantization.
///
/// Takes histogram entries (OKLab color, accumulated weight) and produces
/// up to `max_colors` palette centroids in OKLab space.
///
/// If `refine` is true, performs k-means refinement after median cut.
pub fn median_cut(histogram: Vec<(OKLab, f32)>, max_colors: usize, refine: bool) -> Vec<OKLab> {
    if histogram.is_empty() {
        return Vec::new();
    }

    if histogram.len() <= max_colors {
        return histogram.into_iter().map(|(lab, _)| lab).collect();
    }

    let mut boxes = Vec::with_capacity(max_colors);
    boxes.push(ColorBox::new(histogram));

    while boxes.len() < max_colors {
        // Find the box with highest priority to split
        let best_idx = boxes
            .iter()
            .enumerate()
            .filter(|(_, b)| b.entries.len() >= 2)
            .max_by(|(_, a), (_, b)| {
                a.priority()
                    .partial_cmp(&b.priority())
                    .unwrap_or(core::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i);

        let Some(idx) = best_idx else {
            break; // No more splittable boxes
        };

        let to_split = boxes.swap_remove(idx);
        let (left, right) = to_split.split();
        boxes.push(left);
        boxes.push(right);
    }

    let mut palette: Vec<OKLab> = boxes.iter().map(|b| b.centroid()).collect();

    if refine {
        // Collect all histogram entries for refinement
        let all_entries: Vec<(OKLab, f32)> = boxes.into_iter().flat_map(|b| b.entries).collect();
        palette = kmeans_refine(palette, &all_entries);
    }

    palette
}

/// Weighted k-means refinement with convergence checking.
/// Runs up to 32 iterations, stops early if centroids stabilize.
fn kmeans_refine(mut centroids: Vec<OKLab>, entries: &[(OKLab, f32)]) -> Vec<OKLab> {
    const MAX_ITERS: usize = 32;
    const CONVERGENCE_THRESHOLD: f32 = 1e-6; // max centroid movement² to stop

    let k = centroids.len();

    for _ in 0..MAX_ITERS {
        let mut sums_l = vec![0.0f32; k];
        let mut sums_a = vec![0.0f32; k];
        let mut sums_b = vec![0.0f32; k];
        let mut weights = vec![0.0f32; k];

        // Assign each entry to nearest centroid
        for &(lab, w) in entries {
            let nearest = find_nearest(&centroids, lab);
            sums_l[nearest] += lab.l * w;
            sums_a[nearest] += lab.a * w;
            sums_b[nearest] += lab.b * w;
            weights[nearest] += w;
        }

        // Recompute centroids and track movement
        let mut max_movement = 0.0f32;
        for i in 0..k {
            if weights[i] > 1e-10 {
                let new_centroid = OKLab::new(
                    sums_l[i] / weights[i],
                    sums_a[i] / weights[i],
                    sums_b[i] / weights[i],
                );
                max_movement = max_movement.max(centroids[i].distance_sq(new_centroid));
                centroids[i] = new_centroid;
            }
        }

        if max_movement < CONVERGENCE_THRESHOLD {
            break;
        }
    }

    centroids
}

// =====================================================================
// Variance-minimizing quantization (direct entry-based, Wu-style splitting)
// =====================================================================

/// Get a specific axis value from an OKLab color.
#[inline]
fn get_axis_3d(lab: &OKLab, axis: u8) -> f32 {
    match axis {
        0 => lab.l,
        1 => lab.a,
        _ => lab.b,
    }
}

/// Compute weighted variance of histogram entries: Var = sum(x²w)/W - (sum(xw)/W)² per axis.
fn variance_from_stats_3d(w: f64, sl: f64, sa: f64, sb: f64, sl2: f64, sa2: f64, sb2: f64) -> f64 {
    if w < 1e-10 {
        return 0.0;
    }
    (sl2 - sl * sl / w) + (sa2 - sa * sa / w) + (sb2 - sb * sb / w)
}

/// Variance-minimizing quantization (Wu-style greedy splitting on histogram entries).
///
/// Operates directly on floating-point OKLab histogram entries — no fixed-resolution
/// grid, so there's no binning precision loss. Greedily splits the highest-variance
/// cluster at the position minimizing total within-cluster variance.
///
/// If `refine` is true, follows up with k-means refinement on histogram entries.
pub fn wu_quantize(histogram: Vec<(OKLab, f32)>, max_colors: usize, refine: bool) -> Vec<OKLab> {
    if histogram.is_empty() {
        return Vec::new();
    }

    if histogram.len() <= max_colors {
        return histogram.into_iter().map(|(lab, _)| lab).collect();
    }

    let n = histogram.len();
    let mut indices: Vec<usize> = (0..n).collect();

    // Compute initial total stats
    let mut tw = 0.0f64;
    let mut tl = 0.0f64;
    let mut ta = 0.0f64;
    let mut tb = 0.0f64;
    let mut tl2 = 0.0f64;
    let mut ta2 = 0.0f64;
    let mut tb2 = 0.0f64;
    for &(lab, w) in &histogram {
        let w64 = w as f64;
        tw += w64;
        let (l, a, b) = (lab.l as f64, lab.a as f64, lab.b as f64);
        tl += l * w64;
        ta += a * w64;
        tb += b * w64;
        tl2 += l * l * w64;
        ta2 += a * a * w64;
        tb2 += b * b * w64;
    }
    let initial_var = variance_from_stats_3d(tw, tl, ta, tb, tl2, ta2, tb2);

    // Each box: (start, end, variance) — contiguous range in `indices`
    let mut boxes: Vec<(usize, usize, f64)> = vec![(0, n, initial_var)];
    let mut buf: Vec<usize> = Vec::with_capacity(n);
    let mut best_sorted: Vec<usize> = Vec::with_capacity(n);

    while boxes.len() < max_colors {
        // Find splittable box with highest variance
        let best = boxes
            .iter()
            .enumerate()
            .filter(|(_, (s, e, _))| e - s >= 2)
            .max_by(|(_, a), (_, b)| a.2.partial_cmp(&b.2).unwrap_or(core::cmp::Ordering::Equal));
        let Some((box_idx, &(start, end, box_var))) = best else {
            break;
        };
        if box_var < 1e-12 {
            break;
        }

        let m = end - start;
        buf.clear();
        buf.extend_from_slice(&indices[start..end]);

        // Total stats for this box
        let (bw, bl, ba, bb, bl2, ba2, bb2) = {
            let (mut w, mut sl, mut sa, mut sb, mut sl2, mut sa2, mut sb2) =
                (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
            for &i in &buf {
                let (lab, wt) = &histogram[i];
                let w64 = *wt as f64;
                w += w64;
                let (l, a, b) = (lab.l as f64, lab.a as f64, lab.b as f64);
                sl += l * w64;
                sa += a * w64;
                sb += b * w64;
                sl2 += l * l * w64;
                sa2 += a * a * w64;
                sb2 += b * b * w64;
            }
            (w, sl, sa, sb, sl2, sa2, sb2)
        };

        let mut best_var = f64::MAX;
        let mut best_split = 1usize;

        for axis in 0..3u8 {
            buf.sort_unstable_by(|&a, &b| {
                get_axis_3d(&histogram[a].0, axis)
                    .partial_cmp(&get_axis_3d(&histogram[b].0, axis))
                    .unwrap_or(core::cmp::Ordering::Equal)
            });

            // Running left accumulation
            let mut lw = 0.0f64;
            let mut ll = 0.0f64;
            let mut la = 0.0f64;
            let mut lb = 0.0f64;
            let mut ll2 = 0.0f64;
            let mut la2 = 0.0f64;
            let mut lb2 = 0.0f64;

            for pos in 0..m - 1 {
                let i = buf[pos];
                let (lab, w) = &histogram[i];
                let w64 = *w as f64;
                lw += w64;
                let (l, a, b) = (lab.l as f64, lab.a as f64, lab.b as f64);
                ll += l * w64;
                la += a * w64;
                lb += b * w64;
                ll2 += l * l * w64;
                la2 += a * a * w64;
                lb2 += b * b * w64;

                let rw = bw - lw;
                if lw < 1e-10 || rw < 1e-10 {
                    continue;
                }

                let lvar = variance_from_stats_3d(lw, ll, la, lb, ll2, la2, lb2);
                let rvar = variance_from_stats_3d(
                    rw,
                    bl - ll,
                    ba - la,
                    bb - lb,
                    bl2 - ll2,
                    ba2 - la2,
                    bb2 - lb2,
                );
                let total = lvar + rvar;

                if total < best_var {
                    best_var = total;
                    best_split = pos + 1;
                    best_sorted.clear();
                    best_sorted.extend_from_slice(&buf);
                }
            }
        }

        if best_var >= f64::MAX {
            break;
        }

        // Apply the best split
        indices[start..end].copy_from_slice(&best_sorted);
        let mid = start + best_split;

        let lvar = {
            let (mut w, mut sl, mut sa, mut sb, mut sl2, mut sa2, mut sb2) =
                (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
            for &i in &indices[start..mid] {
                let (lab, wt) = &histogram[i];
                let w64 = *wt as f64;
                w += w64;
                let (l, a, b) = (lab.l as f64, lab.a as f64, lab.b as f64);
                sl += l * w64;
                sa += a * w64;
                sb += b * w64;
                sl2 += l * l * w64;
                sa2 += a * a * w64;
                sb2 += b * b * w64;
            }
            variance_from_stats_3d(w, sl, sa, sb, sl2, sa2, sb2)
        };
        let rvar = {
            let (mut w, mut sl, mut sa, mut sb, mut sl2, mut sa2, mut sb2) =
                (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
            for &i in &indices[mid..end] {
                let (lab, wt) = &histogram[i];
                let w64 = *wt as f64;
                w += w64;
                let (l, a, b) = (lab.l as f64, lab.a as f64, lab.b as f64);
                sl += l * w64;
                sa += a * w64;
                sb += b * w64;
                sl2 += l * l * w64;
                sa2 += a * a * w64;
                sb2 += b * b * w64;
            }
            variance_from_stats_3d(w, sl, sa, sb, sl2, sa2, sb2)
        };

        boxes.swap_remove(box_idx);
        boxes.push((start, mid, lvar));
        boxes.push((mid, end, rvar));
    }

    // Extract centroids
    let mut palette: Vec<OKLab> = boxes
        .iter()
        .map(|&(s, e, _)| {
            let (mut w, mut sl, mut sa, mut sb) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
            for &i in &indices[s..e] {
                let (lab, wt) = &histogram[i];
                let w64 = *wt as f64;
                w += w64;
                sl += lab.l as f64 * w64;
                sa += lab.a as f64 * w64;
                sb += lab.b as f64 * w64;
            }
            if w < 1e-10 {
                OKLab::new(0.0, 0.0, 0.0)
            } else {
                OKLab::new((sl / w) as f32, (sa / w) as f32, (sb / w) as f32)
            }
        })
        .collect();

    if refine {
        palette = kmeans_refine(palette, &histogram);
    }

    palette
}

// =====================================================================
// Farthest-point seeding (deterministic k-means++)
// =====================================================================

/// Farthest-point seeding: deterministic k-means++ on histogram entries.
///
/// 1. First centroid = highest-weight entry
/// 2. Each subsequent = entry with max `sqrt(weight) * min_dist_sq` to existing centroids
/// 3. O(K*N) — trivial for 256 × 600-4000 histogram entries
#[allow(dead_code)]
pub fn farthest_point_seed(histogram: &[(OKLab, f32)], k: usize) -> Vec<OKLab> {
    if histogram.is_empty() || k == 0 {
        return Vec::new();
    }

    let n = histogram.len();
    if n <= k {
        return histogram.iter().map(|(lab, _)| *lab).collect();
    }

    // min_dist[i] = minimum distance² from entry i to any chosen centroid
    let mut min_dist = vec![f32::MAX; n];
    let mut centroids = Vec::with_capacity(k);

    // First centroid = highest-weight entry
    let first = histogram
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.1.partial_cmp(&b.1).unwrap_or(core::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);
    centroids.push(histogram[first].0);

    // Update min distances from first centroid
    for i in 0..n {
        min_dist[i] = histogram[i].0.distance_sq(centroids[0]);
    }

    for _ in 1..k {
        // Pick entry with max weighted min-distance: sqrt(weight) * min_dist
        let mut best_idx = 0;
        let mut best_score = -1.0f32;
        for i in 0..n {
            let score = histogram[i].1.sqrt() * min_dist[i];
            if score > best_score {
                best_score = score;
                best_idx = i;
            }
        }

        let new_centroid = histogram[best_idx].0;
        centroids.push(new_centroid);

        // Update min distances
        for i in 0..n {
            let d = histogram[i].0.distance_sq(new_centroid);
            if d < min_dist[i] {
                min_dist[i] = d;
            }
        }
    }

    centroids
}

/// Quantize using farthest-point seeding + histogram-level k-means refinement.
///
/// Alternative to [`wu_quantize`]. Seeds centroids via farthest-point (deterministic
/// k-means++) then refines with k-means on histogram entries.
#[allow(dead_code)]
pub fn farthest_point_quantize(
    histogram: Vec<(OKLab, f32)>,
    max_colors: usize,
) -> Vec<OKLab> {
    if histogram.is_empty() {
        return Vec::new();
    }
    if histogram.len() <= max_colors {
        return histogram.into_iter().map(|(lab, _)| lab).collect();
    }

    let centroids = farthest_point_seed(&histogram, max_colors);
    kmeans_refine(centroids, &histogram)
}

// =====================================================================
// Variance-minimizing quantization — 4D alpha-aware variant
// =====================================================================

/// Get a specific axis value from an OKLabA color.
#[inline]
fn get_axis_4d(laba: &OKLabA, axis: u8) -> f32 {
    match axis {
        0 => laba.lab.l,
        1 => laba.lab.a,
        2 => laba.lab.b,
        _ => laba.alpha,
    }
}

/// Compute weighted variance of 4D (L, a, b, alpha) entries.
#[allow(clippy::too_many_arguments)]
fn variance_from_stats_4d(
    w: f64,
    sl: f64,
    sa: f64,
    sb: f64,
    sal: f64,
    sl2: f64,
    sa2: f64,
    sb2: f64,
    sal2: f64,
) -> f64 {
    if w < 1e-10 {
        return 0.0;
    }
    (sl2 - sl * sl / w)
        + (sa2 - sa * sa / w)
        + (sb2 - sb * sb / w)
        + (sal2 - sal * sal / w)
}

/// Variance-minimizing quantization with alpha as a 4th dimension.
///
/// Operates directly on floating-point OKLabA histogram entries.
/// If `refine` is true, follows up with k-means refinement on histogram entries.
pub fn wu_quantize_alpha(
    histogram: Vec<(OKLabA, f32)>,
    max_colors: usize,
    refine: bool,
) -> Vec<OKLabA> {
    if histogram.is_empty() {
        return Vec::new();
    }

    if histogram.len() <= max_colors {
        return histogram.into_iter().map(|(laba, _)| laba).collect();
    }

    let n = histogram.len();
    let mut indices: Vec<usize> = (0..n).collect();

    // Compute initial total stats
    let mut tw = 0.0f64;
    let mut tl = 0.0f64;
    let mut ta = 0.0f64;
    let mut tb = 0.0f64;
    let mut tal = 0.0f64;
    let mut tl2 = 0.0f64;
    let mut ta2 = 0.0f64;
    let mut tb2 = 0.0f64;
    let mut tal2 = 0.0f64;
    for &(laba, w) in &histogram {
        let w64 = w as f64;
        tw += w64;
        let (l, a, b, al) = (
            laba.lab.l as f64,
            laba.lab.a as f64,
            laba.lab.b as f64,
            laba.alpha as f64,
        );
        tl += l * w64;
        ta += a * w64;
        tb += b * w64;
        tal += al * w64;
        tl2 += l * l * w64;
        ta2 += a * a * w64;
        tb2 += b * b * w64;
        tal2 += al * al * w64;
    }
    let initial_var = variance_from_stats_4d(tw, tl, ta, tb, tal, tl2, ta2, tb2, tal2);

    let mut boxes: Vec<(usize, usize, f64)> = vec![(0, n, initial_var)];
    let mut buf: Vec<usize> = Vec::with_capacity(n);
    let mut best_sorted: Vec<usize> = Vec::with_capacity(n);

    while boxes.len() < max_colors {
        let best = boxes
            .iter()
            .enumerate()
            .filter(|(_, (s, e, _))| e - s >= 2)
            .max_by(|(_, a), (_, b)| a.2.partial_cmp(&b.2).unwrap_or(core::cmp::Ordering::Equal));
        let Some((box_idx, &(start, end, box_var))) = best else {
            break;
        };
        if box_var < 1e-12 {
            break;
        }

        let m = end - start;
        buf.clear();
        buf.extend_from_slice(&indices[start..end]);

        // Total stats for this box
        let (bw, bl, ba, bb, bal, bl2, ba2, bb2, bal2) = {
            let (mut w, mut sl, mut sa, mut sb, mut sal, mut sl2, mut sa2, mut sb2, mut sal2) =
                (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
            for &i in &buf {
                let (laba, wt) = &histogram[i];
                let w64 = *wt as f64;
                w += w64;
                let (l, a, b, al) = (
                    laba.lab.l as f64,
                    laba.lab.a as f64,
                    laba.lab.b as f64,
                    laba.alpha as f64,
                );
                sl += l * w64;
                sa += a * w64;
                sb += b * w64;
                sal += al * w64;
                sl2 += l * l * w64;
                sa2 += a * a * w64;
                sb2 += b * b * w64;
                sal2 += al * al * w64;
            }
            (w, sl, sa, sb, sal, sl2, sa2, sb2, sal2)
        };

        let mut best_var = f64::MAX;
        let mut best_split = 1usize;

        for axis in 0..4u8 {
            buf.sort_unstable_by(|&a, &b| {
                get_axis_4d(&histogram[a].0, axis)
                    .partial_cmp(&get_axis_4d(&histogram[b].0, axis))
                    .unwrap_or(core::cmp::Ordering::Equal)
            });

            let mut lw = 0.0f64;
            let mut ll = 0.0f64;
            let mut la = 0.0f64;
            let mut lb = 0.0f64;
            let mut lal = 0.0f64;
            let mut ll2 = 0.0f64;
            let mut la2 = 0.0f64;
            let mut lb2 = 0.0f64;
            let mut lal2 = 0.0f64;

            for pos in 0..m - 1 {
                let i = buf[pos];
                let (laba, w) = &histogram[i];
                let w64 = *w as f64;
                lw += w64;
                let (l, a, b, al) = (
                    laba.lab.l as f64,
                    laba.lab.a as f64,
                    laba.lab.b as f64,
                    laba.alpha as f64,
                );
                ll += l * w64;
                la += a * w64;
                lb += b * w64;
                lal += al * w64;
                ll2 += l * l * w64;
                la2 += a * a * w64;
                lb2 += b * b * w64;
                lal2 += al * al * w64;

                let rw = bw - lw;
                if lw < 1e-10 || rw < 1e-10 {
                    continue;
                }

                let lvar = variance_from_stats_4d(lw, ll, la, lb, lal, ll2, la2, lb2, lal2);
                let rvar = variance_from_stats_4d(
                    rw,
                    bl - ll,
                    ba - la,
                    bb - lb,
                    bal - lal,
                    bl2 - ll2,
                    ba2 - la2,
                    bb2 - lb2,
                    bal2 - lal2,
                );
                let total = lvar + rvar;

                if total < best_var {
                    best_var = total;
                    best_split = pos + 1;
                    best_sorted.clear();
                    best_sorted.extend_from_slice(&buf);
                }
            }
        }

        if best_var >= f64::MAX {
            break;
        }

        indices[start..end].copy_from_slice(&best_sorted);
        let mid = start + best_split;

        let lvar = {
            let (mut w, mut sl, mut sa, mut sb, mut sal, mut sl2, mut sa2, mut sb2, mut sal2) =
                (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
            for &i in &indices[start..mid] {
                let (laba, wt) = &histogram[i];
                let w64 = *wt as f64;
                w += w64;
                let (l, a, b, al) = (
                    laba.lab.l as f64,
                    laba.lab.a as f64,
                    laba.lab.b as f64,
                    laba.alpha as f64,
                );
                sl += l * w64;
                sa += a * w64;
                sb += b * w64;
                sal += al * w64;
                sl2 += l * l * w64;
                sa2 += a * a * w64;
                sb2 += b * b * w64;
                sal2 += al * al * w64;
            }
            variance_from_stats_4d(w, sl, sa, sb, sal, sl2, sa2, sb2, sal2)
        };
        let rvar = {
            let (mut w, mut sl, mut sa, mut sb, mut sal, mut sl2, mut sa2, mut sb2, mut sal2) =
                (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
            for &i in &indices[mid..end] {
                let (laba, wt) = &histogram[i];
                let w64 = *wt as f64;
                w += w64;
                let (l, a, b, al) = (
                    laba.lab.l as f64,
                    laba.lab.a as f64,
                    laba.lab.b as f64,
                    laba.alpha as f64,
                );
                sl += l * w64;
                sa += a * w64;
                sb += b * w64;
                sal += al * w64;
                sl2 += l * l * w64;
                sa2 += a * a * w64;
                sb2 += b * b * w64;
                sal2 += al * al * w64;
            }
            variance_from_stats_4d(w, sl, sa, sb, sal, sl2, sa2, sb2, sal2)
        };

        boxes.swap_remove(box_idx);
        boxes.push((start, mid, lvar));
        boxes.push((mid, end, rvar));
    }

    let mut palette: Vec<OKLabA> = boxes
        .iter()
        .map(|&(s, e, _)| {
            let (mut w, mut sl, mut sa, mut sb, mut sal) =
                (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
            for &i in &indices[s..e] {
                let (laba, wt) = &histogram[i];
                let w64 = *wt as f64;
                w += w64;
                sl += laba.lab.l as f64 * w64;
                sa += laba.lab.a as f64 * w64;
                sb += laba.lab.b as f64 * w64;
                sal += laba.alpha as f64 * w64;
            }
            if w < 1e-10 {
                OKLabA::new(0.0, 0.0, 0.0, 1.0)
            } else {
                OKLabA::new(
                    (sl / w) as f32,
                    (sa / w) as f32,
                    (sb / w) as f32,
                    (sal / w) as f32,
                )
            }
        })
        .collect();

    if refine {
        palette = kmeans_refine_alpha(palette, &histogram);
    }

    palette
}

/// Refine centroids against original pixel data.
///
/// Performs k-means refinement by scanning original pixels and recomputing
/// centroids from pixel-to-centroid assignments. This is more accurate than
/// refining against histogram entries alone, since it accounts for the actual
/// pixel distribution rather than pre-quantized histogram approximations.
///
/// When `pixels.len() > max_samples`, stride-subsamples internally with a
/// rotating offset per iteration to cover different pixels. Stops early if
/// <0.5% of sampled pixels changed assignment.
///
/// Uses three acceleration techniques:
/// 1. **Pre-computed grid**: 4096 sRGB→OKLab values computed once (not per iteration)
/// 2. **Incremental rebuild**: NN cache and neighbors updated seeded/selectively
/// 3. **Triangle-inequality skip**: pixels whose distance to their assigned centroid
///    is less than 1/4 of that centroid's distance to its nearest neighbor are
///    guaranteed to not change cluster and skip the search entirely.
pub fn refine_against_pixels(
    centroids: Vec<OKLab>,
    pixels: &[rgb::RGB<u8>],
    weights: &[f32],
    iterations: usize,
    max_samples: usize,
) -> Vec<OKLab> {
    let labs = crate::simd::batch_srgb_to_oklab_vec(pixels);
    refine_against_pixels_from_labs(centroids, pixels, &labs, weights, iterations, max_samples)
}

/// Pixel-level k-means refinement using pre-computed OKLab values.
///
/// Same as [`refine_against_pixels`] but skips the sRGB→OKLab batch conversion.
/// The `pixels` parameter is still needed for the sRGB NN cache lookup.
pub fn refine_against_pixels_from_labs(
    mut centroids: Vec<OKLab>,
    pixels: &[rgb::RGB<u8>],
    labs: &[OKLab],
    weights: &[f32],
    iterations: usize,
    max_samples: usize,
) -> Vec<OKLab> {
    let k = centroids.len();
    if k == 0 {
        return centroids;
    }

    let n = pixels.len();

    // Pre-compute grid OKLab values once (4096 entries, avoids 4096 srgb_to_oklab per iteration)
    let grid_labs = precompute_nn_grid();

    // Persistent acceleration structures, reused across iterations
    let mut nn_cache = build_centroid_nn_cache(&centroids, &grid_labs);
    let mut neighbors = build_centroid_neighbors(&centroids);
    let mut old_centroids = centroids.clone();

    // Per-pixel assignments for triangle-inequality skip on iterations 1+
    let mut assignments = vec![0u8; n];

    // Per-centroid skip threshold: dist_sq(centroid, nearest_other) / 4.
    // If a pixel is closer than this to its assigned centroid, it can't change.
    let mut skip_threshold = compute_skip_thresholds(&centroids, &neighbors);

    // Subsampling: if we have more pixels than max_samples, stride through them.
    // The stride offset rotates each iteration to cover different pixels.
    let needs_subsample = n > max_samples && max_samples > 0;
    let stride = if needs_subsample {
        n / max_samples
    } else {
        1
    };
    // Prime for rotating offset each iteration
    const OFFSET_PRIME: usize = 7;

    for iter in 0..iterations {
        if iter > 0 {
            // Incremental rebuild: neighbors first (need current centroids),
            // then cache (needs updated neighbors for seeded search).
            rebuild_neighbors_incremental(&centroids, &old_centroids, &mut neighbors);
            rebuild_nn_cache_seeded(&centroids, &grid_labs, &mut nn_cache, &neighbors);
            skip_threshold = compute_skip_thresholds(&centroids, &neighbors);
        }

        let mut sums_l = vec![0.0f64; k];
        let mut sums_a = vec![0.0f64; k];
        let mut sums_b = vec![0.0f64; k];
        let mut total_w = vec![0.0f64; k];
        let mut changed_count: usize = 0;
        let mut sampled_count: usize = 0;

        // Compute iteration offset for subsampling rotation
        let offset = if needs_subsample {
            (iter * OFFSET_PRIME) % stride
        } else {
            0
        };

        let mut i = offset;
        while i < n {
            let pixel = &pixels[i];
            let weight = weights[i];
            let lab = labs[i];
            sampled_count += 1;

            // Triangle-inequality early exit: if the pixel is closer to its
            // current centroid than half the distance to the nearest other
            // centroid, no reassignment is possible.
            let nearest = if iter > 0 {
                let prev = assignments[i] as usize;
                let d = lab.distance_sq(centroids[prev]);
                if d < skip_threshold[prev] {
                    prev
                } else {
                    let seed = centroid_cache_lookup(&nn_cache, pixel.r, pixel.g, pixel.b);
                    find_nearest_seeded(&centroids, lab, seed, &neighbors)
                }
            } else {
                let seed = centroid_cache_lookup(&nn_cache, pixel.r, pixel.g, pixel.b);
                find_nearest_seeded(&centroids, lab, seed, &neighbors)
            };

            if iter > 0 && assignments[i] != nearest as u8 {
                changed_count += 1;
            }
            assignments[i] = nearest as u8;
            let w = weight as f64;
            sums_l[nearest] += lab.l as f64 * w;
            sums_a[nearest] += lab.a as f64 * w;
            sums_b[nearest] += lab.b as f64 * w;
            total_w[nearest] += w;

            i += stride;
        }

        // Save centroids before update (for incremental neighbor rebuild next iter)
        old_centroids.copy_from_slice(&centroids);

        let mut max_movement = 0.0f32;
        for i in 0..k {
            if total_w[i] > 1e-10 {
                let new_centroid = OKLab::new(
                    (sums_l[i] / total_w[i]) as f32,
                    (sums_a[i] / total_w[i]) as f32,
                    (sums_b[i] / total_w[i]) as f32,
                );
                max_movement = max_movement.max(centroids[i].distance_sq(new_centroid));
                centroids[i] = new_centroid;
            }
        }

        if max_movement < 1e-6 {
            break;
        }

        // Early convergence: if <0.5% of sampled pixels changed assignment, stop.
        if iter > 0 && sampled_count > 0 && changed_count * 200 < sampled_count {
            break;
        }
    }

    centroids
}

/// Refine centroids against original RGBA pixel data.
/// Transparent pixels (alpha == 0) are skipped.
/// When `pixels.len() > max_samples`, stride-subsamples internally.
pub fn refine_against_pixels_rgba(
    centroids: Vec<OKLab>,
    pixels: &[rgb::RGBA<u8>],
    weights: &[f32],
    iterations: usize,
    max_samples: usize,
) -> Vec<OKLab> {
    // Batch-convert RGB channels, then zero out transparent pixels
    let rgb_pixels: Vec<rgb::RGB<u8>> = pixels
        .iter()
        .map(|p| rgb::RGB::new(p.r, p.g, p.b))
        .collect();
    let mut labs = crate::simd::batch_srgb_to_oklab_vec(&rgb_pixels);
    for (lab, pixel) in labs.iter_mut().zip(pixels.iter()) {
        if pixel.a == 0 {
            *lab = OKLab::new(0.0, 0.0, 0.0);
        }
    }
    refine_against_pixels_rgba_from_labs(centroids, pixels, &labs, weights, iterations, max_samples)
}

/// Pixel-level k-means refinement for RGBA using pre-computed OKLab values.
///
/// Same as [`refine_against_pixels_rgba`] but skips the sRGB→OKLab batch conversion.
/// Transparent pixels in `labs` should already be zeroed by the caller.
pub fn refine_against_pixels_rgba_from_labs(
    mut centroids: Vec<OKLab>,
    pixels: &[rgb::RGBA<u8>],
    labs: &[OKLab],
    weights: &[f32],
    iterations: usize,
    max_samples: usize,
) -> Vec<OKLab> {
    let k = centroids.len();
    if k == 0 {
        return centroids;
    }

    let n = pixels.len();

    // Pre-compute grid OKLab values once (4096 entries)
    let grid_labs = precompute_nn_grid();

    // Persistent acceleration structures, reused across iterations
    let mut nn_cache = build_centroid_nn_cache(&centroids, &grid_labs);
    let mut neighbors = build_centroid_neighbors(&centroids);
    let mut old_centroids = centroids.clone();
    let mut assignments = vec![0u8; n];
    let mut skip_threshold = compute_skip_thresholds(&centroids, &neighbors);

    let needs_subsample = n > max_samples && max_samples > 0;
    let stride = if needs_subsample {
        n / max_samples
    } else {
        1
    };
    const OFFSET_PRIME: usize = 7;

    for iter in 0..iterations {
        if iter > 0 {
            rebuild_neighbors_incremental(&centroids, &old_centroids, &mut neighbors);
            rebuild_nn_cache_seeded(&centroids, &grid_labs, &mut nn_cache, &neighbors);
            skip_threshold = compute_skip_thresholds(&centroids, &neighbors);
        }

        let mut sums_l = vec![0.0f64; k];
        let mut sums_a = vec![0.0f64; k];
        let mut sums_b = vec![0.0f64; k];
        let mut total_w = vec![0.0f64; k];
        let mut changed_count: usize = 0;
        let mut sampled_count: usize = 0;

        let offset = if needs_subsample {
            (iter * OFFSET_PRIME) % stride
        } else {
            0
        };

        let mut i = offset;
        while i < n {
            let pixel = &pixels[i];
            if pixel.a == 0 {
                i += stride;
                continue;
            }
            let weight = weights[i];
            let lab = labs[i];
            sampled_count += 1;

            let nearest = if iter > 0 {
                let prev = assignments[i] as usize;
                let d = lab.distance_sq(centroids[prev]);
                if d < skip_threshold[prev] {
                    prev
                } else {
                    let seed = centroid_cache_lookup(&nn_cache, pixel.r, pixel.g, pixel.b);
                    find_nearest_seeded(&centroids, lab, seed, &neighbors)
                }
            } else {
                let seed = centroid_cache_lookup(&nn_cache, pixel.r, pixel.g, pixel.b);
                find_nearest_seeded(&centroids, lab, seed, &neighbors)
            };

            if iter > 0 && assignments[i] != nearest as u8 {
                changed_count += 1;
            }
            assignments[i] = nearest as u8;
            let w = weight as f64;
            sums_l[nearest] += lab.l as f64 * w;
            sums_a[nearest] += lab.a as f64 * w;
            sums_b[nearest] += lab.b as f64 * w;
            total_w[nearest] += w;

            i += stride;
        }

        // Save centroids before update (for incremental neighbor rebuild next iter)
        old_centroids.copy_from_slice(&centroids);

        let mut max_movement = 0.0f32;
        for i in 0..k {
            if total_w[i] > 1e-10 {
                let new_centroid = OKLab::new(
                    (sums_l[i] / total_w[i]) as f32,
                    (sums_a[i] / total_w[i]) as f32,
                    (sums_b[i] / total_w[i]) as f32,
                );
                max_movement = max_movement.max(centroids[i].distance_sq(new_centroid));
                centroids[i] = new_centroid;
            }
        }

        if max_movement < 1e-6 {
            break;
        }

        if iter > 0 && sampled_count > 0 && changed_count * 200 < sampled_count {
            break;
        }
    }

    centroids
}

/// Find the index of the nearest centroid to a given color.
#[inline]
fn find_nearest(centroids: &[OKLab], color: OKLab) -> usize {
    let mut best_idx = 0;
    let mut best_dist = f32::MAX;
    for (i, c) in centroids.iter().enumerate() {
        let d = color.distance_sq(*c);
        if d < best_dist {
            best_dist = d;
            best_idx = i;
        }
    }
    best_idx
}

// =====================================================================
// Seeded nearest-neighbor acceleration for k-means
// =====================================================================

/// Pre-compute OKLab values for the 4-bit sRGB grid (16×16×16 = 4096 entries).
/// Called once before the k-means loop to avoid recomputing cube roots every iteration.
fn precompute_nn_grid() -> Vec<OKLab> {
    const BITS: usize = 4;
    const SIZE: usize = 1 << BITS;
    const TOTAL: usize = SIZE * SIZE * SIZE;
    let shift = 8 - BITS;

    let mut grid_rgb = Vec::with_capacity(TOTAL);
    for r_idx in 0..SIZE {
        for g_idx in 0..SIZE {
            for b_idx in 0..SIZE {
                let r = ((r_idx << shift) | (1 << (shift - 1))) as u8;
                let g = ((g_idx << shift) | (1 << (shift - 1))) as u8;
                let b = ((b_idx << shift) | (1 << (shift - 1))) as u8;
                grid_rgb.push(rgb::RGB::new(r, g, b));
            }
        }
    }
    crate::simd::batch_srgb_to_oklab_vec(&grid_rgb)
}

/// Build a 4-bit sRGB→centroid cache (16×16×16 = 4KB) using SIMD brute-force.
/// Used for the first k-means iteration (no previous cache to seed from).
fn build_centroid_nn_cache(centroids: &[OKLab], grid_labs: &[OKLab]) -> Vec<u8> {
    let simd_layout = crate::simd::PaletteSimd::from_oklab_slice(centroids, 0);
    grid_labs
        .iter()
        .map(|lab| simd_layout.nearest(*lab))
        .collect()
}

/// Rebuild the NN cache using the previous cache as seed + neighbor refinement.
/// 4096 × 17 distance checks instead of 4096 × 256 brute-force.
fn rebuild_nn_cache_seeded(
    centroids: &[OKLab],
    grid_labs: &[OKLab],
    cache: &mut [u8],
    neighbors: &[[u8; 16]],
) {
    for (i, lab) in grid_labs.iter().enumerate() {
        let seed = cache[i] as usize;
        cache[i] = find_nearest_seeded(centroids, *lab, seed, neighbors) as u8;
    }
}

/// Compute 16 nearest neighbors for each centroid (brute-force, first iteration).
fn build_centroid_neighbors(centroids: &[OKLab]) -> Vec<[u8; 16]> {
    let n = centroids.len();
    let mut neighbors = vec![[0u8; 16]; n];

    for (i, nbr) in neighbors.iter_mut().enumerate().take(n) {
        rebuild_neighbors_for(i, centroids, nbr);
    }

    neighbors
}

/// Rebuild neighbor list for a single centroid.
fn rebuild_neighbors_for(i: usize, centroids: &[OKLab], out: &mut [u8; 16]) {
    let n = centroids.len();
    const K: usize = 16;

    let mut dists: Vec<(u8, f32)> = (0..n)
        .filter(|&j| j != i)
        .map(|j| (j as u8, centroids[i].distance_sq(centroids[j])))
        .collect();
    dists.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(core::cmp::Ordering::Equal));
    let count = dists.len().min(K);
    for k in 0..count {
        out[k] = dists[k].0;
    }
}

/// Incrementally rebuild neighbor lists only for centroids that moved or whose
/// neighbors moved. Returns the updated neighbors in-place.
fn rebuild_neighbors_incremental(
    centroids: &[OKLab],
    old_centroids: &[OKLab],
    neighbors: &mut [[u8; 16]],
) {
    let k = centroids.len();
    let neighbor_count = (k - 1).min(16);

    // Which centroids moved more than a small threshold?
    let moved: Vec<bool> = (0..k)
        .map(|i| old_centroids[i].distance_sq(centroids[i]) > 1e-5)
        .collect();

    for i in 0..k {
        // Rebuild if this centroid moved, or any of its current neighbors moved
        let needs_rebuild = moved[i]
            || neighbors[i][..neighbor_count]
                .iter()
                .any(|&n| moved[n as usize]);

        if needs_rebuild {
            rebuild_neighbors_for(i, centroids, &mut neighbors[i]);
        }
    }
}

/// Compute per-centroid skip thresholds for triangle-inequality early exit.
///
/// For centroid i, if a pixel's distance² to centroid i is less than
/// `dist²(i, nearest_other) / 4`, then no other centroid can be closer
/// (by triangle inequality). The /4 is because we compare squared distances:
/// `d(p,i) < d(i,j)/2` becomes `d²(p,i) < d²(i,j)/4`.
fn compute_skip_thresholds(centroids: &[OKLab], neighbors: &[[u8; 16]]) -> Vec<f32> {
    centroids
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let nearest_other = neighbors[i][0] as usize;
            c.distance_sq(centroids[nearest_other]) * 0.25
        })
        .collect()
}

/// Find nearest centroid using seed + neighbor refinement (17 checks vs 256).
#[inline]
fn find_nearest_seeded(
    centroids: &[OKLab],
    color: OKLab,
    seed: usize,
    neighbors: &[[u8; 16]],
) -> usize {
    let mut best_idx = seed;
    let mut best_dist = color.distance_sq(centroids[seed]);

    for &nbr in &neighbors[seed] {
        let ni = nbr as usize;
        let d = color.distance_sq(centroids[ni]);
        if d < best_dist {
            best_dist = d;
            best_idx = ni;
        }
    }

    best_idx
}

/// Look up nearest centroid seed from 4-bit sRGB cache.
#[inline]
fn centroid_cache_lookup(cache: &[u8], r: u8, g: u8, b: u8) -> usize {
    const SHIFT: usize = 4; // 8 - 4
    let idx = ((r as usize >> SHIFT) << 8) | ((g as usize >> SHIFT) << 4) | (b as usize >> SHIFT);
    cache[idx] as usize
}

// =====================================================================
// Old median_cut_alpha removed — replaced by wu_quantize_alpha above.

/// K-means refinement in 4D OKLabA space.
fn kmeans_refine_alpha(mut centroids: Vec<OKLabA>, entries: &[(OKLabA, f32)]) -> Vec<OKLabA> {
    const MAX_ITERS: usize = 32;
    const CONVERGENCE_THRESHOLD: f32 = 1e-6;

    let k = centroids.len();

    for _ in 0..MAX_ITERS {
        let mut sums_l = vec![0.0f32; k];
        let mut sums_a = vec![0.0f32; k];
        let mut sums_b = vec![0.0f32; k];
        let mut sums_al = vec![0.0f32; k];
        let mut weights = vec![0.0f32; k];

        for &(laba, w) in entries {
            let nearest = find_nearest_alpha(&centroids, laba);
            sums_l[nearest] += laba.lab.l * w;
            sums_a[nearest] += laba.lab.a * w;
            sums_b[nearest] += laba.lab.b * w;
            sums_al[nearest] += laba.alpha * w;
            weights[nearest] += w;
        }

        let mut max_movement = 0.0f32;
        for i in 0..k {
            if weights[i] > 1e-10 {
                let new_centroid = OKLabA::new(
                    sums_l[i] / weights[i],
                    sums_a[i] / weights[i],
                    sums_b[i] / weights[i],
                    sums_al[i] / weights[i],
                );
                max_movement = max_movement.max(centroids[i].distance_sq(new_centroid));
                centroids[i] = new_centroid;
            }
        }

        if max_movement < CONVERGENCE_THRESHOLD {
            break;
        }
    }

    centroids
}

/// Refine alpha-aware centroids against original RGBA pixel data.
/// When `pixels.len() > max_samples`, stride-subsamples internally.
pub fn refine_against_pixels_alpha(
    mut centroids: Vec<OKLabA>,
    pixels: &[rgb::RGBA<u8>],
    weights: &[f32],
    iterations: usize,
    max_samples: usize,
) -> Vec<OKLabA> {
    let k = centroids.len();
    if k == 0 {
        return centroids;
    }

    let n = pixels.len();

    // Pre-convert all pixels to OKLabA once (avoids repeated cube-root per iteration)
    let rgb_pixels: Vec<rgb::RGB<u8>> = pixels
        .iter()
        .map(|p| rgb::RGB::new(p.r, p.g, p.b))
        .collect();
    let labs = crate::simd::batch_srgb_to_oklab_vec(&rgb_pixels);
    let labas: Vec<OKLabA> = labs
        .iter()
        .zip(pixels.iter())
        .map(|(lab, pixel)| {
            if pixel.a == 0 {
                OKLabA::new(0.0, 0.0, 0.0, 0.0)
            } else {
                OKLabA::new(lab.l, lab.a, lab.b, pixel.a as f32 / 255.0)
            }
        })
        .collect();

    let needs_subsample = n > max_samples && max_samples > 0;
    let stride = if needs_subsample {
        n / max_samples
    } else {
        1
    };
    const OFFSET_PRIME: usize = 7;

    // Per-pixel assignments for early convergence tracking
    let mut assignments = vec![0u8; n];

    for iter in 0..iterations {
        let mut sums_l = vec![0.0f64; k];
        let mut sums_a = vec![0.0f64; k];
        let mut sums_b = vec![0.0f64; k];
        let mut sums_al = vec![0.0f64; k];
        let mut total_w = vec![0.0f64; k];
        let mut changed_count: usize = 0;
        let mut sampled_count: usize = 0;

        let offset = if needs_subsample {
            (iter * OFFSET_PRIME) % stride
        } else {
            0
        };

        let mut i = offset;
        while i < n {
            let laba = labas[i];
            if laba.alpha == 0.0 {
                i += stride;
                continue;
            }
            let weight = weights[i];
            sampled_count += 1;

            let nearest = find_nearest_alpha(&centroids, laba);
            if iter > 0 && assignments[i] != nearest as u8 {
                changed_count += 1;
            }
            assignments[i] = nearest as u8;

            let w = weight as f64;
            sums_l[nearest] += laba.lab.l as f64 * w;
            sums_a[nearest] += laba.lab.a as f64 * w;
            sums_b[nearest] += laba.lab.b as f64 * w;
            sums_al[nearest] += laba.alpha as f64 * w;
            total_w[nearest] += w;

            i += stride;
        }

        let mut max_movement = 0.0f32;
        for i in 0..k {
            if total_w[i] > 1e-10 {
                let new_centroid = OKLabA::new(
                    (sums_l[i] / total_w[i]) as f32,
                    (sums_a[i] / total_w[i]) as f32,
                    (sums_b[i] / total_w[i]) as f32,
                    (sums_al[i] / total_w[i]) as f32,
                );
                max_movement = max_movement.max(centroids[i].distance_sq(new_centroid));
                centroids[i] = new_centroid;
            }
        }

        if max_movement < 1e-6 {
            break;
        }

        if iter > 0 && sampled_count > 0 && changed_count * 200 < sampled_count {
            break;
        }
    }

    centroids
}

/// Find nearest centroid in 4D OKLabA space.
#[inline]
fn find_nearest_alpha(centroids: &[OKLabA], color: OKLabA) -> usize {
    let mut best_idx = 0;
    let mut best_dist = f32::MAX;
    for (i, c) in centroids.iter().enumerate() {
        let d = color.distance_sq(*c);
        if d < best_dist {
            best_dist = d;
            best_idx = i;
        }
    }
    best_idx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_histogram() {
        let result = median_cut(Vec::new(), 16, false);
        assert!(result.is_empty());
    }

    #[test]
    fn fewer_colors_than_max() {
        let hist = vec![
            (OKLab::new(0.5, 0.0, 0.0), 10.0),
            (OKLab::new(0.8, 0.0, 0.0), 10.0),
        ];
        let result = median_cut(hist, 16, false);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn produces_requested_count() {
        let mut hist = Vec::new();
        for i in 0..100 {
            let l = i as f32 / 100.0;
            hist.push((OKLab::new(l, 0.0, 0.0), 1.0));
        }
        let result = median_cut(hist, 8, false);
        assert_eq!(result.len(), 8);
    }

    #[test]
    fn weighted_entries_influence_centroids() {
        // Two color clusters: one with high weight, one with low
        let mut hist = Vec::new();
        // Heavy cluster near L=0.2
        for i in 0..10 {
            let l = 0.2 + i as f32 * 0.01;
            hist.push((OKLab::new(l, 0.0, 0.0), 10.0)); // high weight
        }
        // Light cluster near L=0.8
        for i in 0..10 {
            let l = 0.8 + i as f32 * 0.01;
            hist.push((OKLab::new(l, 0.0, 0.0), 0.1)); // low weight
        }

        let result = median_cut(hist, 4, false);
        assert_eq!(result.len(), 4);

        // The heavy cluster should get more palette entries
        let dark_count = result.iter().filter(|c| c.l < 0.5).count();
        let light_count = result.iter().filter(|c| c.l >= 0.5).count();
        assert!(
            dark_count >= light_count,
            "expected more entries for heavy cluster: dark={dark_count}, light={light_count}"
        );
    }

    #[test]
    fn refinement_improves_centroids() {
        let mut hist = Vec::new();
        for i in 0..50 {
            let l = i as f32 / 50.0;
            hist.push((
                OKLab::new(l, (i as f32).sin() * 0.1, (i as f32).cos() * 0.1),
                1.0,
            ));
        }

        let unrefined = median_cut(hist.clone(), 8, false);
        let refined = median_cut(hist.clone(), 8, true);

        assert_eq!(unrefined.len(), 8);
        assert_eq!(refined.len(), 8);

        // Refinement should produce lower total error
        let err_unrefined = total_error(&hist, &unrefined);
        let err_refined = total_error(&hist, &refined);
        assert!(
            err_refined <= err_unrefined,
            "refinement should not increase error: unrefined={err_unrefined}, refined={err_refined}"
        );
    }

    #[test]
    fn convergence_stops_early() {
        // All identical entries — should converge in 1 iteration
        let hist: Vec<(OKLab, f32)> = (0..100).map(|_| (OKLab::new(0.5, 0.0, 0.0), 1.0)).collect();
        let result = median_cut(hist, 4, true);
        assert!(!result.is_empty());
    }

    /// Helper: compute total weighted quantization error
    fn total_error(entries: &[(OKLab, f32)], centroids: &[OKLab]) -> f32 {
        entries
            .iter()
            .map(|&(lab, w)| {
                let min_dist = centroids
                    .iter()
                    .map(|c| lab.distance_sq(*c))
                    .fold(f32::MAX, f32::min);
                min_dist * w
            })
            .sum()
    }

    // --- Wu's quantization tests ---

    #[test]
    fn wu_empty_histogram() {
        let result = wu_quantize(Vec::new(), 16, false);
        assert!(result.is_empty());
    }

    #[test]
    fn wu_fewer_colors_than_max() {
        let hist = vec![
            (OKLab::new(0.5, 0.0, 0.0), 10.0),
            (OKLab::new(0.8, 0.0, 0.0), 10.0),
        ];
        let result = wu_quantize(hist, 16, false);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn wu_produces_requested_count() {
        let mut hist = Vec::new();
        for i in 0..100 {
            let l = i as f32 / 100.0;
            hist.push((OKLab::new(l, 0.0, 0.0), 1.0));
        }
        let result = wu_quantize(hist, 8, false);
        assert_eq!(result.len(), 8);
    }

    #[test]
    fn wu_refinement_improves_centroids() {
        let mut hist = Vec::new();
        for i in 0..50 {
            let l = i as f32 / 50.0;
            hist.push((
                OKLab::new(l, (i as f32).sin() * 0.1, (i as f32).cos() * 0.1),
                1.0,
            ));
        }

        let unrefined = wu_quantize(hist.clone(), 8, false);
        let refined = wu_quantize(hist.clone(), 8, true);

        assert_eq!(unrefined.len(), 8);
        assert_eq!(refined.len(), 8);

        let err_unrefined = total_error(&hist, &unrefined);
        let err_refined = total_error(&hist, &refined);
        assert!(
            err_refined <= err_unrefined + 1e-6,
            "refinement should not increase error: unrefined={err_unrefined}, refined={err_refined}"
        );
    }

    #[test]
    fn wu_beats_or_matches_median_cut() {
        // Wu should produce equal or lower total error than median cut
        let mut hist = Vec::new();
        for i in 0..200 {
            let l = (i as f32 / 200.0).clamp(0.0, 1.0);
            let a = ((i as f32 * 0.07).sin() * 0.3).clamp(-0.4, 0.4);
            let b = ((i as f32 * 0.13).cos() * 0.3).clamp(-0.4, 0.4);
            hist.push((OKLab::new(l, a, b), (i % 5 + 1) as f32));
        }

        let mc = median_cut(hist.clone(), 16, true);
        let wu = wu_quantize(hist.clone(), 16, true);

        let err_mc = total_error(&hist, &mc);
        let err_wu = total_error(&hist, &wu);
        // Wu should be at least as good (allowing small float tolerance)
        assert!(
            err_wu <= err_mc * 1.05,
            "Wu should not be much worse than median cut: mc={err_mc}, wu={err_wu}"
        );
    }
}
