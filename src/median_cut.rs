extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

use crate::oklab::{OKLab, OKLabA};

/// A box of color entries for median cut subdivision.
#[derive(Debug, Clone)]
struct ColorBox {
    entries: Vec<(OKLab, f32)>, // (color, accumulated_weight)
}

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

/// Refine centroids against original pixel data.
///
/// Performs k-means refinement by scanning original pixels and recomputing
/// centroids from pixel-to-centroid assignments. This is more accurate than
/// refining against histogram entries alone, since it accounts for the actual
/// pixel distribution rather than pre-quantized histogram approximations.
///
/// Uses incremental acceleration: the NN grid OKLab values are computed once,
/// and after the first iteration, the cache and neighbor lists are updated
/// incrementally rather than rebuilt from scratch.
pub fn refine_against_pixels(
    mut centroids: Vec<OKLab>,
    pixels: &[rgb::RGB<u8>],
    weights: &[f32],
    iterations: usize,
) -> Vec<OKLab> {
    use crate::oklab::srgb_to_oklab;

    let k = centroids.len();
    if k == 0 {
        return centroids;
    }

    // Pre-convert all pixels to OKLab once (avoids repeated cube-root per iteration)
    let labs: Vec<OKLab> = pixels
        .iter()
        .map(|p| srgb_to_oklab(p.r, p.g, p.b))
        .collect();

    // Pre-compute grid OKLab values once (4096 entries, avoids 4096 srgb_to_oklab per iteration)
    let grid_labs = precompute_nn_grid();

    // Persistent acceleration structures, reused across iterations
    let mut nn_cache = build_centroid_nn_cache(&centroids, &grid_labs);
    let mut neighbors = build_centroid_neighbors(&centroids);
    let mut old_centroids = centroids.clone();

    for iter in 0..iterations {
        if iter > 0 {
            // Incremental rebuild: neighbors first (need current centroids),
            // then cache (needs updated neighbors for seeded search).
            // Neighbors: only rebuild for centroids that moved or whose neighbors moved.
            // Cache: use previous cache entry as seed + neighbor check (9 checks vs 256).
            rebuild_neighbors_incremental(&centroids, &old_centroids, &mut neighbors);
            rebuild_nn_cache_seeded(&centroids, &grid_labs, &mut nn_cache, &neighbors);
        }

        let mut sums_l = vec![0.0f64; k];
        let mut sums_a = vec![0.0f64; k];
        let mut sums_b = vec![0.0f64; k];
        let mut total_w = vec![0.0f64; k];

        for (i, (pixel, &weight)) in pixels.iter().zip(weights.iter()).enumerate() {
            let lab = labs[i];
            let seed = centroid_cache_lookup(&nn_cache, pixel.r, pixel.g, pixel.b);
            let nearest = find_nearest_seeded(&centroids, lab, seed, &neighbors);
            let w = weight as f64;
            sums_l[nearest] += lab.l as f64 * w;
            sums_a[nearest] += lab.a as f64 * w;
            sums_b[nearest] += lab.b as f64 * w;
            total_w[nearest] += w;
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
    }

    centroids
}

/// Refine centroids against original RGBA pixel data.
/// Transparent pixels (alpha == 0) are skipped.
pub fn refine_against_pixels_rgba(
    mut centroids: Vec<OKLab>,
    pixels: &[rgb::RGBA<u8>],
    weights: &[f32],
    iterations: usize,
) -> Vec<OKLab> {
    use crate::oklab::srgb_to_oklab;

    let k = centroids.len();
    if k == 0 {
        return centroids;
    }

    // Pre-convert all pixels to OKLab once (avoids repeated cube-root per iteration)
    let labs: Vec<OKLab> = pixels
        .iter()
        .map(|p| {
            if p.a == 0 {
                OKLab::new(0.0, 0.0, 0.0) // placeholder for transparent pixels
            } else {
                srgb_to_oklab(p.r, p.g, p.b)
            }
        })
        .collect();

    // Pre-compute grid OKLab values once (4096 entries)
    let grid_labs = precompute_nn_grid();

    // Persistent acceleration structures, reused across iterations
    let mut nn_cache = build_centroid_nn_cache(&centroids, &grid_labs);
    let mut neighbors = build_centroid_neighbors(&centroids);
    let mut old_centroids = centroids.clone();

    for iter in 0..iterations {
        if iter > 0 {
            rebuild_neighbors_incremental(&centroids, &old_centroids, &mut neighbors);
            rebuild_nn_cache_seeded(&centroids, &grid_labs, &mut nn_cache, &neighbors);
        }

        let mut sums_l = vec![0.0f64; k];
        let mut sums_a = vec![0.0f64; k];
        let mut sums_b = vec![0.0f64; k];
        let mut total_w = vec![0.0f64; k];

        for (i, (pixel, &weight)) in pixels.iter().zip(weights.iter()).enumerate() {
            if pixel.a == 0 {
                continue;
            }
            let lab = labs[i];
            let seed = centroid_cache_lookup(&nn_cache, pixel.r, pixel.g, pixel.b);
            let nearest = find_nearest_seeded(&centroids, lab, seed, &neighbors);
            let w = weight as f64;
            sums_l[nearest] += lab.l as f64 * w;
            sums_a[nearest] += lab.a as f64 * w;
            sums_b[nearest] += lab.b as f64 * w;
            total_w[nearest] += w;
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
    use crate::oklab::srgb_to_oklab;

    const BITS: usize = 4;
    const SIZE: usize = 1 << BITS;
    const TOTAL: usize = SIZE * SIZE * SIZE;
    let shift = 8 - BITS;

    let mut grid = Vec::with_capacity(TOTAL);
    for r_idx in 0..SIZE {
        for g_idx in 0..SIZE {
            for b_idx in 0..SIZE {
                let r = ((r_idx << shift) | (1 << (shift - 1))) as u8;
                let g = ((g_idx << shift) | (1 << (shift - 1))) as u8;
                let b = ((b_idx << shift) | (1 << (shift - 1))) as u8;
                grid.push(srgb_to_oklab(r, g, b));
            }
        }
    }
    grid
}

/// Build a 4-bit sRGB→centroid cache (16×16×16 = 4KB) using brute-force search.
/// Used for the first k-means iteration (no previous cache to seed from).
fn build_centroid_nn_cache(centroids: &[OKLab], grid_labs: &[OKLab]) -> Vec<u8> {
    let mut cache = vec![0u8; grid_labs.len()];
    for (i, lab) in grid_labs.iter().enumerate() {
        cache[i] = find_nearest(centroids, *lab) as u8;
    }
    cache
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
// Alpha-aware median cut (4D: L, a, b, alpha)
// =====================================================================

/// A box of color+alpha entries for median cut subdivision.
#[derive(Debug, Clone)]
struct ColorBoxA {
    entries: Vec<(OKLabA, f32)>,
}

impl ColorBoxA {
    fn new(entries: Vec<(OKLabA, f32)>) -> Self {
        Self { entries }
    }

    fn total_weight(&self) -> f32 {
        self.entries.iter().map(|(_, w)| w).sum()
    }

    fn ranges(&self) -> (f32, f32, f32, f32) {
        let (mut l_min, mut l_max) = (f32::MAX, f32::MIN);
        let (mut a_min, mut a_max) = (f32::MAX, f32::MIN);
        let (mut b_min, mut b_max) = (f32::MAX, f32::MIN);
        let (mut al_min, mut al_max) = (f32::MAX, f32::MIN);

        for (laba, _) in &self.entries {
            l_min = l_min.min(laba.lab.l);
            l_max = l_max.max(laba.lab.l);
            a_min = a_min.min(laba.lab.a);
            a_max = a_max.max(laba.lab.a);
            b_min = b_min.min(laba.lab.b);
            b_max = b_max.max(laba.lab.b);
            al_min = al_min.min(laba.alpha);
            al_max = al_max.max(laba.alpha);
        }

        (l_max - l_min, a_max - a_min, b_max - b_min, al_max - al_min)
    }

    fn volume(&self) -> f32 {
        let (rl, ra, rb, ral) = self.ranges();
        rl * ra * rb * (ral + 0.01) // small epsilon so zero-alpha-range doesn't collapse volume
    }

    fn priority(&self) -> f32 {
        self.total_weight() * self.volume()
    }

    fn centroid(&self) -> OKLabA {
        let mut l_sum = 0.0f32;
        let mut a_sum = 0.0f32;
        let mut b_sum = 0.0f32;
        let mut al_sum = 0.0f32;
        let mut w_sum = 0.0f32;

        for (laba, w) in &self.entries {
            l_sum += laba.lab.l * w;
            a_sum += laba.lab.a * w;
            b_sum += laba.lab.b * w;
            al_sum += laba.alpha * w;
            w_sum += w;
        }

        if w_sum < 1e-10 {
            return OKLabA::new(0.0, 0.0, 0.0, 1.0);
        }

        OKLabA::new(l_sum / w_sum, a_sum / w_sum, b_sum / w_sum, al_sum / w_sum)
    }

    fn split(mut self) -> (ColorBoxA, ColorBoxA) {
        let (rl, ra, rb, ral) = self.ranges();

        // Choose split axis (4D)
        let axis = if rl >= ra && rl >= rb && rl >= ral {
            0 // L
        } else if ra >= rb && ra >= ral {
            1 // a
        } else if rb >= ral {
            2 // b
        } else {
            3 // alpha
        };

        self.entries.sort_unstable_by(|a, b| {
            let va = match axis {
                0 => a.0.lab.l,
                1 => a.0.lab.a,
                2 => a.0.lab.b,
                _ => a.0.alpha,
            };
            let vb = match axis {
                0 => b.0.lab.l,
                1 => b.0.lab.a,
                2 => b.0.lab.b,
                _ => b.0.alpha,
            };
            va.partial_cmp(&vb).unwrap_or(core::cmp::Ordering::Equal)
        });

        let half_weight = self.total_weight() / 2.0;
        let mut accumulated = 0.0f32;
        let mut split_idx = 1;

        for (i, (_, w)) in self.entries.iter().enumerate() {
            accumulated += w;
            if accumulated >= half_weight && i + 1 < self.entries.len() {
                split_idx = i + 1;
                break;
            }
        }

        split_idx = split_idx.max(1).min(self.entries.len() - 1);

        let right = self.entries.split_off(split_idx);
        (ColorBoxA::new(self.entries), ColorBoxA::new(right))
    }
}

/// Alpha-aware median cut: quantize in 4D (OKLab + alpha) space.
pub fn median_cut_alpha(
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

    let mut boxes = Vec::with_capacity(max_colors);
    boxes.push(ColorBoxA::new(histogram));

    while boxes.len() < max_colors {
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
            break;
        };

        let to_split = boxes.swap_remove(idx);
        let (left, right) = to_split.split();
        boxes.push(left);
        boxes.push(right);
    }

    let mut palette: Vec<OKLabA> = boxes.iter().map(|b| b.centroid()).collect();

    if refine {
        let all_entries: Vec<(OKLabA, f32)> = boxes.into_iter().flat_map(|b| b.entries).collect();
        palette = kmeans_refine_alpha(palette, &all_entries);
    }

    palette
}

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
pub fn refine_against_pixels_alpha(
    mut centroids: Vec<OKLabA>,
    pixels: &[rgb::RGBA<u8>],
    weights: &[f32],
    iterations: usize,
) -> Vec<OKLabA> {
    use crate::oklab::srgb_to_oklab;

    let k = centroids.len();
    if k == 0 {
        return centroids;
    }

    for _ in 0..iterations {
        let mut sums_l = vec![0.0f64; k];
        let mut sums_a = vec![0.0f64; k];
        let mut sums_b = vec![0.0f64; k];
        let mut sums_al = vec![0.0f64; k];
        let mut total_w = vec![0.0f64; k];

        for (pixel, &weight) in pixels.iter().zip(weights.iter()) {
            if pixel.a == 0 {
                continue;
            }
            let lab = srgb_to_oklab(pixel.r, pixel.g, pixel.b);
            let alpha_f = pixel.a as f32 / 255.0;
            let laba = OKLabA::new(lab.l, lab.a, lab.b, alpha_f);
            let nearest = find_nearest_alpha(&centroids, laba);
            let w = weight as f64;
            sums_l[nearest] += lab.l as f64 * w;
            sums_a[nearest] += lab.a as f64 * w;
            sums_b[nearest] += lab.b as f64 * w;
            sums_al[nearest] += alpha_f as f64 * w;
            total_w[nearest] += w;
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
}
