extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

use crate::oklab::{OKLab, OKLabA, oklab_to_srgb};

/// Strategy for ordering palette entries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaletteSortStrategy {
    /// Greedy nearest-neighbor TSP from darkest entry. Good for delta-coded formats (WebP, JXL).
    DeltaMinimize,
    /// Sort by OKLab L (lightness). Good for PNG scanline filters (sub/up predict neighbors).
    Luminance,
}

/// A quantized color palette with OKLab-space acceleration.
#[derive(Debug, Clone)]
pub struct Palette {
    /// sRGB palette entries, sorted per strategy.
    entries_srgb: Vec<[u8; 3]>,
    /// RGBA palette entries (same order). Alpha from quantization or 255 for opaque.
    entries_rgba: Vec<[u8; 4]>,
    /// OKLab values for each palette entry (same order as entries_srgb).
    entries_oklab: Vec<OKLab>,
    /// Transparent index, if any.
    transparent_index: Option<u8>,
    /// 4-bit sRGB nearest-neighbor cache: 16x16x16 = 4096 entries.
    nn_cache: Option<Vec<u8>>,
    /// Per-entry K nearest palette neighbors (for seeded dither search).
    /// neighbors[i] = up to K closest palette indices to entry i.
    neighbors: Vec<[u8; 16]>,
    neighbor_counts: Vec<u8>,
}

impl Palette {
    /// Build a palette from OKLab centroids with the specified sort strategy.
    #[cfg(test)]
    pub fn from_centroids(centroids: Vec<OKLab>, has_transparency: bool) -> Self {
        Self::from_centroids_sorted(
            centroids,
            has_transparency,
            PaletteSortStrategy::DeltaMinimize,
        )
    }

    /// Build a palette from OKLab centroids, applying the given sort strategy.
    pub fn from_centroids_sorted(
        centroids: Vec<OKLab>,
        has_transparency: bool,
        strategy: PaletteSortStrategy,
    ) -> Self {
        if centroids.is_empty() {
            return Self {
                entries_srgb: Vec::new(),
                entries_rgba: Vec::new(),
                entries_oklab: Vec::new(),
                transparent_index: if has_transparency { Some(0) } else { None },
                nn_cache: None,
                neighbors: Vec::new(),
                neighbor_counts: Vec::new(),
            };
        }

        // Convert to sRGB and keep OKLab paired
        let mut pairs: Vec<(OKLab, [u8; 3])> = centroids
            .into_iter()
            .map(|lab| {
                let (r, g, b) = oklab_to_srgb(lab);
                (lab, [r, g, b])
            })
            .collect();

        let sorted = match strategy {
            PaletteSortStrategy::DeltaMinimize => delta_minimize_sort(&mut pairs),
            PaletteSortStrategy::Luminance => luminance_sort(&mut pairs),
        };

        let mut entries_srgb: Vec<[u8; 3]> = sorted.iter().map(|(_, srgb)| *srgb).collect();
        let mut entries_rgba: Vec<[u8; 4]> = entries_srgb
            .iter()
            .map(|[r, g, b]| [*r, *g, *b, 255])
            .collect();
        let mut entries_oklab: Vec<OKLab> = sorted.iter().map(|(lab, _)| *lab).collect();

        let transparent_index = if has_transparency {
            // Reserve index 0 for transparency
            entries_srgb.insert(0, [0, 0, 0]);
            entries_rgba.insert(0, [0, 0, 0, 0]);
            entries_oklab.insert(0, OKLab::new(0.0, 0.0, 0.0));
            Some(0)
        } else {
            None
        };

        let (neighbors, neighbor_counts) = Self::compute_neighbors(&entries_oklab);

        Self {
            entries_srgb,
            entries_rgba,
            entries_oklab,
            transparent_index,
            nn_cache: None,
            neighbors,
            neighbor_counts,
        }
    }

    /// Build a palette from alpha-aware centroids (4D quantization result).
    ///
    /// Each centroid has its own alpha value which becomes the palette entry's
    /// alpha byte. Sort strategy applies to color channels only.
    pub fn from_centroids_alpha(
        centroids: Vec<OKLabA>,
        has_transparency: bool,
        strategy: PaletteSortStrategy,
    ) -> Self {
        if centroids.is_empty() {
            return Self {
                entries_srgb: Vec::new(),
                entries_rgba: Vec::new(),
                entries_oklab: Vec::new(),
                transparent_index: if has_transparency { Some(0) } else { None },
                nn_cache: None,
                neighbors: Vec::new(),
                neighbor_counts: Vec::new(),
            };
        }

        // Convert to sRGB+alpha, keep OKLab paired
        let pairs: Vec<(OKLab, [u8; 3], u8)> = centroids
            .into_iter()
            .map(|laba| {
                let (r, g, b) = oklab_to_srgb(laba.lab);
                let a = (laba.alpha * 255.0).round().clamp(0.0, 255.0) as u8;
                (laba.lab, [r, g, b], a)
            })
            .collect();

        // Sort by color channels (reusing the sort functions on the OKLab+sRGB part)
        let mut color_pairs: Vec<(OKLab, [u8; 3])> =
            pairs.iter().map(|(lab, srgb, _)| (*lab, *srgb)).collect();

        let sorted_colors = match strategy {
            PaletteSortStrategy::DeltaMinimize => delta_minimize_sort(&mut color_pairs),
            PaletteSortStrategy::Luminance => luminance_sort(&mut color_pairs),
        };

        // Build a mapping from sorted position to original position
        // by matching on OKLab values (they're unique enough)
        let mut used = vec![false; pairs.len()];
        let mut sorted_with_alpha: Vec<(OKLab, [u8; 3], u8)> = Vec::with_capacity(pairs.len());
        for (lab, srgb) in &sorted_colors {
            for (j, (olab, _, alpha)) in pairs.iter().enumerate() {
                if !used[j] && lab.distance_sq(*olab) < 1e-10 {
                    sorted_with_alpha.push((*lab, *srgb, *alpha));
                    used[j] = true;
                    break;
                }
            }
        }

        let mut entries_srgb: Vec<[u8; 3]> = sorted_with_alpha.iter().map(|(_, s, _)| *s).collect();
        let mut entries_rgba: Vec<[u8; 4]> = sorted_with_alpha
            .iter()
            .map(|(_, [r, g, b], a)| [*r, *g, *b, *a])
            .collect();
        let mut entries_oklab: Vec<OKLab> =
            sorted_with_alpha.iter().map(|(lab, _, _)| *lab).collect();

        let transparent_index = if has_transparency {
            entries_srgb.insert(0, [0, 0, 0]);
            entries_rgba.insert(0, [0, 0, 0, 0]);
            entries_oklab.insert(0, OKLab::new(0.0, 0.0, 0.0));
            Some(0)
        } else {
            None
        };

        let (neighbors, neighbor_counts) = Self::compute_neighbors(&entries_oklab);

        Self {
            entries_srgb,
            entries_rgba,
            entries_oklab,
            transparent_index,
            nn_cache: None,
            neighbors,
            neighbor_counts,
        }
    }

    /// Get sRGB palette entries.
    pub fn entries(&self) -> &[[u8; 3]] {
        &self.entries_srgb
    }

    /// Get RGBA palette entries. Alpha is 255 for opaque entries, 0 for the
    /// transparent index (binary transparency), or the quantized alpha value
    /// (full alpha mode).
    pub fn entries_rgba(&self) -> &[[u8; 4]] {
        &self.entries_rgba
    }

    /// Get OKLab palette entries.
    pub fn entries_oklab(&self) -> &[OKLab] {
        &self.entries_oklab
    }

    /// Get transparent index, if any.
    pub fn transparent_index(&self) -> Option<u8> {
        self.transparent_index
    }

    /// Number of palette entries.
    pub fn len(&self) -> usize {
        self.entries_srgb.len()
    }

    /// Whether the palette is empty.
    #[cfg(test)]
    pub fn is_empty(&self) -> bool {
        self.entries_srgb.is_empty()
    }

    /// Find the nearest palette index for an OKLab color.
    /// Uses SoA layout for better cache behavior.
    #[inline]
    pub fn nearest(&self, color: OKLab) -> u8 {
        let start = if self.transparent_index.is_some() {
            1 // skip transparent entry
        } else {
            0
        };
        let mut best_idx = start;
        let mut best_dist = f32::MAX;

        for i in start..self.entries_oklab.len() {
            let d = color.distance_sq(self.entries_oklab[i]);
            if d < best_dist {
                best_dist = d;
                best_idx = i;
            }
        }

        best_idx as u8
    }

    /// Find nearest palette entry using a seed + neighbor refinement.
    /// The seed is a good initial guess (e.g. from NN cache); we only check
    /// the seed and its precomputed K=16 nearest palette neighbors.
    /// Returns (index, distance_sq).
    #[inline]
    pub fn nearest_seeded(&self, color: OKLab, seed: u8) -> u8 {
        let seed_idx = seed as usize;
        let mut best_idx = seed_idx;
        let mut best_dist = color.distance_sq(self.entries_oklab[seed_idx]);

        let count = self.neighbor_counts[seed_idx] as usize;
        let nbrs = &self.neighbors[seed_idx];
        for &nbr in &nbrs[..count] {
            let ni = nbr as usize;
            let d = color.distance_sq(self.entries_oklab[ni]);
            if d < best_dist {
                best_dist = d;
                best_idx = ni;
            }
        }

        best_idx as u8
    }

    /// Find nearest palette entry using two seeds + neighbor refinement.
    /// Checks neighbors of both seeds, deduplicating overlaps.
    /// Use when the error-adjusted pixel may have crossed a palette boundary
    /// relative to the original pixel's cache cell.
    #[inline]
    pub fn nearest_seeded_2(&self, color: OKLab, seed1: u8, seed2: u8) -> u8 {
        // Start with seed1
        let s1 = seed1 as usize;
        let mut best_idx = s1;
        let mut best_dist = color.distance_sq(self.entries_oklab[s1]);

        // Check seed1's neighbors
        let count1 = self.neighbor_counts[s1] as usize;
        let nbrs1 = &self.neighbors[s1];
        for &nbr in &nbrs1[..count1] {
            let ni = nbr as usize;
            let d = color.distance_sq(self.entries_oklab[ni]);
            if d < best_dist {
                best_dist = d;
                best_idx = ni;
            }
        }

        // If seeds differ, also check seed2 and its neighbors
        if seed2 != seed1 {
            let s2 = seed2 as usize;
            let d = color.distance_sq(self.entries_oklab[s2]);
            if d < best_dist {
                best_dist = d;
                best_idx = s2;
            }

            let count2 = self.neighbor_counts[s2] as usize;
            let nbrs2 = &self.neighbors[s2];
            for &nbr in &nbrs2[..count2] {
                let ni = nbr as usize;
                let d = color.distance_sq(self.entries_oklab[ni]);
                if d < best_dist {
                    best_dist = d;
                    best_idx = ni;
                }
            }
        }

        best_idx as u8
    }

    /// Compute K=16 nearest neighbors for each palette entry.
    fn compute_neighbors(entries: &[OKLab]) -> (Vec<[u8; 16]>, Vec<u8>) {
        let n = entries.len();
        let mut neighbors = vec![[0u8; 16]; n];
        let mut counts = vec![0u8; n];
        const K: usize = 16;

        for i in 0..n {
            // Collect all distances to other entries
            let mut dists: Vec<(u8, f32)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| (j as u8, entries[i].distance_sq(entries[j])))
                .collect();
            dists.sort_unstable_by(|a, b| {
                a.1.partial_cmp(&b.1).unwrap_or(core::cmp::Ordering::Equal)
            });
            let count = dists.len().min(K);
            for k in 0..count {
                neighbors[i][k] = dists[k].0;
            }
            counts[i] = count as u8;
        }

        (neighbors, counts)
    }

    /// Find the K nearest palette indices for an OKLab color.
    /// Returns up to K indices sorted by distance (nearest first).
    #[cfg(test)]
    pub fn k_nearest(&self, color: OKLab, k: usize) -> Vec<u8> {
        let start = if self.transparent_index.is_some() {
            1
        } else {
            0
        };

        let mut dists: Vec<(u8, f32)> = (start..self.entries_oklab.len())
            .map(|i| (i as u8, color.distance_sq(self.entries_oklab[i])))
            .collect();

        dists.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(core::cmp::Ordering::Equal));

        dists.iter().take(k).map(|(idx, _)| *idx).collect()
    }

    /// Find the K nearest palette indices without allocation.
    /// Writes indices to `out` and returns how many were written (up to K and out.len()).
    /// Uses partial insertion sort — O(N*K) where N=palette size, K=candidates.
    pub fn k_nearest_into(&self, color: OKLab, out: &mut [u8]) -> usize {
        let k = out.len();
        if k == 0 {
            return 0;
        }

        let start = if self.transparent_index.is_some() {
            1
        } else {
            0
        };

        // Track K best as (index, distance) pairs
        let mut best = [(0u8, f32::MAX); 8]; // Max K=8, stack-allocated
        let slots = k.min(best.len());
        let mut filled = 0usize;

        for i in start..self.entries_oklab.len() {
            let d = color.distance_sq(self.entries_oklab[i]);

            if filled < slots {
                // Still filling — insert in sorted position
                let mut pos = filled;
                while pos > 0 && best[pos - 1].1 > d {
                    best[pos] = best[pos - 1];
                    pos -= 1;
                }
                best[pos] = (i as u8, d);
                filled += 1;
            } else if d < best[slots - 1].1 {
                // Better than worst — insert in sorted position
                let mut pos = slots - 1;
                while pos > 0 && best[pos - 1].1 > d {
                    best[pos] = best[pos - 1];
                    pos -= 1;
                }
                best[pos] = (i as u8, d);
            }
        }

        for i in 0..filled.min(k) {
            out[i] = best[i].0;
        }
        filled.min(k)
    }

    /// Find K nearest palette indices using seed + neighbor refinement.
    /// Searches seed + 16 neighbors (17 candidates), returns top-K.
    /// Much faster than `k_nearest_into()` which scans all 256 entries.
    pub fn k_nearest_seeded(&self, color: OKLab, seed: u8, out: &mut [u8]) -> usize {
        let k = out.len();
        if k == 0 {
            return 0;
        }

        let seed_idx = seed as usize;
        let count = self.neighbor_counts[seed_idx] as usize;
        let nbrs = &self.neighbors[seed_idx];

        // Track K best via insertion sort (same approach as k_nearest_into)
        let mut best = [(0u8, f32::MAX); 8];
        let slots = k.min(best.len());

        // Check seed first
        let d = color.distance_sq(self.entries_oklab[seed_idx]);
        best[0] = (seed, d);
        let mut filled = 1usize;

        // Check all neighbors
        for &nbr in &nbrs[..count] {
            let ni = nbr as usize;
            let d = color.distance_sq(self.entries_oklab[ni]);

            if filled < slots {
                let mut pos = filled;
                while pos > 0 && best[pos - 1].1 > d {
                    best[pos] = best[pos - 1];
                    pos -= 1;
                }
                best[pos] = (nbr, d);
                filled += 1;
            } else if d < best[slots - 1].1 {
                let mut pos = slots - 1;
                while pos > 0 && best[pos - 1].1 > d {
                    best[pos] = best[pos - 1];
                    pos -= 1;
                }
                best[pos] = (nbr, d);
            }
        }

        for i in 0..filled.min(k) {
            out[i] = best[i].0;
        }
        filled.min(k)
    }

    /// Whether the nearest-neighbor cache has been built.
    pub fn has_nn_cache(&self) -> bool {
        self.nn_cache.is_some()
    }

    /// Build the nearest-neighbor cache for fast lookups.
    /// Uses a 4-bit (16x16x16) sRGB grid — 4KB cache that maps each
    /// grid cell center to its nearest palette index. Only used as a seed
    /// for neighbor-refined searches, so lower resolution is fine.
    pub fn build_nn_cache(&mut self) {
        const BITS: usize = 4;
        const SIZE: usize = 1 << BITS; // 16
        const TOTAL: usize = SIZE * SIZE * SIZE; // 4096
        let shift = 8 - BITS; // 4

        let start = if self.transparent_index.is_some() {
            1
        } else {
            0
        };
        let mut cache = vec![0u8; TOTAL];

        for ri in 0..SIZE {
            for gi in 0..SIZE {
                for bi in 0..SIZE {
                    let r = ((ri << shift) | (1 << (shift - 1))) as u8;
                    let g = ((gi << shift) | (1 << (shift - 1))) as u8;
                    let b = ((bi << shift) | (1 << (shift - 1))) as u8;
                    let lab = crate::oklab::srgb_to_oklab(r, g, b);

                    let mut best_idx = start;
                    let mut best_dist = f32::MAX;
                    for i in start..self.entries_oklab.len() {
                        let d = lab.distance_sq(self.entries_oklab[i]);
                        if d < best_dist {
                            best_dist = d;
                            best_idx = i;
                        }
                    }
                    cache[ri * SIZE * SIZE + gi * SIZE + bi] = best_idx as u8;
                }
            }
        }

        self.nn_cache = Some(cache);
    }

    /// Fast nearest-neighbor lookup using the sRGB cache.
    /// Falls back to brute-force if cache isn't built.
    #[inline]
    pub fn nearest_cached(&self, r: u8, g: u8, b: u8) -> u8 {
        if let Some(cache) = &self.nn_cache {
            const SHIFT: usize = 4; // 8 - 4
            const SIZE: usize = 16;
            let ri = (r >> SHIFT) as usize;
            let gi = (g >> SHIFT) as usize;
            let bi = (b >> SHIFT) as usize;
            cache[ri * SIZE * SIZE + gi * SIZE + bi]
        } else {
            let lab = crate::oklab::srgb_to_oklab(r, g, b);
            self.nearest(lab)
        }
    }

    /// Distance from a color to a palette entry.
    pub fn distance_sq(&self, color: OKLab, index: u8) -> f32 {
        color.distance_sq(self.entries_oklab[index as usize])
    }

    /// Find the nearest palette index considering both color and alpha.
    /// Used for alpha-aware quantization paths.
    pub fn nearest_with_alpha(&self, color: OKLab, alpha: f32) -> u8 {
        let start = if self.transparent_index.is_some() {
            1
        } else {
            0
        };

        let query = OKLabA::new(color.l, color.a, color.b, alpha);
        let mut best_idx = start;
        let mut best_dist = f32::MAX;

        for i in start..self.entries_oklab.len() {
            let entry_alpha = self.entries_rgba[i][3] as f32 / 255.0;
            let entry = OKLabA::new(
                self.entries_oklab[i].l,
                self.entries_oklab[i].a,
                self.entries_oklab[i].b,
                entry_alpha,
            );
            let d = query.distance_sq(entry);
            if d < best_dist {
                best_dist = d;
                best_idx = i;
            }
        }

        best_idx as u8
    }
}

/// Delta-minimizing sort: greedy nearest-neighbor TSP.
/// Start from the darkest entry, always jump to the closest unvisited.
fn delta_minimize_sort(pairs: &mut [(OKLab, [u8; 3])]) -> Vec<(OKLab, [u8; 3])> {
    let n = pairs.len();
    if n <= 1 {
        return pairs.to_vec();
    }

    let mut visited = vec![false; n];
    let mut result = Vec::with_capacity(n);

    // Start from darkest (lowest L)
    let start = pairs
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            a.0.l
                .partial_cmp(&b.0.l)
                .unwrap_or(core::cmp::Ordering::Equal)
        })
        .map(|(i, _)| i)
        .unwrap_or(0);

    visited[start] = true;
    result.push(pairs[start]);
    let mut current = start;

    for _ in 1..n {
        let mut best_idx = 0;
        let mut best_dist = f32::MAX;

        for (j, visited_j) in visited.iter().enumerate() {
            if !visited_j {
                let d = pairs[current].0.distance_sq(pairs[j].0);
                if d < best_dist {
                    best_dist = d;
                    best_idx = j;
                }
            }
        }

        visited[best_idx] = true;
        result.push(pairs[best_idx]);
        current = best_idx;
    }

    result
}

/// Luminance sort: order palette entries by OKLab L (lightness), ascending.
/// Good for PNG where scanline filters (sub, up) predict from spatial neighbors,
/// and spatially close pixels tend to have similar lightness.
fn luminance_sort(pairs: &mut [(OKLab, [u8; 3])]) -> Vec<(OKLab, [u8; 3])> {
    pairs.sort_by(|a, b| {
        a.0.l
            .partial_cmp(&b.0.l)
            .unwrap_or(core::cmp::Ordering::Equal)
    });
    pairs.to_vec()
}

/// Reorder palette entries by descending usage frequency, remapping indices.
///
/// Most-common colors get lowest indices, which helps LZW dictionary
/// construction in GIF. The transparent index (if any) stays at index 0.
pub(crate) fn reorder_by_frequency(palette: &Palette, indices: &mut [u8]) -> Palette {
    let transparent_idx = palette.transparent_index();
    let start = if transparent_idx.is_some() { 1 } else { 0 };
    let n = palette.len();

    // Count frequency of each palette index
    let mut freq = vec![0u32; n];
    for &idx in indices.iter() {
        freq[idx as usize] += 1;
    }

    // Build sorted order of non-transparent indices by descending frequency
    let mut order: Vec<usize> = (start..n).collect();
    order.sort_by(|&a, &b| freq[b].cmp(&freq[a]));

    // Build old→new mapping
    let mut old_to_new = vec![0u8; n];
    if transparent_idx.is_some() {
        old_to_new[0] = 0; // transparent stays at 0
    }
    for (new_pos, &old_idx) in order.iter().enumerate() {
        old_to_new[old_idx] = (new_pos + start) as u8;
    }

    // Remap indices
    for idx in indices.iter_mut() {
        *idx = old_to_new[*idx as usize];
    }

    // Build reordered palette
    let mut new_srgb = Vec::with_capacity(n);
    let mut new_rgba = Vec::with_capacity(n);
    let mut new_oklab = Vec::with_capacity(n);

    if transparent_idx.is_some() {
        new_srgb.push(palette.entries_srgb[0]);
        new_rgba.push(palette.entries_rgba[0]);
        new_oklab.push(palette.entries_oklab[0]);
    }
    for &old_idx in &order {
        new_srgb.push(palette.entries_srgb[old_idx]);
        new_rgba.push(palette.entries_rgba[old_idx]);
        new_oklab.push(palette.entries_oklab[old_idx]);
    }

    let (neighbors, neighbor_counts) = Palette::compute_neighbors(&new_oklab);

    Palette {
        entries_srgb: new_srgb,
        entries_rgba: new_rgba,
        entries_oklab: new_oklab,
        transparent_index: transparent_idx,
        nn_cache: None,
        neighbors,
        neighbor_counts,
    }
}

/// Compute sum of squared index deltas — metric for compression friendliness.
pub fn index_delta_score(indices: &[u8]) -> u64 {
    if indices.len() < 2 {
        return 0;
    }
    indices
        .windows(2)
        .map(|w| {
            let delta = (w[1] as i16 - w[0] as i16).unsigned_abs() as u64;
            delta * delta
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_palette() {
        let p = Palette::from_centroids(Vec::new(), false);
        assert!(p.is_empty());
        assert_eq!(p.len(), 0);
    }

    #[test]
    fn single_entry() {
        let p = Palette::from_centroids(vec![OKLab::new(0.5, 0.0, 0.0)], false);
        assert_eq!(p.len(), 1);
        assert_eq!(p.nearest(OKLab::new(0.5, 0.0, 0.0)), 0);
    }

    #[test]
    fn delta_sort_produces_smooth_ordering() {
        // Create entries scattered across lightness range
        let centroids: Vec<OKLab> = (0..8)
            .map(|i| {
                // Deliberately unsorted
                let l = match i {
                    0 => 0.8,
                    1 => 0.2,
                    2 => 0.6,
                    3 => 0.1,
                    4 => 0.9,
                    5 => 0.4,
                    6 => 0.3,
                    7 => 0.7,
                    _ => unreachable!(),
                };
                OKLab::new(l, 0.0, 0.0)
            })
            .collect();

        let palette = Palette::from_centroids(centroids, false);
        let labs = palette.entries_oklab();

        // After delta sort, adjacent entries should be close in lightness
        let mut total_delta = 0.0f32;
        for i in 1..labs.len() {
            total_delta += (labs[i].l - labs[i - 1].l).abs();
        }

        // Worst case (random) delta sum would be much higher than sorted
        // Sorted should traverse the range approximately once: ~0.8
        assert!(
            total_delta < 1.5,
            "delta sort produced high total delta: {total_delta}"
        );
    }

    #[test]
    fn nearest_finds_closest() {
        let centroids = vec![
            OKLab::new(0.2, 0.0, 0.0),
            OKLab::new(0.5, 0.0, 0.0),
            OKLab::new(0.8, 0.0, 0.0),
        ];
        let palette = Palette::from_centroids(centroids, false);

        // Query near 0.2 → should find the dark entry
        let idx = palette.nearest(OKLab::new(0.19, 0.0, 0.0));
        let lab = palette.entries_oklab()[idx as usize];
        assert!(
            (lab.l - 0.2).abs() < 0.05,
            "expected entry near L=0.2, got L={}",
            lab.l
        );
    }

    #[test]
    fn transparency_reserves_index_zero() {
        let centroids = vec![OKLab::new(0.5, 0.0, 0.0), OKLab::new(0.8, 0.0, 0.0)];
        let palette = Palette::from_centroids(centroids, true);
        assert_eq!(palette.len(), 3); // 2 + transparent
        assert_eq!(palette.transparent_index(), Some(0));
    }

    #[test]
    fn k_nearest_returns_sorted() {
        let centroids = vec![
            OKLab::new(0.1, 0.0, 0.0),
            OKLab::new(0.5, 0.0, 0.0),
            OKLab::new(0.9, 0.0, 0.0),
        ];
        let palette = Palette::from_centroids(centroids, false);
        let query = OKLab::new(0.5, 0.0, 0.0);
        let k = palette.k_nearest(query, 3);
        assert_eq!(k.len(), 3);
        // First result should be closest to 0.5
        let first_lab = palette.entries_oklab()[k[0] as usize];
        assert!(
            (first_lab.l - 0.5).abs() < 0.05,
            "expected nearest to L=0.5, got L={}",
            first_lab.l
        );
    }
}
