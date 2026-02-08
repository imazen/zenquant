extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

use crate::oklab::{OKLab, oklab_to_srgb};

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
    /// sRGB palette entries, delta-sorted.
    entries_srgb: Vec<[u8; 3]>,
    /// OKLab values for each palette entry (same order as entries_srgb).
    entries_oklab: Vec<OKLab>,
    /// Transparent index, if any.
    transparent_index: Option<u8>,
}

impl Palette {
    /// Build a palette from OKLab centroids with the specified sort strategy.
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
                entries_oklab: Vec::new(),
                transparent_index: if has_transparency { Some(0) } else { None },
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
        let mut entries_oklab: Vec<OKLab> = sorted.iter().map(|(lab, _)| *lab).collect();

        let transparent_index = if has_transparency {
            // Reserve index 0 for transparency
            entries_srgb.insert(0, [0, 0, 0]);
            entries_oklab.insert(0, OKLab::new(0.0, 0.0, 0.0));
            Some(0)
        } else {
            None
        };

        Self {
            entries_srgb,
            entries_oklab,
            transparent_index,
        }
    }

    /// Get sRGB palette entries.
    pub fn entries(&self) -> &[[u8; 3]] {
        &self.entries_srgb
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
    pub fn is_empty(&self) -> bool {
        self.entries_srgb.is_empty()
    }

    /// Find the nearest palette index for an OKLab color (brute force).
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

    /// Find the K nearest palette indices for an OKLab color.
    /// Returns up to K indices sorted by distance (nearest first).
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

    /// Distance from a color to a palette entry.
    pub fn distance_sq(&self, color: OKLab, index: u8) -> f32 {
        color.distance_sq(self.entries_oklab[index as usize])
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
