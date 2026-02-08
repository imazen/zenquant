extern crate alloc;
use alloc::vec::Vec;

use crate::oklab::OKLab;

/// A box of color entries for median cut subdivision.
#[derive(Debug, Clone)]
struct ColorBox {
    entries: Vec<(OKLab, f32)>, // (centroid, accumulated_weight)
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

    /// Volume of this box (product of ranges, or max range for split criterion).
    fn volume(&self) -> f32 {
        let (rl, ra, rb) = self.ranges();
        rl.max(ra).max(rb) // Use max range as volume proxy for split priority
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
/// Takes histogram entries (OKLab centroid, accumulated weight) and produces
/// up to `max_colors` palette centroids in OKLab space.
///
/// If `refine` is true, performs one round of weighted k-means after median cut.
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
        palette = kmeans_refine(palette, &boxes);
    }

    palette
}

/// One round of weighted k-means refinement.
fn kmeans_refine(mut centroids: Vec<OKLab>, boxes: &[ColorBox]) -> Vec<OKLab> {
    // Collect all entries
    let all_entries: Vec<&(OKLab, f32)> = boxes.iter().flat_map(|b| &b.entries).collect();

    // 3 iterations of k-means
    for _ in 0..3 {
        let k = centroids.len();
        let mut sums_l = vec![0.0f32; k];
        let mut sums_a = vec![0.0f32; k];
        let mut sums_b = vec![0.0f32; k];
        let mut weights = vec![0.0f32; k];

        // Assign each entry to nearest centroid
        for &(lab, w) in &all_entries {
            let nearest = centroids
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    lab.distance_sq(**a)
                        .partial_cmp(&lab.distance_sq(**b))
                        .unwrap_or(core::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i)
                .unwrap_or(0);

            sums_l[nearest] += lab.l * w;
            sums_a[nearest] += lab.a * w;
            sums_b[nearest] += lab.b * w;
            weights[nearest] += w;
        }

        // Recompute centroids
        for i in 0..k {
            if weights[i] > 1e-10 {
                centroids[i] = OKLab::new(
                    sums_l[i] / weights[i],
                    sums_a[i] / weights[i],
                    sums_b[i] / weights[i],
                );
            }
        }
    }

    centroids
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
    fn refinement_doesnt_crash() {
        let mut hist = Vec::new();
        for i in 0..50 {
            let l = i as f32 / 50.0;
            hist.push((
                OKLab::new(l, (i as f32).sin() * 0.1, (i as f32).cos() * 0.1),
                1.0,
            ));
        }
        let result = median_cut(hist, 8, true);
        assert_eq!(result.len(), 8);
    }
}
