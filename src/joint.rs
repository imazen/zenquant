//! Joint deflate+quantization optimization for PNG.
//!
//! After normal quantization produces indices and AQ weights, this module
//! post-processes them: for each pixel with multiple perceptually acceptable
//! palette candidates, it picks the candidate that compresses best under
//! PNG scanline filters.
//!
//! Single-pass approach with a vendored LZ77 predictor:
//!  1. Pre-score all 5 PNG filters per row → keep top K=3.
//!  2. For each top-K filter, run DP index optimization.
//!  3. Feed (filter, optimized indices) through the predictor → size estimate.
//!  4. Row-pair lookahead defers decisions using predictor snapshots.
//!
//! The predictor uses Huffman code length estimation over LZ77 match frequencies,
//! which closely models actual deflate output for relative ranking.

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

use crate::joint_predict::{Predictor, PredictorSnapshot};
use crate::oklab::OKLab;
use crate::palette::Palette;

/// Maximum candidates per pixel. Higher = more options but slower.
const MAX_CANDIDATES: usize = 6;

/// Number of top-scoring filters to fully evaluate per row.
const TOP_K_FILTERS: usize = 3;

// ── PNG bit-depth and packing (duplicated from zenpng to avoid cross-dep) ──

fn select_bit_depth(n_colors: usize) -> u8 {
    if n_colors <= 2 {
        1
    } else if n_colors <= 4 {
        2
    } else if n_colors <= 16 {
        4
    } else {
        8
    }
}

fn packed_row_bytes(width: usize, bit_depth: u8) -> usize {
    match bit_depth {
        8 => width,
        4 => width.div_ceil(2),
        2 => width.div_ceil(4),
        1 => width.div_ceil(8),
        _ => width,
    }
}

/// Pack a single row of indices into bit-packed PNG format.
fn pack_row(indices: &[u8], bit_depth: u8, out: &mut [u8]) {
    if bit_depth == 8 {
        out[..indices.len()].copy_from_slice(indices);
        return;
    }
    let ppb = 8 / bit_depth as usize;
    let mask = (1u8 << bit_depth) - 1;
    for b in out.iter_mut() {
        *b = 0;
    }
    for (x, &idx) in indices.iter().enumerate() {
        let byte_pos = x / ppb;
        let bit_offset = (ppb - 1 - x % ppb) * bit_depth as usize;
        out[byte_pos] |= (idx & mask) << bit_offset;
    }
}

// ── PNG filter math (bpp=1 for indexed) ──

/// Standard PNG Paeth predictor.
#[inline]
fn paeth_predictor(a: u8, b: u8, c: u8) -> u8 {
    let a_i = a as i16;
    let b_i = b as i16;
    let c_i = c as i16;
    let p = a_i + b_i - c_i;
    let pa = (p - a_i).unsigned_abs();
    let pb = (p - b_i).unsigned_abs();
    let pc = (p - c_i).unsigned_abs();
    if pa <= pb && pa <= pc {
        a
    } else if pb <= pc {
        b
    } else {
        c
    }
}

/// Compute the filtered byte value for a given filter type (bpp=1).
#[inline]
fn compute_filtered_byte(filter: u8, raw: u8, left: u8, above: u8, above_left: u8) -> u8 {
    match filter {
        0 => raw,                                                        // None
        1 => raw.wrapping_sub(left),                                     // Sub
        2 => raw.wrapping_sub(above),                                    // Up
        3 => raw.wrapping_sub(((left as u16 + above as u16) / 2) as u8), // Average
        4 => raw.wrapping_sub(paeth_predictor(left, above, above_left)), // Paeth
        _ => raw,
    }
}

/// Apply a PNG filter to a packed row, writing into a pre-allocated slice.
/// Returns the number of bytes written (row.len() + 1).
fn apply_filter_bpp1_into(filter: u8, row: &[u8], prev_row: &[u8], out: &mut [u8]) -> usize {
    out[0] = filter;
    for (x, &raw) in row.iter().enumerate() {
        let left = if x > 0 { row[x - 1] } else { 0 };
        let above = prev_row[x];
        let above_left = if x > 0 { prev_row[x - 1] } else { 0 };
        out[1 + x] = compute_filtered_byte(filter, raw, left, above, above_left);
    }
    row.len() + 1
}

// ── Top-K filter pre-selection ──

/// Cheap heuristic score for a filter (no LZ77, just byte statistics).
/// Higher score = more compressible.
fn filter_prescore(filter: u8, packed: &[u8], prev: &[u8]) -> u32 {
    let mut zeros = 0u32;
    let mut near_zero = 0u32;

    for x in 0..packed.len() {
        let left = if x > 0 { packed[x - 1] } else { 0 };
        let above = prev[x];
        let above_left = if x > 0 { prev[x - 1] } else { 0 };
        let filtered = compute_filtered_byte(filter, packed[x], left, above, above_left);
        if filtered == 0 {
            zeros += 1;
        }
        // Symmetric residual: bytes near 0 or 255 compress well
        let dist_from_zero = filtered.min(filtered.wrapping_neg());
        if dist_from_zero <= 2 {
            near_zero += 1;
        }
    }

    // Weight zeros heavily, near-zero bytes less so
    zeros * 256 + near_zero * 64
}

/// Select top-K filters by pre-scoring. Returns up to TOP_K_FILTERS filter types.
fn select_top_k_filters(packed: &[u8], prev: &[u8]) -> [u8; TOP_K_FILTERS] {
    let mut scores: [(u32, u8); 5] = [
        (filter_prescore(0, packed, prev), 0),
        (filter_prescore(1, packed, prev), 1),
        (filter_prescore(2, packed, prev), 2),
        (filter_prescore(3, packed, prev), 3),
        (filter_prescore(4, packed, prev), 4),
    ];

    // Sort descending by score (higher = better)
    scores.sort_unstable_by(|a, b| b.0.cmp(&a.0));

    [scores[0].1, scores[1].1, scores[2].1]
}

// ── Candidate building ──

/// Per-pixel candidate set: up to MAX_CANDIDATES palette indices within tolerance.
struct Candidates {
    /// For each pixel, up to MAX_CANDIDATES palette indices.
    indices: Vec<[u8; MAX_CANDIDATES]>,
    /// How many candidates each pixel has.
    counts: Vec<u8>,
}

/// Build per-pixel candidate sets.
///
/// For each pixel, finds palette entries within OKLab distance threshold.
/// Tolerance is modulated by AQ weight: smooth areas (weight ~1.0) get tight
/// tolerance, textured areas (weight ~0.1) get ~10x more freedom.
fn build_candidates(
    pixel_oklab: &[OKLab],
    weights: &[f32],
    palette: &Palette,
    initial_indices: &[u8],
    base_tolerance: f32,
) -> Candidates {
    let n = pixel_oklab.len();
    let mut indices = vec![[0u8; MAX_CANDIDATES]; n];
    let mut counts = vec![1u8; n];

    let base_tol_sq = base_tolerance * base_tolerance;

    for i in 0..n {
        let seed = initial_indices[i];
        indices[i][0] = seed;

        let w = weights[i].max(0.01);
        let tol_sq = base_tol_sq / (w * w);

        let mut buf = [0u8; MAX_CANDIDATES];
        let found = palette.k_nearest_seeded(pixel_oklab[i], seed, &mut buf);

        let mut count = 0usize;
        for &cand in &buf[..found] {
            let dist = palette.distance_sq(pixel_oklab[i], cand);
            if dist <= tol_sq && count < MAX_CANDIDATES {
                indices[i][count] = cand;
                count += 1;
            }
        }

        if count == 0 {
            indices[i][0] = seed;
            count = 1;
        } else {
            let has_seed = indices[i][..count].contains(&seed);
            if !has_seed && count < MAX_CANDIDATES {
                indices[i][count] = seed;
                count += 1;
            }
        }

        counts[i] = count as u8;
    }

    Candidates { indices, counts }
}

// ── DP row optimization ──

/// Symmetric distance from zero (wrapping): correlates with compressibility.
/// Zero and near-zero filtered bytes get the shortest Huffman codes.
#[inline]
fn sym_dist(b: u8) -> u16 {
    b.min(b.wrapping_neg()) as u16
}

/// For a given filter type, select the best candidate index per pixel using
/// dynamic programming over left-neighbor dependencies.
///
/// Filters with left-dependency (Sub=1, Average=3, Paeth=4) use full DP:
/// state = candidate chosen at previous pixel, cost = sym_dist of filtered byte.
/// Filters without left-dependency (None=0, Up=2) pick per-pixel minimum.
#[allow(clippy::too_many_arguments)]
fn dp_row_optimize(
    width: usize,
    filter: u8,
    candidates: &Candidates,
    row_start: usize,
    above_packed: &[u8],
    bit_depth: u8,
    row_bytes: usize,
) -> Vec<u8> {
    let mut result_indices = vec![0u8; width];

    // For filters without left-dependency, each pixel is independent
    #[allow(clippy::needless_range_loop)]
    if filter == 0 || filter == 2 {
        let mut packed_row = vec![0u8; row_bytes];
        for x in 0..width {
            let pixel_idx = row_start + x;
            let n_cands = candidates.counts[pixel_idx] as usize;
            let mut best_c = 0usize;
            let mut best_cost = u16::MAX;

            for c in 0..n_cands {
                let cand_idx = candidates.indices[pixel_idx][c];
                let packed_byte = pack_candidate(bit_depth, &packed_row, x, cand_idx);
                let byte_pos = if bit_depth == 8 {
                    x
                } else {
                    x / (8 / bit_depth as usize)
                };
                let above = if !above_packed.is_empty() {
                    above_packed[byte_pos]
                } else {
                    0
                };
                let filtered = compute_filtered_byte(filter, packed_byte, 0, above, 0);
                let cost = sym_dist(filtered);
                if cost < best_cost {
                    best_cost = cost;
                    best_c = c;
                }
            }

            let chosen = candidates.indices[pixel_idx][best_c];
            result_indices[x] = chosen;
            write_packed(bit_depth, &mut packed_row, x, chosen);
        }
        return result_indices;
    }

    // ── DP for filters with left-dependency (Sub=1, Average=3, Paeth=4) ──

    if bit_depth == 8 {
        dp_row_optimize_8bit(
            width,
            filter,
            candidates,
            row_start,
            above_packed,
            &mut result_indices,
        );
    } else {
        dp_row_optimize_subbyte(
            width,
            filter,
            candidates,
            row_start,
            above_packed,
            bit_depth,
            row_bytes,
            &mut result_indices,
        );
    }

    result_indices
}

/// Pack a candidate index into a byte value for the given pixel position.
/// Returns the full packed byte that would result from placing this candidate.
#[inline]
fn pack_candidate(bit_depth: u8, packed_row: &[u8], x: usize, cand_idx: u8) -> u8 {
    if bit_depth == 8 {
        return cand_idx;
    }
    let ppb = 8 / bit_depth as usize;
    let byte_pos = x / ppb;
    let bit_offset = (ppb - 1 - x % ppb) * bit_depth as usize;
    let mask = (1u8 << bit_depth) - 1;
    let mut bv = packed_row[byte_pos];
    bv &= !(mask << bit_offset);
    bv |= (cand_idx & mask) << bit_offset;
    bv
}

/// Write a candidate index into the packed row at pixel position x.
#[inline]
fn write_packed(bit_depth: u8, packed_row: &mut [u8], x: usize, cand_idx: u8) {
    if bit_depth == 8 {
        packed_row[x] = cand_idx;
        return;
    }
    let ppb = 8 / bit_depth as usize;
    let byte_pos = x / ppb;
    let bit_offset = (ppb - 1 - x % ppb) * bit_depth as usize;
    let mask = (1u8 << bit_depth) - 1;
    packed_row[byte_pos] &= !(mask << bit_offset);
    packed_row[byte_pos] |= (cand_idx & mask) << bit_offset;
}

/// DP for 8-bit depth (most common path). Each pixel = one byte, no packing.
#[allow(clippy::needless_range_loop)]
fn dp_row_optimize_8bit(
    width: usize,
    filter: u8,
    candidates: &Candidates,
    row_start: usize,
    above_packed: &[u8],
    result: &mut [u8],
) {
    // dp[c] = minimum cumulative cost choosing candidate c at the current pixel
    // back[x][c] = which candidate was chosen at pixel x-1 to achieve dp[c]
    let mut dp_cur = [u32::MAX; MAX_CANDIDATES];
    let mut dp_prev = [0u32; MAX_CANDIDATES];
    // Backpointers: for each pixel x and candidate c, store the prev candidate index
    let mut backptrs: Vec<[u8; MAX_CANDIDATES]> = vec![[0; MAX_CANDIDATES]; width];

    // Pixel 0: no left neighbor, so left=0, above_left=0
    let above_0 = if !above_packed.is_empty() {
        above_packed[0]
    } else {
        0
    };
    let n0 = candidates.counts[row_start] as usize;
    for c in 0..n0 {
        let raw = candidates.indices[row_start][c];
        let filtered = compute_filtered_byte(filter, raw, 0, above_0, 0);
        dp_prev[c] = sym_dist(filtered) as u32;
    }

    // Forward pass: pixels 1..width-1
    for x in 1..width {
        let pixel_idx = row_start + x;
        let n_cands = candidates.counts[pixel_idx] as usize;
        let n_prev = candidates.counts[pixel_idx - 1] as usize;
        let above = if !above_packed.is_empty() {
            above_packed[x]
        } else {
            0
        };
        let above_left = if x > 0 && !above_packed.is_empty() {
            above_packed[x - 1]
        } else {
            0
        };

        for c in 0..n_cands {
            let raw = candidates.indices[pixel_idx][c];
            let mut best_cost = u32::MAX;
            let mut best_prev = 0u8;

            for pc in 0..n_prev {
                if dp_prev[pc] == u32::MAX {
                    continue;
                }
                let left = candidates.indices[pixel_idx - 1][pc];
                let filtered = compute_filtered_byte(filter, raw, left, above, above_left);
                let cost = dp_prev[pc] + sym_dist(filtered) as u32;
                if cost < best_cost {
                    best_cost = cost;
                    best_prev = pc as u8;
                }
            }

            dp_cur[c] = best_cost;
            backptrs[x][c] = best_prev;
        }
        dp_cur[n_cands..MAX_CANDIDATES].fill(u32::MAX);

        dp_prev = dp_cur;
        dp_cur = [u32::MAX; MAX_CANDIDATES];
    }

    // Backward pass: find minimum-cost final candidate, trace back
    let last_idx = row_start + width - 1;
    let n_last = candidates.counts[last_idx] as usize;
    let mut best_c = 0usize;
    let mut best_cost = u32::MAX;
    for c in 0..n_last {
        if dp_prev[c] < best_cost {
            best_cost = dp_prev[c];
            best_c = c;
        }
    }

    result[width - 1] = candidates.indices[last_idx][best_c];
    let mut cur_c = best_c;
    for x in (1..width).rev() {
        let prev_c = backptrs[x][cur_c] as usize;
        result[x - 1] = candidates.indices[row_start + x - 1][prev_c];
        cur_c = prev_c;
    }
}

/// DP for sub-byte bit depths (<8). Only boundary pixels participate in DP;
/// non-boundary pixels use their initial candidate.
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
fn dp_row_optimize_subbyte(
    width: usize,
    filter: u8,
    candidates: &Candidates,
    row_start: usize,
    above_packed: &[u8],
    bit_depth: u8,
    row_bytes: usize,
    result: &mut [u8],
) {
    let ppb = 8 / bit_depth as usize;

    // Collect boundary pixel positions (last pixel in each packed byte)
    let mut boundaries: Vec<usize> = Vec::new();
    for x in 0..width {
        let is_boundary = (x % ppb == ppb - 1) || x == width - 1;
        if is_boundary {
            boundaries.push(x);
        }
    }

    if boundaries.is_empty() {
        // Shouldn't happen, but safety: just use initial indices
        for x in 0..width {
            result[x] = candidates.indices[row_start + x][0];
        }
        return;
    }

    // For sub-byte DP, state is the packed byte value produced by the boundary
    // pixel's candidate choice. We build a packed row as we go.
    let mut packed_row = vec![0u8; row_bytes];

    // First, fill in all non-boundary pixels with initial candidates
    for x in 0..width {
        let is_boundary = (x % ppb == ppb - 1) || x == width - 1;
        if !is_boundary {
            let initial = candidates.indices[row_start + x][0];
            result[x] = initial;
            write_packed(bit_depth, &mut packed_row, x, initial);
        }
    }

    // DP over boundary pixels
    let mut dp_prev = [u32::MAX; MAX_CANDIDATES];
    let mut dp_cur = [u32::MAX; MAX_CANDIDATES];
    let mut backptrs: Vec<[u8; MAX_CANDIDATES]> = vec![[0; MAX_CANDIDATES]; boundaries.len()];

    // First boundary pixel
    let bx0 = boundaries[0];
    let byte_pos0 = bx0 / ppb;
    let above_0 = if !above_packed.is_empty() {
        above_packed[byte_pos0]
    } else {
        0
    };
    let above_left_0 = if byte_pos0 > 0 && !above_packed.is_empty() {
        above_packed[byte_pos0 - 1]
    } else {
        0
    };
    let left_0 = if byte_pos0 > 0 {
        packed_row[byte_pos0 - 1]
    } else {
        0
    };
    let n0 = candidates.counts[row_start + bx0] as usize;

    for c in 0..n0 {
        let cand = candidates.indices[row_start + bx0][c];
        let packed_byte = pack_candidate(bit_depth, &packed_row, bx0, cand);
        let filtered = compute_filtered_byte(filter, packed_byte, left_0, above_0, above_left_0);
        dp_prev[c] = sym_dist(filtered) as u32;
    }

    // Forward pass over remaining boundaries
    for bi in 1..boundaries.len() {
        let bx = boundaries[bi];
        let byte_pos = bx / ppb;
        let n_cands = candidates.counts[row_start + bx] as usize;
        let n_prev_cands = candidates.counts[row_start + boundaries[bi - 1]] as usize;
        let above = if !above_packed.is_empty() {
            above_packed[byte_pos]
        } else {
            0
        };
        let above_left = if byte_pos > 0 && !above_packed.is_empty() {
            above_packed[byte_pos - 1]
        } else {
            0
        };

        for c in 0..n_cands {
            let cand = candidates.indices[row_start + bx][c];
            let packed_byte = pack_candidate(bit_depth, &packed_row, bx, cand);
            let mut best_cost = u32::MAX;
            let mut best_prev = 0u8;

            for pc in 0..n_prev_cands {
                if dp_prev[pc] == u32::MAX {
                    continue;
                }
                // The left byte is determined by the previous boundary's candidate
                let prev_cand = candidates.indices[row_start + boundaries[bi - 1]][pc];
                let left = pack_candidate(bit_depth, &packed_row, boundaries[bi - 1], prev_cand);
                let filtered = compute_filtered_byte(filter, packed_byte, left, above, above_left);
                let cost = dp_prev[pc] + sym_dist(filtered) as u32;
                if cost < best_cost {
                    best_cost = cost;
                    best_prev = pc as u8;
                }
            }

            dp_cur[c] = best_cost;
            backptrs[bi][c] = best_prev;
        }
        for c in n_cands..MAX_CANDIDATES {
            dp_cur[c] = u32::MAX;
        }

        dp_prev = dp_cur;
        dp_cur = [u32::MAX; MAX_CANDIDATES];
    }

    // Backward pass: find best final boundary candidate, trace back
    let last_bi = boundaries.len() - 1;
    let last_bx = boundaries[last_bi];
    let n_last = candidates.counts[row_start + last_bx] as usize;
    let mut best_c = 0usize;
    let mut best_cost = u32::MAX;
    for c in 0..n_last {
        if dp_prev[c] < best_cost {
            best_cost = dp_prev[c];
            best_c = c;
        }
    }

    // Write boundary choices backward
    let mut cur_c = best_c;
    for bi in (0..=last_bi).rev() {
        let bx = boundaries[bi];
        let chosen = candidates.indices[row_start + bx][cur_c];
        result[bx] = chosen;
        write_packed(bit_depth, &mut packed_row, bx, chosen);
        if bi > 0 {
            cur_c = backptrs[bi][cur_c] as usize;
        }
    }
}

// ── Top-level optimizer ──

/// Joint deflate+quantization optimization for RGB images.
///
/// Returns optimized indices. The downstream PNG encoder handles filter
/// selection and compression through its standard pipeline.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn optimize_rgb(
    pixels: &[rgb::RGB<u8>],
    width: usize,
    height: usize,
    weights: &[f32],
    palette: &Palette,
    initial_indices: &[u8],
    _deflate_effort: u32,
    base_tolerance: f32,
) -> Vec<u8> {
    let pixel_oklab = crate::simd::batch_srgb_to_oklab_vec(pixels);

    optimize_inner(
        &pixel_oklab,
        width,
        height,
        weights,
        palette,
        initial_indices,
        None,
        base_tolerance,
    )
}

/// Joint deflate+quantization optimization for RGB with pre-computed OKLab.
#[allow(clippy::too_many_arguments)]
pub(crate) fn optimize_rgb_with_labs(
    pixel_oklab: &[OKLab],
    width: usize,
    height: usize,
    weights: &[f32],
    palette: &Palette,
    initial_indices: &[u8],
    _deflate_effort: u32,
    base_tolerance: f32,
) -> Vec<u8> {
    optimize_inner(
        pixel_oklab,
        width,
        height,
        weights,
        palette,
        initial_indices,
        None,
        base_tolerance,
    )
}

/// Joint deflate+quantization optimization for RGBA with pre-computed OKLab.
#[allow(clippy::too_many_arguments)]
pub(crate) fn optimize_rgba_with_labs(
    pixel_oklab: &[OKLab],
    width: usize,
    height: usize,
    weights: &[f32],
    palette: &Palette,
    initial_indices: &[u8],
    _deflate_effort: u32,
    base_tolerance: f32,
) -> Vec<u8> {
    let transparent_index = palette.transparent_index();

    optimize_inner(
        pixel_oklab,
        width,
        height,
        weights,
        palette,
        initial_indices,
        transparent_index,
        base_tolerance,
    )
}

/// Core single-pass optimization.
///
/// Merges filter selection and index optimization into one pass using the
/// vendored LZ77 predictor. For each row:
/// 1. Pre-score 5 filters → keep top K=3
/// 2. For each top-K filter: run DP index optimization → get optimized indices
/// 3. Feed (filter, optimized indices) through predictor → size estimate
/// 4. Row-pair lookahead defers decisions using predictor snapshots
#[allow(clippy::too_many_arguments)]
fn optimize_inner(
    pixel_oklab: &[OKLab],
    width: usize,
    height: usize,
    weights: &[f32],
    palette: &Palette,
    initial_indices: &[u8],
    transparent_index: Option<u8>,
    base_tolerance: f32,
) -> Vec<u8> {
    let n_colors = palette.len();
    let bit_depth = select_bit_depth(n_colors);
    let row_bytes = packed_row_bytes(width, bit_depth);
    let filtered_row_size = row_bytes + 1;

    // Build candidate sets
    let mut candidates = build_candidates(
        pixel_oklab,
        weights,
        palette,
        initial_indices,
        base_tolerance,
    );

    // Pin transparent pixels to their index
    if let Some(ti) = transparent_index {
        for (i, &idx) in initial_indices.iter().enumerate() {
            if idx == ti {
                candidates.indices[i] = [ti; MAX_CANDIDATES];
                candidates.counts[i] = 1;
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Single-pass: merged filter selection + index optimization
    // with row-pair lookahead via predictor snapshots
    // ═══════════════════════════════════════════════════════════════════════

    let mut optimized_indices = initial_indices.to_vec();
    let zero_row = vec![0u8; row_bytes];

    // Working buffers
    let mut filtered_buf = vec![0u8; filtered_row_size];

    // The predictor tracks cumulative LZ77 state
    let mut predictor = Predictor::new();

    // Committed state: the packed row for the last committed row (used as "above")
    let mut committed_packed = vec![0u8; row_bytes];
    let mut committed_size: usize = 0;

    // Pending row-pair lookahead state
    struct Pending {
        // Snapshot where previous row used INITIAL indices
        snap_init: PredictorSnapshot,
        packed_init: Vec<u8>,
        size_init: usize,
        // Snapshot where previous row used OPTIMIZED indices
        snap_opt: PredictorSnapshot,
        packed_opt: Vec<u8>,
        size_opt: usize,
        // Pending row info
        pending_y: usize,
        opt_indices: Vec<u8>,
    }

    let mut pending: Option<Pending> = None;

    for y in 0..height {
        // Compact predictor buffer to bound memory (keeps last 32KB window).
        // Skip when pending snapshots exist — they reference data that compact
        // would trim, causing restore() to produce inconsistent state.
        if pending.is_none() {
            predictor.compact();
        }

        let row_start = y * width;

        // Pack initial row
        let initial_row = &initial_indices[row_start..row_start + width];
        let mut packed_init = vec![0u8; row_bytes];
        pack_row(initial_row, bit_depth, &mut packed_init);

        // Determine the "above" row for filter context
        let above = if y > 0 {
            &committed_packed[..]
        } else {
            &zero_row[..]
        };

        // Top-K filter pre-selection
        let top_filters = select_top_k_filters(&packed_init, above);

        // Evaluate all top-K filters with optimized indices
        // Find the best (filter, indices, packed) combination
        struct FilterResult {
            filter: u8,
            indices: Vec<u8>,
            packed: Vec<u8>,
            size: usize,
        }

        let snap_before = predictor.snapshot();
        let mut best_init: Option<FilterResult> = None;
        let mut best_opt: Option<FilterResult> = None;

        for &f in &top_filters {
            // Evaluate initial indices with this filter
            predictor.restore(&snap_before);
            let n = apply_filter_bpp1_into(f, &packed_init, above, &mut filtered_buf);
            let size_init = predictor.feed_row(&filtered_buf[..n]);

            if best_init.as_ref().is_none_or(|b| size_init < b.size) {
                best_init = Some(FilterResult {
                    filter: f,
                    indices: initial_row.to_vec(),
                    packed: packed_init.clone(),
                    size: size_init,
                });
            }

            // Evaluate optimized indices with this filter
            let opt_row = dp_row_optimize(
                width,
                f,
                &candidates,
                row_start,
                above,
                bit_depth,
                row_bytes,
            );
            let mut packed_opt = vec![0u8; row_bytes];
            pack_row(&opt_row, bit_depth, &mut packed_opt);

            if packed_opt != packed_init {
                predictor.restore(&snap_before);
                let n = apply_filter_bpp1_into(f, &packed_opt, above, &mut filtered_buf);
                let size_opt = predictor.feed_row(&filtered_buf[..n]);

                if best_opt.as_ref().is_none_or(|b| size_opt < b.size) {
                    best_opt = Some(FilterResult {
                        filter: f,
                        indices: opt_row,
                        packed: packed_opt,
                        size: size_opt,
                    });
                }
            }
        }

        let bi = best_init.unwrap(); // always exists
        let indices_differ = best_opt.as_ref().is_some_and(|bo| bo.packed != bi.packed);

        if let Some(pend) = pending.take() {
            // Resolve the pending row by evaluating current row under both paths

            // Current row under prev=initial path
            predictor.restore_owned(pend.snap_init);
            let above_init = &pend.packed_init;
            let top_init = select_top_k_filters(&packed_init, above_init);

            let mut best_on_init: Option<(usize, Vec<u8>, Vec<u8>, u8)> = None;
            let snap_init_base = predictor.snapshot();

            for &f in &top_init {
                // Initial indices
                predictor.restore(&snap_init_base);
                let n = apply_filter_bpp1_into(f, &packed_init, above_init, &mut filtered_buf);
                let s = predictor.feed_row(&filtered_buf[..n]);
                if best_on_init.as_ref().is_none_or(|b| s < b.0) {
                    best_on_init = Some((s, initial_row.to_vec(), packed_init.clone(), f));
                }

                // Optimized indices (re-optimize for this context)
                let opt = dp_row_optimize(
                    width,
                    f,
                    &candidates,
                    row_start,
                    above_init,
                    bit_depth,
                    row_bytes,
                );
                let mut po = vec![0u8; row_bytes];
                pack_row(&opt, bit_depth, &mut po);
                if po != packed_init {
                    predictor.restore(&snap_init_base);
                    let n = apply_filter_bpp1_into(f, &po, above_init, &mut filtered_buf);
                    let s = predictor.feed_row(&filtered_buf[..n]);
                    if best_on_init.as_ref().is_none_or(|b| s < b.0) {
                        best_on_init = Some((s, opt, po, f));
                    }
                }
            }

            // Current row under prev=optimized path
            predictor.restore_owned(pend.snap_opt);
            let above_opt = &pend.packed_opt;
            let top_opt = select_top_k_filters(&packed_init, above_opt);

            let mut best_on_opt: Option<(usize, Vec<u8>, Vec<u8>, u8)> = None;
            let snap_opt_base = predictor.snapshot();

            for &f in &top_opt {
                predictor.restore(&snap_opt_base);
                let n = apply_filter_bpp1_into(f, &packed_init, above_opt, &mut filtered_buf);
                let s = predictor.feed_row(&filtered_buf[..n]);
                if best_on_opt.as_ref().is_none_or(|b| s < b.0) {
                    best_on_opt = Some((s, initial_row.to_vec(), packed_init.clone(), f));
                }

                let opt = dp_row_optimize(
                    width,
                    f,
                    &candidates,
                    row_start,
                    above_opt,
                    bit_depth,
                    row_bytes,
                );
                let mut po = vec![0u8; row_bytes];
                pack_row(&opt, bit_depth, &mut po);
                if po != packed_init {
                    predictor.restore(&snap_opt_base);
                    let n = apply_filter_bpp1_into(f, &po, above_opt, &mut filtered_buf);
                    let s = predictor.feed_row(&filtered_buf[..n]);
                    if best_on_opt.as_ref().is_none_or(|b| s < b.0) {
                        best_on_opt = Some((s, opt, po, f));
                    }
                }
            }

            // Compare: (prev=init total, prev=opt total) to decide both rows
            let total_init = pend.size_init + best_on_init.as_ref().map_or(usize::MAX, |b| b.0);
            let total_opt = pend.size_opt + best_on_opt.as_ref().map_or(usize::MAX, |b| b.0);

            let prev_start = pend.pending_y * width;
            if total_opt < total_init {
                // Use optimized for previous row
                optimized_indices[prev_start..prev_start + width]
                    .copy_from_slice(&pend.opt_indices);
                committed_packed.copy_from_slice(&pend.packed_opt);

                if let Some((size, ref cur_indices, ref cur_packed, cur_f)) = best_on_opt {
                    if cur_indices != initial_row {
                        optimized_indices[row_start..row_start + width]
                            .copy_from_slice(cur_indices);
                    }
                    // Commit to predictor on the opt path
                    predictor.restore_owned(snap_opt_base);
                    let n = apply_filter_bpp1_into(cur_f, cur_packed, above_opt, &mut filtered_buf);
                    predictor.feed_row(&filtered_buf[..n]);
                    committed_packed.copy_from_slice(cur_packed);
                    committed_size = size;
                }
            } else {
                // Use initial for previous row (no change needed)
                committed_packed.copy_from_slice(&pend.packed_init);

                if let Some((size, ref cur_indices, ref cur_packed, cur_f)) = best_on_init {
                    if cur_indices != initial_row {
                        optimized_indices[row_start..row_start + width]
                            .copy_from_slice(cur_indices);
                    }
                    predictor.restore_owned(snap_init_base);
                    let n =
                        apply_filter_bpp1_into(cur_f, cur_packed, above_init, &mut filtered_buf);
                    predictor.feed_row(&filtered_buf[..n]);
                    committed_packed.copy_from_slice(cur_packed);
                    committed_size = size;
                }
            }
        } else if indices_differ {
            let bo = best_opt.unwrap();

            // Check if optimization looks promising (2+ bytes smaller)
            if bo.size + 2 < bi.size {
                // Defer decision to next row via row-pair lookahead

                // Create snapshot for init path
                predictor.restore(&snap_before);
                let n = apply_filter_bpp1_into(bi.filter, &bi.packed, above, &mut filtered_buf);
                let size_init = predictor.feed_row(&filtered_buf[..n]);
                let snap_init = predictor.snapshot();

                // Create snapshot for opt path
                predictor.restore(&snap_before);
                let n = apply_filter_bpp1_into(bo.filter, &bo.packed, above, &mut filtered_buf);
                let size_opt = predictor.feed_row(&filtered_buf[..n]);
                let snap_opt = predictor.snapshot();

                pending = Some(Pending {
                    snap_init,
                    packed_init: bi.packed,
                    size_init,
                    snap_opt,
                    packed_opt: bo.packed,
                    size_opt,
                    pending_y: y,
                    opt_indices: bo.indices,
                });
                continue; // Don't update committed yet
            }

            // Not promising enough to defer — commit initial
            predictor.restore(&snap_before);
            let n = apply_filter_bpp1_into(bi.filter, &bi.packed, above, &mut filtered_buf);
            committed_size = predictor.feed_row(&filtered_buf[..n]);
            committed_packed.copy_from_slice(&bi.packed);
        } else {
            // No optimization possible — commit best initial
            predictor.restore(&snap_before);
            let n = apply_filter_bpp1_into(bi.filter, &bi.packed, above, &mut filtered_buf);
            committed_size = predictor.feed_row(&filtered_buf[..n]);
            committed_packed.copy_from_slice(&bi.packed);
        }
    }

    // Resolve any pending last row
    if let Some(pend) = pending {
        let prev_start = pend.pending_y * width;
        if pend.size_opt < pend.size_init {
            optimized_indices[prev_start..prev_start + width].copy_from_slice(&pend.opt_indices);
        }
    }

    let _ = committed_size; // used for size tracking, final value unused
    optimized_indices
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn paeth_predictor_basic() {
        assert_eq!(paeth_predictor(100, 100, 100), 100);
        assert_eq!(paeth_predictor(10, 20, 15), 15);
        assert_eq!(paeth_predictor(0, 0, 0), 0);
    }

    #[test]
    fn filter_none_is_identity() {
        assert_eq!(compute_filtered_byte(0, 42, 0, 0, 0), 42);
        assert_eq!(compute_filtered_byte(0, 255, 100, 200, 50), 255);
    }

    #[test]
    fn filter_sub() {
        assert_eq!(compute_filtered_byte(1, 100, 50, 0, 0), 50);
        assert_eq!(compute_filtered_byte(1, 10, 20, 0, 0), 246);
    }

    #[test]
    fn filter_up() {
        assert_eq!(compute_filtered_byte(2, 100, 0, 50, 0), 50);
    }

    #[test]
    fn select_bit_depth_correct() {
        assert_eq!(select_bit_depth(2), 1);
        assert_eq!(select_bit_depth(4), 2);
        assert_eq!(select_bit_depth(16), 4);
        assert_eq!(select_bit_depth(17), 8);
        assert_eq!(select_bit_depth(256), 8);
    }

    #[test]
    fn pack_row_8bit() {
        let indices = [0, 1, 2, 3];
        let mut out = [0u8; 4];
        pack_row(&indices, 8, &mut out);
        assert_eq!(out, [0, 1, 2, 3]);
    }

    #[test]
    fn pack_row_4bit() {
        let indices = [0x0A, 0x0B, 0x0C];
        let mut out = [0u8; 2];
        pack_row(&indices, 4, &mut out);
        assert_eq!(out[0], 0xAB);
        assert_eq!(out[1], 0xC0);
    }

    #[test]
    fn top_k_filter_selection() {
        // All zeros — None filter should score well (already 0)
        let packed = vec![0u8; 100];
        let prev = vec![0u8; 100];
        let top = select_top_k_filters(&packed, &prev);
        // Filter 0 (None) should be in top 3 since all bytes are 0
        assert!(
            top.contains(&0),
            "filter None should be top-K for all-zero data"
        );
    }

    #[test]
    fn optimize_rgb_produces_valid_output() {
        let pixels: Vec<rgb::RGB<u8>> = vec![
            rgb::RGB::new(255, 0, 0),
            rgb::RGB::new(0, 255, 0),
            rgb::RGB::new(0, 0, 255),
            rgb::RGB::new(255, 255, 0),
            rgb::RGB::new(255, 0, 0),
            rgb::RGB::new(0, 255, 0),
            rgb::RGB::new(0, 0, 255),
            rgb::RGB::new(255, 255, 0),
        ];
        let width = 4;
        let height = 2;
        let weights = vec![1.0f32; 8];

        let centroids: Vec<OKLab> = [
            rgb::RGB::new(255u8, 0, 0),
            rgb::RGB::new(0, 255, 0),
            rgb::RGB::new(0, 0, 255),
            rgb::RGB::new(255, 255, 0),
        ]
        .iter()
        .map(|c| crate::oklab::srgb_to_oklab(c.r, c.g, c.b))
        .collect();
        let pal = Palette::from_centroids_sorted(
            centroids,
            false,
            crate::palette::PaletteSortStrategy::Luminance,
        );

        let indices: Vec<u8> = pixels
            .iter()
            .map(|p| {
                let lab = crate::oklab::srgb_to_oklab(p.r, p.g, p.b);
                pal.nearest(lab)
            })
            .collect();

        let opt_indices = optimize_rgb(&pixels, width, height, &weights, &pal, &indices, 7, 0.01);

        assert_eq!(opt_indices.len(), width * height);
        for &idx in &opt_indices {
            assert!((idx as usize) < pal.len(), "invalid index {idx}");
        }
    }
}
