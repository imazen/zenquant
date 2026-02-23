//! Joint deflate+quantization optimization for PNG.
//!
//! After normal quantization produces indices and AQ weights, this module
//! post-processes them: for each pixel with multiple perceptually acceptable
//! palette candidates, it picks the candidate that compresses best under
//! PNG scanline filters. The optimized indices are returned for the downstream
//! encoder to compress through its standard filter selection + deflate pipeline.
//!
//! Two-pass approach:
//! 1. **Filter selection**: Evaluate multiple filter strategies on the initial
//!    indices using incremental deflate compression forking. Pick the best
//!    per-row filter assignment.
//! 2. **Index optimization**: With filters locked, greedily optimize indices
//!    per pixel to minimize filtered byte residuals (prefer zeros and runs).
//!    The downstream encoder then re-selects filters on these improved indices.

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

use zenflate::{CompressionLevel, Compressor, Unstoppable};

use crate::oklab::OKLab;
use crate::palette::Palette;

/// Maximum candidates per pixel. Higher = more options but slower.
const MAX_CANDIDATES: usize = 6;

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

/// Apply a PNG filter to a packed row (bpp=1). Writes filter byte + filtered data to `out`.
fn apply_filter_bpp1(filter: u8, row: &[u8], prev_row: &[u8], out: &mut Vec<u8>) {
    out.push(filter);
    for (x, &raw) in row.iter().enumerate() {
        let left = if x > 0 { row[x - 1] } else { 0 };
        let above = prev_row[x];
        let above_left = if x > 0 { prev_row[x - 1] } else { 0 };
        out.push(compute_filtered_byte(filter, raw, left, above, above_left));
    }
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

// ── Greedy row optimization ──

/// For a given filter type, greedily select the best candidate index per pixel.
///
/// Conservative strategy: only change an index from the initial value if a
/// candidate produces a zero filtered byte (perfect prediction) or matches
/// the previous filtered byte (run extension). This preserves the
/// compression-friendly patterns from Viterbi while exploiting "free" wins.
#[allow(clippy::too_many_arguments)]
fn greedy_row_optimize(
    width: usize,
    filter: u8,
    candidates: &Candidates,
    row_start: usize,
    above_packed: &[u8],
    bit_depth: u8,
    row_bytes: usize,
) -> Vec<u8> {
    let mut result_indices = vec![0u8; width];
    let mut packed_row = vec![0u8; row_bytes];
    let mut prev_filtered: u8 = 0;

    for x in 0..width {
        let pixel_idx = row_start + x;
        let n_cands = candidates.counts[pixel_idx] as usize;

        // Default: keep the initial index (first candidate)
        let initial = candidates.indices[pixel_idx][0];
        let mut best_candidate = initial;

        // Single candidate — nothing to optimize
        if n_cands <= 1 {
            result_indices[x] = initial;
            // Update packed_row
            if bit_depth == 8 {
                packed_row[x] = initial;
            } else {
                let ppb = 8 / bit_depth as usize;
                let byte_pos = x / ppb;
                let bit_offset = (ppb - 1 - x % ppb) * bit_depth as usize;
                let mask = (1u8 << bit_depth) - 1;
                packed_row[byte_pos] &= !(mask << bit_offset);
                packed_row[byte_pos] |= (initial & mask) << bit_offset;
            }
            // Update prev_filtered for run tracking
            let byte_pos = if bit_depth == 8 {
                x
            } else {
                x / (8 / bit_depth as usize)
            };
            let left = if byte_pos > 0 {
                packed_row[byte_pos - 1]
            } else {
                0
            };
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
            prev_filtered =
                compute_filtered_byte(filter, packed_row[byte_pos], left, above, above_left);
            continue;
        }

        // For sub-byte depths, only score at byte boundaries
        if bit_depth < 8 {
            let ppb = 8 / bit_depth as usize;
            let is_last_in_byte = (x % ppb == ppb - 1) || x == width - 1;
            if !is_last_in_byte {
                // Not a boundary — just use initial
                result_indices[x] = initial;
                let byte_pos = x / ppb;
                let bit_offset = (ppb - 1 - x % ppb) * bit_depth as usize;
                let mask = (1u8 << bit_depth) - 1;
                packed_row[byte_pos] &= !(mask << bit_offset);
                packed_row[byte_pos] |= (initial & mask) << bit_offset;
                continue;
            }
        }

        let byte_pos = if bit_depth == 8 {
            x
        } else {
            x / (8 / bit_depth as usize)
        };
        let left = if byte_pos > 0 {
            packed_row[byte_pos - 1]
        } else {
            0
        };
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

        // Compute what the initial index would produce
        let initial_packed = if bit_depth == 8 {
            initial
        } else {
            let ppb = 8 / bit_depth as usize;
            let bit_offset = (ppb - 1 - x % ppb) * bit_depth as usize;
            let mask = (1u8 << bit_depth) - 1;
            let mut bv = packed_row[byte_pos];
            bv &= !(mask << bit_offset);
            bv |= (initial & mask) << bit_offset;
            bv
        };
        let initial_filtered =
            compute_filtered_byte(filter, initial_packed, left, above, above_left);

        // Only look for alternatives if initial doesn't already give us 0
        if initial_filtered != 0 {
            // Try to find a candidate that gives zero or matches the run
            let mut found_run = false;

            for c in 1..n_cands {
                let cand_idx = candidates.indices[pixel_idx][c];
                let packed_byte = if bit_depth == 8 {
                    cand_idx
                } else {
                    let ppb = 8 / bit_depth as usize;
                    let bit_offset = (ppb - 1 - x % ppb) * bit_depth as usize;
                    let mask = (1u8 << bit_depth) - 1;
                    let mut bv = packed_row[byte_pos];
                    bv &= !(mask << bit_offset);
                    bv |= (cand_idx & mask) << bit_offset;
                    bv
                };

                let filtered = compute_filtered_byte(filter, packed_byte, left, above, above_left);

                if filtered == 0 {
                    // Perfect prediction — always take this
                    best_candidate = cand_idx;
                    break; // can't do better than zero
                }

                if !found_run && byte_pos > 0 && filtered == prev_filtered {
                    // Run extension — good, but keep looking for zero
                    best_candidate = cand_idx;
                    found_run = true;
                }
            }
        }

        result_indices[x] = best_candidate;

        // Update packed row
        if bit_depth == 8 {
            packed_row[x] = best_candidate;
        } else {
            let ppb = 8 / bit_depth as usize;
            let bit_offset = (ppb - 1 - x % ppb) * bit_depth as usize;
            let mask = (1u8 << bit_depth) - 1;
            packed_row[byte_pos] &= !(mask << bit_offset);
            packed_row[byte_pos] |= (best_candidate & mask) << bit_offset;
        }

        // Track prev_filtered
        prev_filtered =
            compute_filtered_byte(filter, packed_row[byte_pos], left, above, above_left);
    }

    result_indices
}

// ── Top-level optimizer ──

/// Joint deflate+quantization optimization for RGB images.
///
/// Returns optimized indices. The downstream PNG encoder handles filter
/// selection and compression through its standard pipeline.
#[allow(clippy::too_many_arguments)]
pub(crate) fn optimize_rgb(
    pixels: &[rgb::RGB<u8>],
    width: usize,
    height: usize,
    weights: &[f32],
    palette: &Palette,
    initial_indices: &[u8],
    deflate_effort: u32,
    base_tolerance: f32,
) -> Vec<u8> {
    let pixel_oklab: Vec<OKLab> = pixels
        .iter()
        .map(|p| crate::oklab::srgb_to_oklab(p.r, p.g, p.b))
        .collect();

    optimize_inner(
        &pixel_oklab,
        width,
        height,
        weights,
        palette,
        initial_indices,
        None,
        deflate_effort,
        base_tolerance,
    )
}

/// Joint deflate+quantization optimization for RGBA images.
///
/// Transparent pixels (at transparent_index) are kept unchanged.
/// Returns optimized indices.
#[allow(clippy::too_many_arguments)]
pub(crate) fn optimize_rgba(
    pixels: &[rgb::RGBA<u8>],
    width: usize,
    height: usize,
    weights: &[f32],
    palette: &Palette,
    initial_indices: &[u8],
    deflate_effort: u32,
    base_tolerance: f32,
) -> Vec<u8> {
    let pixel_oklab: Vec<OKLab> = pixels
        .iter()
        .map(|p| crate::oklab::srgb_to_oklab(p.r, p.g, p.b))
        .collect();

    let transparent_index = palette.transparent_index();

    optimize_inner(
        &pixel_oklab,
        width,
        height,
        weights,
        palette,
        initial_indices,
        transparent_index,
        deflate_effort,
        base_tolerance,
    )
}

/// Core two-pass optimization.
///
/// Pass 1: Select per-row filters using incremental compression evaluation
///         on the initial (unmodified) indices — same quality as the downstream
///         encoder's own filter selection.
///
/// Pass 2: With filters locked, greedily optimize indices per pixel to minimize
///         filtered byte residuals. The downstream encoder will re-select filters
///         on these improved indices, potentially finding even better filters.
#[allow(clippy::too_many_arguments)]
fn optimize_inner(
    pixel_oklab: &[OKLab],
    width: usize,
    height: usize,
    weights: &[f32],
    palette: &Palette,
    initial_indices: &[u8],
    transparent_index: Option<u8>,
    deflate_effort: u32,
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
    // Pass 1: Filter selection via incremental compression forking
    // ═══════════════════════════════════════════════════════════════════════

    let eval_effort = deflate_effort.clamp(1, 22);
    let inc_effort = if eval_effort == 10 { 11 } else { eval_effort };
    let inc_level = CompressionLevel::new(inc_effort);

    let filter_choices = {
        let mut compressor = Compressor::new(inc_level);
        let total_filtered = filtered_row_size * height;
        let compress_bound = Compressor::deflate_compress_bound(total_filtered);
        let mut compress_buf = vec![0u8; compress_bound];
        let mut cumulative: usize = 0;
        let mut filters = Vec::with_capacity(height);
        let mut filtered_stream = Vec::with_capacity(total_filtered);
        let mut prev_packed = vec![0u8; row_bytes];
        let zero_row = vec![0u8; row_bytes];

        for y in 0..height {
            let row_start = y * width;
            let is_final = y == height - 1;
            let row_indices = &initial_indices[row_start..row_start + width];
            let mut packed = vec![0u8; row_bytes];
            pack_row(row_indices, bit_depth, &mut packed);

            let prev = if y > 0 {
                &prev_packed[..]
            } else {
                &zero_row[..]
            };

            let mut best_size = usize::MAX;
            let mut best_filter = 0u8;
            let mut best_fork: Option<(Compressor, usize)> = None;

            for f in 0..5u8 {
                let start = filtered_stream.len();
                apply_filter_bpp1(f, &packed, prev, &mut filtered_stream);

                let mut fork = compressor.clone();
                let result = fork.deflate_compress_incremental(
                    &filtered_stream,
                    &mut compress_buf,
                    is_final,
                    Unstoppable,
                );
                filtered_stream.truncate(start);

                if let Ok(size) = result {
                    let total = cumulative + size;
                    if total < best_size {
                        best_size = total;
                        best_filter = f;
                        best_fork = Some((fork, size));
                    }
                }
            }

            apply_filter_bpp1(best_filter, &packed, prev, &mut filtered_stream);
            filters.push(best_filter);
            if let Some((fork, size)) = best_fork {
                compressor = fork;
                cumulative = cumulative.wrapping_add(size);
            }
            prev_packed.copy_from_slice(&packed);
        }
        filters
    };

    // ═══════════════════════════════════════════════════════════════════════
    // Pass 2: Index optimization with row-pair lookahead.
    //
    // Each row's A/B decision is deferred until the NEXT row is processed.
    // This catches cases where optimizing row N looks good locally but
    // hurts row N+1's compression via changed deflate state.
    //
    // For each row, we evaluate it under both the "prev kept initial" and
    // "prev used optimized" compressor states, then pick the 2-row path
    // that produces the best cumulative output.
    // ═══════════════════════════════════════════════════════════════════════

    let mut optimized_indices = initial_indices.to_vec();
    let zero_row = vec![0u8; row_bytes];

    // New compressor for pass 2 (pass 1's was consumed)
    let pass2_effort = if eval_effort == 10 { 11 } else { eval_effort };
    let pass2_level = CompressionLevel::new(pass2_effort);
    let compress_bound = Compressor::deflate_compress_bound(filtered_row_size * height);

    // A "path" tracks one possible compressor timeline
    struct Path {
        compressor: Compressor,
        filtered: Vec<u8>,
        cumulative: usize,
        packed_row: Vec<u8>,         // last committed packed row (for filter context)
        compress_buf: Vec<u8>,
    }

    impl Path {
        fn eval_row(
            &self,
            filter: u8,
            packed: &[u8],
            above: &[u8],
            is_final: bool,
        ) -> Option<(Compressor, usize)> {
            let mut filtered = self.filtered.clone();
            apply_filter_bpp1(filter, packed, above, &mut filtered);
            let mut fork = self.compressor.clone();
            let mut buf = self.compress_buf.clone();
            let size = fork
                .deflate_compress_incremental(&filtered, &mut buf, is_final, Unstoppable)
                .ok()?;
            Some((fork, size))
        }

        /// Commit a row. Uses `self.packed_row` as the above context before
        /// overwriting it with the new packed row.
        fn commit_row(
            &mut self,
            filter: u8,
            packed: &[u8],
            compressor: Compressor,
            size: usize,
        ) {
            // self.packed_row is the "above" row — use it before overwriting
            let above: Vec<u8> = self.packed_row.clone();
            apply_filter_bpp1(filter, packed, &above, &mut self.filtered);
            self.compressor = compressor;
            self.cumulative = self.cumulative.wrapping_add(size);
            self.packed_row[..packed.len()].copy_from_slice(packed);
        }

        /// Like commit_row but uses a provided above context (for first row).
        fn commit_row_with_above(
            &mut self,
            filter: u8,
            packed: &[u8],
            above: &[u8],
            compressor: Compressor,
            size: usize,
        ) {
            apply_filter_bpp1(filter, packed, above, &mut self.filtered);
            self.compressor = compressor;
            self.cumulative = self.cumulative.wrapping_add(size);
            self.packed_row[..packed.len()].copy_from_slice(packed);
        }
    }

    // Start with a single path (no pending decision yet)
    let mut committed = Path {
        compressor: Compressor::new(pass2_level),
        filtered: Vec::with_capacity(filtered_row_size * height),
        cumulative: 0,
        packed_row: vec![0u8; row_bytes],
        compress_buf: vec![0u8; compress_bound],
    };

    // Pending: the previous row has two possible states (init vs optimized)
    // that we haven't decided between yet.
    struct Pending {
        // Path where previous row used INITIAL indices
        path_init: Path,
        // Path where previous row used OPTIMIZED indices
        path_opt: Path,
        // Row index and optimized indices for the pending row
        pending_y: usize,
        opt_indices: Vec<u8>,
    }

    let mut pending: Option<Pending> = None;

    for (y, &filter) in filter_choices.iter().enumerate() {
        let row_start = y * width;
        let is_final = y == height - 1;

        let initial_row = &initial_indices[row_start..row_start + width];
        let mut packed_init = vec![0u8; row_bytes];
        pack_row(initial_row, bit_depth, &mut packed_init);

        let opt_row = greedy_row_optimize(
            width,
            filter,
            &candidates,
            row_start,
            // above: use committed packed row (or pending's — resolved below)
            if y > 0 {
                &committed.packed_row[..]
            } else {
                &zero_row[..]
            },
            bit_depth,
            row_bytes,
        );
        let mut packed_opt = vec![0u8; row_bytes];
        pack_row(&opt_row, bit_depth, &mut packed_opt);

        let indices_differ = initial_row != &opt_row[..];

        if let Some(pend) = pending.take() {
            // We have a deferred decision from the previous row.
            // Evaluate current row under BOTH previous-row paths to decide
            // which version of the previous row to keep.

            let above_init = &pend.path_init.packed_row[..];
            let above_opt = &pend.path_opt.packed_row[..];

            // Best current-row outcome starting from prev=initial
            let eval_on_init = pend.path_init.eval_row(filter, &packed_init, above_init, is_final);
            let eval_opt_on_init = if indices_differ {
                // Re-optimize for the init context since above row differs
                let opt_for_init = greedy_row_optimize(
                    width, filter, &candidates, row_start,
                    above_init, bit_depth, row_bytes,
                );
                let mut packed_for_init = vec![0u8; row_bytes];
                pack_row(&opt_for_init, bit_depth, &mut packed_for_init);
                let eval = pend.path_init.eval_row(filter, &packed_for_init, above_init, is_final);
                eval.map(|e| (e, opt_for_init, packed_for_init))
            } else {
                None
            };

            // Best current-row outcome starting from prev=optimized
            let eval_on_opt = pend.path_opt.eval_row(filter, &packed_init, above_opt, is_final);
            let eval_opt_on_opt = if indices_differ {
                let opt_for_opt = greedy_row_optimize(
                    width, filter, &candidates, row_start,
                    above_opt, bit_depth, row_bytes,
                );
                let mut packed_for_opt = vec![0u8; row_bytes];
                pack_row(&opt_for_opt, bit_depth, &mut packed_for_opt);
                let eval = pend.path_opt.eval_row(filter, &packed_for_opt, above_opt, is_final);
                eval.map(|e| (e, opt_for_opt, packed_for_opt))
            } else {
                None
            };

            // Find best total across all 4 combinations:
            // (prev=init, cur=init), (prev=init, cur=opt),
            // (prev=opt, cur=init), (prev=opt, cur=opt)
            struct Candidate {
                total: usize,
                use_prev_opt: bool,
                use_cur_opt: bool,
                cur_fork: Compressor,
                cur_size: usize,
                cur_packed: Vec<u8>,
                cur_indices: Vec<u8>,
            }
            let mut best: Option<Candidate> = None;

            // (prev=init, cur=init)
            if let Some((fork, size)) = eval_on_init {
                let total = pend.path_init.cumulative.wrapping_add(size);
                if best.as_ref().map_or(true, |b| total < b.total) {
                    best = Some(Candidate {
                        total,
                        use_prev_opt: false,
                        use_cur_opt: false,
                        cur_fork: fork,
                        cur_size: size,
                        cur_packed: packed_init.clone(),
                        cur_indices: initial_row.to_vec(),
                    });
                }
            }

            // (prev=init, cur=opt)
            if let Some(((fork, size), ref indices, ref packed)) = eval_opt_on_init {
                let total = pend.path_init.cumulative.wrapping_add(size);
                if best.as_ref().map_or(true, |b| total < b.total) {
                    best = Some(Candidate {
                        total,
                        use_prev_opt: false,
                        use_cur_opt: true,
                        cur_fork: fork,
                        cur_size: size,
                        cur_packed: packed.clone(),
                        cur_indices: indices.clone(),
                    });
                }
            }

            // (prev=opt, cur=init)
            if let Some((fork, size)) = eval_on_opt {
                let total = pend.path_opt.cumulative.wrapping_add(size);
                if best.as_ref().map_or(true, |b| total < b.total) {
                    best = Some(Candidate {
                        total,
                        use_prev_opt: true,
                        use_cur_opt: false,
                        cur_fork: fork,
                        cur_size: size,
                        cur_packed: packed_init.clone(),
                        cur_indices: initial_row.to_vec(),
                    });
                }
            }

            // (prev=opt, cur=opt)
            if let Some(((fork, size), ref indices, ref packed)) = eval_opt_on_opt {
                let total = pend.path_opt.cumulative.wrapping_add(size);
                if best.as_ref().map_or(true, |b| total < b.total) {
                    best = Some(Candidate {
                        total,
                        use_prev_opt: true,
                        use_cur_opt: true,
                        cur_fork: fork,
                        cur_size: size,
                        cur_packed: packed.clone(),
                        cur_indices: indices.clone(),
                    });
                }
            }

            if let Some(winner) = best {
                // Decide the previous row
                let prev_start = pend.pending_y * width;
                committed = if winner.use_prev_opt {
                    optimized_indices[prev_start..prev_start + width]
                        .copy_from_slice(&pend.opt_indices);
                    pend.path_opt
                } else {
                    pend.path_init
                };

                // Decide the current row
                if winner.use_cur_opt {
                    optimized_indices[row_start..row_start + width]
                        .copy_from_slice(&winner.cur_indices);
                }

                committed.commit_row(
                    filter,
                    &winner.cur_packed,
                    winner.cur_fork,
                    winner.cur_size,
                );
            }
        } else {
            // No pending decision — first row or after a non-optimizable row.
            let above = if y > 0 {
                committed.packed_row.as_slice()
            } else {
                zero_row.as_slice()
            };

            if indices_differ {
                let eval_init = committed.eval_row(filter, &packed_init, above, is_final);
                let eval_opt = committed.eval_row(filter, &packed_opt, above, is_final);

                match (eval_init, eval_opt) {
                    (Some((fork_a, size_a)), Some((fork_b, size_b))) if size_b + 2 < size_a => {
                        // Optimization looks promising — defer decision
                        let mut path_init = Path {
                            compressor: committed.compressor.clone(),
                            filtered: committed.filtered.clone(),
                            cumulative: committed.cumulative,
                            packed_row: committed.packed_row.clone(),
                            compress_buf: committed.compress_buf.clone(),
                        };
                        if y == 0 {
                            path_init.commit_row_with_above(filter, &packed_init, &zero_row, fork_a, size_a);
                        } else {
                            path_init.commit_row(filter, &packed_init, fork_a, size_a);
                        }

                        let mut path_opt = Path {
                            compressor: committed.compressor.clone(),
                            filtered: committed.filtered.clone(),
                            cumulative: committed.cumulative,
                            packed_row: committed.packed_row.clone(),
                            compress_buf: committed.compress_buf.clone(),
                        };
                        if y == 0 {
                            path_opt.commit_row_with_above(filter, &packed_opt, &zero_row, fork_b, size_b);
                        } else {
                            path_opt.commit_row(filter, &packed_opt, fork_b, size_b);
                        }

                        pending = Some(Pending {
                            path_init,
                            path_opt,
                            pending_y: y,
                            opt_indices: opt_row,
                        });
                        continue; // Don't update committed yet
                    }
                    (Some((fork_a, size_a)), _) => {
                        if y == 0 {
                            committed.commit_row_with_above(filter, &packed_init, &zero_row, fork_a, size_a);
                        } else {
                            committed.commit_row(filter, &packed_init, fork_a, size_a);
                        }
                    }
                    _ => {} // shouldn't happen, but safe fallback
                }
            } else {
                // No optimization possible
                if let Some((fork, size)) = committed.eval_row(filter, &packed_init, above, is_final) {
                    if y == 0 {
                        committed.commit_row_with_above(filter, &packed_init, &zero_row, fork, size);
                    } else {
                        committed.commit_row(filter, &packed_init, fork, size);
                    }
                }
            }
        }
    }

    // If the last row is still pending, resolve it (no next row to compare against)
    if let Some(pend) = pending {
        let prev_start = pend.pending_y * width;
        if pend.path_opt.cumulative < pend.path_init.cumulative {
            optimized_indices[prev_start..prev_start + width]
                .copy_from_slice(&pend.opt_indices);
        }
    }

    optimized_indices
}

#[cfg(test)]
mod tests {
    use super::*;

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
