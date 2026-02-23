//! Joint deflate+quantization optimization for PNG.
//!
//! After normal quantization produces indices and AQ weights, this module
//! post-processes them: for each pixel with multiple perceptually acceptable
//! palette candidates, it picks the candidate that compresses best under
//! PNG scanline filters + deflate. Per-scanline, it forks the deflate
//! compressor to try all 5 PNG filter types, greedily selecting indices
//! that minimize filtered residuals.

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

use zenflate::{CompressionLevel, Compressor, Unstoppable};

use crate::PngZointData;
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
///
/// Parameters:
/// - `filter`: PNG filter type 0–4
/// - `raw`: the raw (unfiltered) byte at this position
/// - `left`: raw byte of the pixel to the left (0 if at column 0)
/// - `above`: raw byte of the pixel above (0 if at row 0)
/// - `above_left`: raw byte of the pixel above-left (0 if at edges)
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

    // Tolerance is base_tolerance / weight. Higher weight = tighter tolerance.
    // Weight ~0.1 (textured) → tolerance ~10x base → more candidates.
    // Weight ~1.0 (smooth) → tolerance ~1x base → fewer candidates.
    let base_tol_sq = base_tolerance * base_tolerance;

    for i in 0..n {
        let seed = initial_indices[i];
        indices[i][0] = seed;

        // Effective tolerance: base / weight, squared for comparison
        let w = weights[i].max(0.01); // avoid division by near-zero
        let tol_sq = base_tol_sq / (w * w);

        // Use seeded K-nearest to find candidates
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

        // Always include the original index if it wasn't in the candidate list
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
/// Scoring prefers:
/// 1. filtered_byte == 0 (perfect prediction) — best
/// 2. Small absolute residuals — good
/// 3. Same filtered byte as previous pixel — run extension bonus
///
/// Returns the optimized row of indices.
#[allow(clippy::too_many_arguments)]
fn greedy_row_optimize(
    width: usize,
    y: usize,
    filter: u8,
    candidates: &Candidates,
    row_start: usize,
    optimized_above: &[u8], // packed row bytes from previous row (empty if y==0)
    bit_depth: u8,
    row_byte_count: usize,
) -> Vec<u8> {
    let mut result_indices = vec![0u8; width];
    let mut packed_row = vec![0u8; row_byte_count];

    // For filters that reference "left", we track the left packed byte.
    // For "above" and "above-left", we use the packed above row.
    let _ = y; // used implicitly via optimized_above

    for x in 0..width {
        let pixel_idx = row_start + x;
        let n_cands = candidates.counts[pixel_idx] as usize;

        let mut best_score = i32::MAX;
        let mut best_candidate = candidates.indices[pixel_idx][0];

        for c in 0..n_cands {
            let cand_idx = candidates.indices[pixel_idx][c];

            // Pack this candidate to see what byte it produces
            // For 8-bit, the packed byte IS the index
            let packed_byte = if bit_depth == 8 {
                cand_idx
            } else {
                // For sub-byte depths, we need the byte that contains this pixel
                let ppb = 8 / bit_depth as usize;
                let byte_pos = x / ppb;
                let bit_offset = (ppb - 1 - x % ppb) * bit_depth as usize;
                let mask = (1u8 << bit_depth) - 1;

                // Start with the current packed byte (which may have prior pixels in it)
                let mut byte_val = packed_row[byte_pos];
                // Clear this pixel's bits
                byte_val &= !(mask << bit_offset);
                // Set this pixel's bits
                byte_val |= (cand_idx & mask) << bit_offset;
                byte_val
            };

            // Only score on byte boundaries for sub-byte depths
            if bit_depth < 8 {
                let ppb = 8 / bit_depth as usize;
                let is_last_in_byte = (x % ppb == ppb - 1) || x == width - 1;
                if !is_last_in_byte {
                    // For non-boundary pixels, just pick the first candidate
                    // and let the boundary pixel do the scoring
                    if c == 0 {
                        best_candidate = cand_idx;
                    }
                    continue;
                }
            }

            let byte_pos = if bit_depth == 8 {
                x
            } else {
                x / (8 / bit_depth as usize)
            };

            // Compute filter context
            let left = if byte_pos > 0 {
                packed_row[byte_pos - 1]
            } else {
                0
            };
            let above = if !optimized_above.is_empty() {
                optimized_above[byte_pos]
            } else {
                0
            };
            let above_left = if byte_pos > 0 && !optimized_above.is_empty() {
                optimized_above[byte_pos - 1]
            } else {
                0
            };

            let filtered = compute_filtered_byte(filter, packed_byte, left, above, above_left);

            // Score: prefer 0, then small residuals
            let score = if filtered == 0 {
                0 // perfect prediction
            } else {
                // Use absolute value of filtered byte as score (closer to 0 = better)
                let abs_val = (filtered as i8).unsigned_abs() as i32;
                abs_val + 1 // +1 so that 0 is strictly best
            };

            if score < best_score {
                best_score = score;
                best_candidate = cand_idx;
            }
        }

        result_indices[x] = best_candidate;

        // Update packed row with the chosen candidate
        if bit_depth == 8 {
            packed_row[x] = best_candidate;
        } else {
            let ppb = 8 / bit_depth as usize;
            let byte_pos = x / ppb;
            let bit_offset = (ppb - 1 - x % ppb) * bit_depth as usize;
            let mask = (1u8 << bit_depth) - 1;
            packed_row[byte_pos] &= !(mask << bit_offset);
            packed_row[byte_pos] |= (best_candidate & mask) << bit_offset;
        }
    }

    result_indices
}

// ── Top-level optimizer ──

/// Joint deflate+quantization optimization for RGB images.
///
/// Returns (optimized_indices, PngZointData).
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
) -> (Vec<u8>, PngZointData) {
    // Convert pixels to OKLab for distance computation
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
/// Transparent pixels (at transparent_index) are kept unchanged — no
/// candidate alternatives are offered for them.
///
/// Returns (optimized_indices, PngZointData).
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
) -> (Vec<u8>, PngZointData) {
    // Convert pixels to OKLab
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

/// Core optimization loop shared by RGB and RGBA paths.
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
) -> (Vec<u8>, PngZointData) {
    let n_colors = palette.len();
    let bit_depth = select_bit_depth(n_colors);
    let row_bytes = packed_row_bytes(width, bit_depth);
    let filtered_row_size = row_bytes + 1; // filter byte + data

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

    // Clamp deflate effort to incremental-compatible range (1-9 or 11-22)
    let eval_effort = deflate_effort.clamp(1, 22);
    let eval_effort = if eval_effort == 10 { 11 } else { eval_effort };
    let eval_level = CompressionLevel::new(eval_effort);

    // Output buffers
    let mut optimized_indices = initial_indices.to_vec();
    let mut filter_choices = Vec::with_capacity(height);
    let mut filtered_stream = Vec::with_capacity(filtered_row_size * height);

    // Compressor for incremental forking
    let mut compressor = Compressor::new(eval_level);
    let compress_bound = Compressor::deflate_compress_bound(filtered_row_size * height);
    let mut compress_buf = vec![0u8; compress_bound];
    let mut cumulative_output: usize = 0;

    // Previous packed row (zero-filled for first row)
    let mut prev_packed_row = vec![0u8; row_bytes];
    let zero_row = vec![0u8; row_bytes];

    for y in 0..height {
        let row_start = y * width;
        let is_final = y == height - 1;
        let above = if y > 0 {
            &prev_packed_row[..]
        } else {
            &[] as &[u8]
        };

        let mut best_size = usize::MAX;
        let mut best_filter = 0u8;
        let mut best_row_indices: Option<Vec<u8>> = None;
        let mut best_compressor: Option<(Compressor, usize)> = None;

        // Try all 5 PNG filter types
        for f in 0..5u8 {
            // Greedy-optimize indices for this filter
            let row_indices = greedy_row_optimize(
                width,
                y,
                f,
                &candidates,
                row_start,
                above,
                bit_depth,
                row_bytes,
            );

            // Pack the row
            let mut packed = vec![0u8; row_bytes];
            pack_row(&row_indices, bit_depth, &mut packed);

            // Apply filter to get the filtered row
            let filtered_start = filtered_stream.len();
            apply_filter_bpp1(
                f,
                &packed,
                if y > 0 { &prev_packed_row } else { &zero_row },
                &mut filtered_stream,
            );

            // Fork compressor and compress incrementally
            let mut fork = compressor.clone();
            let result = fork.deflate_compress_incremental(
                &filtered_stream,
                &mut compress_buf,
                is_final,
                Unstoppable,
            );

            // Remove the candidate row (we haven't committed yet)
            filtered_stream.truncate(filtered_start);

            if let Ok(size) = result {
                let total = cumulative_output + size;
                if total < best_size {
                    best_size = total;
                    best_filter = f;
                    best_row_indices = Some(row_indices);
                    best_compressor = Some((fork, size));
                }
            }
        }

        // Commit winning filter + indices
        let winning_indices = best_row_indices
            .unwrap_or_else(|| initial_indices[row_start..row_start + width].to_vec());

        // Update optimized indices
        optimized_indices[row_start..row_start + width].copy_from_slice(&winning_indices);

        // Pack and filter the winning row
        let mut packed = vec![0u8; row_bytes];
        pack_row(&winning_indices, bit_depth, &mut packed);
        apply_filter_bpp1(
            best_filter,
            &packed,
            if y > 0 { &prev_packed_row } else { &zero_row },
            &mut filtered_stream,
        );

        filter_choices.push(best_filter);

        // Advance compressor state
        if let Some((winner, size)) = best_compressor {
            compressor = winner;
            cumulative_output = cumulative_output.wrapping_add(size);
        }

        prev_packed_row.copy_from_slice(&packed);
    }

    // Compute Adler-32 of the uncompressed filtered stream
    let adler = zenflate::adler32(1, &filtered_stream);

    // Final compression: re-compress the complete filtered stream at a higher
    // effort for optimal block splitting, since the incremental per-row
    // compression was at evaluation effort (potentially lower).
    let final_effort = deflate_effort.clamp(1, 22).max(eval_effort);
    let final_level = CompressionLevel::new(final_effort);
    let mut final_compressor = Compressor::new(final_level);
    let final_bound = Compressor::deflate_compress_bound(filtered_stream.len());
    let mut final_buf = vec![0u8; final_bound];

    let deflate_stream =
        match final_compressor.deflate_compress(&filtered_stream, &mut final_buf, Unstoppable) {
            Ok(size) => final_buf[..size].to_vec(),
            Err(_) => {
                // Fallback: use the incremental output — shouldn't happen but be safe
                let mut fallback_compressor = Compressor::new(eval_level);
                let bound = Compressor::deflate_compress_bound(filtered_stream.len());
                let mut buf = vec![0u8; bound];
                let size = fallback_compressor
                    .deflate_compress(&filtered_stream, &mut buf, Unstoppable)
                    .unwrap_or(0);
                buf[..size].to_vec()
            }
        };

    let zoint_data = PngZointData {
        deflate_stream,
        filter_choices,
        bit_depth,
        adler32: adler,
    };

    (optimized_indices, zoint_data)
}
