//! Vendored lightweight LZ77 predictor for zoint filter ranking.
//!
//! Predicts deflate-compressed sizes without producing valid deflate output.
//! Vendors the minimum matchfinder code from zenflate (HC matchfinder + frequency
//! counting + Huffman code length estimation) to decouple zoint from the zenflate crate.
//!
//! The predictor's purpose is **relative ranking** of PNG filter choices, not
//! exact size prediction. Huffman code lengths (capped at 15 bits per deflate spec)
//! closely model actual deflate output for this use case.

extern crate alloc;
use alloc::sync::Arc;

// ── Vendored fast_bytes (safe paths only) ──────────────────────────────────

#[inline(always)]
fn load_u32_le(data: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(data[off..off + 4].try_into().unwrap())
}

#[inline(always)]
fn load_u64_le(data: &[u8], off: usize) -> u64 {
    u64::from_le_bytes(data[off..off + 8].try_into().unwrap())
}

#[inline(always)]
fn get_byte(data: &[u8], idx: usize) -> u8 {
    data[idx]
}

// ── Vendored matchfinder constants ─────────────────────────────────────────

const MATCHFINDER_WINDOW_ORDER: u32 = 15;
const MATCHFINDER_WINDOW_SIZE: u32 = 1 << MATCHFINDER_WINDOW_ORDER;
const MATCHFINDER_INITVAL: i16 = i16::MIN;

// ── Vendored matchfinder common functions ──────────────────────────────────

#[inline(always)]
fn lz_hash(seq: u32, num_bits: u32) -> u32 {
    seq.wrapping_mul(0x1E35A7BD) >> (32 - num_bits)
}

#[inline(always)]
fn lz_extend(strptr: &[u8], matchptr: &[u8], start_len: u32, max_len: u32) -> u32 {
    let mut len = start_len;
    let max = max_len as usize;

    while (len as usize) + 8 <= max {
        let off = len as usize;
        let sw = load_u64_le(strptr, off);
        let mw = load_u64_le(matchptr, off);
        let xor = sw ^ mw;
        if xor != 0 {
            len += xor.trailing_zeros() >> 3;
            return len.min(max_len);
        }
        len += 8;
    }

    while (len as usize) < max && get_byte(strptr, len as usize) == get_byte(matchptr, len as usize)
    {
        len += 1;
    }
    len
}

#[inline]
#[allow(dead_code)]
fn matchfinder_init(data: &mut [i16]) {
    data.fill(MATCHFINDER_INITVAL);
}

#[inline]
fn matchfinder_rebase(data: &mut [i16]) {
    for entry in data.iter_mut() {
        *entry = entry.saturating_add(i16::MIN);
    }
}

// ── Vendored HC matchfinder ────────────────────────────────────────────────

const HC_MATCHFINDER_HASH3_ORDER: u32 = 15;
const HC_MATCHFINDER_HASH4_ORDER: u32 = 16;
const HC_HASH3_SIZE: usize = 1 << HC_MATCHFINDER_HASH3_ORDER;
const HC_HASH4_SIZE: usize = 1 << HC_MATCHFINDER_HASH4_ORDER;
const WINDOW_MASK: usize = MATCHFINDER_WINDOW_SIZE as usize - 1;

#[derive(Clone)]
struct HcMatchfinder {
    hash3_tab: [i16; HC_HASH3_SIZE],
    hash4_tab: [i16; HC_HASH4_SIZE],
    next_tab: [i16; MATCHFINDER_WINDOW_SIZE as usize],
}

impl HcMatchfinder {
    fn new() -> Self {
        Self {
            hash3_tab: [MATCHFINDER_INITVAL; HC_HASH3_SIZE],
            hash4_tab: [MATCHFINDER_INITVAL; HC_HASH4_SIZE],
            next_tab: [MATCHFINDER_INITVAL; MATCHFINDER_WINDOW_SIZE as usize],
        }
    }

    fn slide_window(&mut self) {
        matchfinder_rebase(&mut self.hash3_tab);
        matchfinder_rebase(&mut self.hash4_tab);
        matchfinder_rebase(&mut self.next_tab);
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn longest_match(
        &mut self,
        input: &[u8],
        in_base_offset: &mut usize,
        in_next: usize,
        best_len: u32,
        max_len: u32,
        nice_len: u32,
        max_search_depth: u32,
        next_hashes: &mut [u32; 2],
    ) -> (u32, u32) {
        let mut best_len = best_len;
        let mut best_offset = 0u32;
        let mut depth_remaining = max_search_depth;

        let mut cur_pos = (in_next - *in_base_offset) as u32;

        if cur_pos >= MATCHFINDER_WINDOW_SIZE {
            self.slide_window();
            *in_base_offset += MATCHFINDER_WINDOW_SIZE as usize;
            cur_pos -= MATCHFINDER_WINDOW_SIZE;
        }

        let in_base = *in_base_offset;
        let cutoff = cur_pos as i32 - MATCHFINDER_WINDOW_SIZE as i32;

        let hash3 = next_hashes[0] as usize;
        let hash4 = next_hashes[1] as usize;

        let cur_node3 = self.hash3_tab[hash3] as i32;
        let mut cur_node4 = self.hash4_tab[hash4] as i32;

        self.hash3_tab[hash3] = cur_pos as i16;
        self.hash4_tab[hash4] = cur_pos as i16;
        self.next_tab[cur_pos as usize] = cur_node4 as i16;

        if in_next + 5 <= input.len() {
            let next_seq = load_u32_le(input, in_next + 1);
            next_hashes[0] = lz_hash(next_seq & 0xFFFFFF, HC_MATCHFINDER_HASH3_ORDER);
            next_hashes[1] = lz_hash(next_seq, HC_MATCHFINDER_HASH4_ORDER);
        }

        if max_len < 5 {
            return (best_len, best_offset);
        }

        // good_match threshold — hardcoded for our greedy use case
        let good_match = 16u32;

        if best_len < 4 {
            if cur_node3 <= cutoff {
                return (best_len, best_offset);
            }

            let seq4 = load_u32_le(input, in_next);

            if best_len < 3 {
                let match_pos = (in_base as isize + cur_node3 as isize) as usize;
                let match_seq = load_u32_le(input, match_pos);
                if (match_seq & 0xFFFFFF) == (seq4 & 0xFFFFFF) {
                    best_len = 3;
                    best_offset = (in_next - match_pos) as u32;
                }
            }

            if cur_node4 <= cutoff {
                return (best_len, best_offset);
            }

            loop {
                let match_pos = (in_base as isize + cur_node4 as isize) as usize;
                let match_seq = load_u32_le(input, match_pos);

                if match_seq == seq4 {
                    best_len = lz_extend(&input[in_next..], &input[match_pos..], 4, max_len);
                    best_offset = (in_next - match_pos) as u32;
                    if best_len >= nice_len {
                        return (best_len, best_offset);
                    }
                    if best_len >= good_match {
                        depth_remaining = (depth_remaining >> 2).max(1);
                    }
                    cur_node4 = self.next_tab[cur_node4 as usize & WINDOW_MASK] as i32;
                    if cur_node4 <= cutoff || {
                        depth_remaining -= 1;
                        depth_remaining == 0
                    } {
                        return (best_len, best_offset);
                    }
                    break;
                }

                cur_node4 = self.next_tab[cur_node4 as usize & WINDOW_MASK] as i32;
                if cur_node4 <= cutoff || {
                    depth_remaining -= 1;
                    depth_remaining == 0
                } {
                    return (best_len, best_offset);
                }
            }
        } else if cur_node4 <= cutoff || best_len >= nice_len {
            return (best_len, best_offset);
        }

        loop {
            let match_pos = (in_base as isize + cur_node4 as isize) as usize;

            let tail_off = (best_len - 3) as usize;
            let m_tail = load_u32_le(input, match_pos + tail_off);
            let s_tail = load_u32_le(input, in_next + tail_off);

            if m_tail == s_tail {
                let m_head = load_u32_le(input, match_pos);
                let s_head = load_u32_le(input, in_next);
                if m_head == s_head {
                    let len = lz_extend(&input[in_next..], &input[match_pos..], 4, max_len);
                    if len > best_len {
                        best_len = len;
                        best_offset = (in_next - match_pos) as u32;
                        if best_len >= nice_len {
                            return (best_len, best_offset);
                        }
                        if best_len >= good_match {
                            depth_remaining = (depth_remaining >> 2).max(1);
                        }
                    }
                }
            }

            cur_node4 = self.next_tab[cur_node4 as usize & WINDOW_MASK] as i32;
            if cur_node4 <= cutoff || {
                depth_remaining -= 1;
                depth_remaining == 0
            } {
                return (best_len, best_offset);
            }
        }
    }

    #[inline(always)]
    fn skip_bytes(
        &mut self,
        input: &[u8],
        in_base_offset: &mut usize,
        in_next: usize,
        in_end: usize,
        count: u32,
        next_hashes: &mut [u32; 2],
    ) {
        if count as usize + 5 > in_end - in_next {
            return;
        }

        let mut cur_pos = (in_next - *in_base_offset) as u32;
        let mut hash3 = next_hashes[0] as usize;
        let mut hash4 = next_hashes[1] as usize;
        let mut pos = in_next;
        let mut remaining = count;

        loop {
            if cur_pos >= MATCHFINDER_WINDOW_SIZE {
                self.slide_window();
                *in_base_offset += MATCHFINDER_WINDOW_SIZE as usize;
                cur_pos -= MATCHFINDER_WINDOW_SIZE;
            }

            self.hash3_tab[hash3] = cur_pos as i16;
            self.next_tab[cur_pos as usize] = self.hash4_tab[hash4];
            self.hash4_tab[hash4] = cur_pos as i16;

            pos += 1;
            cur_pos += 1;
            remaining -= 1;

            let next_seq = load_u32_le(input, pos);
            hash3 = lz_hash(next_seq & 0xFFFFFF, HC_MATCHFINDER_HASH3_ORDER) as usize;
            hash4 = lz_hash(next_seq, HC_MATCHFINDER_HASH4_ORDER) as usize;

            if remaining == 0 {
                break;
            }
        }

        next_hashes[0] = hash3 as u32;
        next_hashes[1] = hash4 as u32;
    }
}

// ── Vendored deflate constants for frequency counting ──────────────────────

const DEFLATE_MIN_MATCH_LEN: u32 = 3;
const DEFLATE_MAX_MATCH_LEN: u32 = 258;
const DEFLATE_NUM_LITLEN_SYMS: usize = 288;
const DEFLATE_NUM_OFFSET_SYMS: usize = 32;
const DEFLATE_FIRST_LEN_SYM: u32 = 257;
#[allow(dead_code)]
const DEFLATE_END_OF_BLOCK: u32 = 256;

const DEFLATE_LENGTH_BASE: [u16; 29] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131,
    163, 195, 227, 258,
];

const DEFLATE_LENGTH_EXTRA_BITS: [u8; 29] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
];

const DEFLATE_OFFSET_BASE: [u32; 32] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
    2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577, 0, 0,
];

const DEFLATE_OFFSET_EXTRA_BITS: [u8; 32] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13,
    13, 0, 0,
];

/// Length slot for each match length (3..=258).
const LENGTH_SLOT: [u8; DEFLATE_MAX_MATCH_LEN as usize + 1] = {
    let mut table = [0u8; DEFLATE_MAX_MATCH_LEN as usize + 1];
    let mut slot = 0u8;
    while slot < 29 {
        let base = DEFLATE_LENGTH_BASE[slot as usize] as usize;
        let extra = DEFLATE_LENGTH_EXTRA_BITS[slot as usize];
        let count = 1usize << extra;
        let mut j = 0usize;
        while j < count && base + j <= DEFLATE_MAX_MATCH_LEN as usize {
            table[base + j] = slot;
            j += 1;
        }
        slot += 1;
    }
    table
};

/// Offset slot lookup for small offsets (offset-1 in [0..255]).
const OFFSET_SLOT_SMALL: [u8; 256] = {
    let mut table = [0u8; 256];
    let mut slot = 0u8;
    while slot < 30 {
        let base = DEFLATE_OFFSET_BASE[slot as usize] as usize;
        let extra = DEFLATE_OFFSET_EXTRA_BITS[slot as usize];
        let count = 1usize << extra;
        let mut j = 0usize;
        while j < count {
            let offset_m1 = base + j - 1;
            if offset_m1 < 256 {
                table[offset_m1] = slot;
            }
            j += 1;
        }
        slot += 1;
    }
    table
};

#[inline(always)]
fn get_offset_slot(offset: u32) -> u32 {
    debug_assert!((1..=32768).contains(&offset));
    let n = (256u32.wrapping_sub(offset)) >> 29;
    OFFSET_SLOT_SMALL[((offset - 1) >> n) as usize] as u32 + (n << 1)
}

// ── Frequency table ────────────────────────────────────────────────────────

#[derive(Clone)]
struct DeflateFreqs {
    litlen: [u32; DEFLATE_NUM_LITLEN_SYMS],
    offset: [u32; DEFLATE_NUM_OFFSET_SYMS],
}

impl DeflateFreqs {
    fn new() -> Self {
        Self {
            litlen: [0; DEFLATE_NUM_LITLEN_SYMS],
            offset: [0; DEFLATE_NUM_OFFSET_SYMS],
        }
    }

    #[allow(dead_code)]
    fn reset(&mut self) {
        self.litlen.fill(0);
        self.offset.fill(0);
    }
}

// ── Huffman code length estimation ──────────────────────────────────────────

/// Compute Huffman code lengths for a frequency table.
///
/// Uses a two-queue merge approach (sorted leaves + internal nodes) for O(n log n)
/// construction. Code lengths are capped at `max_bits` (15 for deflate).
/// Returns a Vec where `result[i]` is the code length for symbol `i` (0 if unused).
fn compute_huffman_lengths(freqs: &[u32], max_bits: u8) -> alloc::vec::Vec<u8> {
    let n = freqs.len();
    let mut lengths = alloc::vec![0u8; n];

    // Collect active symbols sorted by frequency ascending
    let mut symbols: alloc::vec::Vec<(u32, u16)> = alloc::vec::Vec::new();
    for (i, &f) in freqs.iter().enumerate() {
        if f > 0 {
            symbols.push((f, i as u16));
        }
    }

    if symbols.is_empty() {
        return lengths;
    }
    if symbols.len() == 1 {
        lengths[symbols[0].1 as usize] = 1;
        return lengths;
    }

    symbols.sort_unstable_by_key(|&(f, _)| f);

    // Two-queue Huffman construction: track depths instead of building a tree.
    // Each entry is (combined_freq, max_depth_in_subtree).
    // When two entries merge, the new depth = max(depth_a, depth_b) + 1.
    let num_symbols = symbols.len();
    let mut leaf_queue: alloc::collections::VecDeque<(u32, usize)> =
        alloc::collections::VecDeque::with_capacity(num_symbols);
    let mut internal_queue: alloc::collections::VecDeque<(u32, usize)> =
        alloc::collections::VecDeque::new();

    // We need to track which symbols end up at which depth.
    // Use a parallel array: for each leaf, its symbol index; for internal nodes,
    // the indices of the two children merged.
    // Simpler approach: use the "package-merge" insight — just compute depths
    // by tracking how many leaves are at each depth.

    // Actually, the simplest correct approach for our "total cost only" use case:
    // Build a Huffman tree tracking depths, then read off code lengths.

    // Node storage: leaves are indices 0..num_symbols, internal nodes follow.
    // Each node stores: (freq, depth, left_child, right_child, symbol_or_none)
    struct Node {
        depth: u8,
        left: usize,
        right: usize,
        symbol: u16, // u16::MAX means internal node
    }

    let mut nodes: alloc::vec::Vec<Node> = alloc::vec::Vec::with_capacity(2 * num_symbols);

    // Create leaf nodes (sorted by frequency)
    for &(freq, sym) in &symbols {
        leaf_queue.push_back((freq, nodes.len()));
        nodes.push(Node {
            depth: 0,
            left: usize::MAX,
            right: usize::MAX,
            symbol: sym,
        });
    }

    // Pick the minimum from the front of either queue
    let pick_min =
        |leaves: &mut alloc::collections::VecDeque<(u32, usize)>,
         internals: &mut alloc::collections::VecDeque<(u32, usize)>|
         -> (u32, usize) {
            match (leaves.front(), internals.front()) {
                (Some(&l), Some(&i)) => {
                    if l.0 <= i.0 {
                        leaves.pop_front().unwrap()
                    } else {
                        internals.pop_front().unwrap()
                    }
                }
                (Some(_), None) => leaves.pop_front().unwrap(),
                (None, Some(_)) => internals.pop_front().unwrap(),
                (None, None) => unreachable!(),
            }
        };

    // Merge until one node remains
    while leaf_queue.len() + internal_queue.len() > 1 {
        let (f1, idx1) = pick_min(&mut leaf_queue, &mut internal_queue);
        let (f2, idx2) = pick_min(&mut leaf_queue, &mut internal_queue);
        let combined_freq = f1 + f2;
        let new_idx = nodes.len();
        nodes.push(Node {
            depth: 0,
            left: idx1,
            right: idx2,
            symbol: u16::MAX,
        });
        internal_queue.push_back((combined_freq, new_idx));
    }

    // Compute depths via DFS from root
    let root = if let Some(&(_, idx)) = leaf_queue.front() {
        idx
    } else {
        internal_queue.front().unwrap().1
    };

    // Iterative DFS to assign depths
    let mut stack: alloc::vec::Vec<(usize, u8)> = alloc::vec::Vec::new();
    stack.push((root, 0));
    while let Some((idx, depth)) = stack.pop() {
        nodes[idx].depth = depth;
        if nodes[idx].symbol != u16::MAX {
            // Leaf — record code length (capped at max_bits)
            lengths[nodes[idx].symbol as usize] = depth.min(max_bits);
        } else {
            stack.push((nodes[idx].left, depth + 1));
            stack.push((nodes[idx].right, depth + 1));
        }
    }

    lengths
}

/// Estimate compressed size in bits using Huffman code lengths.
///
/// Computes actual Huffman code lengths (capped at 15 bits for deflate) from
/// symbol frequencies, then sums freq * code_length for the total bit cost.
fn estimate_compressed_bits(freqs: &DeflateFreqs) -> u64 {
    let litlen_lengths = compute_huffman_lengths(&freqs.litlen, 15);
    let offset_lengths = compute_huffman_lengths(&freqs.offset, 15);

    let mut bits = 0u64;

    // Sum freq * code_length for litlen symbols
    for (i, &f) in freqs.litlen.iter().enumerate() {
        if f > 0 {
            bits += f as u64 * litlen_lengths[i] as u64;
        }
    }

    // Sum freq * code_length for offset symbols
    for (i, &f) in freqs.offset.iter().enumerate() {
        if f > 0 {
            bits += f as u64 * offset_lengths[i] as u64;
        }
    }

    // Add extra bits for length and offset symbols
    let mut extra_bits = 0u64;
    for (slot, &freq) in freqs.litlen[DEFLATE_FIRST_LEN_SYM as usize..]
        .iter()
        .enumerate()
    {
        if freq > 0 && slot < 29 {
            extra_bits += freq as u64 * DEFLATE_LENGTH_EXTRA_BITS[slot] as u64;
        }
    }
    for (slot, &freq) in freqs.offset.iter().enumerate() {
        if freq > 0 && slot < 30 {
            extra_bits += freq as u64 * DEFLATE_OFFSET_EXTRA_BITS[slot] as u64;
        }
    }

    // Block overhead estimate (header + EOB)
    bits + extra_bits + 80
}

// ── Predictor ──────────────────────────────────────────────────────────────

/// Snapshot of predictor state for fork/restore.
pub(crate) struct PredictorSnapshot {
    mf: Arc<HcMatchfinder>,
    freqs: DeflateFreqs,
    in_base_offset: usize,
    next_hashes: [u32; 2],
    total_bytes_fed: usize,
    estimated_bits: u64,
    data_trimmed: usize,
}

/// Lightweight LZ77 predictor for estimating deflate-compressed sizes.
///
/// Uses a greedy HC matchfinder (equivalent to zenflate effort ~10) to find
/// LZ77 matches, counts symbol frequencies, and estimates compressed size
/// via Shannon entropy. The estimates are only used for **relative ranking**
/// of PNG filter choices, not exact size prediction.
pub(crate) struct Predictor {
    mf: HcMatchfinder,
    freqs: DeflateFreqs,
    in_base_offset: usize,
    next_hashes: [u32; 2],
    total_bytes_fed: usize,
    estimated_bits: u64,
    /// Accumulated filtered data (the matchfinder's input buffer).
    /// Periodically compacted to keep only the last MATCHFINDER_WINDOW_SIZE
    /// bytes reachable by backward references.
    data: alloc::vec::Vec<u8>,
    /// Cumulative bytes trimmed from the front of `data`.
    /// `data[i]` corresponds to logical position `i + data_trimmed`.
    data_trimmed: usize,
}

/// Greedy matchfinder parameters (equivalent to zenflate effort 10).
const NICE_LEN: u32 = 30;
const MAX_SEARCH_DEPTH: u32 = 16;

impl Predictor {
    pub fn new() -> Self {
        Self {
            mf: HcMatchfinder::new(),
            freqs: DeflateFreqs::new(),
            in_base_offset: 0,
            next_hashes: [0; 2],
            total_bytes_fed: 0,
            estimated_bits: 0,
            data: alloc::vec::Vec::with_capacity(4096),
            data_trimmed: 0,
        }
    }

    /// Feed a filtered row (filter byte + filtered data).
    /// Returns estimated cumulative compressed size in bytes.
    pub fn feed_row(&mut self, filtered_row: &[u8]) -> usize {
        let start_pos = self.data.len();
        self.data.extend_from_slice(filtered_row);
        let end_pos = self.data.len();

        // Initialize hashes if this is the first data
        if start_pos == 0 && end_pos >= 4 {
            let seq = load_u32_le(&self.data, 0);
            self.next_hashes[0] = lz_hash(seq & 0xFFFFFF, HC_MATCHFINDER_HASH3_ORDER);
            self.next_hashes[1] = lz_hash(seq, HC_MATCHFINDER_HASH4_ORDER);
        } else if start_pos > 0 && start_pos + 4 <= end_pos {
            // Re-hash at the boundary of new data
            let seq = load_u32_le(&self.data, start_pos);
            self.next_hashes[0] = lz_hash(seq & 0xFFFFFF, HC_MATCHFINDER_HASH3_ORDER);
            self.next_hashes[1] = lz_hash(seq, HC_MATCHFINDER_HASH4_ORDER);
        }

        // Run greedy LZ77 matching on the new data
        let mut pos = start_pos;
        // Need room for: load_u32_le tail check (4 bytes past best_len-3)
        // and load_u64_le in lz_extend (8 bytes). At minimum we need 5 bytes
        // remaining to enter longest_match (it returns early if max_len < 5).
        // We use safe_end = end_pos - 5 to ensure at least 5 bytes available,
        // and cap max_len to remaining - 1 so the 4-byte tail check at
        // (in_next + best_len - 3) doesn't read past end_pos.
        let safe_end = end_pos.saturating_sub(5);

        while pos < safe_end {
            let remaining = (end_pos - pos) as u32;
            // Cap at remaining-1: longest_match does load_u32_le at
            // in_next + best_len - 3, reading 4 bytes → needs best_len+1 bytes
            let max_len = (remaining - 1).min(DEFLATE_MAX_MATCH_LEN);

            let (match_len, match_offset) = self.mf.longest_match(
                &self.data,
                &mut self.in_base_offset,
                pos,
                0,
                max_len,
                NICE_LEN,
                MAX_SEARCH_DEPTH,
                &mut self.next_hashes,
            );

            if match_len >= DEFLATE_MIN_MATCH_LEN {
                // Record match: length symbol + offset symbol
                let len_slot = LENGTH_SLOT[match_len as usize] as usize;
                self.freqs.litlen[DEFLATE_FIRST_LEN_SYM as usize + len_slot] += 1;
                let off_slot = get_offset_slot(match_offset) as usize;
                self.freqs.offset[off_slot] += 1;

                // Skip matched bytes (update hash tables)
                let skip = match_len - 1;
                if skip > 0 {
                    self.mf.skip_bytes(
                        &self.data,
                        &mut self.in_base_offset,
                        pos + 1,
                        end_pos,
                        skip,
                        &mut self.next_hashes,
                    );
                }
                pos += match_len as usize;
            } else {
                // Record literal
                self.freqs.litlen[self.data[pos] as usize] += 1;
                pos += 1;
            }
        }

        // Remaining bytes at end (not enough lookahead for matching) → literals
        while pos < end_pos {
            self.freqs.litlen[self.data[pos] as usize] += 1;
            pos += 1;
        }

        // End-of-block symbol (counted once per estimation, but for relative
        // ranking it doesn't matter since it's constant across filters)
        self.total_bytes_fed += filtered_row.len();
        self.estimated_bits = estimate_compressed_bits(&self.freqs);

        // Return estimated size in bytes (ceiling)
        self.estimated_bits.div_ceil(8) as usize
    }

    /// Compact the data buffer by discarding bytes unreachable by backward
    /// references. Call between rows (not mid-filter-evaluation) to bound
    /// memory growth. Without compaction, the data buffer grows to
    /// `width * height` bytes; with it, stays under ~2 * MATCHFINDER_WINDOW_SIZE.
    pub fn compact(&mut self) {
        let window = MATCHFINDER_WINDOW_SIZE as usize;
        // Keep 1.5x window to avoid compacting too frequently
        if self.data.len() <= window + window / 2 {
            return;
        }
        // The oldest reachable backward reference is at most WINDOW_SIZE bytes
        // before the current write position. Keep WINDOW_SIZE bytes.
        let trim = self.data.len() - window;
        self.data.drain(..trim);
        self.in_base_offset = self.in_base_offset.saturating_sub(trim);
        self.data_trimmed += trim;
    }

    /// Take a snapshot for later fork/restore.
    ///
    /// The heavy matchfinder state is wrapped in an Arc — if only one snapshot
    /// exists when restored, the Arc is unwrapped without copying.
    pub fn snapshot(&self) -> PredictorSnapshot {
        PredictorSnapshot {
            mf: Arc::new(self.mf.clone()),
            freqs: self.freqs.clone(),
            in_base_offset: self.in_base_offset,
            next_hashes: self.next_hashes,
            total_bytes_fed: self.total_bytes_fed,
            estimated_bits: self.estimated_bits,
            data_trimmed: self.data_trimmed,
        }
    }

    /// Restore from a snapshot.
    ///
    /// If the snapshot's Arc has refcount=1, this is a move (no memcpy of the
    /// 256KB matchfinder). Otherwise it clones.
    pub fn restore(&mut self, snap: &PredictorSnapshot) {
        // Try to unwrap the Arc; if refcount > 1, clone
        self.mf = (*snap.mf).clone();
        self.freqs = snap.freqs.clone();
        self.in_base_offset = snap.in_base_offset;
        self.next_hashes = snap.next_hashes;
        self.total_bytes_fed = snap.total_bytes_fed;
        self.estimated_bits = snap.estimated_bits;
        self.data_trimmed = snap.data_trimmed;
        // Truncate data buffer to match the snapshot's state
        self.data.truncate(snap.total_bytes_fed - snap.data_trimmed);
    }

    /// Restore from an owned snapshot (consumes the snapshot).
    ///
    /// If the Arc has refcount=1, avoids the clone entirely.
    pub fn restore_owned(&mut self, snap: PredictorSnapshot) {
        self.mf = match Arc::try_unwrap(snap.mf) {
            Ok(mf) => mf,
            Err(arc) => (*arc).clone(),
        };
        self.freqs = snap.freqs;
        self.in_base_offset = snap.in_base_offset;
        self.next_hashes = snap.next_hashes;
        self.total_bytes_fed = snap.total_bytes_fed;
        self.estimated_bits = snap.estimated_bits;
        self.data_trimmed = snap.data_trimmed;
        self.data.truncate(snap.total_bytes_fed - snap.data_trimmed);
    }

    /// Get the current estimated compressed size in bytes.
    #[allow(dead_code)]
    pub fn estimated_size(&self) -> usize {
        self.estimated_bits.div_ceil(8) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predictor_basic() {
        let mut pred = Predictor::new();

        // Feed a simple filtered row (filter=0 None, then raw bytes)
        let row1 = vec![0u8; 100]; // all zeros — very compressible
        let size1 = pred.feed_row(&row1);
        assert!(size1 > 0, "estimated size should be positive");
        assert!(size1 < 100, "all-zeros should compress well");

        // Feed a random-ish row — should be less compressible
        let row2: alloc::vec::Vec<u8> = (0..100u8).collect();
        let size2 = pred.feed_row(&row2);
        assert!(size2 > size1, "random data should increase cumulative size");
    }

    #[test]
    fn predictor_snapshot_restore() {
        let mut pred = Predictor::new();
        let row = vec![42u8; 200];
        pred.feed_row(&row);

        let snap = pred.snapshot();
        let size_before = pred.estimated_size();

        // Feed more data
        let row2: alloc::vec::Vec<u8> = (0..200u8).collect();
        pred.feed_row(&row2);
        assert_ne!(pred.estimated_size(), size_before);

        // Restore
        pred.restore(&snap);
        assert_eq!(pred.estimated_size(), size_before);
    }

    #[test]
    fn predictor_ranking_zeros_vs_random() {
        // Zeros should estimate smaller than sequential data
        let mut pred_a = Predictor::new();
        let mut pred_b = Predictor::new();

        let zeros = vec![0u8; 500];
        let sequential: alloc::vec::Vec<u8> = (0..500u16).map(|i| (i % 256) as u8).collect();

        let size_zeros = pred_a.feed_row(&zeros);
        let size_seq = pred_b.feed_row(&sequential);

        assert!(
            size_zeros < size_seq,
            "zeros ({size_zeros}) should estimate smaller than sequential ({size_seq})"
        );
    }

    #[test]
    fn entropy_estimation_basic() {
        let mut freqs = DeflateFreqs::new();
        // Uniform distribution over 4 symbols: each should cost 2 bits
        freqs.litlen[0] = 100;
        freqs.litlen[1] = 100;
        freqs.litlen[2] = 100;
        freqs.litlen[3] = 100;
        let bits = estimate_compressed_bits(&freqs);
        // 400 symbols × 2 bits = 800 bits + 80 overhead = 880
        // Allow some tolerance for fixed-point approximation
        assert!(
            bits > 700 && bits < 1100,
            "expected ~880 bits for uniform-4, got {bits}"
        );
    }

    #[test]
    fn offset_slot_table_consistency() {
        // Verify offset slot for small offsets
        assert_eq!(get_offset_slot(1), 0);
        assert_eq!(get_offset_slot(2), 1);
        assert_eq!(get_offset_slot(3), 2);
        assert_eq!(get_offset_slot(4), 3);
        // Larger offsets
        assert_eq!(get_offset_slot(5), 4);
        assert_eq!(get_offset_slot(7), 5);
    }

    #[test]
    fn length_slot_table_consistency() {
        assert_eq!(LENGTH_SLOT[3], 0); // min match
        assert_eq!(LENGTH_SLOT[4], 1);
        assert_eq!(LENGTH_SLOT[258], 28); // max match
    }
}
