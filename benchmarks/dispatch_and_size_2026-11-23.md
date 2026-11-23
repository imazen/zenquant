# zenquant dispatch & code-size optimization — 2026-11-23

Baseline `54da0ca5` (`chore(deps): upgrade archmage/magetypes to 0.9.33`),
leaf-scope change on `main`. Worker `swe-zensim`, lane
`bench-lanes-2026-11-23`. All benchmarks ran on `i134` (Intel i5-13400F)
under `taskset -c 2` + `nice -n 19` + `ZENBENCH_CPU_POLICY=unrestricted`,
serialized by `flock` on the lane lock. No pushes.

## Scope

Three `[LEAF]` items from `briefs/swe-zenquant.md`:

1. **Outline `diffuse_err_3ch` / `diffuse_err_4ch`** (`src/dither.rs`):
   `#[inline(always)]` → `#[inline(never)]`. Each helper is expanded 4–8×
   per pixel inside `diffuse_kernel_3ch!` / `diffuse_kernel_3ch_opaque!` /
   `diffuse_kernel_4ch_opaque!` across six row functions. Bodies unchanged —
   identical IEEE-754 evaluation order, so identical output indices.
2. **Drop per-row `Box<dyn Iterator<Item = usize>>`** at six sites in
   `src/dither.rs` (lines 589/782/990/1165/1276/1398 on the baseline).
   Replaced with `for x_step in 0..width { let x = if forward { x_step }
   else { width - 1 - x_step } }` — same forward/reverse traversal order,
   zero heap traffic and no per-pixel vtable dispatch. `alloc::boxed::Box`
   import removed.
3. **Hoist `incant!` dispatch above the pixel loop** for
   `PaletteSimd::nearest` (`src/simd.rs`): new `nearest_batch(&[OKLab],
   &mut [u8])` resolves the tier once per batch via
   `palette_nearest_batch_dispatch_{v3,neon,wasm128,scalar}`, each looping
   the same `palette_nearest_generic` kernel. The aarch64
   `NEON_MIN_PALETTE_ENTRIES` cutoff is checked once per batch on
   `num_entries` only — identical decisions, identical results.
   Callers converted: `Palette::rebuild_nn_cache` (`src/palette.rs`),
   `build_centroid_nn_cache` (`src/median_cut.rs`). The per-call
   `nearest` + `palette_nearest_dispatch_*` fns are now
   `#[cfg(any(feature = "_dev", test))]` — production never calls them;
   the `kernel_tiers` bench keeps measuring the single-call shape.

## Output parity

`zenquant-parity` probe (deterministic LCG noise, 192×128, weights
0.5–1.0, fixed 64-entry palette): 38 output records carrying **44 FNV-1a
hash comparisons** —

- `dither_image` / `dither_image_rgba` / `dither_image_rgba_alpha` ×
  {None, Adaptive, SierraLite, BlueNoise, Linear, Ordered} at
  strength 1.0 (18 records / 18 hashes),
- the same sweep at reduced strengths {0.3, 0.7} for the three
  error-diffusion modes on `dither_image` and `dither_image_rgba` only
  — `dither_image_rgba_alpha` is **not** covered at reduced strength
  (12 records / 12 hashes),
- `quantize` + `quantize_rgba` index hashes × all six modes at default
  strength (6 records / 12 hashes — the two per record),
- `quantize` RGB only, Adaptive at {0.3, 0.7} — no RGBA
  reduced-strength end-to-end coverage (2 records / 2 hashes).

Baseline export vs working tree: **all 44 hashes identical** (golden +
after logs in `~/tmp/bench-lanes-2026-11-23/zenquant-parity/`).

## Measurements

### Binary / Wasm (i134)

| metric | BEFORE | AFTER | Δ |
|---|---|---|---|
| wasm probe, stripped | 157,905 B | 131,337 B | **−26,568 (−16.8%)** |
| wasm probe, gzip -9 | 55,901 B | 53,854 B | −2,047 (−3.7%) |
| test-bin ELF `.text` section (readelf) | 1,057,477 B | 1,028,613 B | −28,864 (−2.73%) |
| test-bin GNU `size` "text" aggregate | 1,370,305 B | 1,340,885 B | −29,420 (−2.15%) |
| `dither_image` sym | 26,336 B | 16,221 B | −38.4% |
| `dither_image_rgba` sym | 28,476 B | 18,761 B | −34.1% |
| `dither_image_rgba_alpha` sym | 28,525 B | 16,515 B | −42.1% |
| `diffuse_err_3ch`/`_4ch` syms | 0 (inlined) | 1 × 531 + 1 × 672 B | outlined |
| `palette_nearest_dispatch_v3` | 1 × 992 B | 1 × 992 B (cfg `_dev`/`test` only; absent in shipped lib) | |
| `palette_nearest_batch_dispatch_v3` | — | 1 × 1,168 B | new |
| `dyn Iterator`/`Rev<Range>` tokens | 5 | 0 | −100% |

Note: test-bin keeps `palette_nearest_dispatch_v3` because `#[cfg(test)]`
retains the single-call path for unit tests; the shipped rlib drops it
entirely (`_dev`-gate). The wasm gzip delta is much smaller than raw
because deflate already deduplicates the repeated inlined copies.

### Attribution bench — `dither_quant` (256×256 noise, zenbench, cpu 2)

| arm | BEFORE | AFTER | Δ (of means) |
|---|---|---|---|
| `quantize_rgb_256` | 212.56 ±1.0 ms | 212.77 ±1.1 ms | +0.10% |
| `dither_rgb_adaptive` | 17.94 ±0.0 ms | 18.11 ±0.1 ms | +0.95% |
| `dither_rgb_sierralite` | 17.59 ±0.0 ms | 17.72 ±0.1 ms | +0.74% |
| `dither_rgb_linear_1` | 17.54 ±0.0 ms | 17.43 ±0.1 ms | −0.63% |
| `dither_rgba_adaptive` | 17.86 ±0.0 ms | 18.10 ±0.1 ms | +1.34% |
| `dither_rgba_sierralite` | 17.61 ±0.0 ms | 17.97 ±0.2 ms | +2.04% |
| `dither_rgba_linear_1` | 17.30 ±0.0 ms | 17.56 ±0.1 ms | +1.50% |

Observed deltas range −0.6% to +2.0% on the six dither arms; the
outlined `diffuse_err_*` helpers are now one call per neighbor per
pixel (brief item 1) and the boxed-iterator alloc+dispatch per row is
gone (item 2). `quantize_rgb_256` moved +0.10% (±1.0–1.1 ms MAD,
CIs overlap).

### `kernel_tiers` — `palette_nearest/1MP` (per-call dispatch shape, unchanged by design)

| palette | v3 BEFORE | v3 AFTER | scalar BEFORE | scalar AFTER |
|---|---|---|---|---|
| 8 | 21.6 ms | 22.4 ms | 35.8 ms | 35.8 ms |
| 16 | 26.6 ms | 27.8 ms | 43.2 ms | 41.9 ms |
| 24 | 27.1 ms | 28.4 ms | 44.5 ms | 44.4 ms |
| 32 | 30.7 ms | 33.4 ms | 47.3 ms | 45.9 ms |
| 48 | 35.8 ms | 36.5 ms | 53.5 ms | 52.9 ms |
| 64 | 41.5 ms | 42.8 ms | 59.1 ms | 58.0 ms |
| 96 | 50.9 ms | 52.6 ms | 68.9 ms | 68.7 ms |
| 104 | 53.4 ms | 54.1 ms | 72.9 ms | 71.4 ms |
| 112 | 55.9 ms | 58.4 ms | 74.7 ms | 73.4 ms |
| 120 | 57.3 ms | 59.8 ms | 77.6 ms | 76.8 ms |
| 128 | 60.4 ms | 61.5 ms | 79.5 ms | 80.6 ms |
| 192 | 82.7 ms | 82.4 ms | 101.6 ms | 105.7 ms |
| 256 | 102.3 ms | 101.8 ms | 122.0 ms | 132.1 ms |

`batch_srgb_to_oklab/1MP`: v3 2.7 ms (381M ops/s) vs scalar 14.5 ms
(72.4M ops/s) — identical BEFORE/AFTER, untouched kernel sentinel.

This bench measures the **per-call** dispatch shape (`Pal::map` →
`nearest`), which the change leaves intact under `_dev`/`test` — the
kernel source and call shape are identical BEFORE/AFTER. Observed
deltas: v3 −0.5%…+8.8% across the sweep (pal32 +8.8%, pal192 −0.4%,
pal256 −0.5%); scalar −3.0%…+8.3% (pal256 +8.3%, pal192 +4.0%,
pal16/32 −3.0%). These are two different binaries so cross-run spread
is reported as measured, not attributed.

## Verification

Final-tree gates re-run on the committed state into
`~/tmp/bench-lanes-2026-11-23/zenquant-final-gates.log`:

- `cargo fmt --check` — clean
- `cargo clippy --all-targets -- -D warnings` — clean
- `cargo test --release --lib --tests --examples` — 24 suites + lib
  (100 tests), all pass
- `cargo check --target wasm32-unknown-unknown` — clean (default + `_dev`)
- Parity: 44/44 output hashes identical to baseline (see above)
