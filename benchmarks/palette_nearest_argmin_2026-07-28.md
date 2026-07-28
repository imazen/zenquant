# palette_nearest vertical argmin — 2026-07-28

Platform: Apple Silicon (aarch64, NEON), darwin 25.5.0
Bench: `benches/kernel_tiers.rs` (zenbench, interleaved arms), 1 MP mapped per measurement
Gate: `examples/output_fingerprint.rs` — 48 cells (3 image kinds × 8 palette sizes × 2 output
formats), FNV-1a over the index buffer and the palette. One image kind is a smooth gradient
specifically because it maximizes distance-tie density, which is where argmin tie-breaking is
observable at all.

## What shipped

`palette_nearest_generic` went from a per-group `reduce_min()` — plus a `to_array()` and a
scalar lane scan whenever the group improved — to a per-lane running minimum with **one**
horizontal reduce after the loop, and an index accumulator instead of recomputing
`splat(gi*8) + lane_ids` per group.

The per-group horizontal reduce sat in the loop-carried dependency chain: the next group's
compare could not start until it retired. `to_array()` additionally spills the vector to the
stack. The rewrite makes the loop body pure element-wise SIMD.

**Output is byte-identical on all 48 gate cells.**

| palette entries | NEON before | NEON after | speedup | scalar tier | tier winner |
|---|---|---|---|---|---|
| 8 | — | 17.7 ms | — | 11.2 ms | scalar 1.58× |
| 16 | 21.9 ms | 21.0 ms | 1.04× | 12.0 ms | scalar 1.75× |
| 24 | — | 22.6 ms | — | 13.5 ms | scalar 1.67× |
| 32 | — | 23.3 ms | — | 17.1 ms | scalar 1.36× |
| 48 | — | 26.3 ms | — | 21.7 ms | scalar 1.21× |
| 64 | 44.6 ms | 29.1 ms | **1.53×** | 25.7 ms | scalar 1.13× |
| 96 | — | 34.0 ms | — | 31.8 ms | scalar 1.07× |
| 128 | — | 38.3 ms | — | 47.0 ms | neon 1.23× |
| 192 | — | 48.9 ms | — | 58.4 ms | neon 1.19× |
| 256 | 84.4 ms | 59.2 ms | **1.43×** | 71.2 ms | neon 1.20× |

`batch_srgb_to_oklab` was measured at the same time and is healthy: 3.3 ms NEON vs 10.8 ms
scalar (3.3×).

Tie-breaking is preserved exactly — both shapes return the lowest index achieving the minimum
distance. `simd_lt` is strict, so the earliest group wins within a lane, and the final scan
takes the smallest packed index among tied lanes. The dropped `idx < num_entries` bounds check
was safe to drop because both constructors pad unused slots with `f32::INFINITY`, so a padding
lane's distance is `inf` and can never win.

## Measured and deliberately NOT shipped

**1. Size-conditional dispatch to the scalar tier.** The autovectorized magetypes scalar tier
beats NEON below ~112 entries (1.58×–1.75× at 8–16), with a clean monotonic crossover between
96 and 128. This is the same fix that shipped in zenpng/zenresize/zensim during this sweep —
but here it is **not safe**: the two tiers produce different quantizer output. 23 of the 48
gate cells diverge (identical palettes, different per-pixel assignment).

Cause: magetypes' scalar `mul_add` is `a*b + c` (two roundings) by a documented
cross-platform-determinism tradeoff in `nostd_math.rs`, while NEON uses a fused `vfmaq_f32`
(one rounding). The tiers therefore compute different squared distances and pick differently
at near-ties. Changing which pixels map to which palette entry is a pixel change, not a free
speedup.

Writing the distance without `mul_add` would make the tiers agree — Rust does not contract FP
by default — but that changes today's NEON output, so it is also a pixel change. It needs a
decision, not a commit.

**2. Batching the dispatch.** `nearest` is called from five per-pixel loops (`remap.rs:55`,
`median_cut.rs:1216`, `dither.rs:1816`/`1834`, `joint.rs:1123`, `palette.rs:511`), so the
`#[arcane]` `#[target_feature]` boundary is crossed per pixel; archmage's guidance is one
`#[arcane]` per hot loop, not per call. A `nearest_many` batch entry point was built and
measured: **within noise at every palette size** (pal16 −1.8% with CI [−4.2%, +0.7%], pal64
0.0%, pal256 +0.7%). Reverted.

Why it does nothing here: on aarch64 NEON is *baseline*, so `#[target_feature(enable = "neon")]`
adds nothing over the default target and LLVM inlines across the boundary freely. The per-call
`#[arcane]` cost archmage warns about is an x86/AVX2 phenomenon, where AVX2 is not baseline. It
may still be worth doing for x86 — unmeasured on this host, so not claimed.
