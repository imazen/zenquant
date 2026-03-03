# SIMD Acceleration Benchmark — 2026-03-03

Platform: WSL2 Linux 6.6.87.2 (x86_64, AVX2+FMA)
Dataset: CID22-512 validation (41 images, 512x512)

## Phase 1: Batch conversion + palette NN cache (75b1452)

### time_only: Average ms/image (3 interleaved runs)

| Preset   | Before | Phase 1 | Speedup |
|----------|--------|---------|---------|
| fast     | 56.2   | 50.2    | 1.12x   |
| balanced | 75.8   | 70.9    | 1.07x   |
| best     | 123.0  | 117.9   | 1.04x   |

## Phase 2: + K-means SIMD (50427cf)

SIMD batch conversion for pixel pre-conversion, grid precomputation,
and NN cache build inside k-means refinement. Also fixed
refine_against_pixels_alpha to pre-convert once instead of per-iteration.

### time_only: Average ms/image (3 interleaved runs)

| Preset   | Before | Phase 2 | Speedup |
|----------|--------|---------|---------|
| fast     | 56.0   | 50.1    | 1.12x   |
| balanced | 76.0   | 67.5    | 1.13x   |
| best     | 123.0  | 115.8   | 1.06x   |

### Quality verification (speed_test, 41 images)

| Preset      | BA (before) | BA (after) | SS2 (before) | SS2 (after) |
|-------------|-------------|------------|--------------|-------------|
| fast        | 3.920       | 3.920      | 78.70        | 78.70       |
| balanced    | 3.698       | 3.698      | 79.91        | 79.91       |
| best        | 3.571       | 3.571      | 79.74        | 79.75       |

Deflate and run-length metrics also unchanged within noise.

### profile_steps: Single image (1025469.png, 512x512)

| Step              | Before (ms) | Phase 2 (ms) | Change |
|-------------------|-------------|--------------|--------|
| AQ masking        | 9.9         | 10.6         | noise  |
| Histogram         | 10.6        | 7.8          | -26%   |
| Median cut        | 13.7        | 13.3         | noise  |
| K-means (2it)     | 15.2        | 12.0         | -21%   |
| K-means (8it)     | 37.0        | 33.5         | -9%    |
| Palette+cache     | 1.7         | 1.2          | -29%   |
| Dithering         | 25.0        | 23.6         | -6%    |
| Viterbi           | 21.9        | 22.6         | noise  |
| **Total**         | **119.9**   | **112.5**    | **-6%**|

## Analysis

Cumulative SIMD speedup across both phases:
- **fast**: 56→50ms (1.12x) — no k-means, wins from batch conversion + NN cache
- **balanced**: 76→67.5ms (1.13x) — 2-iter k-means benefits most from SIMD
  pre-conversion being a larger fraction of k-means total work
- **best**: 123→116ms (1.06x) — 8-iter k-means + Viterbi dominate, limiting gains

K-means (2it) improved 21% because the one-time SIMD batch conversion
(pixels + grid) and SIMD NN cache build are a large fraction of 2-iteration
work. K-means (8it) improved only 9% because the per-iteration seeded NN
search (still scalar) dominates at higher iteration counts.

Remaining scalar bottlenecks (by wall-time contribution):
1. K-means per-iteration loop (seeded NN search) — ~30ms for 8 iters
2. Viterbi DP — ~22ms, sequential by nature
3. Dither inner loop (error diffusion) — ~24ms, sequential
4. AQ masking — ~10ms
5. Median cut — ~13ms

Potential further optimizations:
- Online/minibatch k-means (cap samples instead of full passes) — algorithmic
- SIMD-accelerate AQ masking (batch conversion of neighborhoods)
- SIMD-accelerate the per-iteration seeded NN search in k-means
