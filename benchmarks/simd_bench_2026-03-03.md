# SIMD Acceleration Benchmark — 2026-03-03

Commit: 75b1452 (feat: add SIMD acceleration via archmage/magetypes)
Platform: WSL2 Linux 6.6.87.2 (x86_64, AVX2+FMA)
Dataset: CID22-512 validation (41 images, 512x512)
Command: `cargo run --release --example time_only --features _dev`

## time_only: Average ms/image (3 interleaved runs)

| Preset   | Before (avg) | After (avg) | Speedup |
|----------|-------------|------------|---------|
| fast     | 56.2        | 50.2       | 1.12x   |
| balanced | 75.8        | 70.9       | 1.07x   |
| best     | 123.0       | 117.9      | 1.04x   |
| iq       | 42.9        | 43.1       | —       |
| qr       | 29.7        | 29.5       | —       |

## profile_steps: Single image (1025469.png, 512x512)

| Step              | Before (ms) | After (ms) | Notes                     |
|-------------------|-------------|------------|---------------------------|
| AQ masking        | 9.8         | 9.6        | No change expected        |
| Histogram         | 10.7        | 7.8        | SIMD batch conversion     |
| Median cut        | 13.2        | 13.8       | No change expected        |
| K-means (8it)     | 35.5        | 36.2       | No change expected        |
| Palette+cache     | 1.6         | 1.2        | SIMD nearest in NN cache  |
| Dithering         | 25.7        | 23.0       | SIMD batch conversion     |
| Viterbi           | 22.0        | 22.4       | No change expected        |
| **Total**         | **118.4**   | **113.9**  | **1.04x**                 |

## Analysis

The SIMD acceleration provides a consistent 4-12% overall speedup depending on
the quality preset. The "fast" preset benefits most (12%) because batch
conversion and NN cache lookup are a larger fraction of total time when k-means
refinement and Viterbi steps are skipped or reduced.

Per-step breakdown shows the wins are concentrated in:
- **Histogram** (batch sRGB→OKLab conversion): ~27% faster
- **Palette+cache** (SIMD nearest-neighbor): ~25% faster
- **Dithering** (batch sRGB→OKLab conversion): ~10% faster

The sequential dither inner loop (error diffusion) and Viterbi refinement
dominate wall time at higher quality presets and remain scalar, limiting
overall speedup to Amdahl's law predictions.

Potential further gains:
- SIMD-accelerate k-means refinement (batch conversion + nearest per iter)
- SIMD-accelerate AQ masking (batch conversion of neighborhoods)
- SIMD-accelerate Viterbi path scoring
