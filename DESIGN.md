# zenquant — AQ-Informed Palette Quantizer

## Core Insight

Existing palette quantizers (imagequant, quantizr, color_quant) treat all pixels equally. zenjpeg's AQ algorithm proves that human vision tolerates more error in textured regions than smooth ones. zenquant applies this to palette quantization:

- **Allocate more palette entries** to smooth gradients (where banding is visible)
- **Suppress dithering** in smooth regions (long runs → better compression)
- **Allow aggressive dithering** in textured regions (invisible, runs already broken)
- **Order palette entries** to minimize index deltas (proven by WebP's VP8L `kMinimizeDelta`)

The goal: better perceptual quality AND smaller compressed output by making smarter per-region decisions.

## Algorithm Pipeline

### 1. OKLab Color Space (`oklab.rs`)

All palette operations happen in OKLab (Bjorn Ottosson). Perceptual uniformity means Euclidean distance ≈ perceived difference. Inline sRGB↔linear transfer + OKLab conversion matrices — no external color science deps.

### 2. Simplified AQ Pipeline (`masking.rs`)

Stripped-down zenjpeg AQ, keeping the codec-agnostic parts:

1. **Local contrast**: per-pixel `(L - avg_4_neighbors)²`, clamped to 0.2. OKLab L is already perceptual, so we skip zenjpeg's `RatioOfDerivatives`.
2. **Erosion to 4×4 blocks**: min-biased weighted average of the 4 smallest values per block. Weights `[0.40, 0.25, 0.20, 0.15]` — heavier min-bias than jpegli for banding protection.
3. **Masking → weight**: `weight = 0.1 + 0.9 / (1.0 + K * sqrt(masking))`. Low contrast (smooth) → high weight; high contrast (texture) → low weight.
4. **Bilinear upscale** back to per-pixel.

Output: per-pixel weights in [0.1, 1.0].

### 3. Weighted Histogram (`histogram.rs`)

Hash from quantized OKLab (12-bit, 4096 buckets max) to accumulated weight + centroid. Each pixel contributes its AQ weight, not 1.0. This biases palette allocation toward perceptually important (smooth) regions.

### 4. Weighted Median Cut (`median_cut.rs`)

Modified median cut in OKLab space:
- **Split criterion**: `weighted_count × volume`
- **Split axis**: largest OKLab range
- **Split point**: weighted median (not population median)
- **Centroid**: weight-averaged OKLab

Optional k-means refinement (3 iterations) when quality ≥ 50.

### 5. Delta-Minimizing Palette Sort (`palette.rs`)

Greedy nearest-neighbor TSP from darkest entry. Ensures similar colors have adjacent indices → small index deltas → better LZW/deflate/PNG compression.

### 6. Run-Biased Remapping (`remap.rs`)

For each pixel, find K=4 nearest palette entries. If the previous pixel's index is among candidates and the quality penalty is within an AQ-modulated threshold, prefer it (extends the run). Smooth regions get lower thresholds (quality matters), textured regions get higher (runs matter, error masked).

### 7. AQ-Adaptive Error Diffusion (`dither.rs`)

Floyd-Steinberg with AQ modulation: error is diffused normally but *received* with the target pixel's AQ weight. Smooth regions get full error correction; textured regions get damped dithering. Optional run-aware dither suppression for compression priority.

## Public API

```rust
let result = zenquant::quantize(pixels, width, height, &QuantizeConfig::default())?;

let config = QuantizeConfig::new()
    .max_colors(256)
    .quality(85)
    .run_priority(RunPriority::Balanced)
    .dither(DitherMode::Adaptive);

let result = zenquant::quantize(pixels, width, height, &config)?;

result.palette()           // &[[u8; 3]] — sRGB, delta-sorted
result.indices()           // &[u8] — palette indices
result.transparent_index() // Option<u8>
```

Input: `&[rgb::RGB<u8>]` or `&[rgb::RGBA<u8>]`. RGBA with alpha=0 gets a dedicated transparent index.
