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

### 8. Format-Specific Optimization

Each output format has different compression characteristics. `OutputFormat` controls palette sort strategy, dither strength, alpha handling, and post-processing:

| Format | Sort | Dither | Alpha | Post-pass |
|--------|------|--------|-------|-----------|
| Generic | DeltaMinimize | 0.5 | Full 4D | — |
| Gif | DeltaMinimize | 0.5 | Binary (0/255) | Frequency reorder |
| Png | Luminance | 0.3 | Full 4D + tRNS | — |
| WebpLossless | DeltaMinimize | 0.5 | Full 4D | — |
| JxlModular | DeltaMinimize | 0.4 | Full 4D | — |

**Luminance sort** (PNG): Orders palette by OKLab L ascending. Spatially neighboring pixels tend to have similar lightness, so scanline filters (sub/up) produce small deltas → better deflate compression.

**Frequency reorder** (GIF): After dithering, sorts palette by descending usage frequency. Most-common colors get lowest indices, helping LZW dictionary construction.

**Full alpha** (Generic/PNG/WebP/JXL): 4D OKLabA quantization. Alpha is a quantizable dimension — the pipeline builds histogram, median-cuts, and k-means-refines in 4D space. Each palette entry has its own alpha value. Distance metric premultiplies by alpha so transparent pixel color differences matter less.

**Binary alpha** (GIF): Classic opaque + transparent index approach. Alpha==0 → transparent, else opaque.

### 9. Already-Paletted Fast Path

At the start of `quantize`/`quantize_rgba`, images with ≤`max_colors` unique colors are detected via BTreeSet scan with early exit. When triggered: skip masking, histogram, median cut, k-means, and dithering. Build palette directly from exact colors, apply format-specific sort, simple nearest-color remap. Lossless for already-paletted images.

## Public API

```rust
use zenquant::{QuantizeConfig, OutputFormat, DitherMode, RunPriority};

// Default: Generic format, 256 colors, quality 85
let result = zenquant::quantize(pixels, width, height, &QuantizeConfig::default())?;

// Format-specific optimization
let config = QuantizeConfig::new()
    .max_colors(256)
    .quality(85)
    .output_format(OutputFormat::Png)      // PNG-optimized defaults
    .dither(DitherMode::Adaptive)          // user override
    .run_priority(RunPriority::Balanced);

let result = zenquant::quantize(pixels, width, height, &config)?;

result.palette()           // &[[u8; 3]] — sRGB entries
result.palette_rgba()      // &[[u8; 4]] — RGBA entries (alpha per index)
result.indices()           // &[u8] — palette indices per pixel
result.transparent_index() // Option<u8>
result.alpha_table()       // Option<Vec<u8>> — for PNG tRNS chunk
result.palette_len()       // usize
```

Input: `&[rgb::RGB<u8>]` or `&[rgb::RGBA<u8>]`. RGBA with alpha=0 gets a dedicated transparent index. Format selection controls whether alpha is binary (GIF) or quantized as a 4th dimension (everything else).
