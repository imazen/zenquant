# Animation with zenquant

How to use zenquant for GIF, APNG, and animated WebP encoding. Covers shared palettes, per-frame remapping, temporal stability, quality control, and the tradeoffs between the four dither modes.

## The pipeline

Animation encoding with palette quantization has two phases:

1. **Build a shared palette** from representative frames. One palette, used across all frames.
2. **Remap each frame** against that palette. Each frame gets its own index buffer; the palette stays fixed.

zenquant handles both phases. Subframe cropping, dirty-rect detection, disposal methods, and LZW/deflate encoding are the caller's responsibility — zenquant operates on full-frame pixel buffers and returns palette + indices.

## Shared palette construction

```rust
use zenquant::{QuantizeConfig, OutputFormat, ImgRef};

let config = QuantizeConfig::new(OutputFormat::Gif);

let frames: Vec<ImgRef<'_, rgb::RGBA<u8>>> = frame_data.iter()
    .map(|f| ImgRef::new(f, width, height))
    .collect();
let shared = zenquant::build_palette_rgba(&frames, &config).unwrap();
```

`build_palette_rgba` (and `build_palette` for RGB) accepts frames with different dimensions. Internally it merges AQ-weighted histograms from all frames, runs median cut, then refines centroids with cross-frame k-means. The result contains only the palette — no indices yet.

For GIF, pass `OutputFormat::Gif`. This sets binary transparency (one palette entry reserved for alpha==0 pixels), delta-minimize palette sort (adjacent palette entries are similar colors, which helps LZW), and tuned dither/run parameters.

## Per-frame remapping

```rust
for frame_pixels in &frame_data {
    let result = shared.remap_rgba(frame_pixels, width, height, &config).unwrap();
    // result.palette_rgba() — same palette every frame
    // result.indices()      — frame-specific index map
    // result.transparent_index() — which index means "transparent" (GIF)
}
```

Each `remap_rgba` call applies dithering and run optimization from the config, but doesn't touch the palette. Palette order is preserved — no frequency reordering happens during remap, so all frames share the same color table.

`remap` works the same way for RGB frames.

## Dither modes for animation

The dither mode is the single biggest decision for animation quality. zenquant has four, each with different temporal behavior.

### Floyd-Steinberg (default: `Adaptive`)

Standard serpentine error diffusion with AQ masking. Best single-frame quality. Worst temporal stability: if any pixel changes, error cascades right and down, potentially changing every downstream pixel. A moving sprite over a static background causes the entire background to flicker.

This is the default because it produces the best results for still images and single-frame use.

### Sierra Lite (`SierraLite`)

Lighter error diffusion — 3 neighbors instead of 4 (no diagonal-forward), with weights 2/4, 1/4, 1/4 instead of Floyd-Steinberg's 7/16, 3/16, 5/16, 1/16. Less error cascade means a changed pixel affects fewer downstream pixels. Temporal flicker is reduced but not eliminated.

Good middle ground: nearly as sharp as Floyd-Steinberg on gradients, noticeably less flicker on animations with partial-frame changes.

```rust
let config = QuantizeConfig::new(OutputFormat::Gif)
    ._sierra_lite_dither();
```

### Blue Noise (`BlueNoise`)

Position-deterministic ordered dithering. Each pixel's dither noise comes from a tiled 64x64 blue noise threshold map, modulated by an edge-aware dither map. No error propagation at all — each pixel is independent.

Zero temporal flicker. If a pixel doesn't change between frames, its index won't change either. Static backgrounds are perfectly stable regardless of what moves on top.

The tradeoff: single-frame quality is lower than error diffusion. Gradients won't be as smooth. But for animation, the temporal stability often matters more than per-frame sharpness.

```rust
let config = QuantizeConfig::new(OutputFormat::Gif)
    ._blue_noise_dither();
```

### No dither (`None`)

Nearest-color mapping with no dither at all. Maximum compression (longest possible runs), zero flicker, but visible banding on gradients. Useful for flat-color content like pixel art or UI elements, or as a fallback for high-motion sequences where dither patterns would be invisible anyway.

```rust
let config = QuantizeConfig::new(OutputFormat::Gif)
    ._no_dither();
```

### Which mode to pick?

| Mode | Single-frame quality | Temporal stability | Compression | Best for |
|------|---------------------|--------------------|-------------|----------|
| Adaptive (F-S) | Best | Poor | Good | Stills, single frame |
| SierraLite | Good | Moderate | Good | Animation with partial changes |
| BlueNoise | Fair | Perfect | Good | Animation where flicker is unacceptable |
| None | Poor (banding) | Perfect | Best | Flat art, pixel art, high-motion |

For most GIF/APNG encoding, start with `SierraLite` or `BlueNoise`. Use `Adaptive` only if temporal artifacts aren't a concern (e.g., every frame changes completely anyway).

## Temporal clamping

Temporal clamping locks pixels to their previous frame's index when the source color hasn't changed. It works with `Adaptive` and `SierraLite` dither modes, and is redundant for `BlueNoise` and `None` (which are already position-deterministic).

Pass the previous frame's indices via `remap_with_prev` or `remap_rgba_with_prev`:

```rust
let config = QuantizeConfig::new(OutputFormat::Gif)
    ._sierra_lite_dither();

let shared = zenquant::build_palette_rgba(&frames, &config).unwrap();

let mut prev_indices: Option<Vec<u8>> = None;

for frame_pixels in &frame_data {
    let result = if let Some(ref prev) = prev_indices {
        shared.remap_rgba_with_prev(frame_pixels, width, height, &config, prev).unwrap()
    } else {
        shared.remap_rgba(frame_pixels, width, height, &config).unwrap()
    };

    prev_indices = Some(result.indices().to_vec());
    // encode frame...
}
```

### How it works

For each pixel, the ditherer computes the undithered nearest palette entry (what you'd get with `DitherMode::None`). If that matches `prev_indices[i]`, the pixel is "locked" — it keeps the previous index. Error still diffuses through locked pixels so neighboring unlocked pixels don't see seams.

This means temporal clamping is conservative: it only locks pixels where the source→palette mapping is identical between frames. Source pixels that change even slightly (e.g., video noise, compression artifacts) get a fresh dither pass.

### Full-frame buffers required

Both `pixels` and `prev_indices` must be full-frame buffers (`width * height` elements). If your encoder does subframe cropping, composite the subframe back onto the full canvas before calling remap. Blue noise relies on frame-absolute pixel coordinates — a cropped subregion would shift the noise tile and break temporal coherence.

## Run priority and compression

Run priority controls how aggressively zenquant extends index runs (consecutive identical indices). Longer runs compress better in LZW (GIF) and entropy coding (WebP), but extending a run means picking a slightly-worse palette entry for some pixels.

Three levels:

- **Quality** — no run extension. Every pixel gets its best match.
- **Balanced** — modest run extension where AQ masking indicates the eye won't notice. Default for most formats.
- **Compression** — aggressive run extension, especially in textured/noisy regions.

```rust
let config = QuantizeConfig::new(OutputFormat::Gif)
    ._run_priority_compression();
```

For animation, `Compression` often makes sense — GIF's LZW codec rewards long runs heavily, and the quality difference is concentrated in regions where the eye is least sensitive (thanks to AQ masking).

### Viterbi optimization

At `Quality::Best` (the default), zenquant runs a per-scanline Viterbi dynamic programming pass after dithering. This finds the globally optimal index sequence balancing color accuracy against run-length extension — similar to trellis quantization in video codecs.

At `Quality::Balanced`, a faster greedy run-extension pass is used instead. At `Quality::Fast`, no post-dither optimization runs.

The Viterbi lambda (tradeoff parameter) scales per format: GIF gets 3x the base lambda because LZW rewards runs so heavily.

## Quality control

### Per-frame quality measurement

Enable quality metrics to get SSIMULACRA2 and butteraugli estimates for each frame:

```rust
let config = QuantizeConfig::new(OutputFormat::Gif)
    .compute_quality_metric(true);

let result = shared.remap_rgba(frame_pixels, width, height, &config).unwrap();
let ssim2 = result.ssimulacra2_estimate().unwrap();  // 0–100, higher = better
let ba = result.butteraugli_estimate().unwrap();       // 0+, lower = better
```

These are fast estimates based on a calibrated MPE→metric lookup table, not full reference metric computations.

### Quality floors

Set `min_ssim2` to enforce a per-frame quality floor. If a frame's estimated SSIMULACRA2 falls below the threshold, `remap` returns `QualityNotMet` instead of a result. The encoder can then fall back to truecolor for that frame:

```rust
let config = QuantizeConfig::new(OutputFormat::Png)
    .min_ssim2(75.0);

match shared.remap_rgba(frame_pixels, width, height, &config) {
    Ok(result) => { /* encode as indexed */ }
    Err(QuantizeError::QualityNotMet { achieved_ssim2, .. }) => {
        // frame needs truecolor — quantization quality too low
    }
    Err(e) => panic!("{e}"),
}
```

This works with both `remap` and `remap_with_prev`.

### Auto-tuned compression tiers

`target_ssim2` auto-selects quality preset, dither strength, and run priority to maximize compression while staying above a target score:

```rust
let config = QuantizeConfig::new(OutputFormat::Gif)
    .target_ssim2(80.0);
```

Internally, this selects from 5 compression tiers (calibrated against the CID22 corpus). Higher targets use conservative settings; lower targets allow aggressive compression.

## GIF-specific features

### Binary transparency

With `OutputFormat::Gif`, one palette entry is reserved for fully transparent pixels (alpha==0). Access it via `result.transparent_index()`. Pixels with any alpha > 0 are treated as opaque — GIF doesn't support partial transparency.

### Palette ordering

GIF uses two palette optimizations:

1. **Delta-minimize sort** — greedy nearest-neighbor ordering so adjacent palette indices map to similar colors. Smaller index deltas → better LZW compression.
2. **Frequency reorder** (quantize path only) — after remapping, palette entries are reordered by usage frequency so the most common colors get low indices. This helps LZW dictionary construction.

Frequency reordering only happens in `quantize` / `quantize_rgba`. The `remap` path skips it because palette order must stay stable across frames when using a shared palette.

## Format tuning reference

zenquant auto-tunes several parameters per output format:

| Parameter | GIF | PNG | WebP Lossless |
|-----------|-----|-----|---------------|
| Default dither strength | 0.35 | 0.50 | 0.40 |
| Palette sort | Delta-minimize | Luminance | Delta-minimize |
| Frequency reorder | Yes (quantize only) | No | No |
| Alpha mode | Binary (0 or 255) | Full (0–255) | Full (0–255) |
| Viterbi lambda scale | 3.0x | 1.0x | 2.0x |

Lower dither strength for GIF because dithering breaks index runs, and LZW compression depends heavily on run length. Higher lambda scale for the same reason.

PNG uses luminance sort because PNG's scanline prediction filters (sub, up, average) exploit spatial locality — pixels near each other tend to have similar lightness.

## Putting it all together

A complete GIF animation encoding loop with temporal clamping and quality control:

```rust
use zenquant::{QuantizeConfig, QuantizeError, OutputFormat, ImgRef};

let config = QuantizeConfig::new(OutputFormat::Gif)
    ._sierra_lite_dither()
    .min_ssim2(70.0);

// Build shared palette
let frame_refs: Vec<ImgRef<'_, rgb::RGBA<u8>>> = frame_data.iter()
    .map(|f| ImgRef::new(f, width, height))
    .collect();
let shared = zenquant::build_palette_rgba(&frame_refs, &config).unwrap();

let mut prev_indices: Option<Vec<u8>> = None;

for frame_pixels in &frame_data {
    let remap_result = if let Some(ref prev) = prev_indices {
        shared.remap_rgba_with_prev(frame_pixels, width, height, &config, prev)
    } else {
        shared.remap_rgba(frame_pixels, width, height, &config)
    };

    match remap_result {
        Ok(result) => {
            let indices = result.indices();
            let palette = result.palette_rgba();
            let transparent = result.transparent_index();
            prev_indices = Some(indices.to_vec());
            // encode indexed frame...
        }
        Err(QuantizeError::QualityNotMet { achieved_ssim2, .. }) => {
            prev_indices = None; // break temporal chain
            // encode as truecolor frame, or use more palette colors
        }
        Err(e) => panic!("{e}"),
    }
}
```

Note: when a frame falls back to truecolor, set `prev_indices` to `None` — the next indexed frame shouldn't clamp against a truecolor frame's indices.

## Expert methods

These `#[doc(hidden)]` methods give fine-grained control. They're stable enough for animation encoders to depend on, but not part of the documented public API yet.

| Method | Effect |
|--------|--------|
| `._no_dither()` | Disable dithering entirely |
| `._blue_noise_dither()` | Position-deterministic blue noise |
| `._sierra_lite_dither()` | Lighter error diffusion |
| `._dither_strength(f32)` | Override dither strength (0.0–1.0) |
| `._run_priority_quality()` | No run extension bias |
| `._run_priority_compression()` | Aggressive run extension |
| `._viterbi_lambda(f32)` | Override Viterbi tradeoff parameter |
| `.remap_with_prev(pixels, w, h, config, prev)` | RGB remap with temporal clamping |
| `.remap_rgba_with_prev(pixels, w, h, config, prev)` | RGBA remap with temporal clamping |
