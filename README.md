# zenquant

Color quantization with perceptual masking. Reduces truecolor images to 256-color indexed palettes in OKLab space, using butteraugli-inspired adaptive quantization (AQ) weights to concentrate palette entries where human vision is most sensitive.

## What it does

Most quantizers treat every pixel equally. zenquant spends palette entries on smooth gradients, skin tones, and other regions where banding is visible — and wastes fewer entries on noisy textures where the eye can't tell the difference.

The pipeline: histogram in OKLab → median cut → k-means refinement with AQ weights → format-aware palette sorting → adaptive Floyd-Steinberg dithering → optional Viterbi DP for run-length optimization.

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
zenquant = "0.1"
```

### Quantize an RGB image

```rust
use zenquant::{QuantizeConfig, OutputFormat};

let config = QuantizeConfig::new(OutputFormat::Png);
let result = zenquant::quantize(&pixels, width, height, &config).unwrap();

let palette = result.palette();   // &[[u8; 3]] — sRGB
let indices = result.indices();   // &[u8] — row-major
```

### Quantize RGBA (GIF with transparency)

```rust
use zenquant::{QuantizeConfig, OutputFormat};

let config = QuantizeConfig::new(OutputFormat::Gif);
let result = zenquant::quantize_rgba(&pixels, width, height, &config).unwrap();

// Binary transparency: one palette entry reserved for transparent pixels
if let Some(idx) = result.transparent_index() {
    // pixels with alpha == 0 map to this index
}
```

### Write an indexed PNG

```rust
use zenquant::{QuantizeConfig, OutputFormat};

let config = QuantizeConfig::new(OutputFormat::Png);
let result = zenquant::quantize(&pixels, width, height, &config).unwrap();

let mut encoder = png::Encoder::new(file, width as u32, height as u32);
encoder.set_color(png::ColorType::Indexed);
encoder.set_depth(png::BitDepth::Eight);
encoder.set_palette(result.palette().iter().flat_map(|c| *c).collect::<Vec<_>>());

if let Some(trns) = result.alpha_table() {
    encoder.set_trns(trns);
}

let mut writer = encoder.write_header().unwrap();
writer.write_image_data(result.indices()).unwrap();
```

### Shared palette for animations

Build one palette from multiple frames, then remap each frame against it:

```rust
use zenquant::{QuantizeConfig, OutputFormat, ImgRef};

let config = QuantizeConfig::new(OutputFormat::Gif);

// Build shared palette from representative frames
let frames: Vec<ImgRef<'_, rgb::RGBA<u8>>> = frame_data.iter()
    .map(|f| ImgRef::new(f, width, height))
    .collect();
let shared = zenquant::build_palette_rgba(&frames, &config).unwrap();

// Remap each frame
for frame_pixels in &frame_data {
    let result = shared.remap_rgba(frame_pixels, width, height, &config).unwrap();
    // result.palette() is the same across all frames
    // result.indices() is frame-specific
}
```

### Quality presets

```rust
use zenquant::Quality;

// Fast — ~30ms for 512x512. No AQ masking or k-means refinement.
let config = QuantizeConfig::new(OutputFormat::Png).quality(Quality::Fast);

// Balanced — ~60ms. AQ masking + 2 k-means iterations.
let config = QuantizeConfig::new(OutputFormat::Png).quality(Quality::Balanced);

// Best — ~120ms. AQ masking + 8 k-means iterations + Viterbi DP. (default)
let config = QuantizeConfig::new(OutputFormat::Png).quality(Quality::Best);
```

### Output formats

The `OutputFormat` controls palette sorting and dither tuning for each format's compression algorithm:

- **`Gif`** — LZW compression. Delta-minimize palette sort + post-remap frequency reorder. Binary transparency.
- **`Png`** — Deflate + scanline filters. Luminance sort for spatial locality. Full alpha via tRNS.
- **`WebpLossless`** — VP8L delta palette encoding. Delta-minimize sort.

## Benchmarks

Averaged over 209 images from the [CID22](https://zenodo.org/records/11186568) 512x512 training set. All quantizers configured for 256 colors with 50% dithering. Measured on an AMD Ryzen 9 7950X, single-threaded.

| Quantizer | Butteraugli | SSIMULACRA2 | Deflate size | Time |
|-----------|-------------|-------------|--------------|------|
| **zenquant** (Best) | **3.36** | 80.5 | **108 KB** | 118 ms |
| quantizr 1.4 | 3.23 | **81.3** | 125 KB | 29 ms |
| imagequant 4.4 | 4.73 | 75.9 | 106 KB | 39 ms |

Lower butteraugli = better. Higher SSIMULACRA2 = better. Smaller deflate = more compressible.

zenquant and quantizr produce similar perceptual quality (within noise on individual images). imagequant produces the most compressible index streams but at notably worse perceptual quality — its LZW-aware dithering sacrifices visual fidelity for smaller files.

zenquant's advantage is most visible on images with smooth gradients and subtle color transitions, where AQ masking prevents banding that other quantizers miss.

### Reproduce the benchmarks

```bash
# Requires CID22 corpus (or provide your own image directory)
cargo run --example compare --release -- /path/to/images
```

### Generate visual comparisons

```bash
cargo run --example compare_images --release -- /path/to/images /tmp/output 5
```

This generates side-by-side montages (requires ImageMagick `montage`).

## Integration

zenquant is used as the default quantizer in:

- [**zengif**](https://github.com/imazen/zengif) — GIF codec (`features = ["zenquant"]`)
- [**zenwebp**](https://github.com/imazen/zenwebp) — WebP codec (`features = ["quantize"]`)

## Features

- `std` (default) — enables `std::error::Error` impl
- `_dev` — exposes internal modules for profiling (not public API)

`no_std` + `alloc` compatible when `std` is disabled.

## Credits

- **Claude** (Anthropic) — AI-assisted development. Not all code manually reviewed.

## License

AGPL-3.0-or-later. Commercial licenses available at [imazen.io/pricing](https://www.imazen.io/pricing).
