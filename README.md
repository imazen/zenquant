# zenquant [![CI](https://img.shields.io/github/actions/workflow/status/imazen/zenquant/ci.yml?style=flat-square&label=CI)](https://github.com/imazen/zenquant/actions/workflows/ci.yml) [![crates.io](https://img.shields.io/crates/v/zenquant?style=flat-square)](https://crates.io/crates/zenquant) [![docs.rs](https://img.shields.io/docsrs/zenquant?style=flat-square)](https://docs.rs/zenquant) [![MSRV](https://img.shields.io/badge/MSRV-1.92-blue?style=flat-square)](https://doc.rust-lang.org/cargo/reference/manifest.html#the-rust-version-field) [![license](https://img.shields.io/badge/license-AGPL--3.0-blue?style=flat-square)](https://github.com/imazen/zenquant#license)

Color quantization with perceptual masking. Reduces truecolor images to 256-color indexed palettes in OKLab space, using butteraugli-inspired adaptive quantization (AQ) weights to concentrate palette entries where human vision is most sensitive.

quantette's k-means mode leads on per-pixel metrics (highest SSIMULACRA2, lowest DSSIM). imagequant consistently looks best to the human eye, even when slightly behind on the numbers. zenquant focuses on file size — the advantage is less obvious when paired with [zenpng](https://github.com/imazen/zenpng)'s aggressive compression, but grows at faster encode speeds or with typical codecs like the `png` crate.

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
use zenquant::{QuantizeConfig, QuantizeError, OutputFormat, ImgRef};

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

For animation encoders (APNG, GIF), you can enforce per-frame quality with `with_min_ssim2` on the remap config. Frames that fail the quality floor return `QualityNotMet`, letting the encoder decide whether to fall back to truecolor for that frame:

```rust
let remap_config = QuantizeConfig::new(OutputFormat::Png)
    .with_min_ssim2(75.0);

for frame_pixels in &frame_data {
    match shared.remap_rgba(frame_pixels, width, height, &remap_config) {
        Ok(result) => {
            let ssim2 = result.ssimulacra2_estimate().unwrap();
            // encode as indexed
        }
        Err(QuantizeError::QualityNotMet { achieved_ssim2, .. }) => {
            // this frame needs truecolor
        }
        Err(e) => panic!("{e}"),
    }
}
```

### Quality targets

Specify quality in SSIMULACRA2 units instead of manually tuning compression knobs. zenquant auto-selects the internal quality preset, dither strength, and run priority to maximize compression while staying above your target.

```rust
use zenquant::{QuantizeConfig, OutputFormat};

// Auto-tune compression: stay above SSIM2 80, compress as hard as possible
let config = QuantizeConfig::new(OutputFormat::Png)
    .with_max_colors(256)
    .with_target_ssim2(80.0);

let result = zenquant::quantize(&pixels, width, height, &config).unwrap();

// Quality metrics are computed automatically when a target is set
let ssim2 = result.ssimulacra2_estimate().unwrap();  // 0–100, higher = better
let ba = result.butteraugli_estimate().unwrap();       // 0+, lower = better
```

Set a hard quality floor with `with_min_ssim2`. Returns `QuantizeError::QualityNotMet` if the result falls below — useful for animation encoders that need to decide per-frame whether to fall back to truecolor:

```rust
use zenquant::{QuantizeConfig, QuantizeError, OutputFormat};

let config = QuantizeConfig::new(OutputFormat::Png)
    .with_max_colors(256)
    .with_min_ssim2(75.0);

match zenquant::quantize(&pixels, width, height, &config) {
    Ok(result) => { /* quality met, use indexed */ }
    Err(QuantizeError::QualityNotMet { min_ssim2, achieved_ssim2 }) => {
        // Fall back to truecolor for this frame
    }
    Err(e) => { /* other error */ }
}
```

Quality metrics and `with_min_ssim2` enforcement also work on the `remap()` path, so you get per-frame quality measurement when using shared palettes for animation.

### Quality presets

```rust
use zenquant::Quality;

// Fast — ~30ms for 512x512. No AQ masking or k-means refinement.
let config = QuantizeConfig::new(OutputFormat::Png).with_quality(Quality::Fast);

// Balanced — ~60ms. AQ masking + 2 k-means iterations.
let config = QuantizeConfig::new(OutputFormat::Png).with_quality(Quality::Balanced);

// Best — ~120ms. AQ masking + 8 k-means iterations + Viterbi DP. (default)
let config = QuantizeConfig::new(OutputFormat::Png).with_quality(Quality::Best);
```

When `target_ssim2` is set, it overrides the quality preset, run priority, and dither strength with auto-tuned values based on calibrated compression tier data.

### Output formats

The `OutputFormat` controls palette sorting and dither tuning for each format's compression algorithm:

- **`Gif`** — LZW compression. Delta-minimize palette sort + post-remap frequency reorder. Binary transparency.
- **`Png`** — Deflate + scanline filters. Luminance sort for spatial locality. Full alpha via tRNS.
- **`WebpLossless`** — VP8L delta palette encoding. Delta-minimize sort.

## Benchmarks

Averaged over 50 images from three corpora ([CID22](https://zenodo.org/records/11186568), [CLIC 2025](https://storage.googleapis.com/clic2025), screenshots). All quantizers configured for 256 colors with default dithering. PNG sizes use aggressive deflate via [zenpng](https://github.com/imazen/zenpng). Sorted by DSSIM.

| Quantizer | Butteraugli | SSIMULACRA2 | DSSIM | PNG size | GIF size | ~ms |
|-----------|-------------|-------------|-------|----------|----------|-----|
| quantette (k-means) | 3.86 | **83.9** | **0.00050** | 616 KB | 799 KB | 265 |
| imagequant s1 d100 | 4.10 | 82.2 | 0.00056 | 637 KB | 848 KB | 546 |
| imagequant s4 d100 | 4.39 | 81.9 | 0.00057 | 640 KB | 854 KB | 315 |
| **zenquant** (Best) | **3.17** | 82.9 | 0.00058 | 586 KB | 764 KB | 542 |
| imagequant s1 d50 | 4.15 | 82.0 | 0.00060 | 627 KB | 836 KB | 465 |
| **zenquant** (Balanced) | 3.21 | 82.9 | 0.00064 | **579 KB** | 751 KB | 453 |
| **zenquant** (Fast) | 3.29 | 82.6 | 0.00069 | 582 KB | **749 KB** | 321 |
| quantizr | 4.44 | 79.7 | 0.00098 | 584 KB | 764 KB | 544 |
| color_quant | 8.96 | 72.1 | 0.00141 | 625 KB | 841 KB | 180 |

Lower butteraugli/DSSIM = better. Higher SSIMULACRA2 = better. Smaller file size = better.

**[Interactive visual comparison (9 configurations of 5 quantizers, 50 images)](https://imageflow-resources.s3.us-west-2.amazonaws.com/demos/zenquant/2026-03-04/index.html)** — slider, diff, and zoom views with per-image metrics. Keyboard shortcuts: 1 = original, 2–0 = variants.

zenquant's advantage is most visible on images with smooth gradients and subtle color transitions, where AQ masking prevents banding that other quantizers miss.

### Reproduce the benchmarks

```bash
cargo run --example quantizer_comparison --release -- gb82-sc,cid22,clic2025 /tmp/output 20
```

The comparison tool generates an interactive HTML report with cached results. Add `--benchmark` for rigorous sequential timing (min-of-5 runs).

## Integration

zenquant is used as the default quantizer in:

- [**zenpng**](https://github.com/imazen/zenpng) — PNG/APNG codec (`features = ["quantize"]`)
- [**zengif**](https://github.com/imazen/zengif) — GIF codec (`features = ["zenquant"]`)
- [**zenwebp**](https://github.com/imazen/zenwebp) — WebP codec (`features = ["quantize"]`)

## Features

- `std` (default) — enables `std` on archmage/magetypes for platform-optimized math
- `zennode` — zennode graph integration (`dep:zennode`)
- `joint` — joint deflate+quantization optimization for PNG
- `_dev` — exposes internal modules for profiling (not public API)

Always `no_std` + `alloc`. Uses `core::error::Error` (Rust 1.81+). SIMD acceleration (AVX2+FMA, NEON) via archmage with automatic scalar fallback. Fully functional without `std`.

## MSRV

The minimum supported Rust version is **1.92**.

## AI-Generated Code Notice

Developed with Claude (Anthropic). Not all code manually reviewed. Review critical paths before production use.

## Image tech I maintain

| | |
|:--|:--|
| State of the art codecs<sup>[1]</sup> | [zenjpeg] · [zenpng] · [zenwebp] · [zengif] · [zenavif] ([rav1d-safe] · [zenrav1e] · [zenavif-parse] · [zenavif-serialize]) · [zenjxl] ([jxl-encoder] · [zenjxl-decoder]) · [zentiff] · [zenbitmaps] · [heic] · [zenraw] · [zenpdf] · [ultrahdr] · [mozjpeg-rs] · [webpx] |
| Compression | [zenflate] · [zenzop] |
| Processing | [zenresize] · [zenfilters] · **zenquant** · [zenblend] |
| Metrics | [zensim] · [fast-ssim2] · [butteraugli] · [resamplescope-rs] · [codec-eval] · [codec-corpus] |
| Pixel types & color | [zenpixels] · [zenpixels-convert] · [linear-srgb] · [garb] |
| Pipeline | [zenpipe] · [zencodec] · [zencodecs] · [zenlayout] · [zennode] |
| ImageResizer | [ImageResizer] (C#) — 24M+ NuGet downloads across all packages |
| [Imageflow][] | Image optimization engine (Rust) — [.NET][imageflow-dotnet] · [node][imageflow-node] · [go][imageflow-go] — 9M+ NuGet downloads across all packages |
| [Imageflow Server][] | [The fast, safe image server](https://www.imazen.io/) (Rust+C#) — 552K+ NuGet downloads, deployed by Fortune 500s and major brands |

<sup>[1]</sup> <sub>as of 2026</sub>

### General Rust awesomeness

[archmage] · [magetypes] · [enough] · [whereat] · [zenbench] · [cargo-copter]

[And other projects](https://www.imazen.io/open-source) · [GitHub @imazen](https://github.com/imazen) · [GitHub @lilith](https://github.com/lilith) · [lib.rs/~lilith](https://lib.rs/~lilith) · [NuGet](https://www.nuget.org/profiles/imazen) (over 30 million downloads / 87 packages)

## License

Dual-licensed: [AGPL-3.0](LICENSE-AGPL3) or [commercial](LICENSE-COMMERCIAL).

I've maintained and developed open-source image server software — and the 40+
library ecosystem it depends on — full-time since 2011. Fifteen years of
continual maintenance, backwards compatibility, support, and the (very rare)
security patch. That kind of stability requires sustainable funding, and
dual-licensing is how we make it work without venture capital or rug-pulls.
Support sustainable and secure software; swap patch tuesday for patch leap-year.

[Our open-source products](https://www.imazen.io/open-source)

**Your options:**

- **Startup license** — $1 if your company has under $1M revenue and fewer
  than 5 employees. [Get a key →](https://www.imazen.io/pricing)
- **Commercial subscription** — Governed by the Imazen Site-wide Subscription
  License v1.1 or later. Apache 2.0-like terms, no source-sharing requirement.
  Sliding scale by company size.
  [Pricing & 60-day free trial →](https://www.imazen.io/pricing)
- **AGPL v3** — Free and open. Share your source if you distribute.

See [LICENSE-COMMERCIAL](LICENSE-COMMERCIAL) for details.

[zenjpeg]: https://github.com/imazen/zenjpeg
[zenpng]: https://github.com/imazen/zenpng
[zenwebp]: https://github.com/imazen/zenwebp
[zengif]: https://github.com/imazen/zengif
[zenavif]: https://github.com/imazen/zenavif
[zenjxl]: https://github.com/imazen/zenjxl
[zentiff]: https://github.com/imazen/zentiff
[zenbitmaps]: https://github.com/imazen/zenbitmaps
[heic]: https://github.com/imazen/heic-decoder-rs
[zenraw]: https://github.com/imazen/zenraw
[zenpdf]: https://github.com/imazen/zenpdf
[ultrahdr]: https://github.com/imazen/ultrahdr
[jxl-encoder]: https://github.com/imazen/jxl-encoder
[zenjxl-decoder]: https://github.com/imazen/zenjxl-decoder
[rav1d-safe]: https://github.com/imazen/rav1d-safe
[zenrav1e]: https://github.com/imazen/zenrav1e
[mozjpeg-rs]: https://github.com/imazen/mozjpeg-rs
[zenavif-parse]: https://github.com/imazen/zenavif-parse
[zenavif-serialize]: https://github.com/imazen/zenavif-serialize
[webpx]: https://github.com/imazen/webpx
[zenflate]: https://github.com/imazen/zenflate
[zenzop]: https://github.com/imazen/zenzop
[zenresize]: https://github.com/imazen/zenresize
[zenfilters]: https://github.com/imazen/zenfilters
[zenblend]: https://github.com/imazen/zenblend
[zensim]: https://github.com/imazen/zensim
[fast-ssim2]: https://github.com/imazen/fast-ssim2
[butteraugli]: https://github.com/imazen/butteraugli
[zenpixels]: https://github.com/imazen/zenpixels
[zenpixels-convert]: https://github.com/imazen/zenpixels
[linear-srgb]: https://github.com/imazen/linear-srgb
[garb]: https://github.com/imazen/garb
[zenpipe]: https://github.com/imazen/zenpipe
[zencodec]: https://github.com/imazen/zencodec
[zencodecs]: https://github.com/imazen/zencodecs
[zenlayout]: https://github.com/imazen/zenlayout
[zennode]: https://github.com/imazen/zennode
[Imageflow]: https://github.com/imazen/imageflow
[Imageflow Server]: https://github.com/imazen/imageflow-server
[imageflow-dotnet]: https://github.com/imazen/imageflow-dotnet
[imageflow-node]: https://github.com/imazen/imageflow-node
[imageflow-go]: https://github.com/imazen/imageflow-go
[ImageResizer]: https://github.com/imazen/resizer
[archmage]: https://github.com/imazen/archmage
[magetypes]: https://github.com/imazen/archmage
[enough]: https://github.com/imazen/enough
[whereat]: https://github.com/lilith/whereat
[zenbench]: https://github.com/imazen/zenbench
[cargo-copter]: https://github.com/imazen/cargo-copter
[resamplescope-rs]: https://github.com/imazen/resamplescope-rs
[codec-eval]: https://github.com/imazen/codec-eval
[codec-corpus]: https://github.com/imazen/codec-corpus
