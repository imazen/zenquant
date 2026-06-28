<!-- GENERATED FROM README.md by zenutils gen-readme-crates.sh — DO NOT EDIT. -->

# zenquant [![CI](https://img.shields.io/github/actions/workflow/status/imazen/zenquant/ci.yml?style=flat-square&label=CI)](https://github.com/imazen/zenquant/actions/workflows/ci.yml)

Color quantization with perceptual masking. Reduces truecolor images to 256-color indexed palettes in OKLab space, using butteraugli-inspired adaptive quantization (AQ) weights to concentrate palette entries where human vision is most sensitive. Pure Rust, `#![forbid(unsafe_code)]`, `no_std` + `alloc`, with runtime SIMD dispatch.

quantette's k-means mode leads on per-pixel metrics (highest SSIMULACRA2, lowest DSSIM). imagequant consistently looks best to the human eye, even when slightly behind on the numbers. zenquant focuses on file size — the advantage is less obvious when paired with [zenpng](https://github.com/imazen/zenpng)'s aggressive compression, but grows at faster encode speeds or with typical codecs like the `png` crate.

## Quick start

```toml
[dependencies]
zenquant = "0.1.3"
rgb = "0.8.53"     # input pixels are typed: &[rgb::RGB<u8>] / &[rgb::RGBA<u8>]
```

```rust
use zenquant::{QuantizeConfig, OutputFormat};
use rgb::FromSlice;            // brings `as_rgb` into scope (zero-copy reinterpret)

// `rgb_bytes` is width * height * 3 interleaved RGB8 bytes from your decoder.
let pixels: &[rgb::RGB<u8>] = rgb_bytes.as_rgb();

let config = QuantizeConfig::new(OutputFormat::Png);
let result = zenquant::quantize(pixels, width, height, &config).unwrap();

let palette = result.palette();   // &[[u8; 3]] — one sRGB entry per palette color
let indices = result.indices();   // &[u8] — one palette index per pixel, row-major
```

## What it does

Most quantizers treat every pixel equally. zenquant spends palette entries on smooth gradients, skin tones, and other regions where banding is visible — and wastes fewer entries on noisy textures where the eye can't tell the difference.

The pipeline: histogram in OKLab → median cut → k-means refinement with AQ weights → format-aware palette sorting → adaptive Floyd-Steinberg dithering → optional Viterbi DP for run-length optimization.

## Usage

### Input pixel types

`quantize` and `quantize_rgba` take **typed** pixel slices, not a raw `&[u8]`
byte buffer:

```rust
pub fn quantize(pixels: &[rgb::RGB<u8>], width: usize, height: usize, config: &QuantizeConfig)
    -> Result<QuantizeResult, QuantizeError>;
pub fn quantize_rgba(pixels: &[rgb::RGBA<u8>], width: usize, height: usize, config: &QuantizeConfig)
    -> Result<QuantizeResult, QuantizeError>;
```

The element types `rgb::RGB<u8>` / `rgb::RGBA<u8>` are re-exported as
`zenquant::RGB` / `zenquant::RGBA`. If you already hold a flat `Vec<u8>` (3 bytes
per RGB pixel, 4 per RGBA), reinterpret it in place with the `rgb` crate's
`FromSlice` adapter — no copy, no allocation:

```rust
use rgb::FromSlice; // brings `as_rgb` / `as_rgba` into scope

// RGB: width * height * 3 bytes
let pixels: &[rgb::RGB<u8>] = rgb_bytes.as_rgb();

// RGBA: width * height * 4 bytes
let pixels_rgba: &[rgb::RGBA<u8>] = rgba_bytes.as_rgba();
```

Add `rgb = "0.8.53"` (the same `rgb` crate zenquant depends on) as a direct
dependency so these element types — and the `FromSlice` adapter — are in scope.

### Quantize RGBA (GIF with transparency)

```rust
use zenquant::{QuantizeConfig, OutputFormat};
use rgb::FromSlice;

// `rgba_bytes` is a Vec<u8> of width * height * 4 bytes
let pixels: &[rgb::RGBA<u8>] = rgba_bytes.as_rgba();

let config = QuantizeConfig::new(OutputFormat::Gif);
let result = zenquant::quantize_rgba(pixels, width, height, &config).unwrap();

// Reading the result:
let palette: &[[u8; 3]] = result.palette();        // sRGB triples
let palette_rgba: &[[u8; 4]] = result.palette_rgba(); // same entries, with alpha
let indices: &[u8] = result.indices();             // one index per pixel, row-major

// Binary transparency: one palette entry reserved for transparent pixels
if let Some(idx) = result.transparent_index() {    // Option<u8>
    // pixels with alpha == 0 map to this index
}
```

`palette()` returns `&[[u8; 3]]` even on the RGBA path; use `palette_rgba()`
(`&[[u8; 4]]`) when you need the per-entry alpha, or `alpha_table()`
(`Option<Vec<u8>>`) for a PNG `tRNS` chunk.

### Write an indexed PNG

```rust
use zenquant::{QuantizeConfig, OutputFormat};
use rgb::FromSlice;

let pixels: &[rgb::RGB<u8>] = rgb_bytes.as_rgb();

let config = QuantizeConfig::new(OutputFormat::Png);
let result = zenquant::quantize(pixels, width, height, &config).unwrap();

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

// `frame_data` is a slice of per-frame RGBA pixel buffers: &[&[rgb::RGBA<u8>]]
// (use `rgb::FromSlice::as_rgba` to get each &[rgb::RGBA<u8>] from a Vec<u8>).
// Build a shared palette from representative frames:
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
        Err(QuantizeError::QualityNotMet { min_ssim2, achieved_ssim2 }) => {
            // this frame needs truecolor (wanted `min_ssim2`, got `achieved_ssim2`)
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
- **`PngJoint`** — like `Png`, plus a post-pass that jointly picks palette indices and PNG filter types per scanline to shrink the deflate stream while staying within each pixel's perceptual tolerance. Requires the `joint` feature.
- **`PngMinSize`** — minimum-file-size PNG: position-deterministic blue-noise dither at very low strength, aggressive run extension, and joint optimization. Requires the `joint` feature.

### Resource limits

`quantize`/`quantize_rgba` allocate several full-image scratch buffers sized from the input dimensions. To bound that, every config carries a pixel-count cap, checked before any allocation:

```rust
use zenquant::{QuantizeConfig, QuantizeError, OutputFormat};

// Default cap is 120 MP (admits ~108 MP photos). Tighten or disable it:
let config = QuantizeConfig::new(OutputFormat::Png)
    .with_max_pixels(Some(64 * 1024 * 1024)); // 64 MP
// .with_max_pixels(None)                     // disable the cap entirely

match zenquant::quantize(&pixels, width, height, &config) {
    Err(QuantizeError::TooManyPixels { pixels, max }) => {
        // width * height (`pixels`) exceeded `max`
    }
    _ => {}
}
```

### Cooperative cancellation

For long-running quantization (large images at `Quality::Best`), `quantize_with_stop` and `quantize_rgba_with_stop` take an [`enough::Stop`](https://github.com/imazen/enough) token and abort the k-means and Viterbi loops promptly when it fires, returning `QuantizeError::Cancelled`:

```rust
use zenquant::{QuantizeConfig, QuantizeError, OutputFormat};

let stop = /* any &dyn enough::Stop, e.g. a deadline or a shared flag */;
match zenquant::quantize_with_stop(&pixels, width, height, &config, stop) {
    Ok(result) => { /* finished before cancellation */ }
    Err(QuantizeError::Cancelled(reason)) => { /* stop fired */ }
    Err(e) => { /* other error */ }
}
```


## Integration

zenquant is used as the default quantizer in:

- [**zenpng**](https://github.com/imazen/zenpng) — PNG/APNG codec (`features = ["quantize"]`)
- [**zengif**](https://github.com/imazen/zengif) — GIF codec (`features = ["zenquant"]`)
- [**zenwebp**](https://github.com/imazen/zenwebp) — WebP codec (`features = ["quantize"]`)

## Features

- `std` (default) — enables `std` on archmage/magetypes for platform-optimized math
- `joint` — joint deflate+quantization optimization for PNG (`OutputFormat::PngJoint` / `PngMinSize`)
- `_dev` — exposes internal modules for profiling (not public API)

Always `no_std` + `alloc`. Uses `core::error::Error`. SIMD acceleration (AVX2+FMA, NEON, WASM SIMD128) via archmage with automatic scalar fallback. Fully functional without `std`.

The minimum supported Rust version (MSRV) is **1.92**.

## AI-Generated Code Notice

Developed with Claude (Anthropic). Not all code manually reviewed. Review critical paths before production use.

## License

Dual-licensed: [AGPL-3.0](https://github.com/imazen/zenquant/blob/main/LICENSE-AGPL3) or [commercial](https://github.com/imazen/zenquant/blob/main/LICENSE-COMMERCIAL).

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

See [LICENSE-COMMERCIAL](https://github.com/imazen/zenquant/blob/main/LICENSE-COMMERCIAL) for details.

## Image tech I maintain

| | |
|:--|:--|
| **Codecs** ¹ | [zenjpeg] · [zenpng] · [zenwebp] · [zengif] · [zenavif] · [zenjxl] · [zenbitmaps] · [heic] · [zentiff] · [zenpdf] · [zensvg] · [zenjp2] · [zenraw] · [ultrahdr] |
| Codec internals | [zenjxl-decoder] · [jxl-encoder] · [zenrav1e] · [rav1d-safe] · [zenavif-parse] · [zenavif-serialize] |
| Compression | [zenflate] · [zenzop] · [zenzstd] |
| Processing | [zenresize] · **zenquant** · [zenblend] · [zenfilters] · [zensally] · [zentone] |
| Pixels & color | [zenpixels] · [zenpixels-convert] · [linear-srgb] · [garb] |
| Pipeline & framework | [zenpipe] · [zencodec] · [zencodecs] · [zenlayout] · [zennode] · [zenwasm] · [zentract] |
| Metrics | [zensim] · [fast-ssim2] · [butteraugli] · [zenmetrics] · [resamplescope-rs] |
| Pickers & ML | [zenanalyze] · [zenpredict] · [zenpicker] |
| Products | [Imageflow] image engine ([.NET][imageflow-dotnet] · [Node][imageflow-node] · [Go][imageflow-go]) · [Imageflow Server] · [ImageResizer] (C#) |

<sub>¹ pure-Rust, `#![forbid(unsafe_code)]` codecs, as of 2026</sub>

### General Rust awesomeness

[zenbench] · [archmage] · [magetypes] · [enough] · [whereat] · [cargo-copter]

[Open source](https://www.imazen.io/open-source) · [@imazen](https://github.com/imazen) · [@lilith](https://github.com/lilith) · [lib.rs/~lilith](https://lib.rs/~lilith)

[zenjpeg]: https://github.com/imazen/zenjpeg
[zenpng]: https://github.com/imazen/zenpng
[zenwebp]: https://github.com/imazen/zenwebp
[zengif]: https://github.com/imazen/zengif
[zenavif]: https://github.com/imazen/zenavif
[zenjxl]: https://github.com/imazen/zenjxl
[zenbitmaps]: https://github.com/imazen/zenbitmaps
[heic]: https://github.com/imazen/heic
[zentiff]: https://github.com/imazen/zentiff
[zenpdf]: https://github.com/imazen/zenpdf
[zensvg]: https://github.com/imazen/zenextras
[zenjp2]: https://github.com/imazen/zenextras
[zenraw]: https://github.com/imazen/zenraw
[ultrahdr]: https://github.com/imazen/ultrahdr
[zenjxl-decoder]: https://github.com/imazen/zenjxl-decoder
[jxl-encoder]: https://github.com/imazen/jxl-encoder
[zenrav1e]: https://github.com/imazen/zenrav1e
[rav1d-safe]: https://github.com/imazen/rav1d-safe
[zenavif-parse]: https://github.com/imazen/zenavif-parse
[zenavif-serialize]: https://github.com/imazen/zenavif-serialize
[zenflate]: https://github.com/imazen/zenflate
[zenzop]: https://github.com/imazen/zenzop
[zenzstd]: https://github.com/imazen/zenzstd
[zenresize]: https://github.com/imazen/zenresize
[zenblend]: https://github.com/imazen/zenblend
[zenfilters]: https://github.com/imazen/zenfilters
[zensally]: https://github.com/imazen/zensally
[zentone]: https://github.com/imazen/zentone
[zenpixels]: https://github.com/imazen/zenpixels
[zenpixels-convert]: https://github.com/imazen/zenpixels
[linear-srgb]: https://github.com/imazen/linear-srgb
[garb]: https://github.com/imazen/garb
[zenpipe]: https://github.com/imazen/zenpipe
[zencodec]: https://github.com/imazen/zencodec
[zencodecs]: https://github.com/imazen/zencodecs
[zenlayout]: https://github.com/imazen/zenlayout
[zennode]: https://github.com/imazen/zennode
[zenwasm]: https://github.com/imazen/zenwasm
[zentract]: https://github.com/imazen/zentract
[zensim]: https://github.com/imazen/zensim
[fast-ssim2]: https://github.com/imazen/fast-ssim2
[butteraugli]: https://github.com/imazen/butteraugli
[zenmetrics]: https://github.com/imazen/zenmetrics
[resamplescope-rs]: https://github.com/imazen/resamplescope-rs
[zenanalyze]: https://github.com/imazen/zenanalyze
[zenpredict]: https://github.com/imazen/zenanalyze
[zenpicker]: https://github.com/imazen/zenanalyze
[zenbench]: https://github.com/imazen/zenbench
[archmage]: https://github.com/imazen/archmage
[magetypes]: https://github.com/imazen/archmage
[enough]: https://github.com/imazen/enough
[whereat]: https://github.com/lilith/whereat
[cargo-copter]: https://github.com/imazen/cargo-copter
[Imageflow]: https://github.com/imazen/imageflow
[Imageflow Server]: https://github.com/imazen/imageflow-dotnet-server
[ImageResizer]: https://github.com/imazen/resizer
[imageflow-dotnet]: https://github.com/imazen/imageflow-dotnet
[imageflow-node]: https://github.com/imazen/imageflow-node
[imageflow-go]: https://github.com/imazen/imageflow-go
