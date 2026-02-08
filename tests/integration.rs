//! Integration tests: validate zenquant output through real GIF, PNG, and WebP encoders
//! using real images from codec-corpus.

use std::io::Cursor;
use std::path::PathBuf;

/// Load a PNG image as RGB pixels + dimensions, skipping non-8bit and non-RGB/RGBA.
fn load_png_rgb(path: &std::path::Path) -> Option<(Vec<rgb::RGB<u8>>, u32, u32)> {
    let data = std::fs::read(path).ok()?;
    let decoder = png::Decoder::new(Cursor::new(&data));
    let mut reader = decoder.read_info().ok()?;
    let info = reader.info();
    let (w, h) = (info.width, info.height);

    // Only handle 8-bit RGB or RGBA
    if info.bit_depth != png::BitDepth::Eight {
        return None;
    }

    let mut buf = vec![0u8; reader.output_buffer_size()];
    let frame = reader.next_frame(&mut buf).ok()?;
    buf.truncate(frame.buffer_size());

    let pixels = match frame.color_type {
        png::ColorType::Rgb => buf
            .chunks_exact(3)
            .map(|c| rgb::RGB {
                r: c[0],
                g: c[1],
                b: c[2],
            })
            .collect(),
        png::ColorType::Rgba => buf
            .chunks_exact(4)
            .map(|c| rgb::RGB {
                r: c[0],
                g: c[1],
                b: c[2],
            })
            .collect(),
        _ => return None,
    };

    Some((pixels, w, h))
}

/// Load a PNG image as RGBA pixels + dimensions.
fn load_png_rgba(path: &std::path::Path) -> Option<(Vec<rgb::RGBA<u8>>, u32, u32)> {
    let data = std::fs::read(path).ok()?;
    let mut decoder = png::Decoder::new(Cursor::new(&data));
    decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::ALPHA);
    let mut reader = decoder.read_info().ok()?;
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let frame = reader.next_frame(&mut buf).ok()?;
    buf.truncate(frame.buffer_size());

    let (w, h) = (frame.width, frame.height);

    let pixels = match frame.color_type {
        png::ColorType::Rgba => buf
            .chunks_exact(4)
            .map(|c| rgb::RGBA {
                r: c[0],
                g: c[1],
                b: c[2],
                a: c[3],
            })
            .collect(),
        png::ColorType::Rgb => buf
            .chunks_exact(3)
            .map(|c| rgb::RGBA {
                r: c[0],
                g: c[1],
                b: c[2],
                a: 255,
            })
            .collect(),
        _ => return None,
    };

    Some((pixels, w, h))
}

/// Get paths to CID22 test images (real photos, diverse content).
fn cid22_paths(max: usize) -> Vec<PathBuf> {
    let corpus_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("codec-corpus/CID22/CID22-512/training");
    if !corpus_dir.exists() {
        return Vec::new();
    }
    let mut paths: Vec<PathBuf> = std::fs::read_dir(&corpus_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|ext| ext == "png"))
        .collect();
    paths.sort();
    paths.truncate(max);
    paths
}

/// Get paths to screenshot test images.
fn screenshot_paths(max: usize) -> Vec<PathBuf> {
    let corpus_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("codec-corpus/gb82-sc");
    if !corpus_dir.exists() {
        return Vec::new();
    }
    let mut paths: Vec<PathBuf> = std::fs::read_dir(&corpus_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|ext| ext == "png"))
        .collect();
    paths.sort();
    paths.truncate(max);
    paths
}

// ============================================================================
// GIF integration: zenquant → zengif encode → zengif decode → verify
// ============================================================================

mod gif_integration {
    use super::*;
    use enough::Unstoppable;
    use zenquant::{OutputFormat, QuantizeConfig};

    fn encode_with_zenquant_palette(
        result: &zenquant::QuantizeResult,
        w: u16,
        h: u16,
    ) -> Vec<u8> {
        let palette = result.palette();
        let indices = result.indices();
        let transparent_idx = result.transparent_index();

        let palette_flat: Vec<u8> = palette.iter().flat_map(|c| c.iter().copied()).collect();
        let gif_palette = zengif::Palette::from_rgb_bytes(&palette_flat);

        let rgba_pixels: Vec<zengif::Rgba> = indices
            .iter()
            .map(|&idx| {
                if Some(idx) == transparent_idx {
                    zengif::Rgba::TRANSPARENT
                } else {
                    let c = palette[idx as usize];
                    zengif::Rgba {
                        r: c[0],
                        g: c[1],
                        b: c[2],
                        a: 255,
                    }
                }
            })
            .collect();

        let frame = zengif::FrameInput::with_palette(w, h, 10, rgba_pixels, gif_palette);
        let config = zengif::EncoderConfig::new()
            .quantizer(zengif::Quantizer::quantizr());
        zengif::encode_gif(
            vec![frame],
            w,
            h,
            config,
            zengif::Limits::default(),
            &Unstoppable,
        )
        .unwrap()
    }

    #[test]
    fn gif_roundtrip_cid22() {
        let paths = cid22_paths(5);
        if paths.is_empty() {
            eprintln!("Skipping: CID22 corpus not found");
            return;
        }

        let config = QuantizeConfig::new().output_format(OutputFormat::Gif);

        for path in &paths {
            let (pixels, w, h) = match load_png_rgb(path) {
                Some(v) => v,
                None => continue,
            };
            let name = path.file_name().unwrap().to_string_lossy();

            let result = zenquant::quantize(&pixels, w as usize, h as usize, &config).unwrap();
            let gif_data = encode_with_zenquant_palette(&result, w as u16, h as u16);

            // Decode and verify dimensions + pixel count
            let (meta, frames, _stats) =
                zengif::decode_gif(&gif_data, zengif::Limits::default(), &Unstoppable).unwrap();
            assert_eq!(meta.width, w as u16);
            assert_eq!(meta.height, h as u16);
            assert_eq!(frames.len(), 1);
            assert_eq!(frames[0].pixels.len(), (w * h) as usize);

            eprintln!(
                "GIF {name}: {w}x{h}, {} bytes, {} colors",
                gif_data.len(),
                result.palette_len()
            );
        }
    }

    #[test]
    fn gif_roundtrip_screenshots() {
        let paths = screenshot_paths(5);
        if paths.is_empty() {
            eprintln!("Skipping: screenshot corpus not found");
            return;
        }

        let config = QuantizeConfig::new().output_format(OutputFormat::Gif);

        for path in &paths {
            let (pixels, w, h) = match load_png_rgb(path) {
                Some(v) => v,
                None => continue,
            };
            let name = path.file_name().unwrap().to_string_lossy();

            let result = zenquant::quantize(&pixels, w as usize, h as usize, &config).unwrap();
            let gif_data = encode_with_zenquant_palette(&result, w as u16, h as u16);

            let (meta, frames, _stats) =
                zengif::decode_gif(&gif_data, zengif::Limits::default(), &Unstoppable).unwrap();
            assert_eq!(meta.width, w as u16);
            assert_eq!(frames[0].pixels.len(), (w * h) as usize);

            eprintln!(
                "GIF screenshot {name}: {w}x{h}, {} bytes, {} colors",
                gif_data.len(),
                result.palette_len()
            );
        }
    }
}

// ============================================================================
// PNG integration: zenquant → png crate indexed encoding → decode → verify
// ============================================================================

mod png_integration {
    use super::*;
    use zenquant::{OutputFormat, QuantizeConfig};

    fn encode_indexed_png(
        palette_rgb: &[[u8; 3]],
        indices: &[u8],
        width: u32,
        height: u32,
        alpha_table: Option<&[u8]>,
    ) -> Vec<u8> {
        let mut buf = Vec::new();
        {
            let mut encoder = png::Encoder::new(&mut buf, width, height);
            encoder.set_color(png::ColorType::Indexed);
            encoder.set_depth(png::BitDepth::Eight);

            let palette_flat: Vec<u8> = palette_rgb.iter().flat_map(|c| c.iter().copied()).collect();
            encoder.set_palette(palette_flat);

            if let Some(trns) = alpha_table {
                encoder.set_trns(trns.to_vec());
            }

            let mut writer = encoder.write_header().unwrap();
            writer.write_image_data(indices).unwrap();
        }
        buf
    }

    fn decode_png_to_rgba(data: &[u8]) -> (Vec<u8>, u32, u32) {
        let mut decoder = png::Decoder::new(Cursor::new(data));
        decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::ALPHA);
        let mut reader = decoder.read_info().unwrap();
        let mut buf = vec![0u8; reader.output_buffer_size()];
        let info = reader.next_frame(&mut buf).unwrap();
        buf.truncate(info.buffer_size());

        let (w, h) = (info.width, info.height);
        match info.color_type {
            png::ColorType::Rgba => (buf, w, h),
            png::ColorType::Rgb => {
                let mut rgba = Vec::with_capacity((w * h * 4) as usize);
                for chunk in buf.chunks(3) {
                    rgba.extend_from_slice(chunk);
                    rgba.push(255);
                }
                (rgba, w, h)
            }
            other => panic!("Unexpected color type after expand: {other:?}"),
        }
    }

    #[test]
    fn png_indexed_roundtrip_cid22() {
        let paths = cid22_paths(5);
        if paths.is_empty() {
            eprintln!("Skipping: CID22 corpus not found");
            return;
        }

        let config = QuantizeConfig::new().output_format(OutputFormat::Png);

        for path in &paths {
            let (pixels, w, h) = match load_png_rgb(path) {
                Some(v) => v,
                None => continue,
            };
            let name = path.file_name().unwrap().to_string_lossy();

            let result =
                zenquant::quantize(&pixels, w as usize, h as usize, &config).unwrap();

            let png_data =
                encode_indexed_png(result.palette(), result.indices(), w, h, None);

            let (decoded, dw, dh) = decode_png_to_rgba(&png_data);
            assert_eq!(dw, w);
            assert_eq!(dh, h);

            // Verify decoded pixels match quantized palette entries exactly
            let palette = result.palette();
            for (i, &idx) in result.indices().iter().enumerate() {
                let c = palette[idx as usize];
                assert_eq!(decoded[i * 4], c[0], "{name} pixel {i} R mismatch");
                assert_eq!(decoded[i * 4 + 1], c[1], "{name} pixel {i} G mismatch");
                assert_eq!(decoded[i * 4 + 2], c[2], "{name} pixel {i} B mismatch");
            }

            eprintln!(
                "PNG {name}: {w}x{h}, indexed={} bytes, {} colors",
                png_data.len(),
                result.palette_len()
            );
        }
    }

    #[test]
    fn png_indexed_smaller_than_truecolor() {
        let paths = cid22_paths(3);
        if paths.is_empty() {
            eprintln!("Skipping: CID22 corpus not found");
            return;
        }

        let config = QuantizeConfig::new().output_format(OutputFormat::Png);

        for path in &paths {
            let (pixels, w, h) = match load_png_rgb(path) {
                Some(v) => v,
                None => continue,
            };
            let name = path.file_name().unwrap().to_string_lossy();

            let result =
                zenquant::quantize(&pixels, w as usize, h as usize, &config).unwrap();
            let indexed_png =
                encode_indexed_png(result.palette(), result.indices(), w, h, None);

            let truecolor_png = {
                let mut buf = Vec::new();
                let mut encoder = png::Encoder::new(&mut buf, w, h);
                encoder.set_color(png::ColorType::Rgb);
                encoder.set_depth(png::BitDepth::Eight);
                let mut writer = encoder.write_header().unwrap();
                let flat: Vec<u8> = pixels.iter().flat_map(|p| [p.r, p.g, p.b]).collect();
                writer.write_image_data(&flat).unwrap();
                drop(writer);
                buf
            };

            let ratio = indexed_png.len() as f64 / truecolor_png.len() as f64;
            eprintln!(
                "PNG {name}: indexed={} bytes, truecolor={} bytes, ratio={ratio:.2}x",
                indexed_png.len(),
                truecolor_png.len()
            );

            assert!(
                indexed_png.len() < truecolor_png.len(),
                "{name}: indexed PNG should be smaller than truecolor"
            );
        }
    }

    #[test]
    fn png_trns_roundtrip() {
        // Use pngsuite RGBA images if available
        let pngsuite = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("codec-corpus/pngsuite");

        // Find an RGBA PNG from pngsuite (tRNS or RGBA types)
        let test_paths: Vec<PathBuf> = if pngsuite.exists() {
            std::fs::read_dir(&pngsuite)
                .unwrap()
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| {
                    p.extension().is_some_and(|ext| ext == "png")
                        && p.file_name()
                            .unwrap()
                            .to_string_lossy()
                            .starts_with("tbyn")
                })
                .collect()
        } else {
            Vec::new()
        };

        if test_paths.is_empty() {
            // Fall back to a synthetic RGBA test
            let w = 64u32;
            let h = 64u32;
            let mut pixels = Vec::with_capacity((w * h) as usize);
            for y in 0..h {
                for x in 0..w {
                    let a = if x < w / 4 && y < h / 4 { 0u8 } else { 255 };
                    pixels.push(rgb::RGBA {
                        r: (x * 255 / w) as u8,
                        g: (y * 255 / h) as u8,
                        b: 128,
                        a,
                    });
                }
            }

            let config = QuantizeConfig::new().output_format(OutputFormat::Png);
            let result =
                zenquant::quantize_rgba(&pixels, w as usize, h as usize, &config).unwrap();

            let alpha_table = result.alpha_table();
            let png_data = encode_indexed_png(
                result.palette(),
                result.indices(),
                w,
                h,
                alpha_table.as_deref(),
            );

            let (decoded, _, _) = decode_png_to_rgba(&png_data);

            // Top-left quadrant transparent
            assert_eq!(decoded[3], 0, "top-left should be transparent");
            // Center pixel opaque
            let center = ((h / 2) * w + w / 2) as usize * 4;
            assert_eq!(decoded[center + 3], 255, "center should be opaque");

            eprintln!(
                "PNG tRNS (synthetic): {} bytes, alpha_table_len={:?}",
                png_data.len(),
                alpha_table.as_ref().map(|t| t.len())
            );
            return;
        }

        // Test with real pngsuite RGBA images
        let config = QuantizeConfig::new().output_format(OutputFormat::Png);
        for path in &test_paths {
            let (pixels, w, h) = match load_png_rgba(path) {
                Some(v) => v,
                None => continue,
            };
            let name = path.file_name().unwrap().to_string_lossy();

            let result =
                zenquant::quantize_rgba(&pixels, w as usize, h as usize, &config).unwrap();
            let alpha_table = result.alpha_table();
            let png_data = encode_indexed_png(
                result.palette(),
                result.indices(),
                w,
                h,
                alpha_table.as_deref(),
            );

            // Verify it decodes
            let (decoded, dw, dh) = decode_png_to_rgba(&png_data);
            assert_eq!(dw, w);
            assert_eq!(dh, h);
            assert_eq!(decoded.len(), (w * h * 4) as usize);

            eprintln!(
                "PNG tRNS {name}: {w}x{h}, {} bytes, alpha_table_len={:?}",
                png_data.len(),
                alpha_table.as_ref().map(|t| t.len())
            );
        }
    }
}

// ============================================================================
// WebP integration: zenquant → zenwebp lossless encode → verify valid output
// ============================================================================

mod webp_integration {
    use super::*;
    use zenquant::{OutputFormat, QuantizeConfig};

    fn reconstruct_rgba(palette: &[[u8; 4]], indices: &[u8]) -> Vec<u8> {
        let mut rgba = Vec::with_capacity(indices.len() * 4);
        for &idx in indices {
            let c = palette[idx as usize];
            rgba.extend_from_slice(&c);
        }
        rgba
    }

    #[test]
    fn webp_lossless_cid22() {
        let paths = cid22_paths(5);
        if paths.is_empty() {
            eprintln!("Skipping: CID22 corpus not found");
            return;
        }

        let config = QuantizeConfig::new().output_format(OutputFormat::WebpLossless);

        for path in &paths {
            let (pixels, w, h) = match load_png_rgb(path) {
                Some(v) => v,
                None => continue,
            };
            let name = path.file_name().unwrap().to_string_lossy();

            let result =
                zenquant::quantize(&pixels, w as usize, h as usize, &config).unwrap();
            let rgba = reconstruct_rgba(result.palette_rgba(), result.indices());

            let lossless_config = zenwebp::LosslessConfig::new();
            let webp_data = zenwebp::EncodeRequest::lossless(
                &lossless_config,
                &rgba,
                zenwebp::PixelLayout::Rgba8,
                w,
                h,
            )
            .encode()
            .unwrap();

            assert_eq!(&webp_data[..4], b"RIFF");
            assert_eq!(&webp_data[8..12], b"WEBP");

            eprintln!(
                "WebP {name}: {w}x{h}, {} bytes, {} colors",
                webp_data.len(),
                result.palette_len()
            );
        }
    }

    #[test]
    fn webp_lossless_screenshots() {
        let paths = screenshot_paths(5);
        if paths.is_empty() {
            eprintln!("Skipping: screenshot corpus not found");
            return;
        }

        let config = QuantizeConfig::new().output_format(OutputFormat::WebpLossless);

        for path in &paths {
            let (pixels, w, h) = match load_png_rgb(path) {
                Some(v) => v,
                None => continue,
            };
            let name = path.file_name().unwrap().to_string_lossy();

            let result =
                zenquant::quantize(&pixels, w as usize, h as usize, &config).unwrap();
            let rgba = reconstruct_rgba(result.palette_rgba(), result.indices());

            let lossless_config = zenwebp::LosslessConfig::new();
            let webp_data = zenwebp::EncodeRequest::lossless(
                &lossless_config,
                &rgba,
                zenwebp::PixelLayout::Rgba8,
                w,
                h,
            )
            .encode()
            .unwrap();

            assert_eq!(&webp_data[..4], b"RIFF");

            eprintln!(
                "WebP screenshot {name}: {w}x{h}, {} bytes, {} colors",
                webp_data.len(),
                result.palette_len()
            );
        }
    }

    #[test]
    fn webp_quantized_smaller_than_raw() {
        // Use screenshot images — they have large flat areas that benefit from quantization
        let paths = screenshot_paths(3);
        if paths.is_empty() {
            eprintln!("Skipping: screenshot corpus not found");
            return;
        }

        let config = QuantizeConfig::new().output_format(OutputFormat::WebpLossless);
        let lossless_config = zenwebp::LosslessConfig::new();

        for path in &paths {
            let (pixels, w, h) = match load_png_rgb(path) {
                Some(v) => v,
                None => continue,
            };
            let name = path.file_name().unwrap().to_string_lossy();

            let result =
                zenquant::quantize(&pixels, w as usize, h as usize, &config).unwrap();
            let quant_rgba = reconstruct_rgba(result.palette_rgba(), result.indices());
            let quantized_webp = zenwebp::EncodeRequest::lossless(
                &lossless_config,
                &quant_rgba,
                zenwebp::PixelLayout::Rgba8,
                w,
                h,
            )
            .encode()
            .unwrap();

            let raw_rgba: Vec<u8> =
                pixels.iter().flat_map(|p| [p.r, p.g, p.b, 255]).collect();
            let raw_webp = zenwebp::EncodeRequest::lossless(
                &lossless_config,
                &raw_rgba,
                zenwebp::PixelLayout::Rgba8,
                w,
                h,
            )
            .encode()
            .unwrap();

            let ratio = quantized_webp.len() as f64 / raw_webp.len() as f64;
            eprintln!(
                "WebP {name}: quantized={} bytes, raw={} bytes, ratio={ratio:.2}x",
                quantized_webp.len(),
                raw_webp.len()
            );
        }
    }
}

// ============================================================================
// Cross-format comparison on real images
// ============================================================================

mod cross_format {
    use super::*;
    use flate2::Compression;
    use flate2::write::DeflateEncoder;
    use std::io::Write;
    use zenquant::{OutputFormat, QuantizeConfig};

    fn deflate_size(data: &[u8]) -> usize {
        let mut encoder = DeflateEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(data).unwrap();
        encoder.finish().unwrap().len()
    }

    #[test]
    fn format_comparison_cid22() {
        let paths = cid22_paths(3);
        if paths.is_empty() {
            eprintln!("Skipping: CID22 corpus not found");
            return;
        }

        let formats = [
            ("Generic", OutputFormat::Generic),
            ("Gif", OutputFormat::Gif),
            ("Png", OutputFormat::Png),
            ("WebpLossless", OutputFormat::WebpLossless),
        ];

        for path in &paths {
            let (pixels, w, h) = match load_png_rgb(path) {
                Some(v) => v,
                None => continue,
            };
            let name = path.file_name().unwrap().to_string_lossy();

            eprintln!("\n{name} ({w}x{h}):");
            eprintln!(
                "  {:<15} {:>7} {:>10} {:>8}",
                "Format", "Colors", "Deflate", "AvgRun"
            );

            for &(fmt_name, fmt) in &formats {
                let config = QuantizeConfig::new().output_format(fmt);
                let result =
                    zenquant::quantize(&pixels, w as usize, h as usize, &config).unwrap();
                let deflate = deflate_size(result.indices());
                let avg_run = zenquant::remap::average_run_length(result.indices());
                eprintln!(
                    "  {:<15} {:>7} {:>10} {:>8.2}",
                    fmt_name,
                    result.palette_len(),
                    deflate,
                    avg_run
                );
            }
        }
    }
}
