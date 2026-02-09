//! Visual comparison of palette quantizers with interactive HTML report.
//!
//! Generates side-by-side comparisons of zenquant, imagequant, quantizr,
//! color_quant, and exoquant with slider, diff, and zoom views.
//!
//! Usage:
//!   cargo run --example quantizer_comparison --release -- <corpus> <output_dir> [max_images]
//!
//! Corpus names:
//!   cid22    → CID22/CID22-512/training
//!   clic2025 → clic2025/final-test
//!   gb82-sc  → gb82-sc

use butteraugli::ButteraugliParams;
use fast_ssim2::compute_ssimulacra2;
use imgref::ImgVec;
use rgb::RGB8;
use std::path::{Path, PathBuf};
use std::time::Instant;
use zenquant::{OutputFormat, QuantizeConfig};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

const QUANTIZER_NAMES: &[&str] = &[
    "zenquant",
    "imagequant",
    "quantizr",
    "color_quant",
    "exoquant",
];

#[derive(Clone)]
struct ImageResult {
    name: String,
    width: usize,
    height: usize,
    quantizers: Vec<QuantizerResult>,
}

#[derive(Clone)]
struct QuantizerResult {
    name: String,
    butteraugli: f64,
    ssimulacra2: f64,
    dssim: f64,
    png_bytes: usize,
    gif_bytes: usize,
    webp_bytes: usize,
    time_ms: f64,
    note: &'static str,
}

struct ReportData {
    corpus: String,
    images: Vec<ImageResult>,
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: quantizer_comparison <corpus> <output_dir> [max_images]");
        eprintln!("  corpus: cid22, clic2025, gb82-sc");
        std::process::exit(1);
    }

    let corpus_name = &args[1];
    let output_dir = PathBuf::from(&args[2]);
    let max_images: usize = args
        .get(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(usize::MAX);

    let corpus_path = match corpus_name.as_str() {
        "cid22" => "CID22/CID22-512/training",
        "clic2025" => "clic2025/final-test",
        "gb82-sc" => "gb82-sc",
        other => {
            eprintln!("Unknown corpus '{other}'. Use: cid22, clic2025, gb82-sc");
            std::process::exit(1);
        }
    };

    // Download corpus via codec-corpus
    eprintln!("Fetching corpus '{corpus_name}' ({corpus_path})...");
    let corpus = codec_corpus::Corpus::new().expect("failed to init codec-corpus");
    let image_dir = corpus.get(corpus_path).expect("failed to download corpus");

    let mut paths: Vec<PathBuf> = std::fs::read_dir(&image_dir)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", image_dir.display()))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .is_some_and(|ext| ext == "png" || ext == "jpg" || ext == "jpeg")
        })
        .collect();

    paths.sort();
    paths.truncate(max_images);

    if paths.is_empty() {
        eprintln!("No images found in {}", image_dir.display());
        std::process::exit(1);
    }

    std::fs::create_dir_all(&output_dir).expect("cannot create output directory");

    eprintln!(
        "Comparing {} quantizers on {} images → {}",
        QUANTIZER_NAMES.len(),
        paths.len(),
        output_dir.display()
    );

    let mut report = ReportData {
        corpus: corpus_name.clone(),
        images: Vec::new(),
    };

    for (i, path) in paths.iter().enumerate() {
        let stem = path.file_stem().unwrap_or_default().to_string_lossy();
        eprintln!("[{}/{}] {stem}...", i + 1, paths.len());

        let img = match image::open(path) {
            Ok(img) => img.to_rgb8(),
            Err(e) => {
                eprintln!("  Skipping: {e}");
                continue;
            }
        };

        let width = img.width() as usize;
        let height = img.height() as usize;
        let pixels: Vec<rgb::RGB<u8>> = img
            .pixels()
            .map(|p| rgb::RGB {
                r: p.0[0],
                g: p.0[1],
                b: p.0[2],
            })
            .collect();
        let ref_rgb: Vec<RGB8> = pixels.iter().map(|p| RGB8::new(p.r, p.g, p.b)).collect();

        // Create image subdirectory
        let img_dir = output_dir.join(&*stem);
        std::fs::create_dir_all(&img_dir).unwrap();

        // Save original as truecolor PNG
        save_truecolor_png(&img_dir.join("original.png"), &pixels, width, height);

        let mut image_result = ImageResult {
            name: stem.to_string(),
            width,
            height,
            quantizers: Vec::new(),
        };

        // --- zenquant ---
        let t0 = Instant::now();
        let zq = zenquant::quantize(
            &pixels,
            width,
            height,
            &QuantizeConfig::new(OutputFormat::Png).max_colors(256),
        )
        .unwrap();
        let zq_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let zq_pal: Vec<[u8; 3]> = zq.palette().to_vec();
        let zq_idx: Vec<u8> = zq.indices().to_vec();
        save_indexed_png(
            &img_dir.join("zenquant.png"),
            &zq_pal,
            &zq_idx,
            width,
            height,
        );
        let zq_metrics = compute_metrics(&ref_rgb, &zq_pal, &zq_idx, width, height);
        let zq_sizes = measure_format_sizes(&zq_pal, &zq_idx, width, height);
        image_result.quantizers.push(QuantizerResult {
            name: "zenquant".into(),
            butteraugli: zq_metrics.0,
            ssimulacra2: zq_metrics.1,
            dssim: zq_metrics.2,
            png_bytes: zq_sizes.0,
            gif_bytes: zq_sizes.1,
            webp_bytes: zq_sizes.2,
            time_ms: zq_ms,
            note: "",
        });
        eprint!("  zq:{:.1}ms", zq_ms);

        // --- imagequant ---
        let t0 = Instant::now();
        let (iq_pal, iq_idx) = run_imagequant(&pixels, width, height);
        let iq_ms = t0.elapsed().as_secs_f64() * 1000.0;
        save_indexed_png(
            &img_dir.join("imagequant.png"),
            &iq_pal,
            &iq_idx,
            width,
            height,
        );
        let iq_metrics = compute_metrics(&ref_rgb, &iq_pal, &iq_idx, width, height);
        let iq_sizes = measure_format_sizes(&iq_pal, &iq_idx, width, height);
        image_result.quantizers.push(QuantizerResult {
            name: "imagequant".into(),
            butteraugli: iq_metrics.0,
            ssimulacra2: iq_metrics.1,
            dssim: iq_metrics.2,
            png_bytes: iq_sizes.0,
            gif_bytes: iq_sizes.1,
            webp_bytes: iq_sizes.2,
            time_ms: iq_ms,
            note: "",
        });
        eprint!("  iq:{:.1}ms", iq_ms);

        // --- quantizr ---
        let t0 = Instant::now();
        let (qr_pal, qr_idx) = run_quantizr(&pixels, width, height);
        let qr_ms = t0.elapsed().as_secs_f64() * 1000.0;
        save_indexed_png(
            &img_dir.join("quantizr.png"),
            &qr_pal,
            &qr_idx,
            width,
            height,
        );
        let qr_metrics = compute_metrics(&ref_rgb, &qr_pal, &qr_idx, width, height);
        let qr_sizes = measure_format_sizes(&qr_pal, &qr_idx, width, height);
        image_result.quantizers.push(QuantizerResult {
            name: "quantizr".into(),
            butteraugli: qr_metrics.0,
            ssimulacra2: qr_metrics.1,
            dssim: qr_metrics.2,
            png_bytes: qr_sizes.0,
            gif_bytes: qr_sizes.1,
            webp_bytes: qr_sizes.2,
            time_ms: qr_ms,
            note: "",
        });
        eprint!("  qr:{:.1}ms", qr_ms);

        // --- color_quant ---
        let t0 = Instant::now();
        let (cq_pal, cq_idx) = run_color_quant(&pixels, width, height);
        let cq_ms = t0.elapsed().as_secs_f64() * 1000.0;
        save_indexed_png(
            &img_dir.join("color_quant.png"),
            &cq_pal,
            &cq_idx,
            width,
            height,
        );
        let cq_metrics = compute_metrics(&ref_rgb, &cq_pal, &cq_idx, width, height);
        let cq_sizes = measure_format_sizes(&cq_pal, &cq_idx, width, height);
        image_result.quantizers.push(QuantizerResult {
            name: "color_quant".into(),
            butteraugli: cq_metrics.0,
            ssimulacra2: cq_metrics.1,
            dssim: cq_metrics.2,
            png_bytes: cq_sizes.0,
            gif_bytes: cq_sizes.1,
            webp_bytes: cq_sizes.2,
            time_ms: cq_ms,
            note: "no dithering",
        });
        eprint!("  cq:{:.1}ms", cq_ms);

        // --- exoquant ---
        let t0 = Instant::now();
        let exo = run_exoquant(&pixels, width, height);
        let exo_ms = t0.elapsed().as_secs_f64() * 1000.0;
        match exo {
            Some((ex_pal, ex_idx)) => {
                save_indexed_png(
                    &img_dir.join("exoquant.png"),
                    &ex_pal,
                    &ex_idx,
                    width,
                    height,
                );
                let ex_metrics = compute_metrics(&ref_rgb, &ex_pal, &ex_idx, width, height);
                let ex_sizes = measure_format_sizes(&ex_pal, &ex_idx, width, height);
                image_result.quantizers.push(QuantizerResult {
                    name: "exoquant".into(),
                    butteraugli: ex_metrics.0,
                    ssimulacra2: ex_metrics.1,
                    dssim: ex_metrics.2,
                    png_bytes: ex_sizes.0,
                    gif_bytes: ex_sizes.1,
                    webp_bytes: ex_sizes.2,
                    time_ms: exo_ms,
                    note: "",
                });
                eprint!("  exo:{:.1}ms", exo_ms);
            }
            None => {
                eprintln!("  exoquant: skipped (compilation issue?)");
            }
        }

        eprintln!();
        report.images.push(image_result);
    }

    // Generate HTML report
    let html = generate_html(&report);
    let html_path = output_dir.join("index.html");
    std::fs::write(&html_path, &html).expect("failed to write index.html");
    eprintln!("\nReport: {}", html_path.display());

    // Print summary
    print_summary(&report);
}

// ---------------------------------------------------------------------------
// Quantizer wrappers
// ---------------------------------------------------------------------------

fn run_imagequant(pixels: &[rgb::RGB<u8>], width: usize, height: usize) -> (Vec<[u8; 3]>, Vec<u8>) {
    let mut attr = imagequant::Attributes::new();
    attr.set_quality(0, 80).unwrap();

    let rgba_pixels: Vec<imagequant::RGBA> = pixels
        .iter()
        .map(|p| imagequant::RGBA::new(p.r, p.g, p.b, 255))
        .collect();

    let mut img = attr.new_image(rgba_pixels, width, height, 0.0).unwrap();
    let mut result = attr.quantize(&mut img).unwrap();
    result.set_dithering_level(0.5).unwrap();

    let (palette, indices) = result.remapped(&mut img).unwrap();
    let palette_rgb: Vec<[u8; 3]> = palette.iter().map(|c| [c.r, c.g, c.b]).collect();
    (palette_rgb, indices)
}

fn run_quantizr(pixels: &[rgb::RGB<u8>], width: usize, height: usize) -> (Vec<[u8; 3]>, Vec<u8>) {
    let pixel_bytes: Vec<u8> = pixels.iter().flat_map(|p| [p.r, p.g, p.b, 255u8]).collect();
    let image = quantizr::Image::new(&pixel_bytes, width, height).unwrap();

    let mut options = quantizr::Options::default();
    options.set_max_colors(256).unwrap();

    let mut result = quantizr::QuantizeResult::quantize(&image, &options);
    result.set_dithering_level(0.5).unwrap();

    let mut indices = vec![0u8; width * height];
    result.remap_image(&image, &mut indices).unwrap();

    let palette = result.get_palette();
    let palette_rgb: Vec<[u8; 3]> = palette.entries[..palette.count as usize]
        .iter()
        .map(|c| [c.r, c.g, c.b])
        .collect();

    (palette_rgb, indices)
}

fn run_color_quant(
    pixels: &[rgb::RGB<u8>],
    _width: usize,
    _height: usize,
) -> (Vec<[u8; 3]>, Vec<u8>) {
    let rgba_bytes: Vec<u8> = pixels.iter().flat_map(|p| [p.r, p.g, p.b, 255u8]).collect();

    let nq = color_quant::NeuQuant::new(10, 256, &rgba_bytes);

    let palette_flat = nq.color_map_rgb();
    let palette: Vec<[u8; 3]> = palette_flat
        .chunks_exact(3)
        .map(|c| [c[0], c[1], c[2]])
        .collect();

    let indices: Vec<u8> = rgba_bytes
        .chunks_exact(4)
        .map(|pix| nq.index_of(pix) as u8)
        .collect();

    (palette, indices)
}

fn run_exoquant(
    pixels: &[rgb::RGB<u8>],
    width: usize,
    _height: usize,
) -> Option<(Vec<[u8; 3]>, Vec<u8>)> {
    use exoquant::{Color, convert_to_indexed, ditherer, optimizer};

    let exo_pixels: Vec<Color> = pixels
        .iter()
        .map(|p| Color::new(p.r, p.g, p.b, 255))
        .collect();

    let (palette, indices) = convert_to_indexed(
        &exo_pixels,
        width,
        256,
        &optimizer::KMeans,
        &ditherer::FloydSteinberg::new(),
    );

    let palette_rgb: Vec<[u8; 3]> = palette.iter().map(|c| [c.r, c.g, c.b]).collect();
    Some((palette_rgb, indices))
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

/// Returns (butteraugli, ssimulacra2, dssim)
fn compute_metrics(
    ref_rgb: &[RGB8],
    palette: &[[u8; 3]],
    indices: &[u8],
    width: usize,
    height: usize,
) -> (f64, f64, f64) {
    let test_rgb: Vec<RGB8> = indices
        .iter()
        .map(|&idx| {
            let c = palette[idx as usize];
            RGB8::new(c[0], c[1], c[2])
        })
        .collect();

    // Butteraugli
    let ref_img = ImgVec::new(ref_rgb.to_vec(), width, height);
    let test_img = ImgVec::new(test_rgb.clone(), width, height);
    let ba = butteraugli::butteraugli(
        ref_img.as_ref(),
        test_img.as_ref(),
        &ButteraugliParams::default(),
    )
    .map(|r| r.score)
    .unwrap_or(f64::NAN);

    // SSIMULACRA2
    let ref_pixels: Vec<[u8; 3]> = ref_rgb.iter().map(|p| [p.r, p.g, p.b]).collect();
    let test_pixels: Vec<[u8; 3]> = test_rgb.iter().map(|p| [p.r, p.g, p.b]).collect();
    let ref_img_ss = ImgVec::new(ref_pixels, width, height);
    let test_img_ss = ImgVec::new(test_pixels, width, height);
    let ss2 = compute_ssimulacra2(ref_img_ss.as_ref(), test_img_ss.as_ref()).unwrap_or(f64::NAN);

    // DSSIM
    let d = dssim_core::Dssim::new();
    let ref_dssim = d.create_image_rgb(ref_rgb, width, height).unwrap();
    let test_dssim = d.create_image_rgb(&test_rgb, width, height).unwrap();
    let (dssim_val, _) = d.compare(&ref_dssim, &test_dssim);
    let dssim: f64 = dssim_val.into();

    (ba, ss2, dssim)
}

// ---------------------------------------------------------------------------
// Format encoding (for size measurement)
// ---------------------------------------------------------------------------

/// Returns (png_bytes, gif_bytes, webp_bytes)
fn measure_format_sizes(
    palette: &[[u8; 3]],
    indices: &[u8],
    width: usize,
    height: usize,
) -> (usize, usize, usize) {
    let png_bytes = encode_indexed_png_bytes(palette, indices, width, height);
    let gif_bytes = encode_gif_bytes(palette, indices, width, height);
    let webp_bytes = encode_webp_bytes(palette, indices, width, height);
    (png_bytes, gif_bytes, webp_bytes)
}

fn encode_indexed_png_bytes(
    palette: &[[u8; 3]],
    indices: &[u8],
    width: usize,
    height: usize,
) -> usize {
    let mut buf = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut buf, width as u32, height as u32);
        encoder.set_color(png::ColorType::Indexed);
        encoder.set_depth(png::BitDepth::Eight);
        let palette_flat: Vec<u8> = palette.iter().flat_map(|c| c.iter().copied()).collect();
        encoder.set_palette(palette_flat);
        let mut writer = encoder.write_header().unwrap();
        writer.write_image_data(indices).unwrap();
    }
    buf.len()
}

fn encode_gif_bytes(palette: &[[u8; 3]], indices: &[u8], width: usize, height: usize) -> usize {
    let mut buf = Vec::new();
    let palette_flat: Vec<u8> = palette.iter().flat_map(|c| c.iter().copied()).collect();
    {
        let mut encoder =
            gif::Encoder::new(&mut buf, width as u16, height as u16, &palette_flat).unwrap();
        let frame =
            gif::Frame::from_indexed_pixels(width as u16, height as u16, indices.to_vec(), None);
        encoder.write_frame(&frame).unwrap();
    }
    buf.len()
}

fn encode_webp_bytes(palette: &[[u8; 3]], indices: &[u8], width: usize, height: usize) -> usize {
    // Reconstruct RGBA from palette + indices
    let mut rgba = Vec::with_capacity(width * height * 4);
    for &idx in indices {
        let c = palette[idx as usize];
        rgba.extend_from_slice(&[c[0], c[1], c[2], 255]);
    }

    let config = zenwebp::LosslessConfig::new();
    let result = zenwebp::EncodeRequest::lossless(
        &config,
        &rgba,
        zenwebp::PixelLayout::Rgba8,
        width as u32,
        height as u32,
    )
    .encode();

    match result {
        Ok(data) => data.len(),
        Err(_) => 0,
    }
}

// ---------------------------------------------------------------------------
// PNG I/O
// ---------------------------------------------------------------------------

fn save_truecolor_png(path: &Path, pixels: &[rgb::RGB<u8>], width: usize, height: usize) {
    let file = std::fs::File::create(path).unwrap();
    let buf = std::io::BufWriter::new(file);
    let mut encoder = png::Encoder::new(buf, width as u32, height as u32);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    let flat: Vec<u8> = pixels.iter().flat_map(|p| [p.r, p.g, p.b]).collect();
    let mut writer = encoder.write_header().unwrap();
    writer.write_image_data(&flat).unwrap();
}

fn save_indexed_png(path: &Path, palette: &[[u8; 3]], indices: &[u8], width: usize, height: usize) {
    let file = std::fs::File::create(path).unwrap();
    let buf = std::io::BufWriter::new(file);
    let mut encoder = png::Encoder::new(buf, width as u32, height as u32);
    encoder.set_color(png::ColorType::Indexed);
    encoder.set_depth(png::BitDepth::Eight);
    let palette_flat: Vec<u8> = palette.iter().flat_map(|c| c.iter().copied()).collect();
    encoder.set_palette(palette_flat);
    let mut writer = encoder.write_header().unwrap();
    writer.write_image_data(indices).unwrap();
}

// ---------------------------------------------------------------------------
// Summary
// ---------------------------------------------------------------------------

fn print_summary(report: &ReportData) {
    if report.images.is_empty() {
        return;
    }

    eprintln!("\n--- Summary ---");
    eprintln!(
        "{:<14} {:>7} {:>7} {:>8} {:>8} {:>8} {:>8} {:>7}",
        "Quantizer", "BA", "SS2", "DSSIM", "PNG", "GIF", "WebP", "ms"
    );
    eprintln!("{}", "-".repeat(72));

    // Collect per-quantizer name
    let mut all_names: Vec<String> = Vec::new();
    for img in &report.images {
        for q in &img.quantizers {
            if !all_names.contains(&q.name) {
                all_names.push(q.name.clone());
            }
        }
    }

    for name in &all_names {
        let mut sum_ba = 0.0f64;
        let mut sum_ss2 = 0.0f64;
        let mut sum_dssim = 0.0f64;
        let mut sum_png = 0usize;
        let mut sum_gif = 0usize;
        let mut sum_webp = 0usize;
        let mut sum_ms = 0.0f64;
        let mut count = 0u32;

        for img in &report.images {
            if let Some(q) = img.quantizers.iter().find(|q| &q.name == name) {
                sum_ba += q.butteraugli;
                sum_ss2 += q.ssimulacra2;
                sum_dssim += q.dssim;
                sum_png += q.png_bytes;
                sum_gif += q.gif_bytes;
                sum_webp += q.webp_bytes;
                sum_ms += q.time_ms;
                count += 1;
            }
        }

        if count > 0 {
            let n = count as f64;
            eprintln!(
                "{:<14} {:>7.2} {:>7.1} {:>8.5} {:>8.0} {:>8.0} {:>8.0} {:>7.1}",
                name,
                sum_ba / n,
                sum_ss2 / n,
                sum_dssim / n,
                sum_png as f64 / n,
                sum_gif as f64 / n,
                sum_webp as f64 / n,
                sum_ms / n,
            );
        }
    }
}

// ---------------------------------------------------------------------------
// HTML generation
// ---------------------------------------------------------------------------

fn generate_html(report: &ReportData) -> String {
    let data_json = report_to_json(report);
    HTML_TEMPLATE.replace("{DATA_JSON}", &data_json)
}

fn json_escape(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

fn report_to_json(report: &ReportData) -> String {
    let mut out = String::with_capacity(4096);
    out.push_str("{\"corpus\":\"");
    out.push_str(&json_escape(&report.corpus));
    out.push_str("\",\"images\":[");

    for (i, img) in report.images.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str("{\"name\":\"");
        out.push_str(&json_escape(&img.name));
        out.push_str("\",\"width\":");
        out.push_str(&img.width.to_string());
        out.push_str(",\"height\":");
        out.push_str(&img.height.to_string());
        out.push_str(",\"quantizers\":[");

        for (j, q) in img.quantizers.iter().enumerate() {
            if j > 0 {
                out.push(',');
            }
            out.push_str("{\"name\":\"");
            out.push_str(&json_escape(&q.name));
            out.push_str("\",\"butteraugli\":");
            out.push_str(&format!("{:.4}", q.butteraugli));
            out.push_str(",\"ssimulacra2\":");
            out.push_str(&format!("{:.2}", q.ssimulacra2));
            out.push_str(",\"dssim\":");
            out.push_str(&format!("{:.6}", q.dssim));
            out.push_str(",\"png_bytes\":");
            out.push_str(&q.png_bytes.to_string());
            out.push_str(",\"gif_bytes\":");
            out.push_str(&q.gif_bytes.to_string());
            out.push_str(",\"webp_bytes\":");
            out.push_str(&q.webp_bytes.to_string());
            out.push_str(",\"time_ms\":");
            out.push_str(&format!("{:.1}", q.time_ms));
            out.push_str(",\"note\":\"");
            out.push_str(&json_escape(q.note));
            out.push_str("\"}");
        }
        out.push_str("]}");
    }
    out.push_str("]}");
    out
}

const HTML_TEMPLATE: &str = r##"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Quantizer Comparison</title>
<style>
:root {
  --bg: #0f0f0f;
  --bg-card: #1a1a1a;
  --bg-hover: #252525;
  --text: #e0e0e0;
  --text-muted: #888;
  --border: #333;
  --primary: #3b82f6;
  --green: #22c55e;
  --red: #ef4444;
}

* { box-sizing: border-box; margin: 0; padding: 0; }

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  background: var(--bg);
  color: var(--text);
  overflow-x: hidden;
}

.header {
  padding: 12px 16px;
  border-bottom: 1px solid var(--border);
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 8px;
}

.header h1 { font-size: 1.1rem; font-weight: 600; }
.header .info { font-size: 0.8rem; color: var(--text-muted); }

/* Image selector strip */
.image-nav {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 16px;
  border-bottom: 1px solid var(--border);
  background: var(--bg-card);
}

.image-nav button {
  min-width: 44px;
  min-height: 44px;
  border: 1px solid var(--border);
  background: var(--bg);
  color: var(--text);
  border-radius: 4px;
  cursor: pointer;
  font-size: 1.2rem;
  flex-shrink: 0;
}

.image-nav button:hover { background: var(--bg-hover); }
.image-nav button:disabled { opacity: 0.3; cursor: default; }

.image-name {
  flex: 1;
  text-align: center;
  font-size: 0.9rem;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.image-counter {
  font-size: 0.75rem;
  color: var(--text-muted);
  flex-shrink: 0;
}

/* Viewport */
.viewport-container {
  position: relative;
  background: #111;
  overflow: hidden;
  touch-action: none;
}

.zoom-bar {
  display: flex;
  gap: 4px;
  padding: 6px 12px;
  background: rgba(0,0,0,0.7);
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  z-index: 20;
  align-items: center;
  flex-wrap: wrap;
}

.zoom-bar button {
  padding: 6px 12px;
  min-height: 36px;
  border: 1px solid var(--border);
  background: var(--bg);
  color: var(--text);
  border-radius: 4px;
  cursor: pointer;
  font-size: 0.75rem;
  white-space: nowrap;
}

.zoom-bar button:hover { background: var(--bg-hover); }
.zoom-bar button.active { background: var(--primary); color: white; border-color: var(--primary); }

.zoom-bar .dpr-info {
  font-size: 0.7rem;
  color: var(--text-muted);
  margin-left: auto;
}

.mode-btn { margin-left: 8px; }
.mode-btn.diff-active { background: var(--red) !important; border-color: var(--red) !important; color: white !important; }

.viewport-scroll {
  overflow: auto;
  display: flex;
  justify-content: center;
  align-items: center;
  cursor: grab;
}

.viewport-scroll.dragging { cursor: grabbing; }

.compare-container {
  position: relative;
  user-select: none;
  -webkit-user-select: none;
}

.compare-container.diff-mode {
  isolation: isolate;
}

.compare-container img {
  display: block;
  user-select: none;
  -webkit-user-drag: none;
}

.compare-left { position: relative; z-index: 1; }

.compare-right {
  position: absolute;
  top: 0;
  left: 0;
  z-index: 2;
}

.diff-overlay {
  position: absolute;
  top: 0;
  left: 0;
  z-index: 2;
  mix-blend-mode: difference;
}

.diff-label {
  position: absolute;
  bottom: 8px;
  left: 50%;
  transform: translateX(-50%);
  background: rgba(0,0,0,0.8);
  color: white;
  padding: 4px 12px;
  border-radius: 4px;
  font-size: 0.75rem;
  z-index: 15;
  pointer-events: none;
}

.slider-handle {
  position: absolute;
  top: 0;
  bottom: 0;
  width: 4px;
  background: white;
  z-index: 10;
  transform: translateX(-50%);
  cursor: ew-resize;
  box-shadow: 0 0 8px rgba(0,0,0,0.5);
}

.slider-handle::before {
  content: '';
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  width: 40px;
  height: 40px;
  background: white;
  border-radius: 50%;
  box-shadow: 0 2px 8px rgba(0,0,0,0.3);
}

.slider-handle::after {
  content: '\2194';
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  font-size: 1.25rem;
  color: #333;
  z-index: 1;
}

.compare-labels {
  display: flex;
  justify-content: space-between;
  padding: 4px 12px;
  font-size: 0.75rem;
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  z-index: 15;
  pointer-events: none;
}

.compare-labels span {
  padding: 2px 8px;
  border-radius: 3px;
}

.compare-labels span:first-child { background: rgba(37,99,235,0.85); }
.compare-labels span:last-child { background: rgba(124,58,237,0.85); }

/* Zoom modes — fit */
.zoom-fit .compare-container img {
  image-rendering: auto;
}

.zoom-pixel .compare-container img {
  image-rendering: pixelated;
  image-rendering: crisp-edges;
}

/* Choice tabs */
.choice-tabs {
  display: flex;
  gap: 4px;
  padding: 8px 12px;
  border-top: 1px solid var(--border);
  background: var(--bg-card);
  overflow-x: auto;
  -webkit-overflow-scrolling: touch;
}

.choice-tabs button {
  padding: 8px 16px;
  min-height: 44px;
  border: 1px solid var(--border);
  background: var(--bg);
  color: var(--text);
  border-radius: 4px;
  cursor: pointer;
  font-size: 0.8rem;
  white-space: nowrap;
  flex-shrink: 0;
}

.choice-tabs button:hover { background: var(--bg-hover); }
.choice-tabs button.sel-left { background: #2563eb; color: white; border-color: #2563eb; }
.choice-tabs button.sel-right { background: #7c3aed; color: white; border-color: #7c3aed; }
.choice-tabs button .note { font-size: 0.65rem; color: var(--text-muted); display: block; }
.choice-tabs button.sel-left .note,
.choice-tabs button.sel-right .note { color: rgba(255,255,255,0.7); }
.side-badge {
  display: inline-block;
  font-size: 0.6rem;
  font-weight: 700;
  width: 16px;
  height: 16px;
  line-height: 16px;
  text-align: center;
  border-radius: 3px;
  margin-right: 4px;
  vertical-align: middle;
  background: rgba(255,255,255,0.25);
}
.choice-tabs button.orig-locked { background: #2563eb; color: white; border-color: #2563eb; }
.choice-tabs button.orig-unlocked { color: var(--text-muted); border-style: dashed; }
.tab-sep {
  width: 1px;
  height: 28px;
  background: var(--border);
  flex-shrink: 0;
  align-self: center;
}

/* Metrics table */
.metrics-section {
  padding: 8px 12px;
  border-top: 1px solid var(--border);
}

.metrics-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.75rem;
}

.metrics-table th, .metrics-table td {
  padding: 4px 8px;
  text-align: right;
  border-bottom: 1px solid var(--border);
}

.metrics-table th { color: var(--text-muted); font-weight: 500; }
.metrics-table td:first-child, .metrics-table th:first-child { text-align: left; }
.metrics-table tr.active { background: rgba(59, 130, 246, 0.15); }
.metrics-table .best { color: var(--green); font-weight: 600; }

/* Summary section */
.summary-toggle {
  padding: 8px 12px;
  border-top: 1px solid var(--border);
  cursor: pointer;
  font-size: 0.8rem;
  color: var(--text-muted);
  background: var(--bg-card);
}

.summary-toggle:hover { background: var(--bg-hover); }

.summary-section {
  padding: 8px 12px;
  border-top: 1px solid var(--border);
  display: none;
}

.summary-section.open { display: block; }

.summary-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.75rem;
}

.summary-table th, .summary-table td {
  padding: 4px 8px;
  text-align: right;
  border-bottom: 1px solid var(--border);
  cursor: pointer;
}

.summary-table th { color: var(--text-muted); font-weight: 500; }
.summary-table td:first-child, .summary-table th:first-child { text-align: left; cursor: default; }
.summary-table .best { color: var(--green); font-weight: 600; }
.summary-table th.sorted { color: var(--primary); }

/* Keyboard help */
.kbd-help {
  padding: 8px 12px;
  font-size: 0.7rem;
  color: var(--text-muted);
  border-top: 1px solid var(--border);
}

.kbd-help kbd {
  background: var(--bg-hover);
  border: 1px solid var(--border);
  border-radius: 3px;
  padding: 1px 4px;
  font-family: monospace;
}

@media (max-width: 540px) {
  .choice-tabs { flex-wrap: wrap; }
  .zoom-bar { flex-wrap: wrap; }
  .kbd-help { display: none; }
}
</style>
</head>
<body>

<script>
// --- Data ---
const DATA = {DATA_JSON};

// --- State ---
let currentImageIdx = 0;
// Selection model:
// originalLocked=true (default): original is always left side, pick one quantizer for right.
// originalLocked=false: pick any two quantizers to compare.
// Choice indices: 0=original, 1..N=quantizers in button order.
let originalLocked = true;
let selected = [0, 1];
let zoomMode = 'fit'; // 'fit', 'native', '2x', '3x'
let viewMode = 'slider'; // 'slider', 'diff'
let sliderPct = 20; // start showing mostly the quantizer (right side)
let isSliding = false;
let isPanning = false;
let panStart = { x: 0, y: 0, scrollLeft: 0, scrollTop: 0 };

function getDPR() { return window.devicePixelRatio || 1; }
function getNativeZoom() { return 1 / getDPR(); }

function getZoomScale() {
  const dpr = getDPR();
  switch (zoomMode) {
    case 'fit': return 'fit';
    case 'native': return 1 / dpr;
    case '2x': return 2 / dpr;
    case '3x': return 3 / dpr;
    default: return 'fit';
  }
}

// --- Helpers ---

// choices[0] = original, choices[1..N] = quantizers
function getChoices(img) {
  const choices = [{ name: 'original', note: '', isOriginal: true }];
  for (const q of img.quantizers) {
    choices.push({ name: q.name, note: q.note, isOriginal: false });
  }
  return choices;
}

function choiceSrc(img, choiceIdx) {
  if (choiceIdx === 0) return `${img.name}/original.png`;
  return `${img.name}/${img.quantizers[choiceIdx - 1].name}.png`;
}

function choiceLabel(img, choiceIdx) {
  if (choiceIdx === 0) return 'original';
  const q = img.quantizers[choiceIdx - 1];
  return q.name + (q.note ? ' (' + q.note + ')' : '');
}

// --- Rendering ---

function render() {
  const img = DATA.images[currentImageIdx];
  if (!img) return;

  // Clamp selected indices to valid range
  const numChoices = img.quantizers.length + 1;
  selected = selected.map(s => Math.min(s, numChoices - 1));

  // Header
  document.getElementById('corpus-name').textContent = DATA.corpus;
  document.getElementById('image-count').textContent = `${DATA.images.length} images`;

  // Image nav
  document.getElementById('img-name').textContent = img.name;
  document.getElementById('img-counter').textContent = `${currentImageIdx + 1} / ${DATA.images.length}`;
  document.getElementById('btn-prev').disabled = currentImageIdx === 0;
  document.getElementById('btn-next').disabled = currentImageIdx === DATA.images.length - 1;

  // Viewport
  renderViewport(img);

  // Choice tabs
  renderChoiceTabs(img);

  // Metrics
  renderMetrics(img);

  // Zoom bar
  renderZoomBar();
}

function renderViewport(img) {
  const container = document.getElementById('compare-container');
  const scroll = document.getElementById('viewport-scroll');
  const scale = getZoomScale();

  const leftIdx = Math.min(...selected);
  const rightIdx = Math.max(...selected);
  const leftSrc = choiceSrc(img, leftIdx);
  const rightSrc = choiceSrc(img, rightIdx);
  const leftLabel = choiceLabel(img, leftIdx);
  const rightLabel = choiceLabel(img, rightIdx);

  scroll.className = 'viewport-scroll' + (scale === 'fit' ? ' zoom-fit' : ' zoom-pixel');

  // Compute viewport available height — use most of the remaining space
  const headerH = document.querySelector('.header').offsetHeight;
  const navH = document.querySelector('.image-nav').offsetHeight;
  const tabsH = document.querySelector('.choice-tabs')?.offsetHeight || 44;
  const metricsH = 120;
  const remaining = window.innerHeight - headerH - navH - tabsH - metricsH;
  const viewportH = Math.max(200, remaining);
  scroll.style.height = viewportH + 'px';

  // Compute explicit pixel dimensions for all modes
  let styleAttr = '';
  if (scale === 'fit') {
    const viewportW = scroll.clientWidth || window.innerWidth;
    const aspect = img.width / img.height;
    let fitW, fitH;
    if (viewportW / viewportH > aspect) {
      fitH = viewportH;
      fitW = Math.round(fitH * aspect);
    } else {
      fitW = viewportW;
      fitH = Math.round(fitW / aspect);
    }
    styleAttr = `width:${fitW}px;height:${fitH}px;`;
  } else {
    const w = Math.round(img.width * scale);
    const h = Math.round(img.height * scale);
    styleAttr = `width:${w}px;height:${h}px;`;
  }

  const isDiff = viewMode === 'diff' && leftIdx !== rightIdx;
  container.className = 'compare-container' + (isDiff ? ' diff-mode' : '');

  if (leftIdx === rightIdx) {
    // Same choice selected twice — just show the single image
    container.innerHTML = `
      <img class="compare-left" src="${leftSrc}" style="${styleAttr}" alt="${leftLabel}" loading="lazy">
      <div class="compare-labels"><span>${leftLabel}</span></div>
    `;
  } else if (isDiff) {
    container.innerHTML = `
      <img class="compare-left" src="${leftSrc}" style="${styleAttr}" alt="${leftLabel}" loading="lazy">
      <img class="diff-overlay" src="${rightSrc}" style="${styleAttr}" alt="${rightLabel}" loading="lazy">
      <div class="diff-label">DIFF: ${leftLabel} vs ${rightLabel} (black = identical)</div>
    `;
  } else {
    // Slider mode
    const clipRight = 100 - sliderPct;
    container.innerHTML = `
      <img class="compare-left" src="${leftSrc}" style="${styleAttr}" alt="${leftLabel}" loading="lazy">
      <img class="compare-right" src="${rightSrc}" style="${styleAttr}clip-path:inset(0 ${clipRight}% 0 0);" alt="${rightLabel}" loading="lazy">
      <div class="slider-handle" style="left:${sliderPct}%;"></div>
      <div class="compare-labels">
        <span>${leftLabel}</span>
        <span>${rightLabel}</span>
      </div>
    `;
  }
}

function renderZoomBar() {
  const dpr = getDPR();
  const nativePct = Math.round((1 / dpr) * 100);
  const zoomLevels = [
    { key: 'fit', label: 'Fit' },
    { key: 'native', label: `1:1 (${nativePct}%)` },
    { key: '2x', label: '2:1' },
    { key: '3x', label: '3:1' },
  ];

  const bar = document.getElementById('zoom-bar');
  let html = '';
  for (const z of zoomLevels) {
    html += `<button class="${zoomMode === z.key ? 'active' : ''}" onclick="setZoom('${z.key}')">${z.label}</button>`;
  }
  html += `<button class="mode-btn ${viewMode === 'diff' ? 'diff-active' : ''}" onclick="toggleDiff()">${viewMode === 'diff' ? 'Slider' : 'Diff'}</button>`;
  html += `<span class="dpr-info">${dpr}x DPR</span>`;
  bar.innerHTML = html;
}

function renderChoiceTabs(img) {
  const tabs = document.getElementById('choice-tabs');
  const choices = getChoices(img);
  const leftIdx = Math.min(...selected);
  const rightIdx = Math.max(...selected);
  let html = '';

  // Original button — checkbox-style lock toggle
  const origCheck = originalLocked ? '&#x2611;' : '&#x2610;';
  const origCls = originalLocked ? 'sel-left orig-locked' : 'orig-unlocked';
  const origBadge = originalLocked ? '<span class="side-badge">L</span>' : '';
  html += `<button class="${origCls}" onclick="toggleChoice(0)">${origBadge}${origCheck} original</button>`;
  html += '<span class="tab-sep"></span>';

  // Quantizer buttons
  for (let i = 1; i < choices.length; i++) {
    const c = choices[i];
    const isLeft = i === leftIdx;
    const isRight = i === rightIdx && leftIdx !== rightIdx;
    let cls = '';
    if (isLeft) cls = 'sel-left';
    else if (isRight) cls = 'sel-right';
    const note = c.note ? `<span class="note">${c.note}</span>` : '';
    const badge = isLeft ? '<span class="side-badge">L</span>' : isRight ? '<span class="side-badge">R</span>' : '';
    html += `<button class="${cls}" onclick="toggleChoice(${i})">${badge}${c.name}${note}</button>`;
  }
  tabs.innerHTML = html;
}

function renderMetrics(img) {
  const section = document.getElementById('metrics-section');

  // Find best values (BA: lower=better, SS2: higher=better, sizes: lower=better, time: lower=better)
  const qs = img.quantizers;
  const bestBA = Math.min(...qs.map(q => q.butteraugli));
  const bestSS2 = Math.max(...qs.map(q => q.ssimulacra2));
  const bestDSSIM = Math.min(...qs.map(q => q.dssim));
  const bestPNG = Math.min(...qs.map(q => q.png_bytes));
  const bestGIF = Math.min(...qs.map(q => q.gif_bytes));
  const bestWebP = Math.min(...qs.filter(q => q.webp_bytes > 0).map(q => q.webp_bytes));
  const bestTime = Math.min(...qs.map(q => q.time_ms));

  // Which quantizer indices (0-based into quantizers[]) are selected?
  // selected[] uses choice indices where 0=original, so quantizer i = choice i+1
  const selQuantIdxs = selected.filter(s => s > 0).map(s => s - 1);

  let html = `<table class="metrics-table">
    <tr><th>Quantizer</th><th>BA ↓</th><th>SS2 ↑</th><th>DSSIM ↓</th><th>PNG</th><th>GIF</th><th>WebP</th><th>ms</th></tr>`;

  const fmtSize = (v) => v > 0 ? (v / 1024).toFixed(1) + 'K' : '-';
  for (let i = 0; i < qs.length; i++) {
    const q = qs[i];
    const active = selQuantIdxs.includes(i) ? ' active' : '';
    html += `<tr class="${active}" onclick="toggleChoice(${i + 1})" style="cursor:pointer">
      <td>${q.name}${q.note ? ' *' : ''}</td>
      <td class="${q.butteraugli === bestBA ? 'best' : ''}">${q.butteraugli.toFixed(2)}</td>
      <td class="${q.ssimulacra2 === bestSS2 ? 'best' : ''}">${q.ssimulacra2.toFixed(1)}</td>
      <td class="${q.dssim === bestDSSIM ? 'best' : ''}">${q.dssim.toFixed(5)}</td>
      <td class="${q.png_bytes === bestPNG ? 'best' : ''}">${fmtSize(q.png_bytes)}</td>
      <td class="${q.gif_bytes === bestGIF ? 'best' : ''}">${fmtSize(q.gif_bytes)}</td>
      <td class="${q.webp_bytes === bestWebP ? 'best' : ''}">${fmtSize(q.webp_bytes)}</td>
      <td class="${q.time_ms === bestTime ? 'best' : ''}">${q.time_ms.toFixed(1)}</td>
    </tr>`;
  }
  html += '</table>';
  section.innerHTML = html;
}

function renderSummary() {
  const section = document.getElementById('summary-section');
  if (DATA.images.length === 0) return;

  // Collect per-quantizer averages
  const names = [];
  const totals = {};

  for (const img of DATA.images) {
    for (const q of img.quantizers) {
      if (!totals[q.name]) {
        totals[q.name] = { ba: 0, ss2: 0, dssim: 0, png: 0, gif: 0, webp: 0, ms: 0, n: 0 };
        names.push(q.name);
      }
      const t = totals[q.name];
      t.ba += q.butteraugli;
      t.ss2 += q.ssimulacra2;
      t.dssim += q.dssim;
      t.png += q.png_bytes;
      t.gif += q.gif_bytes;
      t.webp += q.webp_bytes;
      t.ms += q.time_ms;
      t.n += 1;
    }
  }

  const avgs = names.map(name => {
    const t = totals[name];
    return {
      name,
      ba: t.ba / t.n,
      ss2: t.ss2 / t.n,
      dssim: t.dssim / t.n,
      png: t.png / t.n,
      gif: t.gif / t.n,
      webp: t.webp / t.n,
      ms: t.ms / t.n,
    };
  });

  const bestBA = Math.min(...avgs.map(a => a.ba));
  const bestSS2 = Math.max(...avgs.map(a => a.ss2));
  const bestDSSIM = Math.min(...avgs.map(a => a.dssim));
  const bestPNG = Math.min(...avgs.map(a => a.png));
  const bestGIF = Math.min(...avgs.map(a => a.gif));
  const bestWebP = Math.min(...avgs.filter(a => a.webp > 0).map(a => a.webp));
  const bestTime = Math.min(...avgs.map(a => a.ms));

  const fmtSize = (v) => v > 0 ? (v / 1024).toFixed(1) + 'K' : '-';

  let html = `<table class="summary-table">
    <tr><th>Quantizer</th><th>Avg BA ↓</th><th>Avg SS2 ↑</th><th>Avg DSSIM ↓</th><th>Avg PNG</th><th>Avg GIF</th><th>Avg WebP</th><th>Avg ms</th></tr>`;

  for (const a of avgs) {
    html += `<tr>
      <td>${a.name}</td>
      <td class="${a.ba === bestBA ? 'best' : ''}">${a.ba.toFixed(2)}</td>
      <td class="${a.ss2 === bestSS2 ? 'best' : ''}">${a.ss2.toFixed(1)}</td>
      <td class="${a.dssim === bestDSSIM ? 'best' : ''}">${a.dssim.toFixed(5)}</td>
      <td class="${a.png === bestPNG ? 'best' : ''}">${fmtSize(a.png)}</td>
      <td class="${a.gif === bestGIF ? 'best' : ''}">${fmtSize(a.gif)}</td>
      <td class="${a.webp === bestWebP ? 'best' : ''}">${fmtSize(a.webp)}</td>
      <td class="${a.ms === bestTime ? 'best' : ''}">${a.ms.toFixed(1)}</td>
    </tr>`;
  }
  html += '</table>';
  section.innerHTML = html;
}

// --- Interactions ---

function prevImage() {
  if (currentImageIdx > 0) { currentImageIdx--; render(); }
}
function nextImage() {
  if (currentImageIdx < DATA.images.length - 1) { currentImageIdx++; render(); }
}

function toggleChoice(idx) {
  const img = DATA.images[currentImageIdx];
  if (!img) return;
  const numChoices = img.quantizers.length + 1;

  if (idx === 0) {
    // Clicked "original" — toggle the lock
    if (originalLocked) {
      // Unlock: switch to comparing two quantizers
      originalLocked = false;
      const curQuant = selected.find(s => s > 0) || 1;
      // Pick another quantizer that isn't the current one
      let other = 0;
      for (let i = 1; i < numChoices; i++) {
        if (i !== curQuant) { other = i; break; }
      }
      if (other === 0) other = curQuant; // only one quantizer available
      selected = [curQuant, other];
      sliderPct = 50;
    } else {
      // Lock: go back to original vs one quantizer
      originalLocked = true;
      const curQuant = selected[1]; // keep the most recent
      selected = [0, curQuant];
      sliderPct = 20;
    }
    render();
    return;
  }

  // Clicked a quantizer
  if (originalLocked) {
    // Original stays, just swap in the clicked quantizer
    if (selected.includes(idx)) {
      // Already showing this one — snap slider to emphasize it
      sliderPct = 20;
    } else {
      selected = [0, idx];
      sliderPct = 20;
    }
  } else {
    // Two-quantizer mode
    if (selected.includes(idx)) {
      // Already selected — snap slider to emphasize it
      const leftIdx = Math.min(...selected);
      sliderPct = (idx === leftIdx) ? 80 : 20;
    } else {
      // Replace the older selection
      selected = [selected[1], idx];
      const leftIdx = Math.min(...selected);
      sliderPct = (idx === leftIdx) ? 80 : 20;
    }
  }
  render();
}

function setZoom(mode) {
  zoomMode = mode;
  render();
}

function toggleDiff() {
  viewMode = viewMode === 'diff' ? 'slider' : 'diff';
  render();
}

function toggleSummary() {
  const s = document.getElementById('summary-section');
  s.classList.toggle('open');
  if (s.classList.contains('open')) renderSummary();
}

// Slider drag
function updateSlider(clientX) {
  const container = document.getElementById('compare-container');
  const rect = container.getBoundingClientRect();
  sliderPct = Math.max(0, Math.min(100, ((clientX - rect.left) / rect.width) * 100));

  // Update DOM directly for performance (no full re-render)
  const handle = container.querySelector('.slider-handle');
  const right = container.querySelector('.compare-right');
  if (handle) handle.style.left = sliderPct + '%';
  if (right) right.style.clipPath = `inset(0 ${100 - sliderPct}% 0 0)`;
}

document.addEventListener('mousedown', (e) => {
  const scroll = document.getElementById('viewport-scroll');
  const container = document.getElementById('compare-container');
  if (!scroll || !container) return;

  if (container.contains(e.target) && viewMode === 'slider') {
    isSliding = true;
    updateSlider(e.clientX);
    e.preventDefault();
  } else if (scroll.contains(e.target) && zoomMode !== 'fit') {
    isPanning = true;
    panStart = {
      x: e.pageX, y: e.pageY,
      scrollLeft: scroll.scrollLeft, scrollTop: scroll.scrollTop,
    };
    scroll.classList.add('dragging');
    e.preventDefault();
  }
});

document.addEventListener('mousemove', (e) => {
  if (isSliding) {
    updateSlider(e.clientX);
  } else if (isPanning) {
    const scroll = document.getElementById('viewport-scroll');
    scroll.scrollLeft = panStart.scrollLeft - (e.pageX - panStart.x);
    scroll.scrollTop = panStart.scrollTop - (e.pageY - panStart.y);
  }
});

document.addEventListener('mouseup', () => {
  isSliding = false;
  if (isPanning) {
    isPanning = false;
    const scroll = document.getElementById('viewport-scroll');
    if (scroll) scroll.classList.remove('dragging');
  }
});

// Touch support for slider
document.addEventListener('touchstart', (e) => {
  const container = document.getElementById('compare-container');
  if (!container || !container.contains(e.target)) return;
  if (viewMode !== 'slider') return;

  isSliding = true;
  updateSlider(e.touches[0].clientX);
}, { passive: true });

document.addEventListener('touchmove', (e) => {
  if (isSliding) {
    updateSlider(e.touches[0].clientX);
    e.preventDefault();
  }
}, { passive: false });

document.addEventListener('touchend', () => {
  isSliding = false;
});

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

  switch (e.key) {
    case 'ArrowLeft': prevImage(); e.preventDefault(); break;
    case 'ArrowRight': nextImage(); e.preventDefault(); break;
    // 0 = original, 1-5 = quantizers
    case '0': case '1': case '2': case '3': case '4': case '5': {
      const choiceIdx = parseInt(e.key);
      const img = DATA.images[currentImageIdx];
      if (img && choiceIdx <= img.quantizers.length) toggleChoice(choiceIdx);
      e.preventDefault();
      break;
    }
    case 'f': case 'F': setZoom('fit'); e.preventDefault(); break;
    case 'n': case 'N': setZoom('native'); e.preventDefault(); break;
    case 'd': case 'D': toggleDiff(); e.preventDefault(); break;
  }
});

// Init
window.addEventListener('DOMContentLoaded', () => {
  render();
});
window.addEventListener('resize', () => {
  render();
});
</script>

<div class="header">
  <h1>Quantizer Comparison</h1>
  <span class="info"><span id="corpus-name"></span> &mdash; <span id="image-count"></span></span>
</div>

<div class="image-nav">
  <button id="btn-prev" onclick="prevImage()">&larr;</button>
  <span class="image-name" id="img-name"></span>
  <span class="image-counter" id="img-counter"></span>
  <button id="btn-next" onclick="nextImage()">&rarr;</button>
</div>

<div class="viewport-container">
  <div class="zoom-bar" id="zoom-bar"></div>
  <div class="viewport-scroll zoom-fit" id="viewport-scroll">
    <div class="compare-container" id="compare-container"></div>
  </div>
</div>

<div class="choice-tabs" id="choice-tabs"></div>

<div class="metrics-section" id="metrics-section"></div>

<div class="summary-toggle" onclick="toggleSummary()">&#9660; Summary (averages across all images)</div>
<div class="summary-section" id="summary-section"></div>

<div class="kbd-help">
  <kbd>&larr;</kbd><kbd>&rarr;</kbd> prev/next image &nbsp;
  <kbd>0</kbd> original &nbsp;
  <kbd>1</kbd>-<kbd>5</kbd> quantizers &nbsp;
  <kbd>f</kbd> fit &nbsp;
  <kbd>n</kbd> native 1:1 &nbsp;
  <kbd>d</kbd> diff
</div>

</body>
</html>
"##;
