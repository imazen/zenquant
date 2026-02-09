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
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use zenquant::{OutputFormat, Quality, QuantizeConfig};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

const QUANTIZER_NAMES: &[&str] = &[
    "zq-fast",
    "zq-balanced",
    "zq-best",
    "zq-best-d50",
    "zq-best-d60",
    "zq-best-q-d50",
    "zq-best-q-d60",
    "iq-s1-d50",
    "iq-s4-d100",
    "iq-s1-d100",
    "quantizr",
    "color_quant",
    "exoquant",
];

#[derive(Clone, Debug)]
struct ImageResult {
    name: String,
    group: String,
    width: usize,
    height: usize,
    quantizers: Vec<QuantizerResult>,
}

#[derive(Clone, Debug)]
struct QuantizerResult {
    name: String,
    butteraugli: f64,
    ssimulacra2: f64,
    dssim: f64,
    png_bytes: usize,
    gif_bytes: usize,
    time_ms: f64,
    note: &'static str,
}

struct ReportData {
    label: String,
    images: Vec<ImageResult>,
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn resolve_corpus(name: &str) -> &'static str {
    match name {
        "cid22" => "CID22/CID22-512/training",
        "clic2025" => "clic2025/final-test",
        "gb82-sc" => "gb82-sc",
        _ => {
            eprintln!("Unknown corpus '{name}'. Use: cid22, clic2025, gb82-sc");
            std::process::exit(1);
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "Usage: quantizer_comparison <corpus[,corpus,...]> <output_dir> [max_images] [--recalc=name,...]"
        );
        eprintln!("  corpus: cid22, clic2025, gb82-sc (comma-separated for multiple)");
        eprintln!("  --recalc=zq-best,zq-balanced  invalidate cache for specific quantizers");
        eprintln!("  --recalc=all                   ignore all cached results");
        std::process::exit(1);
    }

    let corpus_arg = &args[1];
    let output_dir = PathBuf::from(&args[2]);
    let max_images: usize = args
        .get(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(usize::MAX);

    // Parse --recalc flag from any position
    let recalc: Vec<String> = args
        .iter()
        .filter_map(|a| a.strip_prefix("--recalc="))
        .flat_map(|v| v.split(',').map(|s| s.to_string()))
        .collect();
    let recalc_all = recalc.iter().any(|s| s == "all");

    // Parse comma-separated corpus names
    let corpus_names: Vec<&str> = corpus_arg.split(',').collect();
    let corpus_specs: Vec<(&str, &str)> = corpus_names
        .iter()
        .map(|name| (*name, resolve_corpus(name)))
        .collect();

    let corpus = codec_corpus::Corpus::new().expect("failed to init codec-corpus");
    std::fs::create_dir_all(&output_dir).expect("cannot create output directory");

    let label = corpus_names.join(", ");
    let mut report = ReportData {
        label: label.clone(),
        images: Vec::new(),
    };

    // Collect all (corpus_name, path) pairs across corpora
    let mut all_tasks: Vec<(String, PathBuf)> = Vec::new();

    for &(corpus_name, corpus_path) in &corpus_specs {
        eprintln!("Fetching corpus '{corpus_name}' ({corpus_path})...");
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
            continue;
        }

        for path in paths {
            all_tasks.push((corpus_name.to_string(), path));
        }
    }

    let num_threads: usize = args
        .iter()
        .filter_map(|a| a.strip_prefix("--threads="))
        .next()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8);

    let total = all_tasks.len();
    eprintln!(
        "Processing {} images with {} quantizers, {} threads",
        total,
        QUANTIZER_NAMES.len(),
        num_threads
    );

    // Process images in parallel
    let results: Arc<Mutex<Vec<ImageResult>>> = Arc::new(Mutex::new(Vec::new()));

    // Process in chunks of num_threads
    for chunk_start in (0..total).step_by(num_threads) {
        let chunk_end = (chunk_start + num_threads).min(total);
        std::thread::scope(|s| {
            for task_idx in chunk_start..chunk_end {
                let (corpus_name, path) = &all_tasks[task_idx];
                let results = Arc::clone(&results);
                let output_dir = &output_dir;
                let recalc = &recalc;

                s.spawn(move || {
                let stem = path.file_stem().unwrap_or_default().to_string_lossy();

                let img = match image::open(path) {
                    Ok(img) => img.to_rgb8(),
                    Err(e) => {
                        eprintln!("[{}/{}] {stem}: skipping ({e})", task_idx + 1, total);
                        return;
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
                let ref_rgb: Vec<RGB8> =
                    pixels.iter().map(|p| RGB8::new(p.r, p.g, p.b)).collect();

                let img_dir = output_dir.join(&*stem);
                std::fs::create_dir_all(&img_dir).unwrap();
                save_truecolor_png(&img_dir.join("original.png"), &pixels, width, height);

                let mut image_result = ImageResult {
                    name: stem.to_string(),
                    group: corpus_name.clone(),
                    width,
                    height,
                    quantizers: Vec::new(),
                };

                // Load cache
                let cache_path = img_dir.join("cache.json");
                let mut cache = load_cache(&cache_path);
                let mut log = format!("[{}/{}] {corpus_name}/{stem}", task_idx + 1, total);

                for &qname in QUANTIZER_NAMES {
                    let should_recalc =
                        recalc_all || recalc.iter().any(|r| r == qname);
                    if !should_recalc {
                        if let Some(cached) = cache.get(qname) {
                            if img_dir.join(format!("{qname}.png")).exists() {
                                image_result.quantizers.push(cached.clone());
                                log.push_str(&format!("  {qname}:cached"));
                                continue;
                            }
                        }
                    }

                    let t0 = Instant::now();
                    let result = run_quantizer(qname, &pixels, width, height);
                    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;

                    let (pal, idx, note) = match result {
                        Some(r) => r,
                        None => {
                            log.push_str(&format!("  {qname}:skip"));
                            continue;
                        }
                    };

                    save_indexed_png(
                        &img_dir.join(format!("{qname}.png")),
                        &pal,
                        &idx,
                        width,
                        height,
                    );
                    let metrics = compute_metrics(&ref_rgb, &pal, &idx, width, height);
                    let sizes = measure_format_sizes(&pal, &idx, width, height);
                    let qr = QuantizerResult {
                        name: qname.into(),
                        butteraugli: metrics.0,
                        ssimulacra2: metrics.1,
                        dssim: metrics.2,
                        png_bytes: sizes.0,
                        gif_bytes: sizes.1,
                        time_ms: elapsed_ms,
                        note,
                    };
                    cache.insert(qname.to_string(), qr.clone());
                    image_result.quantizers.push(qr);
                    log.push_str(&format!("  {qname}:{:.0}ms", elapsed_ms));
                }

                save_cache(&cache_path, &cache);
                eprintln!("{log}");

                results.lock().unwrap().push(image_result);
            });
            }
        });
    }

    // Sort results to maintain deterministic order (by group then name)
    let mut images = Arc::try_unwrap(results).unwrap().into_inner().unwrap();
    images.sort_by(|a, b| a.group.cmp(&b.group).then(a.name.cmp(&b.name)));
    report.images = images;

    // Generate HTML report
    let html = generate_html(&report);
    let html_path = output_dir.join("index.html");
    std::fs::write(&html_path, &html).expect("failed to write index.html");
    eprintln!("\nReport: {}", html_path.display());

    // Print summary
    print_summary(&report);
}

// ---------------------------------------------------------------------------
// Cache
// ---------------------------------------------------------------------------

fn load_cache(path: &Path) -> HashMap<String, QuantizerResult> {
    let mut cache = HashMap::new();
    let data = match std::fs::read_to_string(path) {
        Ok(d) => d,
        Err(_) => return cache,
    };

    // Minimal JSON parser for our cache format: { "name": { fields... }, ... }
    // Each value is a QuantizerResult serialized as JSON object
    for entry in data.split(r#""__entry__":"#).skip(1) {
        // Parse: "name":"...", "butteraugli":..., etc. until "}"
        let parse_str = |key: &str| -> Option<String> {
            let marker = format!("\"{key}\":\"");
            let start = entry.find(&marker)? + marker.len();
            let end = entry[start..].find('"')? + start;
            Some(entry[start..end].to_string())
        };
        let parse_f64 = |key: &str| -> Option<f64> {
            let marker = format!("\"{key}\":");
            let start = entry.find(&marker)? + marker.len();
            let end = entry[start..]
                .find(|c: char| c != '.' && c != '-' && !c.is_ascii_digit())
                .unwrap_or(entry.len() - start)
                + start;
            entry[start..end].parse().ok()
        };
        let parse_usize = |key: &str| -> Option<usize> {
            parse_f64(key).map(|v| v as usize)
        };

        if let (Some(name), Some(ba), Some(ss2), Some(dssim), Some(png), Some(gif), Some(ms)) = (
            parse_str("name"),
            parse_f64("butteraugli"),
            parse_f64("ssimulacra2"),
            parse_f64("dssim"),
            parse_usize("png_bytes"),
            parse_usize("gif_bytes"),
            parse_f64("time_ms"),
        ) {
            let note_str = parse_str("note").unwrap_or_default();
            let note: &'static str = match note_str.as_str() {
                "no dithering" => "no dithering",
                "spd1 q100 d0.5" => "spd1 q100 d0.5",
                "spd4 q100 d1.0" => "spd4 q100 d1.0",
                "spd1 q100 d1.0" => "spd1 q100 d1.0",
                _ => "",
            };
            cache.insert(
                name.clone(),
                QuantizerResult {
                    name,
                    butteraugli: ba,
                    ssimulacra2: ss2,
                    dssim,
                    png_bytes: png,
                    gif_bytes: gif,
                    time_ms: ms,
                    note,
                },
            );
        }
    }
    cache
}

fn save_cache(path: &Path, cache: &HashMap<String, QuantizerResult>) {
    let mut out = String::from("{");
    for (i, (_, q)) in cache.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(&format!(
            r#""__entry__":1,"name":"{}","butteraugli":{:.4},"ssimulacra2":{:.2},"dssim":{:.6},"png_bytes":{},"gif_bytes":{},"time_ms":{:.1},"note":"{}""#,
            json_escape(&q.name),
            q.butteraugli,
            q.ssimulacra2,
            q.dssim,
            q.png_bytes,
            q.gif_bytes,
            q.time_ms,
            json_escape(q.note),
        ));
    }
    out.push('}');
    let _ = std::fs::write(path, out);
}

// ---------------------------------------------------------------------------
// Quantizer wrappers
// ---------------------------------------------------------------------------

/// Dispatch to the right quantizer by name. Returns (palette, indices, note).
fn run_quantizer(
    name: &str,
    pixels: &[rgb::RGB<u8>],
    width: usize,
    height: usize,
) -> Option<(Vec<[u8; 3]>, Vec<u8>, &'static str)> {
    match name {
        "zq-fast" => {
            let cfg = QuantizeConfig::new(OutputFormat::Png)
                .quality(Quality::Fast)
                .max_colors(256);
            let r = zenquant::quantize(pixels, width, height, &cfg).unwrap();
            Some((r.palette().to_vec(), r.indices().to_vec(), ""))
        }
        "zq-balanced" => {
            let cfg = QuantizeConfig::new(OutputFormat::Png)
                .quality(Quality::Balanced)
                .max_colors(256);
            let r = zenquant::quantize(pixels, width, height, &cfg).unwrap();
            Some((r.palette().to_vec(), r.indices().to_vec(), ""))
        }
        "zq-best" => {
            let cfg = QuantizeConfig::new(OutputFormat::Png).max_colors(256);
            let r = zenquant::quantize(pixels, width, height, &cfg).unwrap();
            Some((r.palette().to_vec(), r.indices().to_vec(), "d0.3 (default)"))
        }
        "zq-best-d50" => {
            let cfg = QuantizeConfig::new(OutputFormat::Png)
                .max_colors(256)
                ._dither_strength(0.5);
            let r = zenquant::quantize(pixels, width, height, &cfg).unwrap();
            Some((r.palette().to_vec(), r.indices().to_vec(), "d0.5"))
        }
        "zq-best-d60" => {
            let cfg = QuantizeConfig::new(OutputFormat::Png)
                .max_colors(256)
                ._dither_strength(0.6);
            let r = zenquant::quantize(pixels, width, height, &cfg).unwrap();
            Some((r.palette().to_vec(), r.indices().to_vec(), "d0.6"))
        }
        "zq-best-q-d50" => {
            let cfg = QuantizeConfig::new(OutputFormat::Png)
                .max_colors(256)
                ._dither_strength(0.5)
                ._run_priority_quality();
            let r = zenquant::quantize(pixels, width, height, &cfg).unwrap();
            Some((r.palette().to_vec(), r.indices().to_vec(), "d0.5 quality-runs"))
        }
        "zq-best-q-d60" => {
            let cfg = QuantizeConfig::new(OutputFormat::Png)
                .max_colors(256)
                ._dither_strength(0.6)
                ._run_priority_quality();
            let r = zenquant::quantize(pixels, width, height, &cfg).unwrap();
            Some((r.palette().to_vec(), r.indices().to_vec(), "d0.6 quality-runs"))
        }
        "iq-s1-d50" => {
            let (p, i) = run_imagequant(pixels, width, height, 1, 100, 0.5);
            Some((p, i, "spd1 q100 d0.5"))
        }
        "iq-s4-d100" => {
            let (p, i) = run_imagequant(pixels, width, height, 4, 100, 1.0);
            Some((p, i, "spd4 q100 d1.0"))
        }
        "iq-s1-d100" => {
            let (p, i) = run_imagequant(pixels, width, height, 1, 100, 1.0);
            Some((p, i, "spd1 q100 d1.0"))
        }
        "quantizr" => {
            let (p, i) = run_quantizr(pixels, width, height);
            Some((p, i, ""))
        }
        "color_quant" => {
            let (p, i) = run_color_quant(pixels, width, height);
            Some((p, i, "no dithering"))
        }
        "exoquant" => run_exoquant(pixels, width, height).map(|(p, i)| (p, i, "")),
        _ => None,
    }
}

fn run_imagequant(
    pixels: &[rgb::RGB<u8>],
    width: usize,
    height: usize,
    speed: i32,
    quality: u8,
    dither: f32,
) -> (Vec<[u8; 3]>, Vec<u8>) {
    let mut attr = imagequant::Attributes::new();
    attr.set_quality(0, quality).unwrap();
    attr.set_speed(speed).unwrap();

    let rgba_pixels: Vec<imagequant::RGBA> = pixels
        .iter()
        .map(|p| imagequant::RGBA::new(p.r, p.g, p.b, 255))
        .collect();

    let mut img = attr.new_image(rgba_pixels, width, height, 0.0).unwrap();
    let mut result = attr.quantize(&mut img).unwrap();
    result.set_dithering_level(dither).unwrap();

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

/// Returns (png_bytes, gif_bytes)
fn measure_format_sizes(
    palette: &[[u8; 3]],
    indices: &[u8],
    width: usize,
    height: usize,
) -> (usize, usize) {
    let png_bytes = encode_indexed_png_bytes(palette, indices, width, height);
    let gif_bytes = encode_gif_bytes(palette, indices, width, height);
    (png_bytes, gif_bytes)
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
        "{:<14} {:>7} {:>7} {:>8} {:>8} {:>8} {:>7}",
        "Quantizer", "BA", "SS2", "DSSIM", "PNG", "GIF", "ms"
    );
    eprintln!("{}", "-".repeat(64));

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
        let mut sum_ms = 0.0f64;
        let mut count = 0u32;

        for img in &report.images {
            if let Some(q) = img.quantizers.iter().find(|q| &q.name == name) {
                sum_ba += q.butteraugli;
                sum_ss2 += q.ssimulacra2;
                sum_dssim += q.dssim;
                sum_png += q.png_bytes;
                sum_gif += q.gif_bytes;
                sum_ms += q.time_ms;
                count += 1;
            }
        }

        if count > 0 {
            let n = count as f64;
            eprintln!(
                "{:<14} {:>7.2} {:>7.1} {:>8.5} {:>8.0} {:>8.0} {:>7.1}",
                name,
                sum_ba / n,
                sum_ss2 / n,
                sum_dssim / n,
                sum_png as f64 / n,
                sum_gif as f64 / n,
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
    out.push_str("{\"label\":\"");
    out.push_str(&json_escape(&report.label));
    out.push_str("\",\"images\":[");

    for (i, img) in report.images.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str("{\"name\":\"");
        out.push_str(&json_escape(&img.name));
        out.push_str("\",\"group\":\"");
        out.push_str(&json_escape(&img.group));
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

/* Group filter bar */
.group-bar {
  display: flex;
  gap: 4px;
  padding: 6px 12px;
  border-bottom: 1px solid var(--border);
  background: var(--bg-card);
  overflow-x: auto;
}

.group-bar button {
  padding: 6px 14px;
  min-height: 36px;
  border: 1px solid var(--border);
  background: var(--bg);
  color: var(--text);
  border-radius: 4px;
  cursor: pointer;
  font-size: 0.8rem;
  white-space: nowrap;
}

.group-bar button:hover { background: var(--bg-hover); }
.group-bar button.active { background: var(--primary); color: white; border-color: var(--primary); }

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
.metrics-table th.sorted { color: var(--primary); }
.metrics-table th { cursor: pointer; }

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
// Sort state for metrics table — persists across image changes
// key: field name, asc: true = ascending (best first for lower-is-better)
let metricSort = { key: null, asc: true };
// Group filter — null means "all"
let activeGroup = null;

function getGroups() {
  const seen = new Set();
  const groups = [];
  for (const img of DATA.images) {
    if (!seen.has(img.group)) { seen.add(img.group); groups.push(img.group); }
  }
  return groups;
}

function getFilteredImages() {
  if (activeGroup === null) return DATA.images;
  return DATA.images.filter(img => img.group === activeGroup);
}

function setGroup(g) {
  activeGroup = g;
  currentImageIdx = 0;
  render();
}

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
  const filtered = getFilteredImages();
  const img = filtered[currentImageIdx];
  if (!img) return;

  // Clamp selected indices to valid range
  const numChoices = img.quantizers.length + 1;
  selected = selected.map(s => Math.min(s, numChoices - 1));

  // Header
  document.getElementById('corpus-name').textContent = DATA.label;
  document.getElementById('image-count').textContent = `${filtered.length} images`;

  // Group bar
  renderGroupBar();

  // Image nav
  document.getElementById('img-name').textContent = `${img.group}: ${img.name}`;
  document.getElementById('img-counter').textContent = `${currentImageIdx + 1} / ${filtered.length}`;
  document.getElementById('btn-prev').disabled = currentImageIdx === 0;
  document.getElementById('btn-next').disabled = currentImageIdx === filtered.length - 1;

  // Viewport
  renderViewport(img);

  // Choice tabs
  renderChoiceTabs(img);

  // Metrics
  renderMetrics(img);

  // Zoom bar
  renderZoomBar();
}

function renderGroupBar() {
  const bar = document.getElementById('group-bar');
  const groups = getGroups();
  if (groups.length <= 1) { bar.style.display = 'none'; return; }
  bar.style.display = '';
  let html = `<button class="${activeGroup === null ? 'active' : ''}" onclick="setGroup(null)">All</button>`;
  for (const g of groups) {
    html += `<button class="${activeGroup === g ? 'active' : ''}" onclick="setGroup('${g}')">${g}</button>`;
  }
  bar.innerHTML = html;
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

  // Compute viewport height — leave room for tabs, metrics, and summary below
  const headerH = document.querySelector('.header').offsetHeight;
  const navH = document.querySelector('.image-nav').offsetHeight;
  const belowH = 250; // tabs + metrics + summary toggle + keyboard help
  const available = window.innerHeight - headerH - navH - belowH;
  const viewportH = Math.max(200, Math.min(available, window.innerHeight * 0.55));
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
    // Slider mode — left image is base, right image clips from the left
    // so right image is visible to the RIGHT of the slider
    container.innerHTML = `
      <img class="compare-left" src="${leftSrc}" style="${styleAttr}" alt="${leftLabel}" loading="lazy">
      <img class="compare-right" src="${rightSrc}" style="${styleAttr}clip-path:inset(0 0 0 ${sliderPct}%);" alt="${rightLabel}" loading="lazy">
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

// Column definitions for metrics table
const METRIC_COLS = [
  { key: 'butteraugli', label: 'BA', dir: 'asc', fmt: v => v.toFixed(2) },
  { key: 'ssimulacra2', label: 'SS2', dir: 'desc', fmt: v => v.toFixed(1) },
  { key: 'dssim', label: 'DSSIM', dir: 'asc', fmt: v => v.toFixed(5) },
  { key: 'png_bytes', label: 'PNG', dir: 'asc', fmt: v => v > 0 ? (v / 1024).toFixed(1) + 'K' : '-' },
  { key: 'gif_bytes', label: 'GIF', dir: 'asc', fmt: v => v > 0 ? (v / 1024).toFixed(1) + 'K' : '-' },
  { key: 'time_ms', label: 'ms', dir: 'asc', fmt: v => v.toFixed(1) },
];

function sortMetrics(key) {
  const col = METRIC_COLS.find(c => c.key === key);
  if (!col) return;
  if (metricSort.key === key) {
    // Toggle direction
    metricSort.asc = !metricSort.asc;
  } else {
    // New column — use its natural "best" direction
    metricSort.key = key;
    metricSort.asc = col.dir === 'asc';
  }
  render();
}

function renderMetrics(img) {
  const section = document.getElementById('metrics-section');
  const qs = img.quantizers;

  // Build sorted index array (maps display row → original quantizer index)
  const indices = qs.map((_, i) => i);
  if (metricSort.key) {
    const k = metricSort.key;
    const asc = metricSort.asc;
    indices.sort((a, b) => asc ? qs[a][k] - qs[b][k] : qs[b][k] - qs[a][k]);
  }

  // Find best values
  const best = {};
  for (const col of METRIC_COLS) {
    const vals = qs.map(q => q[col.key]).filter(v => v > 0);
    best[col.key] = col.dir === 'asc' ? Math.min(...vals) : Math.max(...vals);
  }

  // Which quantizer indices (0-based) are selected?
  const selQuantIdxs = selected.filter(s => s > 0).map(s => s - 1);

  // Header row with sortable columns
  let html = '<table class="metrics-table"><tr><th>Quantizer</th>';
  for (const col of METRIC_COLS) {
    const arrow = col.dir === 'asc' ? ' \u2193' : ' \u2191';
    const sorted = metricSort.key === col.key;
    const sortArrow = sorted ? (metricSort.asc ? ' \u25B2' : ' \u25BC') : '';
    const cls = sorted ? ' class="sorted"' : '';
    html += `<th${cls} style="cursor:pointer" onclick="sortMetrics('${col.key}')">${col.label}${arrow}${sortArrow}</th>`;
  }
  html += '</tr>';

  // Data rows in sort order
  for (const i of indices) {
    const q = qs[i];
    const active = selQuantIdxs.includes(i) ? ' active' : '';
    html += `<tr class="${active}" onclick="toggleChoice(${i + 1})" style="cursor:pointer">`;
    html += `<td>${q.name}${q.note ? ' *' : ''}</td>`;
    for (const col of METRIC_COLS) {
      const v = q[col.key];
      const isBest = v === best[col.key] ? ' best' : '';
      html += `<td class="${isBest}">${col.fmt(v)}</td>`;
    }
    html += '</tr>';
  }
  html += '</table>';
  section.innerHTML = html;
}

function renderSummary() {
  const section = document.getElementById('summary-section');
  const filtered = getFilteredImages();
  if (filtered.length === 0) return;

  // Collect per-quantizer averages
  const names = [];
  const totals = {};

  for (const img of filtered) {
    for (const q of img.quantizers) {
      if (!totals[q.name]) {
        totals[q.name] = { ba: 0, ss2: 0, dssim: 0, png: 0, gif: 0, ms: 0, n: 0 };
        names.push(q.name);
      }
      const t = totals[q.name];
      t.ba += q.butteraugli;
      t.ss2 += q.ssimulacra2;
      t.dssim += q.dssim;
      t.png += q.png_bytes;
      t.gif += q.gif_bytes;
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
      ms: t.ms / t.n,
    };
  });

  const bestBA = Math.min(...avgs.map(a => a.ba));
  const bestSS2 = Math.max(...avgs.map(a => a.ss2));
  const bestDSSIM = Math.min(...avgs.map(a => a.dssim));
  const bestPNG = Math.min(...avgs.map(a => a.png));
  const bestGIF = Math.min(...avgs.map(a => a.gif));
  const bestTime = Math.min(...avgs.map(a => a.ms));

  const fmtSize = (v) => v > 0 ? (v / 1024).toFixed(1) + 'K' : '-';

  let html = `<table class="summary-table">
    <tr><th>Quantizer</th><th>Avg BA ↓</th><th>Avg SS2 ↑</th><th>Avg DSSIM ↓</th><th>Avg PNG</th><th>Avg GIF</th><th>Avg ms</th></tr>`;

  for (const a of avgs) {
    html += `<tr>
      <td>${a.name}</td>
      <td class="${a.ba === bestBA ? 'best' : ''}">${a.ba.toFixed(2)}</td>
      <td class="${a.ss2 === bestSS2 ? 'best' : ''}">${a.ss2.toFixed(1)}</td>
      <td class="${a.dssim === bestDSSIM ? 'best' : ''}">${a.dssim.toFixed(5)}</td>
      <td class="${a.png === bestPNG ? 'best' : ''}">${fmtSize(a.png)}</td>
      <td class="${a.gif === bestGIF ? 'best' : ''}">${fmtSize(a.gif)}</td>
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
  const filtered = getFilteredImages();
  if (currentImageIdx < filtered.length - 1) { currentImageIdx++; render(); }
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
  if (right) right.style.clipPath = `inset(0 0 0 ${sliderPct}%)`;
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
    // 0 = original, 1-9 = quantizers
    case '0': case '1': case '2': case '3': case '4': case '5':
    case '6': case '7': case '8': case '9': {
      const choiceIdx = parseInt(e.key);
      const filtered = getFilteredImages();
      const img = filtered[currentImageIdx];
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

<div class="group-bar" id="group-bar" style="display:none"></div>

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
  <kbd>1</kbd>-<kbd>9</kbd> quantizers &nbsp;
  <kbd>f</kbd> fit &nbsp;
  <kbd>n</kbd> native 1:1 &nbsp;
  <kbd>d</kbd> diff
</div>

</body>
</html>
"##;
