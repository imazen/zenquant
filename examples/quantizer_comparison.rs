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
    "zq-best-q-d70",
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
                        let should_recalc = recalc_all || recalc.iter().any(|r| r == qname);
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
    let html = generate_html(&report, &output_dir);
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
        let parse_usize = |key: &str| -> Option<usize> { parse_f64(key).map(|v| v as usize) };

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
            Some((r.palette().to_vec(), r.indices().to_vec(), ""))
        }
        "zq-best-q-d70" => {
            let cfg = QuantizeConfig::new(OutputFormat::Png)
                .max_colors(256)
                ._dither_strength(0.7)
                ._run_priority_quality();
            let r = zenquant::quantize(pixels, width, height, &cfg).unwrap();
            Some((
                r.palette().to_vec(),
                r.indices().to_vec(),
                "d0.7 quality-runs",
            ))
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

fn generate_html(report: &ReportData, output_dir: &Path) -> String {
    // Copy image-compare.js web component to output directory
    let component_src = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("efficient-ui/image-compare.js");
    let component_dst = output_dir.join("image-compare.js");
    std::fs::copy(&component_src, &component_dst).unwrap_or_else(|e| {
        panic!(
            "Failed to copy image-compare.js from {} to {}: {}",
            component_src.display(),
            component_dst.display(),
            e
        )
    });

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
    // Outputs the ImageCompareConfig format expected by <image-compare>
    let mut out = String::with_capacity(4096);
    out.push_str("{\"title\":\"");
    out.push_str(&json_escape(&report.label));
    out.push_str("\",\"mode\":\"slider\",\"images\":[");

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
        out.push_str(",\"variants\":[");

        // First variant is always "original" with no stats
        out.push_str("{\"name\":\"original\",\"url\":\"");
        out.push_str(&json_escape(&img.name));
        out.push_str("/original.png\",\"stats\":{}}");

        for q in &img.quantizers {
            out.push(',');
            out.push_str("{\"name\":\"");
            out.push_str(&json_escape(&q.name));
            out.push_str("\",\"url\":\"");
            out.push_str(&json_escape(&img.name));
            out.push('/');
            out.push_str(&json_escape(&q.name));
            out.push_str(".png\",\"note\":\"");
            out.push_str(&json_escape(q.note));
            out.push_str("\",\"stats\":{");
            out.push_str("\"butteraugli\":");
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
            out.push_str("}}");
        }
        out.push_str("]}");
    }

    // Column definitions
    out.push_str("],\"columns\":[");
    out.push_str("{\"key\":\"butteraugli\",\"label\":\"BA\",\"direction\":\"asc\"},");
    out.push_str("{\"key\":\"ssimulacra2\",\"label\":\"SS2\",\"direction\":\"desc\"},");
    out.push_str("{\"key\":\"dssim\",\"label\":\"DSSIM\",\"direction\":\"asc\"},");
    out.push_str("{\"key\":\"png_bytes\",\"label\":\"PNG\",\"direction\":\"asc\"},");
    out.push_str("{\"key\":\"gif_bytes\",\"label\":\"GIF\",\"direction\":\"asc\"},");
    out.push_str("{\"key\":\"time_ms\",\"label\":\"ms\",\"direction\":\"asc\"}");
    out.push_str("],\"showKeyLegend\":true}");
    out
}

const HTML_TEMPLATE: &str = include_str!("quantizer_comparison.html");
