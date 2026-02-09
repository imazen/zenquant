//! Compare zenquant vs imagequant vs quantizr on real images.
//!
//! Measures: Butteraugli score, SSIMULACRA2 score, OKLab MSE, average run length,
//!           deflate-compressed index size, and quantization time.
//!
//! Usage:
//!   cargo run --example compare --release -- [image_dir] [max_images]
//!
//! Defaults to CID22-512/training corpus.

use butteraugli::ButteraugliParams;
use fast_ssim2::compute_ssimulacra2;
use flate2::Compression;
use flate2::write::DeflateEncoder;
use imgref::ImgVec;
use rgb::RGB8;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use zenquant::_internals::average_run_length;
use zenquant::{OutputFormat, QuantizeConfig};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let image_dir = args
        .get(1)
        .map(|s| s.as_str())
        .unwrap_or("/home/lilith/work/codec-corpus/CID22/CID22-512/training");
    let max_images: usize = args
        .get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(usize::MAX);

    let mut paths: Vec<PathBuf> = std::fs::read_dir(image_dir)
        .unwrap_or_else(|e| panic!("cannot read {image_dir}: {e}"))
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
        eprintln!("No images found in {image_dir}");
        return;
    }

    eprintln!(
        "Comparing quantizers on {} images from {image_dir}",
        paths.len()
    );

    // Accumulators for averages
    let mut totals = [Totals::default(); 3]; // zenquant, imagequant, quantizr
    let names = ["zenquant", "imagequant", "quantizr"];

    // Print header
    println!(
        "{:<36} {:>7} {:>7} {:>7}  {:>7} {:>7} {:>7}  {:>7} {:>7} {:>7}  {:>8} {:>8} {:>8}  {:>7} {:>7} {:>7}",
        "Image",
        "zq_ba",
        "iq_ba",
        "qr_ba",
        "zq_ss2",
        "iq_ss2",
        "qr_ss2",
        "zq_run",
        "iq_run",
        "qr_run",
        "zq_defl",
        "iq_defl",
        "qr_defl",
        "zq_ms",
        "iq_ms",
        "qr_ms",
    );
    println!("{}", "-".repeat(175));

    for path in &paths {
        let img = match image::open(path) {
            Ok(img) => img.to_rgb8(),
            Err(e) => {
                eprintln!("Skipping {}: {e}", path.display());
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

        // RGB8 pixels for metric computation
        let ref_rgb: Vec<RGB8> = pixels.iter().map(|p| RGB8::new(p.r, p.g, p.b)).collect();

        let name = path.file_stem().unwrap_or_default().to_string_lossy();

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
        let zq_stats = compute_stats(&ref_rgb, zq.palette(), zq.indices(), width, height);

        // --- imagequant ---
        let t0 = Instant::now();
        let (iq_pal, iq_idx) = run_imagequant(&pixels, width, height);
        let iq_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let iq_stats = compute_stats(&ref_rgb, &iq_pal, &iq_idx, width, height);

        // --- quantizr ---
        let t0 = Instant::now();
        let (qr_pal, qr_idx) = run_quantizr(&pixels, width, height);
        let qr_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let qr_stats = compute_stats(&ref_rgb, &qr_pal, &qr_idx, width, height);

        println!(
            "{:<36} {:>7.3} {:>7.3} {:>7.3}  {:>7.2} {:>7.2} {:>7.2}  {:>7.1} {:>7.1} {:>7.1}  {:>8} {:>8} {:>8}  {:>7.1} {:>7.1} {:>7.1}",
            &name[..name.len().min(36)],
            zq_stats.butteraugli,
            iq_stats.butteraugli,
            qr_stats.butteraugli,
            zq_stats.ssimulacra2,
            iq_stats.ssimulacra2,
            qr_stats.ssimulacra2,
            zq_stats.avg_run,
            iq_stats.avg_run,
            qr_stats.avg_run,
            zq_stats.deflate_size,
            iq_stats.deflate_size,
            qr_stats.deflate_size,
            zq_ms,
            iq_ms,
            qr_ms,
        );

        totals[0].add(&zq_stats, zq_ms);
        totals[1].add(&iq_stats, iq_ms);
        totals[2].add(&qr_stats, qr_ms);
    }

    let n = totals[0].count as f64;
    if n > 0.0 {
        println!("{}", "=".repeat(175));
        println!(
            "{:<36} {:>7.3} {:>7.3} {:>7.3}  {:>7.2} {:>7.2} {:>7.2}  {:>7.1} {:>7.1} {:>7.1}  {:>8.0} {:>8.0} {:>8.0}  {:>7.1} {:>7.1} {:>7.1}",
            "AVERAGE",
            totals[0].butteraugli / n,
            totals[1].butteraugli / n,
            totals[2].butteraugli / n,
            totals[0].ssimulacra2 / n,
            totals[1].ssimulacra2 / n,
            totals[2].ssimulacra2 / n,
            totals[0].avg_run / n,
            totals[1].avg_run / n,
            totals[2].avg_run / n,
            totals[0].deflate_size / n,
            totals[1].deflate_size / n,
            totals[2].deflate_size / n,
            totals[0].time_ms / n,
            totals[1].time_ms / n,
            totals[2].time_ms / n,
        );

        // Summary comparison
        eprintln!("\n--- Summary (Butteraugli: lower=better, SSIMULACRA2: higher=better) ---");
        for i in 0..3 {
            eprintln!(
                "  {:<12}  BA: {:.3}  SS2: {:.2}  runs: {:.1}  deflate: {:.0}  time: {:.1}ms",
                names[i],
                totals[i].butteraugli / n,
                totals[i].ssimulacra2 / n,
                totals[i].avg_run / n,
                totals[i].deflate_size / n,
                totals[i].time_ms / n,
            );
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct Totals {
    butteraugli: f64,
    ssimulacra2: f64,
    avg_run: f64,
    deflate_size: f64,
    time_ms: f64,
    count: u32,
}

impl Totals {
    fn add(&mut self, stats: &Stats, time_ms: f64) {
        self.butteraugli += stats.butteraugli;
        self.ssimulacra2 += stats.ssimulacra2;
        self.avg_run += stats.avg_run as f64;
        self.deflate_size += stats.deflate_size as f64;
        self.time_ms += time_ms;
        self.count += 1;
    }
}

#[derive(Debug)]
struct Stats {
    butteraugli: f64,
    ssimulacra2: f64,
    avg_run: f32,
    deflate_size: usize,
}

fn compute_stats(
    ref_rgb: &[RGB8],
    palette: &[[u8; 3]],
    indices: &[u8],
    width: usize,
    height: usize,
) -> Stats {
    // Reconstruct quantized image as RGB8 pixels
    let test_rgb: Vec<RGB8> = indices
        .iter()
        .map(|&idx| {
            let c = palette[idx as usize];
            RGB8::new(c[0], c[1], c[2])
        })
        .collect();

    // Butteraugli (ImgRef<RGB8> API)
    let ref_img = ImgVec::new(ref_rgb.to_vec(), width, height);
    let test_img_ba = ImgVec::new(test_rgb.clone(), width, height);
    let butteraugli = butteraugli::butteraugli(
        ref_img.as_ref(),
        test_img_ba.as_ref(),
        &ButteraugliParams::default(),
    )
    .map(|r| r.score)
    .unwrap_or(f64::NAN);

    // SSIMULACRA2 (ImgRef<[u8; 3]> API)
    let ref_pixels: Vec<[u8; 3]> = ref_rgb.iter().map(|p| [p.r, p.g, p.b]).collect();
    let test_pixels: Vec<[u8; 3]> = test_rgb.iter().map(|p| [p.r, p.g, p.b]).collect();
    let ref_img_ss = ImgVec::new(ref_pixels, width, height);
    let test_img_ss = ImgVec::new(test_pixels, width, height);
    let ssimulacra2 =
        compute_ssimulacra2(ref_img_ss.as_ref(), test_img_ss.as_ref()).unwrap_or(f64::NAN);

    let avg_run = average_run_length(indices);
    let deflate_size = deflate_compress(indices);

    Stats {
        butteraugli,
        ssimulacra2,
        avg_run,
        deflate_size,
    }
}

fn deflate_compress(data: &[u8]) -> usize {
    let mut encoder = DeflateEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(data).unwrap();
    encoder.finish().unwrap().len()
}

// --- imagequant comparison ---

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

// --- quantizr comparison ---

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
