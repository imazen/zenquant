//! Compare quantize() vs build_palette()+remap() quality.
//!
//! Tests whether the shared-palette path loses quality for single frames.
//!
//! Usage:
//!   cargo run --example compare_paths --release -- [image_dir] [max_images]

use butteraugli::ButteraugliParams;
use fast_ssim2::compute_ssimulacra2;
use imgref::ImgVec;
use rgb::RGB8;
use std::path::PathBuf;
use std::time::Instant;

use zenquant::{DitherMode, ImgRef, OutputFormat, QuantizeConfig, RunPriority};

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
        "Comparing quantize() vs build_palette()+remap() on {} images",
        paths.len()
    );

    let config = QuantizeConfig::new()
        .max_colors(256)
        .quality(85)
        .dither(DitherMode::Adaptive)
        .run_priority(RunPriority::Balanced)
        .output_format(OutputFormat::Png);

    println!(
        "{:<36} {:>8} {:>8}  {:>8} {:>8}  {:>8} {:>8}",
        "Image", "q_ba", "bp_ba", "q_ss2", "bp_ss2", "q_ms", "bp_ms",
    );
    println!("{}", "-".repeat(100));

    let mut sum_q_ba = 0.0f64;
    let mut sum_bp_ba = 0.0f64;
    let mut sum_q_ss2 = 0.0f64;
    let mut sum_bp_ss2 = 0.0f64;
    let mut sum_q_ms = 0.0f64;
    let mut sum_bp_ms = 0.0f64;
    let mut count = 0u32;

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

        let ref_rgb: Vec<RGB8> = pixels.iter().map(|p| RGB8::new(p.r, p.g, p.b)).collect();
        let name = path.file_stem().unwrap_or_default().to_string_lossy();

        // --- quantize() ---
        let t0 = Instant::now();
        let q_result = zenquant::quantize(&pixels, width, height, &config).unwrap();
        let q_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let (q_ba, q_ss2) = score(&ref_rgb, q_result.palette(), q_result.indices(), width, height);

        // --- build_palette() + remap() ---
        let t0 = Instant::now();
        let frame = ImgRef::new(&pixels, width, height);
        let shared = zenquant::build_palette(&[frame], &config).unwrap();
        let bp_result = shared.remap(&pixels, width, height, &config).unwrap();
        let bp_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let (bp_ba, bp_ss2) = score(&ref_rgb, bp_result.palette(), bp_result.indices(), width, height);

        println!(
            "{:<36} {:>8.3} {:>8.3}  {:>8.2} {:>8.2}  {:>8.1} {:>8.1}",
            &name[..name.len().min(36)],
            q_ba, bp_ba, q_ss2, bp_ss2, q_ms, bp_ms,
        );

        sum_q_ba += q_ba;
        sum_bp_ba += bp_ba;
        sum_q_ss2 += q_ss2;
        sum_bp_ss2 += bp_ss2;
        sum_q_ms += q_ms;
        sum_bp_ms += bp_ms;
        count += 1;
    }

    if count > 0 {
        let n = count as f64;
        println!("{}", "=".repeat(100));
        println!(
            "{:<36} {:>8.3} {:>8.3}  {:>8.2} {:>8.2}  {:>8.1} {:>8.1}",
            "AVERAGE",
            sum_q_ba / n, sum_bp_ba / n,
            sum_q_ss2 / n, sum_bp_ss2 / n,
            sum_q_ms / n, sum_bp_ms / n,
        );
        println!();
        println!("q = quantize(),  bp = build_palette() + remap()");
    }
}

fn score(
    ref_rgb: &[RGB8],
    palette: &[[u8; 3]],
    indices: &[u8],
    width: usize,
    height: usize,
) -> (f64, f64) {
    let test_rgb: Vec<RGB8> = indices
        .iter()
        .map(|&idx| {
            let c = palette[idx as usize];
            RGB8::new(c[0], c[1], c[2])
        })
        .collect();

    let ref_img = ImgVec::new(ref_rgb.to_vec(), width, height);
    let test_img = ImgVec::new(test_rgb.clone(), width, height);
    let ba = butteraugli::butteraugli(
        ref_img.as_ref(),
        test_img.as_ref(),
        &ButteraugliParams::default(),
    )
    .map(|r| r.score)
    .unwrap_or(f64::NAN);

    let ref_pixels: Vec<[u8; 3]> = ref_rgb.iter().map(|p| [p.r, p.g, p.b]).collect();
    let test_pixels: Vec<[u8; 3]> = test_rgb.iter().map(|p| [p.r, p.g, p.b]).collect();
    let ref_img_ss = ImgVec::new(ref_pixels, width, height);
    let test_img_ss = ImgVec::new(test_pixels, width, height);
    let ss2 = compute_ssimulacra2(ref_img_ss.as_ref(), test_img_ss.as_ref()).unwrap_or(f64::NAN);

    (ba, ss2)
}
