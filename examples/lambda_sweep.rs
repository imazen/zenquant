//! Lambda sweep for Viterbi tuning.
//!
//! Usage:
//!   cargo run --example lambda_sweep --release -- [image_dir] [max_images]

use butteraugli::ButteraugliParams;
use fast_ssim2::compute_ssimulacra2;
use flate2::Compression;
use flate2::write::DeflateEncoder;
use imgref::ImgVec;
use rgb::RGB8;
use std::io::Write;
use std::path::PathBuf;

use zenquant::_internals::average_run_length;
use zenquant::{DitherMode, QuantizeConfig, RunPriority};

const LAMBDAS: &[f32] = &[0.0, 0.005, 0.01, 0.02];

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let image_dir = args
        .get(1)
        .map(|s| s.as_str())
        .unwrap_or("/home/lilith/work/codec-corpus/CID22/CID22-512/validation");
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

    eprintln!("Lambda sweep on {} images from {image_dir}", paths.len());

    // Pre-load images
    let images: Vec<(Vec<rgb::RGB<u8>>, Vec<RGB8>, usize, usize)> = paths
        .iter()
        .filter_map(|path| {
            let img = image::open(path).ok()?.to_rgb8();
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
            Some((pixels, ref_rgb, width, height))
        })
        .collect();

    // Header
    println!(
        "{:<10} {:>8} {:>8} {:>8} {:>8}",
        "lambda", "BA", "SS2", "deflate", "runs"
    );
    println!("{}", "-".repeat(44));

    for &lambda in LAMBDAS {
        let mut total_ba = 0.0f64;
        let mut total_ss2 = 0.0f64;
        let mut total_deflate = 0u64;
        let mut total_runs = 0.0f64;
        let mut count = 0u32;

        for (pixels, ref_rgb, width, height) in &images {
            let width = *width;
            let height = *height;

            let config = QuantizeConfig::new()
                .max_colors(256)
                .quality(85)
                .dither(DitherMode::Adaptive)
                .run_priority(RunPriority::Balanced)
                .viterbi_lambda(lambda);

            let result = zenquant::quantize(pixels, width, height, &config).unwrap();

            // Butteraugli
            let test_rgb: Vec<RGB8> = result
                .indices()
                .iter()
                .map(|&idx| {
                    let c = result.palette()[idx as usize];
                    RGB8::new(c[0], c[1], c[2])
                })
                .collect();

            let ref_img = ImgVec::new(ref_rgb.clone(), width, height);
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
            let ss2 =
                compute_ssimulacra2(ref_img_ss.as_ref(), test_img_ss.as_ref()).unwrap_or(f64::NAN);

            let avg_run = average_run_length(result.indices());
            let deflate = deflate_compress(result.indices());

            if ba.is_finite() && ss2.is_finite() {
                total_ba += ba;
                total_ss2 += ss2;
                total_deflate += deflate as u64;
                total_runs += avg_run as f64;
                count += 1;
            }
        }

        let n = count as f64;
        println!(
            "{:<10.4} {:>8.3} {:>8.2} {:>8.0} {:>8.1}",
            lambda,
            total_ba / n,
            total_ss2 / n,
            total_deflate as f64 / n,
            total_runs / n,
        );
    }
}

fn deflate_compress(data: &[u8]) -> usize {
    let mut encoder = DeflateEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(data).unwrap();
    encoder.finish().unwrap().len()
}
