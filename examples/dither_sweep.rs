//! Dither strength sweep — find the highest dither that doesn't hurt DSSIM.
//!
//! Usage:
//!   cargo run --example dither_sweep --release -- [corpus] [max_images]
//!   corpus: cid22, clic2025, gb82-sc (default: cid22)

use butteraugli::ButteraugliParams;
use dssim_core::Dssim;
use fast_ssim2::compute_ssimulacra2;
use imgref::ImgVec;
use rgb::RGB8;
use std::path::PathBuf;

use zenquant::{OutputFormat, QuantizeConfig};

const DITHER_VALUES: &[f32] = &[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];

fn corpus_path(name: &str) -> &'static str {
    match name {
        "cid22" => "/home/lilith/work/codec-corpus/CID22/CID22-512/training",
        "clic2025" => "/home/lilith/work/codec-corpus/clic2025/final-test",
        "gb82-sc" => "/home/lilith/work/codec-corpus/gb82-sc",
        _ => panic!("Unknown corpus: {name}"),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let corpus = args.get(1).map(|s| s.as_str()).unwrap_or("cid22");
    let max_images: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(5);
    let image_dir = corpus_path(corpus);

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

    eprintln!(
        "Dither sweep on {} images from {corpus} ({image_dir})",
        paths.len()
    );

    // Pre-load images
    let images: Vec<(String, Vec<rgb::RGB<u8>>, Vec<RGB8>, usize, usize)> = paths
        .iter()
        .filter_map(|path| {
            let name = path.file_stem()?.to_string_lossy().to_string();
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
            Some((name, pixels, ref_rgb, width, height))
        })
        .collect();

    // Header
    println!(
        "{:<8} {:>8} {:>8} {:>10}",
        "dither", "BA", "SS2", "DSSIM"
    );
    println!("{}", "-".repeat(38));

    for &dither in DITHER_VALUES {
        let mut total_ba = 0.0f64;
        let mut total_ss2 = 0.0f64;
        let mut total_dssim = 0.0f64;
        let mut count = 0u32;

        for (name, pixels, ref_rgb, width, height) in &images {
            let width = *width;
            let height = *height;

            let config = QuantizeConfig::new(OutputFormat::Png)
                .max_colors(256)
                ._dither_strength(dither);

            let result = zenquant::quantize(pixels, width, height, &config).unwrap();

            let test_rgb: Vec<RGB8> = result
                .indices()
                .iter()
                .map(|&idx| {
                    let c = result.palette()[idx as usize];
                    RGB8::new(c[0], c[1], c[2])
                })
                .collect();

            // Butteraugli
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
            let ss2 = compute_ssimulacra2(ref_img_ss.as_ref(), test_img_ss.as_ref())
                .unwrap_or(f64::NAN);

            // DSSIM
            let d = Dssim::new();
            let ref_dssim = d.create_image_rgb(ref_rgb, width, height).unwrap();
            let test_dssim = d.create_image_rgb(&test_rgb, width, height).unwrap();
            let (dssim_val, _) = d.compare(&ref_dssim, &test_dssim);
            let dssim: f64 = dssim_val.into();

            if ba.is_finite() && ss2.is_finite() {
                total_ba += ba;
                total_ss2 += ss2;
                total_dssim += dssim;
                count += 1;
            }

            eprint!(".");
        }
        eprintln!();

        let n = count as f64;
        println!(
            "{:<8.1} {:>8.3} {:>8.2} {:>10.6}",
            dither,
            total_ba / n,
            total_ss2 / n,
            total_dssim / n,
        );
    }
}
