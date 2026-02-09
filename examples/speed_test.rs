//! Speed + quality comparison at different quality presets.
//!
//! Usage:
//!   cargo run --example speed_test --release -- [image_dir] [max_images]

use butteraugli::ButteraugliParams;
use fast_ssim2::compute_ssimulacra2;
use flate2::Compression;
use flate2::write::DeflateEncoder;
use imgref::ImgVec;
use rgb::RGB8;
use std::io::Write;
use std::time::Instant;

use zenquant::_internals::average_run_length;
use zenquant::{OutputFormat, Quality, QuantizeConfig};

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

    let mut paths: Vec<std::path::PathBuf> = std::fs::read_dir(image_dir)
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

    // Pre-load images
    let images: Vec<(Vec<rgb::RGB<u8>>, Vec<RGB8>, usize, usize)> = paths
        .iter()
        .filter_map(|path| {
            let img = image::open(path).ok()?.to_rgb8();
            let w = img.width() as usize;
            let h = img.height() as usize;
            let px: Vec<rgb::RGB<u8>> = img
                .pixels()
                .map(|p| rgb::RGB {
                    r: p.0[0],
                    g: p.0[1],
                    b: p.0[2],
                })
                .collect();
            let rf: Vec<RGB8> = px.iter().map(|p| RGB8::new(p.r, p.g, p.b)).collect();
            Some((px, rf, w, h))
        })
        .collect();

    eprintln!("Testing {} images from {image_dir}", images.len());

    // Presets to test
    let presets: Vec<(&str, QuantizeConfig)> = vec![
        ("zq fast", QuantizeConfig::new(OutputFormat::Png).quality(Quality::Fast)),
        ("zq balanced", QuantizeConfig::new(OutputFormat::Png).quality(Quality::Balanced)),
        ("zq best", QuantizeConfig::new(OutputFormat::Png)),
    ];

    println!(
        "{:<20} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "Preset", "BA", "SS2", "deflate", "runs", "ms"
    );
    println!("{}", "-".repeat(72));

    for (name, config) in &presets {
        let mut total_ba = 0.0f64;
        let mut total_ss2 = 0.0f64;
        let mut total_deflate = 0u64;
        let mut total_runs = 0.0f64;
        let mut total_ms = 0.0f64;
        let mut count = 0u32;

        for (pixels, ref_rgb, width, height) in &images {
            let t = Instant::now();
            let result = zenquant::quantize(pixels, *width, *height, config).unwrap();
            let ms = t.elapsed().as_secs_f64() * 1000.0;

            let test_rgb: Vec<RGB8> = result
                .indices()
                .iter()
                .map(|&idx| {
                    let c = result.palette()[idx as usize];
                    RGB8::new(c[0], c[1], c[2])
                })
                .collect();

            let ref_img = ImgVec::new(ref_rgb.clone(), *width, *height);
            let test_img = ImgVec::new(test_rgb.clone(), *width, *height);
            let ba = butteraugli::butteraugli(
                ref_img.as_ref(),
                test_img.as_ref(),
                &ButteraugliParams::default(),
            )
            .map(|r| r.score)
            .unwrap_or(f64::NAN);

            let ref_px: Vec<[u8; 3]> = ref_rgb.iter().map(|p| [p.r, p.g, p.b]).collect();
            let test_px: Vec<[u8; 3]> = test_rgb.iter().map(|p| [p.r, p.g, p.b]).collect();
            let ss2 = compute_ssimulacra2(
                ImgVec::new(ref_px, *width, *height).as_ref(),
                ImgVec::new(test_px, *width, *height).as_ref(),
            )
            .unwrap_or(f64::NAN);

            let avg_run = average_run_length(result.indices());
            let deflate = deflate_compress(result.indices());

            if ba.is_finite() && ss2.is_finite() {
                total_ba += ba;
                total_ss2 += ss2;
                total_deflate += deflate as u64;
                total_runs += avg_run as f64;
                total_ms += ms;
                count += 1;
            }
        }

        let n = count as f64;
        println!(
            "{:<20} {:>8.3} {:>8.2} {:>8.0} {:>8.1} {:>8.1}",
            name,
            total_ba / n,
            total_ss2 / n,
            total_deflate as f64 / n,
            total_runs / n,
            total_ms / n,
        );
    }

    // Also run imagequant and quantizr for comparison
    println!();
    {
        let mut total_ba = 0.0f64;
        let mut total_ss2 = 0.0f64;
        let mut total_deflate = 0u64;
        let mut total_runs = 0.0f64;
        let mut total_ms = 0.0f64;
        let mut count = 0u32;

        for (pixels, ref_rgb, width, height) in &images {
            let t = Instant::now();
            let (pal, idx) = run_imagequant(pixels, *width, *height);
            let ms = t.elapsed().as_secs_f64() * 1000.0;

            let test_rgb: Vec<RGB8> = idx
                .iter()
                .map(|&i| {
                    let c = pal[i as usize];
                    RGB8::new(c[0], c[1], c[2])
                })
                .collect();

            let ref_img = ImgVec::new(ref_rgb.clone(), *width, *height);
            let test_img = ImgVec::new(test_rgb.clone(), *width, *height);
            let ba = butteraugli::butteraugli(
                ref_img.as_ref(),
                test_img.as_ref(),
                &ButteraugliParams::default(),
            )
            .map(|r| r.score)
            .unwrap_or(f64::NAN);

            let ref_px: Vec<[u8; 3]> = ref_rgb.iter().map(|p| [p.r, p.g, p.b]).collect();
            let test_px: Vec<[u8; 3]> = test_rgb.iter().map(|p| [p.r, p.g, p.b]).collect();
            let ss2 = compute_ssimulacra2(
                ImgVec::new(ref_px, *width, *height).as_ref(),
                ImgVec::new(test_px, *width, *height).as_ref(),
            )
            .unwrap_or(f64::NAN);

            if ba.is_finite() && ss2.is_finite() {
                total_ba += ba;
                total_ss2 += ss2;
                total_deflate += deflate_compress(&idx) as u64;
                total_runs += average_run_length(&idx) as f64;
                total_ms += ms;
                count += 1;
            }
        }
        let n = count as f64;
        println!(
            "{:<20} {:>8.3} {:>8.2} {:>8.0} {:>8.1} {:>8.1}",
            "imagequant",
            total_ba / n,
            total_ss2 / n,
            total_deflate as f64 / n,
            total_runs / n,
            total_ms / n,
        );
    }
    {
        let mut total_ba = 0.0f64;
        let mut total_ss2 = 0.0f64;
        let mut total_deflate = 0u64;
        let mut total_runs = 0.0f64;
        let mut total_ms = 0.0f64;
        let mut count = 0u32;

        for (pixels, ref_rgb, width, height) in &images {
            let t = Instant::now();
            let (pal, idx) = run_quantizr(pixels, *width, *height);
            let ms = t.elapsed().as_secs_f64() * 1000.0;

            let test_rgb: Vec<RGB8> = idx
                .iter()
                .map(|&i| {
                    let c = pal[i as usize];
                    RGB8::new(c[0], c[1], c[2])
                })
                .collect();

            let ref_img = ImgVec::new(ref_rgb.clone(), *width, *height);
            let test_img = ImgVec::new(test_rgb.clone(), *width, *height);
            let ba = butteraugli::butteraugli(
                ref_img.as_ref(),
                test_img.as_ref(),
                &ButteraugliParams::default(),
            )
            .map(|r| r.score)
            .unwrap_or(f64::NAN);

            let ref_px: Vec<[u8; 3]> = ref_rgb.iter().map(|p| [p.r, p.g, p.b]).collect();
            let test_px: Vec<[u8; 3]> = test_rgb.iter().map(|p| [p.r, p.g, p.b]).collect();
            let ss2 = compute_ssimulacra2(
                ImgVec::new(ref_px, *width, *height).as_ref(),
                ImgVec::new(test_px, *width, *height).as_ref(),
            )
            .unwrap_or(f64::NAN);

            if ba.is_finite() && ss2.is_finite() {
                total_ba += ba;
                total_ss2 += ss2;
                total_deflate += deflate_compress(&idx) as u64;
                total_runs += average_run_length(&idx) as f64;
                total_ms += ms;
                count += 1;
            }
        }
        let n = count as f64;
        println!(
            "{:<20} {:>8.3} {:>8.2} {:>8.0} {:>8.1} {:>8.1}",
            "quantizr",
            total_ba / n,
            total_ss2 / n,
            total_deflate as f64 / n,
            total_runs / n,
            total_ms / n,
        );
    }
}

fn deflate_compress(data: &[u8]) -> usize {
    let mut encoder = DeflateEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(data).unwrap();
    encoder.finish().unwrap().len()
}

fn run_imagequant(pixels: &[rgb::RGB<u8>], width: usize, height: usize) -> (Vec<[u8; 3]>, Vec<u8>) {
    let mut attr = imagequant::Attributes::new();
    attr.set_quality(0, 80).unwrap();
    let rgba: Vec<imagequant::RGBA> = pixels
        .iter()
        .map(|p| imagequant::RGBA::new(p.r, p.g, p.b, 255))
        .collect();
    let mut img = attr.new_image(rgba, width, height, 0.0).unwrap();
    let mut result = attr.quantize(&mut img).unwrap();
    result.set_dithering_level(0.5).unwrap();
    let (pal, idx) = result.remapped(&mut img).unwrap();
    (pal.iter().map(|c| [c.r, c.g, c.b]).collect(), idx)
}

fn run_quantizr(pixels: &[rgb::RGB<u8>], width: usize, height: usize) -> (Vec<[u8; 3]>, Vec<u8>) {
    let pixel_bytes: Vec<u8> = pixels.iter().flat_map(|p| [p.r, p.g, p.b, 255u8]).collect();
    let image = quantizr::Image::new(&pixel_bytes, width, height).unwrap();
    let mut options = quantizr::Options::default();
    options.set_max_colors(256).unwrap();
    let mut result = quantizr::QuantizeResult::quantize(&image, &options);
    result.set_dithering_level(0.5).unwrap();
    let mut idx = vec![0u8; width * height];
    result.remap_image(&image, &mut idx).unwrap();
    let pal = result.get_palette();
    (
        pal.entries[..pal.count as usize]
            .iter()
            .map(|c| [c.r, c.g, c.b])
            .collect(),
        idx,
    )
}
