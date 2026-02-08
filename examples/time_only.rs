//! Pure timing — no quality metrics.
//!
//! Usage:
//!   cargo run --example time_only --release -- [image_dir] [max_images]

use std::time::Instant;
use zenquant::{DitherMode, QuantizeConfig, RunPriority};

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

    let images: Vec<(Vec<rgb::RGB<u8>>, usize, usize)> = paths
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
            Some((px, w, h))
        })
        .collect();

    let configs = [
        (
            "q=30",
            QuantizeConfig::new()
                .quality(30)
                .dither(DitherMode::Adaptive)
                .run_priority(RunPriority::Balanced),
        ),
        (
            "q=85",
            QuantizeConfig::new()
                .quality(85)
                .dither(DitherMode::Adaptive)
                .run_priority(RunPriority::Balanced),
        ),
    ];

    println!("{:<8} {:>8}", "Preset", "avg_ms");
    println!("{}", "-".repeat(18));

    for (name, config) in &configs {
        let t = Instant::now();
        for (pixels, width, height) in &images {
            let _ = zenquant::quantize(pixels, *width, *height, config).unwrap();
        }
        let total_ms = t.elapsed().as_secs_f64() * 1000.0;
        println!("{:<8} {:>8.1}", name, total_ms / images.len() as f64);
    }

    // imagequant
    {
        let t = Instant::now();
        for (pixels, width, height) in &images {
            let rgba: Vec<imagequant::RGBA> = pixels
                .iter()
                .map(|p| imagequant::RGBA::new(p.r, p.g, p.b, 255))
                .collect();
            let mut attr = imagequant::Attributes::new();
            attr.set_quality(0, 80).unwrap();
            let mut img = attr.new_image(rgba, *width, *height, 0.0).unwrap();
            let mut result = attr.quantize(&mut img).unwrap();
            result.set_dithering_level(0.5).unwrap();
            let _ = result.remapped(&mut img).unwrap();
        }
        let total_ms = t.elapsed().as_secs_f64() * 1000.0;
        println!("{:<8} {:>8.1}", "iq", total_ms / images.len() as f64);
    }

    // quantizr
    {
        let t = Instant::now();
        for (pixels, width, height) in &images {
            let pixel_bytes: Vec<u8> = pixels.iter().flat_map(|p| [p.r, p.g, p.b, 255u8]).collect();
            let image = quantizr::Image::new(&pixel_bytes, *width, *height).unwrap();
            let mut options = quantizr::Options::default();
            options.set_max_colors(256).unwrap();
            let mut result = quantizr::QuantizeResult::quantize(&image, &options);
            result.set_dithering_level(0.5).unwrap();
            let mut idx = vec![0u8; *width * *height];
            let _ = result.remap_image(&image, &mut idx).unwrap();
        }
        let total_ms = t.elapsed().as_secs_f64() * 1000.0;
        println!("{:<8} {:>8.1}", "qr", total_ms / images.len() as f64);
    }
}
