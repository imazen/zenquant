//! Generate side-by-side visual comparisons of zenquant vs imagequant vs quantizr.
//!
//! Usage:
//!   cargo run --example compare_images --release -- [image_dir] [output_dir] [max_images]

use zenquant::{OutputFormat, Quality, QuantizeConfig};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let image_dir = args
        .get(1)
        .map(|s| s.as_str())
        .unwrap_or("/home/lilith/work/codec-corpus/CID22/CID22-512/validation");
    let output_dir = args
        .get(2)
        .map(|s| s.as_str())
        .unwrap_or("/mnt/v/output/zenquant/compare");
    let max_images: usize = args
        .get(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);

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
    // Pick evenly spaced images for variety
    let step = if paths.len() > max_images {
        paths.len() / max_images
    } else {
        1
    };
    let selected: Vec<_> = paths.iter().step_by(step).take(max_images).collect();

    std::fs::create_dir_all(output_dir).unwrap();

    for path in &selected {
        let stem = path.file_stem().unwrap().to_str().unwrap();
        let img = image::open(path).unwrap().to_rgb8();
        let w = img.width() as usize;
        let h = img.height() as usize;
        let pixels: Vec<rgb::RGB<u8>> = img
            .pixels()
            .map(|p| rgb::RGB {
                r: p.0[0],
                g: p.0[1],
                b: p.0[2],
            })
            .collect();

        eprintln!("Processing {stem} ({w}x{h})...");

        // zenquant balanced
        let config_bal = QuantizeConfig::new(OutputFormat::Png).quality(Quality::Balanced);
        let zq60 = zenquant::quantize(&pixels, w, h, &config_bal).unwrap();

        // zenquant best quality
        let config_best = QuantizeConfig::new(OutputFormat::Png);
        let zq85 = zenquant::quantize(&pixels, w, h, &config_best).unwrap();

        // imagequant
        let (iq_pal, iq_idx) = run_imagequant(&pixels, w, h);

        // quantizr
        let (qr_pal, qr_idx) = run_quantizr(&pixels, w, h);

        // Save individual quantized PNGs
        save_quantized(
            &zq60,
            w,
            h,
            &format!("{output_dir}/{stem}_zq60.png"),
        );
        save_quantized(
            &zq85,
            w,
            h,
            &format!("{output_dir}/{stem}_zq85.png"),
        );
        save_indexed(
            &iq_pal,
            &iq_idx,
            w,
            h,
            &format!("{output_dir}/{stem}_iq.png"),
        );
        save_indexed(
            &qr_pal,
            &qr_idx,
            w,
            h,
            &format!("{output_dir}/{stem}_qr.png"),
        );

        // Also save original
        img.save(format!("{output_dir}/{stem}_orig.png")).unwrap();
    }

    // Create montages using ImageMagick
    for path in &selected {
        let stem = path.file_stem().unwrap().to_str().unwrap();
        let montage_path = format!("{output_dir}/{stem}_montage.png");

        let status = std::process::Command::new("montage")
            .args([
                &format!("{output_dir}/{stem}_orig.png"),
                &format!("{output_dir}/{stem}_zq60.png"),
                &format!("{output_dir}/{stem}_zq85.png"),
                &format!("{output_dir}/{stem}_iq.png"),
                &format!("{output_dir}/{stem}_qr.png"),
                "-tile",
                "5x1",
                "-geometry",
                "+2+2",
                "-title",
                &format!("{stem}"),
                "-label",
                "",
                &montage_path,
            ])
            .status();

        // Label each image individually
        let _ = std::process::Command::new("montage")
            .args([
                "-label", "Original",
                &format!("{output_dir}/{stem}_orig.png"),
                "-label", "zq q=60 (62ms)",
                &format!("{output_dir}/{stem}_zq60.png"),
                "-label", "zq q=85 (125ms)",
                &format!("{output_dir}/{stem}_zq85.png"),
                "-label", "imagequant (42ms)",
                &format!("{output_dir}/{stem}_iq.png"),
                "-label", "quantizr (30ms)",
                &format!("{output_dir}/{stem}_qr.png"),
                "-tile", "5x1",
                "-geometry", "+4+4",
                "-pointsize", "14",
                &montage_path,
            ])
            .status();

        if status.is_ok() {
            eprintln!("Created montage: {montage_path}");
        }
    }
}

fn save_quantized(result: &zenquant::QuantizeResult, w: usize, h: usize, path: &str) {
    let mut imgbuf = image::RgbImage::new(w as u32, h as u32);
    for (i, &idx) in result.indices().iter().enumerate() {
        let c = result.palette()[idx as usize];
        let x = (i % w) as u32;
        let y = (i / w) as u32;
        imgbuf.put_pixel(x, y, image::Rgb([c[0], c[1], c[2]]));
    }
    imgbuf.save(path).unwrap();
}

fn save_indexed(pal: &[[u8; 3]], idx: &[u8], w: usize, h: usize, path: &str) {
    let mut imgbuf = image::RgbImage::new(w as u32, h as u32);
    for (i, &pidx) in idx.iter().enumerate() {
        let c = pal[pidx as usize];
        let x = (i % w) as u32;
        let y = (i / w) as u32;
        imgbuf.put_pixel(x, y, image::Rgb(c));
    }
    imgbuf.save(path).unwrap();
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
