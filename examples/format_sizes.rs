//! Compare file sizes across PNG, GIF, WebP for all three quantizers.

use zenquant::{OutputFormat, Quality, QuantizeConfig};

fn main() {
    let image_dir = "/home/lilith/work/codec-corpus/CID22/CID22-512/validation";

    let mut paths: Vec<std::path::PathBuf> = std::fs::read_dir(image_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|ext| ext == "png"))
        .collect();
    paths.sort();
    let step = paths.len() / 5;
    let selected: Vec<_> = paths.iter().step_by(step).take(5).collect();

    println!(
        "{:<12} | {:>18} | {:>18} | {:>18} | {:>18}",
        "Image", "zq q=60", "zq q=85", "imagequant", "quantizr"
    );
    println!(
        "{:<12} | {:>5} {:>5} {:>5} | {:>5} {:>5} {:>5} | {:>5} {:>5} {:>5} | {:>5} {:>5} {:>5}",
        "", "png", "gif", "webp", "png", "gif", "webp", "png", "gif", "webp", "png", "gif", "webp"
    );
    println!("{}", "-".repeat(90));

    let mut totals = [[0usize; 3]; 4];

    for path in &selected {
        let stem = path.file_stem().unwrap().to_str().unwrap();
        let img = image::open(path).unwrap().to_rgb8();
        let w = img.width() as usize;
        let h = img.height() as usize;
        let pixels: Vec<rgb::RGB<u8>> = img
            .pixels()
            .map(|p| rgb::RGB { r: p.0[0], g: p.0[1], b: p.0[2] })
            .collect();

        eprint!("{stem}...");

        // zenquant balanced — quantize once per FORMAT for proper sort strategy
        let s60 = {
            let png_cfg = QuantizeConfig::new(OutputFormat::Png).quality(Quality::Balanced);
            let gif_cfg = QuantizeConfig::new(OutputFormat::Gif).quality(Quality::Balanced);
            let webp_cfg = QuantizeConfig::new(OutputFormat::WebpLossless).quality(Quality::Balanced);
            let png_r = zenquant::quantize(&pixels, w, h, &png_cfg).unwrap();
            let gif_r = zenquant::quantize(&pixels, w, h, &gif_cfg).unwrap();
            let webp_r = zenquant::quantize(&pixels, w, h, &webp_cfg).unwrap();
            [
                encode_indexed_png(png_r.palette(), png_r.indices(), w, h),
                encode_gif_from_result(gif_r.palette(), gif_r.indices(), w, h, &format!("{stem}_zq60")),
                encode_webp_from_result(webp_r.palette(), webp_r.indices(), w, h, &format!("{stem}_zq60")),
            ]
        };

        // zenquant best
        let s85 = {
            let png_cfg = QuantizeConfig::new(OutputFormat::Png);
            let gif_cfg = QuantizeConfig::new(OutputFormat::Gif);
            let webp_cfg = QuantizeConfig::new(OutputFormat::WebpLossless);
            let png_r = zenquant::quantize(&pixels, w, h, &png_cfg).unwrap();
            let gif_r = zenquant::quantize(&pixels, w, h, &gif_cfg).unwrap();
            let webp_r = zenquant::quantize(&pixels, w, h, &webp_cfg).unwrap();
            [
                encode_indexed_png(png_r.palette(), png_r.indices(), w, h),
                encode_gif_from_result(gif_r.palette(), gif_r.indices(), w, h, &format!("{stem}_zq85")),
                encode_webp_from_result(webp_r.palette(), webp_r.indices(), w, h, &format!("{stem}_zq85")),
            ]
        };

        // imagequant (same palette for all formats — no format-specific sorting)
        let (iq_pal, iq_idx) = run_imagequant(&pixels, w, h);
        let siq = measure_all_4(&iq_pal, &iq_idx, w, h, &format!("{stem}_iq"));

        // quantizr
        let (qr_pal, qr_idx) = run_quantizr(&pixels, w, h);
        let sqr = measure_all_4(&qr_pal, &qr_idx, w, h, &format!("{stem}_qr"));

        eprintln!(" done");

        println!(
            "{:<12} | {:>5} {:>5} {:>5} | {:>5} {:>5} {:>5} | {:>5} {:>5} {:>5} | {:>5} {:>5} {:>5}",
            stem,
            kb(s60[0]), kb(s60[1]), kb(s60[2]),
            kb(s85[0]), kb(s85[1]), kb(s85[2]),
            kb(siq[0]), kb(siq[1]), kb(siq[2]),
            kb(sqr[0]), kb(sqr[1]), kb(sqr[2]),
        );

        for (i, s) in [s60, s85, siq, sqr].iter().enumerate() {
            for j in 0..3 {
                totals[i][j] += s[j];
            }
        }
    }

    println!("{}", "-".repeat(90));
    println!(
        "{:<12} | {:>5} {:>5} {:>5} | {:>5} {:>5} {:>5} | {:>5} {:>5} {:>5} | {:>5} {:>5} {:>5}",
        "AVERAGE",
        kb(totals[0][0] / 5), kb(totals[0][1] / 5), kb(totals[0][2] / 5),
        kb(totals[1][0] / 5), kb(totals[1][1] / 5), kb(totals[1][2] / 5),
        kb(totals[2][0] / 5), kb(totals[2][1] / 5), kb(totals[2][2] / 5),
        kb(totals[3][0] / 5), kb(totals[3][1] / 5), kb(totals[3][2] / 5),
    );
}

fn kb(bytes: usize) -> String {
    format!("{}K", bytes / 1024)
}

fn encode_indexed_png(palette: &[[u8; 3]], indices: &[u8], w: usize, h: usize) -> usize {
    let mut buf = Vec::new();
    let mut encoder = png::Encoder::new(&mut buf, w as u32, h as u32);
    encoder.set_color(png::ColorType::Indexed);
    encoder.set_depth(png::BitDepth::Eight);
    let plte: Vec<u8> = palette.iter().flat_map(|c| [c[0], c[1], c[2]]).collect();
    encoder.set_palette(plte);
    let mut writer = encoder.write_header().unwrap();
    writer.write_image_data(indices).unwrap();
    drop(writer);
    buf.len()
}

fn encode_gif_from_result(palette: &[[u8; 3]], indices: &[u8], w: usize, h: usize, tag: &str) -> usize {
    let tmp_png = format!("/tmp/zq_{tag}_gif.png");
    let tmp_gif = format!("/tmp/zq_{tag}.gif");
    write_indexed_png_file(palette, indices, w, h, &tmp_png);
    let _ = std::process::Command::new("convert").args([&tmp_png, &tmp_gif]).output();
    let size = std::fs::metadata(&tmp_gif).map(|m| m.len() as usize).unwrap_or(0);
    let _ = std::fs::remove_file(&tmp_png);
    let _ = std::fs::remove_file(&tmp_gif);
    size
}

fn encode_webp_from_result(palette: &[[u8; 3]], indices: &[u8], w: usize, h: usize, tag: &str) -> usize {
    let tmp_png = format!("/tmp/zq_{tag}_webp.png");
    let tmp_webp = format!("/tmp/zq_{tag}.webp");
    write_indexed_png_file(palette, indices, w, h, &tmp_png);
    let _ = std::process::Command::new("convert")
        .args([&tmp_png, "-define", "webp:lossless=true", &tmp_webp])
        .output();
    let size = std::fs::metadata(&tmp_webp).map(|m| m.len() as usize).unwrap_or(0);
    let _ = std::fs::remove_file(&tmp_png);
    let _ = std::fs::remove_file(&tmp_webp);
    size
}

fn write_indexed_png_file(palette: &[[u8; 3]], indices: &[u8], w: usize, h: usize, path: &str) {
    let mut buf = Vec::new();
    let mut encoder = png::Encoder::new(&mut buf, w as u32, h as u32);
    encoder.set_color(png::ColorType::Indexed);
    encoder.set_depth(png::BitDepth::Eight);
    let plte: Vec<u8> = palette.iter().flat_map(|c| [c[0], c[1], c[2]]).collect();
    encoder.set_palette(plte);
    let mut writer = encoder.write_header().unwrap();
    writer.write_image_data(indices).unwrap();
    drop(writer);
    std::fs::write(path, &buf).unwrap();
}

/// For imagequant/quantizr: same palette for all formats (no format-specific sorting).
fn measure_all_4(palette: &[[u8; 4]], indices: &[u8], w: usize, h: usize, tag: &str) -> [usize; 3] {
    let pal3: Vec<[u8; 3]> = palette.iter().map(|c| [c[0], c[1], c[2]]).collect();
    [
        encode_indexed_png(&pal3, indices, w, h),
        encode_gif_from_result(&pal3, indices, w, h, tag),
        encode_webp_from_result(&pal3, indices, w, h, tag),
    ]
}

fn run_imagequant(pixels: &[rgb::RGB<u8>], width: usize, height: usize) -> (Vec<[u8; 4]>, Vec<u8>) {
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
    (pal.iter().map(|c| [c.r, c.g, c.b, c.a]).collect(), idx)
}

fn run_quantizr(pixels: &[rgb::RGB<u8>], width: usize, height: usize) -> (Vec<[u8; 4]>, Vec<u8>) {
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
            .map(|c| [c.r, c.g, c.b, c.a])
            .collect(),
        idx,
    )
}
