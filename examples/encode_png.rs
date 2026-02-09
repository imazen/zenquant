//! Quantize an image and encode it as an indexed PNG.
//!
//! Usage:
//!   cargo run --example encode_png --release -- <input.png> [output.png]

use zenquant::{OutputFormat, QuantizeConfig};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let input = args.get(1).expect("usage: encode_png <input.png> [output.png]");
    let output = args
        .get(2)
        .cloned()
        .unwrap_or_else(|| input.replace(".png", "_q.png").replace(".jpg", "_q.png"));

    // Load image as RGB
    let img = image::open(input).unwrap().to_rgb8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let pixels: Vec<rgb::RGB<u8>> = img
        .pixels()
        .map(|p| rgb::RGB { r: p.0[0], g: p.0[1], b: p.0[2] })
        .collect();

    // Quantize with PNG-optimized settings
    let config = QuantizeConfig::new(OutputFormat::Png);
    let result = zenquant::quantize(&pixels, w, h, &config).unwrap();

    // Write indexed PNG
    let file = std::fs::File::create(&output).unwrap();
    let buf = std::io::BufWriter::new(file);
    let mut encoder = png::Encoder::new(buf, w as u32, h as u32);
    encoder.set_color(png::ColorType::Indexed);
    encoder.set_depth(png::BitDepth::Eight);

    // Flatten palette to [R, G, B, R, G, B, ...]
    let palette_flat: Vec<u8> = result.palette().iter().flat_map(|c| c.iter().copied()).collect();
    encoder.set_palette(palette_flat);

    // Set tRNS chunk if any palette entries have alpha < 255
    if let Some(trns) = result.alpha_table() {
        encoder.set_trns(trns);
    }

    let mut writer = encoder.write_header().unwrap();
    writer.write_image_data(result.indices()).unwrap();
    drop(writer);

    eprintln!(
        "{input} ({w}x{h}) → {output} ({} colors)",
        result.palette_len()
    );
}
