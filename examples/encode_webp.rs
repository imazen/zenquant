//! Quantize an image and encode it as a lossless WebP using zenwebp.
//!
//! Usage:
//!   cargo run --example encode_webp --release -- <input.png> [output.webp]

use zenquant::{OutputFormat, QuantizeConfig};
use zenwebp::{EncodeRequest, LosslessConfig, PixelLayout};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let input = args.get(1).expect("usage: encode_webp <input.png> [output.webp]");
    let output = args
        .get(2)
        .cloned()
        .unwrap_or_else(|| input.replace(".png", ".webp").replace(".jpg", ".webp"));

    // Load image as RGB
    let img = image::open(input).unwrap().to_rgb8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let pixels: Vec<rgb::RGB<u8>> = img
        .pixels()
        .map(|p| rgb::RGB { r: p.0[0], g: p.0[1], b: p.0[2] })
        .collect();

    // Quantize with WebP-optimized settings
    let config = QuantizeConfig::new(OutputFormat::WebpLossless);
    let result = zenquant::quantize(&pixels, w, h, &config).unwrap();

    // Reconstruct RGBA from palette + indices for zenwebp
    let palette = result.palette();
    let mut rgba_bytes: Vec<u8> = Vec::with_capacity(w * h * 4);
    for &idx in result.indices() {
        let c = palette[idx as usize];
        rgba_bytes.extend_from_slice(&[c[0], c[1], c[2], 255]);
    }

    // Encode as lossless WebP
    let lossless_config = LosslessConfig::new();
    let webp_data = EncodeRequest::lossless(
        &lossless_config,
        &rgba_bytes,
        PixelLayout::Rgba8,
        w as u32,
        h as u32,
    )
    .encode()
    .unwrap();

    std::fs::write(&output, &webp_data).unwrap();
    eprintln!(
        "{input} ({w}x{h}) → {output} ({} bytes, {} colors)",
        webp_data.len(),
        result.palette_len()
    );
}
