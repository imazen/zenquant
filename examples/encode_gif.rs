//! Quantize an image and encode it as a GIF using zengif.
//!
//! Requires the `zengif-backend` feature:
//!   cargo run --example encode_gif --release --features zengif-backend -- <input.png> [output.gif]

use enough::Unstoppable;
use zenquant::zengif_backend;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let input = args.get(1).expect("usage: encode_gif <input.png> [output.gif]");
    let output = args
        .get(2)
        .cloned()
        .unwrap_or_else(|| input.replace(".png", ".gif").replace(".jpg", ".gif"));

    // Load image as RGBA (GIF needs transparency info)
    let img = image::open(input).unwrap().to_rgba8();
    let (w, h) = (img.width() as u16, img.height() as u16);
    let pixels: Vec<zengif::Rgba> = img
        .pixels()
        .map(|p| zengif::Rgba { r: p.0[0], g: p.0[1], b: p.0[2], a: p.0[3] })
        .collect();

    // Quantize with GIF-optimized settings (binary transparency, LZW sort)
    let frame = zengif_backend::quantize_frame(&pixels, w, h, 85, 0.35).unwrap();

    // Encode with zengif
    let config = zengif::EncoderConfig::new().quantizer(zengif::Quantizer::quantizr());
    let gif_data = zengif::encode_gif(
        vec![frame],
        w,
        h,
        config,
        zengif::Limits::default(),
        &Unstoppable,
    )
    .unwrap();

    std::fs::write(&output, &gif_data).unwrap();
    eprintln!("{input} ({w}x{h}) → {output} ({} bytes)", gif_data.len());
}
