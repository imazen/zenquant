//! Quantize an image and encode it as a GIF using zengif.
//!
//! Usage:
//!   cargo run --example encode_gif --release -- <input.png> [output.gif]

use enough::Unstoppable;
use zenquant::{OutputFormat, QuantizeConfig};

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

    let pixels: Vec<rgb::RGBA<u8>> = img
        .pixels()
        .map(|p| rgb::RGBA { r: p.0[0], g: p.0[1], b: p.0[2], a: p.0[3] })
        .collect();

    // Quantize with GIF-optimized settings (binary transparency, LZW sort)
    let config = QuantizeConfig::new(OutputFormat::Gif);
    let result = zenquant::quantize_rgba(&pixels, w as usize, h as usize, &config).unwrap();

    // Build zengif palette and reconstructed pixels from zenquant result
    let palette_flat: Vec<u8> = result.palette().iter().flat_map(|c| c.iter().copied()).collect();
    let gif_palette = zengif::Palette::from_rgb_bytes(&palette_flat);

    let transparent_idx = result.transparent_index();
    let reconstructed: Vec<zengif::Rgba> = result
        .indices()
        .iter()
        .map(|&idx| {
            if Some(idx) == transparent_idx {
                zengif::Rgba::TRANSPARENT
            } else {
                let c = result.palette()[idx as usize];
                zengif::Rgba { r: c[0], g: c[1], b: c[2], a: 255 }
            }
        })
        .collect();

    let frame = zengif::FrameInput::with_palette(w, h, 0, reconstructed, gif_palette);

    // Encode with zengif (quantizer config is ignored since we provide a palette)
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
