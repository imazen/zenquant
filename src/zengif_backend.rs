//! zengif integration helpers for using zenquant as a GIF quantization backend.
//!
//! Provides convenience functions to quantize images with zenquant and feed
//! the results directly to zengif's encoder via `FrameInput::with_palette`.
//!
//! # Usage
//!
//! ```rust,no_run
//! use zenquant::zengif_backend;
//! use zengif::{encode_gif, EncoderConfig, Limits, Quantizer};
//! use enough::Unstoppable;
//!
//! // Assuming you have RGBA pixels from somewhere
//! # let rgba_pixels: Vec<zengif::Rgba> = vec![];
//! # let (w, h) = (64u16, 64u16);
//!
//! // Quantize with zenquant, get a zengif-ready FrameInput
//! let frame = zengif_backend::quantize_frame(&rgba_pixels, w, h, 85, 0.5);
//! let frame = frame.expect("quantization failed");
//!
//! // Encode directly with zengif (bypasses zengif's internal quantizer)
//! let config = EncoderConfig::new()
//!     .quantizer(Quantizer::quantizr()); // needed for encoder init
//! let gif = encode_gif(vec![frame], w, h, config, Limits::default(), &Unstoppable)
//!     .unwrap();
//! ```

use alloc::vec::Vec;

use zengif::{FrameInput, Palette, Rgba};

use crate::{DitherMode, OutputFormat, QuantizeConfig, QuantizeResult};

/// Quantize RGBA pixels and return a `zengif::FrameInput` with the palette pre-set.
///
/// This bypasses zengif's internal quantization — the frame is passed through
/// directly to the GIF encoder with zenquant's optimized palette.
///
/// # Arguments
/// * `pixels` - RGBA pixels in zengif's `Rgba` format
/// * `width` - Frame width
/// * `height` - Frame height
/// * `quality` - Quantization quality (0-100)
/// * `dithering` - Dithering strength (0.0-1.0)
///
/// # Returns
/// A `FrameInput` with palette set, ready for `zengif::encode_gif()` or
/// `zengif::Encoder::add_frame()`. Returns `None` if quantization fails.
pub fn quantize_frame(
    pixels: &[Rgba],
    width: u16,
    height: u16,
    quality: u32,
    dithering: f32,
) -> Option<FrameInput> {
    quantize_frame_with_delay(pixels, width, height, quality, dithering, 0)
}

/// Like [`quantize_frame`] but with a custom delay (centiseconds).
pub fn quantize_frame_with_delay(
    pixels: &[Rgba],
    width: u16,
    height: u16,
    quality: u32,
    dithering: f32,
    delay: u16,
) -> Option<FrameInput> {
    let dither = if dithering <= 0.0 {
        DitherMode::None
    } else {
        DitherMode::Adaptive
    };

    let config = QuantizeConfig::new()
        .max_colors(256)
        .quality(quality)
        .output_format(OutputFormat::Gif)
        .dither(dither)
        .dither_strength(dithering);

    let rgba: Vec<rgb::RGBA<u8>> = pixels
        .iter()
        .map(|p| rgb::RGBA {
            r: p.r,
            g: p.g,
            b: p.b,
            a: p.a,
        })
        .collect();

    let result = crate::quantize_rgba(&rgba, width as usize, height as usize, &config).ok()?;

    Some(result_to_frame_input(&result, pixels, width, height, delay))
}

/// Convert a `QuantizeResult` into a `zengif::FrameInput` with palette.
///
/// This is useful when you want more control over the quantization config.
///
/// ```rust,no_run
/// use zenquant::{QuantizeConfig, OutputFormat};
/// use zenquant::zengif_backend;
///
/// # let pixels_rgba: Vec<rgb::RGBA<u8>> = vec![];
/// # let pixels_zengif: Vec<zengif::Rgba> = vec![];
/// # let (w, h) = (64usize, 64usize);
/// let config = QuantizeConfig::new()
///     .output_format(OutputFormat::Gif)
///     .max_colors(128);
///
/// let result = zenquant::quantize_rgba(&pixels_rgba, w, h, &config).unwrap();
/// let frame = zengif_backend::result_to_frame_input(
///     &result, &pixels_zengif, w as u16, h as u16, 10,
/// );
/// ```
pub fn result_to_frame_input(
    result: &QuantizeResult,
    original_pixels: &[Rgba],
    width: u16,
    height: u16,
    delay: u16,
) -> FrameInput {
    let palette = result.palette();
    let indices = result.indices();
    let transparent_idx = result.transparent_index();

    // Build zengif Palette
    let palette_flat: Vec<u8> = palette.iter().flat_map(|c| c.iter().copied()).collect();
    let gif_palette = Palette::from_rgb_bytes(&palette_flat);

    // Reconstruct RGBA pixels from palette + indices.
    // zengif's with_palette path will map these back to the palette losslessly.
    let reconstructed: Vec<Rgba> = indices
        .iter()
        .enumerate()
        .map(|(i, &idx)| {
            if Some(idx) == transparent_idx {
                // Preserve original transparent pixel (for frame differencing)
                if i < original_pixels.len() && original_pixels[i].a == 0 {
                    Rgba::TRANSPARENT
                } else {
                    Rgba::TRANSPARENT
                }
            } else {
                let c = palette[idx as usize];
                Rgba {
                    r: c[0],
                    g: c[1],
                    b: c[2],
                    a: 255,
                }
            }
        })
        .collect();

    FrameInput::with_palette(width, height, delay, reconstructed, gif_palette)
}

/// Quantize multiple frames and return them as `FrameInput`s with a shared palette.
///
/// All frames are quantized against a combined histogram, producing a single
/// palette shared across all frames. This is better for GIF animations than
/// per-frame quantization.
///
/// # Arguments
/// * `frames` - Slice of (pixels, delay) pairs
/// * `width` - Canvas width (all frames must match)
/// * `height` - Canvas height (all frames must match)
/// * `quality` - Quantization quality (0-100)
/// * `dithering` - Dithering strength (0.0-1.0)
pub fn quantize_animation(
    frames: &[(&[Rgba], u16)],
    width: u16,
    height: u16,
    quality: u32,
    dithering: f32,
) -> Option<Vec<FrameInput>> {
    if frames.is_empty() {
        return Some(Vec::new());
    }

    let dither = if dithering <= 0.0 {
        DitherMode::None
    } else {
        DitherMode::Adaptive
    };

    let config = QuantizeConfig::new()
        .max_colors(256)
        .quality(quality)
        .output_format(OutputFormat::Gif)
        .dither(dither)
        .dither_strength(dithering);

    // Build a shared palette from sampled frames
    let sample_count = frames.len().min(16);
    let step = frames.len() / sample_count;
    let mut all_rgba: Vec<rgb::RGBA<u8>> = Vec::new();
    for i in 0..sample_count {
        let (pixels, _) = frames[i * step];
        for p in pixels {
            all_rgba.push(rgb::RGBA {
                r: p.r,
                g: p.g,
                b: p.b,
                a: p.a,
            });
        }
    }

    // Quantize combined pixels (no dithering) to get a shared palette
    let palette_config = config.clone().dither(DitherMode::None);
    let combined_w = width as usize * sample_count;
    let combined_h = height as usize;
    if all_rgba.len() != combined_w * combined_h {
        return None;
    }
    let palette_result =
        crate::quantize_rgba(&all_rgba, combined_w, combined_h, &palette_config).ok()?;

    // Now remap each frame using this palette (with dithering)
    let shared_palette = palette_result.palette();
    let centroids: Vec<crate::oklab::OKLab> = shared_palette
        .iter()
        .map(|c| crate::oklab::srgb_to_oklab(c[0], c[1], c[2]))
        .collect();

    let has_transparent = frames
        .iter()
        .any(|(pixels, _)| pixels.iter().any(|p| p.a == 0));
    let pal = crate::palette::Palette::from_centroids_sorted(
        centroids,
        has_transparent,
        crate::palette::PaletteSortStrategy::DeltaMinimize,
    );

    let palette_flat: Vec<u8> = pal
        .entries()
        .iter()
        .flat_map(|c| c.iter().copied())
        .collect();
    let gif_palette = Palette::from_rgb_bytes(&palette_flat);

    let mut result_frames = Vec::with_capacity(frames.len());
    for &(pixels, delay) in frames {
        // Remap with dithering
        let rgba: Vec<rgb::RGBA<u8>> = pixels
            .iter()
            .map(|p| rgb::RGBA {
                r: p.r,
                g: p.g,
                b: p.b,
                a: p.a,
            })
            .collect();

        let weights =
            crate::masking::compute_masking_weights_rgba(&rgba, width as usize, height as usize);

        let indices = crate::dither::dither_image_rgba(
            &rgba,
            width as usize,
            height as usize,
            &weights,
            &pal,
            dither,
            crate::remap::RunPriority::Balanced,
            dithering,
        );

        // Reconstruct RGBA for zengif
        let reconstructed: Vec<Rgba> = indices
            .iter()
            .enumerate()
            .map(|(i, &idx)| {
                if Some(idx) == pal.transparent_index() && i < pixels.len() && pixels[i].a == 0 {
                    Rgba::TRANSPARENT
                } else {
                    let c = pal.entries()[idx as usize];
                    Rgba {
                        r: c[0],
                        g: c[1],
                        b: c[2],
                        a: 255,
                    }
                }
            })
            .collect();

        result_frames.push(FrameInput::with_palette(
            width,
            height,
            delay,
            reconstructed,
            gif_palette.clone(),
        ));
    }

    Some(result_frames)
}
