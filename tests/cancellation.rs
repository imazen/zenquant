//! Cooperative-cancellation tests for the `*_with_stop` quantize entry points.
//!
//! A cancelled [`enough::Stop`] token must surface as
//! [`QuantizeError::Cancelled`] at a phase boundary, while
//! [`enough::Unstoppable`] must let the quantization run to completion.

use zenquant::{OutputFormat, Quality, QuantizeConfig, QuantizeError};

/// Build a many-color RGBA gradient so the quantizer can't take the
/// "image already fits in the palette" fast path (which would short-circuit
/// before any phase-boundary cancellation check).
fn gradient_rgba(width: usize, height: usize) -> Vec<rgb::RGBA<u8>> {
    let mut pixels = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            let r = (x * 255 / width) as u8;
            let g = (y * 255 / height) as u8;
            let b = ((x + y) * 255 / (width + height)) as u8;
            pixels.push(rgb::RGBA { r, g, b, a: 255 });
        }
    }
    pixels
}

#[test]
fn cancelled_stop_returns_cancelled_error() {
    let (width, height) = (32, 32);
    let pixels = gradient_rgba(width, height);
    // Best quality runs the full k-means + Viterbi pipeline, guaranteeing the
    // phase-boundary `stop.check()` is reached. max_colors well below the
    // gradient's unique-color count avoids the exact-palette fast path.
    let config = QuantizeConfig::new(OutputFormat::Png)
        .with_quality(Quality::Best)
        .with_max_colors(16);

    let stop = almost_enough::Stopper::cancelled();
    let result = zenquant::quantize_rgba_with_stop(&pixels, width, height, &config, &stop);

    match result {
        Err(QuantizeError::Cancelled(_)) => {}
        other => panic!("expected Err(QuantizeError::Cancelled(_)), got {other:?}"),
    }
}

#[test]
fn unstoppable_completes_ok() {
    let (width, height) = (32, 32);
    let pixels = gradient_rgba(width, height);
    let config = QuantizeConfig::new(OutputFormat::Png)
        .with_quality(Quality::Best)
        .with_max_colors(16);

    let result =
        zenquant::quantize_rgba_with_stop(&pixels, width, height, &config, &enough::Unstoppable);

    let result = result.expect("Unstoppable token must allow the quantize to complete");
    assert!(
        !result.indices().is_empty(),
        "completed quantize should produce per-pixel indices"
    );
    assert!(
        result.palette_len() >= 2,
        "completed quantize should produce a non-trivial palette"
    );
}
