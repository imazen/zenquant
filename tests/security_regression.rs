//! Regression tests for security-audit-2026-05-06 panic-class fixes.
//!
//! These guard against:
//!   - `remap_with_prev` / `remap_rgba_with_prev` panicking on too-short
//!     `prev_indices` or out-of-range entries.
//!   - `compute_mpe` / `compute_mpe_rgba` panicking on shape-mismatched
//!     palette / index / weight buffers.
//!   - `usize` overflow in `width * height` slipping past `validate_inputs`
//!     and downstream impls.

use zenquant::_internals::{compute_mpe, compute_mpe_rgba};
use zenquant::{ImgRef, OutputFormat, QuantizeConfig, QuantizeError};

fn solid_rgb(width: usize, height: usize) -> Vec<rgb::RGB<u8>> {
    let mut v = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            v.push(rgb::RGB::new(((x ^ y) & 0xff) as u8, 64, 192));
        }
    }
    v
}

fn solid_rgba(width: usize, height: usize) -> Vec<rgb::RGBA<u8>> {
    let mut v = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            v.push(rgb::RGBA::new(((x ^ y) & 0xff) as u8, 64, 192, 255));
        }
    }
    v
}

#[test]
fn remap_with_prev_rejects_short_buffer() {
    let w = 8;
    let h = 8;
    let pixels = solid_rgb(w, h);
    let frame = ImgRef::new(&pixels, w, h);

    let config = QuantizeConfig::new(OutputFormat::Png);
    let palette = zenquant::build_palette(&[frame], &config).unwrap();

    // Too-short prev_indices buffer — must error, not panic in dither.
    let prev = vec![0u8; w * h - 1];
    let err = palette
        .remap_with_prev(&pixels, w, h, &config, &prev)
        .expect_err("short prev_indices must error");
    assert!(matches!(err, QuantizeError::DimensionMismatch { .. }));
}

#[test]
fn remap_with_prev_rejects_oversized_buffer() {
    let w = 8;
    let h = 8;
    let pixels = solid_rgb(w, h);
    let frame = ImgRef::new(&pixels, w, h);

    let config = QuantizeConfig::new(OutputFormat::Png);
    let palette = zenquant::build_palette(&[frame], &config).unwrap();

    let prev = vec![0u8; w * h + 16];
    let err = palette
        .remap_with_prev(&pixels, w, h, &config, &prev)
        .expect_err("oversized prev_indices must error");
    assert!(matches!(err, QuantizeError::DimensionMismatch { .. }));
}

#[test]
fn remap_with_prev_rejects_out_of_range_index() {
    let w = 8;
    let h = 8;
    let pixels = solid_rgb(w, h);
    let frame = ImgRef::new(&pixels, w, h);

    let config = QuantizeConfig::new(OutputFormat::Png);
    let palette = zenquant::build_palette(&[frame], &config).unwrap();

    // Pick a value guaranteed out of range — any value at u8::MAX larger than
    // the palette length will do (palette is small for this synthetic input).
    let pal_len = palette.palette_len();
    assert!(pal_len <= 256);
    let bad = u8::MAX;
    assert!(bad as usize >= pal_len, "u8::MAX must exceed pal_len");

    let mut prev = vec![0u8; w * h];
    prev[3] = bad;

    let err = palette
        .remap_with_prev(&pixels, w, h, &config, &prev)
        .expect_err("out-of-range prev_indices entry must error");
    assert!(matches!(err, QuantizeError::InvalidIndex { .. }));
}

#[test]
fn remap_rgba_with_prev_rejects_short_buffer() {
    let w = 8;
    let h = 8;
    let pixels = solid_rgba(w, h);
    let frame = ImgRef::new(&pixels, w, h);

    let config = QuantizeConfig::new(OutputFormat::Gif);
    let palette = zenquant::build_palette_rgba(&[frame], &config).unwrap();

    let prev = vec![0u8; 1];
    let err = palette
        .remap_rgba_with_prev(&pixels, w, h, &config, &prev)
        .expect_err("short prev_indices must error");
    assert!(matches!(err, QuantizeError::DimensionMismatch { .. }));
}

#[test]
fn remap_rgba_with_prev_rejects_out_of_range_index() {
    let w = 8;
    let h = 8;
    let pixels = solid_rgba(w, h);
    let frame = ImgRef::new(&pixels, w, h);

    let config = QuantizeConfig::new(OutputFormat::Gif);
    let palette = zenquant::build_palette_rgba(&[frame], &config).unwrap();

    let pal_len = palette.palette_len();
    let bad = u8::MAX;
    assert!(bad as usize >= pal_len);

    let mut prev = vec![0u8; w * h];
    prev[7] = bad;

    let err = palette
        .remap_rgba_with_prev(&pixels, w, h, &config, &prev)
        .expect_err("out-of-range prev_indices entry must error");
    assert!(matches!(err, QuantizeError::InvalidIndex { .. }));
}

#[test]
fn remap_with_prev_accepts_valid_buffer() {
    // Sanity check: the validator does NOT reject correctly-sized in-range
    // buffers. Catches over-aggressive validation.
    let w = 4;
    let h = 4;
    let pixels = solid_rgb(w, h);
    let frame = ImgRef::new(&pixels, w, h);

    let config = QuantizeConfig::new(OutputFormat::Png);
    let palette = zenquant::build_palette(&[frame], &config).unwrap();

    let prev = vec![0u8; w * h];
    let _ = palette
        .remap_with_prev(&pixels, w, h, &config, &prev)
        .expect("valid prev_indices must succeed");
}

#[test]
fn compute_mpe_does_not_panic_on_oob_index() {
    // Out-of-range index would previously OOB-panic at palette[idx] in
    // release builds. Now clamps to the last palette entry.
    let w = 4;
    let h = 4;
    let pixels = solid_rgb(w, h);
    let palette = vec![[0u8, 0, 0], [255, 255, 255]];
    let mut indices = vec![0u8; w * h];
    indices[2] = 200; // way beyond palette.len()=2

    let _ = compute_mpe(&pixels, &palette, &indices, w, h, None);
}

#[test]
fn compute_mpe_returns_zero_on_shape_mismatch() {
    let w = 4;
    let h = 4;
    let pixels = solid_rgb(w, h);
    let palette = vec![[0u8, 0, 0], [255, 255, 255]];
    let indices = vec![0u8; 1]; // wrong length

    let r = compute_mpe(&pixels, &palette, &indices, w, h, None);
    assert_eq!(r.score, 0.0);
}

#[test]
fn compute_mpe_returns_zero_on_weights_mismatch() {
    let w = 4;
    let h = 4;
    let pixels = solid_rgb(w, h);
    let palette = vec![[0u8, 0, 0], [255, 255, 255]];
    let indices = vec![0u8; w * h];
    let weights = vec![1.0f32; 3]; // wrong length

    let r = compute_mpe(&pixels, &palette, &indices, w, h, Some(&weights));
    assert_eq!(r.score, 0.0);
}

#[test]
fn compute_mpe_rgba_does_not_panic_on_oob_index() {
    let w = 4;
    let h = 4;
    let pixels = solid_rgba(w, h);
    let palette = vec![[0u8, 0, 0, 255], [255, 255, 255, 255]];
    let mut indices = vec![0u8; w * h];
    indices[2] = 200;

    let _ = compute_mpe_rgba(&pixels, &palette, &indices, w, h, None);
}

#[test]
fn compute_mpe_rgba_returns_zero_on_shape_mismatch() {
    let w = 4;
    let h = 4;
    let pixels = solid_rgba(w, h);
    let palette = vec![[0u8, 0, 0, 255]];
    let indices = vec![0u8; 2];

    let r = compute_mpe_rgba(&pixels, &palette, &indices, w, h, None);
    assert_eq!(r.score, 0.0);
}

#[test]
fn quantize_rejects_dimension_overflow() {
    // width * height that overflows usize must error rather than wrapping
    // and slipping past the equality check (then later panicking in the
    // pipeline). On 32-bit usize this is a small product; on 64-bit it
    // requires usize::MAX-class numbers.
    let w = usize::MAX;
    let h = 2usize;
    let pixels: Vec<rgb::RGB<u8>> = Vec::new();
    let config = QuantizeConfig::new(OutputFormat::Png);
    let err =
        zenquant::quantize(&pixels, w, h, &config).expect_err("overflow must error, not wrap");
    assert!(matches!(err, QuantizeError::DimensionOverflow { .. }));
}

#[test]
fn quantize_rgba_rejects_dimension_overflow() {
    let w = usize::MAX;
    let h = 4usize;
    let pixels: Vec<rgb::RGBA<u8>> = Vec::new();
    let config = QuantizeConfig::new(OutputFormat::Gif);
    let err = zenquant::quantize_rgba(&pixels, w, h, &config).expect_err("overflow must error");
    assert!(matches!(err, QuantizeError::DimensionOverflow { .. }));
}
