extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

use crate::oklab::{OKLab, srgb_to_oklab};
use crate::palette::Palette;
use crate::remap::RunPriority;

/// Dithering mode.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DitherMode {
    /// No dithering — nearest color only.
    None,
    /// AQ-adaptive dithering — modulated by masking weights.
    Adaptive,
}

/// Apply Floyd-Steinberg error diffusion with AQ modulation.
///
/// `dither_strength` controls the fraction of quantization error to diffuse
/// (0.0 = no dithering, 0.5 = half-strength, 1.0 = standard Floyd-Steinberg).
#[allow(clippy::too_many_arguments)]
pub fn dither_image(
    pixels: &[rgb::RGB<u8>],
    width: usize,
    height: usize,
    weights: &[f32],
    palette: &Palette,
    mode: DitherMode,
    run_priority: RunPriority,
    dither_strength: f32,
) -> Vec<u8> {
    if mode == DitherMode::None {
        return simple_remap(pixels, palette);
    }

    let run_bias = run_priority.bias();

    // Working buffer in OKLab (we add error diffusion to this)
    let mut lab_buf: Vec<[f32; 3]> = pixels
        .iter()
        .map(|p| {
            let lab = srgb_to_oklab(p.r, p.g, p.b);
            [lab.l, lab.a, lab.b]
        })
        .collect();

    let mut indices = vec![0u8; pixels.len()];
    let mut prev_index: Option<u8> = None;

    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            let current = OKLab::new(lab_buf[idx][0], lab_buf[idx][1], lab_buf[idx][2]);

            // Find nearest palette entry using seed + neighbor refinement
            let p = pixels[idx];
            let seed = palette.nearest_cached(p.r, p.g, p.b);
            let best = palette.nearest_seeded(current, seed);
            let best_lab = palette.entries_oklab()[best as usize];

            // Run-aware dither suppression: if we'd extend a run, and we're in
            // compression mode, consider using the previous index
            let chosen = if run_bias > 0.0 {
                if let Some(prev) = prev_index {
                    let prev_dist = current.distance_sq(palette.entries_oklab()[prev as usize]);
                    let best_dist = current.distance_sq(best_lab);
                    let w = weights[idx];
                    let threshold = run_bias * best_dist * 2.0 * (1.1 - w);

                    if prev_dist < best_dist + threshold {
                        prev
                    } else {
                        best
                    }
                } else {
                    best
                }
            } else {
                best
            };

            indices[idx] = chosen;
            prev_index = Some(chosen);

            let chosen_lab = palette.entries_oklab()[chosen as usize];

            // Compute quantization error, scaled by dither strength
            let err_l = (current.l - chosen_lab.l) * dither_strength;
            let err_a = (current.a - chosen_lab.a) * dither_strength;
            let err_b = (current.b - chosen_lab.b) * dither_strength;

            // Diffuse error with AQ modulation.
            // Error is generated at reduced strength but *received* with the target pixel's AQ weight.
            let diffuse_err = |buf: &mut [[f32; 3]],
                               target_idx: usize,
                               fraction: f32,
                               el: f32,
                               ea: f32,
                               eb: f32,
                               w_mod: f32| {
                let scale = fraction * w_mod;
                buf[target_idx][0] += el * scale;
                buf[target_idx][1] += ea * scale;
                buf[target_idx][2] += eb * scale;
            };

            let adaptive = mode == DitherMode::Adaptive;

            // Floyd-Steinberg kernel: right 7/16, bottom-left 3/16, bottom 5/16, bottom-right 1/16
            if x + 1 < width {
                let w = if adaptive { weights[idx + 1] } else { 1.0 };
                diffuse_err(&mut lab_buf, idx + 1, 7.0 / 16.0, err_l, err_a, err_b, w);
            }
            if y + 1 < height {
                if x > 0 {
                    let ti = (y + 1) * width + (x - 1);
                    let w = if adaptive { weights[ti] } else { 1.0 };
                    diffuse_err(&mut lab_buf, ti, 3.0 / 16.0, err_l, err_a, err_b, w);
                }
                {
                    let ti = (y + 1) * width + x;
                    let w = if adaptive { weights[ti] } else { 1.0 };
                    diffuse_err(&mut lab_buf, ti, 5.0 / 16.0, err_l, err_a, err_b, w);
                }
                if x + 1 < width {
                    let ti = (y + 1) * width + (x + 1);
                    let w = if adaptive { weights[ti] } else { 1.0 };
                    diffuse_err(&mut lab_buf, ti, 1.0 / 16.0, err_l, err_a, err_b, w);
                }
            }
        }
    }

    indices
}

/// Apply dithering for RGBA images. Transparent pixels pass through unchanged.
#[allow(clippy::too_many_arguments)]
pub fn dither_image_rgba(
    pixels: &[rgb::RGBA<u8>],
    width: usize,
    height: usize,
    weights: &[f32],
    palette: &Palette,
    mode: DitherMode,
    run_priority: RunPriority,
    dither_strength: f32,
) -> Vec<u8> {
    let transparent_idx = palette.transparent_index().unwrap_or(0);

    if mode == DitherMode::None {
        return simple_remap_rgba(pixels, palette, transparent_idx);
    }

    let run_bias = run_priority.bias();

    let mut lab_buf: Vec<[f32; 3]> = pixels
        .iter()
        .map(|p| {
            let lab = srgb_to_oklab(p.r, p.g, p.b);
            [lab.l, lab.a, lab.b]
        })
        .collect();

    let mut indices = vec![0u8; pixels.len()];
    let mut prev_index: Option<u8> = None;

    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;

            if pixels[idx].a == 0 {
                indices[idx] = transparent_idx;
                prev_index = Some(transparent_idx);
                continue;
            }

            let current = OKLab::new(lab_buf[idx][0], lab_buf[idx][1], lab_buf[idx][2]);
            let p = pixels[idx];
            let seed = palette.nearest_cached(p.r, p.g, p.b);
            let best = palette.nearest_seeded(current, seed);
            let best_lab = palette.entries_oklab()[best as usize];

            let chosen = if run_bias > 0.0 {
                if let Some(prev) = prev_index {
                    if prev != transparent_idx {
                        let prev_dist = current.distance_sq(palette.entries_oklab()[prev as usize]);
                        let best_dist = current.distance_sq(best_lab);
                        let w = weights[idx];
                        let threshold = run_bias * best_dist * 2.0 * (1.1 - w);
                        if prev_dist < best_dist + threshold {
                            prev
                        } else {
                            best
                        }
                    } else {
                        best
                    }
                } else {
                    best
                }
            } else {
                best
            };

            indices[idx] = chosen;
            prev_index = Some(chosen);

            let chosen_lab = palette.entries_oklab()[chosen as usize];
            let err_l = (current.l - chosen_lab.l) * dither_strength;
            let err_a = (current.a - chosen_lab.a) * dither_strength;
            let err_b = (current.b - chosen_lab.b) * dither_strength;

            let adaptive = mode == DitherMode::Adaptive;

            let diffuse_err = |buf: &mut [[f32; 3]],
                               ti: usize,
                               fraction: f32,
                               el: f32,
                               ea: f32,
                               eb: f32,
                               w_mod: f32| {
                let scale = fraction * w_mod;
                buf[ti][0] += el * scale;
                buf[ti][1] += ea * scale;
                buf[ti][2] += eb * scale;
            };

            if x + 1 < width && pixels[idx + 1].a > 0 {
                let w = if adaptive { weights[idx + 1] } else { 1.0 };
                diffuse_err(&mut lab_buf, idx + 1, 7.0 / 16.0, err_l, err_a, err_b, w);
            }
            if y + 1 < height {
                if x > 0 {
                    let ti = (y + 1) * width + (x - 1);
                    if pixels[ti].a > 0 {
                        let w = if adaptive { weights[ti] } else { 1.0 };
                        diffuse_err(&mut lab_buf, ti, 3.0 / 16.0, err_l, err_a, err_b, w);
                    }
                }
                {
                    let ti = (y + 1) * width + x;
                    if pixels[ti].a > 0 {
                        let w = if adaptive { weights[ti] } else { 1.0 };
                        diffuse_err(&mut lab_buf, ti, 5.0 / 16.0, err_l, err_a, err_b, w);
                    }
                }
                if x + 1 < width {
                    let ti = (y + 1) * width + (x + 1);
                    if pixels[ti].a > 0 {
                        let w = if adaptive { weights[ti] } else { 1.0 };
                        diffuse_err(&mut lab_buf, ti, 1.0 / 16.0, err_l, err_a, err_b, w);
                    }
                }
            }
        }
    }

    indices
}

/// Alpha-aware dithering: error diffusion includes alpha channel.
/// Used when the palette has per-entry alpha values (PNG, WebP, JXL, Generic).
#[allow(clippy::too_many_arguments)]
pub fn dither_image_rgba_alpha(
    pixels: &[rgb::RGBA<u8>],
    width: usize,
    height: usize,
    weights: &[f32],
    palette: &Palette,
    mode: DitherMode,
    run_priority: RunPriority,
    dither_strength: f32,
) -> Vec<u8> {
    let transparent_idx = palette.transparent_index().unwrap_or(0);

    if mode == DitherMode::None {
        return simple_remap_rgba_alpha(pixels, palette, transparent_idx);
    }

    let run_bias = run_priority.bias();

    // Working buffer: [L, a, b, alpha]
    let mut lab_buf: Vec<[f32; 4]> = pixels
        .iter()
        .map(|p| {
            let lab = srgb_to_oklab(p.r, p.g, p.b);
            let alpha = p.a as f32 / 255.0;
            [lab.l, lab.a, lab.b, alpha]
        })
        .collect();

    let mut indices = vec![0u8; pixels.len()];
    let mut prev_index: Option<u8> = None;

    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            let current_alpha = lab_buf[idx][3];

            // Fully transparent
            if pixels[idx].a == 0 {
                indices[idx] = transparent_idx;
                prev_index = Some(transparent_idx);
                continue;
            }

            let current = OKLab::new(lab_buf[idx][0], lab_buf[idx][1], lab_buf[idx][2]);
            let best = palette.nearest_with_alpha(current, current_alpha);
            let best_lab = palette.entries_oklab()[best as usize];

            let chosen = if run_bias > 0.0 {
                if let Some(prev) = prev_index {
                    if prev != transparent_idx {
                        let prev_lab = palette.entries_oklab()[prev as usize];
                        let prev_dist = current.distance_sq(prev_lab);
                        let best_dist = current.distance_sq(best_lab);
                        let w = weights[idx];
                        let threshold = run_bias * best_dist * 2.0 * (1.1 - w);
                        if prev_dist < best_dist + threshold {
                            prev
                        } else {
                            best
                        }
                    } else {
                        best
                    }
                } else {
                    best
                }
            } else {
                best
            };

            indices[idx] = chosen;
            prev_index = Some(chosen);

            let chosen_lab = palette.entries_oklab()[chosen as usize];
            let chosen_alpha = palette.entries_rgba()[chosen as usize][3] as f32 / 255.0;

            // Compute quantization error in color + alpha
            let err_l = (current.l - chosen_lab.l) * dither_strength;
            let err_a = (current.a - chosen_lab.a) * dither_strength;
            let err_b = (current.b - chosen_lab.b) * dither_strength;
            let err_al = (current_alpha - chosen_alpha) * dither_strength;

            let adaptive = mode == DitherMode::Adaptive;

            let diffuse_err = |buf: &mut [[f32; 4]],
                               ti: usize,
                               fraction: f32,
                               el: f32,
                               ea: f32,
                               eb: f32,
                               eal: f32,
                               w_mod: f32| {
                let scale = fraction * w_mod;
                buf[ti][0] += el * scale;
                buf[ti][1] += ea * scale;
                buf[ti][2] += eb * scale;
                buf[ti][3] += eal * scale;
            };

            // Don't diffuse error to fully-transparent neighbors
            if x + 1 < width && pixels[idx + 1].a > 0 {
                let w = if adaptive { weights[idx + 1] } else { 1.0 };
                diffuse_err(
                    &mut lab_buf,
                    idx + 1,
                    7.0 / 16.0,
                    err_l,
                    err_a,
                    err_b,
                    err_al,
                    w,
                );
            }
            if y + 1 < height {
                if x > 0 {
                    let ti = (y + 1) * width + (x - 1);
                    if pixels[ti].a > 0 {
                        let w = if adaptive { weights[ti] } else { 1.0 };
                        diffuse_err(&mut lab_buf, ti, 3.0 / 16.0, err_l, err_a, err_b, err_al, w);
                    }
                }
                {
                    let ti = (y + 1) * width + x;
                    if pixels[ti].a > 0 {
                        let w = if adaptive { weights[ti] } else { 1.0 };
                        diffuse_err(&mut lab_buf, ti, 5.0 / 16.0, err_l, err_a, err_b, err_al, w);
                    }
                }
                if x + 1 < width {
                    let ti = (y + 1) * width + (x + 1);
                    if pixels[ti].a > 0 {
                        let w = if adaptive { weights[ti] } else { 1.0 };
                        diffuse_err(&mut lab_buf, ti, 1.0 / 16.0, err_l, err_a, err_b, err_al, w);
                    }
                }
            }
        }
    }

    indices
}

/// Simple nearest-color remap for alpha-aware palettes.
fn simple_remap_rgba_alpha(
    pixels: &[rgb::RGBA<u8>],
    palette: &Palette,
    transparent_idx: u8,
) -> Vec<u8> {
    pixels
        .iter()
        .map(|p| {
            if p.a == 0 {
                transparent_idx
            } else {
                let lab = srgb_to_oklab(p.r, p.g, p.b);
                let alpha = p.a as f32 / 255.0;
                palette.nearest_with_alpha(lab, alpha)
            }
        })
        .collect()
}

/// Simple nearest-color remap without dithering.
/// Uses the sRGB nearest-neighbor cache if available.
pub(crate) fn simple_remap(pixels: &[rgb::RGB<u8>], palette: &Palette) -> Vec<u8> {
    if palette.has_nn_cache() {
        pixels
            .iter()
            .map(|p| palette.nearest_cached(p.r, p.g, p.b))
            .collect()
    } else {
        pixels
            .iter()
            .map(|p| {
                let lab = srgb_to_oklab(p.r, p.g, p.b);
                palette.nearest(lab)
            })
            .collect()
    }
}

pub(crate) fn simple_remap_rgba(
    pixels: &[rgb::RGBA<u8>],
    palette: &Palette,
    transparent_idx: u8,
) -> Vec<u8> {
    pixels
        .iter()
        .map(|p| {
            if p.a == 0 {
                transparent_idx
            } else {
                let lab = srgb_to_oklab(p.r, p.g, p.b);
                palette.nearest(lab)
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_gradient(width: usize) -> Vec<rgb::RGB<u8>> {
        (0..width)
            .map(|x| {
                let v = (x * 255 / width.max(1)) as u8;
                rgb::RGB { r: v, g: v, b: v }
            })
            .collect()
    }

    fn make_test_palette() -> Palette {
        let centroids = vec![
            srgb_to_oklab(0, 0, 0),
            srgb_to_oklab(85, 85, 85),
            srgb_to_oklab(170, 170, 170),
            srgb_to_oklab(255, 255, 255),
        ];
        Palette::from_centroids(centroids, false)
    }

    #[test]
    fn no_dither_produces_valid_indices() {
        let palette = make_test_palette();
        let pixels = make_gradient(64);
        let weights = vec![1.0; 64];
        let indices = dither_image(
            &pixels,
            64,
            1,
            &weights,
            &palette,
            DitherMode::None,
            RunPriority::Quality,
            0.5,
        );
        assert_eq!(indices.len(), 64);
        for &idx in &indices {
            assert!((idx as usize) < palette.len());
        }
    }

    #[test]
    fn adaptive_dither_produces_valid_indices() {
        let palette = make_test_palette();
        let width = 16;
        let height = 16;
        let mut pixels = Vec::with_capacity(width * height);
        for y in 0..height {
            for x in 0..width {
                let v = ((x + y) * 255 / (width + height)) as u8;
                pixels.push(rgb::RGB { r: v, g: v, b: v });
            }
        }
        let weights = vec![0.5; width * height];
        let indices = dither_image(
            &pixels,
            width,
            height,
            &weights,
            &palette,
            DitherMode::Adaptive,
            RunPriority::Balanced,
            0.5,
        );
        assert_eq!(indices.len(), width * height);
        for &idx in &indices {
            assert!((idx as usize) < palette.len());
        }
    }
}
