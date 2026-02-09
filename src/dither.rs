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

/// Compute a per-pixel dither ratio based on local edge contrast.
///
/// Pixels near edges (high contrast to neighbors) get reduced dither strength
/// to prevent error diffusion from bleeding across color boundaries. Smooth
/// areas keep full dither strength for clean gradient rendering.
fn compute_dither_map(lab_buf: &[[f32; 3]], width: usize, height: usize) -> Vec<f32> {
    let len = width * height;
    let mut map = vec![1.0f32; len];

    // Thresholds in squared OKLab distance.
    // Below edge_low: fully smooth, ratio = 1.0
    // Above edge_high: hard edge, ratio = min_ratio
    let edge_low: f32 = 0.003;
    let edge_high: f32 = 0.05;
    let min_ratio: f32 = 0.2;
    let range = edge_high - edge_low;

    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            let [l, a, b] = lab_buf[idx];
            let mut max_dist_sq: f32 = 0.0;

            // Check 4 cardinal neighbors
            if x > 0 {
                let [nl, na, nb] = lab_buf[idx - 1];
                let d = (l - nl) * (l - nl) + (a - na) * (a - na) + (b - nb) * (b - nb);
                max_dist_sq = max_dist_sq.max(d);
            }
            if x + 1 < width {
                let [nl, na, nb] = lab_buf[idx + 1];
                let d = (l - nl) * (l - nl) + (a - na) * (a - na) + (b - nb) * (b - nb);
                max_dist_sq = max_dist_sq.max(d);
            }
            if y > 0 {
                let [nl, na, nb] = lab_buf[idx - width];
                let d = (l - nl) * (l - nl) + (a - na) * (a - na) + (b - nb) * (b - nb);
                max_dist_sq = max_dist_sq.max(d);
            }
            if y + 1 < height {
                let [nl, na, nb] = lab_buf[idx + width];
                let d = (l - nl) * (l - nl) + (a - na) * (a - na) + (b - nb) * (b - nb);
                max_dist_sq = max_dist_sq.max(d);
            }

            if max_dist_sq > edge_low {
                let t = ((max_dist_sq - edge_low) / range).min(1.0);
                map[idx] = 1.0 - t * (1.0 - min_ratio);
            }
        }
    }

    map
}

/// 4-channel variant of dither map for RGBA alpha-aware dithering.
fn compute_dither_map_4(lab_buf: &[[f32; 4]], width: usize, height: usize) -> Vec<f32> {
    let len = width * height;
    let mut map = vec![1.0f32; len];

    let edge_low: f32 = 0.003;
    let edge_high: f32 = 0.05;
    let min_ratio: f32 = 0.2;
    let range = edge_high - edge_low;

    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            let [l, a, b, _] = lab_buf[idx];
            let mut max_dist_sq: f32 = 0.0;

            if x > 0 {
                let [nl, na, nb, _] = lab_buf[idx - 1];
                let d = (l - nl) * (l - nl) + (a - na) * (a - na) + (b - nb) * (b - nb);
                max_dist_sq = max_dist_sq.max(d);
            }
            if x + 1 < width {
                let [nl, na, nb, _] = lab_buf[idx + 1];
                let d = (l - nl) * (l - nl) + (a - na) * (a - na) + (b - nb) * (b - nb);
                max_dist_sq = max_dist_sq.max(d);
            }
            if y > 0 {
                let [nl, na, nb, _] = lab_buf[idx - width];
                let d = (l - nl) * (l - nl) + (a - na) * (a - na) + (b - nb) * (b - nb);
                max_dist_sq = max_dist_sq.max(d);
            }
            if y + 1 < height {
                let [nl, na, nb, _] = lab_buf[idx + width];
                let d = (l - nl) * (l - nl) + (a - na) * (a - na) + (b - nb) * (b - nb);
                max_dist_sq = max_dist_sq.max(d);
            }

            if max_dist_sq > edge_low {
                let t = ((max_dist_sq - edge_low) / range).min(1.0);
                map[idx] = 1.0 - t * (1.0 - min_ratio);
            }
        }
    }

    map
}

/// Apply serpentine Floyd-Steinberg error diffusion with AQ modulation.
///
/// `dither_strength` controls the fraction of quantization error to diffuse
/// (0.0 = no dithering, 0.5 = half-strength, 1.0 = standard Floyd-Steinberg).
///
/// Improvements over naive Floyd-Steinberg:
/// - Edge-aware dither map: pixels near edges generate less error, preventing
///   error diffusion from bleeding across color boundaries.
/// - Serpentine scanning: alternates L→R and R→L each row, preventing
///   directional banding artifacts visible at higher dither strengths.
/// - Error magnitude reduction: when accumulated error exceeds a threshold,
///   it's damped by 75% to prevent runaway error accumulation.
/// - OKLab gamut clamping: prevents error-adjusted values from drifting
///   outside the representable color space.
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

    // High-dither features only activate above this threshold.
    // Below it, use simple F-S to preserve the tuned low-strength behavior.
    let high_dither = dither_strength > 0.4;

    // Edge-aware dither map: reduces error generation near edges.
    let dither_map = if high_dither {
        compute_dither_map(&lab_buf, width, height)
    } else {
        vec![1.0f32; pixels.len()]
    };

    // Error magnitude threshold — errors beyond this get damped.
    let max_err_sq = 0.002 * dither_strength;

    let adaptive = mode == DitherMode::Adaptive;

    for y in 0..height {
        // Serpentine: even rows L→R, odd rows R→L (high dither only)
        let forward = !high_dither || y % 2 == 0;
        let mut prev_index: Option<u8> = None;

        let x_iter: Box<dyn Iterator<Item = usize>> = if forward {
            Box::new(0..width)
        } else {
            Box::new((0..width).rev())
        };

        for x in x_iter {
            let idx = y * width + x;
            let current = OKLab::new(lab_buf[idx][0], lab_buf[idx][1], lab_buf[idx][2]);

            // Find nearest palette entry to the error-adjusted pixel
            let p = pixels[idx];
            let orig_lab = srgb_to_oklab(p.r, p.g, p.b);
            let seed = palette.nearest_cached(p.r, p.g, p.b);
            let dithered_best = palette.nearest_seeded(current, seed);

            // Undithered fallback (high dither only): if error diffusion
            // pushed us to a palette entry much farther from the *original*
            // pixel than the naive nearest match, reject it.
            let best = if high_dither {
                let undithered_best = palette.nearest_seeded(orig_lab, seed);
                if dithered_best == undithered_best {
                    dithered_best
                } else {
                    let d_dithered = orig_lab.distance_sq(
                        palette.entries_oklab()[dithered_best as usize],
                    );
                    let d_undithered = orig_lab.distance_sq(
                        palette.entries_oklab()[undithered_best as usize],
                    );
                    if d_dithered <= d_undithered * 2.0 {
                        dithered_best
                    } else {
                        undithered_best
                    }
                }
            } else {
                dithered_best
            };
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

            // Compute quantization error, scaled by dither strength and
            // edge-aware dither map. Edge pixels generate less error to
            // prevent bleeding across color boundaries.
            let scale = dither_strength * dither_map[idx];
            let mut err_l = (current.l - chosen_lab.l) * scale;
            let mut err_a = (current.a - chosen_lab.a) * scale;
            let mut err_b = (current.b - chosen_lab.b) * scale;

            // Error magnitude reduction (high dither only): damp large
            // errors to prevent runaway accumulation.
            if high_dither {
                let err_mag = err_l * err_l + err_a * err_a + err_b * err_b;
                if err_mag > max_err_sq {
                    err_l *= 0.75;
                    err_a *= 0.75;
                    err_b *= 0.75;
                }
            }

            // Diffuse error with AQ modulation and gamut clamping.
            let diffuse_err = |buf: &mut [[f32; 3]],
                               target_idx: usize,
                               fraction: f32,
                               el: f32,
                               ea: f32,
                               eb: f32,
                               w_mod: f32| {
                let f = fraction * w_mod;
                let v = &mut buf[target_idx];
                v[0] = (v[0] + el * f).clamp(0.0, 1.0);
                v[1] = (v[1] + ea * f).clamp(-0.5, 0.5);
                v[2] = (v[2] + eb * f).clamp(-0.5, 0.5);
            };

            // Serpentine Floyd-Steinberg kernel:
            //   Forward (L→R): right 7/16, bottom-left 3/16, bottom 5/16, bottom-right 1/16
            //   Reverse (R→L): left 7/16, bottom-right 3/16, bottom 5/16, bottom-left 1/16
            if forward {
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
            } else {
                if x > 0 {
                    let w = if adaptive { weights[idx - 1] } else { 1.0 };
                    diffuse_err(&mut lab_buf, idx - 1, 7.0 / 16.0, err_l, err_a, err_b, w);
                }
                if y + 1 < height {
                    if x + 1 < width {
                        let ti = (y + 1) * width + (x + 1);
                        let w = if adaptive { weights[ti] } else { 1.0 };
                        diffuse_err(&mut lab_buf, ti, 3.0 / 16.0, err_l, err_a, err_b, w);
                    }
                    {
                        let ti = (y + 1) * width + x;
                        let w = if adaptive { weights[ti] } else { 1.0 };
                        diffuse_err(&mut lab_buf, ti, 5.0 / 16.0, err_l, err_a, err_b, w);
                    }
                    if x > 0 {
                        let ti = (y + 1) * width + (x - 1);
                        let w = if adaptive { weights[ti] } else { 1.0 };
                        diffuse_err(&mut lab_buf, ti, 1.0 / 16.0, err_l, err_a, err_b, w);
                    }
                }
            }
        }
    }

    indices
}

/// Apply serpentine Floyd-Steinberg dithering for RGBA images.
/// Transparent pixels pass through unchanged.
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
    let max_err_sq = 0.002 * dither_strength;
    let adaptive = mode == DitherMode::Adaptive;
    let high_dither = dither_strength > 0.4;

    let mut lab_buf: Vec<[f32; 3]> = pixels
        .iter()
        .map(|p| {
            let lab = srgb_to_oklab(p.r, p.g, p.b);
            [lab.l, lab.a, lab.b]
        })
        .collect();

    let mut indices = vec![0u8; pixels.len()];
    let dither_map = if high_dither {
        compute_dither_map(&lab_buf, width, height)
    } else {
        vec![1.0f32; pixels.len()]
    };

    for y in 0..height {
        let forward = !high_dither || y % 2 == 0;
        let mut prev_index: Option<u8> = None;

        let x_iter: Box<dyn Iterator<Item = usize>> = if forward {
            Box::new(0..width)
        } else {
            Box::new((0..width).rev())
        };

        for x in x_iter {
            let idx = y * width + x;

            if pixels[idx].a == 0 {
                indices[idx] = transparent_idx;
                prev_index = Some(transparent_idx);
                continue;
            }

            let current = OKLab::new(lab_buf[idx][0], lab_buf[idx][1], lab_buf[idx][2]);
            let p = pixels[idx];
            let orig_lab = srgb_to_oklab(p.r, p.g, p.b);
            let seed = palette.nearest_cached(p.r, p.g, p.b);
            let dithered_best = palette.nearest_seeded(current, seed);

            let best = if high_dither {
                let undithered_best = palette.nearest_seeded(orig_lab, seed);
                if dithered_best == undithered_best {
                    dithered_best
                } else {
                    let d_dithered = orig_lab.distance_sq(
                        palette.entries_oklab()[dithered_best as usize],
                    );
                    let d_undithered = orig_lab.distance_sq(
                        palette.entries_oklab()[undithered_best as usize],
                    );
                    if d_dithered <= d_undithered * 2.0 {
                        dithered_best
                    } else {
                        undithered_best
                    }
                }
            } else {
                dithered_best
            };
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
            let scale = dither_strength * dither_map[idx];
            let mut err_l = (current.l - chosen_lab.l) * scale;
            let mut err_a = (current.a - chosen_lab.a) * scale;
            let mut err_b = (current.b - chosen_lab.b) * scale;

            if high_dither {
                let err_mag = err_l * err_l + err_a * err_a + err_b * err_b;
                if err_mag > max_err_sq {
                    err_l *= 0.75;
                    err_a *= 0.75;
                    err_b *= 0.75;
                }
            }

            let diffuse_err = |buf: &mut [[f32; 3]],
                               ti: usize,
                               fraction: f32,
                               el: f32,
                               ea: f32,
                               eb: f32,
                               w_mod: f32| {
                let f = fraction * w_mod;
                let v = &mut buf[ti];
                v[0] = (v[0] + el * f).clamp(0.0, 1.0);
                v[1] = (v[1] + ea * f).clamp(-0.5, 0.5);
                v[2] = (v[2] + eb * f).clamp(-0.5, 0.5);
            };

            let opaque_at = |px: usize| pixels[px].a > 0;

            if forward {
                if x + 1 < width && opaque_at(idx + 1) {
                    let w = if adaptive { weights[idx + 1] } else { 1.0 };
                    diffuse_err(&mut lab_buf, idx + 1, 7.0 / 16.0, err_l, err_a, err_b, w);
                }
                if y + 1 < height {
                    if x > 0 {
                        let ti = (y + 1) * width + (x - 1);
                        if opaque_at(ti) {
                            let w = if adaptive { weights[ti] } else { 1.0 };
                            diffuse_err(&mut lab_buf, ti, 3.0 / 16.0, err_l, err_a, err_b, w);
                        }
                    }
                    {
                        let ti = (y + 1) * width + x;
                        if opaque_at(ti) {
                            let w = if adaptive { weights[ti] } else { 1.0 };
                            diffuse_err(&mut lab_buf, ti, 5.0 / 16.0, err_l, err_a, err_b, w);
                        }
                    }
                    if x + 1 < width {
                        let ti = (y + 1) * width + (x + 1);
                        if opaque_at(ti) {
                            let w = if adaptive { weights[ti] } else { 1.0 };
                            diffuse_err(&mut lab_buf, ti, 1.0 / 16.0, err_l, err_a, err_b, w);
                        }
                    }
                }
            } else {
                if x > 0 && opaque_at(idx - 1) {
                    let w = if adaptive { weights[idx - 1] } else { 1.0 };
                    diffuse_err(&mut lab_buf, idx - 1, 7.0 / 16.0, err_l, err_a, err_b, w);
                }
                if y + 1 < height {
                    if x + 1 < width {
                        let ti = (y + 1) * width + (x + 1);
                        if opaque_at(ti) {
                            let w = if adaptive { weights[ti] } else { 1.0 };
                            diffuse_err(&mut lab_buf, ti, 3.0 / 16.0, err_l, err_a, err_b, w);
                        }
                    }
                    {
                        let ti = (y + 1) * width + x;
                        if opaque_at(ti) {
                            let w = if adaptive { weights[ti] } else { 1.0 };
                            diffuse_err(&mut lab_buf, ti, 5.0 / 16.0, err_l, err_a, err_b, w);
                        }
                    }
                    if x > 0 {
                        let ti = (y + 1) * width + (x - 1);
                        if opaque_at(ti) {
                            let w = if adaptive { weights[ti] } else { 1.0 };
                            diffuse_err(&mut lab_buf, ti, 1.0 / 16.0, err_l, err_a, err_b, w);
                        }
                    }
                }
            }
        }
    }

    indices
}

/// Alpha-aware serpentine dithering: error diffusion includes alpha channel.
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
    let max_err_sq = 0.002 * dither_strength;
    let adaptive = mode == DitherMode::Adaptive;
    let high_dither = dither_strength > 0.4;

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
    let dither_map = if high_dither {
        compute_dither_map_4(&lab_buf, width, height)
    } else {
        vec![1.0f32; pixels.len()]
    };

    for y in 0..height {
        let forward = !high_dither || y % 2 == 0;
        let mut prev_index: Option<u8> = None;

        let x_iter: Box<dyn Iterator<Item = usize>> = if forward {
            Box::new(0..width)
        } else {
            Box::new((0..width).rev())
        };

        for x in x_iter {
            let idx = y * width + x;
            let current_alpha = lab_buf[idx][3];

            // Fully transparent
            if pixels[idx].a == 0 {
                indices[idx] = transparent_idx;
                prev_index = Some(transparent_idx);
                continue;
            }

            let current = OKLab::new(lab_buf[idx][0], lab_buf[idx][1], lab_buf[idx][2]);
            let p = pixels[idx];
            let orig_lab = srgb_to_oklab(p.r, p.g, p.b);
            let orig_alpha = p.a as f32 / 255.0;
            let dithered_best = palette.nearest_with_alpha(current, current_alpha);

            let best = if high_dither {
                let undithered_best = palette.nearest_with_alpha(orig_lab, orig_alpha);
                if dithered_best == undithered_best {
                    dithered_best
                } else {
                    let d_dithered = orig_lab.distance_sq(
                        palette.entries_oklab()[dithered_best as usize],
                    );
                    let d_undithered = orig_lab.distance_sq(
                        palette.entries_oklab()[undithered_best as usize],
                    );
                    if d_dithered <= d_undithered * 2.0 {
                        dithered_best
                    } else {
                        undithered_best
                    }
                }
            } else {
                dithered_best
            };
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

            let scale = dither_strength * dither_map[idx];
            let mut err_l = (current.l - chosen_lab.l) * scale;
            let mut err_a = (current.a - chosen_lab.a) * scale;
            let mut err_b = (current.b - chosen_lab.b) * scale;
            let mut err_al = (current_alpha - chosen_alpha) * scale;

            if high_dither {
                let err_mag = err_l * err_l + err_a * err_a + err_b * err_b;
                if err_mag > max_err_sq {
                    err_l *= 0.75;
                    err_a *= 0.75;
                    err_b *= 0.75;
                    err_al *= 0.75;
                }
            }

            let diffuse_err = |buf: &mut [[f32; 4]],
                               ti: usize,
                               fraction: f32,
                               el: f32,
                               ea: f32,
                               eb: f32,
                               eal: f32,
                               w_mod: f32| {
                let f = fraction * w_mod;
                let v = &mut buf[ti];
                v[0] = (v[0] + el * f).clamp(0.0, 1.0);
                v[1] = (v[1] + ea * f).clamp(-0.5, 0.5);
                v[2] = (v[2] + eb * f).clamp(-0.5, 0.5);
                v[3] = (v[3] + eal * f).clamp(0.0, 1.0);
            };

            let opaque_at = |px: usize| pixels[px].a > 0;

            if forward {
                if x + 1 < width && opaque_at(idx + 1) {
                    let w = if adaptive { weights[idx + 1] } else { 1.0 };
                    diffuse_err(&mut lab_buf, idx + 1, 7.0 / 16.0, err_l, err_a, err_b, err_al, w);
                }
                if y + 1 < height {
                    if x > 0 {
                        let ti = (y + 1) * width + (x - 1);
                        if opaque_at(ti) {
                            let w = if adaptive { weights[ti] } else { 1.0 };
                            diffuse_err(&mut lab_buf, ti, 3.0 / 16.0, err_l, err_a, err_b, err_al, w);
                        }
                    }
                    {
                        let ti = (y + 1) * width + x;
                        if opaque_at(ti) {
                            let w = if adaptive { weights[ti] } else { 1.0 };
                            diffuse_err(&mut lab_buf, ti, 5.0 / 16.0, err_l, err_a, err_b, err_al, w);
                        }
                    }
                    if x + 1 < width {
                        let ti = (y + 1) * width + (x + 1);
                        if opaque_at(ti) {
                            let w = if adaptive { weights[ti] } else { 1.0 };
                            diffuse_err(&mut lab_buf, ti, 1.0 / 16.0, err_l, err_a, err_b, err_al, w);
                        }
                    }
                }
            } else {
                if x > 0 && opaque_at(idx - 1) {
                    let w = if adaptive { weights[idx - 1] } else { 1.0 };
                    diffuse_err(&mut lab_buf, idx - 1, 7.0 / 16.0, err_l, err_a, err_b, err_al, w);
                }
                if y + 1 < height {
                    if x + 1 < width {
                        let ti = (y + 1) * width + (x + 1);
                        if opaque_at(ti) {
                            let w = if adaptive { weights[ti] } else { 1.0 };
                            diffuse_err(&mut lab_buf, ti, 3.0 / 16.0, err_l, err_a, err_b, err_al, w);
                        }
                    }
                    {
                        let ti = (y + 1) * width + x;
                        if opaque_at(ti) {
                            let w = if adaptive { weights[ti] } else { 1.0 };
                            diffuse_err(&mut lab_buf, ti, 5.0 / 16.0, err_l, err_a, err_b, err_al, w);
                        }
                    }
                    if x > 0 {
                        let ti = (y + 1) * width + (x - 1);
                        if opaque_at(ti) {
                            let w = if adaptive { weights[ti] } else { 1.0 };
                            diffuse_err(&mut lab_buf, ti, 1.0 / 16.0, err_l, err_a, err_b, err_al, w);
                        }
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
