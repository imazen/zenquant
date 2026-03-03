extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

use crate::blue_noise;
use crate::metric::MpeAccumulator;
use crate::oklab::{OKLab, oklab_to_srgb, srgb_to_oklab};
use crate::palette::Palette;
use crate::remap::RunPriority;

/// Dithering mode.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DitherMode {
    /// No dithering — nearest color only.
    None,
    /// AQ-adaptive Floyd-Steinberg dithering — modulated by masking weights.
    Adaptive,
    /// Sierra Lite (Sierra-2-4-A) — lighter error diffusion with 3 neighbors,
    /// less temporal cascade than Floyd-Steinberg. Good for animation.
    SierraLite,
    /// Position-deterministic blue noise ordered dithering — zero temporal
    /// flicker, slightly lower single-frame quality than error diffusion.
    BlueNoise,
    /// Unidirectional Floyd-Steinberg — always scans left-to-right, no
    /// edge-aware dither map, no error damping, no undithered fallback.
    /// Creates highly row-coherent error patterns that compress well with
    /// PNG's Up filter and deflate. Best at low strength (0.1–0.3).
    Linear,
    /// Fast OKLab Floyd-Steinberg — streamlined for speed.
    ///
    /// Uses fast_cbrt (~3x faster OKLab conversion), no edge-aware dither map,
    /// no error damping, no undithered fallback. Serpentine scan with 2-row
    /// error buffer. Good quality for 256-color palettes at ~3-5x the speed
    /// of Adaptive.
    Ordered,
}

/// Apply serpentine error diffusion kernel for 3-channel (L,a,b) error.
///
/// With 4 weights: Floyd-Steinberg (right, below-back, below, below-forward)
/// With 3 weights: Sierra Lite (right, below-back, below; no diagonal forward)
///
/// The optional opacity guard wraps each neighbor check in `$pixels[$ti].a > 0`.
macro_rules! diffuse_kernel_3ch {
    // Without opacity guard (RGB)
    ($forward:expr, $x:expr, $y:expr, $width:expr, $height:expr, $idx:expr,
     $lab_buf:expr, $diffuse_err:expr, $weights:expr, $adaptive:expr,
     $err_l:expr, $err_a:expr, $err_b:expr,
     $w_right:expr, $w_below_back:expr, $w_below:expr $(, $w_below_fwd:expr)?) => {{
        #[allow(unused_variables)]
        let (forward, x, y, width, height, idx) = ($forward, $x, $y, $width, $height, $idx);
        if forward {
            if x + 1 < width {
                let w = if $adaptive { $weights[idx + 1] } else { 1.0 };
                ($diffuse_err)(&mut $lab_buf, idx + 1, $w_right, $err_l, $err_a, $err_b, w);
            }
            if y + 1 < height {
                if x > 0 {
                    let ti = (y + 1) * width + (x - 1);
                    let w = if $adaptive { $weights[ti] } else { 1.0 };
                    ($diffuse_err)(&mut $lab_buf, ti, $w_below_back, $err_l, $err_a, $err_b, w);
                }
                {
                    let ti = (y + 1) * width + x;
                    let w = if $adaptive { $weights[ti] } else { 1.0 };
                    ($diffuse_err)(&mut $lab_buf, ti, $w_below, $err_l, $err_a, $err_b, w);
                }
                $(
                    if x + 1 < width {
                        let ti = (y + 1) * width + (x + 1);
                        let w = if $adaptive { $weights[ti] } else { 1.0 };
                        ($diffuse_err)(&mut $lab_buf, ti, $w_below_fwd, $err_l, $err_a, $err_b, w);
                    }
                )?
            }
        } else {
            if x > 0 {
                let w = if $adaptive { $weights[idx - 1] } else { 1.0 };
                ($diffuse_err)(&mut $lab_buf, idx - 1, $w_right, $err_l, $err_a, $err_b, w);
            }
            if y + 1 < height {
                if x + 1 < width {
                    let ti = (y + 1) * width + (x + 1);
                    let w = if $adaptive { $weights[ti] } else { 1.0 };
                    ($diffuse_err)(&mut $lab_buf, ti, $w_below_back, $err_l, $err_a, $err_b, w);
                }
                {
                    let ti = (y + 1) * width + x;
                    let w = if $adaptive { $weights[ti] } else { 1.0 };
                    ($diffuse_err)(&mut $lab_buf, ti, $w_below, $err_l, $err_a, $err_b, w);
                }
                $(
                    if x > 0 {
                        let ti = (y + 1) * width + (x - 1);
                        let w = if $adaptive { $weights[ti] } else { 1.0 };
                        ($diffuse_err)(&mut $lab_buf, ti, $w_below_fwd, $err_l, $err_a, $err_b, w);
                    }
                )?
            }
        }
    }};
}

/// Same as `diffuse_kernel_3ch!` but with an opacity guard — only diffuses
/// to neighbors where `$pixels[ti].a > 0`.
macro_rules! diffuse_kernel_3ch_opaque {
    ($forward:expr, $x:expr, $y:expr, $width:expr, $height:expr, $idx:expr,
     $lab_buf:expr, $diffuse_err:expr, $weights:expr, $adaptive:expr,
     $pixels:expr, $err_l:expr, $err_a:expr, $err_b:expr,
     $w_right:expr, $w_below_back:expr, $w_below:expr $(, $w_below_fwd:expr)?) => {{
        #[allow(unused_variables)]
        let (forward, x, y, width, height, idx) = ($forward, $x, $y, $width, $height, $idx);
        if forward {
            if x + 1 < width && $pixels[idx + 1].a > 0 {
                let w = if $adaptive { $weights[idx + 1] } else { 1.0 };
                ($diffuse_err)(&mut $lab_buf, idx + 1, $w_right, $err_l, $err_a, $err_b, w);
            }
            if y + 1 < height {
                if x > 0 {
                    let ti = (y + 1) * width + (x - 1);
                    if $pixels[ti].a > 0 {
                        let w = if $adaptive { $weights[ti] } else { 1.0 };
                        ($diffuse_err)(&mut $lab_buf, ti, $w_below_back, $err_l, $err_a, $err_b, w);
                    }
                }
                {
                    let ti = (y + 1) * width + x;
                    if $pixels[ti].a > 0 {
                        let w = if $adaptive { $weights[ti] } else { 1.0 };
                        ($diffuse_err)(&mut $lab_buf, ti, $w_below, $err_l, $err_a, $err_b, w);
                    }
                }
                $(
                    if x + 1 < width {
                        let ti = (y + 1) * width + (x + 1);
                        if $pixels[ti].a > 0 {
                            let w = if $adaptive { $weights[ti] } else { 1.0 };
                            ($diffuse_err)(&mut $lab_buf, ti, $w_below_fwd, $err_l, $err_a, $err_b, w);
                        }
                    }
                )?
            }
        } else {
            if x > 0 && $pixels[idx - 1].a > 0 {
                let w = if $adaptive { $weights[idx - 1] } else { 1.0 };
                ($diffuse_err)(&mut $lab_buf, idx - 1, $w_right, $err_l, $err_a, $err_b, w);
            }
            if y + 1 < height {
                if x + 1 < width {
                    let ti = (y + 1) * width + (x + 1);
                    if $pixels[ti].a > 0 {
                        let w = if $adaptive { $weights[ti] } else { 1.0 };
                        ($diffuse_err)(&mut $lab_buf, ti, $w_below_back, $err_l, $err_a, $err_b, w);
                    }
                }
                {
                    let ti = (y + 1) * width + x;
                    if $pixels[ti].a > 0 {
                        let w = if $adaptive { $weights[ti] } else { 1.0 };
                        ($diffuse_err)(&mut $lab_buf, ti, $w_below, $err_l, $err_a, $err_b, w);
                    }
                }
                $(
                    if x > 0 {
                        let ti = (y + 1) * width + (x - 1);
                        if $pixels[ti].a > 0 {
                            let w = if $adaptive { $weights[ti] } else { 1.0 };
                            ($diffuse_err)(&mut $lab_buf, ti, $w_below_fwd, $err_l, $err_a, $err_b, w);
                        }
                    }
                )?
            }
        }
    }};
}

/// 4-channel (L,a,b,alpha) error diffusion kernel with opacity guard.
macro_rules! diffuse_kernel_4ch_opaque {
    ($forward:expr, $x:expr, $y:expr, $width:expr, $height:expr, $idx:expr,
     $lab_buf:expr, $diffuse_err:expr, $weights:expr, $adaptive:expr,
     $pixels:expr, $err_l:expr, $err_a:expr, $err_b:expr, $err_al:expr,
     $w_right:expr, $w_below_back:expr, $w_below:expr $(, $w_below_fwd:expr)?) => {{
        #[allow(unused_variables)]
        let (forward, x, y, width, height, idx) = ($forward, $x, $y, $width, $height, $idx);
        if forward {
            if x + 1 < width && $pixels[idx + 1].a > 0 {
                let w = if $adaptive { $weights[idx + 1] } else { 1.0 };
                ($diffuse_err)(&mut $lab_buf, idx + 1, $w_right, $err_l, $err_a, $err_b, $err_al, w);
            }
            if y + 1 < height {
                if x > 0 {
                    let ti = (y + 1) * width + (x - 1);
                    if $pixels[ti].a > 0 {
                        let w = if $adaptive { $weights[ti] } else { 1.0 };
                        ($diffuse_err)(&mut $lab_buf, ti, $w_below_back, $err_l, $err_a, $err_b, $err_al, w);
                    }
                }
                {
                    let ti = (y + 1) * width + x;
                    if $pixels[ti].a > 0 {
                        let w = if $adaptive { $weights[ti] } else { 1.0 };
                        ($diffuse_err)(&mut $lab_buf, ti, $w_below, $err_l, $err_a, $err_b, $err_al, w);
                    }
                }
                $(
                    if x + 1 < width {
                        let ti = (y + 1) * width + (x + 1);
                        if $pixels[ti].a > 0 {
                            let w = if $adaptive { $weights[ti] } else { 1.0 };
                            ($diffuse_err)(&mut $lab_buf, ti, $w_below_fwd, $err_l, $err_a, $err_b, $err_al, w);
                        }
                    }
                )?
            }
        } else {
            if x > 0 && $pixels[idx - 1].a > 0 {
                let w = if $adaptive { $weights[idx - 1] } else { 1.0 };
                ($diffuse_err)(&mut $lab_buf, idx - 1, $w_right, $err_l, $err_a, $err_b, $err_al, w);
            }
            if y + 1 < height {
                if x + 1 < width {
                    let ti = (y + 1) * width + (x + 1);
                    if $pixels[ti].a > 0 {
                        let w = if $adaptive { $weights[ti] } else { 1.0 };
                        ($diffuse_err)(&mut $lab_buf, ti, $w_below_back, $err_l, $err_a, $err_b, $err_al, w);
                    }
                }
                {
                    let ti = (y + 1) * width + x;
                    if $pixels[ti].a > 0 {
                        let w = if $adaptive { $weights[ti] } else { 1.0 };
                        ($diffuse_err)(&mut $lab_buf, ti, $w_below, $err_l, $err_a, $err_b, $err_al, w);
                    }
                }
                $(
                    if x > 0 {
                        let ti = (y + 1) * width + (x - 1);
                        if $pixels[ti].a > 0 {
                            let w = if $adaptive { $weights[ti] } else { 1.0 };
                            ($diffuse_err)(&mut $lab_buf, ti, $w_below_fwd, $err_l, $err_a, $err_b, $err_al, w);
                        }
                    }
                )?
            }
        }
    }};
}

/// 3-channel error diffusion with proportional overflow control and gamut clamping.
///
/// Preserves error direction (critical for hue consistency) by scaling all
/// channels uniformly when any would overflow the gamut bounds.
#[inline(always)]
fn diffuse_err_3ch(
    buf: &mut [[f32; 3]],
    target_idx: usize,
    fraction: f32,
    el: f32,
    ea: f32,
    eb: f32,
    w_mod: f32,
) {
    let f = fraction * w_mod;
    let v = &mut buf[target_idx];
    let dl = el * f;
    let da = ea * f;
    let db = eb * f;
    let new_l = v[0] + dl;
    let new_a = v[1] + da;
    let new_b = v[2] + db;
    let mut ratio = 1.0f32;
    if dl != 0.0 {
        if new_l > 1.05 {
            ratio = ratio.min((1.05 - v[0]) / dl);
        }
        if new_l < -0.05 {
            ratio = ratio.min((-0.05 - v[0]) / dl);
        }
    }
    if da != 0.0 {
        if new_a > 0.55 {
            ratio = ratio.min((0.55 - v[1]) / da);
        }
        if new_a < -0.55 {
            ratio = ratio.min((-0.55 - v[1]) / da);
        }
    }
    if db != 0.0 {
        if new_b > 0.55 {
            ratio = ratio.min((0.55 - v[2]) / db);
        }
        if new_b < -0.55 {
            ratio = ratio.min((-0.55 - v[2]) / db);
        }
    }
    ratio = ratio.max(0.0);
    v[0] += dl * ratio;
    v[1] += da * ratio;
    v[2] += db * ratio;
}

/// 4-channel error diffusion with proportional overflow control and gamut clamping.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn diffuse_err_4ch(
    buf: &mut [[f32; 4]],
    target_idx: usize,
    fraction: f32,
    el: f32,
    ea: f32,
    eb: f32,
    eal: f32,
    w_mod: f32,
) {
    let f = fraction * w_mod;
    let v = &mut buf[target_idx];
    let dl = el * f;
    let da = ea * f;
    let db = eb * f;
    let dal = eal * f;
    let new_l = v[0] + dl;
    let new_a = v[1] + da;
    let new_b = v[2] + db;
    let new_al = v[3] + dal;
    let mut ratio = 1.0f32;
    if dl != 0.0 {
        if new_l > 1.05 {
            ratio = ratio.min((1.05 - v[0]) / dl);
        }
        if new_l < -0.05 {
            ratio = ratio.min((-0.05 - v[0]) / dl);
        }
    }
    if da != 0.0 {
        if new_a > 0.55 {
            ratio = ratio.min((0.55 - v[1]) / da);
        }
        if new_a < -0.55 {
            ratio = ratio.min((-0.55 - v[1]) / da);
        }
    }
    if db != 0.0 {
        if new_b > 0.55 {
            ratio = ratio.min((0.55 - v[2]) / db);
        }
        if new_b < -0.55 {
            ratio = ratio.min((-0.55 - v[2]) / db);
        }
    }
    if dal != 0.0 {
        if new_al > 1.05 {
            ratio = ratio.min((1.05 - v[3]) / dal);
        }
        if new_al < -0.05 {
            ratio = ratio.min((-0.05 - v[3]) / dal);
        }
    }
    ratio = ratio.max(0.0);
    v[0] += dl * ratio;
    v[1] += da * ratio;
    v[2] += db * ratio;
    v[3] += dal * ratio;
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

/// Bundled parameters for dither functions, reducing argument counts.
///
/// Groups the image dimensions, masking weights, palette, dither mode,
/// run priority, dither strength, and optional previous-frame indices
/// that are shared across all dither entry points.
#[derive(Debug, Clone)]
pub struct DitherParams<'a> {
    /// Image width in pixels.
    pub width: usize,
    /// Image height in pixels.
    pub height: usize,
    /// Per-pixel AQ masking weights (length = width * height).
    pub weights: &'a [f32],
    /// Quantized palette to dither against.
    pub palette: &'a Palette,
    /// Dithering algorithm to use.
    pub mode: DitherMode,
    /// Run-length optimization priority.
    pub run_priority: RunPriority,
    /// Error diffusion strength (0.0 = none, 0.5 = half, 1.0 = full).
    pub dither_strength: f32,
    /// Previous frame's index buffer for temporal consistency (APNG).
    pub prev_indices: Option<&'a [u8]>,
    /// Pre-computed OKLab values. If Some, skip batch sRGB→OKLab conversion.
    /// The buffer is copied internally since error diffusion mutates it.
    pub precomputed_labs: Option<&'a [OKLab]>,
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
pub fn dither_image(
    pixels: &[rgb::RGB<u8>],
    params: &DitherParams<'_>,
    mut mpe_acc: Option<&mut MpeAccumulator>,
) -> Vec<u8> {
    let DitherParams {
        width,
        height,
        weights,
        palette,
        mode,
        run_priority,
        dither_strength,
        prev_indices,
        precomputed_labs,
    } = *params;
    if mode == DitherMode::None {
        let indices = simple_remap(pixels, palette);
        if let Some(ref mut acc) = mpe_acc {
            let oklab_pal = palette.entries_oklab();
            for (i, (pixel, &idx)) in pixels.iter().zip(indices.iter()).enumerate() {
                let orig = srgb_to_oklab(pixel.r, pixel.g, pixel.b);
                let chosen = oklab_pal[idx as usize];
                acc.accumulate(i, orig, chosen, weights[i]);
            }
        }
        return indices;
    }

    if mode == DitherMode::BlueNoise {
        return dither_image_blue_noise(pixels, params, mpe_acc);
    }

    if mode == DitherMode::Ordered {
        return dither_image_ordered(pixels, params, mpe_acc);
    }

    let run_bias = run_priority.bias();
    let linear = mode == DitherMode::Linear;

    // Working buffer in OKLab (we add error diffusion to this)
    let mut lab_buf = vec![[0.0f32; 3]; pixels.len()];
    if let Some(labs) = precomputed_labs {
        for (dst, src) in lab_buf.iter_mut().zip(labs.iter()) {
            *dst = [src.l, src.a, src.b];
        }
    } else {
        crate::simd::batch_srgb_to_oklab(pixels, &mut lab_buf);
    }

    let mut indices = vec![0u8; pixels.len()];

    // Feature gates by dither strength:
    // - Serpentine + dither map: always on for Adaptive/SierraLite, off for Linear
    // - Undithered fallback: > 0.4 (prevents hue-shift speckles at high strength)
    // - Error damping: > 0.3 (prevents runaway accumulation)
    // Linear mode disables all of these for maximum row coherence.
    let use_fallback = !linear && dither_strength > 0.4;
    let use_damping = !linear && dither_strength > 0.3;

    // Edge-aware dither map: reduces error generation near edges.
    // Linear mode skips this (uniform 1.0) for predictable row-coherent patterns.
    let dither_map = if linear {
        vec![1.0f32; pixels.len()]
    } else {
        compute_dither_map(&lab_buf, width, height)
    };

    // Error magnitude threshold — errors beyond this get damped.
    let max_err_sq = 0.002 * dither_strength;

    let adaptive = matches!(mode, DitherMode::Adaptive | DitherMode::SierraLite | DitherMode::Linear);
    let use_sierra = mode == DitherMode::SierraLite;

    for y in 0..height {
        // Serpentine: even rows L→R, odd rows R→L. Linear: always L→R.
        let forward = linear || y % 2 == 0;
        let mut prev_index: Option<u8> = None;

        let x_iter: Box<dyn Iterator<Item = usize>> = if forward {
            Box::new(0..width)
        } else {
            Box::new((0..width).rev())
        };

        for x in x_iter {
            let idx = y * width + x;
            let current = OKLab::new(lab_buf[idx][0], lab_buf[idx][1], lab_buf[idx][2]);

            let p = pixels[idx];
            let orig_lab = srgb_to_oklab(p.r, p.g, p.b);
            let seed = palette.nearest_cached(p.r, p.g, p.b);

            // Temporal clamping: if undithered nearest matches prev frame, lock it
            let locked = prev_indices.is_some_and(|pi| seed == pi[idx]);

            let chosen = if locked {
                prev_indices.unwrap()[idx]
            } else {
                let (adj_r, adj_g, adj_b) = oklab_to_srgb(current);
                let seed2 = palette.nearest_cached(adj_r, adj_g, adj_b);
                let dithered_best = palette.nearest_seeded_2(current, seed, seed2);

                let best = if use_fallback {
                    let undithered_best = palette.nearest_seeded(orig_lab, seed);
                    if dithered_best == undithered_best {
                        dithered_best
                    } else {
                        let d_dithered =
                            orig_lab.distance_sq(palette.entries_oklab()[dithered_best as usize]);
                        let d_undithered =
                            orig_lab.distance_sq(palette.entries_oklab()[undithered_best as usize]);
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

                if run_bias > 0.0 {
                    let best_dist = current.distance_sq(best_lab);
                    let w = weights[idx];
                    let threshold = run_bias * best_dist * 2.0 * (1.1 - w);

                    // Collect candidates: horizontal (prev) and vertical (above)
                    let mut alt_idx = best;
                    let mut alt_dist = best_dist;

                    if let Some(prev) = prev_index {
                        let d = current.distance_sq(palette.entries_oklab()[prev as usize]);
                        if d < best_dist + threshold && d < alt_dist {
                            alt_idx = prev;
                            alt_dist = d;
                        }
                    }

                    // Vertical bias (Linear only): prefer above-row index for
                    // PNG Up filter — zero residual when index matches above.
                    if linear && y > 0 {
                        let above = indices[(y - 1) * width + x];
                        let d = current.distance_sq(palette.entries_oklab()[above as usize]);
                        if d < best_dist + threshold && d < alt_dist {
                            alt_idx = above;
                        }
                    }

                    alt_idx
                } else {
                    best
                }
            };

            indices[idx] = chosen;
            prev_index = Some(chosen);

            let chosen_lab = palette.entries_oklab()[chosen as usize];

            if let Some(ref mut acc) = mpe_acc {
                acc.accumulate(idx, orig_lab, chosen_lab, weights[idx]);
            }

            // Error diffusion: flows through both locked and unlocked pixels
            let scale = dither_strength * dither_map[idx];
            let mut err_l = (current.l - chosen_lab.l) * scale;
            let mut err_a = (current.a - chosen_lab.a) * scale;
            let mut err_b = (current.b - chosen_lab.b) * scale;

            if use_damping {
                let err_mag = err_l * err_l + err_a * err_a + err_b * err_b;
                if err_mag > max_err_sq {
                    err_l *= 0.75;
                    err_a *= 0.75;
                    err_b *= 0.75;
                }
            }

            let diffuse_err = diffuse_err_3ch;

            if use_sierra {
                diffuse_kernel_3ch!(forward, x, y, width, height, idx,
                    lab_buf, diffuse_err, weights, adaptive,
                    err_l, err_a, err_b,
                    2.0 / 4.0, 1.0 / 4.0, 1.0 / 4.0);
            } else {
                diffuse_kernel_3ch!(forward, x, y, width, height, idx,
                    lab_buf, diffuse_err, weights, adaptive,
                    err_l, err_a, err_b,
                    7.0 / 16.0, 3.0 / 16.0, 5.0 / 16.0, 1.0 / 16.0);
            }
        }
    }

    indices
}

/// Apply serpentine Floyd-Steinberg dithering for RGBA images.
/// Transparent pixels pass through unchanged.
pub fn dither_image_rgba(
    pixels: &[rgb::RGBA<u8>],
    params: &DitherParams<'_>,
    mut mpe_acc: Option<&mut MpeAccumulator>,
) -> Vec<u8> {
    let DitherParams {
        width,
        height,
        weights,
        palette,
        mode,
        run_priority,
        dither_strength,
        prev_indices,
        precomputed_labs,
    } = *params;
    let transparent_idx = palette.transparent_index().unwrap_or(0);

    if mode == DitherMode::None {
        let indices = simple_remap_rgba(pixels, palette, transparent_idx);
        if let Some(ref mut acc) = mpe_acc {
            let oklab_pal = palette.entries_oklab();
            for (i, (pixel, &idx)) in pixels.iter().zip(indices.iter()).enumerate() {
                if pixel.a == 0 {
                    continue;
                }
                let orig = srgb_to_oklab(pixel.r, pixel.g, pixel.b);
                let chosen = oklab_pal[idx as usize];
                acc.accumulate(i, orig, chosen, weights[i]);
            }
        }
        return indices;
    }

    if mode == DitherMode::BlueNoise {
        return dither_image_rgba_blue_noise(pixels, params, mpe_acc);
    }

    if mode == DitherMode::Ordered {
        return dither_image_rgba_ordered(pixels, params, mpe_acc);
    }

    let run_bias = run_priority.bias();
    let linear = mode == DitherMode::Linear;
    let max_err_sq = 0.002 * dither_strength;
    let adaptive = matches!(mode, DitherMode::Adaptive | DitherMode::SierraLite | DitherMode::Linear);
    let use_fallback = !linear && dither_strength > 0.4;
    let use_damping = !linear && dither_strength > 0.3;
    let use_sierra = mode == DitherMode::SierraLite;

    let mut lab_buf = vec![[0.0f32; 3]; pixels.len()];
    if let Some(labs) = precomputed_labs {
        for (dst, src) in lab_buf.iter_mut().zip(labs.iter()) {
            *dst = [src.l, src.a, src.b];
        }
    } else {
        let rgb_pixels: Vec<rgb::RGB<u8>> = pixels.iter().map(|p| rgb::RGB::new(p.r, p.g, p.b)).collect();
        crate::simd::batch_srgb_to_oklab(&rgb_pixels, &mut lab_buf);
    }

    let mut indices = vec![0u8; pixels.len()];
    let dither_map = if linear {
        vec![1.0f32; pixels.len()]
    } else {
        compute_dither_map(&lab_buf, width, height)
    };

    for y in 0..height {
        let forward = linear || y % 2 == 0;
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

            // Temporal clamping: if undithered nearest matches prev frame, lock it
            let locked = prev_indices.is_some_and(|pi| seed == pi[idx]);

            let chosen = if locked {
                prev_indices.unwrap()[idx]
            } else {
                let (adj_r, adj_g, adj_b) = oklab_to_srgb(current);
                let seed2 = palette.nearest_cached(adj_r, adj_g, adj_b);
                let dithered_best = palette.nearest_seeded_2(current, seed, seed2);

                let best = if use_fallback {
                    let undithered_best = palette.nearest_seeded(orig_lab, seed);
                    if dithered_best == undithered_best {
                        dithered_best
                    } else {
                        let d_dithered =
                            orig_lab.distance_sq(palette.entries_oklab()[dithered_best as usize]);
                        let d_undithered =
                            orig_lab.distance_sq(palette.entries_oklab()[undithered_best as usize]);
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

                if run_bias > 0.0 {
                    let best_dist = current.distance_sq(best_lab);
                    let w = weights[idx];
                    let threshold = run_bias * best_dist * 2.0 * (1.1 - w);

                    let mut alt_idx = best;
                    let mut alt_dist = best_dist;

                    if let Some(prev) = prev_index
                        && prev != transparent_idx
                    {
                        let d =
                            current.distance_sq(palette.entries_oklab()[prev as usize]);
                        if d < best_dist + threshold && d < alt_dist {
                            alt_idx = prev;
                            alt_dist = d;
                        }
                    }

                    if linear && y > 0 {
                        let above = indices[(y - 1) * width + x];
                        if above != transparent_idx {
                            let d =
                                current.distance_sq(palette.entries_oklab()[above as usize]);
                            if d < best_dist + threshold && d < alt_dist {
                                alt_idx = above;
                            }
                        }
                    }

                    alt_idx
                } else {
                    best
                }
            };

            indices[idx] = chosen;
            prev_index = Some(chosen);

            let chosen_lab = palette.entries_oklab()[chosen as usize];

            if let Some(ref mut acc) = mpe_acc {
                acc.accumulate(idx, orig_lab, chosen_lab, weights[idx]);
            }

            // Error diffusion: flows through both locked and unlocked pixels
            let scale = dither_strength * dither_map[idx];
            let mut err_l = (current.l - chosen_lab.l) * scale;
            let mut err_a = (current.a - chosen_lab.a) * scale;
            let mut err_b = (current.b - chosen_lab.b) * scale;

            if use_damping {
                let err_mag = err_l * err_l + err_a * err_a + err_b * err_b;
                if err_mag > max_err_sq {
                    err_l *= 0.75;
                    err_a *= 0.75;
                    err_b *= 0.75;
                }
            }

            let diffuse_err = diffuse_err_3ch;

            if use_sierra {
                diffuse_kernel_3ch_opaque!(forward, x, y, width, height, idx,
                    lab_buf, diffuse_err, weights, adaptive,
                    pixels, err_l, err_a, err_b,
                    2.0 / 4.0, 1.0 / 4.0, 1.0 / 4.0);
            } else {
                diffuse_kernel_3ch_opaque!(forward, x, y, width, height, idx,
                    lab_buf, diffuse_err, weights, adaptive,
                    pixels, err_l, err_a, err_b,
                    7.0 / 16.0, 3.0 / 16.0, 5.0 / 16.0, 1.0 / 16.0);
            }
        }
    }

    indices
}

/// Alpha-aware serpentine dithering: error diffusion includes alpha channel.
/// Used when the palette has per-entry alpha values (PNG, WebP, JXL, Generic).
pub fn dither_image_rgba_alpha(
    pixels: &[rgb::RGBA<u8>],
    params: &DitherParams<'_>,
    mut mpe_acc: Option<&mut MpeAccumulator>,
) -> Vec<u8> {
    let DitherParams {
        width,
        height,
        weights,
        palette,
        mode,
        run_priority,
        dither_strength,
        prev_indices,
        precomputed_labs,
    } = *params;
    let transparent_idx = palette.transparent_index().unwrap_or(0);

    if mode == DitherMode::None {
        let indices = simple_remap_rgba_alpha(pixels, palette, transparent_idx);
        if let Some(ref mut acc) = mpe_acc {
            let oklab_pal = palette.entries_oklab();
            for (i, (pixel, &idx)) in pixels.iter().zip(indices.iter()).enumerate() {
                if pixel.a == 0 {
                    continue;
                }
                let orig = srgb_to_oklab(pixel.r, pixel.g, pixel.b);
                let chosen = oklab_pal[idx as usize];
                acc.accumulate(i, orig, chosen, weights[i]);
            }
        }
        return indices;
    }

    if mode == DitherMode::BlueNoise {
        return dither_image_rgba_alpha_blue_noise(pixels, params, mpe_acc);
    }

    if mode == DitherMode::Ordered {
        return dither_image_rgba_alpha_ordered(pixels, params, mpe_acc);
    }

    let run_bias = run_priority.bias();
    let linear = mode == DitherMode::Linear;
    let max_err_sq = 0.002 * dither_strength;
    let adaptive = matches!(mode, DitherMode::Adaptive | DitherMode::SierraLite | DitherMode::Linear);
    let use_fallback = !linear && dither_strength > 0.4;
    let use_damping = !linear && dither_strength > 0.3;
    let use_sierra = mode == DitherMode::SierraLite;

    // Working buffer: [L, a, b, alpha] — batch-convert RGB via SIMD, then fill alpha
    let mut lab_buf: Vec<[f32; 4]> = if let Some(labs) = precomputed_labs {
        labs.iter()
            .zip(pixels.iter())
            .map(|(lab, p)| [lab.l, lab.a, lab.b, p.a as f32 / 255.0])
            .collect()
    } else {
        let mut rgb_buf = vec![[0.0f32; 3]; pixels.len()];
        let rgb_pixels: Vec<rgb::RGB<u8>> = pixels.iter().map(|p| rgb::RGB::new(p.r, p.g, p.b)).collect();
        crate::simd::batch_srgb_to_oklab(&rgb_pixels, &mut rgb_buf);
        rgb_buf.into_iter()
            .zip(pixels.iter())
            .map(|([l, a, b], p)| [l, a, b, p.a as f32 / 255.0])
            .collect()
    };

    let mut indices = vec![0u8; pixels.len()];
    let dither_map = if linear {
        vec![1.0f32; pixels.len()]
    } else {
        compute_dither_map_4(&lab_buf, width, height)
    };

    for y in 0..height {
        let forward = linear || y % 2 == 0;
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

            // Temporal clamping: use alpha-aware undithered nearest for lock check
            let undithered_nearest = palette.nearest_with_alpha(orig_lab, orig_alpha);
            let locked = prev_indices.is_some_and(|pi| undithered_nearest == pi[idx]);

            let chosen = if locked {
                prev_indices.unwrap()[idx]
            } else {
                let dithered_best = palette.nearest_with_alpha(current, current_alpha);

                let best = if use_fallback {
                    if dithered_best == undithered_nearest {
                        dithered_best
                    } else {
                        let d_dithered =
                            orig_lab.distance_sq(palette.entries_oklab()[dithered_best as usize]);
                        let d_undithered = orig_lab
                            .distance_sq(palette.entries_oklab()[undithered_nearest as usize]);
                        if d_dithered <= d_undithered * 2.0 {
                            dithered_best
                        } else {
                            undithered_nearest
                        }
                    }
                } else {
                    dithered_best
                };
                let best_lab = palette.entries_oklab()[best as usize];

                if run_bias > 0.0 {
                    let best_dist = current.distance_sq(best_lab);
                    let w = weights[idx];
                    let threshold = run_bias * best_dist * 2.0 * (1.1 - w);

                    let mut alt_idx = best;
                    let mut alt_dist = best_dist;

                    if let Some(prev) = prev_index
                        && prev != transparent_idx
                    {
                        let d = current.distance_sq(palette.entries_oklab()[prev as usize]);
                        if d < best_dist + threshold && d < alt_dist {
                            alt_idx = prev;
                            alt_dist = d;
                        }
                    }

                    if linear && y > 0 {
                        let above = indices[(y - 1) * width + x];
                        if above != transparent_idx {
                            let d =
                                current.distance_sq(palette.entries_oklab()[above as usize]);
                            if d < best_dist + threshold && d < alt_dist {
                                alt_idx = above;
                            }
                        }
                    }

                    alt_idx
                } else {
                    best
                }
            };

            indices[idx] = chosen;
            prev_index = Some(chosen);

            let chosen_lab = palette.entries_oklab()[chosen as usize];
            let chosen_alpha = palette.entries_rgba()[chosen as usize][3] as f32 / 255.0;

            if let Some(ref mut acc) = mpe_acc {
                acc.accumulate(idx, orig_lab, chosen_lab, weights[idx]);
            }

            // Error diffusion: flows through both locked and unlocked pixels
            let scale = dither_strength * dither_map[idx];
            let mut err_l = (current.l - chosen_lab.l) * scale;
            let mut err_a = (current.a - chosen_lab.a) * scale;
            let mut err_b = (current.b - chosen_lab.b) * scale;
            let mut err_al = (current_alpha - chosen_alpha) * scale;

            if use_damping {
                let err_mag = err_l * err_l + err_a * err_a + err_b * err_b;
                if err_mag > max_err_sq {
                    err_l *= 0.75;
                    err_a *= 0.75;
                    err_b *= 0.75;
                    err_al *= 0.75;
                }
            }

            let diffuse_err = diffuse_err_4ch;

            if use_sierra {
                diffuse_kernel_4ch_opaque!(forward, x, y, width, height, idx,
                    lab_buf, diffuse_err, weights, adaptive,
                    pixels, err_l, err_a, err_b, err_al,
                    2.0 / 4.0, 1.0 / 4.0, 1.0 / 4.0);
            } else {
                diffuse_kernel_4ch_opaque!(forward, x, y, width, height, idx,
                    lab_buf, diffuse_err, weights, adaptive,
                    pixels, err_l, err_a, err_b, err_al,
                    7.0 / 16.0, 3.0 / 16.0, 5.0 / 16.0, 1.0 / 16.0);
            }
        }
    }

    indices
}

// ---------------------------------------------------------------------------
// Ordered (fast) dithering — streamlined OKLab Floyd-Steinberg
// ---------------------------------------------------------------------------

/// Fast OKLab Floyd-Steinberg dithering for RGB images.
///
/// Uses fast_cbrt batch conversion, serpentine scan, no edge-aware dither map,
/// no error damping, no undithered fallback. Error is diffused via the standard
/// FS kernel (7/16, 3/16, 5/16, 1/16) in OKLab space using a full-image buffer
/// for maximum quality.
fn dither_image_ordered(
    pixels: &[rgb::RGB<u8>],
    params: &DitherParams<'_>,
    mut mpe_acc: Option<&mut MpeAccumulator>,
) -> Vec<u8> {
    let DitherParams {
        width,
        height,
        weights,
        palette,
        run_priority,
        dither_strength,
        prev_indices,
        precomputed_labs,
        ..
    } = *params;
    let run_bias = run_priority.bias();

    // Use precomputed OKLab or fast batch conversion (~3x faster than exact cbrt)
    let mut lab_buf = vec![[0.0f32; 3]; pixels.len()];
    if let Some(labs) = precomputed_labs {
        for (dst, src) in lab_buf.iter_mut().zip(labs.iter()) {
            *dst = [src.l, src.a, src.b];
        }
    } else {
        crate::simd::batch_srgb_to_oklab_fast(pixels, &mut lab_buf);
    }

    let mut indices = vec![0u8; pixels.len()];

    for y in 0..height {
        let forward = y % 2 == 0;
        let mut prev_index: Option<u8> = None;

        let x_iter: Box<dyn Iterator<Item = usize>> = if forward {
            Box::new(0..width)
        } else {
            Box::new((0..width).rev())
        };

        for x in x_iter {
            let idx = y * width + x;
            let current = OKLab::new(lab_buf[idx][0], lab_buf[idx][1], lab_buf[idx][2]);

            let p = pixels[idx];
            let seed = palette.nearest_cached(p.r, p.g, p.b);

            // Temporal clamping
            let locked = prev_indices.is_some_and(|pi| seed == pi[idx]);

            let chosen = if locked {
                prev_indices.unwrap()[idx]
            } else {
                let (adj_r, adj_g, adj_b) = oklab_to_srgb(current);
                let seed2 = palette.nearest_cached(adj_r, adj_g, adj_b);
                let best = palette.nearest_seeded_2(current, seed, seed2);
                let best_lab = palette.entries_oklab()[best as usize];

                if run_bias > 0.0 {
                    let best_dist = current.distance_sq(best_lab);
                    let w = weights[idx];
                    let threshold = run_bias * best_dist * 2.0 * (1.1 - w);

                    let mut alt_idx = best;
                    let mut alt_dist = best_dist;

                    if let Some(prev) = prev_index {
                        let d = current.distance_sq(palette.entries_oklab()[prev as usize]);
                        if d < best_dist + threshold && d < alt_dist {
                            alt_idx = prev;
                            alt_dist = d;
                        }
                    }
                    let _ = alt_dist;

                    alt_idx
                } else {
                    best
                }
            };

            indices[idx] = chosen;
            prev_index = Some(chosen);

            let chosen_lab = palette.entries_oklab()[chosen as usize];

            if let Some(ref mut acc) = mpe_acc {
                let orig_lab = srgb_to_oklab(p.r, p.g, p.b);
                acc.accumulate(idx, orig_lab, chosen_lab, weights[idx]);
            }

            // Error diffusion — no damping, no dither map modulation
            let err_l = (current.l - chosen_lab.l) * dither_strength;
            let err_a = (current.a - chosen_lab.a) * dither_strength;
            let err_b = (current.b - chosen_lab.b) * dither_strength;

            // FS kernel: diffuse error to neighbors via the existing 3ch function
            // (includes proportional overflow control / gamut clamping)
            diffuse_kernel_3ch!(forward, x, y, width, height, idx,
                lab_buf, diffuse_err_3ch, weights, true,
                err_l, err_a, err_b,
                7.0 / 16.0, 3.0 / 16.0, 5.0 / 16.0, 1.0 / 16.0);
        }
    }

    indices
}

/// Fast OKLab Floyd-Steinberg dithering for RGBA images (binary transparency).
fn dither_image_rgba_ordered(
    pixels: &[rgb::RGBA<u8>],
    params: &DitherParams<'_>,
    mut mpe_acc: Option<&mut MpeAccumulator>,
) -> Vec<u8> {
    let DitherParams {
        width,
        height,
        weights,
        palette,
        run_priority,
        dither_strength,
        prev_indices,
        precomputed_labs,
        ..
    } = *params;
    let transparent_idx = palette.transparent_index().unwrap_or(0);
    let run_bias = run_priority.bias();

    let mut lab_buf = vec![[0.0f32; 3]; pixels.len()];
    if let Some(labs) = precomputed_labs {
        for (dst, src) in lab_buf.iter_mut().zip(labs.iter()) {
            *dst = [src.l, src.a, src.b];
        }
    } else {
        let rgb_pixels: Vec<rgb::RGB<u8>> =
            pixels.iter().map(|p| rgb::RGB::new(p.r, p.g, p.b)).collect();
        crate::simd::batch_srgb_to_oklab_fast(&rgb_pixels, &mut lab_buf);
    }

    let mut indices = vec![0u8; pixels.len()];

    for y in 0..height {
        let forward = y % 2 == 0;
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
            let seed = palette.nearest_cached(p.r, p.g, p.b);

            let locked = prev_indices.is_some_and(|pi| seed == pi[idx]);

            let chosen = if locked {
                prev_indices.unwrap()[idx]
            } else {
                let (adj_r, adj_g, adj_b) = oklab_to_srgb(current);
                let seed2 = palette.nearest_cached(adj_r, adj_g, adj_b);
                let best = palette.nearest_seeded_2(current, seed, seed2);
                let best_lab = palette.entries_oklab()[best as usize];

                if run_bias > 0.0 {
                    let best_dist = current.distance_sq(best_lab);
                    let w = weights[idx];
                    let threshold = run_bias * best_dist * 2.0 * (1.1 - w);

                    let mut alt_idx = best;
                    let mut alt_dist = best_dist;

                    if let Some(prev) = prev_index
                        && prev != transparent_idx
                    {
                        let d = current.distance_sq(palette.entries_oklab()[prev as usize]);
                        if d < best_dist + threshold && d < alt_dist {
                            alt_idx = prev;
                            alt_dist = d;
                        }
                    }
                    let _ = alt_dist;

                    alt_idx
                } else {
                    best
                }
            };

            indices[idx] = chosen;
            prev_index = Some(chosen);

            let chosen_lab = palette.entries_oklab()[chosen as usize];

            if let Some(ref mut acc) = mpe_acc {
                let orig_lab = srgb_to_oklab(p.r, p.g, p.b);
                acc.accumulate(idx, orig_lab, chosen_lab, weights[idx]);
            }

            let err_l = (current.l - chosen_lab.l) * dither_strength;
            let err_a = (current.a - chosen_lab.a) * dither_strength;
            let err_b = (current.b - chosen_lab.b) * dither_strength;

            diffuse_kernel_3ch_opaque!(forward, x, y, width, height, idx,
                lab_buf, diffuse_err_3ch, weights, true,
                pixels, err_l, err_a, err_b,
                7.0 / 16.0, 3.0 / 16.0, 5.0 / 16.0, 1.0 / 16.0);
        }
    }

    indices
}

/// Fast OKLab Floyd-Steinberg dithering for RGBA with full alpha quantization.
fn dither_image_rgba_alpha_ordered(
    pixels: &[rgb::RGBA<u8>],
    params: &DitherParams<'_>,
    mut mpe_acc: Option<&mut MpeAccumulator>,
) -> Vec<u8> {
    let DitherParams {
        width,
        height,
        weights,
        palette,
        run_priority,
        dither_strength,
        prev_indices,
        precomputed_labs,
        ..
    } = *params;
    let transparent_idx = palette.transparent_index().unwrap_or(0);
    let run_bias = run_priority.bias();

    // Working buffer: [L, a, b, alpha]
    let mut lab_buf: Vec<[f32; 4]> = if let Some(labs) = precomputed_labs {
        labs.iter()
            .zip(pixels.iter())
            .map(|(lab, p)| [lab.l, lab.a, lab.b, p.a as f32 / 255.0])
            .collect()
    } else {
        let mut rgb_buf = vec![[0.0f32; 3]; pixels.len()];
        let rgb_pixels: Vec<rgb::RGB<u8>> =
            pixels.iter().map(|p| rgb::RGB::new(p.r, p.g, p.b)).collect();
        crate::simd::batch_srgb_to_oklab_fast(&rgb_pixels, &mut rgb_buf);
        rgb_buf
            .into_iter()
            .zip(pixels.iter())
            .map(|([l, a, b], p)| [l, a, b, p.a as f32 / 255.0])
            .collect()
    };

    let mut indices = vec![0u8; pixels.len()];

    for y in 0..height {
        let forward = y % 2 == 0;
        let mut prev_index: Option<u8> = None;

        let x_iter: Box<dyn Iterator<Item = usize>> = if forward {
            Box::new(0..width)
        } else {
            Box::new((0..width).rev())
        };

        for x in x_iter {
            let idx = y * width + x;
            let current_alpha = lab_buf[idx][3];

            if pixels[idx].a == 0 {
                indices[idx] = transparent_idx;
                prev_index = Some(transparent_idx);
                continue;
            }

            let current = OKLab::new(lab_buf[idx][0], lab_buf[idx][1], lab_buf[idx][2]);
            let p = pixels[idx];

            let undithered_nearest = palette.nearest_with_alpha(
                srgb_to_oklab(p.r, p.g, p.b),
                p.a as f32 / 255.0,
            );
            let locked = prev_indices.is_some_and(|pi| undithered_nearest == pi[idx]);

            let chosen = if locked {
                prev_indices.unwrap()[idx]
            } else {
                let best = palette.nearest_with_alpha(current, current_alpha);
                let best_lab = palette.entries_oklab()[best as usize];

                if run_bias > 0.0 {
                    let best_dist = current.distance_sq(best_lab);
                    let w = weights[idx];
                    let threshold = run_bias * best_dist * 2.0 * (1.1 - w);

                    let mut alt_idx = best;
                    let mut alt_dist = best_dist;

                    if let Some(prev) = prev_index
                        && prev != transparent_idx
                    {
                        let d = current.distance_sq(palette.entries_oklab()[prev as usize]);
                        if d < best_dist + threshold && d < alt_dist {
                            alt_idx = prev;
                            alt_dist = d;
                        }
                    }
                    let _ = alt_dist;

                    alt_idx
                } else {
                    best
                }
            };

            indices[idx] = chosen;
            prev_index = Some(chosen);

            let chosen_lab = palette.entries_oklab()[chosen as usize];
            let chosen_alpha = palette.entries_rgba()[chosen as usize][3] as f32 / 255.0;

            if let Some(ref mut acc) = mpe_acc {
                let orig_lab = srgb_to_oklab(p.r, p.g, p.b);
                acc.accumulate(idx, orig_lab, chosen_lab, weights[idx]);
            }

            let err_l = (current.l - chosen_lab.l) * dither_strength;
            let err_a = (current.a - chosen_lab.a) * dither_strength;
            let err_b = (current.b - chosen_lab.b) * dither_strength;
            let err_al = (current_alpha - chosen_alpha) * dither_strength;

            diffuse_kernel_4ch_opaque!(forward, x, y, width, height, idx,
                lab_buf, diffuse_err_4ch, weights, true,
                pixels, err_l, err_a, err_b, err_al,
                7.0 / 16.0, 3.0 / 16.0, 5.0 / 16.0, 1.0 / 16.0);
        }
    }

    indices
}

// ---------------------------------------------------------------------------
// Blue noise dithering — position-deterministic, zero temporal flicker
// ---------------------------------------------------------------------------

/// Blue noise dithering for RGB images.
///
/// Each pixel gets noise from a tiled 64×64 blue noise map, modulated by the
/// edge-aware dither map. No error propagation — each pixel is independent,
/// so the result is fully deterministic per position.
fn dither_image_blue_noise(
    pixels: &[rgb::RGB<u8>],
    params: &DitherParams<'_>,
    mut mpe_acc: Option<&mut MpeAccumulator>,
) -> Vec<u8> {
    let DitherParams {
        width,
        height,
        weights,
        palette,
        run_priority,
        dither_strength,
        precomputed_labs,
        ..
    } = *params;
    let run_bias = run_priority.bias();

    // Convert to OKLab for dither map computation (read-only after this)
    let mut lab_buf = vec![[0.0f32; 3]; pixels.len()];
    if let Some(labs) = precomputed_labs {
        for (dst, src) in lab_buf.iter_mut().zip(labs.iter()) {
            *dst = [src.l, src.a, src.b];
        }
    } else {
        crate::simd::batch_srgb_to_oklab(pixels, &mut lab_buf);
    }

    let dither_map = compute_dither_map(&lab_buf, width, height);
    let mut indices = vec![0u8; pixels.len()];

    // Noise scales: L has wider perceptual range than a,b
    let noise_l = dither_strength * 0.15;
    let noise_ab = dither_strength * 0.06;

    for y in 0..height {
        let mut prev_index: Option<u8> = None;

        for x in 0..width {
            let idx = y * width + x;
            let p = pixels[idx];
            let orig_lab = srgb_to_oklab(p.r, p.g, p.b);

            // Blue noise threshold modulated by edge-aware map
            let t = blue_noise::threshold(x, y) * dither_map[idx];

            // Add noise in OKLab, clamp to gamut bounds
            let noisy = OKLab::new(
                (orig_lab.l + t * noise_l).clamp(-0.05, 1.05),
                (orig_lab.a + t * noise_ab).clamp(-0.55, 0.55),
                (orig_lab.b + t * noise_ab).clamp(-0.55, 0.55),
            );

            // Two-seed palette search
            let seed = palette.nearest_cached(p.r, p.g, p.b);
            let (adj_r, adj_g, adj_b) = oklab_to_srgb(noisy);
            let seed2 = palette.nearest_cached(adj_r, adj_g, adj_b);
            let best = palette.nearest_seeded_2(noisy, seed, seed2);

            // Run-priority bias
            let chosen = if run_bias > 0.0 {
                if let Some(prev) = prev_index {
                    let best_lab = palette.entries_oklab()[best as usize];
                    let prev_dist = noisy.distance_sq(palette.entries_oklab()[prev as usize]);
                    let best_dist = noisy.distance_sq(best_lab);
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

            if let Some(ref mut acc) = mpe_acc {
                let chosen_lab = palette.entries_oklab()[chosen as usize];
                acc.accumulate(idx, orig_lab, chosen_lab, weights[idx]);
            }
        }
    }

    indices
}

/// Blue noise dithering for RGBA images with binary transparency.
/// Transparent pixels (alpha == 0) map to transparent_idx.
fn dither_image_rgba_blue_noise(
    pixels: &[rgb::RGBA<u8>],
    params: &DitherParams<'_>,
    mut mpe_acc: Option<&mut MpeAccumulator>,
) -> Vec<u8> {
    let DitherParams {
        width,
        height,
        weights,
        palette,
        run_priority,
        dither_strength,
        precomputed_labs,
        ..
    } = *params;
    let transparent_idx = palette.transparent_index().unwrap_or(0);
    let run_bias = run_priority.bias();

    let mut lab_buf = vec![[0.0f32; 3]; pixels.len()];
    if let Some(labs) = precomputed_labs {
        for (dst, src) in lab_buf.iter_mut().zip(labs.iter()) {
            *dst = [src.l, src.a, src.b];
        }
    } else {
        let rgb_pixels: Vec<rgb::RGB<u8>> = pixels.iter().map(|p| rgb::RGB::new(p.r, p.g, p.b)).collect();
        crate::simd::batch_srgb_to_oklab(&rgb_pixels, &mut lab_buf);
    }

    let dither_map = compute_dither_map(&lab_buf, width, height);
    let mut indices = vec![0u8; pixels.len()];

    let noise_l = dither_strength * 0.15;
    let noise_ab = dither_strength * 0.06;

    for y in 0..height {
        let mut prev_index: Option<u8> = None;

        for x in 0..width {
            let idx = y * width + x;

            if pixels[idx].a == 0 {
                indices[idx] = transparent_idx;
                prev_index = Some(transparent_idx);
                continue;
            }

            let p = pixels[idx];
            let orig_lab = srgb_to_oklab(p.r, p.g, p.b);
            let t = blue_noise::threshold(x, y) * dither_map[idx];

            let noisy = OKLab::new(
                (orig_lab.l + t * noise_l).clamp(-0.05, 1.05),
                (orig_lab.a + t * noise_ab).clamp(-0.55, 0.55),
                (orig_lab.b + t * noise_ab).clamp(-0.55, 0.55),
            );

            let seed = palette.nearest_cached(p.r, p.g, p.b);
            let (adj_r, adj_g, adj_b) = oklab_to_srgb(noisy);
            let seed2 = palette.nearest_cached(adj_r, adj_g, adj_b);
            let best = palette.nearest_seeded_2(noisy, seed, seed2);

            let chosen = if run_bias > 0.0 {
                if let Some(prev) = prev_index {
                    if prev != transparent_idx {
                        let best_lab = palette.entries_oklab()[best as usize];
                        let prev_dist = noisy.distance_sq(palette.entries_oklab()[prev as usize]);
                        let best_dist = noisy.distance_sq(best_lab);
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

            if let Some(ref mut acc) = mpe_acc {
                let chosen_lab = palette.entries_oklab()[chosen as usize];
                acc.accumulate(idx, orig_lab, chosen_lab, weights[idx]);
            }
        }
    }

    indices
}

/// Blue noise dithering for RGBA images with full alpha channel.
/// Applies noise to alpha as well (for palettes with per-entry alpha).
fn dither_image_rgba_alpha_blue_noise(
    pixels: &[rgb::RGBA<u8>],
    params: &DitherParams<'_>,
    mut mpe_acc: Option<&mut MpeAccumulator>,
) -> Vec<u8> {
    let DitherParams {
        width,
        height,
        weights,
        palette,
        run_priority,
        dither_strength,
        precomputed_labs,
        ..
    } = *params;
    let transparent_idx = palette.transparent_index().unwrap_or(0);
    let run_bias = run_priority.bias();

    let lab_buf: Vec<[f32; 4]> = if let Some(labs) = precomputed_labs {
        labs.iter()
            .zip(pixels.iter())
            .map(|(lab, p)| [lab.l, lab.a, lab.b, p.a as f32 / 255.0])
            .collect()
    } else {
        let mut rgb_buf = vec![[0.0f32; 3]; pixels.len()];
        let rgb_pixels: Vec<rgb::RGB<u8>> = pixels.iter().map(|p| rgb::RGB::new(p.r, p.g, p.b)).collect();
        crate::simd::batch_srgb_to_oklab(&rgb_pixels, &mut rgb_buf);
        rgb_buf.into_iter()
            .zip(pixels.iter())
            .map(|([l, a, b], p)| [l, a, b, p.a as f32 / 255.0])
            .collect()
    };

    let dither_map = compute_dither_map_4(&lab_buf, width, height);
    let mut indices = vec![0u8; pixels.len()];

    let noise_l = dither_strength * 0.15;
    let noise_ab = dither_strength * 0.06;
    let noise_alpha = dither_strength * 0.10;

    for y in 0..height {
        let mut prev_index: Option<u8> = None;

        for x in 0..width {
            let idx = y * width + x;

            if pixels[idx].a == 0 {
                indices[idx] = transparent_idx;
                prev_index = Some(transparent_idx);
                continue;
            }

            let p = pixels[idx];
            let orig_lab = srgb_to_oklab(p.r, p.g, p.b);
            let orig_alpha = p.a as f32 / 255.0;
            let t = blue_noise::threshold(x, y) * dither_map[idx];

            let noisy = OKLab::new(
                (orig_lab.l + t * noise_l).clamp(-0.05, 1.05),
                (orig_lab.a + t * noise_ab).clamp(-0.55, 0.55),
                (orig_lab.b + t * noise_ab).clamp(-0.55, 0.55),
            );
            let noisy_alpha = (orig_alpha + t * noise_alpha).clamp(0.0, 1.0);

            let best = palette.nearest_with_alpha(noisy, noisy_alpha);

            let chosen = if run_bias > 0.0 {
                if let Some(prev) = prev_index {
                    if prev != transparent_idx {
                        let best_lab = palette.entries_oklab()[best as usize];
                        let prev_dist = noisy.distance_sq(palette.entries_oklab()[prev as usize]);
                        let best_dist = noisy.distance_sq(best_lab);
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

            if let Some(ref mut acc) = mpe_acc {
                let chosen_lab = palette.entries_oklab()[chosen as usize];
                acc.accumulate(idx, orig_lab, chosen_lab, weights[idx]);
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
        let params = DitherParams {
            width: 64,
            height: 1,
            weights: &weights,
            palette: &palette,
            mode: DitherMode::None,
            run_priority: RunPriority::Quality,
            dither_strength: 0.5,
            prev_indices: None,
            precomputed_labs: None,
        };
        let indices = dither_image(&pixels, &params, None);
        assert_eq!(indices.len(), 64);
        for &idx in &indices {
            assert!((idx as usize) < palette.len());
        }
    }

    #[test]
    fn ordered_dither_produces_valid_indices() {
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
        let params = DitherParams {
            width,
            height,
            weights: &weights,
            palette: &palette,
            mode: DitherMode::Ordered,
            run_priority: RunPriority::Balanced,
            dither_strength: 0.5,
            prev_indices: None,
            precomputed_labs: None,
        };
        let indices = dither_image(&pixels, &params, None);
        assert_eq!(indices.len(), width * height);
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
        let params = DitherParams {
            width,
            height,
            weights: &weights,
            palette: &palette,
            mode: DitherMode::Adaptive,
            run_priority: RunPriority::Balanced,
            dither_strength: 0.5,
            prev_indices: None,
            precomputed_labs: None,
        };
        let indices = dither_image(&pixels, &params, None);
        assert_eq!(indices.len(), width * height);
        for &idx in &indices {
            assert!((idx as usize) < palette.len());
        }
    }
}
