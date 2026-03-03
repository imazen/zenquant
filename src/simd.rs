//! SIMD-accelerated batch operations for color conversion and palette search.
//!
//! Uses archmage for runtime CPU dispatch (AVX2+FMA on x86_64, NEON on aarch64,
//! scalar fallback everywhere else) and magetypes for portable SIMD abstractions.

#![allow(unsafe_code)] // #[arcane] generates target_feature wrappers

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

use archmage::incant;
use magetypes::simd::backends::F32x8Convert;
use magetypes::simd::generic::f32x8;

use crate::oklab::{OKLab, srgb_to_oklab};

// ============================================================================
// Batch sRGB → OKLab conversion
// ============================================================================

/// Convert a slice of sRGB pixels to OKLab, writing `[L, a, b]` triples.
///
/// Uses SIMD (AVX2+FMA or NEON) to process 8 pixels at a time. Remainder
/// pixels fall back to scalar conversion.
pub(crate) fn batch_srgb_to_oklab(pixels: &[rgb::RGB<u8>], out: &mut [[f32; 3]]) {
    assert_eq!(pixels.len(), out.len());
    incant!(batch_srgb_to_oklab_dispatch(pixels, out), [v3, neon]);
}

/// Convert a slice of sRGB pixels to a Vec<OKLab>.
pub(crate) fn batch_srgb_to_oklab_vec(pixels: &[rgb::RGB<u8>]) -> Vec<OKLab> {
    let mut buf = vec![[0.0f32; 3]; pixels.len()];
    batch_srgb_to_oklab(pixels, &mut buf);
    buf.into_iter()
        .map(|[l, a, b]| OKLab::new(l, a, b))
        .collect()
}

// --- Platform-specific entry points ---

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn batch_srgb_to_oklab_dispatch_v3(
    token: archmage::X64V3Token,
    pixels: &[rgb::RGB<u8>],
    out: &mut [[f32; 3]],
) {
    batch_srgb_to_oklab_generic(token, pixels, out);
}

#[cfg(target_arch = "aarch64")]
#[archmage::arcane]
fn batch_srgb_to_oklab_dispatch_neon(
    token: archmage::NeonToken,
    pixels: &[rgb::RGB<u8>],
    out: &mut [[f32; 3]],
) {
    batch_srgb_to_oklab_generic(token, pixels, out);
}

fn batch_srgb_to_oklab_dispatch_scalar(
    _token: archmage::ScalarToken,
    pixels: &[rgb::RGB<u8>],
    out: &mut [[f32; 3]],
) {
    for (px, o) in pixels.iter().zip(out.iter_mut()) {
        let lab = srgb_to_oklab(px.r, px.g, px.b);
        *o = [lab.l, lab.a, lab.b];
    }
}

// --- Generic SIMD implementation ---

#[inline(always)]
fn batch_srgb_to_oklab_generic<T: F32x8Convert>(
    token: T,
    pixels: &[rgb::RGB<u8>],
    out: &mut [[f32; 3]],
) {
    let chunks = pixels.len() / 8;
    let remainder = pixels.len() % 8;

    for c in 0..chunks {
        let base = c * 8;

        // Scalar LUT gather: 24 lookups into sRGB→linear table
        let mut lin_r = [0.0f32; 8];
        let mut lin_g = [0.0f32; 8];
        let mut lin_b = [0.0f32; 8];
        for i in 0..8 {
            let px = &pixels[base + i];
            lin_r[i] = linear_srgb::default::srgb_u8_to_linear(px.r);
            lin_g[i] = linear_srgb::default::srgb_u8_to_linear(px.g);
            lin_b[i] = linear_srgb::default::srgb_u8_to_linear(px.b);
        }

        let (l_out, a_out, b_out) = srgb_to_oklab_8x(token, lin_r, lin_g, lin_b);

        for i in 0..8 {
            out[base + i] = [l_out[i], a_out[i], b_out[i]];
        }
    }

    // Remainder: pad to 8 and run through SIMD to keep consistent precision
    if remainder > 0 {
        let base = chunks * 8;
        let mut lin_r = [0.0f32; 8];
        let mut lin_g = [0.0f32; 8];
        let mut lin_b = [0.0f32; 8];
        for i in 0..remainder {
            let px = &pixels[base + i];
            lin_r[i] = linear_srgb::default::srgb_u8_to_linear(px.r);
            lin_g[i] = linear_srgb::default::srgb_u8_to_linear(px.g);
            lin_b[i] = linear_srgb::default::srgb_u8_to_linear(px.b);
        }
        let (l_out, a_out, b_out) = srgb_to_oklab_8x(token, lin_r, lin_g, lin_b);
        for i in 0..remainder {
            out[base + i] = [l_out[i], a_out[i], b_out[i]];
        }
    }
}

/// SIMD OKLab conversion for 8 pixels at once.
///
/// Takes linearized R, G, B arrays (8 values each), returns L, a, b arrays.
/// Uses FMA chains for M1 and M2 matrix multiplies, `cbrt_midp()` for cube root.
#[inline(always)]
#[allow(clippy::excessive_precision)]
fn srgb_to_oklab_8x<T: F32x8Convert>(
    token: T,
    lin_r: [f32; 8],
    lin_g: [f32; 8],
    lin_b: [f32; 8],
) -> ([f32; 8], [f32; 8], [f32; 8]) {
    let r = f32x8::from_array(token, lin_r);
    let g = f32x8::from_array(token, lin_g);
    let b = f32x8::from_array(token, lin_b);

    // M1: linear sRGB → LMS (Ottosson's matrix)
    // Each row: FMA chain = coeff0*r + (coeff1*g + coeff2*b)
    let lms_l = f32x8::splat(token, 0.4122214708).mul_add(
        r,
        f32x8::splat(token, 0.5363325363).mul_add(g, f32x8::splat(token, 0.0514459929) * b),
    );
    let lms_m = f32x8::splat(token, 0.2119034982).mul_add(
        r,
        f32x8::splat(token, 0.6806995451).mul_add(g, f32x8::splat(token, 0.1073969566) * b),
    );
    let lms_s = f32x8::splat(token, 0.0883024619).mul_add(
        r,
        f32x8::splat(token, 0.2817188376).mul_add(g, f32x8::splat(token, 0.6299787005) * b),
    );

    // Cube root (~3 ULP precision, sufficient for perceptual color)
    let l_ = lms_l.cbrt_midp();
    let m_ = lms_m.cbrt_midp();
    let s_ = lms_s.cbrt_midp();

    // M2: LMS^(1/3) → OKLab (Ottosson's matrix)
    let ok_l = f32x8::splat(token, 0.2104542553).mul_add(
        l_,
        f32x8::splat(token, 0.7936177850).mul_add(m_, f32x8::splat(token, -0.0040720468) * s_),
    );
    let ok_a = f32x8::splat(token, 1.9779984951).mul_add(
        l_,
        f32x8::splat(token, -2.4285922050).mul_add(m_, f32x8::splat(token, 0.4505937099) * s_),
    );
    let ok_b = f32x8::splat(token, 0.0259040371).mul_add(
        l_,
        f32x8::splat(token, 0.7827717662).mul_add(m_, f32x8::splat(token, -0.8086757660) * s_),
    );

    (ok_l.to_array(), ok_a.to_array(), ok_b.to_array())
}

// ============================================================================
// PaletteSimd: AoSoA layout for SIMD nearest-neighbor search
// ============================================================================

/// Palette in AoSoA (Array of Structures of Arrays) layout for SIMD search.
///
/// Channels are split into groups of 8 for f32x8 processing. Unused lanes
/// and the transparent index (if any) are padded with `f32::INFINITY` so
/// they never win a nearest-neighbor comparison.
#[derive(Debug, Clone)]
pub(crate) struct PaletteSimd {
    l: Vec<[f32; 8]>,
    a: Vec<[f32; 8]>,
    b: Vec<[f32; 8]>,
    num_entries: usize,
    start: usize, // 1 if transparent index present, else 0
}

impl PaletteSimd {
    /// Create an empty PaletteSimd (placeholder before real data is available).
    pub(crate) fn empty() -> Self {
        Self {
            l: Vec::new(),
            a: Vec::new(),
            b: Vec::new(),
            num_entries: 0,
            start: 0,
        }
    }

    /// Build from a palette's OKLab entries.
    pub(crate) fn from_palette(palette: &crate::palette::Palette) -> Self {
        let entries = palette.entries_oklab();
        let start = if palette.transparent_index().is_some() {
            1
        } else {
            0
        };
        let num_entries = entries.len();
        let num_groups = num_entries.saturating_sub(start).div_ceil(8);

        let mut l = vec![[f32::INFINITY; 8]; num_groups];
        let mut a = vec![[f32::INFINITY; 8]; num_groups];
        let mut b = vec![[f32::INFINITY; 8]; num_groups];

        for (i, entry) in entries[start..].iter().enumerate() {
            let group = i / 8;
            let lane = i % 8;
            l[group][lane] = entry.l;
            a[group][lane] = entry.a;
            b[group][lane] = entry.b;
        }

        Self {
            l,
            a,
            b,
            num_entries,
            start,
        }
    }

    /// Build from a raw slice of OKLab entries (e.g. centroids during k-means).
    pub(crate) fn from_oklab_slice(entries: &[OKLab], start: usize) -> Self {
        let num_entries = entries.len();
        let num_groups = num_entries.saturating_sub(start).div_ceil(8);

        let mut l = vec![[f32::INFINITY; 8]; num_groups];
        let mut a = vec![[f32::INFINITY; 8]; num_groups];
        let mut b = vec![[f32::INFINITY; 8]; num_groups];

        for (i, entry) in entries[start..].iter().enumerate() {
            let group = i / 8;
            let lane = i % 8;
            l[group][lane] = entry.l;
            a[group][lane] = entry.a;
            b[group][lane] = entry.b;
        }

        Self {
            l,
            a,
            b,
            num_entries,
            start,
        }
    }

    /// Find the nearest palette index to the given OKLab color.
    pub(crate) fn nearest(&self, color: OKLab) -> u8 {
        incant!(palette_nearest_dispatch(self, color), [v3, neon])
    }
}

// --- Platform-specific nearest entry points ---

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn palette_nearest_dispatch_v3(token: archmage::X64V3Token, pal: &PaletteSimd, color: OKLab) -> u8 {
    palette_nearest_generic(token, pal, color)
}

#[cfg(target_arch = "aarch64")]
#[archmage::arcane]
fn palette_nearest_dispatch_neon(
    token: archmage::NeonToken,
    pal: &PaletteSimd,
    color: OKLab,
) -> u8 {
    palette_nearest_generic(token, pal, color)
}

fn palette_nearest_dispatch_scalar(
    token: archmage::ScalarToken,
    pal: &PaletteSimd,
    color: OKLab,
) -> u8 {
    palette_nearest_generic(token, pal, color)
}

/// Generic SIMD nearest-neighbor search.
///
/// Processes 8 palette entries per iteration: computes squared distances
/// with FMA, then does a horizontal reduce to find the group minimum.
#[inline(always)]
fn palette_nearest_generic<T: F32x8Convert>(token: T, pal: &PaletteSimd, color: OKLab) -> u8 {
    let ql = f32x8::splat(token, color.l);
    let qa = f32x8::splat(token, color.a);
    let qb = f32x8::splat(token, color.b);

    let mut best_idx = pal.start as u8;
    let mut best_d = f32::MAX;

    for (gi, ((pl, pa), pb)) in pal.l.iter().zip(pal.a.iter()).zip(pal.b.iter()).enumerate() {
        let pl = f32x8::load(token, pl);
        let pa = f32x8::load(token, pa);
        let pb = f32x8::load(token, pb);

        // Squared distance: (ql-pl)^2 + (qa-pa)^2 + (qb-pb)^2
        let dl = ql - pl;
        let da = qa - pa;
        let db = qb - pb;
        let dist = dl.mul_add(dl, da.mul_add(da, db * db));

        let min_d = dist.reduce_min();
        if min_d < best_d {
            // Find which lane holds the minimum
            let arr = dist.to_array();
            for (lane, &d) in arr.iter().enumerate() {
                if d == min_d {
                    let idx = gi * 8 + lane + pal.start;
                    if idx < pal.num_entries {
                        best_d = d;
                        best_idx = idx as u8;
                    }
                    break;
                }
            }
        }
    }

    best_idx
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::oklab;
    use crate::palette::{Palette, PaletteSortStrategy};

    #[test]
    fn batch_conversion_black_white_primaries() {
        let pixels = vec![
            rgb::RGB::new(0, 0, 0),       // black
            rgb::RGB::new(255, 255, 255), // white
            rgb::RGB::new(255, 0, 0),     // red
            rgb::RGB::new(0, 255, 0),     // green
            rgb::RGB::new(0, 0, 255),     // blue
            rgb::RGB::new(128, 128, 128), // mid gray
            rgb::RGB::new(255, 255, 0),   // yellow
            rgb::RGB::new(0, 255, 255),   // cyan
            rgb::RGB::new(255, 0, 255),   // magenta
        ];

        let mut out = vec![[0.0f32; 3]; pixels.len()];
        batch_srgb_to_oklab(&pixels, &mut out);

        for (i, px) in pixels.iter().enumerate() {
            let scalar = oklab::srgb_to_oklab(px.r, px.g, px.b);
            let [l, a, b] = out[i];
            assert!(
                (l - scalar.l).abs() < 2e-4
                    && (a - scalar.a).abs() < 2e-4
                    && (b - scalar.b).abs() < 2e-4,
                "pixel {i} ({},{},{}) SIMD [{l}, {a}, {b}] vs scalar [{}, {}, {}] — diff [{}, {}, {}]",
                px.r,
                px.g,
                px.b,
                scalar.l,
                scalar.a,
                scalar.b,
                (l - scalar.l).abs(),
                (a - scalar.a).abs(),
                (b - scalar.b).abs(),
            );
        }
    }

    #[test]
    fn batch_conversion_all_grays() {
        // Test all 256 gray values — exercises every LUT entry
        let pixels: Vec<rgb::RGB<u8>> = (0..=255).map(|v| rgb::RGB::new(v, v, v)).collect();
        let mut out = vec![[0.0f32; 3]; 256];
        batch_srgb_to_oklab(&pixels, &mut out);

        for (i, px) in pixels.iter().enumerate() {
            let scalar = oklab::srgb_to_oklab(px.r, px.g, px.b);
            let [l, a, b] = out[i];
            assert!(
                (l - scalar.l).abs() < 5e-4
                    && (a - scalar.a).abs() < 5e-4
                    && (b - scalar.b).abs() < 5e-4,
                "gray {i}: SIMD [{l}, {a}, {b}] vs scalar [{}, {}, {}]",
                scalar.l,
                scalar.a,
                scalar.b,
            );
        }
    }

    #[test]
    fn batch_conversion_remainder_handling() {
        // Non-multiple-of-8 lengths
        for len in [1, 3, 7, 9, 15, 17] {
            let pixels: Vec<rgb::RGB<u8>> = (0..len)
                .map(|i| {
                    let v = (i as u16 * 37 % 256) as u8;
                    rgb::RGB::new(v, 255 - v, ((v as u16 * 3) % 256) as u8)
                })
                .collect();
            let mut out = vec![[0.0f32; 3]; len];
            batch_srgb_to_oklab(&pixels, &mut out);

            for (i, px) in pixels.iter().enumerate() {
                let scalar = oklab::srgb_to_oklab(px.r, px.g, px.b);
                let [l, a, b] = out[i];
                assert!(
                    (l - scalar.l).abs() < 5e-4
                        && (a - scalar.a).abs() < 5e-4
                        && (b - scalar.b).abs() < 5e-4,
                    "len={len} pixel {i}: SIMD [{l}, {a}, {b}] vs scalar [{}, {}, {}]",
                    scalar.l,
                    scalar.a,
                    scalar.b,
                );
            }
        }
    }

    #[test]
    fn batch_vec_matches_array_output() {
        let pixels: Vec<rgb::RGB<u8>> = (0..20)
            .map(|i| {
                let v = (i as u16 * 47 % 256) as u8;
                rgb::RGB::new(
                    v,
                    ((v as u16 + 80) % 256) as u8,
                    ((v as u16 + 160) % 256) as u8,
                )
            })
            .collect();

        let vec_result = batch_srgb_to_oklab_vec(&pixels);
        let mut arr_result = vec![[0.0f32; 3]; pixels.len()];
        batch_srgb_to_oklab(&pixels, &mut arr_result);

        for (i, (lab, arr)) in vec_result.iter().zip(arr_result.iter()).enumerate() {
            assert!(
                (lab.l - arr[0]).abs() < 1e-10
                    && (lab.a - arr[1]).abs() < 1e-10
                    && (lab.b - arr[2]).abs() < 1e-10,
                "pixel {i}: vec {:?} vs arr {:?}",
                lab,
                arr,
            );
        }
    }

    #[test]
    fn simd_nearest_matches_scalar() {
        // Build a 200-entry palette from diverse colors
        let centroids: Vec<OKLab> = (0..200)
            .map(|i| {
                let r = ((i * 37) % 256) as u8;
                let g = ((i * 73) % 256) as u8;
                let b = ((i * 131) % 256) as u8;
                oklab::srgb_to_oklab(r, g, b)
            })
            .collect();
        let palette =
            Palette::from_centroids_sorted(centroids, false, PaletteSortStrategy::DeltaMinimize);
        let simd_pal = PaletteSimd::from_palette(&palette);

        // Query a grid of colors
        for r in (0..=255).step_by(17) {
            for g in (0..=255).step_by(51) {
                for b in (0..=255).step_by(51) {
                    let color = oklab::srgb_to_oklab(r as u8, g as u8, b as u8);
                    let scalar_idx = palette.nearest(color);
                    let simd_idx = simd_pal.nearest(color);
                    assert_eq!(
                        scalar_idx, simd_idx,
                        "mismatch for sRGB({r},{g},{b}): scalar={scalar_idx}, simd={simd_idx}"
                    );
                }
            }
        }
    }

    #[test]
    fn simd_nearest_with_transparency() {
        // Palette with transparent index — should skip index 0
        let centroids: Vec<OKLab> = (0..50)
            .map(|i| {
                let v = (i * 5) as u8;
                oklab::srgb_to_oklab(v, v, v)
            })
            .collect();
        let palette = Palette::from_centroids_sorted(
            centroids,
            true, // has_transparency → index 0 reserved
            PaletteSortStrategy::Luminance,
        );
        let simd_pal = PaletteSimd::from_palette(&palette);

        // Every query should return a non-zero index
        for v in (0..=255).step_by(7) {
            let color = oklab::srgb_to_oklab(v as u8, v as u8, v as u8);
            let scalar_idx = palette.nearest(color);
            let simd_idx = simd_pal.nearest(color);
            assert_ne!(
                simd_idx, 0,
                "SIMD nearest returned transparent index for gray {v}"
            );
            assert_eq!(scalar_idx, simd_idx, "mismatch for gray {v}");
        }
    }
}
