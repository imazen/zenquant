#[allow(unused_imports)]
use num_traits::Float;

/// OKLab color representation.
///
/// Bjorn Ottosson's perceptually uniform color space.
/// L: lightness [0, 1], a: green-red, b: blue-yellow.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OKLab {
    pub l: f32,
    pub a: f32,
    pub b: f32,
}

impl OKLab {
    pub const fn new(l: f32, a: f32, b: f32) -> Self {
        Self { l, a, b }
    }

    /// Squared Euclidean distance in OKLab space.
    /// Approximates perceptual difference since OKLab is perceptually uniform.
    pub fn distance_sq(self, other: Self) -> f32 {
        let dl = self.l - other.l;
        let da = self.a - other.a;
        let db = self.b - other.b;
        dl * dl + da * da + db * db
    }

    /// Pair with an alpha value to create an OKLabA.
    pub const fn with_alpha(self, alpha: f32) -> OKLabA {
        OKLabA { lab: self, alpha }
    }
}

/// OKLab color with alpha channel for RGBA quantization.
///
/// Alpha is stored as a linear 0.0–1.0 value. Distance function premultiplies
/// by alpha so color differences matter less for transparent pixels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OKLabA {
    pub lab: OKLab,
    pub alpha: f32,
}

impl OKLabA {
    pub const fn new(l: f32, a: f32, b: f32, alpha: f32) -> Self {
        Self {
            lab: OKLab::new(l, a, b),
            alpha,
        }
    }

    /// Alpha-weighted squared distance.
    ///
    /// Color difference is scaled by average alpha — two transparent pixels
    /// with different RGB values should be "close" since the color isn't visible.
    /// Alpha difference is always significant.
    pub fn distance_sq(self, other: Self) -> f32 {
        let avg_alpha = (self.alpha + other.alpha) * 0.5;
        let color_dist = self.lab.distance_sq(other.lab) * avg_alpha;
        let alpha_diff = self.alpha - other.alpha;
        color_dist + alpha_diff * alpha_diff
    }
}

// --- sRGB transfer function (delegated to linear-srgb crate) ---

/// sRGB gamma → linear (single channel, 0..255 → 0.0..1.0)
/// Uses linear-srgb's const LUT — zero init cost, no powf calls.
#[inline(always)]
fn srgb_to_linear(c: u8) -> f32 {
    linear_srgb::default::srgb_u8_to_linear(c)
}

/// Linear → sRGB gamma (single channel, 0.0..1.0 → 0..255)
#[inline(always)]
fn linear_to_srgb(c: f32) -> u8 {
    linear_srgb::default::linear_to_srgb_u8(c.clamp(0.0, 1.0))
}

// --- Fast cube root (IEEE 754 bit trick + 2 Newton-Raphson) ---
// ~22 bits precision, ~3x faster than f32::cbrt(). Borrowed from zenpixels.

/// Fast approximate cube root via IEEE 754 bit trick + 2 Newton-Raphson iterations.
///
/// ~22 bits of precision — sufficient for perceptual color math where the
/// inputs are already quantized to 8-bit sRGB.
#[inline(always)]
pub(crate) fn fast_cbrt(x: f32) -> f32 {
    if x == 0.0 {
        return 0.0;
    }
    let bits = x.to_bits();
    let mut y = f32::from_bits((bits / 3) + 0x2a51_7d48);
    y = (2.0 * y + x / (y * y)) / 3.0;
    y = (2.0 * y + x / (y * y)) / 3.0;
    y
}

/// OKLab L channel only — uses fast_cbrt, skips a/b rows of M2.
///
/// For masking we only need luminance, so this saves ~2/3 of the M2 multiply.
#[inline(always)]
#[allow(clippy::excessive_precision)]
pub(crate) fn srgb_to_oklab_l_fast(r: u8, g: u8, b: u8) -> f32 {
    let r = srgb_to_linear(r);
    let g = srgb_to_linear(g);
    let b = srgb_to_linear(b);
    let l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b;
    let m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b;
    let s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b;
    0.2104542553 * fast_cbrt(l) + 0.7936177850 * fast_cbrt(m) - 0.0040720468 * fast_cbrt(s)
}

/// Full sRGB→OKLab using fast_cbrt. Same matrix math as [`srgb_to_oklab`] but
/// ~3x faster due to the approximate cube root.
#[inline(always)]
#[allow(clippy::excessive_precision, dead_code)]
pub(crate) fn srgb_to_oklab_fast(r: u8, g: u8, b: u8) -> OKLab {
    let r = srgb_to_linear(r);
    let g = srgb_to_linear(g);
    let b = srgb_to_linear(b);
    let l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b;
    let m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b;
    let s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b;
    let l_ = fast_cbrt(l);
    let m_ = fast_cbrt(m);
    let s_ = fast_cbrt(s);
    OKLab {
        l: 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
        a: 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
        b: 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
    }
}

// --- OKLab conversion (Bjorn Ottosson) ---
// Matrix constants are from the OKLab reference implementation — keep author's
// original values, let the compiler truncate to f32.

/// Convert sRGB (0..255 per channel) to OKLab.
#[allow(clippy::excessive_precision)]
pub fn srgb_to_oklab(r: u8, g: u8, b: u8) -> OKLab {
    let r = srgb_to_linear(r);
    let g = srgb_to_linear(g);
    let b = srgb_to_linear(b);

    // Linear sRGB → LMS (using Ottosson's M1 matrix)
    let l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b;
    let m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b;
    let s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b;

    // Cube root
    let l_ = l.cbrt();
    let m_ = m.cbrt();
    let s_ = s.cbrt();

    // LMS → OKLab (Ottosson's M2 matrix)
    OKLab {
        l: 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
        a: 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
        b: 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
    }
}

/// Convert OKLab to sRGB (0..255 per channel).
#[allow(clippy::excessive_precision)]
pub fn oklab_to_srgb(lab: OKLab) -> (u8, u8, u8) {
    // OKLab → LMS (inverse of M2)
    let l_ = lab.l + 0.3963377774 * lab.a + 0.2158037573 * lab.b;
    let m_ = lab.l - 0.1055613458 * lab.a - 0.0638541728 * lab.b;
    let s_ = lab.l - 0.0894841775 * lab.a - 1.2914855480 * lab.b;

    // Undo cube root
    let l = l_ * l_ * l_;
    let m = m_ * m_ * m_;
    let s = s_ * s_ * s_;

    // LMS → linear sRGB (inverse of M1)
    let r = 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s;
    let g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s;
    let b = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s;

    (linear_to_srgb(r), linear_to_srgb(g), linear_to_srgb(b))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn black_roundtrip() {
        let lab = srgb_to_oklab(0, 0, 0);
        assert!(lab.l.abs() < 0.001);
        assert!(lab.a.abs() < 0.001);
        assert!(lab.b.abs() < 0.001);
        let (r, g, b) = oklab_to_srgb(lab);
        assert_eq!((r, g, b), (0, 0, 0));
    }

    #[test]
    fn white_roundtrip() {
        let lab = srgb_to_oklab(255, 255, 255);
        assert!((lab.l - 1.0).abs() < 0.001);
        assert!(lab.a.abs() < 0.001);
        assert!(lab.b.abs() < 0.001);
        let (r, g, b) = oklab_to_srgb(lab);
        assert_eq!((r, g, b), (255, 255, 255));
    }

    #[test]
    fn red_roundtrip() {
        let lab = srgb_to_oklab(255, 0, 0);
        let (r, g, b) = oklab_to_srgb(lab);
        assert_eq!(r, 255);
        assert!(g <= 1);
        assert!(b <= 1);
    }

    #[test]
    fn green_roundtrip() {
        let lab = srgb_to_oklab(0, 255, 0);
        let (r, g, b) = oklab_to_srgb(lab);
        assert!(r <= 1);
        assert_eq!(g, 255);
        assert!(b <= 1);
    }

    #[test]
    fn blue_roundtrip() {
        let lab = srgb_to_oklab(0, 0, 255);
        let (r, g, b) = oklab_to_srgb(lab);
        assert!(r <= 1);
        assert!(g <= 1);
        assert_eq!(b, 255);
    }

    #[test]
    fn midtone_roundtrip() {
        // Test all channels with a mid-tone gray
        let lab = srgb_to_oklab(128, 128, 128);
        let (r, g, b) = oklab_to_srgb(lab);
        // Allow ±1 for rounding
        assert!((r as i16 - 128).unsigned_abs() <= 1);
        assert!((g as i16 - 128).unsigned_abs() <= 1);
        assert!((b as i16 - 128).unsigned_abs() <= 1);
    }

    #[test]
    fn distance_symmetric() {
        let a = srgb_to_oklab(255, 0, 0);
        let b = srgb_to_oklab(0, 0, 255);
        assert!((a.distance_sq(b) - b.distance_sq(a)).abs() < 1e-10);
    }

    #[test]
    fn distance_identity() {
        let a = srgb_to_oklab(100, 150, 200);
        assert!(a.distance_sq(a) < 1e-10);
    }

    #[test]
    fn similar_colors_small_distance() {
        let a = srgb_to_oklab(100, 100, 100);
        let b = srgb_to_oklab(101, 100, 100);
        let far = srgb_to_oklab(200, 50, 50);
        assert!(a.distance_sq(b) < a.distance_sq(far));
    }

    #[test]
    fn fast_cbrt_precision() {
        // fast_cbrt should be within ~22 bits of f32::cbrt
        for &x in &[0.0f32, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 100.0] {
            let expected = x.cbrt();
            let got = super::fast_cbrt(x);
            let rel_err = if expected == 0.0 {
                got.abs()
            } else {
                (got - expected).abs() / expected
            };
            assert!(
                rel_err < 1e-5,
                "fast_cbrt({x}): expected {expected}, got {got}, rel_err {rel_err}"
            );
        }
    }

    #[test]
    fn srgb_to_oklab_fast_matches_exact() {
        // fast variant should be very close to exact for all test colors
        let test_colors: &[(u8, u8, u8)] = &[
            (0, 0, 0),
            (255, 255, 255),
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (128, 128, 128),
            (200, 100, 50),
            (10, 200, 150),
        ];
        for &(r, g, b) in test_colors {
            let exact = srgb_to_oklab(r, g, b);
            let fast = super::srgb_to_oklab_fast(r, g, b);
            let dist = exact.distance_sq(fast);
            assert!(
                dist < 1e-8,
                "srgb_to_oklab_fast({r},{g},{b}): dist_sq={dist}, exact={exact:?}, fast={fast:?}"
            );
        }
    }

    #[test]
    fn srgb_to_oklab_l_fast_matches_exact() {
        let test_colors: &[(u8, u8, u8)] = &[
            (0, 0, 0),
            (255, 255, 255),
            (128, 128, 128),
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
        ];
        for &(r, g, b) in test_colors {
            let exact = srgb_to_oklab(r, g, b).l;
            let fast = super::srgb_to_oklab_l_fast(r, g, b);
            let err = (exact - fast).abs();
            assert!(
                err < 1e-4,
                "srgb_to_oklab_l_fast({r},{g},{b}): exact={exact}, fast={fast}, err={err}"
            );
        }
    }
}
