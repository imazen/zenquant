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

// --- OKLab conversion (Bjorn Ottosson) ---
// Matrix constants are from the OKLab reference implementation — keep author's
// original values, let the compiler truncate to f32.

/// Batch convert sRGB pixels to OKLab.
/// The LUT lookup + matrix multiply pattern auto-vectorizes well.
#[allow(clippy::excessive_precision)]
pub fn srgb_to_oklab_batch(pixels: &[rgb::RGB<u8>], out: &mut Vec<OKLab>) {
    out.clear();
    out.reserve(pixels.len());
    let conv = linear_srgb::lut::SrgbConverter::new();

    for p in pixels {
        let r = conv.srgb_u8_to_linear(p.r);
        let g = conv.srgb_u8_to_linear(p.g);
        let b = conv.srgb_u8_to_linear(p.b);

        let l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b;
        let m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b;
        let s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b;

        let l_ = l.cbrt();
        let m_ = m.cbrt();
        let s_ = s.cbrt();

        out.push(OKLab {
            l: 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
            a: 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
            b: 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
        });
    }
}

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
}
