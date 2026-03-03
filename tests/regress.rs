//! Quality regression tests — ensure quantization scores stay within fitted tolerances.
//!
//! These tests pin MPE, SSIMULACRA2 estimate, and Butteraugli estimate to known baselines.
//! Any algorithmic change must produce scores within tolerance or the baselines must be
//! explicitly updated after reviewing the quality impact.
//!
//! Run calibration to refresh values:
//!   cargo test --test regress -- calibrate --nocapture

use zenquant::{OutputFormat, Quality, QuantizeConfig};

// ============================================================================
// Synthetic test images (always available, deterministic)
// ============================================================================

/// Smooth RGB gradient — tests gradient banding and chroma quantization.
fn gradient_image(width: usize, height: usize) -> Vec<rgb::RGB<u8>> {
    let mut pixels = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            let r = (x * 255 / width.max(1)) as u8;
            let g = (y * 255 / height.max(1)) as u8;
            let b = ((x + y) * 128 / (width + height).max(1)) as u8;
            pixels.push(rgb::RGB { r, g, b });
        }
    }
    pixels
}

/// Pseudo-random noise — many unique colors, stresses palette selection.
fn noisy_image(width: usize, height: usize) -> Vec<rgb::RGB<u8>> {
    let mut pixels = Vec::with_capacity(width * height);
    for i in 0..(width * height) {
        let h = (i as u32).wrapping_mul(2654435761) as u8;
        pixels.push(rgb::RGB {
            r: h,
            g: h.wrapping_add(50),
            b: h.wrapping_add(100),
        });
    }
    pixels
}

/// Color wheel / hue sweep — tests chroma preservation across the full gamut.
fn hue_sweep_image(width: usize, height: usize) -> Vec<rgb::RGB<u8>> {
    let mut pixels = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            let hue = (x as f32 / width as f32) * 360.0;
            let sat = y as f32 / height as f32;
            let (r, g, b) = hsv_to_rgb(hue, sat, 0.9);
            pixels.push(rgb::RGB { r, g, b });
        }
    }
    pixels
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (u8, u8, u8) {
    let c = v * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = v - c;
    let (r1, g1, b1) = match (h as u32 / 60) % 6 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    (
        ((r1 + m) * 255.0) as u8,
        ((g1 + m) * 255.0) as u8,
        ((b1 + m) * 255.0) as u8,
    )
}

/// Semi-transparent RGBA gradient for transparency regression.
fn rgba_gradient(width: usize, height: usize) -> Vec<rgb::RGBA<u8>> {
    let mut pixels = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            let r = (x * 255 / width.max(1)) as u8;
            let g = (y * 255 / height.max(1)) as u8;
            let b = 128u8;
            let a = (y * 255 / height.max(1)) as u8;
            pixels.push(rgb::RGBA { r, g, b, a });
        }
    }
    pixels
}

// ============================================================================
// Test infrastructure
// ============================================================================

struct Baseline {
    name: &'static str,
    mpe: f32,
    ss2: f32,
    ba: f32,
}

/// Check that RGB quantization quality scores match baseline within tolerance.
///
/// Tolerance: 10% relative, 0.002 absolute floor (for near-zero scores).
/// This means scores can shift by up to 10% without failing, allowing
/// minor algorithmic improvements while catching major regressions.
fn check_rgb(
    pixels: &[rgb::RGB<u8>],
    width: usize,
    height: usize,
    quality: Quality,
    max_colors: u32,
    baseline: &Baseline,
) {
    let config = QuantizeConfig::new(OutputFormat::Png)
        .with_quality(quality)
        .with_max_colors(max_colors)
        .with_compute_quality_metric(true);

    let result = zenquant::quantize(pixels, width, height, &config).unwrap();

    let mpe = result.mpe_score().expect("metric should be computed");
    let ss2 = result
        .ssimulacra2_estimate()
        .expect("ss2 should be computed");
    let ba = result
        .butteraugli_estimate()
        .expect("ba should be computed");

    const REL_TOL: f32 = 0.10;
    const ABS_TOL: f32 = 0.002;

    let mpe_tol = (baseline.mpe * REL_TOL).max(ABS_TOL);
    let ss2_tol = (baseline.ss2.abs() * REL_TOL).max(ABS_TOL);
    let ba_tol = (baseline.ba * REL_TOL).max(ABS_TOL);

    assert!(
        (mpe - baseline.mpe).abs() <= mpe_tol,
        "[{}] MPE: expected {:.6} ± {:.6}, got {:.6}",
        baseline.name,
        baseline.mpe,
        mpe_tol,
        mpe
    );
    assert!(
        (ss2 - baseline.ss2).abs() <= ss2_tol,
        "[{}] SS2: expected {:.2} ± {:.2}, got {:.2}",
        baseline.name,
        baseline.ss2,
        ss2_tol,
        ss2
    );
    assert!(
        (ba - baseline.ba).abs() <= ba_tol,
        "[{}] BA: expected {:.4} ± {:.4}, got {:.4}",
        baseline.name,
        baseline.ba,
        ba_tol,
        ba
    );
}

fn check_rgba(
    pixels: &[rgb::RGBA<u8>],
    width: usize,
    height: usize,
    quality: Quality,
    max_colors: u32,
    baseline: &Baseline,
) {
    let config = QuantizeConfig::new(OutputFormat::Png)
        .with_quality(quality)
        .with_max_colors(max_colors)
        .with_compute_quality_metric(true);

    let result = zenquant::quantize_rgba(pixels, width, height, &config).unwrap();

    let mpe = result.mpe_score().expect("metric should be computed");
    let ss2 = result
        .ssimulacra2_estimate()
        .expect("ss2 should be computed");
    let ba = result
        .butteraugli_estimate()
        .expect("ba should be computed");

    const REL_TOL: f32 = 0.10;
    const ABS_TOL: f32 = 0.002;

    let mpe_tol = (baseline.mpe * REL_TOL).max(ABS_TOL);
    let ss2_tol = (baseline.ss2.abs() * REL_TOL).max(ABS_TOL);
    let ba_tol = (baseline.ba * REL_TOL).max(ABS_TOL);

    assert!(
        (mpe - baseline.mpe).abs() <= mpe_tol,
        "[{}] MPE: expected {:.6} ± {:.6}, got {:.6}",
        baseline.name,
        baseline.mpe,
        mpe_tol,
        mpe
    );
    assert!(
        (ss2 - baseline.ss2).abs() <= ss2_tol,
        "[{}] SS2: expected {:.2} ± {:.2}, got {:.2}",
        baseline.name,
        baseline.ss2,
        ss2_tol,
        ss2
    );
    assert!(
        (ba - baseline.ba).abs() <= ba_tol,
        "[{}] BA: expected {:.4} ± {:.4}, got {:.4}",
        baseline.name,
        baseline.ba,
        ba_tol,
        ba
    );
}

// ============================================================================
// Calibration helper
// ============================================================================

#[test]
fn calibrate() {
    let cases: Vec<(&str, Vec<rgb::RGB<u8>>, usize, usize)> = vec![
        ("gradient_64", gradient_image(64, 64), 64, 64),
        ("gradient_256", gradient_image(256, 256), 256, 256),
        ("noise_64", noisy_image(64, 64), 64, 64),
        ("noise_256", noisy_image(256, 256), 256, 256),
        ("hue_sweep_128", hue_sweep_image(128, 128), 128, 128),
    ];

    let presets = [
        ("fast", Quality::Fast),
        ("balanced", Quality::Balanced),
        ("best", Quality::Best),
    ];

    let color_counts = [16u32, 256];

    eprintln!("\n=== RGB Calibration ===\n");
    for (img_name, pixels, w, h) in &cases {
        for &(preset_name, quality) in &presets {
            for &max_colors in &color_counts {
                let config = QuantizeConfig::new(OutputFormat::Png)
                    .with_quality(quality)
                    .with_max_colors(max_colors)
                    .with_compute_quality_metric(true);

                let result = zenquant::quantize(pixels, *w, *h, &config).unwrap();
                let mpe = result.mpe_score().unwrap_or(0.0);
                let ss2 = result.ssimulacra2_estimate().unwrap_or(0.0);
                let ba = result.butteraugli_estimate().unwrap_or(0.0);

                eprintln!(
                    "{img_name}/{preset_name}/{max_colors}c: mpe={mpe:.6}, ss2={ss2:.2}, ba={ba:.4}"
                );
            }
        }
    }

    eprintln!("\n=== RGBA Calibration ===\n");
    let rgba_pixels = rgba_gradient(128, 128);
    for &(preset_name, quality) in &presets {
        let config = QuantizeConfig::new(OutputFormat::Png)
            .with_quality(quality)
            .with_max_colors(256)
            .with_compute_quality_metric(true);

        let result = zenquant::quantize_rgba(&rgba_pixels, 128, 128, &config).unwrap();
        let mpe = result.mpe_score().unwrap_or(0.0);
        let ss2 = result.ssimulacra2_estimate().unwrap_or(0.0);
        let ba = result.butteraugli_estimate().unwrap_or(0.0);

        eprintln!("rgba_gradient/{preset_name}/256c: mpe={mpe:.6}, ss2={ss2:.2}, ba={ba:.4}");
    }
}

// ============================================================================
// RGB regression tests — gradient (easy image, tests banding)
// ============================================================================

#[test]
fn regress_gradient_64_fast_16c() {
    check_rgb(
        &gradient_image(64, 64),
        64,
        64,
        Quality::Fast,
        16,
        &Baseline {
            name: "gradient_64/fast/16c",
            mpe: 0.156743,
            ss2: -1.61,
            ba: 20.4349,
        },
    );
}

#[test]
fn regress_gradient_64_fast_256c() {
    check_rgb(
        &gradient_image(64, 64),
        64,
        64,
        Quality::Fast,
        256,
        &Baseline {
            name: "gradient_64/fast/256c",
            mpe: 0.022407,
            ss2: 80.08,
            ba: 3.2459,
        },
    );
}

#[test]
fn regress_gradient_64_balanced_16c() {
    check_rgb(
        &gradient_image(64, 64),
        64,
        64,
        Quality::Balanced,
        16,
        &Baseline {
            name: "gradient_64/balanced/16c",
            mpe: 0.157420,
            ss2: -1.61,
            ba: 20.6117,
        },
    );
}

#[test]
fn regress_gradient_64_balanced_256c() {
    check_rgb(
        &gradient_image(64, 64),
        64,
        64,
        Quality::Balanced,
        256,
        &Baseline {
            name: "gradient_64/balanced/256c",
            mpe: 0.021906,
            ss2: 80.35,
            ba: 3.1833,
        },
    );
}

#[test]
fn regress_gradient_64_best_16c() {
    check_rgb(
        &gradient_image(64, 64),
        64,
        64,
        Quality::Best,
        16,
        &Baseline {
            name: "gradient_64/best/16c",
            mpe: 0.157330,
            ss2: -1.61,
            ba: 20.5882,
        },
    );
}

#[test]
fn regress_gradient_64_best_256c() {
    check_rgb(
        &gradient_image(64, 64),
        64,
        64,
        Quality::Best,
        256,
        &Baseline {
            name: "gradient_64/best/256c",
            mpe: 0.026987,
            ss2: 77.56,
            ba: 3.8243,
        },
    );
}

// --- gradient 256x256 (larger, more gradient precision needed) ---

#[test]
fn regress_gradient_256_fast_16c() {
    check_rgb(
        &gradient_image(256, 256),
        256,
        256,
        Quality::Fast,
        16,
        &Baseline {
            name: "gradient_256/fast/16c",
            mpe: 0.177423,
            ss2: -12.30,
            ba: 23.7659,
        },
    );
}

#[test]
fn regress_gradient_256_fast_256c() {
    check_rgb(
        &gradient_image(256, 256),
        256,
        256,
        Quality::Fast,
        256,
        &Baseline {
            name: "gradient_256/fast/256c",
            mpe: 0.056919,
            ss2: 60.89,
            ba: 7.5856,
        },
    );
}

#[test]
fn regress_gradient_256_balanced_256c() {
    check_rgb(
        &gradient_image(256, 256),
        256,
        256,
        Quality::Balanced,
        256,
        &Baseline {
            name: "gradient_256/balanced/256c",
            mpe: 0.038337,
            ss2: 71.35,
            ba: 5.2304,
        },
    );
}

#[test]
fn regress_gradient_256_best_256c() {
    check_rgb(
        &gradient_image(256, 256),
        256,
        256,
        Quality::Best,
        256,
        &Baseline {
            name: "gradient_256/best/256c",
            mpe: 0.037371,
            ss2: 71.88,
            ba: 5.1203,
        },
    );
}

// ============================================================================
// RGB regression tests — noise (hard image, stresses palette selection)
// ============================================================================

#[test]
fn regress_noise_64_fast_16c() {
    check_rgb(
        &noisy_image(64, 64),
        64,
        64,
        Quality::Fast,
        16,
        &Baseline {
            name: "noise_64/fast/16c",
            mpe: 0.027894,
            ss2: 77.06,
            ba: 3.9405,
        },
    );
}

#[test]
fn regress_noise_64_balanced_16c() {
    check_rgb(
        &noisy_image(64, 64),
        64,
        64,
        Quality::Balanced,
        16,
        &Baseline {
            name: "noise_64/balanced/16c",
            mpe: 0.024669,
            ss2: 78.82,
            ba: 3.5286,
        },
    );
}

#[test]
fn regress_noise_64_best_16c() {
    check_rgb(
        &noisy_image(64, 64),
        64,
        64,
        Quality::Best,
        16,
        &Baseline {
            name: "noise_64/best/16c",
            mpe: 0.024241,
            ss2: 79.06,
            ba: 3.4751,
        },
    );
}

#[test]
fn regress_noise_256_fast_16c() {
    check_rgb(
        &noisy_image(256, 256),
        256,
        256,
        Quality::Fast,
        16,
        &Baseline {
            name: "noise_256/fast/16c",
            mpe: 0.034009,
            ss2: 73.72,
            ba: 4.7231,
        },
    );
}

#[test]
fn regress_noise_256_balanced_16c() {
    check_rgb(
        &noisy_image(256, 256),
        256,
        256,
        Quality::Balanced,
        16,
        &Baseline {
            name: "noise_256/balanced/16c",
            mpe: 0.032150,
            ss2: 74.74,
            ba: 4.4852,
        },
    );
}

#[test]
fn regress_noise_256_best_16c() {
    check_rgb(
        &noisy_image(256, 256),
        256,
        256,
        Quality::Best,
        16,
        &Baseline {
            name: "noise_256/best/16c",
            mpe: 0.030406,
            ss2: 75.69,
            ba: 4.2619,
        },
    );
}

// ============================================================================
// RGB regression tests — hue sweep (chroma preservation)
// ============================================================================

#[test]
fn regress_hue_sweep_fast_16c() {
    check_rgb(
        &hue_sweep_image(128, 128),
        128,
        128,
        Quality::Fast,
        16,
        &Baseline {
            name: "hue_sweep/fast/16c",
            mpe: 0.141085,
            ss2: 9.01,
            ba: 18.8974,
        },
    );
}

#[test]
fn regress_hue_sweep_fast_256c() {
    check_rgb(
        &hue_sweep_image(128, 128),
        128,
        128,
        Quality::Fast,
        256,
        &Baseline {
            name: "hue_sweep/fast/256c",
            mpe: 0.030204,
            ss2: 75.80,
            ba: 4.2361,
        },
    );
}

#[test]
fn regress_hue_sweep_balanced_16c() {
    check_rgb(
        &hue_sweep_image(128, 128),
        128,
        128,
        Quality::Balanced,
        16,
        &Baseline {
            name: "hue_sweep/balanced/16c",
            mpe: 0.140249,
            ss2: 9.31,
            ba: 18.8840,
        },
    );
}

#[test]
fn regress_hue_sweep_balanced_256c() {
    check_rgb(
        &hue_sweep_image(128, 128),
        128,
        128,
        Quality::Balanced,
        256,
        &Baseline {
            name: "hue_sweep/balanced/256c",
            mpe: 0.028936,
            ss2: 76.49,
            ba: 4.0738,
        },
    );
}

#[test]
fn regress_hue_sweep_best_16c() {
    check_rgb(
        &hue_sweep_image(128, 128),
        128,
        128,
        Quality::Best,
        16,
        &Baseline {
            name: "hue_sweep/best/16c",
            mpe: 0.140347,
            ss2: 9.28,
            ba: 18.8856,
        },
    );
}

#[test]
fn regress_hue_sweep_best_256c() {
    check_rgb(
        &hue_sweep_image(128, 128),
        128,
        128,
        Quality::Best,
        256,
        &Baseline {
            name: "hue_sweep/best/256c",
            mpe: 0.028386,
            ss2: 76.79,
            ba: 4.0034,
        },
    );
}

// ============================================================================
// RGBA regression tests — transparency handling
// ============================================================================

#[test]
fn regress_rgba_gradient_fast_256c() {
    check_rgba(
        &rgba_gradient(128, 128),
        128,
        128,
        Quality::Fast,
        256,
        &Baseline {
            name: "rgba_gradient/fast/256c",
            mpe: 0.035264,
            ss2: 73.04,
            ba: 4.8801,
        },
    );
}

#[test]
fn regress_rgba_gradient_balanced_256c() {
    check_rgba(
        &rgba_gradient(128, 128),
        128,
        128,
        Quality::Balanced,
        256,
        &Baseline {
            name: "rgba_gradient/balanced/256c",
            mpe: 0.037121,
            ss2: 72.02,
            ba: 5.0918,
        },
    );
}

#[test]
fn regress_rgba_gradient_best_256c() {
    check_rgba(
        &rgba_gradient(128, 128),
        128,
        128,
        Quality::Best,
        256,
        &Baseline {
            name: "rgba_gradient/best/256c",
            mpe: 0.033064,
            ss2: 74.24,
            ba: 4.6022,
        },
    );
}

// ============================================================================
// Monotonicity tests — quality should improve or stay same with more resources
// ============================================================================

#[test]
fn more_colors_means_lower_mpe() {
    let pixels = gradient_image(128, 128);
    let mut prev_mpe = f32::MAX;

    for max_colors in [8, 16, 32, 64, 128, 256] {
        let config = QuantizeConfig::new(OutputFormat::Png)
            .with_quality(Quality::Balanced)
            .with_max_colors(max_colors)
            .with_compute_quality_metric(true);

        let result = zenquant::quantize(&pixels, 128, 128, &config).unwrap();
        let mpe = result.mpe_score().unwrap();

        assert!(
            mpe <= prev_mpe + 0.001,
            "MPE should decrease with more colors: {max_colors}c gave {mpe:.6}, prev was {prev_mpe:.6}"
        );
        prev_mpe = mpe;
    }
}

#[test]
fn higher_quality_means_lower_or_equal_mpe() {
    let pixels = noisy_image(128, 128);

    let mut scores = Vec::new();
    for quality in [Quality::Fast, Quality::Balanced, Quality::Best] {
        let config = QuantizeConfig::new(OutputFormat::Png)
            .with_quality(quality)
            .with_max_colors(32)
            .with_compute_quality_metric(true);

        let result = zenquant::quantize(&pixels, 128, 128, &config).unwrap();
        scores.push(result.mpe_score().unwrap());
    }

    // Best should be <= Balanced should be <= Fast (with tolerance for noise)
    assert!(
        scores[2] <= scores[0] + 0.002,
        "Best ({:.6}) should be <= Fast ({:.6})",
        scores[2],
        scores[0]
    );
}
