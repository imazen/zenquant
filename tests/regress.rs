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
/// Tolerance: 3% relative, 0.002 absolute floor (for near-zero scores).
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

    const REL_TOL: f32 = 0.03;
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

    const REL_TOL: f32 = 0.03;
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
            mpe: 0.159997,
            ss2: -1.61,
            ba: 21.2843,
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
            mpe: 0.022461,
            ss2: 80.05,
            ba: 3.2527,
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
            mpe: 0.160616,
            ss2: -1.61,
            ba: 21.4458,
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
            mpe: 0.029759,
            ss2: 76.04,
            ba: 4.1791,
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
            mpe: 0.159436,
            ss2: -1.61,
            ba: 21.1377,
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
            mpe: 0.029730,
            ss2: 76.06,
            ba: 4.1754,
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
            mpe: 0.174331,
            ss2: -11.59,
            ba: 22.7580,
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
            mpe: 0.050592,
            ss2: 64.42,
            ba: 6.7450,
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
            mpe: 0.047667,
            ss2: 66.13,
            ba: 6.3501,
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
            mpe: 0.042128,
            ss2: 69.27,
            ba: 5.6626,
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
            mpe: 0.022243,
            ss2: 80.17,
            ba: 3.2253,
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
            mpe: 0.020006,
            ss2: 81.41,
            ba: 2.9457,
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
            mpe: 0.020006,
            ss2: 81.41,
            ba: 2.9457,
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
            mpe: 0.026652,
            ss2: 77.74,
            ba: 3.7815,
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
            mpe: 0.024437,
            ss2: 78.95,
            ba: 3.4997,
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
            mpe: 0.024437,
            ss2: 78.95,
            ba: 3.4997,
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
            mpe: 0.141494,
            ss2: 8.87,
            ba: 18.9039,
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
            mpe: 0.030491,
            ss2: 75.64,
            ba: 4.2729,
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
            mpe: 0.141033,
            ss2: 9.03,
            ba: 18.8965,
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
            mpe: 0.029614,
            ss2: 76.12,
            ba: 4.1606,
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
            mpe: 0.141033,
            ss2: 9.03,
            ba: 18.8965,
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
            mpe: 0.028998,
            ss2: 76.46,
            ba: 4.0818,
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
            mpe: 0.029406,
            ss2: 76.23,
            ba: 4.1339,
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
            mpe: 0.027550,
            ss2: 77.25,
            ba: 3.8965,
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
            mpe: 0.025943,
            ss2: 78.13,
            ba: 3.6907,
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
