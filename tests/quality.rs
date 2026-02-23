use zenquant::_internals::{average_run_length, compute_mpe, index_delta_score, srgb_to_oklab};
use zenquant::{OutputFormat, QuantizeConfig, QuantizeError};

/// Compute mean squared error in OKLab space between original pixels and quantized result.
fn compute_mse(pixels: &[rgb::RGB<u8>], palette: &[[u8; 3]], indices: &[u8]) -> f32 {
    let mut total = 0.0f32;
    for (pixel, &idx) in pixels.iter().zip(indices.iter()) {
        let original = srgb_to_oklab(pixel.r, pixel.g, pixel.b);
        let p = palette[idx as usize];
        let quantized = srgb_to_oklab(p[0], p[1], p[2]);
        total += original.distance_sq(quantized);
    }
    total / pixels.len() as f32
}

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

fn noisy_image(width: usize, height: usize) -> Vec<rgb::RGB<u8>> {
    // Pseudo-random noise via simple hash
    let mut pixels = Vec::with_capacity(width * height);
    for i in 0..(width * height) {
        let h = ((i as u32).wrapping_mul(2654435761)) as u8; // Knuth's multiplicative hash
        pixels.push(rgb::RGB {
            r: h,
            g: h.wrapping_add(50),
            b: h.wrapping_add(100),
        });
    }
    pixels
}

#[test]
fn compression_mode_has_longer_runs() {
    let pixels = gradient_image(64, 64);

    let quality_config = QuantizeConfig::new(OutputFormat::Png)
        .max_colors(32)
        ._no_dither()
        ._run_priority_quality();

    let compression_config = QuantizeConfig::new(OutputFormat::Png)
        .max_colors(32)
        ._no_dither()
        ._run_priority_compression();

    let quality_result = zenquant::quantize(&pixels, 64, 64, &quality_config).unwrap();
    let compression_result = zenquant::quantize(&pixels, 64, 64, &compression_config).unwrap();

    let quality_avg_run = average_run_length(quality_result.indices());
    let compression_avg_run = average_run_length(compression_result.indices());

    assert!(
        compression_avg_run >= quality_avg_run * 0.95,
        "compression mode should have similar or longer runs: quality={quality_avg_run:.2}, compression={compression_avg_run:.2}"
    );
}

#[test]
fn delta_sort_reduces_index_deltas() {
    let pixels = gradient_image(32, 32);

    let config = QuantizeConfig::new(OutputFormat::Png)
        .max_colors(16)
        ._no_dither()
        ._run_priority_quality();

    let result = zenquant::quantize(&pixels, 32, 32, &config).unwrap();
    let delta = index_delta_score(result.indices());

    // Just verify it's finite and computed
    assert!(delta < u64::MAX);
    // The score should be relatively modest for a smooth gradient with sorted palette
    // (Hard to set a universal threshold, so just check it's reasonable)
    let avg_delta_per_pixel = delta as f64 / result.indices().len() as f64;
    assert!(
        avg_delta_per_pixel < 1000.0,
        "unexpectedly high delta score: {avg_delta_per_pixel:.2}"
    );
}

#[test]
fn more_colors_lower_mse() {
    let pixels = gradient_image(32, 32);

    let config_8 = QuantizeConfig::new(OutputFormat::Png)
        .max_colors(8)
        ._no_dither()
        ._run_priority_quality();

    let config_32 = QuantizeConfig::new(OutputFormat::Png)
        .max_colors(32)
        ._no_dither()
        ._run_priority_quality();

    let result_8 = zenquant::quantize(&pixels, 32, 32, &config_8).unwrap();
    let result_32 = zenquant::quantize(&pixels, 32, 32, &config_32).unwrap();

    let mse_8 = compute_mse(&pixels, result_8.palette(), result_8.indices());
    let mse_32 = compute_mse(&pixels, result_32.palette(), result_32.indices());

    assert!(
        mse_32 < mse_8,
        "32-color should have lower MSE than 8-color: mse_8={mse_8:.6}, mse_32={mse_32:.6}"
    );
}

#[test]
fn noisy_image_gets_low_weights() {
    // Noisy images should have lower masking weights than smooth ones
    let noisy = noisy_image(32, 32);
    let smooth = vec![
        rgb::RGB {
            r: 128,
            g: 128,
            b: 128
        };
        32 * 32
    ];

    let noisy_weights = zenquant::_internals::compute_masking_weights(&noisy, 32, 32);
    let smooth_weights = zenquant::_internals::compute_masking_weights(&smooth, 32, 32);

    let noisy_mean: f32 = noisy_weights.iter().sum::<f32>() / noisy_weights.len() as f32;
    let smooth_mean: f32 = smooth_weights.iter().sum::<f32>() / smooth_weights.len() as f32;

    assert!(
        smooth_mean > noisy_mean,
        "smooth image should have higher mean weight: smooth={smooth_mean:.3}, noisy={noisy_mean:.3}"
    );
}

#[test]
fn gradient_produces_reasonable_quality() {
    let pixels = gradient_image(64, 64);

    let config = QuantizeConfig::new(OutputFormat::Png).max_colors(256);

    let result = zenquant::quantize(&pixels, 64, 64, &config).unwrap();
    let mse = compute_mse(&pixels, result.palette(), result.indices());

    // MSE should be very low with 256 colors on a gradient
    assert!(mse < 0.001, "MSE too high for 256-color gradient: {mse:.6}");
}

#[test]
fn mpe_lower_with_more_colors() {
    let pixels = gradient_image(32, 32);

    let config_8 = QuantizeConfig::new(OutputFormat::Png)
        .max_colors(8)
        .compute_quality_metric(true);

    let config_64 = QuantizeConfig::new(OutputFormat::Png)
        .max_colors(64)
        .compute_quality_metric(true);

    let result_8 = zenquant::quantize(&pixels, 32, 32, &config_8).unwrap();
    let result_64 = zenquant::quantize(&pixels, 32, 32, &config_64).unwrap();

    let mpe_8 = result_8.mpe_score().expect("metric should be computed");
    let mpe_64 = result_64.mpe_score().expect("metric should be computed");

    assert!(
        mpe_64 < mpe_8,
        "64-color should have lower MPE than 8-color: mpe_8={mpe_8:.6}, mpe_64={mpe_64:.6}"
    );
}

#[test]
fn mpe_inline_matches_standalone() {
    let pixels = gradient_image(32, 32);

    let config = QuantizeConfig::new(OutputFormat::Png)
        .max_colors(16)
        .compute_quality_metric(true);

    let result = zenquant::quantize(&pixels, 32, 32, &config).unwrap();
    let inline_score = result.mpe_score().expect("metric should be computed");

    // Compute standalone MPE from the same result
    let standalone = compute_mpe(&pixels, result.palette(), result.indices(), 32, 32, None);

    // Inline uses per-pixel masking weights; standalone uses uniform weights.
    // So they won't be identical, but both should be finite and positive.
    assert!(inline_score.is_finite() && inline_score > 0.0);
    assert!(standalone.score.is_finite() && standalone.score > 0.0);
}

#[test]
fn mpe_with_masking_weights() {
    let pixels = gradient_image(32, 32);

    let weights = zenquant::_internals::compute_masking_weights(&pixels, 32, 32);

    let config = QuantizeConfig::new(OutputFormat::Png).max_colors(16);

    let result = zenquant::quantize(&pixels, 32, 32, &config).unwrap();

    let mpe = compute_mpe(
        &pixels,
        result.palette(),
        result.indices(),
        32,
        32,
        Some(&weights),
    );

    assert!(
        mpe.score.is_finite() && mpe.score > 0.0,
        "MPE with masking weights should be finite and positive: {}",
        mpe.score
    );
    assert_eq!(mpe.block_cols, 8); // 32 / 4
    assert_eq!(mpe.block_rows, 8);
}

#[test]
fn mpe_disabled_by_default() {
    let pixels = gradient_image(16, 16);
    let config = QuantizeConfig::new(OutputFormat::Png).max_colors(8);
    let result = zenquant::quantize(&pixels, 16, 16, &config).unwrap();
    assert!(
        result.mpe_score().is_none(),
        "MPE should not be computed by default"
    );
}

// ===================== Quality target API tests =====================

#[test]
fn min_ssim2_too_high_returns_quality_not_met() {
    let pixels = gradient_image(32, 32);
    let config = QuantizeConfig::new(OutputFormat::Png)
        .max_colors(2)
        .min_ssim2(99.9);

    let result = zenquant::quantize(&pixels, 32, 32, &config);
    match result {
        Err(QuantizeError::QualityNotMet {
            min_ssim2,
            achieved_ssim2,
        }) => {
            assert!((min_ssim2 - 99.9).abs() < 0.01);
            assert!(
                achieved_ssim2 < 99.9,
                "achieved {achieved_ssim2} should be below 99.9 with only 2 colors"
            );
        }
        Ok(_) => panic!("expected QualityNotMet error with 2 colors and min_ssim2=99.9"),
        Err(e) => panic!("expected QualityNotMet, got {e:?}"),
    }
}

#[test]
fn target_ssim2_with_256_colors_succeeds() {
    let pixels = gradient_image(32, 32);
    let config = QuantizeConfig::new(OutputFormat::Png)
        .max_colors(256)
        .target_ssim2(80.0);

    let result = zenquant::quantize(&pixels, 32, 32, &config).unwrap();

    // Metric should be computed
    assert!(
        result.mpe_score().is_some(),
        "metric should be computed when target_ssim2 is set"
    );
    assert!(
        result.ssimulacra2_estimate().is_some(),
        "ssimulacra2_estimate should be available"
    );
    assert!(
        result.butteraugli_estimate().is_some(),
        "butteraugli_estimate should be available"
    );
}

#[test]
fn min_ssim2_negative_always_passes() {
    let pixels = gradient_image(16, 16);
    let config = QuantizeConfig::new(OutputFormat::Png)
        .max_colors(2)
        .min_ssim2(-100.0);

    // SSIM2 estimates can go negative for terrible quantization, so use -100.0
    // to ensure the floor is never hit.
    let result = zenquant::quantize(&pixels, 16, 16, &config);
    assert!(result.is_ok(), "min_ssim2(-100.0) should always pass");
}

#[test]
fn metric_computed_when_target_set() {
    let pixels = gradient_image(16, 16);

    // Without compute_quality_metric(true), but with target_ssim2
    let config = QuantizeConfig::new(OutputFormat::Png)
        .max_colors(64)
        .target_ssim2(50.0);

    let result = zenquant::quantize(&pixels, 16, 16, &config).unwrap();
    assert!(
        result.mpe_score().is_some(),
        "metric should be computed when target_ssim2 is set"
    );
}

#[test]
fn metric_computed_when_min_set() {
    let pixels = gradient_image(16, 16);

    // Without compute_quality_metric(true), but with min_ssim2
    let config = QuantizeConfig::new(OutputFormat::Png)
        .max_colors(64)
        .min_ssim2(0.0);

    let result = zenquant::quantize(&pixels, 16, 16, &config).unwrap();
    assert!(
        result.mpe_score().is_some(),
        "metric should be computed when min_ssim2 is set"
    );
}

#[test]
fn convenience_accessors_none_when_no_metric() {
    let pixels = gradient_image(16, 16);
    let config = QuantizeConfig::new(OutputFormat::Png).max_colors(16);
    let result = zenquant::quantize(&pixels, 16, 16, &config).unwrap();

    assert!(result.ssimulacra2_estimate().is_none());
    assert!(result.butteraugli_estimate().is_none());
}

#[test]
fn convenience_accessors_present_with_metric() {
    let pixels = gradient_image(32, 32);
    let config = QuantizeConfig::new(OutputFormat::Png)
        .max_colors(16)
        .compute_quality_metric(true);
    let result = zenquant::quantize(&pixels, 32, 32, &config).unwrap();

    let ssim2 = result.ssimulacra2_estimate().expect("should be computed");
    let ba = result.butteraugli_estimate().expect("should be computed");

    assert!(ssim2.is_finite());
    assert!(ba.is_finite());
    assert!(ba >= 0.0);
}

#[test]
fn target_ssim2_selects_lower_tier_for_low_target() {
    // A low target should select a more aggressive compression tier.
    // We verify this indirectly: low target → should still succeed, and
    // the result should have a metric computed.
    let pixels = gradient_image(32, 32);
    let config = QuantizeConfig::new(OutputFormat::Png)
        .max_colors(256)
        .target_ssim2(40.0);

    let result = zenquant::quantize(&pixels, 32, 32, &config).unwrap();
    assert!(result.mpe_score().is_some());
}

#[test]
fn min_ssim2_rgba_returns_quality_not_met() {
    let pixels: Vec<rgb::RGBA<u8>> = (0..1024)
        .map(|i| rgb::RGBA {
            r: (i % 256) as u8,
            g: ((i * 3) % 256) as u8,
            b: ((i * 7) % 256) as u8,
            a: 255,
        })
        .collect();

    let config = QuantizeConfig::new(OutputFormat::Png)
        .max_colors(2)
        .min_ssim2(99.9);

    let result = zenquant::quantize_rgba(&pixels, 32, 32, &config);
    assert!(
        matches!(result, Err(QuantizeError::QualityNotMet { .. })),
        "expected QualityNotMet for RGBA with 2 colors and min_ssim2=99.9, got {result:?}"
    );
}
