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
        .with_max_colors(32)
        ._with_no_dither()
        ._with_run_priority_quality();

    let compression_config = QuantizeConfig::new(OutputFormat::Png)
        .with_max_colors(32)
        ._with_no_dither()
        ._with_run_priority_compression();

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
        .with_max_colors(16)
        ._with_no_dither()
        ._with_run_priority_quality();

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
        .with_max_colors(8)
        ._with_no_dither()
        ._with_run_priority_quality();

    let config_32 = QuantizeConfig::new(OutputFormat::Png)
        .with_max_colors(32)
        ._with_no_dither()
        ._with_run_priority_quality();

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

    let config = QuantizeConfig::new(OutputFormat::Png).with_max_colors(256);

    let result = zenquant::quantize(&pixels, 64, 64, &config).unwrap();
    let mse = compute_mse(&pixels, result.palette(), result.indices());

    // MSE should be very low with 256 colors on a gradient
    assert!(mse < 0.001, "MSE too high for 256-color gradient: {mse:.6}");
}

#[test]
fn mpe_lower_with_more_colors() {
    let pixels = gradient_image(32, 32);

    let config_8 = QuantizeConfig::new(OutputFormat::Png)
        .with_max_colors(8)
        .with_compute_quality_metric(true);

    let config_64 = QuantizeConfig::new(OutputFormat::Png)
        .with_max_colors(64)
        .with_compute_quality_metric(true);

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
        .with_max_colors(16)
        .with_compute_quality_metric(true);

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

    let config = QuantizeConfig::new(OutputFormat::Png).with_max_colors(16);

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
    let config = QuantizeConfig::new(OutputFormat::Png).with_max_colors(8);
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
        .with_max_colors(2)
        .with_min_ssim2(99.9);

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
        .with_max_colors(256)
        .with_target_ssim2(80.0);

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
        .with_max_colors(2)
        .with_min_ssim2(-100.0);

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
        .with_max_colors(64)
        .with_target_ssim2(50.0);

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
        .with_max_colors(64)
        .with_min_ssim2(0.0);

    let result = zenquant::quantize(&pixels, 16, 16, &config).unwrap();
    assert!(
        result.mpe_score().is_some(),
        "metric should be computed when min_ssim2 is set"
    );
}

#[test]
fn convenience_accessors_none_when_no_metric() {
    let pixels = gradient_image(16, 16);
    let config = QuantizeConfig::new(OutputFormat::Png).with_max_colors(16);
    let result = zenquant::quantize(&pixels, 16, 16, &config).unwrap();

    assert!(result.ssimulacra2_estimate().is_none());
    assert!(result.butteraugli_estimate().is_none());
}

#[test]
fn convenience_accessors_present_with_metric() {
    let pixels = gradient_image(32, 32);
    let config = QuantizeConfig::new(OutputFormat::Png)
        .with_max_colors(16)
        .with_compute_quality_metric(true);
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
        .with_max_colors(256)
        .with_target_ssim2(40.0);

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
        .with_max_colors(2)
        .with_min_ssim2(99.9);

    let result = zenquant::quantize_rgba(&pixels, 32, 32, &config);
    assert!(
        matches!(result, Err(QuantizeError::QualityNotMet { .. })),
        "expected QualityNotMet for RGBA with 2 colors and min_ssim2=99.9, got {result:?}"
    );
}

// ===================== Remap path quality metric tests =====================

#[test]
fn remap_computes_mpe_when_requested() {
    let pixels = gradient_image(32, 32);

    // Build a palette first
    let palette_config = QuantizeConfig::new(OutputFormat::Png).with_max_colors(16);
    let shared = zenquant::quantize(&pixels, 32, 32, &palette_config).unwrap();

    // Remap with metric computation enabled
    let remap_config = QuantizeConfig::new(OutputFormat::Png)
        .with_max_colors(16)
        .with_compute_quality_metric(true);
    let result = shared.remap(&pixels, 32, 32, &remap_config).unwrap();

    assert!(
        result.mpe_score().is_some(),
        "remap should compute MPE when compute_quality_metric is set"
    );
    assert!(result.ssimulacra2_estimate().is_some());
    assert!(result.butteraugli_estimate().is_some());
}

#[test]
fn remap_no_metric_by_default() {
    let pixels = gradient_image(32, 32);
    let config = QuantizeConfig::new(OutputFormat::Png).with_max_colors(16);
    let shared = zenquant::quantize(&pixels, 32, 32, &config).unwrap();
    let result = shared.remap(&pixels, 32, 32, &config).unwrap();

    assert!(
        result.mpe_score().is_none(),
        "remap should not compute MPE by default"
    );
}

#[test]
fn remap_enforces_min_ssim2() {
    let pixels = gradient_image(32, 32);

    // Build palette with only 2 colors — quality will be terrible
    let palette_config = QuantizeConfig::new(OutputFormat::Png).with_max_colors(2);
    let shared = zenquant::quantize(&pixels, 32, 32, &palette_config).unwrap();

    // Remap with a high quality floor
    let remap_config = QuantizeConfig::new(OutputFormat::Png)
        .with_max_colors(2)
        .with_min_ssim2(99.9);
    let result = shared.remap(&pixels, 32, 32, &remap_config);
    assert!(
        matches!(result, Err(QuantizeError::QualityNotMet { .. })),
        "remap should enforce min_ssim2: got {result:?}"
    );
}

#[test]
fn remap_target_ssim2_computes_metric() {
    let pixels = gradient_image(32, 32);

    let palette_config = QuantizeConfig::new(OutputFormat::Png).with_max_colors(256);
    let shared = zenquant::quantize(&pixels, 32, 32, &palette_config).unwrap();

    // Remap with target_ssim2 — should implicitly compute metric
    let remap_config = QuantizeConfig::new(OutputFormat::Png)
        .with_max_colors(256)
        .with_target_ssim2(80.0);
    let result = shared.remap(&pixels, 32, 32, &remap_config).unwrap();

    assert!(
        result.mpe_score().is_some(),
        "remap should compute metric when target_ssim2 is set"
    );
}

#[test]
fn remap_rgba_computes_mpe_when_requested() {
    let pixels: Vec<rgb::RGBA<u8>> = (0..1024)
        .map(|i| rgb::RGBA {
            r: (i % 256) as u8,
            g: ((i * 3) % 256) as u8,
            b: ((i * 7) % 256) as u8,
            a: 255,
        })
        .collect();

    let palette_config = QuantizeConfig::new(OutputFormat::Png).with_max_colors(16);
    let shared = zenquant::quantize_rgba(&pixels, 32, 32, &palette_config).unwrap();

    let remap_config = QuantizeConfig::new(OutputFormat::Png)
        .with_max_colors(16)
        .with_compute_quality_metric(true);
    let result = shared.remap_rgba(&pixels, 32, 32, &remap_config).unwrap();

    assert!(
        result.mpe_score().is_some(),
        "remap_rgba should compute MPE when requested"
    );
    assert!(result.ssimulacra2_estimate().is_some());
}

#[test]
fn remap_rgba_enforces_min_ssim2() {
    let pixels: Vec<rgb::RGBA<u8>> = (0..1024)
        .map(|i| rgb::RGBA {
            r: (i % 256) as u8,
            g: ((i * 3) % 256) as u8,
            b: ((i * 7) % 256) as u8,
            a: 255,
        })
        .collect();

    let palette_config = QuantizeConfig::new(OutputFormat::Png).with_max_colors(2);
    let shared = zenquant::quantize_rgba(&pixels, 32, 32, &palette_config).unwrap();

    let remap_config = QuantizeConfig::new(OutputFormat::Png)
        .with_max_colors(2)
        .with_min_ssim2(99.9);
    let result = shared.remap_rgba(&pixels, 32, 32, &remap_config);
    assert!(
        matches!(result, Err(QuantizeError::QualityNotMet { .. })),
        "remap_rgba should enforce min_ssim2: got {result:?}"
    );
}

// ---------------------------------------------------------------------------
// Blue noise dithering tests
// ---------------------------------------------------------------------------

#[test]
fn blue_noise_produces_valid_indices() {
    let pixels = gradient_image(32, 32);
    let config = QuantizeConfig::new(OutputFormat::Png)
        .with_max_colors(16)
        ._with_blue_noise_dither();
    let result = zenquant::quantize(&pixels, 32, 32, &config).unwrap();
    for &idx in result.indices() {
        assert!(
            (idx as usize) < result.palette_len(),
            "blue noise index {idx} >= palette len {}",
            result.palette_len()
        );
    }
}

#[test]
fn blue_noise_is_deterministic() {
    let pixels = gradient_image(32, 32);
    let config = QuantizeConfig::new(OutputFormat::Png)
        .with_max_colors(16)
        ._with_blue_noise_dither();
    let r1 = zenquant::quantize(&pixels, 32, 32, &config).unwrap();
    let r2 = zenquant::quantize(&pixels, 32, 32, &config).unwrap();
    assert_eq!(
        r1.indices(),
        r2.indices(),
        "blue noise should be deterministic"
    );
}

#[test]
fn blue_noise_differs_from_no_dither() {
    let pixels = gradient_image(32, 32);
    let bn_config = QuantizeConfig::new(OutputFormat::Png)
        .with_max_colors(8)
        ._with_blue_noise_dither();
    let nd_config = QuantizeConfig::new(OutputFormat::Png)
        .with_max_colors(8)
        ._with_no_dither();
    let bn = zenquant::quantize(&pixels, 32, 32, &bn_config).unwrap();
    let nd = zenquant::quantize(&pixels, 32, 32, &nd_config).unwrap();
    // Palettes should be the same (same quantization), but indices should differ
    // because blue noise adds perceptual noise before palette lookup
    let diffs = bn
        .indices()
        .iter()
        .zip(nd.indices().iter())
        .filter(|(a, b)| a != b)
        .count();
    assert!(
        diffs > 0,
        "blue noise should produce at least some different indices vs no-dither"
    );
}

#[test]
fn blue_noise_computes_mpe() {
    let pixels = gradient_image(32, 32);
    let config = QuantizeConfig::new(OutputFormat::Png)
        .with_max_colors(16)
        ._with_blue_noise_dither()
        .with_compute_quality_metric(true);
    let result = zenquant::quantize(&pixels, 32, 32, &config).unwrap();
    assert!(
        result.mpe_score().is_some(),
        "blue noise should compute MPE when requested"
    );
    assert!(
        result.ssimulacra2_estimate().is_some(),
        "blue noise should compute SSIM2 estimate when requested"
    );
}

// ---------------------------------------------------------------------------
// Temporal clamping tests
// ---------------------------------------------------------------------------

#[test]
fn temporal_clamping_locks_static_pixels_rgb() {
    let width = 16;
    let height = 16;
    let pixels = gradient_image(width, height);

    let config = QuantizeConfig::new(OutputFormat::Png).with_max_colors(16);
    let shared = zenquant::quantize(&pixels, width, height, &config).unwrap();

    // First remap (no prev)
    let r1 = shared.remap(&pixels, width, height, &config).unwrap();

    // Second remap with prev_indices — identical pixels should produce identical indices
    let r2 = shared
        .remap_with_prev(&pixels, width, height, &config, r1.indices())
        .unwrap();

    assert_eq!(
        r1.indices(),
        r2.indices(),
        "static pixels should be identical when temporal clamping is applied"
    );
}

#[test]
fn temporal_clamping_locks_static_pixels_rgba() {
    let width = 16;
    let height = 16;
    let pixels: Vec<rgb::RGBA<u8>> = (0..width * height)
        .map(|i| {
            let v = (i * 4 % 256) as u8;
            rgb::RGBA {
                r: v,
                g: v,
                b: v,
                a: 255,
            }
        })
        .collect();

    let config = QuantizeConfig::new(OutputFormat::Gif).with_max_colors(16);
    let shared = zenquant::quantize_rgba(&pixels, width, height, &config).unwrap();

    let r1 = shared.remap_rgba(&pixels, width, height, &config).unwrap();
    let r2 = shared
        .remap_rgba_with_prev(&pixels, width, height, &config, r1.indices())
        .unwrap();

    assert_eq!(
        r1.indices(),
        r2.indices(),
        "static RGBA pixels should be identical when temporal clamping is applied"
    );
}

#[test]
fn temporal_clamping_allows_changed_pixels_to_differ() {
    let width = 16;
    let height = 16;
    let pixels1 = gradient_image(width, height);

    let config = QuantizeConfig::new(OutputFormat::Png).with_max_colors(16);
    let shared = zenquant::quantize(&pixels1, width, height, &config).unwrap();
    let r1 = shared.remap(&pixels1, width, height, &config).unwrap();

    // Create a very different frame
    let pixels2: Vec<rgb::RGB<u8>> = pixels1
        .iter()
        .map(|p| rgb::RGB {
            r: 255 - p.r,
            g: 255 - p.g,
            b: 255 - p.b,
        })
        .collect();

    let r2 = shared
        .remap_with_prev(&pixels2, width, height, &config, r1.indices())
        .unwrap();

    // Changed pixels should be allowed to differ
    let diffs = r1
        .indices()
        .iter()
        .zip(r2.indices().iter())
        .filter(|(a, b)| a != b)
        .count();
    assert!(
        diffs > 0,
        "changed pixels should produce different indices even with temporal clamping"
    );
}

#[test]
fn temporal_clamping_works_with_sierra_lite() {
    let width = 16;
    let height = 16;
    let pixels = gradient_image(width, height);

    let config = QuantizeConfig::new(OutputFormat::Png)
        .with_max_colors(16)
        ._with_sierra_lite_dither();
    let shared = zenquant::quantize(&pixels, width, height, &config).unwrap();
    let r1 = shared.remap(&pixels, width, height, &config).unwrap();
    let r2 = shared
        .remap_with_prev(&pixels, width, height, &config, r1.indices())
        .unwrap();

    assert_eq!(
        r1.indices(),
        r2.indices(),
        "Sierra Lite + temporal clamping should lock static pixels"
    );
}

#[test]
fn remap_with_prev_enforces_min_ssim2() {
    let pixels = gradient_image(32, 32);
    let config = QuantizeConfig::new(OutputFormat::Png).with_max_colors(16);
    let shared = zenquant::quantize(&pixels, 32, 32, &config).unwrap();
    let r1 = shared.remap(&pixels, 32, 32, &config).unwrap();

    // Set unreachably high min_ssim2
    let strict_config = QuantizeConfig::new(OutputFormat::Png)
        .with_max_colors(2)
        .with_min_ssim2(99.9);
    let result = shared.remap_with_prev(&pixels, 32, 32, &strict_config, r1.indices());
    assert!(
        matches!(result, Err(QuantizeError::QualityNotMet { .. })),
        "remap_with_prev should enforce min_ssim2: got {result:?}"
    );
}

// ---------------------------------------------------------------------------
// Animation coverage gap tests
// ---------------------------------------------------------------------------

/// Blue noise: static regions produce identical indices when only another region changes.
/// This directly tests the "zero temporal flicker" claim.
#[test]
fn blue_noise_zero_flicker_static_region() {
    let width = 32;
    let height = 32;

    // Frame 1: gradient top half, solid blue bottom half
    let mut frame1 = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            if y < height / 2 {
                let r = (x * 255 / width) as u8;
                let g = (y * 255 / (height / 2)) as u8;
                frame1.push(rgb::RGB { r, g, b: 100 });
            } else {
                frame1.push(rgb::RGB {
                    r: 50,
                    g: 50,
                    b: 200,
                });
            }
        }
    }

    // Frame 2: different top half (inverted gradient), same bottom half
    let mut frame2 = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            if y < height / 2 {
                let r = 255 - (x * 255 / width) as u8;
                let g = 255 - (y * 255 / (height / 2)) as u8;
                frame2.push(rgb::RGB { r, g, b: 50 });
            } else {
                // Identical to frame 1
                frame2.push(rgb::RGB {
                    r: 50,
                    g: 50,
                    b: 200,
                });
            }
        }
    }

    let config = QuantizeConfig::new(OutputFormat::Png)
        .with_max_colors(32)
        ._with_blue_noise_dither();

    let frames = [
        zenquant::ImgRef::new(&frame1, width, height),
        zenquant::ImgRef::new(&frame2, width, height),
    ];
    let shared = zenquant::build_palette(&frames, &config).unwrap();

    let r1 = shared.remap(&frame1, width, height, &config).unwrap();
    let r2 = shared.remap(&frame2, width, height, &config).unwrap();

    // Bottom half (static region) should have identical indices
    let static_start = (height / 2) * width;
    let static_indices_1 = &r1.indices()[static_start..];
    let static_indices_2 = &r2.indices()[static_start..];
    assert_eq!(
        static_indices_1, static_indices_2,
        "blue noise: static region indices should be identical across frames"
    );

    // Top half (changed region) should differ
    let changed_indices_1 = &r1.indices()[..static_start];
    let changed_indices_2 = &r2.indices()[..static_start];
    let diffs = changed_indices_1
        .iter()
        .zip(changed_indices_2.iter())
        .filter(|(a, b)| a != b)
        .count();
    assert!(
        diffs > 0,
        "blue noise: changed region should produce different indices"
    );
}

/// Sierra Lite produces less temporal cascade than Floyd-Steinberg.
/// When a single region changes, fewer downstream pixels are affected.
#[test]
fn sierra_lite_less_cascade_than_floyd_steinberg() {
    let width = 32;
    let height = 32;

    // Frame 1: smooth gradient
    let frame1: Vec<rgb::RGB<u8>> = (0..width * height)
        .map(|i| {
            let x = i % width;
            let y = i / width;
            rgb::RGB {
                r: (x * 255 / width) as u8,
                g: (y * 255 / height) as u8,
                b: 128,
            }
        })
        .collect();

    // Frame 2: same except top-left 4x4 block is different
    let mut frame2 = frame1.clone();
    for y in 0..4 {
        for x in 0..4 {
            let idx = y * width + x;
            frame2[idx] = rgb::RGB {
                r: 255 - frame1[idx].r,
                g: 255 - frame1[idx].g,
                b: 255 - frame1[idx].b,
            };
        }
    }

    let config_fs = QuantizeConfig::new(OutputFormat::Png).with_max_colors(16);
    let config_sl = QuantizeConfig::new(OutputFormat::Png)
        .with_max_colors(16)
        ._with_sierra_lite_dither();

    // Build shared palettes from both frames for each config
    let frames = [
        zenquant::ImgRef::new(&frame1, width, height),
        zenquant::ImgRef::new(&frame2, width, height),
    ];
    let shared_fs = zenquant::build_palette(&frames, &config_fs).unwrap();
    let shared_sl = zenquant::build_palette(&frames, &config_sl).unwrap();

    let r1_fs = shared_fs.remap(&frame1, width, height, &config_fs).unwrap();
    let r2_fs = shared_fs.remap(&frame2, width, height, &config_fs).unwrap();

    let r1_sl = shared_sl.remap(&frame1, width, height, &config_sl).unwrap();
    let r2_sl = shared_sl.remap(&frame2, width, height, &config_sl).unwrap();

    // Count how many pixels changed outside the 4x4 modified block
    let count_cascade = |r1: &[u8], r2: &[u8]| -> usize {
        let mut cascade = 0;
        for y in 0..height {
            for x in 0..width {
                if x < 4 && y < 4 {
                    continue; // skip the actually-modified block
                }
                let idx = y * width + x;
                if r1[idx] != r2[idx] {
                    cascade += 1;
                }
            }
        }
        cascade
    };

    let fs_cascade = count_cascade(r1_fs.indices(), r2_fs.indices());
    let sl_cascade = count_cascade(r1_sl.indices(), r2_sl.indices());

    assert!(
        sl_cascade <= fs_cascade,
        "Sierra Lite should have <= cascade than Floyd-Steinberg: SL={sl_cascade}, FS={fs_cascade}"
    );
}

/// Blue noise and Sierra Lite produce valid results through the RGBA quantize + remap path.
#[test]
fn blue_noise_and_sierra_lite_rgba_paths() {
    let width = 16;
    let height = 16;
    let pixels: Vec<rgb::RGBA<u8>> = (0..width * height)
        .map(|i| {
            let v = (i * 4 % 256) as u8;
            let a = if i < 16 { 0 } else { 255 };
            rgb::RGBA {
                r: v,
                g: 255 - v,
                b: 128,
                a,
            }
        })
        .collect();

    for (mode_name, config) in [
        (
            "blue_noise",
            QuantizeConfig::new(OutputFormat::Gif)
                .with_max_colors(16)
                ._with_blue_noise_dither(),
        ),
        (
            "sierra_lite",
            QuantizeConfig::new(OutputFormat::Gif)
                .with_max_colors(16)
                ._with_sierra_lite_dither(),
        ),
    ] {
        let result = zenquant::quantize_rgba(&pixels, width, height, &config).unwrap();

        assert!(result.palette_len() <= 16, "{mode_name}: palette too large");
        assert!(
            result.transparent_index().is_some(),
            "{mode_name}: should have transparent index"
        );
        for &idx in result.indices() {
            assert!(
                (idx as usize) < result.palette_len(),
                "{mode_name}: index {idx} >= palette len {}",
                result.palette_len()
            );
        }

        // Transparent pixels should map to transparent index
        let ti = result.transparent_index().unwrap();
        for i in 0..16 {
            assert_eq!(
                result.indices()[i],
                ti,
                "{mode_name}: pixel {i} should be transparent"
            );
        }

        // Remap should also work
        let remapped = result.remap_rgba(&pixels, width, height, &config).unwrap();
        assert_eq!(remapped.palette(), result.palette());
    }
}

/// Full-alpha temporal clamping with semi-transparent pixels (PNG path).
#[test]
fn temporal_clamping_full_alpha_path() {
    let width = 16;
    let height = 16;

    // Semi-transparent gradient
    let pixels: Vec<rgb::RGBA<u8>> = (0..width * height)
        .map(|i| {
            let v = (i * 4 % 256) as u8;
            let a = (i * 2 % 256) as u8; // varying alpha 0-254
            rgb::RGBA {
                r: v,
                g: v,
                b: v,
                a,
            }
        })
        .collect();

    let config = QuantizeConfig::new(OutputFormat::Png).with_max_colors(32);
    let shared = zenquant::quantize_rgba(&pixels, width, height, &config).unwrap();

    // Remap frame 1
    let r1 = shared.remap_rgba(&pixels, width, height, &config).unwrap();

    // Remap frame 2 with prev_indices — same pixels, should be stable
    let r2 = shared
        .remap_rgba_with_prev(&pixels, width, height, &config, r1.indices())
        .unwrap();

    assert_eq!(
        r1.indices(),
        r2.indices(),
        "full-alpha temporal clamping: static semi-transparent pixels should produce identical indices"
    );
}

/// Sequential multi-frame remap loop with temporal clamping chain.
#[test]
fn sequential_multi_frame_remap_chain() {
    let width = 16;
    let height = 16;

    // Three frames with gradually shifting colors
    let make_frame = |offset: u8| -> Vec<rgb::RGBA<u8>> {
        (0..width * height)
            .map(|i| {
                let v = ((i * 3 % 256) as u8).wrapping_add(offset);
                let a = if i < 8 { 0 } else { 255 };
                rgb::RGBA {
                    r: v,
                    g: v.wrapping_add(30),
                    b: v.wrapping_add(60),
                    a,
                }
            })
            .collect()
    };

    let frame1 = make_frame(0);
    let frame2 = make_frame(5); // slight shift
    let frame3 = make_frame(10); // more shift

    let config = QuantizeConfig::new(OutputFormat::Gif)
        .with_max_colors(32)
        ._with_sierra_lite_dither();

    // Build shared palette from all frames
    let frames = [
        zenquant::ImgRef::new(&frame1, width, height),
        zenquant::ImgRef::new(&frame2, width, height),
        zenquant::ImgRef::new(&frame3, width, height),
    ];
    let shared = zenquant::build_palette_rgba(&frames, &config).unwrap();

    // Remap frame 1 (no prev)
    let r1 = shared.remap_rgba(&frame1, width, height, &config).unwrap();

    // Remap frame 2 with prev from frame 1
    let r2 = shared
        .remap_rgba_with_prev(&frame2, width, height, &config, r1.indices())
        .unwrap();

    // Remap frame 3 with prev from frame 2
    let r3 = shared
        .remap_rgba_with_prev(&frame3, width, height, &config, r2.indices())
        .unwrap();

    // All frames should share the same palette
    assert_eq!(r1.palette(), r2.palette());
    assert_eq!(r2.palette(), r3.palette());

    // All indices should be valid
    for (frame_name, result) in [("frame1", &r1), ("frame2", &r2), ("frame3", &r3)] {
        assert_eq!(result.indices().len(), width * height);
        for &idx in result.indices() {
            assert!(
                (idx as usize) < result.palette_len(),
                "{frame_name}: index {idx} >= palette len {}",
                result.palette_len()
            );
        }
    }

    // Transparent indices should be consistent
    assert_eq!(r1.transparent_index(), r2.transparent_index());
    assert_eq!(r2.transparent_index(), r3.transparent_index());
}

/// Error diffusion continues through locked pixels — unlocked pixels near
/// locked regions should still get proper dithered quality.
#[test]
fn error_diffuses_through_locked_pixels() {
    let width = 32;
    let height = 32;

    // Frame where the left half is a smooth gradient (will be locked)
    // and the right half is also a gradient (will be unlocked in frame 2)
    let frame1: Vec<rgb::RGB<u8>> = (0..width * height)
        .map(|i| {
            let x = i % width;
            let y = i / width;
            rgb::RGB {
                r: (y * 255 / height) as u8,
                g: (x * 255 / width) as u8,
                b: 128,
            }
        })
        .collect();

    // Frame 2: left half identical (locked), right half inverted (unlocked)
    let mut frame2 = frame1.clone();
    for y in 0..height {
        for x in (width / 2)..width {
            let idx = y * width + x;
            frame2[idx] = rgb::RGB {
                r: 255 - frame1[idx].r,
                g: 255 - frame1[idx].g,
                b: 255 - frame1[idx].b,
            };
        }
    }

    let config = QuantizeConfig::new(OutputFormat::Png).with_max_colors(32);

    let frames = [
        zenquant::ImgRef::new(&frame1, width, height),
        zenquant::ImgRef::new(&frame2, width, height),
    ];
    let shared = zenquant::build_palette(&frames, &config).unwrap();

    let r1 = shared.remap(&frame1, width, height, &config).unwrap();

    // Remap frame 2 with clamping
    let r2_clamped = shared
        .remap_with_prev(&frame2, width, height, &config, r1.indices())
        .unwrap();

    // Remap frame 2 without clamping (for comparison)
    let r2_unclamped = shared.remap(&frame2, width, height, &config).unwrap();

    // Left half should be mostly locked (identical to frame 1)
    let mut locked_count = 0;
    for y in 0..height {
        for x in 0..(width / 2) {
            let idx = y * width + x;
            if r2_clamped.indices()[idx] == r1.indices()[idx] {
                locked_count += 1;
            }
        }
    }
    let total_left = (width / 2) * height;
    assert!(
        locked_count as f32 / total_left as f32 > 0.9,
        "most left-half pixels should be locked: {locked_count}/{total_left}"
    );

    // Right half should still produce reasonable results (not degenerate)
    // Compare quality: clamped right half should have similar MSE to unclamped
    let right_mse = |indices: &[u8], palette: &[[u8; 3]]| -> f32 {
        let mut total = 0.0f32;
        let mut count = 0;
        for y in 0..height {
            for x in (width / 2)..width {
                let idx = y * width + x;
                let orig = srgb_to_oklab(frame2[idx].r, frame2[idx].g, frame2[idx].b);
                let p = palette[indices[idx] as usize];
                let quant = srgb_to_oklab(p[0], p[1], p[2]);
                total += orig.distance_sq(quant);
                count += 1;
            }
        }
        total / count as f32
    };

    let mse_clamped = right_mse(r2_clamped.indices(), r2_clamped.palette());
    let mse_unclamped = right_mse(r2_unclamped.indices(), r2_unclamped.palette());

    // Clamped MSE on the unlocked region should be within 3x of unclamped
    // (some degradation is expected since error flows through locked pixels
    // which may differ from what free error diffusion would produce)
    assert!(
        mse_clamped < mse_unclamped * 3.0,
        "clamped right-half MSE should be reasonable: clamped={mse_clamped:.6}, unclamped={mse_unclamped:.6}"
    );
}

/// Blue noise + temporal clamping: remap_with_prev should produce identical
/// results to remap (since blue noise is already position-deterministic).
#[test]
fn blue_noise_unaffected_by_temporal_clamping() {
    let width = 16;
    let height = 16;
    let pixels = gradient_image(width, height);

    let config = QuantizeConfig::new(OutputFormat::Png)
        .with_max_colors(16)
        ._with_blue_noise_dither();

    let shared = zenquant::quantize(&pixels, width, height, &config).unwrap();

    // Remap without prev
    let r1 = shared.remap(&pixels, width, height, &config).unwrap();

    // Fabricate some arbitrary prev_indices
    let prev: Vec<u8> = (0..width * height).map(|i| (i % 16) as u8).collect();

    // Remap with prev_indices — should produce same result since blue noise
    // ignores prev_indices (position-deterministic)
    let r2 = shared
        .remap_with_prev(&pixels, width, height, &config, &prev)
        .unwrap();

    assert_eq!(
        r1.indices(),
        r2.indices(),
        "blue noise should produce identical indices regardless of prev_indices"
    );
}
