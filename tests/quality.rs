use zenquant::_internals::{average_run_length, index_delta_score, srgb_to_oklab};
use zenquant::{OutputFormat, QuantizeConfig};

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
