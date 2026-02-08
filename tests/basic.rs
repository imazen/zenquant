use zenquant::{DitherMode, QuantizeConfig, QuantizeError, RunPriority};

#[test]
fn smoke_test_rgb() {
    let width = 32;
    let height = 32;
    let mut pixels = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            let r = (x * 255 / width) as u8;
            let g = (y * 255 / height) as u8;
            let b = 128u8;
            pixels.push(rgb::RGB { r, g, b });
        }
    }

    let config = QuantizeConfig::default();
    let result = zenquant::quantize(&pixels, width, height, &config).unwrap();

    assert!(result.palette_len() <= 256);
    assert!(result.palette_len() >= 2);
    assert_eq!(result.indices().len(), width * height);
    assert!(result.transparent_index().is_none());

    // All indices should be valid
    for &idx in result.indices() {
        assert!((idx as usize) < result.palette_len());
    }
}

#[test]
fn smoke_test_rgba_with_transparency() {
    let width = 16;
    let height = 16;
    let mut pixels = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            let r = (x * 255 / width) as u8;
            let g = (y * 255 / height) as u8;
            // Top-left quadrant is transparent
            let a = if x < 8 && y < 8 { 0 } else { 255 };
            pixels.push(rgb::RGBA { r, g, b: 128, a });
        }
    }

    let config = QuantizeConfig::default();
    let result = zenquant::quantize_rgba(&pixels, width, height, &config).unwrap();

    assert!(result.palette_len() <= 256);
    assert!(result.transparent_index().is_some());

    let ti = result.transparent_index().unwrap();
    // Transparent pixels should map to transparent index
    for y in 0..8 {
        for x in 0..8 {
            assert_eq!(result.indices()[y * width + x], ti);
        }
    }
}

#[test]
fn all_config_modes() {
    let width = 8;
    let height = 8;
    let pixels: Vec<rgb::RGB<u8>> = (0..64)
        .map(|i| {
            let v = (i * 4) as u8;
            rgb::RGB { r: v, g: v, b: v }
        })
        .collect();

    // Test all combinations
    for dither in &[DitherMode::None, DitherMode::Full, DitherMode::Adaptive] {
        for rp in &[
            RunPriority::Quality,
            RunPriority::Balanced,
            RunPriority::Compression,
        ] {
            let config = QuantizeConfig::new()
                .max_colors(8)
                .quality(75)
                .dither(*dither)
                .run_priority(*rp);

            let result = zenquant::quantize(&pixels, width, height, &config).unwrap();
            assert!(result.palette_len() <= 8, "mode {:?}/{:?}", dither, rp);
            assert_eq!(result.indices().len(), 64);
        }
    }
}

#[test]
fn error_zero_dimension() {
    let pixels = vec![rgb::RGB { r: 0, g: 0, b: 0 }];
    let config = QuantizeConfig::default();

    assert!(matches!(
        zenquant::quantize(&pixels, 0, 1, &config),
        Err(QuantizeError::ZeroDimension)
    ));
    assert!(matches!(
        zenquant::quantize(&pixels, 1, 0, &config),
        Err(QuantizeError::ZeroDimension)
    ));
}

#[test]
fn error_dimension_mismatch() {
    let pixels = vec![rgb::RGB { r: 0, g: 0, b: 0 }; 10];
    let config = QuantizeConfig::default();

    assert!(matches!(
        zenquant::quantize(&pixels, 4, 4, &config),
        Err(QuantizeError::DimensionMismatch { .. })
    ));
}

#[test]
fn error_invalid_max_colors() {
    let pixels = vec![rgb::RGB { r: 0, g: 0, b: 0 }; 4];
    assert!(matches!(
        zenquant::quantize(&pixels, 2, 2, &QuantizeConfig::new().max_colors(1)),
        Err(QuantizeError::InvalidMaxColors(1))
    ));
    assert!(matches!(
        zenquant::quantize(&pixels, 2, 2, &QuantizeConfig::new().max_colors(257)),
        Err(QuantizeError::InvalidMaxColors(257))
    ));
}

#[test]
fn error_invalid_quality() {
    let pixels = vec![rgb::RGB { r: 0, g: 0, b: 0 }; 4];
    assert!(matches!(
        zenquant::quantize(&pixels, 2, 2, &QuantizeConfig::new().quality(101)),
        Err(QuantizeError::InvalidQuality(101))
    ));
}

#[test]
fn single_color_image() {
    let pixels = vec![
        rgb::RGB {
            r: 128,
            g: 128,
            b: 128
        };
        64
    ];
    let config = QuantizeConfig::new().max_colors(4);
    let result = zenquant::quantize(&pixels, 8, 8, &config).unwrap();

    // Should produce a small palette
    assert!(result.palette_len() <= 4);

    // All indices should be the same
    let first = result.indices()[0];
    for &idx in result.indices() {
        assert_eq!(idx, first);
    }
}

#[test]
fn two_color_image() {
    let mut pixels = Vec::with_capacity(64);
    for i in 0..64 {
        if i < 32 {
            pixels.push(rgb::RGB { r: 0, g: 0, b: 0 });
        } else {
            pixels.push(rgb::RGB {
                r: 255,
                g: 255,
                b: 255,
            });
        }
    }

    let config = QuantizeConfig::new().max_colors(2).dither(DitherMode::None);
    let result = zenquant::quantize(&pixels, 8, 8, &config).unwrap();
    assert_eq!(result.palette_len(), 2);

    // Black pixels should all map to the same index
    let black_idx = result.indices()[0];
    for &idx in &result.indices()[..32] {
        assert_eq!(idx, black_idx);
    }
    // White pixels should all map to a different index
    let white_idx = result.indices()[32];
    assert_ne!(black_idx, white_idx);
    for &idx in &result.indices()[32..] {
        assert_eq!(idx, white_idx);
    }
}

#[test]
fn low_quality_no_refinement() {
    let pixels: Vec<rgb::RGB<u8>> = (0..256)
        .map(|i| {
            let v = i as u8;
            rgb::RGB { r: v, g: v, b: v }
        })
        .collect();

    let config = QuantizeConfig::new().quality(10).max_colors(16);
    let result = zenquant::quantize(&pixels, 16, 16, &config).unwrap();
    assert!(result.palette_len() <= 16);
}
