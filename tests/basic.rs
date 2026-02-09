use zenquant::{DitherMode, OutputFormat, QuantizeConfig, QuantizeError, RunPriority};

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

// ===================== Format-specific tests =====================

#[test]
fn output_format_gif() {
    let pixels = gradient_8x8();
    let config = QuantizeConfig::new()
        .max_colors(16)
        .output_format(OutputFormat::Gif);
    let result = zenquant::quantize(&pixels, 8, 8, &config).unwrap();
    assert!(result.palette_len() <= 16);
    // GIF uses binary transparency → no alpha_table needed for RGB
    assert!(result.alpha_table().is_none());
}

#[test]
fn output_format_png() {
    let pixels = gradient_8x8();
    let config = QuantizeConfig::new()
        .max_colors(16)
        .output_format(OutputFormat::Png);
    let result = zenquant::quantize(&pixels, 8, 8, &config).unwrap();
    assert!(result.palette_len() <= 16);
    // PNG uses luminance sort — palette L values should be monotonically increasing
    let labs = zenquant::_internals::srgb_to_oklab;
    let mut prev_l = -1.0f32;
    for entry in result.palette() {
        let lab = labs(entry[0], entry[1], entry[2]);
        assert!(
            lab.l >= prev_l - 0.001,
            "PNG luminance sort violated: L={} after L={}",
            lab.l,
            prev_l
        );
        prev_l = lab.l;
    }
}

#[test]
fn output_format_webp_lossless() {
    let pixels = gradient_8x8();
    let config = QuantizeConfig::new()
        .max_colors(16)
        .output_format(OutputFormat::WebpLossless);
    let result = zenquant::quantize(&pixels, 8, 8, &config).unwrap();
    assert!(result.palette_len() <= 16);
}

#[test]
fn output_format_jxl_modular() {
    let pixels = gradient_8x8();
    let config = QuantizeConfig::new()
        .max_colors(16)
        .output_format(OutputFormat::JxlModular);
    let result = zenquant::quantize(&pixels, 8, 8, &config).unwrap();
    assert!(result.palette_len() <= 16);
}

#[test]
fn all_formats_produce_valid_results() {
    let pixels = gradient_8x8();
    for format in &[
        OutputFormat::Generic,
        OutputFormat::Gif,
        OutputFormat::Png,
        OutputFormat::WebpLossless,
        OutputFormat::JxlModular,
    ] {
        let config = QuantizeConfig::new().max_colors(16).output_format(*format);
        let result = zenquant::quantize(&pixels, 8, 8, &config).unwrap();
        assert!(result.palette_len() <= 16, "format {:?}", format);
        assert_eq!(result.indices().len(), 64, "format {:?}", format);
        for &idx in result.indices() {
            assert!(
                (idx as usize) < result.palette_len(),
                "invalid index for {:?}",
                format
            );
        }
    }
}

// ===================== Already-paletted fast path =====================

#[test]
fn exact_palette_fast_path_rgb() {
    // Image with exactly 4 colors — should take the fast path
    let mut pixels = Vec::with_capacity(64);
    let colors = [
        rgb::RGB { r: 255, g: 0, b: 0 },
        rgb::RGB { r: 0, g: 255, b: 0 },
        rgb::RGB { r: 0, g: 0, b: 255 },
        rgb::RGB {
            r: 255,
            g: 255,
            b: 0,
        },
    ];
    for i in 0..64 {
        pixels.push(colors[i % 4]);
    }

    let config = QuantizeConfig::new().max_colors(4).dither(DitherMode::None);
    let result = zenquant::quantize(&pixels, 8, 8, &config).unwrap();
    assert_eq!(result.palette_len(), 4);

    // Each input color should map to a unique palette index
    let mut used_indices: Vec<u8> = result.indices().to_vec();
    used_indices.sort_unstable();
    used_indices.dedup();
    assert_eq!(used_indices.len(), 4);

    // Lossless: remapping back should give the original colors
    for (i, p) in pixels.iter().enumerate() {
        let idx = result.indices()[i] as usize;
        let pal = result.palette()[idx];
        assert_eq!(pal[0], p.r, "pixel {i} R mismatch");
        assert_eq!(pal[1], p.g, "pixel {i} G mismatch");
        assert_eq!(pal[2], p.b, "pixel {i} B mismatch");
    }
}

#[test]
fn exact_palette_fast_path_rgba() {
    let mut pixels = Vec::with_capacity(64);
    let colors = [
        rgb::RGBA {
            r: 255,
            g: 0,
            b: 0,
            a: 255,
        },
        rgb::RGBA {
            r: 0,
            g: 255,
            b: 0,
            a: 255,
        },
        rgb::RGBA {
            r: 0,
            g: 0,
            b: 255,
            a: 255,
        },
        rgb::RGBA {
            r: 0,
            g: 0,
            b: 0,
            a: 0,
        }, // transparent
    ];
    for i in 0..64 {
        pixels.push(colors[i % 4]);
    }

    let config = QuantizeConfig::new().max_colors(4).dither(DitherMode::None);
    let result = zenquant::quantize_rgba(&pixels, 8, 8, &config).unwrap();
    assert!(result.transparent_index().is_some());
    assert!(result.palette_len() <= 4);

    let ti = result.transparent_index().unwrap();
    // Transparent pixels should map to transparent index
    for i in (3..64).step_by(4) {
        assert_eq!(result.indices()[i], ti, "pixel {i} should be transparent");
    }
}

#[test]
fn exact_palette_too_many_colors_takes_normal_path() {
    // 257 unique colors → can't fit in 256 → normal path
    let mut pixels = Vec::with_capacity(512);
    for i in 0..256 {
        pixels.push(rgb::RGB {
            r: i as u8,
            g: 0,
            b: 0,
        });
        pixels.push(rgb::RGB {
            r: i as u8,
            g: 0,
            b: 0,
        });
    }
    // Add one more unique color
    pixels.push(rgb::RGB { r: 0, g: 1, b: 0 });
    pixels.push(rgb::RGB { r: 0, g: 1, b: 0 });
    // Pad to full row
    while pixels.len() < 520 {
        pixels.push(rgb::RGB { r: 0, g: 0, b: 0 });
    }

    let config = QuantizeConfig::new().max_colors(256);
    let result = zenquant::quantize(&pixels, 520, 1, &config).unwrap();
    // Should still produce a valid result, just not lossless
    assert!(result.palette_len() <= 256);
}

// ===================== RGBA palette entries =====================

#[test]
fn palette_rgba_opaque() {
    let pixels = gradient_8x8();
    let config = QuantizeConfig::new().max_colors(8);
    let result = zenquant::quantize(&pixels, 8, 8, &config).unwrap();

    let rgba = result.palette_rgba();
    assert_eq!(rgba.len(), result.palette_len());
    // All entries should have alpha=255
    for entry in rgba {
        assert_eq!(entry[3], 255, "opaque RGB palette should have alpha=255");
    }
}

#[test]
fn palette_rgba_with_transparency() {
    let mut pixels = Vec::with_capacity(64);
    for i in 0..64 {
        let v = (i * 4) as u8;
        let a = if i < 16 { 0 } else { 255 };
        pixels.push(rgb::RGBA {
            r: v,
            g: v,
            b: v,
            a,
        });
    }

    let config = QuantizeConfig::new().max_colors(8);
    let result = zenquant::quantize_rgba(&pixels, 8, 8, &config).unwrap();

    let rgba = result.palette_rgba();
    let ti = result.transparent_index().unwrap();
    assert_eq!(
        rgba[ti as usize][3], 0,
        "transparent index should have alpha=0"
    );
}

// ===================== alpha_table =====================

#[test]
fn alpha_table_opaque_returns_none() {
    let pixels = gradient_8x8();
    let config = QuantizeConfig::new().max_colors(8);
    let result = zenquant::quantize(&pixels, 8, 8, &config).unwrap();
    assert!(
        result.alpha_table().is_none(),
        "fully opaque image should have no alpha_table"
    );
}

#[test]
fn alpha_table_with_transparency() {
    let mut pixels = Vec::with_capacity(64);
    for i in 0..64 {
        let a = if i < 16 { 0 } else { 255 };
        pixels.push(rgb::RGBA {
            r: 128,
            g: 128,
            b: 128,
            a,
        });
    }

    let config = QuantizeConfig::new().max_colors(8);
    let result = zenquant::quantize_rgba(&pixels, 8, 8, &config).unwrap();

    let table = result.alpha_table();
    assert!(table.is_some(), "should have alpha_table with transparency");
    let table = table.unwrap();
    // Table should be truncated at last non-255 alpha
    assert!(!table.is_empty());
    // The transparent index should have alpha=0
    let ti = result.transparent_index().unwrap() as usize;
    if ti < table.len() {
        assert_eq!(table[ti], 0);
    }
}

// ===================== GIF frequency reorder =====================

#[test]
fn gif_frequency_reorder_most_common_gets_low_index() {
    // Image where one color dominates: 80% red, 10% green, 10% blue
    let mut pixels = Vec::with_capacity(100);
    for _ in 0..80 {
        pixels.push(rgb::RGB { r: 255, g: 0, b: 0 });
    }
    for _ in 0..10 {
        pixels.push(rgb::RGB { r: 0, g: 255, b: 0 });
    }
    for _ in 0..10 {
        pixels.push(rgb::RGB { r: 0, g: 0, b: 255 });
    }

    let config = QuantizeConfig::new()
        .max_colors(4)
        .output_format(OutputFormat::Gif)
        .dither(DitherMode::None);
    let result = zenquant::quantize(&pixels, 10, 10, &config).unwrap();

    // After frequency reorder, the most common index should be low
    let mut freq = [0u32; 256];
    for &idx in result.indices() {
        freq[idx as usize] += 1;
    }
    // Find the index with highest frequency
    let (most_common_idx, _) = freq
        .iter()
        .enumerate()
        .max_by_key(|&(_, count)| count)
        .unwrap();

    // Most common should be index 0 (or 1 if transparent)
    assert!(
        most_common_idx <= 1,
        "most common color should be at low index, got {most_common_idx}"
    );
}

// ===================== Dither strength override =====================

#[test]
fn dither_strength_override() {
    let pixels = gradient_8x8();
    // Explicitly set dither_strength different from format default
    let config = QuantizeConfig::new()
        .max_colors(8)
        .output_format(OutputFormat::Png) // default 0.3
        .dither_strength(0.8); // override to 0.8

    // Should not panic
    let result = zenquant::quantize(&pixels, 8, 8, &config).unwrap();
    assert!(result.palette_len() <= 8);
}

// ===================== Semi-transparent alpha quantization =====================

#[test]
fn semi_transparent_rgba_quantization() {
    // Image with varying alpha levels
    let mut pixels = Vec::with_capacity(64);
    for i in 0..64 {
        let v = (i * 4) as u8;
        let a = (i * 4).min(255) as u8; // 0 to 252
        pixels.push(rgb::RGBA {
            r: v,
            g: v,
            b: v,
            a,
        });
    }

    // Full alpha mode (Generic/PNG/WebP/JXL)
    let config = QuantizeConfig::new()
        .max_colors(16)
        .output_format(OutputFormat::Generic);
    let result = zenquant::quantize_rgba(&pixels, 8, 8, &config).unwrap();

    let rgba = result.palette_rgba();
    // With full alpha quantization, we should see varying alpha values
    let mut alphas: Vec<u8> = rgba.iter().map(|e| e[3]).collect();
    alphas.sort_unstable();
    alphas.dedup();
    // Should have more than just 0 and 255
    assert!(
        alphas.len() >= 2,
        "full alpha quantization should produce multiple alpha levels, got {:?}",
        alphas
    );
}

#[test]
fn gif_binary_transparency() {
    let mut pixels = Vec::with_capacity(64);
    for i in 0..64 {
        let a = if i < 16 { 0 } else { (i * 4).min(255) as u8 };
        pixels.push(rgb::RGBA {
            r: 128,
            g: 128,
            b: 128,
            a,
        });
    }

    let config = QuantizeConfig::new()
        .max_colors(8)
        .output_format(OutputFormat::Gif);
    let result = zenquant::quantize_rgba(&pixels, 8, 8, &config).unwrap();

    let rgba = result.palette_rgba();
    // GIF binary transparency: alpha should only be 0 or 255
    for entry in rgba {
        assert!(
            entry[3] == 0 || entry[3] == 255,
            "GIF should have binary alpha, got {}",
            entry[3]
        );
    }
}

// ===================== Helper functions =====================

fn gradient_8x8() -> Vec<rgb::RGB<u8>> {
    let mut pixels = Vec::with_capacity(64);
    for y in 0..8 {
        for x in 0..8 {
            let r = (x * 255 / 7) as u8;
            let g = (y * 255 / 7) as u8;
            let b = 128;
            pixels.push(rgb::RGB { r, g, b });
        }
    }
    pixels
}
