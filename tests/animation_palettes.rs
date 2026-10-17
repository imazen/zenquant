use rgb::RGBA;
use zenquant::{ImgRef, OutputFormat, QuantizeConfig, QuantizeResult};

fn assert_valid(result: &QuantizeResult, count: usize, max: usize) {
    assert!(!result.palette().is_empty());
    assert!(result.palette_len() <= max);
    assert_eq!(result.indices().len(), count);
    for &i in result.indices() {
        assert!((i as usize) < result.palette_len());
    }
}

#[test]
fn transparent_only_frames_have_a_real_palette_entry() {
    let pixels: Vec<_> = (0..35).map(|i| RGBA::new(i, 255 - i, 17, 0)).collect();
    for format in [
        OutputFormat::Gif,
        OutputFormat::Png,
        OutputFormat::WebpLossless,
    ] {
        let config = QuantizeConfig::new(format);
        let single = zenquant::quantize_rgba(&pixels, 7, 5, &config).unwrap();
        let shared = zenquant::build_palette_rgba(&[ImgRef::new(&pixels, 7, 5)], &config)
            .unwrap()
            .remap_rgba(&pixels, 7, 5, &config)
            .unwrap();
        for result in [single, shared] {
            assert_valid(&result, pixels.len(), 1);
            assert_eq!(result.transparent_index(), Some(0));
            assert_eq!(result.palette_rgba(), &[[0, 0, 0, 0]]);
            assert!(result.indices().iter().all(|&i| i == 0));
        }
    }
}

#[test]
fn exact_palettes_preserve_distinct_alpha_for_the_same_rgb() {
    let pixels = [0, 1, 32, 127, 128, 254, 255].map(|a| RGBA::new(64, 128, 192, a));
    for format in [OutputFormat::Png, OutputFormat::WebpLossless] {
        let config = QuantizeConfig::new(format);
        let single = zenquant::quantize_rgba(&pixels, 7, 1, &config).unwrap();
        let shared = zenquant::build_palette_rgba(&[ImgRef::new(&pixels, 7, 1)], &config)
            .unwrap()
            .remap_rgba(&pixels, 7, 1, &config)
            .unwrap();
        for result in [single, shared] {
            assert_valid(&result, 7, 7);
            for (p, &idx) in pixels.iter().zip(result.indices()) {
                let actual = result.palette_rgba()[idx as usize];
                assert_eq!(
                    actual[3], p.a,
                    "{format:?}: alpha must survive the exact path"
                );
                if p.a != 0 {
                    assert_eq!(actual, [p.r, p.g, p.b, p.a]);
                }
            }
        }
    }
}

#[test]
fn transparency_consumes_a_slot_even_when_it_is_the_last_pixel() {
    for max in [2, 4, 256] {
        let mut pixels: Vec<_> = (0..max)
            .map(|i| RGBA::new(i as u8, (255 - i) as u8, 64, 255))
            .collect();
        pixels.push(RGBA::new(9, 18, 27, 0));
        for format in [
            OutputFormat::Gif,
            OutputFormat::Png,
            OutputFormat::WebpLossless,
        ] {
            let config = QuantizeConfig::new(format).with_max_colors(max as u32);
            let single = zenquant::quantize_rgba(&pixels, pixels.len(), 1, &config).unwrap();
            let shared =
                zenquant::build_palette_rgba(&[ImgRef::new(&pixels, pixels.len(), 1)], &config)
                    .unwrap()
                    .remap_rgba(&pixels, pixels.len(), 1, &config)
                    .unwrap();
            for result in [single, shared] {
                assert_valid(&result, pixels.len(), max);
                let idx = *result.indices().last().unwrap() as usize;
                assert_eq!(result.palette_rgba()[idx][3], 0, "{format:?} max={max}");
            }
        }
    }
}
