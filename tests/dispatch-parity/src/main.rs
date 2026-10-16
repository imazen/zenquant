//! Independent published-output parity and paired unused-report cost probe.
macro_rules! config {
    ($k:ident,$fmt:expr,$quality:expr,$dither:expr,$target:expr,$colors:expr) => {{
        let f = match $fmt {
            0 => $k::OutputFormat::Png,
            1 => $k::OutputFormat::PngJoint,
            _ => $k::OutputFormat::PngMinSize,
        };
        let q = match $quality {
            0 => $k::Quality::Fast,
            1 => $k::Quality::Balanced,
            _ => $k::Quality::Best,
        };
        let c = $k::QuantizeConfig::new(f)
            .with_quality(q)
            .with_max_colors($colors);
        let c = match $dither {
            0 => c._with_no_dither(),
            1 => c._with_adaptive_dither(),
            2 => c._with_sierra_lite_dither(),
            3 => c._with_blue_noise_dither(),
            4 => c._with_linear_dither(),
            _ => c._with_ordered_dither(),
        };
        if let Some(t) = $target {
            c.with_target_ssim2(t)
        } else {
            c
        }
    }};
}
fn image(w: usize, h: usize, kind: u8) -> Vec<rgb::RGBA<u8>> {
    let mut state = 17u32;
    (0..w * h)
        .map(|i| {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            let (r, g, b) = match kind {
                0 => (((i % w) * 255 / w) as u8, ((i / w) * 255 / h) as u8, 128),
                1 => ((state >> 24) as u8, (state >> 16) as u8, (state >> 8) as u8),
                _ => {
                    if i % 2 == 0 {
                        (0, 0, 0)
                    } else {
                        (255, 255, 255)
                    }
                }
            };
            rgb::RGBA::new(r, g, b, 255)
        })
        .collect()
}
fn parity() {
    let mut count = 0;
    for (w, h) in [(8, 8), (37, 19)] {
        for kind in 0..3 {
            let p = image(w, h, kind);
            let rgb: Vec<_> = p.iter().map(|c| rgb::RGB::new(c.r, c.g, c.b)).collect();
            for fmt in 0..3 {
                for quality in 0..3 {
                    for dither in 0..6 {
                        for target in [None, Some(95.), Some(85.), Some(75.), Some(65.), Some(60.)]
                        {
                            let c = config!(candidate, fmt, quality, dither, target, 8);
                            let r = config!(reference, fmt, quality, dither, target, 8);
                            let normal = candidate::quantize_rgba(&p, w, h, &c).unwrap();
                            let (actual, report) = candidate::quantize_rgba_with_dispatch(
                                &p,
                                w,
                                h,
                                &c,
                                &enough::Unstoppable,
                            )
                            .unwrap();
                            let expected = reference::quantize_rgba(&p, w, h, &r).unwrap();
                            assert_eq!(
                                actual.indices(),
                                expected.indices(),
                                "rgba indices {w} {kind} {fmt} {quality} {dither} {target:?}"
                            );
                            assert_eq!(
                                actual.palette_rgba(),
                                expected.palette_rgba(),
                                "rgba palette"
                            );
                            assert_eq!(actual.indices(), normal.indices());
                            assert_eq!(actual.palette_rgba(), normal.palette_rgba());
                            let (a, _) = candidate::quantize_with_dispatch(
                                &rgb,
                                w,
                                h,
                                &c,
                                &enough::Unstoppable,
                            )
                            .unwrap();
                            let e = reference::quantize(&rgb, w, h, &r).unwrap();
                            assert_eq!(
                                a.indices(),
                                e.indices(),
                                "rgb indices {w} {kind} {fmt} {quality} {dither} {target:?}"
                            );
                            assert_eq!(a.palette_rgba(), e.palette_rgba(), "rgb palette");
                            assert_eq!(
                                report.joint_used,
                                cfg!(feature = "joint") && fmt != 0 && !report.exact_palette
                            );
                            if quality == 0 && dither == 1 && target.is_none() {
                                assert_eq!(report.dither, "Ordered");
                            }
                            if kind == 2 {
                                assert!(
                                    report.exact_palette
                                        && !report.dither_used
                                        && !report.joint_used
                                );
                            }
                            count += 1;
                        }
                    }
                }
            }
        }
    }
    println!(
        "published_0_1_3_parity PASS cases={count} rgb_and_rgba=true joint={}",
        cfg!(feature = "joint")
    );
}
fn main() {
    parity()
}
