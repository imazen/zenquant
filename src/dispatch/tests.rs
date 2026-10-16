use super::*;
use alloc::{format, vec::Vec};
#[derive(Default)]
struct Observer {
    capture: Capture,
    dither: Option<(&'static str, &'static str, f32)>,
    run: Option<(f32, bool)>,
    joint: Option<f32>,
}
impl Sink for Observer {
    fn resolved(&mut self, c: &QuantizeConfig, t: &QuantizeTuning) {
        self.capture.resolved(c, t);
    }
    fn exact(&mut self) {
        self.capture.exact();
    }
    fn masking(&mut self, u: bool, i: usize) {
        self.capture.masking(u, i);
    }
    fn dither(&mut self, p: &dither::DitherParams<'_>) {
        // A distinct test observer reads the actual kernel argument, rather
        // than deriving an expectation from the requested configuration.
        self.dither = Some((
            match p.mode {
                dither::DitherMode::None => "None",
                dither::DitherMode::Adaptive => "Adaptive",
                dither::DitherMode::SierraLite => "SierraLite",
                dither::DitherMode::BlueNoise => "BlueNoise",
                dither::DitherMode::Linear => "Linear",
                dither::DitherMode::Ordered => "Ordered",
            },
            match p.run_priority {
                remap::RunPriority::Quality => "Quality",
                remap::RunPriority::Balanced => "Balanced",
                remap::RunPriority::Compression => "Compression",
            },
            p.dither_strength,
        ));
        self.capture.dither(p);
    }
    fn refinement(&mut self, l: f32, v: bool) {
        self.run = Some((l, v));
        self.capture.refinement(l, v);
    }
    #[cfg(feature = "joint")]
    fn joint(&mut self, t: f32) {
        self.joint = Some(t);
        self.capture.joint(t);
    }
}
fn modes(c: QuantizeConfig, d: usize) -> QuantizeConfig {
    match d {
        0 => c._with_no_dither(),
        1 => c._with_adaptive_dither(),
        2 => c._with_sierra_lite_dither(),
        3 => c._with_blue_noise_dither(),
        4 => c._with_linear_dither(),
        _ => c._with_ordered_dither(),
    }
}
#[test]
fn report_observes_actual_kernel_parameters_across_full_dispatch_grid() {
    let pixels: Vec<_> = (0..128)
        .map(|i| rgb::RGBA::new((i * 17) as u8, (i * 53) as u8, (i * 101) as u8, 255))
        .collect();
    let rgb: Vec<_> = pixels
        .iter()
        .map(|p| rgb::RGB::new(p.r, p.g, p.b))
        .collect();
    for f in [
        OutputFormat::Png,
        OutputFormat::PngJoint,
        OutputFormat::PngMinSize,
        OutputFormat::Gif,
        OutputFormat::WebpLossless,
    ] {
        for q in [Quality::Fast, Quality::Balanced, Quality::Best] {
            for d in 0..6 {
                for target in [None, Some(95.), Some(85.), Some(75.), Some(65.), Some(60.)] {
                    let mut c = modes(QuantizeConfig::new(f).with_quality(q).with_max_colors(4), d);
                    if let Some(t) = target {
                        c = c.with_target_ssim2(t);
                    }
                    for rgba in [false, true] {
                        let mut observed = Observer::default();
                        if rgba {
                            crate::quantize_rgba_impl::<Observer>(
                                &pixels,
                                16,
                                8,
                                &c,
                                &enough::Unstoppable,
                                &mut observed,
                            )
                            .unwrap()
                        } else {
                            crate::quantize_impl::<Observer>(
                                &rgb,
                                16,
                                8,
                                &c,
                                &enough::Unstoppable,
                                &mut observed,
                            )
                            .unwrap()
                        };
                        let report = observed.capture.0.unwrap();
                        let (dither, priority, strength) = observed.dither.unwrap();
                        assert_eq!(
                            (report.dither, report.run_priority, report.dither_strength),
                            (dither, priority, strength)
                        );
                        assert_eq!(report.dither_used, observed.dither.is_some());
                        assert_eq!(report.joint_used, observed.joint.is_some());
                        assert_eq!(report.joint_tolerance, observed.joint);
                        assert_eq!(
                            report.run_refinement,
                            observed
                                .run
                                .map(|(_, v)| if v { "viterbi" } else { "run-extend" })
                        );
                        assert_eq!(report.run_lambda, observed.run.map_or(0., |(l, _)| l));
                        assert!(!report.joint_deflate_effort_used);
                        assert_eq!(report.metric, target.is_some());
                        if q == Quality::Fast && d == 1 && target.is_none() {
                            assert_eq!(report.dither, "Ordered");
                        }
                        if let Some(t) = target {
                            let tier = if t > 90. {
                                "best-quality"
                            } else if t > 82. {
                                "best-balanced"
                            } else if t > 74. {
                                "best-compression"
                            } else if t > 60. {
                                "balanced-compression"
                            } else {
                                "fast-compression"
                            };
                            assert_eq!(report.target_tier, Some(tier));
                        }
                    }
                }
            }
        }
    }
}
#[test]
fn exact_palette_bypass_and_target_strength_scaling_are_explicit() {
    let pixels = alloc::vec![rgb::RGBA::new(1,2,3,255);8];
    let c = QuantizeConfig::new(OutputFormat::PngJoint)
        .with_target_ssim2(50.)
        ._with_dither_strength(0.5);
    let (_, r) =
        crate::quantize_rgba_with_dispatch(&pixels, 4, 2, &c, &enough::Unstoppable).unwrap();
    assert!(r.exact_palette && r.metric && !r.dither_used && !r.masking && !r.joint_used);
    assert_eq!(r.dither_strength, 0.3);
    assert_eq!(r.kmeans_iterations, 0);
    assert_eq!(r.run_refinement, None);
    assert_eq!(format!("{:?}", r.quality), "Fast");
}

#[test]
fn quality_floor_failure_retains_actual_dispatch_but_early_error_does_not() {
    let p: Vec<_> = (0..128)
        .map(|i| rgb::RGBA::new((i * 17) as u8, (i * 53) as u8, (i * 101) as u8, 255))
        .collect();
    let c = QuantizeConfig::new(OutputFormat::PngJoint)
        .with_max_colors(2)
        .with_min_ssim2(100.);
    let failure =
        crate::quantize_rgba_with_dispatch(&p, 16, 8, &c, &enough::Unstoppable).unwrap_err();
    assert!(matches!(
        failure.error,
        crate::QuantizeError::QualityNotMet { .. }
    ));
    let report = failure.dispatch.unwrap();
    assert!(report.metric && report.dither_used && !report.joint_used);
    assert!(
        crate::quantize_rgba_with_dispatch(&p, 0, 8, &c, &enough::Unstoppable)
            .unwrap_err()
            .dispatch
            .is_none()
    );
}
