//! Ordinary calls only: this binary never references the reporting API.
use std::{hint::black_box, time::Instant};
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
fn cost() {
    // Compare the unused API to the exact pre-change source, not to 0.1.3's
    // different cancellation paths. ABBA ordering exposes ordinary timing noise.
    for kind in [1, 2] {
        for side in [8, 64, 256, 1024] {
            let p = image(side, side, kind);
            let a = config!(candidate, 0, 0, 1, None::<f32>, 16);
            let b = config!(before, 0, 0, 1, None::<f32>, 16);
            let reps = if side < 64 {
                200
            } else if side < 256 {
                10
            } else {
                1
            };
            let mut rows = vec![];
            for round in 0..20 {
                let mut sample = [0u128; 4];
                for (slot, elapsed) in sample.iter_mut().enumerate() {
                    let is_a = if round % 2 == 0 {
                        slot == 0 || slot == 3
                    } else {
                        slot == 1 || slot == 2
                    };
                    let start = Instant::now();
                    for _ in 0..reps {
                        if is_a {
                            black_box(
                                candidate::quantize_rgba(black_box(&p), side, side, black_box(&a))
                                    .unwrap(),
                            );
                        } else {
                            black_box(
                                before::quantize_rgba(black_box(&p), side, side, black_box(&b))
                                    .unwrap(),
                            );
                        }
                    }
                    *elapsed = start.elapsed().as_nanos();
                }
                let (a, b) = if round % 2 == 0 {
                    (sample[0] + sample[3], sample[1] + sample[2])
                } else {
                    (sample[1] + sample[2], sample[0] + sample[3])
                };
                rows.push((a, b));
                println!(
                    "cost kind={kind} side={side} round={round} repeats={reps} candidate_ns={a} before_ns={b}"
                );
            }
            let mut ratios: Vec<_> = rows.iter().map(|(a, b)| *a as f64 / *b as f64).collect();
            ratios.sort_by(f64::total_cmp);
            println!(
                "cost_summary kind={kind} side={side} paired_rounds=20 ratio_median={} ratio_min={} ratio_max={}",
                ratios[10], ratios[0], ratios[19]
            );
        }
    }
}

fn main() {
    cost()
}
