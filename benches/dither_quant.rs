// Lane-attribution bench (swe-zenquant): dither_image / dither_image_rgba
// over the 3ch/4ch error-diffusion kernels (Adaptive FS + SierraLite +
// Linear non-adaptive) plus end-to-end quantize, on deterministic
// NON-smooth synthetic noise fixtures.
use zenquant::_dev::{dither, oklab, palette, remap};

fn noise_px(w: usize, h: usize, ch: usize, seed: u32) -> Vec<u8> {
    let mut v = Vec::with_capacity(w * h * ch);
    let mut s: u32 = seed;
    for _ in 0..w * h * ch {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        v.push((s >> 8) as u8);
    }
    v
}

fn rgb_img(w: usize, h: usize, seed: u32) -> Vec<rgb::RGB<u8>> {
    noise_px(w, h, 3, seed)
        .chunks_exact(3)
        .map(|c| rgb::RGB::new(c[0], c[1], c[2]))
        .collect()
}

fn rgba_img(w: usize, h: usize, seed: u32) -> Vec<rgb::RGBA<u8>> {
    noise_px(w, h, 4, seed)
        .chunks_exact(4)
        .map(|c| rgb::RGBA::new(c[0], c[1], c[2], c[3]))
        .collect()
}

fn pal(n: usize) -> palette::Palette {
    let centroids: Vec<oklab::OKLab> = (0..n)
        .map(|i| {
            let v = (i * 255 / n.max(1)) as u8;
            oklab::srgb_to_oklab(v, ((i * 37) % 256) as u8, 255 - v)
        })
        .collect();
    palette::Palette::from_centroids_sorted(
        centroids,
        false,
        palette::PaletteSortStrategy::Luminance,
    )
}

fn params<'a>(
    w: usize,
    h: usize,
    weights: &'a [f32],
    pal: &'a palette::Palette,
    mode: dither::DitherMode,
) -> dither::DitherParams<'a> {
    dither::DitherParams {
        width: w,
        height: h,
        weights,
        palette: pal,
        mode,
        run_priority: remap::RunPriority::Quality,
        dither_strength: 1.0,
        prev_indices: None,
        precomputed_labs: None,
    }
}

fn main() {
    let w = 256;
    let h = 256;
    let rgb_px = rgb_img(w, h, 0x9e3779b9);
    let rgba_px = rgba_img(w, h, 0x85ebca6b);
    let pal = pal(64);
    let weights: Vec<f32> = noise_px(w, h, 1, 0x243f6a88)
        .iter()
        .map(|&v| 0.3 + v as f32 / 255.0 * 0.7)
        .collect();
    let ones = vec![1.0f32; w * h];

    let result = zenbench::run_gated(zenbench::GateConfig::disabled(), |suite| {
        suite.compare("dither_quant", |g| {
            let px = rgb_px.clone();
            g.bench("quantize_rgb_256", move |b| {
                let px = px.clone();
                let cfg = zenquant::QuantizeConfig::new(zenquant::OutputFormat::Png);
                b.iter(move || {
                    zenbench::black_box(
                        zenquant::quantize(&px, w, h, &cfg).unwrap().indices().len(),
                    )
                })
            });
            for (name, mode, wts) in [
                (
                    "dither_rgb_adaptive",
                    dither::DitherMode::Adaptive,
                    weights.clone(),
                ),
                (
                    "dither_rgb_sierralite",
                    dither::DitherMode::SierraLite,
                    weights.clone(),
                ),
                (
                    "dither_rgb_linear_1",
                    dither::DitherMode::Linear,
                    ones.clone(),
                ),
            ] {
                let px = rgb_px.clone();
                let pal = pal.clone();
                g.bench(name, move |b| {
                    let px = px.clone();
                    let wts = wts.clone();
                    let pal = pal.clone();
                    b.iter(move || {
                        let p = params(w, h, &wts, &pal, mode);
                        zenbench::black_box(dither::dither_image(&px, &p, None).len())
                    })
                });
            }
            for (name, mode, wts) in [
                (
                    "dither_rgba_adaptive",
                    dither::DitherMode::Adaptive,
                    weights.clone(),
                ),
                (
                    "dither_rgba_sierralite",
                    dither::DitherMode::SierraLite,
                    weights.clone(),
                ),
                (
                    "dither_rgba_linear_1",
                    dither::DitherMode::Linear,
                    ones.clone(),
                ),
            ] {
                let px = rgba_px.clone();
                let pal = pal.clone();
                g.bench(name, move |b| {
                    let px = px.clone();
                    let wts = wts.clone();
                    let pal = pal.clone();
                    b.iter(move || {
                        let p = params(w, h, &wts, &pal, mode);
                        zenbench::black_box(dither::dither_image_rgba(&px, &p, None).len())
                    })
                });
            }
        });
    });
    zenbench::postprocess_result(&result);
}
