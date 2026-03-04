//! Speed + quality comparison at different quality presets.
//!
//! Usage:
//!   cargo run --example speed_test --release -- [image_dir] [max_images]

use butteraugli::ButteraugliParams;
use fast_ssim2::compute_ssimulacra2;
use flate2::Compression;
use flate2::write::DeflateEncoder;
use imgref::ImgVec;
use rgb::RGB8;
use std::io::Write;
use std::time::Instant;
use zensim::{Zensim, ZensimProfile};

use zenquant::{OutputFormat, Quality, QuantizeConfig};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let image_dir = args
        .get(1)
        .map(|s| s.as_str())
        .unwrap_or("/home/lilith/work/codec-corpus/CID22/CID22-512/validation");
    let max_images: usize = args
        .get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(usize::MAX);

    let mut paths: Vec<std::path::PathBuf> = std::fs::read_dir(image_dir)
        .unwrap_or_else(|e| panic!("cannot read {image_dir}: {e}"))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .is_some_and(|ext| ext == "png" || ext == "jpg" || ext == "jpeg")
        })
        .collect();
    paths.sort();
    paths.truncate(max_images);

    // Pre-load images
    let images: Vec<(Vec<rgb::RGB<u8>>, Vec<RGB8>, usize, usize)> = paths
        .iter()
        .filter_map(|path| {
            let img = image::open(path).ok()?.to_rgb8();
            let w = img.width() as usize;
            let h = img.height() as usize;
            let px: Vec<rgb::RGB<u8>> = img
                .pixels()
                .map(|p| rgb::RGB {
                    r: p.0[0],
                    g: p.0[1],
                    b: p.0[2],
                })
                .collect();
            let rf: Vec<RGB8> = px.iter().map(|p| RGB8::new(p.r, p.g, p.b)).collect();
            Some((px, rf, w, h))
        })
        .collect();

    eprintln!("Testing {} images from {image_dir}", images.len());

    let zsim = Zensim::new(ZensimProfile::latest());

    // Presets to test
    let presets: Vec<(&str, QuantizeConfig)> = vec![
        (
            "zq fast",
            QuantizeConfig::new(OutputFormat::Png)
                .with_quality(Quality::Fast)
                .with_compute_quality_metric(true),
        ),
        (
            "zq balanced",
            QuantizeConfig::new(OutputFormat::Png)
                .with_quality(Quality::Balanced)
                .with_compute_quality_metric(true),
        ),
        (
            "zq best",
            QuantizeConfig::new(OutputFormat::Png).with_compute_quality_metric(true),
        ),
    ];

    println!(
        "{:<20} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "Preset", "BA", "SS2", "zsim", "MPE", "eBA", "eSS2", "defl", "ms", "ms/MP"
    );
    println!("{}", "-".repeat(104));

    // Compute total megapixels for ms/MP
    let total_mp: f64 = images.iter().map(|(_, _, w, h)| (*w * *h) as f64 / 1_000_000.0).sum();

    for (name, config) in &presets {
        let mut vals_ba = Vec::new();
        let mut vals_ss2 = Vec::new();
        let mut vals_zsim = Vec::new();
        let mut vals_mpe = Vec::new();
        let mut vals_eba = Vec::new();
        let mut vals_ess2 = Vec::new();
        let mut vals_deflate = Vec::new();
        let mut vals_ms = Vec::new();

        for (pixels, ref_rgb, width, height) in &images {
            let t = Instant::now();
            let result = zenquant::quantize(pixels, *width, *height, config).unwrap();
            let ms = t.elapsed().as_secs_f64() * 1000.0;

            // Zenquant's internal MPE metric
            let (mpe, eba, ess2) = if let Some(m) = result.mpe_result() {
                (
                    m.score as f64,
                    m.butteraugli_estimate as f64,
                    m.ssimulacra2_estimate as f64,
                )
            } else {
                (f64::NAN, f64::NAN, f64::NAN)
            };

            let test_rgb: Vec<RGB8> = result
                .indices()
                .iter()
                .map(|&idx| {
                    let c = result.palette()[idx as usize];
                    RGB8::new(c[0], c[1], c[2])
                })
                .collect();

            let ref_img = ImgVec::new(ref_rgb.clone(), *width, *height);
            let test_img = ImgVec::new(test_rgb.clone(), *width, *height);
            let ba = butteraugli::butteraugli(
                ref_img.as_ref(),
                test_img.as_ref(),
                &ButteraugliParams::default(),
            )
            .map(|r| r.score)
            .unwrap_or(f64::NAN);

            let ref_px: Vec<[u8; 3]> = ref_rgb.iter().map(|p| [p.r, p.g, p.b]).collect();
            let test_px: Vec<[u8; 3]> = test_rgb.iter().map(|p| [p.r, p.g, p.b]).collect();
            let ss2 = compute_ssimulacra2(
                ImgVec::new(ref_px, *width, *height).as_ref(),
                ImgVec::new(test_px, *width, *height).as_ref(),
            )
            .unwrap_or(f64::NAN);

            // Zensim
            let ref_zsim = ImgVec::new(ref_rgb.clone(), *width, *height);
            let test_zsim = ImgVec::new(test_rgb.clone(), *width, *height);
            let zs = zsim
                .compute(&ref_zsim.as_ref(), &test_zsim.as_ref())
                .map(|r| r.score)
                .unwrap_or(f64::NAN);

            let deflate = deflate_compress(result.indices());

            if ba.is_finite() && ss2.is_finite() {
                vals_ba.push(ba);
                vals_ss2.push(ss2);
                vals_zsim.push(zs);
                vals_mpe.push(mpe);
                vals_eba.push(eba);
                vals_ess2.push(ess2);
                vals_deflate.push(deflate as f64);
                vals_ms.push(ms);
            }
        }

        let n = vals_ba.len() as f64;
        let ms_per_mp = vals_ms.iter().sum::<f64>() / total_mp;
        println!(
            "{:<20} {:>8.3} {:>8.2} {:>8.2} {:>8.4} {:>8.3} {:>8.2} {:>8.0} {:>8.1} {:>8.1}",
            format!("{name} mean"),
            vals_ba.iter().sum::<f64>() / n,
            vals_ss2.iter().sum::<f64>() / n,
            vals_zsim.iter().sum::<f64>() / n,
            vals_mpe.iter().sum::<f64>() / n,
            vals_eba.iter().sum::<f64>() / n,
            vals_ess2.iter().sum::<f64>() / n,
            vals_deflate.iter().sum::<f64>() / n,
            vals_ms.iter().sum::<f64>() / n,
            ms_per_mp,
        );
        let p95_ms_per_mp = percentile_f64(&mut vals_ms.clone()) / total_mp * n;
        println!(
            "{:<20} {:>8.3} {:>8.2} {:>8.2} {:>8.4} {:>8.3} {:>8.2} {:>8.0} {:>8.1} {:>8.1}",
            format!("{name} p95"),
            percentile_f64(&mut vals_ba),
            percentile_f64(&mut vals_ss2),
            percentile_f64(&mut vals_zsim),
            percentile_f64(&mut vals_mpe),
            percentile_f64(&mut vals_eba),
            percentile_f64(&mut vals_ess2),
            percentile_f64(&mut vals_deflate),
            percentile_f64(&mut vals_ms),
            p95_ms_per_mp,
        );
    }

    // Also run competitors for comparison
    println!();
    for (comp_name, comp_fn) in [
        ("imagequant", run_imagequant as fn(&[rgb::RGB<u8>], usize, usize) -> (Vec<[u8; 3]>, Vec<u8>)),
        ("quantizr", run_quantizr as fn(&[rgb::RGB<u8>], usize, usize) -> (Vec<[u8; 3]>, Vec<u8>)),
        ("quantette-wu", run_quantette_wu as fn(&[rgb::RGB<u8>], usize, usize) -> (Vec<[u8; 3]>, Vec<u8>)),
        ("quantette-km", run_quantette_km as fn(&[rgb::RGB<u8>], usize, usize) -> (Vec<[u8; 3]>, Vec<u8>)),
    ] {
        let mut vals_ba = Vec::new();
        let mut vals_ss2 = Vec::new();
        let mut vals_zsim = Vec::new();
        let mut vals_deflate = Vec::new();
        let mut vals_ms = Vec::new();

        for (pixels, ref_rgb, width, height) in &images {
            let t = Instant::now();
            let (pal, idx) = comp_fn(pixels, *width, *height);
            let ms = t.elapsed().as_secs_f64() * 1000.0;

            let test_rgb: Vec<RGB8> = idx
                .iter()
                .map(|&i| {
                    let c = pal[i as usize];
                    RGB8::new(c[0], c[1], c[2])
                })
                .collect();

            let ref_img = ImgVec::new(ref_rgb.clone(), *width, *height);
            let test_img = ImgVec::new(test_rgb.clone(), *width, *height);
            let ba = butteraugli::butteraugli(
                ref_img.as_ref(),
                test_img.as_ref(),
                &ButteraugliParams::default(),
            )
            .map(|r| r.score)
            .unwrap_or(f64::NAN);

            let ref_px: Vec<[u8; 3]> = ref_rgb.iter().map(|p| [p.r, p.g, p.b]).collect();
            let test_px: Vec<[u8; 3]> = test_rgb.iter().map(|p| [p.r, p.g, p.b]).collect();
            let ss2 = compute_ssimulacra2(
                ImgVec::new(ref_px, *width, *height).as_ref(),
                ImgVec::new(test_px, *width, *height).as_ref(),
            )
            .unwrap_or(f64::NAN);

            let ref_zsim = ImgVec::new(ref_rgb.clone(), *width, *height);
            let test_zsim = ImgVec::new(test_rgb.clone(), *width, *height);
            let zs = zsim
                .compute(&ref_zsim.as_ref(), &test_zsim.as_ref())
                .map(|r| r.score)
                .unwrap_or(f64::NAN);

            if ba.is_finite() && ss2.is_finite() {
                vals_ba.push(ba);
                vals_ss2.push(ss2);
                vals_zsim.push(zs);
                vals_deflate.push(deflate_compress(&idx) as f64);
                vals_ms.push(ms);
            }
        }
        let n = vals_ba.len() as f64;
        let ms_per_mp = vals_ms.iter().sum::<f64>() / total_mp;
        println!(
            "{:<20} {:>8.3} {:>8.2} {:>8.2} {:>8} {:>8} {:>8} {:>8.0} {:>8.1} {:>8.1}",
            format!("{comp_name} mean"),
            vals_ba.iter().sum::<f64>() / n,
            vals_ss2.iter().sum::<f64>() / n,
            vals_zsim.iter().sum::<f64>() / n,
            "", "", "",
            vals_deflate.iter().sum::<f64>() / n,
            vals_ms.iter().sum::<f64>() / n,
            ms_per_mp,
        );
        let p95_ms_per_mp = percentile_f64(&mut vals_ms.clone()) / total_mp * n;
        println!(
            "{:<20} {:>8.3} {:>8.2} {:>8.2} {:>8} {:>8} {:>8} {:>8.0} {:>8.1} {:>8.1}",
            format!("{comp_name} p95"),
            percentile_f64(&mut vals_ba),
            percentile_f64(&mut vals_ss2),
            percentile_f64(&mut vals_zsim),
            "", "", "",
            percentile_f64(&mut vals_deflate),
            percentile_f64(&mut vals_ms),
            p95_ms_per_mp,
        );
    }
}

fn percentile_f64(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((values.len() - 1) as f64 * 0.95) as usize;
    values[idx.min(values.len() - 1)]
}

fn deflate_compress(data: &[u8]) -> usize {
    let mut encoder = DeflateEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(data).unwrap();
    encoder.finish().unwrap().len()
}

fn run_imagequant(pixels: &[rgb::RGB<u8>], width: usize, height: usize) -> (Vec<[u8; 3]>, Vec<u8>) {
    let mut attr = imagequant::Attributes::new();
    attr.set_quality(0, 80).unwrap();
    let rgba: Vec<imagequant::RGBA> = pixels
        .iter()
        .map(|p| imagequant::RGBA::new(p.r, p.g, p.b, 255))
        .collect();
    let mut img = attr.new_image(rgba, width, height, 0.0).unwrap();
    let mut result = attr.quantize(&mut img).unwrap();
    result.set_dithering_level(0.5).unwrap();
    let (pal, idx) = result.remapped(&mut img).unwrap();
    (pal.iter().map(|c| [c.r, c.g, c.b]).collect(), idx)
}

fn run_quantette_wu(pixels: &[rgb::RGB<u8>], width: usize, height: usize) -> (Vec<[u8; 3]>, Vec<u8>) {
    run_quantette(pixels, width, height, false)
}

fn run_quantette_km(pixels: &[rgb::RGB<u8>], width: usize, height: usize) -> (Vec<[u8; 3]>, Vec<u8>) {
    run_quantette(pixels, width, height, true)
}

fn run_quantette(
    pixels: &[rgb::RGB<u8>],
    width: usize,
    height: usize,
    use_kmeans: bool,
) -> (Vec<[u8; 3]>, Vec<u8>) {
    use quantette::deps::palette::Srgb;
    use quantette::{ImageBuf, Pipeline, QuantizeMethod};
    use quantette::dither::FloydSteinberg;

    let srgb_pixels: Vec<Srgb<u8>> = pixels.iter().map(|p| Srgb::new(p.r, p.g, p.b)).collect();
    let image = ImageBuf::new(width as u32, height as u32, srgb_pixels)
        .expect("quantette ImageBuf");

    let method = if use_kmeans {
        QuantizeMethod::kmeans()
    } else {
        QuantizeMethod::Wu
    };

    let indexed = Pipeline::new()
        .palette_size(256u16.try_into().unwrap())
        .quantize_method(method)
        .ditherer(Some(FloydSteinberg::new()))
        .input_image(image.as_ref())
        .output_srgb8_indexed_image();

    let palette: Vec<[u8; 3]> = indexed.palette().iter().map(|c| [c.red, c.green, c.blue]).collect();
    let indices = indexed.indices().to_vec();
    (palette, indices)
}

fn run_quantizr(pixels: &[rgb::RGB<u8>], width: usize, height: usize) -> (Vec<[u8; 3]>, Vec<u8>) {
    let pixel_bytes: Vec<u8> = pixels.iter().flat_map(|p| [p.r, p.g, p.b, 255u8]).collect();
    let image = quantizr::Image::new(&pixel_bytes, width, height).unwrap();
    let mut options = quantizr::Options::default();
    options.set_max_colors(256).unwrap();
    let mut result = quantizr::QuantizeResult::quantize(&image, &options);
    result.set_dithering_level(0.5).unwrap();
    let mut idx = vec![0u8; width * height];
    result.remap_image(&image, &mut idx).unwrap();
    let pal = result.get_palette();
    (
        pal.entries[..pal.count as usize]
            .iter()
            .map(|c| [c.r, c.g, c.b])
            .collect(),
        idx,
    )
}
