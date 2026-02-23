//! Calibrate MPE ↔ butteraugli mapping.
//!
//! Loads images from a corpus, quantizes at various color counts,
//! computes both butteraugli and multiple MPE pooling variants, outputs CSV.
//!
//! Usage:
//!   cargo run --example calibrate_mpe --release -- [image_dir] [max_images]
//!
//! Defaults to CID22-512/training corpus.

use butteraugli::ButteraugliParams;
use fast_ssim2::compute_ssimulacra2;
use imgref::ImgVec;
use rgb::RGB8;
use std::path::PathBuf;

use zenquant::_internals::compute_mpe;
use zenquant::{OutputFormat, QuantizeConfig};

/// Minkowski-p pooling: (mean(x^p))^(1/p)
fn minkowski_pool(values: &[f32], p: f64) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sum = 0.0f64;
    for &v in values {
        sum += (v as f64).powf(p);
    }
    let mean = sum / values.len() as f64;
    mean.powf(1.0 / p) as f32
}

/// Percentile of sorted values (0-100).
fn percentile(values: &mut [f32], pct: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((values.len() - 1) as f32 * pct / 100.0) as usize;
    values[idx.min(values.len() - 1)]
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let image_dir = args
        .get(1)
        .map(|s| s.as_str())
        .unwrap_or("/home/lilith/work/codec-corpus/CID22/CID22-512/training");
    let max_images: usize = args
        .get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(usize::MAX);

    let mut paths: Vec<PathBuf> = std::fs::read_dir(image_dir)
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

    if paths.is_empty() {
        eprintln!("No images found in {image_dir}");
        std::process::exit(1);
    }

    let color_counts: &[u32] = &[8, 16, 32, 64, 128, 256];

    // CSV header: multiple pooling variants for comparison
    println!("image,colors,mink4,mink8,mink16,max,p95,p99,butteraugli,ssim2");

    for path in &paths {
        let img = match image::open(path) {
            Ok(img) => img.to_rgb8(),
            Err(e) => {
                eprintln!("skip {}: {e}", path.display());
                continue;
            }
        };
        let width = img.width() as usize;
        let height = img.height() as usize;
        let pixels: Vec<RGB8> = img
            .pixels()
            .map(|p| RGB8 {
                r: p.0[0],
                g: p.0[1],
                b: p.0[2],
            })
            .collect();
        let fname = path.file_name().unwrap().to_string_lossy();

        let orig_img = ImgVec::new(pixels.clone(), width, height);

        for &colors in color_counts {
            let config = QuantizeConfig::new(OutputFormat::Png).max_colors(colors);

            let result = match zenquant::quantize(&pixels, width, height, &config) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("  skip {fname} @ {colors}: {e}");
                    continue;
                }
            };

            let mpe = compute_mpe(
                &pixels,
                result.palette(),
                result.indices(),
                width,
                height,
                None,
            );

            // Compute multiple pooling variants from block scores
            let bs = &mpe.block_scores;
            let mink4 = minkowski_pool(bs, 4.0);
            let mink8 = mpe.score; // internal pooling is mink8
            let mink16 = minkowski_pool(bs, 16.0);
            let max_score = bs.iter().cloned().fold(0.0f32, f32::max);
            let mut bs_copy = bs.clone();
            let p95 = percentile(&mut bs_copy.clone(), 95.0);
            let p99 = percentile(&mut bs_copy, 99.0);

            // Reconstruct quantized image
            let quant_pixels: Vec<RGB8> = result
                .indices()
                .iter()
                .map(|&idx| {
                    let p = result.palette()[idx as usize];
                    RGB8 {
                        r: p[0],
                        g: p[1],
                        b: p[2],
                    }
                })
                .collect();

            // SSIMULACRA2 (takes ImgRef<[u8; 3]>)
            let ref_pixels_ss: Vec<[u8; 3]> = pixels.iter().map(|p| [p.r, p.g, p.b]).collect();
            let test_pixels_ss: Vec<[u8; 3]> =
                quant_pixels.iter().map(|p| [p.r, p.g, p.b]).collect();
            let ref_img_ss = ImgVec::new(ref_pixels_ss, width, height);
            let test_img_ss = ImgVec::new(test_pixels_ss, width, height);
            let ss2 =
                compute_ssimulacra2(ref_img_ss.as_ref(), test_img_ss.as_ref()).unwrap_or(f64::NAN);

            // Butteraugli
            let quant_img = ImgVec::new(quant_pixels, width, height);
            let ba_result = butteraugli::butteraugli(
                orig_img.as_ref(),
                quant_img.as_ref(),
                &ButteraugliParams::default(),
            );

            match ba_result {
                Ok(ba) => {
                    println!(
                        "{fname},{colors},{mink4:.6},{mink8:.6},{mink16:.6},{max_score:.6},{p95:.6},{p99:.6},{:.4},{ss2:.4}",
                        ba.score
                    );
                }
                Err(e) => {
                    eprintln!("  butteraugli error {fname} @ {colors}: {e}");
                }
            }
        }
    }

    eprintln!("\nDone. Pipe output to a CSV file and compute Pearson r to calibrate.");
}
