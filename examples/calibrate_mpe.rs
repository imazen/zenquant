//! Calibrate MPE ↔ butteraugli mapping.
//!
//! Loads images from CID22 corpus, quantizes at various color counts,
//! computes both butteraugli and MPE, outputs CSV for correlation analysis.
//!
//! Usage:
//!   cargo run --example calibrate_mpe --release -- [image_dir] [max_images]
//!
//! Defaults to CID22-512/training corpus.

use butteraugli::ButteraugliParams;
use imgref::ImgVec;
use rgb::RGB8;
use std::path::PathBuf;

use zenquant::_internals::compute_mpe;
use zenquant::{OutputFormat, QuantizeConfig};

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

    println!("image,colors,mpe_score,butteraugli");

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

        // Butteraugli reference image (sRGB u8)
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

            // MPE
            let mpe = compute_mpe(
                &pixels,
                result.palette(),
                result.indices(),
                width,
                height,
                None,
            );

            // Butteraugli — reconstruct quantized image as RGB8
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
            let quant_img = ImgVec::new(quant_pixels, width, height);

            let ba_result = butteraugli::butteraugli(
                orig_img.as_ref(),
                quant_img.as_ref(),
                &ButteraugliParams::default(),
            );

            match ba_result {
                Ok(ba) => {
                    println!("{fname},{colors},{:.6},{:.4}", mpe.score, ba.score);
                }
                Err(e) => {
                    eprintln!("  butteraugli error {fname} @ {colors}: {e}");
                }
            }
        }
    }

    eprintln!("\nDone. Pipe output to a CSV file and compute Pearson r to calibrate.");
}
