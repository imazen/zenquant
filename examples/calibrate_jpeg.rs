//! Calibrate JPEG quality → butteraugli / SSIMULACRA2 mapping.
//!
//! Compresses images at various JPEG quality levels using the `image` crate
//! (libjpeg-turbo compatible tables), then measures distortion with butteraugli
//! and SSIMULACRA2 to produce reference data for MPE ↔ JPEG quality equivalences.
//!
//! Usage:
//!   cargo run --example calibrate_jpeg --release -- [image_dir] [max_images]

use butteraugli::ButteraugliParams;
use fast_ssim2::compute_ssimulacra2;
use imgref::ImgVec;
use rgb::RGB8;
use std::io::Cursor;
use std::path::PathBuf;

const JPEG_QUALITIES: &[u8] = &[95, 90, 85, 80, 75, 70, 60, 50, 40, 30];

fn codec_corpus_dir() -> std::path::PathBuf {
    let dir = std::path::PathBuf::from(
        std::env::var("CODEC_CORPUS_DIR")
            .unwrap_or_else(|_| "/home/lilith/work/codec-corpus".into()),
    );
    assert!(
        dir.is_dir(),
        "Codec corpus not found: {}. Set CODEC_CORPUS_DIR.",
        dir.display()
    );
    dir
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let default_dir = codec_corpus_dir()
        .join("CID22/CID22-512/training")
        .to_string_lossy()
        .into_owned();
    let image_dir = args.get(1).unwrap_or(&default_dir);
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

    println!("image,jpeg_quality,butteraugli,ssim2");

    for path in &paths {
        let img = match image::open(path) {
            Ok(img) => img.to_rgb8(),
            Err(e) => {
                eprintln!("skip {}: {e}", path.display());
                continue;
            }
        };
        let width = img.width();
        let height = img.height();
        let fname = path.file_name().unwrap().to_string_lossy();

        let ref_rgb: Vec<RGB8> = img
            .pixels()
            .map(|p| RGB8::new(p.0[0], p.0[1], p.0[2]))
            .collect();

        for &q in JPEG_QUALITIES {
            // Encode to JPEG in memory
            let mut jpeg_buf = Cursor::new(Vec::new());
            let encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut jpeg_buf, q);
            if let Err(e) = img.write_with_encoder(encoder) {
                eprintln!("  jpeg encode error {fname} @ q{q}: {e}");
                continue;
            }

            // Decode back
            let decoded = match image::load_from_memory_with_format(
                jpeg_buf.get_ref(),
                image::ImageFormat::Jpeg,
            ) {
                Ok(d) => d.to_rgb8(),
                Err(e) => {
                    eprintln!("  jpeg decode error {fname} @ q{q}: {e}");
                    continue;
                }
            };

            let test_rgb: Vec<RGB8> = decoded
                .pixels()
                .map(|p| RGB8::new(p.0[0], p.0[1], p.0[2]))
                .collect();

            // Butteraugli
            let ref_img = ImgVec::new(ref_rgb.clone(), width as usize, height as usize);
            let test_img = ImgVec::new(test_rgb.clone(), width as usize, height as usize);
            let ba = butteraugli::butteraugli(
                ref_img.as_ref(),
                test_img.as_ref(),
                &ButteraugliParams::default(),
            )
            .map(|r| r.score)
            .unwrap_or(f64::NAN);

            // SSIMULACRA2
            let ref_pixels: Vec<[u8; 3]> = ref_rgb.iter().map(|p| [p.r, p.g, p.b]).collect();
            let test_pixels: Vec<[u8; 3]> = test_rgb.iter().map(|p| [p.r, p.g, p.b]).collect();
            let ref_img_ss = ImgVec::new(ref_pixels, width as usize, height as usize);
            let test_img_ss = ImgVec::new(test_pixels, width as usize, height as usize);
            let ss2 =
                compute_ssimulacra2(ref_img_ss.as_ref(), test_img_ss.as_ref()).unwrap_or(f64::NAN);

            println!("{fname},{q},{ba:.4},{ss2:.4}");
        }
    }

    eprintln!("\nDone.");
}
