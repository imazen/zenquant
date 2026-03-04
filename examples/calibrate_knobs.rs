//! Calibrate compression tier quality costs.
//!
//! Runs quantization at 5 compression tier settings across corpus images,
//! measuring SSIM2 via MPE. Outputs CSV showing the quality delta between tiers.
//!
//! Usage:
//!   cargo run --example calibrate_knobs --release -- [image_dir] [max_images]
//!
//! Defaults to CID22-512/training corpus.
//!
//! The 5 tiers:
//!   0: Best quality, RunPriority::Quality, dither 1.0× — max quality
//!   1: Best quality, RunPriority::Balanced, dither 1.0× — default
//!   2: Best quality, RunPriority::Compression, dither 1.0× — aggressive runs
//!   3: Balanced quality, RunPriority::Compression, dither 0.8× — faster
//!   4: Fast quality, RunPriority::Compression, dither 0.6× — max compression

use rgb::RGB8;
use std::path::PathBuf;
use std::time::Instant;

use zenquant::{OutputFormat, Quality, QuantizeConfig};

struct TierConfig {
    name: &'static str,
    quality: Quality,
    run_priority: &'static str,
    dither_mult: f32,
}

const TIERS: [TierConfig; 5] = [
    TierConfig {
        name: "tier0_max_quality",
        quality: Quality::Best,
        run_priority: "quality",
        dither_mult: 1.0,
    },
    TierConfig {
        name: "tier1_default",
        quality: Quality::Best,
        run_priority: "balanced",
        dither_mult: 1.0,
    },
    TierConfig {
        name: "tier2_aggressive_runs",
        quality: Quality::Best,
        run_priority: "compression",
        dither_mult: 1.0,
    },
    TierConfig {
        name: "tier3_balanced_compress",
        quality: Quality::Balanced,
        run_priority: "compression",
        dither_mult: 0.8,
    },
    TierConfig {
        name: "tier4_max_compress",
        quality: Quality::Fast,
        run_priority: "compression",
        dither_mult: 0.6,
    },
];

fn build_config(tier: &TierConfig) -> QuantizeConfig {
    let mut config = QuantizeConfig::new(OutputFormat::Png)
        .with_max_colors(256)
        .with_quality(tier.quality)
        .with_compute_quality_metric(true);

    config = match tier.run_priority {
        "quality" => config._with_run_priority_quality(),
        "compression" => config._with_run_priority_compression(),
        _ => config, // balanced is default
    };

    if (tier.dither_mult - 1.0).abs() > 0.01 {
        // Apply dither multiplier against PNG default (0.5)
        config = config._with_dither_strength(0.5 * tier.dither_mult);
    }

    config
}

fn codec_corpus_dir() -> std::path::PathBuf {
    let dir = std::path::PathBuf::from(
        std::env::var("CODEC_CORPUS_DIR").unwrap_or_else(|_| "/home/lilith/work/codec-corpus".into()),
    );
    assert!(dir.is_dir(), "Codec corpus not found: {}. Set CODEC_CORPUS_DIR.", dir.display());
    dir
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let default_dir = codec_corpus_dir().join("CID22/CID22-512/training").to_string_lossy().into_owned();
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

    eprintln!(
        "Calibrating {} tiers across {} images from {}",
        TIERS.len(),
        paths.len(),
        image_dir
    );

    // CSV header
    println!("image,tier,tier_name,mpe,ssim2_est,ba_est,time_ms");

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

        for (tier_idx, tier) in TIERS.iter().enumerate() {
            let config = build_config(tier);

            let start = Instant::now();
            let result = match zenquant::quantize(&pixels, width, height, &config) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("  skip {fname} @ {}: {e}", tier.name);
                    continue;
                }
            };
            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

            let mpe = result.mpe_score().unwrap_or(0.0);
            let ssim2 = result.ssimulacra2_estimate().unwrap_or(100.0);
            let ba = result.butteraugli_estimate().unwrap_or(0.0);

            println!(
                "{fname},{tier_idx},{},{mpe:.6},{ssim2:.4},{ba:.4},{elapsed_ms:.1}",
                tier.name
            );
        }
    }

    eprintln!("\nDone. Pipe output to a CSV file for analysis.");
    eprintln!("Compute per-tier medians to see quality cost of each tier step.");
}
