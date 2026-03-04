//! Profile time spent in each pipeline step.
//!
//! Usage:
//!   cargo run --example profile_steps --release -- [image_path]

use std::time::Instant;

use zenquant::_dev::dither::{DitherMode as DM, DitherParams, dither_image};
use zenquant::_dev::histogram;
use zenquant::_dev::masking;
use zenquant::_dev::median_cut;
use zenquant::_dev::oklab::{OKLab, srgb_to_oklab};
use zenquant::_dev::palette::{Palette, PaletteSortStrategy};
use zenquant::_dev::remap;

fn codec_corpus_dir() -> std::path::PathBuf {
    let dir = std::path::PathBuf::from(
        std::env::var("CODEC_CORPUS_DIR").unwrap_or_else(|_| "/home/lilith/work/codec-corpus".into()),
    );
    assert!(dir.is_dir(), "Codec corpus not found: {}. Set CODEC_CORPUS_DIR.", dir.display());
    dir
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let default_path = codec_corpus_dir().join("CID22/CID22-512/training/1001682.png").to_string_lossy().into_owned();
    let image_path = args.get(1).unwrap_or(&default_path);

    let img = image::open(image_path).unwrap().to_rgb8();
    let width = img.width() as usize;
    let height = img.height() as usize;
    let pixels: Vec<rgb::RGB<u8>> = img
        .pixels()
        .map(|p| rgb::RGB {
            r: p.0[0],
            g: p.0[1],
            b: p.0[2],
        })
        .collect();

    println!("Image: {image_path}");
    println!("Size: {width}x{height} = {} pixels", pixels.len());
    println!();

    // Step 0: Shared OKLab conversion (computed once, shared across pipeline)
    let t = Instant::now();
    let labs: Vec<OKLab> = pixels
        .iter()
        .map(|p| srgb_to_oklab(p.r, p.g, p.b))
        .collect();
    let oklab_ms = t.elapsed().as_secs_f64() * 1000.0;
    println!("0. OKLab convert:   {:>8.1}ms", oklab_ms);

    // Step 1: AQ masking (from pre-computed labs)
    let t = Instant::now();
    let weights = masking::compute_masking_weights_from_labs(&labs, width, height);
    let masking_ms = t.elapsed().as_secs_f64() * 1000.0;
    println!("1. AQ masking:      {:>8.1}ms", masking_ms);

    // Step 2: Histogram (from pre-computed labs)
    let t = Instant::now();
    let (hist, _bumped) = histogram::build_histogram_from_labs(&labs, &weights, 256);
    let hist_ms = t.elapsed().as_secs_f64() * 1000.0;
    println!(
        "2. Histogram:       {:>8.1}ms  ({} entries)",
        hist_ms,
        hist.len()
    );

    // Step 3: Wu quantize
    let t = Instant::now();
    let centroids_mc = median_cut::wu_quantize(hist, 256, true);
    let mc_ms = t.elapsed().as_secs_f64() * 1000.0;
    println!(
        "3. Wu quantize:     {:>8.1}ms  ({} colors)",
        mc_ms,
        centroids_mc.len()
    );

    // Step 3b: Pixel-level k-means refinement (balanced = 2 iters, from labs)
    let t = Instant::now();
    let _centroids_2 = median_cut::refine_against_pixels_from_labs(
        centroids_mc.clone(),
        &pixels,
        &labs,
        &weights,
        2,
        131_072,
    );
    let kmeans_2_ms = t.elapsed().as_secs_f64() * 1000.0;
    println!("3b. K-means (2it):  {:>8.1}ms", kmeans_2_ms);

    // Step 3c: Pixel-level k-means refinement (quality = 8 iters, from labs)
    let t = Instant::now();
    let centroids = median_cut::refine_against_pixels_from_labs(
        centroids_mc.clone(),
        &pixels,
        &labs,
        &weights,
        8,
        131_072,
    );
    let kmeans_ms = t.elapsed().as_secs_f64() * 1000.0;
    println!("3c. K-means (8it):  {:>8.1}ms", kmeans_ms);

    // Step 4: Palette build + NN cache
    let t = Instant::now();
    let mut pal =
        Palette::from_centroids_sorted(centroids, false, PaletteSortStrategy::DeltaMinimize);
    pal.build_nn_cache();
    let pal_ms = t.elapsed().as_secs_f64() * 1000.0;
    println!("4. Palette+cache:   {:>8.1}ms", pal_ms);

    // Step 5: Dithering (Adaptive, with pre-computed labs)
    let t = Instant::now();
    let params = DitherParams {
        width,
        height,
        weights: &weights,
        palette: &pal,
        mode: DM::Adaptive,
        run_priority: remap::RunPriority::Balanced,
        dither_strength: 0.5,
        prev_indices: None,
        precomputed_labs: Some(&labs),
    };
    let mut indices = dither_image(&pixels, &params, None);
    let dither_ms = t.elapsed().as_secs_f64() * 1000.0;
    println!("5. Dither Adaptive: {:>8.1}ms", dither_ms);

    // Step 5a: Dithering (Ordered, with pre-computed labs)
    let t = Instant::now();
    let params_ordered = DitherParams {
        width,
        height,
        weights: &weights,
        palette: &pal,
        mode: DM::Ordered,
        run_priority: remap::RunPriority::Balanced,
        dither_strength: 0.5,
        prev_indices: None,
        precomputed_labs: Some(&labs),
    };
    let _indices_ordered = dither_image(&pixels, &params_ordered, None);
    let ordered_ms = t.elapsed().as_secs_f64() * 1000.0;
    println!("5a. Dither Ordered: {:>8.1}ms", ordered_ms);

    // Step 5b: Viterbi (with pre-computed labs)
    let t = Instant::now();
    remap::viterbi_refine_with_labs(
        &pixels,
        &labs,
        width,
        height,
        &weights,
        &pal,
        &mut indices,
        0.01,
    );
    let viterbi_ms = t.elapsed().as_secs_f64() * 1000.0;
    println!("5b. Viterbi:        {:>8.1}ms", viterbi_ms);

    println!();
    let total =
        oklab_ms + masking_ms + hist_ms + mc_ms + kmeans_ms + pal_ms + dither_ms + viterbi_ms;
    println!("Total (Adaptive):   {:>8.1}ms", total);
    let total_ordered =
        oklab_ms + masking_ms + hist_ms + mc_ms + kmeans_ms + pal_ms + ordered_ms + viterbi_ms;
    println!("Total (Ordered):    {:>8.1}ms", total_ordered);

    // Also test without k-means
    println!();
    println!("--- Without k-means refinement ---");
    let mut pal_noref =
        Palette::from_centroids_sorted(centroids_mc, false, PaletteSortStrategy::DeltaMinimize);
    pal_noref.build_nn_cache();
    let t = Instant::now();
    let params_noref = DitherParams {
        width,
        height,
        weights: &weights,
        palette: &pal_noref,
        mode: DM::Adaptive,
        run_priority: remap::RunPriority::Balanced,
        dither_strength: 0.5,
        prev_indices: None,
        precomputed_labs: Some(&labs),
    };
    let mut indices2 = dither_image(&pixels, &params_noref, None);
    remap::viterbi_refine_with_labs(
        &pixels,
        &labs,
        width,
        height,
        &weights,
        &pal_noref,
        &mut indices2,
        0.01,
    );
    let noref_ms = t.elapsed().as_secs_f64() * 1000.0;
    println!("Dither+Viterbi (no k-means): {:>8.1}ms", noref_ms);
    let total_noref = oklab_ms + masking_ms + hist_ms + mc_ms + pal_ms + noref_ms;
    println!("Total without k-means:       {:>8.1}ms", total_noref);
}
