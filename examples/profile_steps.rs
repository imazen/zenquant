//! Profile time spent in each pipeline step.
//!
//! Usage:
//!   cargo run --example profile_steps --release -- [image_path]

use std::path::PathBuf;
use std::time::Instant;

use zenquant::dither::{DitherMode as DM, dither_image};
use zenquant::histogram;
use zenquant::masking;
use zenquant::median_cut;
use zenquant::palette::{Palette, PaletteSortStrategy};
use zenquant::remap;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let image_path = args
        .get(1)
        .map(|s| s.as_str())
        .unwrap_or("/home/lilith/work/codec-corpus/CID22/CID22-512/training/1001682.png");

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

    // Step 1: AQ masking
    let t = Instant::now();
    let weights = masking::compute_masking_weights(&pixels, width, height);
    let masking_ms = t.elapsed().as_secs_f64() * 1000.0;
    println!("1. AQ masking:      {:>8.1}ms", masking_ms);

    // Step 2: Histogram
    let t = Instant::now();
    let hist = histogram::build_histogram(&pixels, &weights);
    let hist_ms = t.elapsed().as_secs_f64() * 1000.0;
    println!(
        "2. Histogram:       {:>8.1}ms  ({} entries)",
        hist_ms,
        hist.len()
    );

    // Step 3: Median cut
    let t = Instant::now();
    let centroids_mc = median_cut::median_cut(hist, 256, true);
    let mc_ms = t.elapsed().as_secs_f64() * 1000.0;
    println!(
        "3. Median cut:      {:>8.1}ms  ({} colors)",
        mc_ms,
        centroids_mc.len()
    );

    // Step 3b: Pixel-level k-means refinement
    let t = Instant::now();
    let centroids = if pixels.len() <= 500_000 {
        median_cut::refine_against_pixels(centroids_mc.clone(), &pixels, &weights, 8)
    } else {
        // Subsample
        let step = (pixels.len() / 250_000).max(1);
        let sub_pixels: Vec<rgb::RGB<u8>> = pixels.iter().step_by(step).copied().collect();
        let sub_weights: Vec<f32> = weights.iter().step_by(step).copied().collect();
        median_cut::refine_against_pixels(centroids_mc.clone(), &sub_pixels, &sub_weights, 8)
    };
    let kmeans_ms = t.elapsed().as_secs_f64() * 1000.0;
    println!("3b. K-means refine: {:>8.1}ms  (8 iters)", kmeans_ms);

    // Step 4: Palette build + NN cache
    let t = Instant::now();
    let mut pal =
        Palette::from_centroids_sorted(centroids, false, PaletteSortStrategy::DeltaMinimize);
    pal.build_nn_cache();
    let pal_ms = t.elapsed().as_secs_f64() * 1000.0;
    println!("4. Palette+cache:   {:>8.1}ms", pal_ms);

    // Step 5: Dithering
    let t = Instant::now();
    let mut indices = dither_image(
        &pixels,
        width,
        height,
        &weights,
        &pal,
        DM::Adaptive,
        remap::RunPriority::Balanced,
        0.5,
    );
    let dither_ms = t.elapsed().as_secs_f64() * 1000.0;
    println!("5. Dithering:       {:>8.1}ms", dither_ms);

    // Step 5b: Viterbi
    let t = Instant::now();
    remap::viterbi_refine(&pixels, width, height, &weights, &pal, &mut indices, 0.01);
    let viterbi_ms = t.elapsed().as_secs_f64() * 1000.0;
    println!("5b. Viterbi:        {:>8.1}ms", viterbi_ms);

    println!();
    let total = masking_ms + hist_ms + mc_ms + kmeans_ms + pal_ms + dither_ms + viterbi_ms;
    println!("Total:              {:>8.1}ms", total);

    // Also test without k-means
    println!();
    println!("--- Without k-means refinement ---");
    let mut pal_noref =
        Palette::from_centroids_sorted(centroids_mc, false, PaletteSortStrategy::DeltaMinimize);
    pal_noref.build_nn_cache();
    let t = Instant::now();
    let mut indices2 = dither_image(
        &pixels,
        width,
        height,
        &weights,
        &pal_noref,
        DM::Adaptive,
        remap::RunPriority::Balanced,
        0.5,
    );
    remap::viterbi_refine(
        &pixels,
        width,
        height,
        &weights,
        &pal_noref,
        &mut indices2,
        0.01,
    );
    let noref_ms = t.elapsed().as_secs_f64() * 1000.0;
    println!("Dither+Viterbi (no k-means): {:>8.1}ms", noref_ms);
    let total_noref = masking_ms + hist_ms + mc_ms + pal_ms + noref_ms;
    println!("Total without k-means:       {:>8.1}ms", total_noref);
}
