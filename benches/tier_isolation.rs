//! SIMD-tier isolation: the native top tier vs the same code forced to scalar.
//!
//! zenquant's SIMD is the sRGB→OKLab batch conversion (`src/simd.rs`), which
//! feeds every distance computation in k-means and the Viterbi pass. Nothing in
//! this crate could tell you what that is worth on ARM — there were no benches
//! at all. A kernel slower than its own scalar fallback would be invisible.
//! (The same gap in linear-srgb was hiding a real regression.)
//!
//! Run: `cargo bench --bench tier_isolation --features _dev`
//! Do NOT build with `-C target-cpu=native`: that pins the tier at compile
//! time, after which it cannot be disabled and this bench skips rather than
//! silently reporting the SIMD path under both labels.

use criterion::{Criterion, criterion_group, criterion_main};
use zenquant::{OutputFormat, QuantizeConfig};

#[cfg(target_arch = "aarch64")]
type TierToken = archmage::NeonToken;
#[cfg(target_arch = "x86_64")]
type TierToken = archmage::X64V3Token;

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
const TIER_NAME: &str = if cfg!(target_arch = "aarch64") {
    "neon"
} else {
    "v3(avx2)"
};

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
fn set_simd(enabled: bool) -> bool {
    TierToken::dangerously_disable_token_process_wide(!enabled).is_ok()
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
fn set_simd(_enabled: bool) -> bool {
    false
}

/// Noise + patches, not a gradient: gradients quantize degenerately (few
/// distinct colours) and would understate the palette search entirely.
fn make_image(w: usize, h: usize) -> Vec<rgb::RGB<u8>> {
    let mut px = Vec::with_capacity(w * h);
    let mut state = 0x9e37_79b9u32;
    for y in 0..h {
        for x in 0..w {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let patch = ((x / 32 + y / 32) & 3) as u8;
            px.push(rgb::RGB {
                r: ((state >> 24) as u8).wrapping_add(patch.wrapping_mul(40)),
                g: ((state >> 16) as u8).wrapping_add(patch.wrapping_mul(80)),
                b: ((state >> 8) as u8).wrapping_add(patch.wrapping_mul(120)),
            });
        }
    }
    px
}

fn bench_tiers(c: &mut Criterion) {
    if !set_simd(true) || !set_simd(false) {
        eprintln!(
            "[tier_isolation] no toggleable SIMD tier on this target, or the tier is \
             compile-time guaranteed (drop -C target-cpu=native, build with --features _dev). \
             Skipping."
        );
        return;
    }
    set_simd(true);
    eprintln!("[tier_isolation] comparing {TIER_NAME} vs forced scalar");

    for &(label, w, h) in &[("256x256", 256usize, 256usize), ("1024x1024", 1024, 1024)] {
        let px = make_image(w, h);
        let config = QuantizeConfig::new(OutputFormat::Png).with_max_colors(256);
        let mut group = c.benchmark_group(format!("quantize/{label}"));
        group.sample_size(20);
        for (arm, simd) in [(TIER_NAME, true), ("scalar", false)] {
            group.bench_function(arm, |b| {
                set_simd(simd);
                b.iter(|| {
                    zenquant::quantize(std::hint::black_box(&px), w, h, &config).unwrap()
                })
            });
        }
        set_simd(true);
        group.finish();
    }
    set_simd(true);
}

criterion_group!(benches, bench_tiers);
criterion_main!(benches);
