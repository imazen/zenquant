//! Per-kernel NEON-vs-forced-scalar for zenquant's two SIMD kernels.
//!
//! `tier_isolation.rs` measures `quantize` end-to-end. That cannot reveal a
//! single kernel SLOWER than its own scalar fallback — the faster kernel
//! averages it away. That failure mode was found and fixed in garb, zensim,
//! zentone, zenpng and zenresize during the 2026-07-28 aarch64 sweep, so
//! zenquant's kernels are checked individually rather than inferred.
//!
//! NOTE: on aarch64 NEON is BASELINE, so the "scalar" arm is still fully
//! autovectorized by LLVM. A ratio near 1.00 does NOT mean a kernel is
//! missing — it means both arms compiled to equivalent work.
//!
//! Palette size is swept because `nearest`'s inner loop is over palette
//! entries: a single size would pin one point on that curve.
//!
//! Run: `cargo bench --bench kernel_tiers --features _dev`
//! Do NOT pass `-C target-cpu=native`: that pins the tier at compile time,
//! after which it cannot be disabled and this bench skips rather than
//! silently reporting the SIMD path under both labels.

use zenbench::prelude::*;
use zenquant::_dev::kernels::{Pal, batch_srgb_to_oklab};
use zenquant::_dev::oklab::OKLab;

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

fn labs(n: usize, seed: u32) -> Vec<OKLab> {
    let mut s = seed | 1;
    let mut g = move || {
        s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (s >> 8) as f32 / 16_777_216.0
    };
    (0..n)
        .map(|_| OKLab::new(g(), g() - 0.5, g() - 0.5))
        .collect()
}

fn pixels(n: usize, seed: u32) -> Vec<rgb::RGB<u8>> {
    let mut s = seed | 1;
    let mut g = move || {
        s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (s >> 24) as u8
    };
    (0..n).map(|_| rgb::RGB::new(g(), g(), g())).collect()
}

fn bench_kernels(suite: &mut Suite) {
    if !set_simd(true) || !set_simd(false) {
        eprintln!(
            "[kernel_tiers] no toggleable SIMD tier here, or the tier is \
             compile-time guaranteed (drop -C target-cpu=native, build with \
             --features _dev). Skipping."
        );
        return;
    }
    set_simd(true);
    eprintln!("[kernel_tiers] comparing {TIER_NAME} vs forced scalar");

    const N: usize = 1 << 20; // 1 MP, the realistic remap workload

    // Kernel 1: nearest-palette-entry search, over the GIF/PNG palette range.
    let colors: &'static [OKLab] = Box::leak(labs(N, 7).into_boxed_slice());
    for &npal in &[8usize, 16, 24, 32, 48, 64, 96, 104, 112, 120, 128, 192, 256] {
        let pal: &'static Pal = Box::leak(Box::new(Pal::from_oklab(&labs(npal, 99))));
        suite.compare(format!("palette_nearest/1MP/pal{npal}"), |g| {
            g.throughput(Throughput::Elements(N as u64));
            for (arm, simd) in [(TIER_NAME, true), ("scalar", false)] {
                g.bench(arm, move |b| {
                    let mut out = vec![0u8; N];
                    b.iter(move || {
                        // Must be inside the closure: zenbench interleaves the
                        // arms. Costs one atomic store per iteration and
                        // applies to both arms equally, so it cannot bias the
                        // comparison.
                        set_simd(simd);
                        pal.map(colors, &mut out);
                    })
                });
            }
        });
    }

    // Kernel 2: batch sRGB -> OKLab (cube roots; the front of the pipeline).
    let px: &'static [rgb::RGB<u8>] = Box::leak(pixels(N, 11).into_boxed_slice());
    suite.compare("batch_srgb_to_oklab/1MP", |g| {
        g.throughput(Throughput::Elements(N as u64));
        for (arm, simd) in [(TIER_NAME, true), ("scalar", false)] {
            g.bench(arm, move |b| {
                let mut out = vec![[0.0f32; 3]; N];
                b.iter(move || {
                    set_simd(simd);
                    batch_srgb_to_oklab(px, &mut out);
                })
            });
        }
    });
    set_simd(true);
}

zenbench::main!(bench_kernels);
