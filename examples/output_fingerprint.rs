//! Deterministic fingerprint of quantizer output, for before/after byte
//! comparison across a code change. Prints one hash per (image, palette size).
use zenquant::{OutputFormat, Quality, QuantizeConfig};

fn img(w: usize, h: usize, seed: u32, kind: u32) -> Vec<rgb::RGB<u8>> {
    let mut s = seed | 1;
    let mut g = move || {
        s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (s >> 24) as u8
    };
    (0..w * h)
        .map(|i| {
            let (x, y) = (i % w, i / w);
            match kind {
                // Smooth gradient: maximal tie density in palette distance,
                // which is exactly where argmin tie-breaking is observable.
                0 => rgb::RGB::new((x * 255 / w) as u8, (y * 255 / h) as u8, 128),
                // Noise: exercises the whole palette.
                1 => rgb::RGB::new(g(), g(), g()),
                // Few-color flat regions: drives small effective palettes.
                _ => {
                    let c = (((x / 37) + (y / 41)) % 5) as u8;
                    rgb::RGB::new(c * 50, 255 - c * 50, c * 30)
                }
            }
        })
        .collect()
}

fn fnv(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x1000_0000_01b3);
    }
    h
}

fn main() {
    // `--features _dev` + arg "scalar" forces the magetypes scalar tier, so
    // the two tiers' quantizer output can be compared byte-for-byte. This
    // matters because scalar `mul_add` is `a*b + c` (two roundings) while NEON
    // uses a fused `vfmaq_f32` (one), so the tiers can compute different
    // squared distances and pick different palette entries near ties.
    #[cfg(feature = "_dev")]
    if std::env::args().any(|a| a == "scalar") {
        #[cfg(target_arch = "aarch64")]
        archmage::NeonToken::dangerously_disable_token_process_wide(true)
            .expect("tier must be toggleable; drop -C target-cpu=native");
        #[cfg(target_arch = "x86_64")]
        archmage::X64V3Token::dangerously_disable_token_process_wide(true)
            .expect("tier must be toggleable; drop -C target-cpu=native");
        eprintln!("[fingerprint] SIMD tier disabled — running scalar");
    }

    for kind in 0..3u32 {
        let (w, h) = (256usize, 256usize);
        let px = img(w, h, 7 + kind, kind);
        for &ncol in &[8usize, 16, 32, 64, 96, 128, 192, 256] {
            for (fname, fmt) in [("gif", OutputFormat::Gif), ("png", OutputFormat::Png)] {
                let cfg = QuantizeConfig::new(fmt)
                    .with_max_colors(ncol as u32)
                    .with_quality(Quality::Best);
                let r = zenquant::quantize(&px, w, h, &cfg).expect("quantize");
                let mut pal_bytes = Vec::new();
                for c in r.palette() {
                    pal_bytes.extend_from_slice(c);
                }
                println!(
                    "kind{kind} n{ncol} {fname} idx={:016x} pal={:016x} len={}",
                    fnv(r.indices()),
                    fnv(&pal_bytes),
                    r.palette().len()
                );
            }
        }
    }
}
