use zenquant::oklab::srgb_to_oklab;
use std::collections::BTreeMap;

fn count_bins(pixels: &[rgb::RGB<u8>], bits: u32) -> usize {
    let max_val = (1u32 << bits) - 1;
    let scale = max_val as f32;
    let mut bins: BTreeMap<u32, u32> = BTreeMap::new();
    for p in pixels {
        let lab = srgb_to_oklab(p.r, p.g, p.b);
        let l_bin = ((lab.l * scale).round() as u32).min(max_val);
        let a_bin = (((lab.a + 0.4) * (scale / 0.8)).round() as u32).min(max_val);
        let b_bin = (((lab.b + 0.4) * (scale / 0.8)).round() as u32).min(max_val);
        let key = (l_bin << (bits * 2)) | (a_bin << bits) | b_bin;
        *bins.entry(key).or_default() += 1;
    }
    bins.len()
}

fn main() {
    let dir = std::env::args().nth(1).unwrap_or("/home/lilith/work/codec-corpus/CID22/CID22-512/training".into());
    let max: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(10);

    let mut paths: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|ext| ext == "png" || ext == "jpg" || ext == "jpeg"))
        .collect();
    paths.sort();
    paths.truncate(max);

    eprintln!("{:<36} {:>8} {:>8} {:>8} {:>8}", "Image", "pixels", "4-bit", "5-bit", "6-bit");
    eprintln!("{}", "-".repeat(72));

    for path in &paths {
        let img = image::open(path).unwrap().to_rgb8();
        let pixels: Vec<rgb::RGB<u8>> = img.pixels().map(|p| rgb::RGB { r: p.0[0], g: p.0[1], b: p.0[2] }).collect();
        let name = path.file_stem().unwrap_or_default().to_string_lossy();
        let n4 = count_bins(&pixels, 4);
        let n5 = count_bins(&pixels, 5);
        let n6 = count_bins(&pixels, 6);
        eprintln!("{:<36} {:>8} {:>8} {:>8} {:>8}", &name[..name.len().min(36)], pixels.len(), n4, n5, n6);
    }
}
