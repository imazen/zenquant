# Feedback Log

## 2026-03-03
- User: Research quantette crate thoroughly — k-means, SIMD, color space, dedup, dithering, Wu quantization. Report specific algorithms and optimizations with file paths and line numbers.

## 2026-02-08
- User: "make it optimal at reencoding images that already have 256 colors or less, as well as handling transparency, with modes for optimal gif vs png vs webp vs jxl decisions based on their respective filter and compression algorithms"
- User: "and of their respective level of support for transparency levels"

## 2026-02-09
- User: Research alternatives to k-means for palette refinement — mini-batch k-means, Elkan's triangle inequality, histogram-based refinement, other production approaches

## 2026-02-23
- User: Add OutputFormat::PngMinSize — BlueNoise dithering at 0.1, aggressive run extension (Compression priority, lambda_scale 2.5), joint optimization. For minimum PNG file size.
- User: Research state of the art in PNG optimization tools and libraries — oxipng, pngquant, optipng, zopflipng, ECT, pngcrush, APNG tools, libpng/lodepng/stb_image. Competitive landscape analysis.
- User: "we made a gauntlet of png images that are way better than clic" — Switch benchmarks from CLIC 2025 corpus to the PNG Optimization Gauntlet at /mnt/v/output/gauntlet/ (stratified sampling: 142 truecolor, 144 indexed, 63 APNG, plus reference sets).
- User: "store the interesting file paths. we can figure out how to cut overhead. what was the original png file size?" — Store paths of images where zenpng couldn't beat source size, for future overhead reduction work.

## 2026-03-03
- User: Research https://github.com/IanManske/quantette thoroughly — algorithms, API design, optimizations, dithering, feature set, implementation details. Competitive analysis for zenquant.
- User: Read archmage + magetypes README and source from cargo registry cache. Understand full API for: incant! macro dispatch, #[arcane] attribute, F32x8Backend trait methods, cbrt_midp(), scalar fallback, x86_64 SSE/AVX dispatch, aarch64 NEON dispatch.
