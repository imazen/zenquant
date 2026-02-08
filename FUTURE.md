# Future Work

Items outside the initial scope, ordered roughly by expected impact.

## Near-term

- **zengif integration**: Implement `QuantizerTrait` as a new backend for zengif.
- **SIMD masking pipeline**: Profile the hot path (OKLab conversion, error diffusion, nearest-neighbor search) and port to `wide`/`archmage` where beneficial.
- **Trellis/Viterbi remapping**: Globally optimal scanline-level run optimization, analogous to zenjpeg's trellis for DCT coefficients. Should significantly improve compression ratio for LZW-based formats.

## Medium-term

- **Animation support**: Shared palette across frames with temporal coherence. Key for GIF animation quality.
- **PNG-specific optimization**: Scanline filter selection aware of palette ordering. Delta-sorted palettes should interact well with PNG's sub/up/average filters.
- **VP8L kModifiedZeng sort**: Try Zeng et al.'s algorithm instead of greedy TSP for palette ordering. May produce better results for certain color distributions.

## Research

- **Perceptual loss function**: Replace OKLab Euclidean with a more sophisticated perceptual metric that accounts for spatial masking at the palette selection stage (not just weighting).
- **Adaptive bucket resolution**: Variable histogram resolution — finer buckets in perceptually sensitive regions, coarser in masked regions.
- **Content-adaptive K**: Automatically determine optimal palette size from image content rather than using a fixed max_colors.
