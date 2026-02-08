# zenquant

AQ-informed palette quantizer — perceptual masking meets color quantization.

## Credits

This project was informed by prior work in palette quantization:

- **[libimagequant](https://github.com/ImageOptim/libimagequant)** (GPL-3.0) — The gold standard for perceptual palette quantization. Our approach to adaptive quantization masking and k-means refinement was influenced by libimagequant's algorithms, though our implementation uses a different masking model (butteraugli-derived AQ) and histogram strategy.

- **[quantizr](https://github.com/nicoshev/quantizr)** (MIT) — A clean Rust median-cut quantizer. Used as a reference for Rust palette quantization patterns.

- **Claude** (Anthropic) — AI-assisted development. Not all code manually reviewed — review critical paths before production use.

## License

Sustainable, large-scale open source work requires a funding model, and I have been
doing this full-time for 15 years. If you are using this for closed-source development
AND make over $1 million per year, you'll need to buy a commercial license at
https://www.imazen.io/pricing

Commercial licenses are similar to the Apache 2 license but company-specific, and on
a sliding scale. You can also use this under the AGPL v3.
