# zenquant benchmarks — methodology & reproduction

How zenquant's comparisons are run, and how to read the committed result files.
The headline cross-library table in the root [`README.md`](../README.md) comes
from the `quantizer_comparison` example; the other files here are internal A/B
optimization runs.

## What the cross-library comparison measures

`examples/quantizer_comparison.rs` quantizes the same decoded images with every
contender, then scores and (optionally) times each one. Contenders:

- **zenquant** — `Quality::Fast`, `Balanced`, `Best`
- **quantette** — k-means mode
- **imagequant** — `s1 d50`, `s1 d100`, `s4 d100` (speed/dither variants)
- **quantizr**
- **color_quant**

Quality is scored with three independent metrics so no single bias dominates:
**butteraugli**, **SSIMULACRA2** (via `fast-ssim2`), and **DSSIM** (via
`dssim-core`). Encoded size is measured by actually encoding: PNG through
[zenpng](https://github.com/imazen/zenpng)'s aggressive deflate, GIF through the
`gif` crate. All contenders get identical inputs — same images, same dimensions,
same 256-color budget, same default dithering — so the comparison is
apples-to-apples.

## Fairness guarantees (`--benchmark` mode)

The default run is multi-threaded across images and reports **approximate** times
(printed as `~ms`) — fine for the quality columns, not for timing claims. Pass
`--benchmark` for honest timing; it changes the timing discipline to:

- **Single-thread.** `--benchmark` forces one worker thread, so each quantizer is
  timed without contention. Never compare a thread-pooled run against a
  single-threaded one.
- **Warm-up + best-of.** Each quantizer runs 5 times per image; the first run is
  a discarded warm-up, and the fastest of the remaining timed runs is reported
  (min-of-runs rejects scheduler/turbo noise toward the slow side).
- **No I/O in the timed region.** Images are decoded into in-RAM pixel buffers
  *before* timing starts. The timed closure calls only the quantizer; no file
  open/read/decode/encode/write is measured. The result is kept (not dropped) so
  it isn't optimized away.
- **No `-C target-cpu=native`.** Builds use archmage's runtime SIMD dispatch
  (AVX2+FMA / NEON / WASM128 with scalar fallback), which is what ships. Native
  builds bake in ISA extensions and give misleading numbers.

This harness reports a per-quantizer best-of-N rather than the interleaved,
paired A/B statistics that [zenbench](https://github.com/imazen/zenbench)
provides. Porting `quantizer_comparison`'s timing loop to zenbench is the way to
get tight relative-speed confidence intervals and publishable SVG charts; it's
the preferred path for new timing benchmarks in this family.

## Reproduce

```sh
git clone https://github.com/imazen/zenquant && cd zenquant
git checkout <commit>      # the commit named in the result you're reproducing

# quality table + interactive HTML report (multi-threaded, approximate timing):
cargo run --example quantizer_comparison --release -- gb82-sc,cid22,clic2025 /tmp/output 20

# rigorous timing (single-thread, warm-up + best-of, I/O excluded):
cargo run --example quantizer_comparison --release -- cid22 /tmp/output 20 --benchmark
```

Corpus names (`gb82-sc`, `cid22`, `clic2025`) are resolved through the
`codec-corpus` dev-dependency. Competitors are dev-dependencies, so `cargo` pins
them for you; the versions used for committed results (pin these if reproducing
elsewhere):

| Crate | Version | Role |
|-------|---------|------|
| [`imagequant`](https://crates.io/crates/imagequant) | 4 | competitor quantizer |
| [`quantizr`](https://crates.io/crates/quantizr) | 1 | competitor quantizer |
| [`quantette`](https://crates.io/crates/quantette) | 0.5 (`kmeans`, `std`) | competitor quantizer |
| [`color_quant`](https://crates.io/crates/color_quant) | 1 | competitor quantizer |
| [`butteraugli`](https://crates.io/crates/butteraugli) | 0.9 | quality metric |
| [`fast-ssim2`](https://crates.io/crates/fast-ssim2) | 0.8.0 | SSIMULACRA2 metric |
| [`dssim-core`](https://crates.io/crates/dssim-core) | 3 | DSSIM metric |
| [`zenpng`](https://github.com/imazen/zenpng) | git | PNG size oracle |
| [`gif`](https://crates.io/crates/gif) | 0.14 | GIF size oracle |

The headline table in the root README is from the **2026-03-04** comparison run
(same date as the linked interactive visual comparison). When you regenerate it,
record the CPU/RAM/OS and `rustc -V` of the machine in the result file's header.

## Result files

Each committed run lands as `benchmarks/<topic>_<YYYY-MM-DD>.md` (this directory's
`*.md`/`*.log`/`*.csv`/`*.tsv` are gitignored *except* this README, so result
files are committed deliberately). Current files:

- `simd_bench_2026-03-03.md` — internal SIMD A/B: batch sRGB→OKLab conversion,
  palette nearest-neighbor cache, and k-means SIMD. CID22-512, WSL2/AVX2+FMA.
- `shared_oklab_buffer_2026-03-04.md` — internal A/B: compute the OKLab buffer
  once at pipeline entry and share it across masking/histogram/k-means/dither/
  viterbi/joint instead of re-converting per stage. CID22-512.

Each file states, in its header, the commit(s) and platform the numbers came
from. Do not commit numbers you didn't generate, don't extrapolate one size to
another (measure each), and report memory only from heaptrack / `time -v`, never
from estimates.

## Charts (what to plot for which decision)

| Question | Chart |
|----------|-------|
| "Which quantizer is fastest?" | horizontal **bar**, sorted by throughput (MP/s); single-thread |
| "Size vs quality?" | **RD / Pareto scatter**: x = encoded bytes (PNG/GIF), y = SSIMULACRA2 / butteraugli / DSSIM; one point per quality preset, show the frontier |
| "How does it scale with image size?" | **line**, x = pixels (log); fit `total = α + β·pixels` and report both the fixed overhead and the per-pixel slope |
| "Is the A/B delta real / how noisy?" | **violin** or PDF of per-call times, or a paired 95% CI |

Avoid pie / 3D / dual-axis charts — they obscure the comparison.
