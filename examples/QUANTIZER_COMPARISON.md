# Quantizer Comparison Viewer

Visual comparison tool for palette quantizers. Generates an interactive HTML
report with slider, diff, and zoom views.

## Usage

```bash
cargo run --example quantizer_comparison --release -- <corpus> <output_dir> [max_images]
```

Corpus names: `cid22`, `clic2025`, `gb82-sc`. Images are downloaded lazily
via codec-corpus.

```bash
# Compare on 5 CID22 images
cargo run --example quantizer_comparison --release -- cid22 /mnt/v/output/zenquant/comparison 5

# Full CLIC2025 corpus
cargo run --example quantizer_comparison --release -- clic2025 /mnt/v/output/zenquant/clic
```

Open `output_dir/index.html` in a browser (works from `file://`).

## Quantizers

| Quantizer | Dithering | Notes |
|-----------|-----------|-------|
| zenquant | Floyd-Steinberg 50% | AQ masking, OKLab, Viterbi DP |
| imagequant | Floyd-Steinberg 50% | libimagequant, quality 0-80 |
| quantizr | Floyd-Steinberg 50% | quantizr crate |
| color_quant | None | NeuQuant, no dithering support |
| exoquant | Floyd-Steinberg | Binary on/off, KMeans optimizer |

## Viewer UX

### Selection Model

The button strip shows `☑ original` followed by all quantizer names.

**Original locked (default):** Original is always the left side. Click any
quantizer to compare it against original. The slider starts at 20% so you
see mostly the quantizer you picked.

**Original unlocked:** Click the original checkbox to uncheck it. Now you can
pick any two quantizers to compare against each other. Click original again
to re-lock it.

Whichever two are selected, left/right is determined by button order. Each
selected button shows an `L` or `R` badge and is colored blue (left) or
purple (right). The slider labels at the bottom of the viewport match these
colors.

Clicking a quantizer that's already selected snaps the slider to show 80%
of that side. Clicking a new quantizer replaces the older selection and
snaps to 80% of the new pick.

### Comparison Modes

**Slider** (default): Left image underneath, right image clipped on top.
Drag the slider handle (or click anywhere on the image) to reveal more of
either side. 40px touch target on the handle.

**Diff**: `mix-blend-mode: difference` overlay. Black pixels are identical,
bright pixels differ. Toggle with the Diff button or `d` key.

### Zoom

All zoom levels use explicit pixel dimensions on both images so left and
right stay perfectly aligned.

| Level | Behavior |
|-------|----------|
| Fit | Fill viewport, maintain aspect ratio, `image-rendering: auto` |
| 1:1 | One source pixel = one device pixel (CSS size = `naturalSize / devicePixelRatio`) |
| 2:1 | Two device pixels per source pixel |
| 3:1 | Three device pixels per source pixel |

Zoomed modes use `image-rendering: pixelated` / `crisp-edges`. The current
DPR is shown in the zoom bar.

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `←` `→` | Previous / next image |
| `0` | Toggle original lock |
| `1`-`5` | Select quantizer |
| `f` | Fit zoom |
| `n` | Native 1:1 zoom |
| `d` | Toggle diff mode |

### Metrics

Per-image table shows BA (butteraugli, lower=better), SS2 (SSIMULACRA2,
higher=better), DSSIM (lower=better), PNG/GIF/WebP file sizes, and
quantization time. Best values are highlighted green. Clicking a row selects
that quantizer.

Collapsible summary section shows averages across all images.

## Output Structure

```
output_dir/
  index.html              # Self-contained viewer (inline CSS + JS + JSON data)
  {image_stem}/
    original.png          # Truecolor RGB
    zenquant.png          # Indexed PNG, 256 colors
    imagequant.png
    quantizr.png
    color_quant.png
    exoquant.png
```

GIF and WebP are encoded in-memory for file size measurement only; not saved
to disk.

## Metrics Context

Butteraugli and SSIMULACRA2 were designed for lossy compression artifacts,
not quantization with error diffusion. A quantizer with worse metric scores
may look better visually — that's the whole point of this tool. Use the
slider to judge quality directly; treat the numbers as supplementary data.
