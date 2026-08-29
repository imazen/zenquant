# Changelog

## [Unreleased]

### Added
- `build_palette_with_stop`, `build_palette_rgba_with_stop`,
  `QuantizeResult::remap_with_stop` and `QuantizeResult::remap_rgba_with_stop` —
  the remaining entry points that run k-means refinement and per-scanline
  Viterbi/run-extension work now accept an `enough::Stop` token, matching
  `quantize_with_stop`/`quantize_rgba_with_stop`. Purely additive; the existing
  no-token functions delegate with `enough::Unstoppable`.
- Versioned public-API surface snapshot at `docs/public-api/zenquant.txt`,
  regenerated on every `cargo test` by `tests/public_api_doc.rs`
  (`ZEN_API_DOC=check` verifies in CI's clippy job, `=off` skips elsewhere);
  `just api-doc` / `api-doc-check` recipes.

### Changed
- The `max_pixels` cap (default 120 MP, `QuantizeConfig::with_max_pixels`) now
  covers `QuantizeResult::remap`/`remap_rgba`/`remap_with_prev`/`remap_rgba_with_prev`
  and `build_palette`/`build_palette_rgba`, not just `quantize`/`quantize_rgba` —
  all of them size the same 12-16 B/px scratch buffers from the input dimensions.
  `build_palette*` caps the **sum** across frames, since the frames are
  concatenated into one buffer. The cap is now checked from the declared
  dimensions before the buffer-length check, so an over-cap request reports
  `TooManyPixels` rather than `DimensionMismatch`.
- Cooperative cancellation now also reaches the palette *seeding* and
  histogram-level k-means: the farthest-point seed loop, `kmeans_refine`,
  `wu_quantize_alpha`'s box-split loop and `kmeans_refine_alpha`. These run up to
  32 iterations over the whole histogram with an O(k) nearest search per entry,
  and previously ignored the `stop` token entirely. Cancelling yields the
  centroids seeded/converged so far — a short palette is still a valid palette.
- Cooperative cancellation now reaches the RGBA and full-alpha k-means refine
  loops (`refine_against_pixels_rgba`, `refine_against_pixels_rgba_from_labs`,
  `refine_against_pixels_alpha`), which previously ran all their iterations
  regardless of the `stop` token that `quantize_rgba_with_stop` carries. The RGB
  path already polled. Cancelling returns the centroids refined so far.
- Cooperative cancellation now reaches the `joint` feature's deflate+quantization
  optimizer: the `stop` token passed to `quantize_with_stop` /
  `quantize_rgba_with_stop` is polled at every scanline boundary of the row DP and
  while building the per-pixel candidate table. Cancelling mid-pass returns valid
  indices (already-committed rows keep their optimized values, the rest keep their
  initial ones) rather than running to completion. No public API change — `joint`
  is `pub(crate)`.
- Fixed published package include list: LICENSE-AGPL3 and LICENSE-COMMERCIAL now correctly included; added CHANGELOG.md (bba2630f)

### Fixed
- docs: README overhaul — added `## Quick start`, `lib.rs` badge, dual-license badge, the `with_max_pixels`/`TooManyPixels` resource cap and `quantize_with_stop` cancellation paths, and `PngJoint`/`PngMinSize` formats; split the crates.io README into generated `README.crates.md` (`readme =`), wrapped the benchmark table in `crates.io:skip`, and added `benchmarks/README.md` methodology
- docs(readme): state quantize input element type (`&[rgb::RGB<u8>]` / `&[rgb::RGBA<u8>]`, not `&[u8]`) and show converting a `Vec<u8>` via `rgb::FromSlice::as_rgb`/`as_rgba`; document `palette()`/`palette_rgba()`/`indices()`/`transparent_index()` accessor types; reconcile the `QualityNotMet { min_ssim2, achieved_ssim2 }` field set across examples — fixes a first-try compile gap found by an insulated-developer usability test

## 0.1.1 (2026-03-25)

### Fixed
- Added `scalar` fallback tier to `incant!` dispatch, fixing archmage deprecation warnings
- CI: replaced broken `git =` zensim dependency with published `version = "0.2.0"`

### Changed
- Bumped `archmage` 0.9.3 → 0.9.12
- Bumped `magetypes` 0.9.3 → 0.9.12
- Bumped `linear-srgb` 0.6.0 → 0.6.4 (path dep replaced with crates.io)

## 0.1.0 (2026-03-05)

Initial public release.
