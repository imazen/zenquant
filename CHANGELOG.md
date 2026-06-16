# Changelog

## [Unreleased]

### Added
- Versioned public-API surface snapshot at `docs/public-api/zenquant.txt`,
  regenerated on every `cargo test` by `tests/public_api_doc.rs`
  (`ZEN_API_DOC=check` verifies in CI's clippy job, `=off` skips elsewhere);
  `just api-doc` / `api-doc-check` recipes.

### Changed
- Fixed published package include list: LICENSE-AGPL3 and LICENSE-COMMERCIAL now correctly included; added CHANGELOG.md (bba2630f)

### Fixed
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
