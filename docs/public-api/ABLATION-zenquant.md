# ABLATION-zenquant.md

**Date:** 2026-06-11  
**Snapshot commit:** 20ff49c8 (main@origin)  
**Surface size:** 162 items (default features = all features — no feature-gated additions)  
**Grep template:** `grep -r "<symbol>" /home/lilith/work/zen/{zenpng,zengif,zenpipe} --include="*.rs" --exclude-dir=target`

---

## Summary

**0 items flagged. Surface is coherent.**

162 items reviewed. No public-API mistakes found under the conservative bar.

---

## Consumer Evidence

| Symbol | Consumers confirmed |
|---|---|
| `zenquant::quantize_rgba` / `quantize` | zenpng (quantize.rs), zengif (zenquant_impl.rs) |
| `zenquant::build_palette_rgba` | zenpng (quantize.rs) |
| `zenquant::QuantizeConfig` | zenpng (quantize.rs), zengif (zenquant_impl.rs) |
| `zenquant::QuantizeResult` | zengif (zenquant_impl.rs: `palette_to_bytes`) |
| `zenquant::OutputFormat` | zenpng (quantize.rs, indexed.rs), zengif (zenquant_impl.rs) |
| `zenquant::Quality` | zenpng (quantize.rs), zengif (zenquant_impl.rs) |
| `zenquant::QuantizeError` | zenpng (error.rs: `#[from]`) |
| `zenquant::RGBA` / `ImgRef` | zenpng (quantize.rs), zengif |

---

## Structural observations (not flagged)

### Dual `QuantizeError` path

`zenquant::QuantizeError` (root re-export) and `zenquant::error::QuantizeError` (submodule) refer to the same type. The root re-export (`pub use error::QuantizeError`) is standard convenience: consumers use the root path, the `error::` submodule preserves discoverability. Not a mistake — standard Rust pattern. Both appear in the API surface because `cargo public-api` traces both paths.

### `zenquant::metric::MpeResult` in return signature

`QuantizeResult::mpe_result()` returns `Option<&zenquant::metric::MpeResult>` but the `metric` module is `pub(crate)` (only `pub` under the `_dev` internal feature). The type is reachable through the returned reference and callers can access all fields (`.score`, `.block_scores`, `.block_cols`, `.block_rows`, `.butteraugli_estimate`, `.ssimulacra2_estimate`) through normal field access on the reference.

This is a minor ergonomic gap: callers cannot name the type in their own function signatures without the `metric` feature. However:
- Scalar accessors on `QuantizeResult` cover 90%+ of cases: `mpe_score()`, `ssimulacra2_estimate()`, `butteraugli_estimate()`
- `mpe_result()` exists for access to `block_scores` (spatial heat map) which has no scalar equivalent
- No confirmed consumer uses `mpe_result()` in the scanned codebase
- Fixing requires either making `metric` pub (adds to surface) or introducing a new `pub struct` wrapper (additive change) — either is a deliberate design choice, not a mistake

Not flagged: the conservative bar requires clear mistakes, not ergonomic gaps. This is a known trade-off worth tracking, not a ship-blocker.

---

## Digest

| Metric | Count |
|---|---|
| Items in surface | 162 |
| Items flagged (Action A) | 0 |
| Items flagged (Action B) | 0 |
| Flag rate | 0% |

**Verdict:** Surface is tightly scoped around config + result + error + top-level quantize/build_palette fns. Internals (metric, dither, histogram, masking, median_cut, oklab, palette, remap, blue_noise, simd) are all `pub(crate)`. Enumeration of `OutputFormat` variants is intentional (PNG/GIF/WebpLossless targets). No leaks.
