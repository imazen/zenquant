# zenquant wasm128 SIMD Plan

## Overview

WebAssembly SIMD (wasm128) uses 128-bit v128 registers providing `f32x4` operations. No runtime feature detection — binary either has SIMD or doesn't. Enabled via `RUSTFLAGS="-C target-feature=+simd128"`.

## Crate Strategy

**Use `wide` v1.1** (already a transitive dep via `linear-srgb`). Wide maps `f32x4` to wasm128's v128 on wasm32 targets. On x86-64, the same code uses SSE2/AVX2/AVX-512 via `multiversion`. No dispatch overhead on wasm32 — `multiversed` becomes a no-op.

This means a single codebase works across x86-64 (with runtime dispatch), aarch64, and wasm32 (compile-time feature).

## Hot Paths and Expected Impact

### 1. Palette Nearest-Neighbor Search (HIGH — called per pixel during dithering)

**Current:** `palette.rs:86-105` — scalar brute-force, 256 iterations per pixel.

**SIMD approach:** Process 4 palette entries per iteration with `f32x4`:
```rust
// Pack 4 palette entries' L values into f32x4, same for a, b
// Compute 4 distance_sq values in parallel
// Track min via f32x4 comparison + selection
```

**Expected speedup:** 2-3x on the inner loop. This is the highest-value target since it's called for every pixel during dithering.

### 2. OKLab Batch Conversion (MEDIUM — called once per image)

**Current:** `oklab.rs:50-96` — per-pixel LUT lookup + 3×3 matrix multiply + 3× cbrt.

**SIMD approach:**
- LUT lookups remain scalar (gather not available in wasm128)
- Matrix multiply: 9 FMA operations per pixel → pack 4 pixels' R values, 4 G values, 4 B values
- `cbrt()`: no native wasm128 cbrt — use Newton-Raphson approximation or `wide::f32x4::powf(1.0/3.0)`

**Expected speedup:** 1.5-2x for the matrix multiply portion. cbrt limits gains.

### 3. K-means Distance Computation (MEDIUM — called 16 iterations × histogram_size × 256)

**Current:** `median_cut.rs:222-234` — same pattern as nearest-neighbor.

**SIMD approach:** Same f32x4 distance pattern as palette search. Process 4 centroids per iteration.

**Expected speedup:** 2-3x, but k-means is a smaller fraction of total time.

### 4. Local Contrast Computation (LOW — called once per image)

**Current:** `masking.rs:53-92` — per-pixel neighbor average and squaring.

**SIMD approach:** Process 4 pixels per iteration. Neighbor access is stride-based.

**Expected speedup:** 1.5-2x, but masking is small fraction of total time.

### 5. Floyd-Steinberg Error Diffusion (LOW — data-dependent scatter)

**Not recommended for SIMD.** Error diffusion is inherently sequential (each pixel depends on previous pixels' errors). The 3-component multiply-add per neighbor could use f32x4 (pack L,a,b,0) but the random scatter pattern prevents vectorization of the outer loop.

## Implementation Plan

### Phase 1: Shared SIMD Distance Utility

Add a `simd.rs` module with a portable distance function:

```rust
use wide::f32x4;

/// Compute distance_sq for 4 OKLab colors against a single target.
/// Returns [dist0, dist1, dist2, dist3].
#[inline(always)]
pub fn distance_sq_x4(
    target_l: f32, target_a: f32, target_b: f32,
    pal_l: f32x4, pal_a: f32x4, pal_b: f32x4,
) -> f32x4 {
    let dl = pal_l - f32x4::splat(target_l);
    let da = pal_a - f32x4::splat(target_a);
    let db = pal_b - f32x4::splat(target_b);
    dl * dl + da * da + db * db
}
```

### Phase 2: SIMD Palette Search

Store palette OKLab values in SoA layout (separate L, a, b arrays padded to multiple of 4). Replace the brute-force loop in `Palette::nearest()` with f32x4 distance computation + horizontal min.

### Phase 3: SIMD K-means

Same distance pattern applied to the k-means inner loop in `median_cut.rs`.

### Phase 4: SIMD OKLab Batch Conversion

Structure-of-arrays approach: accumulate 4 pixels' linear RGB values, then perform 4-wide matrix multiply. cbrt approximation via Newton-Raphson:
```rust
fn cbrt_approx_x4(x: f32x4) -> f32x4 {
    // Initial estimate via bit manipulation (Halley's method)
    // 2-3 Newton-Raphson iterations for f32 precision
}
```

## Build Configuration

### For users
```toml
# .cargo/config.toml
[target.wasm32-unknown-unknown]
rustflags = ["-C", "target-feature=+simd128"]
```

### Feature flag
```toml
[features]
default = ["std"]
std = ["linear-srgb/std"]
# No separate wasm feature needed — wide handles it via cfg(target_arch)
```

### Testing
```bash
# Build for wasm32
cargo build --target wasm32-unknown-unknown --no-default-features
# With SIMD
RUSTFLAGS="-C target-feature=+simd128" cargo build --target wasm32-unknown-unknown --no-default-features
```

## Dependencies

- `wide = "1.1"` — add as direct dependency (currently transitive via linear-srgb)
- No other new dependencies needed

## Risk Assessment

- **Low risk:** Distance computation SIMD — pure math, easy to validate
- **Low risk:** SoA palette layout — internal refactor, no API change
- **Medium risk:** cbrt approximation — must validate accuracy doesn't cause visible artifacts
- **No risk:** `wide` dependency — already in the dependency tree via linear-srgb
