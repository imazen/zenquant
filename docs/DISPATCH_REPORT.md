# Effective dispatch observations

`quantize_with_dispatch` and `quantize_rgba_with_dispatch` take the same pixels,
geometry/config and cancellation token as the existing `*_with_stop` calls.
They return `(QuantizeResult, EffectiveDispatch)` on success. `DispatchError`
retains the underlying error and any observation available at failure, including
quality-floor outcomes; pre-resolution validation errors have no report. The non-exhaustive report
observes settings and executed stages; it does not change them. Ordinary calls
expand the same pipeline without observer code, storage or allocation.

The report includes target-tier quality/priority/strength overrides,
Fast+Adaptive → Ordered, format defaults, pixel-level refinement budget,
actual run lambda/refinement, implicit metrics and whether joint optimization
ran. Exact palettes explicitly record their bypass. `joint_available` and
`joint_used` differ when the build contains joint but this call does not use
it. The current joint predictor ignores requested deflate effort, recorded as
`joint_deflate_effort_used=false`. Joint use describes entry into that existing
optimizer; it does not claim every pixel changed or alter its cancellation
behavior. Actual palette and index bytes remain on `QuantizeResult`.

The report APIs cover single-call RGB/RGBA quantization. Shared-palette
construction and remapping retain their existing APIs and behavior.

Generated-only validation:

```
just dispatch-check
just dispatch-api-check
just dispatch-cost
```

The structural observer sees the actual dither/refinement/joint arguments at
call sites. It covers all three presets, six dither modes, five formats,
no target and all five target tiers in both RGB and RGBA. The independent
standalone harness compares actual palette/index bytes to published 0.1.3,
including both joint build variants, multiple generated shapes/content and
exact-palette bypass. No real corpus is accessed.

Main before this change is 1dc144ca (0.1.4), and differs from published 0.1.3
in cancellation propagation, pixel caps and scan/batching implementations.
The inspected target-tier thresholds and Fast/Adaptive promotion are retained.
Published parity is tested rather than inferred from the main version number.

[Unused-report probe](../benchmarks/dispatch_unused_2026-10-16.txt) compares
the ordinary API with that exact pre-change main in one process. The cost
binary never references a reporting entry point, so it tests unused reporting
without linking the parity reader's report calls. On eight
measured size/content cells, no paired 95% log-Student interval lies wholly
above ratio one. This is evidence for these generated cells, not a universal
throughput guarantee or an optimization claim. The plain macro expansion
contains no reporting statements; an earlier generic-return prototype was
replaced after its tiny-input cost probe detected a difference.
