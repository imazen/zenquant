//! Opt-in observation of the same values consumed by the quantizer.
use crate::{AlphaMode, OutputFormat, Quality, QuantizeConfig, QuantizeTuning, dither, remap};

/// Effective settings and executed stages for one successful quantization.
///
/// Returned only by [`crate::quantize_with_dispatch`] and
/// [`crate::quantize_rgba_with_dispatch`]. This observes automatic choices;
/// it does not configure the quantizer. Exact-palette inputs bypass masking,
/// refinement, dithering and joint optimization even when requested.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct EffectiveDispatch {
    /// Requested output format; format defaults have already been resolved below.
    pub format: OutputFormat,
    /// Quality after any target-tier override.
    pub quality: Quality,
    /// `Quality`, `Balanced` or `Compression`, after target-tier override.
    pub run_priority: &'static str,
    /// Target-selected tier: `best-quality`, `best-balanced`,
    /// `best-compression`, `balanced-compression`, `fast-compression`, or None.
    pub target_tier: Option<&'static str>,
    /// Resolved mode: `None`, `Adaptive`, `SierraLite`, `BlueNoise`, `Linear`,
    /// or `Ordered`. Fast+Adaptive resolves to Ordered.
    pub dither: &'static str,
    /// Whether the resolved dither was actually invoked (false for exact palettes).
    pub dither_used: bool,
    /// Resolved strength, including format defaults and target-tier scaling.
    pub dither_strength: f32,
    /// `Full` or `Binary` alpha handling.
    pub alpha: &'static str,
    /// Palette ordering resolved from the format: `Luminance` or `DeltaMinimize`.
    pub palette_sort: &'static str,
    /// Whether GIF frequency reordering is enabled by the format.
    pub frequency_reorder: bool,
    /// Whether an exact palette bypassed the quantization stages.
    pub exact_palette: bool,
    /// Whether perceptual masking was invoked.
    pub masking: bool,
    /// Actual pixel-level k-means refinement iteration budget; zero on bypass.
    pub kmeans_iterations: usize,
    /// Requested per-iteration sample cap; zero means unbounded.
    pub kmeans_sample_cap: usize,
    /// Actual post-dither run penalty, including format scaling; zero on bypass.
    pub run_lambda: f32,
    /// `viterbi`, `run-extend`, or None when no post-dither refinement ran.
    pub run_refinement: Option<&'static str>,
    /// Whether this build contains the joint optimizer.
    pub joint_available: bool,
    /// Whether joint optimization was invoked for this call.
    pub joint_used: bool,
    /// Actual joint tolerance, only when the optimizer was invoked.
    pub joint_tolerance: Option<f32>,
    /// False: the current joint predictor ignores the requested deflate effort.
    pub joint_deflate_effort_used: bool,
    /// Whether quality metrics were requested or implicitly enabled.
    pub metric: bool,
}

pub(crate) trait Sink: Default {
    fn resolved(&mut self, _: &QuantizeConfig, _: &QuantizeTuning) {}
    fn exact(&mut self) {}
    fn masking(&mut self, _: bool, _: usize) {}
    fn dither(&mut self, _: &dither::DitherParams<'_>) {}
    fn refinement(&mut self, _: f32, _: bool) {}
    #[cfg(feature = "joint")]
    fn joint(&mut self, _: f32) {}
}
#[derive(Default)]
pub(crate) struct Capture(Option<EffectiveDispatch>);
impl Sink for Capture {
    fn resolved(&mut self, c: &QuantizeConfig, t: &QuantizeTuning) {
        let priority = match c.run_priority {
            remap::RunPriority::Quality => "Quality",
            remap::RunPriority::Balanced => "Balanced",
            remap::RunPriority::Compression => "Compression",
        };
        self.0 = Some(EffectiveDispatch {
            format: c.output_format,
            quality: c.quality,
            run_priority: priority,
            target_tier: c.target_ssim2.map(|_| match (c.quality, c.run_priority) {
                (Quality::Best, remap::RunPriority::Quality) => "best-quality",
                (Quality::Best, remap::RunPriority::Balanced) => "best-balanced",
                (Quality::Best, remap::RunPriority::Compression) => "best-compression",
                (Quality::Balanced, _) => "balanced-compression",
                (Quality::Fast, _) => "fast-compression",
            }),
            dither: match c.dither_mode {
                dither::DitherMode::None => "None",
                dither::DitherMode::Adaptive => "Adaptive",
                dither::DitherMode::SierraLite => "SierraLite",
                dither::DitherMode::BlueNoise => "BlueNoise",
                dither::DitherMode::Linear => "Linear",
                dither::DitherMode::Ordered => "Ordered",
            },
            dither_used: false,
            dither_strength: t.dither_strength,
            alpha: match t.alpha_mode {
                AlphaMode::Full => "Full",
                AlphaMode::Binary => "Binary",
            },
            palette_sort: match t.sort_strategy {
                crate::palette::PaletteSortStrategy::Luminance => "Luminance",
                crate::palette::PaletteSortStrategy::DeltaMinimize => "DeltaMinimize",
            },
            frequency_reorder: t.gif_frequency_reorder,
            exact_palette: false,
            masking: false,
            kmeans_iterations: 0,
            kmeans_sample_cap: c.kmeans_sample_cap,
            run_lambda: 0.,
            run_refinement: None,
            joint_available: cfg!(feature = "joint"),
            joint_used: false,
            joint_tolerance: None,
            joint_deflate_effort_used: false,
            metric: c.compute_metric,
        });
    }
    fn exact(&mut self) {
        self.0.as_mut().unwrap().exact_palette = true;
    }
    fn masking(&mut self, used: bool, iterations: usize) {
        let r = self.0.as_mut().unwrap();
        r.masking = used;
        r.kmeans_iterations = iterations;
    }
    fn dither(&mut self, p: &dither::DitherParams<'_>) {
        let r = self.0.as_mut().unwrap();
        r.dither_used = true;
        // Observe the actual parameters at the dither call site.
        debug_assert_eq!(r.dither_strength, p.dither_strength);
        r.dither_strength = p.dither_strength;
    }
    fn refinement(&mut self, lambda: f32, viterbi: bool) {
        let r = self.0.as_mut().unwrap();
        r.run_lambda = lambda;
        r.run_refinement = Some(if viterbi { "viterbi" } else { "run-extend" });
    }
    #[cfg(feature = "joint")]
    fn joint(&mut self, tolerance: f32) {
        let r = self.0.as_mut().unwrap();
        r.joint_used = true;
        r.joint_tolerance = Some(tolerance);
    }
}

#[cfg(test)]
mod tests;

/// Quantization error with the observation available at the point of failure.
/// Invalid inputs which fail before configuration resolution have no report.
#[derive(Debug)]
#[non_exhaustive]
pub struct DispatchError {
    /// The unchanged underlying quantizer error.
    pub error: crate::QuantizeError,
    /// Resolved configuration and stages entered before the error.
    pub dispatch: Option<alloc::boxed::Box<EffectiveDispatch>>,
}
impl core::fmt::Display for DispatchError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.error.fmt(f)
    }
}
impl core::error::Error for DispatchError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        Some(&self.error)
    }
}
impl Capture {
    pub(crate) fn complete(
        self,
        result: Result<crate::QuantizeResult, crate::QuantizeError>,
    ) -> Result<(crate::QuantizeResult, EffectiveDispatch), DispatchError> {
        match result {
            Ok(result) => Ok((result, self.0.expect("resolved before successful return"))),
            Err(error) => Err(DispatchError {
                error,
                dispatch: self.0.map(alloc::boxed::Box::new),
            }),
        }
    }
}
