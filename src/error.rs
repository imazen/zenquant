use thiserror::Error;

/// Error returned from palette quantization operations.
#[derive(Debug, Clone, Error)]
#[non_exhaustive]
pub enum QuantizeError {
    /// Image dimensions cannot be zero.
    #[error("image dimensions cannot be zero")]
    ZeroDimension,

    /// Pixel buffer length does not match the declared width and height.
    #[error("pixel buffer length {len} does not match dimensions {width}x{height}")]
    DimensionMismatch {
        /// Actual length of the pixel buffer.
        len: usize,
        /// Declared image width.
        width: usize,
        /// Declared image height.
        height: usize,
    },

    /// `max_colors` must be between 2 and 256.
    #[error("max_colors must be between 2 and 256, got {0}")]
    InvalidMaxColors(u32),

    /// The quantized image did not meet the minimum SSIM2 quality threshold.
    #[error("quality target not met: wanted SSIM2 >= {min_ssim2:.1}, got {achieved_ssim2:.1}")]
    QualityNotMet {
        /// The minimum SSIM2 score that was required.
        min_ssim2: f32,
        /// The SSIM2 score actually achieved.
        achieved_ssim2: f32,
    },

    /// Image dimension product (`width * height`) overflows `usize`.
    #[error("image dimensions {width}x{height} overflow usize")]
    DimensionOverflow {
        /// Declared image width.
        width: usize,
        /// Declared image height.
        height: usize,
    },

    /// Image pixel count exceeds the configured `max_pixels` cap.
    ///
    /// See [`QuantizeConfig::with_max_pixels`](crate::QuantizeConfig::with_max_pixels).
    #[error("image has {pixels} pixels, exceeding the max_pixels cap of {max}")]
    TooManyPixels {
        /// Total pixels (`width * height`) of the input.
        pixels: usize,
        /// The configured maximum.
        max: usize,
    },

    /// A caller-supplied palette index is out of range for the active palette.
    #[error("palette index {index} is out of range for palette of length {palette_len}")]
    InvalidIndex {
        /// The offending index value.
        index: u32,
        /// The current palette length.
        palette_len: usize,
    },

    /// The quantize was cancelled via the cooperative [`enough::Stop`] token
    /// passed to [`quantize_with_stop`](crate::quantize_with_stop) /
    /// [`quantize_rgba_with_stop`](crate::quantize_rgba_with_stop).
    #[error("quantize cancelled: {0:?}")]
    Cancelled(enough::StopReason),
}
