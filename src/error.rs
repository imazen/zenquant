use thiserror::Error;

#[derive(Debug, Error)]
pub enum QuantizeError {
    #[error("image dimensions cannot be zero")]
    ZeroDimension,

    #[error("pixel buffer length {len} does not match dimensions {width}x{height}")]
    DimensionMismatch {
        len: usize,
        width: usize,
        height: usize,
    },

    #[error("max_colors must be between 2 and 256, got {0}")]
    InvalidMaxColors(u32),
}
