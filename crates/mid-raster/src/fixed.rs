// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-raster.md, section "fixed.rs"
// ============================================================================
//! 24.8 fixed point, the only number format the rasterizer's pixel work uses.
//!
//! A coordinate enters as an `f32` pixel position and leaves this module as an
//! integer count of 1/256 pixels. From there on everything is integer
//! arithmetic, so the same input gives the same bytes on every target.

/// Fractional bits of a fixed-point coordinate.
pub const SHIFT: u32 = 8;

/// One pixel in fixed point.
pub const ONE: i64 = 1 << SHIFT;

/// Largest magnitude a converted coordinate can have, 2^20 pixels. It keeps
/// every product the rasterizer forms inside an `i64`.
pub const LIMIT: i64 = 1 << 28;

/// Converts a pixel coordinate to fixed point: `round(v * 256)` with ties away
/// from zero, held inside `-LIMIT..=LIMIT`. A NaN or infinite value gives
/// `None`, which callers treat as "draw nothing".
pub fn from_f32(v: f32) -> Option<i64> {
    if !v.is_finite() {
        return None;
    }
    let scaled = (v * 256.0).round();
    if scaled >= LIMIT as f32 {
        Some(LIMIT)
    } else if scaled <= -(LIMIT as f32) {
        Some(-LIMIT)
    } else {
        Some(scaled as i64)
    }
}

/// `floor(a / b)` for a positive `b`.
#[inline]
pub fn floor_div(a: i64, b: i64) -> i64 {
    a.div_euclid(b)
}

/// `a / b` rounded to the nearest integer with ties up, for a positive `b`.
#[inline]
pub fn div_round(a: i64, b: i64) -> i64 {
    (2 * a + b).div_euclid(2 * b)
}
