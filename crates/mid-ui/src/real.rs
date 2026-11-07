// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-ui.md, section "real.rs"
// ============================================================================
//! The scalar the UI core is generic over.

use core::fmt::Debug;
use core::ops::{Add, Div, Mul, Neg, Sub};

/// Floating-point scalar used for every coordinate and size in the UI core.
/// Implemented for [`f32`] and [`f64`]; the layout, hit-test and draw-list
/// code is written once against this trait.
///
/// Non-finite values (NaN, infinity) propagate through arithmetic and never
/// panic.
pub trait Real:
    Copy
    + Debug
    + PartialEq
    + PartialOrd
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + 'static
{
    /// Additive identity.
    const ZERO: Self;
    /// Multiplicative identity.
    const ONE: Self;
    /// One half.
    const HALF: Self;

    /// Converts from `f32` (exact for both implementations).
    fn from_f32(v: f32) -> Self;
    /// Converts from `f64` (rounds when `Self` is `f32`).
    fn from_f64(v: f64) -> Self;
    /// Converts to `f64` (exact for both implementations).
    fn to_f64(self) -> f64;
    /// `true` when the value is neither NaN nor infinite.
    fn is_finite(self) -> bool;

    /// The smaller of two values; returns `self` when `other` is NaN.
    #[inline]
    fn min(self, other: Self) -> Self {
        if other < self { other } else { self }
    }

    /// The larger of two values; returns `self` when `other` is NaN.
    #[inline]
    fn max(self, other: Self) -> Self {
        if other > self { other } else { self }
    }

    /// `self` clamped to be at least zero. NaN is returned unchanged.
    #[inline]
    fn non_neg(self) -> Self {
        if self < Self::ZERO { Self::ZERO } else { self }
    }
}

impl Real for f32 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const HALF: Self = 0.5;

    #[inline] fn from_f32(v: f32) -> Self { v }
    #[inline] fn from_f64(v: f64) -> Self { v as f32 }
    #[inline] fn to_f64(self) -> f64 { f64::from(self) }
    #[inline] fn is_finite(self) -> bool { f32::is_finite(self) }
}

impl Real for f64 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const HALF: Self = 0.5;

    #[inline] fn from_f32(v: f32) -> Self { f64::from(v) }
    #[inline] fn from_f64(v: f64) -> Self { v }
    #[inline] fn to_f64(self) -> f64 { self }
    #[inline] fn is_finite(self) -> bool { f64::is_finite(self) }
}

#[cfg(test)]
mod tests {
    use super::Real;

    #[test]
    fn min_max_ignore_nan_other() {
        assert_eq!(Real::min(1.0_f32, f32::NAN), 1.0);
        assert_eq!(Real::max(1.0_f64, f64::NAN), 1.0);
        assert_eq!(Real::min(2.0_f32, 1.0), 1.0);
        assert_eq!(Real::max(2.0_f64, 3.0), 3.0);
    }

    #[test]
    fn non_neg_clamps_and_keeps_nan() {
        assert_eq!(Real::non_neg(-3.0_f32), 0.0);
        assert_eq!(Real::non_neg(3.0_f64), 3.0);
        assert!(Real::non_neg(f32::NAN).is_nan());
    }

    #[test]
    fn conversions_round_trip() {
        assert_eq!(<f64 as Real>::from_f32(0.5), 0.5);
        assert_eq!(<f32 as Real>::from_f64(0.25), 0.25);
        assert_eq!(Real::to_f64(0.75_f32), 0.75);
    }
}
