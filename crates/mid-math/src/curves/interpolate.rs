// crates/mid-math/src/curves/interpolate.rs
//! The `Interpolate` trait — implemented by every type a curve can operate on.

use crate::{Vec2, Vec3, Quat};

/// Types that support linear interpolation and scalar scaling.
///
/// Implementing this trait allows any curve type in this module to
/// operate on your custom point type.
pub trait Interpolate: Copy + Clone {
    /// Linear interpolation: `self + (rhs - self) * t`.
    fn lerp(self, rhs: Self, t: f32) -> Self;
    /// Scale by a scalar.
    fn scale(self, s: f32) -> Self;
    /// Add two values.
    fn add(self, rhs: Self) -> Self;
    /// Subtract two values.
    fn sub(self, rhs: Self) -> Self;
}

impl Interpolate for f32 {
    #[inline] fn lerp(self, rhs: f32, t: f32) -> f32 { self + (rhs - self) * t }
    #[inline] fn scale(self, s: f32) -> f32 { self * s }
    #[inline] fn add(self, rhs: f32) -> f32 { self + rhs }
    #[inline] fn sub(self, rhs: f32) -> f32 { self - rhs }
}

impl Interpolate for f64 {
    #[inline] fn lerp(self, rhs: f64, t: f32) -> f64 { self + (rhs - self) * t as f64 }
    #[inline] fn scale(self, s: f32) -> f64 { self * s as f64 }
    #[inline] fn add(self, rhs: f64) -> f64 { self + rhs }
    #[inline] fn sub(self, rhs: f64) -> f64 { self - rhs }
}

impl Interpolate for Vec2 {
    #[inline] fn lerp(self, rhs: Vec2, t: f32) -> Vec2 { self.lerp(rhs, t) }
    #[inline] fn scale(self, s: f32) -> Vec2 { self * s }
    #[inline] fn add(self, rhs: Vec2) -> Vec2 { self + rhs }
    #[inline] fn sub(self, rhs: Vec2) -> Vec2 { self - rhs }
}

impl Interpolate for Vec3 {
    #[inline] fn lerp(self, rhs: Vec3, t: f32) -> Vec3 { self.lerp(rhs, t) }
    #[inline] fn scale(self, s: f32) -> Vec3 { self * s }
    #[inline] fn add(self, rhs: Vec3) -> Vec3 { self + rhs }
    #[inline] fn sub(self, rhs: Vec3) -> Vec3 { self - rhs }
}

/// Quaternion interpolation uses slerp instead of lerp for correctness.
impl Interpolate for Quat {
    #[inline] fn lerp(self, rhs: Quat, t: f32) -> Quat { self.slerp(rhs, t) }
    #[inline] fn scale(self, s: f32) -> Quat { self * s }
    #[inline] fn add(self, rhs: Quat) -> Quat { self + rhs }
    #[inline] fn sub(self, rhs: Quat) -> Quat { self - rhs }
                                                               }
