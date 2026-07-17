// crates/mid-math/src/curves/interpolate.rs
//! The `Interpolate` trait — implemented by every type a curve can operate on.

use crate::{Vec2, Vec3, Quat, DVec2, DVec3, DQuat};

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

// ── f64 (large-world precision) ─────────────────────────────────────────────
//
// The curve parameter `t` stays `f32` here, matching the bare `f64` scalar
// impl above — precision belongs to the position/value being interpolated,
// not the blend weight, so there's no loss from that. Every curve type in
// this module is already generic over `T: Interpolate`, so these three
// impls are the entire cost of `HermiteSpline<DVec3>`,
// `CatmullRom<DVec3>`, etc. working — nothing else in `curves/` needed to
// change.

impl Interpolate for DVec2 {
    #[inline] fn lerp(self, rhs: DVec2, t: f32) -> DVec2 { self.lerp(rhs, t as f64) }
    #[inline] fn scale(self, s: f32) -> DVec2 { self * (s as f64) }
    #[inline] fn add(self, rhs: DVec2) -> DVec2 { self + rhs }
    #[inline] fn sub(self, rhs: DVec2) -> DVec2 { self - rhs }
}

impl Interpolate for DVec3 {
    #[inline] fn lerp(self, rhs: DVec3, t: f32) -> DVec3 { self.lerp(rhs, t as f64) }
    #[inline] fn scale(self, s: f32) -> DVec3 { self * (s as f64) }
    #[inline] fn add(self, rhs: DVec3) -> DVec3 { self + rhs }
    #[inline] fn sub(self, rhs: DVec3) -> DVec3 { self - rhs }
}

/// Quaternion interpolation uses slerp instead of lerp for correctness.
impl Interpolate for DQuat {
    #[inline] fn lerp(self, rhs: DQuat, t: f32) -> DQuat { self.slerp(rhs, t as f64) }
    #[inline] fn scale(self, s: f32) -> DQuat { self * (s as f64) }
    #[inline] fn add(self, rhs: DQuat) -> DQuat { self + rhs }
    #[inline] fn sub(self, rhs: DQuat) -> DQuat { self - rhs }
    }
