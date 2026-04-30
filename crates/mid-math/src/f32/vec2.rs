// crates/mid-math/src/f32/vec2.rs
//! Vec2 — always scalar, 8 bytes, no SIMD benefit.
//!
//! Used for UV coordinates, 2D physics, screen-space positions.
//! No architecture variants needed — SSE2 operates on 128-bit registers
//! (4 floats), making a 2-float type an awkward fit with wasted lanes.

use core::fmt;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use crate::EPSILON;

/// 2D vector. 8 bytes, no padding. Always scalar on all platforms.
///
/// **C interop:** use [`CVec2`][crate::ffi::types::CVec2] at the FFI boundary.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

impl Vec2 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0 };
    pub const ONE:  Self = Self { x: 1.0, y: 1.0 };
    pub const X:    Self = Self { x: 1.0, y: 0.0 };
    pub const Y:    Self = Self { x: 0.0, y: 1.0 };

    // ── Constructors ──────────────────────────────────────────────────────────

    #[inline(always)]
    pub fn new(x: f32, y: f32) -> Self { Self { x, y } }

    #[inline(always)]
    pub fn splat(v: f32) -> Self { Self { x: v, y: v } }

    #[inline(always)]
    pub fn from_array(a: [f32; 2]) -> Self { Self::new(a[0], a[1]) }

    #[inline(always)]
    pub fn to_array(self) -> [f32; 2] { [self.x, self.y] }

    /// Extend to a Vec3 by appending `z`.
    #[inline(always)]
    pub fn extend(self, z: f32) -> crate::Vec3 {
        crate::Vec3::new(self.x, self.y, z)
    }

    // ── Arithmetic ────────────────────────────────────────────────────────────

    #[inline(always)]
    pub fn dot(self, rhs: Self) -> f32 { self.x*rhs.x + self.y*rhs.y }

    #[inline(always)]
    pub fn length_sq(self) -> f32 { self.dot(self) }

    #[inline(always)]
    pub fn length(self) -> f32 { self.length_sq().sqrt() }

    #[inline(always)]
    pub fn length_recip(self) -> f32 {
        let l = self.length();
        if l < EPSILON { 0.0 } else { 1.0 / l }
    }

    #[inline(always)]
    pub fn normalize(self) -> Self {
        let l = self.length();
        if l < EPSILON { Self::ZERO } else { self / l }
    }

    #[inline(always)]
    pub fn try_normalize(self) -> Option<Self> {
        let rcp = self.length_recip();
        if rcp > 0.0 && rcp.is_finite() { Some(self * rcp) } else { None }
    }

    #[inline(always)]
    pub fn normalize_or_zero(self) -> Self {
        self.try_normalize().unwrap_or(Self::ZERO)
    }

    #[inline(always)]
    pub fn is_normalized(self) -> bool { (self.length_sq() - 1.0).abs() <= 2e-4 }

    #[inline(always)]
    pub fn lerp(self, rhs: Self, t: f32) -> Self {
        self + (rhs - self) * t
    }

    #[inline(always)]
    pub fn distance(self, rhs: Self) -> f32 { (self - rhs).length() }

    #[inline(always)]
    pub fn distance_sq(self, rhs: Self) -> f32 { (self - rhs).length_sq() }

    /// Perpendicular vector (rotated 90° counter-clockwise).
    #[inline(always)]
    pub fn perpendicular(self) -> Self { Self::new(-self.y, self.x) }

    /// Signed angle from `self` to `rhs` in radians.
    #[inline(always)]
    pub fn angle_to(self, rhs: Self) -> f32 {
        let cross = self.x * rhs.y - self.y * rhs.x;
        let dot   = self.dot(rhs);
        cross.atan2(dot)
    }

    // ── Component-wise ops ────────────────────────────────────────────────────

    #[inline(always)]
    pub fn abs(self) -> Self { Self::new(self.x.abs(), self.y.abs()) }

    #[inline(always)]
    pub fn min(self, rhs: Self) -> Self {
        Self::new(self.x.min(rhs.x), self.y.min(rhs.y))
    }

    #[inline(always)]
    pub fn max(self, rhs: Self) -> Self {
        Self::new(self.x.max(rhs.x), self.y.max(rhs.y))
    }

    #[inline(always)]
    pub fn clamp(self, lo: Self, hi: Self) -> Self {
        self.max(lo).min(hi)
    }

    #[inline(always)]
    pub fn floor(self) -> Self { Self::new(self.x.floor(), self.y.floor()) }

    #[inline(always)]
    pub fn ceil(self) -> Self { Self::new(self.x.ceil(), self.y.ceil()) }

    #[inline(always)]
    pub fn round(self) -> Self { Self::new(self.x.round(), self.y.round()) }

    // ── Predicates ────────────────────────────────────────────────────────────

    #[inline(always)]
    pub fn is_finite(self) -> bool { self.x.is_finite() && self.y.is_finite() }

    #[inline(always)]
    pub fn is_nan(self) -> bool { self.x.is_nan() || self.y.is_nan() }

    #[inline(always)]
    pub fn approx_eq(self, rhs: Self) -> bool {
        (self.x - rhs.x).abs() < EPSILON && (self.y - rhs.y).abs() < EPSILON
    }// Add to Vec2 impl in f32/vec2.rs (after the existing approx_eq method)
#[inline(always)]
pub fn approx_eq_eps(self, rhs: Self, eps: f32) -> bool {
    (self.x - rhs.x).abs() < eps && (self.y - rhs.y).abs() < eps
}
}

// ── Operators ─────────────────────────────────────────────────────────────────

impl Add for Vec2 {
    type Output = Self;
    #[inline(always)] fn add(self, r: Self) -> Self { Self::new(self.x+r.x, self.y+r.y) }
}
impl Sub for Vec2 {
    type Output = Self;
    #[inline(always)] fn sub(self, r: Self) -> Self { Self::new(self.x-r.x, self.y-r.y) }
}
impl Neg for Vec2 {
    type Output = Self;
    #[inline(always)] fn neg(self) -> Self { Self::new(-self.x, -self.y) }
}
impl Mul<f32> for Vec2 {
    type Output = Self;
    #[inline(always)] fn mul(self, s: f32) -> Self { Self::new(self.x*s, self.y*s) }
}
impl Mul<Vec2> for f32 {
    type Output = Vec2;
    #[inline(always)] fn mul(self, v: Vec2) -> Vec2 { Vec2::new(self*v.x, self*v.y) }
}
impl Mul for Vec2 {
    type Output = Self;
    #[inline(always)] fn mul(self, r: Self) -> Self { Self::new(self.x*r.x, self.y*r.y) }
}
impl Div<f32> for Vec2 {
    type Output = Self;
    #[inline(always)] fn div(self, s: f32) -> Self { Self::new(self.x/s, self.y/s) }
}
impl AddAssign for Vec2 {
    #[inline(always)] fn add_assign(&mut self, r: Self) { self.x+=r.x; self.y+=r.y; }
}
impl SubAssign for Vec2 {
    #[inline(always)] fn sub_assign(&mut self, r: Self) { self.x-=r.x; self.y-=r.y; }
}
impl MulAssign<f32> for Vec2 {
    #[inline(always)] fn mul_assign(&mut self, s: f32) { self.x*=s; self.y*=s; }
}
impl DivAssign<f32> for Vec2 {
    #[inline(always)] fn div_assign(&mut self, s: f32) { self.x/=s; self.y/=s; }
}

impl PartialEq for Vec2 {
    fn eq(&self, r: &Self) -> bool { self.x == r.x && self.y == r.y }
}
impl Default for Vec2 { fn default() -> Self { Self::ZERO } }

impl fmt::Display for Vec2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.x, self.y)
    }
}

impl From<[f32; 2]> for Vec2 {
    #[inline] fn from(a: [f32; 2]) -> Self { Self::new(a[0], a[1]) }
}
impl From<Vec2> for [f32; 2] {
    #[inline] fn from(v: Vec2) -> Self { [v.x, v.y] }
}
impl From<(f32, f32)> for Vec2 {
    #[inline] fn from(t: (f32, f32)) -> Self { Self::new(t.0, t.1) }
}
impl From<Vec2> for (f32, f32) {
    #[inline] fn from(v: Vec2) -> Self { (v.x, v.y) }
}
