// crates/mid-math/src/fixed/vec2.rs
//! 2D fixed-point vector — deterministic on all platforms.

use core::fmt;
use core::ops::{Add, AddAssign, Neg, Sub, SubAssign};
use super::fixed::Fixed;

/// 2D fixed-point vector. All arithmetic is integer-only.
///
/// Convert to/from `Vec2` only at system boundaries (rendering, asset loading).
/// Never mix fixed-point and float in the deterministic simulation loop.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct FixedVec2<const FRAC: u32> {
    pub x: Fixed<FRAC>,
    pub y: Fixed<FRAC>,
}

impl<const FRAC: u32> FixedVec2<FRAC> {
    pub const ZERO: Self = Self { x: Fixed::ZERO, y: Fixed::ZERO };

    #[inline(always)]
    pub const fn new(x: Fixed<FRAC>, y: Fixed<FRAC>) -> Self { Self { x, y } }

    #[inline(always)]
    pub const fn from_raw(x: i64, y: i64) -> Self {
        Self { x: Fixed::from_raw(x), y: Fixed::from_raw(y) }
    }

    /// From `i32` pair — exact.
    #[inline]
    pub fn from_i32(x: i32, y: i32) -> Self {
        Self { x: Fixed::from_i32(x), y: Fixed::from_i32(y) }
    }

    /// From `f32` pair — **boundary use only**.
    #[inline]
    pub fn from_f32(x: f32, y: f32) -> Self {
        Self { x: Fixed::from_f32(x), y: Fixed::from_f32(y) }
    }

    /// To `Vec2` (f32) — **boundary use only** (rendering, debug).
    #[inline]
    pub fn to_vec2(self) -> crate::Vec2 {
        crate::Vec2::new(self.x.to_f32(), self.y.to_f32())
    }

    /// From `Vec2` (f32) — **boundary use only**.
    #[inline]
    pub fn from_vec2(v: crate::Vec2) -> Self {
        Self { x: Fixed::from_f32(v.x), y: Fixed::from_f32(v.y) }
    }

    // ── Arithmetic ────────────────────────────────────────────────────────────

    /// Dot product. Each component mul uses `i128` intermediate.
    #[inline]
    pub fn dot(self, rhs: Self) -> Fixed<FRAC> {
        self.x.fixed_mul(rhs.x) + self.y.fixed_mul(rhs.y)
    }

    /// Squared length.
    #[inline] pub fn length_sq(self) -> Fixed<FRAC> { self.dot(self) }

    /// Scale all components by a fixed-point scalar.
    #[inline]
    pub fn scale(self, s: Fixed<FRAC>) -> Self {
        Self { x: self.x.fixed_mul(s), y: self.y.fixed_mul(s) }
    }

    /// Component-wise fixed-point multiply.
    #[inline]
    pub fn mul_elem(self, rhs: Self) -> Self {
        Self { x: self.x.fixed_mul(rhs.x), y: self.y.fixed_mul(rhs.y) }
    }

    /// Perpendicular (90° CCW): `(-y, x)`.
    #[inline] pub fn perp(self) -> Self { Self { x: -self.y, y: self.x } }

    /// Signed perp-dot: `x*rhs.y - y*rhs.x`. Positive = rhs is CCW from self.
    #[inline]
    pub fn perp_dot(self, rhs: Self) -> Fixed<FRAC> {
        self.x.fixed_mul(rhs.y) - self.y.fixed_mul(rhs.x)
    }

    /// Manhattan distance: `|a.x - b.x| + |a.y - b.y|`.
    #[inline]
    pub fn manhattan_distance(self, rhs: Self) -> Fixed<FRAC> {
        (self.x - rhs.x).abs() + (self.y - rhs.y).abs()
    }

    #[inline] pub fn abs(self) -> Self { Self { x: self.x.abs(), y: self.y.abs() } }
    #[inline] pub fn min(self, r: Self) -> Self { Self { x: self.x.min(r.x), y: self.y.min(r.y) } }
    #[inline] pub fn max(self, r: Self) -> Self { Self { x: self.x.max(r.x), y: self.y.max(r.y) } }
    #[inline] pub fn clamp(self, lo: Self, hi: Self) -> Self {
        Self { x: self.x.clamp(lo.x, hi.x), y: self.y.clamp(lo.y, hi.y) }
    }

    /// Linear interpolation per component. `t` in `[Fixed::ZERO, Fixed::one()]`.
    #[inline]
    pub fn lerp(self, rhs: Self, t: Fixed<FRAC>) -> Self {
        Self { x: self.x.lerp(rhs.x, t), y: self.y.lerp(rhs.y, t) }
    }
}

impl<const FRAC: u32> Add for FixedVec2<FRAC> {
    type Output = Self;
    #[inline] fn add(self, r: Self) -> Self { Self { x: self.x + r.x, y: self.y + r.y } }
}
impl<const FRAC: u32> AddAssign for FixedVec2<FRAC> {
    #[inline] fn add_assign(&mut self, r: Self) { *self = *self + r; }
}
impl<const FRAC: u32> Sub for FixedVec2<FRAC> {
    type Output = Self;
    #[inline] fn sub(self, r: Self) -> Self { Self { x: self.x - r.x, y: self.y - r.y } }
}
impl<const FRAC: u32> SubAssign for FixedVec2<FRAC> {
    #[inline] fn sub_assign(&mut self, r: Self) { *self = *self - r; }
}
impl<const FRAC: u32> Neg for FixedVec2<FRAC> {
    type Output = Self;
    #[inline] fn neg(self) -> Self { Self { x: -self.x, y: -self.y } }
}

impl<const FRAC: u32> fmt::Debug for FixedVec2<FRAC> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FixedVec2<{}>({:.4}, {:.4})", FRAC, self.x.to_f64(), self.y.to_f64())
    }
}
impl<const FRAC: u32> fmt::Display for FixedVec2<FRAC> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:.4}, {:.4})", self.x.to_f64(), self.y.to_f64())
    }
  }
