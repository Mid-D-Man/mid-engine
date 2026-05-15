// crates/mid-math/src/fixed/vec3.rs
//! 3D fixed-point vector — deterministic on all platforms.

use core::fmt;
use core::ops::{Add, AddAssign, Neg, Sub, SubAssign};
use super::fixed::Fixed;

/// 3D fixed-point vector. All arithmetic is integer-only.
///
/// Convert to/from `Vec3` only at system boundaries.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct FixedVec3<const FRAC: u32> {
    pub x: Fixed<FRAC>,
    pub y: Fixed<FRAC>,
    pub z: Fixed<FRAC>,
}

impl<const FRAC: u32> FixedVec3<FRAC> {
    pub const ZERO: Self = Self { x: Fixed::ZERO, y: Fixed::ZERO, z: Fixed::ZERO };

    #[inline(always)]
    pub const fn new(x: Fixed<FRAC>, y: Fixed<FRAC>, z: Fixed<FRAC>) -> Self { Self { x, y, z } }

    #[inline(always)]
    pub const fn from_raw(x: i64, y: i64, z: i64) -> Self {
        Self { x: Fixed::from_raw(x), y: Fixed::from_raw(y), z: Fixed::from_raw(z) }
    }

    #[inline]
    pub fn from_i32(x: i32, y: i32, z: i32) -> Self {
        Self { x: Fixed::from_i32(x), y: Fixed::from_i32(y), z: Fixed::from_i32(z) }
    }

    /// From `f32` triple — **boundary use only**.
    #[inline]
    pub fn from_f32(x: f32, y: f32, z: f32) -> Self {
        Self { x: Fixed::from_f32(x), y: Fixed::from_f32(y), z: Fixed::from_f32(z) }
    }

    /// To `Vec3` (f32) — **boundary use only**.
    #[inline]
    pub fn to_vec3(self) -> crate::Vec3 {
        crate::Vec3::new(self.x.to_f32(), self.y.to_f32(), self.z.to_f32())
    }

    /// From `Vec3` (f32) — **boundary use only**.
    #[inline]
    pub fn from_vec3(v: crate::Vec3) -> Self {
        Self { x: Fixed::from_f32(v.x), y: Fixed::from_f32(v.y), z: Fixed::from_f32(v.z) }
    }

    // ── Arithmetic ────────────────────────────────────────────────────────────

    #[inline]
    pub fn dot(self, rhs: Self) -> Fixed<FRAC> {
        self.x.fixed_mul(rhs.x) + self.y.fixed_mul(rhs.y) + self.z.fixed_mul(rhs.z)
    }

    #[inline] pub fn length_sq(self) -> Fixed<FRAC> { self.dot(self) }

    /// Cross product. Each multiplication uses `i128` intermediate.
    #[inline]
    pub fn cross(self, rhs: Self) -> Self {
        Self {
            x: self.y.fixed_mul(rhs.z) - self.z.fixed_mul(rhs.y),
            y: self.z.fixed_mul(rhs.x) - self.x.fixed_mul(rhs.z),
            z: self.x.fixed_mul(rhs.y) - self.y.fixed_mul(rhs.x),
        }
    }

    #[inline]
    pub fn scale(self, s: Fixed<FRAC>) -> Self {
        Self { x: self.x.fixed_mul(s), y: self.y.fixed_mul(s), z: self.z.fixed_mul(s) }
    }

    #[inline]
    pub fn mul_elem(self, rhs: Self) -> Self {
        Self {
            x: self.x.fixed_mul(rhs.x),
            y: self.y.fixed_mul(rhs.y),
            z: self.z.fixed_mul(rhs.z),
        }
    }

    #[inline]
    pub fn manhattan_distance(self, rhs: Self) -> Fixed<FRAC> {
        (self.x - rhs.x).abs() + (self.y - rhs.y).abs() + (self.z - rhs.z).abs()
    }

    #[inline] pub fn abs(self) -> Self { Self { x: self.x.abs(), y: self.y.abs(), z: self.z.abs() } }
    #[inline] pub fn min(self, r: Self) -> Self { Self { x: self.x.min(r.x), y: self.y.min(r.y), z: self.z.min(r.z) } }
    #[inline] pub fn max(self, r: Self) -> Self { Self { x: self.x.max(r.x), y: self.y.max(r.y), z: self.z.max(r.z) } }
    #[inline] pub fn clamp(self, lo: Self, hi: Self) -> Self {
        Self { x: self.x.clamp(lo.x, hi.x), y: self.y.clamp(lo.y, hi.y), z: self.z.clamp(lo.z, hi.z) }
    }

    #[inline]
    pub fn lerp(self, rhs: Self, t: Fixed<FRAC>) -> Self {
        Self { x: self.x.lerp(rhs.x, t), y: self.y.lerp(rhs.y, t), z: self.z.lerp(rhs.z, t) }
    }
}

impl<const FRAC: u32> Add for FixedVec3<FRAC> {
    type Output = Self;
    #[inline] fn add(self, r: Self) -> Self { Self { x: self.x+r.x, y: self.y+r.y, z: self.z+r.z } }
}
impl<const FRAC: u32> AddAssign for FixedVec3<FRAC> {
    #[inline] fn add_assign(&mut self, r: Self) { *self = *self + r; }
}
impl<const FRAC: u32> Sub for FixedVec3<FRAC> {
    type Output = Self;
    #[inline] fn sub(self, r: Self) -> Self { Self { x: self.x-r.x, y: self.y-r.y, z: self.z-r.z } }
}
impl<const FRAC: u32> SubAssign for FixedVec3<FRAC> {
    #[inline] fn sub_assign(&mut self, r: Self) { *self = *self - r; }
}
impl<const FRAC: u32> Neg for FixedVec3<FRAC> {
    type Output = Self;
    #[inline] fn neg(self) -> Self { Self { x: -self.x, y: -self.y, z: -self.z } }
}

impl<const FRAC: u32> fmt::Debug for FixedVec3<FRAC> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FixedVec3<{}>({:.4}, {:.4}, {:.4})", FRAC,
            self.x.to_f64(), self.y.to_f64(), self.z.to_f64())
    }
}
impl<const FRAC: u32> fmt::Display for FixedVec3<FRAC> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:.4}, {:.4}, {:.4})", self.x.to_f64(), self.y.to_f64(), self.z.to_f64())
    }
  }
