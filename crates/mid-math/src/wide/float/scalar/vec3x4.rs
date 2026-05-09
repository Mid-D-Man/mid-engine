// crates/mid-math/src/wide/float/scalar/vec3x4.rs
//! Scalar fallback Vec3x4 — non-x86 platforms.
//! Keeps SoA layout (3 × [f32;4]) for structural equivalence with SSE2 version.

use core::fmt;
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use super::mask4::Mask4;
use super::f32x4::f32x4;
use crate::Vec3;
use crate::EPSILON;

/// 4 × Vec3 in SoA layout — scalar fallback. 48 bytes, 16-byte aligned.
#[derive(Clone, Copy, PartialEq)]
#[repr(C, align(16))]
pub struct Vec3x4 {
    pub x: [f32; 4],
    pub y: [f32; 4],
    pub z: [f32; 4],
}

impl Vec3x4 {
    pub const ZERO: Self = Self { x: [0.0; 4], y: [0.0; 4], z: [0.0; 4] };
    pub const X:    Self = Self { x: [1.0; 4], y: [0.0; 4], z: [0.0; 4] };
    pub const Y:    Self = Self { x: [0.0; 4], y: [1.0; 4], z: [0.0; 4] };
    pub const Z:    Self = Self { x: [0.0; 4], y: [0.0; 4], z: [1.0; 4] };

    #[inline]
    pub fn from_vec3s(a: Vec3, b: Vec3, c: Vec3, d: Vec3) -> Self {
        Self {
            x: [a.x, b.x, c.x, d.x],
            y: [a.y, b.y, c.y, d.y],
            z: [a.z, b.z, c.z, d.z],
        }
    }
    #[inline(always)]
    pub fn from_slice(s: &[Vec3; 4]) -> Self { Self::from_vec3s(s[0], s[1], s[2], s[3]) }

    #[inline(always)]
    pub fn splat(v: Vec3) -> Self {
        Self { x: [v.x; 4], y: [v.y; 4], z: [v.z; 4] }
    }

    #[inline]
    pub fn to_array(self) -> [Vec3; 4] {
        core::array::from_fn(|i| Vec3::new(self.x[i], self.y[i], self.z[i]))
    }

    #[inline(always)]
    pub fn write_to_slice(self, s: &mut [Vec3; 4]) {
        let a = self.to_array(); s.copy_from_slice(&a);
    }

    #[inline]
    pub fn get(self, lane: usize) -> Vec3 {
        assert!(lane < 4); Vec3::new(self.x[lane], self.y[lane], self.z[lane])
    }

    #[inline]
    pub fn mul_elem(self, r: Self) -> Self {
        Self {
            x: core::array::from_fn(|i| self.x[i] * r.x[i]),
            y: core::array::from_fn(|i| self.y[i] * r.y[i]),
            z: core::array::from_fn(|i| self.z[i] * r.z[i]),
        }
    }

    #[inline]
    pub fn scale(self, s: f32x4) -> Self {
        Self {
            x: core::array::from_fn(|i| self.x[i] * s.0[i]),
            y: core::array::from_fn(|i| self.y[i] * s.0[i]),
            z: core::array::from_fn(|i| self.z[i] * s.0[i]),
        }
    }

    #[inline(always)]
    pub fn scale_uniform(self, s: f32) -> Self { self.scale(f32x4::splat(s)) }

    #[inline]
    pub fn madd(self, b: Self, c: Self) -> Self {
        Self {
            x: core::array::from_fn(|i| self.x[i] * b.x[i] + c.x[i]),
            y: core::array::from_fn(|i| self.y[i] * b.y[i] + c.y[i]),
            z: core::array::from_fn(|i| self.z[i] * b.z[i] + c.z[i]),
        }
    }

    #[inline]
    pub fn dot(self, r: Self) -> f32x4 {
        f32x4(core::array::from_fn(|i| {
            self.x[i] * r.x[i] + self.y[i] * r.y[i] + self.z[i] * r.z[i]
        }))
    }

    #[inline]
    pub fn cross(self, r: Self) -> Self {
        Self {
            x: core::array::from_fn(|i| self.y[i]*r.z[i] - self.z[i]*r.y[i]),
            y: core::array::from_fn(|i| self.z[i]*r.x[i] - self.x[i]*r.z[i]),
            z: core::array::from_fn(|i| self.x[i]*r.y[i] - self.y[i]*r.x[i]),
        }
    }

    #[inline(always)] pub fn length_sq(self) -> f32x4 { self.dot(self) }

    #[inline]
    pub fn length(self) -> f32x4 { f32x4(self.length_sq().0.map(|x| x.sqrt())) }

    #[inline]
    pub fn normalize(self) -> Self {
        let lsq = self.length_sq().0;
        let eps2 = EPSILON * EPSILON;
        Self {
            x: core::array::from_fn(|i| if lsq[i] > eps2 { self.x[i] / lsq[i].sqrt() } else { 0.0 }),
            y: core::array::from_fn(|i| if lsq[i] > eps2 { self.y[i] / lsq[i].sqrt() } else { 0.0 }),
            z: core::array::from_fn(|i| if lsq[i] > eps2 { self.z[i] / lsq[i].sqrt() } else { 0.0 }),
        }
    }

    #[inline(always)] pub fn normalize_precise(self) -> Self { self.normalize() }

    #[inline]
    pub fn lerp(self, rhs: Self, t: f32x4) -> Self {
        Self {
            x: core::array::from_fn(|i| self.x[i] + (rhs.x[i] - self.x[i]) * t.0[i]),
            y: core::array::from_fn(|i| self.y[i] + (rhs.y[i] - self.y[i]) * t.0[i]),
            z: core::array::from_fn(|i| self.z[i] + (rhs.z[i] - self.z[i]) * t.0[i]),
        }
    }

    #[inline] pub fn min(self, r: Self) -> Self {
        Self {
            x: core::array::from_fn(|i| self.x[i].min(r.x[i])),
            y: core::array::from_fn(|i| self.y[i].min(r.y[i])),
            z: core::array::from_fn(|i| self.z[i].min(r.z[i])),
        }
    }
    #[inline] pub fn max(self, r: Self) -> Self {
        Self {
            x: core::array::from_fn(|i| self.x[i].max(r.x[i])),
            y: core::array::from_fn(|i| self.y[i].max(r.y[i])),
            z: core::array::from_fn(|i| self.z[i].max(r.z[i])),
        }
    }

    #[inline] pub fn select(mask: Mask4, t: Self, f: Self) -> Self {
        Self {
            x: core::array::from_fn(|i| if mask.0[i] != 0 { t.x[i] } else { f.x[i] }),
            y: core::array::from_fn(|i| if mask.0[i] != 0 { t.y[i] } else { f.y[i] }),
            z: core::array::from_fn(|i| if mask.0[i] != 0 { t.z[i] } else { f.z[i] }),
        }
    }

    #[inline] pub fn length_lt(self, r: Self) -> Mask4 { self.length_sq().cmplt(r.length_sq()) }
    #[inline] pub fn is_finite(self) -> bool {
        self.x.iter().chain(self.y.iter()).chain(self.z.iter()).all(|x| x.is_finite())
    }
}

impl Add for Vec3x4 { type Output=Self; #[inline] fn add(self, r: Self) -> Self { Self { x: core::array::from_fn(|i| self.x[i]+r.x[i]), y: core::array::from_fn(|i| self.y[i]+r.y[i]), z: core::array::from_fn(|i| self.z[i]+r.z[i]) } } }
impl AddAssign for Vec3x4 { fn add_assign(&mut self, r: Self) { *self = *self + r; } }
impl Sub for Vec3x4 { type Output=Self; #[inline] fn sub(self, r: Self) -> Self { Self { x: core::array::from_fn(|i| self.x[i]-r.x[i]), y: core::array::from_fn(|i| self.y[i]-r.y[i]), z: core::array::from_fn(|i| self.z[i]-r.z[i]) } } }
impl SubAssign for Vec3x4 { fn sub_assign(&mut self, r: Self) { *self = *self - r; } }
impl Neg for Vec3x4 { type Output=Self; #[inline] fn neg(self) -> Self { Self { x: self.x.map(|x| -x), y: self.y.map(|y| -y), z: self.z.map(|z| -z) } } }
impl Mul for Vec3x4 { type Output=Self; #[inline] fn mul(self, r: Self) -> Self { self.mul_elem(r) } }
impl MulAssign for Vec3x4 { fn mul_assign(&mut self, r: Self) { *self = *self * r; } }
impl Mul<f32x4> for Vec3x4 { type Output=Self; #[inline] fn mul(self, s: f32x4) -> Self { self.scale(s) } }
impl Mul<f32> for Vec3x4 { type Output=Self; #[inline] fn mul(self, s: f32) -> Self { self.scale_uniform(s) } }

impl Default for Vec3x4 { fn default() -> Self { Self::ZERO } }

impl fmt::Debug for Vec3x4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let a = self.to_array();
        write!(f, "Vec3x4([{:?},{:?},{:?},{:?}])", a[0],a[1],a[2],a[3])
    }
}
impl fmt::Display for Vec3x4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let a = self.to_array();
        write!(f, "[{},{},{},{}]", a[0],a[1],a[2],a[3])
    }
}
impl From<[Vec3;4]> for Vec3x4 { fn from(a: [Vec3;4]) -> Self { Self::from_slice(&a) } }
impl From<Vec3x4> for [Vec3;4] { fn from(v: Vec3x4) -> Self { v.to_array() } }
