// crates/mid-math/src/wide/float/wasm/vec3x4.rs
//! 4 × Vec3 packed SoA — WASM SIMD128.
//!
//! AoS→SoA transpose: WASM has no unpacklo/unpackhi/movelh/movehl.
//! We emulate them with `i32x4_shuffle` (compile-time lane indices).
//!
//!   unpacklo(a,b) = i32x4_shuffle::<0,4,1,5>(a,b)  → [a0,b0,a1,b1]
//!   unpackhi(a,b) = i32x4_shuffle::<2,6,3,7>(a,b)  → [a2,b2,a3,b3]
//!   movelh(a,b)   = i32x4_shuffle::<0,1,4,5>(a,b)  → [a0,a1,b0,b1]
//!   movehl(a,b)   = i32x4_shuffle::<6,7,2,3>(a,b)  → [b2,b3,a2,a3]
//!                   (note SSE movehl arg order is reversed from WASM shuffle)
//!
//! Same 7-shuffle AoS↔SoA algorithm as SSE2.

use core::fmt;
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use core::arch::wasm32::*;

use super::mask4::Mask4;
use super::f32x4::{f32x4, rsqrt_nr};
use crate::f32::wasm::vec3::Vec3;
use crate::EPSILON;

/// 4 × Vec3 in SoA layout. 48 bytes, 16-byte aligned.
#[derive(Clone, Copy)]
#[repr(C, align(16))]
pub struct Vec3x4 {
    pub x: v128,
    pub y: v128,
    pub z: v128,
}

// ── Shuffle shims (SSE2 semantic names → WASM i32x4_shuffle) ─────────────────

/// [a0, b0, a1, b1]
#[inline(always)]
fn unpacklo(a: v128, b: v128) -> v128 {
    i32x4_shuffle::<0, 4, 1, 5>(a, b)
}
/// [a2, b2, a3, b3]
#[inline(always)]
fn unpackhi(a: v128, b: v128) -> v128 {
    i32x4_shuffle::<2, 6, 3, 7>(a, b)
}
/// [a0, a1, b0, b1]
#[inline(always)]
fn movelh(a: v128, b: v128) -> v128 {
    i32x4_shuffle::<0, 1, 4, 5>(a, b)
}
/// SSE movehl(dst, src) = [src2, src3, dst2, dst3]
/// We use: movehl(lo, hi) to match the SSE2 pattern
#[inline(always)]
fn movehl(dst: v128, src: v128) -> v128 {
    // SSE movehl_ps(a,b) = [b[2], b[3], a[2], a[3]]
    // In WASM shuffle terms with src as second arg:
    i32x4_shuffle::<6, 7, 2, 3>(dst, src)
}

impl Vec3x4 {
    // ── Constants ─────────────────────────────────────────────────────────────

    pub const ZERO: Self = Self {
        x: f32x4(0.0, 0.0, 0.0, 0.0),
        y: f32x4(0.0, 0.0, 0.0, 0.0),
        z: f32x4(0.0, 0.0, 0.0, 0.0),
    };

    pub const X: Self = Self {
        x: f32x4(1.0, 1.0, 1.0, 1.0),
        y: f32x4(0.0, 0.0, 0.0, 0.0),
        z: f32x4(0.0, 0.0, 0.0, 0.0),
    };

    pub const Y: Self = Self {
        x: f32x4(0.0, 0.0, 0.0, 0.0),
        y: f32x4(1.0, 1.0, 1.0, 1.0),
        z: f32x4(0.0, 0.0, 0.0, 0.0),
    };

    pub const Z: Self = Self {
        x: f32x4(0.0, 0.0, 0.0, 0.0),
        y: f32x4(0.0, 0.0, 0.0, 0.0),
        z: f32x4(1.0, 1.0, 1.0, 1.0),
    };

    // ── Constructors ──────────────────────────────────────────────────────────

    /// AoS→SoA transpose from 4 Vec3s. Same 7-shuffle algorithm as SSE2.
    #[inline]
    pub fn from_vec3s(a: Vec3, b: Vec3, c: Vec3, d: Vec3) -> Self {
        // Load each Vec3 as a v128 ([x, y, z, 0.0])
        let av = a.to_v128();
        let bv = b.to_v128();
        let cv = c.to_v128();
        let dv = d.to_v128();

        // Mirrors SSE2 unpacklo/hi + movelh/movehl pattern
        let lo01 = unpacklo(av, bv);  // [x0, x1, y0, y1]
        let lo23 = unpacklo(cv, dv);  // [x2, x3, y2, y3]
        let hi01 = unpackhi(av, bv);  // [z0, z1, 0,  0 ]
        let hi23 = unpackhi(cv, dv);  // [z2, z3, 0,  0 ]

        let x = movelh(lo01, lo23);   // [x0, x1, x2, x3]
        let y = movehl(lo23, lo01);   // [y0, y1, y2, y3]
        let z = movelh(hi01, hi23);   // [z0, z1, z2, z3]

        Self { x, y, z }
    }

    #[inline(always)]
    pub fn from_slice(s: &[Vec3; 4]) -> Self {
        Self::from_vec3s(s[0], s[1], s[2], s[3])
    }

    #[inline(always)]
    pub fn splat(v: Vec3) -> Self {
        Self {
            x: f32x4_splat(v.x),
            y: f32x4_splat(v.y),
            z: f32x4_splat(v.z),
        }
    }

    /// SoA→AoS transpose.
    #[inline]
    pub fn to_array(self) -> [Vec3; 4] {
        // Inverse of from_vec3s
        let xy_lo = unpacklo(self.x, self.y);    // [x0, y0, x1, y1]
        let xy_hi = unpackhi(self.x, self.y);    // [x2, y2, x3, y3]
        let zero  = f32x4(0.0, 0.0, 0.0, 0.0);
        let z_lo  = unpacklo(self.z, zero);       // [z0, 0, z1, 0]
        let z_hi  = unpackhi(self.z, zero);       // [z2, 0, z3, 0]

        let v0 = Vec3::from_v128(movelh(xy_lo, z_lo));   // [x0, y0, z0, 0]
        let v1 = Vec3::from_v128(movehl(z_lo, xy_lo));   // [x1, y1, z1, 0]
        let v2 = Vec3::from_v128(movelh(xy_hi, z_hi));   // [x2, y2, z2, 0]
        let v3 = Vec3::from_v128(movehl(z_hi, xy_hi));   // [x3, y3, z3, 0]

        [v0, v1, v2, v3]
    }

    #[inline(always)]
    pub fn write_to_slice(self, s: &mut [Vec3; 4]) {
        s.copy_from_slice(&self.to_array());
    }

    #[inline]
    pub fn get(self, lane: usize) -> Vec3 {
        assert!(lane < 4, "Vec3x4::get — lane {lane} out of bounds (max 3)");
        self.to_array()[lane]
    }

    // ── Arithmetic ────────────────────────────────────────────────────────────

    #[inline(always)]
    pub fn mul_elem(self, rhs: Self) -> Self {
        Self {
            x: f32x4_mul(self.x, rhs.x),
            y: f32x4_mul(self.y, rhs.y),
            z: f32x4_mul(self.z, rhs.z),
        }
    }

    #[inline(always)]
    pub fn scale(self, s: f32x4) -> Self {
        Self {
            x: f32x4_mul(self.x, s.0),
            y: f32x4_mul(self.y, s.0),
            z: f32x4_mul(self.z, s.0),
        }
    }

    #[inline(always)]
    pub fn scale_uniform(self, s: f32) -> Self { self.scale(f32x4::splat(s)) }

    #[inline(always)]
    pub fn madd(self, b: Self, c: Self) -> Self {
        Self {
            x: f32x4_add(f32x4_mul(self.x, b.x), c.x),
            y: f32x4_add(f32x4_mul(self.y, b.y), c.y),
            z: f32x4_add(f32x4_mul(self.z, b.z), c.z),
        }
    }

    // ── Geometric ops ─────────────────────────────────────────────────────────

    #[inline(always)]
    pub fn dot(self, rhs: Self) -> f32x4 {
        let xx = f32x4_mul(self.x, rhs.x);
        let yy = f32x4_mul(self.y, rhs.y);
        let zz = f32x4_mul(self.z, rhs.z);
        f32x4(f32x4_add(f32x4_add(xx, yy), zz))
    }

    #[inline(always)]
    pub fn cross(self, rhs: Self) -> Self {
        Self {
            x: f32x4_sub(f32x4_mul(self.y, rhs.z), f32x4_mul(self.z, rhs.y)),
            y: f32x4_sub(f32x4_mul(self.z, rhs.x), f32x4_mul(self.x, rhs.z)),
            z: f32x4_sub(f32x4_mul(self.x, rhs.y), f32x4_mul(self.y, rhs.x)),
        }
    }

    #[inline(always)]
    pub fn length_sq(self) -> f32x4 { self.dot(self) }

    #[inline(always)]
    pub fn length(self) -> f32x4 { f32x4(f32x4_sqrt(self.length_sq().0)) }

    #[inline]
    pub fn normalize(self) -> Self {
        let len_sq  = self.length_sq().0;
        let inv_len = rsqrt_nr(len_sq);
        let ok      = f32x4_gt(len_sq, f32x4_splat(EPSILON * EPSILON));
        let inv     = v128_and(inv_len, ok);
        Self {
            x: f32x4_mul(self.x, inv),
            y: f32x4_mul(self.y, inv),
            z: f32x4_mul(self.z, inv),
        }
    }

    #[inline]
    pub fn normalize_precise(self) -> Self {
        let len_sq = self.length_sq().0;
        let len    = f32x4_sqrt(len_sq);
        let ok     = f32x4_gt(len, f32x4_splat(EPSILON));
        let safe   = v128_or(
            v128_and(ok, len),
            v128_andnot(f32x4_splat(1.0), ok),
        );
        let inv    = f32x4_div(f32x4_splat(1.0), safe);
        let inv    = v128_and(inv, ok);
        Self {
            x: f32x4_mul(self.x, inv),
            y: f32x4_mul(self.y, inv),
            z: f32x4_mul(self.z, inv),
        }
    }

    // ── Interpolation ─────────────────────────────────────────────────────────

    #[inline(always)]
    pub fn lerp(self, rhs: Self, t: f32x4) -> Self {
        Self {
            x: f32x4_add(self.x, f32x4_mul(f32x4_sub(rhs.x, self.x), t.0)),
            y: f32x4_add(self.y, f32x4_mul(f32x4_sub(rhs.y, self.y), t.0)),
            z: f32x4_add(self.z, f32x4_mul(f32x4_sub(rhs.z, self.z), t.0)),
        }
    }

    #[inline(always)]
    pub fn min(self, rhs: Self) -> Self {
        Self {
            x: f32x4_min(self.x, rhs.x),
            y: f32x4_min(self.y, rhs.y),
            z: f32x4_min(self.z, rhs.z),
        }
    }

    #[inline(always)]
    pub fn max(self, rhs: Self) -> Self {
        Self {
            x: f32x4_max(self.x, rhs.x),
            y: f32x4_max(self.y, rhs.y),
            z: f32x4_max(self.z, rhs.z),
        }
    }

    // ── Branchless select ─────────────────────────────────────────────────────

    #[inline(always)]
    pub fn select(mask: Mask4, t: Self, f: Self) -> Self {
        Self {
            x: v128_or(v128_and(mask.0, t.x), v128_andnot(f.x, mask.0)),
            y: v128_or(v128_and(mask.0, t.y), v128_andnot(f.y, mask.0)),
            z: v128_or(v128_and(mask.0, t.z), v128_andnot(f.z, mask.0)),
        }
    }

    // ── Comparisons ───────────────────────────────────────────────────────────

    #[inline(always)]
    pub fn length_lt(self, rhs: Self) -> Mask4 {
        self.length_sq().cmplt(rhs.length_sq())
    }

    // ── Predicates ────────────────────────────────────────────────────────────

    #[inline]
    pub fn is_finite(self) -> bool {
        let inf = f32x4_splat(f32::INFINITY);
        let ax  = f32x4_abs(self.x);
        let ay  = f32x4_abs(self.y);
        let az  = f32x4_abs(self.z);
        i32x4_all_true(f32x4_lt(ax, inf))
            && i32x4_all_true(f32x4_lt(ay, inf))
            && i32x4_all_true(f32x4_lt(az, inf))
    }
}

// ── Operators ─────────────────────────────────────────────────────────────────

impl Add for Vec3x4 {
    type Output = Self;
    #[inline(always)]
    fn add(self, r: Self) -> Self {
        Self { x: f32x4_add(self.x, r.x), y: f32x4_add(self.y, r.y), z: f32x4_add(self.z, r.z) }
    }
}
impl AddAssign for Vec3x4 { #[inline(always)] fn add_assign(&mut self, r: Self) { *self = *self + r; } }

impl Sub for Vec3x4 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, r: Self) -> Self {
        Self { x: f32x4_sub(self.x, r.x), y: f32x4_sub(self.y, r.y), z: f32x4_sub(self.z, r.z) }
    }
}
impl SubAssign for Vec3x4 { #[inline(always)] fn sub_assign(&mut self, r: Self) { *self = *self - r; } }

impl Neg for Vec3x4 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self { x: f32x4_neg(self.x), y: f32x4_neg(self.y), z: f32x4_neg(self.z) }
    }
}

impl Mul for Vec3x4 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, r: Self) -> Self { self.mul_elem(r) }
}
impl MulAssign for Vec3x4 { #[inline(always)] fn mul_assign(&mut self, r: Self) { *self = *self * r; } }

impl Mul<f32x4> for Vec3x4 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, s: f32x4) -> Self { self.scale(s) }
}

impl Mul<f32> for Vec3x4 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, s: f32) -> Self { self.scale_uniform(s) }
}

impl PartialEq for Vec3x4 {
    fn eq(&self, r: &Self) -> bool {
        i32x4_all_true(f32x4_eq(self.x, r.x))
            && i32x4_all_true(f32x4_eq(self.y, r.y))
            && i32x4_all_true(f32x4_eq(self.z, r.z))
    }
}

impl Default for Vec3x4 { fn default() -> Self { Self::ZERO } }

impl fmt::Debug for Vec3x4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let a = self.to_array();
        write!(f, "Vec3x4([{:?}, {:?}, {:?}, {:?}])", a[0], a[1], a[2], a[3])
    }
}
impl fmt::Display for Vec3x4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let a = self.to_array();
        write!(f, "[{}, {}, {}, {}]", a[0], a[1], a[2], a[3])
    }
}

impl From<[Vec3; 4]> for Vec3x4 { #[inline] fn from(a: [Vec3; 4]) -> Self { Self::from_slice(&a) } }
impl From<Vec3x4> for [Vec3; 4]  { #[inline] fn from(v: Vec3x4)   -> Self { v.to_array() } }
