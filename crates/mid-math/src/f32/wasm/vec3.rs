// crates/mid-math/src/f32/wasm/vec3.rs
//! Vec3 backed by `v128` on wasm32/wasm64 with simd128 target feature.
//!
//! Lane layout : [x, y, z, 0]  — lane 3 is always 0 (padding).
//!
//! Key WASM advantages over SSE2:
//!   f32x4_abs  — direct instruction, no sign-mask ANDNOT trick needed
//!   f32x4_neg  — direct instruction, no XOR with -0.0 needed
//!
//! No FMA in WASM basic SIMD128 (relaxed-simd adds it but we don't require it).
//! LLVM may fuse mul+add chains at the IR level on supporting hosts anyway.

#[cfg(target_arch = "wasm32")]
use core::arch::wasm32::*;
#[cfg(target_arch = "wasm64")]
use core::arch::wasm64::*;

use core::fmt;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::f32::wasm::vec4::Vec4;
use crate::f32::vec2::Vec2;
use crate::impl_vec3_deref;
use crate::wasm::{dot3, dot3_in_x, dot3_into_v128};
use crate::EPSILON;

// ── Union for const initialization ───────────────────────────────────────────
// `v128` cannot be constructed by ordinary const expressions; we use a union
// transmute identical to the NEON implementation.

#[repr(C)]
union UnionCast {
    f: [f32; 4],
    v: Vec3,
}

// ── Type ──────────────────────────────────────────────────────────────────────

/// 3-dimensional vector. 16 bytes, 16-byte aligned. Backed by `v128`.
///
/// **C interop:** use [`CVec3`][crate::ffi::types::CVec3] at the FFI boundary.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct Vec3(pub(crate) v128);

// Provides .x .y .z access via Deref to XYZ<f32>.
impl_vec3_deref!(Vec3);

// ── Constants ─────────────────────────────────────────────────────────────────

impl Vec3 {
    pub const ZERO:  Self = unsafe { UnionCast { f: [ 0.0,  0.0,  0.0, 0.0] }.v };
    pub const ONE:   Self = unsafe { UnionCast { f: [ 1.0,  1.0,  1.0, 0.0] }.v };
    pub const X:     Self = unsafe { UnionCast { f: [ 1.0,  0.0,  0.0, 0.0] }.v };
    pub const Y:     Self = unsafe { UnionCast { f: [ 0.0,  1.0,  0.0, 0.0] }.v };
    pub const Z:     Self = unsafe { UnionCast { f: [ 0.0,  0.0,  1.0, 0.0] }.v };
    pub const NEG_X: Self = unsafe { UnionCast { f: [-1.0,  0.0,  0.0, 0.0] }.v };
    pub const NEG_Y: Self = unsafe { UnionCast { f: [ 0.0, -1.0,  0.0, 0.0] }.v };
    pub const NEG_Z: Self = unsafe { UnionCast { f: [ 0.0,  0.0, -1.0, 0.0] }.v };

    // ── Constructors ─────────────────────────────────────────────────────────

    /// Create from three components. Lane 3 is zeroed.
    #[inline(always)]
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        unsafe { UnionCast { f: [x, y, z, 0.0] }.v }
    }

    /// Broadcast `v` to all three components (lane 3 = 0).
    #[inline(always)]
    pub fn splat(v: f32) -> Self {
        unsafe { UnionCast { f: [v, v, v, 0.0] }.v }
    }

    #[inline(always)] pub fn from_array(a: [f32; 3]) -> Self { Self::new(a[0], a[1], a[2]) }
    #[inline(always)] pub fn to_array(self)           -> [f32; 3] { [self.x, self.y, self.z] }

    /// Extend to Vec4, setting lane 3 = `w`.
    #[inline(always)]
    pub fn extend(self, w: f32) -> Vec4 {
        Vec4::new(self.x, self.y, self.z, w)
    }

    /// Truncate to Vec2 (drop z).
    #[inline(always)]
    pub fn truncate(self) -> Vec2 { Vec2::new(self.x, self.y) }

    // ── Core arithmetic ───────────────────────────────────────────────────────

    /// Dot product — scalar, faster than WASM horizontal reduce for 3 lanes.
    #[inline(always)]
    pub fn dot(self, rhs: Self) -> f32 {
        unsafe { dot3(self.0, rhs.0) }
    }

    /// Broadcast dot product to a Vec3 (all lanes = dot value).
    #[inline]
    pub fn dot_into_vec(self, rhs: Self) -> Self {
        Self(unsafe { dot3_into_v128(self.0, rhs.0) })
    }

    /// Cross product.
    ///
    /// Uses the cyclic-permutation shuffle approach from glam's WASM implementation.
    ///
    ///   lhszxy = [self.z, self.x, self.y, self.y]   shuffle(2,0,1,1)
    ///   rhszxy = [rhs.z,  rhs.x,  rhs.y,  rhs.y]   shuffle(2,0,1,1)
    ///   sub    = lhszxy*rhs - rhszxy*self           = [res.y, res.z, res.x, ?]
    ///   result = shuffle(2,0,1,1)(sub)               = [res.x, res.y, res.z, res.z]
    #[inline(always)]
    pub fn cross(self, rhs: Self) -> Self {
        unsafe {
            let lhszxy     = i32x4_shuffle::<2, 0, 1, 1>(self.0, self.0);
            let rhszxy     = i32x4_shuffle::<2, 0, 1, 1>(rhs.0,  rhs.0);
            let lhszxy_rhs = f32x4_mul(lhszxy, rhs.0);
            let rhszxy_lhs = f32x4_mul(rhszxy, self.0);
            let sub        = f32x4_sub(lhszxy_rhs, rhszxy_lhs);
            Self(i32x4_shuffle::<2, 0, 1, 1>(sub, sub))
        }
    }

    // ── Length / normalise ────────────────────────────────────────────────────

    #[inline(always)] pub fn length_sq(self) -> f32 { self.dot(self) }

    #[inline]
    pub fn length(self) -> f32 {
        unsafe {
            let dot = dot3_in_x(self.0, self.0);
            f32x4_extract_lane::<0>(f32x4_sqrt(dot))
        }
    }

    #[inline]
    pub fn length_recip(self) -> f32 {
        let l = self.length();
        if l < EPSILON { 0.0 } else { 1.0 / l }
    }

    /// Normalise to unit length.  Returns `ZERO` for near-zero-length vectors.
    #[inline]
    pub fn normalize(self) -> Self {
        unsafe {
            let len_v      = f32x4_sqrt(dot3_into_v128(self.0, self.0));
            let normalized = Self(f32x4_div(self.0, len_v));
            // Zero result lanes where length <= EPSILON
            let ok         = f32x4_gt(len_v, f32x4_splat(EPSILON));
            Self(v128_and(normalized.0, ok))
        }
    }

    #[inline]
    pub fn try_normalize(self) -> Option<Self> {
        let rcp = self.length_recip();
        if rcp > 0.0 && rcp.is_finite() { Some(self * rcp) } else { None }
    }

    #[inline] pub fn normalize_or(self, fallback: Self) -> Self {
        self.try_normalize().unwrap_or(fallback)
    }
    #[inline] pub fn normalize_or_zero(self) -> Self { self.normalize_or(Self::ZERO) }
    #[inline] pub fn is_normalized(self) -> bool { (self.length_sq() - 1.0).abs() <= 2e-4 }

    // ── Interpolation / geometry ──────────────────────────────────────────────

    /// Linear interpolation.  No native FMA in WASM SIMD128 baseline:
    /// `self + (rhs - self) * t`  →  2 muls + 1 add (LLVM may fuse on relaxed-simd hosts).
    #[inline]
    pub fn lerp(self, rhs: Self, t: f32) -> Self {
        unsafe {
            let tt   = f32x4_splat(t);
            let diff = f32x4_sub(rhs.0, self.0);
            Self(f32x4_add(self.0, f32x4_mul(diff, tt)))
        }
    }

    #[inline] pub fn reflect(self, n: Self) -> Self { self - n * (2.0 * self.dot(n)) }
    #[inline] pub fn distance(self, rhs: Self)    -> f32 { (self - rhs).length() }
    #[inline] pub fn distance_sq(self, rhs: Self) -> f32 { (self - rhs).length_sq() }

    // ── Component-wise ────────────────────────────────────────────────────────

    /// `f32x4_pmin`: IEEE-754-2008-compatible pmin (NaN propagation may differ on some hosts).
    #[inline] pub fn min(self, rhs: Self) -> Self { Self(unsafe { f32x4_pmin(self.0, rhs.0) }) }
    #[inline] pub fn max(self, rhs: Self) -> Self { Self(unsafe { f32x4_pmax(self.0, rhs.0) }) }
    #[inline] pub fn clamp(self, lo: Self, hi: Self) -> Self { self.max(lo).min(hi) }
    /// Direct `f32x4_abs` — no sign-mask trick needed unlike SSE2.
    #[inline] pub fn abs(self) -> Self { Self(unsafe { f32x4_abs(self.0) }) }

    // ── Predicates ────────────────────────────────────────────────────────────

    #[inline] pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }
    #[inline] pub fn is_nan(self) -> bool {
        self.x.is_nan() || self.y.is_nan() || self.z.is_nan()
    }

    #[inline]
    pub fn approx_eq(self, rhs: Self) -> bool {
        (self - rhs).abs().length_sq() < EPSILON * EPSILON
    }

    #[inline]
    pub fn approx_eq_eps(self, rhs: Self, eps: f32) -> bool {
        (self.x - rhs.x).abs() < eps
            && (self.y - rhs.y).abs() < eps
            && (self.z - rhs.z).abs() < eps
    }
}

// ── Operators ─────────────────────────────────────────────────────────────────

impl Add for Vec3 {
    type Output = Self;
    #[inline(always)]
    fn add(self, r: Self) -> Self { Self(unsafe { f32x4_add(self.0, r.0) }) }
}
impl Sub for Vec3 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, r: Self) -> Self { Self(unsafe { f32x4_sub(self.0, r.0) }) }
}
impl Mul<f32> for Vec3 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, s: f32) -> Self { Self(unsafe { f32x4_mul(self.0, f32x4_splat(s)) }) }
}
impl Mul<Vec3> for f32 {
    type Output = Vec3;
    #[inline(always)]
    fn mul(self, v: Vec3) -> Vec3 { Vec3(unsafe { f32x4_mul(f32x4_splat(self), v.0) }) }
}
impl Mul for Vec3 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, r: Self) -> Self { Self(unsafe { f32x4_mul(self.0, r.0) }) }
}
impl Div<f32> for Vec3 {
    type Output = Self;
    #[inline(always)]
    fn div(self, s: f32) -> Self { Self(unsafe { f32x4_div(self.0, f32x4_splat(s)) }) }
}
/// Direct `f32x4_neg` — no XOR trick needed unlike SSE2.
impl Neg for Vec3 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self { Self(unsafe { f32x4_neg(self.0) }) }
}

impl AddAssign for Vec3 {
    #[inline(always)]
    fn add_assign(&mut self, r: Self) { self.0 = unsafe { f32x4_add(self.0, r.0) }; }
}
impl SubAssign for Vec3 {
    #[inline(always)]
    fn sub_assign(&mut self, r: Self) { self.0 = unsafe { f32x4_sub(self.0, r.0) }; }
}
impl MulAssign<f32> for Vec3 {
    #[inline(always)]
    fn mul_assign(&mut self, s: f32) { self.0 = unsafe { f32x4_mul(self.0, f32x4_splat(s)) }; }
}
impl DivAssign<f32> for Vec3 {
    #[inline(always)]
    fn div_assign(&mut self, s: f32) { self.0 = unsafe { f32x4_div(self.0, f32x4_splat(s)) }; }
}

// ── PartialEq — compare lanes 0,1,2 only (lane 3 is padding) ─────────────────

impl PartialEq for Vec3 {
    #[inline]
    fn eq(&self, rhs: &Self) -> bool {
        unsafe {
            // u32x4_bitmask returns u16; low 4 bits = sign bits of each lane after comparison.
            // f32x4_eq gives all-1s per lane if equal, 0 otherwise.
            (u32x4_bitmask(f32x4_eq(self.0, rhs.0)) & 0b0111) == 0b0111
        }
    }
}

impl Default for Vec3 { fn default() -> Self { Self::ZERO } }

// ── Display / Debug ───────────────────────────────────────────────────────────

impl fmt::Debug for Vec3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Vec3")
            .field(&self.x).field(&self.y).field(&self.z)
            .finish()
    }
}
impl fmt::Display for Vec3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {}, {})", self.x, self.y, self.z)
    }
}

// ── Conversions ───────────────────────────────────────────────────────────────

impl From<[f32; 3]> for Vec3 {
    #[inline] fn from(a: [f32; 3]) -> Self { Self::new(a[0], a[1], a[2]) }
}
impl From<Vec3> for [f32; 3] {
    #[inline] fn from(v: Vec3) -> Self { [v.x, v.y, v.z] }
}
impl From<(f32, f32, f32)> for Vec3 {
    #[inline] fn from(t: (f32, f32, f32)) -> Self { Self::new(t.0, t.1, t.2) }
}
impl From<Vec3> for (f32, f32, f32) {
    #[inline] fn from(v: Vec3) -> Self { (v.x, v.y, v.z) }
}
