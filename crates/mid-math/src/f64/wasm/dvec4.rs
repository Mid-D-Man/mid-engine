// crates/mid-math/src/f64/wasm/dvec4.rs
//! DVec4 backed by 2× `v128` (f64x2) on wasm32/wasm64 with simd128.
//!
//! Storage layout (32 bytes, 32-byte aligned):
//!   lo: v128 → [x (bytes 0-7),  y (bytes 8-15)]
//!   hi: v128 → [z (bytes 16-23), w (bytes 24-31)]
//!
//! Byte-identical to XYZW<f64> {x@0, y@8, z@16, w@24} — impl_dvec4_deref! is zero-cost.
//!
//! Same 2×v128 pattern as SSE2 DVec4. All ops process both registers in parallel.
//!
//! Note: v128_andnot(a, b) = a & ~b  (WASM — reversed from SSE2 _mm_andnot_pd)
//!
//! ## A note on `unsafe`
//! `core::arch::wasm32`/`wasm64` SIMD intrinsics (`f64x2_*`, `u64x2_*`, `v128_*`)
//! are SAFE functions — unlike x86/ARM intrinsics, WASM SIMD instructions can't
//! cause memory unsafety; a module either validates with simd128 support at
//! load time or fails to load at all, there's no "runs on unsupported hardware
//! and crashes" risk to gate behind `unsafe`. Only two things in this file
//! genuinely need `unsafe`: reading a `union` field (always unsafe in Rust,
//! regardless of payload), and calls to `dot4d`/`dot4d_into_v128` from
//! `crate::wasm`, which ARE declared as `unsafe fn`.

#[cfg(target_arch = "wasm32")]
use core::arch::wasm32::*;
#[cfg(target_arch = "wasm64")]
use core::arch::wasm64::*;

use core::fmt;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::wasm::{dot4d, dot4d_into_v128};
use crate::impl_dvec4_deref;
use crate::DEPSILON;

// ── Union for compile-time constant initialisation ────────────────────────────
// DVec4 {lo@0=[x,y], hi@16=[z,w]} is byte-identical to [f64; 4].
#[repr(C)]
union UnionCast { f: [f64; 4], v: DVec4 }

// ── Type ──────────────────────────────────────────────────────────────────────

/// 4D double-precision vector. 32 bytes, 32-byte aligned.
/// Backed by 2× `v128` (f64x2): `lo=[x,y]`, `hi=[z,w]`.
///
/// **C interop:** use `CDVec4` at the FFI boundary.
#[derive(Clone, Copy)]
#[repr(C, align(32))]
pub struct DVec4 {
    pub(crate) lo: v128, // [x, y]
    pub(crate) hi: v128, // [z, w]
}

// Layout {lo@0: x@0,y@8; hi@16: z@16,w@24} ≡ XYZW<f64>{x@0,y@8,z@16,w@24}.
impl_dvec4_deref!(DVec4);

// ── Constants ─────────────────────────────────────────────────────────────────
// Union field read — genuinely unsafe, kept.

impl DVec4 {
    pub const ZERO: Self = unsafe { UnionCast { f: [0.0; 4] }.v };
    pub const ONE:  Self = unsafe { UnionCast { f: [1.0; 4] }.v };
    pub const X:    Self = unsafe { UnionCast { f: [1.0, 0.0, 0.0, 0.0] }.v };
    pub const Y:    Self = unsafe { UnionCast { f: [0.0, 1.0, 0.0, 0.0] }.v };
    pub const Z:    Self = unsafe { UnionCast { f: [0.0, 0.0, 1.0, 0.0] }.v };
    pub const W:    Self = unsafe { UnionCast { f: [0.0, 0.0, 0.0, 1.0] }.v };

    // ── Constructors ─────────────────────────────────────────────────────────

    #[inline(always)]
    pub fn new(x: f64, y: f64, z: f64, w: f64) -> Self {
        // Union field read — genuinely unsafe, kept.
        unsafe { UnionCast { f: [x, y, z, w] }.v }
    }

    #[inline(always)]
    pub fn splat(v: f64) -> Self {
        let r = f64x2_splat(v);
        Self { lo: r, hi: r }
    }

    #[inline(always)] pub fn from_array(a: [f64; 4]) -> Self { Self::new(a[0], a[1], a[2], a[3]) }
    #[inline(always)] pub fn to_array(self) -> [f64; 4] { [self.x, self.y, self.z, self.w] }

    /// Drop w, return DVec3.
    #[inline(always)]
    pub fn truncate(self) -> crate::DVec3 {
        crate::DVec3::new(self.x, self.y, self.z)
    }

    // ── Arithmetic ────────────────────────────────────────────────────────────

    /// 4-lane f64 dot product from two v128 pairs.
    #[inline(always)]
    pub fn dot(self, rhs: Self) -> f64 {
        // dot4d is an unsafe fn (crate::wasm) — kept.
        unsafe { dot4d(self.lo, self.hi, rhs.lo, rhs.hi) }
    }

    #[inline(always)] pub fn length_sq(self) -> f64 { self.dot(self) }

    #[inline]
    pub fn length(self) -> f64 { self.dot(self).sqrt() }

    #[inline]
    pub fn length_recip(self) -> f64 {
        let l = self.length();
        if l < DEPSILON { 0.0 } else { 1.0 / l }
    }

    /// Normalize. Returns ZERO for near-zero-length vectors.
    ///
    /// Computes `sqrt(dot)` broadcast to both lanes of one v128, then divides
    /// both lo and hi registers simultaneously. Guard mask zeroes degenerate lanes.
    #[inline]
    pub fn normalize(self) -> Self {
        // dot4d_into_v128 is an unsafe fn (crate::wasm) — kept; the rest of
        // this block is safe wasm32 intrinsics riding along inside it.
        unsafe {
            let len_v  = f64x2_sqrt(dot4d_into_v128(self.lo, self.hi, self.lo, self.hi));
            let lo_n   = f64x2_div(self.lo, len_v);
            let hi_n   = f64x2_div(self.hi, len_v);
            let ok     = f64x2_gt(len_v, f64x2_splat(DEPSILON));
            Self {
                lo: v128_and(lo_n, ok),
                hi: v128_and(hi_n, ok),
            }
        }
    }

    #[inline]
    pub fn try_normalize(self) -> Option<Self> {
        let rcp = self.length_recip();
        if rcp > 0.0 && rcp.is_finite() { Some(self * rcp) } else { None }
    }

    #[inline] pub fn normalize_or(self, fb: Self) -> Self { self.try_normalize().unwrap_or(fb) }
    #[inline] pub fn normalize_or_zero(self) -> Self { self.normalize_or(Self::ZERO) }
    #[inline] pub fn is_normalized(self) -> bool { (self.length_sq() - 1.0).abs() <= 2e-10 }

    // ── Interpolation ─────────────────────────────────────────────────────────

    /// Linear interpolation: `self + (rhs - self) * t`.
    /// Applied to both lo and hi registers in parallel.
    #[inline]
    pub fn lerp(self, rhs: Self, t: f64) -> Self {
        let tt      = f64x2_splat(t);
        let lo_diff = f64x2_sub(rhs.lo, self.lo);
        let hi_diff = f64x2_sub(rhs.hi, self.hi);
        Self {
            lo: f64x2_add(self.lo, f64x2_mul(lo_diff, tt)),
            hi: f64x2_add(self.hi, f64x2_mul(hi_diff, tt)),
        }
    }

    #[inline] pub fn distance(self, rhs: Self)    -> f64 { (self - rhs).length() }
    #[inline] pub fn distance_sq(self, rhs: Self) -> f64 { (self - rhs).length_sq() }

    // ── Component-wise ────────────────────────────────────────────────────────

    #[inline]
    pub fn min(self, r: Self) -> Self {
        Self {
            lo: f64x2_min(self.lo, r.lo),
            hi: f64x2_min(self.hi, r.hi),
        }
    }
    #[inline]
    pub fn max(self, r: Self) -> Self {
        Self {
            lo: f64x2_max(self.lo, r.lo),
            hi: f64x2_max(self.hi, r.hi),
        }
    }
    #[inline] pub fn clamp(self, lo: Self, hi: Self) -> Self { self.max(lo).min(hi) }
    /// Direct f64x2_abs — no sign-mask trick needed unlike SSE2.
    #[inline]
    pub fn abs(self) -> Self {
        Self {
            lo: f64x2_abs(self.lo),
            hi: f64x2_abs(self.hi),
        }
    }

    // ── Predicates ────────────────────────────────────────────────────────────

    #[inline]
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite() && self.w.is_finite()
    }
    #[inline]
    pub fn is_nan(self) -> bool {
        self.x.is_nan() || self.y.is_nan() || self.z.is_nan() || self.w.is_nan()
    }

    #[inline]
    pub fn approx_eq(self, rhs: Self) -> bool {
        let lo_diff = f64x2_abs(f64x2_sub(self.lo, rhs.lo));
        let hi_diff = f64x2_abs(f64x2_sub(self.hi, rhs.hi));
        let eps     = f64x2_splat(DEPSILON);
        let lo_ok   = f64x2_lt(lo_diff, eps);
        let hi_ok   = f64x2_lt(hi_diff, eps);
        (u64x2_bitmask(lo_ok) & 0b11) == 0b11
            && (u64x2_bitmask(hi_ok) & 0b11) == 0b11
    }

    /// Lossy cast to single-precision `Vec4`.
    #[inline]
    pub fn as_vec4(self) -> crate::Vec4 {
        crate::Vec4::new(self.x as f32, self.y as f32, self.z as f32, self.w as f32)
    }
}

// ── Operators ─────────────────────────────────────────────────────────────────

impl Add for DVec4 {
    type Output = Self;
    #[inline(always)]
    fn add(self, r: Self) -> Self {
        Self {
            lo: f64x2_add(self.lo, r.lo),
            hi: f64x2_add(self.hi, r.hi),
        }
    }
}
impl Sub for DVec4 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, r: Self) -> Self {
        Self {
            lo: f64x2_sub(self.lo, r.lo),
            hi: f64x2_sub(self.hi, r.hi),
        }
    }
}
impl Mul for DVec4 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, r: Self) -> Self {
        Self {
            lo: f64x2_mul(self.lo, r.lo),
            hi: f64x2_mul(self.hi, r.hi),
        }
    }
}
impl Div for DVec4 {
    type Output = Self;
    #[inline(always)]
    fn div(self, r: Self) -> Self {
        Self {
            lo: f64x2_div(self.lo, r.lo),
            hi: f64x2_div(self.hi, r.hi),
        }
    }
}
impl Mul<f64> for DVec4 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, s: f64) -> Self {
        let sv = f64x2_splat(s);
        Self { lo: f64x2_mul(self.lo, sv), hi: f64x2_mul(self.hi, sv) }
    }
}
impl Mul<DVec4> for f64 {
    type Output = DVec4;
    #[inline(always)] fn mul(self, v: DVec4) -> DVec4 { v * self }
}
impl Div<f64> for DVec4 {
    type Output = Self;
    #[inline(always)]
    fn div(self, s: f64) -> Self {
        let sv = f64x2_splat(s);
        Self { lo: f64x2_div(self.lo, sv), hi: f64x2_div(self.hi, sv) }
    }
}
/// Direct f64x2_neg on both registers.
impl Neg for DVec4 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self {
            lo: f64x2_neg(self.lo),
            hi: f64x2_neg(self.hi),
        }
    }
}

impl AddAssign for DVec4 {
    #[inline(always)]
    fn add_assign(&mut self, r: Self) {
        self.lo = f64x2_add(self.lo, r.lo);
        self.hi = f64x2_add(self.hi, r.hi);
    }
}
impl SubAssign for DVec4 {
    #[inline(always)]
    fn sub_assign(&mut self, r: Self) {
        self.lo = f64x2_sub(self.lo, r.lo);
        self.hi = f64x2_sub(self.hi, r.hi);
    }
}
impl MulAssign<f64> for DVec4 {
    #[inline(always)]
    fn mul_assign(&mut self, s: f64) {
        let sv = f64x2_splat(s);
        self.lo = f64x2_mul(self.lo, sv);
        self.hi = f64x2_mul(self.hi, sv);
    }
}
impl DivAssign<f64> for DVec4 {
    #[inline(always)]
    fn div_assign(&mut self, s: f64) {
        let sv = f64x2_splat(s);
        self.lo = f64x2_div(self.lo, sv);
        self.hi = f64x2_div(self.hi, sv);
    }
}

// ── PartialEq ─────────────────────────────────────────────────────────────────

impl PartialEq for DVec4 {
    #[inline]
    fn eq(&self, rhs: &Self) -> bool {
        let lo_eq = u64x2_bitmask(f64x2_eq(self.lo, rhs.lo));
        let hi_eq = u64x2_bitmask(f64x2_eq(self.hi, rhs.hi));
        (lo_eq & 0b11) == 0b11 && (hi_eq & 0b11) == 0b11
    }
}

impl Default for DVec4 { fn default() -> Self { Self::ZERO } }

// ── Display / Debug ───────────────────────────────────────────────────────────

impl fmt::Debug for DVec4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("DVec4")
            .field(&self.x).field(&self.y).field(&self.z).field(&self.w).finish()
    }
}
impl fmt::Display for DVec4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {}, {}, {})", self.x, self.y, self.z, self.w)
    }
}

impl From<[f64; 4]> for DVec4 { #[inline] fn from(a: [f64;4]) -> Self { Self::new(a[0],a[1],a[2],a[3]) } }
impl From<DVec4> for [f64; 4] { #[inline] fn from(v: DVec4) -> Self { [v.x,v.y,v.z,v.w] } }
impl From<(f64,f64,f64,f64)> for DVec4 { #[inline] fn from(t:(f64,f64,f64,f64))->Self{Self::new(t.0,t.1,t.2,t.3)} }
impl From<DVec4> for (f64,f64,f64,f64) { #[inline] fn from(v:DVec4)->(f64,f64,f64,f64){(v.x,v.y,v.z,v.w)} }
