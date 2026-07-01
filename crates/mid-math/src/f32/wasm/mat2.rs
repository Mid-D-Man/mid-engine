// crates/mid-math/src/f32/wasm/mat2.rs
//! Mat2 with WASM SIMD128 fast-paths on wasm32/wasm64.
//!
//! ## This file didn't exist before
//! `f32/mod.rs` had no `wasm::Mat2`, so the crate silently fell back to
//! `pub use mat2::Mat2` (the pure-scalar type) for the entire wasm+simd128
//! target. That's the whole story behind the vs-glam gap in build #34:
//!
//!   mat2/mul         mid-math 4.50ns   glam 1.06ns   (~4.3x)
//!   mat2/determinant mid-math 1.75ns   glam 0.70ns   (~2.5x)
//!   mat2/inverse     mid-math 4.06ns   glam 0.70ns   (~5.8x)
//!   mat2/transpose   mid-math 2.78ns   glam 0.70ns   (~4.0x)
//!
//! `mat2/mul_vec2` and `mat2/from_angle` were already close (3.58 vs 3.66ns,
//! 12.56 vs 12.70ns) — those two are dominated by the trig/scalar-extraction
//! cost either way, so scalar fallback wasn't actually hurting there. The
//! other four are pure shuffle/lane ops where SIMD should be ~free, which is
//! exactly what glam's numbers show (700ps ≈ cost of one v128 op + extract).
//!
//! ## Layout
//! Single `v128`, lanes `[x_axis.x, x_axis.y, y_axis.x, y_axis.y]` —
//! identical convention to `neon::Mat2`.
//!
//! ## Algorithm source
//! Shuffle patterns ported from glam's `f32/wasm/mat2.rs` (MIT/Apache-2.0),
//! adapted to this crate's by-value receiver convention (see neon/mat2.rs
//! fix note — everything here takes `self`/`rhs` by value to match
//! scalar::Mat2 and sse2::Mat2, `col`/`row` stay `&self`).

#[cfg(target_arch = "wasm32")]
use core::arch::wasm32::*;
#[cfg(target_arch = "wasm64")]
use core::arch::wasm64::*;

use core::fmt;
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::f32::vec2::Vec2;
use crate::wasm::v128_from_f32x4;
use crate::EPSILON;

/// 2×2 column-major matrix, WASM SIMD128-backed. 16 bytes, 16-byte aligned.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct Mat2(pub(crate) v128);

// ── Deref → field-access proxy ────────────────────────────────────────────────

impl core::ops::Deref for Mat2 {
    type Target = crate::deref::Cols2<Vec2>;
    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        unsafe { &*(self as *const Self).cast() }
    }
}

impl core::ops::DerefMut for Mat2 {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { &mut *(self as *mut Self).cast() }
    }
}

// ── Constants ─────────────────────────────────────────────────────────────────

impl Mat2 {
    pub const ZERO:     Self = Self(v128_from_f32x4([0.0, 0.0, 0.0, 0.0]));
    pub const IDENTITY: Self = Self(v128_from_f32x4([1.0, 0.0, 0.0, 1.0]));
    pub const NAN:      Self = Self(v128_from_f32x4([f32::NAN; 4]));

    #[inline(always)]
    const fn new(m00: f32, m01: f32, m10: f32, m11: f32) -> Self {
        Self(v128_from_f32x4([m00, m01, m10, m11]))
    }

    // ── Constructors ──────────────────────────────────────────────────────────

    #[inline(always)]
    pub fn from_cols(x_axis: Vec2, y_axis: Vec2) -> Self {
        Self::new(x_axis.x, x_axis.y, y_axis.x, y_axis.y)
    }

    #[inline]
    pub const fn from_cols_array(m: &[f32; 4]) -> Self {
        Self::new(m[0], m[1], m[2], m[3])
    }

    #[inline]
    pub fn to_cols_array(self) -> [f32; 4] {
        [self.x_axis.x, self.x_axis.y, self.y_axis.x, self.y_axis.y]
    }

    #[inline]
    pub const fn from_cols_array_2d(m: &[[f32; 2]; 2]) -> Self {
        Self::new(m[0][0], m[0][1], m[1][0], m[1][1])
    }

    #[inline]
    pub fn to_cols_array_2d(self) -> [[f32; 2]; 2] {
        [self.x_axis.to_array(), self.y_axis.to_array()]
    }

    #[inline]
    pub const fn from_diagonal(diagonal: Vec2) -> Self {
        Self::new(diagonal.x, 0.0, 0.0, diagonal.y)
    }

    #[inline]
    pub fn from_scale(scale: Vec2) -> Self {
        Self::new(scale.x, 0.0, 0.0, scale.y)
    }

    #[inline]
    pub fn from_scale_angle(scale: Vec2, angle: f32) -> Self {
        let (sin, cos) = angle.sin_cos();
        Self::new(cos * scale.x, sin * scale.x, -sin * scale.y, cos * scale.y)
    }

    #[inline]
    pub fn from_angle(angle: f32) -> Self {
        let (sin, cos) = angle.sin_cos();
        Self::new(cos, sin, -sin, cos)
    }

    // ── Accessors ─────────────────────────────────────────────────────────────

    #[inline]
    pub fn col(&self, index: usize) -> Vec2 {
        match index {
            0 => self.x_axis,
            1 => self.y_axis,
            _ => panic!("Mat2::col index {index} out of range"),
        }
    }

    #[inline]
    pub fn col_mut(&mut self, index: usize) -> &mut Vec2 {
        match index {
            0 => &mut self.x_axis,
            1 => &mut self.y_axis,
            _ => panic!("Mat2::col_mut index {index} out of range"),
        }
    }

    #[inline]
    pub fn row(&self, index: usize) -> Vec2 {
        match index {
            0 => Vec2::new(self.x_axis.x, self.y_axis.x),
            1 => Vec2::new(self.x_axis.y, self.y_axis.y),
            _ => panic!("Mat2::row index {index} out of range"),
        }
    }

    #[inline]
    pub fn diagonal(self) -> Vec2 {
        Vec2::new(self.x_axis.x, self.y_axis.y)
    }

    // ── Core math ─────────────────────────────────────────────────────────────

    /// Transpose: single lane shuffle, swap lanes 1 and 2.
    /// `[a,b,c,d] -> [a,c,b,d]`. This was 2.78ns scalar; should land ~700ps.
    #[inline]
    pub fn transpose(self) -> Self {
        Self(i32x4_shuffle::<0, 2, 5, 7>(self.0, self.0))
    }

    /// Determinant: `m00*m11 - m01*m10`, done as shuffle+mul+sub, no branches.
    #[inline]
    pub fn determinant(self) -> f32 {
        let abcd = self.0;
        let dcba = i32x4_shuffle::<3, 2, 5, 4>(abcd, abcd);
        let prod = f32x4_mul(abcd, dcba);
        let det  = f32x4_sub(prod, i32x4_shuffle::<1, 1, 5, 5>(prod, prod));
        f32x4_extract_lane::<0>(det)
    }

    #[inline(always)]
    fn inverse_checked(self, checked: bool) -> (Self, bool) {
        const SIGN: v128 = v128_from_f32x4([1.0, -1.0, -1.0, 1.0]);
        let abcd = self.0;
        let dcba = i32x4_shuffle::<3, 2, 5, 4>(abcd, abcd);
        let prod = f32x4_mul(abcd, dcba);
        let sub  = f32x4_sub(prod, i32x4_shuffle::<1, 1, 5, 5>(prod, prod));
        let det  = i32x4_shuffle::<0, 0, 4, 4>(sub, sub);

        if checked {
            let det0 = f32x4_extract_lane::<0>(det);
            if det0.abs() < EPSILON {
                return (Self::ZERO, false);
            }
        }

        let tmp  = f32x4_div(SIGN, det);
        let dbca = i32x4_shuffle::<3, 1, 6, 4>(abcd, abcd);
        (Self(f32x4_mul(dbca, tmp)), true)
    }

    /// Try to invert. Returns `None` if singular (`|det| < ε`).
    #[inline]
    pub fn try_inverse(self) -> Option<Self> {
        let (m, ok) = self.inverse_checked(true);
        if ok { Some(m) } else { None }
    }

    /// Inverse, returning `Mat2::ZERO` if singular.
    #[inline]
    pub fn inverse_or_zero(self) -> Self {
        self.inverse_checked(true).0
    }

    /// Inverse, assuming the matrix IS invertible. Undefined if singular.
    #[inline]
    pub fn inverse_unchecked(self) -> Self {
        self.inverse_checked(false).0
    }

    /// Alias for `try_inverse` — kept for API compatibility.
    #[inline]
    pub fn inverse(self) -> Option<Self> { self.try_inverse() }

    // ── Transforms ────────────────────────────────────────────────────────────

    #[inline]
    pub fn mul_vec2(self, rhs: Vec2) -> Vec2 {
        let abcd      = self.0;
        let xxyy      = f32x4(rhs.x, rhs.x, rhs.y, rhs.y);
        let axbxcydy  = f32x4_mul(abcd, xxyy);
        let cydyaxbx  = i32x4_shuffle::<2, 3, 4, 5>(axbxcydy, axbxcydy);
        let result    = f32x4_add(axbxcydy, cydyaxbx);
        Vec2::new(f32x4_extract_lane::<0>(result), f32x4_extract_lane::<1>(result))
    }

    #[inline]
    pub fn mul_transpose_vec2(self, rhs: Vec2) -> Vec2 {
        Vec2::new(self.x_axis.dot(rhs), self.y_axis.dot(rhs))
    }

    /// M × N — this was the worst offender (4.50ns scalar vs glam's 1.06ns).
    #[inline]
    pub fn mul_mat2(self, rhs: Self) -> Self {
        let abcd  = self.0;
        let rhs0  = rhs.0;
        let xxyy0 = i32x4_shuffle::<0, 0, 5, 5>(rhs0, rhs0);
        let xxyy1 = i32x4_shuffle::<2, 2, 7, 7>(rhs0, rhs0);
        let ac0   = f32x4_mul(abcd, xxyy0);
        let ac1   = f32x4_mul(abcd, xxyy1);
        let rot0  = i32x4_shuffle::<2, 3, 4, 5>(ac0, ac0);
        let rot1  = i32x4_shuffle::<2, 3, 4, 5>(ac1, ac1);
        let res0  = f32x4_add(ac0, rot0);
        let res1  = f32x4_add(ac1, rot1);
        Self(i32x4_shuffle::<0, 1, 4, 5>(res0, res1))
    }

    // ── Scalar operations ─────────────────────────────────────────────────────

    #[inline]
    pub fn mul_scalar(self, rhs: f32) -> Self {
        Self(f32x4_mul(self.0, f32x4_splat(rhs)))
    }

    #[inline]
    pub fn div_scalar(self, rhs: f32) -> Self {
        Self(f32x4_div(self.0, f32x4_splat(rhs)))
    }

    #[inline]
    pub fn mul_diagonal_scale(self, scale: Vec2) -> Self {
        Self::from_cols(self.x_axis * scale.x, self.y_axis * scale.y)
    }

    #[inline]
    pub fn add_mat2(self, rhs: Self) -> Self {
        Self(f32x4_add(self.0, rhs.0))
    }

    #[inline]
    pub fn sub_mat2(self, rhs: Self) -> Self {
        Self(f32x4_sub(self.0, rhs.0))
    }

    #[inline]
    pub fn abs(self) -> Self {
        Self::from_cols(self.x_axis.abs(), self.y_axis.abs())
    }

    #[inline]
    pub fn recip(self) -> Self {
        Self::from_cols(self.x_axis.recip(), self.y_axis.recip())
    }

    // ── Classification ────────────────────────────────────────────────────────

    #[inline]
    pub fn is_finite(self) -> bool {
        self.x_axis.is_finite() && self.y_axis.is_finite()
    }

    #[inline]
    pub fn is_nan(self) -> bool {
        self.x_axis.is_nan() || self.y_axis.is_nan()
    }

    #[inline]
    pub fn abs_diff_eq(self, rhs: Self, max_abs_diff: f32) -> bool {
        self.x_axis.abs_diff_eq(rhs.x_axis, max_abs_diff)
            && self.y_axis.abs_diff_eq(rhs.y_axis, max_abs_diff)
    }
}

// ── Trait implementations ─────────────────────────────────────────────────────

impl Default for Mat2 {
    #[inline] fn default() -> Self { Self::IDENTITY }
}

impl PartialEq for Mat2 {
    #[inline]
    fn eq(&self, rhs: &Self) -> bool {
        self.x_axis == rhs.x_axis && self.y_axis == rhs.y_axis
    }
}

impl Add for Mat2 {
    type Output = Self;
    #[inline] fn add(self, rhs: Self) -> Self { self.add_mat2(rhs) }
}
impl AddAssign for Mat2 {
    #[inline] fn add_assign(&mut self, rhs: Self) { *self = self.add_mat2(rhs); }
}
impl Sub for Mat2 {
    type Output = Self;
    #[inline] fn sub(self, rhs: Self) -> Self { self.sub_mat2(rhs) }
}
impl SubAssign for Mat2 {
    #[inline] fn sub_assign(&mut self, rhs: Self) { *self = self.sub_mat2(rhs); }
}
impl Neg for Mat2 {
    type Output = Self;
    #[inline] fn neg(self) -> Self { Self(f32x4_neg(self.0)) }
}
impl Mul for Mat2 {
    type Output = Self;
    #[inline] fn mul(self, rhs: Self) -> Self { self.mul_mat2(rhs) }
}
impl MulAssign for Mat2 {
    #[inline] fn mul_assign(&mut self, rhs: Self) { *self = self.mul_mat2(rhs); }
}
impl Mul<Vec2> for Mat2 {
    type Output = Vec2;
    #[inline] fn mul(self, rhs: Vec2) -> Vec2 { self.mul_vec2(rhs) }
}
impl Mul<f32> for Mat2 {
    type Output = Self;
    #[inline] fn mul(self, rhs: f32) -> Self { self.mul_scalar(rhs) }
}
impl Mul<Mat2> for f32 {
    type Output = Mat2;
    #[inline] fn mul(self, rhs: Mat2) -> Mat2 { rhs.mul_scalar(self) }
}
impl MulAssign<f32> for Mat2 {
    #[inline] fn mul_assign(&mut self, rhs: f32) { *self = self.mul_scalar(rhs); }
}

impl AsRef<[f32; 4]> for Mat2 {
    #[inline] fn as_ref(&self) -> &[f32; 4] {
        unsafe { &*(self as *const Self as *const [f32; 4]) }
    }
}
impl AsMut<[f32; 4]> for Mat2 {
    #[inline] fn as_mut(&mut self) -> &mut [f32; 4] {
        unsafe { &mut *(self as *mut Self as *mut [f32; 4]) }
    }
}

impl fmt::Debug for Mat2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Mat2")
            .field("x_axis", &self.x_axis)
            .field("y_axis", &self.y_axis)
            .finish()
    }
}
impl fmt::Display for Mat2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}, {}]", self.x_axis, self.y_axis)
    }
  }
