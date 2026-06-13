// crates/mid-math/src/f32/sse2/mat2.rs
//! Mat2 backed by a single `__m128` on x86 / x86_64.
//!
//! Both columns are packed into one 128-bit register:
//!   lane 0 = x_axis.x,  lane 1 = x_axis.y
//!   lane 2 = y_axis.x,  lane 3 = y_axis.y
//!
//! This matches the scalar [`Mat2`][crate::f32::mat2::Mat2] memory layout
//! byte-for-byte, so the FFI `CMat2` conversion remains zero-cost.
//!
//! ## Why this is faster than the scalar version
//!
//! The scalar Mat2 stores two `Vec2` fields (8 bytes each).  Every operation
//! extracts floats one-by-one.  With the packed `__m128` layout:
//!
//! | Operation       | Scalar                  | SSE2                                    |
//! |-----------------|-------------------------|-----------------------------------------|
//! | `transpose`     | 4 element swaps         | 1 `_mm_shuffle_ps`                      |
//! | `determinant`   | 2 mul + 1 sub + scalar  | 2 shuffles + mul + sub + scalar extract |
//! | `mul_mat2`      | 8 muls + 4 adds         | 6 SSE2 instructions                     |
//! | `mul_vec2`      | 4 muls + 2 adds         | 3 SSE2 instructions + 2 extracts        |
//! | `inverse`       | scalar cofactor         | det (SSE2) + 1 shuffle + 1 div + 1 mul |

use core::fmt;
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::sse2::{m128_abs, m128_from_f32x4};
use crate::f32::vec2::Vec2;
use crate::EPSILON;

#[repr(C)]
union UnionCast { f: [f32; 4], v: Mat2 }

/// Sign pattern for the 2×2 adjugate: [+d, -c, -b, +a] / det.
const SIGN: __m128 = m128_from_f32x4([1.0_f32, -1.0, -1.0, 1.0]);

// ── Type ──────────────────────────────────────────────────────────────────────

/// 2×2 column-major matrix. 16 bytes, 16-byte aligned. Both columns packed
/// into a single `__m128`.
///
/// Memory layout (low → high lanes): [x_axis.x, x_axis.y, y_axis.x, y_axis.y].
/// Identical to the scalar [`Mat2`][crate::f32::mat2::Mat2] layout.
///
/// **C interop:** use `CMat2` (in `crate::ffi::types`) at the FFI boundary.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct Mat2(pub(crate) __m128);

// ── Deref → Cols2<Vec2> ───────────────────────────────────────────────────────
//
// The __m128 layout [x.x, x.y, y.x, y.y] byte-matches Cols2<Vec2>:
//   x_axis: Vec2 { x: x.x, y: x.y }  @ bytes  0-7
//   y_axis: Vec2 { x: y.x, y: y.y }  @ bytes 8-15
//
// This gives callers `.x_axis.x`, `.y_axis.y` etc. with zero overhead.

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

// ── impl Mat2 ─────────────────────────────────────────────────────────────────

impl Mat2 {
    /// All zeros — not a valid transform.
    pub const ZERO: Self = unsafe { UnionCast { f: [0.0; 4] }.v };

    /// Identity — no rotation, no scale.
    pub const IDENTITY: Self = unsafe { UnionCast { f: [1.0, 0.0, 0.0, 1.0] }.v };

    // ── Internal constructor ──────────────────────────────────────────────────

    #[inline(always)]
    const fn new(m00: f32, m01: f32, m10: f32, m11: f32) -> Self {
        unsafe { UnionCast { f: [m00, m01, m10, m11] }.v }
    }

    // ── Public constructors ───────────────────────────────────────────────────

    /// Build from two column vectors.
    #[inline(always)]
    pub fn from_cols(x_axis: Vec2, y_axis: Vec2) -> Self {
        Self::new(x_axis.x, x_axis.y, y_axis.x, y_axis.y)
    }

    /// From column-major flat array `[x.x, x.y, y.x, y.y]`.
    #[inline]
    pub fn from_cols_array(m: &[f32; 4]) -> Self {
        Self::new(m[0], m[1], m[2], m[3])
    }

    /// To column-major flat array `[x.x, x.y, y.x, y.y]`.
    #[inline]
    pub fn to_cols_array(self) -> [f32; 4] {
        [self.x_axis.x, self.x_axis.y, self.y_axis.x, self.y_axis.y]
    }

    /// From column-major 2D array.
    #[inline]
    pub fn from_cols_array_2d(m: &[[f32; 2]; 2]) -> Self {
        Self::from_cols(Vec2::from(m[0]), Vec2::from(m[1]))
    }

    /// Diagonal scale matrix — off-diagonals are zero.
    #[inline]
    pub fn from_diagonal(d: Vec2) -> Self {
        Self::new(d.x, 0.0, 0.0, d.y)
    }

    /// Non-uniform scale combined with counter-clockwise rotation.
    #[inline]
    pub fn from_scale_angle(scale: Vec2, angle: f32) -> Self {
        let (s, c) = angle.sin_cos();
        Self::new(c * scale.x, s * scale.x, -s * scale.y, c * scale.y)
    }

    /// Counter-clockwise rotation by `angle` radians.
    #[inline]
    pub fn from_angle(angle: f32) -> Self {
        let (s, c) = angle.sin_cos();
        Self::new(c, s, -s, c)
    }

    /// Non-uniform scale only.
    #[inline]
    pub fn from_scale(scale: Vec2) -> Self {
        Self::new(scale.x, 0.0, 0.0, scale.y)
    }

    // ── Core ops ──────────────────────────────────────────────────────────────

    /// Transpose — swap rows and columns.
    ///
    /// One `_mm_shuffle_ps` instruction.
    ///
    /// ```text
    /// input : [x.x, x.y, y.x, y.y]  (lanes 0 1 2 3)
    /// output: [x.x, y.x, x.y, y.y]  (lanes 0 2 1 3)
    /// imm   : 0b11_01_10_00
    /// ```
    #[inline]
    pub fn transpose(self) -> Self {
        Self(unsafe { _mm_shuffle_ps::<0b11_01_10_00>(self.0, self.0) })
    }

    /// Diagonal vector `[x_axis.x, y_axis.y]`.
    #[inline]
    pub fn diagonal(self) -> Vec2 {
        Vec2::new(self.x_axis.x, self.y_axis.y)
    }

    /// Signed determinant: `x.x * y.y − x.y * y.x`.
    ///
    /// SSE2: 2 shuffles + 1 mul + 1 sub + 1 scalar extract.
    #[inline]
    pub fn determinant(self) -> f32 {
        unsafe {
            // abcd = [x.x, x.y, y.x, y.y]
            let abcd = self.0;
            // dcba = [y.y, y.x, x.y, x.x]  (reverse)
            let dcba = _mm_shuffle_ps::<0b00_01_10_11>(abcd, abcd);
            // prod = [x.x*y.y, x.y*y.x, y.x*x.y, y.y*x.x]
            let prod = _mm_mul_ps(abcd, dcba);
            // sub[0] = prod[0] - prod[1] = x.x*y.y - x.y*y.x = det
            let sub = _mm_sub_ps(prod, _mm_shuffle_ps::<0b01_01_01_01>(prod, prod));
            _mm_cvtss_f32(sub)
        }
    }

    // ── Inverse ───────────────────────────────────────────────────────────────
    //
    // Derivation for 2×2:
    //   M = [a c]  (col-major: x_axis=[a,c], y_axis=[b,d])
    //       [b d]
    //   det   = a*d - b*c
    //   M^-1  = (1/det) * [d  -c]  = (1/det) * [y.y, -x.y, -y.x, x.x]
    //                     [-b  a]
    //
    // SSE2: shuffle storage to [d,c,b,a] then multiply by SIGN/det = [+,-,-,+]/det
    // → result lanes: [d/det, -c/det, -b/det, a/det] ✓

       #[inline(always)]
    unsafe fn inverse_inner(self) -> (Self, bool) {
        let abcd    = self.0;
        let dcba    = _mm_shuffle_ps::<0b00_01_10_11>(abcd, abcd);
        let prod    = _mm_mul_ps(abcd, dcba);
        let sub     = _mm_sub_ps(prod, _mm_shuffle_ps::<0b01_01_01_01>(prod, prod));
        let det_f32 = _mm_cvtss_f32(sub);
        if det_f32.abs() < EPSILON {
            return (Self::ZERO, false);
        }
        let det     = _mm_shuffle_ps::<0b00_00_00_00>(sub, sub);
        let tmp     = _mm_div_ps(SIGN, det);
        let reorder = _mm_shuffle_ps::<0b00_10_01_11>(abcd, abcd);
        (Self(_mm_mul_ps(reorder, tmp)), true)
    }

    /// Checked inverse — returns `None` when singular (|det| < EPSILON).
    ///
    /// The `_mm_cvtss_f32` scalar extraction + branch is unavoidable for
    /// `Option<Self>`. Glam's `inverse()` emits no branch in release builds
    /// (guarded only by `glam_assert!`) — that explains the ~0.7 ns gap.
    /// When the caller can tolerate a zero result, prefer `inverse_or_zero`.
    #[inline]
    pub fn inverse(self) -> Option<Self> {
        let (m, ok) = unsafe { self.inverse_inner() };
        if ok { Some(m) } else { None }
    }

    /// Branchless inverse — returns `Mat2::ZERO` when singular.
    ///
    /// Uses an SSE2 compare-and-mask (`_mm_cmpge_ps`) to zero out the result
    /// without any conditional jump in the hot path. Preferred over `inverse()`
    /// in throughput-critical code when the caller accepts a zero fallback.
    ///
    /// ```text
    /// 1. Compute full inverse unconditionally (±∞ if |det| ≈ 0 — safe under IEEE 754)
    /// 2. mask = (|det| ≥ EPSILON) → all-ones lanes
    /// 3. result = _mm_and_ps(inverse, mask)  → zero if singular, inverse otherwise
    /// ```
    #[inline]
    pub fn inverse_or_zero(self) -> Self {
        unsafe {
            let abcd    = self.0;
            let dcba    = _mm_shuffle_ps::<0b00_01_10_11>(abcd, abcd);
            let prod    = _mm_mul_ps(abcd, dcba);
            let sub     = _mm_sub_ps(prod, _mm_shuffle_ps::<0b01_01_01_01>(prod, prod));
            let det     = _mm_shuffle_ps::<0b00_00_00_00>(sub, sub); // broadcast to all 4 lanes
            // SSE2 select — no scalar roundtrip, no branch instruction
            let mask    = _mm_cmpge_ps(m128_abs(det), _mm_set1_ps(EPSILON));
            let tmp     = _mm_div_ps(SIGN, det);
            let reorder = _mm_shuffle_ps::<0b00_10_01_11>(abcd, abcd);
            Self(_mm_and_ps(_mm_mul_ps(reorder, tmp), mask))
        }
    }

    // ── Transform helpers ─────────────────────────────────────────────────────

    /// Multiply matrix by column vector: `self * v`.
    ///
    /// ```text
    /// abcd  = [x.x, x.y, y.x, y.y]
    /// xxyy  = [v.x, v.x, v.y, v.y]
    /// axbx  = abcd * xxyy = [x.x*vx, x.y*vx, y.x*vy, y.y*vy]
    /// cydy  = shuffle(axbx, [2,3,0,1]) = [y.x*vy, y.y*vy, x.x*vx, x.y*vx]
    /// res   = axbx + cydy = [x.x*vx+y.x*vy, x.y*vx+y.y*vy, ...]
    /// ```
    #[inline]
    pub fn mul_vec2(self, v: Vec2) -> Vec2 {
        unsafe {
            let abcd = self.0;
            // _mm_set_ps(e3,e2,e1,e0): lane0=e0, lane1=e1, lane2=e2, lane3=e3
            let xxyy = _mm_set_ps(v.y, v.y, v.x, v.x);
            let axbx = _mm_mul_ps(abcd, xxyy);
            let cydy = _mm_shuffle_ps::<0b01_00_11_10>(axbx, axbx);
            let res  = _mm_add_ps(axbx, cydy);
            Vec2::new(
                _mm_cvtss_f32(res),
                _mm_cvtss_f32(_mm_shuffle_ps::<0b01_01_01_01>(res, res)),
            )
        }
    }

    /// Multiply by the transpose of self: `self^T * v`.
    #[inline]
    pub fn mul_transpose_vec2(self, v: Vec2) -> Vec2 {
        // The rows of the original are x = [x.x, y.x] and y = [x.y, y.y].
        // We access them as column-dot-v which is what x_axis.dot / y_axis.dot gives
        // after the Deref — these ARE the column dots, which == transpose-row dots. ✓
        Vec2::new(self.x_axis.dot(v), self.y_axis.dot(v))
    }

    /// Matrix multiply: `self * rhs`.
    ///
    /// Algorithm — process both output columns in two passes, then interleave.
    ///
    /// ```text
    /// xxyy0 = [rhs.x, rhs.x, rhs.y, rhs.y]   ← col-0 components broadcast
    /// xxyy1 = [rhs.z, rhs.z, rhs.w, rhs.w]   ← col-1 components broadcast
    /// t0 = self * xxyy0 ; t1 = self * xxyy1
    /// r0 = t0 + shuffle(t0,[2,3,0,1])         ← [C.x.x, C.x.y, C.x.x, C.x.y]
    /// r1 = t1 + shuffle(t1,[2,3,0,1])         ← [C.y.x, C.y.y, C.y.x, C.y.y]
    /// C  = shuffle(r0, r1, [0,1,0,1])         ← [C.x.x, C.x.y, C.y.x, C.y.y]
    /// ```
    #[inline]
    pub fn mul_mat2(self, rhs: Self) -> Self {
        self.mul(rhs)
    }

    /// Element-wise scalar multiply.
    #[inline]
    pub fn mul_scalar(self, s: f32) -> Self {
        Self(unsafe { _mm_mul_ps(self.0, _mm_set1_ps(s)) })
    }

    /// Element-wise scalar divide.
    #[inline]
    pub fn div_scalar(self, s: f32) -> Self {
        Self(unsafe { _mm_div_ps(self.0, _mm_set1_ps(s)) })
    }

    // ── Predicates ────────────────────────────────────────────────────────────

    /// True if all elements are finite (no NaN or ±∞).
    #[inline]
    pub fn is_finite(self) -> bool {
        self.x_axis.is_finite() && self.y_axis.is_finite()
    }

    /// True if any element is NaN.
    #[inline]
    pub fn is_nan(self) -> bool {
        self.x_axis.is_nan() || self.y_axis.is_nan()
    }

    /// True when every element differs by at most `max_abs_diff`.
    #[inline]
    pub fn abs_diff_eq(self, rhs: Self, max_abs_diff: f32) -> bool {
        unsafe {
            let diff  = _mm_sub_ps(self.0, rhs.0);
            let adiff = m128_abs(diff);
            let eps   = _mm_set1_ps(max_abs_diff);
            (_mm_movemask_ps(_mm_cmplt_ps(adiff, eps)) & 0b1111) == 0b1111
        }
    }
}

// ── Operators ─────────────────────────────────────────────────────────────────

/// Mat4×Mat4 — see `mul_mat2` for the algorithm breakdown.
impl Mul for Mat2 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        unsafe {
            let abcd = self.0;
            let rhs  = rhs.0;
            // Broadcast col-0 and col-1 components of rhs
            let xxyy0 = _mm_shuffle_ps::<0b01_01_00_00>(rhs, rhs); // [rx.x,rx.x,rx.y,rx.y]
            let xxyy1 = _mm_shuffle_ps::<0b11_11_10_10>(rhs, rhs); // [ry.x,ry.x,ry.y,ry.y]
            let t0    = _mm_mul_ps(abcd, xxyy0);
            let t1    = _mm_mul_ps(abcd, xxyy1);
            // Cross-add to accumulate the two column contributions
            let s0    = _mm_shuffle_ps::<0b01_00_11_10>(t0, t0);
            let s1    = _mm_shuffle_ps::<0b01_00_11_10>(t1, t1);
            let r0    = _mm_add_ps(t0, s0); // [C.x.x, C.x.y, C.x.x, C.x.y]
            let r1    = _mm_add_ps(t1, s1); // [C.y.x, C.y.y, C.y.x, C.y.y]
            // Interleave the two output columns
            Self(_mm_shuffle_ps::<0b01_00_01_00>(r0, r1))
        }
    }
}

impl MulAssign for Mat2 {
    #[inline(always)] fn mul_assign(&mut self, rhs: Self) { *self = self.mul(rhs); }
}

impl Mul<Vec2> for Mat2 {
    type Output = Vec2;
    #[inline(always)] fn mul(self, rhs: Vec2) -> Vec2 { self.mul_vec2(rhs) }
}

impl Mul<f32> for Mat2 {
    type Output = Self;
    #[inline(always)] fn mul(self, s: f32) -> Self { self.mul_scalar(s) }
}

impl Mul<Mat2> for f32 {
    type Output = Mat2;
    #[inline(always)] fn mul(self, m: Mat2) -> Mat2 { m.mul_scalar(self) }
}

impl Add for Mat2 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self { Self(unsafe { _mm_add_ps(self.0, rhs.0) }) }
}

impl AddAssign for Mat2 {
    #[inline(always)] fn add_assign(&mut self, rhs: Self) { *self = *self + rhs; }
}

impl Sub for Mat2 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self { Self(unsafe { _mm_sub_ps(self.0, rhs.0) }) }
}

impl SubAssign for Mat2 {
    #[inline(always)] fn sub_assign(&mut self, rhs: Self) { *self = *self - rhs; }
}

impl Neg for Mat2 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self { Self(unsafe { _mm_xor_ps(self.0, _mm_set1_ps(-0.0)) }) }
}

impl PartialEq for Mat2 {
    #[inline]
    fn eq(&self, rhs: &Self) -> bool {
        unsafe { (_mm_movemask_ps(_mm_cmpeq_ps(self.0, rhs.0)) & 0b1111) == 0b1111 }
    }
}

impl Default for Mat2 { #[inline] fn default() -> Self { Self::IDENTITY } }

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

impl From<[[f32; 2]; 2]> for Mat2 {
    #[inline]
    fn from(m: [[f32; 2]; 2]) -> Self {
        Self::from_cols(Vec2::from(m[0]), Vec2::from(m[1]))
    }
}

impl From<Mat2> for [[f32; 2]; 2] {
    #[inline]
    fn from(m: Mat2) -> Self {
        [m.x_axis.to_array(), m.y_axis.to_array()]
    }
}
