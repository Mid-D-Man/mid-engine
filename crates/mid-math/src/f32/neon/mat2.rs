// crates/mid-math/src/f32/neon/mat2.rs
//! NEON-backed 2×2 matrix — aarch64.
//!
//! ## Layout
//! All 4 floats packed into one `float32x4_t`:
//!   lanes [0,1,2,3] = [x_axis.x, x_axis.y, y_axis.x, y_axis.y]
//!
//! ## Why this matters vs. scalar Mat2
//! Scalar Mat2 stores two separate Vec2 fields. Every operation touches two
//! registers with scalar instructions. NEON Mat2 keeps the entire matrix in
//! ONE 128-bit register — same as glam's approach — so:
//!   transpose  = 2 NEON instructions (vrev64 + vext)
//!   det        = ~4 instructions, no lane extractions
//!   mul_mat2   = 6 NEON instructions (vs ~12 scalar muls + 8 scalar adds)
//!   mul_vec2   = 4 NEON instructions (vs 4 scalar muls + 2 scalar adds)
//!
//! Algorithm taken from glam's neon/mat2.rs (MIT/Apache-2.0).

use core::arch::aarch64::*;
use core::fmt;
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::f32::vec2::Vec2;
use crate::EPSILON;

#[repr(C)]
union UnionCast { a: [f32; 4], v: Mat2 }

/// 2×2 column-major matrix, NEON-backed.
///
/// **Layout** (lane order): `[x_axis.x, x_axis.y, y_axis.x, y_axis.y]`
///
/// Both columns live in a single `float32x4_t` register.
/// 16-byte aligned.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct Mat2(pub(crate) float32x4_t);

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
    pub const ZERO:     Self = unsafe { UnionCast { a: [0.0; 4]            }.v };
    pub const IDENTITY: Self = unsafe { UnionCast { a: [1.0, 0.0, 0.0, 1.0] }.v };
    pub const NAN:      Self = unsafe { UnionCast { a: [f32::NAN; 4]       }.v };

    #[inline(always)]
    const fn new(m00: f32, m01: f32, m10: f32, m11: f32) -> Self {
        unsafe { UnionCast { a: [m00, m01, m10, m11] }.v }
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
    pub const fn to_cols_array(&self) -> [f32; 4] {
        unsafe { *(self as *const Self as *const [f32; 4]) }
    }

    #[inline]
    pub const fn from_cols_array_2d(m: &[[f32; 2]; 2]) -> Self {
        Self::new(m[0][0], m[0][1], m[1][0], m[1][1])
    }

    #[inline]
    pub const fn to_cols_array_2d(&self) -> [[f32; 2]; 2] {
        unsafe { *(self as *const Self as *const [[f32; 2]; 2]) }
    }

    #[inline]
    pub const fn from_diagonal(diagonal: Vec2) -> Self {
        Self::new(diagonal.x, 0.0, 0.0, diagonal.y)
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

    // ── Core math ─────────────────────────────────────────────────────────────

    /// Transpose. 2 NEON instructions.
    #[inline]
    pub fn transpose(&self) -> Self {
        // abcd → badc (vrev64) → cdab (vext 2) → then swap back
        // More precisely: we have [a, b, c, d] where a=m00, b=m01, c=m10, d=m11
        // Transpose means [a, c, b, d] (swap lanes 1 and 2)
        // vrev64: [b,a, d,c]
        // then vsetq_lane to swap lane 1 from original with lane 2
        Self(unsafe {
            vsetq_lane_f32::<1>(
                vgetq_lane_f32::<2>(self.0),
                vsetq_lane_f32::<2>(vgetq_lane_f32::<1>(self.0), self.0),
            )
        })
    }

    /// Diagonal elements `[m00, m11]`.
    #[inline]
    pub fn diagonal(&self) -> Vec2 {
        Vec2::new(self.x_axis.x, self.y_axis.y)
    }

    /// Determinant: `m00*m11 - m01*m10`. ~4 NEON instructions.
    #[inline]
    pub fn determinant(&self) -> f32 {
        unsafe {
            // abcd is [m00, m01, m10, m11]
            // badc = vrev64(abcd) = [m01, m00, m11, m10]
            // dcba = vext(badc, badc, 2) = [m11, m10, m01, m00]
            // prod = abcd * dcba = [m00*m11, m01*m10, m10*m01, m11*m00]
            // det  = prod[0] - prod[1]
            let abcd = self.0;
            let badc = vrev64q_f32(abcd);
            let dcba = vextq_f32::<2>(badc, badc);
            let prod = vmulq_f32(abcd, dcba);
            let det  = vsubq_f32(prod, vdupq_laneq_f32::<1>(prod));
            vgetq_lane_f32::<0>(det)
        }
    }

    /// Inverse. Returns `None` if singular (`|det| < ε`).
    #[inline]
    pub fn try_inverse(&self) -> Option<Self> {
        let det = self.determinant();
        if det.abs() < EPSILON { return None; }
        Some(self.inverse_unchecked())
    }

    /// Inverse, returning `Mat2::ZERO` if singular.
    #[inline]
    pub fn inverse_or_zero(&self) -> Self {
        self.try_inverse().unwrap_or(Self::ZERO)
    }

    /// Inverse, assuming the matrix IS invertible. Undefined if singular.
    #[inline]
    pub fn inverse_unchecked(&self) -> Self {
        unsafe {
            // SIGN pattern for adjugate: [+1, -1, -1, +1]
            // Adjugate of [[a,b],[c,d]] = [[d,-b],[-c,a]]
            // In our lane layout [a,b,c,d]: adjugate = [d,-b,-c,a]
            // = dbca reordered as [d,b,c,a] with sign [+,-,-,+]
            const SIGN: float32x4_t = unsafe {
                core::mem::transmute([1.0f32, -1.0, -1.0, 1.0f32])
            };
            let abcd = self.0;
            let badc = vrev64q_f32(abcd);
            let dcba = vextq_f32::<2>(badc, badc);
            let prod = vmulq_f32(abcd, dcba);
            let sub  = vsubq_f32(prod, vdupq_laneq_f32::<1>(prod));
            let det  = vdupq_laneq_f32::<0>(sub);

            // Swap lanes 0↔3 to get the cofactor layout [d,-b,-c,a]
            let dbca = vsetq_lane_f32::<0>(
                vgetq_lane_f32::<0>(abcd),
                vsetq_lane_f32::<3>(vgetq_lane_f32::<3>(abcd), abcd),
            );
            // Apply sign pattern then divide by det
            let signed = vmulq_f32(dbca, SIGN);
            Self(vdivq_f32(signed, det))
        }
    }

    /// Alias for `inverse_unchecked` — kept for API compatibility.
    #[inline]
    pub fn inverse(self) -> Option<Self> { self.try_inverse() }

    // ── Transforms ────────────────────────────────────────────────────────────

    /// M × v. 4 NEON instructions.
    #[inline]
    pub fn mul_vec2(&self, rhs: Vec2) -> Vec2 {
        unsafe {
            // abcd = [m00, m01, m10, m11]
            // xxyy = [rhs.x, rhs.x, rhs.y, rhs.y]
            // axbxcydy = abcd * xxyy
            // cydyaxbx = vext(axbxcydy, axbxcydy, 2) → rotate by 2
            // result   = axbxcydy + cydyaxbx → [ax+cy, bx+dy, ...]
            let abcd   = self.0;
            let xxyy   = vld1q_f32([rhs.x, rhs.x, rhs.y, rhs.y].as_ptr());
            let prod   = vmulq_f32(abcd, xxyy);
            let rotprd = vextq_f32::<2>(prod, prod);
            let result = vaddq_f32(prod, rotprd);
            // Result is in lanes 0,1
            *(&result as *const float32x4_t as *const Vec2)
        }
    }

    /// M^T × v (cheaper than transposing first).
    #[inline]
    pub fn mul_transpose_vec2(&self, rhs: Vec2) -> Vec2 {
        Vec2::new(self.x_axis.dot(rhs), self.y_axis.dot(rhs))
    }

    /// M × N. 6 NEON instructions.
    #[inline]
    pub fn mul_mat2(&self, rhs: &Self) -> Self {
        unsafe {
            let abcd = self.0;
            // Process both columns of rhs simultaneously
            let xxyy0 = vzip1q_f32(rhs.0, rhs.0); // [x0,x0,y0,y0] (col 0 spread)
            let xxyy1 = vzip2q_f32(rhs.0, rhs.0); // [x1,x1,y1,y1] (col 1 spread)

            let prod0    = vmulq_f32(abcd, xxyy0);
            let prod1    = vmulq_f32(abcd, xxyy1);
            let rotprd0  = vextq_f32::<2>(prod0, prod0);
            let rotprd1  = vextq_f32::<2>(prod1, prod1);
            let result0  = vaddq_f32(prod0, rotprd0); // col 0 result in lanes 0,1
            let result1  = vaddq_f32(prod1, rotprd1); // col 1 result in lanes 0,1

            // Pack both results: lanes [0,1] from result0 into [0,1], result1 into [2,3]
            Self(vreinterpretq_f32_u64(vsetq_lane_u64::<1>(
                vgetq_lane_u64::<0>(vreinterpretq_u64_f32(result1)),
                vreinterpretq_u64_f32(result0),
            )))
        }
    }

    // ── Scalar operations ─────────────────────────────────────────────────────

    #[inline]
    pub fn mul_scalar(&self, rhs: f32) -> Self {
        Self(unsafe { vmulq_n_f32(self.0, rhs) })
    }

    #[inline]
    pub fn div_scalar(&self, rhs: f32) -> Self {
        Self(unsafe { vdivq_f32(self.0, vdupq_n_f32(rhs)) })
    }

    #[inline]
    pub fn mul_diagonal_scale(&self, scale: Vec2) -> Self {
        Self::from_cols(self.x_axis * scale.x, self.y_axis * scale.y)
    }

    #[inline]
    pub fn add_mat2(&self, rhs: &Self) -> Self {
        Self(unsafe { vaddq_f32(self.0, rhs.0) })
    }

    #[inline]
    pub fn sub_mat2(&self, rhs: &Self) -> Self {
        Self(unsafe { vsubq_f32(self.0, rhs.0) })
    }

    #[inline]
    pub fn abs(&self) -> Self {
        Self(unsafe { vabsq_f32(self.0) })
    }

    #[inline]
    pub fn recip(&self) -> Self {
        Self::from_cols(self.x_axis.recip(), self.y_axis.recip())
    }

    // ── Classification ────────────────────────────────────────────────────────

    #[inline]
    pub fn is_finite(&self) -> bool {
        self.x_axis.is_finite() && self.y_axis.is_finite()
    }

    #[inline]
    pub fn is_nan(&self) -> bool {
        self.x_axis.is_nan() || self.y_axis.is_nan()
    }

    #[inline]
    pub fn abs_diff_eq(&self, rhs: Self, max_abs_diff: f32) -> bool {
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
    #[inline] fn add(self, rhs: Self) -> Self { self.add_mat2(&rhs) }
}
impl AddAssign for Mat2 {
    #[inline] fn add_assign(&mut self, rhs: Self) { *self = self.add_mat2(&rhs); }
}
impl Sub for Mat2 {
    type Output = Self;
    #[inline] fn sub(self, rhs: Self) -> Self { self.sub_mat2(&rhs) }
}
impl SubAssign for Mat2 {
    #[inline] fn sub_assign(&mut self, rhs: Self) { *self = self.sub_mat2(&rhs); }
}
impl Neg for Mat2 {
    type Output = Self;
    #[inline] fn neg(self) -> Self { Self(unsafe { vnegq_f32(self.0) }) }
}
impl Mul for Mat2 {
    type Output = Self;
    #[inline] fn mul(self, rhs: Self) -> Self { self.mul_mat2(&rhs) }
}
impl MulAssign for Mat2 {
    #[inline] fn mul_assign(&mut self, rhs: Self) { *self = self.mul_mat2(&rhs); }
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
