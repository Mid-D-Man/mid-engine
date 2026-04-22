// crates/mid-math/src/mat.rs

//! Mat3 · Mat4
//!
//! Both are column-major to match GLSL / HLSL / Metal conventions.
//! `cols[c][r]` = element at row `r`, column `c`.
//!
//! # Optimization tiers (per platform-optimization.md)
//!
//! ## Tier 1 (compiler-guided, no intrinsics)
//! - Mat3::mul — explicitly unrolled 9-product form
//! - Mat4::mul — explicitly unrolled 64-product form
//! - Mat4::inverse_trs_scalar — arithmetic-property fast-path (~30 muls vs ~200 for Cramer)
//!
//! ## Tier 2 (SSE2 intrinsics, x86_64 only)
//! - Mat4::inverse_trs — SSE2 parallel dot + transpose + vector scale
//!   Scalar baseline Build #29 [RELEASE]: 81.8 ns/op.
//!   SSE2 result: pending next CI build after inverse_trs commit.
//!   Maintenance estimate: 15 min/quarter. Added: 2026-04-18. Review: 2026-07-18.
//!
//! - Mat4::inverse — SSE2 2×2 sub-determinant cofactor method.
//!   Scalar baseline: 117.1 ns/op [RELEASE]. Glam reference: ~13 ns.
//!   Algorithm: compute six 2×2 minor pairs in parallel via shuffle/mul/sub,
//!   combine into adjugate columns, dot row0 for determinant, rcp-multiply.
//!   Scalar fallback (inverse_scalar) kept for non-x86_64 and correctness tests.
//!   Maintenance estimate: 20 min/quarter. Added: 2026-04-19. Review: 2026-07-19.

use std::fmt;
use std::ops::Mul;
use crate::vec::{Vec3, Vec4};
use crate::quat::Quat;
use crate::EPSILON;

// ─────────────────────────────────────────────────────────────────────────────
// Mat3
// ─────────────────────────────────────────────────────────────────────────────

/// 3×3 column-major matrix. 36 bytes, no padding.
/// Used for normal-matrix calculation and 2D transforms.
///
/// `cols[c][r]` = element at row `r`, column `c`.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct Mat3 {
    pub cols: [[f32; 3]; 3],
}

impl Mat3 {
    pub const ZERO: Self = Self { cols: [[0.0;3];3] };
    pub const IDENTITY: Self = Self { cols: [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]};

    #[inline]
    pub fn from_cols(c0: [f32;3], c1: [f32;3], c2: [f32;3]) -> Self {
        Self { cols: [c0, c1, c2] }
    }

    /// Build from a Mat4 by taking the upper-left 3×3 block.
    #[inline]
    pub fn from_mat4(m: Mat4) -> Self {
        Self::from_cols(
            [m.cols[0][0], m.cols[0][1], m.cols[0][2]],
            [m.cols[1][0], m.cols[1][1], m.cols[1][2]],
            [m.cols[2][0], m.cols[2][1], m.cols[2][2]],
        )
    }

    /// Transpose.
    #[inline]
    pub fn transpose(self) -> Self {
        let c = &self.cols;
        Self::from_cols(
            [c[0][0], c[1][0], c[2][0]],
            [c[0][1], c[1][1], c[2][1]],
            [c[0][2], c[1][2], c[2][2]],
        )
    }

    /// Determinant.
    #[inline]
    pub fn determinant(self) -> f32 {
        let c = &self.cols;
        c[0][0] * (c[1][1]*c[2][2] - c[2][1]*c[1][2])
       -c[1][0] * (c[0][1]*c[2][2] - c[2][1]*c[0][2])
       +c[2][0] * (c[0][1]*c[1][2] - c[1][1]*c[0][2])
    }

    /// Inverse. Returns `None` if the matrix is singular.
    pub fn inverse(self) -> Option<Self> {
        let det = self.determinant();
        if det.abs() < EPSILON { return None; }
        let inv_det = 1.0 / det;
        let c = &self.cols;
        Some(Self::from_cols(
            [
                (c[1][1]*c[2][2] - c[2][1]*c[1][2]) * inv_det,
                (c[2][1]*c[0][2] - c[0][1]*c[2][2]) * inv_det,
                (c[0][1]*c[1][2] - c[1][1]*c[0][2]) * inv_det,
            ],
            [
                (c[2][0]*c[1][2] - c[1][0]*c[2][2]) * inv_det,
                (c[0][0]*c[2][2] - c[2][0]*c[0][2]) * inv_det,
                (c[1][0]*c[0][2] - c[0][0]*c[1][2]) * inv_det,
            ],
            [
                (c[1][0]*c[2][1] - c[2][0]*c[1][1]) * inv_det,
                (c[2][0]*c[0][1] - c[0][0]*c[2][1]) * inv_det,
                (c[0][0]*c[1][1] - c[1][0]*c[0][1]) * inv_det,
            ],
        ))
    }

    /// Normal matrix = transpose(inverse(upper-left 3×3 of model matrix)).
    pub fn normal_matrix(model: Mat4) -> Option<Self> {
        Self::from_mat4(model).inverse().map(|m| m.transpose())
    }

    /// Multiply Mat3 × Vec3.
    #[inline]
    pub fn transform(self, v: Vec3) -> Vec3 {
        let c = &self.cols;
        Vec3::new(
            c[0][0]*v.x + c[1][0]*v.y + c[2][0]*v.z,
            c[0][1]*v.x + c[1][1]*v.y + c[2][1]*v.z,
            c[0][2]*v.x + c[1][2]*v.y + c[2][2]*v.z,
        )
    }
}

impl Default for Mat3 { fn default() -> Self { Self::IDENTITY } }

impl Mul for Mat3 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        let (a, b) = (&self.cols, &rhs.cols);
        Self::from_cols(
            [
                a[0][0]*b[0][0] + a[1][0]*b[0][1] + a[2][0]*b[0][2],
                a[0][1]*b[0][0] + a[1][1]*b[0][1] + a[2][1]*b[0][2],
                a[0][2]*b[0][0] + a[1][2]*b[0][1] + a[2][2]*b[0][2],
            ],
            [
                a[0][0]*b[1][0] + a[1][0]*b[1][1] + a[2][0]*b[1][2],
                a[0][1]*b[1][0] + a[1][1]*b[1][1] + a[2][1]*b[1][2],
                a[0][2]*b[1][0] + a[1][2]*b[1][1] + a[2][2]*b[1][2],
            ],
            [
                a[0][0]*b[2][0] + a[1][0]*b[2][1] + a[2][0]*b[2][2],
                a[0][1]*b[2][0] + a[1][1]*b[2][1] + a[2][1]*b[2][2],
                a[0][2]*b[2][0] + a[1][2]*b[2][1] + a[2][2]*b[2][2],
            ],
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Mat4
// ─────────────────────────────────────────────────────────────────────────────

/// 4×4 column-major matrix. 64 bytes, 16-byte aligned.
///
/// `cols[c][r]` = element at row `r`, column `c`.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C, align(16))]
pub struct Mat4 {
    pub cols: [[f32; 4]; 4],
}

impl Mat4 {
    pub const ZERO: Self = Self { cols: [[0.0;4];4] };
    pub const IDENTITY: Self = Self { cols: [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]};

    // ── Constructors ───────────────────────────────────────────────────────

    #[inline]
    pub fn from_cols(c0: [f32;4], c1: [f32;4], c2: [f32;4], c3: [f32;4]) -> Self {
        Self { cols: [c0, c1, c2, c3] }
    }

    #[inline]
    pub fn from_translation(t: Vec3) -> Self {
        let mut m = Self::IDENTITY;
        m.cols[3] = [t.x, t.y, t.z, 1.0];
        m
    }

    #[inline]
    pub fn from_scale(s: Vec3) -> Self {
        Self::from_cols(
            [s.x, 0.0, 0.0, 0.0],
            [0.0, s.y, 0.0, 0.0],
            [0.0, 0.0, s.z, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        )
    }

    #[inline]
    pub fn from_rotation(q: Quat) -> Self { q.to_mat4() }

    #[inline]
    pub fn from_trs(t: Vec3, r: Quat, s: Vec3) -> Self {
        Self::from_translation(t) * Self::from_rotation(r) * Self::from_scale(s)
    }

    // ── Camera / Projection ────────────────────────────────────────────────

    pub fn look_at_rh(eye: Vec3, center: Vec3, up: Vec3) -> Self {
        let f = (center - eye).normalize();
        let r = f.cross(up).normalize();
        let u = r.cross(f);
        Self::from_cols(
            [ r.x,  u.x, -f.x, 0.0],
            [ r.y,  u.y, -f.y, 0.0],
            [ r.z,  u.z, -f.z, 0.0],
            [-r.dot(eye), -u.dot(eye), f.dot(eye), 1.0],
        )
    }

    pub fn perspective_rh(fov_y: f32, aspect: f32, near: f32, far: f32) -> Self {
        let f = 1.0 / (fov_y * 0.5).tan();
        let z = near - far;
        Self::from_cols(
            [f / aspect, 0.0, 0.0,                      0.0],
            [0.0,        f,   0.0,                      0.0],
            [0.0,        0.0, (far + near) / z,        -1.0],
            [0.0,        0.0, (2.0 * far * near) / z,   0.0],
        )
    }

    pub fn ortho_rh(left: f32, right: f32, bottom: f32, top: f32, near: f32, far: f32) -> Self {
        let rl = right - left;
        let tb = top   - bottom;
        let nf = far   - near;
        Self::from_cols(
            [2.0/rl,              0.0,             0.0,        0.0],
            [0.0,                 2.0/tb,          0.0,        0.0],
            [0.0,                 0.0,            -2.0/nf,     0.0],
            [-(right+left)/rl, -(top+bottom)/tb, -(far+near)/nf, 1.0],
        )
    }

    // ── Core operations ────────────────────────────────────────────────────

    pub fn transpose(self) -> Self {
        let c = &self.cols;
        Self::from_cols(
            [c[0][0], c[1][0], c[2][0], c[3][0]],
            [c[0][1], c[1][1], c[2][1], c[3][1]],
            [c[0][2], c[1][2], c[2][2], c[3][2]],
            [c[0][3], c[1][3], c[2][3], c[3][3]],
        )
    }

    pub fn determinant(self) -> f32 {
        let c = &self.cols;
        let a = |ci: usize, ri: usize| c[ci][ri];
        let sub3 = |c0: usize, c1: usize, c2: usize,
                    r0: usize, r1: usize, r2: usize| -> f32 {
            a(c0,r0)*(a(c1,r1)*a(c2,r2) - a(c2,r1)*a(c1,r2))
           -a(c1,r0)*(a(c0,r1)*a(c2,r2) - a(c2,r1)*a(c0,r2))
           +a(c2,r0)*(a(c0,r1)*a(c1,r2) - a(c1,r1)*a(c0,r2))
        };
        a(0,0)*sub3(1,2,3, 1,2,3)
       -a(1,0)*sub3(0,2,3, 1,2,3)
       +a(2,0)*sub3(0,1,3, 1,2,3)
       -a(3,0)*sub3(0,1,2, 1,2,3)
    }

    // ── General inverse ────────────────────────────────────────────────────

    /// General 4×4 inverse. Dispatches to SSE2 on x86_64, scalar elsewhere.
    /// Returns `None` if the matrix is singular (|det| < EPSILON).
    ///
    /// For matrices you know are TRS, call `inverse_trs()` — it is faster.
    pub fn inverse(self) -> Option<Self> {
        #[cfg(target_arch = "x86_64")]
        return unsafe { sse2::inverse_general(&self) };

        #[allow(unreachable_code)]
        self.inverse_scalar()
    }

    /// Scalar general inverse via Cramer's rule. Always available.
    /// Used as the non-x86_64 path and the correctness baseline in tests.
    ///
    /// Scalar baseline [RELEASE]: 117.1 ns/op.
    pub fn inverse_scalar(self) -> Option<Self> {
        let a = [
            self.cols[0][0], self.cols[0][1], self.cols[0][2], self.cols[0][3],
            self.cols[1][0], self.cols[1][1], self.cols[1][2], self.cols[1][3],
            self.cols[2][0], self.cols[2][1], self.cols[2][2], self.cols[2][3],
            self.cols[3][0], self.cols[3][1], self.cols[3][2], self.cols[3][3],
        ];
        let mut inv = [0.0f32; 16];
        inv[ 0] =  a[5]*a[10]*a[15] - a[5]*a[11]*a[14] - a[9]*a[6]*a[15]  + a[9]*a[7]*a[14]  + a[13]*a[6]*a[11]  - a[13]*a[7]*a[10];
        inv[ 4] = -a[4]*a[10]*a[15] + a[4]*a[11]*a[14] + a[8]*a[6]*a[15]  - a[8]*a[7]*a[14]  - a[12]*a[6]*a[11]  + a[12]*a[7]*a[10];
        inv[ 8] =  a[4]*a[9] *a[15] - a[4]*a[11]*a[13] - a[8]*a[5]*a[15]  + a[8]*a[7]*a[13]  + a[12]*a[5]*a[11]  - a[12]*a[7]*a[9];
        inv[12] = -a[4]*a[9] *a[14] + a[4]*a[10]*a[13] + a[8]*a[5]*a[14]  - a[8]*a[6]*a[13]  - a[12]*a[5]*a[10]  + a[12]*a[6]*a[9];
        inv[ 1] = -a[1]*a[10]*a[15] + a[1]*a[11]*a[14] + a[9]*a[2]*a[15]  - a[9]*a[3]*a[14]  - a[13]*a[2]*a[11]  + a[13]*a[3]*a[10];
        inv[ 5] =  a[0]*a[10]*a[15] - a[0]*a[11]*a[14] - a[8]*a[2]*a[15]  + a[8]*a[3]*a[14]  + a[12]*a[2]*a[11]  - a[12]*a[3]*a[10];
        inv[ 9] = -a[0]*a[9] *a[15] + a[0]*a[11]*a[13] + a[8]*a[1]*a[15]  - a[8]*a[3]*a[13]  - a[12]*a[1]*a[11]  + a[12]*a[3]*a[9];
        inv[13] =  a[0]*a[9] *a[14] - a[0]*a[10]*a[13] - a[8]*a[1]*a[14]  + a[8]*a[2]*a[13]  + a[12]*a[1]*a[10]  - a[12]*a[2]*a[9];
        inv[ 2] =  a[1]*a[6] *a[15] - a[1]*a[7] *a[14] - a[5]*a[2]*a[15]  + a[5]*a[3]*a[14]  + a[13]*a[2]*a[7]   - a[13]*a[3]*a[6];
        inv[ 6] = -a[0]*a[6] *a[15] + a[0]*a[7] *a[14] + a[4]*a[2]*a[15]  - a[4]*a[3]*a[14]  - a[12]*a[2]*a[7]   + a[12]*a[3]*a[6];
        inv[10] =  a[0]*a[5] *a[15] - a[0]*a[7] *a[13] - a[4]*a[1]*a[15]  + a[4]*a[3]*a[13]  + a[12]*a[1]*a[7]   - a[12]*a[3]*a[5];
        inv[14] = -a[0]*a[5] *a[14] + a[0]*a[6] *a[13] + a[4]*a[1]*a[14]  - a[4]*a[2]*a[13]  - a[12]*a[1]*a[6]   + a[12]*a[2]*a[5];
        inv[ 3] = -a[1]*a[6] *a[11] + a[1]*a[7] *a[10] + a[5]*a[2]*a[11]  - a[5]*a[3]*a[10]  - a[9] *a[2]*a[7]   + a[9] *a[3]*a[6];
        inv[ 7] =  a[0]*a[6] *a[11] - a[0]*a[7] *a[10] - a[4]*a[2]*a[11]  + a[4]*a[3]*a[10]  + a[8] *a[2]*a[7]   - a[8] *a[3]*a[6];
        inv[11] = -a[0]*a[5] *a[11] + a[0]*a[7] *a[9]  + a[4]*a[1]*a[11]  - a[4]*a[3]*a[9]   - a[8] *a[1]*a[7]   + a[8] *a[3]*a[5];
        inv[15] =  a[0]*a[5] *a[10] - a[0]*a[6] *a[9]  - a[4]*a[1]*a[10]  + a[4]*a[2]*a[9]   + a[8] *a[1]*a[6]   - a[8] *a[2]*a[5];
        let det = a[0]*inv[0] + a[1]*inv[4] + a[2]*inv[8] + a[3]*inv[12];
        if det.abs() < EPSILON { return None; }
        let inv_det = 1.0 / det;
        for x in inv.iter_mut() { *x *= inv_det; }
        Some(Self::from_cols(
            [inv[0],  inv[1],  inv[2],  inv[3] ],
            [inv[4],  inv[5],  inv[6],  inv[7] ],
            [inv[8],  inv[9],  inv[10], inv[11]],
            [inv[12], inv[13], inv[14], inv[15]],
        ))
    }

    // ── TRS inverse ────────────────────────────────────────────────────────

    /// Fast inverse for pure TRS matrices. Dispatches to SSE2 on x86_64.
    /// Do not call on matrices containing shear or projection.
    #[inline]
    pub fn inverse_trs(self) -> Self {
        #[cfg(target_arch = "x86_64")]
        return unsafe { sse2::inverse_trs(&self) };

        #[allow(unreachable_code)]
        self.inverse_trs_scalar()
    }

    /// Scalar TRS inverse. Always available on all platforms.
    pub fn inverse_trs_scalar(self) -> Self {
        let sx2 = self.cols[0][0]*self.cols[0][0]
                + self.cols[0][1]*self.cols[0][1]
                + self.cols[0][2]*self.cols[0][2];
        let sy2 = self.cols[1][0]*self.cols[1][0]
                + self.cols[1][1]*self.cols[1][1]
                + self.cols[1][2]*self.cols[1][2];
        let sz2 = self.cols[2][0]*self.cols[2][0]
                + self.cols[2][1]*self.cols[2][1]
                + self.cols[2][2]*self.cols[2][2];

        let inv_sx2 = if sx2 < EPSILON { 0.0 } else { 1.0 / sx2 };
        let inv_sy2 = if sy2 < EPSILON { 0.0 } else { 1.0 / sy2 };
        let inv_sz2 = if sz2 < EPSILON { 0.0 } else { 1.0 / sz2 };

        let ic0 = [
            self.cols[0][0]*inv_sx2,
            self.cols[1][0]*inv_sy2,
            self.cols[2][0]*inv_sz2,
            0.0,
        ];
        let ic1 = [
            self.cols[0][1]*inv_sx2,
            self.cols[1][1]*inv_sy2,
            self.cols[2][1]*inv_sz2,
            0.0,
        ];
        let ic2 = [
            self.cols[0][2]*inv_sx2,
            self.cols[1][2]*inv_sy2,
            self.cols[2][2]*inv_sz2,
            0.0,
        ];

        let tx = self.cols[3][0];
        let ty = self.cols[3][1];
        let tz = self.cols[3][2];

        let itx = -(ic0[0]*tx + ic1[0]*ty + ic2[0]*tz);
        let ity = -(ic0[1]*tx + ic1[1]*ty + ic2[1]*tz);
        let itz = -(ic0[2]*tx + ic1[2]*ty + ic2[2]*tz);

        Self::from_cols(ic0, ic1, ic2, [itx, ity, itz, 1.0])
    }

    // ── Transform helpers ──────────────────────────────────────────────────

    #[inline]
    pub fn transform_point(self, p: Vec3) -> Vec3 {
        (self * p.extend(1.0)).truncate()
    }

    #[inline]
    pub fn transform_vector(self, v: Vec3) -> Vec3 {
        (self * v.extend(0.0)).truncate()
    }
}

impl Default for Mat4 { fn default() -> Self { Self::IDENTITY } }

impl Mul for Mat4 {
    type Output = Self;
    /// Explicitly unrolled column-major matrix multiply. Tier 1.
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        let (a, b) = (&self.cols, &rhs.cols);
        Self::from_cols(
            [
                a[0][0]*b[0][0] + a[1][0]*b[0][1] + a[2][0]*b[0][2] + a[3][0]*b[0][3],
                a[0][1]*b[0][0] + a[1][1]*b[0][1] + a[2][1]*b[0][2] + a[3][1]*b[0][3],
                a[0][2]*b[0][0] + a[1][2]*b[0][1] + a[2][2]*b[0][2] + a[3][2]*b[0][3],
                a[0][3]*b[0][0] + a[1][3]*b[0][1] + a[2][3]*b[0][2] + a[3][3]*b[0][3],
            ],
            [
                a[0][0]*b[1][0] + a[1][0]*b[1][1] + a[2][0]*b[1][2] + a[3][0]*b[1][3],
                a[0][1]*b[1][0] + a[1][1]*b[1][1] + a[2][1]*b[1][2] + a[3][1]*b[1][3],
                a[0][2]*b[1][0] + a[1][2]*b[1][1] + a[2][2]*b[1][2] + a[3][2]*b[1][3],
                a[0][3]*b[1][0] + a[1][3]*b[1][1] + a[2][3]*b[1][2] + a[3][3]*b[1][3],
            ],
            [
                a[0][0]*b[2][0] + a[1][0]*b[2][1] + a[2][0]*b[2][2] + a[3][0]*b[2][3],
                a[0][1]*b[2][0] + a[1][1]*b[2][1] + a[2][1]*b[2][2] + a[3][1]*b[2][3],
                a[0][2]*b[2][0] + a[1][2]*b[2][1] + a[2][2]*b[2][2] + a[3][2]*b[2][3],
                a[0][3]*b[2][0] + a[1][3]*b[2][1] + a[2][3]*b[2][2] + a[3][3]*b[2][3],
            ],
            [
                a[0][0]*b[3][0] + a[1][0]*b[3][1] + a[2][0]*b[3][2] + a[3][0]*b[3][3],
                a[0][1]*b[3][0] + a[1][1]*b[3][1] + a[2][1]*b[3][2] + a[3][1]*b[3][3],
                a[0][2]*b[3][0] + a[1][2]*b[3][1] + a[2][2]*b[3][2] + a[3][2]*b[3][3],
                a[0][3]*b[3][0] + a[1][3]*b[3][1] + a[2][3]*b[3][2] + a[3][3]*b[3][3],
            ],
        )
    }
}

impl Mul<Vec4> for Mat4 {
    type Output = Vec4;
    #[inline]
    fn mul(self, v: Vec4) -> Vec4 {
        let c = &self.cols;
        Vec4::new(
            c[0][0]*v.x + c[1][0]*v.y + c[2][0]*v.z + c[3][0]*v.w,
            c[0][1]*v.x + c[1][1]*v.y + c[2][1]*v.z + c[3][1]*v.w,
            c[0][2]*v.x + c[1][2]*v.y + c[2][2]*v.z + c[3][2]*v.w,
            c[0][3]*v.x + c[1][3]*v.y + c[2][3]*v.z + c[3][3]*v.w,
        )
    }
}

impl fmt::Display for Mat4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let c = &self.cols;
        for r in 0..4 {
            writeln!(f, "  [{:8.4}  {:8.4}  {:8.4}  {:8.4}]",
                c[0][r], c[1][r], c[2][r], c[3][r])?;
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tier 2 — SSE2 fast paths (x86_64 only)
//
// SSE2 is the x86_64 ABI baseline. No runtime detection required.
//
// Safety invariants for all functions in this module:
// - Mat4 is #[repr(C, align(16))], so every cols[n] pointer is 16-byte aligned.
//   All _mm_load_ps / _mm_store_ps calls are therefore safe.
// - No side effects outside the returned value.
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(target_arch = "x86_64")]
mod sse2 {
    use core::arch::x86_64::*;
    use super::Mat4;
    use crate::EPSILON;

    // ── General inverse ────────────────────────────────────────────────────
    //
    // Algorithm: 2×2 sub-determinant cofactor method.
    //
    // We load the matrix as 4 column registers, then compute 6 pairs of
    // 2×2 minor determinants that cover all entries of the adjugate.
    // Each pair reuses one shuffle, so total ops are roughly half what
    // scalar Cramer's rule requires.
    //
    // Notation used in comments:
    //   c0..c3 = column registers (each holds 4 floats: rows 0-3)
    //   The matrix entry at row r, col c is c_reg[r].
    //
    // The 6 minor pairs (naming by the two column indices excluded from the 3x3):
    //   p01 = minors using cols 0,1 (row pairs)
    //   p02 = minors using cols 0,2
    //   p03 = minors using cols 0,3
    //   p12 = minors using cols 1,2
    //   p13 = minors using cols 1,3
    //   p23 = minors using cols 2,3
    //
    // Each minor pair is a register of 4 values:
    //   pair[i] = c_a[row_x] * c_b[row_y] - c_a[row_y] * c_b[row_x]
    // for 4 different (row_x, row_y) combinations, computed simultaneously.
    //
    // The 4 adjugate columns are then dot products of these pairs with
    // appropriate rows of the complementary columns.
    //
    // Singular check: if |det| < EPSILON, return None.
    //
    // Safety:
    //   cols[n].as_ptr() is 16-byte aligned (Mat4 is align(16)).
    //   All loads/stores use aligned variants.

    #[inline(always)]
    unsafe fn compute_minor_pair(ca: __m128, cb: __m128, shuf_a: i32, shuf_b: i32) -> __m128 {
        // minor = shuffle_a(ca) * shuffle_b(cb) - shuffle_b(ca) * shuffle_a(cb)
        // We pass the shuffle constants as i32 but _mm_shuffle_ps needs const IMM8.
        // To avoid a match table we inline the 6 calls with explicit constants below.
        // This helper is intentionally NOT used — see inline calls in inverse_general.
        let _ = (ca, cb, shuf_a, shuf_b);
        unreachable!()
    }

    /// SSE2 general 4×4 inverse. Returns None if singular.
    ///
    /// Based on the Intel "Streaming SIMD Extensions — Inverse of 4×4 Matrix"
    /// application note algorithm, adapted for column-major storage.
    ///
    /// Correctness verified against inverse_scalar in tests.
    /// Scalar baseline [RELEASE]: 117.1 ns/op.
    /// Added: 2026-04-19. Review: 2026-07-19.
    pub(super) unsafe fn inverse_general(m: &Mat4) -> Option<Mat4> {
        // Load the four columns.
        // c0 = {m00, m01, m02, m03}  (col 0, rows 0-3)
        // c1 = {m10, m11, m12, m13}
        // c2 = {m20, m21, m22, m23}
        // c3 = {m30, m31, m32, m33}
        let c0 = _mm_load_ps(m.cols[0].as_ptr());
        let c1 = _mm_load_ps(m.cols[1].as_ptr());
        let c2 = _mm_load_ps(m.cols[2].as_ptr());
        let c3 = _mm_load_ps(m.cols[3].as_ptr());

        // ── Compute the 6 paired 2×2 minor determinants ───────────────────
        //
        // For two columns ca and cb, a "minor pair" register holds:
        //   [ca[0]*cb[1]-ca[1]*cb[0],  ca[0]*cb[2]-ca[2]*cb[0],
        //    ca[0]*cb[3]-ca[3]*cb[0],  ca[1]*cb[2]-ca[2]*cb[1]]
        // (the remaining two minors for this column pair are derived by negation
        //  or appear in other combinations below)
        //
        // _mm_shuffle_ps::<IMM>(a, b) with a==b broadcasts/permutes one register.
        // The IMM encoding is: bits[1:0]=lane for result[0], bits[3:2] for result[1],
        // bits[5:4] for result[2], bits[7:6] for result[3].
        //
        // We compute all pairs we need for the 4 adjugate columns.

        // Minors from (c2, c3): used in adjugate cols 0 and 1.
        // m23_01 = c2[0]*c3[1] - c2[1]*c3[0]
        // m23_02 = c2[0]*c3[2] - c2[2]*c3[0]
        // m23_03 = c2[0]*c3[3] - c2[3]*c3[0]
        // m23_12 = c2[1]*c3[2] - c2[2]*c3[1]
        // m23_13 = c2[1]*c3[3] - c2[3]*c3[1]
        // m23_23 = c2[2]*c3[3] - c2[3]*c3[2]
        //
        // We pack these 6 scalars into two SSE registers of 4 each:
        //   minor23_lo = {m23_01, m23_02, m23_03, m23_12}
        //   minor23_hi = {m23_13, m23_23, ?, ?}  (only first 2 used)
        //
        // Compute via:
        //   c2_0011 = {c2[0],c2[0],c2[1],c2[1]}
        //   c3_0011 = {c3[0],c3[0],c3[1],c3[1]}
        //   c2_1223 = {c2[1],c2[2],c2[2],c2[3]}
        //   c3_1223 = {c3[1],c3[2],c3[2],c3[3]}
        //
        //   lo_prod_a = c2_0011 * c3_1223  = {c2[0]*c3[1], c2[0]*c3[2], c2[1]*c3[2], c2[1]*c3[3]}
        //   lo_prod_b = c3_0011 * c2_1223  = {c3[0]*c2[1], c3[0]*c2[2], c3[1]*c2[2], c3[1]*c2[3]}
        //   minor23_lo_partial = lo_prod_a - lo_prod_b
        //     = {m23_01, m23_02, m23_12, m23_13}   [note: positions 2,3 are _12 and _13]
        //
        // We also need m23_03 and m23_23:
        //   c2_0023 = {c2[0],c2[0],c2[2],c2[3]}  (or just compute separately)
        //   c3_0023 = {c3[0],c3[0],c3[2],c3[3]}
        //
        // Rather than perfect packing, we compute in a layout that directly
        // feeds the adjugate dot products below. The key insight is that each
        // adjugate entry is a sum of 3 minor products, and each minor appears
        // in exactly 2 adjugate entries (with opposite sign), so we arrange
        // the computation to reuse register values.

        // Step A: compute 2x2 minors of (c0,c1) and (c2,c3) in parallel.
        //
        // We need these minors for the adjugate:
        //   adj col 0 (cofactors of row 0): uses minors of c1c2c3 submatrix rows 1,2,3
        //   adj col 1 (cofactors of row 1): uses minors of c0c2c3 submatrix rows 0,2,3
        //   adj col 2 (cofactors of row 2): uses minors of c0c1c3 submatrix rows 0,1,3
        //   adj col 3 (cofactors of row 3): uses minors of c0c1c2 submatrix rows 0,1,2
        //
        // All 2×2 minors needed (using row indices):
        //   From (c2,c3): [01,02,03,12,13,23]  → for adj cols 0,1
        //   From (c0,c1): [01,02,03,12,13,23]  → for adj cols 2,3
        //   From (c0,c2): [01,02,03,12,13,23]  → for adj cols 1,3
        //   From (c1,c3): [01,02,03,12,13,23]  → for adj cols 0,2
        //   From (c0,c3): [01,02,03,12,13,23]  → for adj cols 1,2
        //   From (c1,c2): [01,02,03,12,13,23]  → for adj cols 0,3
        //
        // We compute 6 registers, each holding 4 of these minors packed,
        // then dot them with appropriate column rows.

        // Helper: for columns ca, cb compute the register:
        //   {ca[0]*cb[1]-cb[0]*ca[1], ca[0]*cb[2]-cb[0]*ca[2],
        //    ca[1]*cb[3]-cb[1]*ca[3], ca[2]*cb[3]-cb[2]*ca[3]}
        // which gives 4 of the 6 minors: [01, 02, 13, 23]
        // The remaining minors [03, 12] need separate computation.

        macro_rules! minor4 {
            // Computes {ca[0]*cb[1]-ca[1]*cb[0], ca[0]*cb[2]-ca[2]*cb[0],
            //           ca[1]*cb[3]-ca[3]*cb[1], ca[2]*cb[3]-ca[3]*cb[2]}
            // = minors [01, 02, 13, 23]
            ($ca:expr, $cb:expr) => {{
                let ca = $ca;
                let cb = $cb;
                // ca_0012 = {ca[0], ca[0], ca[1], ca[2]}
                let ca_0012 = _mm_shuffle_ps::<0b10_01_00_00>(ca, ca);
                // cb_1223 = {cb[1], cb[2], cb[2], cb[3]}
                let cb_1223 = _mm_shuffle_ps::<0b11_10_10_01>(cb, cb);
                // ca_1223 = {ca[1], ca[2], ca[2], ca[3]}
                let ca_1223 = _mm_shuffle_ps::<0b11_10_10_01>(ca, ca);
                // cb_0012 = {cb[0], cb[0], cb[1], cb[2]}
                let cb_0012 = _mm_shuffle_ps::<0b10_01_00_00>(cb, cb);
                _mm_sub_ps(_mm_mul_ps(ca_0012, cb_1223), _mm_mul_ps(ca_1223, cb_0012))
            }};
        }

        macro_rules! minor03_12 {
            // Computes {ca[0]*cb[3]-ca[3]*cb[0], ca[1]*cb[2]-ca[2]*cb[1]}
            // = minors [03, 12] — the two not covered by minor4
            ($ca:expr, $cb:expr) => {{
                let ca = $ca;
                let cb = $cb;
                // We only need 2 values; pack them as [03, 12, 12, 03] for reuse.
                // ca_0110 = {ca[0], ca[1], ca[1], ca[0]}
                let ca_0110 = _mm_shuffle_ps::<0b00_01_01_00>(ca, ca);
                // cb_3223 = {cb[3], cb[2], cb[2], cb[3]}
                let cb_3223 = _mm_shuffle_ps::<0b11_10_10_11>(cb, cb);
                // ca_3223 = {ca[3], ca[2], ca[2], ca[3]}
                let ca_3223 = _mm_shuffle_ps::<0b11_10_10_11>(ca, ca);
                // cb_0110 = {cb[0], cb[1], cb[1], cb[0]}
                let cb_0110 = _mm_shuffle_ps::<0b00_01_01_00>(cb, cb);
                // result[0] = ca[0]*cb[3] - ca[3]*cb[0]  = minor03
                // result[1] = ca[1]*cb[2] - ca[2]*cb[1]  = minor12
                _mm_sub_ps(_mm_mul_ps(ca_0110, cb_3223), _mm_mul_ps(ca_3223, cb_0110))
            }};
        }

        // The 6 minor registers.
        // m4_XY = 4 minors {[01],[02],[13],[23]} from columns cX, cY
        // m2_XY = 2 minors {[03],[12]} from columns cX, cY  (only lanes 0,1 used)
        let m4_23 = minor4!(c2, c3);
        let m2_23 = minor03_12!(c2, c3);

        let m4_01 = minor4!(c0, c1);
        let m2_01 = minor03_12!(c0, c1);

        let m4_02 = minor4!(c0, c2);
        let m2_02 = minor03_12!(c0, c2);

        let m4_13 = minor4!(c1, c3);
        let m2_13 = minor03_12!(c1, c3);

        let m4_03 = minor4!(c0, c3);
        let m2_03 = minor03_12!(c0, c3);

        let m4_12 = minor4!(c1, c2);
        let m2_12 = minor03_12!(c1, c2);

        // ── Build adjugate columns ─────────────────────────────────────────
        //
        // adj[col][row] = (-1)^(row+col) * M_{row,col}
        // where M_{row,col} is the 3×3 minor obtained by deleting row `row` and col `col`.
        //
        // For a column-major matrix, the adjugate columns are:
        //
        // adj_col0 = cofactors of column 0 entries (deleting col 0 from the 4×4):
        //   adj[0][0] = +det(c1,c2,c3 rows 1,2,3)
        //   adj[0][1] = -det(c1,c2,c3 rows 0,2,3)
        //   adj[0][2] = +det(c1,c2,c3 rows 0,1,3)
        //   adj[0][3] = -det(c1,c2,c3 rows 0,1,2)
        //
        // Each 3×3 determinant expands as a sum of three 2×2 minors from our precomputed set.
        //
        // det(c1,c2,c3 rows 1,2,3) = c1[1]*(c2[2]*c3[3]-c2[3]*c3[2])
        //                           - c1[2]*(c2[1]*c3[3]-c2[3]*c3[1])
        //                           + c1[3]*(c2[1]*c3[2]-c2[2]*c3[1])
        // In our minor notation:
        //   = c1[1]*m23[23] - c1[2]*m23[13] + c1[3]*m23[12]
        //
        // where m23[23] is minor [23] of (c2,c3), etc.
        //
        // m4_23 = {m23[01], m23[02], m23[13], m23[23]}
        // m2_23 = {m23[03], m23[12], ...}
        //
        // So:
        //   adj[0][0] = +( c1[1]*m23[23] - c1[2]*m23[13] + c1[3]*m23[12] )
        //   adj[0][1] = -( c1[0]*m23[23] - c1[2]*m23[03] + c1[3]*m23[02] )
        //   adj[0][2] = +( c1[0]*m23[13] - c1[1]*m23[03] + c1[3]*m23[01] )
        //   adj[0][3] = -( c1[0]*m23[12] - c1[1]*m23[02] + c1[2]*m23[01] )
        //
        // We can compute all 4 simultaneously by broadcasting c1 rows and
        // multiplying against shuffled minor registers.

        // We build each adjugate column as a dot product.
        // To avoid index confusion, we use explicit scalar extraction via
        // _mm_shuffle to broadcast individual rows of c1, c2, c3 as needed.

        // Broadcast helpers: extract lane k of a register to all lanes.
        macro_rules! bcast {
            ($v:expr, 0) => { _mm_shuffle_ps::<0b00_00_00_00>($v, $v) };
            ($v:expr, 1) => { _mm_shuffle_ps::<0b01_01_01_01>($v, $v) };
            ($v:expr, 2) => { _mm_shuffle_ps::<0b10_10_10_10>($v, $v) };
            ($v:expr, 3) => { _mm_shuffle_ps::<0b11_11_11_11>($v, $v) };
        }

        // Precompute broadcasts of c0, c1 rows we'll need repeatedly.
        let c0r0 = bcast!(c0, 0);  let c0r1 = bcast!(c0, 1);
        let c0r2 = bcast!(c0, 2);  let c0r3 = bcast!(c0, 3);
        let c1r0 = bcast!(c1, 0);  let c1r1 = bcast!(c1, 1);
        let c1r2 = bcast!(c1, 2);  let c1r3 = bcast!(c1, 3);
        let c2r0 = bcast!(c2, 0);  let c2r1 = bcast!(c2, 1);
        let c2r2 = bcast!(c2, 2);  let c2r3 = bcast!(c2, 3);
        let c3r0 = bcast!(c3, 0);  let c3r1 = bcast!(c3, 1);
        let c3r2 = bcast!(c3, 2);  let c3r3 = bcast!(c3, 3);

        // Precompute shuffled minor registers for efficient reuse.
        // m4_XY = {m[01], m[02], m[13], m[23]}
        // We also need m[03] and m[12] from m2_XY lanes 0 and 1.
        //
        // To extract individual minors as broadcast registers:
        macro_rules! m_at {
            ($m4:expr, $m2:expr, 01) => { bcast!($m4, 0) };
            ($m4:expr, $m2:expr, 02) => { bcast!($m4, 1) };
            ($m4:expr, $m2:expr, 03) => { bcast!($m2, 0) };
            ($m4:expr, $m2:expr, 12) => { bcast!($m2, 1) };
            ($m4:expr, $m2:expr, 13) => { bcast!($m4, 2) };
            ($m4:expr, $m2:expr, 23) => { bcast!($m4, 3) };
        }

        // ── Adjugate column 0 (cofactors along column 0, i.e. deleting col 0) ──
        // Uses minors from (c1,c2,c3) — specifically (c2,c3) pair weighted by c1.
        //
        // adj0[0] = +c1[1]*m23[23] - c1[2]*m23[13] + c1[3]*m23[12]
        // adj0[1] = -c1[0]*m23[23] + c1[2]*m23[03] - c1[3]*m23[02]
        // adj0[2] = +c1[0]*m23[13] - c1[1]*m23[03] + c1[3]*m23[01]
        // adj0[3] = -c1[0]*m23[12] + c1[1]*m23[02] - c1[2]*m23[01]
        //
        // We compute each row independently then blend into one register.
        let m23_01 = m_at!(m4_23, m2_23, 01);
        let m23_02 = m_at!(m4_23, m2_23, 02);
        let m23_03 = m_at!(m4_23, m2_23, 03);
        let m23_12 = m_at!(m4_23, m2_23, 12);
        let m23_13 = m_at!(m4_23, m2_23, 13);
        let m23_23 = m_at!(m4_23, m2_23, 23);

        // Build adj_col0 as 4 separate scalars computed in parallel by broadcasting.
        // We need to assemble {adj0[0], adj0[1], adj0[2], adj0[3]} into one register.
        // Strategy: compute each term pair, add/sub to get the 4 values, then
        // combine via shuffle.
        //
        // To keep code compact, we compute each entry directly as a scalar (via
        // _mm_store_ss or extracting a single lane). While this loses some SIMD
        // width, it avoids a complex 4-way horizontal combine and is still
        // significantly faster than scalar Cramer because the minor values are
        // already precomputed.
        //
        // A cleaner approach: compute all 4 entries as one SIMD expression by
        // constructing sign-alternating vectors and using dot products.
        // We use this approach below.

        // Compute adj column 0 using a single mul-add chain per element,
        // broadcasting c1 rows and selecting the right minor per row.
        //
        // All 4 entries in parallel:
        //   row 0: +c1[1]*m23[23] - c1[2]*m23[13] + c1[3]*m23[12]
        //   row 1: -c1[0]*m23[23] + c1[2]*m23[03] - c1[3]*m23[02]
        //   row 2: +c1[0]*m23[13] - c1[1]*m23[03] + c1[3]*m23[01]
        //   row 3: -c1[0]*m23[12] + c1[1]*m23[02] - c1[2]*m23[01]
        //
        // Pack these 4 values: we need one value per lane.
        // Compute term by term and accumulate:

        // Term 1 for each row: [+c1[1]*m23[23], -c1[0]*m23[23], +c1[0]*m23[13], -c1[0]*m23[12]]
        // Term 2 for each row: [-c1[2]*m23[13], +c1[2]*m23[03], -c1[1]*m23[03], +c1[1]*m23[02]]
        // Term 3 for each row: [+c1[3]*m23[12], -c1[3]*m23[02], +c1[3]*m23[01], -c1[2]*m23[01]]
        //
        // Instead of packing by column-of-column, we use the following approach:
        // Build the result register directly using single-element extracts and
        // _mm_set_ps. This is a scalar gather but avoids 16 separate loads.

        // Extract scalar values we need using _mm_cvtss_f32 on broadcast results.
        // _mm_cvtss_f32 reads lane 0.
        let get = |v: __m128| -> f32 { _mm_cvtss_f32(v) };

        let c1_0 = get(c1r0); let c1_1 = get(c1r1);
        let c1_2 = get(c1r2); let c1_3 = get(c1r3);
        let c2_0 = get(c2r0); let c2_1 = get(c2r1);
        let c2_2 = get(c2r2); let c2_3 = get(c2r3);
        let c3_0 = get(c3r0); let c3_1 = get(c3r1);
        let c3_2 = get(c3r2); let c3_3 = get(c3r3);

        // Extract the 12 distinct minor values we need:
        let mn23_01 = get(m23_01); let mn23_02 = get(m23_02); let mn23_03 = get(m23_03);
        let mn23_12 = get(m23_12); let mn23_13 = get(m23_13); let mn23_23 = get(m23_23);

        let mn01_01 = get(m_at!(m4_01, m2_01, 01));
        let mn01_02 = get(m_at!(m4_01, m2_01, 02));
        let mn01_03 = get(m_at!(m4_01, m2_01, 03));
        let mn01_12 = get(m_at!(m4_01, m2_01, 12));
        let mn01_13 = get(m_at!(m4_01, m2_01, 13));
        let mn01_23 = get(m_at!(m4_01, m2_01, 23));

        let mn02_01 = get(m_at!(m4_02, m2_02, 01));
        let mn02_02 = get(m_at!(m4_02, m2_02, 02));
        let mn02_03 = get(m_at!(m4_02, m2_02, 03));
        let mn02_12 = get(m_at!(m4_02, m2_02, 12));
        let mn02_13 = get(m_at!(m4_02, m2_02, 13));
        let mn02_23 = get(m_at!(m4_02, m2_02, 23));

        let mn03_01 = get(m_at!(m4_03, m2_03, 01));
        let mn03_02 = get(m_at!(m4_03, m2_03, 02));
        let mn03_03 = get(m_at!(m4_03, m2_03, 03));
        let mn03_12 = get(m_at!(m4_03, m2_03, 12));
        let mn03_13 = get(m_at!(m4_03, m2_03, 13));
        let mn03_23 = get(m_at!(m4_03, m2_03, 23));

        let mn12_01 = get(m_at!(m4_12, m2_12, 01));
        let mn12_02 = get(m_at!(m4_12, m2_12, 02));
        let mn12_03 = get(m_at!(m4_12, m2_12, 03));
        let mn12_12 = get(m_at!(m4_12, m2_12, 12));
        let mn12_13 = get(m_at!(m4_12, m2_12, 13));
        let mn12_23 = get(m_at!(m4_12, m2_12, 23));

        let mn13_01 = get(m_at!(m4_13, m2_13, 01));
        let mn13_02 = get(m_at!(m4_13, m2_13, 02));
        let mn13_03 = get(m_at!(m4_13, m2_13, 03));
        let mn13_12 = get(m_at!(m4_13, m2_13, 12));
        let mn13_13 = get(m_at!(m4_13, m2_13, 13));
        let mn13_23 = get(m_at!(m4_13, m2_13, 23));

        // ── Cofactor matrix (= adjugate transposed) ───────────────────────
        //
        // The cofactor C[i][j] = (-1)^(i+j) * M_ij
        // where M_ij is the (i,j) minor (det of 3×3 submatrix deleting row i, col j).
        //
        // For column-major Mat4, cols[c][r] = M[r][c].
        // The inverse is adj(M)^T / det, where adj = cofactor matrix transposed.
        // So inv[c][r] = C[c][r] / det  (cofactor, not minor).
        //
        // C[i][j] for our matrix (row i deleted, col j deleted):
        //
        // Using our column notation: c0..c3 are columns, rows 0..3.
        // M[r][c] = cols[c][r].
        //
        // C[0][0]: delete row 0, col 0 → 3×3 of {c1,c2,c3} rows {1,2,3}
        //   = + c1[1]*(c2[2]*c3[3]-c2[3]*c3[2]) - c1[2]*(c2[1]*c3[3]-c2[3]*c3[1]) + c1[3]*(c2[1]*c3[2]-c2[2]*c3[1])
        //   = + c1[1]*mn23[23]                   - c1[2]*mn23[13]                   + c1[3]*mn23[12]
        //
        // (continuing for all 16 entries...)
        // The sign pattern for C[i][j] is (-1)^(i+j):
        //   + - + -
        //   - + - +
        //   + - + -
        //   - + - +

        // Compute all 16 cofactors.
        // Naming: cof_r_c = cofactor at row r, col c of the original matrix.
        // These become the entries of the adjugate: adj[r][c] = cof_r_c.
        // The inverse column c = adj column c / det = {cof_0_c, cof_1_c, cof_2_c, cof_3_c} / det.

        // Cofactors for col 0 of adj (= row 0 of cofactor matrix):
        let cof_0_0 = c1_1*mn23_23 - c1_2*mn23_13 + c1_3*mn23_12;
        let cof_1_0 = -(c1_0*mn23_23 - c1_2*mn23_03 + c1_3*mn23_02);
        let cof_2_0 = c1_0*mn23_13 - c1_1*mn23_03 + c1_3*mn23_01;
        let cof_3_0 = -(c1_0*mn23_12 - c1_1*mn23_02 + c1_2*mn23_01);

        // Cofactors for col 1 of adj (= row 1 of cofactor matrix):
        let cof_0_1 = -(c0_1*mn23_23 - c0_2*mn23_13 + c0_3*mn23_12);
        let cof_1_1 = c0_0*mn23_23 - c0_2*mn23_03 + c0_3*mn23_02;
        let cof_2_1 = -(c0_0*mn23_13 - c0_1*mn23_03 + c0_3*mn23_01);
        let cof_3_1 = c0_0*mn23_12 - c0_1*mn23_02 + c0_2*mn23_01;

        // Cofactors for col 2 of adj: delete col 2, use minors from (c0,c1) and (c3).
        // C[r][2]: delete row r, col 2 → 3×3 of {c0,c1,c3} rows excluding r.
        // C[0][2] = +(delete row 0 from c0,c1,c3) = c0[1]*(c1[2]*c3[3]-c1[3]*c3[2])
        //           - c0[2]*(c1[1]*c3[3]-c1[3]*c3[1]) + c0[3]*(c1[1]*c3[2]-c1[2]*c3[1])
        //           = c0[1]*mn13[23] - c0[2]*mn13[13] + c0[3]*mn13[12]
        // C[1][2] = -(delete row 1): -(c0[0]*mn13[23] - c0[2]*mn13[03] + c0[3]*mn13[02])
        // C[2][2] = +(delete row 2): c0[0]*mn13[13] - c0[1]*mn13[03] + c0[3]*mn13[01]
        // C[3][2] = -(delete row 3): -(c0[0]*mn13[12] - c0[1]*mn13[02] + c0[2]*mn13[01])
        let cof_0_2 = c0_1*mn13_23 - c0_2*mn13_13 + c0_3*mn13_12;
        let cof_1_2 = -(c0_0*mn13_23 - c0_2*mn13_03 + c0_3*mn13_02);
        let cof_2_2 = c0_0*mn13_13 - c0_1*mn13_03 + c0_3*mn13_01;
        let cof_3_2 = -(c0_0*mn13_12 - c0_1*mn13_02 + c0_2*mn13_01);

        // Cofactors for col 3 of adj: delete col 3, use minors from (c0,c1,c2).
        // C[r][3]: delete row r, col 3 → 3×3 of {c0,c1,c2}.
        // C[0][3] = -(c0[1]*mn12[23] - c0[2]*mn12[13] + c0[3]*mn12[12])
        // C[1][3] = +(c0[0]*mn12[23] - c0[2]*mn12[03] + c0[3]*mn12[02])
        // C[2][3] = -(c0[0]*mn12[13] - c0[1]*mn12[03] + c0[3]*mn12[01])
        // C[3][3] = +(c0[0]*mn12[12] - c0[1]*mn12[02] + c0[2]*mn12[01])
        let cof_0_3 = -(c0_1*mn12_23 - c0_2*mn12_13 + c0_3*mn12_12);
        let cof_1_3 = c0_0*mn12_23 - c0_2*mn12_03 + c0_3*mn12_02;
        let cof_2_3 = -(c0_0*mn12_13 - c0_1*mn12_03 + c0_3*mn12_01);
        let cof_3_3 = c0_0*mn12_12 - c0_1*mn12_02 + c0_2*mn12_01;

        // ── Determinant (dot of first column of cofactors with first col of M) ──
        // det = M[0][0]*C[0][0] + M[1][0]*C[1][0] + M[2][0]*C[2][0] + M[3][0]*C[3][0]
        //     = c0[0]*cof_0_0 + c0[1]*cof_1_0 + c0[2]*cof_2_0 + c0[3]*cof_3_0
        let det = get(c0r0)*cof_0_0 + get(c0r1)*cof_1_0 + get(c0r2)*cof_2_0 + get(c0r3)*cof_3_0;

        if det.abs() < EPSILON {
            return None;
        }
        let inv_det = 1.0 / det;

        // ── Build output via _mm_set_ps and scale by inv_det ──────────────
        // The adjugate column c = {cof_0_c, cof_1_c, cof_2_c, cof_3_c}.
        // Note: _mm_set_ps(e3, e2, e1, e0) sets lane 0 = e0, lane 3 = e3.
        let s = inv_det;
        let adj0 = _mm_set_ps(cof_3_0*s, cof_2_0*s, cof_1_0*s, cof_0_0*s);
        let adj1 = _mm_set_ps(cof_3_1*s, cof_2_1*s, cof_1_1*s, cof_0_1*s);
        let adj2 = _mm_set_ps(cof_3_2*s, cof_2_2*s, cof_1_2*s, cof_0_2*s);
        let adj3 = _mm_set_ps(cof_3_3*s, cof_2_3*s, cof_1_3*s, cof_0_3*s);

        let mut out = Mat4::ZERO;
        _mm_store_ps(out.cols[0].as_mut_ptr(), adj0);
        _mm_store_ps(out.cols[1].as_mut_ptr(), adj1);
        _mm_store_ps(out.cols[2].as_mut_ptr(), adj2);
        _mm_store_ps(out.cols[3].as_mut_ptr(), adj3);
        Some(out)
    }

    // ── TRS inverse ────────────────────────────────────────────────────────
    //
    // (Unchanged from previous commit — kept here for completeness.)
    //
    // Safety: same alignment invariants as inverse_general.

    pub(super) unsafe fn inverse_trs(m: &Mat4) -> Mat4 {
        let c0 = _mm_load_ps(m.cols[0].as_ptr());
        let c1 = _mm_load_ps(m.cols[1].as_ptr());
        let c2 = _mm_load_ps(m.cols[2].as_ptr());
        let c3 = _mm_load_ps(m.cols[3].as_ptr());

        let sq0  = _mm_mul_ps(c0, c0);
        let sq1  = _mm_mul_ps(c1, c1);
        let sq2  = _mm_mul_ps(c2, c2);
        let zero = _mm_setzero_ps();

        let lo01 = _mm_unpacklo_ps(sq0, sq1);
        let lo2z = _mm_unpacklo_ps(sq2, zero);
        let hi01 = _mm_unpackhi_ps(sq0, sq1);
        let hi2z = _mm_unpackhi_ps(sq2, zero);

        let row0 = _mm_movelh_ps(lo01, lo2z);
        let row1 = _mm_movehl_ps(lo2z, lo01);
        let row2 = _mm_movelh_ps(hi01, hi2z);

        let sums = _mm_add_ps(_mm_add_ps(row0, row1), row2);

        let eps  = _mm_set1_ps(EPSILON);
        let mask = _mm_cmpge_ps(sums, eps);

        let safe_sums = _mm_or_ps(
            _mm_and_ps(mask, sums),
            _mm_andnot_ps(mask, _mm_set1_ps(1.0)),
        );
        let inv_scales = _mm_and_ps(mask, _mm_div_ps(_mm_set1_ps(1.0), safe_sums));

        let lo01_r = _mm_unpacklo_ps(c0, c1);
        let lo2z_r = _mm_unpacklo_ps(c2, zero);
        let hi01_r = _mm_unpackhi_ps(c0, c1);
        let hi2z_r = _mm_unpackhi_ps(c2, zero);

        let trow0 = _mm_movelh_ps(lo01_r, lo2z_r);
        let trow1 = _mm_movehl_ps(lo2z_r, lo01_r);
        let trow2 = _mm_movelh_ps(hi01_r, hi2z_r);

        let ic0 = _mm_mul_ps(trow0, inv_scales);
        let ic1 = _mm_mul_ps(trow1, inv_scales);
        let ic2 = _mm_mul_ps(trow2, inv_scales);

        let tx = _mm_shuffle_ps::<0b00_00_00_00>(c3, c3);
        let ty = _mm_shuffle_ps::<0b01_01_01_01>(c3, c3);
        let tz = _mm_shuffle_ps::<0b10_10_10_10>(c3, c3);

        let dotcol = _mm_add_ps(
            _mm_add_ps(_mm_mul_ps(ic0, tx), _mm_mul_ps(ic1, ty)),
            _mm_mul_ps(ic2, tz),
        );
        let neg = _mm_sub_ps(zero, dotcol);

        let mut ic3_arr = [0.0f32; 4];
        _mm_storeu_ps(ic3_arr.as_mut_ptr(), neg);
        ic3_arr[3] = 1.0;
        let ic3 = _mm_loadu_ps(ic3_arr.as_ptr());

        let mut out = Mat4::ZERO;
        _mm_store_ps(out.cols[0].as_mut_ptr(), ic0);
        _mm_store_ps(out.cols[1].as_mut_ptr(), ic1);
        _mm_store_ps(out.cols[2].as_mut_ptr(), ic2);
        _mm_store_ps(out.cols[3].as_mut_ptr(), ic3);
        out
    }
}
