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
    // Safety:
    //   cols[n].as_ptr() is 16-byte aligned (Mat4 is align(16)).
    //   All loads/stores use aligned variants.

    #[inline(always)]
    unsafe fn compute_minor_pair(ca: __m128, cb: __m128, shuf_a: i32, shuf_b: i32) -> __m128 {
        let _ = (ca, cb, shuf_a, shuf_b);
        unreachable!()
    }

    /// SSE2 general 4×4 inverse. Returns None if singular.
    ///
    /// Correctness verified against inverse_scalar in tests.
    /// Scalar baseline [RELEASE]: 117.1 ns/op.
    /// Added: 2026-04-19. Review: 2026-07-19.
    pub(super) unsafe fn inverse_general(m: &Mat4) -> Option<Mat4> {
        let c0 = _mm_load_ps(m.cols[0].as_ptr());
        let c1 = _mm_load_ps(m.cols[1].as_ptr());
        let c2 = _mm_load_ps(m.cols[2].as_ptr());
        let c3 = _mm_load_ps(m.cols[3].as_ptr());

        macro_rules! minor4 {
            ($ca:expr, $cb:expr) => {{
                let ca = $ca;
                let cb = $cb;
                let ca_0012 = _mm_shuffle_ps::<0b10_01_00_00>(ca, ca);
                let cb_1223 = _mm_shuffle_ps::<0b11_10_10_01>(cb, cb);
                let ca_1223 = _mm_shuffle_ps::<0b11_10_10_01>(ca, ca);
                let cb_0012 = _mm_shuffle_ps::<0b10_01_00_00>(cb, cb);
                _mm_sub_ps(_mm_mul_ps(ca_0012, cb_1223), _mm_mul_ps(ca_1223, cb_0012))
            }};
        }

        macro_rules! minor03_12 {
            ($ca:expr, $cb:expr) => {{
                let ca = $ca;
                let cb = $cb;
                let ca_0110 = _mm_shuffle_ps::<0b00_01_01_00>(ca, ca);
                let cb_3223 = _mm_shuffle_ps::<0b11_10_10_11>(cb, cb);
                let ca_3223 = _mm_shuffle_ps::<0b11_10_10_11>(ca, ca);
                let cb_0110 = _mm_shuffle_ps::<0b00_01_01_00>(cb, cb);
                _mm_sub_ps(_mm_mul_ps(ca_0110, cb_3223), _mm_mul_ps(ca_3223, cb_0110))
            }};
        }

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

        macro_rules! bcast {
            ($v:expr, 0) => { _mm_shuffle_ps::<0b00_00_00_00>($v, $v) };
            ($v:expr, 1) => { _mm_shuffle_ps::<0b01_01_01_01>($v, $v) };
            ($v:expr, 2) => { _mm_shuffle_ps::<0b10_10_10_10>($v, $v) };
            ($v:expr, 3) => { _mm_shuffle_ps::<0b11_11_11_11>($v, $v) };
        }

        let c0r0 = bcast!(c0, 0);  let c0r1 = bcast!(c0, 1);
        let c0r2 = bcast!(c0, 2);  let c0r3 = bcast!(c0, 3);
        let c1r0 = bcast!(c1, 0);  let c1r1 = bcast!(c1, 1);
        let c1r2 = bcast!(c1, 2);  let c1r3 = bcast!(c1, 3);
        let c2r0 = bcast!(c2, 0);  let c2r1 = bcast!(c2, 1);
        let c2r2 = bcast!(c2, 2);  let c2r3 = bcast!(c2, 3);
        let c3r0 = bcast!(c3, 0);  let c3r1 = bcast!(c3, 1);
        let c3r2 = bcast!(c3, 2);  let c3r3 = bcast!(c3, 3);

        macro_rules! m_at {
            ($m4:expr, $m2:expr, 01) => { bcast!($m4, 0) };
            ($m4:expr, $m2:expr, 02) => { bcast!($m4, 1) };
            ($m4:expr, $m2:expr, 03) => { bcast!($m2, 0) };
            ($m4:expr, $m2:expr, 12) => { bcast!($m2, 1) };
            ($m4:expr, $m2:expr, 13) => { bcast!($m4, 2) };
            ($m4:expr, $m2:expr, 23) => { bcast!($m4, 3) };
        }

        let m23_01 = m_at!(m4_23, m2_23, 01);
        let m23_02 = m_at!(m4_23, m2_23, 02);
        let m23_03 = m_at!(m4_23, m2_23, 03);
        let m23_12 = m_at!(m4_23, m2_23, 12);
        let m23_13 = m_at!(m4_23, m2_23, 13);
        let m23_23 = m_at!(m4_23, m2_23, 23);

        let get = |v: __m128| -> f32 { _mm_cvtss_f32(v) };

        // ── Extract all needed scalar values ──────────────────────────────
        let c0_0 = get(c0r0); let c0_1 = get(c0r1);
        let c0_2 = get(c0r2); let c0_3 = get(c0r3);

        let c1_0 = get(c1r0); let c1_1 = get(c1r1);
        let c1_2 = get(c1r2); let c1_3 = get(c1r3);
        let c2_0 = get(c2r0); let c2_1 = get(c2r1);
        let c2_2 = get(c2r2); let c2_3 = get(c2r3);
        let c3_0 = get(c3r0); let c3_1 = get(c3r1);
        let c3_2 = get(c3r2); let c3_3 = get(c3r3);

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

        // ── Cofactors for adjugate col 0 (deleting col 0, weighted by c1) ──
        let cof_0_0 = c1_1*mn23_23 - c1_2*mn23_13 + c1_3*mn23_12;
        let cof_1_0 = -(c1_0*mn23_23 - c1_2*mn23_03 + c1_3*mn23_02);
        let cof_2_0 = c1_0*mn23_13 - c1_1*mn23_03 + c1_3*mn23_01;
        let cof_3_0 = -(c1_0*mn23_12 - c1_1*mn23_02 + c1_2*mn23_01);

        // ── Cofactors for adjugate col 1 (deleting col 1, weighted by c0) ──
        let cof_0_1 = -(c0_1*mn23_23 - c0_2*mn23_13 + c0_3*mn23_12);
        let cof_1_1 = c0_0*mn23_23 - c0_2*mn23_03 + c0_3*mn23_02;
        let cof_2_1 = -(c0_0*mn23_13 - c0_1*mn23_03 + c0_3*mn23_01);
        let cof_3_1 = c0_0*mn23_12 - c0_1*mn23_02 + c0_2*mn23_01;

        // ── Cofactors for adjugate col 2 (deleting col 2, from c0,c1,c3) ──
        let cof_0_2 = c0_1*mn13_23 - c0_2*mn13_13 + c0_3*mn13_12;
        let cof_1_2 = -(c0_0*mn13_23 - c0_2*mn13_03 + c0_3*mn13_02);
        let cof_2_2 = c0_0*mn13_13 - c0_1*mn13_03 + c0_3*mn13_01;
        let cof_3_2 = -(c0_0*mn13_12 - c0_1*mn13_02 + c0_2*mn13_01);

        // ── Cofactors for adjugate col 3 (deleting col 3, from c0,c1,c2) ──
        let cof_0_3 = -(c0_1*mn12_23 - c0_2*mn12_13 + c0_3*mn12_12);
        let cof_1_3 = c0_0*mn12_23 - c0_2*mn12_03 + c0_3*mn12_02;
        let cof_2_3 = -(c0_0*mn12_13 - c0_1*mn12_03 + c0_3*mn12_01);
        let cof_3_3 = c0_0*mn12_12 - c0_1*mn12_02 + c0_2*mn12_01;

        // ── Determinant ───────────────────────────────────────────────────
        let det = c0_0*cof_0_0 + c0_1*cof_1_0 + c0_2*cof_2_0 + c0_3*cof_3_0;

        if det.abs() < EPSILON {
            return None;
        }
        let inv_det = 1.0 / det;

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
