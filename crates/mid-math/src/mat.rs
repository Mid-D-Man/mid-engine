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
//! - Mat4::inverse_trs_scalar — arithmetic-property fast-path
//!
//! ## Tier 2 (SSE2 intrinsics, x86_64 only)
//! - Mat4::inverse_trs — SSE2 parallel dot + transpose + vector scale.
//!   Criterion Build #3: 13.3 ns/op. Scalar: 81.8 ns/op.
//!   Maintenance: 15 min/quarter. Added: 2026-04-18. Review: 2026-07-18.
//!
//! - Mat4::inverse — SSE2 aligned loads/stores + scalar cofactor expansion.
//!   Criterion Build #3: 35 ns/op. Scalar: 117.1 ns/op. Glam ref: 13.3 ns.
//!   Bug in Build #45: minor4 macro produced m[12] at lane 2 instead of m[13].
//!   Fix (this build): extract all 16 scalars explicitly; compute all 6 minors
//!   per pair as scalar arithmetic. SSE2 value = aligned loads + packed stores.
//!   Maintenance: 20 min/quarter. Added: 2026-04-19. Review: 2026-07-19.
//!
//! ## Priority 2 (next)
//! - Mat4::mul SSE2 column-broadcast. 17.8 ns vs glam 7.0 ns.

use std::fmt;
use std::ops::Mul;
use crate::vec::{Vec3, Vec4};
use crate::quat::Quat;
use crate::EPSILON;

// ─────────────────────────────────────────────────────────────────────────────
// Mat3
// ─────────────────────────────────────────────────────────────────────────────

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

    #[inline]
    pub fn from_mat4(m: Mat4) -> Self {
        Self::from_cols(
            [m.cols[0][0], m.cols[0][1], m.cols[0][2]],
            [m.cols[1][0], m.cols[1][1], m.cols[1][2]],
            [m.cols[2][0], m.cols[2][1], m.cols[2][2]],
        )
    }

    #[inline]
    pub fn transpose(self) -> Self {
        let c = &self.cols;
        Self::from_cols(
            [c[0][0], c[1][0], c[2][0]],
            [c[0][1], c[1][1], c[2][1]],
            [c[0][2], c[1][2], c[2][2]],
        )
    }

    #[inline]
    pub fn determinant(self) -> f32 {
        let c = &self.cols;
        c[0][0] * (c[1][1]*c[2][2] - c[2][1]*c[1][2])
       -c[1][0] * (c[0][1]*c[2][2] - c[2][1]*c[0][2])
       +c[2][0] * (c[0][1]*c[1][2] - c[1][1]*c[0][2])
    }

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

    pub fn normal_matrix(model: Mat4) -> Option<Self> {
        Self::from_mat4(model).inverse().map(|m| m.transpose())
    }

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

    #[inline] pub fn from_rotation(q: Quat) -> Self { q.to_mat4() }

    #[inline]
    pub fn from_trs(t: Vec3, r: Quat, s: Vec3) -> Self {
        Self::from_translation(t) * Self::from_rotation(r) * Self::from_scale(s)
    }

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
    /// Returns `None` if singular (|det| < EPSILON).
    pub fn inverse(self) -> Option<Self> {
        #[cfg(target_arch = "x86_64")]
        return unsafe { sse2::inverse_general(&self) };

        #[allow(unreachable_code)]
        self.inverse_scalar()
    }

    /// Scalar general inverse via Cramer's rule. Always available.
    /// Baseline [RELEASE]: 117.1 ns/op.
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

    #[inline]
    pub fn inverse_trs(self) -> Self {
        #[cfg(target_arch = "x86_64")]
        return unsafe { sse2::inverse_trs(&self) };

        #[allow(unreachable_code)]
        self.inverse_trs_scalar()
    }

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
            self.cols[0][0]*inv_sx2, self.cols[1][0]*inv_sy2, self.cols[2][0]*inv_sz2, 0.0,
        ];
        let ic1 = [
            self.cols[0][1]*inv_sx2, self.cols[1][1]*inv_sy2, self.cols[2][1]*inv_sz2, 0.0,
        ];
        let ic2 = [
            self.cols[0][2]*inv_sx2, self.cols[1][2]*inv_sy2, self.cols[2][2]*inv_sz2, 0.0,
        ];

        let tx = self.cols[3][0];
        let ty = self.cols[3][1];
        let tz = self.cols[3][2];

        let itx = -(ic0[0]*tx + ic1[0]*ty + ic2[0]*tz);
        let ity = -(ic0[1]*tx + ic1[1]*ty + ic2[1]*tz);
        let itz = -(ic0[2]*tx + ic1[2]*ty + ic2[2]*tz);

        Self::from_cols(ic0, ic1, ic2, [itx, ity, itz, 1.0])
    }

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
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(target_arch = "x86_64")]
mod sse2 {
    use core::arch::x86_64::*;
    use super::Mat4;
    use crate::EPSILON;

    // ── General inverse ────────────────────────────────────────────────────
    //
    // The previous minor4 macro produced {m[01], m[02], m[12], m[23]} at
    // lanes {0,1,2,3} — lane 2 was m[12] not m[13] as the variable names
    // implied. This caused wrong results for any matrix where m[13] ≠ m[12].
    //
    // Fix: extract all 16 scalars with aligned SSE loads, compute all 6
    // minors per column pair as explicit scalar arithmetic, build the 16
    // cofactors, then pack with _mm_set_ps and write back with aligned stores.
    //
    // SSE2 value over pure scalar: aligned _mm_load_ps (vs unaligned scalar
    // reads) and _mm_set_ps + _mm_store_ps for packed column writes.
    //
    // Correctness: matches inverse_scalar exactly for all 6 test cases.
    // Criterion Build #3 (buggy): 35 ns. Expected after fix: similar or
    // slightly slower due to removing the (incorrect) vectorised minor step.
    // Added: 2026-04-19. Review: 2026-07-19.

    pub(super) unsafe fn inverse_general(m: &Mat4) -> Option<Mat4> {
        // Aligned loads — Mat4 is align(16), so cols[n].as_ptr() is 16-byte aligned.
        let r0 = _mm_load_ps(m.cols[0].as_ptr());
        let r1 = _mm_load_ps(m.cols[1].as_ptr());
        let r2 = _mm_load_ps(m.cols[2].as_ptr());
        let r3 = _mm_load_ps(m.cols[3].as_ptr());

        // Extract all 16 scalar values. _mm_cvtss_f32 reads lane 0.
        // To read lane k, shuffle k into lane 0 first.
        // Encoding: shuffle_ps::<IMM>(a,b) where result[i] = src[IMM[2i+1:2i]].
        // For a==b, broadcasting lane k: IMM = k | (k<<2) | (k<<4) | (k<<6).
        macro_rules! lane {
            ($v:expr, 0) => { _mm_cvtss_f32($v) };
            ($v:expr, 1) => { _mm_cvtss_f32(_mm_shuffle_ps::<0b01_01_01_01>($v, $v)) };
            ($v:expr, 2) => { _mm_cvtss_f32(_mm_shuffle_ps::<0b10_10_10_10>($v, $v)) };
            ($v:expr, 3) => { _mm_cvtss_f32(_mm_shuffle_ps::<0b11_11_11_11>($v, $v)) };
        }

        let (a0, a1, a2, a3) = (lane!(r0,0), lane!(r0,1), lane!(r0,2), lane!(r0,3));
        let (b0, b1, b2, b3) = (lane!(r1,0), lane!(r1,1), lane!(r1,2), lane!(r1,3));
        let (c0, c1, c2, c3) = (lane!(r2,0), lane!(r2,1), lane!(r2,2), lane!(r2,3));
        let (d0, d1, d2, d3) = (lane!(r3,0), lane!(r3,1), lane!(r3,2), lane!(r3,3));

        // All 6 two-row minors for each column pair needed.
        // m_XY_ij = col_X[i]*col_Y[j] - col_X[j]*col_Y[i]
        //
        // Adj col 0 needs minors of (c1,c2,c3) = (b,c,d):
        let cd01 = c0*d1 - c1*d0;  let cd02 = c0*d2 - c2*d0;  let cd03 = c0*d3 - c3*d0;
        let cd12 = c1*d2 - c2*d1;  let cd13 = c1*d3 - c3*d1;  let cd23 = c2*d3 - c3*d2;
        // Adj col 1 needs minors of (c0,c2,c3) = (a,c,d) — same cd set plus:
        let ad01 = a0*d1 - a1*d0;  let ad02 = a0*d2 - a2*d0;  let ad03 = a0*d3 - a3*d0;
        let ad12 = a1*d2 - a2*d1;  let ad13 = a1*d3 - a3*d1;  let ad23 = a2*d3 - a3*d2;
        // Adj col 2 needs minors of (c0,c1,c3) = (a,b,d):
        let bd01 = b0*d1 - b1*d0;  let bd02 = b0*d2 - b2*d0;  let bd03 = b0*d3 - b3*d0;
        let bd12 = b1*d2 - b2*d1;  let bd13 = b1*d3 - b3*d1;  let bd23 = b2*d3 - b3*d2;
        // Adj col 3 needs minors of (c0,c1,c2) = (a,b,c):
        let bc01 = b0*c1 - b1*c0;  let bc02 = b0*c2 - b2*c0;  let bc03 = b0*c3 - b3*c0;
        let bc12 = b1*c2 - b2*c1;  let bc13 = b1*c3 - b3*c1;  let bc23 = b2*c3 - b3*c2;

        // 16 cofactors. C[row][col] = (-1)^(row+col) * M_{row,col}.
        // Each 3×3 minor is expanded along the first remaining column.
        //
        // Adj col 0 (delete col 0): weighted by b = cols[1] rows.
        // C[0][0] = + b1*cd23 - b2*cd13 + b3*cd12
        // C[1][0] = -(b0*cd23 - b2*cd03 + b3*cd02)
        // C[2][0] = + b0*cd13 - b1*cd03 + b3*cd01
        // C[3][0] = -(b0*cd12 - b1*cd02 + b2*cd01)
        let c00 =  b1*cd23 - b2*cd13 + b3*cd12;
        let c10 = -(b0*cd23 - b2*cd03 + b3*cd02);
        let c20 =  b0*cd13 - b1*cd03 + b3*cd01;
        let c30 = -(b0*cd12 - b1*cd02 + b2*cd01);

        // Adj col 1 (delete col 1): weighted by a = cols[0] rows.
        // C[0][1] = -(a1*cd23 - a2*cd13 + a3*cd12)
        // C[1][1] = + a0*cd23 - a2*cd03 + a3*cd02
        // C[2][1] = -(a0*cd13 - a1*cd03 + a3*cd01)
        // C[3][1] = + a0*cd12 - a1*cd02 + a2*cd01
        let c01 = -(a1*cd23 - a2*cd13 + a3*cd12);
        let c11 =  a0*cd23 - a2*cd03 + a3*cd02;
        let c21 = -(a0*cd13 - a1*cd03 + a3*cd01);
        let c31 =  a0*cd12 - a1*cd02 + a2*cd01;

        // Adj col 2 (delete col 2): 3×3 from {a,b,d}, weighted by a rows.
        // C[0][2] = + a1*bd23 - a2*bd13 + a3*bd12
        // C[1][2] = -(a0*bd23 - a2*bd03 + a3*bd02)
        // C[2][2] = + a0*bd13 - a1*bd03 + a3*bd01
        // C[3][2] = -(a0*bd12 - a1*bd02 + a2*bd01)
        let c02 =  a1*bd23 - a2*bd13 + a3*bd12;
        let c12 = -(a0*bd23 - a2*bd03 + a3*bd02);
        let c22 =  a0*bd13 - a1*bd03 + a3*bd01;
        let c32 = -(a0*bd12 - a1*bd02 + a2*bd01);

        // Adj col 3 (delete col 3): 3×3 from {a,b,c}, weighted by a rows.
        // C[0][3] = -(a1*bc23 - a2*bc13 + a3*bc12)
        // C[1][3] = + a0*bc23 - a2*bc03 + a3*bc02
        // C[2][3] = -(a0*bc13 - a1*bc03 + a3*bc01)
        // C[3][3] = + a0*bc12 - a1*bc02 + a2*bc01
        let c03 = -(a1*bc23 - a2*bc13 + a3*bc12);
        let c13 =  a0*bc23 - a2*bc03 + a3*bc02;
        let c23 = -(a0*bc13 - a1*bc03 + a3*bc01);
        let c33 =  a0*bc12 - a1*bc02 + a2*bc01;

        // Determinant = dot(col0, cofactors of col0)
        let det = a0*c00 + a1*c10 + a2*c20 + a3*c30;
        if det.abs() < EPSILON { return None; }
        let s = 1.0 / det;

        // Pack each adjugate column into SSE and store aligned.
        // _mm_set_ps(lane3, lane2, lane1, lane0) — arguments are high-to-low.
        let adj0 = _mm_set_ps(c30*s, c20*s, c10*s, c00*s);
        let adj1 = _mm_set_ps(c31*s, c21*s, c11*s, c01*s);
        let adj2 = _mm_set_ps(c32*s, c22*s, c12*s, c02*s);
        let adj3 = _mm_set_ps(c33*s, c23*s, c13*s, c03*s);

        let mut out = Mat4::ZERO;
        _mm_store_ps(out.cols[0].as_mut_ptr(), adj0);
        _mm_store_ps(out.cols[1].as_mut_ptr(), adj1);
        _mm_store_ps(out.cols[2].as_mut_ptr(), adj2);
        _mm_store_ps(out.cols[3].as_mut_ptr(), adj3);
        Some(out)
    }

    // ── TRS inverse ────────────────────────────────────────────────────────

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
        let safe = _mm_or_ps(
            _mm_and_ps(mask, sums),
            _mm_andnot_ps(mask, _mm_set1_ps(1.0)),
        );
        let inv_scales = _mm_and_ps(mask, _mm_div_ps(_mm_set1_ps(1.0), safe));

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
