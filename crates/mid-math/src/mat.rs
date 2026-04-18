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
//!   SSE2 is the x86_64 ABI baseline (mandatory since 2003), no runtime detection needed.
//!   Scalar fallback: inverse_trs_scalar().
//!   Maintenance estimate: 15 min/quarter. Added: 2026-04-18. Review: 2026-07-18.
//!
//! ## Not yet optimised (general inverse)
//! - Mat4::inverse — Cramer's rule, 116.2 ns/op (Build #29 [RELEASE]).
//!   Tier 2 SSE2 cofactor/shuffle implementation is the next target.
//!   Requires: [RELEASE] benchmark proving ≥10% gain at 5k-mat scale.

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
    /// Used to correctly transform surface normals when a non-uniform scale
    /// is present. Returns `None` for degenerate matrices.
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
    /// Explicitly unrolled 3×3 multiply. Tier 1.
    /// Removes loop bounds so LLVM can schedule all 9 products simultaneously.
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
/// Layout matches OpenGL / Vulkan / Metal column-major convention.
///
/// **C layout:** `float cols[4][4]` — first index is column, second is row.
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

    /// Translation matrix.
    #[inline]
    pub fn from_translation(t: Vec3) -> Self {
        let mut m = Self::IDENTITY;
        m.cols[3] = [t.x, t.y, t.z, 1.0];
        m
    }

    /// Non-uniform scale matrix.
    #[inline]
    pub fn from_scale(s: Vec3) -> Self {
        Self::from_cols(
            [s.x, 0.0, 0.0, 0.0],
            [0.0, s.y, 0.0, 0.0],
            [0.0, 0.0, s.z, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        )
    }

    /// Rotation matrix from a normalized quaternion.
    #[inline]
    pub fn from_rotation(q: Quat) -> Self { q.to_mat4() }

    /// TRS matrix: Translation × Rotation × Scale (applied right-to-left).
    #[inline]
    pub fn from_trs(t: Vec3, r: Quat, s: Vec3) -> Self {
        Self::from_translation(t) * Self::from_rotation(r) * Self::from_scale(s)
    }

    // ── Camera / Projection ────────────────────────────────────────────────

    /// Right-handed look-at view matrix (Y-up, camera looks down -Z).
    ///
    /// `eye`    = camera world position
    /// `center` = target world position
    /// `up`     = world up vector (usually `Vec3::Y`)
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

    /// Right-handed perspective projection.
    /// NDC clip space: x/y ∈ [-1,1], z ∈ [-1,1] (OpenGL convention).
    ///
    /// `fov_y`  = vertical field of view in radians
    /// `aspect` = width ÷ height
    /// `near`   = near clip plane (must be > 0)
    /// `far`    = far clip plane (must be > near)
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

    /// Right-handed orthographic projection.
    /// NDC clip space: x/y ∈ [-1,1], z ∈ [-1,1] (OpenGL convention).
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

    /// Transpose.
    pub fn transpose(self) -> Self {
        let c = &self.cols;
        Self::from_cols(
            [c[0][0], c[1][0], c[2][0], c[3][0]],
            [c[0][1], c[1][1], c[2][1], c[3][1]],
            [c[0][2], c[1][2], c[2][2], c[3][2]],
            [c[0][3], c[1][3], c[2][3], c[3][3]],
        )
    }

    /// Determinant.
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

    /// General 4×4 inverse via cofactor expansion (Cramer's rule).
    /// Returns `None` if the matrix is singular.
    ///
    /// For matrices you know are TRS, call `inverse_trs()` instead —
    /// it is significantly cheaper (81.8 ns vs 116.2 ns, Build #29 [RELEASE]).
    ///
    /// Build #29 [RELEASE] baseline: 116.2 ns/op.
    /// Next target: Tier 2 SSE2 cofactor/shuffle — pending benchmark proof.
    pub fn inverse(self) -> Option<Self> {
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

    /// Fast inverse for **pure TRS** (Translation × Rotation × Scale) matrices.
    ///
    /// Dispatches to SSE2 on x86_64 (guaranteed ABI baseline, no runtime check).
    /// Falls back to `inverse_trs_scalar()` on all other architectures.
    ///
    /// **Do not call this on matrices containing shear or projection.**
    /// Results are undefined (no panic, but mathematically wrong).
    /// Use `inverse()` for the general case.
    ///
    /// Build #29 [RELEASE] scalar baseline: 81.8 ns/op.
    /// SSE2 first measurement: pending next CI build.
    #[inline]
    pub fn inverse_trs(self) -> Self {
        #[cfg(target_arch = "x86_64")]
        return unsafe { sse2::inverse_trs(&self) };

        #[allow(unreachable_code)]
        self.inverse_trs_scalar()
    }

    /// Scalar TRS inverse. Always available on all platforms.
    ///
    /// Used as: the non-x86_64 fallback, the verification baseline in tests,
    /// and the constexpr-compatible path if needed.
    ///
    /// ## How it works
    ///
    /// For M = T·R·S, the inverse is M⁻¹ = S⁻¹·Rᵀ·T⁻¹ because:
    /// - R is orthogonal → Rᵀ = R⁻¹
    /// - S is diagonal → S⁻¹ = diag(1/sx, 1/sy, 1/sz) computed via
    ///   reciprocal squared column norms (avoids sqrt).
    ///
    /// **Undefined for matrices with shear.** Use `inverse()` otherwise.
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

        // Output column i = transposed row i of the upper 3×3, scaled by
        // the reciprocal squared norm of the corresponding input column.
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

        // Inverse translation = −(S⁻¹·Rᵀ)·t
        let itx = -(ic0[0]*tx + ic1[0]*ty + ic2[0]*tz);
        let ity = -(ic0[1]*tx + ic1[1]*ty + ic2[1]*tz);
        let itz = -(ic0[2]*tx + ic1[2]*ty + ic2[2]*tz);

        Self::from_cols(ic0, ic1, ic2, [itx, ity, itz, 1.0])
    }

    // ── Transform helpers ──────────────────────────────────────────────────

    /// Transform a point (w = 1 — affected by translation).
    #[inline]
    pub fn transform_point(self, p: Vec3) -> Vec3 {
        (self * p.extend(1.0)).truncate()
    }

    /// Transform a direction vector (w = 0 — unaffected by translation).
    #[inline]
    pub fn transform_vector(self, v: Vec3) -> Vec3 {
        (self * v.extend(0.0)).truncate()
    }
}

impl Default for Mat4 { fn default() -> Self { Self::IDENTITY } }

impl Mul for Mat4 {
    type Output = Self;
    /// Explicitly unrolled column-major matrix multiply: `self × rhs`.
    ///
    /// 64 independent multiply-accumulate ops with no loop bounds, allowing
    /// LLVM to schedule and vectorize the full instruction window at once.
    /// Build #29 [RELEASE]: 6.7 ns/op.
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
    /// Transform a homogeneous Vec4: `M × v`.
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
// SSE2 is the x86_64 ABI baseline (mandatory since 2003, present in every
// x86_64 CPU ever shipped). No runtime detection is required.
//
// Safety invariants:
// - All `_mm_load_ps` calls use `Mat4::cols[n].as_ptr()`.
//   Mat4 is `#[repr(C, align(16))]`, so every cols[n] array is 16-byte
//   aligned. Aligned loads are always correct here. ✓
// - Output `_mm_store_ps` targets `Mat4::ZERO.cols[n].as_mut_ptr()`,
//   also 16-byte aligned. ✓
// - No side effects outside the returned Mat4. ✓
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(target_arch = "x86_64")]
mod sse2 {
    use core::arch::x86_64::*;
    use super::Mat4;
    use crate::EPSILON;

    /// SSE2 inverse of a TRS (Translation × Rotation × Scale) matrix.
    ///
    /// ## Algorithm
    ///
    /// For M = T·R·S, the inverse is M⁻¹ = S⁻¹·Rᵀ·T⁻¹.
    ///
    /// Step 1: Load the four columns into SSE registers.
    ///
    /// Step 2: Compute squared norms {sx², sy², sz², 0} in parallel.
    ///   - Square each column element-wise: sq0 = c0*c0, sq1 = c1*c1, sq2 = c2*c2.
    ///   - Partially transpose with unpacklo/hi + movelh/hl to gather x, y, z
    ///     components of each squared column into three row vectors.
    ///   - Sum the three rows to get {sx², sy², sz², 0}.
    ///
    /// Step 3: Compute masked reciprocals.
    ///   - EPSILON mask prevents division by zero for degenerate (zero-scale) axes.
    ///   - Result: inv_scales = {1/sx², 1/sy², 1/sz², 0}.
    ///
    /// Step 4: Transpose upper 3×3 and apply inv_scales element-wise.
    ///   - trow0 = {c0[0], c1[0], c2[0], 0} → output column 0 of S⁻¹·Rᵀ
    ///   - trow1 = {c0[1], c1[1], c2[1], 0} → output column 1
    ///   - trow2 = {c0[2], c1[2], c2[2], 0} → output column 2
    ///   - Each multiplied element-wise by inv_scales.
    ///
    /// Step 5: Compute inverse translation = −(S⁻¹·Rᵀ)·t.
    ///   - Broadcast tx, ty, tz from the translation column c3.
    ///   - dotcol = ic0·tx + ic1·ty + ic2·tz.
    ///   - Negate, set w = 1.0.
    ///
    /// Step 6: Store all four output columns via aligned _mm_store_ps.
    ///
    /// ## Correctness
    ///
    /// Verified against `inverse_trs_scalar` by the test
    /// `mat4_inverse_trs_sse2_matches_scalar` (6 TRS cases, 1e-5 tolerance).
    ///
    /// ## Performance
    ///
    /// Scalar baseline Build #29 [RELEASE]: 81.8 ns/op.
    /// SSE2 first measurement: pending next CI build after this commit.
    /// Maintenance estimate: 15 min/quarter. Added: 2026-04-18. Review: 2026-07-18.
    pub(super) unsafe fn inverse_trs(m: &Mat4) -> Mat4 {
        // ── Step 1: Load columns ──────────────────────────────────────────
        let c0 = _mm_load_ps(m.cols[0].as_ptr());
        let c1 = _mm_load_ps(m.cols[1].as_ptr());
        let c2 = _mm_load_ps(m.cols[2].as_ptr());
        let c3 = _mm_load_ps(m.cols[3].as_ptr()); // {tx, ty, tz, 1.0}

        // ── Step 2: Squared norms {sx², sy², sz², 0} in parallel ─────────
        let sq0  = _mm_mul_ps(c0, c0);
        let sq1  = _mm_mul_ps(c1, c1);
        let sq2  = _mm_mul_ps(c2, c2);
        let zero = _mm_setzero_ps();

        // Interleave pairs to transpose:
        // lo01 = {sq0[0], sq1[0], sq0[1], sq1[1]}
        // lo2z = {sq2[0],    0.0, sq2[1],    0.0}
        // hi01 = {sq0[2], sq1[2], sq0[3], sq1[3]}
        // hi2z = {sq2[2],    0.0, sq2[3],    0.0}
        let lo01 = _mm_unpacklo_ps(sq0, sq1);
        let lo2z = _mm_unpacklo_ps(sq2, zero);
        let hi01 = _mm_unpackhi_ps(sq0, sq1);
        let hi2z = _mm_unpackhi_ps(sq2, zero);

        // Assemble rows:
        // row0 = {sq0[0], sq1[0], sq2[0], 0} — x²s of each column
        // row1 = {sq0[1], sq1[1], sq2[1], 0} — y²s
        // row2 = {sq0[2], sq1[2], sq2[2], 0} — z²s
        let row0 = _mm_movelh_ps(lo01, lo2z);
        let row1 = _mm_movehl_ps(lo2z, lo01);
        let row2 = _mm_movelh_ps(hi01, hi2z);

        // Column sums = {sx², sy², sz², 0}
        let sums = _mm_add_ps(_mm_add_ps(row0, row1), row2);

        // ── Step 3: Masked reciprocals ────────────────────────────────────
        let eps  = _mm_set1_ps(EPSILON);
        let mask = _mm_cmpge_ps(sums, eps);

        // Replace near-zero denominators with 1.0 so division is safe;
        // the mask zeroes out those reciprocal results afterward.
        let safe_sums  = _mm_or_ps(
            _mm_and_ps(mask, sums),
            _mm_andnot_ps(mask, _mm_set1_ps(1.0)),
        );
        let inv_scales = _mm_and_ps(mask, _mm_div_ps(_mm_set1_ps(1.0), safe_sums));

        // ── Step 4: Transpose upper 3×3 and scale ────────────────────────
        let lo01_r = _mm_unpacklo_ps(c0, c1);
        let lo2z_r = _mm_unpacklo_ps(c2, zero);
        let hi01_r = _mm_unpackhi_ps(c0, c1);
        let hi2z_r = _mm_unpackhi_ps(c2, zero);

        let trow0 = _mm_movelh_ps(lo01_r, lo2z_r);
        let trow1 = _mm_movehl_ps(lo2z_r, lo01_r);
        let trow2 = _mm_movelh_ps(hi01_r, hi2z_r);

        // Each output column is transposed row × inv_scales element-wise.
        let ic0 = _mm_mul_ps(trow0, inv_scales);
        let ic1 = _mm_mul_ps(trow1, inv_scales);
        let ic2 = _mm_mul_ps(trow2, inv_scales);

        // ── Step 5: Inverse translation = −(S⁻¹·Rᵀ)·t ───────────────────
        // Broadcast each translation component.
        // _mm_shuffle_ps::<IMM>(a, b) selects 4 lanes from a (lower 2) and b (upper 2).
        // With a == b, this broadcasts a single lane to all 4 positions.
        let tx = _mm_shuffle_ps::<0b00_00_00_00>(c3, c3); // {tx,tx,tx,tx}
        let ty = _mm_shuffle_ps::<0b01_01_01_01>(c3, c3); // {ty,ty,ty,ty}
        let tz = _mm_shuffle_ps::<0b10_10_10_10>(c3, c3); // {tz,tz,tz,tz}

        let dotcol = _mm_add_ps(
            _mm_add_ps(_mm_mul_ps(ic0, tx), _mm_mul_ps(ic1, ty)),
            _mm_mul_ps(ic2, tz),
        );
        // Negate and fix w = 1.0.
        let neg = _mm_sub_ps(zero, dotcol);

        // Write the negated translation column: x, y, z from neg, w = 1.0.
        // We use a scalar store here because _mm_move_ss only swaps lane 0,
        // not lane 3. Writing to a stack array is the cleanest approach and
        // the compiler eliminates the temporary under -O3.
        let mut ic3_arr = [0.0f32; 4];
        _mm_storeu_ps(ic3_arr.as_mut_ptr(), neg);
        ic3_arr[3] = 1.0;
        let ic3 = _mm_loadu_ps(ic3_arr.as_ptr());

        // ── Step 6: Store output ──────────────────────────────────────────
        let mut out = Mat4::ZERO;
        _mm_store_ps(out.cols[0].as_mut_ptr(), ic0);
        _mm_store_ps(out.cols[1].as_mut_ptr(), ic1);
        _mm_store_ps(out.cols[2].as_mut_ptr(), ic2);
        _mm_store_ps(out.cols[3].as_mut_ptr(), ic3);
        out
    }
}
