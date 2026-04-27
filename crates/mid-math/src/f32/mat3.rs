// crates/mid-math/src/f32/mat3.rs
//! Mat3 — always scalar, column-major.
//!
//! Used for normal matrices, 2D transforms, and extracting the rotation
//! sub-matrix from a Mat4. No SIMD variant — the 3×3 layout does not
//! map cleanly to 128-bit registers without wasted computation.

use core::fmt;
use core::ops::Mul;
use crate::Vec3;
use crate::f32::sse2::mat4::Mat4;
use crate::EPSILON;

/// 3×3 column-major matrix. 36 bytes. Always scalar on all platforms.
///
/// `cols[c][r]` = element at column `c`, row `r`.
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

    // ── Constructors ──────────────────────────────────────────────────────────

    #[inline]
    pub fn from_cols(c0: [f32;3], c1: [f32;3], c2: [f32;3]) -> Self {
        Self { cols: [c0, c1, c2] }
    }

    /// Extract the upper-left 3×3 sub-matrix from a Mat4.
    #[inline]
    pub fn from_mat4(m: &Mat4) -> Self {
        Self::from_cols(
            [m.cols[0][0], m.cols[0][1], m.cols[0][2]],
            [m.cols[1][0], m.cols[1][1], m.cols[1][2]],
            [m.cols[2][0], m.cols[2][1], m.cols[2][2]],
        )
    }

    /// Build from three column vectors.
    #[inline]
    pub fn from_vecs(c0: Vec3, c1: Vec3, c2: Vec3) -> Self {
        Self::from_cols(
            [c0.x, c0.y, c0.z],
            [c1.x, c1.y, c1.z],
            [c2.x, c2.y, c2.z],
        )
    }

    /// Build a 2D rotation matrix (rotation in the XY plane by `angle` radians).
    #[inline]
    pub fn from_rotation_z(angle: f32) -> Self {
        let (s, c) = (angle.sin(), angle.cos());
        Self::from_cols(
            [c,  s,  0.0],
            [-s, c,  0.0],
            [0.0, 0.0, 1.0],
        )
    }

    /// Build a scale matrix.
    #[inline]
    pub fn from_scale(s: Vec3) -> Self {
        Self::from_cols(
            [s.x, 0.0, 0.0],
            [0.0, s.y, 0.0],
            [0.0, 0.0, s.z],
        )
    }

    // ── Operations ────────────────────────────────────────────────────────────

    pub fn transpose(self) -> Self {
        let c = &self.cols;
        Self::from_cols(
            [c[0][0], c[1][0], c[2][0]],
            [c[0][1], c[1][1], c[2][1]],
            [c[0][2], c[1][2], c[2][2]],
        )
    }

    pub fn determinant(self) -> f32 {
        let c = &self.cols;
        c[0][0] * (c[1][1]*c[2][2] - c[2][1]*c[1][2])
       -c[1][0] * (c[0][1]*c[2][2] - c[2][1]*c[0][2])
       +c[2][0] * (c[0][1]*c[1][2] - c[1][1]*c[0][2])
    }

    pub fn inverse(self) -> Option<Self> {
        let det = self.determinant();
        if det.abs() < EPSILON { return None; }
        let id = 1.0 / det;
        let c = &self.cols;
        Some(Self::from_cols(
            [
                 (c[1][1]*c[2][2] - c[2][1]*c[1][2]) * id,
                -(c[0][1]*c[2][2] - c[2][1]*c[0][2]) * id,
                 (c[0][1]*c[1][2] - c[1][1]*c[0][2]) * id,
            ],
            [
                -(c[1][0]*c[2][2] - c[2][0]*c[1][2]) * id,
                 (c[0][0]*c[2][2] - c[2][0]*c[0][2]) * id,
                -(c[0][0]*c[1][2] - c[1][0]*c[0][2]) * id,
            ],
            [
                 (c[1][0]*c[2][1] - c[2][0]*c[1][1]) * id,
                -(c[0][0]*c[2][1] - c[2][0]*c[0][1]) * id,
                 (c[0][0]*c[1][1] - c[1][0]*c[0][1]) * id,
            ],
        ))
    }

    /// Compute the normal matrix for a model transform.
    ///
    /// The normal matrix is the inverse-transpose of the upper-left 3×3 of the
    /// model matrix. Required when the model has non-uniform scale, so normals
    /// transform correctly.
    pub fn normal_matrix(model: &Mat4) -> Option<Self> {
        Self::from_mat4(model).inverse().map(|m| m.transpose())
    }

    /// Transform a Vec3.
    #[inline]
    pub fn transform(self, v: Vec3) -> Vec3 {
        let c = &self.cols;
        Vec3::new(
            c[0][0]*v.x + c[1][0]*v.y + c[2][0]*v.z,
            c[0][1]*v.x + c[1][1]*v.y + c[2][1]*v.z,
            c[0][2]*v.x + c[1][2]*v.y + c[2][2]*v.z,
        )
    }

    // ── Column / row accessors ─────────────────────────────────────────────────

    #[inline]
    pub fn col(&self, index: usize) -> Vec3 {
        Vec3::new(self.cols[index][0], self.cols[index][1], self.cols[index][2])
    }

    #[inline]
    pub fn row(&self, index: usize) -> Vec3 {
        Vec3::new(self.cols[0][index], self.cols[1][index], self.cols[2][index])
    }
}

// ── Operator impls ─────────────────────────────────────────────────────────────

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

impl Mul<Vec3> for Mat3 {
    type Output = Vec3;
    #[inline]
    fn mul(self, v: Vec3) -> Vec3 { self.transform(v) }
}

impl fmt::Display for Mat3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let c = &self.cols;
        for r in 0..3 {
            writeln!(f, "  [{:8.4}  {:8.4}  {:8.4}]",
                c[0][r], c[1][r], c[2][r])?;
        }
        Ok(())
    }
      }
