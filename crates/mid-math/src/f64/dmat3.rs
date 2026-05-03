// crates/mid-math/src/f64/dmat3.rs
//! Double-precision 3×3 column-major matrix. 72 bytes (no padding), align(8).
//!
//! Used for normal matrices, 3D rotation-only transforms, and the inner
//! 3×3 block of DAffine3. Scalar only.

use core::fmt;
use core::ops::Mul;

use super::dvec3::DVec3;
use super::dmat4::DMat4;
use super::dvec2::DEPSILON;

/// 3×3 column-major double-precision matrix. 72 bytes, align(8).
///
/// `cols[c][r]` = element at column `c`, row `r`.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct DMat3 {
    pub cols: [[f64; 3]; 3],
}

impl DMat3 {
    pub const ZERO: Self = Self { cols: [[0.0; 3]; 3] };
    pub const IDENTITY: Self = Self { cols: [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]};

    // ── Constructors ──────────────────────────────────────────────────────────

    #[inline]
    pub fn from_cols(c0: [f64;3], c1: [f64;3], c2: [f64;3]) -> Self {
        Self { cols: [c0, c1, c2] }
    }

    #[inline]
    pub fn from_vecs(c0: DVec3, c1: DVec3, c2: DVec3) -> Self {
        Self::from_cols(
            [c0.x, c0.y, c0.z],
            [c1.x, c1.y, c1.z],
            [c2.x, c2.y, c2.z],
        )
    }

    /// Extract upper-left 3×3 from a DMat4.
    #[inline]
    pub fn from_mat4(m: &DMat4) -> Self {
        Self::from_cols(
            [m.cols[0][0], m.cols[0][1], m.cols[0][2]],
            [m.cols[1][0], m.cols[1][1], m.cols[1][2]],
            [m.cols[2][0], m.cols[2][1], m.cols[2][2]],
        )
    }

    #[inline]
    pub fn from_scale(s: DVec3) -> Self {
        Self::from_cols(
            [s.x, 0.0, 0.0],
            [0.0, s.y, 0.0],
            [0.0, 0.0, s.z],
        )
    }

    #[inline]
    pub fn from_rotation_z(angle: f64) -> Self {
        let (s, c) = angle.sin_cos();
        Self::from_cols([ c,  s, 0.0], [-s,  c, 0.0], [0.0, 0.0, 1.0])
    }

    // ── Ops ───────────────────────────────────────────────────────────────────

    pub fn transpose(self) -> Self {
        let c = &self.cols;
        Self::from_cols(
            [c[0][0], c[1][0], c[2][0]],
            [c[0][1], c[1][1], c[2][1]],
            [c[0][2], c[1][2], c[2][2]],
        )
    }

    pub fn determinant(self) -> f64 {
        let c = &self.cols;
        c[0][0] * (c[1][1]*c[2][2] - c[2][1]*c[1][2])
       -c[1][0] * (c[0][1]*c[2][2] - c[2][1]*c[0][2])
       +c[2][0] * (c[0][1]*c[1][2] - c[1][1]*c[0][2])
    }

    pub fn inverse(self) -> Option<Self> {
        let det = self.determinant();
        if det.abs() < DEPSILON { return None; }
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

    /// Normal matrix = inverse-transpose of upper-left 3×3 of the model matrix.
    ///
    /// Use this to correctly transform normals when non-uniform scale is present.
    pub fn normal_matrix(model: &DMat4) -> Option<Self> {
        Self::from_mat4(model).inverse().map(|m| m.transpose())
    }

    pub fn transform(self, v: DVec3) -> DVec3 {
        let c = &self.cols;
        DVec3::new(
            c[0][0]*v.x + c[1][0]*v.y + c[2][0]*v.z,
            c[0][1]*v.x + c[1][1]*v.y + c[2][1]*v.z,
            c[0][2]*v.x + c[1][2]*v.y + c[2][2]*v.z,
        )
    }

    #[inline]
    pub fn col(&self, i: usize) -> DVec3 {
        DVec3::new(self.cols[i][0], self.cols[i][1], self.cols[i][2])
    }

    #[inline]
    pub fn row(&self, i: usize) -> DVec3 {
        DVec3::new(self.cols[0][i], self.cols[1][i], self.cols[2][i])
    }

    pub fn is_finite(self) -> bool {
        self.cols.iter().flatten().all(|v| v.is_finite())
    }

    /// Lossy cast to single-precision `Mat3`.
    pub fn as_mat3(self) -> crate::Mat3 {
        crate::Mat3::from_cols(
            [self.cols[0][0] as f32, self.cols[0][1] as f32, self.cols[0][2] as f32],
            [self.cols[1][0] as f32, self.cols[1][1] as f32, self.cols[1][2] as f32],
            [self.cols[2][0] as f32, self.cols[2][1] as f32, self.cols[2][2] as f32],
        )
    }
}

impl Default for DMat3 { fn default() -> Self { Self::IDENTITY } }

impl Mul for DMat3 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        let (a, b) = (&self.cols, &rhs.cols);
        Self::from_cols(
            [
                a[0][0]*b[0][0]+a[1][0]*b[0][1]+a[2][0]*b[0][2],
                a[0][1]*b[0][0]+a[1][1]*b[0][1]+a[2][1]*b[0][2],
                a[0][2]*b[0][0]+a[1][2]*b[0][1]+a[2][2]*b[0][2],
            ],
            [
                a[0][0]*b[1][0]+a[1][0]*b[1][1]+a[2][0]*b[1][2],
                a[0][1]*b[1][0]+a[1][1]*b[1][1]+a[2][1]*b[1][2],
                a[0][2]*b[1][0]+a[1][2]*b[1][1]+a[2][2]*b[1][2],
            ],
            [
                a[0][0]*b[2][0]+a[1][0]*b[2][1]+a[2][0]*b[2][2],
                a[0][1]*b[2][0]+a[1][1]*b[2][1]+a[2][1]*b[2][2],
                a[0][2]*b[2][0]+a[1][2]*b[2][1]+a[2][2]*b[2][2],
            ],
        )
    }
}

impl Mul<DVec3> for DMat3 {
    type Output = DVec3;
    #[inline] fn mul(self, v: DVec3) -> DVec3 { self.transform(v) }
}

impl fmt::Display for DMat3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let c = &self.cols;
        for r in 0..3 {
            writeln!(f, "  [{:12.6}  {:12.6}  {:12.6}]",
                c[0][r], c[1][r], c[2][r])?;
        }
        Ok(())
    }
                                     }
