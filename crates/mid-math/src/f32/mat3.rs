// crates/mid-math/src/f32/mat3.rs
// Updated for Build 8: use Vec4 field names (x_axis, y_axis, z_axis)
// instead of cols[i][j] indexing for from_mat4 and normal_matrix.

use core::fmt;
use core::ops::Mul;
use crate::{Vec3, Mat4};
use crate::EPSILON;

/// 3×3 column-major matrix. 36 bytes. Always scalar on all platforms.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct Mat3 {
    pub cols: [[f32; 3]; 3],
}

impl Mat3 {
    pub const ZERO: Self = Self { cols: [[0.0; 3]; 3] };
    pub const IDENTITY: Self = Self { cols: [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]};

    #[inline]
    pub fn from_cols(c0: [f32; 3], c1: [f32; 3], c2: [f32; 3]) -> Self {
        Self { cols: [c0, c1, c2] }
    }

    /// Extract upper-left 3×3 from a Mat4.
    ///
    /// Updated for Build 8: accesses the named Vec4 fields (x_axis, y_axis, z_axis)
    /// via their Deref components rather than the removed cols[i][j] indexing.
    #[inline]
    pub fn from_mat4(m: &Mat4) -> Self {
        Self::from_cols(
            [m.x_axis.x, m.x_axis.y, m.x_axis.z],
            [m.y_axis.x, m.y_axis.y, m.y_axis.z],
            [m.z_axis.x, m.z_axis.y, m.z_axis.z],
        )
    }

    #[inline]
    pub fn from_vecs(c0: Vec3, c1: Vec3, c2: Vec3) -> Self {
        Self::from_cols(
            [c0.x, c0.y, c0.z],
            [c1.x, c1.y, c1.z],
            [c2.x, c2.y, c2.z],
        )
    }

    #[inline]
    pub fn from_rotation_z(angle: f32) -> Self {
        let (s, c) = (angle.sin(), angle.cos());
        Self::from_cols(
            [ c,  s, 0.0],
            [-s,  c, 0.0],
            [0.0, 0.0, 1.0],
        )
    }

    #[inline]
    pub fn from_scale(s: Vec3) -> Self {
        Self::from_cols([s.x, 0.0, 0.0], [0.0, s.y, 0.0], [0.0, 0.0, s.z])
    }

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

    /// Inverse via cross-product method — leverages SSE2 Vec3 cross/dot
    /// instead of the old scalar cofactor expansion.
    ///
    /// ## Algorithm
    /// ```text
    /// col0 = x, col1 = y, col2 = z
    ///
    /// tmp0 = y × z   ← cofactor column 0 of adj(M)
    /// tmp1 = z × x   ← cofactor column 1 of adj(M)
    /// tmp2 = x × y   ← cofactor column 2 of adj(M)
    /// det  = x · tmp0
    ///
    /// M⁻¹ = [tmp0 tmp1 tmp2]ᵀ / det   (transpose = column-major storage)
    /// ```
    ///
    /// ## Instruction count (x86/x86_64 with SSE2 Vec3)
    /// 3 × cross (~6 SSE2 each) + 1 × dot (~3 SSE2) + 9 scalar muls
    /// ≈ 30 SSE2 + 9 scalar vs old ~28 all-scalar — but SSE2 crosses
    /// process 3 floats in parallel, so wall-clock latency drops significantly.
    #[inline]
    pub fn inverse(self) -> Option<Self> {
        use crate::Vec3;
        // Load the three columns into SSE2-backed Vec3 (one _mm_set_ps each).
        let x = Vec3::new(self.cols[0][0], self.cols[0][1], self.cols[0][2]);
        let y = Vec3::new(self.cols[1][0], self.cols[1][1], self.cols[1][2]);
        let z = Vec3::new(self.cols[2][0], self.cols[2][1], self.cols[2][2]);

        // Cofactor rows of the adjugate via cross products.
        let tmp0 = y.cross(z); // row 0 of adj(M)
        let tmp1 = z.cross(x); // row 1 of adj(M)
        let tmp2 = x.cross(y); // row 2 of adj(M)

        let det = x.dot(tmp0);
        if det.abs() < crate::EPSILON {
            return None;
        }
        let inv = det.recip();

        // adj(M) rows become columns of adj(M)ᵀ = M⁻¹ (column-major storage).
        //   col 0 = [tmp0.x, tmp1.x, tmp2.x] / det
        //   col 1 = [tmp0.y, tmp1.y, tmp2.y] / det
        //   col 2 = [tmp0.z, tmp1.z, tmp2.z] / det
        Some(Self::from_cols(
            [tmp0.x * inv, tmp1.x * inv, tmp2.x * inv],
            [tmp0.y * inv, tmp1.y * inv, tmp2.y * inv],
            [tmp0.z * inv, tmp1.z * inv, tmp2.z * inv],
        ))
}
    /// Normal matrix = inverse-transpose of upper-left 3×3 of the model matrix.
    pub fn normal_matrix(model: &Mat4) -> Option<Self> {
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

    #[inline]
    pub fn col(&self, i: usize) -> Vec3 {
        Vec3::new(self.cols[i][0], self.cols[i][1], self.cols[i][2])
    }

    #[inline]
    pub fn row(&self, i: usize) -> Vec3 {
        Vec3::new(self.cols[0][i], self.cols[1][i], self.cols[2][i])
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

impl Mul<Vec3> for Mat3 {
    type Output = Vec3;
    #[inline] fn mul(self, v: Vec3) -> Vec3 { self.transform(v) }
}

impl fmt::Display for Mat3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let c = &self.cols;
        for r in 0..3 {
            writeln!(f, "  [{:8.4}  {:8.4}  {:8.4}]", c[0][r], c[1][r], c[2][r])?;
        }
        Ok(())
    }
         }
