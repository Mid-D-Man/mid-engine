// crates/mid-math/src/f32/mat3.rs
//! Mat3 — always scalar, 3×3 column-major.
//! Serves as both 3D rotation/scale matrix and 2D affine (homogeneous 3×3).

use core::fmt;
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use crate::{Vec2, Vec3, EPSILON};

/// 3×3 column-major matrix. 36 bytes. Always scalar on all targets.
///
/// Storage: `cols[col_idx][row_idx]` — three packed `[f32; 3]` column arrays.
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

    // ── Constructors ──────────────────────────────────────────────────────────

    /// Build from three column Vec3s.
    #[inline]
    pub fn from_cols(x: Vec3, y: Vec3, z: Vec3) -> Self {
        Self { cols: [
            [x.x, x.y, x.z],
            [y.x, y.y, y.z],
            [z.x, z.y, z.z],
        ]}
    }

    /// Build from 9 floats in column-major order.
    #[inline]
    pub fn from_cols_array(m: &[f32; 9]) -> Self {
        Self { cols: [
            [m[0], m[1], m[2]],
            [m[3], m[4], m[5]],
            [m[6], m[7], m[8]],
        ]}
    }

    /// Flatten to column-major `[f32; 9]`.
    #[inline]
    pub fn to_cols_array(self) -> [f32; 9] {
        let c = self.cols;
        [c[0][0],c[0][1],c[0][2], c[1][0],c[1][1],c[1][2], c[2][0],c[2][1],c[2][2]]
    }

    #[inline] pub fn from_cols_array_2d(m: &[[f32; 3]; 3]) -> Self { Self { cols: *m } }
    #[inline] pub fn to_cols_array_2d(self) -> [[f32; 3]; 3] { self.cols }

    /// Write 9 floats to slice in column-major order. Panics if `slice.len() < 9`.
    #[inline]
    pub fn write_cols_to_slice(self, s: &mut [f32]) {
        let c = self.cols;
        s[0]=c[0][0]; s[1]=c[0][1]; s[2]=c[0][2];
        s[3]=c[1][0]; s[4]=c[1][1]; s[5]=c[1][2];
        s[6]=c[2][0]; s[7]=c[2][1]; s[8]=c[2][2];
    }

    /// Diagonal matrix — off-diagonals zero.
    #[inline]
    pub fn from_diagonal(d: Vec3) -> Self {
        Self { cols: [[d.x,0.0,0.0],[0.0,d.y,0.0],[0.0,0.0,d.z]] }
    }

    /// Embed a `Mat2` in the upper-left — third row/col is identity.
    #[inline]
    pub fn from_mat2(m: crate::Mat2) -> Self {
        Self { cols: [
            [m.x_axis.x, m.x_axis.y, 0.0],
            [m.y_axis.x, m.y_axis.y, 0.0],
            [0.0,        0.0,        1.0],
        ]}
    }

    /// Extract upper-left 3×3 from a Mat4.
    #[inline]
    pub fn from_mat4(m: crate::Mat4) -> Self {
        Self { cols: [
            [m.x_axis.x, m.x_axis.y, m.x_axis.z],
            [m.y_axis.x, m.y_axis.y, m.y_axis.z],
            [m.z_axis.x, m.z_axis.y, m.z_axis.z],
        ]}
    }

    // ── 3D rotation constructors ───────────────────────────────────────────────

    /// Build from a normalized quaternion. Fields assumed public `(x, y, z, w)`.
    #[inline]
    pub fn from_quat(q: crate::Quat) -> Self {
        let x2=q.x+q.x; let y2=q.y+q.y; let z2=q.z+q.z;
        let xx=q.x*x2; let xy=q.x*y2; let xz=q.x*z2;
        let yy=q.y*y2; let yz=q.y*z2; let zz=q.z*z2;
        let wx=q.w*x2; let wy=q.w*y2; let wz=q.w*z2;
        Self { cols: [
            [1.0-(yy+zz), xy+wz,       xz-wy       ],
            [xy-wz,       1.0-(xx+zz), yz+wx       ],
            [xz+wy,       yz-wx,       1.0-(xx+yy) ],
        ]}
    }

    /// Build from an axis (need not be unit) and angle in radians (Rodrigues).
    #[inline]
    pub fn from_axis_angle(axis: Vec3, angle: f32) -> Self {
        let (s, c) = angle.sin_cos();
        let t = 1.0 - c;
        let k = axis.normalize();
        let (x, y, z) = (k.x, k.y, k.z);
        Self { cols: [
            [t*x*x+c,   t*x*y+s*z, t*x*z-s*y],
            [t*x*y-s*z, t*y*y+c,   t*y*z+s*x],
            [t*x*z+s*y, t*y*z-s*x, t*z*z+c  ],
        ]}
    }

    #[inline]
    pub fn from_rotation_x(angle: f32) -> Self {
        let (s, c) = angle.sin_cos();
        Self { cols: [[1.0,0.0,0.0],[0.0,c,s],[0.0,-s,c]] }
    }
    #[inline]
    pub fn from_rotation_y(angle: f32) -> Self {
        let (s, c) = angle.sin_cos();
        Self { cols: [[c,0.0,-s],[0.0,1.0,0.0],[s,0.0,c]] }
    }
    #[inline]
    pub fn from_rotation_z(angle: f32) -> Self {
        let (s, c) = angle.sin_cos();
        Self { cols: [[c,s,0.0],[-s,c,0.0],[0.0,0.0,1.0]] }
    }

    /// 3D non-uniform scale.
    #[inline] pub fn from_scale(scale: Vec3) -> Self { Self::from_diagonal(scale) }

    // ── 2D affine constructors (Mat3 as homogeneous 2D transform) ─────────────

    /// 2D CCW rotation (column 2 = translation zero).
    #[inline]
    pub fn from_angle(angle: f32) -> Self {
        let (s, c) = angle.sin_cos();
        Self { cols: [[c,s,0.0],[-s,c,0.0],[0.0,0.0,1.0]] }
    }

    /// 2D translation embedded in column 2.
    #[inline]
    pub fn from_translation(t: Vec2) -> Self {
        Self { cols: [[1.0,0.0,0.0],[0.0,1.0,0.0],[t.x,t.y,1.0]] }
    }

    /// 2D non-uniform scale.
    #[inline]
    pub fn from_scale_2d(s: Vec2) -> Self {
        Self { cols: [[s.x,0.0,0.0],[0.0,s.y,0.0],[0.0,0.0,1.0]] }
    }

    /// Full 2D affine: scale → rotate → translate (scale applied first).
    #[inline]
    pub fn from_scale_angle_translation(scale: Vec2, angle: f32, t: Vec2) -> Self {
        let (s, c) = angle.sin_cos();
        Self { cols: [
            [c*scale.x,   s*scale.x, 0.0],
            [-s*scale.y,  c*scale.y, 0.0],
            [t.x,         t.y,       1.0],
        ]}
    }

    // ── Camera orientation (rotation-only, no translation) ────────────────────

    /// Right-handed look-to rotation matrix (world → camera orientation only).
    #[inline]
    pub fn look_to_rh(dir: Vec3, up: Vec3) -> Self {
        let f = dir.normalize();
        let s = f.cross(up).normalize();
        let u = s.cross(f);
        Self::from_cols(
            Vec3::new(s.x, u.x, -f.x),
            Vec3::new(s.y, u.y, -f.y),
            Vec3::new(s.z, u.z, -f.z),
        )
    }
    #[inline] pub fn look_to_lh(dir: Vec3, up: Vec3) -> Self { Self::look_to_rh(-dir, up) }
    #[inline] pub fn look_at_rh(eye: Vec3, center: Vec3, up: Vec3) -> Self { Self::look_to_rh(center-eye, up) }
    #[inline] pub fn look_at_lh(eye: Vec3, center: Vec3, up: Vec3) -> Self { Self::look_to_lh(center-eye, up) }

    // ── Column / row accessors ─────────────────────────────────────────────────

    /// Column `i` as Vec3 (0-2). Panics if `i >= 3`.
    #[inline]
    pub fn col(&self, i: usize) -> Vec3 {
        assert!(i < 3, "Mat3::col index {i} out of bounds");
        Vec3::new(self.cols[i][0], self.cols[i][1], self.cols[i][2])
    }

    /// Set column `i` from a Vec3. Panics if `i >= 3`.
    #[inline]
    pub fn set_col(&mut self, i: usize, v: Vec3) {
        assert!(i < 3, "Mat3::set_col index {i} out of bounds");
        self.cols[i] = [v.x, v.y, v.z];
    }

    /// Row `i` as Vec3 (assembled — rows are not contiguous). Panics if `i >= 3`.
    #[inline]
    pub fn row(&self, i: usize) -> Vec3 {
        assert!(i < 3, "Mat3::row index {i} out of bounds");
        Vec3::new(self.cols[0][i], self.cols[1][i], self.cols[2][i])
    }

    /// Main diagonal `(m00, m11, m22)`.
    #[inline]
    pub fn diagonal(&self) -> Vec3 { Vec3::new(self.cols[0][0], self.cols[1][1], self.cols[2][2]) }

    // ── Core matrix operations ─────────────────────────────────────────────────

    #[inline]
    pub fn transpose(self) -> Self {
        let c = self.cols;
        Self { cols: [
            [c[0][0], c[1][0], c[2][0]],
            [c[0][1], c[1][1], c[2][1]],
            [c[0][2], c[1][2], c[2][2]],
        ]}
    }

    #[inline]
    pub fn determinant(self) -> f32 {
        let c = self.cols;
        c[0][0]*(c[1][1]*c[2][2] - c[2][1]*c[1][2])
      - c[1][0]*(c[0][1]*c[2][2] - c[2][1]*c[0][2])
      + c[2][0]*(c[0][1]*c[1][2] - c[1][1]*c[0][2])
    }

    /// Try to invert. Returns `None` if singular (|det| < ε).
    #[inline]
    pub fn try_inverse(self) -> Option<Self> {
        let c = self.cols;
        let det = c[0][0]*(c[1][1]*c[2][2] - c[2][1]*c[1][2])
                - c[1][0]*(c[0][1]*c[2][2] - c[2][1]*c[0][2])
                + c[2][0]*(c[0][1]*c[1][2] - c[1][1]*c[0][2]);
        if det.abs() < EPSILON { return None; }
        let r = 1.0 / det;
        Some(Self { cols: [
            [   // col 0 of inverse
                 (c[1][1]*c[2][2] - c[2][1]*c[1][2]) * r,
                -(c[0][1]*c[2][2] - c[2][1]*c[0][2]) * r,
                 (c[0][1]*c[1][2] - c[1][1]*c[0][2]) * r,
            ],
            [   // col 1
                -(c[1][0]*c[2][2] - c[2][0]*c[1][2]) * r,
                 (c[0][0]*c[2][2] - c[2][0]*c[0][2]) * r,
                -(c[0][0]*c[1][2] - c[1][0]*c[0][2]) * r,
            ],
            [   // col 2
                 (c[1][0]*c[2][1] - c[2][0]*c[1][1]) * r,
                -(c[0][0]*c[2][1] - c[2][0]*c[0][1]) * r,
                 (c[0][0]*c[1][1] - c[1][0]*c[0][1]) * r,
            ],
        ]})
    }

    /// Alias for `try_inverse` (original mid-math name).
    #[inline] pub fn inverse(self) -> Option<Self> { self.try_inverse() }
    #[inline] pub fn inverse_or_zero(self) -> Self { self.try_inverse().unwrap_or(Self::ZERO) }

    /// Normal matrix — inverse-transpose of the upper-left 3×3 of a model Mat4.
    ///
    /// Used every frame in shaders to correctly transform surface normals when
    /// the model matrix contains non-uniform scale. Returns `None` if the 3×3
    /// sub-matrix is singular (i.e. a degenerate transform with zero scale on
    /// any axis).
    ///
    /// Equivalent to `Mat3::from_mat4(model).inverse().map(|m| m.transpose())`.
    #[inline]
    pub fn normal_matrix(model: &crate::Mat4) -> Option<Self> {
        Self::from_mat4(*model).try_inverse().map(|m| m.transpose())
    }

    // ── Transform ─────────────────────────────────────────────────────────────

    /// `M * v` — column vector transform.
    #[inline]
    pub fn mul_vec3(self, v: Vec3) -> Vec3 {
        let c = self.cols;
        Vec3::new(
            c[0][0]*v.x + c[1][0]*v.y + c[2][0]*v.z,
            c[0][1]*v.x + c[1][1]*v.y + c[2][1]*v.z,
            c[0][2]*v.x + c[1][2]*v.y + c[2][2]*v.z,
        )
    }

    /// Alias for `mul_vec3`. Matches the naming convention used in the bench
    /// harness and mirrors `DMat3::transform`.
    #[inline]
    pub fn transform(self, v: Vec3) -> Vec3 { self.mul_vec3(v) }

    /// `M^T * v` — cheaper than `.transpose().mul_vec3(v)`.
    #[inline]
    pub fn mul_transpose_vec3(self, v: Vec3) -> Vec3 {
        let c = self.cols;
        Vec3::new(
            c[0][0]*v.x + c[0][1]*v.y + c[0][2]*v.z,
            c[1][0]*v.x + c[1][1]*v.y + c[1][2]*v.z,
            c[2][0]*v.x + c[2][1]*v.y + c[2][2]*v.z,
        )
    }

    /// Matrix × matrix.
    #[inline]
    pub fn mul_mat3(self, rhs: Self) -> Self {
        // AVX + FMA path: 2 output columns per YMM pass, ~2× fewer SIMD ops.
        #[cfg(all(
            any(target_arch = "x86", target_arch = "x86_64"),
            target_feature = "avx",
            target_feature = "fma",
        ))]
        {
            // SAFETY: cfg gate proves avx+fma are available at compile time.
            return unsafe { crate::f32::avx::mat3::mat3_mul_avx(self, rhs) };
        }
        // Scalar fallback — three matrix-vector products.
        Self::from_cols(
            self.mul_vec3(rhs.col(0)),
            self.mul_vec3(rhs.col(1)),
            self.mul_vec3(rhs.col(2)),
        )
    }

    /// Scale every element by `s`.
    #[inline]
    pub fn mul_scalar(self, s: f32) -> Self {
        let c = self.cols;
        Self { cols: [
            [c[0][0]*s,c[0][1]*s,c[0][2]*s],
            [c[1][0]*s,c[1][1]*s,c[1][2]*s],
            [c[2][0]*s,c[2][1]*s,c[2][2]*s],
        ]}
    }
    #[inline] pub fn div_scalar(self, s: f32) -> Self { self.mul_scalar(1.0/s) }

    /// Scale column `i` by `scale[i]`. Equivalent to `self * Mat3::from_diagonal(scale)`.
    #[inline]
    pub fn mul_diagonal_scale(self, scale: Vec3) -> Self {
        Self::from_cols(self.col(0)*scale.x, self.col(1)*scale.y, self.col(2)*scale.z)
    }

    #[inline]
    pub fn add_mat3(self, rhs: Self) -> Self {
        let (a, b) = (self.cols, rhs.cols);
        Self { cols: [
            [a[0][0]+b[0][0],a[0][1]+b[0][1],a[0][2]+b[0][2]],
            [a[1][0]+b[1][0],a[1][1]+b[1][1],a[1][2]+b[1][2]],
            [a[2][0]+b[2][0],a[2][1]+b[2][1],a[2][2]+b[2][2]],
        ]}
    }
    #[inline]
    pub fn sub_mat3(self, rhs: Self) -> Self {
        let (a, b) = (self.cols, rhs.cols);
        Self { cols: [
            [a[0][0]-b[0][0],a[0][1]-b[0][1],a[0][2]-b[0][2]],
            [a[1][0]-b[1][0],a[1][1]-b[1][1],a[1][2]-b[1][2]],
            [a[2][0]-b[2][0],a[2][1]-b[2][1],a[2][2]-b[2][2]],
        ]}
    }

    #[inline]
    pub fn abs(self) -> Self {
        let c = self.cols;
        Self { cols: [
            [c[0][0].abs(),c[0][1].abs(),c[0][2].abs()],
            [c[1][0].abs(),c[1][1].abs(),c[1][2].abs()],
            [c[2][0].abs(),c[2][1].abs(),c[2][2].abs()],
        ]}
    }
    #[inline]
    pub fn recip(self) -> Self {
        let c = self.cols;
        Self { cols: [
            [c[0][0].recip(),c[0][1].recip(),c[0][2].recip()],
            [c[1][0].recip(),c[1][1].recip(),c[1][2].recip()],
            [c[2][0].recip(),c[2][1].recip(),c[2][2].recip()],
        ]}
    }

    // ── 2D affine transforms ──────────────────────────────────────────────────

    /// Transform a 2D point (applies full affine including translation in col 2).
    #[inline]
    pub fn transform_point2(self, p: Vec2) -> Vec2 {
        let c = self.cols;
        Vec2::new(
            c[0][0]*p.x + c[1][0]*p.y + c[2][0],
            c[0][1]*p.x + c[1][1]*p.y + c[2][1],
        )
    }

    /// Transform a 2D direction (linear part only, ignores translation column).
    #[inline]
    pub fn transform_vector2(self, v: Vec2) -> Vec2 {
        let c = self.cols;
        Vec2::new(
            c[0][0]*v.x + c[1][0]*v.y,
            c[0][1]*v.x + c[1][1]*v.y,
        )
    }

    // ── Quaternion conversion ─────────────────────────────────────────────────

    /// Convert a rotation matrix to quaternion (Shepperd's largest-component method).
    /// Matrix must be orthonormal with det = +1.
    #[inline]
    pub fn to_quat(self) -> crate::Quat {
        let c = self.cols;
        let trace = c[0][0] + c[1][1] + c[2][2];
        // Shepperd: pick the largest diagonal to maximise numerical stability
        if trace > 0.0 {
            let four_w = 2.0 * (trace + 1.0).sqrt();
            crate::Quat::new(
                (c[1][2]-c[2][1]) / four_w,
                (c[2][0]-c[0][2]) / four_w,
                (c[0][1]-c[1][0]) / four_w,
                four_w * 0.25,
            )
        } else if c[0][0] > c[1][1] && c[0][0] > c[2][2] {
            let four_x = 2.0 * (1.0 + c[0][0] - c[1][1] - c[2][2]).sqrt();
            crate::Quat::new(
                four_x * 0.25,
                (c[0][1]+c[1][0]) / four_x,
                (c[2][0]+c[0][2]) / four_x,
                (c[1][2]-c[2][1]) / four_x,
            )
        } else if c[1][1] > c[2][2] {
            let four_y = 2.0 * (1.0 + c[1][1] - c[0][0] - c[2][2]).sqrt();
            crate::Quat::new(
                (c[0][1]+c[1][0]) / four_y,
                four_y * 0.25,
                (c[1][2]+c[2][1]) / four_y,
                (c[2][0]-c[0][2]) / four_y,
            )
        } else {
            let four_z = 2.0 * (1.0 + c[2][2] - c[0][0] - c[1][1]).sqrt();
            crate::Quat::new(
                (c[2][0]+c[0][2]) / four_z,
                (c[1][2]+c[2][1]) / four_z,
                four_z * 0.25,
                (c[0][1]-c[1][0]) / four_z,
            )
        }
    }

    // ── Embed / extract ───────────────────────────────────────────────────────

    /// Embed this 3×3 in the top-left of a Mat4 (rest stays identity).
    #[inline]
    pub fn to_mat4(self) -> crate::Mat4 {
        crate::Mat4::from_cols(
            [self.cols[0][0], self.cols[0][1], self.cols[0][2], 0.0],
            [self.cols[1][0], self.cols[1][1], self.cols[1][2], 0.0],
            [self.cols[2][0], self.cols[2][1], self.cols[2][2], 0.0],
            [0.0, 0.0, 0.0, 1.0],
        )
    }

    // ── Predicates ────────────────────────────────────────────────────────────

    #[inline] pub fn is_finite(self) -> bool { self.cols.iter().flatten().all(|v| v.is_finite()) }
    #[inline] pub fn is_nan(self)    -> bool { self.cols.iter().flatten().any(|v| v.is_nan())    }

    #[inline]
    pub fn approx_eq(self, rhs: Self) -> bool { self.abs_diff_eq(rhs, EPSILON) }

    #[inline]
    pub fn abs_diff_eq(self, rhs: Self, max_abs_diff: f32) -> bool {
        for i in 0..3 { for j in 0..3 {
            if (self.cols[i][j] - rhs.cols[i][j]).abs() >= max_abs_diff { return false; }
        }}
        true
    }
}

// ── Default / Display ─────────────────────────────────────────────────────────

impl Default for Mat3 { fn default() -> Self { Self::IDENTITY } }

impl fmt::Display for Mat3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let c = self.cols;
        write!(f, "[({},{},{}), ({},{},{}), ({},{},{})]",
            c[0][0],c[0][1],c[0][2], c[1][0],c[1][1],c[1][2], c[2][0],c[2][1],c[2][2])
    }
}

// ── Operators ─────────────────────────────────────────────────────────────────

impl Mul for Mat3 {
    type Output = Self;
    #[inline] fn mul(self, rhs: Self) -> Self { self.mul_mat3(rhs) }
}
impl MulAssign for Mat3 {
    #[inline] fn mul_assign(&mut self, rhs: Self) { *self = self.mul_mat3(rhs); }
}
impl Mul<Vec3> for Mat3 {
    type Output = Vec3;
    #[inline] fn mul(self, rhs: Vec3) -> Vec3 { self.mul_vec3(rhs) }
}
impl Mul<f32> for Mat3 {
    type Output = Self;
    #[inline] fn mul(self, rhs: f32) -> Self { self.mul_scalar(rhs) }
}
impl Mul<Mat3> for f32 {
    type Output = Mat3;
    #[inline] fn mul(self, rhs: Mat3) -> Mat3 { rhs.mul_scalar(self) }
}
impl MulAssign<f32> for Mat3 {
    #[inline] fn mul_assign(&mut self, rhs: f32) { *self = self.mul_scalar(rhs); }
}
impl Add for Mat3 {
    type Output = Self;
    #[inline] fn add(self, rhs: Self) -> Self { self.add_mat3(rhs) }
}
impl AddAssign for Mat3 {
    #[inline] fn add_assign(&mut self, rhs: Self) { *self = self.add_mat3(rhs); }
}
impl Sub for Mat3 {
    type Output = Self;
    #[inline] fn sub(self, rhs: Self) -> Self { self.sub_mat3(rhs) }
}
impl SubAssign for Mat3 {
    #[inline] fn sub_assign(&mut self, rhs: Self) { *self = self.sub_mat3(rhs); }
}
impl Neg for Mat3 {
    type Output = Self;
    #[inline] fn neg(self) -> Self { self.mul_scalar(-1.0) }
}

// ── Conversions ───────────────────────────────────────────────────────────────

impl From<[[f32;3];3]> for Mat3 { fn from(m: [[f32;3];3]) -> Self { Self::from_cols_array_2d(&m) } }
impl From<Mat3> for [[f32;3];3] { fn from(m: Mat3) -> Self { m.to_cols_array_2d() } }
impl From<[f32;9]> for Mat3 { fn from(m: [f32;9]) -> Self { Self::from_cols_array(&m) } }
impl From<Mat3> for [f32;9] { fn from(m: Mat3) -> Self { m.to_cols_array() } }
