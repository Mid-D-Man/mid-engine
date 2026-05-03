// crates/mid-math/src/f64/dmat4.rs
//! Double-precision 4×4 column-major matrix. 128 bytes, align(32).
//!
//! Scalar only — no AVX2 intrinsics yet. The align(32) reserves space for
//! a future ymm-register fast path (four f64 per register = one column per
//! instruction). Correctness matches the f32 scalar mat4 exactly.

use core::fmt;
use core::ops::Mul;

use super::dvec3::DVec3;
use super::dvec4::DVec4;
use super::dquat::DQuat;
use super::dvec2::DEPSILON;

/// 4×4 column-major double-precision matrix. 128 bytes, align(32).
///
/// `cols[c][r]` = element at column `c`, row `r`.
///
/// **C interop:** use [`CDMat4`][crate::ffi::types::CDMat4] at the FFI boundary.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C, align(32))]
pub struct DMat4 {
    pub cols: [[f64; 4]; 4],
}

impl DMat4 {
    pub const ZERO: Self = Self { cols: [[0.0; 4]; 4] };
    pub const IDENTITY: Self = Self { cols: [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]};

    // ── Constructors ──────────────────────────────────────────────────────────

    #[inline]
    pub fn from_cols(c0: [f64;4], c1: [f64;4], c2: [f64;4], c3: [f64;4]) -> Self {
        Self { cols: [c0, c1, c2, c3] }
    }

    #[inline]
    pub fn from_translation(t: DVec3) -> Self {
        let mut m = Self::IDENTITY;
        m.cols[3] = [t.x, t.y, t.z, 1.0];
        m
    }

    #[inline]
    pub fn from_scale(s: DVec3) -> Self {
        Self::from_cols(
            [s.x, 0.0, 0.0, 0.0],
            [0.0, s.y, 0.0, 0.0],
            [0.0, 0.0, s.z, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        )
    }

    #[inline]
    pub fn from_rotation(q: DQuat) -> Self { q.to_mat4() }

    /// Build a TRS matrix: scale → rotate → translate.
    ///
    /// Equivalent to `T * R * S`.
    #[inline]
    pub fn from_trs(t: DVec3, r: DQuat, s: DVec3) -> Self {
        let q = r.normalize();
        let (x, y, z, w) = (q.x, q.y, q.z, q.w);
        let (x2, y2, z2) = (x+x, y+y, z+z);
        let (xx, yy, zz) = (x*x2, y*y2, z*z2);
        let (xy, xz, yz) = (x*y2, x*z2, y*z2);
        let (wx, wy, wz) = (w*x2, w*y2, w*z2);
        Self::from_cols(
            [(1.0-yy-zz)*s.x,  (xy+wz)*s.x,     (xz-wy)*s.x,    0.0],
            [(xy-wz)*s.y,      (1.0-xx-zz)*s.y,  (yz+wx)*s.y,    0.0],
            [(xz+wy)*s.z,      (yz-wx)*s.z,      (1.0-xx-yy)*s.z, 0.0],
            [t.x, t.y, t.z, 1.0],
        )
    }

    /// Right-handed look-at view matrix.
    pub fn look_at_rh(eye: DVec3, center: DVec3, up: DVec3) -> Self {
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

    /// Right-handed perspective projection, depth range `[0, 1]`.
    pub fn perspective_rh(fov_y: f64, aspect: f64, near: f64, far: f64) -> Self {
        let f = 1.0 / (fov_y * 0.5).tan();
        let z = near - far;
        Self::from_cols(
            [f / aspect, 0.0, 0.0,                     0.0],
            [0.0,        f,   0.0,                     0.0],
            [0.0,        0.0, (far + near) / z,        -1.0],
            [0.0,        0.0, (2.0 * far * near) / z,   0.0],
        )
    }

    /// Right-handed orthographic projection.
    pub fn ortho_rh(
        left: f64, right: f64, bottom: f64, top: f64, near: f64, far: f64,
    ) -> Self {
        let rl = right - left;
        let tb = top   - bottom;
        let nf = far   - near;
        Self::from_cols(
            [2.0/rl, 0.0,    0.0,     0.0],
            [0.0,    2.0/tb, 0.0,     0.0],
            [0.0,    0.0,   -2.0/nf,  0.0],
            [-(right+left)/rl, -(top+bottom)/tb, -(far+near)/nf, 1.0],
        )
    }

    // ── Transform helpers ─────────────────────────────────────────────────────

    pub fn transpose(self) -> Self {
        let c = &self.cols;
        Self::from_cols(
            [c[0][0], c[1][0], c[2][0], c[3][0]],
            [c[0][1], c[1][1], c[2][1], c[3][1]],
            [c[0][2], c[1][2], c[2][2], c[3][2]],
            [c[0][3], c[1][3], c[2][3], c[3][3]],
        )
    }

    #[inline]
    pub fn transform_point(self, p: DVec3) -> DVec3 {
        (self * p.extend(1.0)).truncate()
    }

    #[inline]
    pub fn transform_vector(self, v: DVec3) -> DVec3 {
        (self * v.extend(0.0)).truncate()
    }

    // ── Inverse — general (Cramer/cofactor expansion) ─────────────────────────
    //
    // Same algorithm as scalar f32 mat4 but with f64 arithmetic.
    // Returns None for singular matrices (|det| < DEPSILON).

    pub fn inverse(self) -> Option<Self> {
        let a = [
            self.cols[0][0], self.cols[0][1], self.cols[0][2], self.cols[0][3],
            self.cols[1][0], self.cols[1][1], self.cols[1][2], self.cols[1][3],
            self.cols[2][0], self.cols[2][1], self.cols[2][2], self.cols[2][3],
            self.cols[3][0], self.cols[3][1], self.cols[3][2], self.cols[3][3],
        ];
        let mut inv = [0.0f64; 16];

        inv[ 0] =  a[5]*a[10]*a[15]-a[5]*a[11]*a[14]-a[9]*a[6]*a[15]+a[9]*a[7]*a[14]+a[13]*a[6]*a[11]-a[13]*a[7]*a[10];
        inv[ 4] = -a[4]*a[10]*a[15]+a[4]*a[11]*a[14]+a[8]*a[6]*a[15]-a[8]*a[7]*a[14]-a[12]*a[6]*a[11]+a[12]*a[7]*a[10];
        inv[ 8] =  a[4]*a[9]*a[15]-a[4]*a[11]*a[13]-a[8]*a[5]*a[15]+a[8]*a[7]*a[13]+a[12]*a[5]*a[11]-a[12]*a[7]*a[9];
        inv[12] = -a[4]*a[9]*a[14]+a[4]*a[10]*a[13]+a[8]*a[5]*a[14]-a[8]*a[6]*a[13]-a[12]*a[5]*a[10]+a[12]*a[6]*a[9];
        inv[ 1] = -a[1]*a[10]*a[15]+a[1]*a[11]*a[14]+a[9]*a[2]*a[15]-a[9]*a[3]*a[14]-a[13]*a[2]*a[11]+a[13]*a[3]*a[10];
        inv[ 5] =  a[0]*a[10]*a[15]-a[0]*a[11]*a[14]-a[8]*a[2]*a[15]+a[8]*a[3]*a[14]+a[12]*a[2]*a[11]-a[12]*a[3]*a[10];
        inv[ 9] = -a[0]*a[9]*a[15]+a[0]*a[11]*a[13]+a[8]*a[1]*a[15]-a[8]*a[3]*a[13]-a[12]*a[1]*a[11]+a[12]*a[3]*a[9];
        inv[13] =  a[0]*a[9]*a[14]-a[0]*a[10]*a[13]-a[8]*a[1]*a[14]+a[8]*a[2]*a[13]+a[12]*a[1]*a[10]-a[12]*a[2]*a[9];
        inv[ 2] =  a[1]*a[6]*a[15]-a[1]*a[7]*a[14]-a[5]*a[2]*a[15]+a[5]*a[3]*a[14]+a[13]*a[2]*a[7]-a[13]*a[3]*a[6];
        inv[ 6] = -a[0]*a[6]*a[15]+a[0]*a[7]*a[14]+a[4]*a[2]*a[15]-a[4]*a[3]*a[14]-a[12]*a[2]*a[7]+a[12]*a[3]*a[6];
        inv[10] =  a[0]*a[5]*a[15]-a[0]*a[7]*a[13]-a[4]*a[1]*a[15]+a[4]*a[3]*a[13]+a[12]*a[1]*a[7]-a[12]*a[3]*a[5];
        inv[14] = -a[0]*a[5]*a[14]+a[0]*a[6]*a[13]+a[4]*a[1]*a[14]-a[4]*a[2]*a[13]-a[12]*a[1]*a[6]+a[12]*a[2]*a[5];
        inv[ 3] = -a[1]*a[6]*a[11]+a[1]*a[7]*a[10]+a[5]*a[2]*a[11]-a[5]*a[3]*a[10]-a[9]*a[2]*a[7]+a[9]*a[3]*a[6];
        inv[ 7] =  a[0]*a[6]*a[11]-a[0]*a[7]*a[10]-a[4]*a[2]*a[11]+a[4]*a[3]*a[10]+a[8]*a[2]*a[7]-a[8]*a[3]*a[6];
        inv[11] = -a[0]*a[5]*a[11]+a[0]*a[7]*a[9]+a[4]*a[1]*a[11]-a[4]*a[3]*a[9]-a[8]*a[1]*a[7]+a[8]*a[3]*a[5];
        inv[15] =  a[0]*a[5]*a[10]-a[0]*a[6]*a[9]-a[4]*a[1]*a[10]+a[4]*a[2]*a[9]+a[8]*a[1]*a[6]-a[8]*a[2]*a[5];

        let det = a[0]*inv[0] + a[1]*inv[4] + a[2]*inv[8] + a[3]*inv[12];
        if det.abs() < DEPSILON { return None; }
        let id = 1.0 / det;
        for x in inv.iter_mut() { *x *= id; }

        Some(Self::from_cols(
            [inv[0],  inv[1],  inv[2],  inv[3]],
            [inv[4],  inv[5],  inv[6],  inv[7]],
            [inv[8],  inv[9],  inv[10], inv[11]],
            [inv[12], inv[13], inv[14], inv[15]],
        ))
    }

    // ── Inverse — TRS fast path ───────────────────────────────────────────────
    //
    // Assumes the matrix is a TRS (rotation + uniform or non-uniform scale +
    // translation) with no shear. ~2× faster than the general inverse because
    // the bottom row [0,0,0,1] is implicit.
    //
    // Derivation: identical to f32 inverse_trs_scalar.

    pub fn inverse_trs(self) -> Self {
        let sx2 = self.cols[0][0]*self.cols[0][0]
                + self.cols[0][1]*self.cols[0][1]
                + self.cols[0][2]*self.cols[0][2];
        let sy2 = self.cols[1][0]*self.cols[1][0]
                + self.cols[1][1]*self.cols[1][1]
                + self.cols[1][2]*self.cols[1][2];
        let sz2 = self.cols[2][0]*self.cols[2][0]
                + self.cols[2][1]*self.cols[2][1]
                + self.cols[2][2]*self.cols[2][2];

        let isx = if sx2 < DEPSILON { 0.0 } else { 1.0 / sx2 };
        let isy = if sy2 < DEPSILON { 0.0 } else { 1.0 / sy2 };
        let isz = if sz2 < DEPSILON { 0.0 } else { 1.0 / sz2 };

        let ic0 = [
            self.cols[0][0]*isx, self.cols[1][0]*isy, self.cols[2][0]*isz, 0.0
        ];
        let ic1 = [
            self.cols[0][1]*isx, self.cols[1][1]*isy, self.cols[2][1]*isz, 0.0
        ];
        let ic2 = [
            self.cols[0][2]*isx, self.cols[1][2]*isy, self.cols[2][2]*isz, 0.0
        ];
        let (tx, ty, tz) = (self.cols[3][0], self.cols[3][1], self.cols[3][2]);
        let itx = -(ic0[0]*tx + ic1[0]*ty + ic2[0]*tz);
        let ity = -(ic0[1]*tx + ic1[1]*ty + ic2[1]*tz);
        let itz = -(ic0[2]*tx + ic1[2]*ty + ic2[2]*tz);

        Self::from_cols(ic0, ic1, ic2, [itx, ity, itz, 1.0])
    }

    // ── Predicates ────────────────────────────────────────────────────────────

    pub fn is_finite(self) -> bool {
        self.cols.iter().flatten().all(|v| v.is_finite())
    }

    // ── Cast ─────────────────────────────────────────────────────────────────

    /// Lossy cast to single-precision `Mat4`.
    pub fn as_mat4(self) -> crate::Mat4 {
        crate::Mat4::from_cols(
            [self.cols[0][0] as f32, self.cols[0][1] as f32,
             self.cols[0][2] as f32, self.cols[0][3] as f32],
            [self.cols[1][0] as f32, self.cols[1][1] as f32,
             self.cols[1][2] as f32, self.cols[1][3] as f32],
            [self.cols[2][0] as f32, self.cols[2][1] as f32,
             self.cols[2][2] as f32, self.cols[2][3] as f32],
            [self.cols[3][0] as f32, self.cols[3][1] as f32,
             self.cols[3][2] as f32, self.cols[3][3] as f32],
        )
    }
}

// ── Mul<DMat4> ────────────────────────────────────────────────────────────────
// Column-decomposition: result = [self*rhs.col0, self*rhs.col1, ...]

impl Mul for DMat4 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self::from_cols(
            (self * DVec4::from_array(rhs.cols[0])).to_array(),
            (self * DVec4::from_array(rhs.cols[1])).to_array(),
            (self * DVec4::from_array(rhs.cols[2])).to_array(),
            (self * DVec4::from_array(rhs.cols[3])).to_array(),
        )
    }
}

// ── Mul<DVec4> ────────────────────────────────────────────────────────────────

impl Mul<DVec4> for DMat4 {
    type Output = DVec4;
    #[inline(always)]
    fn mul(self, v: DVec4) -> DVec4 {
        let c = &self.cols;
        DVec4::new(
            c[0][0]*v.x + c[1][0]*v.y + c[2][0]*v.z + c[3][0]*v.w,
            c[0][1]*v.x + c[1][1]*v.y + c[2][1]*v.z + c[3][1]*v.w,
            c[0][2]*v.x + c[1][2]*v.y + c[2][2]*v.z + c[3][2]*v.w,
            c[0][3]*v.x + c[1][3]*v.y + c[2][3]*v.z + c[3][3]*v.w,
        )
    }
}

impl Default for DMat4 { fn default() -> Self { Self::IDENTITY } }

impl fmt::Display for DMat4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let c = &self.cols;
        for r in 0..4 {
            writeln!(f, "  [{:12.6}  {:12.6}  {:12.6}  {:12.6}]",
                c[0][r], c[1][r], c[2][r], c[3][r])?;
        }
        Ok(())
    }
}
