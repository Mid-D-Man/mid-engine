// crates/mid-math/src/f32/scalar/mat4.rs
//! Scalar Mat4 — fallback and reference implementation.
//! Build 8: storage changed from [[f32;4];4] to four Vec4 fields.

use core::fmt;
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use crate::f32::scalar::vec3::Vec3;
use crate::f32::scalar::vec4::Vec4;
use crate::f32::scalar::quat::Quat;
use crate::EPSILON;

/// 4×4 column-major matrix. 64 bytes, 16-byte aligned. Scalar storage.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct Mat4 {
    pub x_axis: Vec4,
    pub y_axis: Vec4,
    pub z_axis: Vec4,
    pub w_axis: Vec4,
}

impl Mat4 {
    pub const ZERO: Self = Self {
        x_axis: Vec4::ZERO, y_axis: Vec4::ZERO,
        z_axis: Vec4::ZERO, w_axis: Vec4::ZERO,
    };
    pub const IDENTITY: Self = Self {
        x_axis: Vec4::X, y_axis: Vec4::Y,
        z_axis: Vec4::Z, w_axis: Vec4::W,
    };

    // ── Constructors ──────────────────────────────────────────────────────────

    #[inline]
    pub fn from_cols(c0: [f32;4], c1: [f32;4], c2: [f32;4], c3: [f32;4]) -> Self {
        Self {
            x_axis: Vec4::from_array(c0),
            y_axis: Vec4::from_array(c1),
            z_axis: Vec4::from_array(c2),
            w_axis: Vec4::from_array(c3),
        }
    }

    /// Build from four Vec4 column vectors.
    #[inline]
    pub fn from_cols_vec4(x: Vec4, y: Vec4, z: Vec4, w: Vec4) -> Self {
        Self { x_axis: x, y_axis: y, z_axis: z, w_axis: w }
    }

    /// Build from a flat array in column-major order (col0, col1, col2, col3).
    #[inline]
    pub fn from_cols_array(m: &[f32; 16]) -> Self {
        Self::from_cols(
            [m[0],m[1],m[2],m[3]],
            [m[4],m[5],m[6],m[7]],
            [m[8],m[9],m[10],m[11]],
            [m[12],m[13],m[14],m[15]],
        )
    }

    /// Flatten to column-major `[f32; 16]`.
    #[inline]
    pub fn to_cols_array(self) -> [f32; 16] {
        let (x,y,z,w) = (self.x_axis, self.y_axis, self.z_axis, self.w_axis);
        [x.x,x.y,x.z,x.w, y.x,y.y,y.z,y.w, z.x,z.y,z.z,z.w, w.x,w.y,w.z,w.w]
    }

    /// Build from nested `[[col0], [col1], [col2], [col3]]`.
    #[inline]
    pub fn from_cols_array_2d(m: &[[f32;4];4]) -> Self {
        Self::from_cols(m[0], m[1], m[2], m[3])
    }

    #[inline]
    pub fn to_cols_array_2d(self) -> [[f32;4];4] {
        [self.x_axis.to_array(), self.y_axis.to_array(),
         self.z_axis.to_array(), self.w_axis.to_array()]
    }

    /// Write 16 floats to slice in column-major order. Panics if `slice.len() < 16`.
    #[inline]
    pub fn write_cols_to_slice(self, s: &mut [f32]) {
        let a = self.to_cols_array();
        s[..16].copy_from_slice(&a);
    }

    // ── Column / row accessors ────────────────────────────────────────────────

    /// Column `i` as Vec4 (0-3). Panics if `i >= 4`.
    #[inline]
    pub fn col(&self, i: usize) -> Vec4 {
        match i {
            0 => self.x_axis, 1 => self.y_axis,
            2 => self.z_axis, 3 => self.w_axis,
            _ => panic!("Mat4::col index {i} out of bounds"),
        }
    }

    /// Mutable column reference. Panics if `i >= 4`.
    #[inline]
    pub fn col_mut(&mut self, i: usize) -> &mut Vec4 {
        match i {
            0 => &mut self.x_axis, 1 => &mut self.y_axis,
            2 => &mut self.z_axis, 3 => &mut self.w_axis,
            _ => panic!("Mat4::col_mut index {i} out of bounds"),
        }
    }

    /// Row `i` as Vec4 (assembled — not contiguous). Panics if `i >= 4`.
    #[inline]
    pub fn row(&self, i: usize) -> Vec4 {
        let get = |v: Vec4| match i { 0=>v.x, 1=>v.y, 2=>v.z, _=>v.w };
        assert!(i < 4, "Mat4::row index {i} out of bounds");
        Vec4::new(get(self.x_axis), get(self.y_axis), get(self.z_axis), get(self.w_axis))
    }

    /// Main diagonal `(m00, m11, m22, m33)`.
    #[inline]
    pub fn diagonal(&self) -> Vec4 {
        Vec4::new(self.x_axis.x, self.y_axis.y, self.z_axis.z, self.w_axis.w)
    }

    // ── Transform constructors ────────────────────────────────────────────────

    #[inline]
    pub fn from_translation(t: Vec3) -> Self {
        Self {
            x_axis: Vec4::X, y_axis: Vec4::Y, z_axis: Vec4::Z,
            w_axis: Vec4::new(t.x, t.y, t.z, 1.0),
        }
    }

    #[inline]
    pub fn from_scale(s: Vec3) -> Self {
        Self {
            x_axis: Vec4::new(s.x, 0.0, 0.0, 0.0),
            y_axis: Vec4::new(0.0, s.y, 0.0, 0.0),
            z_axis: Vec4::new(0.0, 0.0, s.z, 0.0),
            w_axis: Vec4::W,
        }
    }

    /// Build from a normalized quaternion (rotation only, no scale/translation).
    #[inline]
    pub fn from_quat(q: Quat) -> Self { q.to_mat4() }

    /// Alias for `from_quat`.
    #[inline] pub fn from_rotation(q: Quat) -> Self { Self::from_quat(q) }

    #[inline]
    pub fn from_rotation_x(angle: f32) -> Self {
        let (s, c) = angle.sin_cos();
        Self::from_cols(
            [1.0,0.0,0.0,0.0], [0.0,c,s,0.0], [0.0,-s,c,0.0], [0.0,0.0,0.0,1.0],
        )
    }
    #[inline]
    pub fn from_rotation_y(angle: f32) -> Self {
        let (s, c) = angle.sin_cos();
        Self::from_cols(
            [c,0.0,-s,0.0], [0.0,1.0,0.0,0.0], [s,0.0,c,0.0], [0.0,0.0,0.0,1.0],
        )
    }
    #[inline]
    pub fn from_rotation_z(angle: f32) -> Self {
        let (s, c) = angle.sin_cos();
        Self::from_cols(
            [c,s,0.0,0.0], [-s,c,0.0,0.0], [0.0,0.0,1.0,0.0], [0.0,0.0,0.0,1.0],
        )
    }

    /// Build from axis (any length) and angle in radians.
    #[inline]
    pub fn from_axis_angle(axis: Vec3, angle: f32) -> Self {
        let (s, c) = angle.sin_cos();
        let t = 1.0 - c;
        let k = {
            let l = (axis.x*axis.x+axis.y*axis.y+axis.z*axis.z).sqrt();
            if l < EPSILON { Vec3::new(0.0,0.0,1.0) } else { Vec3::new(axis.x/l,axis.y/l,axis.z/l) }
        };
        let (x,y,z) = (k.x,k.y,k.z);
        Self::from_cols(
            [t*x*x+c,   t*x*y+s*z, t*x*z-s*y, 0.0],
            [t*x*y-s*z, t*y*y+c,   t*y*z+s*x, 0.0],
            [t*x*z+s*y, t*y*z-s*x, t*z*z+c,   0.0],
            [0.0, 0.0, 0.0, 1.0],
        )
    }

    /// Build from separate scale, rotation (quat) and translation (glam's TRS order).
    #[inline]
    pub fn from_scale_rotation_translation(scale: Vec3, rotation: Quat, translation: Vec3) -> Self {
        Self::from_trs(translation, rotation, scale)
    }

    /// Build from rotation + translation (uniform scale = 1).
    #[inline]
    pub fn from_rotation_translation(rotation: Quat, translation: Vec3) -> Self {
        Self::from_trs(translation, rotation, Vec3::new(1.0,1.0,1.0))
    }

    /// Embed a `crate::Mat3` (3×3 scalar) into the top-left — w column = W axis.
    #[inline]
    pub fn from_mat3(m: crate::Mat3) -> Self {
        Self::from_cols(
            [m.cols[0][0], m.cols[0][1], m.cols[0][2], 0.0],
            [m.cols[1][0], m.cols[1][1], m.cols[1][2], 0.0],
            [m.cols[2][0], m.cols[2][1], m.cols[2][2], 0.0],
            [0.0, 0.0, 0.0, 1.0],
        )
    }

    /// Embed a `crate::Mat3` plus a translation Vec3.
    #[inline]
    pub fn from_mat3_translation(m: crate::Mat3, t: Vec3) -> Self {
        Self::from_cols(
            [m.cols[0][0], m.cols[0][1], m.cols[0][2], 0.0],
            [m.cols[1][0], m.cols[1][1], m.cols[1][2], 0.0],
            [m.cols[2][0], m.cols[2][1], m.cols[2][2], 0.0],
            [t.x, t.y, t.z, 1.0],
        )
    }

    /// The existing full TRS constructor.
    #[inline]
    pub fn from_trs(t: Vec3, r: Quat, s: Vec3) -> Self {
        let q = r.normalize();
        let (x,y,z,w) = (q.x,q.y,q.z,q.w);
        let (x2,y2,z2) = (x+x,y+y,z+z);
        let (xx,yy,zz) = (x*x2,y*y2,z*z2);
        let (xy,xz,yz) = (x*y2,x*z2,y*z2);
        let (wx,wy,wz) = (w*x2,w*y2,w*z2);
        Self {
            x_axis: Vec4::new((1.0-yy-zz)*s.x, (xy+wz)*s.x,     (xz-wy)*s.x,     0.0),
            y_axis: Vec4::new((xy-wz)*s.y,     (1.0-xx-zz)*s.y,  (yz+wx)*s.y,     0.0),
            z_axis: Vec4::new((xz+wy)*s.z,     (yz-wx)*s.z,      (1.0-xx-yy)*s.z, 0.0),
            w_axis: Vec4::new(t.x, t.y, t.z, 1.0),
        }
    }

    // ── View matrices ─────────────────────────────────────────────────────────

    /// Right-hand look-to (full view matrix, includes translation for `eye`).
    #[inline]
    pub fn look_to_rh(eye: Vec3, dir: Vec3, up: Vec3) -> Self {
        let f = dir.normalize();
        let r = f.cross(up).normalize();
        let u = r.cross(f);
        Self {
            x_axis: Vec4::new(r.x, u.x, -f.x, 0.0),
            y_axis: Vec4::new(r.y, u.y, -f.y, 0.0),
            z_axis: Vec4::new(r.z, u.z, -f.z, 0.0),
            w_axis: Vec4::new(-r.dot(eye), -u.dot(eye), f.dot(eye), 1.0),
        }
    }

    /// Left-hand look-to.
    #[inline]
    pub fn look_to_lh(eye: Vec3, dir: Vec3, up: Vec3) -> Self { Self::look_to_rh(eye, -dir, up) }

    pub fn look_at_rh(eye: Vec3, center: Vec3, up: Vec3) -> Self {
        let f = (center - eye).normalize();
        let r = f.cross(up).normalize();
        let u = r.cross(f);
        Self {
            x_axis: Vec4::new(r.x, u.x, -f.x, 0.0),
            y_axis: Vec4::new(r.y, u.y, -f.y, 0.0),
            z_axis: Vec4::new(r.z, u.z, -f.z, 0.0),
            w_axis: Vec4::new(-r.dot(eye), -u.dot(eye), f.dot(eye), 1.0),
        }
    }

    pub fn look_at_lh(eye: Vec3, center: Vec3, up: Vec3) -> Self {
        let f = (center - eye).normalize();
        let r = up.cross(f).normalize();
        let u = f.cross(r);
        Self {
            x_axis: Vec4::new(r.x, u.x, f.x, 0.0),
            y_axis: Vec4::new(r.y, u.y, f.y, 0.0),
            z_axis: Vec4::new(r.z, u.z, f.z, 0.0),
            w_axis: Vec4::new(-r.dot(eye), -u.dot(eye), -f.dot(eye), 1.0),
        }
    }

    // ── Projection matrices ───────────────────────────────────────────────────
    // perspective_rh/lh, perspective_rh_gl/lh_gl, ortho_rh/lh, ortho_rh_gl/lh_gl,
    // and frustum_rh/lh, frustum_rh_gl/lh_gl now live once in
    // `f32/mat4_projection.rs` (see that file's doc comment for why — this used
    // to be duplicated 5 times across every backend and had drifted).

    /// RH infinite perspective (no far plane), z ∈ [0,1].
    /// Near plane maps to 1, far approaches 0.
    pub fn perspective_infinite_rh(fov_y: f32, aspect: f32, near: f32) -> Self {
        let f = 1.0 / (fov_y*0.5).tan();
        Self {
            x_axis: Vec4::new(f/aspect, 0.0, 0.0, 0.0),
            y_axis: Vec4::new(0.0, f, 0.0, 0.0),
            z_axis: Vec4::new(0.0, 0.0, 0.0, -1.0),
            w_axis: Vec4::new(0.0, 0.0, near, 0.0),
        }
    }

    /// RH infinite perspective with reversed-z, z ∈ [0,1].
    /// Near plane maps to 0, far approaches 1 — optimal for floating-point depth.
    pub fn perspective_infinite_reverse_rh(fov_y: f32, aspect: f32, near: f32) -> Self {
        let f = 1.0 / (fov_y*0.5).tan();
        Self {
            x_axis: Vec4::new(f/aspect, 0.0, 0.0, 0.0),
            y_axis: Vec4::new(0.0, f, 0.0, 0.0),
            z_axis: Vec4::new(0.0, 0.0, 0.0, -1.0),
            w_axis: Vec4::new(0.0, 0.0, near, 0.0),
        }
    }

    // ── Core matrix operations ────────────────────────────────────────────────

    pub fn transpose(self) -> Self {
        Self::from_cols(
            [self.x_axis.x, self.y_axis.x, self.z_axis.x, self.w_axis.x],
            [self.x_axis.y, self.y_axis.y, self.z_axis.y, self.w_axis.y],
            [self.x_axis.z, self.y_axis.z, self.z_axis.z, self.w_axis.z],
            [self.x_axis.w, self.y_axis.w, self.z_axis.w, self.w_axis.w],
        )
    }

    /// Determinant.
    #[inline]
    pub fn determinant(self) -> f32 {
        // Extract flat column-major array then use the same cofactor expansion
        // as inverse_scalar for consistency.
        let a = self.to_cols_array();
        a[0]*(a[5]*a[10]*a[15]-a[5]*a[11]*a[14]-a[9]*a[6]*a[15]+a[9]*a[7]*a[14]+a[13]*a[6]*a[11]-a[13]*a[7]*a[10])
       -a[4]*(a[1]*a[10]*a[15]-a[1]*a[11]*a[14]-a[9]*a[2]*a[15]+a[9]*a[3]*a[14]+a[13]*a[2]*a[11]-a[13]*a[3]*a[10])
       +a[8]*(a[1]*a[6]*a[15]-a[1]*a[7]*a[14]-a[5]*a[2]*a[15]+a[5]*a[3]*a[14]+a[13]*a[2]*a[7]-a[13]*a[3]*a[6])
       -a[12]*(a[1]*a[6]*a[11]-a[1]*a[7]*a[10]-a[5]*a[2]*a[11]+a[5]*a[3]*a[10]+a[9]*a[2]*a[7]-a[9]*a[3]*a[6])
    }

    /// Try to invert. Returns `None` if singular. Uses full 16-cofactor MESA formula.
    pub fn try_inverse(self) -> Option<Self> { self.inverse_scalar() }
    /// Alias for `try_inverse` (original name).
    pub fn inverse(self) -> Option<Self> { self.inverse_scalar() }
    /// Invert or return `ZERO` if singular.
    pub fn inverse_or_zero(self) -> Self { self.inverse_scalar().unwrap_or(Self::ZERO) }

    pub fn inverse_scalar(self) -> Option<Self> {
        let a = self.to_cols_array();
        let mut inv = [0.0f32; 16];
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
        let det = a[0]*inv[0]+a[1]*inv[4]+a[2]*inv[8]+a[3]*inv[12];
        if det.abs() < EPSILON { return None; }
        let id = 1.0/det;
        for v in inv.iter_mut() { *v *= id; }
        Some(Self::from_cols(
            [inv[0],inv[1],inv[2],inv[3]],
            [inv[4],inv[5],inv[6],inv[7]],
            [inv[8],inv[9],inv[10],inv[11]],
            [inv[12],inv[13],inv[14],inv[15]],
        ))
    }

    /// Fast TRS-only inverse (avoids full 16-cofactor computation).
    pub fn inverse_trs(self) -> Self { self.inverse_trs_scalar() }
    pub fn inverse_trs_scalar(self) -> Self {
        let sx2 = self.x_axis.x*self.x_axis.x+self.x_axis.y*self.x_axis.y+self.x_axis.z*self.x_axis.z;
        let sy2 = self.y_axis.x*self.y_axis.x+self.y_axis.y*self.y_axis.y+self.y_axis.z*self.y_axis.z;
        let sz2 = self.z_axis.x*self.z_axis.x+self.z_axis.y*self.z_axis.y+self.z_axis.z*self.z_axis.z;
        let isx = if sx2 < EPSILON { 0.0 } else { 1.0/sx2 };
        let isy = if sy2 < EPSILON { 0.0 } else { 1.0/sy2 };
        let isz = if sz2 < EPSILON { 0.0 } else { 1.0/sz2 };
        let ic0 = [self.x_axis.x*isx, self.y_axis.x*isy, self.z_axis.x*isz, 0.0];
        let ic1 = [self.x_axis.y*isx, self.y_axis.y*isy, self.z_axis.y*isz, 0.0];
        let ic2 = [self.x_axis.z*isx, self.y_axis.z*isy, self.z_axis.z*isz, 0.0];
        let (tx,ty,tz) = (self.w_axis.x, self.w_axis.y, self.w_axis.z);
        let itx = -(ic0[0]*tx + ic1[0]*ty + ic2[0]*tz);
        let ity = -(ic0[1]*tx + ic1[1]*ty + ic2[1]*tz);
        let itz = -(ic0[2]*tx + ic1[2]*ty + ic2[2]*tz);
        Self::from_cols(ic0, ic1, ic2, [itx,ity,itz,1.0])
    }

    // ── Transform helpers ─────────────────────────────────────────────────────

    /// `M * v` — full 4D vector transform.
    #[inline]
    pub fn mul_vec4(self, v: Vec4) -> Vec4 { self * v }

    /// `M^T * v`.
    #[inline]
    pub fn mul_transpose_vec4(self, v: Vec4) -> Vec4 {
        Vec4::new(self.x_axis.dot(v), self.y_axis.dot(v), self.z_axis.dot(v), self.w_axis.dot(v))
    }

    /// Transform a 3D point (divides result by w if w ≠ 0).
    #[inline]
    pub fn transform_point(self, p: Vec3) -> Vec3 {
        Vec3::new(
            self.x_axis.x*p.x + self.y_axis.x*p.y + self.z_axis.x*p.z + self.w_axis.x,
            self.x_axis.y*p.x + self.y_axis.y*p.y + self.z_axis.y*p.z + self.w_axis.y,
            self.x_axis.z*p.x + self.y_axis.z*p.y + self.z_axis.z*p.z + self.w_axis.z,
        )
    }
    /// Alias for `transform_point` (glam-compat name).
    #[inline] pub fn transform_point3(self, p: Vec3) -> Vec3 { self.transform_point(p) }

    /// Transform a 3D direction vector (ignores translation column).
    #[inline]
    pub fn transform_vector(self, v: Vec3) -> Vec3 {
        Vec3::new(
            self.x_axis.x*v.x + self.y_axis.x*v.y + self.z_axis.x*v.z,
            self.x_axis.y*v.x + self.y_axis.y*v.y + self.z_axis.y*v.z,
            self.x_axis.z*v.x + self.y_axis.z*v.y + self.z_axis.z*v.z,
        )
    }
    /// Alias for `transform_vector` (glam-compat name).
    #[inline] pub fn transform_vector3(self, v: Vec3) -> Vec3 { self.transform_vector(v) }

    /// Full perspective transform: transform then divide by w (for projection matrices).
    #[inline]
    pub fn project_point3(self, p: Vec3) -> Vec3 {
        let v = self * Vec4::new(p.x, p.y, p.z, 1.0);
        let rw = if v.w.abs() < EPSILON { 0.0 } else { 1.0 / v.w };
        Vec3::new(v.x*rw, v.y*rw, v.z*rw)
    }

    // ── Named arithmetic ──────────────────────────────────────────────────────

    #[inline]
    pub fn mul_mat4(self, rhs: Self) -> Self { self * rhs }

    #[inline]
    pub fn add_mat4(self, rhs: Self) -> Self {
        Self {
            x_axis: self.x_axis + rhs.x_axis, y_axis: self.y_axis + rhs.y_axis,
            z_axis: self.z_axis + rhs.z_axis, w_axis: self.w_axis + rhs.w_axis,
        }
    }
    #[inline]
    pub fn sub_mat4(self, rhs: Self) -> Self {
        Self {
            x_axis: self.x_axis - rhs.x_axis, y_axis: self.y_axis - rhs.y_axis,
            z_axis: self.z_axis - rhs.z_axis, w_axis: self.w_axis - rhs.w_axis,
        }
    }
    #[inline]
    pub fn mul_scalar(self, s: f32) -> Self {
        Self {
            x_axis: self.x_axis*s, y_axis: self.y_axis*s,
            z_axis: self.z_axis*s, w_axis: self.w_axis*s,
        }
    }
    #[inline] pub fn div_scalar(self, s: f32) -> Self { self.mul_scalar(1.0/s) }

    /// Scale column `i` by `scale[i]`. Equivalent to `self * Mat4::from_scale(scale)`.
    #[inline]
    pub fn mul_diagonal_scale(self, s: Vec3) -> Self {
        Self {
            x_axis: self.x_axis * s.x, y_axis: self.y_axis * s.y,
            z_axis: self.z_axis * s.z, w_axis: self.w_axis,
        }
    }

    #[inline]
    pub fn abs(self) -> Self {
        Self {
            x_axis: self.x_axis.abs(), y_axis: self.y_axis.abs(),
            z_axis: self.z_axis.abs(), w_axis: self.w_axis.abs(),
        }
    }
    #[inline]
    pub fn recip(self) -> Self {
        Self {
            x_axis: self.x_axis.recip(), y_axis: self.y_axis.recip(),
            z_axis: self.z_axis.recip(), w_axis: self.w_axis.recip(),
        }
    }

    // ── Decompose ─────────────────────────────────────────────────────────────

    /// Decompose into (translation, rotation, scale) — glam argument order.
    /// Uses Shepperd's method for quat extraction (same as `decompose_trs`).
    #[inline]
    pub fn to_scale_rotation_translation(self) -> (Vec3, Quat, Vec3) {
        let (t, r, s) = self.decompose_trs();
        (s, r, t)
    }

    pub fn decompose_trs(self) -> (Vec3, Quat, Vec3) {
        let t = Vec3::new(self.w_axis.x, self.w_axis.y, self.w_axis.z);
        let sx = Vec3::new(self.x_axis.x, self.x_axis.y, self.x_axis.z).length();
        let sy = Vec3::new(self.y_axis.x, self.y_axis.y, self.y_axis.z).length();
        let sz = Vec3::new(self.z_axis.x, self.z_axis.y, self.z_axis.z).length();
        let det =
            self.x_axis.x*(self.y_axis.y*self.z_axis.z - self.z_axis.y*self.y_axis.z)
          - self.y_axis.x*(self.x_axis.y*self.z_axis.z - self.z_axis.y*self.x_axis.z)
          + self.z_axis.x*(self.x_axis.y*self.y_axis.z - self.y_axis.y*self.x_axis.z);
        let sx = if det < 0.0 { -sx } else { sx };
        let inv_sx = if sx.abs() < EPSILON { 0.0 } else { 1.0/sx };
        let inv_sy = if sy      < EPSILON { 0.0 } else { 1.0/sy };
        let inv_sz = if sz      < EPSILON { 0.0 } else { 1.0/sz };
        let m00=self.x_axis.x*inv_sx; let m10=self.x_axis.y*inv_sx; let m20=self.x_axis.z*inv_sx;
        let m01=self.y_axis.x*inv_sy; let m11=self.y_axis.y*inv_sy; let m21=self.y_axis.z*inv_sy;
        let m02=self.z_axis.x*inv_sz; let m12=self.z_axis.y*inv_sz; let m22=self.z_axis.z*inv_sz;
        let r = if m22 <= 0.0 {
            let dif10=m11-m00; let omm22=1.0-m22;
            if dif10 <= 0.0 {
                let four_xsq=omm22-dif10; let inv4x=0.5/four_xsq.sqrt();
                Quat::new(four_xsq*inv4x,(m10+m01)*inv4x,(m20+m02)*inv4x,(m12-m21)*inv4x)
            } else {
                let four_ysq=omm22+dif10; let inv4y=0.5/four_ysq.sqrt();
                Quat::new((m10+m01)*inv4y,four_ysq*inv4y,(m21+m12)*inv4y,(m20-m02)*inv4y)
            }
        } else {
            let sum10=m11+m00; let opm22=1.0+m22;
            if sum10 <= 0.0 {
                let four_zsq=opm22-sum10; let inv4z=0.5/four_zsq.sqrt();
                Quat::new((m20+m02)*inv4z,(m21+m12)*inv4z,four_zsq*inv4z,(m01-m10)*inv4z)
            } else {
                let four_wsq=opm22+sum10; let inv4w=0.5/four_wsq.sqrt();
                Quat::new((m12-m21)*inv4w,(m20-m02)*inv4w,(m01-m10)*inv4w,four_wsq*inv4w)
            }
        };
        (t, r.normalize(), Vec3::new(sx, sy, sz))
    }

    // ── Predicates ────────────────────────────────────────────────────────────

    #[inline]
    pub fn is_finite(self) -> bool {
        self.x_axis.is_finite() && self.y_axis.is_finite()
            && self.z_axis.is_finite() && self.w_axis.is_finite()
    }
    #[inline]
    pub fn is_nan(self) -> bool {
        self.x_axis.is_nan() || self.y_axis.is_nan()
            || self.z_axis.is_nan() || self.w_axis.is_nan()
    }
    #[inline]
    pub fn approx_eq(self, rhs: Self) -> bool { self.abs_diff_eq(rhs, EPSILON) }
    #[inline]
    pub fn abs_diff_eq(self, rhs: Self, max_abs_diff: f32) -> bool {
        self.x_axis.abs_diff_eq(rhs.x_axis, max_abs_diff)
            && self.y_axis.abs_diff_eq(rhs.y_axis, max_abs_diff)
            && self.z_axis.abs_diff_eq(rhs.z_axis, max_abs_diff)
            && self.w_axis.abs_diff_eq(rhs.w_axis, max_abs_diff)
    }
}

// ── Default / PartialEq / Debug / Display ─────────────────────────────────────

impl Default for Mat4 { fn default() -> Self { Self::IDENTITY } }
impl PartialEq for Mat4 {
    fn eq(&self, r: &Self) -> bool {
        self.x_axis==r.x_axis && self.y_axis==r.y_axis
            && self.z_axis==r.z_axis && self.w_axis==r.w_axis
    }
}
impl fmt::Debug for Mat4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Mat4")
            .field("x_axis", &self.x_axis).field("y_axis", &self.y_axis)
            .field("z_axis", &self.z_axis).field("w_axis", &self.w_axis)
            .finish()
    }
}
impl fmt::Display for Mat4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for r in 0..4 {
            let g = |v: Vec4| match r { 0=>v.x, 1=>v.y, 2=>v.z, _=>v.w };
            writeln!(f, "  [{:8.4}  {:8.4}  {:8.4}  {:8.4}]",
                g(self.x_axis), g(self.y_axis), g(self.z_axis), g(self.w_axis))?;
        }
        Ok(())
    }
}

// ── Operators ─────────────────────────────────────────────────────────────────

impl Mul<Vec4> for Mat4 {
    type Output = Vec4;
    #[inline(always)]
    fn mul(self, v: Vec4) -> Vec4 {
        Vec4::new(
            self.x_axis.x*v.x + self.y_axis.x*v.y + self.z_axis.x*v.z + self.w_axis.x*v.w,
            self.x_axis.y*v.x + self.y_axis.y*v.y + self.z_axis.y*v.z + self.w_axis.y*v.w,
            self.x_axis.z*v.x + self.y_axis.z*v.y + self.z_axis.z*v.z + self.w_axis.z*v.w,
            self.x_axis.w*v.x + self.y_axis.w*v.y + self.z_axis.w*v.z + self.w_axis.w*v.w,
        )
    }
}
impl Mul for Mat4 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self {
            x_axis: self * rhs.x_axis, y_axis: self * rhs.y_axis,
            z_axis: self * rhs.z_axis, w_axis: self * rhs.w_axis,
        }
    }
}
impl MulAssign for Mat4 { #[inline] fn mul_assign(&mut self, r: Self) { *self = *self * r; } }
impl Mul<f32> for Mat4 { type Output=Self; #[inline] fn mul(self,s:f32)->Self{self.mul_scalar(s)} }
impl Mul<Mat4> for f32 { type Output=Mat4; #[inline] fn mul(self,m:Mat4)->Mat4{m.mul_scalar(self)} }
impl MulAssign<f32> for Mat4 { #[inline] fn mul_assign(&mut self,s:f32){*self=self.mul_scalar(s);} }
impl Add for Mat4 { type Output=Self; #[inline] fn add(self,r:Self)->Self{self.add_mat4(r)} }
impl AddAssign for Mat4 { #[inline] fn add_assign(&mut self,r:Self){*self=self.add_mat4(r);} }
impl Sub for Mat4 { type Output=Self; #[inline] fn sub(self,r:Self)->Self{self.sub_mat4(r)} }
impl SubAssign for Mat4 { #[inline] fn sub_assign(&mut self,r:Self){*self=self.sub_mat4(r);} }
impl Neg for Mat4 { type Output=Self; #[inline] fn neg(self)->Self{self.mul_scalar(-1.0)} }
