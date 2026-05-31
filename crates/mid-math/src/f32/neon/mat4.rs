// crates/mid-math/src/f32/neon/mat4.rs
//! Mat4 with NEON fast-paths on aarch64.

use core::fmt;
use core::ops::Mul;
use core::arch::aarch64::*;

use crate::f32::neon::vec3::Vec3;
use crate::f32::neon::vec4::Vec4;
use crate::f32::neon::quat::Quat;
use crate::EPSILON;

/// 4×4 column-major matrix. 64 bytes, 16-byte aligned.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C, align(16))]
pub struct Mat4 {
    pub cols: [[f32; 4]; 4],
}

impl Mat4 {
    pub const ZERO: Self = Self { cols: [[0.0; 4]; 4] };
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
        let q = r.normalize();
        let (x,y,z,w) = (q.x,q.y,q.z,q.w);
        let (x2,y2,z2) = (x+x,y+y,z+z);
        let (xx,yy,zz) = (x*x2,y*y2,z*z2);
        let (xy,xz,yz) = (x*y2,x*z2,y*z2);
        let (wx,wy,wz) = (w*x2,w*y2,w*z2);
        Self::from_cols(
            [(1.0-yy-zz)*s.x, (xy+wz)*s.x,    (xz-wy)*s.x,    0.0],
            [(xy-wz)*s.y,    (1.0-xx-zz)*s.y,  (yz+wx)*s.y,    0.0],
            [(xz+wy)*s.z,    (yz-wx)*s.z,      (1.0-xx-yy)*s.z,0.0],
            [t.x, t.y, t.z, 1.0],
        )
    }

    // ── View matrices ─────────────────────────────────────────────────────────

    /// Right-handed look-at view matrix.
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

    /// Left-handed look-at view matrix. Camera looks along +Z.
    pub fn look_at_lh(eye: Vec3, center: Vec3, up: Vec3) -> Self {
        let f = (center - eye).normalize();
        let r = up.cross(f).normalize();
        let u = f.cross(r);
        Self::from_cols(
            [ r.x,  u.x,  f.x, 0.0],
            [ r.y,  u.y,  f.y, 0.0],
            [ r.z,  u.z,  f.z, 0.0],
            [-r.dot(eye), -u.dot(eye), -f.dot(eye), 1.0],
        )
    }

    // ── Projection matrices ───────────────────────────────────────────────────

    /// Right-handed perspective projection, depth `[-1, 1]`.
    pub fn perspective_rh(fov_y: f32, aspect: f32, near: f32, far: f32) -> Self {
        let f = 1.0 / (fov_y * 0.5).tan();
        let z = near - far;
        Self::from_cols(
            [f/aspect, 0.0, 0.0,               0.0],
            [0.0,      f,   0.0,               0.0],
            [0.0, 0.0, (far+near)/z,           -1.0],
            [0.0, 0.0, (2.0*far*near)/z,        0.0],
        )
    }

    /// Left-handed perspective projection, depth `[0, 1]` (DX12/Metal/Vulkan LH).
    pub fn perspective_lh(fov_y: f32, aspect: f32, near: f32, far: f32) -> Self {
        let f = 1.0 / (fov_y * 0.5).tan();
        let z = far - near;
        Self::from_cols(
            [f / aspect, 0.0,  0.0,               0.0],
            [0.0,        f,    0.0,               0.0],
            [0.0,        0.0,  far / z,            1.0],
            [0.0,        0.0, -(far * near) / z,   0.0],
        )
    }

    /// Right-handed orthographic projection.
    pub fn ortho_rh(left:f32, right:f32, bottom:f32, top:f32, near:f32, far:f32) -> Self {
        let rl=right-left; let tb=top-bottom; let nf=far-near;
        Self::from_cols(
            [2.0/rl, 0.0, 0.0, 0.0],
            [0.0, 2.0/tb, 0.0, 0.0],
            [0.0, 0.0, -2.0/nf, 0.0],
            [-(right+left)/rl, -(top+bottom)/tb, -(far+near)/nf, 1.0],
        )
    }

    /// Left-handed orthographic projection, depth `[0, 1]`.
    pub fn ortho_lh(left: f32, right: f32, bottom: f32, top: f32, near: f32, far: f32) -> Self {
        let rl = right - left;
        let tb = top   - bottom;
        let nf = far   - near;
        Self::from_cols(
            [2.0 / rl, 0.0,      0.0,       0.0],
            [0.0,      2.0 / tb, 0.0,       0.0],
            [0.0,      0.0,      1.0 / nf,  0.0],
            [-(right + left) / rl, -(top + bottom) / tb, -near / nf, 1.0],
        )
    }

    // ── Transpose / transform ─────────────────────────────────────────────────

    pub fn transpose(self) -> Self {
        let c = &self.cols;
        Self::from_cols(
            [c[0][0],c[1][0],c[2][0],c[3][0]],
            [c[0][1],c[1][1],c[2][1],c[3][1]],
            [c[0][2],c[1][2],c[2][2],c[3][2]],
            [c[0][3],c[1][3],c[2][3],c[3][3]],
        )
    }

    #[inline]
    pub fn transform_point(self, p: Vec3) -> Vec3 {
        (self * p.extend(1.0)).truncate()
    }

    #[inline]
    pub fn transform_vector(self, v: Vec3) -> Vec3 {
        (self * v.extend(0.0)).truncate()
    }

    // ── Decompose ─────────────────────────────────────────────────────────────

    /// Decompose a TRS matrix into `(translation, rotation, scale)`.
    ///
    /// The rotation quaternion is always normalised. `scale.x` may be negative
    /// when the matrix encodes a reflection (determinant < 0). Undefined for
    /// matrices containing shear.
    pub fn decompose_trs(self) -> (Vec3, Quat, Vec3) {
        let t = Vec3::new(self.cols[3][0], self.cols[3][1], self.cols[3][2]);

        let sx = Vec3::new(self.cols[0][0], self.cols[0][1], self.cols[0][2]).length();
        let sy = Vec3::new(self.cols[1][0], self.cols[1][1], self.cols[1][2]).length();
        let sz = Vec3::new(self.cols[2][0], self.cols[2][1], self.cols[2][2]).length();

        // Encode reflection into the sign of sx via the upper-3×3 determinant.
        let det =
            self.cols[0][0] * (self.cols[1][1]*self.cols[2][2] - self.cols[2][1]*self.cols[1][2])
          - self.cols[1][0] * (self.cols[0][1]*self.cols[2][2] - self.cols[2][1]*self.cols[0][2])
          + self.cols[2][0] * (self.cols[0][1]*self.cols[1][2] - self.cols[1][1]*self.cols[0][2]);

        let sx = if det < 0.0 { -sx } else { sx };

        let inv_sx = if sx.abs() < EPSILON { 0.0 } else { 1.0 / sx };
        let inv_sy = if sy      < EPSILON { 0.0 } else { 1.0 / sy };
        let inv_sz = if sz      < EPSILON { 0.0 } else { 1.0 / sz };

        let c0 = Vec3::new(
            self.cols[0][0] * inv_sx,
            self.cols[0][1] * inv_sx,
            self.cols[0][2] * inv_sx,
        );
        let c1 = Vec3::new(
            self.cols[1][0] * inv_sy,
            self.cols[1][1] * inv_sy,
            self.cols[1][2] * inv_sy,
        );
        let c2 = Vec3::new(
            self.cols[2][0] * inv_sz,
            self.cols[2][1] * inv_sz,
            self.cols[2][2] * inv_sz,
        );

        use crate::helpers::euler::QuatExt as _;
        let r = Quat::from_rotation_axes(c0, c1, c2);

        (t, r, Vec3::new(sx, sy, sz))
    }

    // ── Inverse ───────────────────────────────────────────────────────────────

    pub fn inverse(self) -> Option<Self> { self.inverse_scalar() }

    pub fn inverse_scalar(self) -> Option<Self> {
        let a = [
            self.cols[0][0],self.cols[0][1],self.cols[0][2],self.cols[0][3],
            self.cols[1][0],self.cols[1][1],self.cols[1][2],self.cols[1][3],
            self.cols[2][0],self.cols[2][1],self.cols[2][2],self.cols[2][3],
            self.cols[3][0],self.cols[3][1],self.cols[3][2],self.cols[3][3],
        ];
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
        let id = 1.0 / det;
        for x in inv.iter_mut() { *x *= id; }
        Some(Self::from_cols(
            [inv[0],inv[1],inv[2],inv[3]],
            [inv[4],inv[5],inv[6],inv[7]],
            [inv[8],inv[9],inv[10],inv[11]],
            [inv[12],inv[13],inv[14],inv[15]],
        ))
    }

    pub fn inverse_trs(self) -> Self { self.inverse_trs_scalar() }

    pub fn inverse_trs_scalar(self) -> Self {
        let sx2 = self.cols[0][0]*self.cols[0][0]+self.cols[0][1]*self.cols[0][1]+self.cols[0][2]*self.cols[0][2];
        let sy2 = self.cols[1][0]*self.cols[1][0]+self.cols[1][1]*self.cols[1][1]+self.cols[1][2]*self.cols[1][2];
        let sz2 = self.cols[2][0]*self.cols[2][0]+self.cols[2][1]*self.cols[2][1]+self.cols[2][2]*self.cols[2][2];
        let isx = if sx2<EPSILON{0.0}else{1.0/sx2};
        let isy = if sy2<EPSILON{0.0}else{1.0/sy2};
        let isz = if sz2<EPSILON{0.0}else{1.0/sz2};
        let ic0 = [self.cols[0][0]*isx,self.cols[1][0]*isy,self.cols[2][0]*isz,0.0];
        let ic1 = [self.cols[0][1]*isx,self.cols[1][1]*isy,self.cols[2][1]*isz,0.0];
        let ic2 = [self.cols[0][2]*isx,self.cols[1][2]*isy,self.cols[2][2]*isz,0.0];
        let (tx,ty,tz) = (self.cols[3][0],self.cols[3][1],self.cols[3][2]);
        let itx = -(ic0[0]*tx+ic1[0]*ty+ic2[0]*tz);
        let ity = -(ic0[1]*tx+ic1[1]*ty+ic2[1]*tz);
        let itz = -(ic0[2]*tx+ic1[2]*ty+ic2[2]*tz);
        Self::from_cols(ic0, ic1, ic2, [itx,ity,itz,1.0])
    }
}

// ── Mul<Vec4> — 1 FMUL + 3 FMLA (FMA mandatory on AArch64) ──────────────────

impl Mul<Vec4> for Mat4 {
    type Output = Vec4;
    #[inline(always)]
    fn mul(self, v: Vec4) -> Vec4 {
        unsafe {
            let a0 = vld1q_f32(self.cols[0].as_ptr());
            let a1 = vld1q_f32(self.cols[1].as_ptr());
            let a2 = vld1q_f32(self.cols[2].as_ptr());
            let a3 = vld1q_f32(self.cols[3].as_ptr());

            let vx = vdupq_laneq_f32::<0>(v.0);
            let vy = vdupq_laneq_f32::<1>(v.0);
            let vz = vdupq_laneq_f32::<2>(v.0);
            let vw = vdupq_laneq_f32::<3>(v.0);

            let mut res = vmulq_f32(a0, vx);
            res = vfmaq_f32(res, a1, vy);
            res = vfmaq_f32(res, a2, vz);
            res = vfmaq_f32(res, a3, vw);
            Vec4(res)
        }
    }
}

impl Mul for Mat4 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        unsafe {
            let c0 = self * Vec4(vld1q_f32(rhs.cols[0].as_ptr()));
            let c1 = self * Vec4(vld1q_f32(rhs.cols[1].as_ptr()));
            let c2 = self * Vec4(vld1q_f32(rhs.cols[2].as_ptr()));
            let c3 = self * Vec4(vld1q_f32(rhs.cols[3].as_ptr()));
            let mut out = Self::ZERO;
            vst1q_f32(out.cols[0].as_mut_ptr(), c0.0);
            vst1q_f32(out.cols[1].as_mut_ptr(), c1.0);
            vst1q_f32(out.cols[2].as_mut_ptr(), c2.0);
            vst1q_f32(out.cols[3].as_mut_ptr(), c3.0);
            out
        }
    }
}

impl Default for Mat4 { fn default() -> Self { Self::IDENTITY } }

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
