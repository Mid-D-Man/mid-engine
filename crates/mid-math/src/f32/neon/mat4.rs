// crates/mid-math/src/f32/neon/mat4.rs
//! Mat4 with NEON fast-paths on aarch64.
//! Build 8: storage changed to four Vec4 (float32x4_t) fields.
//! Mul<Vec4> now accesses self.x_axis.0 etc. directly — no vld1q_f32 for LHS.
//! FMA (vfmaq_f32) is mandatory on AArch64; all four column multiplies use it.

use core::fmt;
use core::ops::Mul;
use core::arch::aarch64::*;

use crate::f32::neon::vec3::Vec3;
use crate::f32::neon::vec4::Vec4;
use crate::f32::neon::quat::Quat;
use crate::EPSILON;

/// 4×4 column-major matrix. 64 bytes, 16-byte aligned.
/// Columns are `float32x4_t` fields via Vec4 — zero vld1q_f32 for LHS of multiply.
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

    #[inline]
    pub fn from_cols(c0: [f32;4], c1: [f32;4], c2: [f32;4], c3: [f32;4]) -> Self {
        Self {
            x_axis: Vec4::from_array(c0),
            y_axis: Vec4::from_array(c1),
            z_axis: Vec4::from_array(c2),
            w_axis: Vec4::from_array(c3),
        }
    }

    #[inline]
    pub fn from_translation(t: Vec3) -> Self {
        Self {
            x_axis: Vec4::X,
            y_axis: Vec4::Y,
            z_axis: Vec4::Z,
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

    #[inline] pub fn from_rotation(q: Quat) -> Self { q.to_mat4() }

    #[inline]
    pub fn from_trs(t: Vec3, r: Quat, s: Vec3) -> Self {
        let q = r.normalize();
        let (x,y,z,w) = (q.x,q.y,q.z,q.w);
        let (x2,y2,z2) = (x+x,y+y,z+z);
        let (xx,yy,zz) = (x*x2,y*y2,z*z2);
        let (xy,xz,yz) = (x*y2,x*z2,y*z2);
        let (wx,wy,wz) = (w*x2,w*y2,w*z2);
        Self {
            x_axis: Vec4::new((1.0-yy-zz)*s.x, (xy+wz)*s.x,    (xz-wy)*s.x,    0.0),
            y_axis: Vec4::new((xy-wz)*s.y,    (1.0-xx-zz)*s.y,  (yz+wx)*s.y,    0.0),
            z_axis: Vec4::new((xz+wy)*s.z,    (yz-wx)*s.z,      (1.0-xx-yy)*s.z, 0.0),
            w_axis: Vec4::new(t.x, t.y, t.z, 1.0),
        }
    }

    // ── View matrices ─────────────────────────────────────────────────────────

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

    pub fn perspective_rh(fov_y: f32, aspect: f32, near: f32, far: f32) -> Self {
        let f = 1.0 / (fov_y * 0.5).tan();
        let z = near - far;
        Self {
            x_axis: Vec4::new(f/aspect, 0.0, 0.0, 0.0),
            y_axis: Vec4::new(0.0, f, 0.0, 0.0),
            z_axis: Vec4::new(0.0, 0.0, (far+near)/z, -1.0),
            w_axis: Vec4::new(0.0, 0.0, (2.0*far*near)/z, 0.0),
        }
    }

    pub fn perspective_lh(fov_y: f32, aspect: f32, near: f32, far: f32) -> Self {
        let f = 1.0 / (fov_y * 0.5).tan();
        let z = far - near;
        Self {
            x_axis: Vec4::new(f/aspect, 0.0, 0.0, 0.0),
            y_axis: Vec4::new(0.0, f, 0.0, 0.0),
            z_axis: Vec4::new(0.0, 0.0, far/z, 1.0),
            w_axis: Vec4::new(0.0, 0.0, -(far*near)/z, 0.0),
        }
    }

    pub fn ortho_rh(left:f32, right:f32, bottom:f32, top:f32, near:f32, far:f32) -> Self {
        let rl=right-left; let tb=top-bottom; let nf=far-near;
        Self {
            x_axis: Vec4::new(2.0/rl, 0.0, 0.0, 0.0),
            y_axis: Vec4::new(0.0, 2.0/tb, 0.0, 0.0),
            z_axis: Vec4::new(0.0, 0.0, -2.0/nf, 0.0),
            w_axis: Vec4::new(-(right+left)/rl, -(top+bottom)/tb, -(far+near)/nf, 1.0),
        }
    }

    pub fn ortho_lh(left: f32, right: f32, bottom: f32, top: f32, near: f32, far: f32) -> Self {
        let rl = right - left; let tb = top - bottom; let nf = far - near;
        Self {
            x_axis: Vec4::new(2.0/rl, 0.0, 0.0, 0.0),
            y_axis: Vec4::new(0.0, 2.0/tb, 0.0, 0.0),
            z_axis: Vec4::new(0.0, 0.0, 1.0/nf, 0.0),
            w_axis: Vec4::new(-(right+left)/rl, -(top+bottom)/tb, -near/nf, 1.0),
        }
    }

    // ── Transpose ─────────────────────────────────────────────────────────────

    pub fn transpose(self) -> Self {
        Self::from_cols(
            [self.x_axis.x, self.y_axis.x, self.z_axis.x, self.w_axis.x],
            [self.x_axis.y, self.y_axis.y, self.z_axis.y, self.w_axis.y],
            [self.x_axis.z, self.y_axis.z, self.z_axis.z, self.w_axis.z],
            [self.x_axis.w, self.y_axis.w, self.z_axis.w, self.w_axis.w],
        )
    }

    // ── Transform helpers ─────────────────────────────────────────────────────
    //
    // Direct NEON multiply-accumulate, skipping extend/truncate round-trip.
    // Lane 3 of the Vec3 result is a don't-care; Vec3 ops never read it.

    #[inline]
    pub fn transform_point(self, p: Vec3) -> Vec3 {
        unsafe {
            let vx = vdupq_laneq_f32::<0>(p.0);
            let vy = vdupq_laneq_f32::<1>(p.0);
            let vz = vdupq_laneq_f32::<2>(p.0);
            let mut res = vmulq_f32(self.x_axis.0, vx);
            res = vfmaq_f32(res, self.y_axis.0, vy);
            res = vfmaq_f32(res, self.z_axis.0, vz);
            Vec3(vaddq_f32(res, self.w_axis.0))
        }
    }

    #[inline]
    pub fn transform_vector(self, v: Vec3) -> Vec3 {
        unsafe {
            let vx = vdupq_laneq_f32::<0>(v.0);
            let vy = vdupq_laneq_f32::<1>(v.0);
            let vz = vdupq_laneq_f32::<2>(v.0);
            let mut res = vmulq_f32(self.x_axis.0, vx);
            res = vfmaq_f32(res, self.y_axis.0, vy);
            Vec3(vfmaq_f32(res, self.z_axis.0, vz))
        }
    }

    // ── Decompose ─────────────────────────────────────────────────────────────

    pub fn decompose_trs(self) -> (Vec3, Quat, Vec3) {
        let t = self.w_axis.truncate();

        let sx = self.x_axis.truncate().length();
        let sy = self.y_axis.truncate().length();
        let sz = self.z_axis.truncate().length();

        let det =
            self.x_axis.x * (self.y_axis.y*self.z_axis.z - self.z_axis.y*self.y_axis.z)
          - self.y_axis.x * (self.x_axis.y*self.z_axis.z - self.z_axis.y*self.x_axis.z)
          + self.z_axis.x * (self.x_axis.y*self.y_axis.z - self.y_axis.y*self.x_axis.z);
        let sx = if det < 0.0 { -sx } else { sx };

        let inv_sx = if sx.abs() < EPSILON { 0.0 } else { 1.0/sx };
        let inv_sy = if sy      < EPSILON { 0.0 } else { 1.0/sy };
        let inv_sz = if sz      < EPSILON { 0.0 } else { 1.0/sz };

        let c0 = self.x_axis.truncate() * inv_sx;
        let c1 = self.y_axis.truncate() * inv_sy;
        let c2 = self.z_axis.truncate() * inv_sz;

        use crate::helpers::euler::QuatExt as _;
        let r = Quat::from_rotation_axes(c0, c1, c2);
        (t, r, Vec3::new(sx, sy, sz))
    }

    // ── Inverse (scalar cofactor) ─────────────────────────────────────────────

    pub fn inverse(self) -> Option<Self> { self.inverse_scalar() }

    pub fn inverse_scalar(self) -> Option<Self> {
        let a = [
            self.x_axis.x, self.x_axis.y, self.x_axis.z, self.x_axis.w,
            self.y_axis.x, self.y_axis.y, self.y_axis.z, self.y_axis.w,
            self.z_axis.x, self.z_axis.y, self.z_axis.z, self.z_axis.w,
            self.w_axis.x, self.w_axis.y, self.w_axis.z, self.w_axis.w,
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
        let id = 1.0/det;
        for v in inv.iter_mut() { *v *= id; }
        Some(Self::from_cols(
            [inv[0],inv[1],inv[2],inv[3]],
            [inv[4],inv[5],inv[6],inv[7]],
            [inv[8],inv[9],inv[10],inv[11]],
            [inv[12],inv[13],inv[14],inv[15]],
        ))
    }

    pub fn inverse_trs(self) -> Self { self.inverse_trs_scalar() }

    pub fn inverse_trs_scalar(self) -> Self {
        let sx2 = self.x_axis.x*self.x_axis.x + self.x_axis.y*self.x_axis.y + self.x_axis.z*self.x_axis.z;
        let sy2 = self.y_axis.x*self.y_axis.x + self.y_axis.y*self.y_axis.y + self.y_axis.z*self.y_axis.z;
        let sz2 = self.z_axis.x*self.z_axis.x + self.z_axis.y*self.z_axis.y + self.z_axis.z*self.z_axis.z;
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
}

// ── Mul<Vec4> — zero vld1q_f32 for LHS (fields are already float32x4_t) ──────
//
// Pattern: 1 vmulq_f32 + 3 vfmaq_f32. FMA is mandatory on AArch64.
// Total: 4 broadcasts (vdupq_laneq_f32) + 1 mul + 3 FMA = 8 NEON ops.

impl Mul<Vec4> for Mat4 {
    type Output = Vec4;
    #[inline(always)]
    fn mul(self, v: Vec4) -> Vec4 {
        unsafe {
            let vx = vdupq_laneq_f32::<0>(v.0);
            let vy = vdupq_laneq_f32::<1>(v.0);
            let vz = vdupq_laneq_f32::<2>(v.0);
            let vw = vdupq_laneq_f32::<3>(v.0);
            let mut res = vmulq_f32(self.x_axis.0, vx);
            res = vfmaq_f32(res, self.y_axis.0, vy);
            res = vfmaq_f32(res, self.z_axis.0, vz);
            Vec4(vfmaq_f32(res, self.w_axis.0, vw))
        }
    }
}

impl Mul for Mat4 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self {
            x_axis: self * rhs.x_axis,
            y_axis: self * rhs.y_axis,
            z_axis: self * rhs.z_axis,
            w_axis: self * rhs.w_axis,
        }
    }
}

impl Default for Mat4 { fn default() -> Self { Self::IDENTITY } }

impl PartialEq for Mat4 {
    fn eq(&self, rhs: &Self) -> bool {
        self.x_axis == rhs.x_axis && self.y_axis == rhs.y_axis
            && self.z_axis == rhs.z_axis && self.w_axis == rhs.w_axis
    }
}

impl fmt::Debug for Mat4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Mat4")
            .field("x_axis", &self.x_axis)
            .field("y_axis", &self.y_axis)
            .field("z_axis", &self.z_axis)
            .field("w_axis", &self.w_axis)
            .finish()
    }
}

impl fmt::Display for Mat4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for r in 0..4 {
            let get = |v: Vec4| match r { 0=>v.x, 1=>v.y, 2=>v.z, _=>v.w };
            writeln!(f, "  [{:8.4}  {:8.4}  {:8.4}  {:8.4}]",
                get(self.x_axis), get(self.y_axis), get(self.z_axis), get(self.w_axis))?;
        }
        Ok(())
    }
                          }
