// crates/mid-math/src/f32/scalar/mat4.rs
//! Scalar Mat4 — fallback and reference implementation.
//! Build 8: storage changed from [[f32;4];4] to four Vec4 fields.
//! No SIMD benefit here (scalar Vec4), but API is consistent with all backends
//! and LLVM can still auto-vectorise the hot paths.
#![allow(dead_code)]
use core::fmt;
use core::ops::Mul;
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

    #[inline]
    pub fn from_rotation(q: Quat) -> Self { q.to_mat4() }

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

    pub fn perspective_rh(fov_y:f32, aspect:f32, near:f32, far:f32) -> Self {
        let f = 1.0 / (fov_y*0.5).tan();
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

    pub fn ortho_rh(left:f32,right:f32,bottom:f32,top:f32,near:f32,far:f32) -> Self {
        let rl=right-left; let tb=top-bottom; let nf=far-near;
        Self {
            x_axis: Vec4::new(2.0/rl, 0.0, 0.0, 0.0),
            y_axis: Vec4::new(0.0, 2.0/tb, 0.0, 0.0),
            z_axis: Vec4::new(0.0, 0.0, -2.0/nf, 0.0),
            w_axis: Vec4::new(-(right+left)/rl, -(top+bottom)/tb, -(far+near)/nf, 1.0),
        }
    }

    pub fn ortho_lh(left: f32, right: f32, bottom: f32, top: f32, near: f32, far: f32) -> Self {
        let rl = right - left;
        let tb = top   - bottom;
        let nf = far   - near;
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

    #[inline]
    pub fn transform_point(self, p: Vec3) -> Vec3 {
        Vec3::new(
            self.x_axis.x*p.x + self.y_axis.x*p.y + self.z_axis.x*p.z + self.w_axis.x,
            self.x_axis.y*p.x + self.y_axis.y*p.y + self.z_axis.y*p.z + self.w_axis.y,
            self.x_axis.z*p.x + self.y_axis.z*p.y + self.z_axis.z*p.z + self.w_axis.z,
        )
    }

    #[inline]
    pub fn transform_vector(self, v: Vec3) -> Vec3 {
        Vec3::new(
            self.x_axis.x*v.x + self.y_axis.x*v.y + self.z_axis.x*v.z,
            self.x_axis.y*v.x + self.y_axis.y*v.y + self.z_axis.y*v.z,
            self.x_axis.z*v.x + self.y_axis.z*v.y + self.z_axis.z*v.z,
        )
    }

    // ── Decompose ─────────────────────────────────────────────────────────────
    //
    // Shoemake largest-component quaternion extraction inlined — avoids the
    // QuatExt::from_rotation_axes type-mismatch on x86_64 where crate::Vec3
    // resolves to the SSE2 type, not this scalar Vec3.

    pub fn decompose_trs(self) -> (Vec3, Quat, Vec3) {
        let t = Vec3::new(self.w_axis.x, self.w_axis.y, self.w_axis.z);

        let sx = Vec3::new(self.x_axis.x, self.x_axis.y, self.x_axis.z).length();
        let sy = Vec3::new(self.y_axis.x, self.y_axis.y, self.y_axis.z).length();
        let sz = Vec3::new(self.z_axis.x, self.z_axis.y, self.z_axis.z).length();

        let det =
            self.x_axis.x * (self.y_axis.y*self.z_axis.z - self.z_axis.y*self.y_axis.z)
          - self.y_axis.x * (self.x_axis.y*self.z_axis.z - self.z_axis.y*self.x_axis.z)
          + self.z_axis.x * (self.x_axis.y*self.y_axis.z - self.y_axis.y*self.x_axis.z);
        let sx = if det < 0.0 { -sx } else { sx };

        let inv_sx = if sx.abs() < EPSILON { 0.0 } else { 1.0 / sx };
        let inv_sy = if sy      < EPSILON { 0.0 } else { 1.0 / sy };
        let inv_sz = if sz      < EPSILON { 0.0 } else { 1.0 / sz };

        let m00 = self.x_axis.x * inv_sx;
        let m10 = self.x_axis.y * inv_sx;
        let m20 = self.x_axis.z * inv_sx;
        let m01 = self.y_axis.x * inv_sy;
        let m11 = self.y_axis.y * inv_sy;
        let m21 = self.y_axis.z * inv_sy;
        let m02 = self.z_axis.x * inv_sz;
        let m12 = self.z_axis.y * inv_sz;
        let m22 = self.z_axis.z * inv_sz;

        let r = if m22 <= 0.0 {
            let dif10 = m11 - m00;
            let omm22 = 1.0 - m22;
            if dif10 <= 0.0 {
                let four_xsq = omm22 - dif10;
                let inv4x = 0.5 / four_xsq.sqrt();
                Quat::new(four_xsq*inv4x, (m10+m01)*inv4x, (m20+m02)*inv4x, (m12-m21)*inv4x)
            } else {
                let four_ysq = omm22 + dif10;
                let inv4y = 0.5 / four_ysq.sqrt();
                Quat::new((m10+m01)*inv4y, four_ysq*inv4y, (m21+m12)*inv4y, (m20-m02)*inv4y)
            }
        } else {
            let sum10 = m11 + m00;
            let opm22 = 1.0 + m22;
            if sum10 <= 0.0 {
                let four_zsq = opm22 - sum10;
                let inv4z = 0.5 / four_zsq.sqrt();
                Quat::new((m20+m02)*inv4z, (m21+m12)*inv4z, four_zsq*inv4z, (m01-m10)*inv4z)
            } else {
                let four_wsq = opm22 + sum10;
                let inv4w = 0.5 / four_wsq.sqrt();
                Quat::new((m12-m21)*inv4w, (m20-m02)*inv4w, (m01-m10)*inv4w, four_wsq*inv4w)
            }
        };

        (t, r.normalize(), Vec3::new(sx, sy, sz))
    }

    // ── Inverse ───────────────────────────────────────────────────────────────

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

impl Default for Mat4 { fn default() -> Self { Self::IDENTITY } }

impl PartialEq for Mat4 {
    fn eq(&self, rhs: &Self) -> bool {
        self.x_axis == rhs.x_axis && self.y_axis == rhs.y_axis
            && self.z_axis == rhs.z_axis && self.w_axis == rhs.w_axis
    }
}

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
            x_axis: self * rhs.x_axis,
            y_axis: self * rhs.y_axis,
            z_axis: self * rhs.z_axis,
            w_axis: self * rhs.w_axis,
        }
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
