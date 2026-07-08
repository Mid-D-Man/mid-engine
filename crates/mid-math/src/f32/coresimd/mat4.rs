// crates/mid-math/src/f32/coresimd/mat4.rs
//! Mat4 using Rust portable SIMD (`f32x4`).
//! Build 8: storage changed to four Vec4 (f32x4) fields.
//! Mul<Vec4> accesses self.x_axis.0 directly — no f32x4::from_array for LHS.

use core::fmt;
use core::ops::Mul;
use core::simd::prelude::*;
use core::simd::{cmp::SimdPartialOrd, num::SimdFloat};

use super::{dot4, f32x4_bitor, f32x4_bitand};
use crate::f32::coresimd::vec3::Vec3;
use crate::f32::coresimd::vec4::Vec4;
use crate::f32::coresimd::quat::Quat;
use crate::EPSILON;

/// 4×4 column-major matrix. 64 bytes, 16-byte aligned. Backed by `f32x4`.
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

    // ── View / projection matrices ────────────────────────────────────────────

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

    // ── Transpose ─────────────────────────────────────────────────────────────
    //
    // Portable SIMD 4×4 transpose using simd_swizzle.

    pub fn transpose(self) -> Self {
        let x = self.x_axis.0;
        let y = self.y_axis.0;
        let z = self.z_axis.0;
        let w = self.w_axis.0;
        let tmp0 = simd_swizzle!(x, y, [0, 1, 4, 5]);
        let tmp1 = simd_swizzle!(x, y, [2, 3, 6, 7]);
        let tmp2 = simd_swizzle!(z, w, [0, 1, 4, 5]);
        let tmp3 = simd_swizzle!(z, w, [2, 3, 6, 7]);
        Self {
            x_axis: Vec4(simd_swizzle!(tmp0, tmp2, [0, 2, 4, 6])),
            y_axis: Vec4(simd_swizzle!(tmp0, tmp2, [1, 3, 5, 7])),
            z_axis: Vec4(simd_swizzle!(tmp1, tmp3, [0, 2, 4, 6])),
            w_axis: Vec4(simd_swizzle!(tmp1, tmp3, [1, 3, 5, 7])),
        }
    }

    // ── Transform helpers ─────────────────────────────────────────────────────

    #[inline]
    pub fn transform_point(self, p: Vec3) -> Vec3 {
        let vx = simd_swizzle!(p.0, [0, 0, 0, 0]);
        let vy = simd_swizzle!(p.0, [1, 1, 1, 1]);
        let vz = simd_swizzle!(p.0, [2, 2, 2, 2]);
        let res = self.x_axis.0 * vx + self.y_axis.0 * vy + self.z_axis.0 * vz;
        Vec3(res + self.w_axis.0)
    }

    #[inline]
    pub fn transform_vector(self, v: Vec3) -> Vec3 {
        let vx = simd_swizzle!(v.0, [0, 0, 0, 0]);
        let vy = simd_swizzle!(v.0, [1, 1, 1, 1]);
        let vz = simd_swizzle!(v.0, [2, 2, 2, 2]);
        Vec3(self.x_axis.0 * vx + self.y_axis.0 * vy + self.z_axis.0 * vz)
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
        let r = super::quat::quat_from_rotation_axes(c0, c1, c2);
        (t, r, Vec3::new(sx, sy, sz))
    }

    // ── Inverse (portable SIMD cofactor) ──────────────────────────────────────
    // Columns accessed directly as f32x4 — no from_array loads.

    pub fn inverse(self) -> Option<Self> {
        let x = self.x_axis.0;
        let y = self.y_axis.0;
        let z = self.z_axis.0;
        let w = self.w_axis.0;

        let fac0 = {
            let swp0a = simd_swizzle!(w, z, [3, 3, 7, 7]);
            let swp0b = simd_swizzle!(w, z, [2, 2, 6, 6]);
            let swp00 = simd_swizzle!(z, y, [2, 2, 6, 6]);
            let swp01 = simd_swizzle!(swp0a, [0, 0, 0, 2]);
            let swp02 = simd_swizzle!(swp0b, [0, 0, 0, 2]);
            let swp03 = simd_swizzle!(z, y, [3, 3, 7, 7]);
            swp00 * swp01 - swp02 * swp03
        };
        let fac1 = {
            let swp0a = simd_swizzle!(w, z, [3, 3, 7, 7]);
            let swp0b = simd_swizzle!(w, z, [1, 1, 5, 5]);
            let swp00 = simd_swizzle!(z, y, [1, 1, 5, 5]);
            let swp01 = simd_swizzle!(swp0a, [0, 0, 0, 2]);
            let swp02 = simd_swizzle!(swp0b, [0, 0, 0, 2]);
            let swp03 = simd_swizzle!(z, y, [3, 3, 7, 7]);
            swp00 * swp01 - swp02 * swp03
        };
        let fac2 = {
            let swp0a = simd_swizzle!(w, z, [2, 2, 6, 6]);
            let swp0b = simd_swizzle!(w, z, [1, 1, 5, 5]);
            let swp00 = simd_swizzle!(z, y, [1, 1, 5, 5]);
            let swp01 = simd_swizzle!(swp0a, [0, 0, 0, 2]);
            let swp02 = simd_swizzle!(swp0b, [0, 0, 0, 2]);
            let swp03 = simd_swizzle!(z, y, [2, 2, 6, 6]);
            swp00 * swp01 - swp02 * swp03
        };
        let fac3 = {
            let swp0a = simd_swizzle!(w, z, [3, 3, 7, 7]);
            let swp0b = simd_swizzle!(w, z, [0, 0, 4, 4]);
            let swp00 = simd_swizzle!(z, y, [0, 0, 4, 4]);
            let swp01 = simd_swizzle!(swp0a, [0, 0, 0, 2]);
            let swp02 = simd_swizzle!(swp0b, [0, 0, 0, 2]);
            let swp03 = simd_swizzle!(z, y, [3, 3, 7, 7]);
            swp00 * swp01 - swp02 * swp03
        };
        let fac4 = {
            let swp0a = simd_swizzle!(w, z, [2, 2, 6, 6]);
            let swp0b = simd_swizzle!(w, z, [0, 0, 4, 4]);
            let swp00 = simd_swizzle!(z, y, [0, 0, 4, 4]);
            let swp01 = simd_swizzle!(swp0a, [0, 0, 0, 2]);
            let swp02 = simd_swizzle!(swp0b, [0, 0, 0, 2]);
            let swp03 = simd_swizzle!(z, y, [2, 2, 6, 6]);
            swp00 * swp01 - swp02 * swp03
        };
        let fac5 = {
            let swp0a = simd_swizzle!(w, z, [1, 1, 5, 5]);
            let swp0b = simd_swizzle!(w, z, [0, 0, 4, 4]);
            let swp00 = simd_swizzle!(z, y, [0, 0, 4, 4]);
            let swp01 = simd_swizzle!(swp0a, [0, 0, 0, 2]);
            let swp02 = simd_swizzle!(swp0b, [0, 0, 0, 2]);
            let swp03 = simd_swizzle!(z, y, [1, 1, 5, 5]);
            swp00 * swp01 - swp02 * swp03
        };

        let sign_a = f32x4::from_array([-1.0,  1.0, -1.0,  1.0]);
        let sign_b = f32x4::from_array([ 1.0, -1.0,  1.0, -1.0]);

        let tmp0 = simd_swizzle!(y, x, [0, 0, 4, 4]);
        let vec0 = simd_swizzle!(tmp0, [0, 2, 2, 2]);
        let tmp1 = simd_swizzle!(y, x, [1, 1, 5, 5]);
        let vec1 = simd_swizzle!(tmp1, [0, 2, 2, 2]);
        let tmp2 = simd_swizzle!(y, x, [2, 2, 6, 6]);
        let vec2 = simd_swizzle!(tmp2, [0, 2, 2, 2]);
        let tmp3 = simd_swizzle!(y, x, [3, 3, 7, 7]);
        let vec3 = simd_swizzle!(tmp3, [0, 2, 2, 2]);

        let inv0 = sign_b * (vec1*fac0 - vec2*fac1 + vec3*fac2);
        let inv1 = sign_a * (vec0*fac0 - vec2*fac3 + vec3*fac4);
        let inv2 = sign_b * (vec0*fac1 - vec1*fac3 + vec3*fac5);
        let inv3 = sign_a * (vec0*fac2 - vec1*fac4 + vec2*fac5);

        let row0_lo = simd_swizzle!(inv0, inv1, [0, 0, 4, 4]);
        let row0_hi = simd_swizzle!(inv2, inv3, [0, 0, 4, 4]);
        let row0    = simd_swizzle!(row0_lo, row0_hi, [0, 2, 4, 6]);
        let det     = dot4(x, row0);

        if det.abs() < EPSILON { return None; }
        let rcp = f32x4::splat(1.0 / det);
        Some(Self {
            x_axis: Vec4(inv0 * rcp),
            y_axis: Vec4(inv1 * rcp),
            z_axis: Vec4(inv2 * rcp),
            w_axis: Vec4(inv3 * rcp),
        })
    }

    pub fn inverse_trs(self) -> Self {
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

// ── Mul<Vec4> — LHS columns are f32x4 fields, zero from_array overhead ────────
//
// simd_swizzle! broadcasts each lane of v to a full f32x4.
// LLVM lowers to a single shuffle instruction on each target ISA.

impl Mul<Vec4> for Mat4 {
    type Output = Vec4;
    #[inline(always)]
    fn mul(self, v: Vec4) -> Vec4 {
        let vx = simd_swizzle!(v.0, [0, 0, 0, 0]);
        let vy = simd_swizzle!(v.0, [1, 1, 1, 1]);
        let vz = simd_swizzle!(v.0, [2, 2, 2, 2]);
        let vw = simd_swizzle!(v.0, [3, 3, 3, 3]);
        Vec4(self.x_axis.0 * vx + self.y_axis.0 * vy + self.z_axis.0 * vz + self.w_axis.0 * vw)
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
            .field("x_axis", &self.x_axis).field("y_axis", &self.y_axis)
            .field("z_axis", &self.z_axis).field("w_axis", &self.w_axis)
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
