// crates/mid-math/src/f32/neon/mat4.rs
//! Mat4 with NEON fast-paths on aarch64.
//!
//! NEON-optimised paths:
//!   Mul<Vec4>  — 1 FMUL + 3 FMLA = 4 instructions (FMA mandatory on AArch64)
//!   Mul<Mat4>  — delegates 4× to Mul<Vec4>, four independent chains
//!
//! Inverse   — scalar path (correct, ~117 ns on x86 baseline reference).
//!              NEON shuffle-based inverse is Phase 2 — same priority as x86.
//!
//! Cross-compilation testing:
//!   cross test -p mid-math --target aarch64-unknown-linux-gnu --release

use core::fmt;
use core::ops::Mul;

use core::arch::aarch64::*;

use crate::f32::neon::vec3::Vec3;
use crate::f32::neon::vec4::Vec4;
use crate::f32::neon::quat::Quat;
use crate::EPSILON;

/// 4×4 column-major matrix. 64 bytes, 16-byte aligned.
/// `cols[c][r]` = element at column `c`, row `r`.
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

    pub fn ortho_rh(left:f32,right:f32,bottom:f32,top:f32,near:f32,far:f32) -> Self {
        let rl=right-left; let tb=top-bottom; let nf=far-near;
        Self::from_cols(
            [2.0/rl, 0.0, 0.0, 0.0],
            [0.0, 2.0/tb, 0.0, 0.0],
            [0.0, 0.0, -2.0/nf, 0.0],
            [-(right+left)/rl, -(top+bottom)/tb, -(far+near)/nf, 1.0],
        )
    }

    pub fn transpose(self) -> Self {
        let c = &self.cols;
        Self::from_cols(
            [c[0][0],c[1][0],c[2][0],c[3][0]],
            [c[0][1],c[1][1],c[2][1],c[3][1]],
            [c[0][2],c[1][2],c[2][2],c[3][2]],
            [c[0][3],c[1][3],c[2][3],c[3][3]],
        )
    }

    /// Transform a point (w=1). Uses NEON Mul<Vec4>.
    #[inline]
    pub fn transform_point(self, p: Vec3) -> Vec3 {
        (self * p.extend(1.0)).truncate()
    }

    /// Transform a direction vector (w=0). Uses NEON Mul<Vec4>.
    #[inline]
    pub fn transform_vector(self, v: Vec3) -> Vec3 {
        (self * v.extend(0.0)).truncate()
    }

    // ── Inverse — scalar (Phase 2: NEON shuffle path) ─────────────────────────

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
//
// Each column is loaded as float32x4_t. The vector's lane is broadcast with
// vdupq_laneq_f32, then accumulated via vfmaq_f32 (a + b*c).
// On AArch64 this is 4 SIMD instructions total (1 FMUL + 3 FMLA).
// Compare: SSE2 needs 8 instructions (4 FMUL + 4 FADD) without FMA3.

impl Mul<Vec4> for Mat4 {
    type Output = Vec4;
    #[inline(always)]
    fn mul(self, v: Vec4) -> Vec4 {
        unsafe {
            let a0 = vld1q_f32(self.cols[0].as_ptr());
            let a1 = vld1q_f32(self.cols[1].as_ptr());
            let a2 = vld1q_f32(self.cols[2].as_ptr());
            let a3 = vld1q_f32(self.cols[3].as_ptr());

            // Broadcast each component of v across all 4 lanes.
            let vx = vdupq_laneq_f32::<0>(v.0);
            let vy = vdupq_laneq_f32::<1>(v.0);
            let vz = vdupq_laneq_f32::<2>(v.0);
            let vw = vdupq_laneq_f32::<3>(v.0);

            // result = col0*vx + col1*vy + col2*vz + col3*vw
            // Using FMA chain: vfmaq_f32(acc, b, c) = acc + b*c
            let mut res = vmulq_f32(a0, vx);
            res = vfmaq_f32(res, a1, vy);
            res = vfmaq_f32(res, a2, vz);
            res = vfmaq_f32(res, a3, vw);
            Vec4(res)
        }
    }
}

// ── Mul<Mat4> — four independent Mul<Vec4> chains ────────────────────────────
//
// Four columns of rhs processed independently. OOO scheduler overlaps them.
// AArch64 has more SIMD execution ports than x86 SSE2, so this scales well.

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
