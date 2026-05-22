// crates/mid-math/src/f32/coresimd/mat4.rs
//! Mat4 using Rust portable SIMD (`f32x4`).
//!
//! `simd_swizzle!` replaces all `_mm_shuffle_ps` calls from the SSE2 version.
//! The mapping is: `_mm_shuffle_ps::<IMM>(a, b)` where
//!   result[k] = a[((IMM >> (k*2)) & 3)]   for k in 0..2
//!   result[k] = b[((IMM >> (k*2)) & 3)]   for k in 2..4
//!
//! In portable SIMD: `simd_swizzle!(a, b, [i0,i1,i2,i3])` where
//!   indices 0-3 select from `a`, indices 4-7 select from `b`.
//!
//! All operations are safe (no `unsafe` blocks). The compiler lowers
//! `simd_swizzle!` to the optimal shuffle instruction on each target.

use core::fmt;
use core::ops::Mul;
use core::simd::prelude::*;
use core::simd::{cmp::SimdPartialOrd, num::SimdFloat};

use super::{dot4, f32x4_bitor, f32x4_bitand};
use crate::f32::coresimd::vec3::Vec3;
use crate::f32::coresimd::vec4::Vec4;
use crate::f32::coresimd::quat::Quat;
use crate::EPSILON;

/// 4×4 column-major matrix. 64 bytes, 16-byte aligned.
/// `cols[c][r]` = element at column `c`, row `r`.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C, align(16))]
pub struct Mat4 {
    pub cols: [[f32; 4]; 4],
}

impl Mat4 {
    pub const ZERO: Self = Self { cols: [[0.0;4];4] };
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
            [s.x, 0.0, 0.0, 0.0], [0.0, s.y, 0.0, 0.0],
            [0.0, 0.0, s.z, 0.0], [0.0, 0.0, 0.0, 1.0],
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
            [(1.0-yy-zz)*s.x, (xy+wz)*s.x,     (xz-wy)*s.x,    0.0],
            [(xy-wz)*s.y,     (1.0-xx-zz)*s.y,  (yz+wx)*s.y,    0.0],
            [(xz+wy)*s.z,     (yz-wx)*s.z,      (1.0-xx-yy)*s.z,0.0],
            [t.x, t.y, t.z, 1.0],
        )
    }

    pub fn look_at_rh(eye: Vec3, center: Vec3, up: Vec3) -> Self {
        let f = (center - eye).normalize();
        let r = f.cross(up).normalize();
        let u = r.cross(f);
        Self::from_cols(
            [ r.x,  u.x, -f.x, 0.0], [ r.y,  u.y, -f.y, 0.0],
            [ r.z,  u.z, -f.z, 0.0], [-r.dot(eye), -u.dot(eye), f.dot(eye), 1.0],
        )
    }

    pub fn perspective_rh(fov_y: f32, aspect: f32, near: f32, far: f32) -> Self {
        let f = 1.0 / (fov_y * 0.5).tan();
        let z = near - far;
        Self::from_cols(
            [f/aspect, 0.0, 0.0, 0.0], [0.0, f, 0.0, 0.0],
            [0.0, 0.0, (far+near)/z, -1.0], [0.0, 0.0, (2.0*far*near)/z, 0.0],
        )
    }

    pub fn ortho_rh(left:f32,right:f32,bottom:f32,top:f32,near:f32,far:f32)->Self{
        let rl=right-left; let tb=top-bottom; let nf=far-near;
        Self::from_cols(
            [2.0/rl,0.0,0.0,0.0], [0.0,2.0/tb,0.0,0.0],
            [0.0,0.0,-2.0/nf,0.0], [-(right+left)/rl,-(top+bottom)/tb,-(far+near)/nf,1.0],
        )
    }

    /// Transpose using 2×2 block interleave with `simd_swizzle!`.
    ///
    /// `simd_swizzle!(x, y, [0,1,4,5])` = [x0,x1,y0,y1] interleaves the
    /// first halves of two columns, enabling a 4×4 transpose in 8 swizzles.
    pub fn transpose(self) -> Self {
        let x = f32x4::from_array(self.cols[0]);
        let y = f32x4::from_array(self.cols[1]);
        let z = f32x4::from_array(self.cols[2]);
        let w = f32x4::from_array(self.cols[3]);

        let tmp0 = simd_swizzle!(x, y, [0, 1, 4, 5]); // [x0,x1,y0,y1]
        let tmp1 = simd_swizzle!(x, y, [2, 3, 6, 7]); // [x2,x3,y2,y3]
        let tmp2 = simd_swizzle!(z, w, [0, 1, 4, 5]); // [z0,z1,w0,w1]
        let tmp3 = simd_swizzle!(z, w, [2, 3, 6, 7]); // [z2,z3,w2,w3]

        Self::from_cols(
            simd_swizzle!(tmp0, tmp2, [0, 2, 4, 6]).to_array(), // col 0 of T
            simd_swizzle!(tmp0, tmp2, [1, 3, 5, 7]).to_array(), // col 1 of T
            simd_swizzle!(tmp1, tmp3, [0, 2, 4, 6]).to_array(), // col 2 of T
            simd_swizzle!(tmp1, tmp3, [1, 3, 5, 7]).to_array(), // col 3 of T
        )
    }

    pub fn determinant(self) -> f32 {
        let c = &self.cols;
        let a = |ci: usize, ri: usize| c[ci][ri];
        let sub3 = |c0,c1,c2,r0,r1,r2| -> f32 {
            a(c0,r0)*(a(c1,r1)*a(c2,r2) - a(c2,r1)*a(c1,r2))
           -a(c1,r0)*(a(c0,r1)*a(c2,r2) - a(c2,r1)*a(c0,r2))
           +a(c2,r0)*(a(c0,r1)*a(c1,r2) - a(c1,r1)*a(c0,r2))
        };
        a(0,0)*sub3(1,2,3,1,2,3) - a(1,0)*sub3(0,2,3,1,2,3)
       +a(2,0)*sub3(0,1,3,1,2,3) - a(3,0)*sub3(0,1,2,1,2,3)
    }

    #[inline]
    pub fn transform_point(self, p: Vec3) -> Vec3 {
        (self * p.extend(1.0)).truncate()
    }

    #[inline]
    pub fn transform_vector(self, v: Vec3) -> Vec3 {
        (self * v.extend(0.0)).truncate()
    }

    /// General inverse via cofactor expansion (fac0-fac5) with `simd_swizzle!`.
    ///
    /// Each `simd_swizzle!(a, b, [i0,i1,i2,i3])` index mapping:
    ///   0-3  → lanes from `a`,  4-7 → lanes from `b`.
    /// This directly replaces `_mm_shuffle_ps::<IMM>(a, b)` where:
    ///   result[k] = a[((IMM >> 2k) & 3)]  for k<2,  b[...]  for k>=2.
    pub fn inverse(self) -> Option<Self> {
        let x = f32x4::from_array(self.cols[0]);
        let y = f32x4::from_array(self.cols[1]);
        let z = f32x4::from_array(self.cols[2]);
        let w = f32x4::from_array(self.cols[3]);

        // ── fac0: cofactors for (z2·w3 - z3·w2) family ───────────────────────
        let fac0 = {
            let swp0a = simd_swizzle!(w, z, [3, 3, 7, 7]); // [w3,w3,z3,z3]
            let swp0b = simd_swizzle!(w, z, [2, 2, 6, 6]); // [w2,w2,z2,z2]
            let swp00 = simd_swizzle!(z, y, [2, 2, 6, 6]); // [z2,z2,y2,y2]
            let swp01 = simd_swizzle!(swp0a, [0, 0, 0, 2]); // [w3,w3,w3,z3]
            let swp02 = simd_swizzle!(swp0b, [0, 0, 0, 2]); // [w2,w2,w2,z2]
            let swp03 = simd_swizzle!(z, y, [3, 3, 7, 7]);  // [z3,z3,y3,y3]
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

        // ── Alternating sign patterns ─────────────────────────────────────────
        let sign_a = f32x4::from_array([-1.0,  1.0, -1.0,  1.0]);
        let sign_b = f32x4::from_array([ 1.0, -1.0,  1.0, -1.0]);

        // ── vec0..vec3: broadcast each row element of (y,x) combined ─────────
        // simd_swizzle!(y, x, [0,0,4,4]) = [y0,y0,x0,x0]
        // simd_swizzle!(that, [0,2,2,2]) = [y0,x0,x0,x0]
        let tmp0 = simd_swizzle!(y, x, [0, 0, 4, 4]);
        let vec0 = simd_swizzle!(tmp0, [0, 2, 2, 2]);
        let tmp1 = simd_swizzle!(y, x, [1, 1, 5, 5]);
        let vec1 = simd_swizzle!(tmp1, [0, 2, 2, 2]);
        let tmp2 = simd_swizzle!(y, x, [2, 2, 6, 6]);
        let vec2 = simd_swizzle!(tmp2, [0, 2, 2, 2]);
        let tmp3 = simd_swizzle!(y, x, [3, 3, 7, 7]);
        let vec3 = simd_swizzle!(tmp3, [0, 2, 2, 2]);

        // ── Cofactor columns ──────────────────────────────────────────────────
        let inv0 = sign_b * (vec1*fac0 - vec2*fac1 + vec3*fac2);
        let inv1 = sign_a * (vec0*fac0 - vec2*fac3 + vec3*fac4);
        let inv2 = sign_b * (vec0*fac1 - vec1*fac3 + vec3*fac5);
        let inv3 = sign_a * (vec0*fac2 - vec1*fac4 + vec2*fac5);

        // ── Determinant: dot(col0, first_row_of_cofactor) ─────────────────────
        // first row of cofactor = [inv0[0], inv1[0], inv2[0], inv3[0]]
        let row0_lo = simd_swizzle!(inv0, inv1, [0, 0, 4, 4]); // [i0_0,i0_0,i1_0,i1_0]
        let row0_hi = simd_swizzle!(inv2, inv3, [0, 0, 4, 4]); // [i2_0,i2_0,i3_0,i3_0]
        let row0    = simd_swizzle!(row0_lo, row0_hi, [0, 2, 4, 6]); // [i0_0,i1_0,i2_0,i3_0]
        let det     = dot4(x, row0);

        if det.abs() < EPSILON { return None; }
        let rcp = f32x4::splat(1.0 / det);

        Some(Self::from_cols(
            (inv0 * rcp).to_array(),
            (inv1 * rcp).to_array(),
            (inv2 * rcp).to_array(),
            (inv3 * rcp).to_array(),
        ))
    }

    pub fn inverse_trs(self) -> Self {
        // TRS structure exploit: R^T / |col|². Same as scalar version.
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

// ── Mul<Vec4>: broadcast each component + accumulate ─────────────────────────
//
// `simd_swizzle!(v, [k,k,k,k])` broadcasts lane k — exactly like _mm_shuffle_ps.
// Four such broadcasts times four columns, accumulated. Compiler fuses to
// FMA on FMA3/FMA4/NEON targets automatically from the abstract LLVM IR.

impl Mul<Vec4> for Mat4 {
    type Output = Vec4;
    #[inline(always)]
    fn mul(self, v: Vec4) -> Vec4 {
        let vx = simd_swizzle!(v.0, [0, 0, 0, 0]);
        let vy = simd_swizzle!(v.0, [1, 1, 1, 1]);
        let vz = simd_swizzle!(v.0, [2, 2, 2, 2]);
        let vw = simd_swizzle!(v.0, [3, 3, 3, 3]);

        let a0 = f32x4::from_array(self.cols[0]);
        let a1 = f32x4::from_array(self.cols[1]);
        let a2 = f32x4::from_array(self.cols[2]);
        let a3 = f32x4::from_array(self.cols[3]);

        Vec4(a0*vx + a1*vy + a2*vz + a3*vw)
    }
}

// ── Mul<Mat4>: delegate to 4× Mul<Vec4> ──────────────────────────────────────
//
// Four independent chains that the compiler/OOO scheduler can overlap.

impl Mul for Mat4 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        let c0 = self * Vec4::from_array(rhs.cols[0]);
        let c1 = self * Vec4::from_array(rhs.cols[1]);
        let c2 = self * Vec4::from_array(rhs.cols[2]);
        let c3 = self * Vec4::from_array(rhs.cols[3]);
        Self::from_cols(c0.to_array(), c1.to_array(), c2.to_array(), c3.to_array())
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
