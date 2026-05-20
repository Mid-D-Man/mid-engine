// crates/mid-math/src/f32/wasm/mat4.rs
//! Mat4 with WASM SIMD128 fast-paths on wasm32/wasm64.
//!
//! Reference: cglm include/cglm/simd/wasm/mat4.h + glam src/f32/wasm/mat4.rs
//!
//! WASM SIMD128 shuffle note:
//!   i32x4_shuffle::<L0,L1,L2,L3>(a, b)
//!   Lane i of result = (Li < 4) ? a[Li] : b[Li-4]
//!
//!   vs SSE2 _mm_shuffle_ps::<IMM>(a, b):
//!   result[0] = a[IMM & 3]
//!   result[1] = a[(IMM>>2) & 3]
//!   result[2] = b[(IMM>>4) & 3]
//!   result[3] = b[(IMM>>6) & 3]
//!
//! v128_andnot(a, b) = a & ~b   ← WASM
//! _mm_andnot_ps(a, b) = ~a & b ← SSE2  (argument order REVERSED)
//!
//! No native FMA in WASM SIMD128 base spec.
//! LLVM may fuse mul+add chains on relaxed-simd capable hosts.
//!
//! Mul<Vec4>  — splat+multiply+accumulate: 4 mul + 3 add = 7 instructions
//! Mul<Mat4>  — 4× Mul<Vec4>, four independent chains
//! inverse    — cofactor expansion (fac0-fac5), ported from cglm/glam WASM reference
//! inverse_trs — scalar TRS-specific path (structure exploit, no SIMD needed)

#[cfg(target_arch = "wasm32")]
use core::arch::wasm32::*;
#[cfg(target_arch = "wasm64")]
use core::arch::wasm64::*;

use core::fmt;
use core::ops::Mul;

use crate::f32::wasm::vec3::Vec3;
use crate::f32::wasm::vec4::Vec4;
use crate::f32::wasm::quat::Quat;
use crate::wasm::v128_from_f32x4;
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

    pub fn ortho_rh(left:f32, right:f32, bottom:f32, top:f32, near:f32, far:f32) -> Self {
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

    #[inline]
    pub fn transform_point(self, p: Vec3) -> Vec3 {
        (self * p.extend(1.0)).truncate()
    }

    #[inline]
    pub fn transform_vector(self, v: Vec3) -> Vec3 {
        (self * v.extend(0.0)).truncate()
    }

    // ── Inverse ───────────────────────────────────────────────────────────────

    /// General Mat4 inverse via cofactor expansion with WASM SIMD shuffles.
    ///
    /// Ported from cglm `glm_mat4_inv_wasm` / glam `src/f32/wasm/mat4.rs`.
    /// Returns `None` if singular.
    pub fn inverse(self) -> Option<Self> {
        unsafe { wasm_inverse_general(&self) }
    }

    /// Scalar general inverse — correctness reference, also used for debug.
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
        let id = 1.0/det;
        for x in inv.iter_mut() { *x *= id; }
        Some(Self::from_cols(
            [inv[0],inv[1],inv[2],inv[3]],
            [inv[4],inv[5],inv[6],inv[7]],
            [inv[8],inv[9],inv[10],inv[11]],
            [inv[12],inv[13],inv[14],inv[15]],
        ))
    }

    pub fn inverse_trs(self) -> Self {
        // TRS structure: R^-1 = R^T, S^-1 = 1/diag.
        // No SIMD benefit over the scalar exploit; keep scalar.
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

// ── Mul<Vec4> — splat + multiply + accumulate (cglm glm_mat4_mulv_wasm pattern) ──
//
// col0*v.x + col1*v.y + col2*v.z + col3*v.w
// No native FMA in WASM SIMD128 base spec — LLVM may fuse mul+add at IR level.

impl Mul<Vec4> for Mat4 {
    type Output = Vec4;
    #[inline(always)]
    fn mul(self, v: Vec4) -> Vec4 {
        unsafe {
            let a0 = v128_load(self.cols[0].as_ptr() as *const v128);
            let a1 = v128_load(self.cols[1].as_ptr() as *const v128);
            let a2 = v128_load(self.cols[2].as_ptr() as *const v128);
            let a3 = v128_load(self.cols[3].as_ptr() as *const v128);

            // Broadcast each component of v to all 4 lanes.
            // i32x4_shuffle::<0,0,0,0>(x,x) = splat lane 0
            let vx = i32x4_shuffle::<0, 0, 4, 4>(v.0, v.0);
            let vx = i32x4_shuffle::<0, 2, 4, 6>(vx, vx);
            let vy = i32x4_shuffle::<1, 1, 5, 5>(v.0, v.0);
            let vy = i32x4_shuffle::<0, 2, 4, 6>(vy, vy);
            let vz = i32x4_shuffle::<2, 2, 6, 6>(v.0, v.0);
            let vz = i32x4_shuffle::<0, 2, 4, 6>(vz, vz);
            let vw = i32x4_shuffle::<3, 3, 7, 7>(v.0, v.0);
            let vw = i32x4_shuffle::<0, 2, 4, 6>(vw, vw);

            let mut res = f32x4_mul(a0, vx);
            res = f32x4_add(res, f32x4_mul(a1, vy));
            res = f32x4_add(res, f32x4_mul(a2, vz));
            res = f32x4_add(res, f32x4_mul(a3, vw));
            Vec4(res)
        }
    }
}

// ── Mul<Mat4> — four independent Mul<Vec4> chains ────────────────────────────

impl Mul for Mat4 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        unsafe {
            let c0 = self * Vec4(v128_load(rhs.cols[0].as_ptr() as *const v128));
            let c1 = self * Vec4(v128_load(rhs.cols[1].as_ptr() as *const v128));
            let c2 = self * Vec4(v128_load(rhs.cols[2].as_ptr() as *const v128));
            let c3 = self * Vec4(v128_load(rhs.cols[3].as_ptr() as *const v128));
            let mut out = Self::ZERO;
            v128_store(out.cols[0].as_mut_ptr() as *mut v128, c0.0);
            v128_store(out.cols[1].as_mut_ptr() as *mut v128, c1.0);
            v128_store(out.cols[2].as_mut_ptr() as *mut v128, c2.0);
            v128_store(out.cols[3].as_mut_ptr() as *mut v128, c3.0);
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

// ── WASM cofactor inverse — ported from cglm glm_mat4_inv_wasm ───────────────
//
// Shuffle translation key (SSE2 IMM → WASM lane indices):
//   _mm_shuffle_ps::<0b11_11_11_11>(a,a) = [a3,a3,a3,a3] → i32x4_shuffle::<3,3,7,7>(a,a) then collapse
//   _mm_movehl_ps(a,b) = [b2,b3,a2,a3]  → i32x4_shuffle::<6,7,2,3>(a,b)
//   _mm_movelh_ps(a,b) = [a0,a1,b0,b1]  → i32x4_shuffle::<0,1,4,5>(a,b)
//
// v128_andnot(a,b) = a & ~b  ← NOTE: WASM argument order OPPOSITE to SSE2
// In SSE2: _mm_andnot_ps(a,b) = ~a & b
// To get "~a & b" in WASM: v128_andnot(b, a)  (b & ~a)

unsafe fn wasm_inverse_general(m: &Mat4) -> Option<Mat4> {
    let x = v128_load(m.cols[0].as_ptr() as *const v128);
    let y = v128_load(m.cols[1].as_ptr() as *const v128);
    let z = v128_load(m.cols[2].as_ptr() as *const v128);
    let w = v128_load(m.cols[3].as_ptr() as *const v128);

    // fac0 = cofactors involving z,w rows and cols 2,3
    let fac0 = {
        // swp0a = [w3, w3, z3, z3]
        let swp0a = i32x4_shuffle::<3, 3, 7, 7>(w, z);
        // swp0b = [w2, w2, z2, z2]
        let swp0b = i32x4_shuffle::<2, 2, 6, 6>(w, z);
        // swp00 = [z2, z2, y2, y2]
        let swp00 = i32x4_shuffle::<2, 2, 6, 6>(z, y);
        // swp01 = [swp0a[0], swp0a[0], swp0a[4], swp0a[6]]
        let swp01 = i32x4_shuffle::<0, 0, 4, 6>(swp0a, swp0a);
        let swp02 = i32x4_shuffle::<0, 0, 4, 6>(swp0b, swp0b);
        let swp03 = i32x4_shuffle::<3, 3, 7, 7>(z, y);
        f32x4_sub(f32x4_mul(swp00, swp01), f32x4_mul(swp02, swp03))
    };

    let fac1 = {
        let swp0a = i32x4_shuffle::<3, 3, 7, 7>(w, z);
        let swp0b = i32x4_shuffle::<1, 1, 5, 5>(w, z);
        let swp00 = i32x4_shuffle::<1, 1, 5, 5>(z, y);
        let swp01 = i32x4_shuffle::<0, 0, 4, 6>(swp0a, swp0a);
        let swp02 = i32x4_shuffle::<0, 0, 4, 6>(swp0b, swp0b);
        let swp03 = i32x4_shuffle::<3, 3, 7, 7>(z, y);
        f32x4_sub(f32x4_mul(swp00, swp01), f32x4_mul(swp02, swp03))
    };

    let fac2 = {
        let swp0a = i32x4_shuffle::<2, 2, 6, 6>(w, z);
        let swp0b = i32x4_shuffle::<1, 1, 5, 5>(w, z);
        let swp00 = i32x4_shuffle::<1, 1, 5, 5>(z, y);
        let swp01 = i32x4_shuffle::<0, 0, 4, 6>(swp0a, swp0a);
        let swp02 = i32x4_shuffle::<0, 0, 4, 6>(swp0b, swp0b);
        let swp03 = i32x4_shuffle::<2, 2, 6, 6>(z, y);
        f32x4_sub(f32x4_mul(swp00, swp01), f32x4_mul(swp02, swp03))
    };

    let fac3 = {
        let swp0a = i32x4_shuffle::<3, 3, 7, 7>(w, z);
        let swp0b = i32x4_shuffle::<0, 0, 4, 4>(w, z);
        let swp00 = i32x4_shuffle::<0, 0, 4, 4>(z, y);
        let swp01 = i32x4_shuffle::<0, 0, 4, 6>(swp0a, swp0a);
        let swp02 = i32x4_shuffle::<0, 0, 4, 6>(swp0b, swp0b);
        let swp03 = i32x4_shuffle::<3, 3, 7, 7>(z, y);
        f32x4_sub(f32x4_mul(swp00, swp01), f32x4_mul(swp02, swp03))
    };

    let fac4 = {
        let swp0a = i32x4_shuffle::<2, 2, 6, 6>(w, z);
        let swp0b = i32x4_shuffle::<0, 0, 4, 4>(w, z);
        let swp00 = i32x4_shuffle::<0, 0, 4, 4>(z, y);
        let swp01 = i32x4_shuffle::<0, 0, 4, 6>(swp0a, swp0a);
        let swp02 = i32x4_shuffle::<0, 0, 4, 6>(swp0b, swp0b);
        let swp03 = i32x4_shuffle::<2, 2, 6, 6>(z, y);
        f32x4_sub(f32x4_mul(swp00, swp01), f32x4_mul(swp02, swp03))
    };

    let fac5 = {
        let swp0a = i32x4_shuffle::<1, 1, 5, 5>(w, z);
        let swp0b = i32x4_shuffle::<0, 0, 4, 4>(w, z);
        let swp00 = i32x4_shuffle::<0, 0, 4, 4>(z, y);
        let swp01 = i32x4_shuffle::<0, 0, 4, 6>(swp0a, swp0a);
        let swp02 = i32x4_shuffle::<0, 0, 4, 6>(swp0b, swp0b);
        let swp03 = i32x4_shuffle::<1, 1, 5, 5>(z, y);
        f32x4_sub(f32x4_mul(swp00, swp01), f32x4_mul(swp02, swp03))
    };

    // sign patterns: col0=[+,-,+,-]  col1=[-,+,-,+]
    let sign_a = v128_from_f32x4([ 1.0, -1.0,  1.0, -1.0]);
    let sign_b = v128_from_f32x4([-1.0,  1.0, -1.0,  1.0]);

    // Build vec0..vec3: broadcast element from (y,x) combined for each row
    let tmp0  = i32x4_shuffle::<0, 0, 4, 4>(y, x);
    let vec0  = i32x4_shuffle::<0, 2, 4, 6>(tmp0, tmp0);
    let tmp1  = i32x4_shuffle::<1, 1, 5, 5>(y, x);
    let vec1  = i32x4_shuffle::<0, 2, 4, 6>(tmp1, tmp1);
    let tmp2  = i32x4_shuffle::<2, 2, 6, 6>(y, x);
    let vec2  = i32x4_shuffle::<0, 2, 4, 6>(tmp2, tmp2);
    let tmp3  = i32x4_shuffle::<3, 3, 7, 7>(y, x);
    let vec3  = i32x4_shuffle::<0, 2, 4, 6>(tmp3, tmp3);

    // Compute cofactor columns
    let inv0 = f32x4_mul(sign_b,
        f32x4_add(f32x4_sub(f32x4_mul(vec1, fac0), f32x4_mul(vec2, fac1)),
                  f32x4_mul(vec3, fac2)));
    let inv1 = f32x4_mul(sign_a,
        f32x4_add(f32x4_sub(f32x4_mul(vec0, fac0), f32x4_mul(vec2, fac3)),
                  f32x4_mul(vec3, fac4)));
    let inv2 = f32x4_mul(sign_b,
        f32x4_add(f32x4_sub(f32x4_mul(vec0, fac1), f32x4_mul(vec1, fac3)),
                  f32x4_mul(vec3, fac5)));
    let inv3 = f32x4_mul(sign_a,
        f32x4_add(f32x4_sub(f32x4_mul(vec0, fac2), f32x4_mul(vec1, fac4)),
                  f32x4_mul(vec2, fac5)));

    // Determinant: dot of first column of m with first row of cofactor matrix
    // row0 of cofactors = [inv0[0], inv1[0], inv2[0], inv3[0]]
    let row0_lo = i32x4_shuffle::<0, 0, 4, 4>(inv0, inv1); // [inv0[0], inv0[0], inv1[0], inv1[0]]
    let row0_hi = i32x4_shuffle::<0, 0, 4, 4>(inv2, inv3);
    let row0    = i32x4_shuffle::<0, 2, 4, 6>(row0_lo, row0_hi);

    // dot(x, row0)
    let dot_v  = f32x4_mul(x, row0);
    // horizontal sum
    let s0 = f32x4_add(dot_v, i32x4_shuffle::<1, 0, 3, 2>(dot_v, dot_v));
    let s1 = f32x4_add(s0,    i32x4_shuffle::<2, 3, 0, 1>(s0,    s0));
    let det = f32x4_extract_lane::<0>(s1);

    if det.abs() < EPSILON { return None; }

    let rcp = f32x4_splat(1.0 / det);
    let mut out = Mat4::ZERO;
    v128_store(out.cols[0].as_mut_ptr() as *mut v128, f32x4_mul(inv0, rcp));
    v128_store(out.cols[1].as_mut_ptr() as *mut v128, f32x4_mul(inv1, rcp));
    v128_store(out.cols[2].as_mut_ptr() as *mut v128, f32x4_mul(inv2, rcp));
    v128_store(out.cols[3].as_mut_ptr() as *mut v128, f32x4_mul(inv3, rcp));
    Some(out)
  }
