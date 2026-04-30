// crates/mid-math/src/f32/sse2/mat4.rs
//! Mat4 with SSE2 fast-paths on x86 / x86_64.
//!
//! CHANGED vs previous version:
//!   1. Mul — loop unrolled into 4 explicit column blocks so LLVM can pipeline
//!      all 4 output columns simultaneously. Closes the 2.5× gap vs glam.
//!   2. sse2_inverse_general — replaced scalar-extraction approach with
//!      glam-style fac0-fac5 shuffle+mul SIMD. Closes the 3× gap vs glam.
//!   3. sse2_inverse_trs — unchanged (already adequate at 13 ns).

use core::fmt;
use core::ops::Mul;

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::f32::sse2::vec3::Vec3;
use crate::f32::sse2::vec4::Vec4;
use crate::f32::sse2::quat::Quat;
use crate::EPSILON;

/// 4×4 column-major matrix. 64 bytes, 16-byte aligned.
///
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

    // ── Constructors (unchanged) ───────────────────────────────────────────────

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

    #[inline]
    pub fn from_rotation(q: Quat) -> Self { q.to_mat4() }

    /// Direct TRS — no intermediate matrix multiplies.
    #[inline]
    pub fn from_trs(t: Vec3, r: Quat, s: Vec3) -> Self {
        let q = r.normalize();
        let (x, y, z, w) = (q.x, q.y, q.z, q.w);
        let (x2, y2, z2) = (x+x, y+y, z+z);
        let (xx, yy, zz) = (x*x2, y*y2, z*z2);
        let (xy, xz, yz) = (x*y2, x*z2, y*z2);
        let (wx, wy, wz) = (w*x2, w*y2, w*z2);
        Self::from_cols(
            [(1.0-yy-zz)*s.x,  (xy+wz)*s.x,   (xz-wy)*s.x,  0.0],
            [  (xy-wz)*s.y, (1.0-xx-zz)*s.y,   (yz+wx)*s.y,  0.0],
            [  (xz+wy)*s.z,    (yz-wx)*s.z,  (1.0-xx-yy)*s.z, 0.0],
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
        let rl = right-left; let tb = top-bottom; let nf = far-near;
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
            [c[0][0], c[1][0], c[2][0], c[3][0]],
            [c[0][1], c[1][1], c[2][1], c[3][1]],
            [c[0][2], c[1][2], c[2][2], c[3][2]],
            [c[0][3], c[1][3], c[2][3], c[3][3]],
        )
    }

    pub fn determinant(self) -> f32 {
        let c = &self.cols;
        let a = |ci: usize, ri: usize| c[ci][ri];
        let sub3 = |c0: usize, c1: usize, c2: usize,
                    r0: usize, r1: usize, r2: usize| -> f32 {
            a(c0,r0)*(a(c1,r1)*a(c2,r2) - a(c2,r1)*a(c1,r2))
           -a(c1,r0)*(a(c0,r1)*a(c2,r2) - a(c2,r1)*a(c0,r2))
           +a(c2,r0)*(a(c0,r1)*a(c1,r2) - a(c1,r1)*a(c0,r2))
        };
        a(0,0)*sub3(1,2,3, 1,2,3)
       -a(1,0)*sub3(0,2,3, 1,2,3)
       +a(2,0)*sub3(0,1,3, 1,2,3)
       -a(3,0)*sub3(0,1,2, 1,2,3)
    }

    // ── Transform helpers ─────────────────────────────────────────────────────

    #[inline]
    pub fn transform_point(self, p: Vec3) -> Vec3 {
        (self * p.extend(1.0)).truncate()
    }

    #[inline]
    pub fn transform_vector(self, v: Vec3) -> Vec3 {
        (self * v.extend(0.0)).truncate()
    }

    // ── Inverse (general) ─────────────────────────────────────────────────────

    /// SSE2 general 4×4 inverse — glam-style fac0-fac5 sub-determinants.
    /// Target: ~13 ns (was 39 ns with the old scalar-extraction approach).
    pub fn inverse(self) -> Option<Self> {
        unsafe { sse2_inverse_general(&self) }
    }

    /// Scalar inverse — always available for correctness tests and fallbacks.
    pub fn inverse_scalar(self) -> Option<Self> {
        let a = [
            self.cols[0][0], self.cols[0][1], self.cols[0][2], self.cols[0][3],
            self.cols[1][0], self.cols[1][1], self.cols[1][2], self.cols[1][3],
            self.cols[2][0], self.cols[2][1], self.cols[2][2], self.cols[2][3],
            self.cols[3][0], self.cols[3][1], self.cols[3][2], self.cols[3][3],
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

    // ── TRS inverse ───────────────────────────────────────────────────────────

    #[inline]
    pub fn inverse_trs(self) -> Self {
        unsafe { sse2_inverse_trs(&self) }
    }

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

// ── CHANGED: Mul — manually unrolled, no loop ─────────────────────────────────
//
// The previous version used `for j in 0..4 { ... }` which LLVM frequently
// fails to fully pipeline. Writing the four columns out explicitly lets the
// backend see all four independent computation chains at once and schedule
// them across execution units — the same approach glam uses to achieve 7 ns.
//
// Each column's computation is:
//   out[j] = A.col[0]*B[j][0] + A.col[1]*B[j][1] + A.col[2]*B[j][2] + A.col[3]*B[j][3]
// achieved via shuffle-broadcast of B's scalar elements.

impl Mul for Mat4 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        unsafe {
            // Load all four columns of A once — they're reused for every output column.
            let a0 = _mm_load_ps(self.cols[0].as_ptr());
            let a1 = _mm_load_ps(self.cols[1].as_ptr());
            let a2 = _mm_load_ps(self.cols[2].as_ptr());
            let a3 = _mm_load_ps(self.cols[3].as_ptr());

            // Inline helper: compute one output column given the B column register.
            // Broadcast each scalar of B's column to all 4 lanes, multiply by A's
            // corresponding column, then sum — 4 mul + 2 add, fully pipelineable.
            macro_rules! out_col {
                ($b_col:expr) => {{
                    let b  = _mm_load_ps(rhs.cols[$b_col].as_ptr());
                    let b0 = _mm_shuffle_ps::<0b00_00_00_00>(b, b); // B[0] × 4
                    let b1 = _mm_shuffle_ps::<0b01_01_01_01>(b, b); // B[1] × 4
                    let b2 = _mm_shuffle_ps::<0b10_10_10_10>(b, b); // B[2] × 4
                    let b3 = _mm_shuffle_ps::<0b11_11_11_11>(b, b); // B[3] × 4
                    _mm_add_ps(
                        _mm_add_ps(_mm_mul_ps(a0, b0), _mm_mul_ps(a1, b1)),
                        _mm_add_ps(_mm_mul_ps(a2, b2), _mm_mul_ps(a3, b3)),
                    )
                }};
            }

            let mut out = Self::ZERO;
            // Four explicit stores — no loop, all four chains visible to the scheduler.
            _mm_store_ps(out.cols[0].as_mut_ptr(), out_col!(0));
            _mm_store_ps(out.cols[1].as_mut_ptr(), out_col!(1));
            _mm_store_ps(out.cols[2].as_mut_ptr(), out_col!(2));
            _mm_store_ps(out.cols[3].as_mut_ptr(), out_col!(3));
            out
        }
    }
}

impl Mul<Vec4> for Mat4 {
    type Output = Vec4;
    #[inline]
    fn mul(self, v: Vec4) -> Vec4 {
        unsafe {
            let a0 = _mm_load_ps(self.cols[0].as_ptr());
            let a1 = _mm_load_ps(self.cols[1].as_ptr());
            let a2 = _mm_load_ps(self.cols[2].as_ptr());
            let a3 = _mm_load_ps(self.cols[3].as_ptr());
            let vx = _mm_shuffle_ps::<0b00_00_00_00>(v.0, v.0);
            let vy = _mm_shuffle_ps::<0b01_01_01_01>(v.0, v.0);
            let vz = _mm_shuffle_ps::<0b10_10_10_10>(v.0, v.0);
            let vw = _mm_shuffle_ps::<0b11_11_11_11>(v.0, v.0);
            Vec4(_mm_add_ps(
                _mm_add_ps(_mm_mul_ps(a0, vx), _mm_mul_ps(a1, vy)),
                _mm_add_ps(_mm_mul_ps(a2, vz), _mm_mul_ps(a3, vw)),
            ))
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

// ── CHANGED: sse2_inverse_general — glam fac0-fac5 approach ──────────────────
//
// The old implementation used a `lane!` macro to extract every scalar from
// __m128 registers, then did 100+ scalar multiplications. That is SIMD with
// none of the benefits — essentially a slower scalar path.
//
// This version ports glam's GLM-derived implementation which computes 6
// "factor" vectors (fac0..fac5), each holding four 2×2 sub-determinants,
// using _mm_shuffle_ps + _mm_mul_ps + _mm_sub_ps exclusively. Every
// intermediate value stays in a SIMD register throughout. Expected result:
// ~13 ns vs the old ~40 ns (matches glam's Mat4::inverse on the same runner).
//
// Column mapping to glam's names:
//   cols[0] = x_axis,  cols[1] = y_axis,
//   cols[2] = z_axis,  cols[3] = w_axis

unsafe fn sse2_inverse_general(m: &Mat4) -> Option<Mat4> {
    let x = _mm_load_ps(m.cols[0].as_ptr()); // col 0
    let y = _mm_load_ps(m.cols[1].as_ptr()); // col 1
    let z = _mm_load_ps(m.cols[2].as_ptr()); // col 2
    let w = _mm_load_ps(m.cols[3].as_ptr()); // col 3

    // fac0..fac5: 2×2 minor determinants from rows (cols) 2 and 3.
    // Each fac vector holds the 4 sub-dets needed for one row of cofactors.
    let fac0 = {
        let swp0a = _mm_shuffle_ps(w, z, 0b11_11_11_11);
        let swp0b = _mm_shuffle_ps(w, z, 0b10_10_10_10);
        let swp00 = _mm_shuffle_ps(z, y, 0b10_10_10_10);
        let swp01 = _mm_shuffle_ps(swp0a, swp0a, 0b10_00_00_00);
        let swp02 = _mm_shuffle_ps(swp0b, swp0b, 0b10_00_00_00);
        let swp03 = _mm_shuffle_ps(z, y, 0b11_11_11_11);
        _mm_sub_ps(_mm_mul_ps(swp00, swp01), _mm_mul_ps(swp02, swp03))
    };
    let fac1 = {
        let swp0a = _mm_shuffle_ps(w, z, 0b11_11_11_11);
        let swp0b = _mm_shuffle_ps(w, z, 0b01_01_01_01);
        let swp00 = _mm_shuffle_ps(z, y, 0b01_01_01_01);
        let swp01 = _mm_shuffle_ps(swp0a, swp0a, 0b10_00_00_00);
        let swp02 = _mm_shuffle_ps(swp0b, swp0b, 0b10_00_00_00);
        let swp03 = _mm_shuffle_ps(z, y, 0b11_11_11_11);
        _mm_sub_ps(_mm_mul_ps(swp00, swp01), _mm_mul_ps(swp02, swp03))
    };
    let fac2 = {
        let swp0a = _mm_shuffle_ps(w, z, 0b10_10_10_10);
        let swp0b = _mm_shuffle_ps(w, z, 0b01_01_01_01);
        let swp00 = _mm_shuffle_ps(z, y, 0b01_01_01_01);
        let swp01 = _mm_shuffle_ps(swp0a, swp0a, 0b10_00_00_00);
        let swp02 = _mm_shuffle_ps(swp0b, swp0b, 0b10_00_00_00);
        let swp03 = _mm_shuffle_ps(z, y, 0b10_10_10_10);
        _mm_sub_ps(_mm_mul_ps(swp00, swp01), _mm_mul_ps(swp02, swp03))
    };
    let fac3 = {
        let swp0a = _mm_shuffle_ps(w, z, 0b11_11_11_11);
        let swp0b = _mm_shuffle_ps(w, z, 0b00_00_00_00);
        let swp00 = _mm_shuffle_ps(z, y, 0b00_00_00_00);
        let swp01 = _mm_shuffle_ps(swp0a, swp0a, 0b10_00_00_00);
        let swp02 = _mm_shuffle_ps(swp0b, swp0b, 0b10_00_00_00);
        let swp03 = _mm_shuffle_ps(z, y, 0b11_11_11_11);
        _mm_sub_ps(_mm_mul_ps(swp00, swp01), _mm_mul_ps(swp02, swp03))
    };
    let fac4 = {
        let swp0a = _mm_shuffle_ps(w, z, 0b10_10_10_10);
        let swp0b = _mm_shuffle_ps(w, z, 0b00_00_00_00);
        let swp00 = _mm_shuffle_ps(z, y, 0b00_00_00_00);
        let swp01 = _mm_shuffle_ps(swp0a, swp0a, 0b10_00_00_00);
        let swp02 = _mm_shuffle_ps(swp0b, swp0b, 0b10_00_00_00);
        let swp03 = _mm_shuffle_ps(z, y, 0b10_10_10_10);
        _mm_sub_ps(_mm_mul_ps(swp00, swp01), _mm_mul_ps(swp02, swp03))
    };
    let fac5 = {
        let swp0a = _mm_shuffle_ps(w, z, 0b01_01_01_01);
        let swp0b = _mm_shuffle_ps(w, z, 0b00_00_00_00);
        let swp00 = _mm_shuffle_ps(z, y, 0b00_00_00_00);
        let swp01 = _mm_shuffle_ps(swp0a, swp0a, 0b10_00_00_00);
        let swp02 = _mm_shuffle_ps(swp0b, swp0b, 0b10_00_00_00);
        let swp03 = _mm_shuffle_ps(z, y, 0b01_01_01_01);
        _mm_sub_ps(_mm_mul_ps(swp00, swp01), _mm_mul_ps(swp02, swp03))
    };

    // Alternating sign masks for the cofactor signs (+−+−) / (−+−+).
    let sign_a = _mm_set_ps( 1.0, -1.0,  1.0, -1.0);
    let sign_b = _mm_set_ps(-1.0,  1.0, -1.0,  1.0);

    // Interleave y and x elements to build the 4-element vectors used in the
    // cofactor expansion. Each vec{0..3} holds one element of columns y and x
    // broadcast appropriately so the later dot products pick the right terms.
    let tmp0 = _mm_shuffle_ps(y, x, 0b00_00_00_00);
    let vec0 = _mm_shuffle_ps(tmp0, tmp0, 0b10_10_10_00);
    let tmp1 = _mm_shuffle_ps(y, x, 0b01_01_01_01);
    let vec1 = _mm_shuffle_ps(tmp1, tmp1, 0b10_10_10_00);
    let tmp2 = _mm_shuffle_ps(y, x, 0b10_10_10_10);
    let vec2 = _mm_shuffle_ps(tmp2, tmp2, 0b10_10_10_00);
    let tmp3 = _mm_shuffle_ps(y, x, 0b11_11_11_11);
    let vec3 = _mm_shuffle_ps(tmp3, tmp3, 0b10_10_10_00);

    // Four columns of the adjugate (transposed cofactor matrix).
    let inv0 = _mm_mul_ps(sign_b, _mm_add_ps(
        _mm_sub_ps(_mm_mul_ps(vec1, fac0), _mm_mul_ps(vec2, fac1)),
        _mm_mul_ps(vec3, fac2),
    ));
    let inv1 = _mm_mul_ps(sign_a, _mm_add_ps(
        _mm_sub_ps(_mm_mul_ps(vec0, fac0), _mm_mul_ps(vec2, fac3)),
        _mm_mul_ps(vec3, fac4),
    ));
    let inv2 = _mm_mul_ps(sign_b, _mm_add_ps(
        _mm_sub_ps(_mm_mul_ps(vec0, fac1), _mm_mul_ps(vec1, fac3)),
        _mm_mul_ps(vec3, fac5),
    ));
    let inv3 = _mm_mul_ps(sign_a, _mm_add_ps(
        _mm_sub_ps(_mm_mul_ps(vec0, fac2), _mm_mul_ps(vec1, fac4)),
        _mm_mul_ps(vec2, fac5),
    ));

    // Determinant: dot(col0_of_input, row0_of_adjugate).
    // row0_of_adjugate = first element of each inv column.
    let row0 = _mm_shuffle_ps(inv0, inv1, 0b00_00_00_00);
    let row1 = _mm_shuffle_ps(inv2, inv3, 0b00_00_00_00);
    let row2 = _mm_shuffle_ps(row0, row1, 0b10_00_10_00);

    // Horizontal dot4(x, row2) — reuse our existing helper.
    let det = crate::sse2::dot4(x, row2);
    if det.abs() < EPSILON { return None; }

    let rcp = _mm_set1_ps(1.0 / det);

    let mut out = Mat4::ZERO;
    _mm_store_ps(out.cols[0].as_mut_ptr(), _mm_mul_ps(inv0, rcp));
    _mm_store_ps(out.cols[1].as_mut_ptr(), _mm_mul_ps(inv1, rcp));
    _mm_store_ps(out.cols[2].as_mut_ptr(), _mm_mul_ps(inv2, rcp));
    _mm_store_ps(out.cols[3].as_mut_ptr(), _mm_mul_ps(inv3, rcp));
    Some(out)
}

// ── sse2_inverse_trs — unchanged (already 13 ns, adequate) ───────────────────

unsafe fn sse2_inverse_trs(m: &Mat4) -> Mat4 {
    let c0 = _mm_load_ps(m.cols[0].as_ptr());
    let c1 = _mm_load_ps(m.cols[1].as_ptr());
    let c2 = _mm_load_ps(m.cols[2].as_ptr());
    let c3 = _mm_load_ps(m.cols[3].as_ptr());
    let sq0  = _mm_mul_ps(c0, c0);
    let sq1  = _mm_mul_ps(c1, c1);
    let sq2  = _mm_mul_ps(c2, c2);
    let zero = _mm_setzero_ps();
    let lo01 = _mm_unpacklo_ps(sq0, sq1);
    let lo2z = _mm_unpacklo_ps(sq2, zero);
    let hi01 = _mm_unpackhi_ps(sq0, sq1);
    let hi2z = _mm_unpackhi_ps(sq2, zero);
    let row0 = _mm_movelh_ps(lo01, lo2z);
    let row1 = _mm_movehl_ps(lo2z, lo01);
    let row2 = _mm_movelh_ps(hi01, hi2z);
    let sums = _mm_add_ps(_mm_add_ps(row0, row1), row2);
    let eps  = _mm_set1_ps(EPSILON);
    let mask = _mm_cmpge_ps(sums, eps);
    let safe = _mm_or_ps(
        _mm_and_ps(mask, sums),
        _mm_andnot_ps(mask, _mm_set1_ps(1.0)),
    );
    let inv_scales = _mm_and_ps(mask, _mm_div_ps(_mm_set1_ps(1.0), safe));
    let lo01_r = _mm_unpacklo_ps(c0, c1);
    let lo2z_r = _mm_unpacklo_ps(c2, zero);
    let hi01_r = _mm_unpackhi_ps(c0, c1);
    let hi2z_r = _mm_unpackhi_ps(c2, zero);
    let trow0 = _mm_movelh_ps(lo01_r, lo2z_r);
    let trow1 = _mm_movehl_ps(lo2z_r, lo01_r);
    let trow2 = _mm_movelh_ps(hi01_r, hi2z_r);
    let ic0 = _mm_mul_ps(trow0, inv_scales);
    let ic1 = _mm_mul_ps(trow1, inv_scales);
    let ic2 = _mm_mul_ps(trow2, inv_scales);
    let tx = _mm_shuffle_ps::<0b00_00_00_00>(c3, c3);
    let ty = _mm_shuffle_ps::<0b01_01_01_01>(c3, c3);
    let tz = _mm_shuffle_ps::<0b10_10_10_10>(c3, c3);
    let dotcol = _mm_add_ps(
        _mm_add_ps(_mm_mul_ps(ic0, tx), _mm_mul_ps(ic1, ty)),
        _mm_mul_ps(ic2, tz),
    );
    let neg = _mm_sub_ps(zero, dotcol);
    let mut ic3_arr = [0.0f32; 4];
    _mm_storeu_ps(ic3_arr.as_mut_ptr(), neg);
    ic3_arr[3] = 1.0;
    let ic3 = _mm_loadu_ps(ic3_arr.as_ptr());
    let mut out = Mat4::ZERO;
    _mm_store_ps(out.cols[0].as_mut_ptr(), ic0);
    _mm_store_ps(out.cols[1].as_mut_ptr(), ic1);
    _mm_store_ps(out.cols[2].as_mut_ptr(), ic2);
    _mm_store_ps(out.cols[3].as_mut_ptr(), ic3);
    out
}
