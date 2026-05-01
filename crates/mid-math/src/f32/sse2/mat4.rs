// crates/mid-math/src/f32/sse2/mat4.rs
//! Mat4 with SSE2 fast-paths on x86 / x86_64.
//!
//! CHANGED vs previous version:
//!   1. Mul — ported glam/cglm column-parallel FMA pattern.
//!      Old: load A cols first, tree-reduce per output col.
//!      New: load R cols once, load L col-by-col, 4 independent accumulators.
//!      Adds `sse2_fmadd` helper: emits vfmadd213ps on +fma, falls back on SSE2.
//!   2. Mul<Vec4> — sequential FMA chain (glam mul_vec4 pattern).
//!      Reduces 4-mul+3-add to 1-mul+3-fmadd, same dependency structure as glam.
//!   3. sse2_inverse_general — unchanged (already glam fac0-fac5 approach).
//!   4. sse2_inverse_trs — unchanged (already adequate).

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

// ── CHANGED: fmadd helper ─────────────────────────────────────────────────────
//
// Emits a single vfmadd213ps on FMA3 targets (Haswell+ / Zen+).
// Enable with RUSTFLAGS="-C target-feature=+fma" or "-C target-cpu=native".
// Falls back to mul+add on SSE2-only targets (2010 MBP Sandy Bridge baseline).
//
// The FMA path is the key reason glam achieves ~7 ns on CI while SSE2-only
// targets will land closer to ~10-13 ns — still a substantial win over the old
// ~17 ns tree-reduction approach.
#[inline(always)]
unsafe fn sse2_fmadd(a: __m128, b: __m128, c: __m128) -> __m128 {
    // cfg!(target_feature) is resolved at compile time, so one branch is dead code.
    #[cfg(target_feature = "fma")]
    {
        // _mm_fmadd_ps(a, b, c) = a*b + c
        // Available in core::arch::x86_64 when fma target_feature is set.
        // Emits: vfmadd213ps (or vfmadd231ps depending on operand order LLVM picks).
        _mm_fmadd_ps(a, b, c)
    }
    #[cfg(not(target_feature = "fma"))]
    {
        // SSE2 fallback: two instructions. LLVM cannot combine them into FMA
        // without the feature flag even with -O3.
        _mm_add_ps(c, _mm_mul_ps(a, b))
    }
}

// ── CHANGED: Mul — column-parallel FMA pattern (glam / cglm approach) ────────
//
// Previous version loaded all 4 lhs columns first, then processed each output
// column with a tree-reduction: (a0*b0 + a1*b1) + (a2*b2 + a3*b3).
// The tree tip add created an extra latency stall.
//
// This version ports the pattern used in both glam (src/f32/sse2/mat4.rs) and
// cglm (include/cglm/simd/sse2/mat4.h glm_mat4_mul_sse2):
//
//   1. Load all 4 rhs columns as broadcast sources.
//   2. Load each lhs column once per pass.
//   3. In each pass, FMA-accumulate into 4 independent output registers (v0..v3).
//
// The four accumulator chains (v0, v1, v2, v3) are data-independent throughout.
// The CPU's out-of-order engine pipelines all 4 chains simultaneously.
//
// Instruction count per multiply:
//   FMA  targets: 4 loads(rhs) + 4×(1 load(lhs) + 4 ops) = 4 + 16 = 20 ops
//   SSE2 targets: same structure but fmadd = mul+add → 4 + 28 = 32 ops
//
// Expected perf: FMA ≈ 7-8 ns, SSE2 ≈ 11-13 ns (was 17.58 ns).

impl Mul for Mat4 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        unsafe {
            // Load all four columns of rhs once.
            // These stay in SIMD registers for the entire computation — no re-loads.
            let r0 = _mm_load_ps(rhs.cols[0].as_ptr());
            let r1 = _mm_load_ps(rhs.cols[1].as_ptr());
            let r2 = _mm_load_ps(rhs.cols[2].as_ptr());
            let r3 = _mm_load_ps(rhs.cols[3].as_ptr());

            // Pass 0: lhs col 0 × scalar element [0] of each rhs column.
            // _mm_shuffle_ps::<0b00_00_00_00> broadcasts lane 0 to all 4 lanes.
            // v0..v3 are fully independent: zero data-dependency between them.
            let l = _mm_load_ps(self.cols[0].as_ptr());
            let mut v0 = _mm_mul_ps(_mm_shuffle_ps::<0b00_00_00_00>(r0, r0), l);
            let mut v1 = _mm_mul_ps(_mm_shuffle_ps::<0b00_00_00_00>(r1, r1), l);
            let mut v2 = _mm_mul_ps(_mm_shuffle_ps::<0b00_00_00_00>(r2, r2), l);
            let mut v3 = _mm_mul_ps(_mm_shuffle_ps::<0b00_00_00_00>(r3, r3), l);

            // Pass 1: FMA with lhs col 1 × element [1] of each rhs column.
            // Each vN only depends on its own prior value — the 4 chains stay parallel.
            let l = _mm_load_ps(self.cols[1].as_ptr());
            v0 = sse2_fmadd(_mm_shuffle_ps::<0b01_01_01_01>(r0, r0), l, v0);
            v1 = sse2_fmadd(_mm_shuffle_ps::<0b01_01_01_01>(r1, r1), l, v1);
            v2 = sse2_fmadd(_mm_shuffle_ps::<0b01_01_01_01>(r2, r2), l, v2);
            v3 = sse2_fmadd(_mm_shuffle_ps::<0b01_01_01_01>(r3, r3), l, v3);

            // Pass 2: FMA with lhs col 2 × element [2] of each rhs column.
            let l = _mm_load_ps(self.cols[2].as_ptr());
            v0 = sse2_fmadd(_mm_shuffle_ps::<0b10_10_10_10>(r0, r0), l, v0);
            v1 = sse2_fmadd(_mm_shuffle_ps::<0b10_10_10_10>(r1, r1), l, v1);
            v2 = sse2_fmadd(_mm_shuffle_ps::<0b10_10_10_10>(r2, r2), l, v2);
            v3 = sse2_fmadd(_mm_shuffle_ps::<0b10_10_10_10>(r3, r3), l, v3);

            // Pass 3: FMA with lhs col 3 × element [3] of each rhs column.
            let l = _mm_load_ps(self.cols[3].as_ptr());
            v0 = sse2_fmadd(_mm_shuffle_ps::<0b11_11_11_11>(r0, r0), l, v0);
            v1 = sse2_fmadd(_mm_shuffle_ps::<0b11_11_11_11>(r1, r1), l, v1);
            v2 = sse2_fmadd(_mm_shuffle_ps::<0b11_11_11_11>(r2, r2), l, v2);
            v3 = sse2_fmadd(_mm_shuffle_ps::<0b11_11_11_11>(r3, r3), l, v3);

            let mut out = Self::ZERO;
            _mm_store_ps(out.cols[0].as_mut_ptr(), v0);
            _mm_store_ps(out.cols[1].as_mut_ptr(), v1);
            _mm_store_ps(out.cols[2].as_mut_ptr(), v2);
            _mm_store_ps(out.cols[3].as_mut_ptr(), v3);
            out
        }
    }
}

// ── CHANGED: Mul<Vec4> — sequential FMA chain (glam mul_vec4 pattern) ────────
//
// Previous version used a tree-reduction:
//   (a0*vx + a1*vy) + (a2*vz + a3*vw)   → 4 mul + 3 add = 7 ops
//
// glam uses a sequential FMA chain (src/f32/sse2/mat4.rs mul_vec4):
//   res  = a0 * vx
//   res += a1 * vy   (fmadd: a1*vy + res)
//   res += a2 * vz
//   res += a3 * vw
//
// FMA targets: 1 mul + 3 fmadd = 4 ops.
// SSE2 targets: 1 mul + 3 (mul+add) = 7 ops (same count, but LLVM schedules
//               the sequential form better than the tree for single-vector use).
//
// Used by transform_point / transform_vector via the * operator.

impl Mul<Vec4> for Mat4 {
    type Output = Vec4;
    #[inline]
    fn mul(self, v: Vec4) -> Vec4 {
        unsafe {
            let a0 = _mm_load_ps(self.cols[0].as_ptr());
            let a1 = _mm_load_ps(self.cols[1].as_ptr());
            let a2 = _mm_load_ps(self.cols[2].as_ptr());
            let a3 = _mm_load_ps(self.cols[3].as_ptr());

            // Broadcast each scalar component of v to all 4 lanes.
            let vx = _mm_shuffle_ps::<0b00_00_00_00>(v.0, v.0);
            let vy = _mm_shuffle_ps::<0b01_01_01_01>(v.0, v.0);
            let vz = _mm_shuffle_ps::<0b10_10_10_10>(v.0, v.0);
            let vw = _mm_shuffle_ps::<0b11_11_11_11>(v.0, v.0);

            // Sequential FMA chain — same pattern as glam's mul_vec4.
            // Single dependency chain but FMA halves instruction count on modern CPUs.
            let mut res = _mm_mul_ps(a0, vx);
            res = sse2_fmadd(a1, vy, res);
            res = sse2_fmadd(a2, vz, res);
            res = sse2_fmadd(a3, vw, res);
            Vec4(res)
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

// ── sse2_inverse_general — glam fac0-fac5 approach (unchanged) ───────────────
//
// Already at target performance (~16 ns). No changes.

unsafe fn sse2_inverse_general(m: &Mat4) -> Option<Mat4> {
    let x = _mm_load_ps(m.cols[0].as_ptr()); // col 0
    let y = _mm_load_ps(m.cols[1].as_ptr()); // col 1
    let z = _mm_load_ps(m.cols[2].as_ptr()); // col 2
    let w = _mm_load_ps(m.cols[3].as_ptr()); // col 3

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

    let sign_a = _mm_set_ps( 1.0, -1.0,  1.0, -1.0);
    let sign_b = _mm_set_ps(-1.0,  1.0, -1.0,  1.0);

    let tmp0 = _mm_shuffle_ps(y, x, 0b00_00_00_00);
    let vec0 = _mm_shuffle_ps(tmp0, tmp0, 0b10_10_10_00);
    let tmp1 = _mm_shuffle_ps(y, x, 0b01_01_01_01);
    let vec1 = _mm_shuffle_ps(tmp1, tmp1, 0b10_10_10_00);
    let tmp2 = _mm_shuffle_ps(y, x, 0b10_10_10_10);
    let vec2 = _mm_shuffle_ps(tmp2, tmp2, 0b10_10_10_00);
    let tmp3 = _mm_shuffle_ps(y, x, 0b11_11_11_11);
    let vec3 = _mm_shuffle_ps(tmp3, tmp3, 0b10_10_10_00);

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

    let row0 = _mm_shuffle_ps(inv0, inv1, 0b00_00_00_00);
    let row1 = _mm_shuffle_ps(inv2, inv3, 0b00_00_00_00);
    let row2 = _mm_shuffle_ps(row0, row1, 0b10_00_10_00);

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
