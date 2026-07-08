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

#[repr(C)]
union SignCast { f: [f32; 4], v: float32x4_t }

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

    /// Full TRS — scale, then rotate, then translate.
    ///
    /// **Precondition: `r` must already be normalized** — same contract as
    /// your SSE2 `Mat4::from_trs` (which already does exactly this: assert,
    /// don't renormalize). The NEON path hadn't gotten the same treatment —
    /// it was calling `r.normalize()` unconditionally, which is most of why
    /// `mat4_construction/from_trs` was ~2x slower than glam here despite
    /// SSE2 already beating/matching glam on the same benchmark.
    #[inline]
    pub fn from_trs(t: Vec3, r: Quat, s: Vec3) -> Self {
        debug_assert!(r.is_normalized(), "from_trs: r must be normalized");
        let (x,y,z,w) = (r.x,r.y,r.z,r.w);
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

    // ── Wide SIMD batch transforms ────────────────────────────────────────────
    // Lane-by-lane via vgetq_lane_f32/vsetq_lane_f32 rather than
    // to_array()/from_vec3s() — those route through crate::Vec3 (the
    // dispatched alias), which this module can't assume matches its own
    // local Vec3 (this module always compiles for cross-referencing
    // regardless of which backend crate::Mat4 actually resolves to).

    /// Transform 4 points packed in a `Vec3x4` (SoA layout) by this matrix.
    #[inline]
    pub fn transform_vec3x4(self, v: crate::wide::float::neon::vec3x4::Vec3x4) -> crate::wide::float::neon::vec3x4::Vec3x4 {
        unsafe {
            let mut out_x = [0.0f32; 4];
            let mut out_y = [0.0f32; 4];
            let mut out_z = [0.0f32; 4];
            let (xs, ys, zs) = (
                [vgetq_lane_f32::<0>(v.x), vgetq_lane_f32::<1>(v.x), vgetq_lane_f32::<2>(v.x), vgetq_lane_f32::<3>(v.x)],
                [vgetq_lane_f32::<0>(v.y), vgetq_lane_f32::<1>(v.y), vgetq_lane_f32::<2>(v.y), vgetq_lane_f32::<3>(v.y)],
                [vgetq_lane_f32::<0>(v.z), vgetq_lane_f32::<1>(v.z), vgetq_lane_f32::<2>(v.z), vgetq_lane_f32::<3>(v.z)],
            );
            for i in 0..4 {
                let r = self.transform_point(Vec3::new(xs[i], ys[i], zs[i]));
                out_x[i] = vgetq_lane_f32::<0>(r.0);
                out_y[i] = vgetq_lane_f32::<1>(r.0);
                out_z[i] = vgetq_lane_f32::<2>(r.0);
            }
            crate::wide::float::neon::vec3x4::Vec3x4 {
                x: vld1q_f32(out_x.as_ptr()),
                y: vld1q_f32(out_y.as_ptr()),
                z: vld1q_f32(out_z.as_ptr()),
            }
        }
    }

    /// Transform 4 direction vectors packed in a `Vec3x4` (ignores translation).
    #[inline]
    pub fn transform_vec3x4_dir(self, v: crate::wide::float::neon::vec3x4::Vec3x4) -> crate::wide::float::neon::vec3x4::Vec3x4 {
        unsafe {
            let mut out_x = [0.0f32; 4];
            let mut out_y = [0.0f32; 4];
            let mut out_z = [0.0f32; 4];
            let (xs, ys, zs) = (
                [vgetq_lane_f32::<0>(v.x), vgetq_lane_f32::<1>(v.x), vgetq_lane_f32::<2>(v.x), vgetq_lane_f32::<3>(v.x)],
                [vgetq_lane_f32::<0>(v.y), vgetq_lane_f32::<1>(v.y), vgetq_lane_f32::<2>(v.y), vgetq_lane_f32::<3>(v.y)],
                [vgetq_lane_f32::<0>(v.z), vgetq_lane_f32::<1>(v.z), vgetq_lane_f32::<2>(v.z), vgetq_lane_f32::<3>(v.z)],
            );
            for i in 0..4 {
                let r = self.transform_vector(Vec3::new(xs[i], ys[i], zs[i]));
                out_x[i] = vgetq_lane_f32::<0>(r.0);
                out_y[i] = vgetq_lane_f32::<1>(r.0);
                out_z[i] = vgetq_lane_f32::<2>(r.0);
            }
            crate::wide::float::neon::vec3x4::Vec3x4 {
                x: vld1q_f32(out_x.as_ptr()),
                y: vld1q_f32(out_y.as_ptr()),
                z: vld1q_f32(out_z.as_ptr()),
            }
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

        let r = super::quat::quat_from_rotation_axes(c0, c1, c2);
        (t, r, Vec3::new(sx, sy, sz))
    }

    // ── Inverse ────────────────────────────────────────────────────────────────

    /// Inverse via vectorized cofactor expansion — see `inverse_raw` below.
    /// `inverse_scalar` is kept as a reference/fallback implementation; it's
    /// the one that was wired up here before (pure scalar, ~3x slower on
    /// this bench — 48ns vs glam's 16ns for `mat4/inverse_general`).
    pub fn inverse(self) -> Option<Self> {
        let (m, det) = self.inverse_raw();
        if det.abs() < EPSILON { return None; }
        let rcp = 1.0 / det;
        unsafe {
            Some(Self {
                x_axis: Vec4(vmulq_n_f32(m.x_axis.0, rcp)),
                y_axis: Vec4(vmulq_n_f32(m.y_axis.0, rcp)),
                z_axis: Vec4(vmulq_n_f32(m.z_axis.0, rcp)),
                w_axis: Vec4(vmulq_n_f32(m.w_axis.0, rcp)),
            })
        }
    }

    /// Vectorized cofactor-expansion inverse. Ported from glam's NEON
    /// `Mat4::inverse_checked` (itself based on g-truc/glm's
    /// `glm_mat4_inverse`) — adapted to mid-math's `Vec4`/`Mat4` field
    /// layout and const-generic intrinsic call style.
    ///
    /// Returns the *unscaled* cofactor matrix plus the raw determinant
    /// (`a[0]*row2[0] + a[1]*row2[1] + ...` via `self.x_axis · row2`) —
    /// the caller divides by `det` once at the end. This is the same
    /// split `inverse_scalar` uses (`inv[]` array computed first, `det`
    /// from a handful of its entries, then one division pass at the end),
    /// just done with zero scalar lane extraction until that final dot.
    ///
    /// `swizzle0266`'s single-lane patch (`vsetq_lane_f32::<2>`) is the one
    /// spot that touches an individual lane rather than shuffling whole
    /// vectors — everything else here is full-width vector ops.
    fn inverse_raw(self) -> (Self, f32) {
        unsafe {
            let swizzle3377 = |a: float32x4_t, b: float32x4_t| -> float32x4_t {
                let r = vuzp2q_f32(a, b);
                vtrn2q_f32(r, r)
            };
            let swizzle2266 = |a: float32x4_t, b: float32x4_t| -> float32x4_t {
                let r = vuzp1q_f32(a, b);
                vtrn2q_f32(r, r)
            };
            let swizzle0046 = |a: float32x4_t, b: float32x4_t| -> float32x4_t {
                let r = vuzp1q_f32(a, a);
                vuzp1q_f32(r, b)
            };
            let swizzle1155 = |a: float32x4_t, b: float32x4_t| -> float32x4_t {
                let r = vzip1q_f32(a, b);
                vzip2q_f32(r, r)
            };
            let swizzle0044 = |a: float32x4_t, b: float32x4_t| -> float32x4_t {
                let r = vuzp1q_f32(a, b);
                vtrn1q_f32(r, r)
            };
            let swizzle0266 = |a: float32x4_t, b: float32x4_t| -> float32x4_t {
                let r = vuzp1q_f32(a, b);
                vsetq_lane_f32::<2>(vgetq_lane_f32::<2>(b), r)
            };
            let swizzle0246 = |a: float32x4_t, b: float32x4_t| -> float32x4_t { vuzp1q_f32(a, b) };

            let fac0 = {
                let swp0a = swizzle3377(self.w_axis.0, self.z_axis.0);
                let swp0b = swizzle2266(self.w_axis.0, self.z_axis.0);
                let swp00 = swizzle2266(self.z_axis.0, self.y_axis.0);
                let swp01 = swizzle0046(swp0a, swp0a);
                let swp02 = swizzle0046(swp0b, swp0b);
                let swp03 = swizzle3377(self.z_axis.0, self.y_axis.0);
                vsubq_f32(vmulq_f32(swp00, swp01), vmulq_f32(swp02, swp03))
            };
            let fac1 = {
                let swp0a = swizzle3377(self.w_axis.0, self.z_axis.0);
                let swp0b = swizzle1155(self.w_axis.0, self.z_axis.0);
                let swp00 = swizzle1155(self.z_axis.0, self.y_axis.0);
                let swp01 = swizzle0046(swp0a, swp0a);
                let swp02 = swizzle0046(swp0b, swp0b);
                let swp03 = swizzle3377(self.z_axis.0, self.y_axis.0);
                vsubq_f32(vmulq_f32(swp00, swp01), vmulq_f32(swp02, swp03))
            };
            let fac2 = {
                let swp0a = swizzle2266(self.w_axis.0, self.z_axis.0);
                let swp0b = swizzle1155(self.w_axis.0, self.z_axis.0);
                let swp00 = swizzle1155(self.z_axis.0, self.y_axis.0);
                let swp01 = swizzle0046(swp0a, swp0a);
                let swp02 = swizzle0046(swp0b, swp0b);
                let swp03 = swizzle2266(self.z_axis.0, self.y_axis.0);
                vsubq_f32(vmulq_f32(swp00, swp01), vmulq_f32(swp02, swp03))
            };
            let fac3 = {
                let swp0a = swizzle3377(self.w_axis.0, self.z_axis.0);
                let swp0b = swizzle0044(self.w_axis.0, self.z_axis.0);
                let swp00 = swizzle0044(self.z_axis.0, self.y_axis.0);
                let swp01 = swizzle0046(swp0a, swp0a);
                let swp02 = swizzle0046(swp0b, swp0b);
                let swp03 = swizzle3377(self.z_axis.0, self.y_axis.0);
                vsubq_f32(vmulq_f32(swp00, swp01), vmulq_f32(swp02, swp03))
            };
            let fac4 = {
                let swp0a = swizzle2266(self.w_axis.0, self.z_axis.0);
                let swp0b = swizzle0044(self.w_axis.0, self.z_axis.0);
                let swp00 = swizzle0044(self.z_axis.0, self.y_axis.0);
                let swp01 = swizzle0046(swp0a, swp0a);
                let swp02 = swizzle0046(swp0b, swp0b);
                let swp03 = swizzle2266(self.z_axis.0, self.y_axis.0);
                vsubq_f32(vmulq_f32(swp00, swp01), vmulq_f32(swp02, swp03))
            };
            let fac5 = {
                let swp0a = swizzle1155(self.w_axis.0, self.z_axis.0);
                let swp0b = swizzle0044(self.w_axis.0, self.z_axis.0);
                let swp00 = swizzle0044(self.z_axis.0, self.y_axis.0);
                let swp01 = swizzle0046(swp0a, swp0a);
                let swp02 = swizzle0046(swp0b, swp0b);
                let swp03 = swizzle1155(self.z_axis.0, self.y_axis.0);
                vsubq_f32(vmulq_f32(swp00, swp01), vmulq_f32(swp02, swp03))
            };

            const SIGN_A: float32x4_t = unsafe { SignCast { f: [-1.0, 1.0, -1.0, 1.0] }.v };
            const SIGN_B: float32x4_t = unsafe { SignCast { f: [1.0, -1.0, 1.0, -1.0] }.v };

            let temp0 = swizzle0044(self.y_axis.0, self.x_axis.0);
            let vec0  = swizzle0266(temp0, temp0);
            let temp1 = swizzle1155(self.y_axis.0, self.x_axis.0);
            let vec1  = swizzle0266(temp1, temp1);
            let temp2 = swizzle2266(self.y_axis.0, self.x_axis.0);
            let vec2  = swizzle0266(temp2, temp2);
            let temp3 = swizzle3377(self.y_axis.0, self.x_axis.0);
            let vec3  = swizzle0266(temp3, temp3);

            let inv0 = vmulq_f32(SIGN_B, vaddq_f32(
                vsubq_f32(vmulq_f32(vec1, fac0), vmulq_f32(vec2, fac1)), vmulq_f32(vec3, fac2),
            ));
            let inv1 = vmulq_f32(SIGN_A, vaddq_f32(
                vsubq_f32(vmulq_f32(vec0, fac0), vmulq_f32(vec2, fac3)), vmulq_f32(vec3, fac4),
            ));
            let inv2 = vmulq_f32(SIGN_B, vaddq_f32(
                vsubq_f32(vmulq_f32(vec0, fac1), vmulq_f32(vec1, fac3)), vmulq_f32(vec3, fac5),
            ));
            let inv3 = vmulq_f32(SIGN_A, vaddq_f32(
                vsubq_f32(vmulq_f32(vec0, fac2), vmulq_f32(vec1, fac4)), vmulq_f32(vec2, fac5),
            ));

            let row0 = swizzle0044(inv0, inv1);
            let row1 = swizzle0044(inv2, inv3);
            let row2 = swizzle0246(row0, row1);

            let det = vaddvq_f32(vmulq_f32(self.x_axis.0, row2));

            (Self {
                x_axis: Vec4(inv0), y_axis: Vec4(inv1),
                z_axis: Vec4(inv2), w_axis: Vec4(inv3),
            }, det)
        }
    }

    /// Pure-scalar cofactor inverse. Kept for reference/fallback — see
    /// `inverse_raw` for the vectorized path `inverse()` actually uses now.
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
