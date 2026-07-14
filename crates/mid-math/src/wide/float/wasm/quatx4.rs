// crates/mid-math/src/wide/float/wasm/quatx4.rs
//! 4 quaternions SoA — WASM SIMD128.

use core::fmt;
use core::ops::Mul;

#[cfg(target_arch = "wasm32")]
use core::arch::wasm32::{
    v128,
    v128_and, v128_or, v128_andnot, v128_xor,
    i32x4_all_true,
    f32x4_add, f32x4_sub, f32x4_mul, f32x4_div,
    f32x4_neg, f32x4_sqrt,
    f32x4_eq, f32x4_gt,
    f32x4_splat, f32x4_extract_lane,
};
#[cfg(target_arch = "wasm64")]
use core::arch::wasm64::{
    v128,
    v128_and, v128_or, v128_andnot, v128_xor,
    i32x4_all_true,
    f32x4_add, f32x4_sub, f32x4_mul, f32x4_div,
    f32x4_neg, f32x4_sqrt,
    f32x4_eq, f32x4_gt,
    f32x4_splat, f32x4_extract_lane,
};

use crate::wasm::v128_from_f32x4;
use crate::EPSILON;
use crate::f32::wasm::quat::Quat;
use super::f32x4::f32x4;
use super::vec3x4::Vec3x4;

/// 4 quaternions in SoA layout. 64 bytes, 16-byte aligned.
#[derive(Clone, Copy)]
#[repr(C, align(16))]
pub struct QuatX4 {
    pub x: v128,
    pub y: v128,
    pub z: v128,
    pub w: v128,
}

impl QuatX4 {
    pub const IDENTITY: Self = Self {
        x: v128_from_f32x4([0.0; 4]),
        y: v128_from_f32x4([0.0; 4]),
        z: v128_from_f32x4([0.0; 4]),
        w: v128_from_f32x4([1.0; 4]),
    };

    // ── Constructors ──────────────────────────────────────────────────────────

    /// Build from 4 individual quaternions. Quat.x/y/z/w are via Deref.
    #[inline]
    pub fn from_quats(a: Quat, b: Quat, c: Quat, d: Quat) -> Self {
        Self {
            x: v128_from_f32x4([a.x, b.x, c.x, d.x]),
            y: v128_from_f32x4([a.y, b.y, c.y, d.y]),
            z: v128_from_f32x4([a.z, b.z, c.z, d.z]),
            w: v128_from_f32x4([a.w, b.w, c.w, d.w]),
        }
    }

    #[inline(always)]
    pub fn splat(q: Quat) -> Self {
        Self {
            x: f32x4_splat(q.x),
            y: f32x4_splat(q.y),
            z: f32x4_splat(q.z),
            w: f32x4_splat(q.w),
        }
    }

    #[inline(always)]
    pub fn from_slice(s: &[Quat; 4]) -> Self {
        Self::from_quats(s[0], s[1], s[2], s[3])
    }

    #[inline]
    pub fn to_array(self) -> [Quat; 4] {
        let xs = [
            f32x4_extract_lane::<0>(self.x), f32x4_extract_lane::<1>(self.x),
            f32x4_extract_lane::<2>(self.x), f32x4_extract_lane::<3>(self.x),
        ];
        let ys = [
            f32x4_extract_lane::<0>(self.y), f32x4_extract_lane::<1>(self.y),
            f32x4_extract_lane::<2>(self.y), f32x4_extract_lane::<3>(self.y),
        ];
        let zs = [
            f32x4_extract_lane::<0>(self.z), f32x4_extract_lane::<1>(self.z),
            f32x4_extract_lane::<2>(self.z), f32x4_extract_lane::<3>(self.z),
        ];
        let ws = [
            f32x4_extract_lane::<0>(self.w), f32x4_extract_lane::<1>(self.w),
            f32x4_extract_lane::<2>(self.w), f32x4_extract_lane::<3>(self.w),
        ];
        core::array::from_fn(|i| Quat::new(xs[i], ys[i], zs[i], ws[i]))
    }

    #[inline]
    pub fn get(self, lane: usize) -> Quat {
        assert!(lane < 4, "QuatX4::get — lane {lane} out of bounds (max 3)");
        self.to_array()[lane]
    }

    // ── Core ops ──────────────────────────────────────────────────────────────

    #[inline(always)]
    pub fn dot(self, r: Self) -> f32x4 {
        f32x4(f32x4_add(
            f32x4_add(f32x4_mul(self.x,r.x), f32x4_mul(self.y,r.y)),
            f32x4_add(f32x4_mul(self.z,r.z), f32x4_mul(self.w,r.w)),
        ))
    }

    #[inline(always)] pub fn length_sq(self) -> f32x4 { self.dot(self) }

    #[inline]
    pub fn normalize(self) -> Self {
        let lsq      = self.length_sq().0;
        let len      = f32x4_sqrt(lsq);
        let ok       = f32x4_gt(len, f32x4_splat(EPSILON));
        // v128_andnot(a, b) = a & !b
        let safe_len = v128_or(v128_and(ok, len), v128_andnot(f32x4_splat(1.0), ok));
        let inv      = f32x4_div(f32x4_splat(1.0), safe_len);

        let nx = v128_and(ok, f32x4_mul(self.x, inv));
        let ny = v128_and(ok, f32x4_mul(self.y, inv));
        let nz = v128_and(ok, f32x4_mul(self.z, inv));
        let nw = v128_or(
            v128_and(ok, f32x4_mul(self.w, inv)),
            v128_andnot(f32x4_splat(1.0), ok),   // identity w=1 for degenerate lanes
        );
        Self { x:nx, y:ny, z:nz, w:nw }
    }

    #[inline(always)]
    pub fn conjugate(self) -> Self {
        Self {
            x: f32x4_neg(self.x),
            y: f32x4_neg(self.y),
            z: f32x4_neg(self.z),
            w: self.w,
        }
    }

    // ── Hamilton product ───────────────────────────────────────────────────────

    #[inline(always)]
    pub fn mul_quatx4(self, r: Self) -> Self {
        let (lx,ly,lz,lw) = (self.x,self.y,self.z,self.w);
        let (rx,ry,rz,rw) = (r.x,r.y,r.z,r.w);

        Self {
            x: f32x4_add(
                f32x4_add(f32x4_mul(lw,rx), f32x4_mul(lx,rw)),
                f32x4_sub(f32x4_mul(ly,rz), f32x4_mul(lz,ry)),
            ),
            y: f32x4_add(
                f32x4_sub(f32x4_mul(lw,ry), f32x4_mul(lx,rz)),
                f32x4_add(f32x4_mul(ly,rw), f32x4_mul(lz,rx)),
            ),
            z: f32x4_add(
                f32x4_add(f32x4_mul(lw,rz), f32x4_mul(lx,ry)),
                f32x4_sub(f32x4_mul(lz,rw), f32x4_mul(ly,rx)),
            ),
            w: f32x4_sub(
                f32x4_mul(lw,rw),
                f32x4_add(
                    f32x4_add(f32x4_mul(lx,rx), f32x4_mul(ly,ry)),
                    f32x4_mul(lz,rz),
                ),
            ),
        }
    }

    // ── Interpolation ──────────────────────────────────────────────────────────

    #[inline]
    pub fn nlerp(self, rhs: Self, t: f32x4) -> Self {
        let dot = self.dot(rhs).0;
        // sign_mask: 0x80000000 per lane where dot < 0
        let sign_mask = v128_and(dot, f32x4_splat(-0.0));

        let rx = v128_xor(rhs.x, sign_mask);
        let ry = v128_xor(rhs.y, sign_mask);
        let rz = v128_xor(rhs.z, sign_mask);
        let rw = v128_xor(rhs.w, sign_mask);

        let tt = t.0;
        let lerped = Self {
            x: f32x4_add(self.x, f32x4_mul(f32x4_sub(rx,self.x), tt)),
            y: f32x4_add(self.y, f32x4_mul(f32x4_sub(ry,self.y), tt)),
            z: f32x4_add(self.z, f32x4_mul(f32x4_sub(rz,self.z), tt)),
            w: f32x4_add(self.w, f32x4_mul(f32x4_sub(rw,self.w), tt)),
        };
        lerped.normalize()
    }

    // ── Rotation ──────────────────────────────────────────────────────────────

    #[inline]
    pub fn rotate(self, v: Vec3x4) -> Vec3x4 {
        let qxyz = Vec3x4 { x:self.x, y:self.y, z:self.z };
        let two  = f32x4_splat(2.0);

        let cross1 = qxyz.cross(v);
        let t = Vec3x4 {
            x: f32x4_mul(two, cross1.x),
            y: f32x4_mul(two, cross1.y),
            z: f32x4_mul(two, cross1.z),
        };
        let wt = Vec3x4 {
            x: f32x4_mul(self.w, t.x),
            y: f32x4_mul(self.w, t.y),
            z: f32x4_mul(self.w, t.z),
        };
        let cross2 = qxyz.cross(t);
        Vec3x4 {
            x: f32x4_add(v.x, f32x4_add(wt.x, cross2.x)),
            y: f32x4_add(v.y, f32x4_add(wt.y, cross2.y)),
            z: f32x4_add(v.z, f32x4_add(wt.z, cross2.z)),
        }
    }

    #[inline]
    pub fn is_finite(self) -> bool {
        self.to_array().iter().all(|q| q.is_finite())
    }
}

// ── Operators ─────────────────────────────────────────────────────────────────

impl Mul for QuatX4 {
    type Output = Self;
    #[inline(always)] fn mul(self, r: Self) -> Self { self.mul_quatx4(r) }
}

impl PartialEq for QuatX4 {
    fn eq(&self, r: &Self) -> bool {
        i32x4_all_true(f32x4_eq(self.x,r.x))
            && i32x4_all_true(f32x4_eq(self.y,r.y))
            && i32x4_all_true(f32x4_eq(self.z,r.z))
            && i32x4_all_true(f32x4_eq(self.w,r.w))
    }
}
impl Default for QuatX4 { fn default() -> Self { Self::IDENTITY } }

impl fmt::Debug for QuatX4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let a = self.to_array();
        write!(f, "QuatX4([{:?}, {:?}, {:?}, {:?}])", a[0], a[1], a[2], a[3])
    }
}
impl fmt::Display for QuatX4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let a = self.to_array();
        write!(f, "[{}, {}, {}, {}]", a[0], a[1], a[2], a[3])
    }
}

impl From<[Quat; 4]> for QuatX4 { #[inline] fn from(a:[Quat;4])->Self { Self::from_slice(&a) } }
impl From<QuatX4> for [Quat; 4]  { #[inline] fn from(v:QuatX4) ->Self { v.to_array() } }
