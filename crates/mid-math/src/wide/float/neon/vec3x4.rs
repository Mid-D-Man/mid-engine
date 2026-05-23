// crates/mid-math/src/wide/float/neon/vec3x4.rs
//! 4 × Vec3 packed in SoA layout — NEON, aarch64.
//!
//! ## vld4q_f32 / vst4q_f32 — the headline optimization
//!
//! Each Vec3 is `#[repr(transparent)] float32x4_t` with layout `[x, y, z, 0.0]`.
//! Four Vec3s stored contiguously in memory:
//!   `[x0,y0,z0,0, x1,y1,z1,0, x2,y2,z2,0, x3,y3,z3,0]`  — 16 floats.
//!
//! `vld4q_f32(ptr)` deinterleaves into SoA in ONE instruction:
//!   val[0]=[x0,x1,x2,x3], val[1]=[y0,y1,y2,y3], val[2]=[z0,z1,z2,z3]
//!
//! vs SSE2's 7-shuffle sequence (unpacklo×2, unpackhi×2, movelh×2, movehl×1).
//! `vst4q_f32(ptr, packed)` reverses the operation in ONE instruction.
//!
//! ## Other NEON wins
//!   - `vfmaq_f32` (mandatory FMA): lerp is 1 instruction per component
//!   - `vbslq_f32`: select is 1 instruction per component
//!   - `vrsqrteq_f32` + `vrsqrtsq_f32`: rsqrt NR uses dedicated hardware step

use core::arch::aarch64::*;
use core::fmt;
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use super::mask4::Mask4;
use super::f32x4::{f32x4, rsqrt_nr};
use crate::Vec3;
use crate::EPSILON;

/// 4 × Vec3 in SoA layout. 48 bytes, 16-byte aligned. Backed by 3 × `float32x4_t`.
#[derive(Clone, Copy)]
#[repr(C, align(16))]
pub struct Vec3x4 {
    /// x-components: [x0, x1, x2, x3]
    pub x: float32x4_t,
    /// y-components: [y0, y1, y2, y3]
    pub y: float32x4_t,
    /// z-components: [z0, z1, z2, z3]
    pub z: float32x4_t,
}

impl Vec3x4 {
    // ── Constants ─────────────────────────────────────────────────────────────

    pub const ZERO: Self = Self {
        x: unsafe { core::mem::transmute([0.0f32; 4]) },
        y: unsafe { core::mem::transmute([0.0f32; 4]) },
        z: unsafe { core::mem::transmute([0.0f32; 4]) },
    };
    pub const X: Self = Self {
        x: unsafe { core::mem::transmute([1.0f32, 1.0, 1.0, 1.0]) },
        y: unsafe { core::mem::transmute([0.0f32; 4]) },
        z: unsafe { core::mem::transmute([0.0f32; 4]) },
    };
    pub const Y: Self = Self {
        x: unsafe { core::mem::transmute([0.0f32; 4]) },
        y: unsafe { core::mem::transmute([1.0f32, 1.0, 1.0, 1.0]) },
        z: unsafe { core::mem::transmute([0.0f32; 4]) },
    };
    pub const Z: Self = Self {
        x: unsafe { core::mem::transmute([0.0f32; 4]) },
        y: unsafe { core::mem::transmute([0.0f32; 4]) },
        z: unsafe { core::mem::transmute([1.0f32, 1.0, 1.0, 1.0]) },
    };

    // ── Constructors ──────────────────────────────────────────────────────────

    /// Build from a slice of 4 Vec3s using `vld4q_f32` — ONE instruction AoS→SoA.
    ///
    /// Vec3 is `#[repr(transparent)] float32x4_t` with layout `[x, y, z, 0.0]`.
    /// `vld4q_f32` loads 16 floats and deinterleaves into 4 channels simultaneously.
    #[inline]
    pub fn from_slice(s: &[Vec3; 4]) -> Self {
        unsafe {
            // s = [x0,y0,z0,0, x1,y1,z1,0, x2,y2,z2,0, x3,y3,z3,0]
            // vld4q_f32 deinterleaves into (val[0]..val[3]) in ONE instruction.
            let loaded = vld4q_f32(s.as_ptr() as *const f32);
            Self {
                x: loaded.0, // [x0, x1, x2, x3]
                y: loaded.1, // [y0, y1, y2, y3]
                z: loaded.2, // [z0, z1, z2, z3]
                             // loaded.3 (padding zeros) discarded
            }
        }
    }

    /// Build from 4 individual Vec3s. Stores to stack then calls `from_slice`.
    #[inline(always)]
    pub fn from_vec3s(a: Vec3, b: Vec3, c: Vec3, d: Vec3) -> Self {
        Self::from_slice(&[a, b, c, d])
    }

    /// Broadcast a single Vec3 to all 4 lanes.
    #[inline(always)]
    pub fn splat(v: Vec3) -> Self {
        unsafe {
            Self {
                x: vdupq_n_f32(v.x),
                y: vdupq_n_f32(v.y),
                z: vdupq_n_f32(v.z),
            }
        }
    }

    /// Extract all 4 vectors using `vst4q_f32` — ONE instruction SoA→AoS.
    #[inline]
    pub fn to_array(self) -> [Vec3; 4] {
        unsafe {
            let packed = float32x4x4_t(self.x, self.y, self.z, vdupq_n_f32(0.0));
            // Allocate output on stack.  Vec3 is repr(transparent) float32x4_t.
            let mut out = [Vec3::ZERO; 4];
            // vst4q_f32 interleaves channels back to AoS:
            // [x0,y0,z0,0, x1,y1,z1,0, x2,y2,z2,0, x3,y3,z3,0] = 4 Vec3s ✓
            vst4q_f32(out.as_mut_ptr() as *mut f32, packed);
            out
        }
    }

    #[inline(always)]
    pub fn write_to_slice(self, s: &mut [Vec3; 4]) {
        let a = self.to_array();
        s.copy_from_slice(&a);
    }

    #[inline]
    pub fn get(self, lane: usize) -> Vec3 {
        assert!(lane < 4, "Vec3x4::get — lane {lane} out of bounds (max 3)");
        self.to_array()[lane]
    }

    // ── Arithmetic ────────────────────────────────────────────────────────────

    /// Element-wise multiply (NOT dot product — use `.dot()`).
    #[inline(always)]
    pub fn mul_elem(self, r: Self) -> Self {
        unsafe {
            Self {
                x: vmulq_f32(self.x, r.x),
                y: vmulq_f32(self.y, r.y),
                z: vmulq_f32(self.z, r.z),
            }
        }
    }

    /// Scale each of the 4 vectors by a per-lane scalar.
    #[inline(always)]
    pub fn scale(self, s: f32x4) -> Self {
        unsafe {
            Self {
                x: vmulq_f32(self.x, s.0),
                y: vmulq_f32(self.y, s.0),
                z: vmulq_f32(self.z, s.0),
            }
        }
    }

    #[inline(always)]
    pub fn scale_uniform(self, s: f32) -> Self {
        unsafe {
            Self {
                x: vmulq_n_f32(self.x, s),
                y: vmulq_n_f32(self.y, s),
                z: vmulq_n_f32(self.z, s),
            }
        }
    }

    /// Fused multiply-add: `self * b + c`. Uses mandatory AArch64 FMA.
    ///
    /// `vfmaq_f32(c, self, b)` = `c + self*b` — one instruction per component.
    #[inline(always)]
    pub fn madd(self, b: Self, c: Self) -> Self {
        unsafe {
            Self {
                x: vfmaq_f32(c.x, self.x, b.x),
                y: vfmaq_f32(c.y, self.y, b.y),
                z: vfmaq_f32(c.z, self.z, b.z),
            }
        }
    }

    // ── Geometric ops ─────────────────────────────────────────────────────────

    /// 4 independent dot products simultaneously. Returns `f32x4[i] = dot(self[i], rhs[i])`.
    #[inline(always)]
    pub fn dot(self, rhs: Self) -> f32x4 {
        unsafe {
            let xx = vmulq_f32(self.x, rhs.x);
            let yy = vmulq_f32(self.y, rhs.y);
            let zz = vmulq_f32(self.z, rhs.z);
            f32x4(vaddq_f32(vaddq_f32(xx, yy), zz))
        }
    }

    /// 4 independent cross products simultaneously.
    #[inline(always)]
    pub fn cross(self, rhs: Self) -> Self {
        unsafe {
            Self {
                x: vsubq_f32(vmulq_f32(self.y, rhs.z), vmulq_f32(self.z, rhs.y)),
                y: vsubq_f32(vmulq_f32(self.z, rhs.x), vmulq_f32(self.x, rhs.z)),
                z: vsubq_f32(vmulq_f32(self.x, rhs.y), vmulq_f32(self.y, rhs.x)),
            }
        }
    }

    #[inline(always)] pub fn length_sq(self) -> f32x4 { self.dot(self) }

    #[inline(always)]
    pub fn length(self) -> f32x4 {
        f32x4(unsafe { vsqrtq_f32(self.length_sq().0) })
    }

    /// Normalize all 4 vectors via `vrsqrteq_f32` + Newton-Raphson step.
    ///
    /// Degenerate lanes (length ≈ 0) produce zero — guarded by EPSILON mask.
    #[inline]
    pub fn normalize(self) -> Self {
        unsafe {
            let len_sq  = self.length_sq().0;
            let inv_len = rsqrt_nr(len_sq);
            // Zero out degenerate lanes
            let ok  = vcgtq_f32(len_sq, vdupq_n_f32(EPSILON * EPSILON));
            let inv = vreinterpretq_f32_u32(vandq_u32(
                vreinterpretq_u32_f32(inv_len), ok,
            ));
            Self {
                x: vmulq_f32(self.x, inv),
                y: vmulq_f32(self.y, inv),
                z: vmulq_f32(self.z, inv),
            }
        }
    }

    /// Normalize via full sqrt + divide (IEEE754-accurate).
    #[inline]
    pub fn normalize_precise(self) -> Self {
        unsafe {
            let len_sq = self.length_sq().0;
            let len    = vsqrtq_f32(len_sq);
            let ok     = vcgtq_f32(len, vdupq_n_f32(EPSILON));
            // Safe divisor: 1.0 in degenerate lanes (result zeroed by mask anyway)
            let safe = vbslq_f32(ok, len, vdupq_n_f32(1.0));
            let inv  = vdivq_f32(vdupq_n_f32(1.0), safe);
            let inv  = vreinterpretq_f32_u32(vandq_u32(
                vreinterpretq_u32_f32(inv), ok,
            ));
            Self {
                x: vmulq_f32(self.x, inv),
                y: vmulq_f32(self.y, inv),
                z: vmulq_f32(self.z, inv),
            }
        }
    }

    // ── Interpolation / component-wise ────────────────────────────────────────

    /// Per-lane lerp using mandatory AArch64 FMA.
    ///
    /// `vfmaq_f32(self, diff, t)` = `self + diff*t` — one instruction per component.
    #[inline(always)]
    pub fn lerp(self, rhs: Self, t: f32x4) -> Self {
        unsafe {
            let dx = vsubq_f32(rhs.x, self.x);
            let dy = vsubq_f32(rhs.y, self.y);
            let dz = vsubq_f32(rhs.z, self.z);
            Self {
                x: vfmaq_f32(self.x, dx, t.0),
                y: vfmaq_f32(self.y, dy, t.0),
                z: vfmaq_f32(self.z, dz, t.0),
            }
        }
    }

    #[inline(always)]
    pub fn min(self, r: Self) -> Self {
        unsafe {
            Self {
                x: vminq_f32(self.x, r.x),
                y: vminq_f32(self.y, r.y),
                z: vminq_f32(self.z, r.z),
            }
        }
    }
    #[inline(always)]
    pub fn max(self, r: Self) -> Self {
        unsafe {
            Self {
                x: vmaxq_f32(self.x, r.x),
                y: vmaxq_f32(self.y, r.y),
                z: vmaxq_f32(self.z, r.z),
            }
        }
    }

    // ── Branchless select ─────────────────────────────────────────────────────

    /// Per-lane bitselect using `vbslq_f32` — ONE instruction per component.
    #[inline(always)]
    pub fn select(mask: Mask4, if_true: Self, if_false: Self) -> Self {
        unsafe {
            Self {
                x: vbslq_f32(mask.0, if_true.x, if_false.x),
                y: vbslq_f32(mask.0, if_true.y, if_false.y),
                z: vbslq_f32(mask.0, if_true.z, if_false.z),
            }
        }
    }

    #[inline(always)]
    pub fn length_lt(self, r: Self) -> Mask4 {
        self.length_sq().cmplt(r.length_sq())
    }

    #[inline]
    pub fn is_finite(self) -> bool {
        self.to_array().iter().all(|v| v.is_finite())
    }
}

// ── Operators ─────────────────────────────────────────────────────────────────

impl Add for Vec3x4 {
    type Output = Self;
    #[inline(always)]
    fn add(self, r: Self) -> Self {
        unsafe { Self { x: vaddq_f32(self.x, r.x), y: vaddq_f32(self.y, r.y), z: vaddq_f32(self.z, r.z) } }
    }
}
impl AddAssign for Vec3x4 { #[inline(always)] fn add_assign(&mut self, r: Self) { *self = *self + r; } }

impl Sub for Vec3x4 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, r: Self) -> Self {
        unsafe { Self { x: vsubq_f32(self.x, r.x), y: vsubq_f32(self.y, r.y), z: vsubq_f32(self.z, r.z) } }
    }
}
impl SubAssign for Vec3x4 { #[inline(always)] fn sub_assign(&mut self, r: Self) { *self = *self - r; } }

impl Neg for Vec3x4 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        unsafe { Self { x: vnegq_f32(self.x), y: vnegq_f32(self.y), z: vnegq_f32(self.z) } }
    }
}
impl Mul for Vec3x4 { type Output = Self; #[inline(always)] fn mul(self, r: Self) -> Self { self.mul_elem(r) } }
impl MulAssign for Vec3x4 { #[inline(always)] fn mul_assign(&mut self, r: Self) { *self = *self * r; } }
impl Mul<f32x4> for Vec3x4 { type Output = Self; #[inline(always)] fn mul(self, s: f32x4) -> Self { self.scale(s) } }
impl Mul<f32> for Vec3x4 { type Output = Self; #[inline(always)] fn mul(self, s: f32) -> Self { self.scale_uniform(s) } }

impl PartialEq for Vec3x4 {
    fn eq(&self, r: &Self) -> bool {
        unsafe {
            vminvq_u32(vceqq_f32(self.x, r.x)) == u32::MAX
                && vminvq_u32(vceqq_f32(self.y, r.y)) == u32::MAX
                && vminvq_u32(vceqq_f32(self.z, r.z)) == u32::MAX
        }
    }
}

impl Default for Vec3x4 { fn default() -> Self { Self::ZERO } }

impl fmt::Debug for Vec3x4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let a = self.to_array();
        write!(f, "Vec3x4([{:?}, {:?}, {:?}, {:?}])", a[0], a[1], a[2], a[3])
    }
}
impl fmt::Display for Vec3x4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let a = self.to_array();
        write!(f, "[{}, {}, {}, {}]", a[0], a[1], a[2], a[3])
    }
}
impl From<[Vec3; 4]> for Vec3x4 { #[inline] fn from(a: [Vec3; 4]) -> Self { Self::from_slice(&a) } }
impl From<Vec3x4> for [Vec3; 4] { #[inline] fn from(v: Vec3x4) -> Self { v.to_array() } }
