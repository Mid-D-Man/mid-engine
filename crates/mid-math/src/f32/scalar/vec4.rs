// crates/mid-math/src/f32/scalar/vec4.rs
//! Scalar Vec4 — fallback for non-SIMD targets and correctness reference.
//!
//! On x86/x86_64, `sse2/vec4.rs` supersedes this — additions made here
//! must be mirrored there too (and to neon/vec4.rs, wasm/vec4.rs) for parity.

use core::fmt;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use crate::{BVec4, EPSILON};

/// 4D vector. 16 bytes, align(16). Scalar storage.
#[derive(Debug, Clone, Copy)]
#[repr(C, align(16))]
pub struct Vec4 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl Vec4 {
    pub const ZERO:         Self = Self { x: 0.0,  y: 0.0,  z: 0.0,  w: 0.0  };
    pub const ONE:          Self = Self { x: 1.0,  y: 1.0,  z: 1.0,  w: 1.0  };
    pub const NEG_ONE:      Self = Self { x:-1.0,  y:-1.0,  z:-1.0,  w:-1.0  };
    pub const X:            Self = Self { x: 1.0,  y: 0.0,  z: 0.0,  w: 0.0  };
    pub const Y:            Self = Self { x: 0.0,  y: 1.0,  z: 0.0,  w: 0.0  };
    pub const Z:            Self = Self { x: 0.0,  y: 0.0,  z: 1.0,  w: 0.0  };
    pub const W:            Self = Self { x: 0.0,  y: 0.0,  z: 0.0,  w: 1.0  };
    pub const NEG_X:        Self = Self { x:-1.0,  y: 0.0,  z: 0.0,  w: 0.0  };
    pub const NEG_Y:        Self = Self { x: 0.0,  y:-1.0,  z: 0.0,  w: 0.0  };
    pub const NEG_Z:        Self = Self { x: 0.0,  y: 0.0,  z:-1.0,  w: 0.0  };
    pub const NEG_W:        Self = Self { x: 0.0,  y: 0.0,  z: 0.0,  w:-1.0  };
    pub const MIN:          Self = Self { x: f32::MIN, y: f32::MIN, z: f32::MIN, w: f32::MIN };
    pub const MAX:          Self = Self { x: f32::MAX, y: f32::MAX, z: f32::MAX, w: f32::MAX };
    pub const NAN:          Self = Self { x: f32::NAN, y: f32::NAN, z: f32::NAN, w: f32::NAN };
    pub const INFINITY:     Self = Self { x: f32::INFINITY,     y: f32::INFINITY,     z: f32::INFINITY,     w: f32::INFINITY     };
    pub const NEG_INFINITY: Self = Self { x: f32::NEG_INFINITY, y: f32::NEG_INFINITY, z: f32::NEG_INFINITY, w: f32::NEG_INFINITY };

    // ── Constructors ──────────────────────────────────────────────────────────

    #[inline(always)] pub fn new(x: f32, y: f32, z: f32, w: f32) -> Self { Self { x, y, z, w } }
    #[inline(always)] pub fn splat(v: f32) -> Self { Self { x: v, y: v, z: v, w: v } }
    #[inline(always)] pub fn from_array(a: [f32; 4]) -> Self { Self::new(a[0], a[1], a[2], a[3]) }
    #[inline(always)] pub fn to_array(self) -> [f32; 4] { [self.x, self.y, self.z, self.w] }

    /// Apply a closure to each component.
    #[inline]
    pub fn map<F: Fn(f32) -> f32>(self, f: F) -> Self {
        Self::new(f(self.x), f(self.y), f(self.z), f(self.w))
    }

    /// Conditional select component-wise based on `mask`.
    #[inline]
    pub fn select(mask: BVec4, if_true: Self, if_false: Self) -> Self {
        Self::new(
            if mask.x { if_true.x } else { if_false.x },
            if mask.y { if_true.y } else { if_false.y },
            if mask.z { if_true.z } else { if_false.z },
            if mask.w { if_true.w } else { if_false.w },
        )
    }

    /// Write components to `slice[0..4]`. Panics if slice is too short.
    #[inline]
    pub fn write_to_slice(self, slice: &mut [f32]) {
        slice[0] = self.x; slice[1] = self.y; slice[2] = self.z; slice[3] = self.w;
    }

    #[inline(always)] pub fn with_x(mut self, x: f32) -> Self { self.x = x; self }
    #[inline(always)] pub fn with_y(mut self, y: f32) -> Self { self.y = y; self }
    #[inline(always)] pub fn with_z(mut self, z: f32) -> Self { self.z = z; self }
    #[inline(always)] pub fn with_w(mut self, w: f32) -> Self { self.w = w; self }

    /// Truncate to Vec3 by dropping `w`.
    #[inline(always)]
    pub fn truncate(self) -> super::vec3::Vec3 { super::vec3::Vec3::new(self.x, self.y, self.z) }

    /// Perspective divide: `(x/w, y/w, z/w)`. Returns `Vec3::ZERO` if `w ≈ 0`.
    #[inline]
    pub fn project(self) -> super::vec3::Vec3 {
        let recip = if self.w.abs() < EPSILON { 0.0 } else { 1.0 / self.w };
        super::vec3::Vec3::new(self.x * recip, self.y * recip, self.z * recip)
    }

    // ── Dot ───────────────────────────────────────────────────────────────────

    #[inline(always)]
    pub fn dot(self, r: Self) -> f32 { self.x*r.x + self.y*r.y + self.z*r.z + self.w*r.w }
    /// Dot product broadcast into a `Vec4`.
    #[inline(always)] pub fn dot_into_vec(self, r: Self) -> Self { Self::splat(self.dot(r)) }

    // ── Length ────────────────────────────────────────────────────────────────

    #[inline(always)] pub fn length_sq(self) -> f32 { self.dot(self) }
    /// Alias for `length_sq` (glam-compat name).
    #[inline(always)] pub fn length_squared(self) -> f32 { self.length_sq() }
    #[inline(always)] pub fn length(self) -> f32 { self.length_sq().sqrt() }
    #[inline(always)]
    pub fn length_recip(self) -> f32 {
        let l = self.length();
        if l < EPSILON { 0.0 } else { 1.0 / l }
    }

    // ── Normalize ─────────────────────────────────────────────────────────────

    #[inline(always)]
    pub fn normalize(self) -> Self {
        let l = self.length();
        if l < EPSILON { Self::ZERO } else { self / l }
    }
    #[inline(always)]
    pub fn try_normalize(self) -> Option<Self> {
        let rcp = self.length_recip();
        if rcp > 0.0 && rcp.is_finite() { Some(self * rcp) } else { None }
    }
    /// Normalize, returning `fallback` if zero-length.
    #[inline(always)]
    pub fn normalize_or(self, fallback: Self) -> Self { self.try_normalize().unwrap_or(fallback) }
    #[inline(always)] pub fn normalize_or_zero(self) -> Self { self.normalize_or(Self::ZERO) }
    #[inline(always)] pub fn is_normalized(self) -> bool { (self.length_sq() - 1.0).abs() <= 2e-4 }
    /// Normalize and return `(normalized, original_length)` in one pass.
    #[inline]
    pub fn normalize_and_length(self) -> (Self, f32) {
        let l = self.length();
        if l < EPSILON { (Self::ZERO, 0.0) } else { (self / l, l) }
    }

    // ── Distance ──────────────────────────────────────────────────────────────

    #[inline(always)] pub fn distance(self, r: Self) -> f32 { (self - r).length() }
    #[inline(always)] pub fn distance_sq(self, r: Self) -> f32 { (self - r).length_sq() }
    /// Alias for `distance_sq` (glam-compat name).
    #[inline(always)] pub fn distance_squared(self, r: Self) -> f32 { self.distance_sq(r) }

    // ── Reduction ─────────────────────────────────────────────────────────────

    #[inline(always)] pub fn min_element(self) -> f32 { self.x.min(self.y).min(self.z).min(self.w) }
    #[inline(always)] pub fn max_element(self) -> f32 { self.x.max(self.y).max(self.z).max(self.w) }
    #[inline(always)] pub fn element_sum(self) -> f32 { self.x + self.y + self.z + self.w }
    #[inline(always)] pub fn element_product(self) -> f32 { self.x * self.y * self.z * self.w }

    /// Index (0-3) of the minimum component.
    #[inline]
    pub fn min_position(self) -> usize {
        let (mut idx, mut val) = (0usize, self.x);
        if self.y < val { idx = 1; val = self.y; }
        if self.z < val { idx = 2; val = self.z; }
        if self.w < val { idx = 3; }
        idx
    }
    /// Index (0-3) of the maximum component.
    #[inline]
    pub fn max_position(self) -> usize {
        let (mut idx, mut val) = (0usize, self.x);
        if self.y > val { idx = 1; val = self.y; }
        if self.z > val { idx = 2; val = self.z; }
        if self.w > val { idx = 3; }
        idx
    }

    // ── Comparisons → BVec4 ───────────────────────────────────────────────────

    #[inline(always)] pub fn cmpeq(self, r: Self) -> BVec4 { BVec4::new(self.x==r.x, self.y==r.y, self.z==r.z, self.w==r.w) }
    #[inline(always)] pub fn cmpne(self, r: Self) -> BVec4 { BVec4::new(self.x!=r.x, self.y!=r.y, self.z!=r.z, self.w!=r.w) }
    #[inline(always)] pub fn cmpge(self, r: Self) -> BVec4 { BVec4::new(self.x>=r.x, self.y>=r.y, self.z>=r.z, self.w>=r.w) }
    #[inline(always)] pub fn cmpgt(self, r: Self) -> BVec4 { BVec4::new(self.x> r.x, self.y> r.y, self.z> r.z, self.w> r.w) }
    #[inline(always)] pub fn cmple(self, r: Self) -> BVec4 { BVec4::new(self.x<=r.x, self.y<=r.y, self.z<=r.z, self.w<=r.w) }
    #[inline(always)] pub fn cmplt(self, r: Self) -> BVec4 { BVec4::new(self.x< r.x, self.y< r.y, self.z< r.z, self.w< r.w) }

    // ── Sign ──────────────────────────────────────────────────────────────────

    #[inline(always)]
    pub fn signum(self) -> Self { Self::new(self.x.signum(), self.y.signum(), self.z.signum(), self.w.signum()) }
    #[inline(always)]
    pub fn copysign(self, rhs: Self) -> Self {
        Self::new(self.x.copysign(rhs.x), self.y.copysign(rhs.y),
                  self.z.copysign(rhs.z), self.w.copysign(rhs.w))
    }
    /// Bitmask of sign bits: bit 0=x, bit 1=y, bit 2=z, bit 3=w.
    #[inline]
    pub fn is_negative_bitmask(self) -> u32 {
          (self.x.is_sign_negative() as u32)
        | ((self.y.is_sign_negative() as u32) << 1)
        | ((self.z.is_sign_negative() as u32) << 2)
        | ((self.w.is_sign_negative() as u32) << 3)
    }

    // ── Predicates ────────────────────────────────────────────────────────────

    #[inline(always)]
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite() && self.w.is_finite()
    }
    #[inline(always)]
    pub fn is_nan(self) -> bool { self.x.is_nan() || self.y.is_nan() || self.z.is_nan() || self.w.is_nan() }
    #[inline(always)]
    pub fn is_finite_mask(self) -> BVec4 {
        BVec4::new(self.x.is_finite(), self.y.is_finite(), self.z.is_finite(), self.w.is_finite())
    }
    #[inline(always)]
    pub fn is_nan_mask(self) -> BVec4 {
        BVec4::new(self.x.is_nan(), self.y.is_nan(), self.z.is_nan(), self.w.is_nan())
    }

    // ── Component-wise math ───────────────────────────────────────────────────

    #[inline(always)] pub fn abs(self) -> Self { Self::new(self.x.abs(), self.y.abs(), self.z.abs(), self.w.abs()) }
    #[inline(always)] pub fn min(self, r: Self) -> Self { Self::new(self.x.min(r.x), self.y.min(r.y), self.z.min(r.z), self.w.min(r.w)) }
    #[inline(always)] pub fn max(self, r: Self) -> Self { Self::new(self.x.max(r.x), self.y.max(r.y), self.z.max(r.z), self.w.max(r.w)) }
    #[inline(always)] pub fn clamp(self, lo: Self, hi: Self) -> Self { self.max(lo).min(hi) }
    #[inline(always)] pub fn floor(self) -> Self { Self::new(self.x.floor(), self.y.floor(), self.z.floor(), self.w.floor()) }
    #[inline(always)] pub fn ceil(self)  -> Self { Self::new(self.x.ceil(),  self.y.ceil(),  self.z.ceil(),  self.w.ceil())  }
    #[inline(always)] pub fn round(self) -> Self { Self::new(self.x.round(), self.y.round(), self.z.round(), self.w.round()) }
    #[inline(always)] pub fn trunc(self) -> Self { Self::new(self.x.trunc(), self.y.trunc(), self.z.trunc(), self.w.trunc()) }
    #[inline(always)] pub fn fract(self) -> Self { Self::new(self.x.fract(), self.y.fract(), self.z.fract(), self.w.fract()) }
    /// GLSL `fract` — always non-negative.
    #[inline(always)] pub fn fract_gl(self) -> Self { self - self.floor() }
    /// Heaviside step: 0.0 if `self < rhs`, else 1.0.
    #[inline(always)]
    pub fn step(self, rhs: Self) -> Self {
        Self::new(
            if self.x < rhs.x { 0.0 } else { 1.0 },
            if self.y < rhs.y { 0.0 } else { 1.0 },
            if self.z < rhs.z { 0.0 } else { 1.0 },
            if self.w < rhs.w { 0.0 } else { 1.0 },
        )
    }
    /// Clamp each component to `[0.0, 1.0]`.
    #[inline(always)] pub fn saturate(self) -> Self { self.clamp(Self::ZERO, Self::ONE) }
    #[inline(always)] pub fn recip(self)  -> Self { Self::new(self.x.recip(), self.y.recip(), self.z.recip(), self.w.recip()) }
    #[inline(always)] pub fn sqrt(self)   -> Self { Self::new(self.x.sqrt(),  self.y.sqrt(),  self.z.sqrt(),  self.w.sqrt())  }
    #[inline(always)] pub fn exp(self)    -> Self { Self::new(self.x.exp(),   self.y.exp(),   self.z.exp(),   self.w.exp())   }
    #[inline(always)] pub fn exp2(self)   -> Self { Self::new(self.x.exp2(),  self.y.exp2(),  self.z.exp2(),  self.w.exp2())  }
    #[inline(always)] pub fn ln(self)     -> Self { Self::new(self.x.ln(),    self.y.ln(),    self.z.ln(),    self.w.ln())    }
    #[inline(always)] pub fn log2(self)   -> Self { Self::new(self.x.log2(),  self.y.log2(),  self.z.log2(),  self.w.log2())  }
    #[inline(always)]
    pub fn powf(self, n: f32) -> Self { Self::new(self.x.powf(n), self.y.powf(n), self.z.powf(n), self.w.powf(n)) }
    #[inline(always)] pub fn sin(self)    -> Self { Self::new(self.x.sin(),   self.y.sin(),   self.z.sin(),   self.w.sin())   }
    #[inline(always)] pub fn cos(self)    -> Self { Self::new(self.x.cos(),   self.y.cos(),   self.z.cos(),   self.w.cos())   }
    #[inline(always)]
    pub fn sin_cos(self) -> (Self, Self) {
        let (sx, cx) = self.x.sin_cos(); let (sy, cy) = self.y.sin_cos();
        let (sz, cz) = self.z.sin_cos(); let (sw, cw) = self.w.sin_cos();
        (Self::new(sx, sy, sz, sw), Self::new(cx, cy, cz, cw))
    }
    #[inline(always)]
    pub fn div_euclid(self, rhs: Self) -> Self {
        Self::new(self.x.div_euclid(rhs.x), self.y.div_euclid(rhs.y),
                  self.z.div_euclid(rhs.z), self.w.div_euclid(rhs.w))
    }
    #[inline(always)]
    pub fn rem_euclid(self, rhs: Self) -> Self {
        Self::new(self.x.rem_euclid(rhs.x), self.y.rem_euclid(rhs.y),
                  self.z.rem_euclid(rhs.z), self.w.rem_euclid(rhs.w))
    }

    // ── FMA ───────────────────────────────────────────────────────────────────

    /// Fused multiply-add: `self * a + b`.
    #[inline(always)]
    pub fn mul_add(self, a: Self, b: Self) -> Self {
        Self::new(
            self.x.mul_add(a.x, b.x), self.y.mul_add(a.y, b.y),
            self.z.mul_add(a.z, b.z), self.w.mul_add(a.w, b.w),
        )
    }

    // ── Geometry ──────────────────────────────────────────────────────────────

    #[inline(always)] pub fn lerp(self, r: Self, t: f32) -> Self { self + (r - self) * t }
    #[inline(always)] pub fn midpoint(self, rhs: Self) -> Self { (self + rhs) * 0.5 }

    /// Move toward `target` by at most `max_dist`. Never overshoots.
    #[inline(always)]
    pub fn move_towards(self, target: Self, max_dist: f32) -> Self {
        let d = target - self;
        let len = d.length();
        if len <= max_dist || len < EPSILON { target } else { self + d / len * max_dist }
    }

    /// Reflect `self` off a surface defined by unit `normal`.
    #[inline(always)]
    pub fn reflect(self, normal: Self) -> Self { self - 2.0 * self.dot(normal) * normal }

    /// Refract `self` through a surface with unit `normal` and ratio `eta` (n_i/n_t).
    /// Returns `ZERO` on total internal reflection.
    #[inline]
    pub fn refract(self, normal: Self, eta: f32) -> Self {
        let n_dot_i = normal.dot(self);
        let k = 1.0 - eta * eta * (1.0 - n_dot_i * n_dot_i);
        if k < 0.0 { Self::ZERO } else { self * eta - normal * (eta * n_dot_i + k.sqrt()) }
    }

    /// Clamp vector length to `[min, max]`.
    #[inline(always)]
    pub fn clamp_length(self, min: f32, max: f32) -> Self {
        let len = self.length();
        if len < EPSILON { return Self::ZERO; }
        let c = len.clamp(min, max);
        if (c - len).abs() < EPSILON { self } else { self * (c / len) }
    }
    #[inline(always)]
    pub fn clamp_length_max(self, max: f32) -> Self {
        let len = self.length();
        if len > max && len > EPSILON { self * (max / len) } else { self }
    }
    #[inline(always)]
    pub fn clamp_length_min(self, min: f32) -> Self {
        let len = self.length();
        if len < min && len > EPSILON { self * (min / len) } else { self }
    }

    /// Project `self` onto `rhs`. Returns `ZERO` if `rhs` is zero-length.
    #[inline(always)]
    pub fn project_onto(self, rhs: Self) -> Self {
        let d = rhs.length_sq();
        if d < EPSILON { Self::ZERO } else { rhs * (self.dot(rhs) / d) }
    }
    #[inline(always)] pub fn reject_from(self, rhs: Self) -> Self { self - self.project_onto(rhs) }
    /// Project assuming `rhs` is already unit length (no division).
    #[inline(always)] pub fn project_onto_normalized(self, rhs: Self) -> Self { rhs * self.dot(rhs) }
    #[inline(always)] pub fn reject_from_normalized(self, rhs: Self) -> Self { self - self.project_onto_normalized(rhs) }

    // ── Approx equality ───────────────────────────────────────────────────────

    #[inline(always)]
    pub fn approx_eq(self, r: Self) -> bool {
        (self.x-r.x).abs() < EPSILON && (self.y-r.y).abs() < EPSILON
            && (self.z-r.z).abs() < EPSILON && (self.w-r.w).abs() < EPSILON
    }
    #[inline(always)]
    pub fn approx_eq_eps(self, rhs: Self, eps: f32) -> bool {
        (self.x-rhs.x).abs() < eps && (self.y-rhs.y).abs() < eps
            && (self.z-rhs.z).abs() < eps && (self.w-rhs.w).abs() < eps
    }
    /// Approximate equality with explicit `max_abs_diff` (glam-compat name).
    #[inline(always)]
    pub fn abs_diff_eq(self, rhs: Self, max_abs_diff: f32) -> bool {
        self.approx_eq_eps(rhs, max_abs_diff)
    }
}

// ── PartialEq / Default / Display ─────────────────────────────────────────────

impl PartialEq for Vec4 {
    fn eq(&self, r: &Self) -> bool { self.x==r.x && self.y==r.y && self.z==r.z && self.w==r.w }
}
impl Default for Vec4 { fn default() -> Self { Self::ZERO } }
impl fmt::Display for Vec4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {}, {}, {})", self.x, self.y, self.z, self.w)
    }
}

// ── Arithmetic operators ──────────────────────────────────────────────────────

impl Add  for Vec4 { type Output=Self; #[inline(always)] fn add(self,r:Self)->Self{Self::new(self.x+r.x,self.y+r.y,self.z+r.z,self.w+r.w)} }
impl Sub  for Vec4 { type Output=Self; #[inline(always)] fn sub(self,r:Self)->Self{Self::new(self.x-r.x,self.y-r.y,self.z-r.z,self.w-r.w)} }
impl Neg  for Vec4 { type Output=Self; #[inline(always)] fn neg(self)->Self{Self::new(-self.x,-self.y,-self.z,-self.w)} }
// scalar
impl Mul<f32> for Vec4 { type Output=Self; #[inline(always)] fn mul(self,s:f32)->Self{Self::new(self.x*s,self.y*s,self.z*s,self.w*s)} }
impl Mul<Vec4> for f32 { type Output=Vec4; #[inline(always)] fn mul(self,v:Vec4)->Vec4{Vec4::new(self*v.x,self*v.y,self*v.z,self*v.w)} }
// component-wise
impl Mul for Vec4 { type Output=Self; #[inline(always)] fn mul(self,r:Self)->Self{Self::new(self.x*r.x,self.y*r.y,self.z*r.z,self.w*r.w)} }
impl Div<f32> for Vec4 { type Output=Self; #[inline(always)] fn div(self,s:f32)->Self{Self::new(self.x/s,self.y/s,self.z/s,self.w/s)} }
impl Div for Vec4 { type Output=Self; #[inline(always)] fn div(self,r:Self)->Self{Self::new(self.x/r.x,self.y/r.y,self.z/r.z,self.w/r.w)} }

impl AddAssign for Vec4 { #[inline(always)] fn add_assign(&mut self,r:Self){self.x+=r.x;self.y+=r.y;self.z+=r.z;self.w+=r.w;} }
impl SubAssign for Vec4 { #[inline(always)] fn sub_assign(&mut self,r:Self){self.x-=r.x;self.y-=r.y;self.z-=r.z;self.w-=r.w;} }
impl MulAssign<f32> for Vec4 { #[inline(always)] fn mul_assign(&mut self,s:f32){self.x*=s;self.y*=s;self.z*=s;self.w*=s;} }
impl MulAssign for Vec4 { #[inline(always)] fn mul_assign(&mut self,r:Self){self.x*=r.x;self.y*=r.y;self.z*=r.z;self.w*=r.w;} }
impl DivAssign<f32> for Vec4 { #[inline(always)] fn div_assign(&mut self,s:f32){self.x/=s;self.y/=s;self.z/=s;self.w/=s;} }
impl DivAssign for Vec4 { #[inline(always)] fn div_assign(&mut self,r:Self){self.x/=r.x;self.y/=r.y;self.z/=r.z;self.w/=r.w;} }

// ── Conversions ───────────────────────────────────────────────────────────────

impl From<[f32;4]> for Vec4      { fn from(a:[f32;4])->Self{Self::new(a[0],a[1],a[2],a[3])} }
impl From<Vec4> for [f32;4]      { fn from(v:Vec4)->[f32;4]{[v.x,v.y,v.z,v.w]} }
impl From<(f32,f32,f32,f32)> for Vec4 { fn from(t:(f32,f32,f32,f32))->Self{Self::new(t.0,t.1,t.2,t.3)} }
impl From<Vec4> for (f32,f32,f32,f32) { fn from(v:Vec4)->(f32,f32,f32,f32){(v.x,v.y,v.z,v.w)} }
