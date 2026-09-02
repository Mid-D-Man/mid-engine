// crates/mid-math/src/f32/vec2.rs
//! Vec2 — always scalar, 8 bytes, no SIMD benefit on any target.

use core::fmt;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use crate::{BVec2, EPSILON};

/// 2D vector. 8 bytes, no padding. Always scalar on all platforms.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

impl Vec2 {
    pub const ZERO:         Self = Self { x: 0.0, y: 0.0 };
    pub const ONE:          Self = Self { x: 1.0, y: 1.0 };
    pub const NEG_ONE:      Self = Self { x: -1.0, y: -1.0 };
    pub const X:            Self = Self { x: 1.0, y: 0.0 };
    pub const Y:            Self = Self { x: 0.0, y: 1.0 };
    pub const NEG_X:        Self = Self { x: -1.0, y: 0.0 };
    pub const NEG_Y:        Self = Self { x: 0.0, y: -1.0 };
    pub const MIN:          Self = Self { x: f32::MIN, y: f32::MIN };
    pub const MAX:          Self = Self { x: f32::MAX, y: f32::MAX };
    pub const NAN:          Self = Self { x: f32::NAN, y: f32::NAN };
    pub const INFINITY:     Self = Self { x: f32::INFINITY,     y: f32::INFINITY };
    pub const NEG_INFINITY: Self = Self { x: f32::NEG_INFINITY, y: f32::NEG_INFINITY };

    // ── Constructors ──────────────────────────────────────────────────────────

    #[inline(always)] pub fn new(x: f32, y: f32) -> Self { Self { x, y } }
    #[inline(always)] pub fn splat(v: f32) -> Self { Self { x: v, y: v } }
    #[inline(always)] pub fn from_array(a: [f32; 2]) -> Self { Self::new(a[0], a[1]) }
    #[inline(always)] pub fn to_array(self) -> [f32; 2] { [self.x, self.y] }

    /// Apply a closure to each component independently.
    #[inline]
    pub fn map<F: Fn(f32) -> f32>(self, f: F) -> Self { Self::new(f(self.x), f(self.y)) }

    /// Conditional select component-wise: `if mask { if_true } else { if_false }`.
    #[inline]
    pub fn select(mask: BVec2, if_true: Self, if_false: Self) -> Self {
        Self::new(
            if mask.x { if_true.x } else { if_false.x },
            if mask.y { if_true.y } else { if_false.y },
        )
    }

    /// Write components to `slice[0..2]`. Panics if `slice.len() < 2`.
    #[inline]
    pub fn write_to_slice(self, slice: &mut [f32]) {
        slice[0] = self.x;
        slice[1] = self.y;
    }

    /// Return `self` with `x` replaced.
    #[inline(always)] pub fn with_x(mut self, x: f32) -> Self { self.x = x; self }
    /// Return `self` with `y` replaced.
    #[inline(always)] pub fn with_y(mut self, y: f32) -> Self { self.y = y; self }

    /// Extend to Vec3 by appending `z`.
    #[inline(always)]
    pub fn extend(self, z: f32) -> crate::Vec3 { crate::Vec3::new(self.x, self.y, z) }

    // ── Dot ───────────────────────────────────────────────────────────────────

    #[inline(always)] pub fn dot(self, rhs: Self) -> f32 { self.x * rhs.x + self.y * rhs.y }
    /// Dot product broadcast into a `Vec2`: `Self::splat(self.dot(rhs))`.
    #[inline(always)] pub fn dot_into_vec(self, rhs: Self) -> Self { Self::splat(self.dot(rhs)) }

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
    pub fn normalize_or(self, fallback: Self) -> Self {
        self.try_normalize().unwrap_or(fallback)
    }
    #[inline(always)] pub fn normalize_or_zero(self) -> Self { self.normalize_or(Self::ZERO) }
    #[inline(always)] pub fn is_normalized(self) -> bool { (self.length_sq() - 1.0).abs() <= 2e-4 }
    /// Normalize and return `(normalized_self, original_length)` in one pass.
    #[inline]
    pub fn normalize_and_length(self) -> (Self, f32) {
        let l = self.length();
        if l < EPSILON { (Self::ZERO, 0.0) } else { (self / l, l) }
    }

    // ── Distance ──────────────────────────────────────────────────────────────

    #[inline(always)] pub fn distance(self, rhs: Self) -> f32 { (self - rhs).length() }
    #[inline(always)] pub fn distance_sq(self, rhs: Self) -> f32 { (self - rhs).length_sq() }
    /// Alias for `distance_sq` (glam-compat name).
    #[inline(always)] pub fn distance_squared(self, rhs: Self) -> f32 { self.distance_sq(rhs) }

    // ── Reduction ─────────────────────────────────────────────────────────────

    #[inline(always)] pub fn min_element(self) -> f32 { self.x.min(self.y) }
    #[inline(always)] pub fn max_element(self) -> f32 { self.x.max(self.y) }
    #[inline(always)] pub fn element_sum(self) -> f32 { self.x + self.y }
    #[inline(always)] pub fn element_product(self) -> f32 { self.x * self.y }
    /// Index (0 or 1) of the minimum component.
    #[inline] pub fn min_position(self) -> usize { if self.x <= self.y { 0 } else { 1 } }
    /// Index (0 or 1) of the maximum component.
    #[inline] pub fn max_position(self) -> usize { if self.x >= self.y { 0 } else { 1 } }

    // ── Component-wise comparisons → BVec2 ────────────────────────────────────

    #[inline(always)] pub fn cmpeq(self, rhs: Self) -> BVec2 { BVec2::new(self.x == rhs.x, self.y == rhs.y) }
    #[inline(always)] pub fn cmpne(self, rhs: Self) -> BVec2 { BVec2::new(self.x != rhs.x, self.y != rhs.y) }
    #[inline(always)] pub fn cmpge(self, rhs: Self) -> BVec2 { BVec2::new(self.x >= rhs.x, self.y >= rhs.y) }
    #[inline(always)] pub fn cmpgt(self, rhs: Self) -> BVec2 { BVec2::new(self.x >  rhs.x, self.y >  rhs.y) }
    #[inline(always)] pub fn cmple(self, rhs: Self) -> BVec2 { BVec2::new(self.x <= rhs.x, self.y <= rhs.y) }
    #[inline(always)] pub fn cmplt(self, rhs: Self) -> BVec2 { BVec2::new(self.x <  rhs.x, self.y <  rhs.y) }

    // ── Sign ──────────────────────────────────────────────────────────────────

    #[inline(always)] pub fn signum(self) -> Self { Self::new(self.x.signum(), self.y.signum()) }
    #[inline(always)] pub fn copysign(self, rhs: Self) -> Self { Self::new(self.x.copysign(rhs.x), self.y.copysign(rhs.y)) }
    /// Bitmask of sign bits: bit 0 = x, bit 1 = y. Set when component is negative.
    #[inline]
    pub fn is_negative_bitmask(self) -> u32 {
        (self.x.is_sign_negative() as u32) | ((self.y.is_sign_negative() as u32) << 1)
    }

    // ── Predicates ────────────────────────────────────────────────────────────

    #[inline(always)] pub fn is_finite(self) -> bool { self.x.is_finite() && self.y.is_finite() }
    #[inline(always)] pub fn is_nan(self) -> bool { self.x.is_nan() || self.y.is_nan() }
    #[inline(always)] pub fn is_finite_mask(self) -> BVec2 { BVec2::new(self.x.is_finite(), self.y.is_finite()) }
    #[inline(always)] pub fn is_nan_mask(self) -> BVec2 { BVec2::new(self.x.is_nan(), self.y.is_nan()) }

    // ── Component-wise math ───────────────────────────────────────────────────

    #[inline(always)] pub fn abs(self) -> Self { Self::new(self.x.abs(),   self.y.abs())   }
    #[inline(always)] pub fn min(self, rhs: Self) -> Self { Self::new(self.x.min(rhs.x), self.y.min(rhs.y)) }
    #[inline(always)] pub fn max(self, rhs: Self) -> Self { Self::new(self.x.max(rhs.x), self.y.max(rhs.y)) }
    #[inline(always)] pub fn clamp(self, lo: Self, hi: Self) -> Self { self.max(lo).min(hi) }
    #[inline(always)] pub fn floor(self) -> Self { Self::new(self.x.floor(), self.y.floor()) }
    #[inline(always)] pub fn ceil(self)  -> Self { Self::new(self.x.ceil(),  self.y.ceil())  }
    #[inline(always)] pub fn round(self) -> Self { Self::new(self.x.round(), self.y.round()) }
    #[inline(always)] pub fn trunc(self) -> Self { Self::new(self.x.trunc(), self.y.trunc()) }
    /// Fractional part via truncation (negative inputs give negative result).
    #[inline(always)] pub fn fract(self) -> Self { Self::new(self.x.fract(), self.y.fract()) }
    /// GLSL `fract` — always non-negative (uses `floor`, not truncation).
    #[inline(always)] pub fn fract_gl(self) -> Self { self - self.floor() }
    /// Heaviside step: 0.0 if `self < rhs`, else 1.0.
    #[inline(always)]
    pub fn step(self, rhs: Self) -> Self {
        Self::new(if self.x < rhs.x { 0.0 } else { 1.0 },
                  if self.y < rhs.y { 0.0 } else { 1.0 })
    }
    /// Clamp each component to `[0.0, 1.0]`.
    #[inline(always)] pub fn saturate(self) -> Self { self.clamp(Self::ZERO, Self::ONE) }
    /// Component-wise reciprocal.
    #[inline(always)] pub fn recip(self)  -> Self { Self::new(self.x.recip(), self.y.recip()) }
    #[inline(always)] pub fn sqrt(self)   -> Self { Self::new(self.x.sqrt(),  self.y.sqrt())  }
    #[inline(always)] pub fn exp(self)    -> Self { Self::new(self.x.exp(),   self.y.exp())   }
    #[inline(always)] pub fn exp2(self)   -> Self { Self::new(self.x.exp2(),  self.y.exp2())  }
    #[inline(always)] pub fn ln(self)     -> Self { Self::new(self.x.ln(),    self.y.ln())    }
    #[inline(always)] pub fn log2(self)   -> Self { Self::new(self.x.log2(),  self.y.log2())  }
    #[inline(always)] pub fn powf(self, n: f32) -> Self { Self::new(self.x.powf(n), self.y.powf(n)) }
    #[inline(always)] pub fn sin(self)    -> Self { Self::new(self.x.sin(),   self.y.sin())   }
    #[inline(always)] pub fn cos(self)    -> Self { Self::new(self.x.cos(),   self.y.cos())   }
    #[inline(always)]
    pub fn sin_cos(self) -> (Self, Self) {
        let (sx, cx) = self.x.sin_cos();
        let (sy, cy) = self.y.sin_cos();
        (Self::new(sx, sy), Self::new(cx, cy))
    }
    #[inline(always)] pub fn div_euclid(self, rhs: Self) -> Self { Self::new(self.x.div_euclid(rhs.x), self.y.div_euclid(rhs.y)) }
    #[inline(always)] pub fn rem_euclid(self, rhs: Self) -> Self { Self::new(self.x.rem_euclid(rhs.x), self.y.rem_euclid(rhs.y)) }

    // ── FMA ───────────────────────────────────────────────────────────────────

    /// Fused multiply-add: `self * a + b` (one rounding step when hardware-supported).
    #[inline(always)]
    pub fn mul_add(self, a: Self, b: Self) -> Self {
        Self::new(self.x.mul_add(a.x, b.x), self.y.mul_add(a.y, b.y))
    }

    // ── Geometry ──────────────────────────────────────────────────────────────

    #[inline(always)] pub fn lerp(self, rhs: Self, t: f32) -> Self { self + (rhs - self) * t }
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
    /// Component of `self` perpendicular to `rhs`.
    #[inline(always)]
    pub fn reject_from(self, rhs: Self) -> Self { self - self.project_onto(rhs) }
    /// Like `project_onto` but `rhs` is assumed to already be unit length (no division).
    #[inline(always)]
    pub fn project_onto_normalized(self, rhs: Self) -> Self { rhs * self.dot(rhs) }
    /// Like `reject_from` but `rhs` is assumed to already be unit length.
    #[inline(always)]
    pub fn reject_from_normalized(self, rhs: Self) -> Self { self - self.project_onto_normalized(rhs) }

    // ── 2D-specific ───────────────────────────────────────────────────────────

    /// Perpendicular vector (rotated 90° CCW): `(-y, x)`. Alias: `perp`.
    #[inline(always)] pub fn perpendicular(self) -> Self { Self::new(-self.y, self.x) }
    /// Alias for `perpendicular` (glam-compat).
    #[inline(always)] pub fn perp(self) -> Self { self.perpendicular() }

    /// Signed 2D "cross product" (perp-dot product): `x*rhs.y - y*rhs.x`. Alias: `perp_dot`.
    #[inline(always)] pub fn cross(self, rhs: Self) -> f32 { self.x * rhs.y - self.y * rhs.x }
    /// Alias for `cross` (glam-compat).
    #[inline(always)] pub fn perp_dot(self, rhs: Self) -> f32 { self.cross(rhs) }

    /// Signed angle from `self` to `rhs` in radians, range `[-π, π]`.
    #[inline(always)]
    pub fn angle_to(self, rhs: Self) -> f32 {
        let c = self.x * rhs.y - self.y * rhs.x;
        c.atan2(self.dot(rhs))
    }
    /// Angle of this vector from the +X axis: `y.atan2(x)`, range `[-π, π]`.
    #[inline(always)] pub fn to_angle(self) -> f32 { self.y.atan2(self.x) }
    /// Unit vector at `angle` radians from +X axis: `(cos, sin)`.
    #[inline(always)]
    pub fn from_angle(angle: f32) -> Self { let (s, c) = angle.sin_cos(); Self::new(c, s) }

    // ── Polar coordinates ─────────────────────────────────────────────────────

    /// Convert to `(radius, angle)`. `angle` ∈ `[-π, π]`.
    #[inline(always)] pub fn to_polar(self) -> (f32, f32) { (self.length(), self.y.atan2(self.x)) }
    /// Build from polar `(radius, angle)`.
    #[inline(always)]
    pub fn from_polar(radius: f32, angle: f32) -> Self {
        let (s, c) = angle.sin_cos();
        Self::new(radius * c, radius * s)
    }

    // ── Approx equality ───────────────────────────────────────────────────────

    #[inline(always)]
    pub fn approx_eq(self, rhs: Self) -> bool {
        (self.x - rhs.x).abs() < EPSILON && (self.y - rhs.y).abs() < EPSILON
    }
    #[inline(always)]
    pub fn approx_eq_eps(self, rhs: Self, eps: f32) -> bool {
        (self.x - rhs.x).abs() < eps && (self.y - rhs.y).abs() < eps
    }
    /// Approximate equality with explicit `max_abs_diff` (glam-compat name for `approx_eq_eps`).
    #[inline(always)]
    pub fn abs_diff_eq(self, rhs: Self, max_abs_diff: f32) -> bool {
        self.approx_eq_eps(rhs, max_abs_diff)
    }
}

// ── PartialEq / Default / Display ─────────────────────────────────────────────

impl PartialEq for Vec2 {
    fn eq(&self, r: &Self) -> bool { self.x == r.x && self.y == r.y }
}
impl Default for Vec2 { fn default() -> Self { Self::ZERO } }
impl fmt::Display for Vec2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.x, self.y)
    }
}

// ── Arithmetic operators ──────────────────────────────────────────────────────

impl Add  for Vec2 { type Output=Self; #[inline(always)] fn add(self,r:Self)->Self{Self::new(self.x+r.x,self.y+r.y)} }
impl Sub  for Vec2 { type Output=Self; #[inline(always)] fn sub(self,r:Self)->Self{Self::new(self.x-r.x,self.y-r.y)} }
impl Neg  for Vec2 { type Output=Self; #[inline(always)] fn neg(self)->Self{Self::new(-self.x,-self.y)} }
// scalar × Vec2
impl Mul<f32> for Vec2 { type Output=Self; #[inline(always)] fn mul(self,s:f32)->Self{Self::new(self.x*s,self.y*s)} }
impl Mul<Vec2> for f32 { type Output=Vec2; #[inline(always)] fn mul(self,v:Vec2)->Vec2{Vec2::new(self*v.x,self*v.y)} }
// component-wise Vec2 × Vec2
impl Mul for Vec2 { type Output=Self; #[inline(always)] fn mul(self,r:Self)->Self{Self::new(self.x*r.x,self.y*r.y)} }
impl Div<f32> for Vec2 { type Output=Self; #[inline(always)] fn div(self,s:f32)->Self{Self::new(self.x/s,self.y/s)} }
impl Div for Vec2 { type Output=Self; #[inline(always)] fn div(self,r:Self)->Self{Self::new(self.x/r.x,self.y/r.y)} }

impl AddAssign for Vec2 { #[inline(always)] fn add_assign(&mut self,r:Self){self.x+=r.x;self.y+=r.y;} }
impl SubAssign for Vec2 { #[inline(always)] fn sub_assign(&mut self,r:Self){self.x-=r.x;self.y-=r.y;} }
impl MulAssign<f32> for Vec2 { #[inline(always)] fn mul_assign(&mut self,s:f32){self.x*=s;self.y*=s;} }
impl MulAssign for Vec2 { #[inline(always)] fn mul_assign(&mut self,r:Self){self.x*=r.x;self.y*=r.y;} }
impl DivAssign<f32> for Vec2 { #[inline(always)] fn div_assign(&mut self,s:f32){self.x/=s;self.y/=s;} }
impl DivAssign for Vec2 { #[inline(always)] fn div_assign(&mut self,r:Self){self.x/=r.x;self.y/=r.y;} }

// ── Conversions ───────────────────────────────────────────────────────────────

impl From<[f32; 2]>  for Vec2 { #[inline] fn from(a: [f32; 2]) -> Self  { Self::new(a[0], a[1]) } }
impl From<Vec2> for [f32; 2]  { #[inline] fn from(v: Vec2)     -> Self  { [v.x, v.y] } }
impl From<(f32, f32)> for Vec2 { #[inline] fn from(t: (f32,f32)) -> Self { Self::new(t.0, t.1) } }
impl From<Vec2> for (f32, f32) { #[inline] fn from(v: Vec2) -> Self      { (v.x, v.y) } }

