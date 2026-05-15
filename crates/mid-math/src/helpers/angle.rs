// crates/mid-math/src/angle.rs
//! Type-safe angle wrappers — eliminate unit-mismatch bugs.
//!
//! `Radians` and `Degrees` are distinct types. The compiler rejects mixing them.
//! All trig functions (`sin`, `cos`, `tan`) are methods on `Radians` only —
//! you cannot accidentally pass degrees to a function expecting radians.
//!
//! Conversion: `Degrees::to_radians()` and `Radians::to_degrees()`.
//! From f32: `Radians::from(1.5_f32)` or `Degrees::from(90.0_f32)`.

use core::fmt;
use core::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

// ── Radians ───────────────────────────────────────────────────────────────────

/// An angle in radians. Wraps `f32`. Distinct from `Degrees` at compile time.
#[derive(Clone, Copy, PartialEq, PartialOrd, Default)]
#[repr(transparent)]
pub struct Radians(pub f32);

impl Radians {
    pub const ZERO:     Self = Radians(0.0);
    pub const PI:       Self = Radians(core::f32::consts::PI);
    pub const TAU:      Self = Radians(core::f32::consts::TAU);
    pub const FRAC_PI_2:Self = Radians(core::f32::consts::FRAC_PI_2);
    pub const FRAC_PI_4:Self = Radians(core::f32::consts::FRAC_PI_4);

    #[inline(always)] pub const fn new(v: f32) -> Self { Radians(v) }
    #[inline(always)] pub const fn value(self) -> f32 { self.0 }

    /// Wrap to `[-π, π)`.
    #[inline]
    pub fn wrap(self) -> Self {
        let v = self.0.rem_euclid(core::f32::consts::TAU);
        Radians(if v > core::f32::consts::PI { v - core::f32::consts::TAU } else { v })
    }

    /// Wrap to `[0, 2π)`.
    #[inline] pub fn wrap_positive(self) -> Self {
        Radians(self.0.rem_euclid(core::f32::consts::TAU))
    }

    /// Convert to degrees.
    #[inline] pub fn to_degrees(self) -> Degrees { Degrees(self.0.to_degrees()) }

    // ── Trig ─────────────────────────────────────────────────────────────────

    #[inline] pub fn sin(self)          -> f32 { self.0.sin() }
    #[inline] pub fn cos(self)          -> f32 { self.0.cos() }
    #[inline] pub fn tan(self)          -> f32 { self.0.tan() }
    #[inline] pub fn sin_cos(self)      -> (f32, f32) { self.0.sin_cos() }
    #[inline] pub fn asin(v: f32)       -> Self { Radians(v.asin()) }
    #[inline] pub fn acos(v: f32)       -> Self { Radians(v.acos()) }
    #[inline] pub fn atan2(y: f32, x: f32) -> Self { Radians(y.atan2(x)) }
    #[inline] pub fn lerp(self, rhs: Self, t: f32) -> Self {
        Radians(self.0 + (rhs.0 - self.0) * t)
    }
    #[inline] pub fn abs(self)  -> Self { Radians(self.0.abs()) }
    #[inline] pub fn is_finite(self) -> bool { self.0.is_finite() }
}

// ── Degrees ───────────────────────────────────────────────────────────────────

/// An angle in degrees. Wraps `f32`. Distinct from `Radians` at compile time.
#[derive(Clone, Copy, PartialEq, PartialOrd, Default)]
#[repr(transparent)]
pub struct Degrees(pub f32);

impl Degrees {
    pub const ZERO:      Self = Degrees(0.0);
    pub const FULL:      Self = Degrees(360.0);
    pub const HALF:      Self = Degrees(180.0);
    pub const QUARTER:   Self = Degrees(90.0);
    pub const EIGHTH:    Self = Degrees(45.0);

    #[inline(always)] pub const fn new(v: f32) -> Self { Degrees(v) }
    #[inline(always)] pub const fn value(self) -> f32 { self.0 }

    /// Wrap to `(-180, 180]`.
    #[inline] pub fn wrap(self) -> Self {
        let v = self.0.rem_euclid(360.0);
        Degrees(if v > 180.0 { v - 360.0 } else { v })
    }
    /// Wrap to `[0, 360)`.
    #[inline] pub fn wrap_positive(self) -> Self { Degrees(self.0.rem_euclid(360.0)) }

    /// Convert to radians.
    #[inline] pub fn to_radians(self) -> Radians { Radians(self.0.to_radians()) }

    #[inline] pub fn lerp(self, rhs: Self, t: f32) -> Self {
        Degrees(self.0 + (rhs.0 - self.0) * t)
    }
    #[inline] pub fn abs(self) -> Self { Degrees(self.0.abs()) }
}

// ── Operators — Radians ───────────────────────────────────────────────────────

impl Add  for Radians { type Output=Self; #[inline] fn add(self,r:Self)->Self{Radians(self.0+r.0)} }
impl Sub  for Radians { type Output=Self; #[inline] fn sub(self,r:Self)->Self{Radians(self.0-r.0)} }
impl Neg  for Radians { type Output=Self; #[inline] fn neg(self)->Self{Radians(-self.0)} }
impl Mul<f32> for Radians { type Output=Self; #[inline] fn mul(self,s:f32)->Self{Radians(self.0*s)} }
impl Div<f32> for Radians { type Output=Self; #[inline] fn div(self,s:f32)->Self{Radians(self.0/s)} }
impl AddAssign for Radians { #[inline] fn add_assign(&mut self,r:Self){self.0+=r.0;} }
impl SubAssign for Radians { #[inline] fn sub_assign(&mut self,r:Self){self.0-=r.0;} }
impl MulAssign<f32> for Radians { #[inline] fn mul_assign(&mut self,s:f32){self.0*=s;} }

// ── Operators — Degrees ───────────────────────────────────────────────────────

impl Add  for Degrees { type Output=Self; #[inline] fn add(self,r:Self)->Self{Degrees(self.0+r.0)} }
impl Sub  for Degrees { type Output=Self; #[inline] fn sub(self,r:Self)->Self{Degrees(self.0-r.0)} }
impl Neg  for Degrees { type Output=Self; #[inline] fn neg(self)->Self{Degrees(-self.0)} }
impl Mul<f32> for Degrees { type Output=Self; #[inline] fn mul(self,s:f32)->Self{Degrees(self.0*s)} }
impl Div<f32> for Degrees { type Output=Self; #[inline] fn div(self,s:f32)->Self{Degrees(self.0/s)} }
impl AddAssign for Degrees { #[inline] fn add_assign(&mut self,r:Self){self.0+=r.0;} }
impl SubAssign for Degrees { #[inline] fn sub_assign(&mut self,r:Self){self.0-=r.0;} }

// ── Conversions ───────────────────────────────────────────────────────────────

impl From<f32> for Radians { #[inline] fn from(v: f32) -> Self { Radians(v) } }
impl From<f32> for Degrees { #[inline] fn from(v: f32) -> Self { Degrees(v) } }
impl From<Radians> for f32 { #[inline] fn from(v: Radians) -> f32 { v.0 } }
impl From<Degrees> for f32 { #[inline] fn from(v: Degrees) -> f32 { v.0 } }
impl From<Degrees> for Radians { #[inline] fn from(d: Degrees) -> Self { d.to_radians() } }
impl From<Radians> for Degrees { #[inline] fn from(r: Radians) -> Self { r.to_degrees() } }

impl fmt::Debug for Radians {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "{}rad", self.0) }
}
impl fmt::Display for Radians {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "{}rad", self.0) }
}
impl fmt::Debug for Degrees {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "{}°", self.0) }
}
impl fmt::Display for Degrees {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "{}°", self.0) }
}
