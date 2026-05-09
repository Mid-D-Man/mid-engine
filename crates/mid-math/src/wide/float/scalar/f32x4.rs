// crates/mid-math/src/wide/float/scalar/f32x4.rs
//! Scalar fallback 4-lane f32 — non-x86 platforms.

#![allow(non_camel_case_types)]

use core::fmt;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use super::mask4::Mask4;

/// 4-lane independent f32 scalar — scalar fallback.
#[derive(Clone, Copy, PartialEq)]
#[repr(C, align(16))]
pub struct f32x4(pub(crate) [f32; 4]);

impl f32x4 {
    pub const ZERO:         Self = f32x4([0.0; 4]);
    pub const ONE:          Self = f32x4([1.0; 4]);
    pub const NEG_ONE:      Self = f32x4([-1.0; 4]);
    pub const INFINITY:     Self = f32x4([f32::INFINITY; 4]);
    pub const NEG_INFINITY: Self = f32x4([f32::NEG_INFINITY; 4]);

    #[inline(always)] pub fn splat(v: f32) -> Self { f32x4([v; 4]) }
    #[inline(always)] pub fn new(a: f32, b: f32, c: f32, d: f32) -> Self { f32x4([a, b, c, d]) }
    #[inline(always)] pub fn from_array(a: [f32; 4]) -> Self { f32x4(a) }
    #[inline(always)] pub fn to_array(self) -> [f32; 4] { self.0 }
    #[inline] pub fn get(self, i: usize) -> f32 { assert!(i < 4); self.0[i] }

    #[inline] pub fn sqrt(self) -> Self { f32x4(self.0.map(|x| x.sqrt())) }
    #[inline] pub fn recip_sqrt(self) -> Self { f32x4(self.0.map(|x| 1.0 / x.sqrt())) }
    #[inline] pub fn recip(self) -> Self { f32x4(self.0.map(|x| 1.0 / x)) }
    #[inline] pub fn abs(self) -> Self { f32x4(self.0.map(|x| x.abs())) }
    #[inline] pub fn min(self, r: Self) -> Self { f32x4(core::array::from_fn(|i| self.0[i].min(r.0[i]))) }
    #[inline] pub fn max(self, r: Self) -> Self { f32x4(core::array::from_fn(|i| self.0[i].max(r.0[i]))) }
    #[inline] pub fn clamp(self, lo: Self, hi: Self) -> Self { self.max(lo).min(hi) }
    #[inline] pub fn min_element(self) -> f32 { self.0.iter().copied().reduce(f32::min).unwrap() }
    #[inline] pub fn max_element(self) -> f32 { self.0.iter().copied().reduce(f32::max).unwrap() }
    #[inline] pub fn mul_add(self, b: Self, c: Self) -> Self {
        f32x4(core::array::from_fn(|i| self.0[i] * b.0[i] + c.0[i]))
    }
    #[inline] pub fn blend(mask: Mask4, t: Self, f: Self) -> Self {
        f32x4(core::array::from_fn(|i| if mask.0[i] != 0 { t.0[i] } else { f.0[i] }))
    }

    #[inline] pub fn cmpeq(self, r: Self) -> Mask4 { Mask4(core::array::from_fn(|i| if self.0[i]==r.0[i] { u32::MAX } else { 0 })) }
    #[inline] pub fn cmpne(self, r: Self) -> Mask4 { !self.cmpeq(r) }
    #[inline] pub fn cmplt(self, r: Self) -> Mask4 { Mask4(core::array::from_fn(|i| if self.0[i]<r.0[i] { u32::MAX } else { 0 })) }
    #[inline] pub fn cmple(self, r: Self) -> Mask4 { Mask4(core::array::from_fn(|i| if self.0[i]<=r.0[i] { u32::MAX } else { 0 })) }
    #[inline] pub fn cmpgt(self, r: Self) -> Mask4 { Mask4(core::array::from_fn(|i| if self.0[i]>r.0[i] { u32::MAX } else { 0 })) }
    #[inline] pub fn cmpge(self, r: Self) -> Mask4 { Mask4(core::array::from_fn(|i| if self.0[i]>=r.0[i] { u32::MAX } else { 0 })) }

    #[inline] pub fn is_finite(self) -> bool { self.0.iter().all(|x| x.is_finite()) }
    #[inline] pub fn is_nan(self) -> bool { self.0.iter().any(|x| x.is_nan()) }
}

impl Add for f32x4 { type Output=Self; #[inline] fn add(self, r: Self) -> Self { f32x4(core::array::from_fn(|i| self.0[i]+r.0[i])) } }
impl AddAssign for f32x4 { fn add_assign(&mut self, r: Self) { *self = *self + r; } }
impl Sub for f32x4 { type Output=Self; #[inline] fn sub(self, r: Self) -> Self { f32x4(core::array::from_fn(|i| self.0[i]-r.0[i])) } }
impl SubAssign for f32x4 { fn sub_assign(&mut self, r: Self) { *self = *self - r; } }
impl Mul for f32x4 { type Output=Self; #[inline] fn mul(self, r: Self) -> Self { f32x4(core::array::from_fn(|i| self.0[i]*r.0[i])) } }
impl MulAssign for f32x4 { fn mul_assign(&mut self, r: Self) { *self = *self * r; } }
impl Div for f32x4 { type Output=Self; #[inline] fn div(self, r: Self) -> Self { f32x4(core::array::from_fn(|i| self.0[i]/r.0[i])) } }
impl DivAssign for f32x4 { fn div_assign(&mut self, r: Self) { *self = *self / r; } }
impl Neg for f32x4 { type Output=Self; #[inline] fn neg(self) -> Self { f32x4(self.0.map(|x| -x)) } }

impl fmt::Debug for f32x4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "f32x4({},{},{},{})", self.0[0],self.0[1],self.0[2],self.0[3])
    }
}
impl fmt::Display for f32x4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{},{},{},{}]", self.0[0],self.0[1],self.0[2],self.0[3])
    }
}
impl From<[f32;4]> for f32x4 { fn from(a: [f32;4]) -> Self { Self::from_array(a) } }
impl From<f32x4> for [f32;4] { fn from(v: f32x4) -> Self { v.to_array() } }
impl From<f32> for f32x4 { fn from(v: f32) -> Self { Self::splat(v) } }
