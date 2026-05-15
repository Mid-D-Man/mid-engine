// crates/mid-math/src/color/hsl.rs
//! HSL (Hue-Saturation-Lightness) — CSS-compatible artist color in sRGB space.
//!
//! H ∈ [0, 360), S ∈ [0, 1], L ∈ [0, 1].
//! HSL vs HSV: pure colors appear at L=0.5 (HSL) vs V=1.0 (HSV).
//! Use HSL for CSS-compatible pickers. Use HSV for brightness-focused tools.

use core::fmt;
use super::rgb::Rgb;
use super::hsv::Hsv;

/// HSL color in sRGB space.
#[derive(Clone, Copy, PartialEq, Default)]
#[repr(C)]
pub struct Hsl { pub h: f32, pub s: f32, pub l: f32 }

impl Hsl {
    pub const BLACK:   Self = Self { h:   0.0, s: 0.0, l: 0.0 };
    pub const WHITE:   Self = Self { h:   0.0, s: 0.0, l: 1.0 };
    pub const RED:     Self = Self { h:   0.0, s: 1.0, l: 0.5 };
    pub const GREEN:   Self = Self { h: 120.0, s: 1.0, l: 0.5 };
    pub const BLUE:    Self = Self { h: 240.0, s: 1.0, l: 0.5 };

    #[inline(always)] pub const fn new(h: f32, s: f32, l: f32) -> Self { Self { h, s, l } }

    pub fn from_srgb(r: f32, g: f32, b: f32) -> Self {
        let max   = r.max(g).max(b);
        let min   = r.min(g).min(b);
        let delta = max - min;
        let l = (max + min) * 0.5;
        let s = if delta < 1e-6 { 0.0 }
                else { delta / (1.0 - (2.0 * l - 1.0).abs()) };
        let h = if delta < 1e-6 { 0.0 }
                else if max == r { 60.0 * ((g - b) / delta).rem_euclid(6.0) }
                else if max == g { 60.0 * ((b - r) / delta + 2.0) }
                else             { 60.0 * ((r - g) / delta + 4.0) };
        Self { h, s, l }
    }

    pub fn to_srgb(self) -> (f32, f32, f32) {
        if self.s < 1e-6 { return (self.l, self.l, self.l); }
        let c = (1.0 - (2.0 * self.l - 1.0).abs()) * self.s;
        let h = self.h / 60.0;
        let x = c * (1.0 - (h.rem_euclid(2.0) - 1.0).abs());
        let m = self.l - c * 0.5;
        let (r1, g1, b1) = match h.floor() as i32 {
            0 => (c, x, 0.0), 1 => (x, c, 0.0), 2 => (0.0, c, x),
            3 => (0.0, x, c), 4 => (x, 0.0, c), _ => (c, 0.0, x),
        };
        (r1 + m, g1 + m, b1 + m)
    }

    #[inline] pub fn from_linear(c: Rgb) -> Self {
        let s = c.to_srgb(); Self::from_srgb(s.r, s.g, s.b)
    }
    #[inline] pub fn to_linear(self) -> Rgb {
        let (r, g, b) = self.to_srgb(); Rgb { r, g, b }.from_srgb()
    }

    #[inline] pub fn shift_hue(self, delta: f32) -> Self {
        Self { h: (self.h + delta).rem_euclid(360.0), ..self }
    }
    #[inline] pub fn scale_saturation(self, f: f32) -> Self {
        Self { s: (self.s * f).clamp(0.0, 1.0), ..self }
    }
    #[inline] pub fn desaturate(self, amount: f32) -> Self {
        self.scale_saturation(1.0 - amount.clamp(0.0, 1.0))
    }
    #[inline] pub fn lighten(self, amount: f32) -> Self {
        Self { l: (self.l + amount).clamp(0.0, 1.0), ..self }
    }
    #[inline] pub fn darken(self, amount: f32) -> Self { self.lighten(-amount) }
    #[inline] pub fn complement(self) -> Self { self.shift_hue(180.0) }

    pub fn lerp(self, rhs: Self, t: f32) -> Self {
        let dh = ((rhs.h - self.h + 180.0).rem_euclid(360.0)) - 180.0;
        Self {
            h: (self.h + dh * t).rem_euclid(360.0),
            s: self.s + (rhs.s - self.s) * t,
            l: self.l + (rhs.l - self.l) * t,
        }
    }

    /// Convert to HSV (same hue, recomputed S/V).
    pub fn to_hsv(self) -> Hsv {
        let (r, g, b) = self.to_srgb(); Hsv::from_srgb(r, g, b)
    }
}

impl fmt::Debug for Hsl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Hsl(h={:.1}°, s={:.3}, l={:.3})", self.h, self.s, self.l)
    }
}
impl fmt::Display for Hsl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "hsl({:.1},{:.3},{:.3})", self.h, self.s, self.l)
    }
}
impl From<Rgb> for Hsl { #[inline] fn from(c: Rgb) -> Self { Self::from_linear(c) } }
impl From<Hsl> for Rgb { #[inline] fn from(c: Hsl) -> Self { c.to_linear() } }
impl From<Hsv> for Hsl {
    fn from(c: Hsv) -> Self {
        let (r, g, b) = c.to_srgb(); Self::from_srgb(r, g, b)
    }
}
impl From<Hsl> for Hsv {
    fn from(c: Hsl) -> Self { c.to_hsv() }
  }
