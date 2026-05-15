// crates/mid-math/src/color/hsv.rs
//! HSV (Hue-Saturation-Value) — artist-friendly color in sRGB space.
//!
//! H ∈ [0, 360), S ∈ [0, 1], V ∈ [0, 1].
//! Operates in **sRGB** not linear — convert to Rgb before any lighting math.
//! Primary uses: color pickers, hue rotation, saturation FX, palette generation.

use core::fmt;
use super::rgb::Rgb;

/// HSV color in sRGB space.
#[derive(Clone, Copy, PartialEq, Default)]
#[repr(C)]
pub struct Hsv { pub h: f32, pub s: f32, pub v: f32 }

impl Hsv {
    pub const BLACK:   Self = Self { h:   0.0, s: 0.0, v: 0.0 };
    pub const WHITE:   Self = Self { h:   0.0, s: 0.0, v: 1.0 };
    pub const RED:     Self = Self { h:   0.0, s: 1.0, v: 1.0 };
    pub const GREEN:   Self = Self { h: 120.0, s: 1.0, v: 1.0 };
    pub const BLUE:    Self = Self { h: 240.0, s: 1.0, v: 1.0 };
    pub const YELLOW:  Self = Self { h:  60.0, s: 1.0, v: 1.0 };
    pub const CYAN:    Self = Self { h: 180.0, s: 1.0, v: 1.0 };
    pub const MAGENTA: Self = Self { h: 300.0, s: 1.0, v: 1.0 };

    #[inline(always)]
    pub const fn new(h: f32, s: f32, v: f32) -> Self { Self { h, s, v } }

    // ── Conversion: sRGB f32 ──────────────────────────────────────────────────

    /// Convert from sRGB float triple `(r, g, b) ∈ [0, 1]`.
    pub fn from_srgb(r: f32, g: f32, b: f32) -> Self {
        let max   = r.max(g).max(b);
        let min   = r.min(g).min(b);
        let delta = max - min;
        let v = max;
        let s = if max > 1e-6 { delta / max } else { 0.0 };
        let h = if delta < 1e-6 {
            0.0
        } else if max == r {
            60.0 * ((g - b) / delta).rem_euclid(6.0)
        } else if max == g {
            60.0 * ((b - r) / delta + 2.0)
        } else {
            60.0 * ((r - g) / delta + 4.0)
        };
        Self { h, s, v }
    }

    /// Convert to sRGB float triple `(r, g, b) ∈ [0, 1]`.
    pub fn to_srgb(self) -> (f32, f32, f32) {
        if self.s < 1e-6 { return (self.v, self.v, self.v); }
        let h = self.h / 60.0;
        let i = h.floor() as i32;
        let f = h - h.floor();
        let p = self.v * (1.0 - self.s);
        let q = self.v * (1.0 - self.s * f);
        let t = self.v * (1.0 - self.s * (1.0 - f));
        match i.rem_euclid(6) {
            0 => (self.v, t, p),
            1 => (q, self.v, p),
            2 => (p, self.v, t),
            3 => (p, q, self.v),
            4 => (t, p, self.v),
            _ => (self.v, p, q),
        }
    }

    // ── Conversion: linear Rgb ────────────────────────────────────────────────

    /// From linear `Rgb` — internally converts to sRGB first.
    #[inline] pub fn from_linear(c: Rgb) -> Self {
        let s = c.to_srgb(); Self::from_srgb(s.r, s.g, s.b)
    }

    /// To linear `Rgb` — decodes from sRGB.
    #[inline] pub fn to_linear(self) -> Rgb {
        let (r, g, b) = self.to_srgb(); Rgb { r, g, b }.from_srgb()
    }

    // ── Operations ────────────────────────────────────────────────────────────

    /// Rotate hue by `delta` degrees. Wraps around [0, 360).
    #[inline] pub fn shift_hue(self, delta: f32) -> Self {
        Self { h: (self.h + delta).rem_euclid(360.0), ..self }
    }
    /// Scale saturation by factor. Clamped to [0, 1].
    #[inline] pub fn scale_saturation(self, f: f32) -> Self {
        Self { s: (self.s * f).clamp(0.0, 1.0), ..self }
    }
    /// Desaturate by `amount` toward grey. `1.0` = fully grey.
    #[inline] pub fn desaturate(self, amount: f32) -> Self {
        self.scale_saturation(1.0 - amount.clamp(0.0, 1.0))
    }
    /// Scale brightness. Clamped to [0, 1].
    #[inline] pub fn scale_value(self, f: f32) -> Self {
        Self { v: (self.v * f).clamp(0.0, 1.0), ..self }
    }

    /// Complementary hue (180° opposite).
    #[inline] pub fn complement(self) -> Self { self.shift_hue(180.0) }

    /// Lerp in HSV space — takes shortest hue arc.
    pub fn lerp(self, rhs: Self, t: f32) -> Self {
        let dh = ((rhs.h - self.h + 180.0).rem_euclid(360.0)) - 180.0;
        Self {
            h: (self.h + dh * t).rem_euclid(360.0),
            s: self.s + (rhs.s - self.s) * t,
            v: self.v + (rhs.v - self.v) * t,
        }
    }
}

impl fmt::Debug for Hsv {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Hsv(h={:.1}°, s={:.3}, v={:.3})", self.h, self.s, self.v)
    }
}
impl fmt::Display for Hsv {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "hsv({:.1},{:.3},{:.3})", self.h, self.s, self.v)
    }
}
impl From<Rgb> for Hsv { #[inline] fn from(c: Rgb) -> Self { Self::from_linear(c) } }
impl From<Hsv> for Rgb { #[inline] fn from(c: Hsv) -> Self { c.to_linear() } }
