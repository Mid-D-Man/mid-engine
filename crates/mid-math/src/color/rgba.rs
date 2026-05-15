// crates/mid-math/src/color/rgba.rs
//! Linear-light RGBA color — 4 × f32.

use core::fmt;
use core::ops::{Add, AddAssign, Mul, MulAssign};
use super::rgb::Rgb;
use super::color32::Color32;

/// Linear-light RGBA color. RGB in `[0, 1]` physical range (HDR > 1.0 valid); A in `[0, 1]`.
///
/// Alpha is always linear — never gamma-encoded.
/// Convert to `Color32` (sRGB u8) only at GPU upload / texture write boundaries.
#[derive(Clone, Copy, PartialEq, Default)]
#[repr(C)]
pub struct Rgba {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl Rgba {
    pub const TRANSPARENT: Self = Self { r: 0.0, g: 0.0, b: 0.0, a: 0.0 };
    pub const BLACK:       Self = Self { r: 0.0, g: 0.0, b: 0.0, a: 1.0 };
    pub const WHITE:       Self = Self { r: 1.0, g: 1.0, b: 1.0, a: 1.0 };
    pub const RED:         Self = Self { r: 1.0, g: 0.0, b: 0.0, a: 1.0 };
    pub const GREEN:       Self = Self { r: 0.0, g: 1.0, b: 0.0, a: 1.0 };
    pub const BLUE:        Self = Self { r: 0.0, g: 0.0, b: 1.0, a: 1.0 };

    // ── Constructors ──────────────────────────────────────────────────────────

    #[inline(always)] pub const fn new(r: f32, g: f32, b: f32, a: f32) -> Self { Self { r, g, b, a } }
    #[inline(always)] pub fn from_rgb(rgb: Rgb, a: f32) -> Self { Self { r: rgb.r, g: rgb.g, b: rgb.b, a } }
    #[inline(always)] pub fn to_rgb(self) -> Rgb { Rgb { r: self.r, g: self.g, b: self.b } }
    #[inline(always)] pub fn from_array(a: [f32; 4]) -> Self { Self { r: a[0], g: a[1], b: a[2], a: a[3] } }
    #[inline(always)] pub fn to_array(self) -> [f32; 4] { [self.r, self.g, self.b, self.a] }

    /// Build from `Color32` (sRGB u8) — converts RGB to linear; alpha stays linear.
    #[inline]
    pub fn from_color32(c: Color32) -> Self {
        Self {
            r: Rgb::srgb_to_linear_ch(c.r as f32 / 255.0),
            g: Rgb::srgb_to_linear_ch(c.g as f32 / 255.0),
            b: Rgb::srgb_to_linear_ch(c.b as f32 / 255.0),
            a: c.a as f32 / 255.0,
        }
    }

    // ── sRGB conversion ───────────────────────────────────────────────────────

    /// Interpret RGB channels as sRGB-encoded and return linear equivalent.
    /// Alpha is unchanged.
    #[inline] pub fn from_srgb(self) -> Self { Self::from_rgb(self.to_rgb().from_srgb(), self.a) }

    /// Convert linear RGB channels to sRGB encoding. Alpha is unchanged.
    #[inline] pub fn to_srgb(self) -> Self { Self::from_rgb(self.to_rgb().to_srgb(), self.a) }

    // ── Tone mapping ──────────────────────────────────────────────────────────

    /// Per-channel Reinhard on RGB. Alpha preserved.
    #[inline] pub fn tone_map_reinhard(self) -> Self {
        Self::from_rgb(self.to_rgb().tone_map_reinhard(), self.a)
    }

    /// Luminance-based Reinhard on RGB. Alpha preserved.
    #[inline] pub fn tone_map_reinhard_luminance(self) -> Self {
        Self::from_rgb(self.to_rgb().tone_map_reinhard_luminance(), self.a)
    }

    /// ACES filmic on RGB. Alpha preserved.
    #[inline] pub fn tone_map_aces(self) -> Self {
        Self::from_rgb(self.to_rgb().tone_map_aces(), self.a)
    }

    // ── Alpha operations ──────────────────────────────────────────────────────

    /// Premultiply: `rgb *= a`. Required before many blend operations.
    #[inline]
    pub fn premultiply_alpha(self) -> Self {
        Self { r: self.r * self.a, g: self.g * self.a, b: self.b * self.a, a: self.a }
    }

    /// Un-premultiply: `rgb /= a`. Returns `TRANSPARENT` when `a ≈ 0`.
    #[inline]
    pub fn unpremultiply_alpha(self) -> Self {
        if self.a < 1e-6 { return Self::TRANSPARENT; }
        let inv = 1.0 / self.a;
        Self { r: self.r * inv, g: self.g * inv, b: self.b * inv, a: self.a }
    }

    #[inline] pub fn with_alpha(self, a: f32) -> Self { Self { a, ..self } }

    // ── Operations ────────────────────────────────────────────────────────────

    #[inline]
    pub fn lerp(self, rhs: Self, t: f32) -> Self {
        Self {
            r: self.r + (rhs.r - self.r) * t,
            g: self.g + (rhs.g - self.g) * t,
            b: self.b + (rhs.b - self.b) * t,
            a: self.a + (rhs.a - self.a) * t,
        }
    }

    #[inline]
    pub fn clamp(self, lo: f32, hi: f32) -> Self {
        Self {
            r: self.r.clamp(lo, hi), g: self.g.clamp(lo, hi),
            b: self.b.clamp(lo, hi), a: self.a.clamp(lo, hi),
        }
    }

    #[inline] pub fn scale(self, s: f32) -> Self { Self { r: self.r*s, g: self.g*s, b: self.b*s, a: self.a*s } }
    #[inline] pub fn luminance(self) -> f32 { self.to_rgb().luminance() }

    // ── Hex parsing ───────────────────────────────────────────────────────────

    /// Parse from `"#RRGGBB"` (alpha=1.0) or `"#RRGGBBAA"`.
    /// RGB is sRGB-decoded to linear; alpha is linear as-is.
    pub fn from_hex(s: &str) -> Option<Self> {
        let s = s.strip_prefix('#')?;
        match s.len() {
            6 => {
                let r = u8::from_str_radix(&s[0..2], 16).ok()? as f32 / 255.0;
                let g = u8::from_str_radix(&s[2..4], 16).ok()? as f32 / 255.0;
                let b = u8::from_str_radix(&s[4..6], 16).ok()? as f32 / 255.0;
                Some(Self::from_rgb(Rgb { r, g, b }.from_srgb(), 1.0))
            }
            8 => {
                let r = u8::from_str_radix(&s[0..2], 16).ok()? as f32 / 255.0;
                let g = u8::from_str_radix(&s[2..4], 16).ok()? as f32 / 255.0;
                let b = u8::from_str_radix(&s[4..6], 16).ok()? as f32 / 255.0;
                let a = u8::from_str_radix(&s[6..8], 16).ok()? as f32 / 255.0;
                Some(Self::from_rgb(Rgb { r, g, b }.from_srgb(), a))
            }
            _ => None,
        }
    }

    /// Convert to packed `Color32` (sRGB u8). Clamps HDR values to [0, 1].
    ///
    /// RGB channels are gamma-encoded. Alpha is linear-scaled.
    #[inline]
    pub fn to_color32(self) -> Color32 {
        let srgb = self.to_srgb().clamp(0.0, 1.0);
        Color32::new(
            (srgb.r * 255.0 + 0.5) as u8,
            (srgb.g * 255.0 + 0.5) as u8,
            (srgb.b * 255.0 + 0.5) as u8,
            (self.a.clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
        )
    }
}

impl Add  for Rgba { type Output=Self; #[inline] fn add(self,r:Self)->Self { Self{r:self.r+r.r,g:self.g+r.g,b:self.b+r.b,a:self.a+r.a} } }
impl AddAssign for Rgba { #[inline] fn add_assign(&mut self,r:Self){*self=*self+r;} }
impl Mul<f32> for Rgba { type Output=Self; #[inline] fn mul(self,s:f32)->Self { self.scale(s) } }
impl MulAssign<f32> for Rgba { #[inline] fn mul_assign(&mut self,s:f32){*self=*self*s;} }
impl Mul for Rgba { type Output=Self; #[inline] fn mul(self,r:Self)->Self { Self{r:self.r*r.r,g:self.g*r.g,b:self.b*r.b,a:self.a*r.a} } }

impl From<Rgb> for Rgba { #[inline] fn from(rgb: Rgb) -> Self { Self::from_rgb(rgb, 1.0) } }
impl From<Rgba> for Rgb  { #[inline] fn from(c: Rgba) -> Rgb  { c.to_rgb() } }
impl From<Color32> for Rgba { #[inline] fn from(c: Color32) -> Self { Self::from_color32(c) } }
impl From<Rgba> for Color32 { #[inline] fn from(c: Rgba) -> Color32 { c.to_color32() } }
impl From<[f32;4]> for Rgba { fn from(a:[f32;4])->Self{Self::from_array(a)} }
impl From<Rgba> for [f32;4] { fn from(v:Rgba)->[f32;4]{v.to_array()} }

impl fmt::Debug for Rgba {
    fn fmt(&self,f:&mut fmt::Formatter<'_>)->fmt::Result {
        write!(f,"Rgba({:.4},{:.4},{:.4},{:.4})",self.r,self.g,self.b,self.a)
    }
}
impl fmt::Display for Rgba {
    fn fmt(&self,f:&mut fmt::Formatter<'_>)->fmt::Result {
        write!(f,"rgba({:.4},{:.4},{:.4},{:.4})",self.r,self.g,self.b,self.a)
    }
                                        }
