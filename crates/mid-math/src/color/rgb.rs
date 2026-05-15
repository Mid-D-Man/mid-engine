// crates/mid-math/src/color/rgb.rs
//! Linear-light RGB color — 3 × f32.

use core::fmt;
use core::ops::{Add, AddAssign, Mul, MulAssign, Sub};

/// Linear-light RGB color. Components in `[0, 1]` for physical range; HDR > 1.0 is valid.
///
/// This is **linear** RGB — not sRGB gamma-encoded.
/// Always work in linear space; convert to sRGB only at output boundaries
/// (texture writes, framebuffer output, UI widgets).
#[derive(Clone, Copy, PartialEq, Default)]
#[repr(C)]
pub struct Rgb {
    pub r: f32,
    pub g: f32,
    pub b: f32,
}

impl Rgb {
    pub const BLACK:  Self = Self { r: 0.0, g: 0.0, b: 0.0 };
    pub const WHITE:  Self = Self { r: 1.0, g: 1.0, b: 1.0 };
    pub const RED:    Self = Self { r: 1.0, g: 0.0, b: 0.0 };
    pub const GREEN:  Self = Self { r: 0.0, g: 1.0, b: 0.0 };
    pub const BLUE:   Self = Self { r: 0.0, g: 0.0, b: 1.0 };
    pub const YELLOW: Self = Self { r: 1.0, g: 1.0, b: 0.0 };
    pub const CYAN:   Self = Self { r: 0.0, g: 1.0, b: 1.0 };
    pub const MAGENTA:Self = Self { r: 1.0, g: 0.0, b: 1.0 };

    // ── Constructors ──────────────────────────────────────────────────────────

    #[inline(always)] pub const fn new(r: f32, g: f32, b: f32) -> Self { Self { r, g, b } }
    #[inline(always)] pub fn splat(v: f32) -> Self { Self { r: v, g: v, b: v } }
    #[inline(always)] pub fn from_array(a: [f32; 3]) -> Self { Self { r: a[0], g: a[1], b: a[2] } }
    #[inline(always)] pub fn to_array(self) -> [f32; 3] { [self.r, self.g, self.b] }

    // ── sRGB conversion ───────────────────────────────────────────────────────

    /// Linear → sRGB for a single channel. IEC 61966-2-1 standard formula.
    #[inline]
    pub fn linear_to_srgb_ch(c: f32) -> f32 {
        if c <= 0.0031308 {
            12.92 * c
        } else {
            1.055 * c.powf(1.0 / 2.4) - 0.055
        }
    }

    /// sRGB → linear for a single channel. IEC 61966-2-1 standard formula.
    #[inline]
    pub fn srgb_to_linear_ch(c: f32) -> f32 {
        if c <= 0.04045 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        }
    }

    /// Interpret `self` as sRGB-encoded and return linear-light equivalent.
    /// Call at input boundaries (texture load, hex parse).
    #[inline]
    pub fn from_srgb(self) -> Self {
        Self {
            r: Self::srgb_to_linear_ch(self.r),
            g: Self::srgb_to_linear_ch(self.g),
            b: Self::srgb_to_linear_ch(self.b),
        }
    }

    /// Convert linear `self` to sRGB encoding for output.
    #[inline]
    pub fn to_srgb(self) -> Self {
        Self {
            r: Self::linear_to_srgb_ch(self.r),
            g: Self::linear_to_srgb_ch(self.g),
            b: Self::linear_to_srgb_ch(self.b),
        }
    }

    // ── Tone mapping ──────────────────────────────────────────────────────────

    /// Per-channel Reinhard: `c / (1 + c)`. Maps HDR `[0, ∞)` to LDR `[0, 1)`.
    #[inline]
    pub fn tone_map_reinhard(self) -> Self {
        Self { r: self.r / (1.0 + self.r), g: self.g / (1.0 + self.g), b: self.b / (1.0 + self.b) }
    }

    /// Luminance-based Reinhard — preserves hue/saturation better than per-channel.
    ///
    /// Rec. 709 luminance: `L = 0.2126r + 0.7152g + 0.0722b`.
    /// Scale: `s = 1 / (1 + L)`.
    #[inline]
    pub fn tone_map_reinhard_luminance(self) -> Self {
        let lum = self.luminance();
        let s = 1.0 / (1.0 + lum);
        Self { r: self.r * s, g: self.g * s, b: self.b * s }
    }

    /// ACES filmic approximation (Narkowicz 2015).
    ///
    /// `f(x) = (x*(2.51x + 0.03)) / (x*(2.43x + 0.59) + 0.14)`
    /// Better highlight rolloff than simple Reinhard.
    #[inline]
    pub fn tone_map_aces(self) -> Self {
        let aces_ch = |x: f32| -> f32 {
            (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)
        };
        Self { r: aces_ch(self.r), g: aces_ch(self.g), b: aces_ch(self.b) }
    }

    // ── Operations ────────────────────────────────────────────────────────────

    #[inline]
    pub fn lerp(self, rhs: Self, t: f32) -> Self {
        Self {
            r: self.r + (rhs.r - self.r) * t,
            g: self.g + (rhs.g - self.g) * t,
            b: self.b + (rhs.b - self.b) * t,
        }
    }

    #[inline]
    pub fn clamp(self, lo: f32, hi: f32) -> Self {
        Self { r: self.r.clamp(lo, hi), g: self.g.clamp(lo, hi), b: self.b.clamp(lo, hi) }
    }

    #[inline] pub fn scale(self, s: f32) -> Self { Self { r: self.r*s, g: self.g*s, b: self.b*s } }

    /// Rec. 709 luminance: `0.2126r + 0.7152g + 0.0722b`.
    #[inline] pub fn luminance(self) -> f32 { 0.2126 * self.r + 0.7152 * self.g + 0.0722 * self.b }

    /// Component-wise multiply (modulate / tint).
    #[inline]
    pub fn modulate(self, rhs: Self) -> Self {
        Self { r: self.r*rhs.r, g: self.g*rhs.g, b: self.b*rhs.b }
    }

    // ── Hex parsing ───────────────────────────────────────────────────────────

    /// Parse from `"#RRGGBB"` sRGB hex string and return **linear** RGB.
    ///
    /// Returns `None` if not a valid 7-character hex color.
    pub fn from_hex(s: &str) -> Option<Self> {
        let s = s.strip_prefix('#')?;
        if s.len() != 6 { return None; }
        let r = u8::from_str_radix(&s[0..2], 16).ok()? as f32 / 255.0;
        let g = u8::from_str_radix(&s[2..4], 16).ok()? as f32 / 255.0;
        let b = u8::from_str_radix(&s[4..6], 16).ok()? as f32 / 255.0;
        Some(Self { r, g, b }.from_srgb())
    }

    /// Convert to `Vec3(r, g, b)`.
    #[inline] pub fn to_vec3(self) -> crate::Vec3 { crate::Vec3::new(self.r, self.g, self.b) }
    /// Convert from `Vec3(r, g, b)`.
    #[inline] pub fn from_vec3(v: crate::Vec3) -> Self { Self { r: v.x, g: v.y, b: v.z } }
}

impl Add  for Rgb { type Output=Self; #[inline] fn add(self,r:Self)->Self { Self{r:self.r+r.r,g:self.g+r.g,b:self.b+r.b} } }
impl AddAssign for Rgb { #[inline] fn add_assign(&mut self,r:Self){*self=*self+r;} }
impl Sub  for Rgb { type Output=Self; #[inline] fn sub(self,r:Self)->Self { Self{r:self.r-r.r,g:self.g-r.g,b:self.b-r.b} } }
impl Mul<f32> for Rgb { type Output=Self; #[inline] fn mul(self,s:f32)->Self { self.scale(s) } }
impl MulAssign<f32> for Rgb { #[inline] fn mul_assign(&mut self,s:f32){*self=*self*s;} }
impl Mul for Rgb { type Output=Self; #[inline] fn mul(self,r:Self)->Self { self.modulate(r) } }

impl fmt::Debug for Rgb {
    fn fmt(&self,f:&mut fmt::Formatter<'_>)->fmt::Result { write!(f,"Rgb({:.4},{:.4},{:.4})",self.r,self.g,self.b) }
}
impl fmt::Display for Rgb {
    fn fmt(&self,f:&mut fmt::Formatter<'_>)->fmt::Result { write!(f,"rgb({:.4},{:.4},{:.4})",self.r,self.g,self.b) }
}
impl From<[f32;3]> for Rgb { fn from(a:[f32;3])->Self { Self::from_array(a) } }
impl From<Rgb> for [f32;3] { fn from(v:Rgb)->[f32;3] { v.to_array() } }
