// crates/mid-math/src/color/color32.rs
//! Packed sRGB u8 color — 4 bytes, the standard GPU upload format.

use core::fmt;

/// Packed 8-bit RGBA color in **sRGB** space. 4 bytes, align 1.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(C)]
pub struct Color32 {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl Color32 {
    pub const TRANSPARENT: Self = Self { r: 0,   g: 0,   b: 0,   a: 0   };
    pub const BLACK:       Self = Self { r: 0,   g: 0,   b: 0,   a: 255 };
    pub const WHITE:       Self = Self { r: 255, g: 255, b: 255, a: 255 };
    pub const RED:         Self = Self { r: 255, g: 0,   b: 0,   a: 255 };
    pub const GREEN:       Self = Self { r: 0,   g: 255, b: 0,   a: 255 };
    pub const BLUE:        Self = Self { r: 0,   g: 0,   b: 255, a: 255 };
    pub const YELLOW:      Self = Self { r: 255, g: 255, b: 0,   a: 255 };
    pub const CYAN:        Self = Self { r: 0,   g: 255, b: 255, a: 255 };
    pub const MAGENTA:     Self = Self { r: 255, g: 0,   b: 255, a: 255 };

    #[inline(always)]
    pub const fn new(r: u8, g: u8, b: u8, a: u8) -> Self { Self { r, g, b, a } }

    #[inline(always)]
    pub const fn new_opaque(r: u8, g: u8, b: u8) -> Self { Self { r, g, b, a: 255 } }

    #[inline(always)]
    pub const fn from_array(a: [u8; 4]) -> Self { Self { r: a[0], g: a[1], b: a[2], a: a[3] } }

    #[inline(always)]
    pub const fn to_array(self) -> [u8; 4] { [self.r, self.g, self.b, self.a] }

    #[inline(always)]
    pub const fn to_u32_rgba(self) -> u32 {
        (self.r as u32) << 24 | (self.g as u32) << 16 | (self.b as u32) << 8 | self.a as u32
    }

    #[inline(always)]
    pub const fn to_u32_argb(self) -> u32 {
        (self.a as u32) << 24 | (self.r as u32) << 16 | (self.g as u32) << 8 | self.b as u32
    }

    #[inline(always)]
    pub const fn from_u32_rgba(v: u32) -> Self {
        Self {
            r: (v >> 24) as u8,
            g: (v >> 16) as u8,
            b: (v >>  8) as u8,
            a: (v      ) as u8,
        }
    }

    /// Build from a packed `0xRRGGBBAA` hex literal — same as `from_u32_rgba`.
    ///
    /// ```rust
    /// let red = Color32::from_hex_u32(0xFF0000FF);
    /// assert_eq!(red, Color32::RED);
    /// ```
    #[inline(always)]
    pub const fn from_hex_u32(hex: u32) -> Self {
        Self::from_u32_rgba(hex)
    }

    /// Pack to a `0xRRGGBBAA` u32. Same as `to_u32_rgba`.
    #[inline(always)]
    pub const fn to_hex_u32(self) -> u32 {
        self.to_u32_rgba()
    }

    /// Parse from a CSS-style hex string: `"#RRGGBB"` or `"#RRGGBBAA"`.
    ///
    /// Returns `None` if the string is not a valid 7- or 9-character hex color.
    /// Alpha defaults to `255` for 6-digit forms.
    pub fn from_hex(s: &str) -> Option<Self> {
        let s = s.strip_prefix('#')?;
        match s.len() {
            6 => {
                let r = u8::from_str_radix(&s[0..2], 16).ok()?;
                let g = u8::from_str_radix(&s[2..4], 16).ok()?;
                let b = u8::from_str_radix(&s[4..6], 16).ok()?;
                Some(Self { r, g, b, a: 255 })
            }
            8 => {
                let r = u8::from_str_radix(&s[0..2], 16).ok()?;
                let g = u8::from_str_radix(&s[2..4], 16).ok()?;
                let b = u8::from_str_radix(&s[4..6], 16).ok()?;
                let a = u8::from_str_radix(&s[6..8], 16).ok()?;
                Some(Self { r, g, b, a })
            }
            _ => None,
        }
    }

    /// Format as `"#RRGGBB"` (ignores alpha).
    pub fn to_hex_rgb(self) -> String {
        format!("#{:02X}{:02X}{:02X}", self.r, self.g, self.b)
    }

    /// Format as `"#RRGGBBAA"`.
    pub fn to_hex_rgba(self) -> String {
        format!("#{:02X}{:02X}{:02X}{:02X}", self.r, self.g, self.b, self.a)
    }

    #[inline]
    pub fn blend_over(self, src: Self) -> Self {
        if src.a == 255 { return src; }
        if src.a == 0   { return self; }
        let a  = src.a as u32;
        let ia = 255 - a;
        Self {
            r: ((src.r as u32 * a + self.r as u32 * ia) / 255) as u8,
            g: ((src.g as u32 * a + self.g as u32 * ia) / 255) as u8,
            b: ((src.b as u32 * a + self.b as u32 * ia) / 255) as u8,
            a: src.a.saturating_add(((self.a as u32 * ia) / 255) as u8),
        }
    }

    #[inline]
    pub fn scale_alpha(self, factor: u8) -> Self {
        Self { a: ((self.a as u16 * factor as u16) / 255) as u8, ..self }
    }

    #[inline]
    pub fn premultiply(self) -> Self {
        let a = self.a as u32;
        Self {
            r: ((self.r as u32 * a) / 255) as u8,
            g: ((self.g as u32 * a) / 255) as u8,
            b: ((self.b as u32 * a) / 255) as u8,
            a: self.a,
        }
    }

    #[inline] pub fn with_alpha(self, a: u8) -> Self { Self { a, ..self } }
    #[inline] pub fn is_opaque(self) -> bool { self.a == 255 }
    #[inline] pub fn is_transparent(self) -> bool { self.a == 0 }
}

impl fmt::Debug for Color32 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Color32(#{:02X}{:02X}{:02X}{:02X})", self.r, self.g, self.b, self.a)
    }
}

impl fmt::Display for Color32 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "#{:02X}{:02X}{:02X}{:02X}", self.r, self.g, self.b, self.a)
    }
}

impl From<[u8; 4]> for Color32 { #[inline] fn from(a: [u8; 4]) -> Self { Self::from_array(a) } }
impl From<Color32> for [u8; 4] { #[inline] fn from(c: Color32) -> Self { c.to_array() } }
impl From<(u8, u8, u8, u8)> for Color32 {
    #[inline] fn from((r,g,b,a): (u8,u8,u8,u8)) -> Self { Self::new(r,g,b,a) }
}
