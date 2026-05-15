// crates/mid-math/src/color/loglux.rs
//! HDR color encodings: RGBE and LogLuv32.
//!
//! RGBE  — Radiance .hdr format (Ward 1991). 4 bytes, ~1% quantisation. Fast.
//!          Good for: environment maps, IBL, skyboxes.
//!
//! LogLuv32 — CIE 1976 perceptually-uniform HDR (Ward 1998). 4 bytes.
//!            Separates luminance from chromaticity so quantisation error
//!            matches human perception. Good for: physics-based lighting,
//!            bloom thresholds, HDR tone mapping pipelines.
//!
//! Both accept **linear** RGB only. Never pass sRGB-encoded values.

use core::fmt;
use super::rgb::Rgb;

// ── RGBE ─────────────────────────────────────────────────────────────────────

/// Radiance HDR (RGBE) packed color. 4 bytes.
///
/// Shared exponent in alpha byte allows ~6 orders of magnitude range.
/// Decode precision: ~1/256 relative error per channel.
#[derive(Clone, Copy, PartialEq, Eq, Default)]
#[repr(C)]
pub struct Rgbe { pub r: u8, pub g: u8, pub b: u8, pub e: u8 }

impl Rgbe {
    pub const BLACK: Self = Self { r: 0, g: 0, b: 0, e: 0 };

    /// Encode linear RGB to RGBE. Input must be non-negative.
    pub fn encode(r: f32, g: f32, b: f32) -> Self {
        let m = r.max(g).max(b);
        if m < 1e-32 { return Self::BLACK; }

        // Extract IEEE 754 biased exponent of m.
        // frexp exponent e: m = mantissa * 2^e, 0.5 ≤ mantissa < 1.
        // biased_exp = e + 126 for normals, so e = biased_exp - 126.
        let biased = ((m.to_bits() >> 23) & 0xFF) as i32;
        let e = biased - 126;

        // Scale so that m maps to [128, 256). scale = 2^(-e) * 256.
        let scale = 2f32.powi(-e) * 256.0;
        Self {
            r: (r * scale).min(255.0) as u8,
            g: (g * scale).min(255.0) as u8,
            b: (b * scale).min(255.0) as u8,
            e: (e + 128) as u8,
        }
    }

    /// Decode RGBE to linear RGB.
    pub fn decode(self) -> (f32, f32, f32) {
        if self.e == 0 { return (0.0, 0.0, 0.0); }
        // Undo the scale: scale^-1 = 2^(e - 128) / 256 = 2^(e - 128 - 8)
        let scale = 2f32.powi(self.e as i32 - 128 - 8);
        (
            self.r as f32 * scale,
            self.g as f32 * scale,
            self.b as f32 * scale,
        )
    }

    #[inline] pub fn encode_rgb(c: Rgb) -> Self { Self::encode(c.r, c.g, c.b) }
    #[inline] pub fn decode_rgb(self) -> Rgb {
        let (r, g, b) = self.decode(); Rgb { r, g, b }
    }

    #[inline] pub const fn from_array(a: [u8; 4]) -> Self { Self { r: a[0], g: a[1], b: a[2], e: a[3] } }
    #[inline] pub const fn to_array(self) -> [u8; 4] { [self.r, self.g, self.b, self.e] }
}

impl fmt::Debug for Rgbe {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (r, g, b) = self.decode();
        write!(f, "Rgbe(r={:.3}, g={:.3}, b={:.3}, e={})", r, g, b, self.e as i32 - 128)
    }
}

// ── LogLuv32 ─────────────────────────────────────────────────────────────────
//
// Encoding layout (32 bits):
//   bits [31:16] = L_int (u16): (log2(Y) + 64) * 256, range Y ∈ [2^-64, 2^192)
//   bits [15:8]  = u_int (u8): u' * 255  (CIE 1976 u' chromaticity)
//   bits [7:0]   = v_int (u8): v' * 255  (CIE 1976 v' chromaticity)
//
// RGB → XYZ (D65, linear sRGB primaries):
//   X = 0.4124564r + 0.3575761g + 0.1804375b
//   Y = 0.2126729r + 0.7151522g + 0.0721750b
//   Z = 0.0193339r + 0.1191920g + 0.9503041b
//
// XYZ → u'v' (CIE 1976 UCS):
//   denom = X + 15Y + 3Z
//   u' = 4X / denom
//   v' = 9Y / denom
//
// Decoding u'v'Y → XYZ:
//   W = 9Y / v' ... see decode() impl for full derivation

/// LogLuv32 HDR packed color. 4 bytes.
///
/// Perceptually uniform luminance-chromaticity separation.
/// ~10 bits effective luminance precision, ~6 bits chromaticity each channel.
#[derive(Clone, Copy, PartialEq, Eq, Default)]
#[repr(transparent)]
pub struct LogLuv32(pub u32);

impl LogLuv32 {
    pub const BLACK: Self = LogLuv32(0);

    /// Encode linear RGB to LogLuv32. Input must be non-negative.
    pub fn encode(r: f32, g: f32, b: f32) -> Self {
        // Linear RGB → CIE XYZ (D65 white point)
        let x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b;
        let y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b;
        let z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b;

        if y < 1e-32 { return Self::BLACK; }

        // XYZ → CIE 1976 u'v' chromaticity
        let denom = x + 15.0 * y + 3.0 * z;
        let (up, vp) = if denom < 1e-32 {
            (0.0f32, 0.0f32)
        } else {
            (4.0 * x / denom, 9.0 * y / denom)
        };

        // Log2 luminance, biased by 64, scaled by 256 → u16
        let l_f = (y.log2() + 64.0) * 256.0;
        let l_int = l_f.clamp(0.0, 65535.0) as u16;

        let u_int = (up.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        let v_int = (vp.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;

        LogLuv32(((l_int as u32) << 16) | ((u_int as u32) << 8) | v_int as u32)
    }

    /// Decode LogLuv32 to linear RGB.
    pub fn decode(self) -> (f32, f32, f32) {
        if self.0 == 0 { return (0.0, 0.0, 0.0); }

        let l_int = (self.0 >> 16) as u16;
        let u_int = ((self.0 >> 8) & 0xFF) as u8;
        let v_int = (self.0 & 0xFF) as u8;

        let l = l_int as f32 / 256.0 - 64.0;
        let y = 2f32.powf(l);

        let up = u_int as f32 / 255.0;
        let vp = v_int as f32 / 255.0;

        if vp < 1e-6 { return (0.0, 0.0, 0.0); }

        // Recover XYZ from Y, u', v'
        // denom W = 9Y / v' (from v' = 9Y/W)
        // X = u'W/4 = 9u'Y / (4v')
        // Z = Y*(36 - 9u' - 60v') / (12v')
        let x = 9.0 * up * y / (4.0 * vp);
        let z = y * (36.0 - 9.0 * up - 60.0 * vp) / (12.0 * vp);

        // CIE XYZ → linear sRGB (D65 inverse matrix)
        let r =  3.2404542 * x - 1.5371385 * y - 0.4985314 * z;
        let g = -0.9692660 * x + 1.8760108 * y + 0.0415560 * z;
        let b =  0.0556434 * x - 0.2040259 * y + 1.0572252 * z;

        (r.max(0.0), g.max(0.0), b.max(0.0))
    }

    #[inline] pub fn encode_rgb(c: Rgb) -> Self { Self::encode(c.r, c.g, c.b) }
    #[inline] pub fn decode_rgb(self) -> Rgb {
        let (r, g, b) = self.decode(); Rgb { r, g, b }
    }
    #[inline] pub fn raw(self) -> u32 { self.0 }
    #[inline] pub fn luminance_log2(self) -> f32 {
        (self.0 >> 16) as f32 / 256.0 - 64.0
    }
}

impl fmt::Debug for LogLuv32 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (r, g, b) = self.decode();
        write!(f, "LogLuv32(r={:.3}, g={:.3}, b={:.3})", r, g, b)
    }
}
impl From<Rgb> for LogLuv32 { fn from(c: Rgb) -> Self { Self::encode_rgb(c) } }
impl From<LogLuv32> for Rgb  { fn from(c: LogLuv32) -> Self { c.decode_rgb() } }
impl From<Rgb> for Rgbe      { fn from(c: Rgb) -> Self { Self::encode_rgb(c) } }
impl From<Rgbe> for Rgb      { fn from(c: Rgbe) -> Self { c.decode_rgb() } }
