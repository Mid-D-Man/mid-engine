// crates/mid-math/src/color/ycbcr.rs
//! YCbCr color space — luminance + chrominance separation.
//!
//! Standard for digital video encoding (JPEG, H.264, H.265, HEVC).
//! Separates brightness (Y) from color information (Cb/Cr), matching human
//! vision which is more sensitive to luminance than chrominance.
//!
//! Two standards supported:
//!   BT.601 — Standard Definition (480i/576i). Used by JPEG.
//!   BT.709 — High Definition (720p/1080p). Used by most modern video.
//!
//! **Important:** YCbCr operates in **sRGB** space, not linear.
//! Convert linear Rgb → sRGB before encoding; sRGB → linear after decoding.
//!
//! Range: all components in [0, 1] (full range / "PC levels").

use core::fmt;
use super::rgb::Rgb;

/// ITU-R standard selector for YCbCr matrix coefficients.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum YCbCrStandard {
    /// BT.601 — SD video / JPEG. Luma: 0.299R + 0.587G + 0.114B.
    Bt601,
    /// BT.709 — HD video / sRGB. Luma: 0.2126R + 0.7152G + 0.0722B.
    Bt709,
}

/// YCbCr color. All components in [0, 1] (full range).
///
/// Y  ∈ [0, 1]: luma (perceptual brightness)
/// Cb ∈ [0, 1]: blue-difference chroma (0.5 = neutral)
/// Cr ∈ [0, 1]: red-difference chroma  (0.5 = neutral)
#[derive(Clone, Copy, PartialEq, Default)]
#[repr(C)]
pub struct YCbCr {
    pub y:  f32,
    pub cb: f32,
    pub cr: f32,
}

impl YCbCr {
    pub const BLACK: Self = Self { y: 0.0, cb: 0.5, cr: 0.5 };
    pub const WHITE: Self = Self { y: 1.0, cb: 0.5, cr: 0.5 };

    #[inline(always)] pub const fn new(y: f32, cb: f32, cr: f32) -> Self { Self { y, cb, cr } }

    // ── BT.601 ────────────────────────────────────────────────────────────────

    /// Encode sRGB → YCbCr using BT.601 (JPEG, SD video).
    ///
    /// Input: sRGB float `(r, g, b) ∈ [0, 1]` (NOT linear).
    pub fn from_srgb_bt601(r: f32, g: f32, b: f32) -> Self {
        Self {
            y:   0.299       * r + 0.587       * g + 0.114       * b,
            cb: -0.168736    * r - 0.331264    * g + 0.5         * b + 0.5,
            cr:  0.5         * r - 0.418688    * g - 0.081312    * b + 0.5,
        }
    }

    /// Decode BT.601 → sRGB float `(r, g, b) ∈ [0, 1]`.
    pub fn to_srgb_bt601(self) -> (f32, f32, f32) {
        let cb = self.cb - 0.5;
        let cr = self.cr - 0.5;
        let r = (self.y                  + 1.402       * cr).clamp(0.0, 1.0);
        let g = (self.y - 0.344136 * cb - 0.714136    * cr).clamp(0.0, 1.0);
        let b = (self.y + 1.772    * cb                   ).clamp(0.0, 1.0);
        (r, g, b)
    }

    // ── BT.709 ────────────────────────────────────────────────────────────────

    /// Encode sRGB → YCbCr using BT.709 (HD video).
    ///
    /// Input: sRGB float `(r, g, b) ∈ [0, 1]` (NOT linear).
    pub fn from_srgb_bt709(r: f32, g: f32, b: f32) -> Self {
        Self {
            y:   0.2126     * r + 0.7152     * g + 0.0722     * b,
            cb: -0.114572   * r - 0.385428   * g + 0.5        * b + 0.5,
            cr:  0.5        * r - 0.454153   * g - 0.045847   * b + 0.5,
        }
    }

    /// Decode BT.709 → sRGB float.
    pub fn to_srgb_bt709(self) -> (f32, f32, f32) {
        let cb = self.cb - 0.5;
        let cr = self.cr - 0.5;
        let r = (self.y                   + 1.5748   * cr).clamp(0.0, 1.0);
        let g = (self.y - 0.187324 * cb  - 0.468124 * cr).clamp(0.0, 1.0);
        let b = (self.y + 1.8556   * cb                  ).clamp(0.0, 1.0);
        (r, g, b)
    }

    // ── Generic dispatch ──────────────────────────────────────────────────────

    /// Encode from linear `Rgb` using the given standard.
    ///
    /// Converts to sRGB internally — linear values must not be passed directly.
    pub fn from_linear(c: Rgb, standard: YCbCrStandard) -> Self {
        let s = c.to_srgb();
        match standard {
            YCbCrStandard::Bt601 => Self::from_srgb_bt601(s.r, s.g, s.b),
            YCbCrStandard::Bt709 => Self::from_srgb_bt709(s.r, s.g, s.b),
        }
    }

    /// Decode to linear `Rgb` using the given standard.
    pub fn to_linear(self, standard: YCbCrStandard) -> Rgb {
        let (r, g, b) = match standard {
            YCbCrStandard::Bt601 => self.to_srgb_bt601(),
            YCbCrStandard::Bt709 => self.to_srgb_bt709(),
        };
        Rgb { r, g, b }.from_srgb()
    }

    // ── Pack/unpack 8-bit ─────────────────────────────────────────────────────

    /// Pack to 8-bit digital (Y: 0-255, Cb/Cr: 0-255 centered at 128).
    #[inline]
    pub fn to_u8(self) -> (u8, u8, u8) {
        (
            (self.y  * 255.0 + 0.5).clamp(0.0, 255.0) as u8,
            (self.cb * 255.0 + 0.5).clamp(0.0, 255.0) as u8,
            (self.cr * 255.0 + 0.5).clamp(0.0, 255.0) as u8,
        )
    }

    /// Unpack from 8-bit digital.
    #[inline]
    pub fn from_u8(y: u8, cb: u8, cr: u8) -> Self {
        Self {
            y:  y  as f32 / 255.0,
            cb: cb as f32 / 255.0,
            cr: cr as f32 / 255.0,
        }
    }

    // ── Utilities ─────────────────────────────────────────────────────────────

    /// Extract luminance only (greyscale proxy).
    #[inline] pub fn to_greyscale(self) -> Rgb { Rgb::splat(self.y) }
    /// Check if effectively achromatic (Cb≈Cr≈0.5).
    #[inline] pub fn is_achromatic(self) -> bool {
        (self.cb - 0.5).abs() < 0.01 && (self.cr - 0.5).abs() < 0.01
    }
}

impl fmt::Debug for YCbCr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "YCbCr(y={:.3}, cb={:.3}, cr={:.3})", self.y, self.cb, self.cr)
    }
}
impl fmt::Display for YCbCr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ycbcr({:.3},{:.3},{:.3})", self.y, self.cb, self.cr)
    }
}
