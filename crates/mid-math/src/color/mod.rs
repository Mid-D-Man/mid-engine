// crates/mid-math/src/color/mod.rs
//! Color types for Mid Engine.
//!
//! ## Representation guide
//!
//! | Type       | Space       | When to use                                     |
//! |------------|-------------|--------------------------------------------------|
//! | `Rgb`      | Linear f32  | All lighting math: lerp, blend, tone map         |
//! | `Rgba`     | Linear f32  | Same + alpha compositing                         |
//! | `Color32`  | sRGB u8     | GPU upload, PNG/texture I/O, UI widgets          |
//! | `Hsv`      | sRGB f32    | Color pickers, hue rotation, saturation FX       |
//! | `Hsl`      | sRGB f32    | CSS-compatible pickers, lighten/darken           |
//! | `Rgbe`     | Linear HDR  | Environment maps, IBL (Radiance .hdr format)     |
//! | `LogLuv32` | Linear HDR  | Physics lighting, perceptually-accurate HDR      |
//! | `YCbCr`    | sRGB chroma | Video encoding/decoding, texture compression     |
//!
//! ## Conversion pipeline
//! ```text
//! PNG/hex → Color32 → Rgba::from_color32() → [math] → .to_color32() → GPU
//! HDR file → Rgbe   → Rgbe::decode_rgb()   → [math] → Rgbe::encode_rgb()
//! Video    → YCbCr  → .to_linear(BT709)    → [math] → YCbCr::from_linear(BT709)
//! ```
//! Never do lighting math in sRGB, HSV, HSL, or YCbCr space.

mod color32;
mod rgb;
mod rgba;
mod hsv;
mod hsl;
mod loglux;
mod ycbcr;

pub use color32::Color32;
pub use rgb::Rgb;
pub use rgba::Rgba;
pub use hsv::Hsv;
pub use hsl::Hsl;
pub use loglux::{Rgbe, LogLuv32};
pub use ycbcr::{YCbCr, YCbCrStandard};

// ── Standalone sRGB ↔ linear conversion helpers ───────────────────────────────
//
// These duplicate the channel-wise logic already on `Rgb`, but as free
// functions so callers that work with raw f32 values don't need to construct
// a full `Rgb` struct just to convert a single channel.

/// Convert one channel from sRGB gamma-encoded to linear light.
///
/// Uses the IEC 61966-2-1 standard formula — the same formula as
/// [`Rgb::srgb_to_linear_ch`].
///
/// Input is clamped implicitly by the formula (negative inputs map to negative
/// linear values, which is technically valid for HDR blending purposes).
#[inline]
pub fn srgb_to_linear(v: f32) -> f32 {
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

/// Convert one channel from linear light to sRGB gamma-encoded.
///
/// Uses the IEC 61966-2-1 standard formula — the same formula as
/// [`Rgb::linear_to_srgb_ch`].
#[inline]
pub fn linear_to_srgb(v: f32) -> f32 {
    if v <= 0.003_130_8 {
        v * 12.92
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}
