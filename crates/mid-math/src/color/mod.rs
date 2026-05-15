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
