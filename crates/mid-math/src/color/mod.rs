// crates/mid-math/src/color/mod.rs
//! Color types for Mid Engine.
//!
//! Three representations, one clear rule:
//!
//! | Type      | Space       | When to use                                    |
//! |-----------|-------------|------------------------------------------------|
//! | `Rgb`     | Linear f32  | All math: lerp, blend, tone map, lighting     |
//! | `Rgba`    | Linear f32  | Same + alpha compositing                       |
//! | `Color32` | sRGB u8     | GPU upload, PNG/texture I/O, UI widgets        |
//!
//! # Conversion path
//! ```text
//! PNG/hex → Color32 → Rgba::from_color32() → [math] → .to_color32() → GPU
//! ```
//! Never do math in sRGB space — results will be physically wrong.

mod color32;
mod rgb;
mod rgba;

pub use color32::Color32;
pub use rgb::Rgb;
pub use rgba::Rgba;
