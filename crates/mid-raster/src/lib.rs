// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-raster.md, section "lib.rs"
// ============================================================================
//! Integer-exact CPU coverage rasterizer for Mid Engine.
//!
//! Draws rectangles, rounded rectangles, thick lines and filled paths into a
//! premultiplied `RGBA8` surface through an axis-aligned clip stack, in sRGB
//! or linear-light blending. Anti-aliasing is exact area coverage computed in
//! 24.8 fixed point, so the same input produces the same bytes on every target.
//! The contracts it follows are in `docs/mid-ui-contracts.md`. It depends on
//! `mid-math` only and has no `unsafe` code.
//!
//! ```
//! use mid_raster::{Canvas, Surface};
//! use mid_math::Color32;
//!
//! let mut surface = Surface::new(8, 8);
//! let mut canvas = Canvas::new(surface.as_mut());
//! canvas.fill_rect(1.0, 1.0, 4.0, 4.0, Color32::new(255, 0, 0, 255));
//! assert_eq!(surface.pixel(2, 2), Color32::new(255, 0, 0, 255));
//! assert_eq!(surface.pixel(6, 6), Color32::new(0, 0, 0, 0));
//! ```

pub mod canvas;
pub mod clip;
pub mod cover;
pub mod fixed;
pub mod linear;
pub mod path;
pub mod pixel;
pub mod shapes;
pub mod tables;

pub use canvas::{Canvas, Stats};
pub use clip::{ClipStack, IRect};
pub use cover::{FillRule, Mask};
pub use path::Path;
pub use pixel::{BlendSpace, Surface, SurfaceMut};
