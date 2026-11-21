// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-editor.md, section "lib.rs"
// ============================================================================
//! mid-editor: the platform-independent core of the Mid Engine editor.
//!
//! The editor's panels are a [`mid_ui`] tree, its viewport is a scene
//! projected on the CPU with [`mid_math`], and its output is one flat
//! [`PaintList`] per frame for a host to draw. A host (an Android activity, a
//! browser page) turns platform events into [`Editor::pointer`] calls, calls
//! [`Editor::frame`], and paints [`Editor::paint`]. There is no renderer, no
//! windowing and no `unsafe` here.

#![forbid(unsafe_code)]

pub mod camera;
pub mod editor;
pub mod input;
pub mod paint;
pub mod scene;
pub mod stats;
pub mod text;
pub mod viewport;

pub use editor::{Editor, Mode, Tab};
pub use input::Phase;
pub use paint::{PaintItem, PaintList};
pub use text::{FixedWidth, TextMeasure};
