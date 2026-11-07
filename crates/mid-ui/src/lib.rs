// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-ui.md, section "lib.rs"
// ============================================================================
//! mid-ui — standalone UI core for Mid Engine.
//!
//! A widget [`Tree`], anchor/margin [`layout`], pointer [`hit`] testing and a
//! backend-agnostic [`DrawCmd`] list, all generic over [`f32`] and [`f64`]
//! through [`Real`]. It has no renderer, no input loop and no ECS: a host
//! owns the tree, runs [`layout::compute`] when the viewport or a style
//! changes, asks [`hit::claims_pointer`] before routing pointer input to a
//! game world, and feeds [`draw::build_draw_list`] to whichever backend it
//! uses.
//!
//! ```
//! use mid_ui::{draw, layout, Anchor, DrawCmd, Sizing, Size, Style, Tree, Color32};
//!
//! let mut tree = Tree::<f32>::new(Style::default());
//! let panel = tree.add_child(tree.root(), Style {
//!     anchor: Anchor::Center,
//!     width: Sizing::Fixed(100.0),
//!     height: Sizing::Fixed(40.0),
//!     background: Color32::WHITE,
//!     ..Style::default()
//! }).unwrap();
//!
//! layout::compute(&mut tree, Size::new(200.0, 100.0));
//! assert_eq!(tree.rect(panel).unwrap().x, 50.0);
//!
//! let mut list = Vec::new();
//! draw::build_draw_list(&tree, &mut list);
//! assert!(matches!(list[0], DrawCmd::FillRect { .. }));
//! ```

pub mod draw;
pub mod geom;
pub mod hit;
pub mod layout;
pub mod real;
pub mod style;
pub mod tree;

pub use draw::DrawCmd;
pub use geom::{Insets, Point, Rect, Size};
pub use mid_math::Color32;
pub use real::Real;
pub use style::{Anchor, Flow, Sizing, Style};
pub use tree::{NodeId, Tree};
