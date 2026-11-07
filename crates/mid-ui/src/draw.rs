// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-ui.md, section "draw.rs"
// ============================================================================
//! The draw-command list: the only thing a renderer backend consumes.

use crate::geom::Rect;
use crate::real::Real;
use crate::tree::{NodeId, Tree};
use mid_math::Color32;

/// One backend-agnostic drawing instruction.
///
/// The list is in paint order. `PushClip` and `PopClip` are always balanced;
/// a backend keeps a clip stack and intersects each pushed rectangle with
/// the one below it.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum DrawCmd<R: Real = f32> {
    /// Fill a rectangle, with rounded corners when `radius` is above zero.
    FillRect {
        /// Rectangle to fill.
        rect: Rect<R>,
        /// Fill color.
        color: Color32,
        /// Corner radius.
        radius: R,
    },
    /// Start clipping to a rectangle.
    PushClip(Rect<R>),
    /// End the most recent clip.
    PopClip,
}

enum Item {
    Visit(NodeId),
    PopClip,
}

/// Appends the draw commands for `tree` to `out`, using the rectangles from
/// the last layout pass. `out` is not cleared, so a caller can reuse one
/// buffer across frames by clearing it itself.
///
/// Hidden subtrees produce nothing. Fully transparent and empty fills are
/// skipped.
pub fn build_draw_list<R: Real>(tree: &Tree<R>, out: &mut Vec<DrawCmd<R>>) {
    let mut stack = vec![Item::Visit(tree.root())];
    while let Some(item) = stack.pop() {
        let id = match item {
            Item::PopClip => {
                out.push(DrawCmd::PopClip);
                continue;
            }
            Item::Visit(id) => id,
        };
        let (style, rect) = match (tree.style(id), tree.rect(id)) {
            (Some(s), Some(r)) => (s, r),
            _ => continue,
        };
        if !style.visible {
            continue;
        }
        if !style.background.is_transparent() && rect.w > R::ZERO && rect.h > R::ZERO {
            out.push(DrawCmd::FillRect { rect, color: style.background, radius: style.radius });
        }
        if style.clip {
            out.push(DrawCmd::PushClip(rect));
            stack.push(Item::PopClip);
        }
        stack.extend(tree.children(id).iter().rev().map(|c| Item::Visit(*c)));
    }
}
