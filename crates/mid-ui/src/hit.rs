// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-ui.md, section "hit.rs"
// ============================================================================
//! Pointer hit testing and the UI-versus-app arbitration seam.

use crate::geom::Point;
use crate::real::Real;
use crate::tree::{NodeId, Tree};

/// The topmost interactive node under `p`, using the rectangles from the
/// last layout pass.
///
/// Paint order decides what is on top: later siblings and descendants win.
/// Hidden subtrees are skipped. A clipping node hides its whole subtree from
/// pointers outside its own rectangle. Non-interactive nodes never win, so
/// the pointer falls through them to whatever is below.
pub fn hit_test<R: Real>(tree: &Tree<R>, p: Point<R>) -> Option<NodeId> {
    let mut best = None;
    let mut stack = vec![tree.root()];
    while let Some(id) = stack.pop() {
        let (style, rect) = match (tree.style(id), tree.rect(id)) {
            (Some(s), Some(r)) => (s, r),
            _ => continue,
        };
        if !style.visible {
            continue;
        }
        let inside = rect.contains(p);
        if style.clip && !inside {
            continue;
        }
        if style.interactive && inside {
            best = Some(id);
        }
        // Reversed so the first child is visited first; the last visited hit
        // overwrites `best`, which is what makes later siblings win.
        stack.extend(tree.children(id).iter().rev().copied());
    }
    best
}

/// `true` when the UI wants the pointer at `p`. Input code that also feeds a
/// game world checks this first and skips the world when it returns `true`.
pub fn claims_pointer<R: Real>(tree: &Tree<R>, p: Point<R>) -> bool {
    hit_test(tree, p).is_some()
}
