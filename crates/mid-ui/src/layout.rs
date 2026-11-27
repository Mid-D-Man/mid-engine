// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-ui.md, section "layout.rs"
// ============================================================================
//! Anchor/margin layout with row and column flow, content-driven sizes and
//! minimum and maximum limits.

use crate::geom::{Rect, Size};
use crate::measure::{Measure, NoMeasure};
use crate::real::Real;
use crate::solve::{solve, Flat};
use crate::tree::{NodeId, Tree};

/// Computes every visible node's rectangle in viewport space, with no content
/// measurements: a leaf's content has no size, so `Fit` leaves collapse to
/// their padding. Same as [`compute_with`] and [`NoMeasure`].
pub fn compute<R: Real>(tree: &mut Tree<R>, viewport: Size<R>) {
    compute_with(tree, viewport, &NoMeasure);
}

/// Computes every visible node's rectangle in viewport space, asking
/// `measure` about the content of leaves.
///
/// The root is placed as if it were the only child of a stack the size of
/// `viewport`, so its own anchor, margin and sizing apply. Hidden nodes and
/// their subtrees are skipped and keep their previous rectangles. The tree is
/// gathered once, solved as a flat list (widths first, then heights, so a
/// wrapped text is measured at the width it is given) and written back. No
/// step recurses, so tree depth never grows the call stack. The measure is
/// asked about a node only while it has no visible children.
pub fn compute_with<R: Real, M: Measure<R> + ?Sized>(tree: &mut Tree<R>, viewport: Size<R>, measure: &M) {
    let root = tree.root();
    if tree.style(root).is_none() {
        return;
    }
    // Breadth first, so the visible children of a node are consecutive.
    let mut ids: Vec<NodeId> = vec![root];
    let mut flat = Flat { style: Vec::new(), content: Vec::new(), first: Vec::new(), count: Vec::new() };
    let mut i = 0;
    while i < ids.len() {
        let id = ids[i];
        match tree.style(id) {
            Some(s) => flat.style.push(*s),
            None => break,
        }
        flat.content.push(tree.content(id));
        flat.first.push(ids.len());
        let before = ids.len();
        for &c in tree.children(id) {
            if let Some(cs) = tree.style(c) {
                if cs.visible {
                    ids.push(c);
                }
            }
        }
        flat.count.push(ids.len() - before);
        i += 1;
    }
    let rects: Vec<Rect<R>> = solve(&flat, viewport, measure);
    for (id, rect) in ids.into_iter().zip(rects) {
        tree.set_rect(id, rect);
    }
}
