// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-ui.md, section "layout.rs"
// ============================================================================
//! Anchor/margin layout with row and column flow.

use crate::geom::{Rect, Size};
use crate::real::Real;
use crate::style::{Flow, Sizing, Style};
use crate::tree::{NodeId, Tree};

/// Computes every visible node's rectangle in viewport space.
///
/// The root is placed as if it were the only child of a stack the size of
/// `viewport`, so its own anchor, margin and sizing apply. Hidden nodes and
/// their subtrees are skipped and keep their previous rectangles. The pass
/// is a single top-down walk with an explicit stack, so tree depth never
/// grows the call stack.
pub fn compute<R: Real>(tree: &mut Tree<R>, viewport: Size<R>) {
    let root = tree.root();
    let root_style = match tree.style(root) {
        Some(s) => *s,
        None => return,
    };
    let area = Rect::new(R::ZERO, R::ZERO, viewport.w, viewport.h);
    tree.set_rect(root, stack_place(&root_style, area));

    let mut work = vec![root];
    let mut kids: Vec<(NodeId, Style<R>)> = Vec::new();
    let mut placed: Vec<(NodeId, Rect<R>)> = Vec::new();
    while let Some(id) = work.pop() {
        let (style, rect) = match (tree.style(id), tree.rect(id)) {
            (Some(s), Some(r)) => (*s, r),
            _ => continue,
        };
        kids.clear();
        placed.clear();
        for &c in tree.children(id) {
            if let Some(cs) = tree.style(c) {
                if cs.visible {
                    kids.push((c, *cs));
                }
            }
        }
        let content = rect.inset(style.padding);
        match style.flow {
            Flow::Stack => {
                for (c, cs) in &kids {
                    placed.push((*c, stack_place(cs, content)));
                }
            }
            Flow::Row => linear(&kids, content, style.gap, Axis::X, &mut placed),
            Flow::Column => linear(&kids, content, style.gap, Axis::Y, &mut placed),
        }
        for &(c, r) in &placed {
            tree.set_rect(c, r);
            work.push(c);
        }
    }
}

fn resolve<R: Real>(sizing: Sizing<R>, available: R) -> R {
    match sizing {
        Sizing::Fixed(v) => v.non_neg(),
        Sizing::Fill(_) => available.non_neg(),
    }
}

// Places one node inside `content`, honouring its margin, sizing and anchor.
fn stack_place<R: Real>(s: &Style<R>, content: Rect<R>) -> Rect<R> {
    let area = content.inset(s.margin);
    let w = resolve(s.width, area.w);
    let h = resolve(s.height, area.h);
    let (fx, fy) = s.anchor.factors::<R>();
    Rect::new(area.x + fx * (area.w - w), area.y + fy * (area.h - h), w, h)
}

#[derive(Clone, Copy)]
enum Axis {
    X,
    Y,
}

// One child's numbers split into main-axis and cross-axis terms, so the row
// and column cases share a single code path.
struct Lane<R: Real> {
    main_before: R,
    main_after: R,
    cross_before: R,
    cross_after: R,
    main: Sizing<R>,
    cross: Sizing<R>,
    cross_factor: R,
}

fn lane<R: Real>(s: &Style<R>, axis: Axis) -> Lane<R> {
    let (fx, fy) = s.anchor.factors::<R>();
    match axis {
        Axis::X => Lane {
            main_before: s.margin.left,
            main_after: s.margin.right,
            cross_before: s.margin.top,
            cross_after: s.margin.bottom,
            main: s.width,
            cross: s.height,
            cross_factor: fy,
        },
        Axis::Y => Lane {
            main_before: s.margin.top,
            main_after: s.margin.bottom,
            cross_before: s.margin.left,
            cross_after: s.margin.right,
            main: s.height,
            cross: s.width,
            cross_factor: fx,
        },
    }
}

fn linear<R: Real>(
    kids: &[(NodeId, Style<R>)],
    content: Rect<R>,
    gap: R,
    axis: Axis,
    out: &mut Vec<(NodeId, Rect<R>)>,
) {
    if kids.is_empty() {
        return;
    }
    let (main_start, main_len, cross_start, cross_len) = match axis {
        Axis::X => (content.x, content.w, content.y, content.h),
        Axis::Y => (content.y, content.h, content.x, content.w),
    };

    // Pass 1: everything that does not flex (fixed sizes, margins, gaps) and
    // the total flex weight. Weights that are zero, negative or NaN are
    // ignored, so such a node ends up with no main-axis space.
    let mut fixed = R::ZERO;
    let mut weights = R::ZERO;
    for (_, s) in kids {
        let l = lane(s, axis);
        fixed = fixed + l.main_before + l.main_after;
        match l.main {
            Sizing::Fixed(v) => fixed = fixed + v.non_neg(),
            Sizing::Fill(w) => {
                if w > R::ZERO {
                    weights = weights + w;
                }
            }
        }
    }
    let gaps = gap * R::from_f64((kids.len() - 1) as f64);
    let leftover = (main_len - fixed - gaps).non_neg();

    // Pass 2: place children in order along the main axis.
    let mut cursor = main_start;
    for (id, s) in kids {
        let l = lane(s, axis);
        cursor = cursor + l.main_before;
        let main_size = match l.main {
            Sizing::Fixed(v) => v.non_neg(),
            Sizing::Fill(w) => {
                if w > R::ZERO && weights > R::ZERO { leftover * w / weights } else { R::ZERO }
            }
        };
        let cross_avail = (cross_len - l.cross_before - l.cross_after).non_neg();
        let cross_size = resolve(l.cross, cross_avail);
        let cross_pos = cross_start + l.cross_before + l.cross_factor * (cross_avail - cross_size);
        let rect = match axis {
            Axis::X => Rect::new(cursor, cross_pos, main_size, cross_size),
            Axis::Y => Rect::new(cross_pos, cursor, cross_size, main_size),
        };
        out.push((*id, rect));
        cursor = cursor + main_size + l.main_after + gap;
    }
}
