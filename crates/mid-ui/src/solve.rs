// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-ui.md, section "solve.rs"
// ============================================================================
//! The size solver: a pure function from a flat list of nodes to rectangles.
//!
//! The model is a parent that proposes room and children that pick a size
//! inside it. Per axis the solver runs a bottom-up pass that finds what each
//! node wants (its natural size and the floor it can shrink to), then a
//! top-down pass that assigns sizes and positions. Width goes first, because
//! the height of wrapped text depends on the width it is given and nothing
//! about a width depends on a height, except through an aspect ratio, which
//! reads the width that is already settled.
//!
//! The nodes arrive in breadth-first order, so the visible children of a node
//! are consecutive entries and a bottom-up pass is a loop from the end. No
//! pass recurses; depth costs nothing on the call stack.

use crate::geom::{Rect, Size};
use crate::measure::{ContentKey, Measure};
use crate::real::Real;
use crate::style::{Flow, Sizing, Style};

/// The visible part of a tree, gathered breadth-first. Entry `i` has its
/// visible children at `first[i] .. first[i] + count[i]`. The root is entry 0.
pub(crate) struct Flat<R: Real> {
    pub style: Vec<Style<R>>,
    pub content: Vec<Option<ContentKey>>,
    pub first: Vec<usize>,
    pub count: Vec<usize>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Axis {
    X,
    Y,
}

fn sizing_of<R: Real>(s: &Style<R>, a: Axis) -> Sizing<R> {
    match a {
        Axis::X => s.width,
        Axis::Y => s.height,
    }
}

fn margin_of<R: Real>(s: &Style<R>, a: Axis) -> (R, R) {
    match a {
        Axis::X => (s.margin.left, s.margin.right),
        Axis::Y => (s.margin.top, s.margin.bottom),
    }
}

fn padding_of<R: Real>(s: &Style<R>, a: Axis) -> (R, R) {
    match a {
        Axis::X => (s.padding.left, s.padding.right),
        Axis::Y => (s.padding.top, s.padding.bottom),
    }
}

fn limits_of<R: Real>(s: &Style<R>, a: Axis) -> (R, Option<R>) {
    match a {
        Axis::X => (s.min_width, s.max_width),
        Axis::Y => (s.min_height, s.max_height),
    }
}

fn factor_of<R: Real>(s: &Style<R>, a: Axis) -> R {
    let (fx, fy) = s.anchor.factors::<R>();
    match a {
        Axis::X => fx,
        Axis::Y => fy,
    }
}

/// Whether `axis` is the direction the flow lays its children along.
fn is_main(flow: Flow, a: Axis) -> bool {
    matches!((flow, a), (Flow::Row, Axis::X) | (Flow::Column, Axis::Y))
}

/// Holds `v` inside `min` and `max`. The minimum wins when they conflict.
fn clamp<R: Real>(v: R, min: R, max: Option<R>) -> R {
    let v = match max {
        Some(m) => v.min(m),
        None => v,
    };
    v.max(min)
}

/// A host answer that is not finite or is negative counts as zero.
fn clean<R: Real>(v: R) -> R {
    if v.is_finite() && v >= R::ZERO { v } else { R::ZERO }
}

/// The aspect ratio when it is usable.
fn aspect_of<R: Real>(s: &Style<R>) -> Option<R> {
    s.aspect.filter(|a| a.is_finite() && *a > R::ZERO)
}

/// One axis, every node: where it starts, how long it is, what it wants and
/// the least it can shrink to.
struct Lane<R: Real> {
    pos: Vec<R>,
    len: Vec<R>,
    nat: Vec<R>,
    floor: Vec<R>,
}

impl<R: Real> Lane<R> {
    fn new(n: usize) -> Self {
        Self {
            pos: vec![R::ZERO; n],
            len: vec![R::ZERO; n],
            nat: vec![R::ZERO; n],
            floor: vec![R::ZERO; n],
        }
    }
}

/// Work arrays for laying out one container's children, reused across nodes.
struct Scratch<R: Real> {
    size: Vec<R>,
    weight: Vec<R>,
    target: Vec<R>,
    kind: Vec<Kind>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    /// A fixed size, or a flexible child that cannot flex (weight not positive).
    Rigid,
    /// Sized to its content; may shrink to its floor.
    Fit,
    /// Flexible and still taking part in the split.
    Flex,
    /// Flexible, held at a minimum or maximum.
    Held,
}

/// Solves the whole tree. The result has one rectangle per entry of `flat`.
pub(crate) fn solve<R: Real, M: Measure<R> + ?Sized>(
    flat: &Flat<R>,
    viewport: Size<R>,
    measure: &M,
) -> Vec<Rect<R>> {
    let n = flat.style.len();
    let mut cols = Lane::<R>::new(n);
    let mut rows = Lane::<R>::new(n);
    let mut scratch = Scratch { size: Vec::new(), weight: Vec::new(), target: Vec::new(), kind: Vec::new() };

    natural_widths(flat, &mut cols, measure);
    assign(flat, Axis::X, viewport.w, &mut cols, &mut scratch);
    natural_heights(flat, &cols, &mut rows, measure);
    assign(flat, Axis::Y, viewport.h, &mut rows, &mut scratch);

    (0..n).map(|i| Rect::new(cols.pos[i], rows.pos[i], cols.len[i], rows.len[i])).collect()
}

// What a container wants along `axis`, from its children's wants.
fn container_natural<R: Real>(flat: &Flat<R>, i: usize, axis: Axis, lane: &Lane<R>) -> (R, R) {
    let s = &flat.style[i];
    let (pb, pa) = padding_of(s, axis);
    let main = is_main(s.flow, axis);
    let (first, count) = (flat.first[i], flat.count[i]);
    let (mut nat, mut floor) = (R::ZERO, R::ZERO);
    for k in first..first + count {
        let ks = &flat.style[k];
        let (mb, ma) = margin_of(ks, axis);
        let (lo, hi) = limits_of(ks, axis);
        let (cn, cf) = match sizing_of(ks, axis) {
            Sizing::Fixed(v) => {
                let c = clamp(v.non_neg(), lo, hi);
                (c, c)
            }
            _ => (clamp(lane.nat[k], lo, hi), clamp(lane.floor[k], lo, hi)),
        };
        let (on, of) = (mb + cn + ma, mb + cf + ma);
        if main {
            nat = nat + on;
            floor = floor + of;
        } else {
            nat = nat.max(on);
            floor = floor.max(of);
        }
    }
    if main && count > 1 {
        let gaps = s.gap * R::from_f64((count - 1) as f64);
        nat = nat + gaps;
        floor = floor + gaps;
    }
    (nat + pb + pa, floor + pb + pa)
}

fn natural_widths<R: Real, M: Measure<R> + ?Sized>(flat: &Flat<R>, cols: &mut Lane<R>, measure: &M) {
    for i in (0..flat.style.len()).rev() {
        let s = &flat.style[i];
        let (pb, pa) = padding_of(s, Axis::X);
        let (mut nat, mut floor) = if flat.count[i] > 0 {
            container_natural(flat, i, Axis::X, cols)
        } else {
            match flat.content[i] {
                Some(k) => (
                    clean(measure.line_width(k)) + pb + pa,
                    clean(measure.min_content_width(k)) + pb + pa,
                ),
                None => (pb + pa, pb + pa),
            }
        };
        // A fit width over a fixed height with an aspect ratio is that height
        // times the ratio, whatever the content says.
        if let (Some(a), Sizing::Fit, Sizing::Fixed(v)) = (aspect_of(s), s.width, s.height) {
            nat = clamp(v.non_neg(), s.min_height, s.max_height) * a;
            floor = nat;
        }
        cols.nat[i] = nat;
        cols.floor[i] = floor;
    }
}

fn natural_heights<R: Real, M: Measure<R> + ?Sized>(
    flat: &Flat<R>,
    cols: &Lane<R>,
    rows: &mut Lane<R>,
    measure: &M,
) {
    for i in (0..flat.style.len()).rev() {
        let s = &flat.style[i];
        let (pt, pb) = padding_of(s, Axis::Y);
        let mut nat = if flat.count[i] > 0 {
            container_natural(flat, i, Axis::Y, rows).0
        } else {
            match flat.content[i] {
                Some(k) => {
                    let (pl, pr) = padding_of(s, Axis::X);
                    let inner = (cols.len[i] - (pl + pr)).non_neg();
                    clean(measure.wrapped_height(k, inner)) + pt + pb
                }
                None => pt + pb,
            }
        };
        // A fit height with an aspect ratio is the settled width over it.
        if let (Some(a), Sizing::Fit) = (aspect_of(s), s.height) {
            nat = cols.len[i] / a;
        }
        rows.nat[i] = nat;
        // A height does not shrink below what its content needs.
        rows.floor[i] = nat;
    }
}

// The size a node takes from `avail` when it is placed on its own, in a stack
// or on the cross axis of a row or column.
fn resolve<R: Real>(sizing: Sizing<R>, avail: R, nat: R, floor: R, lo: R, hi: Option<R>) -> R {
    let base = match sizing {
        Sizing::Fixed(v) => v.non_neg(),
        Sizing::Fill(_) => avail.non_neg(),
        Sizing::Fit => nat.min(avail).max(floor),
    };
    clamp(base, lo, hi)
}

// Top-down: fixes `pos` and `len` of every node on `axis`.
fn assign<R: Real>(flat: &Flat<R>, axis: Axis, view_len: R, lane: &mut Lane<R>, scratch: &mut Scratch<R>) {
    // The root is placed as the only child of a stack the size of the viewport.
    {
        let s = &flat.style[0];
        let (mb, ma) = margin_of(s, axis);
        let (lo, hi) = limits_of(s, axis);
        let avail = (view_len - (mb + ma)).non_neg();
        let size = resolve(sizing_of(s, axis), avail, lane.nat[0], lane.floor[0], lo, hi);
        lane.len[0] = size;
        lane.pos[0] = (R::ZERO + mb) + factor_of(s, axis) * (avail - size);
    }
    for i in 0..flat.style.len() {
        let count = flat.count[i];
        if count == 0 {
            continue;
        }
        let s = &flat.style[i];
        let (pb, pa) = padding_of(s, axis);
        let cstart = lane.pos[i] + pb;
        let clen = (lane.len[i] - (pb + pa)).non_neg();
        let first = flat.first[i];
        if is_main(s.flow, axis) {
            linear(flat, i, axis, cstart, clen, lane, scratch);
        } else {
            let stacked = s.flow == Flow::Stack;
            for k in first..first + count {
                let ks = &flat.style[k];
                let (mb, ma) = margin_of(ks, axis);
                let (lo, hi) = limits_of(ks, axis);
                // A stack insets by the summed margin, a row or column's cross
                // axis by one margin and then the other; the sums can differ in
                // the last bit, so each keeps its own order.
                let avail = if stacked { (clen - (mb + ma)).non_neg() } else { (clen - mb - ma).non_neg() };
                let size = resolve(sizing_of(ks, axis), avail, lane.nat[k], lane.floor[k], lo, hi);
                lane.len[k] = size;
                lane.pos[k] = (cstart + mb) + factor_of(ks, axis) * (avail - size);
            }
        }
    }
}

// Lays the children of node `i` along its flow direction.
fn linear<R: Real>(
    flat: &Flat<R>,
    i: usize,
    axis: Axis,
    cstart: R,
    clen: R,
    lane: &mut Lane<R>,
    sc: &mut Scratch<R>,
) {
    let (first, count) = (flat.first[i], flat.count[i]);
    let gap = flat.style[i].gap;

    sc.size.clear();
    sc.weight.clear();
    sc.target.clear();
    sc.kind.clear();
    sc.size.resize(count, R::ZERO);
    sc.weight.resize(count, R::ZERO);
    sc.target.resize(count, R::ZERO);
    sc.kind.resize(count, Kind::Rigid);

    // Pass 1: everything that does not flex (margins, fixed and fit sizes,
    // gaps) and the total flex weight. A weight that is zero, negative or NaN
    // cannot flex, so such a node keeps its minimum, which is zero unless set.
    let mut fixed = R::ZERO;
    for j in 0..count {
        let ks = &flat.style[first + j];
        let (mb, ma) = margin_of(ks, axis);
        let (lo, hi) = limits_of(ks, axis);
        fixed = fixed + mb + ma;
        match sizing_of(ks, axis) {
            Sizing::Fixed(v) => {
                let c = clamp(v.non_neg(), lo, hi);
                sc.size[j] = c;
                fixed = fixed + c;
            }
            Sizing::Fit => {
                let c = clamp(lane.nat[first + j], lo, hi);
                sc.size[j] = c;
                sc.kind[j] = Kind::Fit;
                fixed = fixed + c;
            }
            Sizing::Fill(w) => {
                if w > R::ZERO {
                    sc.weight[j] = w;
                    sc.kind[j] = Kind::Flex;
                } else {
                    let c = clamp(R::ZERO, lo, hi);
                    sc.size[j] = c;
                    fixed = fixed + c;
                }
            }
        }
    }
    let gaps = gap * R::from_f64((count - 1) as f64);
    let raw = (clen - fixed) - gaps;

    if raw < R::ZERO {
        // Short of room: fit children give up space down to their floors, in
        // proportion to how much each can give. Flexible children get none.
        let deficit = -raw;
        let mut room = R::ZERO;
        for j in 0..count {
            if sc.kind[j] == Kind::Fit {
                room = room + shrinkable(flat, lane, axis, first + j, sc.size[j]);
            }
        }
        for j in 0..count {
            match sc.kind[j] {
                Kind::Fit if room > R::ZERO => {
                    let k = first + j;
                    let give = shrinkable(flat, lane, axis, k, sc.size[j]);
                    sc.size[j] = if deficit >= room {
                        sc.size[j] - give
                    } else {
                        sc.size[j] - deficit * give / room
                    };
                }
                Kind::Flex => {
                    let (lo, hi) = limits_of(&flat.style[first + j], axis);
                    sc.size[j] = clamp(R::ZERO, lo, hi);
                }
                _ => {}
            }
        }
    } else {
        split_flex(flat, first, count, axis, raw, sc);
    }

    // Pass 2: place children in order along the main axis.
    let mut cursor = cstart;
    for j in 0..count {
        let k = first + j;
        let (mb, ma) = margin_of(&flat.style[k], axis);
        cursor = cursor + mb;
        lane.pos[k] = cursor;
        lane.len[k] = sc.size[j];
        cursor = cursor + sc.size[j] + ma + gap;
    }
}

// How far a fit child can shrink from `size`: down to its floor, which is held
// inside its own limits like every other size.
fn shrinkable<R: Real>(flat: &Flat<R>, lane: &Lane<R>, axis: Axis, k: usize, size: R) -> R {
    let (lo, hi) = limits_of(&flat.style[k], axis);
    (size - clamp(lane.floor[k], lo, hi)).non_neg()
}

// Shares `raw` between the flexible children by weight, holding each inside
// its minimum and maximum. A child that would leave its bounds is fixed at the
// bound and the rest is shared again among the others, until nobody is
// outside. Each round fixes at least one child, so it ends.
fn split_flex<R: Real>(flat: &Flat<R>, first: usize, count: usize, axis: Axis, raw: R, sc: &mut Scratch<R>) {
    loop {
        let (mut held, mut weights, mut any) = (R::ZERO, R::ZERO, false);
        for j in 0..count {
            match sc.kind[j] {
                Kind::Held => held = held + sc.size[j],
                Kind::Flex => {
                    weights = weights + sc.weight[j];
                    any = true;
                }
                _ => {}
            }
        }
        if !any {
            return;
        }
        let leftover = (raw - held).non_neg();
        let mut violation = R::ZERO;
        for j in 0..count {
            if sc.kind[j] == Kind::Flex {
                let (lo, hi) = limits_of(&flat.style[first + j], axis);
                let target = leftover * sc.weight[j] / weights;
                let size = clamp(target, lo, hi);
                sc.target[j] = target;
                sc.size[j] = size;
                violation = violation + (size - target);
            }
        }
        // Fix the children that went past a minimum when the total violation
        // is positive, past a maximum when it is negative, all when none.
        for j in 0..count {
            if sc.kind[j] == Kind::Flex {
                let (size, target) = (sc.size[j], sc.target[j]);
                let fix = if violation > R::ZERO {
                    size > target
                } else if violation < R::ZERO {
                    size < target
                } else {
                    true
                };
                if fix {
                    sc.kind[j] = Kind::Held;
                }
            }
        }
    }
}
