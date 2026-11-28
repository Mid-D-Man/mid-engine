// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-raster.md, section "clip.rs"
// ============================================================================
//! Integer pixel rectangles and the axis-aligned clip stack.

use crate::fixed::{self, ONE};

/// A half-open rectangle of whole pixels: `x0 <= x < x1`, `y0 <= y < y1`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IRect {
    /// Left edge, inclusive.
    pub x0: i32,
    /// Top edge, inclusive.
    pub y0: i32,
    /// Right edge, exclusive.
    pub x1: i32,
    /// Bottom edge, exclusive.
    pub y1: i32,
}

impl IRect {
    /// A rectangle from its edges.
    pub const fn new(x0: i32, y0: i32, x1: i32, y1: i32) -> Self {
        Self { x0, y0, x1, y1 }
    }

    /// Whether it covers no pixel.
    pub const fn is_empty(&self) -> bool {
        self.x1 <= self.x0 || self.y1 <= self.y0
    }

    /// The part both rectangles share; empty when they do not meet.
    pub fn intersect(&self, o: &IRect) -> IRect {
        IRect {
            x0: self.x0.max(o.x0),
            y0: self.y0.max(o.y0),
            x1: self.x1.min(o.x1),
            y1: self.y1.min(o.y1),
        }
    }

    /// Width in pixels, zero when empty.
    pub fn width(&self) -> u32 {
        (self.x1 - self.x0).max(0) as u32
    }

    /// Height in pixels, zero when empty.
    pub fn height(&self) -> u32 {
        (self.y1 - self.y0).max(0) as u32
    }
}

/// Rounds a fixed-point coordinate to the nearest pixel edge, ties up.
fn snap(fx: i64) -> i32 {
    fixed::floor_div(fx + ONE / 2, ONE) as i32
}

/// The stack of clip rectangles. The bottom entry is the surface itself and is
/// never popped. A pushed clip is intersected with the one below it, and its
/// edges snap to whole pixels, so a clip never produces partial coverage.
#[derive(Debug, Clone)]
pub struct ClipStack {
    stack: Vec<IRect>,
}

impl ClipStack {
    /// A stack whose bottom entry is `bounds`.
    pub fn new(bounds: IRect) -> Self {
        Self { stack: vec![bounds] }
    }

    /// The clip in force.
    pub fn current(&self) -> IRect {
        *self.stack.last().expect("the bottom entry is never popped")
    }

    /// How many clips are open above the bottom entry.
    pub fn depth(&self) -> usize {
        self.stack.len() - 1
    }

    /// Opens a clip from fixed-point edges: each snaps to the nearest pixel
    /// edge and the result is intersected with the current clip. `None` edges
    /// (a non-finite coordinate) open an empty clip, which hides everything
    /// until its pop, so the pushes and pops stay paired.
    pub fn push(&mut self, edges: Option<(i64, i64, i64, i64)>) {
        let cur = self.current();
        let next = match edges {
            Some((x0, y0, x1, y1)) => {
                cur.intersect(&IRect::new(snap(x0), snap(y0), snap(x1), snap(y1)))
            }
            None => IRect::new(cur.x0, cur.y0, cur.x0, cur.y0),
        };
        self.stack.push(next);
    }

    /// Closes the newest clip. Returns `false` when none is open.
    pub fn pop(&mut self) -> bool {
        if self.stack.len() > 1 {
            self.stack.pop();
            true
        } else {
            false
        }
    }

    /// Closes every open clip and returns how many there were.
    pub fn close_all(&mut self) -> usize {
        let open = self.depth();
        self.stack.truncate(1);
        open
    }
}
