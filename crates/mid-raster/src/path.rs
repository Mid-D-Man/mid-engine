// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-raster.md, section "path.rs"
// ============================================================================
//! Paths and their flattening into line segments.
//!
//! A [`Path`] holds `f32` pixel coordinates. Flattening converts every point to
//! fixed point first and subdivides curves with integer arithmetic only, so the
//! segments, and therefore the pixels, are the same on every target.

use crate::fixed::{div_round, from_f32};

/// One flattened line segment in fixed point: `[x0, y0, x1, y1]`.
pub type Segment = [i64; 4];

/// How far a flattened curve may stray from the true curve, in 1/256 pixel.
pub const FLATNESS: i64 = 16;

/// Most pieces one curve is cut into.
const MAX_STEPS: i64 = 128;

#[derive(Debug, Clone, Copy, PartialEq)]
enum Cmd {
    Move(f32, f32),
    Line(f32, f32),
    Quad(f32, f32, f32, f32),
    Cubic(f32, f32, f32, f32, f32, f32),
    Close,
}

/// A sequence of contours made of lines, quadratic and cubic curves.
///
/// A fill closes every open contour implicitly. A segment drawn with no current
/// point starts a contour at its end point.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Path {
    cmds: Vec<Cmd>,
}

impl Path {
    /// An empty path.
    pub fn new() -> Self {
        Self::default()
    }

    /// Starts a new contour at `(x, y)`.
    pub fn move_to(&mut self, x: f32, y: f32) -> &mut Self {
        self.cmds.push(Cmd::Move(x, y));
        self
    }

    /// A straight line to `(x, y)`.
    pub fn line_to(&mut self, x: f32, y: f32) -> &mut Self {
        self.cmds.push(Cmd::Line(x, y));
        self
    }

    /// A quadratic curve with control point `(cx, cy)` to `(x, y)`.
    pub fn quad_to(&mut self, cx: f32, cy: f32, x: f32, y: f32) -> &mut Self {
        self.cmds.push(Cmd::Quad(cx, cy, x, y));
        self
    }

    /// A cubic curve with control points `(c1x, c1y)` and `(c2x, c2y)` to `(x, y)`.
    pub fn cubic_to(&mut self, c1x: f32, c1y: f32, c2x: f32, c2y: f32, x: f32, y: f32) -> &mut Self {
        self.cmds.push(Cmd::Cubic(c1x, c1y, c2x, c2y, x, y));
        self
    }

    /// Closes the current contour with a line back to its start.
    pub fn close(&mut self) -> &mut Self {
        self.cmds.push(Cmd::Close);
        self
    }

    /// Whether the path has no commands.
    pub fn is_empty(&self) -> bool {
        self.cmds.is_empty()
    }

    /// Flattens the path into fixed-point segments, closing every contour.
    /// Returns `None` when any coordinate is not finite, so such a path draws
    /// nothing instead of drawing part of itself.
    pub fn flatten(&self) -> Option<Vec<Segment>> {
        let pt = |x: f32, y: f32| -> Option<(i64, i64)> { Some((from_f32(x)?, from_f32(y)?)) };
        let mut out: Vec<Segment> = Vec::new();
        let (mut start, mut cur) = ((0i64, 0i64), (0i64, 0i64));
        let mut open = false;

        let close = |out: &mut Vec<Segment>, cur: (i64, i64), start: (i64, i64), open: bool| {
            if open && cur != start {
                out.push([cur.0, cur.1, start.0, start.1]);
            }
        };

        for cmd in &self.cmds {
            match *cmd {
                Cmd::Move(x, y) => {
                    close(&mut out, cur, start, open);
                    start = pt(x, y)?;
                    cur = start;
                    open = true;
                }
                Cmd::Line(x, y) => {
                    let p = pt(x, y)?;
                    if !open {
                        start = p;
                        cur = p;
                        open = true;
                    } else {
                        out.push([cur.0, cur.1, p.0, p.1]);
                        cur = p;
                    }
                }
                Cmd::Quad(cx, cy, x, y) => {
                    let (c, p) = (pt(cx, cy)?, pt(x, y)?);
                    if !open {
                        start = p;
                        cur = p;
                        open = true;
                    } else {
                        flatten_quad(&mut out, cur, c, p);
                        cur = p;
                    }
                }
                Cmd::Cubic(ax, ay, bx, by, x, y) => {
                    let (a, b, p) = (pt(ax, ay)?, pt(bx, by)?, pt(x, y)?);
                    if !open {
                        start = p;
                        cur = p;
                        open = true;
                    } else {
                        flatten_cubic(&mut out, cur, a, b, p);
                        cur = p;
                    }
                }
                Cmd::Close => {
                    close(&mut out, cur, start, open);
                    cur = start;
                }
            }
        }
        close(&mut out, cur, start, open);
        Some(out)
    }
}

/// The smallest `n` in `1..=MAX_STEPS` with `n * n * den >= num`.
fn steps(num: i64, den: i64) -> i64 {
    let mut n = 1;
    while n < MAX_STEPS && n * n * den < num {
        n += 1;
    }
    n
}

fn flatten_quad(out: &mut Vec<Segment>, p0: (i64, i64), p1: (i64, i64), p2: (i64, i64)) {
    // The deviation of a quadratic cut into n equal parameter steps is at most
    // |p0 - 2 p1 + p2| / (4 n^2); the L1 norm is no smaller than the true one.
    let dd = (p0.0 - 2 * p1.0 + p2.0).abs() + (p0.1 - 2 * p1.1 + p2.1).abs();
    let n = steps(dd, 4 * FLATNESS);
    let n2 = n * n;
    let mut prev = p0;
    for k in 1..n {
        let (a, b, c) = ((n - k) * (n - k), 2 * k * (n - k), k * k);
        let x = div_round(a * p0.0 + b * p1.0 + c * p2.0, n2);
        let y = div_round(a * p0.1 + b * p1.1 + c * p2.1, n2);
        out.push([prev.0, prev.1, x, y]);
        prev = (x, y);
    }
    out.push([prev.0, prev.1, p2.0, p2.1]);
}

fn flatten_cubic(
    out: &mut Vec<Segment>,
    p0: (i64, i64),
    p1: (i64, i64),
    p2: (i64, i64),
    p3: (i64, i64),
) {
    // With m the larger L1 second difference of the control points, n equal
    // parameter steps leave a deviation of at most 3 m / (4 n^2).
    let d1 = (p0.0 - 2 * p1.0 + p2.0).abs() + (p0.1 - 2 * p1.1 + p2.1).abs();
    let d2 = (p1.0 - 2 * p2.0 + p3.0).abs() + (p1.1 - 2 * p2.1 + p3.1).abs();
    let n = steps(3 * d1.max(d2), 4 * FLATNESS);
    let n3 = n * n * n;
    let mut prev = p0;
    for k in 1..n {
        let m = n - k;
        let (a, b, c, d) = (m * m * m, 3 * m * m * k, 3 * m * k * k, k * k * k);
        let x = div_round(a * p0.0 + b * p1.0 + c * p2.0 + d * p3.0, n3);
        let y = div_round(a * p0.1 + b * p1.1 + c * p2.1 + d * p3.1, n3);
        out.push([prev.0, prev.1, x, y]);
        prev = (x, y);
    }
    out.push([prev.0, prev.1, p3.0, p3.1]);
}
