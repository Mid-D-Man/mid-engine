// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-ui.md, section "geom.rs"
// ============================================================================
//! UI-space geometry: points, sizes, rectangles and edge insets.

use crate::real::Real;
use mid_math::{DVec2, Vec2};

/// A position in UI space (y grows downward).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point<R: Real = f32> {
    /// Horizontal coordinate.
    pub x: R,
    /// Vertical coordinate.
    pub y: R,
}

impl<R: Real> Point<R> {
    /// The origin.
    pub const ZERO: Self = Self { x: R::ZERO, y: R::ZERO };

    /// Creates a point.
    #[inline]
    pub const fn new(x: R, y: R) -> Self { Self { x, y } }
}

/// A width/height pair.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Size<R: Real = f32> {
    /// Horizontal extent.
    pub w: R,
    /// Vertical extent.
    pub h: R,
}

impl<R: Real> Size<R> {
    /// Zero width and height.
    pub const ZERO: Self = Self { w: R::ZERO, h: R::ZERO };

    /// Creates a size.
    #[inline]
    pub const fn new(w: R, h: R) -> Self { Self { w, h } }
}

/// Per-edge distances, used for margins and padding.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Insets<R: Real = f32> {
    /// Left edge.
    pub left: R,
    /// Top edge.
    pub top: R,
    /// Right edge.
    pub right: R,
    /// Bottom edge.
    pub bottom: R,
}

impl<R: Real> Insets<R> {
    /// No inset on any edge.
    pub const ZERO: Self = Self { left: R::ZERO, top: R::ZERO, right: R::ZERO, bottom: R::ZERO };

    /// Creates insets from explicit edges.
    #[inline]
    pub const fn new(left: R, top: R, right: R, bottom: R) -> Self {
        Self { left, top, right, bottom }
    }

    /// The same distance on all four edges.
    #[inline]
    pub fn all(v: R) -> Self { Self::new(v, v, v, v) }

    /// `h` on the left and right edges, `v` on the top and bottom edges.
    #[inline]
    pub fn symmetric(h: R, v: R) -> Self { Self::new(h, v, h, v) }

    /// Sum of the left and right edges.
    #[inline]
    pub fn horizontal(self) -> R { self.left + self.right }

    /// Sum of the top and bottom edges.
    #[inline]
    pub fn vertical(self) -> R { self.top + self.bottom }
}

/// An axis-aligned rectangle stored as origin plus extent.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rect<R: Real = f32> {
    /// Left edge.
    pub x: R,
    /// Top edge.
    pub y: R,
    /// Width.
    pub w: R,
    /// Height.
    pub h: R,
}

impl<R: Real> Rect<R> {
    /// A zero-size rectangle at the origin.
    pub const ZERO: Self = Self { x: R::ZERO, y: R::ZERO, w: R::ZERO, h: R::ZERO };

    /// Creates a rectangle from origin and extent.
    #[inline]
    pub const fn new(x: R, y: R, w: R, h: R) -> Self { Self { x, y, w, h } }

    /// Right edge (`x + w`).
    #[inline]
    pub fn right(self) -> R { self.x + self.w }

    /// Bottom edge (`y + h`).
    #[inline]
    pub fn bottom(self) -> R { self.y + self.h }

    /// Center point.
    #[inline]
    pub fn center(self) -> Point<R> {
        Point::new(self.x + self.w * R::HALF, self.y + self.h * R::HALF)
    }

    /// Half-open containment: the left and top edges are inside, the right
    /// and bottom edges are not. A zero-size rectangle contains nothing.
    #[inline]
    pub fn contains(self, p: Point<R>) -> bool {
        p.x >= self.x && p.x < self.right() && p.y >= self.y && p.y < self.bottom()
    }

    /// Overlap of two rectangles. When they do not overlap the result has
    /// zero width and/or height.
    pub fn intersect(self, other: Self) -> Self {
        let x0 = self.x.max(other.x);
        let y0 = self.y.max(other.y);
        let x1 = self.right().min(other.right());
        let y1 = self.bottom().min(other.bottom());
        Self::new(x0, y0, (x1 - x0).non_neg(), (y1 - y0).non_neg())
    }

    /// Shrinks the rectangle by `i` on each edge. Width and height are
    /// clamped at zero rather than going negative.
    pub fn inset(self, i: Insets<R>) -> Self {
        Self::new(
            self.x + i.left,
            self.y + i.top,
            (self.w - i.horizontal()).non_neg(),
            (self.h - i.vertical()).non_neg(),
        )
    }
}

impl From<Vec2> for Point<f32> {
    #[inline]
    fn from(v: Vec2) -> Self { Self::new(v.x, v.y) }
}

impl From<Point<f32>> for Vec2 {
    #[inline]
    fn from(p: Point<f32>) -> Self { Vec2::new(p.x, p.y) }
}

impl From<DVec2> for Point<f64> {
    #[inline]
    fn from(v: DVec2) -> Self { Self::new(v.x, v.y) }
}

impl From<Point<f64>> for DVec2 {
    #[inline]
    fn from(p: Point<f64>) -> Self { DVec2::new(p.x, p.y) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contains_is_half_open() {
        let r = Rect::new(10.0_f32, 20.0, 30.0, 40.0);
        assert!(r.contains(Point::new(10.0, 20.0)));
        assert!(r.contains(Point::new(39.9, 59.9)));
        assert!(!r.contains(Point::new(40.0, 30.0)));
        assert!(!r.contains(Point::new(20.0, 60.0)));
        assert!(!r.contains(Point::new(9.9, 30.0)));
    }

    #[test]
    fn zero_size_rect_contains_nothing() {
        let r = Rect::new(5.0_f64, 5.0, 0.0, 10.0);
        assert!(!r.contains(Point::new(5.0, 5.0)));
    }

    #[test]
    fn intersect_overlap_and_disjoint() {
        let a = Rect::new(0.0_f32, 0.0, 10.0, 10.0);
        let b = Rect::new(5.0, 5.0, 10.0, 10.0);
        assert_eq!(a.intersect(b), Rect::new(5.0, 5.0, 5.0, 5.0));
        let far = Rect::new(20.0, 20.0, 5.0, 5.0);
        let none = a.intersect(far);
        assert_eq!((none.w, none.h), (0.0, 0.0));
    }

    #[test]
    fn inset_shrinks_and_clamps() {
        let r = Rect::new(0.0_f32, 0.0, 100.0, 50.0);
        assert_eq!(r.inset(Insets::all(10.0)), Rect::new(10.0, 10.0, 80.0, 30.0));
        let tiny = Rect::new(0.0_f32, 0.0, 10.0, 10.0).inset(Insets::all(20.0));
        assert_eq!((tiny.w, tiny.h), (0.0, 0.0));
    }

    #[test]
    fn insets_helpers() {
        let i = Insets::symmetric(3.0_f32, 4.0);
        assert_eq!((i.left, i.top, i.right, i.bottom), (3.0, 4.0, 3.0, 4.0));
        assert_eq!((i.horizontal(), i.vertical()), (6.0, 8.0));
    }

    #[test]
    fn center_and_edges() {
        let r = Rect::new(10.0_f64, 20.0, 30.0, 40.0);
        assert_eq!((r.right(), r.bottom()), (40.0, 60.0));
        assert_eq!(r.center(), Point::new(25.0, 40.0));
    }

    #[test]
    fn mid_math_conversions_round_trip() {
        let p = Point::<f32>::from(Vec2::new(1.5, 2.5));
        assert_eq!(p, Point::new(1.5, 2.5));
        let v = Vec2::from(p);
        assert_eq!((v.x, v.y), (1.5, 2.5));
        let d = DVec2::from(Point::<f64>::new(3.0, 4.0));
        assert_eq!((d.x, d.y), (3.0, 4.0));
    }
}
