// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-ui.md, section "style.rs"
// ============================================================================
//! Per-node layout and paint properties.

use crate::geom::Insets;
use crate::real::Real;
use mid_math::Color32;

/// Where a node sits inside the free space its parent gives it.
///
/// In a [`Flow::Stack`] parent the anchor positions the node on both axes.
/// In a [`Flow::Row`] or [`Flow::Column`] parent it only positions the node
/// on the cross axis (vertical for a row, horizontal for a column).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Anchor {
    /// Top-left corner.
    #[default]
    TopLeft,
    /// Top edge, centered horizontally.
    Top,
    /// Top-right corner.
    TopRight,
    /// Left edge, centered vertically.
    Left,
    /// Centered on both axes.
    Center,
    /// Right edge, centered vertically.
    Right,
    /// Bottom-left corner.
    BottomLeft,
    /// Bottom edge, centered horizontally.
    Bottom,
    /// Bottom-right corner.
    BottomRight,
}

impl Anchor {
    /// Horizontal and vertical placement as fractions of the free space:
    /// `0` pins to the left/top, `0.5` centers, `1` pins to the right/bottom.
    pub fn factors<R: Real>(self) -> (R, R) {
        let (z, h, o) = (R::ZERO, R::HALF, R::ONE);
        match self {
            Anchor::TopLeft => (z, z),
            Anchor::Top => (h, z),
            Anchor::TopRight => (o, z),
            Anchor::Left => (z, h),
            Anchor::Center => (h, h),
            Anchor::Right => (o, h),
            Anchor::BottomLeft => (z, o),
            Anchor::Bottom => (h, o),
            Anchor::BottomRight => (o, o),
        }
    }
}

/// How a node arranges its children.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Flow {
    /// Children overlap; each is placed independently by its own anchor.
    #[default]
    Stack,
    /// Children are laid out left to right.
    Row,
    /// Children are laid out top to bottom.
    Column,
}

/// How a node's width or height is decided.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Sizing<R: Real = f32> {
    /// An exact size. Negative values are treated as zero.
    Fixed(R),
    /// Take the available space. On a row's or column's main axis the
    /// available space left after fixed-size and fit siblings is shared
    /// between `Fill` siblings in proportion to their weights, with each
    /// share held inside the node's minimum and maximum; on every other axis
    /// the weight is ignored and the node takes all the available space.
    Fill(R),
    /// Size to the content: the widest line of a leaf's text, the children of
    /// a container, plus padding. When the space is short the node gives up
    /// room down to its minimum content width (a height never shrinks below
    /// its content) and no further.
    Fit,
}

impl<R: Real> Sizing<R> {
    /// `Fill` with weight one.
    #[inline]
    pub fn fill() -> Self { Sizing::Fill(R::ONE) }
}

/// Layout and paint properties of one node.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Style<R: Real = f32> {
    /// Placement inside the parent's free space.
    pub anchor: Anchor,
    /// Width rule.
    pub width: Sizing<R>,
    /// Height rule.
    pub height: Sizing<R>,
    /// Space kept clear around this node, inside its parent's content area.
    pub margin: Insets<R>,
    /// Space kept clear between this node's edge and its children.
    pub padding: Insets<R>,
    /// Space between consecutive children of a row or column.
    pub gap: R,
    /// How this node arranges its children.
    pub flow: Flow,
    /// Hidden nodes take no space and neither they nor their subtree are
    /// laid out, hit-tested or drawn.
    pub visible: bool,
    /// Whether this node itself can be the target of a pointer hit.
    /// Non-interactive nodes let the pointer fall through to what is below.
    pub interactive: bool,
    /// Whether children are clipped to this node's rectangle, for drawing
    /// and for hit testing.
    pub clip: bool,
    /// Background fill. Fully transparent draws nothing.
    pub background: Color32,
    /// Corner radius of the background fill.
    pub radius: R,
    /// Smallest width, whatever the width rule says. Wins over `max_width`.
    pub min_width: R,
    /// Smallest height, whatever the height rule says. Wins over `max_height`.
    pub min_height: R,
    /// Largest width; `None` for no limit.
    pub max_width: Option<R>,
    /// Largest height; `None` for no limit.
    pub max_height: Option<R>,
    /// Width divided by height. With `height: Fit` the height becomes the
    /// width over this ratio; with `width: Fit` and a fixed height the width
    /// becomes the height times it. Zero, negative and non-finite values are
    /// ignored.
    pub aspect: Option<R>,
}

impl<R: Real> Default for Style<R> {
    fn default() -> Self {
        Self {
            anchor: Anchor::TopLeft,
            width: Sizing::fill(),
            height: Sizing::fill(),
            margin: Insets::ZERO,
            padding: Insets::ZERO,
            gap: R::ZERO,
            flow: Flow::Stack,
            visible: true,
            interactive: false,
            clip: false,
            background: Color32::TRANSPARENT,
            radius: R::ZERO,
            min_width: R::ZERO,
            min_height: R::ZERO,
            max_width: None,
            max_height: None,
            aspect: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anchor_factors_cover_the_grid() {
        let all: [(Anchor, (f64, f64)); 9] = [
            (Anchor::TopLeft, (0.0, 0.0)),
            (Anchor::Top, (0.5, 0.0)),
            (Anchor::TopRight, (1.0, 0.0)),
            (Anchor::Left, (0.0, 0.5)),
            (Anchor::Center, (0.5, 0.5)),
            (Anchor::Right, (1.0, 0.5)),
            (Anchor::BottomLeft, (0.0, 1.0)),
            (Anchor::Bottom, (0.5, 1.0)),
            (Anchor::BottomRight, (1.0, 1.0)),
        ];
        for (a, want) in all {
            assert_eq!(a.factors::<f32>(), (want.0 as f32, want.1 as f32));
            assert_eq!(a.factors::<f64>(), want);
        }
    }

    #[test]
    fn default_style_fills_and_is_inert() {
        let s = Style::<f32>::default();
        assert_eq!(s.width, Sizing::Fill(1.0));
        assert_eq!(s.height, Sizing::Fill(1.0));
        assert!(s.visible && !s.interactive && !s.clip);
        assert_eq!(s.flow, Flow::Stack);
        assert!(s.background.is_transparent());
        assert_eq!((s.min_width, s.min_height), (0.0, 0.0));
        assert_eq!((s.max_width, s.max_height, s.aspect), (None, None, None));
    }
}
