// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-ui.md, section "measure.rs"
// ============================================================================
//! What the solver asks a host about the content of a leaf.
//!
//! The solver never sees a font. A leaf carries a [`ContentKey`], and the
//! host's [`Measure`] turns that key into three numbers: the width of the
//! content on one line, the narrowest width it can be given, and its height at
//! a chosen width. `docs/mid-ui-contracts.md` (section 4) states the rules
//! for these numbers; the text crate's `TextMeasure` is the usual source.

use crate::real::Real;

/// An opaque handle a host attaches to a leaf so that its [`Measure`] knows
/// what to measure. The solver stores and passes it back without looking at it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct ContentKey(pub u64);

/// Sizes of leaf content, asked by the solver while it lays a tree out.
///
/// Each answer must be finite and not negative; the solver replaces any
/// other value with zero. Answers must not depend on anything but the
/// arguments and the host's own font data, so that solving twice gives the
/// same rectangles.
pub trait Measure<R: Real> {
    /// Width of the content when it is not wrapped (the widest line when only
    /// hard breaks apply).
    fn line_width(&self, key: ContentKey) -> R;

    /// Width of the widest piece that cannot be broken. A node is never given
    /// less than this unless a maximum size says so.
    fn min_content_width(&self, key: ContentKey) -> R;

    /// Height of the content wrapped at `max_width`.
    fn wrapped_height(&self, key: ContentKey, max_width: R) -> R;
}

/// A measure that reports nothing: every leaf has no content size. Used by
/// [`layout::compute`](crate::layout::compute), which keeps its old behavior.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoMeasure;

impl<R: Real> Measure<R> for NoMeasure {
    #[inline]
    fn line_width(&self, _key: ContentKey) -> R { R::ZERO }
    #[inline]
    fn min_content_width(&self, _key: ContentKey) -> R { R::ZERO }
    #[inline]
    fn wrapped_height(&self, _key: ContentKey, _max_width: R) -> R { R::ZERO }
}
