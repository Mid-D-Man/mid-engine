// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-ui.md, section "lazy.rs"
// ============================================================================
//! Which rows of a long, uniform list are in view.
//!
//! A hierarchy panel can hold tens of thousands of rows of one height. Only the
//! rows in view need nodes. [`LazyRows`] answers which range that is, and how
//! tall the whole list is for a scroller, so the host can create exactly those
//! rows and keep the rest as numbers.

use core::ops::Range;

use crate::real::Real;

/// A list of `count` rows of one height with a gap between neighbors.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LazyRows<R: Real = f32> {
    /// Number of rows.
    pub count: usize,
    /// Height of one row.
    pub row_height: R,
    /// Space between two neighboring rows.
    pub gap: R,
}

impl<R: Real> LazyRows<R> {
    /// Creates a list description.
    pub const fn new(count: usize, row_height: R, gap: R) -> Self {
        Self { count, row_height, gap }
    }

    /// Distance from the top of one row to the top of the next.
    fn pitch(&self) -> R {
        self.row_height.non_neg() + self.gap.non_neg()
    }

    /// Height of the whole list: all rows and the gaps between them.
    pub fn extent(&self) -> R {
        if self.count == 0 {
            return R::ZERO;
        }
        let n = R::from_f64(self.count as f64);
        self.row_height.non_neg() * n + self.gap.non_neg() * R::from_f64((self.count - 1) as f64)
    }

    /// Top of row `index` measured from the top of the list.
    pub fn row_top(&self, index: usize) -> R {
        self.pitch() * R::from_f64(index as f64)
    }

    /// The rows that touch a window `view` tall that starts `scroll` below the
    /// top of the list. The scroll position is clamped into the list, so an
    /// overscroll still shows the last rows. A row counts when any part of it
    /// is inside the window; a gap alone does not.
    pub fn visible(&self, scroll: R, view: R) -> Range<usize> {
        let pitch = self.pitch();
        let usable = self.count > 0 && pitch > R::ZERO && pitch.is_finite();
        if !usable {
            return 0..0;
        }
        let view = if view.is_finite() { view.non_neg() } else { R::ZERO };
        let max_scroll = (self.extent() - view).non_neg();
        let mut scroll = if scroll.is_finite() { scroll.non_neg() } else { R::ZERO };
        if scroll > max_scroll {
            scroll = max_scroll;
        }
        // `as usize` truncates toward zero, which is the floor for the
        // non-negative values here, and saturates for huge ones.
        let mut first = (scroll / pitch).to_f64() as usize;
        if self.row_top(first) + self.row_height.non_neg() <= scroll {
            first += 1;
        }
        if view <= R::ZERO {
            let at = first.min(self.count);
            return at..at;
        }
        let edge = ((scroll + view) / pitch).to_f64();
        let mut end = edge as usize;
        if (end as f64) < edge {
            end += 1;
        }
        let first = first.min(self.count);
        let end = end.min(self.count).max(first);
        first..end
    }
}
