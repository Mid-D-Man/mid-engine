// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-ecs.md, section "Parallel iteration (`batching.rs`,
// `QueryBatch`, `Query::par_iter`)"
// ============================================================================
//! How a parallel query is cut into pieces of work.
//!
//! A parallel query does not hand rows to threads one at a time. It cuts
//! the matched archetypes into *batches*: each batch is a list of row
//! windows (a row range of one archetype) that one thread iterates start
//! to finish. [`BatchingStrategy`] says how big a batch should be;
//! [`plan_batches`] does the cutting. The cutting is a pure function of
//! the archetype lengths, so it is tested without a world or a thread.
//!
//! The shape is bevy's (`bevy_ecs::batching::BatchingStrategy`, and the
//! queueing in `QueryState::par_fold_init_unchecked_manual`): an archetype
//! at least one batch long is split into batch-sized windows, shorter
//! ones are queued together until the queue reaches a batch, so a world
//! of many tiny archetypes does not become many tiny tasks.

use std::ops::Range;

/// How a parallel query is divided into batches.
///
/// The default splits the largest matched archetype into about one batch
/// per thread. Raise [`Self::batches_per_thread`] for work whose cost
/// varies a lot per row (more, smaller batches balance better); set
/// [`Self::min_batch_size`] when one row is so cheap that a task per few
/// rows would cost more than the rows.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BatchingStrategy {
    /// The smallest and largest batch (in rows) the strategy may pick.
    /// An empty range (`fixed`) means "exactly `start` rows".
    pub batch_size_limits: Range<usize>,
    /// How many batches per thread to aim for when sizing from the
    /// largest archetype. At least 1.
    pub batches_per_thread: usize,
}

impl Default for BatchingStrategy {
    fn default() -> Self {
        Self::new()
    }
}

impl BatchingStrategy {
    /// One batch per thread for the largest archetype, no size limits.
    pub const fn new() -> Self {
        Self {
            batch_size_limits: 1..usize::MAX,
            batches_per_thread: 1,
        }
    }

    /// Batches of exactly `batch_size` rows (the last of an archetype may
    /// be shorter). Ignores thread count and archetype size.
    pub const fn fixed(batch_size: usize) -> Self {
        Self {
            batch_size_limits: batch_size..batch_size,
            batches_per_thread: 1,
        }
    }

    /// Batches of at least `batch_size` rows.
    pub const fn min_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size_limits.start = batch_size;
        self
    }

    /// Batches of at most `batch_size` rows.
    pub const fn max_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size_limits.end = batch_size;
        self
    }

    /// Aim for this many batches per thread.
    ///
    /// # Panics
    /// If `batches_per_thread` is zero.
    pub fn batches_per_thread(mut self, batches_per_thread: usize) -> Self {
        assert!(
            batches_per_thread > 0,
            "The number of batches per thread must be non-zero."
        );
        self.batches_per_thread = batches_per_thread;
        self
    }

    /// The batch size for a query whose largest archetype has
    /// `max_items()` rows, run on `thread_count` threads. `max_items` is
    /// only called when the size is not fixed.
    ///
    /// # Panics
    /// If `thread_count` is zero and the size is not fixed.
    #[inline]
    pub fn calc_batch_size(&self, max_items: impl FnOnce() -> usize, thread_count: usize) -> usize {
        if self.batch_size_limits.is_empty() {
            return self.batch_size_limits.start;
        }
        assert!(
            thread_count > 0,
            "Attempted to run parallel iteration with zero threads"
        );
        let batches = thread_count.saturating_mul(self.batches_per_thread);
        let batch_size = max_items().div_ceil(batches);
        batch_size.clamp(self.batch_size_limits.start, self.batch_size_limits.end)
    }
}

/// A row window: rows `rows` of the archetype at position `archetype` in
/// the list `plan_batches` was given.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct WindowSpec {
    pub(crate) archetype: usize,
    pub(crate) rows: Range<usize>,
}

/// Cuts archetypes of `lens[i]` rows into batches of windows.
///
/// * An archetype of at least `batch_size` rows becomes windows of
///   `batch_size` rows (the last may be shorter), one batch each.
/// * A shorter one is queued; the queue becomes one batch as soon as it
///   holds `batch_size` rows or more, and whatever is left at the end
///   becomes the last batch.
/// * Empty archetypes produce nothing.
///
/// Every row of every archetype lands in exactly one window of exactly
/// one batch. A `batch_size` of zero is treated as one.
pub(crate) fn plan_batches(lens: &[usize], batch_size: usize) -> Vec<Vec<WindowSpec>> {
    let batch_size = batch_size.max(1);
    let mut batches: Vec<Vec<WindowSpec>> = Vec::new();
    let mut queue: Vec<WindowSpec> = Vec::new();
    let mut queued = 0usize;
    for (archetype, &len) in lens.iter().enumerate() {
        if len == 0 {
            continue;
        }
        if len >= batch_size {
            let mut start = 0;
            while start < len {
                let end = start + batch_size.min(len - start);
                batches.push(vec![WindowSpec {
                    archetype,
                    rows: start..end,
                }]);
                start = end;
            }
            continue;
        }
        queue.push(WindowSpec {
            archetype,
            rows: 0..len,
        });
        queued += len;
        if queued >= batch_size {
            batches.push(std::mem::take(&mut queue));
            queued = 0;
        }
    }
    if !queue.is_empty() {
        batches.push(queue);
    }
    batches
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every row of every archetype appears in exactly one window, and no
    /// window is empty or reaches past its archetype.
    fn assert_exact_cover(lens: &[usize], batches: &[Vec<WindowSpec>]) {
        let mut seen: Vec<Vec<u32>> = lens.iter().map(|&len| vec![0; len]).collect();
        for batch in batches {
            assert!(!batch.is_empty(), "an empty batch was produced");
            for window in batch {
                assert!(window.rows.start < window.rows.end, "empty window");
                assert!(window.rows.end <= lens[window.archetype], "window past end");
                for row in window.rows.clone() {
                    seen[window.archetype][row] += 1;
                }
            }
        }
        for (archetype, rows) in seen.iter().enumerate() {
            for (row, &count) in rows.iter().enumerate() {
                assert_eq!(count, 1, "archetype {archetype} row {row} covered {count}x");
            }
        }
    }

    #[test]
    fn exhaustive_small_cases_cover_every_row_exactly_once() {
        // Every length vector of up to 4 archetypes with lengths 0..=5,
        // against every batch size 0..=7.
        let mut lens = [0usize; 4];
        let total = 6usize.pow(4);
        for n in 0..total {
            let mut x = n;
            for slot in lens.iter_mut() {
                *slot = x % 6;
                x /= 6;
            }
            for batch_size in 0..=7 {
                let batches = plan_batches(&lens, batch_size);
                assert_exact_cover(&lens, &batches);
            }
        }
    }

    #[test]
    fn big_archetype_splits_into_batch_sized_windows() {
        let batches = plan_batches(&[10], 4);
        let windows: Vec<_> = batches
            .iter()
            .map(|b| {
                assert_eq!(b.len(), 1);
                b[0].rows.clone()
            })
            .collect();
        assert_eq!(windows, vec![0..4, 4..8, 8..10]);
    }

    #[test]
    fn exact_multiple_has_no_short_tail() {
        let batches = plan_batches(&[8], 4);
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[1][0].rows, 4..8);
    }

    #[test]
    fn archetype_exactly_one_batch_long_is_not_queued() {
        // len == batch_size takes the split path: one window of the
        // whole archetype, alone in its batch (not merged with the
        // small one before it).
        let batches = plan_batches(&[1, 4], 4);
        assert_eq!(batches.len(), 2);
        let sizes: Vec<usize> = batches.iter().map(|b| b.len()).collect();
        assert!(sizes.contains(&1));
        assert!(batches.iter().any(|b| b.len() == 1 && b[0].archetype == 1));
        assert!(batches.iter().any(|b| b.len() == 1 && b[0].archetype == 0));
    }

    #[test]
    fn small_archetypes_are_queued_into_one_batch() {
        // 1+1+1+1 reaches batch_size 4 exactly on the fourth: one batch of
        // four windows, not four batches.
        let batches = plan_batches(&[1, 1, 1, 1], 4);
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].len(), 4);
    }

    #[test]
    fn queue_flushes_when_it_reaches_batch_size_not_before() {
        // 3 then 2: after the first the queue holds 3 (< 4), after the
        // second 5 (>= 4) and flushes as ONE batch; the trailing 1 is the
        // last batch.
        let batches = plan_batches(&[3, 2, 1], 4);
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].len(), 2);
        assert_eq!(batches[1].len(), 1);
        // And a queue of 3 rows (< 4) alone is still emitted at the end.
        let batches = plan_batches(&[3], 4);
        assert_eq!(batches.len(), 1);
    }

    #[test]
    fn queue_flushes_the_moment_it_equals_batch_size() {
        // 2 + 2 hits batch_size 4 exactly: that is one full batch, and
        // the archetype after it starts the next one (it does not get
        // folded into the full queue).
        let batches = plan_batches(&[2, 2, 1], 4);
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].len(), 2);
        assert_eq!(batches[1].len(), 1);
        assert_eq!(batches[1][0].archetype, 2);
    }

    #[test]
    fn empty_archetypes_and_empty_input_produce_nothing() {
        assert!(plan_batches(&[], 4).is_empty());
        assert!(plan_batches(&[0, 0, 0], 4).is_empty());
        let batches = plan_batches(&[0, 3, 0], 4);
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0][0].archetype, 1);
    }

    #[test]
    fn zero_batch_size_is_treated_as_one() {
        let batches = plan_batches(&[3], 0);
        assert_eq!(batches.len(), 3);
    }

    #[test]
    fn huge_batch_size_does_not_overflow() {
        let batches = plan_batches(&[5, 9], usize::MAX);
        assert_exact_cover(&[5, 9], &batches);
        // Both are shorter than the batch size: one queued batch.
        assert_eq!(batches.len(), 1);
    }

    #[test]
    fn archetype_index_is_the_position_in_the_input() {
        let batches = plan_batches(&[0, 0, 7], 100);
        assert_eq!(batches[0][0].archetype, 2);
    }

    // ── BatchingStrategy ─────────────────────────────────────────────

    #[test]
    fn default_is_one_batch_per_thread_of_the_largest_archetype() {
        let s = BatchingStrategy::new();
        assert_eq!(s.calc_batch_size(|| 100, 4), 25);
        // Rounds up so there are never more batches than threads.
        assert_eq!(s.calc_batch_size(|| 101, 4), 26);
        assert_eq!(s.calc_batch_size(|| 0, 4), 1);
        assert_eq!(BatchingStrategy::default(), s);
    }

    #[test]
    fn batches_per_thread_multiplies_the_batch_count() {
        let s = BatchingStrategy::new().batches_per_thread(4);
        assert_eq!(s.calc_batch_size(|| 160, 4), 10);
    }

    #[test]
    fn limits_clamp_the_computed_size() {
        let s = BatchingStrategy::new().min_batch_size(50);
        assert_eq!(s.calc_batch_size(|| 100, 4), 50);
        let s = BatchingStrategy::new().max_batch_size(10);
        assert_eq!(s.calc_batch_size(|| 100, 4), 10);
        // Within the limits: untouched.
        let s = BatchingStrategy::new().min_batch_size(5).max_batch_size(30);
        assert_eq!(s.calc_batch_size(|| 100, 4), 25);
    }

    #[test]
    fn fixed_ignores_threads_and_never_asks_for_the_max() {
        let s = BatchingStrategy::fixed(7);
        assert_eq!(
            s.calc_batch_size(|| panic!("max_items must not be called"), 0),
            7
        );
        assert_eq!(s.calc_batch_size(|| 1000, 16), 7);
    }

    #[test]
    #[should_panic(expected = "zero threads")]
    fn zero_threads_panics_when_the_size_is_not_fixed() {
        BatchingStrategy::new().calc_batch_size(|| 10, 0);
    }

    #[test]
    #[should_panic(expected = "non-zero")]
    fn zero_batches_per_thread_panics() {
        let _ = BatchingStrategy::new().batches_per_thread(0);
    }

    #[test]
    fn thread_count_times_batches_does_not_overflow() {
        let s = BatchingStrategy::new().batches_per_thread(usize::MAX);
        // saturating: batches = usize::MAX, so ceil(10 / MAX) = 1.
        assert_eq!(s.calc_batch_size(|| 10, 8), 1);
    }
}
