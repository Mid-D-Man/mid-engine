// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-alloc.md, section "tracking.rs"
// ============================================================================
//! [`Tracked<A>`], wrapping any [`RawAlloc`] with lock-free counters for
//! every allocation and deallocation that passes through it. Combines
//! two real, independently read sources rather than inventing a shape
//! fresh:
//!
//! - The hook points themselves match
//!   `foonathan::memory::tracked_allocator<Tracker, RawAllocator>`'s
//!   real interface (`tracking.hpp`, source read): call a hook after a
//!   successful allocation, and one before a deallocation, collapsed
//!   from that trait's four hooks (`on_node_allocation`/
//!   `on_node_deallocation`/`on_array_allocation`/
//!   `on_array_deallocation` — `RawAlloc` has no separate array path)
//!   onto `RawAlloc`'s own two methods.
//! - The counters themselves, and their exact update sequence, are
//!   `mod_alloc::ModAlloc`'s real, already-shipped design (source
//!   read, `mod-alloc` 1.0.0 from crates.io, MSRV 1.75 — this
//!   project's own toolchain floor): six `AtomicU64` fields updated
//!   with `Ordering::Relaxed`, peaks tracked via `fetch_max` rather
//!   than a compare-exchange loop. Ported directly rather than
//!   reinvented, since it was already right.
//!
//! Using atomics instead of a mutex is the one real reason this module
//! didn't have to wait on the `no_std` spinlock a
//! `ThreadSafeAllocator`-style wrapper still needs (see
//! `docs/mid-alloc.md`'s "Zig re-survey" section). **Stated plainly
//! rather than implied:** this does not make `Tracked<A>` usable from
//! multiple threads by itself. None of this crate's current `RawAlloc`
//! implementors are `Sync` (`StackAllocator` uses bare `Cell`s), so
//! `Tracked<A>` isn't either — the atomics here are correctness
//! infrastructure for whenever a `Sync` `RawAlloc` exists, not a
//! working concurrent tracker today. Also inherited as-is from
//! `mod_alloc`'s real design, not newly introduced by this port: the
//! peak counters update via two separate atomic operations
//! (`fetch_add` then `fetch_max`), not one combined atomic step, so
//! under real concurrent access the recorded peak can very rarely
//! under-report by one interleaved update. A known, accepted tradeoff
//! for lock-free counters generally, not a bug specific to this port.

use crate::raw_alloc::RawAlloc;
use core::ptr::NonNull;
use core::sync::atomic::{AtomicU64, Ordering};

/// A snapshot of [`Tracked`]'s counters at one point in time. Matches
/// `mod_alloc::AllocStats`'s real field set (source read) with one
/// real addition stated rather than left unmarked: `dealloc_count`,
/// which `mod_alloc` does not track separately (its `alloc_count`
/// only counts allocations and growing reallocations) but
/// `foonathan::tracked_allocator`'s own hook interface tracks both
/// directions symmetrically, and `RawAlloc::try_dealloc_raw` already
/// reports success/failure per call, so counting it costs nothing.
/// Every other field is `mod_alloc`'s own, not renamed or reshaped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct AllocStats {
    /// Number of successful allocations recorded.
    pub alloc_count: u64,
    /// Number of successful deallocations recorded.
    pub dealloc_count: u64,
    /// Total bytes ever allocated (deallocated bytes are not
    /// subtracted back out of this one — see `current_bytes`).
    pub total_bytes: u64,
    /// The highest `current_bytes` has ever been.
    pub peak_bytes: u64,
    /// Currently-allocated bytes (allocations minus deallocations).
    pub current_bytes: u64,
    /// Currently-live allocation count.
    pub live_count: u64,
    /// The highest `live_count` has ever been.
    pub peak_live_count: u64,
}

impl AllocStats {
    /// The difference between this snapshot and an earlier `baseline`
    /// one, matching `mod_alloc::Profiler::stop`'s real math: counting
    /// fields (`alloc_count`, `dealloc_count`, `total_bytes`,
    /// `current_bytes`) become real deltas via `saturating_sub`; the
    /// two peak fields carry over as this snapshot's own absolute
    /// values, since "peak" has no meaningful delta.
    pub fn since(&self, baseline: AllocStats) -> AllocStats {
        AllocStats {
            alloc_count: self.alloc_count.saturating_sub(baseline.alloc_count),
            dealloc_count: self.dealloc_count.saturating_sub(baseline.dealloc_count),
            total_bytes: self.total_bytes.saturating_sub(baseline.total_bytes),
            current_bytes: self.current_bytes.saturating_sub(baseline.current_bytes),
            peak_bytes: self.peak_bytes,
            live_count: self.live_count.saturating_sub(baseline.live_count),
            peak_live_count: self.peak_live_count,
        }
    }
}

/// Wraps a [`RawAlloc`] and records [`AllocStats`] for everything that
/// passes through it. See this module's own doc comment for exactly
/// which real sources this design came from and what it does and does
/// not guarantee.
pub struct Tracked<A> {
    inner: A,
    alloc_count: AtomicU64,
    dealloc_count: AtomicU64,
    total_bytes: AtomicU64,
    peak_bytes: AtomicU64,
    current_bytes: AtomicU64,
    live_count: AtomicU64,
    peak_live_count: AtomicU64,
}

impl<A> Tracked<A> {
    /// Wraps `inner`. All counters start at zero.
    pub fn new(inner: A) -> Self {
        Self {
            inner,
            alloc_count: AtomicU64::new(0),
            dealloc_count: AtomicU64::new(0),
            total_bytes: AtomicU64::new(0),
            peak_bytes: AtomicU64::new(0),
            current_bytes: AtomicU64::new(0),
            live_count: AtomicU64::new(0),
            peak_live_count: AtomicU64::new(0),
        }
    }

    /// The wrapped allocator.
    pub fn inner(&self) -> &A {
        &self.inner
    }

    /// Reads every counter. Each is read independently with `Relaxed`
    /// ordering, so the result is a coherent best-effort view, not a
    /// single atomic moment in time -- matches `mod_alloc::ModAlloc::
    /// snapshot`'s own documented caveat exactly.
    pub fn snapshot(&self) -> AllocStats {
        AllocStats {
            alloc_count: self.alloc_count.load(Ordering::Relaxed),
            dealloc_count: self.dealloc_count.load(Ordering::Relaxed),
            total_bytes: self.total_bytes.load(Ordering::Relaxed),
            peak_bytes: self.peak_bytes.load(Ordering::Relaxed),
            current_bytes: self.current_bytes.load(Ordering::Relaxed),
            live_count: self.live_count.load(Ordering::Relaxed),
            peak_live_count: self.peak_live_count.load(Ordering::Relaxed),
        }
    }

    /// Resets every counter to zero. Matches `mod_alloc::ModAlloc::
    /// reset`'s own documented caveat: calling this while allocations
    /// are still live can make `current_bytes` wrap on a later
    /// deallocation, since the live count this allocator doesn't know
    /// about anymore still tries to subtract itself back out.
    pub fn reset(&self) {
        self.alloc_count.store(0, Ordering::Relaxed);
        self.dealloc_count.store(0, Ordering::Relaxed);
        self.total_bytes.store(0, Ordering::Relaxed);
        self.peak_bytes.store(0, Ordering::Relaxed);
        self.current_bytes.store(0, Ordering::Relaxed);
        self.live_count.store(0, Ordering::Relaxed);
        self.peak_live_count.store(0, Ordering::Relaxed);
    }

    #[inline]
    fn record_alloc(&self, size: u64) {
        self.alloc_count.fetch_add(1, Ordering::Relaxed);
        self.total_bytes.fetch_add(size, Ordering::Relaxed);
        let new_current = self.current_bytes.fetch_add(size, Ordering::Relaxed) + size;
        self.peak_bytes.fetch_max(new_current, Ordering::Relaxed);
        let new_live = self.live_count.fetch_add(1, Ordering::Relaxed) + 1;
        self.peak_live_count.fetch_max(new_live, Ordering::Relaxed);
    }

    #[inline]
    fn record_dealloc(&self, size: u64) {
        self.dealloc_count.fetch_add(1, Ordering::Relaxed);
        self.current_bytes.fetch_sub(size, Ordering::Relaxed);
        self.live_count.fetch_sub(1, Ordering::Relaxed);
    }
}

impl<A: RawAlloc> RawAlloc for Tracked<A> {
    #[inline]
    fn try_alloc_raw(&self, size: usize, align: usize) -> Option<NonNull<u8>> {
        let ptr = self.inner.try_alloc_raw(size, align)?;
        self.record_alloc(size as u64);
        Some(ptr)
    }

    #[inline]
    unsafe fn try_dealloc_raw(&self, ptr: NonNull<u8>, size: usize, align: usize) -> bool {
        // SAFETY: forwarding the caller's own contract straight
        // through to `inner`.
        let freed = self.inner.try_dealloc_raw(ptr, size, align);
        if freed {
            self.record_dealloc(size as u64);
        }
        freed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::raw_alloc::HeapAlloc;

    #[test]
    fn snapshot_starts_at_zero() {
        let t = Tracked::new(HeapAlloc);
        assert_eq!(t.snapshot(), AllocStats::default());
    }

    #[test]
    fn alloc_and_dealloc_update_the_matching_counters() {
        let t = Tracked::new(HeapAlloc);
        let ptr = t.try_alloc_raw(64, 8).expect("a small heap allocation should not fail");

        let after_alloc = t.snapshot();
        assert_eq!(after_alloc.alloc_count, 1);
        assert_eq!(after_alloc.total_bytes, 64);
        assert_eq!(after_alloc.current_bytes, 64);
        assert_eq!(after_alloc.live_count, 1);
        assert_eq!(after_alloc.peak_bytes, 64);
        assert_eq!(after_alloc.peak_live_count, 1);
        assert_eq!(after_alloc.dealloc_count, 0, "not freed yet");

        unsafe {
            assert!(t.try_dealloc_raw(ptr, 64, 8));
        }
        let after_dealloc = t.snapshot();
        assert_eq!(after_dealloc.dealloc_count, 1);
        assert_eq!(after_dealloc.current_bytes, 0);
        assert_eq!(after_dealloc.live_count, 0);
        assert_eq!(
            after_dealloc.peak_bytes, 64,
            "peak must not drop back down just because current usage did"
        );
        assert_eq!(after_dealloc.total_bytes, 64, "total is cumulative, dealloc doesn't undo it");
    }

    #[test]
    fn peak_tracks_the_high_water_mark_not_the_latest_value() {
        let t = Tracked::new(HeapAlloc);
        let a = t.try_alloc_raw(100, 8).unwrap();
        let b = t.try_alloc_raw(50, 8).unwrap();
        unsafe {
            assert!(t.try_dealloc_raw(a, 100, 8));
        }
        // current_bytes is now 50, but peak must still remember 150.
        let snap = t.snapshot();
        assert_eq!(snap.current_bytes, 50);
        assert_eq!(snap.peak_bytes, 150);
        assert_eq!(snap.peak_live_count, 2);
        assert_eq!(snap.live_count, 1);
        unsafe {
            assert!(t.try_dealloc_raw(b, 50, 8));
        }
    }

    #[test]
    fn reset_zeroes_every_counter() {
        let t = Tracked::new(HeapAlloc);
        let ptr = t.try_alloc_raw(8, 8).unwrap();
        t.reset();
        assert_eq!(t.snapshot(), AllocStats::default());
        // Real cleanup so this test doesn't leak, even though the
        // counters no longer know about this allocation.
        unsafe {
            assert!(t.inner().try_dealloc_raw(ptr, 8, 8));
        }
    }

    #[test]
    fn since_computes_a_real_delta_and_keeps_peak_absolute() {
        let t = Tracked::new(HeapAlloc);
        let a = t.try_alloc_raw(10, 1).unwrap();
        let baseline = t.snapshot();

        let b = t.try_alloc_raw(20, 1).unwrap();
        let delta = t.snapshot().since(baseline);

        assert_eq!(delta.alloc_count, 1);
        assert_eq!(delta.total_bytes, 20);
        assert_eq!(delta.current_bytes, 20);
        assert_eq!(
            delta.peak_bytes, 30,
            "peak is the snapshot's own absolute value, not a delta"
        );

        unsafe {
            assert!(t.try_dealloc_raw(a, 10, 1));
            assert!(t.try_dealloc_raw(b, 20, 1));
        }
    }

    #[test]
    fn a_failed_dealloc_does_not_get_counted() {
        // NullAlloc-wrapped: try_dealloc_raw always reports false, so
        // no dealloc should ever be recorded regardless of how many
        // times it's called.
        use crate::raw_alloc::NullAlloc;
        let t = Tracked::new(NullAlloc);
        let fake = core::ptr::NonNull::<u8>::dangling();
        unsafe {
            assert!(!t.try_dealloc_raw(fake, 8, 8));
        }
        assert_eq!(t.snapshot().dealloc_count, 0);
    }
}
