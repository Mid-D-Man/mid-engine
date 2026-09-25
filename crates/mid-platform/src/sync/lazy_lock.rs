// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-platform.md, section "sync/lazy_lock.rs"
// ============================================================================

//! Provides `LazyLock`.

pub use implementation::LazyLock;

#[cfg(feature = "std")]
use std::sync as implementation;

#[cfg(not(feature = "std"))]
mod implementation {
    use crate::cell::SyncUnsafeCell;
    use crate::sync::OnceLock;
    use core::fmt;
    use core::ops::Deref;

    /// `no_std` fallback implementation of `LazyLock` from the standard
    /// library.
    ///
    /// Real `std::sync::LazyLock` stores its initializer and its result in
    /// the same memory (a hand-rolled union, swapped via raw pointer writes)
    /// to avoid paying for both an `Option<F>` and an `Option<T>` at once.
    /// This type does not replicate that: it builds directly on top of
    /// [`OnceLock`] (`docs/mid-platform.md`'s own Phase 2 plan names this
    /// exact composition -- "builds directly on top of OnceLock once that
    /// exists") plus this crate's own [`SyncUnsafeCell`](crate::cell::SyncUnsafeCell)
    /// (Phase 1, already built and tested) to hold the initializer until
    /// it's consumed. No new unsafe algorithm gets invented for this type at
    /// all -- every piece of synchronization it needs, `OnceLock::get_or_init`
    /// already provides.
    pub struct LazyLock<T, F = fn() -> T> {
        cell: OnceLock<T>,
        // Holds the initializer until `force` consumes it. Reachable from
        // `&self` (a `Deref` call only ever gets `&self`, never `&mut
        // self`), through the same kind of "I can prove exclusive access
        // for one specific operation, without a lock" argument `SyncCell`
        // itself exists for -- see the SAFETY comment on `force` below for
        // exactly why the `.take()` this cell allows can never race.
        init: SyncUnsafeCell<Option<F>>,
    }

    // SAFETY (Send/Sync bounds): the initializer closure `F` has to be
    // `Send` since whichever thread wins the `OnceLock` race is the one that
    // actually calls it, not necessarily the thread that constructed this
    // `LazyLock`. The resulting `T` needs the same `Send + Sync` bound
    // `OnceLock<T>` itself needs to be `Sync`, for the same reason stated on
    // that type.
    unsafe impl<T: Send + Sync, F: Send> Sync for LazyLock<T, F> {}
    unsafe impl<T: Send, F: Send> Send for LazyLock<T, F> {}

    impl<T, F> LazyLock<T, F> {
        /// Creates a new lazy value with the given initializing function.
        pub const fn new(f: F) -> LazyLock<T, F> {
            Self {
                cell: OnceLock::new(),
                init: SyncUnsafeCell::new(Some(f)),
            }
        }
    }

    impl<T, F: FnOnce() -> T> LazyLock<T, F> {
        /// Forces evaluation of this lazy value and returns a reference to
        /// the result.
        ///
        /// This is equivalent to the `Deref` implementation, but is
        /// explicit and doesn't require an implicit reborrow.
        pub fn force(this: &LazyLock<T, F>) -> &T {
            this.cell.get_or_init(|| {
                // SAFETY: `OnceLock::get_or_init` guarantees the closure
                // passed to it runs on at most one thread, at most once, for
                // the lifetime of `this.cell` -- its own `init_lock`
                // serializes every caller down to a single winner before any
                // of them can reach this closure body. So whichever thread
                // is running this closure right now is the only thread that
                // will ever touch `this.init` through `force`, which makes
                // this raw-pointer access exclusive in practice even though
                // it goes through `&self`.
                let f = unsafe { &mut *this.init.get() }
                    .take()
                    .expect("LazyLock initializer already consumed");
                f()
            })
        }
    }

    impl<T, F> Deref for LazyLock<T, F>
    where
        F: FnOnce() -> T,
    {
        type Target = T;

        fn deref(&self) -> &T {
            LazyLock::force(self)
        }
    }

    impl<T: fmt::Debug, F> fmt::Debug for LazyLock<T, F> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            let mut d = f.debug_tuple("LazyLock");
            match self.cell.get() {
                Some(v) => d.field(v),
                None => d.field(&format_args!("<uninit>")),
            };
            d.finish()
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn deref_runs_the_initializer_exactly_once() {
            use core::cell::Cell;

            let calls = Cell::new(0u32);
            let lazy: LazyLock<u32, _> = LazyLock::new(|| {
                calls.set(calls.get() + 1);
                42
            });

            assert_eq!(*lazy, 42);
            assert_eq!(*lazy, 42);
            assert_eq!(calls.get(), 1);
        }

        #[test]
        fn force_and_deref_agree() {
            let lazy: LazyLock<u32, _> = LazyLock::new(|| 7);
            assert_eq!(*LazyLock::force(&lazy), 7);
            assert_eq!(*lazy, 7);
        }

        #[test]
        fn many_real_threads_racing_deref_agree_on_one_result() {
            extern crate std;
            use std::sync::Arc;
            use std::sync::atomic::{AtomicU32, Ordering};
            use std::thread;

            let calls = Arc::new(AtomicU32::new(0));
            let calls_for_init = Arc::clone(&calls);
            let lazy: Arc<LazyLock<u32, _>> = Arc::new(LazyLock::new(move || {
                calls_for_init.fetch_add(1, Ordering::SeqCst);
                99
            }));

            let handles: std::vec::Vec<_> = (0..8)
                .map(|_| {
                    let lazy = Arc::clone(&lazy);
                    thread::spawn(move || **lazy)
                })
                .collect();

            for h in handles {
                assert_eq!(h.join().unwrap(), 99);
            }
            assert_eq!(calls.load(Ordering::SeqCst), 1);
        }
    }
}
