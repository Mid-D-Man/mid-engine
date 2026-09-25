// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-platform.md, section "sync/once.rs"
// ============================================================================

//! Provides `Once`, `OnceState`, `OnceLock`.

pub use implementation::{Once, OnceLock, OnceState};

#[cfg(feature = "std")]
use std::sync as implementation;

#[cfg(not(feature = "std"))]
mod implementation {
    use crate::sync::Mutex;
    use core::cell::UnsafeCell;
    use core::fmt;
    use core::mem::MaybeUninit;
    use core::panic::{RefUnwindSafe, UnwindSafe};
    use core::sync::atomic::{AtomicBool, Ordering};

    /// `no_std` fallback implementation of `OnceLock` from the standard
    /// library.
    ///
    /// Real `spin::Once` (source read directly, `spin` 0.10.0's `once.rs`,
    /// from crates.io) hand-rolls its own four-state (`Incomplete` /
    /// `Running` / `Complete` / `Panicked`) atomic state machine with a
    /// `compare_exchange`-driven wait loop. This type deliberately does not
    /// port that: `docs/mid-platform.md`'s own Phase 2 plan for this file
    /// explicitly floats building on top of `sync::mutex::Mutex` instead of
    /// a new from-scratch primitive, and with no compiler in this project's
    /// working environment to catch a subtle state-machine bug, reusing the
    /// crate's own already-tested `Mutex` for the actual mutual exclusion
    /// is the simpler and more trustworthy choice. This crate's `Mutex`
    /// never poisons either (see its own doc comment), so there is no
    /// `Panicked` state to track in the first place -- one fewer state than
    /// `spin::Once`'s real four needs modeling at all.
    ///
    /// The design actually used is the standard double-checked-locking
    /// shape, not a bare `Mutex<Option<T>>`: a separate `completed:
    /// AtomicBool`, checked with `Acquire` before ever touching the lock,
    /// keeps `get()` -- expected to be the hot path once a value exists --
    /// fully lock-free after initialization, matching `std::sync::OnceLock`'s
    /// real performance characteristic. Wrapping the data straight in
    /// `Mutex<Option<T>>` would have been simpler still, but would force
    /// every single `get()` call to spin-acquire the lock even long after
    /// initialization finished, which is the wrong trade for a type whose
    /// entire purpose is "pay the synchronization cost once."
    pub struct OnceLock<T> {
        // Guards only the *initializing* race -- who gets to actually call
        // the closure and write `data`. Once `completed` is `true`, `get()`
        // never touches this at all.
        init_lock: Mutex<()>,
        completed: AtomicBool,
        data: UnsafeCell<MaybeUninit<T>>,
    }

    // SAFETY (Send/Sync bounds): matches `std::sync::OnceLock`'s own real
    // bounds -- `Sync` requires `T: Send + Sync` (once initialized, multiple
    // threads get concurrent `&T` via `get()`'s lock-free fast path, and
    // whichever thread's closure happens to win the race hands its `T` off
    // to every other thread that reads it afterward), `Send` requires only
    // `T: Send`.
    unsafe impl<T: Send + Sync> Sync for OnceLock<T> {}
    unsafe impl<T: Send> Send for OnceLock<T> {}

    // `UnsafeCell` is not `RefUnwindSafe`/`UnwindSafe` by default; restoring
    // both here matches `std::sync::OnceLock`'s own explicit impls (its
    // interior mutability is only ever observed in a fully-initialized state
    // from `get()`'s perspective, same reasoning that lets `Mutex` and
    // `RwLock` stay usable across an `catch_unwind` boundary).
    impl<T: RefUnwindSafe + UnwindSafe> RefUnwindSafe for OnceLock<T> {}
    impl<T: UnwindSafe> UnwindSafe for OnceLock<T> {}

    impl<T> OnceLock<T> {
        /// Creates a new empty cell.
        #[must_use]
        pub const fn new() -> Self {
            Self {
                init_lock: Mutex::new(()),
                completed: AtomicBool::new(false),
                data: UnsafeCell::new(MaybeUninit::uninit()),
            }
        }

        /// Gets a reference to the underlying value, or `None` if the cell
        /// is empty. Never blocks.
        pub fn get(&self) -> Option<&T> {
            if self.completed.load(Ordering::Acquire) {
                // SAFETY: `completed == true` is only ever stored (with
                // `Release`) after `data` has been fully written in
                // `get_or_init` below, so this `Acquire` load establishes a
                // happens-before edge with that write -- the value here is
                // fully initialized and no one holds a `&mut` to it (the
                // only way to get one is `get_mut`/`into_inner`, both of
                // which take `&mut self`/`self` and so cannot alias a live
                // `&self` call to `get`).
                Some(unsafe { (*self.data.get()).assume_init_ref() })
            } else {
                None
            }
        }

        /// Gets a mutable reference to the underlying value, or `None` if
        /// the cell is empty.
        pub fn get_mut(&mut self) -> Option<&mut T> {
            if *self.completed.get_mut() {
                // SAFETY: `&mut self` proves exclusive access; `completed`
                // being `true` proves `data` was fully initialized.
                Some(unsafe { (*self.data.get()).assume_init_mut() })
            } else {
                None
            }
        }

        /// Sets the contents of this cell to `value`. Returns `Err(value)`
        /// (handing the value back) if the cell was already full.
        pub fn set(&self, value: T) -> Result<(), T> {
            let mut value = Some(value);
            self.get_or_init(|| value.take().expect("closure invoked at most once"));
            match value {
                Some(v) => Err(v),
                None => Ok(()),
            }
        }

        /// Gets the contents of the cell, initializing it with `f` if the
        /// cell was empty. Blocks (by spinning) if another thread is
        /// concurrently initializing it.
        pub fn get_or_init<F>(&self, f: F) -> &T
        where
            F: FnOnce() -> T,
        {
            // Fast path: already initialized, no locking at all.
            if let Some(v) = self.get() {
                return v;
            }

            // Slow path: race for `init_lock`. Whoever wins actually calls
            // `f`; everyone else just waits for the lock and then finds
            // `completed == true` already.
            let _guard = self.init_lock.lock().expect("this crate's Mutex never poisons");

            // Double-checked: another thread may have finished
            // initializing between the fast-path check above and actually
            // winning `init_lock`.
            if !self.completed.load(Ordering::Acquire) {
                let value = f();
                // SAFETY: `init_lock` is held, and `completed` is still
                // `false` -- no other thread can be inside this branch (any
                // racing caller is either still spinning on `init_lock`, or
                // already saw `completed == true` and returned via the fast
                // path above), so this write has no concurrent observer.
                unsafe {
                    (*self.data.get()).write(value);
                }
                self.completed.store(true, Ordering::Release);
            }

            // SAFETY: `completed` is now `true` -- either this call just set
            // it (write happened immediately above, same thread), or another
            // thread did before we acquired `init_lock` (`Mutex::lock`'s own
            // acquire semantics establish happens-before with that thread's
            // release of the lock, which happened after its `Release` store
            // to `completed`).
            unsafe { (*self.data.get()).assume_init_ref() }
        }

        /// Consumes the `OnceLock`, returning the wrapped value, or `None`
        /// if the cell was empty.
        pub fn into_inner(mut self) -> Option<T> {
            self.take()
        }

        /// Takes the value out of this `OnceLock`, moving it back to an
        /// uninitialized state.
        pub fn take(&mut self) -> Option<T> {
            if *self.completed.get_mut() {
                *self.completed.get_mut() = false;
                let data = core::mem::replace(&mut self.data, UnsafeCell::new(MaybeUninit::uninit()));
                // SAFETY: `&mut self` proves exclusive access, and
                // `completed` was `true` before the reset above, so `data`
                // held a fully-initialized `T`.
                Some(unsafe { data.into_inner().assume_init() })
            } else {
                None
            }
        }
    }

    impl<T> Default for OnceLock<T> {
        fn default() -> OnceLock<T> {
            OnceLock::new()
        }
    }

    impl<T: fmt::Debug> fmt::Debug for OnceLock<T> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            let mut d = f.debug_tuple("OnceLock");
            match self.get() {
                Some(v) => d.field(v),
                None => d.field(&format_args!("<uninit>")),
            };
            d.finish()
        }
    }

    /// `no_std` fallback implementation of `Once` from the standard library.
    /// A thin `OnceLock<()>` wrapper -- same composition real `spin`/
    /// `bevy_platform` both use, since a plain "did this run yet" flag is
    /// exactly what `OnceLock<()>` already is.
    pub struct Once {
        inner: OnceLock<()>,
    }

    impl Once {
        /// Creates a new `Once` value.
        // `#[allow]`, not `#[expect]` -- `#[expect]` needs rustc 1.81+, past
        // this crate's real 1.75 floor (unlike the bench target, nothing
        // else in this crate's own library code needs a newer toolchain,
        // and this lint doesn't need one either). Matches `std::sync::Once`,
        // which also has no `Default` impl.
        #[allow(clippy::new_without_default)]
        pub const fn new() -> Self {
            Self { inner: OnceLock::new() }
        }

        /// Performs an initialization routine once and only once. The given
        /// closure runs if this is the first call to `call_once`;
        /// otherwise it does not run at all.
        pub fn call_once<F: FnOnce()>(&self, f: F) {
            self.inner.get_or_init(f);
        }

        /// Performs the same function as [`call_once`](Self::call_once)
        /// except ignores poisoning -- a no-op distinction here, since this
        /// type never poisons in the first place.
        pub fn call_once_force<F: FnOnce(&OnceState)>(&self, f: F) {
            const STATE: OnceState = OnceState { _private: () };
            self.call_once(move || f(&STATE));
        }

        /// Returns `true` if [`call_once`](Self::call_once) has completed
        /// successfully at least once.
        pub fn is_completed(&self) -> bool {
            self.inner.get().is_some()
        }
    }

    impl RefUnwindSafe for Once {}
    impl UnwindSafe for Once {}

    impl fmt::Debug for Once {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("Once").finish_non_exhaustive()
        }
    }

    /// `no_std` fallback implementation of `OnceState` from the standard
    /// library.
    pub struct OnceState {
        _private: (),
    }

    impl OnceState {
        /// Returns `true` if the associated [`Once`] was poisoned prior to
        /// this call. Always `false` -- this crate's primitives never
        /// poison.
        pub fn is_poisoned(&self) -> bool {
            false
        }
    }

    impl fmt::Debug for OnceState {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("OnceState").field("poisoned", &self.is_poisoned()).finish()
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn get_or_init_runs_the_closure_exactly_once() {
            let cell = OnceLock::new();
            let mut calls = 0u32;
            for _ in 0..5 {
                cell.get_or_init(|| {
                    calls += 1;
                    42u32
                });
            }
            assert_eq!(calls, 1);
            assert_eq!(*cell.get().unwrap(), 42);
        }

        #[test]
        fn get_returns_none_before_init_and_some_after() {
            let cell: OnceLock<u32> = OnceLock::new();
            assert!(cell.get().is_none());
            cell.get_or_init(|| 7);
            assert_eq!(cell.get(), Some(&7));
        }

        #[test]
        fn set_fails_and_returns_the_value_once_already_set() {
            let cell = OnceLock::new();
            assert_eq!(cell.set(1u32), Ok(()));
            assert_eq!(cell.set(2u32), Err(2));
            assert_eq!(*cell.get().unwrap(), 1);
        }

        #[test]
        fn take_empties_the_cell_and_allows_reinitialization() {
            let mut cell = OnceLock::new();
            cell.get_or_init(|| 9u32);
            assert_eq!(cell.take(), Some(9));
            assert!(cell.get().is_none());
            cell.get_or_init(|| 10u32);
            assert_eq!(*cell.get().unwrap(), 10);
        }

        #[test]
        fn once_call_once_runs_exactly_once() {
            let once = Once::new();
            let mut calls = 0u32;
            for _ in 0..5 {
                once.call_once(|| calls += 1);
            }
            assert_eq!(calls, 1);
            assert!(once.is_completed());
        }

        #[test]
        fn once_state_is_never_poisoned() {
            let once = Once::new();
            once.call_once_force(|state| assert!(!state.is_poisoned()));
        }

        #[test]
        fn once_lock_is_send_and_sync_when_t_is_send_and_sync() {
            fn assert_send_sync<T: Send + Sync>() {}
            assert_send_sync::<OnceLock<u32>>();
        }

        #[test]
        fn many_real_threads_racing_get_or_init_agree_on_one_winner() {
            extern crate std;
            use std::sync::Arc;
            use std::sync::atomic::{AtomicU32, Ordering as StdOrdering};
            use std::thread;

            let cell: Arc<OnceLock<u32>> = Arc::new(OnceLock::new());
            let calls = Arc::new(AtomicU32::new(0));
            let threads = 8;

            let handles: std::vec::Vec<_> = (0..threads)
                .map(|i| {
                    let cell = Arc::clone(&cell);
                    let calls = Arc::clone(&calls);
                    thread::spawn(move || {
                        *cell.get_or_init(|| {
                            calls.fetch_add(1, StdOrdering::SeqCst);
                            i
                        })
                    })
                })
                .collect();

            let results: std::vec::Vec<u32> = handles.into_iter().map(|h| h.join().unwrap()).collect();
            assert_eq!(calls.load(StdOrdering::SeqCst), 1, "exactly one thread's closure should have run");
            let winner = results[0];
            assert!(results.iter().all(|&r| r == winner), "every thread must observe the same winning value");
        }
    }
}
