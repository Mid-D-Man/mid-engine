// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-platform.md, section "sync/rwlock.rs"
// ============================================================================

//! Provides `RwLock`, `RwLockReadGuard`, `RwLockWriteGuard`.

pub use implementation::{RwLock, RwLockReadGuard, RwLockWriteGuard};

#[cfg(feature = "std")]
use std::sync as implementation;

#[cfg(not(feature = "std"))]
mod implementation {
    use crate::sync::{LockResult, TryLockError, TryLockResult};
    use core::cell::UnsafeCell;
    use core::fmt;
    use core::ops::{Deref, DerefMut};
    use core::sync::atomic::{AtomicUsize, Ordering};

    /// `no_std` fallback implementation of `RwLock` from the standard library.
    ///
    /// A single `AtomicUsize` packs the whole state: bit 0 is the `WRITER`
    /// flag, every reader adds `READER` (`2`) on top of that. This is the
    /// same real shape the `spin` crate's own `RwLock` uses (source read
    /// directly, `spin` 0.10.0's `rwlock.rs`, from crates.io -- not ported
    /// from memory), minus its `RwLockUpgradableGuard`/upgrade mechanism:
    /// this crate's Phase 2 scope is `read`/`write`/`try_read`/`try_write`
    /// only (`docs/mid-platform.md`, "Build order"), so the extra `UPGRADED`
    /// bit and the upgrade/downgrade methods it exists for are left out
    /// rather than carried along unused -- a simpler correct design was
    /// prioritized over a more feature-complete port, same reasoning
    /// `sync::mutex` already applied when it took `mid_alloc::sync::SpinLock`'s
    /// algorithm rather than `spin::Mutex`'s own generic-relax-strategy form.
    ///
    /// Two correctness details kept from the real algorithm despite the
    /// simplification, both because getting either wrong is a real bug, not
    /// a style choice:
    /// - `try_read` increments the reader count *before* checking the
    ///   `WRITER` bit, then backs the increment out on failure. A racing
    ///   `try_read` can therefore transiently bump the reader count while a
    ///   writer holds the lock, before undoing it -- which is exactly why
    ///   the next point matters.
    /// - `RwLockWriteGuard::drop` clears the `WRITER` bit with
    ///   `fetch_and(!WRITER, ..)`, never a blind `store(0, ..)`. A blind
    ///   store would race the transient bump above: if a losing `try_read`'s
    ///   `fetch_add` lands after the writer's store but its own compensating
    ///   `fetch_sub` hasn't landed yet, a `store(0)` would get stomped back
    ///   to a stale reader count by that pending `fetch_sub`, corrupting the
    ///   state. `fetch_and` only ever clears the one bit it names, so it's
    ///   safe regardless of what a racing reader increment/decrement pair is
    ///   doing to the other bits at the same time.
    ///
    /// Unfair to writers under continuous read pressure, same disclosed
    /// trade-off `spin::RwLock`'s own doc comment states for itself -- there
    /// is no fairness mechanism here (no upgradeable-guard, no
    /// writer-priority flag) to alleviate it, matching the "favor the
    /// simplest correct design" call this type's own design note in
    /// `docs/mid-platform.md` asked for. Worth a real fairness pass if
    /// mid-engine ever has a workload that hits this in practice -- not
    /// speculatively built now.
    pub struct RwLock<T: ?Sized> {
        state: AtomicUsize,
        data: UnsafeCell<T>,
    }

    const WRITER: usize = 1;
    const READER: usize = 1 << 1;
    // Matches `spin::rwlock`'s own cap exactly (`usize::MAX / READER / 2`) --
    // an arbitrary but cheap guard against a reader count ever growing large
    // enough to corrupt the `WRITER` bit, checked on every successful
    // `try_read` rather than gated behind `debug_assert!`, since the check
    // itself costs one comparison and this project would rather fail loudly
    // than silently miscompute under an extreme, unanticipated reader count.
    const MAX_READERS: usize = usize::MAX / READER / 2;

    // Same unsafe impls `std::sync::RwLock` and `spin::RwLock` both carry:
    // `Send` whenever `T` is (moving the lock to another thread is fine
    // regardless of concurrent access), `Sync` only when `T` is also `Sync`
    // -- unlike `Mutex`, `RwLock` hands out concurrent `&T` to multiple
    // readers at once, so `T` itself has to tolerate that.
    unsafe impl<T: ?Sized + Send> Send for RwLock<T> {}
    unsafe impl<T: ?Sized + Send + Sync> Sync for RwLock<T> {}

    impl<T> RwLock<T> {
        /// Creates a new instance of an `RwLock<T>` which is unlocked.
        pub const fn new(t: T) -> RwLock<T> {
            Self {
                state: AtomicUsize::new(0),
                data: UnsafeCell::new(t),
            }
        }
    }

    impl<T: ?Sized> RwLock<T> {
        /// Locks this `RwLock` with shared read access, blocking (by
        /// spinning) the current thread until it can be acquired.
        pub fn read(&self) -> LockResult<RwLockReadGuard<'_, T>> {
            loop {
                if let Ok(guard) = self.try_read() {
                    return Ok(guard);
                }
                // Cheap non-exclusive wait, same reasoning as `sync::mutex`'s
                // own `lock()`: don't retry the RMW `fetch_add` while a
                // writer is visibly still holding the lock.
                while self.state.load(Ordering::Relaxed) & WRITER != 0 {
                    core::hint::spin_loop();
                }
            }
        }

        /// Attempts to acquire this `RwLock` with shared read access.
        pub fn try_read(&self) -> TryLockResult<RwLockReadGuard<'_, T>> {
            let prev = self.state.fetch_add(READER, Ordering::Acquire);

            if prev > MAX_READERS * READER {
                self.state.fetch_sub(READER, Ordering::Relaxed);
                panic!("mid_platform::sync::RwLock: too many concurrent readers");
            }

            if prev & WRITER != 0 {
                // A writer holds (or is racing to acquire) the lock -- undo
                // the speculative increment above.
                self.state.fetch_sub(READER, Ordering::Relaxed);
                Err(TryLockError::WouldBlock)
            } else {
                Ok(RwLockReadGuard { lock: &self.state, data: self.data.get() })
            }
        }

        /// Locks this `RwLock` with exclusive write access, blocking (by
        /// spinning) the current thread until it can be acquired.
        pub fn write(&self) -> LockResult<RwLockWriteGuard<'_, T>> {
            loop {
                if let Ok(guard) = self.try_write() {
                    return Ok(guard);
                }
                while self.state.load(Ordering::Relaxed) != 0 {
                    core::hint::spin_loop();
                }
            }
        }

        /// Attempts to lock this `RwLock` with exclusive write access.
        pub fn try_write(&self) -> TryLockResult<RwLockWriteGuard<'_, T>> {
            // Only succeeds from the fully-unlocked state (`0`): no readers,
            // no writer. `compare_exchange_weak`, matching `sync::mutex`'s
            // own `try_lock` -- allowed to spuriously fail even when free,
            // fine since both `write()`/`lock()` just retry.
            self.state
                .compare_exchange_weak(0, WRITER, Ordering::Acquire, Ordering::Relaxed)
                .ok()
                .map(|_| RwLockWriteGuard { lock: self })
                .ok_or(TryLockError::WouldBlock)
        }

        /// Determines whether the lock is poisoned. Always `false` -- see
        /// this type's own doc comment.
        pub fn is_poisoned(&self) -> bool {
            false
        }

        /// Clears the poisoned state from a lock. No-op -- see this type's
        /// own doc comment.
        pub fn clear_poison(&self) {}

        /// Consumes this `RwLock`, returning the underlying data.
        pub fn into_inner(self) -> LockResult<T>
        where
            T: Sized,
        {
            Ok(self.data.into_inner())
        }

        /// Returns a mutable reference to the underlying data.
        pub fn get_mut(&mut self) -> LockResult<&mut T> {
            Ok(self.data.get_mut())
        }
    }

    impl<T> From<T> for RwLock<T> {
        fn from(t: T) -> Self {
            RwLock::new(t)
        }
    }

    impl<T: Default> Default for RwLock<T> {
        fn default() -> RwLock<T> {
            RwLock::new(Default::default())
        }
    }

    impl<T: ?Sized + fmt::Debug> fmt::Debug for RwLock<T> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            let mut d = f.debug_struct("RwLock");
            match self.try_read() {
                Ok(guard) => {
                    d.field("data", &&*guard);
                }
                Err(TryLockError::Poisoned(err)) => {
                    d.field("data", &&**err.get_ref());
                }
                Err(TryLockError::WouldBlock) => {
                    d.field("data", &format_args!("<locked>"));
                }
            }
            d.field("poisoned", &false);
            d.finish_non_exhaustive()
        }
    }

    /// Grants shared read access to an [`RwLock`]'s data. Releases that
    /// share of the lock when dropped.
    pub struct RwLockReadGuard<'a, T: ?Sized> {
        lock: &'a AtomicUsize,
        data: *const T,
    }

    // SAFETY (Send/Sync bounds): a read guard is just a shared reference in
    // disguise -- safe to send/share across threads under exactly the same
    // `T: Sync` bound a plain `&T` would need. Matches `spin::RwLockReadGuard`'s
    // own bounds exactly (verified against its real source, not assumed).
    unsafe impl<T: ?Sized + Sync> Send for RwLockReadGuard<'_, T> {}
    unsafe impl<T: ?Sized + Sync> Sync for RwLockReadGuard<'_, T> {}

    impl<'a, T: ?Sized> Deref for RwLockReadGuard<'a, T> {
        type Target = T;

        fn deref(&self) -> &T {
            // SAFETY: holding this guard is proof no writer holds the lock
            // (the `WRITER` bit was observed clear when this guard's reader
            // slot was reserved, and a writer can only acquire once every
            // reader slot -- including this one -- has released).
            unsafe { &*self.data }
        }
    }

    impl<'a, T: ?Sized> Drop for RwLockReadGuard<'a, T> {
        fn drop(&mut self) {
            self.lock.fetch_sub(READER, Ordering::Release);
        }
    }

    impl<'a, T: ?Sized + fmt::Debug> fmt::Debug for RwLockReadGuard<'a, T> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            fmt::Debug::fmt(&**self, f)
        }
    }

    /// Grants exclusive write access to an [`RwLock`]'s data. Releases the
    /// lock when dropped.
    pub struct RwLockWriteGuard<'a, T: ?Sized> {
        lock: &'a RwLock<T>,
    }

    // SAFETY (Send/Sync bounds): matches `spin::RwLockWriteGuard`'s own
    // bounds exactly (verified against its real source) -- a write guard
    // behaves like `&mut T` handed to whichever thread holds it, so both
    // `Send` and `Sync` on the guard itself require `T: Send + Sync`.
    unsafe impl<T: ?Sized + Send + Sync> Send for RwLockWriteGuard<'_, T> {}
    unsafe impl<T: ?Sized + Send + Sync> Sync for RwLockWriteGuard<'_, T> {}

    impl<'a, T: ?Sized> Deref for RwLockWriteGuard<'a, T> {
        type Target = T;

        fn deref(&self) -> &T {
            // SAFETY: holding a `RwLockWriteGuard` is proof this lock has no
            // readers and no other writer -- `try_write` only succeeds from
            // the fully-unlocked (`0`) state.
            unsafe { &*self.lock.data.get() }
        }
    }

    impl<'a, T: ?Sized> DerefMut for RwLockWriteGuard<'a, T> {
        fn deref_mut(&mut self) -> &mut T {
            // SAFETY: same reasoning as `deref` above.
            unsafe { &mut *self.lock.data.get() }
        }
    }

    impl<'a, T: ?Sized> Drop for RwLockWriteGuard<'a, T> {
        fn drop(&mut self) {
            // `fetch_and`, not `store(0, ..)` -- see this module's own top
            // doc comment for exactly why a blind store would be unsound
            // here.
            self.lock.state.fetch_and(!WRITER, Ordering::Release);
        }
    }

    impl<'a, T: ?Sized + fmt::Debug> fmt::Debug for RwLockWriteGuard<'a, T> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            fmt::Debug::fmt(&**self, f)
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn write_grants_exclusive_mutable_access() {
            let l = RwLock::new(0u32);
            *l.write().unwrap() += 1;
            *l.write().unwrap() += 1;
            assert_eq!(*l.read().unwrap(), 2);
        }

        #[test]
        fn multiple_readers_can_hold_the_lock_at_once() {
            let l = RwLock::new(5u32);
            let r1 = l.read().unwrap();
            let r2 = l.read().unwrap();
            assert_eq!(*r1, 5);
            assert_eq!(*r2, 5);
            assert!(l.try_write().is_err(), "readers still held, write must not succeed");
        }

        #[test]
        fn try_write_fails_while_a_reader_is_held() {
            let l = RwLock::new(0u32);
            let r = l.read().unwrap();
            assert!(l.try_write().is_err());
            drop(r);
            assert!(l.try_write().is_ok());
        }

        #[test]
        fn try_read_fails_while_a_writer_is_held() {
            let l = RwLock::new(0u32);
            let w = l.write().unwrap();
            assert!(l.try_read().is_err(), "writer still held, read must not succeed");
            drop(w);
            assert!(l.try_read().is_ok());
        }

        #[test]
        fn try_write_fails_while_a_writer_is_held() {
            let l = RwLock::new(0u32);
            let w = l.write().unwrap();
            assert!(l.try_write().is_err());
            drop(w);
            assert!(l.try_write().is_ok());
        }

        #[test]
        fn rwlock_is_send_and_sync_when_t_is_send_and_sync() {
            fn assert_send_sync<T: Send + Sync>() {}
            assert_send_sync::<RwLock<u32>>();
        }

        #[test]
        fn is_poisoned_and_clear_poison_are_inert() {
            let l = RwLock::new(0u32);
            assert!(!l.is_poisoned());
            l.clear_poison();
            assert!(!l.is_poisoned());
        }

        #[test]
        fn many_real_threads_racing_reads_and_writes_lose_no_updates() {
            extern crate std;
            use std::sync::Arc;
            use std::thread;

            let counter = Arc::new(RwLock::new(0u64));
            let writers = 4;
            let increments_per_writer = 1_000;

            let mut handles = std::vec::Vec::new();
            for _ in 0..writers {
                let counter = Arc::clone(&counter);
                handles.push(thread::spawn(move || {
                    for _ in 0..increments_per_writer {
                        *counter.write().unwrap() += 1;
                    }
                }));
            }
            // A few concurrent readers racing the writers above -- this
            // mainly exercises that `try_read`'s speculative-increment/undo
            // pair never corrupts the writers' own view of the lock.
            for _ in 0..4 {
                let counter = Arc::clone(&counter);
                handles.push(thread::spawn(move || {
                    for _ in 0..500 {
                        let _ = *counter.read().unwrap();
                    }
                }));
            }

            for h in handles {
                h.join().unwrap();
            }

            assert_eq!(*counter.read().unwrap(), writers * increments_per_writer);
        }
    }
}
