// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-platform.md, section "sync/mutex.rs"
// ============================================================================

//! Provides `Mutex` and `MutexGuard`.

pub use implementation::{Mutex, MutexGuard};

#[cfg(feature = "std")]
use std::sync as implementation;

#[cfg(not(feature = "std"))]
mod implementation {
    use crate::sync::{LockResult, TryLockError, TryLockResult};
    use core::cell::UnsafeCell;
    use core::fmt;
    use core::ops::{Deref, DerefMut};
    use core::sync::atomic::{AtomicBool, Ordering};

    /// `no_std` fallback implementation of `Mutex` from the standard library.
    ///
    /// A spinlock, not an OS-backed mutex — there is no OS to ask for a real wait
    /// here. This is an independent copy of the same algorithm
    /// `mid_alloc::sync::SpinLock` already validated under real multi-threaded
    /// stress tests (`crates/mid-alloc/src/sync.rs`), not a Cargo dependency on
    /// that crate — see `docs/mid-platform.md`, "Why not just depend on
    /// mid-alloc for SpinLock." "Test, then test-and-set": attempt the real
    /// atomic compare-exchange first, and only on contention fall back to
    /// spinning on a plain `Relaxed` load (cheaper than repeatedly retrying the
    /// exclusive compare-exchange, which would otherwise hammer the cache line
    /// with read-modify-write traffic while the lock is genuinely held
    /// elsewhere).
    ///
    /// Never actually poisons: a panic while holding the guard just unwinds past
    /// `Drop` and releases the lock normally, same as `std::sync::Mutex` when its
    /// own poisoning is bypassed. `is_poisoned`/`clear_poison` exist to keep this
    /// type a drop-in replacement for `std::sync::Mutex` call sites, not because
    /// poisoning is actually tracked.
    pub struct Mutex<T: ?Sized> {
        locked: AtomicBool,
        data: UnsafeCell<T>,
    }

    // Same unsafe impls `std::sync::Mutex` and `mid_alloc::sync::SpinLock` both
    // carry: safe to move to another thread whenever `T` is, safe to share
    // across threads whenever `T` is `Send` -- `Sync` is not required of `T`,
    // since the lock itself is what provides the exclusion a bare `T: Sync`
    // bound would otherwise have to.
    unsafe impl<T: ?Sized + Send> Send for Mutex<T> {}
    unsafe impl<T: ?Sized + Send> Sync for Mutex<T> {}

    impl<T> Mutex<T> {
        /// Creates a new mutex in an unlocked state ready for use.
        pub const fn new(t: T) -> Self {
            Self {
                locked: AtomicBool::new(false),
                data: UnsafeCell::new(t),
            }
        }
    }

    impl<T: ?Sized> Mutex<T> {
        /// Acquires the mutex, blocking (by spinning) the current thread until it
        /// is able to do so.
        pub fn lock(&self) -> LockResult<MutexGuard<'_, T>> {
            loop {
                if let Ok(guard) = self.try_lock() {
                    return Ok(guard);
                }
                // Cheap, non-exclusive wait: a plain `Relaxed` load doesn't need
                // to win cache-line ownership the way the compare-exchange below
                // does, so spinning here first avoids flooding the bus with
                // expensive RMW traffic while the lock is genuinely held by
                // someone else.
                while self.locked.load(Ordering::Relaxed) {
                    core::hint::spin_loop();
                }
            }
        }

        /// Attempts to acquire this lock without blocking.
        pub fn try_lock(&self) -> TryLockResult<MutexGuard<'_, T>> {
            // `compare_exchange_weak`: allowed to spuriously fail even when the
            // lock is free, which is fine (and can be cheaper on some platforms)
            // since `lock` above just retries in that case anyway.
            self.locked
                .compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
                .ok()
                .map(|_| MutexGuard { lock: self })
                .ok_or(TryLockError::WouldBlock)
        }

        /// Determines whether the mutex is poisoned. Always `false` — see this
        /// type's own doc comment.
        pub fn is_poisoned(&self) -> bool {
            false
        }

        /// Clears the poisoned state from a mutex. No-op — see this type's own
        /// doc comment.
        pub fn clear_poison(&self) {}

        /// Consumes this mutex, returning the underlying data.
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

    impl<T> From<T> for Mutex<T> {
        fn from(t: T) -> Self {
            Mutex::new(t)
        }
    }

    impl<T: Default> Default for Mutex<T> {
        fn default() -> Mutex<T> {
            Mutex::new(Default::default())
        }
    }

    impl<T: ?Sized + fmt::Debug> fmt::Debug for Mutex<T> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            let mut d = f.debug_struct("Mutex");
            match self.try_lock() {
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

    /// Grants exclusive access to a [`Mutex`]'s data. Releases the lock when
    /// dropped.
    pub struct MutexGuard<'a, T: ?Sized> {
        lock: &'a Mutex<T>,
    }

    impl<'a, T: ?Sized> Deref for MutexGuard<'a, T> {
        type Target = T;

        fn deref(&self) -> &T {
            // SAFETY: holding a `MutexGuard` is proof this lock is held by no one
            // else -- `try_lock`/`lock` are the only ways to construct one, and
            // both require winning the compare-exchange above first.
            unsafe { &*self.lock.data.get() }
        }
    }

    impl<'a, T: ?Sized> DerefMut for MutexGuard<'a, T> {
        fn deref_mut(&mut self) -> &mut T {
            // SAFETY: same reasoning as `deref` above; `&mut self` here
            // additionally rules out another live `&T` from this same guard
            // existing simultaneously.
            unsafe { &mut *self.lock.data.get() }
        }
    }

    impl<'a, T: ?Sized> Drop for MutexGuard<'a, T> {
        fn drop(&mut self) {
            self.lock.locked.store(false, Ordering::Release);
        }
    }

    impl<'a, T: ?Sized + fmt::Debug> fmt::Debug for MutexGuard<'a, T> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            fmt::Debug::fmt(&**self, f)
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn lock_grants_exclusive_mutable_access() {
            let l = Mutex::new(0u32);
            *l.lock().unwrap() += 1;
            *l.lock().unwrap() += 1;
            assert_eq!(*l.lock().unwrap(), 2);
        }

        #[test]
        fn try_lock_fails_while_a_guard_is_still_held() {
            let l = Mutex::new(0u32);
            let guard = l.lock().unwrap();
            assert!(l.try_lock().is_err(), "already held, must not succeed");
            drop(guard);
            assert!(l.try_lock().is_ok(), "released, must succeed now");
        }

        #[test]
        fn mutex_is_send_and_sync_when_t_is_send() {
            // Compile-time check, no_std-friendly -- same idiom
            // mid-alloc::sync's own SpinLock test uses.
            fn assert_send_sync<T: Send + Sync>() {}
            assert_send_sync::<Mutex<u32>>();
        }

        #[test]
        fn is_poisoned_and_clear_poison_are_inert() {
            let l = Mutex::new(0u32);
            assert!(!l.is_poisoned());
            l.clear_poison();
            assert!(!l.is_poisoned());
        }
    }
}
