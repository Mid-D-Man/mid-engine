// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-alloc.md, section "sync.rs"
// ============================================================================
//! [`SpinLock<T>`], the `no_std` mutual-exclusion primitive every other
//! module in this crate has been waiting on, plus [`SyncAlloc<A>`],
//! which uses it to turn any [`RawAlloc`] into one safe to share
//! across threads -- the same real shape as Zig's
//! `std.heap.ThreadSafeAllocator` (source read, `ThreadSafeAllocator.zig`,
//! from this crate's own Zig re-survey pass): lock, forward to the
//! wrapped allocator, unlock, on every call.
//!
//! `SpinLock` itself is not modeled on Zig (`std.Thread.Mutex` is an
//! OS-backed mutex, not a spinlock, so there was nothing to port
//! there) or on foonathan/memory (whose own `no_std`-compatible
//! locking story this survey never found a matching piece for). It is
//! grounded instead in the `spin` crate's real, widely used
//! `SpinMutex` (source read, `spin` 0.10.0 from crates.io): a "test,
//! then test-and-set" loop -- attempt the actual atomic
//! compare-exchange first, and only if that fails, spin on a plain
//! `Relaxed` load (cheaper, avoids hammering the cache line with
//! repeated read-modify-write traffic under contention) until the lock
//! *looks* free before attempting the compare-exchange again. Ported
//! directly rather than reinvented, simplified by dropping `spin`'s
//! generic relax-strategy type parameter (`core::hint::spin_loop()`
//! hardcoded, not made pluggable) since this crate has no use for more
//! than the one backend.

use crate::raw_alloc::RawAlloc;
use core::cell::UnsafeCell;
use core::ops::{Deref, DerefMut};
use core::ptr::NonNull;
use core::sync::atomic::{AtomicBool, Ordering};

/// A spinlock-guarded `T`. See this module's doc comment for the real
/// source the locking algorithm is grounded in.
pub struct SpinLock<T> {
    locked: AtomicBool,
    data: UnsafeCell<T>,
}

// Same unsafe impls `spin::mutex::SpinMutex` and `std::sync::Mutex`
// both carry: a `SpinLock<T>` is safe to move to another thread
// whenever `T` is (auto-derived already, stated here explicitly rather
// than left implicit), and safe to share across threads whenever `T`
// is `Send` -- `Sync` is not required of `T`, since the lock itself is
// what provides the exclusion a bare `T: Sync` bound would otherwise
// have to.
unsafe impl<T: Send> Send for SpinLock<T> {}
unsafe impl<T: Send> Sync for SpinLock<T> {}

impl<T> SpinLock<T> {
    /// Wraps `data` behind a new, unlocked spinlock.
    pub const fn new(data: T) -> Self {
        Self {
            locked: AtomicBool::new(false),
            data: UnsafeCell::new(data),
        }
    }

    /// Blocks (by spinning, not sleeping -- there is no OS to ask for
    /// a real wait here) until the lock is acquired, then returns a
    /// guard granting exclusive access.
    pub fn lock(&self) -> SpinLockGuard<'_, T> {
        loop {
            if let Some(guard) = self.try_lock() {
                return guard;
            }
            // The cheaper, non-exclusive wait matches `spin`'s own
            // real reasoning directly: a plain `Relaxed` load doesn't
            // need to win any cache-line ownership the way the
            // compare-exchange below does, so spinning here first
            // avoids flooding the bus with expensive RMW traffic while
            // the lock is genuinely held by someone else.
            while self.locked.load(Ordering::Relaxed) {
                core::hint::spin_loop();
            }
        }
    }

    /// Attempts to acquire the lock without blocking. Returns `None`
    /// if it is already held.
    pub fn try_lock(&self) -> Option<SpinLockGuard<'_, T>> {
        // `compare_exchange_weak`, matching `spin`'s own `try_lock_weak`:
        // allowed to spuriously fail even when the lock is free, which
        // is fine (and can be cheaper on some platforms) since `lock`
        // above just retries in that case anyway.
        self.locked
            .compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
            .ok()
            .map(|_| SpinLockGuard { lock: self })
    }
}

/// Grants exclusive access to a [`SpinLock`]'s data. Releases the lock
/// when dropped.
pub struct SpinLockGuard<'a, T> {
    lock: &'a SpinLock<T>,
}

impl<'a, T> Deref for SpinLockGuard<'a, T> {
    type Target = T;

    fn deref(&self) -> &T {
        // SAFETY: holding a `SpinLockGuard` is proof this lock is
        // held by no one else -- `try_lock`/`lock` are the only ways
        // to construct one, and both require winning the
        // compare-exchange above first.
        unsafe { &*self.lock.data.get() }
    }
}

impl<'a, T> DerefMut for SpinLockGuard<'a, T> {
    fn deref_mut(&mut self) -> &mut T {
        // SAFETY: same reasoning as `deref` above; `&mut self` here
        // additionally rules out another live `&T` from this same
        // guard existing simultaneously.
        unsafe { &mut *self.lock.data.get() }
    }
}

impl<'a, T> Drop for SpinLockGuard<'a, T> {
    fn drop(&mut self) {
        self.lock.locked.store(false, Ordering::Release);
    }
}

/// Wraps any [`RawAlloc`] behind a [`SpinLock`], making it safe to
/// share across threads regardless of what interior mutability the
/// wrapped allocator itself uses (`StackAllocator`'s bare `Cell`s
/// included). Matches Zig's `std.heap.ThreadSafeAllocator`'s real
/// shape exactly: lock, forward, unlock, on every call -- see this
/// module's doc comment.
pub struct SyncAlloc<A> {
    inner: SpinLock<A>,
}

impl<A> SyncAlloc<A> {
    /// Wraps `inner` behind a new spinlock.
    pub const fn new(inner: A) -> Self {
        Self {
            inner: SpinLock::new(inner),
        }
    }
}

impl<A: RawAlloc> RawAlloc for SyncAlloc<A> {
    fn try_alloc_raw(&self, size: usize, align: usize) -> Option<NonNull<u8>> {
        self.inner.lock().try_alloc_raw(size, align)
    }

    unsafe fn try_dealloc_raw(&self, ptr: NonNull<u8>, size: usize, align: usize) -> bool {
        // SAFETY: forwarding the caller's own contract straight
        // through to `inner`, now serialized by the lock.
        self.inner.lock().try_dealloc_raw(ptr, size, align)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::raw_alloc::HeapAlloc;
    use crate::stack_allocator::StackAllocator;

    #[test]
    fn lock_grants_exclusive_mutable_access() {
        let l = SpinLock::new(0u32);
        *l.lock() += 1;
        *l.lock() += 1;
        assert_eq!(*l.lock(), 2);
    }

    #[test]
    fn try_lock_fails_while_a_guard_is_still_held() {
        let l = SpinLock::new(0u32);
        let guard = l.lock();
        assert!(l.try_lock().is_none(), "already held, must not succeed");
        drop(guard);
        assert!(l.try_lock().is_some(), "released, must succeed now");
    }

    #[test]
    fn spin_lock_is_send_and_sync_when_t_is_send() {
        // Compile-time check, no_std-friendly -- see BumpArena's own
        // `bump_arena_is_send_when_t_is_send` in mid-arena for the
        // same idiom applied to `Send` alone.
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<SpinLock<u32>>();
        // `StackAllocator` itself is `Send` (nothing in it blocks
        // that) but not `Sync` (its `Cell`s block it) -- the real
        // property this whole module exists to add back.
        assert_send_sync::<SpinLock<StackAllocator>>();
    }

    #[test]
    fn many_real_threads_racing_a_spin_lock_lose_no_updates() {
        extern crate std;
        use std::sync::Arc;
        use std::thread;

        let counter = Arc::new(SpinLock::new(0u64));
        let threads = 8;
        let increments_per_thread = 2_000;

        let handles: std::vec::Vec<_> = (0..threads)
            .map(|_| {
                let counter = Arc::clone(&counter);
                thread::spawn(move || {
                    for _ in 0..increments_per_thread {
                        *counter.lock() += 1;
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(*counter.lock(), threads * increments_per_thread);
    }

    #[test]
    fn many_real_threads_allocating_through_sync_alloc_never_overlap() {
        extern crate std;
        use std::sync::Arc;
        use std::thread;

        // Small enough that, without real mutual exclusion, concurrent
        // `alloc_raw` calls racing `StackAllocator`'s own `top` `Cell`
        // would be expected to corrupt it (hand two callers the same
        // bytes, or lose a bump) well within this many attempts.
        let threads = 8;
        let per_thread = 200;
        let cap = threads * per_thread; // one byte per allocation
        let alloc = Arc::new(SyncAlloc::new(StackAllocator::with_capacity(cap)));

        let handles: std::vec::Vec<_> = (0..threads)
            .map(|_| {
                let alloc = Arc::clone(&alloc);
                thread::spawn(move || {
                    let mut got = std::vec::Vec::with_capacity(per_thread);
                    for _ in 0..per_thread {
                        if let Some(ptr) = alloc.try_alloc_raw(1, 1) {
                            got.push(ptr.as_ptr() as usize);
                        }
                    }
                    got
                })
            })
            .collect();

        let mut all_addrs: std::vec::Vec<usize> =
            handles.into_iter().flat_map(|h| h.join().unwrap()).collect();
        assert_eq!(
            all_addrs.len(),
            threads * per_thread,
            "every allocation across every thread should have succeeded -- capacity was sized for exactly this many"
        );
        all_addrs.sort_unstable();
        all_addrs.dedup();
        assert_eq!(
            all_addrs.len(),
            threads * per_thread,
            "no two threads should ever have been handed the same address"
        );
    }

    #[test]
    fn sync_alloc_wraps_heap_alloc_too() {
        let a = SyncAlloc::new(HeapAlloc);
        let ptr = a.try_alloc_raw(16, 8).expect("a small heap allocation should not fail");
        unsafe {
            assert!(a.try_dealloc_raw(ptr, 16, 8));
        }
    }
}
