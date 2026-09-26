//! C-compatible FFI exports for mid-platform.
//!
//! Scope for this pass: `Once` and `Barrier` only. Both cross the FFI
//! boundary cleanly because neither ever hands a caller a *borrowed*
//! guard object that has to be explicitly, correctly paired with a
//! later release call: `Barrier::wait` returns a plain `bool` by value,
//! and `Once::call_once` just runs a callback and returns `()` — nothing
//! for a C caller to hold onto, nothing that can be freed too early or
//! used after its data went away.
//!
//! `Mutex`, `RwLock`, `OnceLock`, and `LazyLock` are deliberately **not**
//! covered here — every one of them hands back a live reference (a
//! `MutexGuard`/`RwLock*Guard`, or a `&T` straight out of `OnceLock`/
//! `LazyLock`) into memory this crate owns. Exposing that to C means an
//! opaque guard handle that must be explicitly unlocked (C has no
//! `Drop`), plus a real hazard if the underlying handle gets freed while
//! a guard is still outstanding — genuinely the same class of problem
//! `mid-ecs`'s own `ffi.rs` named and deliberately deferred for
//! component-data access in its first pass (a live pointer into
//! caller-visible memory that a later call can invalidate out from
//! under it), not attempted in the same pass as this crate's own first
//! slice either. Tracked as open work in `docs/mid-platform.md`.
//!
//! `std`-only: building a `cdylib`/`staticlib` for C to link against
//! needs a real OS underneath it anyway (an allocator, a panic runtime),
//! so this module doesn't build under `--no-default-features` — the
//! `no_std` fallback path was never meant to be called from C directly
//! in the first place.
//!
//! ## Conventions — copied directly from `mid-net`'s/`mid-ecs`'s real
//! `ffi.rs` files, not reinvented
//! - Every function checks its pointer arguments for null before
//!   dereferencing and returns a defined error code (or a safe default —
//!   `false` for `mid_platform_barrier_wait`/`mid_platform_once_is_completed`,
//!   neither of which has a natural "distinct failure reason" shape)
//!   instead of dereferencing a null pointer.
//! - Every function's body runs inside [`std::panic::catch_unwind`] via
//!   `ffi_guard` (or the same pattern inlined for the two `bool`-
//!   returning functions, which don't fit `ffi_guard`'s `i32` shape) —
//!   unwinding across an `extern "C"` boundary is undefined behavior, so
//!   a panic here becomes `MidPlatformStatus::InternalPanic`/`false`
//!   instead.
//! - Every function taking a raw pointer is `unsafe fn` with a `# Safety`
//!   doc comment, matching `clippy::not_unsafe_ptr_arg_deref`'s
//!   requirement.
//! - Neither `Once` nor `Barrier` is `repr(C)` (both wrap real Rust
//!   types with a `Mutex`/`OnceLock` inside), so each crosses the
//!   boundary as an opaque heap-allocated handle (`Box::into_raw`/
//!   `Box::from_raw`), matching `mid-ecs`'s own `MidEcsWorld` handle
//!   pattern exactly.

use std::ffi::c_void;
use std::panic::{catch_unwind, AssertUnwindSafe};

use crate::sync::{Barrier, Once};

/// Status codes. `mid_platform_barrier_wait` and
/// `mid_platform_once_is_completed` return a plain `bool` (pure queries/
/// waits, not operations with distinct failure reasons) — everything
/// else that can fail returns one of these (`MidPlatformStatus::Ok` is
/// always `0`), matching `mid-ecs`'s/`mid-net`'s own status-enum
/// convention.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MidPlatformStatus {
    Ok = 0,
    NullPointer = -1,
    /// Something inside this crate panicked. Should never happen for
    /// well-formed input per each function's documented contract, or a
    /// C callback passed to `mid_platform_once_call` that itself never
    /// panics (it can't panic in Rust's sense, but Rust code invoked
    /// through it — e.g. from this file's own tests — can) — exists so
    /// a caller gets a defined code instead of UB from an unwind
    /// crossing the FFI boundary.
    InternalPanic = -2,
}

fn ffi_guard(f: impl FnOnce() -> i32) -> i32 {
    catch_unwind(AssertUnwindSafe(f)).unwrap_or(MidPlatformStatus::InternalPanic as i32)
}

// ============================================================================
// Barrier
// ============================================================================

/// Opaque handle to a [`Barrier`]. Always heap-allocated by this crate;
/// every handle returned by `mid_platform_barrier_new` must be freed
/// with `mid_platform_barrier_free` exactly once.
pub struct MidPlatformBarrier(Barrier);

/// Creates a new barrier that blocks `num_threads` calls to
/// `mid_platform_barrier_wait` before releasing all of them at once.
/// Never returns NULL — allocation failure aborts the process the same
/// way any other Rust `Box` allocation failure would, matching
/// `mid_ecs_world_new`'s own convention.
#[no_mangle]
pub extern "C" fn mid_platform_barrier_new(num_threads: usize) -> *mut MidPlatformBarrier {
    Box::into_raw(Box::new(MidPlatformBarrier(Barrier::new(num_threads))))
}

/// Frees a handle returned by `mid_platform_barrier_new`. NULL is a safe
/// no-op.
///
/// # Safety
/// `barrier` must either be NULL, or a handle previously returned by
/// `mid_platform_barrier_new` that hasn't been freed yet and that no
/// other thread is currently blocked inside `mid_platform_barrier_wait`
/// on (freeing a barrier out from under a thread still waiting on it is
/// undefined behavior — this crate has no way to detect that misuse any
/// more than freeing any other shared object out from under a live
/// reference to it would).
#[no_mangle]
pub unsafe extern "C" fn mid_platform_barrier_free(barrier: *mut MidPlatformBarrier) {
    if barrier.is_null() {
        return;
    }
    drop(unsafe { Box::from_raw(barrier) });
}

/// Blocks the calling thread until `num_threads` total calls to this
/// function (across however many threads) have been made on this
/// `barrier`, then releases all of them at once. Returns `true` for
/// exactly one (arbitrary) call per rendezvous — the "leader" — `false`
/// for every other. A barrier is reusable: calling this again after a
/// full rendezvous works the same way a second time.
///
/// Returns `false` on a NULL `barrier` without blocking — matches
/// `mid_ecs_world_is_alive`'s own "null handle getters return safe
/// defaults, not a crash" convention, since this has no natural
/// distinct-failure-code shape to report through instead.
///
/// # Safety
/// `barrier` must either be NULL, or a valid handle from
/// `mid_platform_barrier_new` that stays alive for as long as any
/// thread might still be blocked inside this function on it.
#[no_mangle]
pub unsafe extern "C" fn mid_platform_barrier_wait(barrier: *const MidPlatformBarrier) -> bool {
    if barrier.is_null() {
        return false;
    }
    let result = catch_unwind(AssertUnwindSafe(|| {
        let barrier = unsafe { &*barrier };
        barrier.0.wait().is_leader()
    }));
    result.unwrap_or(false)
}

// ============================================================================
// Once
// ============================================================================

/// C function-pointer type for the callback passed to
/// `mid_platform_once_call` — see that function's own doc comment.
pub type MidPlatformOnceFn = unsafe extern "C" fn(ctx: *mut c_void);

/// Opaque handle to an [`Once`]. Always heap-allocated by this crate;
/// every handle returned by `mid_platform_once_new` must be freed with
/// `mid_platform_once_free` exactly once.
pub struct MidPlatformOnce(Once);

/// Creates a new `Once`, not yet completed. Never returns NULL, matching
/// `mid_platform_barrier_new`'s own convention.
#[no_mangle]
pub extern "C" fn mid_platform_once_new() -> *mut MidPlatformOnce {
    Box::into_raw(Box::new(MidPlatformOnce(Once::new())))
}

/// Frees a handle returned by `mid_platform_once_new`. NULL is a safe
/// no-op.
///
/// # Safety
/// `once` must either be NULL, or a handle previously returned by
/// `mid_platform_once_new` that hasn't been freed yet.
#[no_mangle]
pub unsafe extern "C" fn mid_platform_once_free(once: *mut MidPlatformOnce) {
    if once.is_null() {
        return;
    }
    drop(unsafe { Box::from_raw(once) });
}

/// Calls `f(ctx)` exactly once across every call to this function made
/// on this `once`, however many times (or from however many threads) it
/// is called — every call after the first one that actually runs `f`
/// just waits for that first call to finish, then returns, without
/// running `f` again. Matches `Once::call_once`'s own semantics exactly
/// (see `sync::once`'s own doc comment): this function doesn't tell the
/// caller whether *this particular call* was the one that ran `f` —
/// neither does the underlying Rust method.
///
/// Returns `MidPlatformStatus::NullPointer` on a NULL `once` or a NULL
/// `f`, `MidPlatformStatus::InternalPanic` if `f` (or anything it calls
/// back into on the Rust side) panics, `MidPlatformStatus::Ok`
/// otherwise.
///
/// # Safety
/// `once` must be a valid, non-null handle from `mid_platform_once_new`.
/// `f`, if non-null, must be a valid function pointer safe to call with
/// `ctx`; `ctx` may be NULL only if `f` is documented to accept that.
#[no_mangle]
pub unsafe extern "C" fn mid_platform_once_call(
    once: *const MidPlatformOnce,
    f: Option<MidPlatformOnceFn>,
    ctx: *mut c_void,
) -> i32 {
    ffi_guard(|| {
        if once.is_null() {
            return MidPlatformStatus::NullPointer as i32;
        }
        let Some(f) = f else {
            return MidPlatformStatus::NullPointer as i32;
        };
        let once = unsafe { &*once };
        // SAFETY: `f`/`ctx` are exactly the pointer and context this
        // function's own `# Safety` contract requires the caller to
        // have supplied validly; calling `f` here is the whole point of
        // this function.
        once.0.call_once(|| unsafe { f(ctx) });
        MidPlatformStatus::Ok as i32
    })
}

/// Returns `true` if `mid_platform_once_call` has completed
/// successfully at least once on this `once`. `false` on a NULL `once`.
///
/// # Safety
/// `once` must either be NULL, or a valid handle from
/// `mid_platform_once_new`.
#[no_mangle]
pub unsafe extern "C" fn mid_platform_once_is_completed(once: *const MidPlatformOnce) -> bool {
    if once.is_null() {
        return false;
    }
    let result = catch_unwind(AssertUnwindSafe(|| {
        let once = unsafe { &*once };
        once.0.is_completed()
    }));
    result.unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    // --- Barrier ---

    #[test]
    fn barrier_new_free_round_trips() {
        let b = mid_platform_barrier_new(1);
        assert!(!b.is_null());
        // SAFETY: `b` is non-null, just created, not yet freed.
        unsafe { mid_platform_barrier_free(b) };
    }

    #[test]
    fn barrier_free_null_is_a_safe_no_op() {
        // SAFETY: NULL is the documented safe-no-op case.
        unsafe { mid_platform_barrier_free(std::ptr::null_mut()) };
    }

    #[test]
    fn barrier_of_one_is_immediately_its_own_leader() {
        let b = mid_platform_barrier_new(1);
        // SAFETY: `b` is non-null and not yet freed for this whole block.
        unsafe {
            assert!(mid_platform_barrier_wait(b));
            mid_platform_barrier_free(b);
        }
    }

    #[test]
    fn barrier_wait_on_null_returns_false_without_blocking() {
        // SAFETY: NULL is the documented safe path.
        assert!(!unsafe { mid_platform_barrier_wait(std::ptr::null()) });
    }

    #[test]
    fn barrier_releases_every_real_thread_together_through_the_c_surface() {
        let barrier = mid_platform_barrier_new(4);
        let ptr = barrier as usize; // Send the raw address across threads explicitly.
        let leaders = std::sync::Arc::new(AtomicU32::new(0));

        let handles: Vec<_> = (0..4)
            .map(|_| {
                let leaders = std::sync::Arc::clone(&leaders);
                std::thread::spawn(move || {
                    // SAFETY: `ptr` points at `barrier`, which stays alive
                    // for this whole test (freed only after every thread
                    // below has joined).
                    let is_leader = unsafe { mid_platform_barrier_wait(ptr as *const MidPlatformBarrier) };
                    if is_leader {
                        leaders.fetch_add(1, Ordering::SeqCst);
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(leaders.load(Ordering::SeqCst), 1, "exactly one thread must be leader");

        // SAFETY: every thread above has joined; nothing is waiting on
        // `barrier` anymore.
        unsafe { mid_platform_barrier_free(barrier) };
    }

    // --- Once ---

    #[test]
    fn once_new_free_round_trips() {
        let o = mid_platform_once_new();
        assert!(!o.is_null());
        // SAFETY: `o` is non-null, just created, not yet freed.
        unsafe { mid_platform_once_free(o) };
    }

    #[test]
    fn once_free_null_is_a_safe_no_op() {
        // SAFETY: NULL is the documented safe-no-op case.
        unsafe { mid_platform_once_free(std::ptr::null_mut()) };
    }

    static CALL_COUNT: AtomicU32 = AtomicU32::new(0);
    unsafe extern "C" fn increment_call_count(_ctx: *mut c_void) {
        CALL_COUNT.fetch_add(1, Ordering::SeqCst);
    }

    #[test]
    fn once_call_runs_the_callback_exactly_once() {
        CALL_COUNT.store(0, Ordering::SeqCst);
        let o = mid_platform_once_new();
        // SAFETY: `o` non-null and not yet freed; `increment_call_count`
        // is a real, valid `extern "C" fn` accepting a possibly-NULL ctx.
        unsafe {
            for _ in 0..5 {
                let status =
                    mid_platform_once_call(o, Some(increment_call_count), std::ptr::null_mut());
                assert_eq!(status, MidPlatformStatus::Ok as i32);
            }
            assert_eq!(CALL_COUNT.load(Ordering::SeqCst), 1);
            assert!(mid_platform_once_is_completed(o));
            mid_platform_once_free(o);
        }
    }

    #[test]
    fn once_call_on_null_once_or_null_fn_is_null_pointer() {
        let o = mid_platform_once_new();
        // SAFETY: exercising the documented NULL-pointer error paths.
        unsafe {
            assert_eq!(
                mid_platform_once_call(std::ptr::null(), Some(increment_call_count), std::ptr::null_mut()),
                MidPlatformStatus::NullPointer as i32
            );
            assert_eq!(
                mid_platform_once_call(o, None, std::ptr::null_mut()),
                MidPlatformStatus::NullPointer as i32
            );
            mid_platform_once_free(o);
        }
    }

    #[test]
    fn once_is_completed_on_null_returns_false() {
        // SAFETY: NULL is the documented safe path.
        assert!(!unsafe { mid_platform_once_is_completed(std::ptr::null()) });
    }

    #[test]
    fn once_is_completed_reflects_real_state() {
        let o = mid_platform_once_new();
        // SAFETY: `o` non-null and not yet freed for this whole block.
        unsafe {
            assert!(!mid_platform_once_is_completed(o));
            mid_platform_once_call(o, Some(increment_call_count), std::ptr::null_mut());
            assert!(mid_platform_once_is_completed(o));
            mid_platform_once_free(o);
        }
    }
}
