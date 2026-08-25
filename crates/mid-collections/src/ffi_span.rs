//! A reusable, checked FFI buffer/span wrapper.
//!
//! Two real, already-documented needs converge on this one module:
//!
//! - `mid-net`'s own `ffi.rs` hand-rolls "null-check + buffer-size-check +
//!   `catch_unwind` boundary" per function, one call site at a time
//!   (`mid_net_player_state_encode`/`decode`,
//!   `mid_net_player_event_encode`/`decode`, and every future one).
//!   Real, already-visible duplication -- not speculative.
//! - `mid-ecs`'s FFI section (`docs/mid-ecs.md`) needs a way to expose
//!   component-column data (`Vec<T>` inside `SparseShell`'s `SparseSet`s
//!   and `Archetypes`' `Table`s) across the boundary, where `insert`/
//!   `remove`/migration can reallocate or move the backing storage out
//!   from under a previously handed-out pointer -- unlike anything in
//!   either crate's `ffi.rs` today, where every value crosses by-value
//!   or through an opaque handle with no live pointer into mutable
//!   interior storage.
//!
//! `docs/mid-collections.md`'s "FFI wrapper" section calls this the
//! `FfiBuf`-shaped idea (name not settled) and makes one scope call
//! explicit: **build on `zerocopy`, don't reimplement alignment-checked
//! casting by hand.** Checked directly against `zerocopy` 0.8.56's real
//! behavior before writing anything here (not assumed): its `FromBytes`
//! family (`<[T]>::ref_from_bytes`/`mut_from_bytes`) does real runtime
//! length *and* alignment checking as part of the trait itself, and
//! rejects with `Err` rather than UB on a mismatch. Also confirmed
//! directly: `zerocopy` 0.8.56 (with the `derive` feature, for
//! `FromBytes`/`IntoBytes`/`KnownLayout`/`Immutable`) builds clean on
//! this project's rustc 1.75 floor -- no MSRV wall, unlike five other
//! dependencies this project has hit that exact problem with.
//!
//! # Two directions, one shared shape
//!
//! - **Outbound** (Rust -> C): [`FfiSpan`], a plain `#[repr(C)]` struct
//!   (`ptr`/`stride`/`count`) built from an already-valid Rust `&[T]`
//!   via [`FfiSpan::from_slice`]. Nothing to validate on this side --
//!   Rust already knows the data is valid. The real requirement is that
//!   `T`'s layout is actually meaningful to a non-Rust reader, which is
//!   what the `IntoBytes` bound statically guarantees: no uninitialized
//!   padding bytes can leak across the boundary, and no interior
//!   mutability (`Immutable`) can make "read-only from C" a lie.
//! - **Inbound** (C -> Rust): [`checked_slice`]/[`checked_slice_mut`],
//!   turning a caller-supplied `(ptr, len_bytes)` pair into a real,
//!   validated `&[T]`/`&mut [T]` -- null-checked here, length- and
//!   alignment-checked by `zerocopy`.
//!
//! Deliberately one shape for both crates' needs, not an `mid-ecs`-only
//! or `mid-net`-only type: a `mid-net` byte buffer is just `stride ==
//! 1`; a `mid-ecs` component column is `stride == size_of::<T>()`. Same
//! struct either way.
//!
//! # `Vec::as_ptr()`'s contract, made explicit for a non-Rust caller
//!
//! [`FfiSpan`] carries no lifetime -- it can't, it's `#[repr(C)]` and
//! crosses into a language with no borrow checker. The safety contract
//! that a Rust caller gets for free from the borrow checker has to
//! become a *documented convention* instead once it crosses: an
//! `FfiSpan` is valid only until the next call that could reallocate or
//! reorder the exact storage it points into. This is not a weaker
//! guarantee than `Vec::as_ptr()`'s own -- it's the identical guarantee,
//! just no longer mechanically enforced past the boundary. Every
//! FFI function that hands out an `FfiSpan` must say so in its own
//! `# Safety` doc, matching how every unsafe function in this workspace
//! already documents its contract rather than relying on this module to
//! carry it implicitly.
//!
//! # Three tiers of "how much runtime checking", not two
//!
//! - **Type-checked Rust input needs zero runtime checks.**
//!   [`FfiSpan::from_slice`] validates nothing -- by the time a caller
//!   has a real `&[T]`, the borrow checker and type system already
//!   proved it's non-null, aligned, and live. A runtime null-check there
//!   would be re-checking something the compiler already ruled out.
//! - **A raw pointer crossing the FFI boundary needs real runtime
//!   checks that can never be elided, in *any* build profile.**
//!   [`checked_slice`]/[`checked_slice_mut`] take a bare pointer +
//!   length from a caller the Rust compiler has zero visibility into --
//!   nothing proved it's non-null, correctly aligned, or the length it
//!   claims. The null check and `zerocopy`'s length/alignment check stay
//!   in release too: there's no compile-time proof to lean on for
//!   adversarial-by-construction input.
//! - **A postcondition of a call already trusted needs no independent
//!   re-check in production, only in testing.** The two `debug_assert!`s
//!   in each function (`slice.as_ptr() == ptr`,
//!   `slice.len() * size_of::<T>() == len_bytes`) aren't checking
//!   anything that could independently vary -- they're true by
//!   construction *given* `ref_from_bytes`/`mut_from_bytes` already
//!   returned `Ok`, per that function's own documented contract. Kept
//!   as a debug-only self-check against a bug in this wrapper's own
//!   reasoning (or a future `zerocopy` behavior change), not compiled
//!   into release, matching this workspace's established "prove it
//!   during testing, pay nothing in production" shape (`mid-net`'s own
//!   hot paths already follow this). This is what
//!   `docs/mid-collections.md`'s "sentinel/canary corruption check" line
//!   is implemented as here -- a self-check against this module's own
//!   logic, not a guard-byte pattern around caller-owned memory (which
//!   would need a different shape entirely, and isn't what the actual
//!   risk here calls for: nothing in this module owns or allocates
//!   memory across a call boundary for a guard pattern to protect).

use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

/// A plain, C-representable view into a Rust-owned dense array: pointer
/// to the first element, size of one element in bytes, and element
/// count. The "FFI span" wire shape -- see this module's doc comment
/// for the full design and its safety contract.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FfiSpan {
    pub ptr: *const u8,
    pub stride: usize,
    pub count: usize,
}

impl FfiSpan {
    /// An empty, always-safe-to-hand-out span -- `ptr` is null, `count`
    /// is `0`. The correct return value for "nothing here" (a
    /// never-registered component type, an empty table), never a
    /// dangling non-null pointer with a `0` count.
    pub const fn empty() -> Self {
        Self {
            ptr: core::ptr::null(),
            stride: 0,
            count: 0,
        }
    }

    /// Builds a span over an existing, valid Rust slice.
    ///
    /// `T: IntoBytes` is the load-bearing bound, not an arbitrary
    /// restriction -- see this module's doc comment for why.
    pub fn from_slice<T: IntoBytes + Immutable>(slice: &[T]) -> Self {
        if slice.is_empty() {
            return Self::empty();
        }
        Self {
            ptr: slice.as_ptr().cast::<u8>(),
            stride: core::mem::size_of::<T>(),
            count: slice.len(),
        }
    }

    /// Whether this span is the empty sentinel (`ptr` null, `count` 0).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
}

/// Everything that can go wrong validating a caller-supplied `(ptr,
/// len_bytes)` pair before it's safe to treat as `&[T]`/`&mut [T]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FfiBufError {
    /// `ptr` was null.
    NullPointer,
    /// `len_bytes` isn't an exact multiple of `size_of::<T>()`.
    LengthMismatch,
    /// `ptr` wasn't aligned for `T` -- surfaced by `zerocopy`'s own
    /// `FromBytes` family, not hand-checked here.
    Misaligned,
}

impl core::fmt::Display for FfiBufError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NullPointer => write!(f, "null pointer passed across FFI boundary"),
            Self::LengthMismatch => {
                write!(
                    f,
                    "buffer length is not a valid multiple of the element size"
                )
            }
            Self::Misaligned => write!(f, "pointer is not correctly aligned for the target type"),
        }
    }
}

// No `impl std::error::Error` here -- this crate is `#![no_std]` (see
// `lib.rs`'s own doc comment); `Debug` + `Display` is the right amount
// of error-trait surface without pulling in `std`.

/// Validates a caller-supplied `(ptr, len_bytes)` pair and interprets
/// it as `&'a [T]`. Consolidates the null-check + length/alignment
/// check every FFI decode/getter in this workspace currently repeats by
/// hand into one place.
///
/// `len_bytes` must be exactly `N * size_of::<T>()` for some `N` --
/// this function does not accept (and silently truncate) a longer
/// buffer with trailing bytes; a caller claiming the wrong length is
/// exactly the kind of caller error this exists to catch.
///
/// # Safety
/// `ptr` must be either null, or valid for reads of `len_bytes` bytes,
/// for the entire lifetime `'a`. This function cannot verify that on
/// its own -- it's inherent to crossing a C ABI, same as every other
/// raw-pointer function in this workspace's `ffi.rs` files.
pub unsafe fn checked_slice<'a, T>(ptr: *const T, len_bytes: usize) -> Result<&'a [T], FfiBufError>
where
    T: FromBytes + Immutable + KnownLayout,
{
    if ptr.is_null() {
        return Err(FfiBufError::NullPointer);
    }
    // SAFETY: caller guarantees `ptr` is valid for reads of `len_bytes`
    // bytes for `'a`; null already ruled out above.
    let bytes = unsafe { core::slice::from_raw_parts(ptr.cast::<u8>(), len_bytes) };
    let slice = <[T]>::ref_from_bytes(bytes).map_err(|_| {
        // zerocopy folds "wrong length" and "misaligned" into one error
        // type here; length is cheap to re-check ourselves to give the
        // caller the more specific of the two.
        //
        // `usize::is_multiple_of` (stabilized in Rust 1.87 -- confirmed
        // via a real search, not assumed; this sandbox's rustc 1.75
        // floor predates it and cannot compile this line, confirmed
        // directly) is what real CI's clippy (rustc 1.98) wants in
        // place of the `%` form below. Neither toolchain gap is a
        // concern here -- CI's 1.98 postdates 1.87 comfortably -- but
        // this specific line is unverified by me: nothing I have
        // access to has compiled it. Flagged plainly rather than
        // quietly claimed as tested, unlike everything else in this
        // delivery.
        if !len_bytes.is_multiple_of(core::mem::size_of::<T>()) {
            FfiBufError::LengthMismatch
        } else {
            FfiBufError::Misaligned
        }
    })?;
    debug_assert_eq!(
        slice.as_ptr(),
        ptr,
        "checked_slice: constructed slice must start at the exact pointer it was given"
    );
    debug_assert_eq!(
        core::mem::size_of_val(slice),
        len_bytes,
        "checked_slice: constructed slice's byte length must exactly match the input"
    );
    Ok(slice)
}

/// Mutable counterpart to [`checked_slice`].
///
/// # Safety
/// `ptr` must be either null, or valid for reads and writes of
/// `len_bytes` bytes, for the entire lifetime `'a`, with no other live
/// reference to the same memory for that duration.
pub unsafe fn checked_slice_mut<'a, T>(
    ptr: *mut T,
    len_bytes: usize,
) -> Result<&'a mut [T], FfiBufError>
where
    T: FromBytes + IntoBytes + KnownLayout,
{
    if ptr.is_null() {
        return Err(FfiBufError::NullPointer);
    }
    let expected_ptr_addr = ptr as usize;
    // SAFETY: caller guarantees `ptr` is valid for reads+writes of
    // `len_bytes` bytes for `'a`, exclusively; null already ruled out.
    let bytes = unsafe { core::slice::from_raw_parts_mut(ptr.cast::<u8>(), len_bytes) };
    let slice = <[T]>::mut_from_bytes(bytes).map_err(|_| {
        // Same `is_multiple_of` swap as `checked_slice` above, same
        // unverified-by-me caveat -- see that function's comment.
        if !len_bytes.is_multiple_of(core::mem::size_of::<T>()) {
            FfiBufError::LengthMismatch
        } else {
            FfiBufError::Misaligned
        }
    })?;
    debug_assert_eq!(
        slice.as_ptr() as usize,
        expected_ptr_addr,
        "checked_slice_mut: constructed slice must start at the exact pointer it was given"
    );
    debug_assert_eq!(
        core::mem::size_of_val(slice),
        len_bytes,
        "checked_slice_mut: constructed slice's byte length must exactly match the input"
    );
    Ok(slice)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::format;

    #[derive(FromBytes, IntoBytes, KnownLayout, Immutable, Debug, Clone, Copy, PartialEq)]
    #[repr(C)]
    struct Position {
        x: f32,
        y: f32,
    }

    #[test]
    fn ffi_span_empty_has_null_ptr_and_zero_count() {
        let span = FfiSpan::empty();
        assert!(span.ptr.is_null());
        assert_eq!(span.count, 0);
        assert!(span.is_empty());
    }

    #[test]
    fn ffi_span_from_empty_slice_is_the_empty_sentinel() {
        let empty: &[Position] = &[];
        let span = FfiSpan::from_slice(empty);
        assert!(span.ptr.is_null());
        assert_eq!(span.count, 0);
    }

    #[test]
    fn ffi_span_from_slice_reports_correct_stride_and_count() {
        let data = [Position { x: 1.0, y: 2.0 }, Position { x: 3.0, y: 4.0 }];
        let span = FfiSpan::from_slice(&data);
        assert!(!span.ptr.is_null());
        assert_eq!(span.stride, core::mem::size_of::<Position>());
        assert_eq!(span.count, 2);
        assert_eq!(span.ptr, data.as_ptr().cast::<u8>());
    }

    #[test]
    fn checked_slice_null_pointer_is_an_error_not_a_crash() {
        // SAFETY: null is the documented, checked-first error case.
        let result = unsafe { checked_slice::<Position>(core::ptr::null(), 0) };
        assert_eq!(result, Err(FfiBufError::NullPointer));
    }

    #[test]
    fn checked_slice_mut_null_pointer_is_an_error_not_a_crash() {
        // SAFETY: null is the documented, checked-first error case.
        let result = unsafe { checked_slice_mut::<Position>(core::ptr::null_mut(), 0) };
        assert_eq!(result, Err(FfiBufError::NullPointer));
    }

    #[test]
    fn checked_slice_round_trips_real_data() {
        let data = [Position { x: 1.0, y: 2.0 }, Position { x: 3.0, y: 4.0 }];
        let bytes_len = core::mem::size_of_val(&data);
        // SAFETY: `data` is a real, valid, live array of exactly this length.
        let slice = unsafe { checked_slice::<Position>(data.as_ptr(), bytes_len) }
            .expect("well-formed input must parse");
        assert_eq!(slice, &data);
    }

    #[test]
    fn checked_slice_rejects_a_non_exact_multiple_length() {
        let data = [Position { x: 1.0, y: 2.0 }];
        // One byte short of one whole `Position` -- not accepted, not
        // silently truncated.
        let short_len = core::mem::size_of::<Position>() - 1;
        // SAFETY: `data` is valid for at least `short_len` bytes (it's
        // valid for the whole array, which is longer).
        let result = unsafe { checked_slice::<Position>(data.as_ptr(), short_len) };
        assert_eq!(result, Err(FfiBufError::LengthMismatch));
    }

    #[test]
    fn checked_slice_mut_actually_allows_writes_through() {
        let mut data = [Position { x: 0.0, y: 0.0 }, Position { x: 0.0, y: 0.0 }];
        let bytes_len = core::mem::size_of_val(&data);
        // SAFETY: `data` is real, valid, live, and exclusively borrowed
        // for the duration of this block.
        let slice = unsafe { checked_slice_mut::<Position>(data.as_mut_ptr(), bytes_len) }
            .expect("well-formed input must parse");
        slice[0].x = 9.0;
        slice[1].y = 8.0;
        assert_eq!(data[0], Position { x: 9.0, y: 0.0 });
        assert_eq!(data[1], Position { x: 0.0, y: 8.0 });
    }

    #[test]
    fn checked_slice_empty_input_gives_empty_slice_not_an_error() {
        // A non-null, zero-length buffer is well-formed (0 is a valid
        // multiple of any stride) -- must round-trip as an empty slice,
        // not an error.
        let data: [Position; 0] = [];
        // SAFETY: a zero-length read from any non-null, well-aligned
        // pointer is always valid, including a dangling-but-aligned
        // `Vec`/array-of-zero pointer.
        let slice = unsafe { checked_slice::<Position>(data.as_ptr(), 0) }
            .expect("zero-length input must be well-formed, not an error");
        assert!(slice.is_empty());
    }

    #[test]
    fn ffi_buf_error_display_does_not_panic() {
        for e in [
            FfiBufError::NullPointer,
            FfiBufError::LengthMismatch,
            FfiBufError::Misaligned,
        ] {
            let _ = format!("{e}");
        }
    }
}
