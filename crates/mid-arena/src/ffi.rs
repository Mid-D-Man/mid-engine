// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-arena.md, section "ffi.rs"
// ============================================================================
//! Checked access to arena-owned memory across an FFI boundary. Behind
//! the `ffi` feature.
//!
//! One rule runs through the module: nothing here trusts a pointer, a
//! length, or a key it was handed.
//!
//! - Handle in, bytes across: [`read_value`], [`write_value`] and
//!   [`value_span`] take an `ArenaKey::as_ffi()` handle (a bare index for
//!   `UncheckedSlotArena`) through [`FfiKeyed`]. `read_value` and
//!   `write_value` copy bytes, so C never holds a pointer into slot
//!   storage that a later `insert` could move.
//! - Regions out: `BumpArena::region_spans` (feature `bump`) hands each
//!   contiguous region to C as an [`ArenaSpan`].
//!
//! The slot arenas never export a span over many slots. Their storage is
//! a `Vec` of slot enums or unions, so the stride is not
//! `size_of::<T>()` and a span would describe memory that is not an
//! array of `T`.
//!
//! [`ArenaSpan`] has the same `repr(C)` layout as
//! `mid_collections::FfiSpan` (`ptr`, `stride`, `count`), so one C
//! struct covers both. It is a separate type on purpose: depending on
//! `mid-collections` would add a crate edge, and `ffi_span.rs` there
//! calls `usize::is_multiple_of`, which needs Rust 1.87, above this
//! crate's 1.75 floor.

use zerocopy::{FromBytes, Immutable, IntoBytes};

use crate::slot_arena::{ArenaKey, SlotArena};

#[cfg(feature = "bump")]
use crate::bump_arena::BumpArena;
#[cfg(feature = "compact")]
use crate::compact_slot_arena::CompactSlotArena;
#[cfg(feature = "unchecked")]
use crate::unchecked_slot_arena::UncheckedSlotArena;

/// A read-only view of `count` consecutive `stride`-byte elements,
/// laid out for C.
///
/// The pointer is valid only as long as the arena it came from is not
/// mutated through `&mut self` and not dropped. Each producing function
/// states its own exact validity window.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArenaSpan {
    pub ptr: *const u8,
    pub stride: usize,
    pub count: usize,
}

impl ArenaSpan {
    /// The "nothing here" span: null `ptr`, `count` of `0`.
    pub const fn empty() -> Self {
        Self {
            ptr: core::ptr::null(),
            stride: 0,
            count: 0,
        }
    }

    /// Builds a span over an existing Rust slice. An empty slice gives
    /// [`empty`](Self::empty), never a dangling pointer with a zero
    /// count.
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

    /// Whether this span holds no elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
}

/// Everything that can go wrong in the checked access functions below.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FfiArenaError {
    /// The caller's buffer pointer was null.
    NullPointer,
    /// The element type has size zero, so there are no bytes to copy.
    ZeroSized,
    /// The caller's buffer length is not exactly `size_of::<T>()`.
    LengthMismatch,
    /// The key does not name a live value: never issued, already
    /// removed, or its slot has since been reused under a newer
    /// generation. For `UncheckedSlotArena`, also a key above
    /// `u32::MAX`.
    StaleKey,
}

/// A keyed arena whose handles cross FFI as a plain `u64`.
///
/// Lookups never panic and never read out of bounds on a bad key: a key
/// that names no live value returns `None`.
pub trait FfiKeyed {
    type Value;

    fn ffi_get(&self, key: u64) -> Option<&Self::Value>;

    fn ffi_get_mut(&mut self, key: u64) -> Option<&mut Self::Value>;
}

impl<T> FfiKeyed for SlotArena<T> {
    type Value = T;

    #[inline]
    fn ffi_get(&self, key: u64) -> Option<&T> {
        self.get(ArenaKey::from_ffi(key))
    }

    #[inline]
    fn ffi_get_mut(&mut self, key: u64) -> Option<&mut T> {
        self.get_mut(ArenaKey::from_ffi(key))
    }
}

#[cfg(feature = "compact")]
impl<T> FfiKeyed for CompactSlotArena<T> {
    type Value = T;

    #[inline]
    fn ffi_get(&self, key: u64) -> Option<&T> {
        self.get(ArenaKey::from_ffi(key))
    }

    #[inline]
    fn ffi_get_mut(&mut self, key: u64) -> Option<&mut T> {
        self.get_mut(ArenaKey::from_ffi(key))
    }
}

/// `UncheckedSlotArena` has no generation, so a stale index that was
/// reused reads back the new occupant. That is the type's own contract,
/// unchanged here. Only a key that does not fit a `u32` is rejected.
#[cfg(feature = "unchecked")]
impl<T> FfiKeyed for UncheckedSlotArena<T> {
    type Value = T;

    #[inline]
    fn ffi_get(&self, key: u64) -> Option<&T> {
        self.get(u32::try_from(key).ok()?)
    }

    #[inline]
    fn ffi_get_mut(&mut self, key: u64) -> Option<&mut T> {
        self.get_mut(u32::try_from(key).ok()?)
    }
}

/// Validates a caller buffer meant to hold exactly one `T`.
fn check_one<T>(ptr: *const u8, len_bytes: usize) -> Result<(), FfiArenaError> {
    if ptr.is_null() {
        return Err(FfiArenaError::NullPointer);
    }
    let size = core::mem::size_of::<T>();
    if size == 0 {
        return Err(FfiArenaError::ZeroSized);
    }
    if len_bytes != size {
        return Err(FfiArenaError::LengthMismatch);
    }
    Ok(())
}

/// Copies the value behind `key` into the caller's buffer.
///
/// The buffer is plain bytes, so it needs no particular alignment.
///
/// # Safety
///
/// `out` must be valid for writes of `out_len` bytes and must not
/// overlap memory owned by `arena`.
pub unsafe fn read_value<A>(
    arena: &A,
    key: u64,
    out: *mut u8,
    out_len: usize,
) -> Result<(), FfiArenaError>
where
    A: FfiKeyed,
    A::Value: IntoBytes + Immutable,
{
    check_one::<A::Value>(out, out_len)?;
    let value = arena.ffi_get(key).ok_or(FfiArenaError::StaleKey)?;
    // SAFETY: the caller guarantees `out` is valid for `out_len` byte
    // writes and disjoint from the arena; null and length were checked
    // above.
    let dst = unsafe { core::slice::from_raw_parts_mut(out, out_len) };
    dst.copy_from_slice(value.as_bytes());
    Ok(())
}

/// Overwrites the value behind `key` with the caller's bytes.
///
/// `T: FromBytes` means every bit pattern is a valid `T`, so no input
/// can build an invalid value. The old value is overwritten without a
/// destructor run, which is sound because `IntoBytes + FromBytes` types
/// are plain data.
///
/// # Safety
///
/// `src` must be valid for reads of `src_len` bytes and must not
/// overlap memory owned by `arena`.
pub unsafe fn write_value<A>(
    arena: &mut A,
    key: u64,
    src: *const u8,
    src_len: usize,
) -> Result<(), FfiArenaError>
where
    A: FfiKeyed,
    A::Value: FromBytes + IntoBytes,
{
    check_one::<A::Value>(src, src_len)?;
    let value = arena.ffi_get_mut(key).ok_or(FfiArenaError::StaleKey)?;
    // SAFETY: the caller guarantees `src` is valid for `src_len` byte
    // reads and disjoint from the arena; null and length were checked
    // above.
    let bytes = unsafe { core::slice::from_raw_parts(src, src_len) };
    value.as_mut_bytes().copy_from_slice(bytes);
    Ok(())
}

/// A one-element span over the value behind `key`, for callers that
/// want to read in place instead of copying.
///
/// The pointer is valid until the next call that takes `&mut` on the
/// arena (`insert` can grow and move the slot storage, `remove` ends
/// the value's life) or until the arena drops.
pub fn value_span<A>(arena: &A, key: u64) -> Result<ArenaSpan, FfiArenaError>
where
    A: FfiKeyed,
    A::Value: IntoBytes + Immutable,
{
    if core::mem::size_of::<A::Value>() == 0 {
        return Err(FfiArenaError::ZeroSized);
    }
    let value = arena.ffi_get(key).ok_or(FfiArenaError::StaleKey)?;
    Ok(ArenaSpan::from_slice(core::slice::from_ref(value)))
}

#[cfg(feature = "bump")]
impl<T: IntoBytes + Immutable> BumpArena<T> {
    /// One [`ArenaSpan`] per non-empty region, oldest region first, so
    /// walking the spans in order visits values in allocation order, the
    /// same order `iter_mut` uses.
    ///
    /// Takes `&mut self`. That rules out any `&mut T` from an earlier
    /// `alloc` still being alive when C starts reading, which a `&self`
    /// version could not promise.
    ///
    /// Each span covers the values allocated so far in its region, so
    /// the set of spans is a snapshot. Regions never move, so later
    /// `alloc` calls leave existing spans valid (they only write past
    /// each span's end). Spans become invalid on `reset` or drop, and a
    /// Rust caller must not call `iter_mut` or `reset` while C is still
    /// reading.
    pub fn region_spans(&mut self) -> impl Iterator<Item = ArenaSpan> + '_ {
        self.initialized_regions()
            .map(ArenaSpan::from_slice)
            .filter(|span| !span.is_empty())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zerocopy::KnownLayout;

    #[derive(Debug, Clone, Copy, PartialEq, FromBytes, IntoBytes, Immutable, KnownLayout)]
    #[repr(C)]
    struct Pair {
        a: u32,
        b: u32,
    }

    fn bytes_of(p: &Pair) -> [u8; 8] {
        let mut out = [0u8; 8];
        out.copy_from_slice(p.as_bytes());
        out
    }

    #[test]
    fn span_layout_is_three_words() {
        assert_eq!(
            core::mem::size_of::<ArenaSpan>(),
            3 * core::mem::size_of::<usize>()
        );
        assert_eq!(core::mem::align_of::<ArenaSpan>(), core::mem::align_of::<usize>());
    }

    #[test]
    fn span_from_empty_slice_is_the_null_sentinel() {
        let empty: [u32; 0] = [];
        let s = ArenaSpan::from_slice(&empty);
        assert!(s.is_empty());
        assert!(s.ptr.is_null());
        assert_eq!(s, ArenaSpan::empty());
    }

    #[test]
    fn span_from_slice_reports_stride_and_count() {
        let v = [Pair { a: 1, b: 2 }, Pair { a: 3, b: 4 }];
        let s = ArenaSpan::from_slice(&v);
        assert_eq!(s.stride, 8);
        assert_eq!(s.count, 2);
        assert_eq!(s.ptr, v.as_ptr().cast::<u8>());
    }

    #[test]
    fn slot_read_round_trips() {
        let mut arena = SlotArena::new();
        let key = arena.insert(Pair { a: 7, b: 9 });
        let mut out = [0u8; 8];
        // SAFETY: `out` is a live 8-byte local, disjoint from the arena.
        unsafe { read_value(&arena, key.as_ffi(), out.as_mut_ptr(), out.len()) }.unwrap();
        assert_eq!(out, bytes_of(&Pair { a: 7, b: 9 }));
    }

    #[test]
    fn slot_read_accepts_an_unaligned_buffer() {
        let mut arena = SlotArena::new();
        let key = arena.insert(Pair { a: 0x0102_0304, b: 0x0506_0708 });
        let mut backing = [0u8; 9];
        // Offset 1 is misaligned for u32 on every real target.
        // SAFETY: `backing[1..9]` is 8 live bytes, disjoint from the arena.
        unsafe { read_value(&arena, key.as_ffi(), backing.as_mut_ptr().add(1), 8) }.unwrap();
        assert_eq!(&backing[1..], &bytes_of(&Pair { a: 0x0102_0304, b: 0x0506_0708 }));
        assert_eq!(backing[0], 0);
    }

    #[test]
    fn slot_read_rejects_bad_buffers_before_touching_the_key() {
        let mut arena = SlotArena::new();
        let key = arena.insert(Pair { a: 1, b: 2 });
        let mut out = [0u8; 16];
        // SAFETY (all calls below): pointers are live locals or null,
        // which `read_value` rejects before any dereference.
        let null = unsafe { read_value(&arena, key.as_ffi(), core::ptr::null_mut(), 8) };
        assert_eq!(null, Err(FfiArenaError::NullPointer));
        let short = unsafe { read_value(&arena, key.as_ffi(), out.as_mut_ptr(), 7) };
        assert_eq!(short, Err(FfiArenaError::LengthMismatch));
        let long = unsafe { read_value(&arena, key.as_ffi(), out.as_mut_ptr(), 9) };
        assert_eq!(long, Err(FfiArenaError::LengthMismatch));
        assert_eq!(out, [0u8; 16], "a rejected call must not write");
    }

    #[test]
    fn slot_read_rejects_zero_sized_values() {
        let mut arena = SlotArena::new();
        let key = arena.insert(());
        let mut out = [0u8; 1];
        // SAFETY: `out` is a live local; ZST is rejected before any copy.
        let r = unsafe { read_value(&arena, key.as_ffi(), out.as_mut_ptr(), 0) };
        assert_eq!(r, Err(FfiArenaError::ZeroSized));
        assert_eq!(value_span(&arena, key.as_ffi()), Err(FfiArenaError::ZeroSized));
    }

    #[test]
    fn slot_read_reports_removed_reused_and_bogus_keys_as_stale() {
        let mut arena = SlotArena::new();
        let first = arena.insert(Pair { a: 1, b: 1 });
        arena.remove(first);
        let mut out = [0u8; 8];

        // SAFETY: `out` is a live 8-byte local, disjoint from the arena.
        let removed = unsafe { read_value(&arena, first.as_ffi(), out.as_mut_ptr(), 8) };
        assert_eq!(removed, Err(FfiArenaError::StaleKey));

        let second = arena.insert(Pair { a: 2, b: 2 });
        let reused = unsafe { read_value(&arena, first.as_ffi(), out.as_mut_ptr(), 8) };
        assert_eq!(reused, Err(FfiArenaError::StaleKey), "old generation must not alias the new value");
        assert!(unsafe { read_value(&arena, second.as_ffi(), out.as_mut_ptr(), 8) }.is_ok());

        let bogus = unsafe { read_value(&arena, u64::MAX, out.as_mut_ptr(), 8) };
        assert_eq!(bogus, Err(FfiArenaError::StaleKey));
        assert_eq!(out, bytes_of(&Pair { a: 2, b: 2 }));
    }

    #[test]
    fn slot_write_round_trips_and_lands_in_the_arena() {
        let mut arena = SlotArena::new();
        let key = arena.insert(Pair { a: 0, b: 0 });
        let src = bytes_of(&Pair { a: 11, b: 22 });
        // SAFETY: `src` is a live 8-byte local, disjoint from the arena.
        unsafe { write_value(&mut arena, key.as_ffi(), src.as_ptr(), src.len()) }.unwrap();
        assert_eq!(arena.get(key), Some(&Pair { a: 11, b: 22 }));
    }

    #[test]
    fn slot_write_rejects_bad_input_and_stale_keys_without_writing() {
        let mut arena = SlotArena::new();
        let key = arena.insert(Pair { a: 5, b: 6 });
        let src = [0xFFu8; 8];

        // SAFETY (all calls below): `src` is a live local; null is
        // rejected before any dereference.
        let null = unsafe { write_value(&mut arena, key.as_ffi(), core::ptr::null(), 8) };
        assert_eq!(null, Err(FfiArenaError::NullPointer));
        let short = unsafe { write_value(&mut arena, key.as_ffi(), src.as_ptr(), 4) };
        assert_eq!(short, Err(FfiArenaError::LengthMismatch));
        let bogus = unsafe { write_value(&mut arena, u64::MAX, src.as_ptr(), 8) };
        assert_eq!(bogus, Err(FfiArenaError::StaleKey));
        assert_eq!(arena.get(key), Some(&Pair { a: 5, b: 6 }));

        arena.remove(key);
        let gone = unsafe { write_value(&mut arena, key.as_ffi(), src.as_ptr(), 8) };
        assert_eq!(gone, Err(FfiArenaError::StaleKey));
    }

    #[test]
    fn slot_value_span_points_at_the_live_value() {
        let mut arena = SlotArena::new();
        let key = arena.insert(Pair { a: 3, b: 4 });
        let span = value_span(&arena, key.as_ffi()).unwrap();
        assert_eq!(span.count, 1);
        assert_eq!(span.stride, 8);
        let live = arena.get(key).unwrap() as *const Pair as *const u8;
        assert_eq!(span.ptr, live);
        assert_eq!(value_span(&arena, u64::MAX), Err(FfiArenaError::StaleKey));
    }

    #[cfg(feature = "compact")]
    #[test]
    fn compact_round_trips_and_rejects_stale_keys() {
        let mut arena = CompactSlotArena::new();
        let key = arena.insert(Pair { a: 8, b: 9 });
        let mut out = [0u8; 8];
        // SAFETY: `out` is a live 8-byte local, disjoint from the arena.
        unsafe { read_value(&arena, key.as_ffi(), out.as_mut_ptr(), 8) }.unwrap();
        assert_eq!(out, bytes_of(&Pair { a: 8, b: 9 }));

        let src = bytes_of(&Pair { a: 1, b: 2 });
        unsafe { write_value(&mut arena, key.as_ffi(), src.as_ptr(), 8) }.unwrap();
        assert_eq!(arena.get(key), Some(&Pair { a: 1, b: 2 }));

        arena.remove(key);
        let stale = unsafe { read_value(&arena, key.as_ffi(), out.as_mut_ptr(), 8) };
        assert_eq!(stale, Err(FfiArenaError::StaleKey));
    }

    #[cfg(feature = "unchecked")]
    #[test]
    fn unchecked_round_trips_and_rejects_oversized_keys() {
        let mut arena = UncheckedSlotArena::new();
        let index = arena.insert(Pair { a: 4, b: 5 });
        let mut out = [0u8; 8];
        // SAFETY: `out` is a live 8-byte local, disjoint from the arena.
        unsafe { read_value(&arena, u64::from(index), out.as_mut_ptr(), 8) }.unwrap();
        assert_eq!(out, bytes_of(&Pair { a: 4, b: 5 }));

        // A key with high bits set must not truncate onto a live index.
        let aliased = (1u64 << 32) | u64::from(index);
        let r = unsafe { read_value(&arena, aliased, out.as_mut_ptr(), 8) };
        assert_eq!(r, Err(FfiArenaError::StaleKey));
    }

    #[cfg(feature = "bump")]
    mod bump {
        use super::*;

        extern crate std;
        use std::vec::Vec;

        fn collect(arena: &mut BumpArena<u32>) -> Vec<u32> {
            let mut out = Vec::new();
            for span in arena.region_spans() {
                assert_eq!(span.stride, 4);
                // SAFETY: the span covers initialized `u32`s in a live
                // region, and nothing mutates the arena while this reads.
                let slice =
                    unsafe { core::slice::from_raw_parts(span.ptr.cast::<u32>(), span.count) };
                out.extend_from_slice(slice);
            }
            out
        }

        #[test]
        fn fresh_and_reset_arenas_yield_no_spans() {
            let mut arena: BumpArena<u32> = BumpArena::with_capacity(4);
            assert_eq!(arena.region_spans().count(), 0);
            arena.alloc(1);
            arena.reset();
            assert_eq!(arena.region_spans().count(), 0);
        }

        #[test]
        fn spans_cover_every_value_in_allocation_order() {
            let mut arena: BumpArena<u32> = BumpArena::with_capacity(2);
            for i in 0..9u32 {
                arena.alloc(i * 10);
            }
            assert!(arena.region_count() >= 2, "test needs more than one region");
            let expected: Vec<u32> = (0..9u32).map(|i| i * 10).collect();
            assert_eq!(collect(&mut arena), expected);
            let span_total: usize = arena.region_spans().map(|s| s.count).sum();
            assert_eq!(span_total, arena.len());
        }

        #[test]
        fn existing_spans_stay_valid_across_later_allocations() {
            let mut arena: BumpArena<u32> = BumpArena::with_capacity(2);
            for i in 0..2u32 {
                arena.alloc(i + 100);
            }
            let first: Vec<ArenaSpan> = arena.region_spans().collect();
            assert_eq!(first.len(), 1);

            // Forces a second region and more writes after the first
            // span was taken.
            for i in 0..10u32 {
                arena.alloc(i);
            }
            // SAFETY: regions never move and the first region's initialized
            // prefix is never rewritten by `alloc`.
            let old = unsafe { core::slice::from_raw_parts(first[0].ptr.cast::<u32>(), first[0].count) };
            assert_eq!(old, &[100, 101]);
        }

        #[test]
        fn spans_taken_after_growth_reflect_the_new_snapshot() {
            let mut arena: BumpArena<u32> = BumpArena::with_capacity(4);
            arena.alloc(1);
            let before: usize = arena.region_spans().map(|s| s.count).sum();
            arena.alloc(2);
            let after: usize = arena.region_spans().map(|s| s.count).sum();
            assert_eq!((before, after), (1, 2));
        }
    }
}
