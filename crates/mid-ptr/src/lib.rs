// crates/mid-ptr/src/lib.rs
//! Type-erased raw pointer wrappers, ported from Bevy's `bevy_ptr` crate
//! (MIT/Apache-2.0) and adapted to this workspace's own conventions.
//!
//! The crate gives mid-engine a small, `no_std`, zero-dependency toolkit for
//! holding onto a pointer to *some* value without the compiler knowing what type
//! that value actually is, while still tracking the invariants a raw pointer on its
//! own throws away: whether it's still properly aligned, how long it stays valid
//! for, and who is allowed to read, write, or drop through it.
//!
//! Three families of type live here:
//!
//! - [`Ptr`], [`PtrMut`], and [`OwningPtr`] are the borrow-shaped, mutable-borrow-
//!   shaped, and owning-pointer-shaped erased pointers, mirroring `&T`, `&mut T`,
//!   and `Box<T>` respectively, minus the compile-time type information.
//! - [`MovingPtr`] moves a value to a new location without ever passing it by
//!   value, and can be deconstructed into per-field `MovingPtr`s via
//!   [`deconstruct_moving_ptr`] — useful for things like migrating a component's
//!   bytes from one archetype table column to another without an extra copy.
//! - [`ThinSlicePtr`] is a `&[T]` with the length stripped out, for callers that
//!   already know (or separately track) how long the slice is.
//!
//! [`Aligned`] and [`Unaligned`] are marker types threaded through all of the above
//! as a type parameter, so the alignment guarantee (or lack of one) a given pointer
//! carries is visible in its type rather than only in a doc comment.
//!
//! Every public item here is `unsafe fn` or backed by one internally — this crate
//! exists to make the unsafety explicit and centrally reviewed, not to remove it.
//! See `docs/mid-ptr.md` for the full port notes, including which parts of the
//! upstream crate this leaves out and why.

#![no_std]
// This crate's own [lints] workspace = true (Cargo.toml) pulls in the root
// workspace's `unsafe_code = "deny"` — opted back out of here, explicitly, per
// docs/RUST_AND_CRATE_GUIDELINES.md §3: mid-ptr's entire purpose is wrapping raw
// pointers, so the exception is this one visible line rather than the crate
// silently failing to compile. mid-math was the only crate that had opted into
// `[lints] workspace = true` before this one; every `unsafe` block below carries
// its own `// SAFETY:` comment or `# Safety` doc section, matching the model
// mid-math's `ffi/camera.rs` set, rather than adding to the backlog
// `undocumented_unsafe_blocks = "warn"` is tracking for mid-math.
#![allow(unsafe_code)]

mod aligned;
mod const_non_null;
mod debug_align;
mod erased;
mod moving;
mod moving_macros;
mod thin_slice;
mod unsafe_cell;

pub use aligned::{Aligned, IsAligned, Unaligned};
pub use const_non_null::ConstNonNull;
pub use erased::{OwningPtr, Ptr, PtrMut};
pub use moving::MovingPtr;
pub use thin_slice::ThinSlicePtr;
pub use unsafe_cell::UnsafeCellDeref;
