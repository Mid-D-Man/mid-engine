//! mid-collections — hand-rolled data structures for Mid Engine.
//!
//! Not a general-purpose collections crate. Built piecemeal, one structure
//! at a time, only when `mid-ecs` actually needs it — see
//! `docs/mid-collections.md` for the full ranked list and the reasoning
//! behind each entry. `mid-geom`'s own history is the model: gaps get
//! filled when a real consumer needs them, not speculatively.
//!
//! `#![no_std]` + `alloc` on purpose, matching `mid-common` — this crate
//! sits low enough in the dependency graph (under `mid-ecs`, which has to
//! run on `wasm32` in-browser as well as native) that it shouldn't assume
//! a `std` environment it doesn't actually need. Every structure here is
//! built on `alloc::vec::Vec` alone — zero external dependencies, not just
//! minimal ones.
//!
//! # Modules
//! - `sparse_set` — the first piece, and the foundation the others build
//!   on. O(1) insert/remove/lookup, contiguous iteration over live
//!   elements, no tombstones. This is the storage mid-ecs's "Sparse Shell"
//!   (volatile/toggle components — status effects, tags, anything added
//!   and removed constantly) is built on; see `docs/mid-ecs.md`'s Hybrid
//!   ECS Architecture section for how it fits alongside the Archetype
//!   Core.
//! - `generational_index` — the second piece. Issues handles that detect
//!   their own staleness after the slot they point at is freed and
//!   reused — `mid-ecs`'s `Entity` type and `World::spawn`/`despawn` are
//!   a thin wrapper over this. Implements `SparseSetIndex` directly, so
//!   it composes with `sparse_set` above with no adapter needed.
//! - `ffi_span` — third piece, **behind the `ffi` feature, off by
//!   default.** `docs/mid-collections.md`'s "FFI wrapper" section calls
//!   for building this on `zerocopy` — a real external dependency,
//!   which is real tension against this crate's own "zero external
//!   dependencies, not just minimal ones" line two paragraphs up.
//!   Resolved as a feature gate rather than by picking one doc over the
//!   other: a plain `cargo build -p mid-collections` (what every
//!   non-FFI consumer, including every `wasm32` build of `mid-ecs` that
//!   never touches the boundary, actually does) stays exactly as
//!   zero-dependency as stated; `zerocopy` only enters the graph for
//!   whichever crate explicitly opts in with `features = ["ffi"]` —
//!   `mid-net`'s and `mid-ecs`'s own `ffi.rs` files, specifically.
//!   Matches this workspace's own established precedent for exactly
//!   this shape of problem (`rayon` gated to non-`wasm32` targets in
//!   `mid-ecs`; `tokio`/`reqwest` gated behind DixScript-Rust's
//!   `cloud-import` feature) rather than introducing a new pattern.

#![no_std]
extern crate alloc;

#[cfg(feature = "ffi")]
pub mod ffi_span;
pub mod generational_index;
pub mod sparse_set;

#[cfg(feature = "ffi")]
pub use ffi_span::{checked_slice, checked_slice_mut, FfiBufError, FfiSpan};
pub use generational_index::{GenerationalIndex, GenerationalIndexAllocator};
pub use sparse_set::{SparseSet, SparseSetIndex};
