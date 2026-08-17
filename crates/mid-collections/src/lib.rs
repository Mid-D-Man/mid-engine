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

#![no_std]
extern crate alloc;

pub mod generational_index;
pub mod sparse_set;

pub use generational_index::{GenerationalIndex, GenerationalIndexAllocator};
pub use sparse_set::{SparseSet, SparseSetIndex};
