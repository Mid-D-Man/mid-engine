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

#![no_std]
extern crate alloc;

pub mod sparse_set;

pub use sparse_set::{SparseSet, SparseSetIndex};
