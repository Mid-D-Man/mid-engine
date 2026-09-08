// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-alloc.md, section "lib.rs"
// ============================================================================
//! mid-alloc — composable allocator strategies for Mid Engine, built
//! from a real source read of foonathan/memory (C++, the library this
//! survey was pointed at directly) and the Rust `GlobalAlloc`-adapter
//! ecosystem (`dhat`, `mod-alloc`). See `docs/mid-alloc.md` for the
//! full survey and the reasoning behind every module below.
//!
//! `#![no_std]` + `alloc`, zero external dependencies in the default
//! build — same stance as `mid-arena`/`mid-collections`, same reason.
//!
//! # Why this is a separate crate from `mid-arena`
//!
//! `mid-arena` is scoped to arena/slot allocators specifically
//! (generational value storage, chunked bump allocation). What
//! foonathan/memory's real structure showed is a second, genuinely
//! different thing worth having its own crate for: **composable
//! allocator combinators** — a shared interface, a handful of base
//! strategies (stack, pool), and wrapper types that combine them
//! (fallback, size-based routing, tracking) rather than each strategy
//! being an island. That composability is the actual reusable idea
//! `mid-alloc` exists to bring in, not "a few more allocator structs."
//!
//! # Modules
//! - [`stack_allocator`] — the first piece, built for real.
//!   Fixed-capacity, marker/rewind bump allocator for per-frame/
//!   per-scope scratch storage, directly modeled on foonathan/memory's
//!   `memory_stack`, with `bumpalo`'s `Cell`-based interior-mutability
//!   pattern applied so multiple allocations can be held live
//!   simultaneously. See that module's doc comment for the full design
//!   and its real, checked tradeoffs.
//! - `pool_allocator` (behind the `pool` feature) — fixed-node-size,
//!   free-list [`PoolAllocator<T>`](pool_allocator::PoolAllocator):
//!   typed `create`/`destroy` (Zig's `std.heap.MemoryPool` shape, real
//!   source read), backed by an owned, chunk-linked region chain
//!   (`mid-arena`'s `BumpArena` pattern, reimplemented locally) with
//!   `mid-arena`'s own `CompactSlotArena` union trick threading the
//!   free list through unused slots. See that module's doc comment for
//!   the full design and why it departs from foonathan's `memory_pool`
//!   shape on purpose.
//!
//! # Module plan (catalogued in docs/mid-alloc.md, not yet built)
//! - **`fallback`** — a generic `FallbackAllocator<Primary, Secondary>`
//!   combinator: try `Primary`, fall back to `Secondary` on failure.
//!   Directly modeled on foonathan's `fallback_allocator` — a real
//!   ~10-line dispatch function in the source, not a complex feature.
//! - **`segregator`** — routes an allocation to one of several
//!   allocators based on its size (small → pool, large → heap), modeled
//!   on foonathan's `segregator`.
//! - **`tracking`** — wraps any allocator with alloc/dealloc hooks for
//!   profiling, matching both foonathan's `tracking_allocator` and
//!   Rust's own `dhat`/`mod-alloc` `GlobalAlloc`-wrapper pattern — the
//!   same design in both ecosystems, confirmed by reading both, not
//!   assumed from the resemblance alone.
//!
//! Every one of these traces to a specific real function this survey
//! actually read (`docs/mid-alloc.md`), not to "allocator libraries
//! tend to have this."

#![no_std]
extern crate alloc;

pub mod stack_allocator;

#[cfg(feature = "pool")]
pub mod pool_allocator;

pub use stack_allocator::{StackAllocator, StackMarker};

#[cfg(feature = "pool")]
pub use pool_allocator::PoolAllocator;
