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
//! - [`raw_alloc`] — [`RawAlloc`], the shared, always-fallible
//!   allocation interface combinators build against, plus
//!   [`HeapAlloc`], the natural terminal allocator for a combinator
//!   chain. A real, stated simplification of `foonathan::memory`'s
//!   `RawAllocator`/`allocator_traits` concept -- see that module's
//!   doc comment for exactly what carries over and what doesn't.
//! - `fallback` (behind the `fallback` feature) —
//!   [`fallback::FallbackAllocator<Primary, Secondary>`]: try
//!   `Primary`, fall back to `Secondary` on failure, routing
//!   deallocation back to whichever side actually owns a given
//!   pointer. Directly modeled on foonathan's `fallback_allocator`.
//! - `segregator` (behind the `segregator` feature) —
//!   [`segregator::Segregator<Small, Large>`]: routes an allocation to
//!   `Small` or `Large` purely by comparing its size against a
//!   threshold, no fallthrough if the chosen side fails -- a real,
//!   deliberate difference from `FallbackAllocator`'s try-then-fall-
//!   back behavior, not an oversight. Directly modeled on foonathan's
//!   `binary_segregator`.
//! - `tracking` (behind the `tracking` feature) —
//!   [`tracking::Tracked<A>`]: wraps any `RawAlloc` with lock-free
//!   `AtomicU64` counters (allocation/deallocation counts, current and
//!   peak bytes, current and peak live count). Combines two real
//!   sources: the hook shape from foonathan's `tracked_allocator`, the
//!   counters and their update sequence ported directly from
//!   `mod_alloc::ModAlloc` (crates.io, MSRV 1.75).
//! - `sync` (behind the `sync` feature) — [`sync::SpinLock<T>`], a
//!   `no_std` mutual-exclusion primitive grounded in the real `spin`
//!   crate's algorithm, plus [`sync::SyncAlloc<A>`], which uses it to
//!   turn any `RawAlloc` into one safe to share across threads.
//!   Matches Zig's `std.heap.ThreadSafeAllocator`'s real shape (from
//!   this crate's own Zig re-survey): lock, forward, unlock.
//! - `backed` (behind the `backed` feature) —
//!   [`backed::BackedStack<'p, P>`]: a `StackAllocator`-shaped bump
//!   allocator whose backing block is provisioned by a parent
//!   `RawAlloc` instead of the global allocator -- one allocator
//!   carving its buffer out of another, rather than every allocator
//!   reaching for the heap independently.
//! - `bump_vec` (behind the `bump_vec` feature) —
//!   [`bump_vec::BumpVec<'p, T, P>`]: a growable, `Vec`-shaped
//!   collection backed by any `RawAlloc`. The real, correct home for
//!   "does this crate have something like `bumpalo::collections::Vec`"
//!   -- `mid-arena`'s `BumpArena<T>` cannot host one soundly (see this
//!   module's own doc comment for exactly why), `RawAlloc` can, for
//!   the same real reason `bumpalo::Bump` can.
//!
//! # Module plan
//! Nothing left uncatalogued. Every module foonathan/memory's own real
//! source suggested has shipped, both items the Zig re-survey pass
//! added (`sync`, `backed`) have too, and `bump_vec` closes the one
//! real question left over from `mid-arena`'s own feature-gap pass.
//! See `docs/mid-alloc.md` for the full history and every real source
//! each module traces back to.

#![no_std]
extern crate alloc;

pub mod raw_alloc;
pub mod stack_allocator;

#[cfg(feature = "pool")]
pub mod pool_allocator;

#[cfg(feature = "fallback")]
pub mod fallback;

#[cfg(feature = "segregator")]
pub mod segregator;

#[cfg(feature = "tracking")]
pub mod tracking;

#[cfg(feature = "sync")]
pub mod sync;

#[cfg(feature = "backed")]
pub mod backed;

#[cfg(feature = "bump_vec")]
pub mod bump_vec;

pub use raw_alloc::{HeapAlloc, NullAlloc, RawAlloc};
pub use stack_allocator::{StackAllocator, StackMarker};

#[cfg(feature = "pool")]
pub use pool_allocator::PoolAllocator;

#[cfg(feature = "fallback")]
pub use fallback::FallbackAllocator;

#[cfg(feature = "segregator")]
pub use segregator::Segregator;

#[cfg(feature = "tracking")]
pub use tracking::{AllocStats, Tracked};

#[cfg(feature = "sync")]
pub use sync::{SpinLock, SpinLockGuard, SyncAlloc};

#[cfg(feature = "backed")]
pub use backed::{BackedStack, BackedStackMarker};

#[cfg(feature = "bump_vec")]
pub use bump_vec::BumpVec;
