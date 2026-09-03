// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-arena.md, section "lib.rs"
// ============================================================================
//! mid-arena — arena/slot allocators for Mid Engine, built from a real
//! survey of the Rust arena-crate ecosystem (28 crates) and three
//! established C arena/pool libraries, rather than from a single
//! reference implementation. See `docs/mid-arena.md` for the full
//! comparison, the real benchmark numbers behind every claim below,
//! and the reasoning behind every scope decision.
//!
//! `#![no_std]` + `alloc`, zero external dependencies in the default
//! build — matching `mid-collections`' own precedent exactly, for the
//! same reason: this sits low enough in the dependency graph (a
//! plausible `mid-ecs`/`mid-net`/`tools/mdix-compiler` consumer, and
//! anything that has to run on `wasm32` in-browser) that it shouldn't
//! assume a `std` environment it doesn't need.
//!
//! # What this is, and isn't, for
//!
//! Motivated by `mid-ecs` wanting an arena allocator, but scoped wider
//! on purpose (`docs/mid-arena.md`'s survey covers general-purpose
//! object storage, not just ECS component columns) — see that doc's
//! "Relationship to mid-collections' GenerationalIndex" section for the
//! honest version of where this does and doesn't touch `mid-ecs` today
//! (nowhere yet; candidate consumers are listed, not wired in).
//!
//! # Modules
//! - [`slot_arena`] — the first piece, built for real. Generational,
//!   value-storing [`SlotArena<T>`](slot_arena::SlotArena), directly
//!   extending `mid_collections::GenerationalIndexAllocator`'s own
//!   verified even/odd-generation LIFO-freelist algorithm to actually
//!   own a `T` per slot. See that module's doc comment for the full
//!   design.
//! - `bump_arena` (behind the `bump` feature) — single-typed,
//!   chunk-linked bump allocator, `BumpArena<T>`. See that module's doc
//!   comment for the full design.
//!
//! # Feature gates (`bump` built, rest still planned — see
//! `docs/mid-arena.md` "Feature gates" for the reasoning behind each)
//! - `compact` — `slotmap`-style unsafe union slot layout, trading the
//!   enum tag's per-slot overhead away for types where that matters.
//! - `intern` — hashset-of-boxes dedup arena (`internment`'s
//!   `ArenaIntern` approach), for string/path/asset-key interning.
//! - `concurrent` — sharded lock-free slab (`sharded-slab`'s approach).
//!   Deliberately not default: this crate's own real benchmark shows it
//!   costing roughly an order of magnitude more than plain `slab` when
//!   accessed from a single thread, matching `sharded-slab`'s own
//!   documented caveat that the lock-free design only pays for itself
//!   once actually shared across threads.
//! - `ffi` — checked FFI access, matching `mid_collections`'s own `ffi`
//!   feature shape (optional `zerocopy` dependency, off by default).
//!   [`ArenaKey::as_ffi`](slot_arena::ArenaKey::as_ffi)/`from_ffi`
//!   already exist unconditionally today (cheap, no dependency) — this
//!   feature is specifically for a `checked_slice`-equivalent over
//!   arena-owned memory, not built yet.
//!
//! # Explicitly out of scope: garbage collection
//! `gc`, `gc-arena`, `shredder`, and `elise` (`docs/mid-arena.md`'s
//! survey) all solve a real problem — but a tracing GC's collection
//! pause is, by construction, not a cost a caller can bound in advance.
//! That's in direct conflict with `docs/architecture.md`'s hard
//! 128 Hz/60 Hz frame budgets: a single collection spike landing inside
//! a physics or network tick isn't a slowdown to optimize later, it's a
//! dropped frame. If a scripting sandbox ever needs real tracing-GC
//! semantics, that belongs in its own crate with its own explicit
//! latency contract, not blended into an allocator every other system
//! is assumed to be able to call without a pause budget.

#![no_std]
extern crate alloc;

pub mod slot_arena;

#[cfg(feature = "bump")]
pub mod bump_arena;

pub use slot_arena::{ArenaKey, SlotArena};

#[cfg(feature = "bump")]
pub use bump_arena::BumpArena;
