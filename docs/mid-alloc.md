# mid-alloc

Composable allocator strategies for Mid Engine. Split out from
`mid-arena` on purpose: `mid-arena` stays scoped to arena/slot
allocators (generational value storage, chunked bump allocation);
`mid-alloc` is for everything else the C++ survey turned up —
specifically the **combinator** pattern (a shared interface, a few base
strategies, and wrapper types that compose them) that turned out to be
the real reusable idea in `foonathan/memory`, not just "a few more
allocator structs to clone."

## `lib.rs`

`#![no_std]` + `alloc`, zero mandatory dependencies. Wires up each
module behind its own Cargo feature once built (`stack_allocator` and
`raw_alloc` are unconditional -- the latter joined the former once
`fallback` needed a shared interface to build against; `pool_allocator`
behind `pool`; `fallback` behind `fallback`) and re-exports each
module's public type at the crate root. The crate-level doc comment
there is the short version of "what's built vs. planned" — this file
is the long version, with the actual survey and real source behind
every claim.

## Survey

Pointed at `github.com/foonathan/memory` directly (cloned, real source
read — not the docs, not a blog post about it) plus a quick, targeted
check of `dhat` and `mod-alloc` on crates.io to confirm a specific,
named parallel (tracking/profiling allocator adapters exist in both
ecosystems as the same pattern). Full repo structure:

**Base strategies** — `heap_allocator`, `malloc_allocator`,
`new_allocator` (thin OS/libc wrappers), `static_allocator` (fixed
buffer), `memory_stack` (marker/rewind bump — what `mid-alloc` built
first, see below), `memory_pool`/`memory_pool_collection`/
`memory_pool_type` (fixed-node-size free-list allocator, size-class
buckets), `memory_arena` (the general block-management abstraction
these strategies sit on top of), `temporary_allocator`/
`iteration_allocator` (narrower, scoped variants).

**Combinators** — `fallback_allocator` (try A, fall back to B),
`segregator` (route by allocation size to different allocators),
`joint_allocator` (allocate an object plus trailing extra memory in one
call), `allocator_storage`/`allocator_traits` (the shared interface/
type-erasure layer everything else is built against), `threading`
(synchronized vs. unsynchronized wrapper, same combinator shape as the
other two).

**Debug/interop** — `tracking` (wrap any allocator with alloc/dealloc
hooks), `debugging`/`debug_helpers` (fill-pattern poisoning for
use-after-free detection), `std_allocator`/`memory_resource_adapter`
(bridges to `std::allocator`/`std::pmr`), `smart_ptr`/`deleter`/
`container` (STL integration), `virtual_memory`/`lowlevel_allocator`
(pluggable raw block source — the same idea as tsoding/arena.h's
compile-time-selectable backend, `docs/mid-arena.md`'s C survey,
confirmed independently here rather than assumed to generalize).

## What the real source actually showed (not assumed from names alone)

**`memory_pool::allocate_node()`** (`memory_pool.hpp`): pop a node off
the free list; if the list is empty, allocate one new block from the
underlying (swappable) block allocator sized by `next_capacity()`, then
retry. Straightforward, and confirms the free-list-backed fixed-size
pool is exactly the shape it sounds like — no surprises here, unlike
`apr_pools.c`'s segregated-free-list design in the C survey.

**`memory_stack`/`stack_marker`** (`memory_stack.hpp`): a plain bump
allocator over a fixed block, plus a `stack_marker` capturing the
current bump position and an `unwind(marker)` that resets it. This is
the classic frame/scratch-allocator pattern (allocate temporaries during
a scope, rewind them all at once) — directly what `mid-alloc` built
first as `StackAllocator` (below).

**`fallback_allocator<Default, Fallback>::allocate_node()`**
(`fallback_allocator.hpp`): tries `Default` via a `try_`-prefixed
variant that returns null on failure instead of throwing, falls back to
`Fallback` only if that returns null. About ten real lines — the
combinator pattern is genuinely simple once you see the real
implementation, not the architectural complexity the *idea* of
"composable allocators" might suggest.

**`tracking_allocator<Tracker, Allocator>`** (`tracking.hpp`): wraps any
allocator, calls `Tracker::on_node_allocation()`/
`on_node_deallocation()`/etc. immediately before forwarding to the real
allocate/deallocate call. Checked this against `dhat`'s real Rust
`Alloc` struct (a unit struct implementing `GlobalAlloc`, same
before-forward hook shape in `alloc`/`dealloc`/`realloc`) — same
pattern in both ecosystems, confirmed by reading both sides, not
inferred from the resemblance alone. Also surfaced `mod-alloc`
(crates.io), an already-Rust, already-rustc-1.75-targeting "lean dhat
replacement" — worth reading before building `mid-alloc`'s own
`tracking` module, since it's solving the exact problem in the exact
language and toolchain floor this workspace already has.

**`segregator.hpp`** (real source read, upgraded from a name-only
listing): not a single N-way size table, a chain of *binary* decisions.
`threshold_segregatable<RawAllocator>` pairs a `size <= max_size`
predicate with an allocator; `binary_segregator<Segregatable,
RawAllocator>` tries the `Segregatable` first and falls to the second
allocator otherwise; `make_segregator(a, b, c, ...)` nests these
recursively so the last argument is the final fallback and everything
before it is tried in order. `null_allocator` (always fails) is the
default terminal fallback when none is given. Real shape to build
`mid-alloc`'s own `segregator` module against once it's started, not
assumed from the name.

## What's built: `StackAllocator`

`crates/mid-alloc/src/stack_allocator.rs`. Fixed-capacity, directly
modeled on `memory_stack`/`stack_marker` above, with one real,
necessary departure: `foonathan::memory_stack::allocate()` takes an
ordinary mutable `this` because C++ has no borrow checker to fight. A
literal Rust port taking `&mut self` per `alloc()` call would make the
allocator nearly useless — you could never hold a reference to an
earlier allocation while making a new one, which is the entire point of
a scratch allocator. Fixed by applying `bumpalo`'s own proven pattern
instead (already verified and benched — `docs/mid-arena.md`): a
`Cell<usize>` bump position, so `alloc(&self, ...)` returns `&'a mut T`
tied to the allocator's own lifetime, not to a per-call borrow.

Matches `bumpalo`'s other real tradeoff too, not selectively: no
per-value `Drop` tracking. `rewind()`/`reset()` reclaim bytes, not
destructors — tracking per-allocation type info to run `Drop` would
mean every allocation carries drop-glue, defeating the reason this
allocator exists. Fixed capacity, not chunk-linked, unlike `mid-arena`'s
planned `bump` feature: a scratch allocator's whole point is a known
budget reused every frame; growing on demand means either invalidating
live markers (unsound) or chunk-linking (real complexity this use case
doesn't need).

`resize_raw` added later (see "Fixes and Problems" below): grow or
shrink a previously-returned allocation in place when it is genuinely
the *most recent* one, directly ported from `std.heap.
FixedBufferAllocator`'s real `isLastAllocation`/`resize` (fresh source
read of `lib/std/heap/FixedBufferAllocator.zig`, not the version of
this file read in an earlier pass — Zig's own repo has since split what
used to live inside `heap.zig` into its own per-allocator files, see
the Survey section's own note on this). A non-last allocation can still
report a successful logical shrink (a smaller size is always a valid
view of the same bytes) but can never grow, since real live data may
sit immediately after it in the buffer.

**Tests:** 13, real, passing on rustc 1.75 — including one that actually
matters for trusting the `unsafe` in `alloc_raw()`:
`alignment_is_actually_respected_not_just_assumed` forces a misaligned
starting position with a 1-byte allocation first, then checks a
16-byte-aligned type's returned pointer against
`align_of::<T>()` directly, rather than trusting the arithmetic by
inspection. Also covers: multiple simultaneous live allocations (the
actual property the `Cell` design exists for), overflow returning the
value back unwritten rather than losing it, marker/rewind reclaiming
exactly the reserved range and no more, reset, that rewinding to the
start hands back the literal same address on the next allocation, a
`RawAlloc::try_dealloc_raw` pointer-ownership check (added alongside
`fallback`/`segregator`, see their own "What's built" sections), and
`resize_raw`'s four real cases (grow the last allocation in place,
shrink it and reclaim the tail for real rather than only logically,
a non-last allocation shrinking logically but refusing to grow, and a
grow past total capacity failing cleanly without moving anything).

**Verification honestly scoped, not overstated:** this sandbox's rustc
1.75 has no rustup/nightly component, so no Miri and no
AddressSanitizer were available to check the `unsafe` blocks — hand
review against `bumpalo`'s well-precedented pattern plus the tests
above is what backs this, not a stronger tool. Worth a real Miri pass
on a toolchain that has it before this ships anywhere that isn't itself
still under active development. Said plainly rather than left implied,
matching this project's own standard for what "verified" gets to mean.

## What's built: `PoolAllocator`

`crates/mid-alloc/src/pool_allocator.rs`, behind the `pool` feature.
Fixed-node-size, free-list allocator: typed `create()`/`destroy()`,
reuses a freed slot before growing. Two real sources pulled in
different directions here, and this module takes a real position on
both:

- **API shape** follows Zig's `std.heap.MemoryPool` (real source
  read), not `foonathan::memory_pool`'s `void* allocate_node()` +
  `allocator_traits` specialization. Zig's own maintainers' reasoning
  (a fixed-single-type pool already knows its size/alignment at
  compile time, so a byte-oriented generic interface buys nothing) is
  the reason given, not just "Zig did it this way."
- **Growth mechanic** follows `foonathan::memory_pool::allocate_node()`
  exactly (real source read this pass: pop the free list, or grow by
  one whole block sized for the next region and retry) and
  `std.heap.MemoryPool`'s real choice to grow from an *owned* arena
  rather than a raw block list — but the region chain itself is
  `mid-arena`'s own `BumpArena` pattern (`Cell<NonNull<RegionNode<T>>>`,
  geometric growth), reimplemented locally rather than taken as a
  dependency, keeping `mid-alloc` at zero mandatory dependencies
  including on its own sibling crate.
- **Free-slot storage** reuses `mid-arena`'s `CompactSlotArena` union
  trick (`union Slot<T> { value: ManuallyDrop<T>, next: ... }`) rather
  than inventing a new one.

Deliberately does **not** run `T`'s destructor for an item that's still
live when the pool itself drops — only an explicit `destroy()` call
does. Matches `std.heap.MemoryPool.deinit()`'s real behavior (Zig has
no destructors to run in the first place) and this crate's own
`StackAllocator` tradeoff for the same underlying reason: tracking
which of a pool's non-contiguous slots are still live would mean a
live/dead flag per slot, real cost a fixed-size pool exists to avoid.
Said directly in the module's own doc comment, with a test
(`drop_of_still_live_items_does_not_run_their_destructor`) that exists
specifically to keep that tradeoff honest going forward, not just
documented once and left to drift.

`with_capacity_bounded()` gives a non-growing pool (`create()` returns
`Err(value)` once full and the free list is empty), matching Zig's
real `Options{ .growable = false }` and `StackAllocator`'s own
fixed-budget precedent; `new()`/`with_capacity()` grow forever, Zig's
default.

**Tests:** 13, real, actually run on this sandbox's rustc 1.75 this
pass (`cargo test -p mid-alloc --features pool`) — a step up from
`StackAllocator`'s hand-review-only verification, since a working
`rustc`/`cargo` (matching the project's own MSRV floor via
`apt install rustc cargo`) happened to be available in this session's
sandbox. Covers: create/read-back, multiple simultaneous live
allocations, destroy running `Drop` exactly once and freeing the slot,
freed-slot reuse ahead of growth, LIFO reuse order across three
outstanding frees, growing past the first region, a bounded pool
refusing a third live slot then still reusing one it frees, 200 real
create/destroy cycles staying internally consistent, `Default`, the
no-drop-on-pool-drop tradeoff above, and zero-sized `T`. Existing
`StackAllocator` tests (9) re-run clean alongside these with no
regressions, both with and without the `pool` feature enabled.

**Verification honestly scoped:** same as `StackAllocator` — no Miri
or AddressSanitizer available (this sandbox's rustc has no
rustup/nightly component even though it now has a real `rustc`/`cargo`
via `apt`). Checked by hand against `CompactSlotArena`'s and
`BumpArena`'s already-shipped unsafe shapes, which this module
recombines rather than inventing new ones, plus the real, actually-run
tests above.

## What's built: `RawAlloc` / `HeapAlloc`

`crates/mid-alloc/src/raw_alloc.rs`, unconditional (joined
`stack_allocator` once `fallback` needed a shared interface to build
against). `RawAlloc` is a real, stated simplification of
`foonathan::memory`'s `RawAllocator`/`allocator_traits` concept, not an
attempt to port it in full:

- **Collapses the composable-vs-throwing split.**
  `fallback_allocator.hpp`'s real dispatch (source read) only reaches
  for the fallback allocator through a `try_`-prefixed, fallible
  variant kept separate from a plain one allowed to throw/abort. Every
  allocator this crate has is already `Option`-returning
  (`StackAllocator::alloc_raw`, `PoolAllocator::create`), so there is
  no second variant to distinguish — `RawAlloc::try_alloc_raw` is the
  only allocation function, always fallible.
- **Keeps real pointer-ownership checking on the deallocation side.**
  `memory_stack`'s own composable `try_deallocate_node` (source read)
  calls `state.arena_.owns(ptr)` and reports whether it handled the
  request, rather than unconditionally claiming every call.
  `RawAlloc::try_dealloc_raw` mirrors that exactly, `bool` return
  included — confirmed to matter, not just faithful for its own sake:
  it's what lets `FallbackAllocator` route a deallocation to whichever
  side actually owns a given pointer instead of guessing (see below).

`HeapAlloc` is a bare unit struct deferring to the global allocator via
`alloc`/`dealloc` directly, matching `foonathan::heap_allocator`'s own
role as a thin `malloc`/`free`-equivalent wrapper (source read). A real
limitation stated in its own doc comment rather than left implicit: the
global heap exposes no "is this pointer mine" query the way
`StackAllocator`'s own contiguous buffer does, so `HeapAlloc::
try_dealloc_raw` always claims and frees — sound only as the *last*
allocator in a chain, the same real property `foonathan::heap_allocator`
has and is used for the same way there.

`StackAllocator` itself gained a `RawAlloc` impl in the same pass:
`try_alloc_raw` forwards straight to the existing `alloc_raw`;
`try_dealloc_raw` checks whether `ptr` falls inside `self.buf`'s own
address range and reports that, doing nothing else, matching
`memory_stack`'s real `owns(ptr)` check directly rather than the naive
"always claim it" shortcut a first pass might reach for.

**Tests:** 4 new (`heap_alloc_round_trips_a_value`,
`heap_alloc_zero_sized_request_returns_a_dangling_aligned_pointer`,
`heap_alloc_rejects_a_non_power_of_two_alignment` in `raw_alloc.rs`;
`raw_alloc_try_dealloc_recognizes_only_its_own_pointers` in
`stack_allocator.rs`, added to that module's existing suite). All run
directly against this workspace's real `cargo test -p mid-alloc` on
rustc 1.75 — no scratch crate needed, since (unlike `mid-arena`)
`mid-alloc` carries no `criterion` dev-dependency to block it.

## What's built: `FallbackAllocator`

`crates/mid-alloc/src/fallback.rs`, behind the `fallback` feature.
`FallbackAllocator<Primary, Secondary>`: tries `Primary` first, falls
back to `Secondary` on failure, routing deallocation to whichever side
`try_dealloc_raw` reports actually owns a given pointer. Directly
modeled on `foonathan::fallback_allocator<Default, Fallback>`
(`fallback_allocator.hpp`, source read this pass) — its real dispatch
really is close to the "about ten real lines" the survey section above
already flagged, once the composable/throwing split collapses into
`RawAlloc`'s single always-fallible interface (see "What's built:
`RawAlloc`/`HeapAlloc`" above for that simplification stated in full).

**Tests:** 3, real, run alongside the rest of the crate:
`fallback_allocator_uses_secondary_once_primary_is_full` (both sides
actually written to and read back correctly, not just checked for
non-null), `fallback_allocator_routes_dealloc_to_whichever_side_really_owns_it`
(the one property this whole design exists for — asserts `primary`
itself refuses a pointer the heap fallback actually served, proving the
routing is checked, not guessed), and
`fallback_allocator_falls_all_the_way_through_when_primary_never_succeeds`
(a zero-capacity primary, forcing every allocation through the
fallback).

**Verification honestly scoped:** same standard as every module above
— no Miri/AddressSanitizer available in this sandbox. Checked by hand
against `StackAllocator`'s and `HeapAlloc`'s own already-reviewed
`unsafe` blocks, which this module composes rather than adding new raw
pointer manipulation of its own; `try_alloc_raw`/`try_dealloc_raw`
here contain no `unsafe` at all, only dispatch between two calls whose
own safety was established where they're defined.

## What's built: `Segregator`

`crates/mid-alloc/src/segregator.rs`, behind the `segregator` feature.
`Segregator<Small, Large>`: routes an allocation to `Small` when
`size <= threshold`, to `Large` otherwise. Directly modeled on
`foonathan::binary_segregator<Segregatable, RawAllocator>`
(`segregator.hpp`, source read this pass), with `threshold_segregatable`'s
own `size <= max_size` check folded straight into a `threshold: usize`
field on `Segregator` itself rather than kept as a separate wrapper
type — the same composable-vs-throwing collapse `RawAlloc` already
made once, applied consistently rather than re-litigated per module.

One real, deliberate behavioral difference from `FallbackAllocator`,
confirmed against the real source rather than assumed to match:
`binary_segregator::allocate_node` does **not** fall through to the
other allocator if the size-selected one fails — it calls that one
allocator's allocation function and returns whatever that gives, full
stop. A segregator's whole point is a deterministic size-based
partition, not a resilience mechanism; `Segregator` keeps that same
hard partition, proven by a dedicated test
(`does_not_fall_through_when_the_selected_side_fails`) rather than
just stated in a comment. Deallocation routes the same way —
re-checking the threshold against the given `size`, not an ownership
probe like `FallbackAllocator` needs — sound exactly as long as the
caller passes the same `size` it originally allocated with, the same
contract any `Layout`-based deallocation already requires.

`NullAlloc` (added to `raw_alloc.rs` in this same pass, not a separate
module — it's a general-purpose terminal `RawAlloc`, not specific to
segregators) matches `foonathan::null_allocator`'s real role: always
fails, useful as an explicit "refuse everything past this point"
terminal, e.g. `Segregator<Pool, NullAlloc>` to cap what a caller can
allocate rather than silently spilling to the heap.

**A real scope note on the original "small → pool, large → heap"
framing**, stated rather than quietly dropped: `PoolAllocator<T>` is
monomorphic per `T` (fixed-node-size `create`/`destroy`, not raw
bytes), so it does not implement `RawAlloc` and cannot plug into
`Segregator<Small, _>` directly — a real type-shape mismatch this
survey's original one-line framing glossed over. The tests below use
`Segregator<StackAllocator, HeapAlloc>` instead, which does typecheck
and is a real, common combinator shape (small requests from a fast
fixed buffer, large ones from the heap); a pool-backed small side
would need either a `RawAlloc`-shaped pool (a different, byte-oriented
design than `PoolAllocator` deliberately is, per its own "What's
built" section above) or a segregator generic over a typed allocator
instead of `RawAlloc` — neither built, both left as open, named
possibilities rather than silently assumed away.

**Tests:** 4, real, run alongside the rest of the crate: at-threshold
routes to `small`, above-threshold routes to `large` (checked by
asking each side directly whether it recognizes the resulting pointer,
not by trusting which branch ran), the no-fallthrough behavior above,
and dealloc routing recomputing the threshold rather than probing
ownership (using `NullAlloc` specifically so the test exercises only
the routing decision, not a real pointer).

## Zig re-survey, this pass — real file layout has changed since the
first read, two more borrowable ideas found and catalogued

Re-cloned `ziglang/zig`'s GitHub mirror fresh rather than trusting the
first pass's notes (that earlier read was itself already flagged as a
frozen, pre-Nov-2025-migration snapshot). The real layout has moved:
what the first pass read as pieces of one `heap.zig` file has since
split into `lib/std/heap/FixedBufferAllocator.zig`,
`lib/std/heap/PageAllocator.zig`, `lib/std/heap/debug_allocator.zig`
(the earlier pass's own note that this was "renamed from
GeneralPurposeAllocator" still holds), plus two files the first pass
never saw at all: `lib/std/heap/ThreadSafeAllocator.zig` and
`lib/std/heap/SmpAllocator.zig`. `resize_raw` above (`StackAllocator`)
is the one concrete thing built from this pass; two more real,
source-grounded ideas came out of it, catalogued here rather than
built yet:

- **`ThreadSafeAllocator`** (full source read, 56 lines): a plain
  `std.Thread.Mutex`-guarded wrapper — lock, forward to a
  `child_allocator`, unlock, on every `alloc`/`resize`/`remap`/`free`
  call. Nothing more. This is the concrete Rust-side match for
  `foonathan::memory`'s own already-catalogued `threading` combinator
  (survey section above) — two independent real sources now pointing
  at the same simple pattern. Not built: `#![no_std]` without a mutex
  dependency means this crate would need its own small spinlock
  primitive first (a real, separate piece of work, not assumed to be
  trivial), which is why this stays catalogued rather than built in
  the same pass as reading about it.
- **`debug_allocator.zig`'s design** (1479 lines; read the doc comment,
  `Config` struct, and the size-class/canary logic specifically, not
  every line): small allocations route through power-of-two size-class
  buckets (1, 2, 4, 8, 16... bytes) — a real, independent second source
  for the `memory_pool_collection` size-class-bucket idea the original
  foonathan survey already flagged, now confirmed by an entirely
  different codebase rather than resting on foonathan alone. Large
  allocations keep metadata in a side hash map rather than an inline
  header, specifically to avoid corrupting the returned pointer's
  alignment — directly relevant if `tracking` (still catalogued below)
  ever needs to record per-allocation metadata without disturbing what
  the caller gets back. Also uses a fixed `canary: usize` value written
  into each bucket's own header and checked on every free
  (`if (bucket.canary != config.canary) @panic("Invalid free")`), a
  concrete, cheap corruption-detection technique distinct from (and a
  real complement to) foonathan's fill-pattern poisoning already noted
  in the survey above. Neither the size-class buckets nor the canary
  check is built; both are real candidates for `tracking` once that
  module starts, not for `Segregator`, `Segregator`'s own binary
  (`threshold_segregatable`-shaped) design directly.
- **Confirms an existing design choice independently**, worth stating
  since it wasn't why this pass happened: `FixedBufferAllocator`'s own
  `ownsPtr`/`ownsSlice` (pointer-range containment) is the same check
  `StackAllocator::try_dealloc_raw` already does, and the same one
  `foonathan::memory_stack`'s `owns(ptr)` does — three independent real
  sources landing on the identical technique, not just two.

## What's built: `Tracked`

`crates/mid-alloc/src/tracking.rs`, behind the `tracking` feature.
`Tracked<A>`: wraps any `RawAlloc` with lock-free counters for every
allocation and deallocation that passes through it. Combines two real
sources rather than inventing a shape fresh:

- **The hook points** match `foonathan::tracked_allocator<Tracker,
  RawAllocator>`'s real interface (`tracking.hpp`, source read this
  pass) — call a hook after a successful allocation, one before a
  deallocation — collapsed from that trait's four hooks
  (`on_node_allocation`/`on_node_deallocation`/`on_array_allocation`/
  `on_array_deallocation`) onto `RawAlloc`'s own two methods, since
  `RawAlloc` has no separate array path (same collapse `RawAlloc`
  itself already made from the composable/throwing split — applied
  consistently rather than re-litigated per module, same as
  `Segregator`'s own note above).
- **The counters, and their exact update sequence**, are
  `mod_alloc::ModAlloc`'s real, already-shipped design (source read,
  `mod-alloc` 1.0.0 downloaded fresh from crates.io — MSRV 1.75,
  confirmed against its own `Cargo.toml`, this project's exact
  toolchain floor): six `AtomicU64` fields, `Ordering::Relaxed`
  throughout, peaks tracked with `fetch_max` rather than a
  compare-exchange loop. Ported directly rather than reworked, since
  the real design was already right for this crate's needs.

**One real addition beyond either source, stated rather than left
unmarked:** `AllocStats::dealloc_count`. `mod_alloc` doesn't track
deallocations as their own counter (its `alloc_count` only counts
allocations and growing reallocations, since it exists to profile a
`#[global_allocator]` where `dealloc` never fails); `RawAlloc::
try_dealloc_raw` already reports success or failure per call, so
counting it was free, and `foonathan::tracked_allocator`'s own
hook interface already tracks both directions symmetrically.

**Why atomics instead of a mutex, stated as a real consequence, not
just a style choice:** this is the one module from the whole plan
that didn't end up needing the `no_std` spinlock `ThreadSafeAllocator`-
shaped work is still waiting on. Said plainly rather than left to
imply more than it does: none of this crate's current `RawAlloc`
implementors are `Sync` (`StackAllocator` uses bare `Cell`s), so
`Tracked<A>` isn't `Sync` either yet — the atomics here are correct,
forward-compatible infrastructure for whenever a `Sync` `RawAlloc`
exists, not a working concurrent tracker today. Also inherited as-is
from `mod_alloc`'s real design, not introduced by this port: updating
a peak via two separate atomic operations (`fetch_add` then
`fetch_max`) rather than one combined step means the recorded peak can
very rarely under-report by one interleaved update under real
concurrent access — a known, accepted tradeoff for lock-free counters
generally, not a defect specific to this port, and currently unreachable
anyway given the `Sync` gap just above.

**Tests:** 6, real: snapshot starts at all zeros, alloc/dealloc update
exactly the counters they should (including that `total_bytes` stays
cumulative — dealloc doesn't undo it — and `peak_bytes` doesn't drop
back down when `current_bytes` does), peak tracks the high-water mark
across an alloc/alloc/dealloc sequence rather than the latest value,
`reset` zeroes every field, `since` computes a real delta while keeping
the two peak fields absolute (matching `mod_alloc::Profiler::stop`'s
own real math), and a failed deallocation (routed through `NullAlloc`,
which always refuses) is confirmed to never get counted.

## Module plan (catalogued, not built)

Every module from the original foonathan/memory survey has now
shipped (`stack_allocator`, `pool_allocator`, `fallback`, `segregator`,
`tracking` above) — nothing from that original plan was ruled out or
left behind. What's left below is what the Zig re-survey pass added,
not the original plan:

- **A `no_std` spinlock, then `ThreadSafeAllocator`-style
  `SyncStackAllocator`/similar** — `tracking`'s own atomics turned out
  to sidestep this for that one module, but a mutex-wrapped combinator
  still needs a real spinlock primitive first; none exists in this
  crate yet.
- **Hierarchical "backed" allocators** — letting one allocator
  provision another's backing memory (e.g. a `PoolAllocator` carving
  its regions out of a parent `StackAllocator` instead of the global
  heap) rather than every allocator reaching for the heap
  independently. Real, useful, not built; would need `StackAllocator`/
  `PoolAllocator` parameterized over a `RawAlloc` backing source
  instead of hard-coding `Vec`.
- **`debug_allocator.zig`'s two remaining techniques** (size-class
  buckets, canary-word corruption checks) — noted in the Zig re-survey
  above as candidates for `tracking`, not acted on in `tracking`'s own
  first pass; `Tracked<A>` wraps an existing allocator's alloc/dealloc
  calls, it doesn't itself route by size class or write guard bytes.
  Real future extensions, not silently folded in.

Every module above traces to a specific real function this survey
actually read, not to "allocator libraries tend to have this."

## Relationship to `mid-arena`

No overlap in scope, deliberately: `mid-arena`'s `SlotArena<T>` is
generational value storage with ABA-safe handles (`docs/mid-arena.md`);
`mid-alloc`'s allocators hand out raw/typed memory with no handle
indirection at all, closer to what a `Vec`/`Box` sit on top of than to
what `SlotArena` is. A plausible future point of contact: `mid-alloc`'s
`pool` module could become the block source `mid-arena`'s planned
`bump` feature grows into, the same way `foonathan::memory_pool` takes
a swappable `BlockOrRawAllocator` — not built, not assumed necessary,
just the one concrete place these two crates' scopes could eventually
touch.

## Open item

The library name for this survey came with a trailing "and alloca-t" in
the request that cut off before finishing — asked directly, not yet
answered. Leading guess is `allocator-api2` (the stable-Rust backport of
the nightly `Allocator` trait, which would be the natural way to make
`mid-alloc`'s types usable as the backing allocator for `Vec`/`Box`
directly, given `foonathan::memory`'s own `std_allocator.hpp` does the
same interop job for `std::allocator`) — not acted on since it's a
guess, not a confirmed one.

## Fixes and Problems

### `pool_allocator.rs`

- First pass. Two real compile errors caught by actually running
  `cargo test -p mid-alloc --features pool` on this sandbox's rustc
  1.75 rather than relying on hand-review alone: an unused
  `PoolRegion::remaining()` method (`dead_code` warning — removed,
  `bump()` already self-checks capacity) and three `DropCounter` test
  structs missing `#[derive(Debug)]` (`create()` returns
  `Result<&mut T, T>`, and `.unwrap()` on a `Result` needs `E: Debug`
  — `StackAllocator`'s own tests never hit this because its test
  payloads are all `Copy`/`Debug` primitives). Both fixed same pass;
  all 22 tests in the crate (13 new, 9 existing `StackAllocator`)
  pass clean, with and without the `pool` feature enabled.

### `raw_alloc.rs`

- First pass, new file. One real design decision made and stated
  rather than defaulted into: `RawAlloc::try_dealloc_raw` returns
  `bool` (ownership claimed or not) rather than being a plain
  `unsafe fn(...)` with no return value. Checked against
  `foonathan::memory_stack`'s own composable `try_deallocate_node`
  first (`state.arena_.owns(ptr)`, a real ownership check, not an
  unconditional no-op) before writing this trait, specifically because
  an earlier draft without the `bool` return had no sound way for
  `FallbackAllocator::try_dealloc_raw` to know which side to free
  through — stated here so a future reader doesn't "simplify" it back
  out. `HeapAlloc`'s zero-sized-request handling (a dangling, aligned
  pointer, never calling the real allocator) was checked against how
  `Vec` itself handles zero-sized types internally, the same
  precedent `mid-arena`'s own zero-sized-type test
  (`zero_sized_types_do_not_panic_or_loop_forever`) already
  established for this workspace. All 3 new tests pass on this
  sandbox's rustc 1.75 via a direct `cargo test -p mid-alloc
  --features fallback` (no scratch crate needed — see this module's
  own "What's built" entry for why).

- Added `NullAlloc` in the same pass as `segregator.rs`, needed as a
  real terminal for a `Segregator` chain (`Segregator<Pool,
  NullAlloc>`-shaped usage) rather than added speculatively. Matches
  `foonathan::null_allocator`'s real behavior (always fails) but not
  its throwing non-composable path, since `RawAlloc` has only the one,
  always-fallible interface (see this file's own doc comment). One
  test (`null_alloc_always_fails`) added to this file's existing
  suite.

### `stack_allocator.rs`

- Added a `RawAlloc` impl in the same pass as `raw_alloc.rs` and
  `fallback.rs`. One real risk considered and ruled out: whether
  adding `try_dealloc_raw`'s pointer-range check could regress
  `alloc_raw`'s own already-tested, already-reviewed hot path — it
  does not, since the new impl only reads `self.buf`'s existing
  `as_ptr()`/`len()` and never touches `top`, `alloc_raw`, or `alloc`
  at all. One new test
  (`raw_alloc_try_dealloc_recognizes_only_its_own_pointers`) added to
  the existing suite rather than a separate module, since it tests
  this type's own behavior. All 9 pre-existing `StackAllocator` tests
  re-ran clean alongside it, with and without every feature
  combination (`--features pool,fallback`, `pool` alone, `fallback`
  alone, neither).
- Added `resize_raw`, from a fresh Zig re-survey pass (see this file's
  own "Zig re-survey" section above for the full findings). One real
  correctness detail double-checked rather than assumed from the Zig
  source's shape alone: `FixedBufferAllocator.resize`'s real signature
  takes a `[]u8` slice (pointer *and* length together), but this
  crate's `alloc_raw` only ever hands back a bare pointer -- so
  `resize_raw` takes `old_size` as an explicit parameter rather than
  trying to recover it, the same design already used for
  `try_dealloc_raw`'s `size` parameter elsewhere in this file, kept
  consistent rather than solved two different ways. Four new tests
  (grow the last allocation, shrink and reclaim the tail for real,
  shrink-only on a non-last allocation, grow-past-capacity failing
  cleanly) bring this file to 13; all 38 in the crate re-ran clean
  across every feature combination.

### `fallback.rs`

- First pass, new file. No `unsafe` of its own — `try_alloc_raw`/
  `try_dealloc_raw` here are pure dispatch between two already-checked
  `RawAlloc` implementors, deliberately kept that way rather than
  reaching into either side's internals directly. Verified the actual
  claimed behavior, not just that it compiles: one test asserts
  `primary()` on its own refuses a pointer the heap fallback actually
  served (`fallback_allocator_routes_dealloc_to_whichever_side_really_owns_it`),
  the one property that would silently break if `RawAlloc::
  try_dealloc_raw`'s `bool` return were ever dropped in favor of an
  unconditional claim. All 3 tests pass on this sandbox's rustc 1.75,
  alongside the rest of the crate's 29 total: 22 pre-existing (9
  `StackAllocator` + 13 `PoolAllocator`) + 3 new in `raw_alloc.rs` + 1
  new in `stack_allocator.rs` + 3 new here.

### `segregator.rs`

- First pass, new file. Same "no `unsafe` of its own" property as
  `fallback.rs` — pure size-comparison dispatch between two
  already-checked `RawAlloc` implementors. The one real design
  question this pass had to settle by going back to the real source
  rather than guessing: whether a failed allocation on the
  size-selected side should fall through to the other allocator the
  way `FallbackAllocator` does. Re-read `binary_segregator::
  allocate_node` specifically to check, confirmed it does not (calls
  the chosen side's plain allocation function, returns whatever that
  gives), and wrote `does_not_fall_through_when_the_selected_side_fails`
  as a real test for that specific property rather than trusting the
  read and moving on. Also caught, while writing the "small → pool"
  example from the original survey framing, that `PoolAllocator<T>`'s
  monomorphic-per-`T` shape does not actually implement `RawAlloc` and
  cannot plug into `Segregator` directly — a real type-shape mismatch
  the original one-line module-plan description glossed over, stated
  in full in this file's "What's built" section above rather than
  quietly worked around by picking a different example without
  comment. All 4 tests pass on this sandbox's rustc 1.75, alongside
  the rest of the crate's 34 total (the 29 from `fallback.rs`'s entry
  above + 1 new `NullAlloc` test in `raw_alloc.rs` + 4 here).

### `tracking.rs`

- First pass, new file. The one real design decision this pass made
  and stated rather than defaulted into: whether to track
  `dealloc_count` even though `mod_alloc`'s real source doesn't. Traced
  it back to *why* `mod_alloc` skips it before deciding --
  `mod_alloc::ModAlloc` profiles a `#[global_allocator]`, where `free`
  is infallible by the trait's own contract, so a dealloc count would
  just equal an implicit assumption anyway; `RawAlloc::try_dealloc_raw`
  is not infallible (`FallbackAllocator`/`Segregator` compose multiple
  possible owners), so the count is real, new information here, not
  redundant the way it would be for `mod_alloc`. Added `AllocStats`'s
  `dealloc_count` field and a dedicated test
  (`a_failed_dealloc_does_not_get_counted`, routed through `NullAlloc`
  specifically so the "false" path is exercised for real) rather than
  silently matching the field, since silently matching it would have
  meant a) not thinking through whether the omission was deliberate
  upstream and b) leaving `RawAlloc`'s richer contract underused. All 6
  tests pass on this sandbox's rustc 1.75, alongside the rest of the
  crate's 44 total.
