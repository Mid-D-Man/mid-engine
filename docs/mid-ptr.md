# mid-ptr

## Overview

Type-erased raw pointer wrappers for mid-engine, ported from Bevy's `bevy_ptr`
crate (MIT/Apache-2.0). `no_std`, zero dependencies. Gives the rest of the engine
a small toolkit for holding a pointer to a value without the compiler knowing its
type — `Ptr`/`PtrMut`/`OwningPtr` mirror `&T`/`&mut T`/`Box<T>` with the type
erased, `MovingPtr` moves a value to a new location (and can be deconstructed
field-by-field) without passing it by value, and `ThinSlicePtr` is a `&[T]` with
the length stripped out for callers that track it separately.

## Provenance and scope note

This is a close port, not a from-scratch design — the type layouts, safety
invariants, and most of the doc comments come directly from `bevy_ptr` as it
stands today in `Mid-D-Man/bevy` (`crates/bevy_ptr/src/lib.rs`, 1,616 lines as of
this port). Everything upstream's file has has been carried over: the alignment
marker system, `ConstNonNull`, `Ptr`/`PtrMut`/`OwningPtr`, `MovingPtr` with its
full field-deconstruction machinery, `ThinSlicePtr`, and `UnsafeCellDeref`. The
one thing intentionally dropped is the `#[deprecated(since = "0.18.0", ...)]`
shim method on `ThinSlicePtr` (`get`, superseded by `get_unchecked` upstream) —
that only existed for Bevy's own semver back-compat and has no reason to exist in
a brand-new crate.

**`docs/roadmap.md`'s Decision 6 said not to build this crate right now** (the
FFI-facing half is already covered by `mid_collections::ffi_span::FfiSpan`, and
`archetype.rs`'s own module doc explicitly turned down the `OwningPtr`-style
migration technique for lack of profiled need). This port went ahead anyway, on
direct instruction, after surfacing that conflict. Decision 6 itself hasn't been
deleted — see its own entry in `docs/roadmap.md` for the reopened note. Nothing in
mid-ecs or archetype.rs currently calls into this crate; it's available for when
that's revisited.

**MSRV wall:** the `deconstruct_moving_ptr!` macro's field-projection arms use the
`&raw mut`/`&raw const` operators (RFC 2582), stabilized in Rust 1.82 — above this
workspace's usual rustc-1.75 floor. Full note: `docs/workspace-cargo.md`,
"MSRV / toolchain walls" (moved there from the root Cargo.toml's own
comments this pass — that file now carries only a one-line pointer).
Nothing else in this crate needs anything newer than the workspace floor.

**License note, unresolved:** `bevy_ptr` is dual MIT/Apache-2.0. This crate
carries no license file or `license` field yet, matching every other crate in
this workspace (no license decision has been made project-wide). Once one is
made, this crate specifically needs either a compatible license or an upstream
attribution/notice, since it's a substantial derivative of licensed code, not an
independent reimplementation. Flagging this now rather than letting it get lost.

**One documentation-convention mismatch found while doing this port, still
open:** `docs/DOCUMENTATION_AND_COMMENTING_GUIDELINES.md` §1 says a crate's doc
file lives "inside that crate's own directory, not a shared repo-wide docs
folder" — but all eight existing per-crate doc files (`mid-math.md`,
`mid-ecs.md`, etc.) live at the top-level `docs/`, none nested under
`crates/<name>/docs/`. This file follows the actual, universal practice
(top-level `docs/mid-ptr.md`) over the written rule, same as the other eight.
Not resolved either way — still worth a decision.

**Versioning mismatch found in the same pass — resolved:** `docs/
RUST_AND_CRATE_GUIDELINES.md` §5 says every crate starts at `0.0.1` and stays
there until an official `1.0.0` release; all 19 pre-existing crates are
actually at `0.1.0`, none of them touched here. mid-ptr's own `Cargo.toml`
now follows the documented rule (`0.0.1`) rather than the older crates'
practice — on direct instruction, not a default I picked on my own. The other
19 crates are unchanged; bringing them in line (or updating the doc to match
them instead) is a separate, disclosed follow-up.

## Modules

### `lib.rs`
**What it does:** Crate root — `#![no_std]`, the `#![allow(unsafe_code)]` opt-out
(per `RUST_AND_CRATE_GUIDELINES.md` §3; mid-ptr is the second crate to opt into
`[lints] workspace = true` after mid-math), module wiring, and the flat
`pub use` re-exports that keep the public API surface identical to upstream's
single-file layout (`mid_ptr::Ptr`, `mid_ptr::MovingPtr`, and so on).

**Decisions:**
- Split upstream's single 1,616-line `lib.rs` into 8 files by concern (alignment
  markers, `ConstNonNull`, the erased-pointer family, `MovingPtr`, its macros,
  `ThinSlicePtr`, the internal debug-alignment helper, and `UnsafeCellDeref`),
  per this project's own "favor modular sub-files" convention. Upstream keeps it
  as one file; this project's convention doesn't.
- The split forced `DebugEnsureAligned` (private in upstream, since everything
  shared one file) to become `pub(crate)` here — it's used from `erased.rs`,
  `moving.rs`, and `thin_slice.rs`, which are siblings, not descendants, of the
  module that defines it.

### `aligned.rs`
**What it does:** `Aligned`/`Unaligned` marker types and the sealed `IsAligned`
trait that dispatches `read_ptr`/`copy_nonoverlapping`/`drop_in_place` to either
the aligned or byte-wise-unaligned version, so every other type parameterized
over `A: IsAligned` only has to write the aligned/unaligned split once.

**Tests:** inline `#[cfg(test)] mod tests`. Covers aligned and deliberately
misaligned round-trip reads, and an aligned `copy_nonoverlapping` call.

### `const_non_null.rs`
**What it does:** `ConstNonNull<T>` — the `*const T` counterpart to
`NonNull<T>`'s `*mut T`; only allows conversion to read-only borrows.

**Tests:** inline. Covers null rejection, `as_ref` round-tripping, and the
`From<&T>`/`From<&mut T>` impls.

### `debug_align.rs`
**What it does:** `pub(crate)` `DebugEnsureAligned` trait — asserts a `*mut T` is
properly aligned for `T` in debug builds (skipped under miri, which already
checks this) and is a no-op in release. Every erased-pointer deref/read/drop-as
call routes through it first.

**Decisions:**
- Upstream keeps this private since it lives in the same file as every call
  site. Splitting the crate forced it to `pub(crate)` — see the `lib.rs` note
  above.

### `erased.rs`
**What it does:** `Ptr`, `PtrMut`, `OwningPtr` — the borrow-shaped,
mutable-borrow-shaped, and owning-pointer-shaped type-erased pointers. Shares a
local `impl_ptr!` macro for the common `byte_offset`/`byte_add`/`Debug`/`Pointer`
impls across all three.

**Decisions:**
- `OwningPtr::cast<T>() -> MovingPtr<'a, T, A>` originally constructed
  `MovingPtr` via its tuple field directly, since both types shared a file
  upstream. Here it goes through `MovingPtr::new` instead, since the field is
  private to `moving.rs` now — same safety contract, different call.

**Tests:** inline. Covers `Ptr`/`PtrMut` deref round-trips, `OwningPtr::make` +
`read`, `drop_as` running `Drop` exactly once, and `to_unaligned` +
`read_unaligned`.

### `moving.rs`
**What it does:** `MovingPtr<'a, T, A>` — moves a value to a new location without
passing it by value. Full method surface carried over: `from_value`, `read`,
`write_to`, `assign_to` (via an internal drop-guard to avoid a double memcpy),
`move_field`/`move_maybe_uninit_field` (the primitives
`deconstruct_moving_ptr!` builds on), `partial_move`, `assume_init`,
`to_unaligned`, plus `Deref`/`DerefMut` (aligned only), `Drop`, and the
`From<MovingPtr> for OwningPtr` / `TryFrom<Unaligned> for Aligned` conversions.

**Tests:** inline. Covers `read` skipping the pointee's `Drop`, `assign_to`
dropping the old value exactly once, and `to_unaligned` + `TryFrom` recovering
`Aligned`.

### `moving_macros.rs`
**What it does:** `move_as_ptr!`, `get_pattern!` (macro-internal helper), and
`deconstruct_moving_ptr!` — the three `#[macro_export]` macros that make
`MovingPtr` usable without hand-writing raw pointer arithmetic at every call
site. `deconstruct_moving_ptr!` supports structs, tuples (via the `tuple`
identifier), and projecting through `MaybeUninit::<_>`.

**Decisions:**
- This is the one file transcribed with extra care and re-verified directly
  against the upstream source line-for-line rather than from a first pass,
  specifically because of the `&raw mut`/`assume_init_mut` combination below —
  worth a permanent note here in case this file needs touching again.
- `assume_init_mut()` inside two of the five macro arms isn't a method this
  crate defines: it's `core::mem::MaybeUninit::assume_init_mut`, reached via
  auto-deref through `moving.rs`'s own `DerefMut for MovingPtr<'_, T, Aligned>`
  impl (`Target = T`, and here `T` is instantiated as `MaybeUninit<Inner>`). Easy
  to misread as a missing method; it isn't one.

### `thin_slice.rs`
**What it does:** `ThinSlicePtr<'a, T>` — a `&'a [T]` with the length cut out.
Debug builds still track the length internally and bounds-check
`get_unchecked`; release builds don't. A specialized `impl` block for
`ThinSlicePtr<'a, UnsafeCell<T>>` adds mutable-slice access and a `cast` back to
the plain `T` form.

**Decisions:**
- Dropped upstream's `#[deprecated(since = "0.18.0", ...)] get()` shim — see the
  scope note above.

**Tests:** inline. Covers `get_unchecked` across a full slice, `as_slice_unchecked`
reconstructing the original slice, `slice_unchecked` on a subrange, and the
`UnsafeCell` specialization's mutable-slice + `cast` round-trip.

### `unsafe_cell.rs`
**What it does:** `UnsafeCellDeref` — a sealed extension trait adding
`deref`/`deref_mut`/`read` to `&UnsafeCell<T>`, for call sites that have already
established the aliasing invariants some other way.

**Tests:** inline. Covers `deref`/`deref_mut` reaching the same cell and `read`
returning a copy without disturbing it.

## CI and Workflows

- `.github/workflows/mid-ptr-test.yml` — build, clippy, fmt check, unit +
  doc-tests, `workflow_dispatch` only. **Corrected**: this workflow originally
  ran on push/pull_request with a fake "summary" job that only echoed a few
  lines to the job log. Both were wrong, never checked against
  `mid-ecs-test.yml`'s own already-established pattern (dispatch-only +
  a real, parsed `$GITHUB_STEP_SUMMARY`) before this workflow was first
  written — see `docs/RUST_AND_CRATE_GUIDELINES.md` §7 for the full
  writeup. Now matches that pattern: a Python step parses the raw
  `cargo test` output into JSON, a second step writes a real markdown
  report to the job summary (pass/fail table, failure details, collapsible
  per-suite breakdowns), and the raw log + JSON upload as a build artifact.
  **Still not replicated**: the HTML-report-plus-`gh-pages`-deploy half of
  `mid-ecs-test.yml`'s pattern — the whole `gh-pages` reporting pipeline is
  being migrated to Cloudflare Pages, so building more `gh-pages`-specific
  tooling onto this workflow right now would likely be thrown away almost
  immediately.
- **First real run failed on a cache bug, fixed same pass**: the cache step
  originally cached `~/.cargo/bin/` under a bare `cargo-registry-` key prefix
  (copied from `mid-log-test.yml`), which matched a different, ARM-runner
  workflow's cache and replaced this run's real `cargo` binary with an
  incompatible one (`Exec format error` on every `cargo` call). Fixed by
  dropping `~/.cargo/bin/` from the cached paths and scoping the cache key to
  `${{ runner.os }}` plus `mid-ptr` specifically. Full root-cause writeup in
  `docs/RUST_AND_CRATE_GUIDELINES.md` §7 — `mid-log-test.yml` carries the same
  latent risk, not fixed there in this pass.

## Fixes and Problems

### `.github/workflows/mid-ptr-test.yml`
- **Fixed:** first real CI run failed on every `cargo` invocation with
  `cannot execute binary file: Exec format error`. Root cause: the cache step
  cached `~/.cargo/bin/` (architecture-specific binaries) under a
  workspace-wide-generic `cargo-registry-` key prefix, copied from
  `mid-log-test.yml`'s own cache block. `restore-keys` fell through to a
  different workflow's cache (`benches/ecs-vs-bevy-ecs`, built on an
  `ubuntu-24.04-arm` runner) and silently swapped in an ARM `cargo` binary on
  this run's x86_64 `ubuntu-latest` runner. Fixed by dropping `~/.cargo/bin/`
  from the cached paths and scoping the cache key to `${{ runner.os }}` plus
  `mid-ptr` specifically — see `docs/RUST_AND_CRATE_GUIDELINES.md` §7 for the
  full writeup.
- **Fixed:** the workflow ran on push/pull_request with a log-echo "summary"
  job — wrong on both counts, against a convention (`workflow_dispatch` only,
  real `$GITHUB_STEP_SUMMARY`) already established in `mid-ecs-test.yml`
  before this workflow existed, not checked at the time. See
  `docs/RUST_AND_CRATE_GUIDELINES.md` §7.

### FFI — open gap, not yet fixed
- The root `README.md`'s own Design Mandates state "every crate exposes a
  strict `#[repr(C)]` FFI boundary" — confirmed directly, not assumed. This
  crate currently has neither an FFI module nor a `cdylib`/`staticlib`
  `crate-type` in its `Cargo.toml` (only the default `rlib`). `mid-ecs` and
  `mid-net` both have the real pattern to follow (an `ffi.rs` module + a C
  smoke test under a `ffi-smoke-test/` directory, wired into their own test
  workflow). Not done in the pass that built this crate; flagged here rather
  than left silently missing.
