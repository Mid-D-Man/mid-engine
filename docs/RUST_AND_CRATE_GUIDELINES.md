# Rust and Crate Guidelines

## Purpose

`DOCUMENTATION_AND_COMMENTING_GUIDELINES.md` covers how to write comments and
docs, in any language. This one covers Rust and Cargo specifics: how a crate
in this workspace is supposed to be laid out, when something gets a feature
flag, how unsafe code gets handled, what a dev build looks like versus a
release build, and how versions move.

This is a first pass. Some of it (the workspace section especially) depends
on a decision that has not been made yet, and is written that way on purpose.

## 1. Workspace Structure

mid-engine is meant to be one Cargo workspace: 14 crates under `crates/`,
already wired together with `path = "../..."` dependencies (`mid-anim` on
`mid-math`, `mid-app` on `mid-ecs`, `mid-ecs` on `mid-collections` and
`mid-math`, `mid-physics` on `mid-math` and `mid-geom`, and more). Right now
there is no `[workspace]` manifest anywhere in the repo, and the file sitting
at the repo root as `Cargo.toml` is a stale near-duplicate of
`crates/mid-math/Cargo.toml` (same package name, same version, same bench
list) that predates mid-math's move into `crates/`. Building at the repo
root right now tries to rebuild mid-math a second time, from paths that do
not exist there.

Once this is set up as a real workspace, the root `Cargo.toml` becomes:

```toml
[workspace]
resolver = "2"
members = ["crates/*"]

[workspace.package]
edition = "2021"
license = "MIT OR Apache-2.0"

[workspace.lints.rust]
unsafe_code = "deny"
missing_docs = "warn"

[workspace.lints.clippy]
undocumented_unsafe_blocks = "warn"
```

A crate opts into the shared lint table with `[lints] workspace = true`
instead of repeating the list. `[workspace.dependencies]` holds one pinned
version for anything more than one crate depends on (`glam` for
benchmarking, `criterion`, and so on), so a version bump happens once at the
workspace level instead of drifting crate by crate. That is the exact kind
of drift that already happened once with mid-math's own `glam` dev-dependency going
five minor releases stale before anyone noticed.

## 2. Feature Gating

If a piece of a crate is substantial and not everyone who depends on that
crate needs it, it gets a Cargo feature. This is already happening in two
places in this repo:

- `mid-collections`: `ffi = ["dep:zerocopy"]`, `default = []`. The
  `zerocopy` dependency and whatever it enables in `component.rs` only
  compile in for a consumer that actually asked for the FFI span mechanism.
- `mid-math`: `mint = ["dep:mint"]`, gating the mint interop conversions.

`mid-math`'s own `ffi/` module (C-ABI exports for essentially every type in
the crate) is the next candidate: most consumers of mid-math are other Rust
crates in this workspace and never touch the C ABI at all, so compiling in
several hundred `#[no_mangle] extern "C"` functions for them is wasted
compile time and wasted binary size. `mid-ecs` made the opposite call for
its own, much smaller `ffi.rs`: always compiled, no separate feature,
explained directly in its `mid-collections` dependency comment. That is
the right call for something that small. mid-math's `ffi/` is a different
scale, so it gets its own feature:

```toml
[features]
default = []
ffi = []
```

A few rules for feature flags in this workspace:

- One-line comment above every feature explaining what it turns on, not just
  what it's named. `# Enable interoperation with the real mint crate's
  Vector/Point/Quaternion/Matrix types.`, not silence.
- Optional dependencies always use the `dep:` prefix in the feature list
  (`ffi = ["dep:zerocopy"]`), never the older implicit
  same-named-feature-from-an-optional-dependency form. Explicit here costs
  nothing and avoids Cargo's own feature-unification surprises.
- `default = []` unless there's a genuinely good reason a bare
  `dependency = { path = "..." }` with no `features = [...]` should still
  get something extra. Most of this workspace's crates should default to
  the smallest useful surface, not the largest.
- A feature that only forwards to an optional dependency's own feature of
  the same purpose uses the `?` form so requesting it does not silently turn
  the dependency on: `serde = ["dep:serde", "some-dep?/serde"]`.

## 3. Unsafe Code

mid-math in particular carries a lot of unsafe SIMD intrinsic code, and none
of it is currently required to carry a `# Safety` doc comment or a `// SAFETY:`
comment at the call site. `docs/roadmap.md`'s own Decision 5 already flagged
this as real, unenforced debt. Once the workspace lint table exists (section
1), the rule is:

- `unsafe_code = "deny"` at the workspace level. A crate that has a real
  reason to use unsafe (mid-math's SIMD intrinsics, an FFI boundary,
  anything touching raw pointers for performance) opts back in explicitly
  with `#![allow(unsafe_code)]` at its crate root, so the exception is
  visible in that one line rather than silently inherited.
- `undocumented_unsafe_blocks = "warn"` (clippy). Every `unsafe { ... }`
  block gets a `// SAFETY:` comment immediately above it explaining why the
  invariant the block depends on actually holds. Every `unsafe fn` gets a
  `# Safety` doc section explaining what its caller has to guarantee.
  `camera.rs`'s `mid_frustum_from_planes` in mid-math's own `ffi/` already
  does this correctly. That is the model, not a new invention.

## 4. Debug and Release Builds

Cargo's own `dev`/`release` profile defaults (unoptimized + debug assertions
on, versus optimized + debug assertions off) are the baseline and do not
need overriding for most of this workspace. Two additions are worth making
once the workspace exists:

```toml
[profile.wasm-release]
inherits = "release"
opt-level = "z"
lto = "fat"
codegen-units = 1
```

Already recommended in `docs/roadmap.md` for mid-math specifically, never
applied. Belongs at the workspace level now so every crate that ships to
wasm gets it, not just mid-math.

`debug_assert!`/`debug_assert_eq!` are the right tool for an invariant that
is expensive enough to skip in release but genuinely useful to catch in
day-to-day dev and testing builds. Prefer them over a plain `assert!` for
anything on a hot path that is not also a safety invariant (a safety
invariant backing an `unsafe` block still gets a real `assert!`, or the
`unsafe` block does not get to make that assumption at all).

## 5. Versioning

Every crate starts and stays at `0.0.1` for as long as it's in active
dev/testing. No incrementing 0.1.0, 0.2.0, and so on along the way. The
first actual, official release jumps straight to `1.0.0`. Nothing in
between.

## 6. Still Open

- Whether to set up the real `[workspace]` now or later (section 1). This
  changes what `[workspace.lints]`/`[workspace.dependencies]` actually look
  like once it happens, so it's worth deciding before those get written
  down as settled.
- Whether every crate that could reasonably split an `ffi` feature out
  (mid-collections and mid-math already do; anything else with a sizeable
  `ffi.rs`/`ffi/` is a candidate) should, or whether mid-ecs's
  "small enough to always compile" call is the right default and `ffi` as
  a feature is the exception, not the rule.
- A per-crate `[package.metadata.docs.rs] all-features = true` (bevy's
  convention) so generated docs never hide something behind a feature that
  is not enabled by default.
