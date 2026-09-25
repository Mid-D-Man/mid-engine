# Root workspace Cargo.toml

## Overview

This is the companion doc for the root `Cargo.toml` — the file itself now
carries only terse one-line pointers into the sections below, per this
project's own "no fix history or decision logs inline" rule
(`docs/DOCUMENTATION_AND_COMMENTING_GUIDELINES.md`). Everything that used to
live as long dated `#` comment blocks directly in the TOML file has moved
here verbatim; nothing was reworded or summarized in the move, only
reformatted from `#`-prefixed comments into prose.

## Sections

### `[workspace] members`

mid-engine is a real Cargo workspace: `[workspace]`, `resolver = "2"`, 28
members under `crates/`, `benches/`, and `examples/` (recounted directly
against the member list this pass — up from 24 as of this doc's own last
count, itself already one behind reality: `benches/query2-ref-isolated`
had been added without updating this line), already wired together with
`path = "../.."`-style dependencies (`mid-anim` on `mid-math`, `mid-app` on
`mid-ecs`, `mid-ecs` on `mid-collections` and `mid-math`, `mid-physics` on
`mid-math` and `mid-geom`, and more).

New in a later pass (`docs/roadmap.md`, "What can be built in parallel"):
`mid-time`, `mid-physics`, `mid-anim`, `mid-app` were added as v0 stubs, not
yet compiled/verified in this sandbox (no rustc available here at all — see
that roadmap section's own honesty note).

New this pass: `crates/mid-platform` (`docs/roadmap.md` Decision 3 reopened
— see that section and `docs/mid-platform.md`). Unlike every other crate in
this workspace, it has real, non-default-off Cargo features (`std`, on by
default, and `alloc`) rather than being unconditionally `no_std` — its whole
purpose is switching between a `std` passthrough and a hand-rolled fallback,
so the feature gate is structural, not incidental. `cargo test -p
mid-platform` alone only exercises the default (`std`) path; CI also runs
`--no-default-features` to exercise the fallback path, since it's a
genuinely different code path default features would never even compile.

**Fixed:** `tools/mdix-compiler` and `examples/headless-server` were each
listed twice in the members array. `docs/RUST_AND_CRATE_GUIDELINES.md` §1
already claimed this was "found and fixed," but the actual file still had
both duplicates until this was caught directly. Harmless in practice (Cargo
doesn't error on an exact-duplicate member path), but real drift between the
doc and the file it describes. Removed the second occurrence of each; the
doc's own description matches the file again now.

### MSRV / toolchain walls

The root `Cargo.toml` carries a running, dated record of every per-crate
MSRV/toolchain wall found so far — read this before assuming a bare `cargo
build`/`cargo test` with no `-p` flag will resolve cleanly; several members
deliberately need a newer toolchain than this project's rustc-1.75 floor.

**`mid-net-transport-quinn`:** its `web-transport-quinn` dependency needs a
Cargo that understands `edition2024` (~1.85+) to even resolve, transitively
(confirmed directly: `cargo check -p mid-net-transport-quinn` fails on rustc
1.75 at the `cpufeatures` manifest with exactly that error). `cargo check -p
<anything-else>` and `cargo test -p <anything-else>` are UNAFFECTED —
verified directly, resolver 2 scopes lockfile resolution to each `-p`
target's own dependency closure, not the whole members list. Only a bare
`cargo check`/`cargo build`/`cargo test` with NO `-p` (i.e. "build
everything") pulls this crate in and needs the newer toolchain. If you're on
an older Rust and something inexplicably won't resolve, this is almost
certainly why — build with explicit `-p` flags instead.

*Update, later pass:* `examples/headless-server` now depends on
`mid-net-transport-quinn` directly too (it's the actual thing exercising it
over a real bind/dial, not just compiling it in isolation), so `-p
headless-server` needs the same newer toolchain now as well.

*Update, later pass:* `mid-net-transport-wasm` added, and it behaves
DIFFERENTLY from the two above — confirmed empirically, not assumed. `cargo
check -p mid-net-transport-wasm` with NO `--target` flag resolves cleanly
even on rustc 1.75 (its `web-transport-wasm` dependency lives under
`[target.'cfg(target_arch = "wasm32")'.dependencies]`, and resolver 2
doesn't fetch platform-specific deps for a platform that isn't actually
being built). The wall only shows up with an explicit `--target
wasm32-unknown-unknown` — same edition2024 problem, this time via
`idna_adapter` rather than `cpufeatures`, pulled in through
`web-transport-wasm`'s own tree. So: `-p mid-net-transport-wasm` alone is
fine anywhere; add `--target wasm32-unknown-unknown` and it needs the same
~1.85+ toolchain as the other two.

*Update, later pass:* `mid-net-transport-wasm/wasm-test-server` added — a
native-only fixture server for the real browser test (see
`.github/workflows/mid-net-transport-wasm-test.yml`). Depends on
`mid-net-transport-quinn` directly, so `-p
mid-net-transport-wasm-test-server` needs the ~1.85+ toolchain
unconditionally, same as `-p mid-net-transport-quinn` itself — not the
conditional wasm32-only situation `mid-net-transport-wasm` is in.

**`mid-ecs` (rayon dependency):** needs rustc 1.80+, confirmed directly
(`cargo check -p mid-ecs` fails on rustc 1.75 at `rayon-core`'s manifest).
Independent of the edition2024 walls above — ordinary MSRV drift, not the
same root cause. `-p mid-collections` alone is unaffected (zero deps).

**`mid-collections`' bench** (criterion dev-dependency, added when the
SparseSet bench suite was written): pulls in criterion → clap_builder, which
needs edition2024 same as the walls above — confirmed directly, fails at
`clap_builder`'s manifest on rustc 1.75. A FIFTH independent instance of this
exact class of ordinary upstream MSRV drift, not a new root cause. REAL
REGRESSION worth flagging explicitly: this affects `cargo test -p
mid-collections` too, not just `--bench` — verified directly (first assumed
dev-dependencies were resolved per-target, that assumption was WRONG and
corrected after actually testing it). Cargo resolves a package's full
manifest, dev-dependencies included, before building ANY target from it, so
`mid-collections` lost the "zero-MSRV-wall, fully testable in this sandbox"
property this same record used to get to claim uniquely for it, the moment
criterion was added as a dev-dep. Real tests were still run and verified
before this dependency was added (18/18, see `docs/mid-collections.md`) —
this note is about local re-verification going forward, not about doubting
that earlier result.

**`mid-arena`'s bench** (criterion dev-dependency, added when
`vs_arena_crates` was written): same edition2024 wall as every entry above —
confirmed directly, fails at clap_builder's manifest on rustc 1.75. A SIXTH
independent instance of this exact class of ordinary upstream MSRV drift.
Same real regression as `mid-collections` took, same reason, not
re-litigated — `cargo test -p mid-arena` alone now needs the newer toolchain
too, not just `--bench`. Real tests were run and verified BEFORE this
dependency was added (16/16, see `docs/mid-arena.md`) — same sequencing
`mid-collections` used, same reason: verify everything the current toolchain
still can, before adding the thing that closes that window.

**`mid-platform`'s bench** (criterion dev-dependency, added when
`sync_bench.rs` was written to cover Phase 2's `RwLock`/`Once`/`OnceLock`/
`LazyLock`/`Barrier`): same edition2024-via-clap_builder wall as
`mid-collections`'s and `mid-arena`'s own bench dev-dependencies above — the
third crate in this workspace to hit specifically this wall via a criterion
dev-dependency (confirmed by checking each entry above directly, not
assumed). Same real regression those two took, same reason: `cargo test -p
mid-platform` with no other flags is affected too, not just `--bench` —
Cargo resolves a package's full manifest, dev-dependencies included, before
building any target from it. Unlike the crates above, `mid-platform` also
has real `std`/`no_std` cargo features of its own (`docs/mid-platform.md`),
independent of this wall — `--no-default-features` still needs the same
newer toolchain once this dependency is in the manifest, the two concerns
don't cancel each other out.

**`mid-ptr`** (`docs/roadmap.md` Decision 6 reopened — see that section): a
ported crate, not an upstream-dependency wall like every entry above it.
`crates/mid-ptr/src/moving_macros.rs`'s `deconstruct_moving_ptr!`
field-projection arms use the `&raw mut`/`&raw const` operators (RFC 2582),
stabilized in Rust 1.82.0 (confirmed against the actual 1.82.0 release
notes, not assumed from the RFC alone — the raw-reference operators
themselves are edition-independent, so this is a straight rustc-version
floor, not an edition2024 situation like the walls above). A SEVENTH
independent MSRV-wall instance, but the only one so far caused by this
workspace's own code rather than a dependency's manifest. `-p mid-ptr` alone
needs 1.82+; nothing else in this crate does, and nothing else in the
workspace is affected. No rustc at all was available to verify this by
actually compiling it — see `docs/mid-ptr.md` for the full note.

**`benches/ecs-vs-bevy-ecs`:** its `bevy_ecs` dependency declares
`rust-version = "1.95.0"` (confirmed directly: tried building `bevy_ecs`
0.19.1 from crates.io). Same class of wall as `mid-net-transport-quinn`'s
edition2024 requirement above, same isolation reasoning — kept as its own
workspace member specifically so a bare `cargo test`/`cargo build`/`cargo
bench` without `-p` doesn't pull the newer-toolchain requirement into
`mid-ecs`'s own, otherwise-1.91-compatible build. `-p ecs-vs-bevy-ecs` needs
the newer toolchain; nothing else does.

### `[workspace.lints]` and `[workspace.dependencies]`

*Restored 2026-09-08:* `docs/RUST_AND_CRATE_GUIDELINES.md` §1 already
documented this exact table as existing ("The workspace now has a
`[workspace.lints]` table and a `[workspace.dependencies]` table"), but it
was actually missing from the file — confirmed directly (grepped for
`[workspace.lints]`/`[workspace.dependencies]` in the file before concluding
that, not assumed from the doc alone), which is what made `mid-math`'s
`criterion = { workspace = true, ... }` fail to resolve
("`workspace.dependencies` was not defined") and would have failed the same
way on its `[lints]` / `workspace = true` opt-in right after, since
`[workspace.lints]` was equally absent. Content restored matches the
guidelines doc's own quoted block exactly, not reconstructed from
guesswork. `missing_docs`/`undocumented_unsafe_blocks` are `"warn"`, not
`"deny"` — opting in (`mid-math` only, so far, per that same doc's §1) won't
fail anyone's build even if it surfaces new warnings.

`mid-math`'s wasm32-target dev-dependency overrides `criterion` to
`default-features = false` — Cargo ignores that override (a real, unresolved
upstream limitation: a member can only override `default-features` to
`false` when the *workspace* entry already says `false`, confirmed against
Cargo's own docs and rust-lang/cargo#11329/#12162, not guessed at — an
earlier attempt here to "fix" this by setting `default-features = true`
explicitly didn't actually resolve it either, just reworded the same
warning). Harmless and unrelated to any of the passes above: nothing here
touches the wasm32 target, and the warning says "could become a hard error
in the future," not that it is one. Real follow-up for whoever next touches
`mid-math`'s wasm bench setup, not fixed here.

### `[profile.release]` and `[profile.bench]`

`cargo bench` does NOT use `[profile.release]` — it has its own default
profile (`opt-level=3`, but WITHOUT `lto`/`codegen-units=1`). That gap
matters for cross-crate calls into external comparison crates in this
project's benches (e.g. `wide`): this project's own `mid-math` methods are
all `#[inline(always)]`, which survives regardless, but a crate only called
via `#[inline]` (not `#[inline(always)]`) through a multi-hop call chain
depends on aggressive cross-crate inlining to collapse down to the same
single instruction this project's own code gets — added 2026-08-23 after
`wide::i32x4::add` benched ~14x slower than this project's own `i32x4::add`,
tracked to exactly this profile gap (see `docs/platform-optimization.md`
§9).

### `[profile.bench-nolto]`

Added after checking two things directly, both prompted by the
`query2_static` investigation (`docs/mid-ecs.md`) reaching a point where
every source-level `Iter2` variant had been tried:

1. `[profile.bench]` already sets `codegen-units=1` + `lto=true`
   workspace-wide, so `benches/ecs-vs-bevy-ecs` compiles real `bevy_ecs`
   under this same regime too (it's a workspace member, not a separate
   workspace — checked directly).
2. Bevy's OWN root `Cargo.toml` (`Mid-D-Man/bevy`, real source, not assumed)
   has no bare `[profile.release]` and no `[profile.bench]` at all — its own
   benchmarks (the same `benches/benches/bevy_ecs/iteration/*.rs` already
   read for the `#[inline(never)]`-wrapper convention) run under cargo's
   plain, unconfigured default: `codegen-units=16`, no LTO. Bevy relies on
   explicit `#[inline(always)]` on the functions that need it (confirmed on
   `QueryIterationCursor::next` itself) rather than whole-program LTO to
   guarantee their codegen — a different strategy than this workspace's "LTO
   everything, let the compiler see the whole graph" choice, one this
   project never deliberately chose FOR `mid-ecs`'s own query iterators
   specifically, it's just what the `wide`-crate fix above happens to apply
   to everything.

Searched before adding this profile: LTO making a *specific* loop's own
codegen worse, independent of any codegen-unit-reshuffling question, is real
and confirmed upstream, not a stretch — rust-lang/rust#106609 ("LTO produces
worse codegen for a loop," a fat-vs-thin-vs-off comparison on real
disassembly) and rust-lang/rust#146497 (a 2025 nalgebra/criterion
reproduction showing >4000% degradation from `lto="fat"` alone, no
codegen-units change needed). Separately, vortex-data/vortex#9259 hit the
*other* mechanism this project already avoids (`codegen-units=16`
reshuffling which functions share a unit when unrelated code is added) and
fixed it by moving TO `codegen-units=1`+`lto=true` — the setting this
workspace already has. So: two distinct, independently-real mechanisms
exist, this project's current profile is already the fix for one of them,
and might be the active cause of the other for this specific loop shape.
Worth an actual A/B test on the one thing this investigation hadn't varied
yet: the profile itself, not another `Iter2` source variant.

`inherits = "release"` is required by cargo for any custom profile; every
field is then set explicitly anyway, so nothing unintended carries over from
release (`strip`/`debug` in particular).

## Fixes and Problems

*(see the MSRV / toolchain walls section above — every entry there is
already a dated fix/finding in its own right; nothing further to log here
yet)*
