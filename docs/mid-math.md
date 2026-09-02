# mid-math

## Scope of this doc

`mid-math` is a large, mature crate — SIMD-backed `f32`/`f64` vector and
matrix types, fixed-point, curves, noise, color spaces, camera math.
This doc does not attempt to cover all of it. It covers one specific,
real subsystem: the **Large World Coordinates (LWC)** primitives —
what exists, what was added this pass, and what's still ahead. A
full crate-wide design doc is a separate, later undertaking.

## The problem: jitter

A `f32` has roughly 7 significant decimal digits. At a coordinate
magnitude of `100_000.0`, the gap between two representable `f32`
values is already close to `0.01` — smaller offsets than that get
rounded to the same bit pattern, which reads as visual jitter/snapping
once an object or camera is far enough from the world origin. This
is real, not theoretical: `f64_tests.rs`'s
`dvec3_to_view_relative_is_the_actual_fix_for_the_jitter_this_exists_for`
test asserts the precision loss directly at a `100_000.0`-magnitude
coordinate before proving the fix addresses it.

`f64` doesn't eliminate the problem, it just moves the threshold much
further out (~15-16 significant digits) — which is why LWC in most
engines, this one included, is a hybrid: store world-scale state in
`f64` where it actually needs the range, keep everything downstream of
the camera (rendering, most gameplay logic) in `f32`, and have one
deliberate conversion step in between.

## What already exists: the f64 primitive layer

`mid-math/src/f64/` — `DVec2/3/4`, `DQuat`, `DMat2/3/4`, `DAffine2/3`.
Real, tested (71 dedicated tests in `tests/f64_tests.rs`, not stubs),
not newly added this pass. Each has an `as_*` lossy-cast counterpart
into the matching `f32` type (`DVec3::as_vec3`, `DAffine3::as_affine3`,
etc.) — a direct truncation, correct when the value is already
small-magnitude, *not* a fix for jitter on its own if called on a
raw world-space value.

`mid-math/src/camera/` — frustum culling, projection decompose/resize,
screen-space unprojection, cascaded shadow map splits. All `f32`,
operating on `Mat4`/camera-relative data — this is deliberate:
once a value is in view space (post-LWC-shift), `f32` is correct and
sufficient, so nothing in this module needed `f64` in the first place.
This is **not** the planned `mid-camera` crate (see below) — it's
math utilities that crate will consume, the same relationship
`mid-ecs` has to `mid-collections`.

## What was added this pass: the view-space shift

The one primitive that didn't exist yet: composing "shift by camera
origin" with "cast to f32" into a single, named, correct operation,
rather than expecting every call site to get the shift direction and
ordering right by hand.

- **`DVec3::to_view_relative(self, origin: DVec3) -> Vec3`** — for
  position-only data. `(self - origin).as_vec3()`.
- **`DAffine3::to_view_relative(self, origin: DVec3) -> Affine3`** —
  for full transforms. `(DAffine3::from_translation(-origin) *
  self).as_affine3()`. Rotation and scale (`matrix3`) pass through
  completely unaffected — they were never position-magnitude-dependent,
  so they never needed `f64` in the first place; only `translation`
  gets shifted, from a world-magnitude value down to a small,
  camera-relative one, which is what makes truncating it to `f32`
  safe.

Both are real, tested against the actual precision-loss scenario, not
just the arithmetic in isolation — see `daffine3_to_view_relative_shifts_translation_only`
and its sibling tests in `f64_tests.rs`.

Precision is highest exactly where `origin` is. Calling this once per
frame with the camera's own current position/transform, right before
building per-vertex or per-instance GPU data, is the intended use —
everything downstream of that point stays `f32`.

## What's still ahead

- **`mid-ecs` integration** — a real `GlobalTransform` component doesn't
  exist yet. See `docs/mid-ecs.md`'s own new section for the design
  (two component types, `f32` default + `f64` opt-in), which is where
  these primitives actually get used.
- **`mid-camera`** — planned, not started. This engine's equivalent of
  Unity's Cinemachine: camera rigs, follow/orbit/look-at behavior,
  blending between virtual cameras. Sits on top of both `mid-math`'s
  camera math (frustum/projection/unprojection) and whatever
  `mid-ecs` transform system it tracks — the same "math primitives
  below, ECS-facing behavior above" split this engine already uses
  elsewhere (`mid-collections` → `mid-ecs`).
- **Render Core** — the actual per-frame call site that takes a
  camera's current `DAffine3`, calls `to_view_relative` on every
  visible entity's `GlobalTransformLWC`, and hands the result to the
  GPU. Downstream of both of the above; not started.

## Known issue found and fixed this pass: `tests/mod.rs`

`crates/mid-math/src/tests/mod.rs` declared `mod mid_vec;`, expecting
`crates/mid-math/src/tests/mid_vec.rs` — a file that didn't exist,
which meant **`cargo test -p mid-math` could not compile at all**,
for any test, regardless of this LWC work. The actual file (526 lines
of real `#[test]` coverage for `MidVec`'s drop/alignment/spill
semantics) was sitting at `crates/mid-math/src/mid_vec/mid_vec.rs` —
inside the *implementation* directory, alongside `mod.rs`/`raw.rs`/
`iter.rs`, where it wasn't declared by that directory's own `mod.rs`
either, so it was simultaneously dead code there. Moved to its
evidently-intended location; both problems resolved by the one move.
`cargo test -p mid-math --lib` now passes 659/659, including the 65
`MidVec` tests this recovers and everything added this pass.

## Known issue found, not fixed this pass: clippy and fmt debt

`cargo clippy -p mid-math -- -D warnings` — the exact command
`mid-math-test.yml` runs — currently reports **182 errors** on rustc
1.91 (closer to CI's real 1.98 than this sandbox's default 1.75 has
ever been able to check). `cargo fmt -p mid-math --check` reports
**264 files** with formatting drift from plain `rustfmt` defaults —
essentially crate-wide; there's no `rustfmt.toml` anywhere in the repo,
so this looks like a deliberate, hand-maintained dense/aligned style
(e.g. `pub const ZERO: Self = Self { x: 0.0, ... }` kept on one line,
struct fields column-aligned) that plain `rustfmt` was never actually
run against, not accidental drift. Neither is in any file this pass
touched (confirmed directly: the four files this pass edited/moved
show up in `--files-with-diff` only because they live inside
already-non-conformant files, not because of anything added here —
zero *new* clippy warnings from this pass's own code).

**Why this was never visible as a CI failure:** both steps in
`mid-math-test.yml` are `continue-on-error: true` — and the one step
that *is* blocking (`cargo build -p mid-math`, "Build all crate
types") only runs a plain `cargo build`, which never compiles the
`#[cfg(test)]` module at all, so it wouldn't have caught the `mid_vec`
issue above either even if fmt/clippy were blocking. The two steps
that actually run `cargo test` (debug and release) are *also*
`continue-on-error: true`. This means the real state of this crate's
tests, lints, and formatting has effectively not been enforced by CI —
only whatever a person happened to check locally would have caught
any of this, which is exactly how the `mid_vec` compile blocker went
unnoticed. Worth a deliberate decision (tighten the workflow, or leave
it soft on purpose) rather than being rediscovered by accident again —
flagging plainly, not fixing the workflow here, since changing what
CI is allowed to fail on is a real call for you to make, not mine to
make silently.

Clippy categories found, not exhaustive: `needless_range_loop` (dozens,
mostly `curves/`), `clone_on_copy` (dozens, `curves/kochanek_bartels.rs`/
`bspline.rs`/`hermite.rs`/`cardinal.rs`), `missing_safety_doc` on a
large number of `extern "C"` functions across `ffi/*.rs` (real
API-documentation gaps, not stylistic), `should_implement_trait`
(`neg`/`shl`/`shr` methods that collide with std trait names),
`excessive_precision`/`approx_constant` (float literals, `noise/`,
`color/loglux.rs`, `f32/math.rs`), `cast_slice_from_raw_parts`
(`mid_vec/`), `doc_lazy_continuation`/`doc_overindented_list_items`
(same family of lint this project's own `mid-collections` fix already
dealt with once this cycle). A dedicated pass, not something to fix
inline here.

## Fixes and Problems

### `f32/vec2.rs`, `f32/{scalar,sse2,neon,wasm,coresimd}/vec3.rs`, `f32/{scalar,sse2,neon,wasm,coresimd}/vec4.rs`

The move to the `swizzle/` module directory (see that module's own top
comment) left the old per-type macro invocations behind in these 11
files. Both the old and new locations invoked
`impl_vec2_swizzle!`/`impl_vec3_swizzle!`/`impl_vec4_swizzle!` for the
same concrete types, a conflicting trait implementation (`E0119`) that
blocked every build touching `mid-math`, including real CI runs with
nothing to do with swizzle at all. Fixed by deleting the 11 old
invocations, each a standalone `crate::impl_vecN_swizzle!(...)` call
under a `// ── Swizzle ──` header and nothing else in the block, since
`swizzle/f32.rs` already covers every one of those types across every
backend. Verified with a direct test exercising `.xy()`/`.xyz()`/
`.xyzw()` on real `Vec2`/`Vec3`/`Vec4` values after the fix, not just a
clean build. Full `cargo test -p mid-math --lib` still passes 659/659
afterward.

Also noticed while running the full suite, unrelated to this fix: 5 of
the crate's doctests fail to compile (`camera/frustum.rs`,
`color/color32.rs`, `fixed/mod.rs`, `helpers/euler.rs`, `noise/fbm.rs`),
each missing an import or referencing an undefined variable in the
example code itself. Pre-existing, not touched here.
