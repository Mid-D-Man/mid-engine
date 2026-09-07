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

Note on this doc's own writing: the sections above this one predate
`DOCUMENTATION_AND_COMMENTING_GUIDELINES.md` and use em dashes
throughout. The guideline says not to. New entries below follow the
real rule; the older sections were left as they were rather than
rewritten, since that would be an unrelated sweep.

### `wide/float/avx2/vec3x8.rs`, `swizzle/wide_float.rs`

`wide_float.rs` invoked `impl_vec3_axis_swizzle!` on `Vec3x8`, the same
macro used for `Vec3x4`. That macro builds a struct literal assuming
public `x`/`y`/`z` fields, which is true for `Vec3x4` (its own doc
comment says the fields are public for exactly this kind of use) but
not for `Vec3x8`. `Vec3x8` stores two `Vec3x4` halves instead (`lo`,
`hi`), on purpose, to avoid holding a raw `__m256` outside a
`target_feature`-gated scope. The macro invocation produced `E0560`/
`E0609` on every field access, blocking any build that compiled
`mid-math` at all, including `ecs-vs-bevy-ecs`'s real CI bench run
(build #9).

This invocation could not have caused a failure before now. `Vec3x8`,
`f32x8`, and `Mask8` used to be compiled only behind the crate's `avx2`
target feature; `wide/float/mod.rs`'s own doc comment explains the
recent change to always compile them on x86/x86_64 instead, so a
non-AVX2 build wouldn't link-error on types it never touches. That
change is what exposed this: the broken invocation was sitting in the
tree the whole time, just never previously type-checked on a normal
x86/x86_64 build.

Fixed by writing `Vec3AxisSwizzle` for `Vec3x8` directly in
`vec3x8.rs`, one method per axis permutation, each delegating to
`self.lo`/`self.hi` (both already implement the trait via the same
macro, since `Vec3x4`'s fields are public) and recombining with
`from_halves`. This is the same shape every other method on `Vec3x8`
already uses (`mul_elem`, `scale`, `min`, `max`, and so on), so it's
not a new pattern for this file. The broken macro invocation in
`wide_float.rs` was removed and replaced with a comment pointing to
where the real impl lives and why.

`Vec3x8` and `f32x8` had no test coverage at all before this fix.
Added 5 tests to `tests/wide_tests.rs`: two checking `Vec3x4`'s own
axis swizzle against scalar math (also previously untested), and three
for `Vec3x8` checking a swizzle result against scalar math, checking
`xyz()` is the identity, and cross-checking `Vec3x8`'s composed result
against applying the same swizzle to its two `Vec3x4` halves
independently. `cargo test -p mid-math --lib` passes 664/664
afterward.

Also fixed while in this crate: 5 unused imports left over from the
earlier swizzle duplicate-invocation fix (`f32/scalar/vec3.rs`,
`f32/scalar/vec4.rs`, `f32/sse2/vec4.rs`) were still present on the
real repo. The tarball that landed was an earlier version of that fix,
before the import cleanup. Removed here; `cargo check -p mid-math`
reports zero unused-import warnings now.

Verified `mid-anim`, `mid-collections`, and `mid-ecs` (the three other
workspace crates depending on `mid-math`) all still build clean against
this fix. `mid-ecs`'s own test suite still passes 176/176.

Not run: `ecs-vs-bevy-ecs` itself. Its `bevy_ecs` pin needs rustc
1.95+; the sandbox's best available apt package is rustc-1.91, so this
crate has never been buildable here (documented in its own `Cargo.toml`
already). The real numbers from the query2_static diagnostic bench
still need an actual CI run once this fix lands.

### `f32/mat4.rs` (found, not fixed this pass)

With `mid-math` compiling again, `mid-geom` and `mid-physics` (both
depend on it) fail to build: 21 real errors, all `Mat4` missing
`x_axis`/`y_axis`/`z_axis`/`w_axis` fields at various call sites in
`mid-geom`. This was hidden entirely behind `mid-math`'s own compile
failure until now. Unrelated to the swizzle fix above and doesn't
block `ecs-vs-bevy-ecs` (which only needs `mid-ecs`, not `mid-geom`),
so it wasn't touched here. Flagging plainly rather than fixing it
without being asked, since it's a real, separate break.
