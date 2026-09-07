# mid-geom

## Scope of this doc

This crate had no doc file before this pass. Rather than retrofitting a
full Overview and Modules breakdown for the whole crate in one sweep,
this starts with just the Fixes and Problems entry for what was
actually touched. The rest gets filled in incrementally as other files
in this crate are next touched for their own reasons, per
`DOCUMENTATION_AND_COMMENTING_GUIDELINES.md`'s own incremental update
rule.

## Fixes and Problems

### `d3/shapes/aabb.rs`, `d3/planes/frustum.rs`, `d3/transform/transform3d.rs`

All three files called `m.cols[c][r]` on a `mid_math::Mat4`. `Mat4`
has never stored a `cols` array; every backend (scalar, sse2, neon,
wasm, coresimd) stores four named `Vec4` fields instead: `x_axis`,
`y_axis`, `z_axis`, `w_axis`. `mid-math`'s own doc comment says this
storage changed from `[[f32; 4]; 4]` to the four named fields in an
earlier pass. These three call sites were never updated to match, and
nothing caught it because `mid-geom` was hidden behind `mid-math`'s own
unrelated compile failure (see `docs/mid-math.md`'s
`wide/float/avx2/vec3x8.rs` entry) until that failure was fixed. With
`mid-math` compiling again, these were the next thing blocking the
build, 21 real errors across the three files.

Fixed by switching every call site to the named fields. Where the
column and row were both compile-time constants (translation and
per-axis scale extraction in `aabb.rs`'s `transform` and
`transform3d.rs`'s `From<Mat4> for Transform`), this is a direct
one-to-one swap: `cols[3][0..2]` is `w_axis.x/.y/.z`, `cols[0][0..2]` is
`x_axis.x/.y/.z`, and so on. Column 0 is `x_axis` and so on all the way
through, confirmed against `Mat4::from_cols`'s own constructor, which
assigns its four `[f32; 4]` arguments to `x_axis`/`y_axis`/`z_axis`/
`w_axis` in that order.

Where the column and/or row were runtime loop variables
(`aabb.rs`'s 3x3 linear-part loop, `frustum.rs`'s row-extraction
closure), a small local closure picks the right named field by index
and then the right component of it. `Mat4::col()`/`Mat4::row()` exist
on the scalar backend and would have been a shorter fix, but they
aren't defined on sse2/neon/wasm/coresimd, so relying on them would
have broken again the moment this crate builds for anything other than
x86 scalar. The local closures use only the fields every backend has.

`mid-geom` had zero tests before this fix. Added 3 to
`d3/shapes/aabb.rs`: identity transform is a no-op, pure translation
shifts both corners equally, and non-uniform per-axis scale lands on
the right axis (this last one is the one that would have caught a
wrong column/row mapping immediately). `Frustum::from_mat4` and
`Transform::from<Mat4>` are still untested; flagging rather than
building out full coverage for the whole crate in this same pass.

`cargo test -p mid-geom --lib` passes 3/3. `mid-physics` (the only
other workspace crate depending on `mid-geom`) builds clean against
this fix.

Also noticed while in this crate, unrelated to the fix: 3 unused-import
warnings in `ffi.rs` (`Mat4`, `Vec2`, `Vec3`, `Frustum`) and in
`d2/shapes/rect.rs` (`EPSILON`). Pre-existing, not touched here.
