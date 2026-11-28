# mid-raster

## Overview

Integer-exact CPU coverage rasterizer for Mid Engine, the first slice of A2 of `docs/mid-ui-roadmap.md`. It
draws rectangles, rounded rectangles, thick lines and filled paths (lines, quadratic and cubic curves) into a
premultiplied `RGBA8` surface, through an axis-aligned clip stack, blending in sRGB or in linear light.
Anti-aliasing is exact area coverage computed in 24.8 fixed point, so the same input gives the same bytes on
every target. It follows `docs/mid-ui-contracts.md` (sections 1 and 2.5). It depends on `mid-math` only, has no
`unsafe` code, and the crate name is provisional (D7 of the roadmap leaves crate names open).

What this slice is, against the A2 text of the roadmap:

- **Built:** the path coverage rasterizer with both fill rules; a closed-form rectangle; rounded rectangles and
  thick lines (as paths through the same rasterizer, which gives exact area on their flattened outlines instead
  of a separate closed form); an axis-aligned clip stack; blending into `RGBA8` with the sRGB and the linear
  option; scalar code; the `tiny-skia` oracle in dev-dependencies; byte equality of golden scenes on x86_64
  and wasm32.
- **Not built yet:** a closed-form rounded rectangle and thick line as separate analytic functions; round caps,
  joins and dashes; gradients, images, shadows; text (A4 feeds glyph outlines to the path rasterizer); a
  translucent destination (A12); SIMD; sparse accumulation for shapes much larger than the screen; the backend
  that reads a `PaintItem` list (A3, which also adds the C API and the wasm host).

## Decisions

- **Signed-area cells, not supersampling.** Each line segment adds signed area to a grid of cells, and the
  running sum along a row is the winding area at a pixel. It is exact for axis-aligned rectangles and conserves
  every segment's contribution exactly, so nothing leaks whatever the rounding does. It is weakest where
  contours cross inside one pixel, where a sum cannot tell which part of the pixel is inside (section
  `cover.rs`).
- **All pixel work is integer.** A coordinate enters as `f32`, becomes 24.8 fixed point once (`round(v * 256)`,
  ties away from zero, held inside 2^20 pixels, non-finite means "draw nothing"), and nothing after that is
  floating point. Before that conversion only add, subtract, multiply, divide and square root are used, which
  IEEE 754 defines the same on every target.
- **One rounding function.** The blend uses `Color32::div255_round` and the `_rounded` helpers of `mid-math`, so
  there is one reference for the division by 255 in the workspace.
- **Coverage scales alpha, then the over.** `a' = div255(a * k)`, then the over of the contract onto an opaque
  destination (`docs/mid-ui-contracts.md`, section 1.2).
- **Clips snap to pixels.** A clip edge goes to the nearest pixel edge, ties up, so a clip never makes partial
  coverage. A clip with a non-finite value is an empty clip, which keeps pushes and pops paired.
- **The blend space is a canvas setting,** sRGB by default and linear light on request (the maintainer's answer
  to question 1 of B1).
- **Memory.** The accumulator is dense, `(width + 1) * height` `i32` cells for the region a shape touches inside
  the clip. That is fine for anything that fits the screen and wasteful for a shape far larger than the surface;
  a sparse form is later work.

## Modules

### `lib.rs`

**What it does:** crate docs, the module list, flat re-exports (`Canvas`, `Stats`, `ClipStack`, `IRect`,
`FillRule`, `Mask`, `Path`, `BlendSpace`, `Surface`, `SurfaceMut`) and one doctest.

### `fixed.rs`

**What it does:** the 24.8 fixed-point helpers: `from_f32`, `floor_div`, `div_round`, and the constants `SHIFT`,
`ONE` and `LIMIT`.

**Decisions:**
- `from_f32` is `round(v * 256)` with `f32::round` (ties away from zero), saturating at `LIMIT = 2^28`
  (2^20 pixels), so every product the rasterizer forms stays inside an `i64`. NaN and infinities give `None`.
- `div_round` is round-half-up for a positive divisor, `floor((2a + b) / 2b)`, and `floor_div` is Euclidean
  division, so negative coordinates round the same way as positive ones.

**Tests:** `tests/coverage.rs` (ties, saturation, non-finite, the helpers).

### `clip.rs`

**What it does:** `IRect` (half-open integer pixel rectangle) and `ClipStack`. The bottom entry is the surface
and is never popped. `push` intersects with the clip below and snaps the edges; `pop` reports whether anything
was open; `close_all` closes what is left and says how many.

**Tests:** `tests/canvas.rs` (nesting, snapping with ties up, a clip off the surface, a non-finite clip, the
bottom entry).

### `pixel.rs`

**What it does:** `SurfaceMut` (a borrowed buffer with an explicit stride, validated at construction), `Surface`
(an owned, tightly packed buffer), `BlendSpace` and `over(dst, src, space)`.

**Decisions:**
- The stride is never assumed to be `width * 4`. `SurfaceMut::new` returns `None` when the stride is below a
  row or the slice is too short; the last row needs no padding.
- Pixels are stored in the order R, G, B, A. The window surface is opaque in practice, so a destination counts
  as opaque and the color channels of `over` ignore its alpha, as `Color32::blend_over_rounded` does.

**Tests:** `tests/blend.rs`.

### `linear.rs`

**What it does:** the linear-light option of `docs/mid-ui-contracts.md`, section 1.3: `decode`, `encode`,
`over_channel` and `over`. Both colors are decoded to 16-bit linear light, mixed as
`(L(s) * a + L(d) * (255 - a) + 127) / 255` and encoded back by a binary search over 255 thresholds (nearest
level in the linear domain, ties to the lower). Alpha accumulates as `a + div255(dst.a * (255 - a))`.

**Decisions:**
- Opaque destinations only, as in the contract; translucent layers wait for A12.
- A source with alpha 255 or 0 returns early, so the endpoints are exact.

**Tests:** `tests/blend.rs` (known points, `E(L(v)) = v`, equality with a brute-force nearest search over all
65,536 inputs, 188 against 128 for half white over black, monotonicity, endpoints).

### `tables.rs`

**What it does:** generated constants `DECODE` (256 `u16`), `THRESHOLDS` (255 `u16`) and the FNV-1a checksum of
each. `tools/gen_linear_tables.py` writes the file in exact decimal arithmetic (the `decimal` module, no
floating point) and asserts that the decode table is strictly increasing and that encoding round-trips; its
header records the SHA-256 of both tables.

**Decisions:**
- The SHA-256 values in the header are the ones pinned in `docs/mid-ui-contracts.md`, section 1.3, and the
  generator reproduced them: `fdb7af3c01815a2118db8e50aac3c2304205ccc3f381f8976c588642c1aa60b6` for the decode
  table and `c094f907c4217d7ab90eb3ad0cc86caf2a04077855a5cce354562b3d7acce9c3` for the thresholds [run].
- The Rust side checks the FNV-1a of each table in a test, so a hand edit is caught without a hash dependency.

**Tests:** `tests/blend.rs`.

### `path.rs`

**What it does:** `Path` (move, line, quad, cubic, close; `f32` points) and `flatten`, which converts to fixed
point and cuts curves into line segments with integer arithmetic only.

**Decisions:**
- A curve is cut into `n` equal parameter steps. `n` is the smallest value in `1..=128` for which the
  deviation bound `|p0 - 2 p1 + p2| / (4 n^2)` (quadratic) or `3 m / (4 n^2)` with `m` the larger second
  difference (cubic) is within `FLATNESS`, 16 units or 1/16 pixel. The L1 norm stands in for the length, which
  is never smaller than the true one, so the bound holds. Points are evaluated exactly from the control points
  and rounded once.
- The chords lie inside the arcs, so a flattened curve is slightly smaller than the true one. On a rounded
  rectangle of side 20 and radius 6 the raster has 368.13 square pixels of coverage against 369.10 for the true
  shape, under one square pixel.
- Every contour closes implicitly, at a `move_to`, at a `close` and at the end. A segment with no current point
  starts a contour at its end point. A path with a non-finite coordinate flattens to `None`, so it draws
  nothing instead of part of itself.

**Tests:** `tests/coverage.rs` (deviation against the true curve in `f64`, exact end points, the step cap,
implicit closing, non-finite input).

### `cover.rs`

**What it does:** `rasterize(segments, region, rule)` returns a `Mask` of 8-bit coverage; `bounds` gives the
pixels a segment list touches; `FillRule` is `NonZero` or `EvenOdd`.

**How it works:** each segment is cut at the rows of the region, and within a row at the cell boundaries. Each
piece adds `d * (256 - f)` to the cell it lies in and `d * f` to the next, where `d` is its height and `f` the
mean x inside the cell, so the pair always sums to `d * 256`. The running sum of a row, in 1/65536 pixel, is the
winding area: non-zero clamps its magnitude to one pixel, even-odd folds it modulo two. Coverage is
`(area * 255 + 32768) >> 16`, which maps a full pixel to 255.

**Decisions:**
- Positions inside a segment come from one rounding formula applied to the original end points (`x_at`,
  `y_at`), and every piece is the difference of two such values, so the pieces add up to the segment exactly.
- A region is just a window. The part of a segment left of it counts as a vertical edge at its left side, the
  part right of it is dropped, and rows above and below are not visited, so the pixels inside are the same
  whatever region is asked for. A test rasterizes 200 random curved polygons twice, in full and in a window, and
  compares every pixel.
- Cells accumulate with wrapping adds. Intermediate cells can leave the `i32` range while the running sum stays
  inside it, and modular arithmetic gives the right sum anyway.
- A guard at the top of `add` skips a segment wholly above or below the region. It is redundant (the row range
  is empty then) and is kept as a cheap early exit; one mutant that removes it survives for that reason.

**Known limit:** where contours cross inside one pixel, or a self-intersecting contour overlaps itself there, the
sum is approximate. The oracle numbers below show it: the worst differences are at the crossings of a star and of
random self-crossing polygons.

**Tests:** `tests/coverage.rs` (26 tests: exact rectangles by closed form against the path rasterizer on 300
random rectangles, areas, both fill rules, opposite windings, the window property, flattening, shapes) and
`tests/oracle.rs`.

### `shapes.rs`

**What it does:** `rounded_rect` and `line` as paths, `KAPPA`, and the closed-form rectangle: `rect_coverage`
(one pixel), `rect_bounds` and `rect_mask`.

**Decisions:**
- A rounded rectangle is four lines and four cubic quarter circles with the control offset `KAPPA * r`. The
  radius is held between zero and half of the shorter side; a radius that is not finite counts as zero.
- A line is a rectangle with butt caps, centered on the segment. The normal needs one `sqrt` and two divisions,
  which IEEE defines exactly. A line with no length, no width or a non-finite value is `None`.
- The closed-form rectangle is the exact overlap area of the rectangle with the pixel, mapped with the same
  rounding as the rasterizer, so the two agree on every pixel (tested on random rectangles).

**Tests:** `tests/coverage.rs`, `tests/oracle.rs`.

### `canvas.rs`

**What it does:** `Canvas` ties a surface, a clip stack and a blend space together: `fill_rect`,
`fill_rounded_rect`, `stroke_line`, `fill_path`, `push_clip`, `pop_clip`, `finish`, and `Stats`, which counts
what had to be ignored or repaired (pops with nothing open, clips closed at the end, shapes that drew nothing)
as `docs/mid-ui-contracts.md` section 2.5 asks.

**Decisions:**
- A rectangle takes the closed-form path and writes each pixel by its own coverage; every other shape builds a
  mask for the part of its bounds that lies inside the clip and blends it.
- Random garbage (3,000 shapes and clips built from random `f32` bit patterns, including NaN, infinities and
  huge values) neither panics nor hangs; that is tested.

**Tests:** `tests/canvas.rs` (19 tests) and `tests/golden.rs` with `tests/golden/scenes.rs`: a scene of rectangles,
a rounded rectangle, a thick line, a star under even-odd, a ring, nested clips and translucent colors, drawn in
both blend spaces, plus 300 random curved paths under both fill rules, each hashed with FNV-1a 64 and compared
with a pinned value. `tools/wasm-hash` builds the same file into a wasm32 module and `check.mjs` compares the
hashes it computes with the pinned ones.

## Not built yet

See the list in the overview. Also open: the tolerance against `tiny-skia` is a measurement on a fixed set, not a
bound; performance has not been measured at all; and the `PaintItem` list is not consumed yet (A3).

## Verification status

Sandbox only, rustc 1.97.1:

- **69 tests pass:** 15 blend, 19 canvas, 26 coverage, 3 golden, 5 oracle, 1 doctest.
- **The oracle, `tiny-skia` 0.11.4,** on a 32 by 32 surface, white on transparent, anti-aliasing on, the same
  control points. The table is the largest and the mean difference over the pixels either side touches, on the
  0 to 255 scale, and the ceiling the test holds:

  | Shape | Max | Mean | Ceiling max, mean |
  |---|---|---|---|
  | rectangle, integer edges | 0 | 0.00 | 0, 0 |
  | rectangle, fractional edges | 26 | 8.30 | 32, 10 |
  | rectangle, thin (0.9 of a row) | 26 | 25.52 | 32, 28 |
  | rounded rectangle | 26 | 2.58 | 32, 3.5 |
  | circle (radius 11.5) | 22 | 1.36 | 28, 2 |
  | circle (radius 3) | 24 | 4.51 | 30, 5.5 |
  | ring, non-zero and even-odd | 17 | 2.09 | 24, 3 |
  | triangle | 20 | 1.45 | 26, 2 |
  | quadratic petal | 30 | 2.11 | 36, 3 |
  | star, non-zero | 35 | 2.85 | 44, 3.5 |
  | star, even-odd | 83 | 4.96 | 96, 6 |
  | line 1 px diagonal | 12 | 5.01 | 16, 6 |
  | line 3 px shallow | 19 | 3.96 | 24, 5 |
  | line 2 px steep | 20 | 4.99 | 26, 6 |
  | 40 random polygons, both rules | 147 | 4.43 (mean of means) | 160, 5.5 |

  `tiny-skia` is a port of Skia and quantizes coverage by supersampling, four rows to a pixel, so a pixel that
  is 0.9 covered can come back as 0.8, a difference of 26; the thin rectangle shows it on every pixel. This
  rasterizer computes the exact area, so on those pixels the difference is `tiny-skia`'s, not an error here. The
  large values are at crossings of contours (the star, the random polygons), where the signed-area sum is the
  weaker method.
- **Mutants:** 25 hand-made mutants, 24 caught by a failing test and 1 equivalent (the redundant guard in
  `cover.rs`, see above). They cover truncating instead of rounding, both fill-rule folds, the cell split, the
  mean x, the winding left of a region, the right-edge clamp, bounds rounding, flatness, cubic point rounding,
  closing of contours (two mutants, one of which survived until the test for a `move_to` closing the contour
  before it was added), snapping ties in the clip, clip intersection, `from_f32` flooring, rectangle rounding,
  the circle constant, the line half width, the clip size, pop counting, the encode tie rule, the linear
  rounding offset and the surface length check.
- **clippy** with `--all-targets --no-deps`: no diagnostics in `mid-raster`.
- **wasm32:** `cargo check -p mid-raster --target wasm32-unknown-unknown` passes. A wasm32 module built from
  `tools/wasm-hash` and run in node (V8) computes the same three hashes as the x86_64 test: the golden scene in
  sRGB (`0x2a105a8f37715748`) and in linear light (`0xcbc2246818347507`), and the 300 random curved paths under
  both fill rules (`0xf856c062fb265d6f`) [run]. That is integer exactness between x86_64 and wasm32, on this
  scene and this path set.
- **Not run:** aarch64 or 32-bit ARM (so the phone), rustc 1.75, rustfmt, real CI, speed.

## CI and Workflows

`.github/workflows/mid-raster-test.yml` (`workflow_dispatch` only, like the others) runs clippy, the tests and
the wasm32 check, and a third job builds `tools/wasm-hash` for `wasm32-unknown-unknown` and runs
`node crates/mid-raster/tools/wasm-hash/check.mjs` on it. It has not run on CI yet.

## Fixes and Problems

### The test runner for wasm32-wasip1 does not work in the sandbox

`cargo test --target wasm32-wasip1` failed to link here. The first failure was the linker call itself (the
distribution's `rust-lld` rejects the `-flavor wasm` arguments rustc passes); with a small wrapper that drops
them, the link failed again on `pthread_join` and `pthread_detach`, which the distribution's `std` for that
target needs from a wasi libc that is not installed. The cross-target check therefore uses a
`wasm32-unknown-unknown` module and node, which needs neither.

### A flattened curve is a little smaller than the true one

Found while testing a rounded rectangle: 368.13 against 369.10 square pixels. It follows from the chord
tolerance of 1/16 pixel and is documented under `path.rs`; a finer `FLATNESS` costs more segments.
