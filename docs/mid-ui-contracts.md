# mid-ui contracts: pixel format, paint list, text types, measure

Status: **proposed**, for review by the maintainer. Written 2026-11-26 as slice B1 of
`docs/mid-ui-roadmap.md`. Evidence tags as in the roadmap: **[src]** read in the source of this repository
(the file is named), **[doc]** stated in documentation, **[run]** measured in the sandbox, *reasoned*
follows from what was read but was not tested. *Proposed* marks a rule that this document introduces and
the maintainer has not yet confirmed. The provisional defaults of D3, D4, D9 and D14 in the roadmap's table
are the starting point and are marked where they apply; each can change before A2 starts.

## 0. The short answer

Four contracts, each small enough to test:

1. **Pixels.** Surfaces are premultiplied `RGBA8`. Colors in the paint list stay straight sRGB and are
   converted once, at the edge. One integer rounding function, `div255`, is used everywhere, and every
   float coordinate becomes fixed point before any pixel work, so the output is the same bytes on every
   architecture (D4, D14).
2. **The paint list.** `PaintItem` (36 bytes, five kinds) and `DrawCmd` stay two vocabularies. The CPU
   backend reads `PaintItem` first (D3). Two kinds are added, a positioned-glyph run and a slot for
   host-drawn content, and every consumer skips and counts a kind it does not know.
3. **Text types.** Glyph ids, clusters as byte ranges, and a script and direction on every run, with no
   ASCII or one-glyph-per-character assumption (Decision 16).
4. **Measure.** A trait of four methods, the three of the harness stack plus the minimum content width. The
   measure returns what the renderer draws (D9).

Nothing here needs code outside tests to settle. The open points are in section 5.

## 1. Pixel format and color

### 1.1 What exists

- `Color32` is `repr(C)`, four `u8` fields in the order r, g, b, a, documented as sRGB
  [src, `crates/mid-math/src/color/color32.rs`]. `PaintItem.color` is `0xRRGGBBAA`, built by `rgba()`
  [src, `crates/mid-editor/src/paint.rs`]. That is the packing of `Color32::to_u32_rgba`, so converting
  between the two is a bit-for-bit no-op.
- `Color32::premultiply` and `Color32::blend_over` divide by 255 with truncation. `blend_over` mixes the
  color channels as if the destination were opaque, which is right for an opaque framebuffer and is not a
  general over operator for translucent layers [src, same file].
- `Rgba` is linear-light `f32` and is not on the paint path [src, `crates/mid-math/src/color/rgba.rs`].

### 1.2 The contract (proposed, D4 and D14 default)

| Item | Rule |
|---|---|
| Surface | `RGBA8`, 4 bytes per pixel, bytes in the order R, G, B, A, origin top left, rows `stride` bytes apart. The stride is always passed explicitly and is never assumed to be `width * 4`. |
| Alpha in surfaces | Premultiplied. The window surface is opaque in practice, where premultiplied and straight alpha agree. |
| Colors in the paint list | Straight sRGB `Color32` or `0xRRGGBBAA`. Converted to premultiplied once, when the backend reads an item. |
| Rounding | One function, `div255(x) = (x + 128 + ((x + 128) >> 8)) >> 8`, for `x` in `0..=65025`. It equals round-half-up of `x / 255`; all 65,026 inputs were compared against the exact rational in the sandbox [run]. |
| Premultiply | `c' = div255(c * a)` per color channel; alpha unchanged. |
| Over | Premultiplied source `s` on premultiplied destination `d`: `d' = s + div255(d * (255 - s.a))` per channel, alpha included. |
| Coverage | An 8-bit value, 255 meaning fully covered. Applied to a premultiplied color as `div255(c * k)` per channel. |
| Blend space | The encoded sRGB numbers are blended directly, with no conversion to linear light. Open, see section 5, question 1. |
| Pixel geometry | Pixel `(i, j)` covers `[i, i + 1)` by `[j, j + 1)`, y down. A rectangle with integer edges covers whole pixels at coverage 255 and adds no blending beyond its own alpha. |
| Fixed point | Item coordinates are `f32`. Each is converted by `round(x * 256.0)` (ties away from zero) to a 24.8 fixed-point integer, saturating. A non-finite coordinate makes the item draw nothing. After that conversion only integer arithmetic is used. |
| Float operations before the conversion | Add, subtract, multiply, divide, square root and comparison only. Their results are defined by IEEE 754 and agree on every target; sine, cosine, exponential, logarithm and power are library-defined and are not used before the conversion (*reasoned*). |
| Golden images | Stored as raw `RGBA8` bytes with a hash. The bytes must be equal on x86_64, aarch64, armv7 and wasm32. This applies to the CPU backend only; the raylib host draws with its own float code and is not integer-exact. |

The backend has its own `div255` and does not call `Color32::premultiply` or `Color32::blend_over`. Those
keep truncating, because changing them changes mid-math results and benchmark numbers. For the same
quotient, truncation and rounding differ by at most one level (*reasoned*), so the two agree on every value
that is a multiple of 255 and are one level apart on some others. Whether mid-math should round too is
question 2 in section 5.

Not in scope: images, gradients, shadows and layers with their own alpha (A12).

## 2. The paint list

### 2.1 What exists

`PaintItem` is `repr(C)`, nine 4-byte fields: `kind`, `color`, `a`, `b`, `c`, `d`, `extra`, `text_start`,
`text_len`, 36 bytes [src, `paint.rs`]. Five kinds:

| Kind | Value | `a`, `b` | `c`, `d` | `extra` | Text pool |
|---|---|---|---|---|---|
| rect | 0 | x, y | width, height | corner radius | none |
| push clip | 1 | x, y | width, height | unused | none |
| pop clip | 2 | unused | unused | unused | none |
| line | 3 | start x, y | end x, y | width | none |
| text | 4 | x, y (top left) | unused | size | bytes `text_start..text_start + text_len` |

`PaintList` holds `items` and a UTF-8 `text` pool that is not NUL-terminated [src, `paint.rs`].

`DrawCmd<R>` has three variants, `FillRect { rect, color, radius }`, `PushClip(Rect)` and `PopClip`. It is
`#[non_exhaustive]`, generic over `f32` and `f64`, and `build_draw_list` skips hidden subtrees, fully
transparent fills and empty rectangles [src, `crates/mid-ui/src/draw.rs`].

### 2.2 Two vocabularies, one conversion (Decision 14)

`DrawCmd` converts to `PaintItem` for the three kinds both have:

| `DrawCmd` | `PaintItem` |
|---|---|
| `FillRect { rect, color, radius }` | kind 0; `a, b` = rect x, y; `c, d` = width, height; `extra` = radius; `color` = `to_u32_rgba` |
| `PushClip(rect)` | kind 1; `a, b, c, d` as above |
| `PopClip` | kind 2 |

For `Real = f64` each coordinate is narrowed to `f32` by rounding to nearest. The reverse direction is
defined only for these three kinds, because lines and text have no `DrawCmd` form today. `DrawCmd` is
non-exhaustive, so variants can be added when `mid-ui` starts to produce them without breaking users.

### 2.3 First consumer (D3 default)

The CPU backend reads `PaintItem` first. It is flat and C-compatible, already has lines and text, and every
host speaks it. A new draw kind therefore lands in `PaintItem` first and in `DrawCmd` when `mid-ui`
produces it.

### 2.4 Kinds added (proposed)

The item stays 36 bytes and kinds 0 to 4 keep their meaning.

| Kind | Value | Fields |
|---|---|---|
| glyphs | 5 | `a, b` = baseline origin of the run; `extra` = size in pixels; `c` = font index stored as an integer in `f32` (exact up to 16,777,216); `d` reserved, zero; `color`; `text_start`, `text_len` = a range in the glyph pool, counted in entries |
| custom | 6 | `a, b, c, d` = rectangle; `text_start` = slot id; the rest zero |

The glyph pool is a second vector in `PaintList`. Its entry is `GlyphPos { glyph: u32, x: f32, y: f32 }`,
12 bytes, `repr(C)`, with the offset from the run's origin in pixels after shaping and after hinting
rounding, if any.

- **Custom** is the slot for content the host draws itself, the 3D viewport first. It obeys paint order and
  clips. The CPU backend draws nothing in it and returns the slot rectangles of the frame, so the host can
  fill them.
- **Unknown kinds.** A consumer that meets a kind it does not know skips the item and counts it. It never
  fails and never panics.
- **Reserved fields** are written as zero. In particular `extra` on a push clip stays zero, which keeps a
  rounded clip open as a later extension with no new kind. Rectangular clips only for now.
- **`KIND_TEXT` stays** for hosts that draw strings with their own fonts. The CPU backend does not read it.
  A lowering pass in the text crate turns a list with text items into one with glyph items; it is a pure
  function from list to list. A text item that reaches the backend is skipped and counted.
- **Script and direction are not in the paint item.** Positioned glyphs are already in drawing order.
  Direction and script matter to shaping and measuring, section 3 (*reasoned*).

### 2.5 Behavior on bad lists (proposed)

- A pop with no open clip is ignored and counted.
- Clips still open at the end of the list are closed.
- A clip rectangle is intersected with the one below it. An empty intersection clips everything until its
  matching pop.
- A rectangle with a negative or non-finite size draws nothing.
- Lines have butt caps. The width is centered on the segment and the edge is antialiased by coverage.

## 3. Text types

### 3.1 What exists

- `TextMeasure` has one method, `width(&self, text: &str, size: f32) -> f32`. `FixedWidth` multiplies the
  character count by a factor and `fit` shortens a string with three dots one character at a time
  [src, `crates/mid-editor/src/text.rs`].
- The C host gives the core a callback of the shape `(user, text pointer, length, size) -> f32`, and a
  result that is not finite falls back to a plain estimate [src, `crates/mid-editor-ffi/src/lib.rs` and its
  test `a_measure_that_returns_nonsense_falls_back_to_a_plain_estimate`].
- `mid-ui` has no text at all: none of its eight files mentions text, a measure or an intrinsic size
  [src, `crates/mid-ui/src`].
- The raylib host loads one font atlas at 64 pixels and scales it at draw time [doc, `docs/mid-ui-status.md`].

All of these assume one font, left to right, and a character count that stands in for a width. Decision 16
rules that out.

### 3.2 The types (proposed; crate names are open, D7)

| Type | Shape | Notes |
|---|---|---|
| `FontId` | `u32` | An index into the font set; also the `c` field of a glyph item |
| `GlyphId` | `u32` | Room for formats with more than 65,535 glyphs |
| `Script` | four ASCII bytes, ISO 15924 | `Latn`, `Arab`, `Deva`, and so on |
| `Direction` | `Ltr` or `Rtl` | A vertical value is reserved and not implemented |
| `Hinting` | `Off` today | Light vertical-only hinting is the first step after it (D5) |
| `TextStyle` | `{ font: FontId, size: f32, hinting: Hinting }` | The same value goes to measuring and to drawing |
| `Run` | `{ start: u32, end: u32, script, direction, level: u8, font }` | A stretch of one font, one script and one direction, as byte offsets into the UTF-8 text; `level` is the bidirectional embedding level |
| `ShapedGlyph` | `{ glyph, cluster: u32, x_advance, y_advance, x_offset, y_offset }` | Advances and offsets in `f32` pixels; `cluster` is the byte offset where the glyph's cluster starts |
| `ShapedRun` | `{ run, glyphs range, width: f32 }` | The output of shaping one run |

Rules:

1. Text is UTF-8. Offsets are bytes, always on `char` boundaries.
2. No API takes a character count as a width. `FixedWidth` stays a test double and is never a default.
3. A cluster maps many to many. One glyph can stand for several characters (a ligature) and several glyphs
   for one character (a base and marks). The shaped output keeps the cluster so that the caret, the
   selection and hit testing can map a position back to the text. In a right-to-left run the cluster values
   decrease along the glyph order.
4. Tier T1 (Latin) fills the types with script `Latn`, direction `Ltr`, level 0, kerning applied and one
   glyph per scalar value. The fields exist from the start so that tiers T2 to T4 are added behind the same
   interfaces (Decision 16).
5. A character with no glyph in the font draws the font's missing-glyph shape, glyph 0, and keeps its
   advance, so a width stays honest (*reasoned*).
6. A run has one font. Fallback to another font splits the run. Where the fallback logic lives is open.

## 4. The measure contract

### 4.1 The trait (proposed)

The harness stack's typesetter side is three methods: the width of a string on one line, the height of a
string wrapped at a width, and the first and last baselines [src, `docs/research-ui-harness.md`]. The size
model of A1 also asks for minimum sizes, and the width of the widest unbreakable piece of text cannot be
derived from those three, so this contract adds a fourth.

```rust
pub trait TextMeasure {
    /// Width of the widest line when only hard breaks apply.
    fn line_width(&self, text: &str, style: &TextStyle) -> f32;
    /// Total height when wrapped at `max_width`.
    fn wrapped_height(&self, text: &str, style: &TextStyle, max_width: f32) -> f32;
    /// Distance from the top to the first and to the last baseline.
    fn baselines(&self, text: &str, style: &TextStyle, max_width: f32) -> (f32, f32);
    /// Width of the widest piece that cannot be broken.
    fn min_content_width(&self, text: &str, style: &TextStyle) -> f32;
}
```

### 4.2 Rules (proposed)

1. **Pure.** A result depends only on the arguments and the font data. Internal caches are allowed.
2. **Finite and not negative.** A host callback that returns a negative or non-finite number is replaced by
   a plain estimate, as the C callback is today.
3. **Edges.** `line_width("")` is 0. `wrapped_height` with `f32::INFINITY` is the height with hard breaks
   only. A `max_width` below `min_content_width` is treated as `min_content_width`.
4. **Order.** `min_content_width <= line_width`, and the first baseline is at most the last baseline, which
   is at most `wrapped_height` at the same width.
5. **Monotone.** `wrapped_height` never grows when `max_width` grows. This is required of the first
   implementation and tested as a property.
6. **Hinted against unhinted (D9 default).** The measure returns what the renderer draws. The measure and
   the renderer share one advance function, and `line_width` of a single-line string is exactly the sum of
   the advances of the glyphs the renderer would draw, accumulated left to right in `f32`. When hinting
   rounds advances to whole pixels, the measure rounds the same way. The test is equality, not a tolerance.

### 4.3 What A1 takes from this

A leaf reports `Intrinsic { min_width, max_width }`, which are `min_content_width` and `line_width`, and the
solver asks `wrapped_height` for a width it has chosen. The solver never sees a font. A1 is built against a
fixed-width test double, so the solver and the real text code can be developed apart.

### 4.4 The C boundary

The single-method callback stays for hosts at tier T0. A second form is a struct of four function pointers
with a version field first. A host that supplies only the old callback gets the other three methods
estimated from its width. Existing hosts keep working (*reasoned*).

## 5. Open points for the maintainer

| # | Question, in plain words | Default taken here |
|---|---|---|
| 1 | Blend colors as the plain numbers stored in the image (sRGB), or convert to linear light and back? | Plain numbers. Exact, cheap, and no tables. Revisit if small text looks thin |
| 2 | The backend rounds when it divides by 255, the mid-math color helpers truncate. Keep both, or change mid-math to round? | Keep both. The backend has its own function |
| 3 | Add the glyph run and the custom slot as new paint kinds now? | Yes |
| 4 | A fourth measure method, `min_content_width`? | Yes |
| 5 | Where do the caret, the selection and composition (IME) live: the text crate, or the widget layer above it? | Not decided here. Nothing in sections 1 to 4 blocks either answer |

D3, D4, D9 and D14 are used as the roadmap's provisional defaults. D5 and D6 do not change anything in this
document.

## 6. What later slices check

- **A2:** `div255` over all 65,026 inputs against the exact rational; coverage of an integer-edged
  rectangle is 255 on its pixels and 0 outside; the fixed-point conversion at ties, saturation and
  non-finite input; equal bytes on x86_64, aarch64, armv7 and wasm32.
- **A3:** `DrawCmd` to `PaintItem` conversion on all three kinds; bad lists as in section 2.5; unknown
  kinds counted and skipped.
- **A4:** the measure rules of section 4.2 as property tests, including the advance equality of rule 6.

## Sources

Read for this document: `crates/mid-editor/src/paint.rs`, `crates/mid-editor/src/text.rs`,
`crates/mid-editor-ffi/src/lib.rs`, `crates/mid-ui/src/draw.rs`, `crates/mid-ui/src/style.rs`,
`crates/mid-math/src/color/color32.rs`, `crates/mid-math/src/color/rgba.rs`,
`docs/research-ui-harness.md`, `docs/research-ui-text.md`, `docs/mid-ui-roadmap.md`,
`docs/mid-ui-status.md`.
