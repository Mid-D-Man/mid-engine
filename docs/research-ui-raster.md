# CPU rasterizer comparison: tiny-skia, Vello CPU, zeno, fontdue, font-rs and ab_glyph_rasterizer

## Overview

Fifth research document of the UI series. Decision 14 puts a CPU backend first in the order of
implementation, with the GPU backend taking priority in a shipped build, and Decision 15 requires an
FFI surface and wasm testing for each new UI crate. This document reads the Rust rasterizers that
exist, against what the editor's paint list needs, so that the first backend crate is sized before it
is written.

This is reading, not measuring. Nothing was built, run or benchmarked, and the only speed figures are
ones the projects state about themselves, marked **[doc]**. Other tags: **[src]** read in source or a
manifest, **[gap]** searched for and not found. Where a statement is reasoning and not a finding, the
text says so.

## Method

| Source | Commit or version | What was read |
|---|---|---|
| `linebender/tiny-skia` (0.12.0) | `5d47547`, 2026-08-20 | `README.md`, `Cargo.toml`, `src/wide`, `src/scan/path_aa.rs` header, sizes |
| `linebender/vello`: `vello_cpu` 0.4.0, `vello_common`, `glifo` 0.5.0 | `a8d6e27`, 2026-11-25 | `vello_cpu/src/lib.rs` documentation, manifests, `ARCHITECTURE.md` mentions, sizes |
| `dfrg/zeno` (0.3.3) | `334a74a`, 2026-10-27 | `README.md`, `src/raster.rs` header, sizes |
| `mooman219/fontdue` (0.9.5) | `1a84e6d`, 2026-10-18 | `src/raster.rs`, `src/platform/mod.rs` |
| `raphlinus/font-rs` | `297f749`, 2020-09-05 | `README.md`, size |
| `alexheretic/ab-glyph`, crate `ab_glyph_rasterizer` 0.1.10 | `5db324d`, 2026-11-23 | `README.md`, manifest, size |
| `Mid-D-Man/mid-engine` | `a9c1810`, 2026-11-25 | `docs/mid-editor.md` (paint list), `docs/mid-ui-wasm.md`, `docs/roadmap.md` (Decisions 11, 12, 14, 15) |
| `Mid-D-Man/msx` | `490b204`, 2026-11-24 | the manifest of its CPU renderer |

Not read: `fearless_simd` (Vello CPU's SIMD layer), the sparse-strips algorithm in code, tiny-skia's
pipeline stages in detail, and any of these libraries on a 32-bit ARM device.

## What the paint list needs

The editor's paint list has five kinds: rectangle, clip push, clip pop, line and text. A color is
`0xRRGGBBAA`, an item has four coordinates and one extra number (radius, width or text size), and text
is a range into a pool [doc, `docs/mid-editor.md`]. The C host draws rounded rectangles, scissor clips,
thick lines and text from it [doc]. The browser playground draws `mid-ui`'s own list with Canvas 2D
[doc, `docs/mid-ui-wasm.md`]. Later needs named in the decisions: images, gradients and shadows
(roadmap U3 and Decision 12), glyphs with hinting (Decision 15), and the vector content of msx, whose
own CPU renderer is built on `tiny-skia` [src].

## Summary

| Library | Scope | Rust source lines | Rust floor | AA method | Fill rules | Strokes | Blend and paint | Clip |
|---|---|---|---|---|---|---|---|---|
| tiny-skia | Complete 2D: fill, stroke, dash, gradients, patterns, images | 15,469 + 7,092 (`path`) | stable (README) | Supersampled scan conversion, `SUPERSAMPLE_SHIFT` 2, from Skia [src] | Both | Yes, with dashing | Blend modes, `lowp` and `highp` pipelines [src] | Masks and rectangles |
| Vello CPU | Complete 2D with the Sparse Strips architecture, glyph runs through `glifo` | 14,061 + 16,399 (`vello_common`) | 1.89 | Not read in code [gap] | Not read | `stroke_path` [src] | A `u8` pipeline for speed and an `f32` pipeline for quality [src] | Not read |
| zeno | Path to alpha or subpixel mask | 6,027 | 1.63 | Integer cell coverage, 256 levels [src] | Non-zero and even-odd [src] | Yes, with dashes | None | None |
| fontdue `raster.rs` | Glyph outline to alpha | 125 | none declared | `f32` accumulation buffer [src] | One | None | None | None |
| font-rs | Glyph outline to alpha | 1,653 | not read | Dense accumulation buffer, like libart [doc] | not read | None | None | None |
| `ab_glyph_rasterizer` | Lines, quadratic and cubic curves to coverage | 553 | not read | Coverage accumulation [doc] | not read | None | None | None |

| Library | SIMD targets | `no_std` | License |
|---|---|---|---|
| tiny-skia | x86 (SSE to AVX2), WebAssembly SIMD128 and relaxed SIMD, AArch64 NEON; "no effect on other targets" [src, manifest] | `no-std-float` feature [src] | BSD-3-Clause |
| Vello CPU | through `fearless_simd`; the targets it supports were not read [gap] | `libm` feature in place of `std` [src] | Apache-2.0 OR MIT |
| zeno | not checked [gap] | attribute present [src] | Apache-2.0 OR MIT |
| fontdue | x86 and x86_64 only; everything else uses a portable fallback [src] | yes [doc] | MIT OR Apache-2.0 OR Zlib |
| font-rs | an optional SSE3 module written in C [doc] | not read | not read |
| `ab_glyph_rasterizer` | none stated | `libm` feature [src] | Apache-2.0 |

## Findings by area

### Four ways to get anti-aliased coverage

1. **A dense accumulation buffer** (font-rs, `ab_glyph_rasterizer`, fontdue's `raster.rs`). Each edge adds
   signed area to a buffer the size of the glyph; a running sum turns it into coverage. It is the
   smallest approach (125 to 1,653 lines) and made for glyphs. It gives coverage only, with no strokes,
   paint or clipping [src, doc]. fontdue's header warns against reusing its code because it relies on
   unsafe invariants that its callers sanitize [src].
2. **Integer cell coverage** (zeno). Edges are walked into cells with an exact-area coverage in integer
   math, 256 levels, with both fill rules, and strokes and dashes on top, in 6,027 lines [src].
3. **Supersampled scan conversion with a blit pipeline** (tiny-skia, a port of Skia). A full renderer:
   paint, gradients, blend modes, clips, in 15,469 lines plus 7,092 for the path library [src]. Its
   README says it must produce the same output as Skia [doc].
4. **Sparse strips** (Vello CPU, and Vello GPU which shares the architecture). Not read in code; the
   projects describe it as the shared architecture of both renderers [doc, `ARCHITECTURE.md`].

Reasoning, not a finding: the paint list needs rectangles, rounded rectangles, lines, an
axis-aligned clip stack and glyph masks. Rectangle, rounded rectangle and line coverage can be
computed in closed form, a scissor clip is a rectangle intersection, and only glyph outlines need a
path rasterizer. That is much less than tiny-skia or Vello CPU provide, and closer to what family 1 or
2 plus a small own renderer would cover. Vector content such as msx's would need family 3 or 4, or
the own rasterizer grown to match.

### 32-bit ARM has no SIMD path in anything read

tiny-skia's README gives performance as 20 to 100 percent slower than Skia on x86-64 and about 100 to
300 percent slower on ARM, and enables SIMD for x86, WebAssembly and AArch64 only [doc, src]. A
32-bit ARM target therefore runs its scalar code. fontdue selects its SIMD module for x86 and x86_64
only [src]. Vello CPU's SIMD layer was not read [gap]. This matches `mid-math`, which runs scalar on
`armv7` for now (Decision 15), so the phone's first CPU backend is scalar end to end, and the later
hand-written armv7 backend would serve the rasterizer as well as the math.

### Pixel math: `u8` and `f32` pipelines

Vello CPU offers two pipelines and a render mode for each, `OptimizeSpeed` on `u8` math and
`OptimizeQuality` on `f32` math, and one of the two features must be enabled [src]. tiny-skia has a
`lowp` and a `highp` pipeline [src]. The shared idea is premultiplied pixels in 8 bits for speed and
`f32` for accuracy. Which pixel format `mid-math`'s `Color32` and its color spaces map to was not
checked here [gap] (question 3).

### Glyph rendering

`glifo` is Vello's answer for text: positioned glyphs in, a glyph cache, shared hinting instance and
hinted advance, an atlas directory, and 4,441 lines [src, doc]; it calls itself experimental [doc]. The
other pattern read is egui's, where a CPU renderer fills each glyph once into a bitmap that goes to an
atlas (`docs/research-ui-text.md`). Both keep rasterization behind a cache, so the rasterizer's speed
matters once per glyph and size, not per frame.

### Testing and determinism

Integer coverage and `u8` pipelines have no floating-point rounding that depends on the CPU; whether
the candidates produce identical bytes on every architecture was not checked [gap]. If an own
rasterizer is integer-exact, the same pixels come out on x86_64, in the browser and on the phone, so
the editor's self-test scenes can become golden images in CI and in the wasm test. That is reasoning,
not a finding. The Skia-exactness claim of tiny-skia [doc] shows the pattern: a reference
implementation as the oracle.

### The existing msx renderer

msx's CPU renderer uses `tiny-skia` [src], and the DixScript-driven document there is the same family
of idea as `.mxui`. An own rasterizer in this repository could later serve msx, or msx could keep
`tiny-skia`; this is a question of scope (question 4), not of reading.

## A first-crates sketch, proposal only

Names are placeholders; crate names are open in Decision 14.

| Piece | Contents | Dependencies | FFI and wasm (Decision 15) |
|---|---|---|---|
| A coverage rasterizer | Closed-form rectangle, rounded rectangle and line coverage; a path coverage rasterizer for glyph outlines; blending into premultiplied `RGBA8`; a rectangle clip stack | `mid-math` only | The core is pure Rust and wasm32-clean like `mid-ui` |
| The text crate | Font reader, outlines, hinting, glyph cache, as in `docs/research-ui-text.md` and `docs/research-ui-hinting.md` | `mid-math` only | A C API with the shape of `mid-editor-ffi`; a wasm test job like `mid-ui-wasm-test.yml` |
| The CPU backend | Draws a paint list into a framebuffer | the two above | A C API: an opaque context, a framebuffer pointer, a paint list in, NULL-tolerant, entry points in `catch_unwind`, a header checked against the symbols; in the browser, the same core writing into `ImageData` |

Tests, using the dev-dependency permission of Decision 15: `tiny-skia` as the oracle for shape coverage
within a tolerance, Skrifa and FreeType for glyphs, hand-made mutants as in the other crates, and
golden images from the editor's self-test scenes.

## A probe, designed and not built

Three parts, in order.

1. **Coverage.** The own coverage rasterizer against `tiny-skia`, `zeno` and `ab_glyph_rasterizer` on a set
   of rounded rectangles, lines and glyph outlines: maximum and mean coverage difference per shape.
2. **The paint list.** The editor's own benchmark scenes (its 1000 and 5000 cube scenes) rendered by the
   CPU backend and by the raylib host, on x86_64 CI, in the browser and on the phone through the
   Console: frame time and pixels per second, with the CPU line and the build flags in the header.
3. **Glyphs.** The Latin glyph probe of `docs/research-ui-text.md`, with the hinted run of
   `docs/research-ui-hinting.md`.

The phone and real CI are the authority; a sandbox x86_64 number says nothing about a 32-bit phone.

## Gaps, ranked

1. **A coverage rasterizer,** shared by shapes and glyphs, scalar first.
2. **The paint list renderer:** rounded rectangle, line, scissor clip, glyph mask blit, premultiplied
   `RGBA8` blending.
3. **The pixel format and color conventions** against `mid-math`'s `Color32`.
4. **The FFI crate and header** for the backend, following `mid-editor-ffi`.
5. **The wasm path:** the same core into a canvas, with a test job.
6. **Golden-image tests** from the editor's self-test scenes.
7. **Images, gradients and shadows** after the first screens.
8. **General paths with strokes** if msx content or vector widgets are to render on the CPU backend.
9. **Threading:** Vello CPU has an optional multi-threaded mode through `rayon` [src]; whether the
   phone or the browser should use threads was not looked at [gap].
10. **An armv7 SIMD path,** which is the nightly backend of Decision 15.

## Questions for the maintainer

1. Is the first CPU backend a small renderer for the paint list (closed-form shapes, a glyph
   rasterizer), or a general path renderer from the start?
2. Should the paint list be the one input of the CPU backend, with `mid-ui`'s `DrawCmd` converted as
   the editor does today (Decision 14, answer 3)?
3. Which pixel format: premultiplied `RGBA8`? And how should `Color32` and its color spaces map to it?
4. Should the rasterizer later serve msx, replacing its `tiny-skia` use, or stay separate?
5. Is integer-exact output (the same bytes on every architecture) a requirement, for golden images?

## Sources

- tiny-skia: `README.md`, `Cargo.toml`, `src/wide/`, `src/scan/path_aa.rs`.
- Vello: `vello_cpu/src/lib.rs`, `vello_cpu/Cargo.toml`, `README.md`, `ARCHITECTURE.md`, `glifo/README.md`.
- zeno: `README.md`, `src/raster.rs`. fontdue: `src/raster.rs`, `src/platform/mod.rs`.
- font-rs: `README.md`. `ab_glyph_rasterizer`: `README.md`, `Cargo.toml`, `src/`.
- mid-engine: `docs/mid-editor.md`, `docs/mid-ui-wasm.md`, `docs/roadmap.md`,
  `docs/research-ui-text.md`, `docs/research-ui-hinting.md`.
- msx: the manifest of `render/msx-render-cpu`.
