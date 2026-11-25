# Text stack comparison: Parley, HarfRust, Skrifa, cosmic-text, swash, fontdue and the egui text code

## Overview

Third research document, after `docs/research-ecs-comparison.md` and `docs/research-ui-comparison.md`.
Decision 14 (answer 4) set the method for text: read the candidates' repositories, take what is
needed, and write an own stack. This document is that reading. It sizes every stage of text
handling by the Rust source of the libraries that do it, so that the cost of each level of language
support is visible before anything is written.

This is reading, not measuring. Nothing was built, run or benchmarked here, and no number of speed
or memory is quoted. Counts are lines of Rust source, tests excluded where the directory layout
allowed it; they include generated tables in some crates, so they size things, they do not rank
quality. The scope of scripts that `mid-ui` must support is not decided (question 1), and it moves
the cost by an order of magnitude.

A side finding on `mid-math` for 32-bit ARM, asked during this work, is at the end.

## Method and evidence levels

| Source | Commit | What was read |
|---|---|---|
| `linebender/parley` (0.13.0, rust-version 1.96) | `f58f168`, 2026-11-24 | `README.md`; the workspace manifest; `parley_engine/src`, `parley/src/{layout,editing}`; `fontique/src/backend/android.rs`; the CPU render example |
| `harfbuzz/harfrust` (0.16.1, rust-version 1.85) | `9849d82`, 2026-11-23 | `README.md`; the module layout and sizes of `harfrust/src` |
| `googlefonts/fontations` (rust-version 1.85) | `4f11548`, 2026-11-23 | `skrifa/README.md`, `skrifa/src` sizes and manifest; sizes of `read-fonts`, `font-types` |
| `pop-os/cosmic-text` (0.19.0, rust-version 1.89) | `b07cb90`, 2026-10-27 | `README.md`; the manifest; `src` sizes; a search of `src/edit` for composition |
| `swash` (0.2.10) and `zeno` (0.3.3, rust-version 1.63) | `db9d673`, 2026-11-13; `334a74a`, 2026-10-27 | `README.md` of both; module sizes; manifests |
| `mooman219/fontdue` (0.9.5) | `1a84e6d`, 2026-10-18 | `README.md`; `raster.rs`, `platform/mod.rs`, `table/`, `unicode/` |
| `harfbuzz/ttf-parser` (0.25.1, rust-version 1.88) | `0c72912`, 2026-08-06 | `README.md`, source size |
| `unicode-rs/unicode-segmentation` (1.13.3), `servo/unicode-bidi` (0.3.18), `axelf4/unicode-linebreak` (0.1.5) | `2368a49`, 2026-11-10; `9a8c7b0`, 2026-09-15; `829adee`, 2023-07-25 | manifests, source sizes |
| `unicode-org/icu4x` (sparse: `components/segmenter`, `components/properties`) | `d998db9`, 2026-11-24 | the segmenter `README.md`, features, source size |
| `emilk/egui` (0.37.0) | `ea6b094`, 2026-11-23 | `crates/epaint/src/text/` (sizes, `font_face.rs`), the `epaint` manifest |
| `Mid-D-Man/msx` | `490b204`, 2026-11-24 | `primitives/msx-text/src/lib.rs` size |
| `Mid-D-Man/mid-engine` | `aa80d48`, 2026-11-24 | `docs/roadmap.md` (Decisions 11 and 14), `docs/mid-editor.md` (text), `crates/mid-math/src/lib.rs`, `crates/mid-math/src/f32/neon/` |
| Rust 1.97.1 sources (Ubuntu `rust-1.97-src`) | n/a | `library/stdarch` NEON attributes, `compiler/rustc_target/src/target_features.rs`, target specs through `rustc --print target-spec-json` |

Every claim carries one tag:

- **[src]** read in the source or manifest of that project.
- **[doc]** stated in that project's own documentation, not checked against its code.
- **[gap]** searched for and not found, with the search named. A gap means "not found in what was
  read", not "does not exist".

Versions move together and apart. `harfrust` is 0.12 in `cosmic-text`, 0.15 in Parley's workspace
and 0.16.1 on its own main; `skrifa` is accepted as `>= 0.40, <= 0.44` by `cosmic-text`,
`>= 0.31.1, <= 0.50` by swash and pinned at 0.49 by Parley [src]. All were read at main, not at
the crates.io release a project would take. Not read: `fontdb`, `roxmltree`, `kurbo`, `tiny-skia`,
`vello_cpu` internals, egui's `TextEdit`, and any of these libraries on a 32-bit ARM device.

## Summary

| Library | Role | Rust source lines | Rust floor | `no_std` | License | Notes |
|---|---|---|---|---|---|---|
| Fontique (Parley repo) | Font enumeration, matching, fallback | 6,217 | 1.96 | `std` feature [src] | Apache-2.0 OR MIT | Backends for Android, Core Text, DirectWrite, fontconfig [src] |
| Parley | Layout, line breaking, bidi resolution, selection, editing | 11,894 (+ 3,370 in `parley_engine`) | 1.96 | `std` feature [src] | Apache-2.0 OR MIT | Four key dependencies: Fontique, HarfRust, Skrifa, ICU4X [doc] |
| HarfRust | Shaping, a port of HarfBuzz | 47,375 | 1.85 | `std` feature, needs `alloc` [src] | MIT | Matches HarfBuzz 15.2.0 [doc] |
| `read-fonts` + `skrifa` | Parsing, metrics, outlines, hinting, variations, color | 88,781 + 41,467 | 1.85 | `std` feature [src] | MIT OR Apache-2.0 | Chrome's font backend on Android and Linux [doc] |
| cosmic-text | Shaping (HarfRust), layout, fallback, editor, raster (swash) | 12,523 | 1.89 | `no_std` feature [src] | MIT OR Apache-2.0 | Other platforms than Linux, macOS, Windows may need their own fallback [doc] |
| swash | Introspection, shaping, scaling, rendering | 26,528 | none declared | `std` feature [src] | Apache-2.0 OR MIT | Text layout and glyph caching are non-goals [doc] |
| zeno | Path rasterizer for alpha and subpixel masks, strokes, dashes | 6,027 | 1.63 | attribute present [src] | Apache-2.0 OR MIT | swash's rasterizer [src] |
| fontdue | Rasterizer, naive layout, kern table | 5,060 | none declared | yes [doc] | MIT OR Apache-2.0 OR Zlib | No shaping, and the author does not expect to add it [doc] |
| `ttf-parser` | Parsing, zero allocation, zero dependencies | 23,792 | 1.88 | yes [doc] | MIT OR Apache-2.0 | In maintenance mode, bug fixes only [doc] |
| `unicode-linebreak` | UAX #14 line breaking | 294 + tables | 1.56 | not found [gap] | Apache-2.0 | Last commit 2023-07-25 [src] |
| `unicode-segmentation` | UAX #29 clusters, words, sentences | 6,432 | 1.85 | not found [gap] | MIT OR Apache-2.0 | |
| `unicode-bidi` | UAX #9 | 5,978 | 1.47 | `std` feature, `hardcoded-data` [src] | MIT OR Apache-2.0 | |
| `icu_segmenter` | UAX #14 and #29 with CSS tailoring, LSTM models for complex scripts (feature `lstm`) | 10,493 | not read | data in a separate crate [src] | not read | |
| egui text (epaint) | Own layout, galley cache, glyph atlas | 8,987 | 1.95 | not applicable | MIT OR Apache-2.0 | Shapes with HarfRust, outlines from Skrifa, glyphs rasterized on the CPU with `vello_cpu` [src] |
| `msx-text` | Text as path outlines, no shaping | 712 | none | not read | not read | Uses `ttf-parser` [src] |

## Findings by area

### The ecosystem has one center of gravity

Parley (and through it Slint and Blitz, per `docs/research-ui-comparison.md`), cosmic-text (and
through it Iced) and egui now all shape with HarfRust [src]. Outlines and metrics come from Skrifa
in Parley and egui, and from Skrifa through swash in cosmic-text's raster path [src]. `ttf-parser`,
which HarfRust was forked away from, is in maintenance mode and points new projects at fontations
[doc]; fontdue and `msx-text` still use it [src]. The practical reading: the parts that are most
used, tested against FreeType glyph by glyph (Skrifa, [doc]) and still changing are fontations and
HarfRust, and the version ranges above show the API moving between releases.

### What each stage costs

| Stage | In mid today | Shown by (lines) |
|---|---|---|
| Font data, matching, fallback | One bundled font, DejaVu Sans, ASCII only [doc] | Fontique 6,217, of which the Android backend is 218; cosmic-text's own fallback with static lists taken from Chromium and Firefox [doc] |
| Parsing, metrics, outlines | The host does it [doc] | `read-fonts` 88,781, `skrifa` 41,467 (outline 30,353 of them); `ttf-parser` 23,792; fontdue's own `kern` reader 200 over `ttf-parser` |
| Shaping | None | HarfRust 47,375; swash `shape` 4,721 |
| Segmentation, bidi, line breaking | None; `fit` shortens with `...` [doc] | `unicode-linebreak` 294 + tables; `unicode-segmentation` 6,432; `unicode-bidi` 5,978; `icu_segmenter` 10,493; `parley_engine` 3,370 |
| Layout | The host measures through a callback [doc] | Parley `layout` and `editing` 8,067 (`line_break.rs` alone 2,143); egui `text_layout.rs` 2,768; fontdue 567 |
| Editing, caret, composition | Not built (E1) [doc] | Parley `editing` (editor 1,192, selection 656, cursor 420) |
| Rasterizing glyphs | The host [doc] | zeno 6,027; fontdue `raster.rs` 125; `vello_cpu` in egui; `tiny-skia` in Parley's example |
| Caching and atlas | The host | egui `glyph_atlas.rs` 409 and `galley_cache.rs` 478; cosmic-text `glyph_cache.rs` 170 |

### Shaping is where the languages cost

HarfRust's 47,375 lines split as: `ot` 23,061 (lookup application 7,917, the complex-script shapers
9,936, layout 1,028, apply 1,374, map 972), `aat` 7,330 (Apple tables), own Unicode tables 3,921,
buffer 3,165, tag table 2,618 [src]. The complex shapers are Arabic, Hebrew, Hangul, Indic (alone
1,937 lines), Khmer, Myanmar, Thai and the Universal Shaping Engine, several of them generated
state machines [src]. fontdue shows the other end: kerning from the `kern` table in 200 lines,
a GSUB loader of 74 lines, and no shaping engine [src]. Between the two sits swash's
own shaper at 4,721 lines [src, scope not read].

So the work divides cleanly by what must be supported:

| Tier | Covers | What the reading shows as the size |
|---|---|---|
| T0 | ASCII with a bundled font (today) | Host fonts, `fit` in the editor [doc] |
| T1 | Latin with kerning, advance widths, a glyph cache | fontdue does rasterizing, layout and kerning over a parser in 5,060 lines [src] |
| T2 | Latin with ligatures, mark positioning, proper clusters and line breaks | GSUB and GPOS application about 11,300 lines in HarfRust (lookups, apply, layout, map); `unicode-segmentation` 6,432 [src] |
| T3 | Right-to-left scripts (bidi, joining) | `unicode-bidi` 5,978; HarfRust's Arabic and Hebrew shapers, size not separated [src] |
| T4 | Indic, Southeast Asian, CJK segmentation | `indic.rs` 1,937; `icu_segmenter` 10,493 plus a separate data crate [src] |

### Rasterizing is shared with the CPU backend

A glyph is a path filled with coverage. The CPU backend that Decision 14 puts first has to fill
anti-aliased rounded rectangles, thick lines and clips for the paint list anyway. The candidates
show three ways to get coverage: zeno, a general path rasterizer with strokes and dashes in 6,027
lines [src]; fontdue's 125-line accumulation buffer on `f32`, whose own header says not to reuse it
because of its unsafe invariants [src]; and `vello_cpu`, which egui calls once per glyph to fill an
outline into a bitmap before putting it in an atlas [src]. Parley's example does the same with
`tiny-skia` and notes that outlines can be cached [src]. The reading is one rasterizer for shapes
and glyphs, with a glyph cache above it.

SIMD in fontdue is selected only for x86 and x86_64; every other architecture, ARM included, gets
its portable fallback [src, `platform/mod.rs`]. zeno's use of SIMD was not checked [gap].

### Hinting

Skrifa carries a TrueType interpreter and an automatic hinter, 10,805 lines for the automatic one
alone, and egui exposes hinting targets (`Smooth`, `Mono`) that mirror Skrifa's [src]. Whether hinting
matters at the phone's pixel density was not measured; nothing in the reading says (question 3).

### Font discovery and fallback on Android

Fontique has an Android backend of 218 lines that parses the system font configuration with
`roxmltree`. A comment in it says the generic families should come from the parsed file once that is done, and its
defaults name Roboto Flex, Roboto and Noto Sans, with Noto Color Emoji for emoji [src]. cosmic-text
states that platforms other than Linux, macOS and Windows may need their own fallback [doc]; whether
its `fontdb` finds Android system fonts was not read [gap]. Skrifa ships as Chrome's font backend on
Android [doc]. None of this was run on a device. The editor today avoids the problem with one bundled
font [doc].

### Editing and the soft keyboard

Parley's editor keeps an IME composition range (the preedit text) and lets the IME hide the cursor
[src, `editing/editor.rs`]. cosmic-text's `src/edit` has no preedit, IME or composition term [gap,
searched for `preedit`, `ime_` and `compose`]. This connects to Decision 14 (answer 6): an in-UI
keypad needs no composition model; an OS keyboard through a helper does, and Parley's editor is the
one candidate that has the shape of it.

### Layout and caching styles

Two styles appear. Parley and cosmic-text keep a layout object of lines, runs and clusters, with
cursor and selection logic over it [src]. egui
caches finished layouts ("galleys") by content because its UI is immediate [src, `galley_cache.rs`].
Decision 14 (answer 9) asks for both styles in `mid-ui`: a retained tree with an immediate authoring
layer, so a layout cache keyed by text, font, size and width serves both.

## What to take from each

Proposals, not decisions. Reading is the rule from Decision 14; nothing here is vendored.

| From | Idea |
|---|---|
| Parley | The split between itemization, bidi, shaping, line breaking and cursor logic as separate modules (`parley_engine` against `parley`); the composition range in the editor; the Android family defaults |
| HarfRust | The boundary: no font size, shaping always in font units and scaled by the caller; no platform or ICU integration inside the shaper |
| Skrifa | Outlines drawn into any path sink, so one outline reader can feed a rasterizer or a GPU tessellator; hinting as an option, not a requirement |
| fontdue | The smallest end-to-end example (parse, kern, layout, rasterize) as the T1 scale reference |
| zeno | A general coverage rasterizer with masks as alpha or subpixel RGBA, strokes and dashes |
| egui | Glyphs filled once into bitmaps by a CPU rasterizer, packed in an atlas, with a layout cache keyed by content |
| cosmic-text | A glyph cache and a shape-run cache as separate files (`glyph_cache.rs` 170 lines, `shape_run_cache.rs`); layout over spans with different styles |
| ICU4X and the `unicode-*` crates | Generated tables from the Unicode data files (`gen-tables` in `unicode-linebreak`) rather than hand-written ones |

## Gaps, ranked

1. **The script scope.** T1 to T4 differ by about two orders of magnitude (question 1).
2. **A coverage rasterizer shared by shapes and glyphs** in the CPU backend.
3. **A font reader and outline extractor** for the tables the editor font needs (`cmap`, `head`,
   `hhea`, `hmtx`, `maxp`, `loca`, `glyf`, `kern`), with an oracle for tests (question 2).
4. **Measuring and layout in the core** behind the `TextMeasure` seam of Decision 11: advances,
   wrapping, ellipsis, caret positions.
5. **A glyph cache,** coverage masks for the CPU backend and an atlas for the GPU backend.
6. **Line breaking and grapheme clusters,** UAX #14 and #29, from generated tables.
7. **An editing model:** caret, selection, composition range, tied to the soft keyboard decision.
8. **Font sources on Android:** bundled only, or system fonts with fallback (question 4).
9. **Hinting:** wanted or not (question 3).
10. **Color glyphs and emoji** (`COLR`, `CBDT`, `sbix` in Skrifa's table), later.
11. **A shaper,** if T2 or beyond is chosen: an own port of the needed lookups, with HarfRust as the
    reference for output.

## A probe, designed and not built

The Latin glyph probe answers whether T1 is as small as fontdue suggests and how it runs on the phone.

- **Scope.** Parse the bundled DejaVu: `cmap` (formats 4 and 12), `head`, `hhea`, `hmtx`, `maxp`, `loca`,
  `glyf` with composites, `kern`. Fill coverage with an own scanline rasterizer. Keep a mask cache.
- **Oracles.** Outlines and advances against Skrifa; coverage against fontdue and zeno within a
  tolerance. The oracle code lives outside the core crates (question 2).
- **What to record.** Glyphs per second at 12, 16 and 24 px, cache hit and miss cost, memory per
  cached glyph, and source lines. On x86_64 CI and on the phone through the editor's Console. The CPU
  line goes in the header, as in the other probes.
- **How to read it.** The phone is the authority. A sandbox x86_64 number says nothing about a
  32-bit phone.

## Questions for the maintainer

1. Which scripts must `mid-ui` render, and by when? Latin only for the editor, or Arabic, Hebrew and
   CJK soon? Since `.mxui` is meant for anything that uses `mid-ui` (Decision 14), is the answer
   "Latin now, a path to the rest"?
2. May tests use a dependency as an oracle (Skrifa, `ttf-parser`, HarfRust) while the core crates
   stay at zero dependencies, or must tests be dependency-free too?
3. Is hinting wanted, or are unhinted outlines enough at the phone's density?
4. On Android: bundled fonts only, or system fonts (Roboto, Noto) with a fallback chain?
5. Where does the text code live: a crate of its own, or inside `mid-ui`? Crate names are open in
   Decision 14.
6. Should the CPU backend's shape rasterizer and the glyph rasterizer be one piece of code?
7. Should the editing model (caret, selection, composition) be in the text crate or in the UI layer?

## Side finding: `mid-math` on 32-bit ARM

Asked on 2026-11-24 whether platform optimization for Android is possible in `mid-math`. It is, with
three caveats read below. The CPU backend and any glyph or shape math on the phone depend on it.

- **Targets.** `armv7-linux-androideabi` has the features `+v7,+thumb-mode,+thumb2,+vfp3d16,-neon`.
  `thumbv7neon-linux-androideabi` has `+v7,+thumb-mode,+thumb2,+vfp3,+neon`. `aarch64-linux-android`
  has `+v8a,+neon` [src, `rustc --print target-spec-json` on 1.97.1].
- **Intrinsics.** In the 1.97.1 sources the NEON intrinsics checked (`vaddq_f32`, `vfmaq_f32`) are stable on every
  architecture but 32-bit ARM, where they carry `unstable(feature = "stdarch_arm_neon_intrinsics", issue = "111800")`.
  The `neon` target feature on 32-bit ARM is unstable behind `arm_target_feature` [src,
  `library/stdarch`, `compiler/rustc_target/src/target_features.rs`].
- **What `mid-math` uses.** Its NEON backend is selected for `aarch64` only and imports
  `core::arch::aarch64` [src, `lib.rs`, `f32/neon/*.rs`]. It calls `vdivq_f32`, `vsqrtq_f32`,
  `vaddvq_f32`, `vaddvq_f64` and `vuzp1q_f32`, which exist only in the `aarch64` module of the
  standard library's intrinsics; `vfmaq_f32`, `vrsqrteq_f32` and `vrsqrtsq_f32` exist in the module
  shared with 32-bit ARM [src]. A port is therefore a second backend, not a rename.
- **Dispatch.** Order is SSE2, NEON, WASM, `coresimd`, scalar. `coresimd` is the nightly
  `portable_simd` backend, chosen by a Cargo feature [src, `lib.rs`].

Ways to get SIMD on the phone, none tried here:

| Way | Toolchain | Cost |
|---|---|---|
| Stay scalar | stable | None; the status quo |
| Build for `thumbv7neon-linux-androideabi` so LLVM may vectorize scalar code | stable | A different target triple in the Android workflow; the gain is unmeasured |
| `coresimd` feature with that target | nightly | No hand-written intrinsics; nightly only for that build |
| An armv7 backend on `core::arch::arm` | nightly, two feature gates | A second backend with `vpadd` where `mid-math` uses `vaddv`, and estimates plus Newton steps where it uses `vdiv` and `vsqrt` |

Not checked: whether every device with a 32-bit userland has NEON, and how ARMv7 NEON treats
denormals and rounding compared with the scalar path, which matters for results that must match
(open, to settle with the first backend).

## Sources

- Parley: `README.md`, `Cargo.toml`, `parley_engine/src/`, `parley/src/layout/`, `parley/src/editing/`,
  `fontique/src/backend/android.rs`, `examples/tiny_skia_render/src/main.rs`.
- HarfRust: `README.md`, `harfrust/src/` (module sizes).
- fontations: `skrifa/README.md`, `skrifa/Cargo.toml`, `skrifa/src/`, `read-fonts/src/`, `font-types/src/`.
- cosmic-text: `README.md`, `Cargo.toml`, `src/`, `src/edit/`.
- swash: `README.md`, `Cargo.toml`, `src/`. zeno: `README.md`, `Cargo.toml`, `src/`.
- fontdue: `README.md`, `src/raster.rs`, `src/platform/mod.rs`, `src/table/`, `src/unicode/`.
- `ttf-parser`: `README.md`. `unicode-linebreak`, `unicode-segmentation`, `unicode-bidi`: manifests and `src/`.
- ICU4X: `components/segmenter/README.md`, `components/segmenter/Cargo.toml`.
- egui: `crates/epaint/src/text/`, `crates/epaint/Cargo.toml`. msx: `primitives/msx-text/src/lib.rs`.
- mid-engine: `docs/roadmap.md`, `docs/mid-editor.md`, `crates/mid-math/src/lib.rs`, `crates/mid-math/src/f32/neon/`.
- Rust 1.97.1: `library/stdarch/crates/core_arch/src/`, `compiler/rustc_target/src/target_features.rs`.
