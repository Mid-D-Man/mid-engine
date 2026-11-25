# Hinting scoping: Skrifa's engines, the bundled font and the sizes the editor draws at

## Overview

Fourth research document of the UI series. Decision 15 records that hinting is wanted. This document
sizes the work from the Rust source of Skrifa, describes what the bundled font offers, lists the pixel
sizes the editor draws at, and adds one measurement: how far hinting changes rendered text at those
sizes, using FreeType, a different library, on the sandbox machine. The measurement says where hinting
pays and in what order the work could go. It does not argue against the decision.

Hinting fits a glyph outline to the pixel grid at one size, so that stems and heights land on whole
pixels. It changes the shape of each glyph and, in some modes, its advance width, so it touches layout
and caching as well as drawing.

Evidence tags: **[src]** read in source or a manifest, **[doc]** stated in a project's documentation,
**[gap]** searched for and not found, and **[run]** a measurement made in this work, with its limits
stated beside it. Nothing about speed was measured.

## Method

| Source | Commit or version | What was read |
|---|---|---|
| `googlefonts/fontations` | `4f11548`, 2026-11-23 | `skrifa/src/outline/hint.rs`; sizes of `glyf/hint`, `autohint`, `cff` |
| `linebender/vello`, crate `glifo` 0.5.0 | `a8d6e27`, 2026-11-25 | `glifo/README.md`; the hinting mentions in `glyph.rs` |
| `Mid-D-Man/mid-engine` | `a9c1810`, 2026-11-25 | `apps/editor-android/assets/fonts/DejaVuSans.ttf`; `apps/editor-android/main.c` (font loading); `crates/mid-editor/src/editor.rs` (text sizes and the pixel unit) |
| FreeType 2.13.2 through `freetype-py`, `fontTools` 4.67.0 | the sandbox, x86_64 | the experiment below; the table dump of the font |

## Findings

### Skrifa has two hinting engines and a default rule

| Engine | What it is | Size in Rust source |
|---|---|---|
| Interpreter, TrueType | Runs the font's own instructions (`fpgm`, `prep`, `cvt`, per-glyph programs) | `glyf/hint` 9,985 lines, of which the opcode engine is 6,166 [src] |
| Interpreter, PostScript | Applies the hints in CFF and CFF2 outlines | inside `cff`, 2,508 lines with the outline code [src] |
| Automatic hinter | Computes hints from the outline shapes, per script | `autohint` 10,805 lines: the top level 3,906, `topo` 2,468, `hint` 2,139, `metrics` 2,292 [src] |

The default engine choice copies FreeType's rule: PostScript fonts use the PostScript interpreter; a
TrueType font uses the interpreter if its `fpgm` or `prep` table is non-empty, and the automatic
hinter otherwise [src, `outline/hint.rs`]. The target modes mirror FreeType's: `Smooth` with `Normal`,
`Light` (a lighter touch, less horizontal adjustment) and `Lcd`, plus `Mono` [src]. Crate `glifo`, which
caches and draws glyphs for Vello, states as a goal that the hinting instance and the hinted advance
are shared between the shaper and the renderer [doc], and keeps a cache of hinting instances [src].

### The bundled font

`DejaVuSans.ttf` (759,720 bytes) has 6,253 glyphs, of which 2,607 are composite and 1,130 carry their
own instructions. It has an `fpgm` table of 171 bytes, a `prep` table of 1,384 bytes and 255 `cvt`
entries. Its `gasp` table asks for grid-fitting above 8 pixels per em. It also has `GPOS`, `GSUB`,
`kern` and `GDEF`. Under the default rule above it would go to the TrueType interpreter [src, table dump
and rule]. That Skrifa's interpreter output matches FreeType's for this font was not checked here [gap].

### The sizes the editor draws at

The editor draws text at 13, 15 and 17 UI units, and one more size from a variable that was not
traced [src, `editor.rs`]. A unit is the short side of the screen divided by 411, so about 2.63 pixels
on the Galaxy A13 4G [src, doc]. The sizes are therefore:

| Unit | Pixels on the A13 (2.63) | Pixels at one pixel per unit |
|---|---|---|
| 13 | about 34 | 13 |
| 15 | about 40 | 15 |
| 17 | about 45 | 17 |

The host loads the font once at 64 pixels with raylib and scales that bitmap at draw time with a
bilinear filter [src, `main.c`]. Nothing today is fitted to the size actually drawn.

### A measurement: how much hinting changes the text at those sizes [run]

Setup: FreeType 2.13.2 through `freetype-py`, `DejaVuSans.ttf`, the printable ASCII characters 33 to
126, at 13, 17, 34, 40 and 45 pixels per em. Four renderings per size: unhinted; FreeType's light
automatic mode (`FT_LOAD_TARGET_LIGHT`); its default mode, which here runs the font's TrueType
instructions; and the forced automatic hinter in normal mode. The difference per glyph is the sum of
absolute coverage differences against the unhinted rendering, divided by the ink of the unhinted
glyph, with both placed at the same origin. Line width is the sum of advances of one sample string of
64 characters, hinted advances being whole pixels and unhinted ones linear. The last columns are the
top row of the letters `x` and `H` above the baseline, unhinted to hinted.

| Pixels per em | Light: mean, p90 (%) | Instructions: mean, p90 (%) | Forced auto: mean, p90 (%) | Line width change, light, instructions, forced auto (%) | `x` top | `H` top |
|---|---|---|---|---|---|---|
| 13 | 21.0, 36.8 | 26.5, 44.1 | 43.6, 68.6 | -0.82, -0.82, -2.23 | 8 to 7 in all | 10 to 9 in all |
| 17 | 26.1, 40.7 | 19.3, 34.8 | 44.9, 72.6 | -0.30, -0.30, +4.36 | 10 to 10, 9, 10 | 13 to 13, 12, 13 |
| 34 | 6.8, 14.3 | 10.4, 21.1 | 15.4, 24.1 | +0.68, +0.68, +1.58 | 19 in all | 25 in all |
| 40 | 11.1, 20.1 | 9.1, 18.2 | 19.2, 31.2 | -0.18, -0.18, +1.20 | 22 in all | 30 to 30, 29, 30 |
| 45 | 6.1, 12.0 | 11.1, 23.3 | 15.0, 25.7 | +0.93, +0.93, -0.29 | 25 to 24, 25, 24 | 33 in all |

What it shows, with its limits:

- At the A13's sizes (34 to 45) the per-glyph change is 6 to 19 percent of the ink, the line width
  moves by less than 1.6 percent, and a snapped height moves by at most one pixel. In the contact sheet
  made from the same renderings, the four rows look alike at 34 and 45 pixels.
- At 13 and 17 pixels, which is the editor at one pixel per unit (a desktop window, or a phone at low
  density), the change is 19 to 45 percent of the ink, the line width moves by up to 4.4 percent, and
  the `x` height snaps by a pixel at 13. The forced automatic hinter widens the sample line visibly at
  17 pixels.
- The metric counts sub-pixel shifts as well as shape change, so it overstates visible change. It was
  looked at on one sheet and one font; legibility was not judged beyond that.
- This is FreeType, not Skrifa and not an own implementation. The hardware does not matter for
  geometry, but nothing here says how fast any hinting is.

So the benefit of hinting is mostly at small sizes, and at the phone's sizes it is small for this
font. That is a reading of one experiment, offered for ordering the work.

### Hinting reaches into layout and caches

- Hinted advances are whole pixels in FreeType's modes, so the width a text measures depends on the
  hinting mode. The `TextMeasure` seam of Decision 11 has to return the same advances the renderer
  draws with, which is why `glifo` shares the hinted advance between shaper and renderer [doc].
- A glyph cache key needs the size and the hinting mode, and a subpixel offset if glyphs are
  positioned between pixels. Hinted glyphs are normally placed on whole pixels. This is reasoning from
  the cache designs read in `docs/research-ui-text.md`, not a source finding.
- The font decides the engine. An instruction-bearing font such as DejaVu needs the interpreter; a font
  without instructions needs the automatic hinter; CFF fonts need the PostScript hints [src, rule].
  Supporting system fonts (Decision 15) may therefore mean all three eventually. Which engines the
  Android system fonts need was not checked [gap].

## Options

| Option | What it is | Size shown by the reading | Notes |
|---|---|---|---|
| H0 | No hinting; outlines scaled to the size, optional fractional positions | 0 | The experiment above shows small change at 34 to 45 pixels for this font |
| H1 | A light, vertical-only automatic hinter for Latin: snap baseline, `x` height, cap height | Skrifa's whole automatic hinter is 10,805 lines for all scripts; the Latin share was not separated [gap] | The idea is FreeType's light target [doc]; widths stay near unhinted |
| H2 | A TrueType bytecode interpreter | 9,985 lines in Skrifa [src] | Needs the font's own programs; exact behavior is large and font-specific |
| H3 | PostScript hints for CFF | part of 2,508 lines [src] | Needed for CFF system fonts |

## Gaps, ranked

1. **An unhinted pipeline first:** outlines, coverage, cache, advances. Every hinting option sits on it.
2. **Where hinted and unhinted widths differ,** the `TextMeasure` contract must say which one is used.
3. **H1 for Latin,** if the maintainer's reason for hinting is the small sizes of a desktop or
   low-density window.
4. **H2,** if exact fidelity to a font's own instructions is wanted.
5. **H3,** when CFF fonts come in with system fonts.
6. **Per-script automatic hinting** for non-Latin scripts, with the script scope of Decision 16.

## A probe addition, designed and not built

Add a hinted run to the Latin glyph probe of `docs/research-ui-text.md`: the same glyphs and sizes as
the table above, own output against Skrifa (a dev-dependency oracle, Decision 15) and, in the sandbox
only, against FreeType, reporting the same two metrics so the numbers are comparable.

## Questions for the maintainer

1. Which sizes come first: the phone's (34 to 45 pixels) or the one-pixel-per-unit sizes of a desktop
   window (13 to 17)? The measurement above says the benefit is in the second.
2. Is the first hinting step the light vertical-only kind (H1), with the interpreter later, or the
   interpreter first because the bundled font carries instructions?
3. Must hinted output match FreeType or Skrifa pixel for pixel, or closely?
4. Is LCD subpixel rendering wanted? Skrifa has a target for it [src]; nothing in the editor asks for
   it.

## Sources

- fontations: `skrifa/src/outline/hint.rs`, `skrifa/src/outline/glyf/hint/`, `skrifa/src/outline/autohint/`,
  `skrifa/src/outline/cff/`.
- Vello: `glifo/README.md`, `glifo/src/glyph.rs`.
- mid-engine: `apps/editor-android/assets/fonts/DejaVuSans.ttf`, `apps/editor-android/main.c`,
  `crates/mid-editor/src/editor.rs`, `docs/research-ui-text.md`, `docs/roadmap.md`.
- The experiment: `freetype-py` with FreeType 2.13.2, `fontTools` 4.67.0, run in the sandbox.
