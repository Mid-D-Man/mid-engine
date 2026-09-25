# Site Documentation Guidelines

## Purpose

`DOCUMENTATION_AND_COMMENTING_GUIDELINES.md` covers the internal,
AI-facing docs: `docs/<CRATE_NAME>.md`, decision logs, fix history, the
stuff future work on the crate itself needs. `web/site/` (the mdBook under
`.github/workflows/deploy-site.yml`, Cloudflare Pages) is a different
audience entirely — someone deciding whether a crate does what they need,
or how to actually call it, not someone about to edit its source. This doc
covers that second surface: what a crate's page on the site should contain
and how it's structured, so it doesn't have to be reinvented per crate.

The two are related but not the same file re-purposed: a site page is a
short, derived summary written for an external reader, never a copy of
`docs/<CRATE_NAME>.md`'s decision log or fix history. Link back to the
internal doc for "why it looks the way it does" (see section 2); don't
restate it here.

## 1. Structure — one overview page per crate, one page per module

Every crate gets an overview page at `web/site/src/crates/<crate>.md` —
what exists today already (`mid-math.md`, `mid-platform.md`, and so on),
kept short: what the crate is for, in a few lines, plus a list of its
modules linking into the pages below. It no longer needs to carry the full
module-by-module detail itself once a crate has been moved onto this
structure — that detail lives one level down.

Each of the crate's public modules gets its own page at
`web/site/src/crates/<crate>/<module>.md`. "Module," here, tracks the same
file/module the source and `docs/<CRATE_NAME>.md`'s own "Modules" section
already use — one site page per `### <file>.rs` section that doc has, not
a different grouping invented for the site.

`SUMMARY.md` nests the module pages under the crate's own entry so they
render as a foldable section in the sidebar — `book.toml` already has
`[output.html.fold] enable = true, level = 1` turned on for exactly this,
it just isn't used yet since every crate today is still a single flat
page. Nesting looks like:

```markdown
- [mid-platform](crates/mid-platform.md)
  - [cell](crates/mid-platform/cell.md)
  - [sync::atomic](crates/mid-platform/atomic.md)
  - [sync::Mutex](crates/mid-platform/mutex.md)
  - [sync::RwLock](crates/mid-platform/rwlock.md)
  - [sync::Once / OnceLock](crates/mid-platform/once.md)
  - [sync::LazyLock](crates/mid-platform/lazy-lock.md)
  - [sync::Barrier](crates/mid-platform/barrier.md)
  - [sync::Arc / Weak](crates/mid-platform/arc.md)
```

`level = 1` folds anything nested one level or deeper by default, so a
crate with modules under it shows up collapsed in the sidebar until
someone actually opens it — matching what "a dropdown per crate" means in
mdBook terms. A crate with no sub-pages yet (not retrofitted — see section
5) just keeps its current flat `- [crate-name](crates/crate-name.md)` line,
no empty dropdown for nothing.

## 2. What goes on a module page

Every module page carries the same shape:

```markdown
# sync::RwLock

One line: what this type/function/module is for.

## What it does

A short paragraph — the real behavior, not a restatement of the heading.
Name the actual types/functions this page covers.

## Example usage

\`\`\`rust
// A short, realistic call site. A few lines, not a tutorial.
\`\`\`

## Status

Done / in progress / planned, plus a link to the full engineering
write-up: see `docs/mid-platform.md`, "sync/rwlock.rs" for the design
decisions and trade-offs.
```

Rules for what goes in each part:

- **What it does** stays behavior-level: what the type is, what it
  guarantees, what it costs. Not a decision log — no "we chose X over Y
  because," no fix history, no benchmark table. That content already has a
  home (`docs/<CRATE_NAME>.md`'s own "Modules" section, "Decisions" and
  "Benchmarks" — see `DOCUMENTATION_AND_COMMENTING_GUIDELINES.md` §4); the
  site page links to it, once, in **Status**, rather than duplicating it.
  A reader who wants the "why" clicks through; a reader who just wants to
  know what's available and how to call it never has to wade through it.
- **Example usage** is real, working-looking code, not pseudocode — even
  though nothing in this project's own toolchain can compile-check it any
  more than the crate's actual source can (`docs/mid-platform.md`'s own
  "sandbox_constraint" note applies here too: write it carefully against
  the real API, don't guess at a shape that looks plausible). One example
  covering the type's main use case is enough; add a second only if the
  module genuinely has two distinct common uses (a lock's `try_`-prefixed
  non-blocking variant is not automatically a second example — only include
  it if it's actually a different enough use case to be worth showing, not
  just because it exists).
- **Status** is one line plus the link — not a changelog. "Done" doesn't
  need elaboration beyond the link; if something real and user-visible is
  still missing (an FFI boundary, say), say so in one line here too, same
  as `docs/<CRATE_NAME>.md`'s own "Fixes and Problems" would, but shorter.

Writing style matches `DOCUMENTATION_AND_COMMENTING_GUIDELINES.md` §5
exactly — third person or first-person plural, human phrasing, no em
dashes, no "leverage"/"utilize"/"seamless," state what the thing does
plainly. `mid-math.md`'s existing overview page (bold key type names in
backticks, short paragraphs, no marketing language) is the tone to match,
not a special "site voice" invented separately.

## 3. Update discipline

Update a module's site page in the same pass that lands the module's code
— not a follow-up, not a batch pass done later. The trigger is the same
one `DOCUMENTATION_AND_COMMENTING_GUIDELINES.md` §6 already uses for
`docs/<CRATE_NAME>.md`: if you touched the file and it doesn't have a
section yet, add one, right then. Letting the internal doc and the public
site drift apart independently is the exact failure mode per-file NOTICE
headers and the incremental-update rule already exist to prevent for
source comments vs. the internal doc — the same discipline extends one
layer further out to the site now that the site has real per-module
content worth keeping in sync.

## 4. Retrofit policy

Existing crates whose site page is still a single flat overview
(`mid-math.md` and most others, as of this doc) are not rewritten in one
pass just because this rule now exists. Same trigger-based, disclosed-gap
discipline this project already applies everywhere else (`docs/roadmap.md`
Decision 3's own named-trigger approach; `docs/mid-collections.md`'s
pieces built "the moment `mid-ecs`'s real storage work started, nothing
before it"): retrofit a crate's site page the next real time that crate is
actively being worked on, not speculatively ahead of that. Named here so
it's an open item, not a silently-abandoned intention — see "Still open"
below.

`mid-platform` is the first crate actually moved onto this structure (this
pass, alongside its Phase 2 primitives — see `docs/mid-platform.md`) and is
the concrete worked example everything above was checked against, not just
an illustration.

## Still open

- Every other crate's site page (`mid-math.md` included, despite being the
  example this doc's own section 1 describes the target shape against) is
  still the old flat-page format. Retrofitting them is real future work,
  not decided to happen on any particular schedule — section 4's
  trigger-based rule governs when each one actually gets touched.
- Whether a crate's FFI surface (where one exists and is built —
  `docs/RUST_AND_CRATE_GUIDELINES.md` §2) deserves its own site module page
  alongside the Rust-facing ones, or belongs folded into the crate overview
  page instead. Not decided — no crate has reached this doc's structure
  with a built FFI module yet to force the question.
