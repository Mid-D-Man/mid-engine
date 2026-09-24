# mid-engine-site

## Overview

The public-facing site (`web/`), deployed to Cloudflare Pages, separate from
GitHub Pages — confirmed directly (`MidManStudio/ubel_stratum`'s real
`.github/workflows/pipeline-dashboard.yml`, read in full) that GitHub Pages
was already being replaced with a `cloudflare/wrangler-action@v3`-based
deploy for that project, so this repo follows the same real, working
pattern rather than reinventing one. `docs/*.md` (this file included) stays
the internal, AI-facing working-notes layer; `web/` is the separate,
user-facing site — see `web/site/src/introduction.md`'s own "Where this
book fits" section for the split stated on the site itself.

**Deploy is `workflow_dispatch` only**, per direct instruction (twice,
correcting an initial wrong guess that it might be safe to auto-deploy on
push, matching `ubel_stratum`'s own choice for its equivalent workflow —
that assumption was wrong for this project specifically, not universally).

## Structure

```text
web/
  home/index.html         — landing page (hero, nav cards, mandates, perf targets)
  site/                    — mdBook: the actual user-facing docs
    book.toml
    src/SUMMARY.md
    src/*.md               — top-level pages
    src/crates/*.md        — one page per crate
    theme/mid-book.css      — accent-color layer on mdBook's built-in "ayu" theme
  tests/index.html         — live per-crate test results (fetches JSON, see below)
  benchmarks/index.html    — live per-suite benchmark results (same pattern, no
                              producer wired up yet — see that page's own note box)
  shared/
    logo.svg                — vectorized from the uploaded logo (color-separated
                               potrace, 7 flat layers)
    theme.css                — shared variables/classes for home, tests, benchmarks
                                (NOT used by site/ — see mid-book.css's own header
                                comment for why that one carries its own copy)

scripts/
  parse_test_results.py    — turns a raw `cargo test --show-output` log into the
                              JSON web/tests/index.html fetches (see below)

.github/workflows/deploy-site.yml  — builds web/site with mdBook, assembles
                                      dist/, deploys via wrangler-action
```

`dist/` (build output, not checked in) mirrors `web/` with `site/book/`'s
mdBook output remapped to `dist/docs/`, and per-crate JSON copied into
`dist/tests/<crate>/`.

## JSON schemas

### Test results (`<crate>-test-results.json`)

```json
{
  "build": "42", "branch": "main", "commit": "abc1234",
  "rust_version": "rustc 1.90.0 (...)", "crate": "mid-ptr",
  "summary": { "total": 20, "passed": 20, "failed": 0, "ignored": 0, "duration_s": 0.12 },
  "suites": [
    { "name": "mid_ptr", "passed": 20, "failed": 0, "ignored": 0, "duration_s": 0.12,
      "tests": [ { "name": "erased::tests::...", "status": "passed",
                   "duration_ms": 0, "output": "" } ] }
  ]
}
```

Produced today by `scripts/parse_test_results.py`, used by both
`deploy-site.yml` and (as of this pass) nothing else yet — `mid-ptr-test.yml`
and `mid-platform-test.yml` still carry their own inline copy of the same
parsing logic, written before this script existed. Not refactored to call
the shared script in this pass — a small, disclosed, low-risk follow-up
(they already work correctly; this is a duplication cleanup, not a
correctness fix).

### Benchmark results (`<suite>-bench-results.json`) — proposed, not produced yet

```json
{
  "build": "42", "branch": "main", "commit": "abc1234",
  "rust_version": "rustc 1.90.0 (...)", "suite": "ecs-vs-bevy-ecs",
  "benchmarks": [
    { "name": "iter_1000_entities", "mean_ns": 1234,
      "baseline_name": "bevy_ecs", "baseline_mean_ns": 1500 }
  ]
}
```

`web/benchmarks/index.html` already fetches this shape; no `bench-vs-*.yml`
workflow emits it yet (they currently write an HTML report directly — see
`docs/benching-standards.md`). Wiring one suite's workflow to also emit this
JSON is the next real step here, deliberately not done speculatively for
all of them at once.

## `scripts/parse_test_results.py`

Pulled out as a standalone script (rather than a third inline copy inside
`deploy-site.yml`) specifically so it has exactly one implementation instead
of accumulating a third copy-pasted-until-corrected version — the same class
of drift `docs/RUST_AND_CRATE_GUIDELINES.md` §7 already names for CI
patterns generally. Takes one or more `path[:label]` raw-log arguments; the
optional label (`mid-platform-test.yml`'s own two feature configurations,
for instance) gets appended to every suite name parsed from that file.

## `.github/workflows/deploy-site.yml`

Two jobs: `build` (optionally re-runs `mid-ptr`/`mid-platform` tests behind
a `run_tests_first` dispatch input, builds the mdBook docs via a pinned
`mdbook` release binary — not `cargo install`, which would spend real
minutes compiling it fresh on every run — then assembles `dist/`) and
`deploy` (downloads that artifact, ships it with
`cloudflare/wrangler-action@v3` using the `CLOUDFLARE_API_TOKEN` /
`CLOUDFLARE_ACCOUNT_ID` repo secrets, already configured). Project name:
`mid-engine-docs`.

## Fixes and Problems

*(none yet — this is the initial build)*
