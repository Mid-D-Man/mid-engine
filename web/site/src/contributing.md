# Contributing & CI

## Manual CI only

Every test and benchmark workflow in this repository is `workflow_dispatch`
only. Nothing runs automatically on push, pull request, or a schedule —
tests and benchmarks take real time and shouldn't run on every commit. Run a
workflow from the repository's Actions tab when you actually want its
results.

Each workflow writes a real, structured summary to its own run's Job
Summary (parsed from the raw test/bench output, not a raw log dump), and
uploads the raw log plus a JSON results file as a build artifact.

## Publishing the site

This site itself deploys the same way: `Deploy Site`
(`.github/workflows/deploy-site.yml`) is also `workflow_dispatch` only. It
rebuilds this book, gathers the latest JSON results each crate's own test
workflow produced, and deploys the whole thing to Cloudflare Pages. Running
a crate's test workflow updates that crate's *own* results; running `Deploy
Site` afterward is what actually publishes them here.

## Documentation conventions

- Every source file starts with a `NOTICE` header comment pointing to its
  crate's own `docs/<crate-name>.md` file and section there. Inline code
  comments describe what/how only — no fix history or decision logs inline.
- Each crate's `docs/<crate-name>.md` (in the repository's top-level
  `docs/`, not nested per crate) holds per-file sections, a CI/workflow
  reference section, and a bottom "Fixes and Problems" section.
- This book (`web/site/`) is the separate, user-facing counterpart to those
  internal docs — see [Introduction](introduction.md) for the split.
