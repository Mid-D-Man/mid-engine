# Contributing to Mid Engine

Thanks for your interest in Mid Engine — a purely ECS-based,
multiplayer-first game engine, built with two non-negotiable mandates:
every crate stays FFI-compatible enough to be usable from any game
engine, and peak performance is a primary design goal, not an
afterthought. Minimal external dependencies runs through the whole
project — foundational crates (`mid-collections`, `mid-ecs`) aim for
zero regular dependencies where possible.

Please read our [Code of Conduct](./CODE_OF_CONDUCT.md) — we want this
to be a welcoming place to work.

## Before you start

Get oriented first, not by asking — the answer is very likely already
written down:

- `docs/architecture.md` — the high-level shape of the engine.
- `docs/roadmap.md` — what's built, what's next, in what order.
- `docs/dev-setup.md` — toolchain, how to build and test locally.
- `docs/bevy-comparison.md` — why (and where) we look at Bevy's
  `Mid-D-Man/bevy` fork as a reference, and where our design
  deliberately diverges from it.
- Each crate's own `docs/mid-*.md` — the real, current design and
  status of that crate, including known issues that are tracked but
  not yet fixed. If a doc and the code disagree, that's a real bug in
  the doc — flag it, don't just trust whichever one you read first.

## What we're looking for

- **Small, focused contributions** over large ones. A change that
  does one real thing and is easy to review is worth more to this
  project than a big one nobody has the context to check line by
  line.
- **Real verification, not assertions.** "This should work" isn't a
  finished contribution. Run the tests. Run clippy. If you touched
  anything FFI-facing, run the real C smoke test. Paste actual output,
  not a description of expected output.
- **Zero assumptions about existing structure.** Check the actual
  current file/API before building on top of it — this project moves
  fast enough that memory (yours, an AI's, or an old doc's) goes stale
  quickly.

## Using a reference crate (Bevy or otherwise)

Where a Bevy file is doing essentially the same job we'd otherwise
build from scratch, adapt it directly — rename, relocate, make the
minimal changes needed — rather than reinventing it. Where adapting a
file would pull in a Bevy-internal crate we haven't built yet
(`bevy_reflect`, `bevy_tasks`, `bevy_platform`, `bevy_ecs_macros`,
`bevy_ptr`, `bevy_utils`), hold off and leave a note to come back to it
rather than half-porting something that won't compile. And when an
adapted file pulls in an *external* dependency, check first whether
it's something we can reasonably hand-roll ourselves before adding it
— that's the whole reason the dependency budget exists.

## Pull requests

1. Fork, branch, make your change.
2. Make sure it actually builds and passes tests locally — see
   `docs/dev-setup.md` for the toolchain this project verifies against.
3. Open a PR describing *what* changed and *why*. Link the doc section
   your change affects, if any, and update that doc in the same PR if
   it goes stale.
4. CI has to be green (or, where a step is a known, separately-tracked
   soft-failure, explain why in the PR).

## AI Usage Policy

Mid Engine's own development leans heavily on AI-assisted work — this
project doesn't pretend otherwise, and isn't opposed to it. But
"assisted" is the operative word, and that's what this policy exists
to make concrete.

- **A human is the author.** Every contribution needs a human who
  understands it well enough to explain and defend it in review,
  regardless of how much of the first draft came from an AI tool.
  "The AI wrote it and I didn't check" is not an acceptable state for
  anything submitted here.
- **AI does not get final say.** A model's own claim that something
  is correct, performant, or complete is not evidence of any of those
  things. Nothing gets merged on the strength of "the AI said it
  works" — it gets merged on the strength of real test output, real
  benchmark numbers, and real CI runs that anyone can re-check.
  Co-designing with an AI tool is welcome; treating its output as
  self-certifying is not.
- **No unreviewed, un-self-reviewed dumps.** Reviewing a large,
  AI-generated change is expensive, and it's not fair to ask
  maintainers to do the understanding you skipped. If a contribution
  is substantially AI-assisted, read it yourself first, and be ready
  to answer questions about any part of it — including the parts that
  looked fine at a glance.
- **Disclose significant AI involvement.** A short note in the PR
  description or a commit trailer (e.g. `Assisted-by: <tool>`) is
  enough. This isn't about gatekeeping the tool, it's about giving
  reviewers the right context.
- **Benchmarks and perf claims need real numbers, from a real run on
  this project's actual verification setup** — not a model's estimate
  of what the numbers would probably look like. Given this project's
  own performance mandate, an unverified perf claim is worse than no
  claim at all.

If you're unsure whether a specific case fits this policy, ask — that's
a much better outcome than either silently skipping AI assistance
you'd have found useful, or shipping something nobody actually
checked.
