# Introduction

Mid Engine is a modular, high-speed Rust engine toolkit — "The Middle Man."

Traditional monolithic engines hide their internals behind layers of
abstraction, while others demand unlimited hardware just to render a basic
scene. Mid Engine takes a different approach: a modular toolkit for
developers who want complete control over their hardware stack, with every
crate designed to be dropped into a larger project rather than forcing an
all-or-nothing commitment.

It's a true "Middle Man" in two senses. First, every crate is FFI-compatible
out of the box, built with as few external dependencies as possible, so it
can sit between a game's logic and a lower-level system (or a completely
different language's codebase) without friction. Second, it's meant to be
the layer between "roll everything yourself" and "adopt a full engine" —
whether that means assembling a sovereign engine from these crates, or
injecting individual modules into an existing C++/C# environment through raw
pointers.

## Where this book fits

This is the user-facing documentation site — architecture, how to use each
crate, and the project's own conventions. It's a different thing from the
`docs/*.md` files in the repository itself, which are internal, AI-facing
working notes (design decisions, fix histories, MSRV walls) meant for
whoever — human or AI — is actively developing the engine, not for someone
using it. If something here seems to assume familiarity a repo-only reader
wouldn't have, that's the split working as intended, not a mistake.

See [Tests](/tests/) and [Benchmarks](/benchmarks/) for live results from
each crate's own CI, and the [Getting Started](getting-started.md) page to
build the workspace yourself.
