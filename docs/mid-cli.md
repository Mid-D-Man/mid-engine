# mid-cli: the `mid` command line

**Status: design only. Nothing is built, and nothing here has been verified.** Written on
2026-11-16 as proposals for the maintainer to confirm or change. No crate exists yet.

## Overview

One binary, `mid`, so that the whole engine is usable from a terminal: the development
workflow (check, test, bench, docs, delivering and applying changes) and, through a remote
protocol, a running world (list entities, read and edit components, step the simulation).

Two audiences, in this order: **scripts and AI**, then people. Every design choice below
favours output a program can rely on over output that reads nicely. A person at a prompt
gets a readable form by default; `--json` gives the stable one.

It is also the first half of the engine app. The app and the command line are two clients of
the same remote protocol (`docs/roadmap.md`, Decision 8), so a screen in the app has a
command-line twin, which keeps both honest and gives the app a way to be tested without a
screen.

## Why it matters now

Work on this repository goes through a shell and git, one command at a time, with the same
conventions each time: which toolchain, how to run the tests, how to summarise them, how to
build a delivery tarball, how to apply it to a fresh clone and compare. Those conventions
live in a person's or an assistant's head. A command that does them, with output a program can
read, removes a class of mistake and makes the next session cheaper.

## Principles

1. **`--json` everywhere.** One JSON document for a command, or JSON lines for a stream. A
   `schema` field with a version number. Fields are added, never repurposed.
2. **Exit codes mean something.** 0 success; 1 the thing asked failed (tests failed, lint
   found problems); 2 usage error; 3 the environment is wrong (a toolchain or tool is
   missing); 4 a remote error from a running world.
3. **No prompts unless there is a terminal,** and `--yes` always skips them. Every command
   that writes has `--dry-run`.
4. **Bounded output.** Test and build output is reduced to a summary (counts, failures, a
   path to the full log) unless `--full` is given. A tool that floods its caller is a tool
   that gets truncated at the worst moment.
5. **Self-describing.** `mid help --json` lists commands and flags; `mid rpc discover` lists
   a running world's methods and component schemas.
6. **Standard library only.** Hand-written argument parsing and hand-written JSON, as
   `web/mid-ui-wasm` already does without `serde`. The binary stays small and builds on every
   target the engine does.
7. **No network unless the command says so.** Only `mid ci` and `mid world --tcp` touch one.

## Command groups

Phase numbers refer to "Phases", below.

| Command | What it does | Needs | Phase |
|---|---|---|---|
| `mid doctor` | Reports the toolchain (rustc, cargo, the wasm32 target, `mdix`, `git`), the repo state and the runner, and exits 3 if something required is missing | nothing | 1 |
| `mid check [crate]` | `cargo check`, `rustfmt --check` and `clippy --no-deps`, summarised per crate | a toolchain | 1 |
| `mid test [crate] [--features ..]` | Runs the tests and reports counts (lib, doc, per feature set), failures, and the change against the previous run | a toolchain | 1 |
| `mid docs lint` | Checks the documentation standard: a NOTICE header in source files pointing to a real `docs/<name>.md`, no em dashes in new doc text, links to other docs resolve | nothing | 1 |
| `mid apply <tarball>` | Applies a delivery tarball the way the repository's sync does, using the real `mdix` tool, then verifies every file is byte-identical to the tarball's | `mdix` | 1 |
| `mid deliver` | Builds a delivery tarball from the files that differ from `origin/main` (the current convention: one tarball, `.github/` files and the root `Cargo.toml` separate) | `git` | 1 |
| `mid rpc call <method> <json>` | Raw call to a running world; `mid rpc discover` | remote protocol | 2 |
| `mid world start [project]` | Starts a headless world with the remote protocol on stdio or TCP | remote protocol | 2 |
| `mid world entities`, `get`, `set`, `query`, `spawn`, `despawn`, `reparent`, `components`, `resources`, `schema` | The world commands; names follow the protocol's methods | remote protocol | 2 |
| `mid world step [n]` | Advances a headless world by `n` frames, so a script or an AI gets deterministic stepping | remote protocol | 2 |
| `mid world watch` | A stream of changes, as JSON lines | remote protocol, TCP | 3 |
| `mid ci status`, `dispatch`, `results` | Reads and triggers workflow runs, and fetches the bench and test summaries they produce | a token, the network | 4 |
| `mid bench [crate]` | Runs a bench locally and says plainly that real CI is the authoritative number | a toolchain | 4 |
| `mid scene load`, `save`, `instantiate` | Scene operations | `mid-nodes`, reflection | 4 |
| `mid new <template>` | Scaffolds a project, using `mdix-scaffold` | `mdix` | 4 |
| `mid explain <topic>` | Prints the relevant doc section for a crate or topic from `docs/`, so a program can look things up without scraping | nothing | 4 |

## How it relates to what exists

- **`mdix-scaffold`** is a Node tool today (`node bin/mdix-scaffold.js generate ...`). `mid apply`
  and `mid new` call it rather than reimplement it. DixScript-Rust needs Rust 1.80 or later, so
  the CLI does not depend on it; keeping the CLI free of DixScript also keeps it buildable
  wherever the engine is.
- **The remote protocol** (`mid-remote`, Decision 8) is what `mid world` speaks. Until it
  exists only the phase 1 commands can be built, and they need nothing from the engine.
- **AI skills.** The repositories already carry skill files for assistants
  (`mdix-scaffold/ai/claude/`). A skill file describing `mid`'s commands and conventions would
  let an assistant use it consistently. Where that file lives is open (below).

## Where it runs

- The workflow commands need the repository and a Rust toolchain: a desktop, a CI job, or a
  cloud development environment reached from a browser.
- The world commands need only the `mid` binary and a world to talk to (a child process on
  stdio, or a TCP address).
- **Unverified:** whether the binary runs in a terminal app on an Android phone (Termux is the
  usual candidate), and whether a cloud development environment is usable from a phone
  browser. Neither has been tried.

## Decisions (all proposed)

- **Name:** the binary is `mid`, the crate `mid-cli`.
- **Location:** `tools/mid-cli`, as its own workspace to start with (the `web/mid-ui-wasm`
  precedent), so it adds no member to the engine's workspace and cannot be broken by, or
  break, the engine's MSRV walls. Joining the main workspace is a later, separate change.
- **Phase 1 depends on nothing in the engine.** It can be built and used before reflection or
  the remote protocol exist.
- **`mid deliver` encodes the current delivery convention** rather than inventing one. If the
  convention changes, the command changes with it.
- **Exit codes and `--json` are part of the interface from the first version,** not added later.

## Phases

| Phase | What | Needs | Done when |
|---|---|---|---|
| 1 | Skeleton, argument parsing, JSON output; `doctor`, `check`, `test`, `docs lint`, `apply`, `deliver` | nothing | Each command has tests; `mid apply` reproduces the byte-identical check done by hand today |
| 2 | `rpc` and the `world` commands over stdio | `mid-reflect`, `mid-remote` | `mid world start` plus `get` and `set` round-trip a component in a headless world |
| 3 | TCP, and `watch` | phase 2 | A second process drives a running world |
| 4 | `ci`, `bench`, `scene`, `new`, `explain` | their dependencies | One at a time, as each dependency lands |

## Not built, on purpose

- A text user interface, colour themes, shell completions beyond the basics.
- Anything that needs a renderer.
- Publishing to crates.io.
- A plugin system for third-party subcommands.

## Open questions for the maintainer

1. Is `mid` the right binary name?
2. Standalone workspace under `tools/` to start (proposed), or a member of the main workspace?
3. Should `mid deliver` exist, or do you want to keep assembling delivery tarballs the current way?
4. Where should the skill file for `mid` live: `mdix-scaffold/ai/claude/` with the other skills,
   or in this repository?
5. Phase 1 first (proposed, no dependencies, immediate use), or wait and build it together with
   the remote protocol?

## Verification status

None. This is a design document. No code, no tests, no CI.
