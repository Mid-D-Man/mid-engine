# mid-cli: the `mid` command line

## Overview

One binary, `mid`, so that the whole engine is usable from a terminal: the development workflow
(check, test, documentation lint, delivering and applying changes) and, later, through a remote
protocol, a running world (list entities, read and edit components, step the simulation).

Two audiences, in this order: **scripts and AI**, then people. Every design choice favours output
a program can rely on over output that reads nicely. A person at a prompt gets a readable form by
default; `--json` gives the stable one.

The engine app is an editor with its own embedded world (`docs/mid-engine-app.md`), so it does not
need the remote protocol. The command line and an editor attached to a remote engine would be the
two clients of that protocol (`docs/roadmap.md`, Decision 8). Both rely on `mid-reflect` to read and
write a world by name, which keeps what the command line can do close to what the editor can do.

**Status.** Phase 1 is built: `doctor`, `check`, `test`, `docs lint`, `apply` and `deliver`. The
world commands (phase 2 onward) wait for `mid-reflect`'s ECS bridge and for the remote protocol.
The crate lives in `tools/mid-cli` as its own workspace, so it adds nothing to the engine's root
`Cargo.toml`.

## Why it matters now

Work on this repository goes through a shell and git, one command at a time, with the same
conventions each time: which toolchain, how to run the tests, how to summarise them, how to build
a delivery tarball, how to apply it to a fresh clone and compare. Those conventions live in a
person's or an assistant's head. A command that does them, with output a program can read,
removes a class of mistake and makes the next session cheaper.

## Principles

1. **`--json` everywhere.** One JSON document per command, on one line, with a `schema` field
   (currently 1). Fields are added, never repurposed.
2. **Exit codes mean something.** 0 success; 1 the thing asked failed (tests failed, lint found
   problems); 2 usage error; 3 the environment is wrong (a toolchain or tool is missing). 4 is
   reserved for a remote error from a running world.
3. **No prompts, and `--dry-run` on every command that writes.**
4. **Bounded output.** Test and build output is reduced to a summary (counts, failures, a path to
   the full log). A tool that floods its caller gets truncated at the worst moment.
5. **Self-describing.** `mid help` lists the commands; `mid help --json` lists them for a
   program.
6. **Standard library plus `mid-reflect`.** Argument parsing is written by hand. JSON output uses
   `mid-reflect`'s `Value`, so the crate has no external dependencies and does not carry a second
   JSON writer.
7. **No network unless the command says so.** None of the phase 1 commands touches one (`apply`
   runs `node`, which does not either).

## Commands

| Command | What it does | Needs | Phase |
|---|---|---|---|
| `mid doctor` | Reports the toolchain, the wasm32 target, the mdix tool and the repository state; exits 3 if a required tool is missing | nothing | 1, built |
| `mid check` | `cargo check`, `rustfmt --check` and `clippy --no-deps` for the crates given, summarised per crate | a toolchain | 1, built |
| `mid test` | Runs the tests for the crates given and reports counts per target, failures, and the change since the last run | a toolchain | 1, built |
| `mid docs lint` | Checks the documentation standard on the files that changed | git | 1, built |
| `mid apply <tarball>` | Applies a delivery tarball with the real mdix tool, then verifies every member is byte-identical in the repository | node, mdix | 1, built |
| `mid deliver` | Builds a delivery tarball from the files that differ from a base | git, tar | 1, built |
| `mid rpc call`, `mid rpc discover` | Raw calls to a running world | remote protocol | 2 |
| `mid world start`, `entities`, `get`, `set`, `query`, `spawn`, `despawn`, `reparent`, `components`, `resources`, `schema`, `step` | The world commands; names follow the protocol's methods | remote protocol, reflection | 2 |
| `mid world watch` | A stream of changes, as JSON lines | remote protocol, TCP | 3 |
| `mid ci status`, `dispatch`, `results` | Reads and triggers workflow runs, and fetches the bench and test summaries they produce | a token, the network | 4 |
| `mid bench [crate]` | Runs a bench locally and says plainly that real CI is the authoritative number | a toolchain | 4 |
| `mid scene load`, `save`, `instantiate` | Scene operations | `mid-nodes`, reflection | 4 |
| `mid new <template>` | Scaffolds a project, using `mdix-scaffold` | mdix | 4 |
| `mid explain <topic>` | Prints the relevant doc section for a crate or topic from `docs/` | nothing | 4 |

## Interface

- **Global flags:** `--json`, `--repo DIR` (or `--repo=DIR`), `--help`/`-h`, `--version`/`-V`. They
  may come before or after the command. Nothing after a bare `--` is read as one.
- **Environment:** `MID_MDIX` names the directory of the mdix tool, for `apply` and `doctor`.
- **JSON:** every report has `schema`, `command` and `ok`, then the command's own fields. A
  failure of how the command was run (a bad flag, a missing tool) has an `error` field with a
  `kind` and a `message`.
- **Logs and state:** full tool output goes to `<temp>/mid-cli/logs/`, and the report carries the
  path. `mid test` keeps the counts of its last run under `<temp>/mid-cli/state/`.

## How it relates to what exists

- **`mdix-scaffold`** is a Node tool (`node bin/mdix-scaffold.js generate ...`). `mid apply` calls
  it rather than reimplementing it. DixScript-Rust needs Rust 1.80 or later, so the command line
  does not depend on it, which keeps the binary buildable wherever the engine is.
- **`mid-reflect`** gives the command line its `Value` and JSON writer. It is the only engine
  crate the phase 1 commands use.
- **The remote protocol** (`mid-remote`, Decision 8) is what `mid world` will speak. Until it
  exists only the phase 1 commands are possible, and they need nothing else from the engine.
- **AI skills.** The repositories carry skill files for assistants (`mdix-scaffold/ai/claude/`). A
  skill file describing `mid`'s commands and conventions would let an assistant use it
  consistently. Where that file lives is open (below).

## Where it runs

- The workflow commands need the repository and a Rust toolchain: a desktop, a CI job, or a cloud
  development environment reached from a browser.
- The world commands will need only the `mid` binary and a world to talk to.
- **Verified:** a Linux sandbox with rustc 1.97.1 and 1.75.0, GNU tar 1.35, node 22 and the real
  mdix tool.
- **Not verified:** macOS and Windows (`apply` and `deliver` call `tar`, and path handling is
  untested there), and whether the binary runs in a terminal app on an Android phone (Termux is
  the usual candidate).

## Decisions

All proposed, for the maintainer to confirm or change.

- **Name:** the binary is `mid`, the crate `mid-cli`.
- **Location:** `tools/mid-cli`, as its own workspace, so it adds no member to the engine's
  workspace and cannot be broken by, or break, the engine's MSRV walls. Joining the main
  workspace is a later, separate change.
- **A dependency on `mid-reflect`.** The first design said phase 1 would depend on nothing in the
  engine and write its own JSON. With `mid-reflect` built, the crate uses its `Value` and JSON
  instead of carrying a second writer. The cost is that `tools/mid-cli` no longer builds without
  `crates/mid-reflect` beside it.
- **`mid deliver` encodes the current delivery convention** rather than inventing one: one
  tarball of the files that differ from the base; `.github/` files and the root `Cargo.toml`
  listed as separate; lockfiles, tarballs and `target/` never included. If the convention
  changes, the command changes with it.
- **`mid apply` verifies by bytes.** It trusts the mdix tool to place files and then compares
  every member of the tarball with the repository.
- **Exit codes and `--json` are part of the interface from the first version.**

## Phases

| Phase | What | Needs | Done when |
|---|---|---|---|
| 1 | Skeleton, argument parsing, JSON output; `doctor`, `check`, `test`, `docs lint`, `apply`, `deliver` | `mid-reflect` | Each command has tests, and `mid apply` reproduces the byte-identical check done by hand. **Done.** |
| 2 | `rpc` and the `world` commands over stdio | the ECS bridge, `mid-remote` | `mid world start` plus `get` and `set` round-trip a component in a headless world |
| 3 | TCP, and `watch` | phase 2 | A second process drives a running world |
| 4 | `ci`, `bench`, `scene`, `new`, `explain` | their dependencies | One at a time, as each dependency lands |

## Not built, on purpose

- A text user interface, colour themes, shell completions beyond the basics.
- Anything that needs a renderer.
- Publishing to crates.io (`publish = false`).
- A plugin system for third-party subcommands.

## Open questions for the maintainer

1. Is `mid` the right binary name?
2. A standalone workspace under `tools/` (as built), or a member of the main workspace?
3. Where should the skill file for `mid` live: `mdix-scaffold/ai/claude/` with the other skills,
   or in this repository?
4. Should `mid docs lint` run in CI on pull requests? It needs the base branch fetched.
5. Is the dependency on `mid-reflect` acceptable, or should the crate keep a private JSON writer
   so that it builds alone?

## Modules

### `main.rs`

**What it does:** the `mid` binary. It reads the process environment, calls `mid_cli::run` with
the real process runner, prints what comes back and exits with its code. Nothing else lives here.

**Tests:** none of its own. `tests/basics.rs` runs the library entry point.

### `lib.rs`

**What it does:** `run`, the entry point, and the types around it. `Env` holds the working
directory, the mdix tool's location (`MID_MDIX`) and a temp directory. `RunResult` is the text to
print and the exit code. `COMMANDS` is the command list that `help` prints. `run` splits the
global flags out of the arguments, answers `--help` and `--version`, finds the repository
directory, and hands the rest to the command.

**Decisions:**
- `run` takes the arguments, a `Runner` and an `Env`, and returns text and a code. It never reads
  the process environment and never prints, so every command can be tested without a terminal.

**Tests:** 18 unit tests across the library, and `tests/basics.rs` (9 tests) for the global flags,
help and version.

### `args.rs`

**What it does:** argument parsing by hand. A command gives a spec, a list of
`(long name, kind, short letter)` entries; `Flag` is a switch, a single value, or a repeatable
value. `Parsed` answers `has`, `value` (the last one given), `values`, `positional` and
`passthrough` (everything after a bare `--`, untouched). Long flags take `--flag=value` or
`--flag value`; short flags are a single letter.

**Decisions:** an unknown flag is a usage error naming the flag, never ignored.

**Tests:** unit tests in the file, and every command's usage-error cases.

### `proc.rs`

**What it does:** running other programs behind a trait. `Cmd` is a program, arguments and an
optional directory. `Outcome` is the exit code, the two output streams and, if the program could
not be started, why. `Runner` runs a `Cmd`; `SystemRunner` runs it for real.

**Decisions:** a program that is not found is an `Outcome` with a spawn error and no code, not a
panic and not an exit code, so a command can tell "not installed" from "failed". Tests replace the
runner with a scripted one.

**Tests:** used by every test file; `tests/common/mod.rs` holds the scripted runner.

### `report.rs`

**What it does:** `Report`, what a command returns. It renders as plain lines for people and as
one JSON line for programs. It also holds the exit codes (`EXIT_OK` 0, `EXIT_FAILED` 1,
`EXIT_USAGE` 2, `EXIT_ENV` 3), `SCHEMA`, and the helpers `strings` and `count`. `Report::error`
builds the failure-of-how-it-was-run form.

**Decisions:** one structure with both renderings, so the text and the JSON cannot drift apart.

**Tests:** unit tests, and the JSON shape is asserted in every command's tests.

### `util.rs`

**What it does:** small helpers: `fnv_hex` (a stable short hash used for state file names),
`write_log` (a new file under `<temp>/mid-cli/logs/`, named by time and a cleaned label),
`path_text`, and durations.

**Tests:** unit tests in the file.

### `git.rs`

**What it does:** what the commands need from git. `toplevel` finds the repository root,
`ref_exists` checks a reference, `changes_since` lists the files that differ from a base (changed,
deleted and untracked-but-not-ignored, with the new name for a rename), and `added_lines` with
`parse_added_lines` give the lines a diff added, with file and line number.

**Tests:** unit tests for the diff parsing; `tests/deliver.rs` and `tests/docs_lint.rs` use real
temporary repositories.

### `cargo_out.rs`

**What it does:** reading what cargo prints. `parse_test_output` reads `cargo test` output into a
result per target, the names of failed tests and the lines that start a compile error.
`parse_diags` reads diagnostics in the short message format (`path:line:col: level: message`), and
`count_unlocated_errors` counts errors that carry no file position.

**Decisions:**
- Target labels (`Running ...`, `Doc-tests ...`) are matched to results by order, not by position
  in the text, because cargo prints the labels on stderr and the results on stdout and the two
  arrive joined, one stream after the other (see "Fixes and Problems").
- Cargo's own summary lines (`could not compile`, `test failed`, `N targets failed`) are not
  counted as build errors, only the error that caused them.

**Tests:** unit tests, including the split-streams case, and `tests/real_cargo.rs`.

### `cmds/mod.rs`

**What it does:** the command table and what commands share: `help`, the usage-error report, and
`repo_root`, which turns "not inside a git repository" into an environment failure (exit 3).

**Tests:** `tests/basics.rs`.

### `cmds/doctor.rs`

**What it does:** `mid doctor`. It probes `rustc`, `cargo` and `git` (required) and `rustfmt`,
`clippy`, `node` and `tar` (optional), whether the wasm32 target is installed, whether `MID_MDIX`
points at a directory with `bin/mdix-scaffold.js`, and the repository state (root, short commit,
branch, number of changed paths, whether `origin/main` exists). Exit 3 if a required tool is
missing.

**Tests:** `tests/doctor_test_check.rs` (17 tests, with a scripted runner for all three of
doctor, test and check).

### `cmds/check.rs`

**What it does:** `mid check`. For each crate given (`-p`, repeatable) or the whole workspace
(`--all`), it runs `cargo check --all-targets`, `cargo fmt --check` and `cargo clippy --no-deps
--all-targets`, and reports warnings and errors per step, the first `--limit` findings (default
10) and the path of the full log. `-F`/`--features` and `--target` pass through. `--no-fmt` and
`--no-clippy` skip a step. `--deny-warnings` fails the step on a warning.

**Decisions:**
- Only the package's own diagnostics count. A crate's path dependencies in the same repository
  report warnings of their own when cargo checks them, and those are reported separately as
  `other_warnings`. The package's directory comes from `cargo metadata --no-deps`.
- Clippy runs with `--no-deps` for the same reason.
- If the directory cannot be found, every diagnostic is counted against the package and the report
  says so in a note line.

**Tests:** `tests/doctor_test_check.rs`, and `tests/real_cargo.rs`, which runs real cargo in a
two-crate workspace with no lockfile.

### `cmds/test.rs`

**What it does:** `mid test`. It runs `cargo test --color=never --no-fail-fast` for the crates
given, with `-F`, `--lib`, `--doc`, and anything after `--` passed to the test binary. It reports
passed, failed and ignored counts per target, the names of failed tests, build errors, the log
path, and the change since the last run of the same scope (same crates, features and filters).

**Decisions:**
- `--no-fail-fast` always, so a failure in one target does not hide the others.
- A run that executed no target is not a success, even if cargo exited 0.
- The last counts are stored under the temp directory, keyed by repository and scope, so the
  comparison needs no file in the repository.

**Tests:** `tests/doctor_test_check.rs`, and `tests/real_cargo.rs` for the labels.

### `cmds/docs_lint.rs`

**What it does:** `mid docs lint`, the documentation standard
(`docs/DOCUMENTATION_AND_COMMENTING_GUIDELINES.md`) checked on the lines that changed. `--base`
names the reference to compare with (default `origin/main`, falling back to `HEAD`); `--all` checks
everything. Four rules:

- `em-dash`: an em dash in Markdown text added since the base.
- `banned-word`: one of the filler words the style guide rules out, in any of its forms, in added
  Markdown text.
- `notice-header`: a source file under `crates/*/src` or `tools/*/src` whose opening NOTICE header
  is missing, or names a `docs/` file that does not exist.
- `doc-section`: a source file with no heading for its name in its crate's documentation.

A line containing `mid-lint: allow` is skipped. The exit code is 1 when there are problems.

**Decisions:** only added lines are checked by default, so old documents are not flagged for text
that was already there.

**Tests:** `tests/docs_lint.rs` (13 tests, real temporary repositories).

### `cmds/apply.rs`

**What it does:** `mid apply <file.tar.gz>`. It lists the tarball's members and refuses the
delivery if any would leave the repository (an absolute path, a `..` part, a backslash). It copies
the tarball into `.mdix/replacements/`, runs the mdix tool through `node` with the overwrite
strategy (`--dry-run` is passed on), removes the copy (unless `--keep`), reads the tool's `REP` and
`NEW` lines, and then extracts the tarball and compares every member with the file in the
repository. The run fails if the tool exits nonzero, reports an error, did not place a member, or
left different bytes. `--mdix DIR` or `MID_MDIX` locates the tool.

**Decisions:**
- The comparison is by bytes and independent of the tool's own report, so a tool that says
  "replaced" but wrote something else is caught.
- A member the tool did not report is a failure even when its bytes match.
- `.mdix/replacements/gitkeep` is not compared, because the tool rewrites it on every run.
- A tarball of the same name already in the replacements folder stops the run before anything is
  copied or removed.

**Tests:** `tests/apply.rs` (22 tests). Real `tar` and git are used, and the `node` run is answered
by a stand-in that unpacks the tarball and prints the tool's lines, with variants that skip a
member, write wrong bytes, exit nonzero or count an error. The real mdix tool was run once, end to
end, on a fresh clone (see "Verification status"). Mutation testing: 26 mutants of this file, all
killed.

### `cmds/deliver.rs`

**What it does:** `mid deliver`. It lists the files that differ from `--base` (default
`origin/main`), sorts them by the delivery convention, writes one tarball of the deliverable
ones (`--out` for the path, `--name` for the base name, default `mid-delivery`), and checks that the
tarball's members are exactly those files. `--dry-run` reports without writing. It also lists the
files to deliver by hand, the files skipped, and the deleted files, because a tarball cannot
delete.

**Decisions:** the convention is `classify`: `.github/` and the root `Cargo.toml` are
**separate**; a `Cargo.lock` at any depth, anything under `.mdix/replacements/`, `.tar.gz` files
and `target/` directories are **skipped**; everything else is delivered.

**Tests:** `tests/deliver.rs` (11 tests, real temporary repositories and real `tar`).

## Verification status

- 92 tests pass on rustc 1.97.1: 18 unit, 9 basics, 22 apply, 11 deliver, 13 docs lint, 17
  doctor, test and check with a scripted runner, and 2 that run real cargo.
- `clippy --no-deps --all-targets` and `rustfmt` are clean.
- A real end-to-end run: `mid deliver` built a tarball of this change, and `mid apply` landed it on
  a fresh clone of `main` through the real mdix tool, with every member byte-identical.
- `mid doctor`, `mid check` and `mid test` were run for real against the repository, which is how
  the two bugs below were found.
- Mutation testing covers `cmds/apply.rs` only (26 mutants, all killed). The other commands have
  not been mutation tested.
- Not verified: other operating systems, Android, and any run on real CI before this delivery.

## CI and Workflows

- `.github/workflows/mid-cli-test.yml` runs the formatting check, clippy and the tests on stable,
  and the tests on Rust 1.75.0, from `tools/mid-cli`. Formatting and clippy do not fail the job,
  as in the other crate workflows. A run starts on `workflow_dispatch`, or when a commit message
  or pull request title contains `--mid-cli` or `--mid-all`.

## Fixes and Problems

### `cmds/deliver.rs`

- **A nested lockfile was delivered.** The first real `mid deliver` of this change put
  `tools/mid-cli/Cargo.lock` in the tarball, because only a root `Cargo.lock` was skipped. The
  repository commits no lockfile, and one written by a newer cargo (version 4) breaks the 1.75
  build. `classify` now skips a `Cargo.lock` at any depth.

### `cmds/check.rs`

- **Dependency warnings were counted against the package on a fresh clone.** The package's
  directory came from `cargo pkgid`, which exits 101 when there is no `Cargo.lock`, and the
  repository commits none. With the directory unknown, every diagnostic counted as the package's:
  `mid check -p mid-reflect -F math` reported 3096 warnings, all from `mid-math`. The directory
  now comes from `cargo metadata --no-deps`, which needs no lockfile. `tests/real_cargo.rs` fails
  against the old approach.

### `cargo_out.rs`

- **Every target had an empty label.** The labels were matched to results by position in the text,
  but cargo prints them on stderr and the results on stdout, and the two arrive joined, so every
  label came after every result. The scripted tests used interleaved output and passed. Labels
  are now matched to results by order.

### `args.rs`, `lib.rs`, `cmds/docs_lint.rs`

- Clippy's `type_complexity` was resolved with a `FlagSpec` alias and a `GlobalSplit` alias. Two
  `map_or(true, ...)` calls use `is_some_and`, because the `is_none_or` that clippy suggests needs
  Rust 1.82 and the workspace floor is 1.75.

### Tests

- **`tests/apply.rs`:** the first mutation run left two mutants alive, an errors count that was
  hidden by a second condition failing for the same reason, and an unasserted cleanup of the
  comparison folder. A test was added for each.
- **Scripted tests hid two bugs.** The check and test bugs above passed every scripted test,
  because the scripts supplied the output the code expected. `tests/real_cargo.rs` runs real cargo
  so that cargo's actual behaviour is part of the suite.
