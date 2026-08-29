# Benching Standards

How this project's `bench-vs-*.yml`/`Abench-*.yml` workflows are
structured, so a new one doesn't have to rediscover the pattern (or skip
part of it) each time. Written after `bench-vs-bevy-ecs.yml` shipped
without the structured-summary step this doc describes — checked
directly against every existing bench workflow rather than assumed.

## The real, current state: two patterns, not one

Grepping every `bench-vs-*.yml`/`Abench-*.yml` file turned up two
genuinely different approaches in active use, not one consistent
standard:

1. **Python parser** (`bench-vs-mat4-fastest.yml`, `bench-vs-color.yml`,
   `bench-vs-c-libs.yml`, `bench-vs-mid-vec.yml`, and others) — a
   `scripts/bench_vs_*.py` script parses the raw criterion output into
   real per-group markdown tables, optionally with a ratio-vs-baseline
   column and colored badges.
2. **Bash grep + collapsible raw dump** (`bench-mid-collections-sparse-
   set.yml`) — no Python, just `grep -B 1 "time:"` for a quick headline
   table plus the full raw log behind a `<details>` fold.

Both are legitimate for what they show. Neither is "wrong." But a bench
workflow written without deliberately picking one (what `bench-vs-bevy-
ecs.yml` did originally) ends up with neither — just a raw, unparsed
wall of criterion text in the step summary, which is the actual problem
this doc exists to prevent from happening again.

**Recommendation for new bench workflows:** use the Python parser
pattern when there's a natural per-group baseline to compute a ratio
against (a "vs X" comparison, which is most of them) — it's the more
information-dense summary and the one worth writing once, correctly,
and reusing the shape of. Fall back to the bash pattern only for
something that doesn't have a clean baseline concept.

## The correctness detail that actually matters more than formatting

**`set -o pipefail` is not optional.** `cargo bench ... | tee raw.txt`
without it means the step's exit code is `tee`'s (always `0`), not
`cargo bench`'s — a real benchmark failure shows green. This was a real
incident (see `Abench-vs-all-f64.yml`'s own "Run f64 criterion
benchmarks" step comment and `mid-ecs-test.yml`'s), and it's why
`mid-ecs-test.yml`, `bench-mid-collections-sparse-set.yml`, and four of
the `bench-vs-*.yml` files have it. **Checked directly: 12 of the
`bench-vs-*.yml` files — including `bench-vs-mat4-fastest.yml`, the one
`bench-vs-bevy-ecs.yml` was originally modeled on — do not have it.**
Not fixed as part of this pass (real, but out of scope for what was
asked); worth a deliberate cleanup pass, not a silent gap to
rediscover the hard way later.

## The recommended shape, end to end

What `bench-vs-bevy-ecs.yml` now does, as the canonical reference:

1. `workflow_dispatch` only, with a free-text `baseline_note` input.
   **Never** `push`/`pull_request` — benchmarks take real time and
   shouldn't run on every commit.
2. `dtolnay/rust-toolchain@stable` (tracks true latest stable) +
   `rustc --version` captured to a step output for the summary header.
   If the target crate has a real MSRV wall above whatever the sandbox
   this was written in could reach, say so explicitly in the workflow's
   own header comment — this project doesn't hide toolchain gaps.
3. Cache the cargo registry, keyed on the relevant `Cargo.toml`(s).
4. Run the bench with **`set -o pipefail`** set first, `tee`'d to a
   `bench-<name>-raw.txt` file.
5. `if: always()` **diagnostics step** — `grep` the raw log for
   criterion's own warning strings (`took zero time`, `Unable to
   complete`, `Warning:`, `panicked`, `Completed N iterations`) into
   their own step-summary block. Cheap, and it surfaces real sampling
   problems (a real one showed up in `bench-vs-bevy-ecs.yml`'s first
   actual run — criterion couldn't complete 100 samples in 5s for the
   `bevy_ecs` structural-churn case, extended to 7.5s on its own) that
   would otherwise sit buried in a wall of raw text.
6. `if: always()` **summary step** — either the Python parser or the
   bash-grep pattern above, not the raw dump alone.
7. `if: always()` **upload the raw log as an artifact**, 30-day
   retention. The step summary is for skimming; the raw file is for
   when someone actually needs the full confidence intervals.

## Writing a `scripts/bench_vs_*.py`

Real, working shape (see `scripts/bench_vs_bevy_ecs.py` for the current
reference implementation, adapted from `bench_vs_mat4_fastest.py`):

- Strip ANSI codes first (`RE_ANSI`), criterion colors its output.
- `RE_RESULT` matches criterion's real `name\n    time: [lo mid hi]`
  shape — grounded in `report.rs`'s actual source at whatever criterion
  version is pinned, not memory. `bench-mid-collections-sparse-set.yml`
  found this the hard way (see its own header comment): it wrote a
  parser it could never actually run against real output at the time,
  since the sandbox that wrote it couldn't get `cargo bench` running at
  all — grounded in real source, but a real workflow trigger was still
  the first actual proof. Where possible, test the parser against real
  captured output before trusting it, the way `bench_vs_bevy_ecs.py`
  was checked against this workflow's own first real run.
- Group results by everything before the final `/` in the benchmark
  name (`spawn_n_entities_two_components/mid-ecs` → group
  `spawn_n_entities_two_components`, variant `mid-ecs`).
- If one variant per group is the natural baseline (a reference
  implementation, or — for a two-engine comparison like `vs_bevy_ecs`
  — the *other* engine), compute a ratio and badge it:
  `≤1.05× parity`, `≤1.5× warn`, `≤5.0× error`, `>5.0×` flagged as
  overhead-dominated. Don't hardcode a *reason* for a bad ratio in the
  script unless it's a real, separately-diagnosed root cause (the way
  `bench_vs_mat4_fastest.py` does for its own, already-profiled
  storage-layout issue) — a first-ever run's script should report
  honestly, not guess.

## Related

- `docs/bevy-comparison.md` / `docs/bevy-file-adoption.md` — why some
  benches exist at all (comparing against a reference implementation).
- `docs/mid-ecs.md`, `docs/mid-math.md` — the design context a given
  bench's numbers should get read against.

## Benchmarking across runners and platforms

This is a real, separate axis from the summary-formatting stuff above,
and `bench-vs-bevy-ecs.yml` didn't have it either on first write —
checked directly against what's actually in this repo (`Abench-vs-all-
f64.yml`, `mid-math-test-neon.yml`, `test-geom.yml`) rather than
assumed. Two genuinely different concerns, easy to conflate:

### 1. ISA-tier SIMD dispatch matrix — only for vectorized, dispatched-backend code

`Abench-vs-all-f64.yml` (and its f32 counterpart) sweep a real
`target_cpu` `workflow_dispatch` choice input — `native`, `x86-64-v4`
(AVX-512), `x86-64-v3` (AVX2+FMA), `x86-64-v2` (SSE4.2), `x86-64`
(SSE2-only), `neon`, `wasm`, `scalar` — because mid-math genuinely
*has* a different dispatched SIMD backend per ISA tier
(`f32/{sse2,avx,neon,wasm}/`), and the whole point of the matrix is
proving the right backend gets picked and actually performs on each
real tier. Real mechanics worth reusing wherever this pattern actually
applies:

- **`CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUSTFLAGS`, not plain
  `RUSTFLAGS`**, for the target-cpu flag — plain `RUSTFLAGS` also
  applies to build-script (host) compilation, which `SIGILL`s if the
  runner's *real* CPU doesn't have the requested ISA (GitHub-hosted
  `ubuntu-latest` runners are AMD EPYC, no AVX-512).
- **Soft-gate hardware you can't guarantee**: `x86-64-v4` checks
  `/proc/cpuinfo` for `avx512f` first and marks the run
  skipped-not-failed with an explanatory step summary if it's absent,
  rather than actually attempting a `SIGILL`ing run.
- **Real ARM coverage exists and is cheap**: `mid-math-test-neon.yml`
  runs on `macos-14` (Apple Silicon M1) — GitHub's free-tier hosted
  ARM runner, not a self-hosted rig. `neon` in the f64/f32 bench
  matrices runs on `ubuntu-24.04-arm` instead (also free-tier hosted),
  with `-C target-cpu=native` as plain `RUSTFLAGS` — safe there
  specifically because host and target are the same architecture, so
  no cross-compile SIGILL risk the x86 tiers have.
- **wasm needs `wasmtime` + a `.cargo/config.toml` runner shim** to
  actually execute — see the `wasm` branch for the exact
  `[target.wasm32-wasip1]` config.

**When a bench does NOT need this matrix — real, already-established
precedent, not new**: `bench-mid-collections-sparse-set.yml`'s own
header comment explains it directly — `SparseSet`'s performance is
governed by cache locality and pointer-chasing, not vectorizable
arithmetic, so there's no ISA tier to sweep. **The same reasoning
applies to `vs_bevy_ecs.rs`** — `spawn`/`dense_query_iteration`/
`structural_churn` are all archetype-migration, hashmap-lookup, and
`Box<dyn Any>`-boxing bound, not SIMD arithmetic bound. Neither
`bench-mid-collections-sparse-set.yml` nor `bench-vs-bevy-ecs.yml` carry
the `target_cpu` matrix, and that's a deliberate omission, not a gap —
don't add one without a real, dispatched-SIMD-backend reason to.

### 2. Cross-OS/cross-arch portability — a different, still-real concern

Independent of SIMD: does the thing actually build and behave the same
on Linux, macOS, and Windows? `test-geom.yml`'s pattern is the
reference — a plain `strategy.matrix.os: [ubuntu-latest, macos-latest,
windows-latest]` with `fail-fast: false`, no ISA-tier logic at all,
just "does this work everywhere Mid Engine cares about."

This is the piece `bench-vs-bevy-ecs.yml` was missing and now has: an
opt-in `platforms` `workflow_dispatch` choice (`ubuntu-only` fast
default, `all` for the full `ubuntu-latest` + `macos-latest` +
`ubuntu-24.04-arm` sweep) — opt-in rather than every-run, because
`bevy_ecs`'s ~400-crate dependency tree makes each platform's build
alone take real minutes, and this bench doesn't need to pay that on
every invocation the way a plain correctness test-matrix (`test-
geom.yml`) does on every dispatch. Reach for `all` before trusting a
cross-platform performance claim from this bench specifically, not by
default.
