<!-- docs/platform-optimization.md -->

# Mid Engine — Platform Optimization Rules

> **Rule 0:** The compiler is smarter than you think. Profile before you optimize.
> **Rule 1:** An optimization that cannot be benchmarked does not exist.
> **Rule 2:** An optimization that breaks FFI is a bug, not a feature.
> **Rule 3:** Debug-mode timing is not a benchmark. All decisions require [RELEASE] numbers.
> **Rule 4:** Every significant performance measurement must be published as a GitHub
>             Step Summary so it is permanently visible in the CI run history.

---

## 0. Build Mode Discipline

`cargo test` without `--release` runs with `opt-level = 0`. Numbers are
5–50× slower than release for math-heavy code. The label `[RELEASE]` or
`[DEBUG]` in every stress test output comes from `cfg!(debug_assertions)`
resolved at compile time. If you see `[DEBUG]` on a run you believed was
release, `cargo test --release` was not actually invoked.

The CI produces two passes per build:
- **Debug pass** → HTML dashboard (correctness: pass/fail counts only).
- **Release pass** → Job summary + artifact `release-perf.txt` (performance numbers).

Any claim of "X is Yns/op" must cite the build number and `[RELEASE]` label.

---

## 1. GitHub Step Summary Policy

**All significant performance measurements must be published as a GitHub Step Summary.**

A Step Summary is permanently linked to its CI run. It cannot be lost when
artifacts expire. It shows directly in the GitHub Actions UI without
downloading anything. It is the authoritative public record for any
optimization decision.

### What must appear in a Step Summary

Every crate that has performance stress tests must emit a Step Summary
containing at minimum:

- Build number, branch, commit, Rust version.
- All `[RELEASE]` stress test results (ns/op, ms total, budget check).
- Any `speedup vs baseline` comparisons.
- A clear label indicating the build mode of every number shown.

### How to emit a Step Summary in a workflow step

```yaml
- name: Publish performance summary
  run: |
    echo "## mid-math Release Performance — Build #${{ github.run_number }}" >> $GITHUB_STEP_SUMMARY
    echo "" >> $GITHUB_STEP_SUMMARY
    echo "Branch: \`${{ github.ref_name }}\`  Commit: \`${GITHUB_SHA:0:8}\`" >> $GITHUB_STEP_SUMMARY
    echo "" >> $GITHUB_STEP_SUMMARY
    echo "| Operation | ns/op | Budget |" >> $GITHUB_STEP_SUMMARY
    echo "|---|---|---|" >> $GITHUB_STEP_SUMMARY
    # Parse and emit rows from the release raw log.
    grep '\[RELEASE\]' mid-math-test-release-raw.txt \
      | grep 'ns/op\|ns/entity\|µs/tick' \
      >> $GITHUB_STEP_SUMMARY || true
```

### What a compliant summary looks like

Need to optimize a hot path?
│
├─ Do you have a [RELEASE] number from cargo test --release?
│   └─ No → Get it first. Then check the Step Summary for context.
│
├─ Is it in the Priority Queue (Section 5)?
│   └─ No → It probably does not need optimizing. Check the baseline table.
│
├─ Is the bottleneck in a #[repr(C)] type layout?
│   └─ Yes → Forbidden. FFI contract is immutable.
│
├─ Can a Tier 1 approach solve it (unroll, math property)?
│   └─ Yes → Do that first. Free wins before unsafe code.
│
├─ Tier 2: Are you using core::arch intrinsics (not asm!)?
│   └─ No → Switch to intrinsics. asm! is forbidden for math.
│
├─ Does a scalar fallback exist?
│   └─ No → Write it first.
│
├─ Is gain ≥10% at 100k scale in [RELEASE] mode?
│   └─ No → Not worth the maintenance cost. Close the PR.
│
└─ Add benchmark comment, Step Summary, run --mid-all, merge.
