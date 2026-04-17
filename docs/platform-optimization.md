<!-- docs/platform-optimization.md -->

# Mid Engine — Platform Optimization Rules

> **Rule 0:** The compiler is smarter than you think. Profile before you optimize.
> **Rule 1:** An optimization that cannot be benchmarked does not exist.
> **Rule 2:** An optimization that breaks FFI is a bug, not a feature.
> **Rule 3:** Debug-mode timing is not a benchmark. All optimization decisions
>             require `cargo test --release` or `cargo bench` numbers labeled [RELEASE].

---

## 0. Build Mode Discipline

`cargo test` without `--release` runs with `opt-level = 0`. LLVM performs
no inlining, no auto-vectorization, and no loop unrolling. These numbers
are between 5× and 50× slower than release for math-heavy code.

The label `[RELEASE]` or `[DEBUG]` in every stress test output comes from
`cfg!(debug_assertions)` resolved at compile time. If you see `[DEBUG]` on
a run you believed was release, check whether `cargo test --release` was
actually invoked.

The CI produces two passes per build:
- **Debug pass** → HTML dashboard (correctness: pass/fail counts only).
- **Release pass** → Job summary + artifact `release-perf.txt` (performance numbers).

Any claim of "X is Yns/op" must cite the build number and `[RELEASE]` label.

---

## 1. Build #19 Baseline (Debug)

These are confirmed debug numbers. They are not performance targets —
they are regression anchors. If a future debug run is significantly
slower, that indicates a correctness or structure problem.

| Operation | Build #13 debug | Build #19 debug | Change | Note |
|---|---|---|---|---|
| Vec3 Add (200k) | 15.9 ns/op | 15.4 ns/op | ~same | scalar, expected |
| Vec3 Dot (100k) | 14.5 ns/op | 20.8 ns/op | slower | CI machine variance |
| Quat Mul (200k) | 20.4 ns/op | 20.2 ns/op | ~same | scalar |
| Mat4 Mul (20k) | 1149.9 ns/op | 49.4 ns/op | **23× faster** | explicit unroll (Tier 1) |
| Mat4 Inverse general (5k) | 2751.6 ns/op | 707.6 ns/op | ~4× faster | unroll benefit |
| Mat4 inverse_trs (5k) | — | 290.7 ns/op | **2.4× vs general** | new Tier 1 fast-path |
| 100k entity transforms | 42.3 ns/entity | 44.3 ns/entity | ~same | scalar, within budget |

**Key finding:** The explicit Mat4 unroll (Tier 1) produced a 23× debug speedup.
This tells us LLVM was struggling with loop analysis in debug mode. In release,
the loop and unrolled versions should converge. The first `[RELEASE]` numbers
from Build #20+ will confirm this.

**Verdict on Tier 2:** Not authorized yet. We need `[RELEASE]` numbers. The
rule is ≥10% improvement at 100k scale in release mode — we don't have release
numbers yet.

---

## 2. The Three Tiers

### Tier 1 — Compiler-guided (default)

Use `#[repr(C, align(16))]`, `#[inline(always)]`, explicit loop unrolling,
and arithmetic-property fast-paths. No intrinsics, no unsafe math code.

**Cost:** Zero maintenance. **Required proof:** None.

### Tier 2 — Intrinsics (`core::arch`)

Hand-written SIMD using `core::arch::x86_64::*` or `core::arch::aarch64::*`.
**Use `core::arch` intrinsics. Never use `asm!` for math.** Intrinsics are
visible to LLVM for register allocation and inlining. Inline assembly is a
black box that prevents these optimizations.

**Cost:** Per-architecture maintenance + unsafe review + fallback path.
**Required proof:** ≥10% improvement at 100k-entity scale in `[RELEASE]` build.

### Tier 3 — Inline assembly (`asm!`)

Reserved **only** for: `rdtsc`, `prefetcht0`, `pause`.
**Forbidden for all linear algebra operations.**

---

## 3. When Optimization Is Forbidden

### 3.1 FFI contract — immutable layout

| Type | Size | Alignment |
|------|------|-----------|
| Vec2 | 8 B  | 4         |
| Vec3 | 16 B | 16        |
| Vec4 | 16 B | 16        |
| Quat | 16 B | 16        |
| Mat3 | 36 B | 4         |
| Mat4 | 64 B | 16        |

Verified by `size_of` / `align_of` assertions in tests. Do not remove them.

### 3.2 No fallback path

Every Tier 2 function must have a working scalar fallback:

```rust
pub fn dot(a: Vec3, b: Vec3) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("sse4.1") {
        return unsafe { dot_sse41(a, b) };
    }
    // Scalar fallback — always compiled, always tested.
    a.x*b.x + a.y*b.y + a.z*b.z
}
```

### 3.3 AVX policy

The 2010 MacBook Pro dev machine does not support AVX. Any AVX path must
be gated behind `--features avx` and must not be required for CI to pass.

---

## 4. Tier 1 Work Log

### Mat4::mul — explicit unroll

Replaced 3-level nested loop with 64 explicit multiply-accumulate terms.
Debug result: 1149.9 → 49.4 ns/op (23×). Release numbers pending.
Maintenance estimate: 0 min/quarter. Added: 2026-04-17.

### Mat4::inverse_trs — TRS arithmetic fast-path

For M = T·R·S, exploits: R is orthogonal (Rᵀ = R⁻¹), S is diagonal.
Inverse = diag(1/sx²,1/sy²,1/sz²) · Rᵀ, with translation = −(that) · t.
Debug result: 290.7 ns/op vs 707.6 ns general in same build (2.4×).
Cross-build comparison (Build #13 general): 9.5× — misleading, different
builds, different loop implementations. Use same-build ratio (2.4×).
Maintenance estimate: 0 min/quarter. Added: 2026-04-17.

### Math constants — additions

Added: `FRAC_PI_3`, `FRAC_PI_4`, `FRAC_PI_6`, `FRAC_1_PI`, `SQRT_2`,
`FRAC_1_SQRT_2`. Used in lighting (Lambert: `FRAC_1_PI`), hexagonal
grids, and normalized diagonal math. Zero runtime cost.
Added: 2026-04-17.

---

## 5. Next Optimization Decision Point

After Build #20 CI runs with the fixed label:

1. Record `[RELEASE]` numbers for Mat4 mul, general inverse, inverse_trs,
   Vec3 ops, Quat rotate.
2. Update the baseline table in this document.
3. If Mat4 mul `[RELEASE]` is still above ~10 ns/op, that triggers Tier 2
   evaluation for SSE2 intrinsics per the mandatory benchmark protocol.
4. If Vec3 add `[RELEASE]` is below ~2 ns/op, auto-vectorization is working
   and Tier 2 for vector ops is not warranted.

---

## 6. Mandatory Benchmark Protocol (Tier 2+)

1. Write stress test first. Run `cargo test --release`. Record `[RELEASE]` baseline.
2. Implement optimization behind `#[cfg]` gate.
3. Run same stress test. Must show ≥10% improvement at 100k scale.
4. Add benchmark comment above function: numbers, date, review date.
5. Confirm `size_of` / `align_of` assertions pass.
6. Run `--mid-all`. Zero regressions.

---

## 7. Decision Tree

```
Need to optimize a hot path?
│
├─ Do you have a [RELEASE] number from cargo test --release?
│   └─ No → Get it. Debug numbers are not actionable.
│
├─ Is the bottleneck in a #[repr(C)] type's layout?
│   └─ Yes → Forbidden. The FFI contract is immutable.
│
├─ Can explicit unrolling or a math property (like inverse_trs) help?
│   └─ Yes → Tier 1 first. Free wins, zero maintenance cost.
│
├─ Does the compiler already auto-vectorize at -O3 with align(16)?
│   └─ Check: if [RELEASE] Vec3 add < 3 ns/op, auto-vectorization works.
│
├─ Is the gain ≥10% at 100k scale in [RELEASE] mode?
│   └─ No → Not worth the Tier 2 cost.
│
├─ Are you using core::arch intrinsics (not asm!)?
│   └─ No → Switch. asm! is forbidden for math.
│
├─ Does a scalar fallback exist?
│   └─ No → Write it first.
│
└─ Add benchmark comment, run --mid-all, merge.
```
