<!-- docs/platform-optimization.md -->

# Mid Engine — Platform Optimization Rules

> **Rule 0:** The compiler is smarter than you think. Profile before you optimize.
> **Rule 1:** An optimization that cannot be benchmarked does not exist.
> **Rule 2:** An optimization that breaks FFI is a bug, not a feature.
> **Rule 3:** Debug-mode timing is not a benchmark. All optimization decisions
>             require `cargo test --release` or `cargo bench` numbers.

---

## 0. Build Mode Discipline

`cargo test` runs in **debug mode** (opt-level = 0, no inlining, no auto-vectorization).
Debug numbers are useful for catching logic regressions. They are **not** comparable
to any release-mode library (glam, nalgebra, etc.).

The CI runs two test passes:
- `cargo test -p mid-math` — debug, confirms correctness, timing ignored for perf decisions.
- `cargo test -p mid-math --release` — release, provides the actual performance baseline.

Any claim of "X is Yns/op" must cite which build mode produced it.

---

## 1. The Three Tiers

Mid Engine uses a strict hierarchy. Each tier requires a higher bar of justification
to enter the codebase and carries a higher maintenance cost.

### Tier 1 — Compiler-guided (default)

Use `#[repr(C, align(16))]`, `#[inline(always)]`, and explicit loop unrolling.
Let the compiler auto-vectorize. This is the default for all math types.

With SSE2 enabled (the x86_64 baseline since 2003), the compiler with `-O3`
will emit equivalent instructions to hand-written intrinsics for simple
arithmetic (add, dot, lerp, unrolled mat-mul).

**Cost:** Zero maintenance. **Benefit:** Immediate. **Required proof:** None.

Examples of Tier 1 work:
- Replacing a loop-based `Mat4::mul` with an explicitly unrolled sum-of-products.
- Adding a `Mat4::inverse_trs()` fast-path using arithmetic properties of TRS matrices.

### Tier 2 — Intrinsics (`std::arch`)

Hand-written SIMD using `core::arch::x86_64::*` or `core::arch::aarch64::*`.
Required only when profiling proves the compiler is leaving measurable performance
on the table (>10% wall-clock gap between the auto-vectorized and hand-written paths
on the same inputs).

**Cost:** Per-architecture maintenance, unsafe code review, and a fallback path.
**Benefit:** Required to be measurable in a benchmark under the CI stress tests.
**Required proof:** A benchmark showing ≥10% improvement at the relevant data scale
(e.g. 100k entity operations, not 10).

**Important:** Use `core::arch` intrinsics, NOT `asm!`. Intrinsics are visible to
LLVM for register allocation and inlining. Inline assembly is a black box that
prevents these optimizations. See Section 3 for the asm! policy.

### Tier 3 — Inline assembly (`asm!`)

Raw CPU instructions via the stable `asm!` macro. Reserved **only** for:

- `rdtsc` — CPU cycle counter for micro-benchmarks inside the engine itself.
- `prefetcht0` — Cache prefetch ahead of bulk SoA iteration in mid-ecs.
- `pause` — Spin-wait hint in any future lock-free synchronization primitives.

**`asm!` is explicitly forbidden for linear algebra.** The correct tool for
math acceleration is `core::arch` intrinsics (Tier 2), not raw assembly.

**Cost:** Architecture-locked, unsafe, zero compiler assistance.
**Benefit:** Must be documented with a specific instruction-count or latency measurement.
**Required proof:** Both a before/after benchmark AND an explanation of why the
compiler cannot generate the equivalent instruction sequence automatically.

---

## 2. When Optimization Is Forbidden

### 2.1 FFI contract violation

Any optimization that changes the memory layout of a `#[repr(C)]` type is
**forbidden**. The following are invariants enforced across the FFI boundary:

| Type | Size | Alignment | First field |
|------|------|-----------|-------------|
| Vec2 | 8 B  | 4         | x: f32      |
| Vec3 | 16 B | 16        | x: f32      |
| Vec4 | 16 B | 16        | x: f32      |
| Quat | 16 B | 16        | x: f32      |
| Mat3 | 36 B | 4         | cols[0][0]  |
| Mat4 | 64 B | 16        | cols[0][0]  |

All verified by `size_of` and `align_of` assertions in the test suite.
Do not remove those assertions.

### 2.2 No fallback path

Tier 2 and Tier 3 code must always have a scalar fallback:

```rust
pub fn operation(a: Vec3, b: Vec3) -> Vec3 {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("sse2") {
        return unsafe { operation_sse2(a, b) };
    }
    // Scalar fallback — always present, always correct.
    Vec3::new(a.x + b.x, a.y + b.y, a.z + b.z)
}
```

### 2.3 Maintenance cost exceeds benefit

Track maintenance cost above the function:

```rust
// SIMD path — maintenance estimate: 15 min/quarter.
// Benchmark: 100k dot products: scalar=2.1ms, sse2=0.9ms (-57%).
// Added: 2026-04-17. Review: 2026-07-17.
```

---

## 3. Architecture Targets

### Supported (CI-verified)

| Architecture | SIMD baseline | Tier 2 instruction set |
|---|---|---|
| x86_64 (CI, Linux) | SSE2 (mandatory since 2003) | SSE4.1 optional |
| x86_64 (dev, 2010 Mac) | SSE4.1 (Core i5/i7) | AVX blocked (no AVX on 2010 MBP) |
| aarch64 (ARM) | NEON | NEON |

### AVX / AVX-512 policy

The 2010 MacBook Pro does not have AVX. AVX optimizations require:
1. A fallback path (SSE2 or scalar) is present and tested.
2. Gated behind `--features avx` Cargo feature flag.
3. CI passes without the feature flag.

---

## 4. The Mandatory Benchmark Protocol

Before any Tier 2 or Tier 3 optimization can be merged:

**Step 1:** Write the stress test first. Run `cargo test --release`. Record baseline.

**Step 2:** Implement the optimization behind a `#[cfg]` gate.

**Step 3:** Run the same stress test. Improvement must be ≥10% at 100k-entity scale.

**Step 4:** Add the benchmark comment above the function.

**Step 5:** Confirm `size_of` and `align_of` assertions still pass.

**Step 6:** Run the full test suite (`--mid-all`). Zero regressions.

---

## 5. Tier 1 Work Log

These are pure-scalar, zero-intrinsic improvements applied to the codebase.
No benchmark proof required per the rules (Tier 1 is free).

### Mat4::mul — explicit unroll (Build #14+)

Replaced the 3-level nested loop with 64 explicit multiply-accumulate terms.

In release mode, LLVM produces identical assembly either way. In debug mode,
the explicit form avoids loop overhead and makes auto-vectorization intent clear.
No maintenance cost. Zero risk to FFI layout.

### Mat4::inverse_trs — TRS fast-path (Build #14+)

Added `inverse_trs()` for pure Translation × Rotation × Scale matrices.

**Correctness basis:** For M = T·R·S, the inverse M⁻¹ = S⁻¹·Rᵀ·T⁻¹.
This exploits that R is orthogonal (Rᵀ = R⁻¹) and S is diagonal (S⁻¹ = diag(1/sx, 1/sy, 1/sz)).
The result is ~30 multiplications vs ~200 for the general Cramer's rule path.

**Debug baseline (Build #13):** 2751.6 ns/op (general inverse, 5k mats).
**Release baseline:** TBD after Build #14 CI run. See `stress_5k_mat4_inverse_trs`.
**Maintenance estimate:** 0 min/quarter. Added: 2026-04-17. Review: 2026-07-17.

Usage in engine code:
```rust
// Entity world transform inverse — always TRS, use the fast path.
let inv = entity.world_transform.inverse_trs();

// Projection matrix inverse — general case, may contain shear.
let inv = projection.inverse().unwrap_or(Mat4::IDENTITY);
```

### Math constants — additions (Build #14+)

Added: `FRAC_PI_3`, `FRAC_PI_4`, `FRAC_PI_6`, `FRAC_1_PI`, `SQRT_2`, `FRAC_1_SQRT_2`.

These appear in lighting equations (`FRAC_1_PI` for Lambert normalisation),
hexagonal grid math (`FRAC_PI_3`, `FRAC_PI_6`), and normalized diagonal
computations (`FRAC_1_SQRT_2`). All are free `f32` constants — zero cost.

---

## 6. Next Optimization Decision Point

After Build #14 CI runs, compare `stress_5k_mat4_inverse_trs` release numbers
against the Build #13 general inverse baseline (2751.6 ns debug / TBD release).

If after release mode the general `Mat4::mul` is still above ~10 ns/op,
that is the target for a Tier 2 SSE2 implementation. The mandatory protocol
in Section 4 applies.

---

## 7. Decision Tree

```
Need to optimize a hot path?
│
├─ Have you run cargo test --release and recorded the number?
│   └─ No → Get the release-mode number first. Debug numbers are meaningless.
│
├─ Is the bottleneck in a #[repr(C)] type's layout?
│   └─ Yes → Forbidden. The FFI contract is immutable.
│
├─ Have you tried explicit unrolling or inverse_trs? (Tier 1)
│   └─ Not yet → Try Tier 1 first. Free wins come before unsafe code.
│
├─ Does the compiler already auto-vectorize with align(16)?
│   └─ Check with `cargo asm` or LLVM MCA → If yes, stop.
│
├─ Is the gain ≥10% at 100k-entity scale in release mode?
│   └─ No → Not worth the maintenance cost.
│
├─ Does a scalar fallback exist?
│   └─ No → Write the fallback first.
│
├─ Are you using core::arch intrinsics (not asm!)?
│   └─ No → Switch to intrinsics. asm! is forbidden for math.
│
└─ Add the benchmark comment, run --mid-all, merge.
```
