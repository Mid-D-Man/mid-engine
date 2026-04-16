<!-- docs/platform-optimization.md -->

# Mid Engine — Platform Optimization Rules

> **Rule 0:** The compiler is smarter than you think. Profile before you optimize.  
> **Rule 1:** An optimization that cannot be benchmarked does not exist.  
> **Rule 2:** An optimization that breaks FFI is a bug, not a feature.

---

## 1. The Three Tiers

Mid Engine uses a strict hierarchy. Each tier requires a higher bar of justification
to enter the codebase and carries a higher maintenance cost.

### Tier 1 — Compiler-guided (default)

Use `#[repr(C, align(16))]` and let the compiler auto-vectorize. This is the
default for all math types. The compiler with SSE2 enabled will emit equivalent
instructions to hand-written intrinsics for simple arithmetic (add, dot, lerp).

**Cost:** Zero maintenance. **Benefit:** Immediate. **Required proof:** None.

### Tier 2 — Intrinsics (`std::arch`)

Hand-written SIMD using `std::arch::x86_64::*` or `std::arch::aarch64::*`.
Required only when profiling proves the compiler is leaving measurable performance
on the table (>10% wall-clock gap between the auto-vectorized and hand-written paths
on the same inputs).

**Cost:** Per-architecture maintenance, unsafe code review, and a fallback path.  
**Benefit:** Required to be measurable in a benchmark under the CI stress tests.  
**Required proof:** A benchmark showing ≥10% improvement at the relevant data scale
(e.g. 100k entity operations, not 10).

### Tier 3 — Inline assembly (`asm!`)

Raw CPU instructions via the stable `asm!` macro. Reserved for operations the
compiler cannot express at all:

- `rdtsc` — CPU cycle counter for micro-benchmarks inside the engine itself.
- `prefetcht0` — Cache prefetch ahead of bulk SoA iteration in mid-ecs.
- `pause` — Spin-wait hint in any future lock-free synchronization primitives.

**Cost:** Architecture-locked, unsafe, zero compiler assistance.  
**Benefit:** Must be documented with a specific instruction-count or latency measurement.  
**Required proof:** Both a before/after benchmark AND an explanation of why the
compiler cannot generate the equivalent instruction sequence automatically.

---

## 2. When Optimization Is Forbidden

These situations block any Tier 2 or Tier 3 optimization regardless of the speedup:

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

If an optimization would change any of the above values, it is rejected
unconditionally. Unity, Unreal, Godot, and any C host depend on these numbers
being stable forever. Run `size_of` and `align_of` assertions in the test suite
for every type (they are already there — do not remove them).

### 2.2 No fallback path

Tier 2 and Tier 3 code must always have a scalar fallback. The pattern is:

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

A Tier 2 or Tier 3 function with no fallback is a build error, not a warning.

### 2.3 Maintenance cost exceeds benefit

If maintaining an optimized path requires more than one hour of engineering
time per calendar quarter (updating for new Rust versions, fixing LLVM regressions,
handling new target triples), the optimized path is removed and replaced with the
scalar fallback until the cost/benefit ratio improves.

Track this in a comment above the function:

```rust
// SIMD path — maintenance estimate: 15 min/quarter.
// Benchmark: 100k dot products: scalar=2.1ms, sse2=0.9ms (-57%).
// Added: 2026-04-16. Review: 2026-07-16.
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

AVX requires runtime detection (`is_x86_feature_detected!("avx")`). The 2010 MacBook
Pro does not have AVX. AVX optimizations are permitted only if:

1. The fallback path (SSE2 or scalar) is present and tested.
2. The AVX path is gated behind a `--features avx` Cargo feature flag.
3. CI continues to pass on `ubuntu-latest` without the feature flag.

### ARM / iOS

All iOS builds use `staticlib` only (App Store rule). Tier 2 ARM NEON paths are
permitted under the same rules as x86_64 SSE2 but require a separate benchmark
on an actual ARM device before merging.

---

## 4. The Mandatory Benchmark Protocol

Before any Tier 2 or Tier 3 optimization can be merged:

**Step 1:** Write the stress test first (before the optimization). Run it. Record
the baseline timing printed by the test runner in the HTML report.

**Step 2:** Implement the optimization behind a `#[cfg]` gate.

**Step 3:** Run the same stress test again. The optimized path must show ≥10%
improvement at the scale relevant to the engine's targets:
- Math operations: 100k iterations (ECS entity count).
- Network: 128 iterations per second (tick rate).
- Logging: 1000 entries per tick.

**Step 4:** Add a comment above the function with the benchmark numbers and date.

**Step 5:** Confirm `size_of` and `align_of` assertions still pass.

**Step 6:** Run the full test suite (`--mid-all`). Zero regressions.

---

## 5. `asm!` Safety Checklist

Every use of `asm!` must pass this checklist before merging:

- [ ] The block is inside `unsafe {}`.
- [ ] Input and output registers are declared explicitly (no implicit clobbers).
- [ ] The `options(nostack)` flag is set if the instruction does not touch the stack.
- [ ] The fallback scalar path exists and is tested separately.
- [ ] The `#[cfg(target_arch = "x86_64")]` guard prevents compilation on other architectures.
- [ ] A comment explains what the instruction does and why the compiler cannot generate it.
- [ ] A benchmark confirms the instruction produces measurable benefit.

Example of a correctly annotated `asm!` block:

```rust
/// Returns the CPU timestamp counter. Used only inside benchmarks — never in
/// production game-loop code. The compiler cannot generate `rdtsc` without
/// this because `std::time::Instant` has ~100 ns overhead vs ~1 ns for rdtsc.
#[cfg(target_arch = "x86_64")]
pub fn cpu_cycles() -> u64 {
    let lo: u32;
    let hi: u32;
    // SAFETY: rdtsc is available on all x86_64 CPUs.
    // options(nostack, nomem): does not touch memory or the stack.
    unsafe {
        std::arch::asm!(
            "rdtsc",
            out("eax") lo,
            out("edx") hi,
            options(nostack, nomem, pure),
        );
    }
    ((hi as u64) << 32) | (lo as u64)
}
```

---

## 6. Decision Tree

```
Need to optimize a hot path?
│
├─ Have you profiled it?
│   └─ No → Profile first. Use the stress tests in the HTML report.
│
├─ Is the bottleneck in a #[repr(C)] type's layout?
│   └─ Yes → Forbidden. The FFI contract is immutable.
│
├─ Does the compiler already auto-vectorize with align(16)?
│   └─ Yes → Stop. The compiler wins here.
│
├─ Is the gain ≥10% at 100k-entity scale?
│   └─ No → Not worth the maintenance cost.
│
├─ Does a scalar fallback exist?
│   └─ No → Write the fallback first.
│
└─ Add the benchmark comment, run --mid-all, merge.
```
