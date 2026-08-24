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

mid-math Release Performance — Build #20
Branch: main  Commit: abc12345  Rust: rustc 1.95.0
Operationns/opBudget checkVec3 add (200k)1.5✅Vec3 dot (100k)1.3✅Quat rotate (100k)0.9✅ within 16.6ms at 100kMat4 mul (20k)7.1✅Mat4 inverse (5k)117.1⚠️ Tier 2 candidateMat4 inverse_trs (5k)78.4⚠️ Tier 2 candidate

This format is required for any build that touches a performance-sensitive crate.

---

## 2. Build #20 Release Baseline (Official)

These are the first confirmed `[RELEASE]` numbers. They are the authoritative
scalar peak for mid-math and the required comparison point for all Tier 2 work.

| Operation | ns/op [RELEASE] | vs glam reference | Status |
|---|---|---|---|
| Vec3 add (200k ops) | 1.5 | ~1–2 ns | ✅ parity |
| Vec3 dot (100k) | 1.3 | ~1–2 ns | ✅ parity |
| Vec3 cross (100k) | 0.9 | ~1–2 ns | ✅ parity |
| Vec3 lerp (100k) | 2.8 | ~2–3 ns | ✅ parity |
| Vec3 normalize (100k) | 3.8 | ~3–5 ns | ✅ parity |
| Quat rotate (100k) | 0.9 | ~2–4 ns | ✅ beating reference |
| Quat mul (200k) | 7.2 | ~2–3 ns | ⚠️ ~3× gap |
| Quat slerp (50k) | 20.0 | ~10–15 ns | ⚠️ trig-bound |
| Euler round-trip (50k) | 104.9 | ~80–120 ns | ✅ trig-bound, expected |
| Mat4 mul (20k) | 7.1 | ~2–5 ns | ⚠️ ~2× gap |
| Mat4 transform_point (10k) | 2.4 | ~2–4 ns | ✅ parity |
| Mat4 inverse general (5k) | 117.1 | ~15–20 ns | 🔴 ~6× gap — Tier 2 target |
| Mat4 inverse_trs (5k) | 78.4 | ~15 ns | 🔴 ~5× gap — Tier 2 target |
| 100k entity transforms | 1.9 ns/entity | ~1–3 ns | ✅ parity |
| 1k frame simulation | 0.1 µs/tick | — | ✅ well within budget |

**Interpretation:**

All vector primitives have reached auto-vectorization parity with glam.
LLVM's optimizer, given `#[repr(C, align(16))]`, `#[inline(always)]`, and
explicitly unrolled loops, is correctly emitting SSE2 instructions for
contiguous float operations. No Tier 2 work is needed for Vec2/Vec3/Vec4
basic ops.

The gap is entirely in matrix inversion. The general 4×4 inverse via
Cramer's rule involves branching, a determinant check, and ~200
multiplications. glam's SIMD implementation uses shuffle-based 2×2
sub-determinants to compute this with vectorized 4-wide operations.
This is the only justified Tier 2 target.

**Note on Fused Multiply-Add (FMA):** Modern x86 CPUs since Haswell (2013)
support FMA3 (`_mm_fmadd_ps`): `a*b + c` in one instruction with one
rounding step instead of two. This is why glam's Mat4 multiply reaches
~2–3 ns — it halves instruction count for the 64 multiply-accumulate
operations per matrix multiply. The 2010 MacBook Pro dev machine does not
have FMA3 (Sandy Bridge, 2011). Any FMA path requires a runtime gate
(`is_x86_feature_detected!("fma")`) and a non-FMA fallback. This is a
Tier 2 future target once the inverse is solved.

---

## 3. The Three Tiers

### Tier 1 — Compiler-guided (default)

`#[repr(C, align(16))]`, `#[inline(always)]`, explicit loop unrolling,
arithmetic-property fast-paths (e.g. `inverse_trs`).

**Cost:** Zero maintenance. **Required proof:** None.

### Tier 2 — Intrinsics (`core::arch`)

`core::arch::x86_64::*`, `core::arch::aarch64::*`, `core::arch::wasm32::*`.

**Never use `asm!` for math.** Intrinsics are visible to LLVM for register
allocation and inlining. Inline assembly creates a black box that prevents
these optimizations.

**Cost:** Per-architecture maintenance + unsafe review + fallback.
**Required proof:** ≥10% improvement at 100k-entity scale in `[RELEASE]` build.

### Tier 3 — Inline assembly (`asm!`)

Reserved **only** for: `rdtsc`, `prefetcht0`, `pause`.
**Forbidden for all linear algebra.**

---

## 4. Platform-Specific Implementation Rules

### x86_64 (CI / desktop)

Baseline: SSE2 — available on every x86_64 CPU since 2003. Always safe,
no detection needed when gating on `#[cfg(target_arch = "x86_64")]`.

Optional: SSE4.1 (`_mm_dp_ps` for dot product, `_mm_blendv_ps` for select).
Requires: `is_x86_feature_detected!("sse4.1")` at runtime.

Optional: FMA3 (`_mm_fmadd_ps`). Requires: Haswell (2013+).
Requires: `is_x86_feature_detected!("fma")` at runtime.
**Not available on the 2010 MacBook Pro dev machine.**

Optional: AVX (`__m256`, 256-bit). Requires: Sandy Bridge (2011) or later
with `is_x86_feature_detected!("avx")`. **Not available on 2010 MBP.**
Must be gated behind `--features avx` Cargo flag AND runtime detection.

Pattern for a guarded SSE2 path:
```rust
#[cfg(target_arch = "x86_64")]
mod sse2 {
    use core::arch::x86_64::*;
    // ... implementation
}

pub fn mat4_inverse(m: &Mat4) -> Option<Mat4> {
    #[cfg(target_arch = "x86_64")]
    {
        // SSE2 is guaranteed on x86_64, no runtime check needed.
        return Some(unsafe { sse2::mat4_inverse_sse2(m) });
    }
    // Scalar fallback — always compiled, always tested.
    m.inverse_scalar()
}
```

### aarch64 (ARM / Apple Silicon / mobile)

Baseline: NEON — mandatory on all AArch64 targets. `float32x4_t`,
`vaddq_f32`, `vmulq_f32`. No runtime detection needed on aarch64.

```rust
#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;
```

### wasm32 (browser / Blazor integration)

WASM SIMD is not automatic. Requires explicit target feature flag:
`RUSTFLAGS="-C target-feature=+simd128"` or in `.cargo/config.toml`:

```toml
[target.wasm32-unknown-unknown]
rustflags = ["-C", "target-feature=+simd128"]
```

Use `core::arch::wasm32` — type is `v128`, ops are `f32x4_add` etc.

```rust
#[cfg(target_arch = "wasm32")]
use core::arch::wasm32::*;
```

**The WASM path is non-negotiable if mid-math is used in web contexts.**
Scalar math in WASM is ~5–10× slower than native and will bottleneck
any 3D web rendering.

### iOS (staticlib only)

App Store policy forbids dynamic libraries. mid-math already builds
`staticlib` for this reason. NEON is mandatory on all modern iOS devices.
All aarch64 intrinsics apply. No special handling needed beyond ensuring
`crate-type = ["rlib", "cdylib", "staticlib"]` remains in `Cargo.toml`.

---

## 5. Optimization Priority Queue (Post Build #20)

Based on the release baseline, here is the authorized work order:

**Priority 1 — Mat4 general inverse (117.1 ns → target <20 ns)**

Biggest gap. Justified by the 6× difference from glam.
Approach: SSE2 shuffle-based 2×2 sub-determinant method.
Calculate pairs of 2×2 determinants simultaneously using `_mm_shuffle_ps`.
This reduces ~200 scalar multiplications to ~50 SIMD ops.
Must have scalar fallback. Must pass `mat4_inverse_roundtrip` test.

**Priority 2 — Mat4 inverse_trs (78.4 ns → target <20 ns)**

The TRS path is currently scalar. With SSE2, the column squared-length
and dot products are 4-wide. Straightforward SIMD port of the existing
arithmetic. Scalar fallback is the current `inverse_trs()`.

**Priority 3 — Mat4 mul (7.1 ns → target <3 ns, future)**

Currently 2× from glam. Not urgent — already within any reasonable frame
budget. FMA3 would close most of this gap but requires a Haswell+ gate.
Defer until after the inverse work is complete.

**Not needed — Vec ops (already at parity)**

Vec3/Vec4 add, dot, cross, lerp, normalize are all at or beating glam
reference numbers. LLVM's auto-vectorizer is already emitting SSE2 here
due to `align(16)`. Adding manual intrinsics would provide no benefit and
would add maintenance burden. These are closed.

---

## 6. Tier 1 Work Log

### Mat4::mul — explicit unroll (Build #14+)
Replaced nested loop. Debug: 1149.9 → 49.4 ns. Release: 7.1 ns.
Maintenance: 0 min/quarter. Added: 2026-04-17.

### Mat4::inverse_trs — TRS fast-path (Build #14+)
Debug: 290.7 ns vs 707.6 ns general (same build). Release: 78.4 ns vs 117.1 ns.
Maintenance: 0 min/quarter. Added: 2026-04-17.

### Math constants — additions (Build #14+)
`FRAC_PI_3`, `FRAC_PI_4`, `FRAC_PI_6`, `FRAC_1_PI`, `SQRT_2`, `FRAC_1_SQRT_2`.
Added: 2026-04-17.

---

## 7. Mandatory Benchmark Protocol (Tier 2+)

1. Write stress test. Run `cargo test --release`. Record `[RELEASE]` baseline.
2. Implement behind `#[cfg]` gate with scalar fallback.
3. ≥10% improvement at 100k scale in `[RELEASE]` mode.
4. Add benchmark comment: numbers, date, review date (3 months out).
5. Confirm `size_of` / `align_of` pass.
6. Publish results in GitHub Step Summary per Section 1.
7. Run `--mid-all`. Zero regressions.

---

## 8. Decision Tree

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
---

## 9. Wide Integer SIMD (`wide/int/`)

Tracks `crates/mid-math/src/wide/int/` specifically — batch integer types
(`i32x4`, `u32x4`, `i16x8`, `u16x8`, `i8x16`, `u8x16` and the AVX2
additions below), not the narrow per-component vectors (`IVec4` etc.,
which stay scalar-only, matching glam's own convention — see Section 4
for why that split exists).

### Platform matrix

| Backend | Target                        | Types                                                    | Status |
|---------|-------------------------------|-----------------------------------------------------------|--------|
| SSE2    | x86 / x86_64                  | i32x4, u32x4, i16x8, u16x8, i8x16, u8x16, IMask4/8/16     | ✅ |
| NEON    | aarch64                       | i32x4, u32x4, i16x8, u16x8, i8x16, u8x16, IMask4/8/16     | ✅ fixed — was implemented but not wired into dispatch, silently falling back to scalar |
| AVX2    | x86 / x86_64 + avx2 feature   | i32x8, u32x8, i16x16, u16x16, i8x32, u8x32, IMask32x8/16x16/8x32 | ✅ additive, not a replacement — same relationship as `Vec3x8` to `Vec3x4` |
| WASM    | wasm32/64 + simd128 feature   | i32x4, u32x4, i16x8, u16x8, i8x16, u8x16                 | ⬜ not started |
| Scalar  | all others                    | i32x4, u32x4, i16x8, u16x8, i8x16, u8x16, IMask4/8/16     | ✅ |

### AVX2 native wins over SSE2

AVX2 has native `_mm256_min/max_epi32`, `_mm256_min/max_epu32`,
`_mm256_min/max_epu16`, `_mm256_mullo_epi32`, `_mm256_abs_epi8/16/32` —
none of these exist in SSE2, which emulates each via a
compare+blend or shuffle/unpack chain. The AVX2 tier is therefore
simpler code in several places, not just wider.

### Cross-lane fix (this pass)

`as_i32x8_lo`/`as_i32x8_hi` (i16x16→i32x8), `as_i16x16_lo`/`as_i16x16_hi`
(i8x32→i16x16), and their unsigned equivalents are now implemented —
turns out the permute-based lane-fixup flagged as a follow-up wasn't
needed at all. AVX2 has dedicated widen/convert instructions
(`_mm256_cvtepi16_epi32`, `_mm256_cvtepi8_epi16`, and the `epu` unsigned
variants) that aren't shuffles in the first place, so the per-128-bit-lane
hazard that affects `unpacklo`/`unpackhi`/`shuffle_epi8` doesn't apply to
them — extract each 128-bit half (`_mm256_castsi256_si128` /
`_mm256_extracti128_si256`) and convert it directly.

`shuffle_bytes` (i8x32/u8x32) uses `_mm256_shuffle_epi8` directly. That
one genuinely IS per-128-bit-lane — it's a real shuffle, not a widen —
so it's documented on the method itself with the real (correct, just
scoped) semantics rather than faked into looking like a flat 32-byte
shuffle. `to_i8x16_pair`/`from_i8x16_pair` (and the `u8x16` equivalents)
are provided for splitting into two independently-shuffleable SSE2
halves if a true cross-half shuffle is ever needed.

### Bench comparison targets

Investigated whether nalgebra/ultraviolet have anything comparable to
bench wide-int against, alongside glam and the `wide` crate (new dev-dep
this pass, version 1.6):
- **ultraviolet**: checked its published `int.rs` directly — only narrow
  `IVec2/3/4`/`UVec2/3/4`, same category as glam. Its `x4`/`x8` wide
  concept is float-only (`Vec3x4`/`Vec3x8` rotor batch types).
- **nalgebra**: SIMD goes through `simba`. `simba`'s `wide`-crate
  integration (`WideF32x4` etc.) is float-only. Integer SIMD in `simba`
  exists only behind Rust's nightly `#![feature(portable_simd)]` —
  incompatible with this project's stable-only toolchain.
- **`wide` crate itself**: the actual apples-to-apples comparison —
  same concept as mid-math's wide/int (N packed scalar lanes), covers
  every width mid-math has plus i64/u64 lanes mid-math doesn't have yet.
  Added as a dev-dependency; `vs_wide_int.rs` now benches against it
  alongside glam for ops verified against its published source (`add`,
  `sub`, `mul`, `min`, `max`, `abs`, `saturating_add`, `saturating_sub`)
  — its shift and equality-comparison calling conventions weren't
  verified confidently enough to bench against blind.

### Bench coverage

`crates/mid-math/benches/vs_wide_int.rs` now covers all 12 wide-int
types: the SSE2/NEON/scalar 6-type family (i32x4/u32x4/i16x8/u16x8/
i8x16/u8x16, vs glam + `wide`) plus the AVX2 6-type family (i32x8/
u32x8/i16x16/u16x16/i8x32/u8x32, vs `wide` only — no glam equivalent
exists at these widths), plus two bulk ECS-shaped workloads
(`bulk_entity_id_scan`, `bulk_u8_sum`). The AVX2 bench functions are
`#[cfg(target_feature = "avx2")]`-gated and called through a small
always-present dispatch wrapper so `criterion_group!` doesn't need a
conditionally-assembled function list.

`.github/workflows/bench-vs-wide-int.yml` has the same `target_cpu`
platform-firing dropdown as `A-mid-math — bench vs all`
(native/x86-64-v4/x86-64-v3/x86-64-v2/x86-64/neon/wasm/scalar) plus a
`wide_type` dropdown covering all 12 types (mapped to a criterion name
filter). Selecting an AVX2 type with a non-AVX2 target_cpu just matches
nothing (criterion doesn't error on an empty filter match) — documented
in the dropdown description rather than added as extra yml logic.
`wasm` currently exercises the scalar fallback cross-compiled to
wasm32 — no WASM SIMD128 backend exists for wide/int yet (see above).

### `wide`-crate comparison numbers were misleading — missing `[profile.bench]`

Build #5's `wide-i32x4/add` benched ~14ns — 13x slower than our own
`i32x4/add`, and even slower than glam's scalar `IVec4/add`. That's not
a real property of the `wide` crate. Traced it to its actual source:

`wide::i32x4::add` doesn't call a raw intrinsic directly — it does
`cast::<i32x4,u32x4>(self) + cast::<i32x4,u32x4>(rhs)` then casts back
(wrapping add is identical for signed/unsigned, so it shares the impl),
which means it delegates into **`u32x4`'s own `Add` impl** as a second
hop. Compare to `min`/`max`, which call `pick! { if #[cfg(...)] { ...
raw intrinsic... } }` directly, one hop, no delegation. Both are marked
`#[inline]` (not `#[inline(always)]`), so collapsing that longer add/sub
chain down to a single instruction depends on the compiler's cross-crate
inlining being aggressive enough to see through it.

`cargo bench` does **not** use `[profile.release]` — it has its own
default profile with `opt-level=3` but no `lto`/`codegen-units=1`. Our
own mid-math methods are all `#[inline(always)]`, which survives
regardless of LTO; a 2-hop cross-crate `#[inline]` chain into another
crate is exactly the case that needs LTO to reliably collapse. Added a
`[profile.bench]` section to the workspace root `Cargo.toml` mirroring
`[profile.release]`'s `opt-level`/`lto`/`codegen-units` (keeping
`debug = true` instead of `release`'s `false` — useful for profiling
bench binaries with `perf`, costs nothing at runtime).

**Not yet confirmed this actually fixes it** — needs a re-run to see if
`wide-i32x4/add` drops into the same range as `min`/`max` (~2-7ns) once
the profile change is applied. If it doesn't move, the hypothesis above
is wrong and it's something else — worth checking before trusting any
absolute `wide-*` timing, though the *relative* ranking (our types
consistently fastest, glam's scalar-storage types consistently slowest)
held across both this run and the previous one, so that comparison is
probably still meaningful even if the exact `wide` numbers move on
the next run.
