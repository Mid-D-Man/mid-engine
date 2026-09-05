// crates/mid-math/src/wide/float/mod.rs
//! Float wide type platform dispatch.
//!
//! ## Platform matrix
//!
//! | Backend | Target                        | Types                      |
//! |---------|-------------------------------|----------------------------|
//! | SSE2    | x86 / x86_64                  | f32x4, Mask4, Vec3x4, QuatX4 |
//! | AVX2    | x86 / x86_64                  | f32x8, Mask8, Vec3x8 (additional) |
//! | NEON    | aarch64                       | f32x4, Mask4, Vec3x4, QuatX4 |
//! | WASM    | wasm32/64 + simd128 feature   | f32x4, Mask4, Vec3x4, QuatX4 |
//! | Scalar  | all others                    | f32x4, Mask4, Vec3x4, QuatX4 |
//!
//! ## Vec3x8 availability
//!
//! Vec3x8, f32x8, and Mask8 are x86-only — there is no NEON or WASM
//! equivalent worth adding (NEON's vld4q_f32 only gives a 4-wide transpose;
//! WASM SIMD128 has no 256-bit registers). All three are now always
//! compiled on x86/x86_64 rather than gated behind the crate's own `avx2`
//! target-feature baseline: storage is two portable width-4 halves, and
//! each arithmetic method checks `crate::wide::avx2_available()` at
//! runtime, calling a `#[target_feature(enable = "avx2")]`-gated fast path
//! when true and the portable halves' own methods otherwise — see
//! `wide/int/avx2/i32x8.rs`'s doc comment for the full reasoning (this was
//! previously a compile-time gate on the whole module, which meant a C
//! caller linking a non-AVX2 build got a link error referencing these
//! types at all, not just a slower runtime path).
//!
//! ## WASM (next)
//!
//! WASM f32x4_* ops mirror SSE2 conceptually. No vld4q equivalent,
//! so AoS→SoA transpose uses the same 7-shuffle approach as SSE2.
//! No baseline FMA (relaxed-simd adds it). Implementation lives in
//! `wide/float/wasm/` (see next commit).

// ── Scalar fallback — always compiled ────────────────────────────────────────
pub(crate) mod scalar;

// ── SSE2 — x86 / x86_64 ──────────────────────────────────────────────────────

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub(crate) mod sse2;

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), not(feature = "force-scalar")))]
pub use sse2::{Mask4, Mask4LaneIter, Vec3x4, QuatX4};

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), not(feature = "force-scalar")))]
#[allow(non_camel_case_types)]
pub use sse2::f32x4::f32x4;

// ── AVX2 — x86 / x86_64 + avx2 ───────────────────────────────────────────────

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub(crate) mod avx2;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub use avx2::{Vec3x8, Mask8};

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(non_camel_case_types)]
pub use avx2::f32x8::f32x8;

// ── NEON — aarch64 ────────────────────────────────────────────────────────────
//
// float32x4_t is mandatory on all AArch64 targets — no runtime check needed.
// vld4q_f32/vst4q_f32 give 1-instruction AoS↔SoA transpose.
// FMA (vfmaq_f32, vfmsq_f32) is mandatory — no target_feature gate required.

#[cfg(target_arch = "aarch64")]
pub(crate) mod neon;

#[cfg(all(target_arch = "aarch64", not(feature = "force-scalar")))]
pub use neon::{Mask4, Mask4LaneIter, Vec3x4, QuatX4};

#[cfg(all(target_arch = "aarch64", not(feature = "force-scalar")))]
#[allow(non_camel_case_types)]
pub use neon::f32x4::f32x4;

// ── WASM SIMD128 ──────────────────────────────────────────────────────────────
//
// Implemented in wide/float/wasm/ — see next commit.
// Build with: RUSTFLAGS="-C target-feature=+simd128"

#[cfg(all(
    any(target_arch = "wasm32", target_arch = "wasm64"),
    target_feature = "simd128",
))]
pub(crate) mod wasm;

#[cfg(all(
    any(target_arch = "wasm32", target_arch = "wasm64"),
    target_feature = "simd128",
    not(feature = "force-scalar"),
))]
pub use wasm::{Mask4, Mask4LaneIter, Vec3x4, QuatX4};

#[cfg(all(
    any(target_arch = "wasm32", target_arch = "wasm64"),
    target_feature = "simd128",
    not(feature = "force-scalar"),
))]
#[allow(non_camel_case_types)]
pub use wasm::f32x4::f32x4;

// ── Scalar fallback ───────────────────────────────────────────────────────────
//
// Active when no SIMD backend applies.

#[cfg(any(
    feature = "force-scalar",
    not(any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "aarch64",
        all(
            any(target_arch = "wasm32", target_arch = "wasm64"),
            target_feature = "simd128",
        ),
    )),
))]
pub use scalar::{Mask4, Mask4LaneIter, Vec3x4, QuatX4};

#[cfg(any(
    feature = "force-scalar",
    not(any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "aarch64",
        all(
            any(target_arch = "wasm32", target_arch = "wasm64"),
            target_feature = "simd128",
        ),
    )),
))]
#[allow(non_camel_case_types)]
pub use scalar::f32x4::f32x4;
