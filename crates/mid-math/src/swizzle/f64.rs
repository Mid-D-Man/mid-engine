// crates/mid-math/src/swizzle/f64.rs
//! Swizzle impls for the f64 vector family.
//!
//! Mirror image of `f32.rs`'s split, but with the canonical/backend-split
//! roles swapped: `DVec3` is canonical here (one impl), `DVec2`/`DVec4` are
//! backend-split. There's no f64 `coresimd` entry — `f64/mod.rs` only uses
//! `coresimd` internally for an alternate, non-public `DVec3`, never for a
//! public `DVec2`/`DVec4`, so there's nothing on that path for this file to
//! reach. Same rule as `f32.rs` for the `#[cfg(...)]` on each backend impl:
//! copied from that backend module's own declaration in `f64/mod.rs`, not
//! from the alias re-export's extra `not(force-scalar)` clause.

// ── DVec3 (canonical) ────────────────────────────────────────────────────────
crate::impl_vec3_swizzle!(crate::f64::DVec3, crate::f64::DVec2, crate::f64::DVec4);

// ── x86 / x86_64 (SSE2) ──────────────────────────────────────────────────────
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
crate::impl_vec2_swizzle!(crate::f64::sse2::DVec2, crate::f64::DVec3, crate::f64::sse2::DVec4);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
crate::impl_vec4_swizzle!(crate::f64::sse2::DVec4, crate::f64::sse2::DVec2, crate::f64::DVec3);

// ── aarch64 (NEON) ───────────────────────────────────────────────────────────
#[cfg(target_arch = "aarch64")]
crate::impl_vec2_swizzle!(crate::f64::neon::DVec2, crate::f64::DVec3, crate::f64::neon::DVec4);
#[cfg(target_arch = "aarch64")]
crate::impl_vec4_swizzle!(crate::f64::neon::DVec4, crate::f64::neon::DVec2, crate::f64::DVec3);

// ── wasm32 / wasm64 + simd128 ────────────────────────────────────────────────
#[cfg(all(any(target_arch = "wasm32", target_arch = "wasm64"), target_feature = "simd128"))]
crate::impl_vec2_swizzle!(crate::f64::wasm::DVec2, crate::f64::DVec3, crate::f64::wasm::DVec4);
#[cfg(all(any(target_arch = "wasm32", target_arch = "wasm64"), target_feature = "simd128"))]
crate::impl_vec4_swizzle!(crate::f64::wasm::DVec4, crate::f64::wasm::DVec2, crate::f64::DVec3);

// ── Scalar fallback (unconditional — `f64/mod.rs` declares `dvec2`/`dvec4`
//    with no cfg gate; these double as both the always-available scalar
//    backend AND the home of `DEPSILON`) ─────────────────────────────────────
crate::impl_vec2_swizzle!(crate::f64::dvec2::DVec2, crate::f64::DVec3, crate::f64::dvec4::DVec4);
crate::impl_vec4_swizzle!(crate::f64::dvec4::DVec4, crate::f64::dvec2::DVec2, crate::f64::DVec3);
