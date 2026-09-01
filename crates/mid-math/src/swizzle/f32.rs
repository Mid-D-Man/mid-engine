// crates/mid-math/src/swizzle/f32.rs
//! Swizzle impls for the f32 vector family.
//!
//! `Vec2` is canonical (one impl). `Vec3`/`Vec4` are backend-split, so each
//! gets one impl per backend, and each `#[cfg(...)]` here is copied verbatim
//! from that backend's own module declaration in `f32/mod.rs` — NOT from the
//! narrower `not(force-scalar)` gate on the *alias* re-export. The concrete
//! backend type (e.g. `crate::f32::sse2::Vec3`) exists whenever its module
//! compiles, regardless of whether `force-scalar` also wins the top-level
//! `crate::f32::Vec3` alias for that build — so gating on the module's own
//! condition, not the alias's, is what's actually correct here.

// ── Vec2 (canonical) ─────────────────────────────────────────────────────────
crate::impl_vec2_swizzle!(crate::f32::Vec2, crate::f32::Vec3, crate::f32::Vec4);

// ── x86 / x86_64 (SSE2) ──────────────────────────────────────────────────────
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
crate::impl_vec3_swizzle!(crate::f32::sse2::Vec3, crate::f32::Vec2, crate::f32::sse2::Vec4);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
crate::impl_vec4_swizzle!(crate::f32::sse2::Vec4, crate::f32::Vec2, crate::f32::sse2::Vec3);

// ── aarch64 (NEON) ───────────────────────────────────────────────────────────
#[cfg(target_arch = "aarch64")]
crate::impl_vec3_swizzle!(crate::f32::neon::Vec3, crate::f32::Vec2, crate::f32::neon::Vec4);
#[cfg(target_arch = "aarch64")]
crate::impl_vec4_swizzle!(crate::f32::neon::Vec4, crate::f32::Vec2, crate::f32::neon::Vec3);

// ── wasm32 / wasm64 + simd128 ────────────────────────────────────────────────
#[cfg(all(any(target_arch = "wasm32", target_arch = "wasm64"), target_feature = "simd128"))]
crate::impl_vec3_swizzle!(crate::f32::wasm::Vec3, crate::f32::Vec2, crate::f32::wasm::Vec4);
#[cfg(all(any(target_arch = "wasm32", target_arch = "wasm64"), target_feature = "simd128"))]
crate::impl_vec4_swizzle!(crate::f32::wasm::Vec4, crate::f32::Vec2, crate::f32::wasm::Vec3);

// ── Scalar fallback (unconditional — `f32/mod.rs` declares this module with
//    no cfg gate at all, so no gate belongs here either) ──────────────────────
crate::impl_vec3_swizzle!(crate::f32::scalar::Vec3, crate::f32::Vec2, crate::f32::scalar::Vec4);
crate::impl_vec4_swizzle!(crate::f32::scalar::Vec4, crate::f32::Vec2, crate::f32::scalar::Vec3);

// ── Portable SIMD (coresimd), opt-in via the `coresimd` feature ─────────────
#[cfg(feature = "coresimd")]
crate::impl_vec3_swizzle!(crate::f32::coresimd::Vec3, crate::f32::Vec2, crate::f32::coresimd::Vec4);
#[cfg(feature = "coresimd")]
crate::impl_vec4_swizzle!(crate::f32::coresimd::Vec4, crate::f32::Vec2, crate::f32::coresimd::Vec3);
