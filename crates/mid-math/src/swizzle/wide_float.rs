// crates/mid-math/src/swizzle/wide_float.rs
// NOTICE: Full documentation, design decisions, and fix history for this
// file live in docs/mid-math.md, section "swizzle/wide_float.rs".
//! Wide-swizzle impls for the f32 wide family: axis-swizzle for `Vec3x4`
//! (via `Vec3AxisSwizzle`, see `wide_axis_engine.rs`) and lane-shuffle for
//! `f32x4`/`f32x8` (via `LaneShuffle4`/`LaneShuffle8`, see
//! `wide_lane_engine.rs`). No `QuatX4` — see `wide_axis_engine.rs`'s doc
//! comment for why. Each `#[cfg(...)]` here is copied from that backend's
//! own module declaration in `wide/float/mod.rs`.
//!
//! `Vec3x8`'s `Vec3AxisSwizzle` impl is NOT generated here — see
//! `wide/float/avx2/vec3x8.rs` directly. `impl_vec3_axis_swizzle!` builds a
//! struct literal assuming public `x`/`y`/`z` fields, which holds for
//! `Vec3x4` but not `Vec3x8` (private `lo`/`hi` halves instead), so it's
//! written by hand there instead of macro-generated here.

// ── x86 / x86_64 (SSE2) ──────────────────────────────────────────────────────
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
crate::impl_vec3_axis_swizzle!(crate::wide::float::sse2::vec3x4::Vec3x4);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
crate::impl_lane_shuffle4!(crate::wide::float::sse2::f32x4::f32x4);

// ── aarch64 (NEON) ───────────────────────────────────────────────────────────
#[cfg(target_arch = "aarch64")]
crate::impl_vec3_axis_swizzle!(crate::wide::float::neon::vec3x4::Vec3x4);
#[cfg(target_arch = "aarch64")]
crate::impl_lane_shuffle4!(crate::wide::float::neon::f32x4::f32x4);

// ── wasm32 / wasm64 + simd128 ────────────────────────────────────────────────
#[cfg(all(any(target_arch = "wasm32", target_arch = "wasm64"), target_feature = "simd128"))]
crate::impl_vec3_axis_swizzle!(crate::wide::float::wasm::vec3x4::Vec3x4);
#[cfg(all(any(target_arch = "wasm32", target_arch = "wasm64"), target_feature = "simd128"))]
crate::impl_lane_shuffle4!(crate::wide::float::wasm::f32x4::f32x4);

// ── Scalar fallback (unconditional) ──────────────────────────────────────────
crate::impl_vec3_axis_swizzle!(crate::wide::float::scalar::vec3x4::Vec3x4);
crate::impl_lane_shuffle4!(crate::wide::float::scalar::f32x4::f32x4);

// ── AVX2 (additive — Vec3x8/f32x8 have no scalar/neon/wasm equivalent,
//    always compiled now, not gated on the avx2 target feature — see
//    wide/float/mod.rs's doc comment) ─────────────────────────────────────────
// Vec3x8's Vec3AxisSwizzle impl lives in wide/float/avx2/vec3x8.rs — see this
// file's own header comment for why.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
crate::impl_lane_shuffle8!(crate::wide::float::avx2::f32x8::f32x8);
