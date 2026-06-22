// crates/mid-math/src/f32/wasm/relaxed.rs
//! WASM Relaxed SIMD fast paths — STUB (not yet implemented).
//!
//! ## Gate
//! #[cfg(all(any(target_arch = "wasm32", target_arch = "wasm64"),
//!           target_feature = "relaxed-simd"))]
//! (Declared in wasm/mod.rs.)
//!
//! ## Availability
//! Chrome 114+ (2023-05), Firefox 117+ (2023-08), Safari 16.4+
//! Node.js 20.x: --experimental-wasm-relaxed-simd flag
//! Node.js 22.x: enabled by default (confirm before removing flag from CI)
//!
//! Build:
//!   RUSTFLAGS="-C target-feature=+simd128,+relaxed-simd" \
//!   cargo build --target wasm32-wasip1
//!
//! Rust feature gate (nightly as of 2026-06):
//!   No stable API yet. The target-feature flag compiles but intrinsics
//!   require nightly via core::arch::wasm32::* relaxed variants.
//!
//! ## New instructions vs simd128
//! | Instruction                    | Game math use case                 |
//! |--------------------------------|------------------------------------|
//! | f32x4_relaxed_madd             | FMA: normalize, dot, lerp chains   |
//! | f32x4_relaxed_nmadd            | FMA with negation: cross product   |
//! | i8x16_relaxed_swizzle          | byte shuffle for color/normal pack |
//! | f32x4_relaxed_min/max          | branchless clamp (IEEE-safe)       |
//! | i32x4_relaxed_trunc_f32x4_s/u | float→int without trap             |
//!
//! ## Expected gains over simd128
//! normalize:  ~2.4 ns → ~2.1 ns (rsqrt + fmadd instead of sqrt + div)
//! mat4/mul:   ~1-5% gain from fmadd chains replacing mul+add pairs
//! lerp/dot:   minimal (already memory/shuffle bound)
//!
//! Gains are larger on ARM WASM (mobile, Chromebook) where relaxed-simd
//! maps to NEON FMA. On x86, simd128 already maps to SSE2+.
//!
//! ## When to implement
//!   1. Relaxed SIMD intrinsics stabilize in Rust (track rust-lang/rust)
//!   2. CI wasm test workflow updated to add relaxed-simd target-feature
//!   3. Bench confirms >10% gain over simd128 on ARM WASM targets
//!
//! ## Future items in this file
//! - Overrides for Vec3::normalize using f32x4_relaxed_madd (rsqrt path)
//! - Overrides for Vec4::dot using relaxed dot sequence
//! - Overrides for Quat::nlerp using fmadd chain
//! (These would wrap the existing wasm/vec3.rs etc. types, not replace them)
