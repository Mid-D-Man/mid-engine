// crates/mid-math/src/f64/wasm/mod.rs
//! WASM SIMD128 f64 types — wasm32/wasm64 with target_feature = "simd128".
//!
//! `v128` used as `f64x2`: lane 0 = x (bytes 0-7), lane 1 = y (bytes 8-15).
//! Memory layout is byte-identical to [f64; 2] and `XY<f64>` — Deref is zero-cost.
//!
//! Types backed here:
//!   DVec2 — 1× v128 (perfect 2-lane fit, same as SSE2 __m128d)
//!   DVec4 — 2× v128: lo=[x,y], hi=[z,w]  (same as SSE2 2×__m128d)
//!   DQuat — 2× v128: lo=[x,y], hi=[z,w]
//!
//! Types remaining scalar (no SIMD gain):
//!   DVec3   — 3 f64 with no padding; 3-lane f64x2 needs 2 registers + masking
//!   DMat2/3/4/DAffine3 — scalar (AVX2 f64 is the justified target, not SIMD128)
//!
//! Key WASM f64x2 advantages vs scalar:
//!   f64x2_add/sub/mul/div — 2 ops per instruction
//!   f64x2_sqrt            — parallel sqrt
//!   f64x2_abs / f64x2_neg — direct (no sign-mask trick unlike SSE2)
//!   f64x2_gt / f64x2_eq   — comparison to v128 mask
//!   v128_and/or/xor       — boolean masking
//!
//! Note: v128_andnot(a, b) = a & ~b  (WASM — opposite argument order to SSE2)
//!
//! Build / test:
//!   RUSTFLAGS="-C target-feature=+simd128" \
//!   cargo build --target wasm32-unknown-unknown

pub mod dvec2;
pub mod dvec4;
pub mod dquat;

pub use dvec2::DVec2;
pub use dvec4::DVec4;
pub use dquat::DQuat;
