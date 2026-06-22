// crates/mid-math/src/f32/wasm/mod.rs
//! WASM SIMD128 implementations — wasm32/wasm64 with simd128 target feature.
//!
//! Status:
//!   Vec3   v128, full SIMD
//!   Vec4   v128, full SIMD
//!   Quat   v128, full SIMD mul_quat + slerp
//!   Mat4   SIMD Mul<Vec4> + Mul<Mat4> + cofactor inverse
//!
//! Build with:
//!   RUSTFLAGS="-C target-feature=+simd128" cargo build --target wasm32-wasip1
//!
//! Test with wasmtime (see .github/workflows/mid-math-test-wasm.yml):
//!   cargo test -p mid-math --target wasm32-wasip1 --release
//!
//! Cross-compile check from x86_64 dev:
//!   cargo check --target wasm32-unknown-unknown \
//!     --config 'build.rustflags=["-C","target-feature=+simd128"]'

pub mod vec3;
pub mod vec4;
pub mod quat;
pub mod mat4;

// ── WASM Relaxed SIMD extension ───────────────────────────────────────────────
// Compiled when relaxed-simd target feature is present (alongside simd128).
// Build: RUSTFLAGS="-C target-feature=+simd128,+relaxed-simd"
// Nightly-only intrinsics as of 2026-06 — stable build is a no-op.
#[cfg(target_feature = "relaxed-simd")]
pub mod relaxed;

pub use vec3::Vec3;
pub use vec4::Vec4;
pub use quat::Quat;
pub use mat4::Mat4;
