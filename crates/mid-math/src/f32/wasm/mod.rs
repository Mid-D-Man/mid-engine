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
//!   RUSTFLAGS="-C target-feature=+simd128" cargo build --target wasm32-unknown-unknown
//!
//! Test with wasm-pack:
//!   RUSTFLAGS="-C target-feature=+simd128" \
//!   wasm-pack test --node -- -p mid-math
//!
//! Cross-compile check from x86_64 dev:
//!   cargo check --target wasm32-unknown-unknown \
//!     --config 'build.rustflags=["-C","target-feature=+simd128"]'

pub mod vec3;
pub mod vec4;
pub mod quat;
pub mod mat4;

pub use vec3::Vec3;
pub use vec4::Vec4;
pub use quat::Quat;
pub use mat4::Mat4;
