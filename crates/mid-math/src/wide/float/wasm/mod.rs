// crates/mid-math/src/wide/float/wasm/mod.rs
//! WASM SIMD128 wide float backend.
//!
//! Mirrors the SSE2 backend conceptually:
//!   - `v128` plays the role of `__m128`
//!   - No dedicated rsqrt/rcp instructions; we use Newton-Raphson on top of
//!     `f32x4_sqrt` for `recip_sqrt`, and division for `recip`
//!   - No `movemask`; we use `i32x4_bitmask` for mask extraction
//!   - AoS→SoA transpose uses the same 7-shuffle pattern as SSE2,
//!     mapped to `i32x4_shuffle` (compile-time-constant lane indices)
//!   - FMA: WASM relaxed-simd adds `f32x4_relaxed_madd`, but baseline
//!     simd128 doesn't have it — we emit two ops and let LLVM fuse if able
//!
//! Build with: RUSTFLAGS="-C target-feature=+simd128"

pub mod mask4;
pub mod f32x4;
pub mod vec3x4;
pub mod quatx4;

pub use mask4::Mask4;
pub use vec3x4::Vec3x4;
pub use quatx4::QuatX4;
