// crates/mid-math/src/wide/float/wasm/mod.rs
//! WASM SIMD128 wide float backend.
//!
//! Mirrors SSE2 layout:
//!   v128         ↔  __m128
//!   f32x4_*      ↔  _mm_*_ps
//!   i32x4_shuffle↔  _mm_shuffle_ps / movelh / movehl / unpack*
//!
//! Differences from SSE2:
//!   v128_andnot(a, b)  = a & !b   (SSE2 _mm_andnot_ps(a,b) = !a & b)
//!   no rsqrt/rcp hardware — use sqrt+div+NR
//!   no movemask — use i32x4_bitmask / i32x4_all_true
//!   AoS↔SoA transpose via i32x4_shuffle (same 7-op algorithm as SSE2)

pub mod mask4;
pub mod f32x4;
pub mod vec3x4;
pub mod quatx4;

pub use mask4::{Mask4, Mask4LaneIter};
pub use vec3x4::Vec3x4;
pub use quatx4::QuatX4;
