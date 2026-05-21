// crates/mid-math/src/f64/sse2/mod.rs
//! SSE2-backed f64 types — x86 / x86_64.
//!
//! `__m128d` holds 2× f64 (128 bits, 16-byte aligned).
//!
//! | Type   | Storage           | Size  | Align | Notes                    |
//! |--------|-------------------|-------|-------|--------------------------|
//! | DVec2  | 1× __m128d        | 16 B  | 16 B  | Perfect 2-lane fit       |
//! | DVec4  | 2× __m128d lo+hi  | 32 B  | 32 B  | lo=[x,y] hi=[z,w]        |
//! | DQuat  | 2× __m128d lo+hi  | 32 B  | 32 B  | lo=[x,y] hi=[z,w]        |
//!
//! DVec3, DMat2, DMat3, DMat4, DAffine3 remain scalar — 3-lane f64 SIMD
//! requires 2 registers with awkward masking, and the matrix types benefit
//! more from AVX2 (4-lane f64) which is gated separately.
//!
//! Cross-compilation check from x86_64:
//!   cargo check -p mid-math
//!   cargo test  -p mid-math --release

pub mod dvec2;
pub mod dvec4;
pub mod dquat;

pub use dvec2::DVec2;
pub use dvec4::DVec4;
pub use dquat::DQuat;
