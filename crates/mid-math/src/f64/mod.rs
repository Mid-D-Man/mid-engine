// crates/mid-math/src/f64/mod.rs
//! Double-precision (f64) math types.
//!
//! All types are scalar-only for now. Alignment is set to 32 bytes on
//! DVec3/DVec4/DQuat/DMat4/DAffine3 to reserve space for a future AVX2
//! fast path (four f64 per ymm register).
//!
//! DEPSILON = 1e-12  (vs 1e-6 for f32 types).

pub mod dvec2;
pub mod dvec3;
pub mod dvec4;
pub mod dquat;
pub mod dmat2;
pub mod dmat3;
pub mod dmat4;
pub mod daffine3;

pub use dvec2::DVec2;
pub use dvec3::DVec3;
pub use dvec4::DVec4;
pub use dquat::DQuat;
pub use dmat2::DMat2;
pub use dmat3::DMat3;
pub use dmat4::DMat4;
pub use daffine3::DAffine3;

/// f64 comparison epsilon.
pub const DEPSILON: f64 = dvec2::DEPSILON;
