// crates/mid-math/src/helpers/mod.rs
//! Supplementary math helpers built on top of the core types.
//!
//! DualQuat has moved to crate::f32::dual_quat — it is re-exported here
//! for backwards compatibility so existing code using
//! `use mid_math::helpers::DualQuat` continues to work.

pub mod angle;
pub mod euler;
pub mod rotor;
pub mod spatial;
pub mod tangent;

// DualQuat now lives in f32/ — re-export for convenience.
// DDualQuat lives in f64/ — also re-exported here for symmetry.
pub use crate::f32::dual_quat::DualQuat;
pub use crate::f64::ddual_quat::DDualQuat;

pub use angle::{Radians, Degrees};
pub use euler::{EulerRot, QuatExt};
pub use rotor::Rotor3;
pub use spatial::{SpatialVelocity, SpatialForce, SpatialInertia};
pub use tangent::{TangentFrame, PackedTangent};
