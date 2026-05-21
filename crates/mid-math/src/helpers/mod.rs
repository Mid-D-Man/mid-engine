// crates/mid-math/src/helpers/mod.rs
//! Supplementary math helpers built on top of the core types.

pub mod angle;
pub mod dual_quat;
pub mod euler;
pub mod rotor;
pub mod spatial;
pub mod tangent;

// Re-export everything callers need from each sub-module.
pub use angle::{Radians, Degrees};
pub use dual_quat::DualQuat;
pub use euler::{EulerRot, QuatExt};
pub use rotor::Rotor3;
pub use spatial::{SpatialVelocity, SpatialForce, SpatialInertia};
pub use tangent::{TangentFrame, PackedTangent};
