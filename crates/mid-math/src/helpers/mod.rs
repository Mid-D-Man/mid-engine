// crates/mid-math/src/helpers/mod.rs
//! Supplementary math helpers built on top of the core types.

pub mod angle;
pub mod euler;
pub mod octahedral;
pub mod rotor;
pub mod spatial;
pub mod tangent;

// DualQuat lives in f32/ — re-export for convenience.
// DDualQuat lives in f64/ — also re-exported here for symmetry.
pub use crate::f32::dual_quat::DualQuat;
pub use crate::f64::ddual_quat::DDualQuat;

pub use angle::{Radians, Degrees};
pub use euler::{EulerRot, QuatExt};
pub use octahedral::{
    encode_octahedral, decode_octahedral,
    encode_octahedral_snorm8,  decode_octahedral_snorm8,
    encode_octahedral_snorm16, decode_octahedral_snorm16,
    encode_octahedral_u32, decode_octahedral_u32,
};
pub use rotor::Rotor3;
pub use spatial::{SpatialVelocity, SpatialForce, SpatialInertia};
pub use tangent::{TangentFrame, PackedTangent};
