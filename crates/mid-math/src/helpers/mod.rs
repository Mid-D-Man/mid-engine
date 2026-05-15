// crates/mid-math/src/helpers/mod.rs
//! High-level math helpers — animation, physics, shading, algebra.
//!
//! Grouped here to keep lib.rs clean. All types are re-exported at the
//! crate root via `pub use helpers::*`-style exports in lib.rs.

pub mod angle;
pub mod dual_quat;
pub mod rotor;
pub mod spatial;
pub mod tangent;

pub use angle::{Degrees, Radians};
pub use dual_quat::DualQuat;
pub use rotor::Rotor3;
pub use spatial::{SpatialForce, SpatialInertia, SpatialVelocity};
pub use tangent::{PackedTangent, TangentFrame};
