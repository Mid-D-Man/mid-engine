// crates/mid-geom/src/lib.rs
//! Geometry primitives for Mid Engine.
//!
//! Organised by dimension (d2 / d3) and domain (shapes, raycast, planes, transform).
//! All types are re-exported flat from this root.
//!
//! **FFI boundary:** see [`ffi`] for C-ABI types and `#[no_mangle]` exports.
//!
//! **Dependency:** mid-geom → mid-math (one-way, no cycle).
//! Math primitives (Vec3, Mat4, Quat, etc.) always come from mid-math.

pub mod d2;
pub mod d3;
pub mod ffi;

// ── 2D flat re-exports ────────────────────────────────────────────────────────

pub use d2::shapes::{Circle, Rect};
pub use d2::raycast::Ray2;
pub use d2::transform::Transform2D;

// ── 3D flat re-exports ────────────────────────────────────────────────────────

pub use d3::shapes::{AABB, Capsule, Sphere};
pub use d3::planes::{Frustum, Plane};
pub use d3::raycast::Ray3;
pub use d3::transform::Transform;
