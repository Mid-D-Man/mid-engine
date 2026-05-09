// crates/mid-math/src/geometry/mod.rs
//! Geometry primitives for Mid Engine.
//!
//! Organised by dimension (d2 / d3) and domain (shapes, raycast, planes, transform).
//!
//! **Extraction note:** this module will move to `mid-geom` when BVH
//! construction begins. Internal imports use `crate::` — update to
//! `mid_math::` at that point. The public API surface is intentionally flat
//! (re-exported from here) so call sites need zero changes.

pub mod d2;
pub mod d3;

// ── Flat re-exports — all geometry types at crate::geometry::* ───────────────

// 2D
pub use d2::shapes::{Circle, Rect};
pub use d2::raycast::Ray2;
pub use d2::transform::Transform2D;

// 3D
pub use d3::shapes::{AABB, Capsule, Sphere};
pub use d3::planes::{Frustum, Plane};
pub use d3::raycast::Ray3;
pub use d3::transform::Transform;
