// crates/mid-math/src/geom/mod.rs
//! Geometric primitives — pure math, no allocations, `no_std` compatible.
//!
//! Higher-level geometry operations (mesh processing, CSG, Delaunay
//! triangulation, convex hull) belong in the `mid-geom` crate. This module
//! contains only the foundational math that everything else depends on.
//!
//! | Module        | Contents                                              |
//! |---------------|-------------------------------------------------------|
//! | `barycentric` | Barycentric coordinates, triangle tests, interpolation|

pub mod barycentric;

pub use barycentric::{
    BarycentricCoords,
    Triangle2,
    Triangle3,
    signed_area_2d,
    triangle_area_3d,
};
