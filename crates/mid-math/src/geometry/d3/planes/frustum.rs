// crates/mid-math/src/geometry/d3/planes/frustum.rs
//! View frustum — 6 clipping planes extracted from a view-projection matrix.

use crate::{Mat4, Vec3, EPSILON};
use super::plane::Plane;
use super::super::shapes::{AABB, Sphere};

/// View frustum defined by 6 planes. 96 bytes.
///
/// Plane order (index):
///   0 = Left, 1 = Right, 2 = Bottom, 3 = Top, 4 = Near, 5 = Far
///
/// Points inside the frustum have positive signed distance from ALL 6 planes.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct Frustum {
    pub planes: [Plane; 6],
}

impl Frustum {
    /// Extract frustum planes from a combined view-projection matrix.
    ///
    /// Uses the Gribb/Hartmann method — works for any hand / depth convention.
    /// Pass `view_proj = projection * view`.
    ///
    /// Column-major `cols[c][r]` maps to row r as:
    /// `[cols[0][r], cols[1][r], cols[2][r], cols[3][r]]`
    pub fn from_mat4(m: &Mat4) -> Self {
        // Row helpers
        let row = |r: usize| {
            [m.cols[0][r], m.cols[1][r], m.cols[2][r], m.cols[3][r]]
        };
        let r0 = row(0); let r1 = row(1);
        let r2 = row(2); let r3 = row(3);

        // Gribb/Hartmann plane extraction (normalised)
        let left   = Plane::from_coefficients(r3[0]+r0[0], r3[1]+r0[1], r3[2]+r0[2], r3[3]+r0[3]);
        let right  = Plane::from_coefficients(r3[0]-r0[0], r3[1]-r0[1], r3[2]-r0[2], r3[3]-r0[3]);
        let bottom = Plane::from_coefficients(r3[0]+r1[0], r3[1]+r1[1], r3[2]+r1[2], r3[3]+r1[3]);
        let top    = Plane::from_coefficients(r3[0]-r1[0], r3[1]-r1[1], r3[2]-r1[2], r3[3]-r1[3]);
        let near   = Plane::from_coefficients(r3[0]+r2[0], r3[1]+r2[1], r3[2]+r2[2], r3[3]+r2[3]);
        let far    = Plane::from_coefficients(r3[0]-r2[0], r3[1]-r2[1], r3[2]-r2[2], r3[3]-r2[3]);

        Self { planes: [left, right, bottom, top, near, far] }
    }

    // ── Visibility tests ──────────────────────────────────────────────────────

    /// True if `point` is inside (or on the boundary of) the frustum.
    #[inline]
    pub fn contains_point(self, p: Vec3) -> bool {
        self.planes.iter().all(|plane| plane.signed_distance(p) >= 0.0)
    }

    /// Conservative sphere test.
    ///
    /// Returns `false` only when the sphere is definitely outside.
    /// May return `true` when the sphere is just outside a corner (false positive).
    #[inline]
    pub fn intersects_sphere(self, sphere: &Sphere) -> bool {
        self.planes.iter().all(|plane| plane.signed_distance(sphere.center) >= -sphere.radius)
    }

    /// Conservative AABB test — positive vertex method.
    ///
    /// For each plane, tests the AABB vertex that is most in the direction
    /// of the plane normal (the "positive vertex"). If the positive vertex
    /// is behind any plane, the AABB is fully outside.
    #[inline]
    pub fn intersects_aabb(self, aabb: &AABB) -> bool {
        for plane in &self.planes {
            let n = plane.normal;
            // Choose the vertex most in the normal's direction
            let px = if n.x >= 0.0 { aabb.max.x } else { aabb.min.x };
            let py = if n.y >= 0.0 { aabb.max.y } else { aabb.min.y };
            let pz = if n.z >= 0.0 { aabb.max.z } else { aabb.min.z };
            // signed_distance for (px,py,pz)
            let d = n.x * px + n.y * py + n.z * pz + plane.d;
            if d < 0.0 { return false; }
        }
        true
    }

    /// Full containment test for AABB — true only if the AABB is fully inside.
    #[inline]
    pub fn contains_aabb(self, aabb: &AABB) -> bool {
        for plane in &self.planes {
            let n = plane.normal;
            // Negative vertex: the vertex most against the normal
            let nx = if n.x >= 0.0 { aabb.min.x } else { aabb.max.x };
            let ny = if n.y >= 0.0 { aabb.min.y } else { aabb.max.y };
            let nz = if n.z >= 0.0 { aabb.min.z } else { aabb.max.z };
            let d = n.x * nx + n.y * ny + n.z * nz + plane.d;
            if d < 0.0 { return false; }
        }
        true
    }
          }
