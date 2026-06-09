// crates/mid-math/src/camera/frustum.rs
//! View frustum — 6-plane representation with culling tests.
//!
//! Planes are extracted from a combined view-projection matrix using the
//! Gribb-Hartmann method (2001). Each plane is stored as `Vec4(nx, ny, nz, d)`
//! where a point `P` is INSIDE the half-space when:
//!   `nx*Px + ny*Py + nz*Pz + d >= 0`
//!
//! Planes are normalised on construction so that `plane_distance` returns
//! true metric distances in world units.
//!
//! ## Culling workflow
//! ```text
//! let vp   = proj * view;
//! let frust = Frustum::from_view_proj(vp);
//!
//! for entity in scene {
//!     if frust.test_aabb(entity.aabb_min, entity.aabb_max) {
//!         render(entity);
//!     }
//! }
//! ```

use crate::{Vec3, Vec4, Mat4};

// ── Plane indices ─────────────────────────────────────────────────────────────

pub const FRUSTUM_LEFT:   usize = 0;
pub const FRUSTUM_RIGHT:  usize = 1;
pub const FRUSTUM_BOTTOM: usize = 2;
pub const FRUSTUM_TOP:    usize = 3;
pub const FRUSTUM_NEAR:   usize = 4;
pub const FRUSTUM_FAR:    usize = 5;

// ── Visibility result ─────────────────────────────────────────────────────────

/// Result of a frustum visibility test.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Visibility {
    /// Entirely outside at least one plane — safe to cull completely.
    Outside,
    /// Entirely inside all six planes — no need to test children.
    Inside,
    /// Straddles the frustum boundary — children must be tested individually.
    Intersect,
}

// ── Frustum ───────────────────────────────────────────────────────────────────

/// A view frustum defined by 6 half-spaces.
///
/// Planes are in the same coordinate space as the matrix used to construct
/// the frustum. Construct from a view-projection matrix to get world-space
/// planes, or from a plain projection matrix to get view-space planes.
#[derive(Clone, Copy, Debug)]
pub struct Frustum {
    /// Normalised planes in order: [Left, Right, Bottom, Top, Near, Far].
    ///
    /// Each `Vec4(nx, ny, nz, d)` satisfies:
    ///   `dot(normal, P) + d >= 0` iff P is on the inside half-space.
    pub planes: [Vec4; 6],
}

impl Frustum {
    /// Extract frustum planes from a combined **view-projection** matrix.
    ///
    /// Uses Gribb-Hartmann (2001): planes are derived directly from matrix
    /// row combinations — no additional matrix inverse is required.
    ///
    /// Clip space convention: `[-1, 1]` on all axes (OpenGL / Vulkan with
    /// the RH perspective matrices used by mid-math).
    ///
    /// Typical usage:
    /// ```
    /// let view_proj = proj * view;
    /// let frustum   = Frustum::from_view_proj(view_proj);
    /// ```
    pub fn from_view_proj(m: Mat4) -> Self {
        // Mat4 is column-major with named Vec4 fields (x_axis, y_axis, z_axis, w_axis).
        // Row r is formed by taking component r from each column field:
        //   row 0: (.x from each col), row 1: (.y from each col), etc.
        let r0 = Vec4::new(m.x_axis.x, m.y_axis.x, m.z_axis.x, m.w_axis.x);
        let r1 = Vec4::new(m.x_axis.y, m.y_axis.y, m.z_axis.y, m.w_axis.y);
        let r2 = Vec4::new(m.x_axis.z, m.y_axis.z, m.z_axis.z, m.w_axis.z);
        let r3 = Vec4::new(m.x_axis.w, m.y_axis.w, m.z_axis.w, m.w_axis.w);

        // Gribb-Hartmann plane extraction (clip-space [-1, 1]):
        //   Left:   w + x >= 0  →  row3 + row0
        //   Right:  w - x >= 0  →  row3 - row0
        //   Bottom: w + y >= 0  →  row3 + row1
        //   Top:    w - y >= 0  →  row3 - row1
        //   Near:   w + z >= 0  →  row3 + row2
        //   Far:    w - z >= 0  →  row3 - row2
        Self {
            planes: [
                normalise_plane(v4_add(r3, r0)), // left
                normalise_plane(v4_sub(r3, r0)), // right
                normalise_plane(v4_add(r3, r1)), // bottom
                normalise_plane(v4_sub(r3, r1)), // top
                normalise_plane(v4_add(r3, r2)), // near
                normalise_plane(v4_sub(r3, r2)), // far
            ],
        }
    }

    /// Build from pre-computed planes. Each is normalised on entry.
    pub fn from_planes(planes: [Vec4; 6]) -> Self {
        Self { planes: planes.map(normalise_plane) }
    }

    // ── Signed distance ───────────────────────────────────────────────────────

    /// Signed distance from point `p` to a plane.
    ///
    /// Positive = p is inside (in front of) the plane.
    /// Negative = p is outside (behind) the plane.
    /// Zero     = p lies exactly on the plane.
    ///
    /// Because planes are normalised, the magnitude equals the true metric
    /// distance in world units.
    #[inline]
    pub fn plane_distance(plane: Vec4, p: Vec3) -> f32 {
        plane.x * p.x + plane.y * p.y + plane.z * p.z + plane.w
    }

    // ── Point tests ───────────────────────────────────────────────────────────

    /// Returns `true` if the point `p` is inside (or on the boundary of) the frustum.
    #[inline]
    pub fn test_point(&self, p: Vec3) -> bool {
        for plane in &self.planes {
            if Self::plane_distance(*plane, p) < 0.0 { return false; }
        }
        true
    }

    // ── Sphere tests ──────────────────────────────────────────────────────────

    /// Fast conservative sphere test.
    ///
    /// Returns `false` only when the sphere is definitely outside.
    /// May return `true` for spheres that barely miss a frustum corner
    /// (false positive) — use `test_sphere_visibility` for exact results.
    ///
    /// This is the standard culling path: one dot-product per plane, no sqrt.
    #[inline]
    pub fn test_sphere(&self, center: Vec3, radius: f32) -> bool {
        for plane in &self.planes {
            if Self::plane_distance(*plane, center) < -radius { return false; }
        }
        true
    }

    /// Precise sphere test that returns a `Visibility` classification.
    ///
    /// A sphere is `Inside` only when every plane distance exceeds `radius`.
    /// The early-out for `Outside` is the same fast test as `test_sphere`.
    pub fn test_sphere_visibility(&self, center: Vec3, radius: f32) -> Visibility {
        let mut fully_inside = 0usize;
        for plane in &self.planes {
            let d = Self::plane_distance(*plane, center);
            if d < -radius  { return Visibility::Outside; }
            if d >=  radius { fully_inside += 1; }
        }
        if fully_inside == 6 { Visibility::Inside } else { Visibility::Intersect }
    }

    // ── AABB tests ────────────────────────────────────────────────────────────

    /// Fast conservative AABB test.
    ///
    /// For each plane, projects the AABB corner most aligned with the plane
    /// normal (the "positive vertex"). If that corner is outside, the whole
    /// box is outside. No square roots, no branch-heavy loop.
    ///
    /// False positives are possible near frustum edges; use
    /// `test_aabb_visibility` for an exact `Visibility` result.
    #[inline]
    pub fn test_aabb(&self, min: Vec3, max: Vec3) -> bool {
        for plane in &self.planes {
            // Positive vertex: the AABB corner most in the direction of normal.
            let px = if plane.x >= 0.0 { max.x } else { min.x };
            let py = if plane.y >= 0.0 { max.y } else { min.y };
            let pz = if plane.z >= 0.0 { max.z } else { min.z };
            if plane.x * px + plane.y * py + plane.z * pz + plane.w < 0.0 {
                return false;
            }
        }
        true
    }

    /// Precise AABB test returning a `Visibility` classification.
    ///
    /// A box is `Inside` only when even its most-outside corner (the
    /// "negative vertex" per plane) is inside every plane.
    pub fn test_aabb_visibility(&self, min: Vec3, max: Vec3) -> Visibility {
        let mut fully_inside = 0usize;
        for plane in &self.planes {
            // Positive vertex (most aligned with normal — used for rejection).
            let px = if plane.x >= 0.0 { max.x } else { min.x };
            let py = if plane.y >= 0.0 { max.y } else { min.y };
            let pz = if plane.z >= 0.0 { max.z } else { min.z };

            // Negative vertex (most opposed to normal — used for full-inside test).
            let nx = if plane.x >= 0.0 { min.x } else { max.x };
            let ny = if plane.y >= 0.0 { min.y } else { max.y };
            let nz = if plane.z >= 0.0 { min.z } else { max.z };

            let dp = plane.x * px + plane.y * py + plane.z * pz + plane.w;
            let dn = plane.x * nx + plane.y * ny + plane.z * nz + plane.w;

            if dp < 0.0 { return Visibility::Outside; }
            if dn >= 0.0 { fully_inside += 1; }
        }
        if fully_inside == 6 { Visibility::Inside } else { Visibility::Intersect }
    }

    // ── World-space corners ───────────────────────────────────────────────────

    /// Extract the 8 world-space corners of the frustum.
    ///
    /// Requires the **inverse** view-projection matrix:
    ///   `let inv_vp = (proj * view).inverse().unwrap();`
    ///
    /// Corner order: `[LBN, RBN, RTN, LTN, LBF, RBF, RTF, LTF]`
    ///   L/R = left/right, B/T = bottom/top, N/F = near/far.
    ///
    /// Useful for: debug visualisation, shadow-cascade bounding, sky-box sizing.
    pub fn world_corners(inv_view_proj: Mat4) -> [Vec3; 8] {
        // The 8 corners of the NDC cube in clip space, w=1 before divide.
        let ndc: [Vec4; 8] = [
            Vec4::new(-1., -1., -1., 1.), // LBN
            Vec4::new( 1., -1., -1., 1.), // RBN
            Vec4::new( 1.,  1., -1., 1.), // RTN
            Vec4::new(-1.,  1., -1., 1.), // LTN
            Vec4::new(-1., -1.,  1., 1.), // LBF
            Vec4::new( 1., -1.,  1., 1.), // RBF
            Vec4::new( 1.,  1.,  1., 1.), // RTF
            Vec4::new(-1.,  1.,  1., 1.), // LTF
        ];

        let mut out = [Vec3::ZERO; 8];
        for (i, &c) in ndc.iter().enumerate() {
            let w = inv_view_proj * c;    // Mat4 * Vec4 — implemented on all backends
            let iw = 1.0 / w.w;
            out[i] = Vec3::new(w.x * iw, w.y * iw, w.z * iw);
        }
        out
    }

    /// World-space centroid of the frustum.
    pub fn world_center(inv_view_proj: Mat4) -> Vec3 {
        let c = Self::world_corners(inv_view_proj);
        let mut s = Vec3::ZERO;
        for v in &c { s = s + *v; }
        s * (1.0 / 8.0)
    }

    /// Axis-aligned bounding box (min, max) enclosing the frustum in world space.
    pub fn world_aabb(inv_view_proj: Mat4) -> (Vec3, Vec3) {
        let c = Self::world_corners(inv_view_proj);
        let mut mn = c[0];
        let mut mx = c[0];
        for v in &c[1..] {
            mn = mn.min(*v);
            mx = mx.max(*v);
        }
        (mn, mx)
    }
}

// ── Private helpers ───────────────────────────────────────────────────────────

/// Normalise a plane so that |normal| = 1.
/// After normalisation, `plane_distance` returns true metric distances.
#[inline]
fn normalise_plane(p: Vec4) -> Vec4 {
    let len = (p.x * p.x + p.y * p.y + p.z * p.z).sqrt();
    if len < 1e-10 { return p; }
    let inv = 1.0 / len;
    Vec4::new(p.x * inv, p.y * inv, p.z * inv, p.w * inv)
}

#[inline] fn v4_add(a: Vec4, b: Vec4) -> Vec4 {
    Vec4::new(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w)
}
#[inline] fn v4_sub(a: Vec4, b: Vec4) -> Vec4 {
    Vec4::new(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w)
            }
