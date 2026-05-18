// crates/mid-math/src/geom/barycentric.rs
//! Barycentric coordinates — the foundational math for triangle operations.
//!
//! ## What are barycentric coordinates?
//!
//! Any point P inside (or outside) a triangle ABC can be expressed as a
//! weighted combination of the three vertices:
//!
//! ```text
//! P = u·A + v·B + w·C     where  u + v + w = 1
//! ```
//!
//! The weights (u, v, w) are the **barycentric coordinates**. They encode
//! *where* the point sits relative to the triangle:
//!
//! - `(1, 0, 0)` → exactly at A
//! - `(0, 1, 0)` → exactly at B
//! - `(0, 0, 1)` → exactly at C
//! - `(1/3, 1/3, 1/3)` → centroid
//! - All three in `[0, 1]` → point is INSIDE the triangle
//! - Any coordinate < 0 → point is OUTSIDE the triangle
//!
//! ## Why is this foundational?
//!
//! Barycentric coordinates are the basis of:
//!   - **Mesh picking** — determine exactly which triangle a ray hit
//!   - **Texture interpolation** — blend UV coords across a triangle face
//!   - **Normal interpolation** — Phong/Gouraud shading smooth normals
//!   - **Color / attribute interpolation** — vertex colours, bone weights
//!   - **Point-in-triangle tests** — collision, selection, hit detection
//!   - **Einstein tile / aperiodic tiling** — testing point membership
//!     in complex polygon shapes built from triangulated patches
//!   - **Delaunay / Voronoi** — circumcircle tests use bary math
//!   - **Physics** — contact point attribution to vertices
//!   - **Animation** — blend shape weights, IK target projection
//!
//! ## Method: Cramer's Rule (exact, robust)
//!
//! We solve the 2×2 linear system formed by the edge vectors. This gives
//! exact rational results and handles all degenerate cases gracefully by
//! checking the determinant.
//!
//! For 3D triangles we project onto the dominant axis plane (the plane
//! where the triangle has maximum area) to keep full precision.
//!
//! ## Notation throughout this file
//! ```text
//! Triangle: A, B, C  (vertices, in counter-clockwise order for +normal)
//! Point:    P        (the point we are testing or interpolating at)
//! Weights:  u (for A), v (for B), w (for C)   — always u+v+w = 1
//! ```

use core::fmt;
use crate::{Vec2, Vec3};

// ── BarycentricCoords ─────────────────────────────────────────────────────────

/// The three barycentric weights `(u, v, w)` corresponding to vertices
/// `(A, B, C)` of a triangle. Guaranteed `u + v + w ≈ 1.0` when valid.
///
/// Weights are NOT clamped — values outside `[0, 1]` indicate a point
/// that is outside the triangle, which is useful information.
#[derive(Clone, Copy, PartialEq)]
pub struct BarycentricCoords {
    /// Weight for vertex A.
    pub u: f32,
    /// Weight for vertex B.
    pub v: f32,
    /// Weight for vertex C (always `1 - u - v` to maintain the sum=1 invariant).
    pub w: f32,
}

impl BarycentricCoords {
    // ── Construction ─────────────────────────────────────────────────────────

    /// Construct directly from weights. No normalisation is performed.
    /// `u + v + w` should equal `1.0` for a valid barycentric triple.
    #[inline(always)]
    pub const fn new(u: f32, v: f32, w: f32) -> Self { Self { u, v, w } }

    /// The centroid `(1/3, 1/3, 1/3)`.
    pub const CENTROID: Self = Self { u: 1.0/3.0, v: 1.0/3.0, w: 1.0/3.0 };

    /// Vertex A: `(1, 0, 0)`.
    pub const VERTEX_A: Self = Self { u: 1.0, v: 0.0, w: 0.0 };
    /// Vertex B: `(0, 1, 0)`.
    pub const VERTEX_B: Self = Self { u: 0.0, v: 1.0, w: 0.0 };
    /// Vertex C: `(0, 0, 1)`.
    pub const VERTEX_C: Self = Self { u: 0.0, v: 0.0, w: 1.0 };

    // ── Classification ────────────────────────────────────────────────────────

    /// Returns `true` if the point is strictly inside the triangle.
    ///
    /// "Strictly inside" means all weights are in the open interval `(0, 1)`.
    /// Points exactly on an edge have one weight equal to `0.0`.
    /// Use `is_inside_or_on_edge` if edge points should count as inside.
    #[inline]
    pub fn is_inside(&self) -> bool {
        self.u > 0.0 && self.v > 0.0 && self.w > 0.0
    }

    /// Returns `true` if the point is inside the triangle OR exactly on an edge.
    ///
    /// All weights must be non-negative. A weight of exactly `0.0` means the
    /// point lies on the edge opposite that vertex.
    #[inline]
    pub fn is_inside_or_on_edge(&self) -> bool {
        self.u >= 0.0 && self.v >= 0.0 && self.w >= 0.0
    }

    /// Returns `true` if the point is outside the triangle.
    #[inline]
    pub fn is_outside(&self) -> bool { !self.is_inside_or_on_edge() }

    /// Returns `true` if the point lies exactly on an edge (one weight is zero).
    #[inline]
    pub fn is_on_edge(&self) -> bool {
        self.is_inside_or_on_edge()
            && (self.u.abs() < f32::EPSILON
                || self.v.abs() < f32::EPSILON
                || self.w.abs() < f32::EPSILON)
    }

    /// Returns `true` if the point coincides with a vertex (two weights are zero).
    #[inline]
    pub fn is_vertex(&self) -> bool {
        let e = f32::EPSILON;
        (self.u > 1.0 - e) || (self.v > 1.0 - e) || (self.w > 1.0 - e)
    }

    // ── Interpolation ─────────────────────────────────────────────────────────

    /// Interpolate a `f32` scalar across the triangle.
    ///
    /// Typical use: depth, temperature, height, any per-vertex float.
    /// ```text
    /// value = u·va + v·vb + w·vc
    /// ```
    #[inline]
    pub fn interpolate_f32(&self, va: f32, vb: f32, vc: f32) -> f32 {
        self.u * va + self.v * vb + self.w * vc
    }

    /// Interpolate a 2D vector (e.g. UV texture coordinates).
    ///
    /// This is how rasterisers compute UVs for every fragment inside a triangle.
    /// ```text
    /// uv = u·uv_a + v·uv_b + w·uv_c
    /// ```
    #[inline]
    pub fn interpolate_vec2(&self, va: Vec2, vb: Vec2, vc: Vec2) -> Vec2 {
        va * self.u + vb * self.v + vc * self.w
    }

    /// Interpolate a 3D vector (e.g. normals, positions, colours as Vec3).
    ///
    /// Used for Phong shading (normal interpolation), smooth colour gradients,
    /// and reconstructing world-space positions from triangle + bary coords.
    /// ```text
    /// result = u·va + v·vb + w·vc
    /// ```
    #[inline]
    pub fn interpolate_vec3(&self, va: Vec3, vb: Vec3, vc: Vec3) -> Vec3 {
        va * self.u + vb * self.v + vc * self.w
    }

    /// Interpolate 4 scalars packed as `[f32; 4]` (e.g. RGBA, bone weights).
    #[inline]
    pub fn interpolate_f32x4(&self, va: [f32;4], vb: [f32;4], vc: [f32;4]) -> [f32;4] {
        [
            self.u * va[0] + self.v * vb[0] + self.w * vc[0],
            self.u * va[1] + self.v * vb[1] + self.w * vc[1],
            self.u * va[2] + self.v * vb[2] + self.w * vc[2],
            self.u * va[3] + self.v * vb[3] + self.w * vc[3],
        ]
    }

    // ── Validity ──────────────────────────────────────────────────────────────

    /// Returns `true` if the coords sum to approximately `1.0` and contain
    /// no NaN or Inf values. A degenerate triangle returns all-NaN/zero coords
    /// which this check will catch.
    #[inline]
    pub fn is_valid(&self) -> bool {
        let sum = self.u + self.v + self.w;
        self.u.is_finite()
            && self.v.is_finite()
            && self.w.is_finite()
            && (sum - 1.0).abs() < 1e-4
    }
}

impl fmt::Debug for BarycentricCoords {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Bary(u={:.4}, v={:.4}, w={:.4})", self.u, self.v, self.w)
    }
}
impl fmt::Display for BarycentricCoords {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:.4}, {:.4}, {:.4})", self.u, self.v, self.w)
    }
}

// ── Triangle2 ─────────────────────────────────────────────────────────────────

/// A 2D triangle with vertices A, B, C.
///
/// Used for screen-space picking, 2D physics, UV-space operations,
/// Spine2D bone-mesh deformation, and Einstein tile membership tests.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Triangle2 {
    pub a: Vec2,
    pub b: Vec2,
    pub c: Vec2,
}

impl Triangle2 {
    #[inline(always)]
    pub const fn new(a: Vec2, b: Vec2, c: Vec2) -> Self { Self { a, b, c } }

    // ── Barycentric computation ───────────────────────────────────────────────

    /// Compute barycentric coordinates of point `p` with respect to this triangle.
    ///
    /// Uses Cramer's rule on the 2×2 system formed by edge vectors AB and AC.
    /// This is exact (no square roots) and numerically stable.
    ///
    /// ## Derivation
    /// ```text
    /// P = A + v·(B-A) + w·(C-A)     expand:
    ///
    /// [B.x-A.x  C.x-A.x] [v]   [P.x-A.x]
    /// [B.y-A.y  C.y-A.y] [w] = [P.y-A.y]
    ///
    /// det = (B.x-A.x)·(C.y-A.y) - (C.x-A.x)·(B.y-A.y)
    ///
    /// v = ((P.x-A.x)·(C.y-A.y) - (C.x-A.x)·(P.y-A.y)) / det
    /// w = ((B.x-A.x)·(P.y-A.y) - (P.x-A.x)·(B.y-A.y)) / det
    /// u = 1 - v - w
    /// ```
    ///
    /// Returns `None` if the triangle is degenerate (zero area, collinear vertices).
    pub fn barycentric(&self, p: Vec2) -> Option<BarycentricCoords> {
        let ab = self.b - self.a;  // edge AB
        let ac = self.c - self.a;  // edge AC
        let ap = p     - self.a;   // A→P

        // Determinant = signed area * 2.  Zero → degenerate triangle.
        let det = ab.x * ac.y - ac.x * ab.y;
        if det.abs() < f32::EPSILON { return None; }

        let inv_det = 1.0 / det;
        let v = (ap.x * ac.y - ac.x * ap.y) * inv_det;
        let w = (ab.x * ap.y - ap.x * ab.y) * inv_det;
        let u = 1.0 - v - w;

        Some(BarycentricCoords::new(u, v, w))
    }

    /// Returns `true` if `p` is inside or on the boundary of the triangle.
    ///
    /// Faster than computing full coordinates when you only need a boolean.
    /// Uses the sign-of-cross-product method — O(1), no division.
    ///
    /// ## Method
    /// A point is inside iff it is on the same side of all three edges AB, BC, CA.
    /// The signed area (cross product z-component) has the same sign for all
    /// three sub-triangles PAB, PBC, PCA when P is inside.
    #[inline]
    pub fn contains(&self, p: Vec2) -> bool {
        let sign = |p1: Vec2, p2: Vec2, p3: Vec2| -> f32 {
            (p1.x - p3.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p3.y)
        };
        let d1 = sign(p, self.a, self.b);
        let d2 = sign(p, self.b, self.c);
        let d3 = sign(p, self.c, self.a);
        let has_neg = (d1 < 0.0) || (d2 < 0.0) || (d3 < 0.0);
        let has_pos = (d1 > 0.0) || (d2 > 0.0) || (d3 > 0.0);
        !(has_neg && has_pos)
    }

    // ── Area and geometry ─────────────────────────────────────────────────────

    /// Signed area of the 2D triangle.
    ///
    /// Positive → vertices in counter-clockwise order.
    /// Negative → vertices in clockwise order.
    /// Zero     → degenerate (collinear) triangle.
    ///
    /// `|signed_area| = area`.
    #[inline]
    pub fn signed_area(&self) -> f32 {
        signed_area_2d(self.a, self.b, self.c)
    }

    /// Unsigned area. Always non-negative.
    #[inline]
    pub fn area(&self) -> f32 { self.signed_area().abs() }

    /// Returns `true` if the vertices are in counter-clockwise winding order.
    #[inline]
    pub fn is_ccw(&self) -> bool { self.signed_area() > 0.0 }

    /// Centroid `(A + B + C) / 3`.
    #[inline]
    pub fn centroid(&self) -> Vec2 {
        Vec2::new(
            (self.a.x + self.b.x + self.c.x) / 3.0,
            (self.a.y + self.b.y + self.c.y) / 3.0,
        )
    }

    /// Returns the circumcircle centre and squared radius of the triangle.
    ///
    /// The circumcircle passes through all three vertices.
    /// Used in Delaunay triangulation to check if a point D is inside
    /// the circumcircle of triangle ABC (the Delaunay condition).
    ///
    /// Returns `None` if the triangle is degenerate.
    pub fn circumcircle(&self) -> Option<(Vec2, f32)> {
        let ax = self.a.x as f64; let ay = self.a.y as f64;
        let bx = self.b.x as f64; let by = self.b.y as f64;
        let cx = self.c.x as f64; let cy = self.c.y as f64;

        let d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));
        if d.abs() < 1e-10 { return None; }

        let ux = ((ax*ax + ay*ay) * (by - cy)
                + (bx*bx + by*by) * (cy - ay)
                + (cx*cx + cy*cy) * (ay - by)) / d;
        let uy = ((ax*ax + ay*ay) * (cx - bx)
                + (bx*bx + by*by) * (ax - cx)
                + (cx*cx + cy*cy) * (bx - ax)) / d;

        let centre = Vec2::new(ux as f32, uy as f32);
        let dx = centre.x - self.a.x;
        let dy = centre.y - self.a.y;
        let r2 = dx * dx + dy * dy;

        Some((centre, r2))
    }

    /// Returns `true` if point `d` lies strictly inside the circumcircle.
    ///
    /// This is the core predicate for Delaunay triangulation.
    /// Uses f64 internally to avoid precision loss near the circle boundary.
    pub fn circumcircle_contains(&self, d: Vec2) -> bool {
        // Uses the exact 4×4 determinant formulation. Sign tells us if D
        // is inside (+), on (-), or outside (0) the circumcircle of ABC
        // assuming CCW winding. We use f64 for robustness.
        let ax = (self.a.x - d.x) as f64;
        let ay = (self.a.y - d.y) as f64;
        let bx = (self.b.x - d.x) as f64;
        let by = (self.b.y - d.y) as f64;
        let cx = (self.c.x - d.x) as f64;
        let cy = (self.c.y - d.y) as f64;

        let det = ax * (by * (cx*cx + cy*cy) - cy * (bx*bx + by*by))
                - ay * (bx * (cx*cx + cy*cy) - cx * (bx*bx + by*by))
                + (ax*ax + ay*ay) * (bx * cy - by * cx);

        // For CCW winding, positive det = D inside circle.
        // For CW winding, flip.
        if self.is_ccw() { det > 0.0 } else { det < 0.0 }
    }

    // ── Conversion ────────────────────────────────────────────────────────────

    /// Lift to a 3D triangle with z = 0.
    #[inline]
    pub fn to_triangle3(&self) -> Triangle3 {
        Triangle3::new(
            self.a.extend(0.0),
            self.b.extend(0.0),
            self.c.extend(0.0),
        )
    }

    /// Flip winding order (swap B and C).
    #[inline]
    pub fn flip_winding(&self) -> Self { Self::new(self.a, self.c, self.b) }

    /// Ensure the triangle has CCW winding. Returns self unchanged if already CCW.
    #[inline]
    pub fn ensure_ccw(&self) -> Self {
        if self.is_ccw() { *self } else { self.flip_winding() }
    }
}

// ── Triangle3 ─────────────────────────────────────────────────────────────────

/// A 3D triangle with vertices A, B, C.
///
/// Used for ray-mesh intersection, normal computation, 3D picking, tangent
/// frame generation, and all 3D geometric operations on triangle surfaces.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Triangle3 {
    pub a: Vec3,
    pub b: Vec3,
    pub c: Vec3,
}

impl Triangle3 {
    #[inline(always)]
    pub const fn new(a: Vec3, b: Vec3, c: Vec3) -> Self { Self { a, b, c } }

    // ── Normal ────────────────────────────────────────────────────────────────

    /// Face normal (unnormalised). Direction follows CCW vertex order.
    ///
    /// `normal = (B - A) × (C - A)`
    #[inline]
    pub fn normal_unnormalised(&self) -> Vec3 {
        (self.b - self.a).cross(self.c - self.a)
    }

    /// Unit face normal. Returns `Vec3::ZERO` for degenerate triangles.
    #[inline]
    pub fn normal(&self) -> Vec3 { self.normal_unnormalised().normalize() }

    // ── Area ──────────────────────────────────────────────────────────────────

    /// Area of the 3D triangle. Always non-negative.
    ///
    /// `area = |AB × AC| / 2`
    #[inline]
    pub fn area(&self) -> f32 { self.normal_unnormalised().length() * 0.5 }

    /// Centroid `(A + B + C) / 3`.
    #[inline]
    pub fn centroid(&self) -> Vec3 {
        (self.a + self.b + self.c) * (1.0 / 3.0)
    }

    // ── Barycentric ───────────────────────────────────────────────────────────

    /// Compute barycentric coordinates of point `p` relative to this 3D triangle.
    ///
    /// ## Method: projection onto dominant axis
    ///
    /// We find the axis (X, Y, or Z) along which the triangle has the LARGEST
    /// projected area, then drop that component and compute 2D barycentric
    /// coordinates in the remaining 2D subspace.
    ///
    /// This is the standard "most stable axis" projection and avoids precision
    /// loss that would occur if we projected onto an axis the triangle is
    /// nearly edge-on to.
    ///
    /// Returns `None` if the triangle is degenerate (zero area).
    ///
    /// Note: `p` does NOT need to lie on the triangle's plane.
    /// If it does, the coords accurately describe membership.
    /// If it doesn't (e.g. a ray-triangle intersection point),
    /// the coords are still valid for attribute interpolation at the
    /// closest projected point.
    pub fn barycentric(&self, p: Vec3) -> Option<BarycentricCoords> {
        let n = self.normal_unnormalised();
        let ax = n.x.abs();
        let ay = n.y.abs();
        let az = n.z.abs();

        // Project onto the dominant axis plane.
        if ax >= ay && ax >= az {
            // Drop X: project onto YZ plane.
            let t = Triangle2::new(
                Vec2::new(self.a.y, self.a.z),
                Vec2::new(self.b.y, self.b.z),
                Vec2::new(self.c.y, self.c.z),
            );
            t.barycentric(Vec2::new(p.y, p.z))
        } else if ay >= ax && ay >= az {
            // Drop Y: project onto XZ plane.
            let t = Triangle2::new(
                Vec2::new(self.a.x, self.a.z),
                Vec2::new(self.b.x, self.b.z),
                Vec2::new(self.c.x, self.c.z),
            );
            t.barycentric(Vec2::new(p.x, p.z))
        } else {
            // Drop Z: project onto XY plane.
            let t = Triangle2::new(
                Vec2::new(self.a.x, self.a.y),
                Vec2::new(self.b.x, self.b.y),
                Vec2::new(self.c.x, self.c.y),
            );
            t.barycentric(Vec2::new(p.x, p.y))
        }
    }

    // ── Point containment ─────────────────────────────────────────────────────

    /// Returns `true` if point `p` (assumed to lie ON the triangle's plane)
    /// is inside or on the edge of the triangle.
    ///
    /// For points NOT on the plane, projects to the nearest point first.
    #[inline]
    pub fn contains_coplanar(&self, p: Vec3) -> bool {
        self.barycentric(p)
            .map(|b| b.is_inside_or_on_edge())
            .unwrap_or(false)
    }

    // ── Ray intersection ─────────────────────────────────────────────────────

    /// Möller–Trumbore ray-triangle intersection.
    ///
    /// Tests if the ray `origin + t·direction` hits this triangle.
    /// Returns `Some((t, bary))` where:
    ///   - `t` is the distance along the ray to the hit point (`t > 0` = forward)
    ///   - `bary` gives barycentric coordinates at the hit point
    ///
    /// Returns `None` if the ray is parallel to the triangle or misses it.
    ///
    /// `culling`:
    ///   - `true`  → back-face hits return `None` (standard for opaque meshes)
    ///   - `false` → both faces return a result (for transparent meshes)
    ///
    /// ## Usage for mesh picking
    /// ```text
    /// let (origin, dir) = picking_ray(mouse_x, mouse_y, inv_vp, viewport);
    /// let dir = dir.normalize();
    /// for tri in mesh.triangles() {
    ///     if let Some((t, bary)) = tri.ray_intersect(origin, dir, true) {
    ///         // t = distance, bary interpolates UVs / normals at hit
    ///         let uv = bary.interpolate_vec2(tri_uv_a, tri_uv_b, tri_uv_c);
    ///     }
    /// }
    /// ```
    pub fn ray_intersect(
        &self,
        origin: Vec3,
        direction: Vec3,
        culling: bool,
    ) -> Option<(f32, BarycentricCoords)> {
        const EPSILON: f32 = 1e-7;

        let ab = self.b - self.a;
        let ac = self.c - self.a;

        // h = dir × AC
        let h = direction.cross(ac);
        let det = ab.dot(h);

        // Parallel ray test. det ≈ 0 → ray is in (or parallel to) the plane.
        if culling {
            // Back-face culling: det must be positive (facing us).
            if det < EPSILON { return None; }
        } else {
            if det.abs() < EPSILON { return None; }
        }

        let inv_det = 1.0 / det;

        // u coordinate (weight for B in standard MT notation,
        // which we map to our v weight for vertex B).
        let s = origin - self.a;
        let v = s.dot(h) * inv_det;
        if !(0.0..=1.0).contains(&v) { return None; }

        // v coordinate (weight for C).
        let q = s.cross(ab);
        let w = direction.dot(q) * inv_det;
        if w < 0.0 || v + w > 1.0 { return None; }

        // t: distance along ray.
        let t = ac.dot(q) * inv_det;
        if t < EPSILON { return None; }  // intersection behind origin

        let u = 1.0 - v - w;
        Some((t, BarycentricCoords::new(u, v, w)))
    }

    // ── Closest point ─────────────────────────────────────────────────────────

    /// Find the closest point on (or inside) the triangle to point `p`.
    ///
    /// Returns `(closest_point, bary_coords)`.
    ///
    /// If `p` projects inside the triangle the result is the projection.
    /// If `p` is outside, the result is on the nearest edge or vertex.
    ///
    /// Useful for: soft-body collision, distance queries, cloth simulation.
    pub fn closest_point(&self, p: Vec3) -> (Vec3, BarycentricCoords) {
        let ab = self.b - self.a;
        let ac = self.c - self.a;
        let ap = p - self.a;

        let d1 = ab.dot(ap);
        let d2 = ac.dot(ap);

        // P projects before A in both edges — closest is A.
        if d1 <= 0.0 && d2 <= 0.0 {
            return (self.a, BarycentricCoords::VERTEX_A);
        }

        let bp = p - self.b;
        let d3 = ab.dot(bp);
        let d4 = ac.dot(bp);

        // P projects past B — closest is B.
        if d3 >= 0.0 && d4 <= d3 {
            return (self.b, BarycentricCoords::VERTEX_B);
        }

        // P projects onto edge AB.
        let vc = d1 * d4 - d3 * d2;
        if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
            let t = d1 / (d1 - d3);
            let pt = self.a + ab * t;
            return (pt, BarycentricCoords::new(1.0 - t, t, 0.0));
        }

        let cp = p - self.c;
        let d5 = ab.dot(cp);
        let d6 = ac.dot(cp);

        // P projects past C — closest is C.
        if d6 >= 0.0 && d5 <= d6 {
            return (self.c, BarycentricCoords::VERTEX_C);
        }

        // P projects onto edge AC.
        let vb = d5 * d2 - d1 * d6;
        if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
            let t = d2 / (d2 - d6);
            let pt = self.a + ac * t;
            return (pt, BarycentricCoords::new(1.0 - t, 0.0, t));
        }

        // P projects onto edge BC.
        let va = d3 * d6 - d5 * d4;
        if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
            let t = (d4 - d3) / ((d4 - d3) + (d5 - d6));
            let pt = self.b + (self.c - self.b) * t;
            return (pt, BarycentricCoords::new(0.0, 1.0 - t, t));
        }

        // P projects inside — full barycentric projection.
        let denom = 1.0 / (va + vb + vc);
        let v = vb * denom;
        let w = vc * denom;
        let u = 1.0 - v - w;
        let pt = self.a + ab * v + ac * w;
        (pt, BarycentricCoords::new(u, v, w))
    }

    // ── Plane ─────────────────────────────────────────────────────────────────

    /// The plane equation `(normal, d)` such that `dot(normal, P) + d = 0`
    /// for all points P on the triangle's plane.
    ///
    /// Returns `None` for degenerate triangles.
    #[inline]
    pub fn plane_equation(&self) -> Option<(Vec3, f32)> {
        let n = self.normal();
        if n.length_sq() < 1e-10 { return None; }
        let d = -n.dot(self.a);
        Some((n, d))
    }

    /// Signed distance from point `p` to the triangle's supporting plane.
    ///
    /// Positive = p is on the side the normal points toward.
    /// Zero     = p is on the plane.
    pub fn plane_distance(&self, p: Vec3) -> Option<f32> {
        let (n, d) = self.plane_equation()?;
        Some(n.dot(p) + d)
    }

    // ── Conversion ────────────────────────────────────────────────────────────

    /// Project to 2D by dropping the Z component.
    #[inline]
    pub fn to_triangle2_xy(&self) -> Triangle2 {
        Triangle2::new(
            Vec2::new(self.a.x, self.a.y),
            Vec2::new(self.b.x, self.b.y),
            Vec2::new(self.c.x, self.c.y),
        )
    }

    /// Flip winding order (swap B and C, negates the normal).
    #[inline]
    pub fn flip_winding(&self) -> Self { Self::new(self.a, self.c, self.b) }
}

// ── Free functions ────────────────────────────────────────────────────────────

/// Signed area of a 2D triangle defined by three points.
///
/// Positive → CCW winding.
/// Negative → CW winding.
/// Zero     → collinear (degenerate).
///
/// `|result| = area`.  Can be used directly without constructing `Triangle2`.
#[inline]
pub fn signed_area_2d(a: Vec2, b: Vec2, c: Vec2) -> f32 {
    0.5 * ((b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y))
}

/// Area of a 3D triangle defined by three points.
///
/// Uses the cross-product formula: `area = |AB × AC| / 2`.
/// Always non-negative.
#[inline]
pub fn triangle_area_3d(a: Vec3, b: Vec3, c: Vec3) -> f32 {
    (b - a).cross(c - a).length() * 0.5
}
