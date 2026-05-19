// crates/mid-math/src/ffi/geom.rs
//! C-ABI exports for geometric primitives — triangles and barycentric coords.
//!
//! C types:
//!   CBarycentricCoords — (u, v, w) weights for vertices (A, B, C)
//!   CTriangle2         — three 2D vertices
//!   CTriangle3         — three 3D vertices
//!   CCircumcircle      — center (CVec2) + squared radius
//!   CRayHit3           — t (distance), barycentric coords at hit point
//!
//! Functions that can fail return u32 (1=success, 0=failure).
//! Output parameters are only written on success.

use crate::{Vec2, Vec3};
use crate::geom::barycentric::{
    BarycentricCoords, Triangle2, Triangle3,
    signed_area_2d, triangle_area_3d,
};
use crate::ffi::float32::{CVec2, CVec3};

// ═══════════════════════════════════════════════════════════════════════════
//  C types
// ═══════════════════════════════════════════════════════════════════════════

/// Barycentric weights (u, v, w) for triangle vertices (A, B, C).
/// Valid when u + v + w ≈ 1. All ≥ 0 → point is inside or on edge.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub struct CBarycentricCoords {
    pub u: f32,
    pub v: f32,
    pub w: f32,
}

/// 2D triangle — three CCW-ordered vertices.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct CTriangle2 {
    pub a: CVec2,
    pub b: CVec2,
    pub c: CVec2,
}

/// 3D triangle — three vertices.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct CTriangle3 {
    pub a: CVec3,
    pub b: CVec3,
    pub c: CVec3,
}

/// Circumcircle result from `mid_triangle2_circumcircle`.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct CCircumcircle {
    pub center:     CVec2,
    /// Squared radius (avoid sqrt — compare with distance_sq for Delaunay tests).
    pub radius_sq:  f32,
}

/// Result of a successful 3D ray-triangle intersection.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct CRayHit3 {
    /// Distance along the ray to the hit point. Always > 0.
    pub t:    f32,
    /// Barycentric coordinates at the hit point.
    pub bary: CBarycentricCoords,
}

// ── conversions ───────────────────────────────────────────────────────────────

#[inline(always)]
fn bary_to_c(b: BarycentricCoords) -> CBarycentricCoords {
    CBarycentricCoords { u: b.u, v: b.v, w: b.w }
}

#[inline(always)]
fn tri2_from_c(t: CTriangle2) -> Triangle2 {
    Triangle2::new(Vec2::new(t.a.x, t.a.y), Vec2::new(t.b.x, t.b.y), Vec2::new(t.c.x, t.c.y))
}

#[inline(always)]
fn tri3_from_c(t: CTriangle3) -> Triangle3 {
    Triangle3::new(Vec3::from(t.a), Vec3::from(t.b), Vec3::from(t.c))
}

// ═══════════════════════════════════════════════════════════════════════════
//  BarycentricCoords — standalone interpolation
// ═══════════════════════════════════════════════════════════════════════════

/// Interpolate a f32 scalar across three vertex values.
#[no_mangle]
pub extern "C" fn mid_bary_interpolate_f32(
    bary: CBarycentricCoords, va: f32, vb: f32, vc: f32,
) -> f32 {
    BarycentricCoords::new(bary.u, bary.v, bary.w).interpolate_f32(va, vb, vc)
}

/// Interpolate a 2D vector (e.g. UV texture coordinates).
#[no_mangle]
pub extern "C" fn mid_bary_interpolate_vec2(
    bary: CBarycentricCoords, va: CVec2, vb: CVec2, vc: CVec2,
) -> CVec2 {
    let r = BarycentricCoords::new(bary.u, bary.v, bary.w)
        .interpolate_vec2(Vec2::new(va.x, va.y), Vec2::new(vb.x, vb.y), Vec2::new(vc.x, vc.y));
    CVec2 { x: r.x, y: r.y }
}

/// Interpolate a 3D vector (e.g. world-space normals, positions).
#[no_mangle]
pub extern "C" fn mid_bary_interpolate_vec3(
    bary: CBarycentricCoords, va: CVec3, vb: CVec3, vc: CVec3,
) -> CVec3 {
    let r = BarycentricCoords::new(bary.u, bary.v, bary.w)
        .interpolate_vec3(Vec3::from(va), Vec3::from(vb), Vec3::from(vc));
    CVec3::new(r.x, r.y, r.z)
}

/// Interpolate four f32 values packed as arrays (e.g. RGBA, bone weights).
/// `va`, `vb`, `vc` must each point to 4 f32 values. `out` receives 4 f32.
///
/// # Safety — va, vb, vc, out non-null, valid for 4 f32 each.
#[no_mangle]
pub unsafe extern "C" fn mid_bary_interpolate_f32x4(
    bary: CBarycentricCoords,
    va: *const f32, vb: *const f32, vc: *const f32,
    out: *mut f32,
) {
    let va = core::slice::from_raw_parts(va, 4);
    let vb = core::slice::from_raw_parts(vb, 4);
    let vc = core::slice::from_raw_parts(vc, 4);
    let out = core::slice::from_raw_parts_mut(out, 4);
    let b   = BarycentricCoords::new(bary.u, bary.v, bary.w);
    let r   = b.interpolate_f32x4(
        [va[0], va[1], va[2], va[3]],
        [vb[0], vb[1], vb[2], vb[3]],
        [vc[0], vc[1], vc[2], vc[3]],
    );
    out.copy_from_slice(&r);
}

/// True if all weights are ≥ 0 and sum ≈ 1 (valid in or on edge of triangle).
#[no_mangle]
pub extern "C" fn mid_bary_is_inside_or_on_edge(bary: CBarycentricCoords) -> bool {
    BarycentricCoords::new(bary.u, bary.v, bary.w).is_inside_or_on_edge()
}

/// True if all weights are > 0 (strictly inside triangle).
#[no_mangle]
pub extern "C" fn mid_bary_is_inside(bary: CBarycentricCoords) -> bool {
    BarycentricCoords::new(bary.u, bary.v, bary.w).is_inside()
}

/// True if the coords sum to ≈ 1 and contain no NaN/Inf.
#[no_mangle]
pub extern "C" fn mid_bary_is_valid(bary: CBarycentricCoords) -> bool {
    BarycentricCoords::new(bary.u, bary.v, bary.w).is_valid()
}

// ═══════════════════════════════════════════════════════════════════════════
//  Triangle2 — 2D triangle operations
// ═══════════════════════════════════════════════════════════════════════════

/// Compute barycentric coordinates of `p` in triangle `t`.
/// Returns 1 on success, 0 if triangle is degenerate (zero area).
///
/// # Safety — out non-null.
#[no_mangle]
pub unsafe extern "C" fn mid_triangle2_barycentric(
    t: CTriangle2, p: CVec2, out: *mut CBarycentricCoords,
) -> u32 {
    match tri2_from_c(t).barycentric(Vec2::new(p.x, p.y)) {
        Some(b) => { *out = bary_to_c(b); 1 }
        None    => 0,
    }
}

/// Returns `true` if point `p` is inside or on the boundary of triangle `t`.
/// Faster than computing full barycentric coordinates.
#[no_mangle]
pub extern "C" fn mid_triangle2_contains(t: CTriangle2, p: CVec2) -> bool {
    tri2_from_c(t).contains(Vec2::new(p.x, p.y))
}

/// Signed area of the triangle. Positive = CCW winding.
#[no_mangle]
pub extern "C" fn mid_triangle2_signed_area(t: CTriangle2) -> f32 {
    tri2_from_c(t).signed_area()
}

/// Unsigned area of the triangle.
#[no_mangle]
pub extern "C" fn mid_triangle2_area(t: CTriangle2) -> f32 {
    tri2_from_c(t).area()
}

/// Returns `true` if the vertices are in counter-clockwise order.
#[no_mangle]
pub extern "C" fn mid_triangle2_is_ccw(t: CTriangle2) -> bool {
    tri2_from_c(t).is_ccw()
}

/// Centroid of the triangle: (A + B + C) / 3.
#[no_mangle]
pub extern "C" fn mid_triangle2_centroid(t: CTriangle2) -> CVec2 {
    let c = tri2_from_c(t).centroid();
    CVec2 { x: c.x, y: c.y }
}

/// Compute the circumcircle (passes through all three vertices).
/// Returns 1 on success, 0 for degenerate triangle.
/// Used for Delaunay triangulation.
///
/// # Safety — out non-null.
#[no_mangle]
pub unsafe extern "C" fn mid_triangle2_circumcircle(
    t: CTriangle2, out: *mut CCircumcircle,
) -> u32 {
    match tri2_from_c(t).circumcircle() {
        Some((center, r2)) => {
            *out = CCircumcircle { center: CVec2 { x: center.x, y: center.y }, radius_sq: r2 };
            1
        }
        None => 0,
    }
}

/// Returns `true` if point `d` lies strictly inside the circumcircle of `t`.
/// Core Delaunay predicate — uses f64 internally for robustness.
#[no_mangle]
pub extern "C" fn mid_triangle2_circumcircle_contains(t: CTriangle2, d: CVec2) -> bool {
    tri2_from_c(t).circumcircle_contains(Vec2::new(d.x, d.y))
}

/// Return triangle with winding flipped (swap B and C).
#[no_mangle]
pub extern "C" fn mid_triangle2_flip_winding(t: CTriangle2) -> CTriangle2 {
    let f = tri2_from_c(t).flip_winding();
    CTriangle2 {
        a: CVec2 { x: f.a.x, y: f.a.y },
        b: CVec2 { x: f.b.x, y: f.b.y },
        c: CVec2 { x: f.c.x, y: f.c.y },
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Triangle3 — 3D triangle operations
// ═══════════════════════════════════════════════════════════════════════════

/// Unit face normal of the triangle. Returns ZERO for degenerate triangles.
#[no_mangle]
pub extern "C" fn mid_triangle3_normal(t: CTriangle3) -> CVec3 {
    let n = tri3_from_c(t).normal();
    CVec3::new(n.x, n.y, n.z)
}

/// Area of the 3D triangle.
#[no_mangle]
pub extern "C" fn mid_triangle3_area(t: CTriangle3) -> f32 {
    tri3_from_c(t).area()
}

/// Centroid of the 3D triangle.
#[no_mangle]
pub extern "C" fn mid_triangle3_centroid(t: CTriangle3) -> CVec3 {
    let c = tri3_from_c(t).centroid();
    CVec3::new(c.x, c.y, c.z)
}

/// Compute barycentric coordinates of `p` projected onto this 3D triangle.
/// Uses dominant-axis projection for numerical stability.
/// Returns 1 on success, 0 for degenerate triangle.
///
/// # Safety — out non-null.
#[no_mangle]
pub unsafe extern "C" fn mid_triangle3_barycentric(
    t: CTriangle3, p: CVec3, out: *mut CBarycentricCoords,
) -> u32 {
    match tri3_from_c(t).barycentric(Vec3::from(p)) {
        Some(b) => { *out = bary_to_c(b); 1 }
        None    => 0,
    }
}

/// Returns `true` if point `p` (assumed coplanar) is inside or on the triangle edge.
#[no_mangle]
pub extern "C" fn mid_triangle3_contains_coplanar(t: CTriangle3, p: CVec3) -> bool {
    tri3_from_c(t).contains_coplanar(Vec3::from(p))
}

/// Möller–Trumbore ray-triangle intersection.
///
/// `culling`: 1 = back-face culling enabled (standard for opaque meshes).
/// Returns 1 on hit, 0 on miss.
/// Writes `t` (distance) and barycentric coords to `out` on hit.
///
/// # Safety — out non-null.
#[no_mangle]
pub unsafe extern "C" fn mid_triangle3_ray_intersect(
    tri:       CTriangle3,
    origin:    CVec3,
    direction: CVec3,
    culling:   u32,
    out:       *mut CRayHit3,
) -> u32 {
    match tri3_from_c(tri).ray_intersect(Vec3::from(origin), Vec3::from(direction), culling != 0) {
        Some((t, bary)) => {
            *out = CRayHit3 { t, bary: bary_to_c(bary) };
            1
        }
        None => 0,
    }
}

/// Find the closest point on or inside the triangle to world-space point `p`.
/// Writes the closest point to `out_point` and its barycentric coords to `out_bary`.
///
/// # Safety — out_point, out_bary non-null.
#[no_mangle]
pub unsafe extern "C" fn mid_triangle3_closest_point(
    tri:       CTriangle3,
    p:         CVec3,
    out_point: *mut CVec3,
    out_bary:  *mut CBarycentricCoords,
) {
    let (pt, bary) = tri3_from_c(tri).closest_point(Vec3::from(p));
    *out_point = CVec3::new(pt.x, pt.y, pt.z);
    *out_bary  = bary_to_c(bary);
}

/// The supporting plane equation: (normal, d) such that dot(normal, P) + d = 0.
/// Returns 1 on success, 0 for degenerate triangle.
/// Writes unit normal to `out_normal` and offset to `out_d`.
///
/// # Safety — out_normal, out_d non-null.
#[no_mangle]
pub unsafe extern "C" fn mid_triangle3_plane_equation(
    tri: CTriangle3, out_normal: *mut CVec3, out_d: *mut f32,
) -> u32 {
    match tri3_from_c(tri).plane_equation() {
        Some((n, d)) => { *out_normal = CVec3::new(n.x, n.y, n.z); *out_d = d; 1 }
        None => 0,
    }
}

/// Signed distance from point `p` to the triangle's supporting plane.
/// Positive = same side as normal.
/// Returns NaN for degenerate triangles.
#[no_mangle]
pub extern "C" fn mid_triangle3_plane_distance(tri: CTriangle3, p: CVec3) -> f32 {
    tri3_from_c(tri).plane_distance(Vec3::from(p)).unwrap_or(f32::NAN)
}

/// Flip winding of a 3D triangle (negates the normal).
#[no_mangle]
pub extern "C" fn mid_triangle3_flip_winding(t: CTriangle3) -> CTriangle3 {
    let f = tri3_from_c(t).flip_winding();
    CTriangle3 {
        a: CVec3::new(f.a.x, f.a.y, f.a.z),
        b: CVec3::new(f.b.x, f.b.y, f.b.z),
        c: CVec3::new(f.c.x, f.c.y, f.c.z),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Batch ray-triangle intersection — mesh picking hot path
// ═══════════════════════════════════════════════════════════════════════════

/// Test `count` triangles against a single ray. Finds the closest hit.
/// Returns 1 if any triangle was hit, 0 otherwise.
/// Writes the closest hit triangle index to `out_index` and the hit to `out_hit`.
/// `culling`: 1 = back-face culling.
///
/// # Safety — tris, out_index, out_hit non-null; tris valid for count.
#[no_mangle]
pub unsafe extern "C" fn mid_triangle3_ray_intersect_batch(
    tris:      *const CTriangle3,
    count:     u32,
    origin:    CVec3,
    direction: CVec3,
    culling:   u32,
    out_index: *mut u32,
    out_hit:   *mut CRayHit3,
) -> u32 {
    let n    = count as usize;
    let tris = core::slice::from_raw_parts(tris, n);
    let orig = Vec3::from(origin);
    let dir  = Vec3::from(direction);
    let cull = culling != 0;

    let mut best_t: f32 = f32::MAX;
    let mut best_idx: u32 = 0;
    let mut best_hit = CRayHit3 { t: 0.0, bary: CBarycentricCoords { u: 0.0, v: 0.0, w: 0.0 } };
    let mut found = false;

    for (i, &tri) in tris.iter().enumerate() {
        if let Some((t, bary)) = tri3_from_c(tri).ray_intersect(orig, dir, cull) {
            if t < best_t {
                best_t   = t;
                best_idx = i as u32;
                best_hit = CRayHit3 { t, bary: bary_to_c(bary) };
                found    = true;
            }
        }
    }

    if found {
        *out_index = best_idx;
        *out_hit   = best_hit;
        1
    } else {
        0
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Free functions
// ═══════════════════════════════════════════════════════════════════════════

/// Signed area of a 2D triangle defined by three points.
/// Positive = CCW winding. |result| = area.
#[no_mangle]
pub extern "C" fn mid_signed_area_2d(a: CVec2, b: CVec2, c: CVec2) -> f32 {
    signed_area_2d(Vec2::new(a.x, a.y), Vec2::new(b.x, b.y), Vec2::new(c.x, c.y))
}

/// Area of a 3D triangle defined by three points. Always non-negative.
#[no_mangle]
pub extern "C" fn mid_triangle_area_3d(a: CVec3, b: CVec3, c: CVec3) -> f32 {
    triangle_area_3d(Vec3::from(a), Vec3::from(b), Vec3::from(c))
}
