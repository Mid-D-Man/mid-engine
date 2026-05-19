// crates/mid-math/src/tests/geom.rs
//! Tests for geometric primitives — BarycentricCoords, Triangle2, Triangle3.

use mid_math::{Vec2, Vec3};
use mid_math::geom::barycentric::{
    BarycentricCoords, Triangle2, Triangle3,
    signed_area_2d, triangle_area_3d,
};

fn approx(a: f32, b: f32) -> bool { (a - b).abs() < 1e-4 }
fn approx3(a: Vec3, b: Vec3) -> bool {
    (a - b).length() < 1e-4
}

// ─────────────────────────────────────────────────────────────────────────────
// BarycentricCoords — interpolation and classification
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn bary_interpolate_f32_vertex_a() {
    let b = BarycentricCoords::new(1.0, 0.0, 0.0);
    assert!(approx(b.interpolate_f32(10.0, 20.0, 30.0), 10.0));
}

#[test]
fn bary_interpolate_f32_vertex_b() {
    let b = BarycentricCoords::new(0.0, 1.0, 0.0);
    assert!(approx(b.interpolate_f32(10.0, 20.0, 30.0), 20.0));
}

#[test]
fn bary_interpolate_f32_vertex_c() {
    let b = BarycentricCoords::new(0.0, 0.0, 1.0);
    assert!(approx(b.interpolate_f32(10.0, 20.0, 30.0), 30.0));
}

#[test]
fn bary_interpolate_f32_centroid() {
    let b = BarycentricCoords::CENTROID;
    let v = b.interpolate_f32(0.0, 6.0, 12.0);
    // (0 + 6 + 12) / 3 = 6
    assert!(approx(v, 6.0), "centroid interpolation: {}", v);
}

#[test]
fn bary_interpolate_vec3_vertex() {
    let b = BarycentricCoords::new(0.0, 1.0, 0.0);
    let va = Vec3::new(1.0, 0.0, 0.0);
    let vb = Vec3::new(0.0, 1.0, 0.0);
    let vc = Vec3::new(0.0, 0.0, 1.0);
    assert!(approx3(b.interpolate_vec3(va, vb, vc), vb));
}

#[test]
fn bary_is_inside_true_for_centroid() {
    assert!(BarycentricCoords::CENTROID.is_inside());
}

#[test]
fn bary_is_inside_false_for_outside() {
    let outside = BarycentricCoords::new(-0.1, 0.6, 0.5);
    assert!(!outside.is_inside());
    assert!(!outside.is_inside_or_on_edge());
}

#[test]
fn bary_on_edge_has_one_zero_weight() {
    // Point on edge BC: u=0.
    let edge = BarycentricCoords::new(0.0, 0.5, 0.5);
    assert!(edge.is_inside_or_on_edge());
    assert!(edge.is_on_edge());
    assert!(!edge.is_inside());
}

#[test]
fn bary_is_valid_for_centroid() {
    assert!(BarycentricCoords::CENTROID.is_valid());
}

#[test]
fn bary_is_valid_false_for_nan() {
    let bad = BarycentricCoords::new(f32::NAN, 0.0, 0.0);
    assert!(!bad.is_valid());
}

// ─────────────────────────────────────────────────────────────────────────────
// Triangle2 — barycentric
// ─────────────────────────────────────────────────────────────────────────────

fn unit_tri2() -> Triangle2 {
    Triangle2::new(Vec2::new(0.0, 0.0), Vec2::new(1.0, 0.0), Vec2::new(0.0, 1.0))
}

#[test]
fn triangle2_bary_at_vertex_a() {
    let b = unit_tri2().barycentric(Vec2::new(0.0, 0.0)).unwrap();
    assert!(approx(b.u, 1.0) && approx(b.v, 0.0) && approx(b.w, 0.0),
        "vertex A bary: u={} v={} w={}", b.u, b.v, b.w);
}

#[test]
fn triangle2_bary_at_vertex_b() {
    let b = unit_tri2().barycentric(Vec2::new(1.0, 0.0)).unwrap();
    assert!(approx(b.v, 1.0) && approx(b.u, 0.0) && approx(b.w, 0.0),
        "vertex B bary: u={} v={} w={}", b.u, b.v, b.w);
}

#[test]
fn triangle2_bary_at_vertex_c() {
    let b = unit_tri2().barycentric(Vec2::new(0.0, 1.0)).unwrap();
    assert!(approx(b.w, 1.0) && approx(b.u, 0.0) && approx(b.v, 0.0),
        "vertex C bary: u={} v={} w={}", b.u, b.v, b.w);
}

#[test]
fn triangle2_bary_at_centroid() {
    let c = unit_tri2().centroid();
    let b = unit_tri2().barycentric(c).unwrap();
    assert!(approx(b.u, 1.0 / 3.0) && approx(b.v, 1.0 / 3.0) && approx(b.w, 1.0 / 3.0),
        "centroid bary: u={} v={} w={}", b.u, b.v, b.w);
}

#[test]
fn triangle2_bary_outside_returns_negative_weight() {
    let b = unit_tri2().barycentric(Vec2::new(2.0, 2.0)).unwrap();
    assert!(!b.is_inside_or_on_edge(), "External point should have a negative weight");
}

#[test]
fn triangle2_bary_degenerate_returns_none() {
    // Collinear triangle — zero area.
    let degenerate = Triangle2::new(Vec2::new(0.0, 0.0), Vec2::new(1.0, 0.0), Vec2::new(2.0, 0.0));
    assert!(degenerate.barycentric(Vec2::new(0.5, 0.0)).is_none());
}

#[test]
fn triangle2_bary_weights_sum_to_one() {
    for i in 0..10 {
        let p = Vec2::new(i as f32 * 0.05 + 0.1, 0.1);
        if let Some(b) = unit_tri2().barycentric(p) {
            assert!(approx(b.u + b.v + b.w, 1.0), "weights sum = {}", b.u + b.v + b.w);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Triangle2 — contains
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn triangle2_contains_centroid() {
    assert!(unit_tri2().contains(unit_tri2().centroid()));
}

#[test]
fn triangle2_contains_vertex() {
    assert!(unit_tri2().contains(Vec2::new(0.0, 0.0)));
}

#[test]
fn triangle2_not_contains_outside_point() {
    assert!(!unit_tri2().contains(Vec2::new(1.0, 1.0)));
}

#[test]
fn triangle2_not_contains_far_outside() {
    assert!(!unit_tri2().contains(Vec2::new(10.0, -5.0)));
}

// ─────────────────────────────────────────────────────────────────────────────
// Triangle2 — area, winding, centroid
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn triangle2_area_unit_triangle() {
    // Right triangle with legs of length 1 → area = 0.5.
    assert!(approx(unit_tri2().area(), 0.5));
}

#[test]
fn triangle2_signed_area_positive_ccw() {
    assert!(unit_tri2().signed_area() > 0.0, "CCW triangle should have positive signed area");
}

#[test]
fn triangle2_signed_area_negative_cw() {
    let cw = unit_tri2().flip_winding();
    assert!(cw.signed_area() < 0.0, "CW triangle should have negative signed area");
}

#[test]
fn triangle2_area_equals_abs_signed_area() {
    let t = unit_tri2();
    assert!(approx(t.area(), t.signed_area().abs()));
}

#[test]
fn triangle2_centroid_correct() {
    let c = unit_tri2().centroid();
    // centroid of (0,0) (1,0) (0,1) = (1/3, 1/3)
    assert!(approx(c.x, 1.0 / 3.0) && approx(c.y, 1.0 / 3.0),
        "centroid: ({}, {})", c.x, c.y);
}

#[test]
fn triangle2_flip_winding_reverses_area_sign() {
    let t = unit_tri2();
    let f = t.flip_winding();
    assert!(approx(t.signed_area(), -f.signed_area()));
}

// ─────────────────────────────────────────────────────────────────────────────
// Triangle2 — circumcircle (Delaunay)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn triangle2_circumcircle_passes_through_vertices() {
    let t = unit_tri2();
    let (center, r2) = t.circumcircle().expect("non-degenerate triangle");
    for v in [t.a, t.b, t.c] {
        let d2 = (Vec2::new(center.x, center.y) - v).length_sq();
        assert!(approx(d2, r2), "Vertex {:?} not on circumcircle: d2={} r2={}", v, d2, r2);
    }
}

#[test]
fn triangle2_circumcircle_contains_interior_point() {
    // Interior point of unit triangle should be inside circumcircle.
    let t = unit_tri2();
    assert!(t.circumcircle_contains(t.centroid()), "Centroid should be inside circumcircle");
}

#[test]
fn triangle2_circumcircle_does_not_contain_far_point() {
    let t = unit_tri2();
    assert!(!t.circumcircle_contains(Vec2::new(100.0, 100.0)));
}

#[test]
fn triangle2_degenerate_circumcircle_returns_none() {
    let degenerate = Triangle2::new(Vec2::new(0.0, 0.0), Vec2::new(1.0, 0.0), Vec2::new(2.0, 0.0));
    assert!(degenerate.circumcircle().is_none());
}

// ─────────────────────────────────────────────────────────────────────────────
// Triangle3 — normal and area
// ─────────────────────────────────────────────────────────────────────────────

fn unit_tri3() -> Triangle3 {
    Triangle3::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0))
}

#[test]
fn triangle3_normal_is_z_axis_for_xy_triangle() {
    let n = unit_tri3().normal();
    // CCW triangle in XY plane → normal should be +Z.
    assert!(approx(n.x, 0.0) && approx(n.y, 0.0) && approx(n.z, 1.0),
        "normal: ({}, {}, {})", n.x, n.y, n.z);
}

#[test]
fn triangle3_normal_is_unit_length() {
    let n = unit_tri3().normal();
    assert!(approx(n.length(), 1.0), "normal length: {}", n.length());
}

#[test]
fn triangle3_area_is_half() {
    // Same shape as unit_tri2: area = 0.5.
    assert!(approx(unit_tri3().area(), 0.5));
}

#[test]
fn triangle3_centroid_correct() {
    let c = unit_tri3().centroid();
    assert!(approx(c.x, 1.0 / 3.0) && approx(c.y, 1.0 / 3.0) && approx(c.z, 0.0));
}

// ─────────────────────────────────────────────────────────────────────────────
// Triangle3 — barycentric
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn triangle3_bary_at_vertex_a() {
    let b = unit_tri3().barycentric(Vec3::ZERO).unwrap();
    assert!(approx(b.u, 1.0) && approx(b.v, 0.0) && approx(b.w, 0.0));
}

#[test]
fn triangle3_bary_weights_sum_to_one() {
    let p = Vec3::new(0.25, 0.25, 0.0);
    let b = unit_tri3().barycentric(p).unwrap();
    assert!(approx(b.u + b.v + b.w, 1.0), "sum={}", b.u + b.v + b.w);
}

#[test]
fn triangle3_bary_inside_for_interior_point() {
    let p = Vec3::new(0.2, 0.2, 0.0);
    let b = unit_tri3().barycentric(p).unwrap();
    assert!(b.is_inside_or_on_edge(), "Interior point should have non-negative weights");
}

// ─────────────────────────────────────────────────────────────────────────────
// Triangle3 — ray intersection (Möller–Trumbore)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn triangle3_ray_hit_from_above() {
    let tri    = unit_tri3();
    let origin = Vec3::new(0.2, 0.2, 5.0);
    let dir    = Vec3::new(0.0, 0.0, -1.0);

    let result = tri.ray_intersect(origin, dir, false);
    assert!(result.is_some(), "Ray from above should hit XY triangle");
    let (t, bary) = result.unwrap();
    assert!(t > 0.0, "t must be positive: {}", t);
    assert!(bary.is_inside_or_on_edge(), "Hit bary must be inside");
}

#[test]
fn triangle3_ray_miss_outside_triangle() {
    let tri    = unit_tri3();
    let origin = Vec3::new(5.0, 5.0, 5.0);  // Far outside the triangle footprint.
    let dir    = Vec3::new(0.0, 0.0, -1.0);
    assert!(tri.ray_intersect(origin, dir, false).is_none(), "Ray outside footprint should miss");
}

#[test]
fn triangle3_ray_miss_parallel_to_plane() {
    let tri    = unit_tri3();
    let origin = Vec3::new(0.2, 0.2, 0.5);
    let dir    = Vec3::new(1.0, 0.0, 0.0);  // Parallel to XY plane.
    assert!(tri.ray_intersect(origin, dir, false).is_none(), "Parallel ray should miss");
}

#[test]
fn triangle3_ray_miss_behind_origin() {
    let tri    = unit_tri3();
    let origin = Vec3::new(0.2, 0.2, -5.0);  // Below the XY plane.
    let dir    = Vec3::new(0.0, 0.0, -1.0);  // Going further away.
    assert!(tri.ray_intersect(origin, dir, false).is_none(), "Ray going away should miss");
}

#[test]
fn triangle3_ray_back_face_culling_rejects_back_hit() {
    let tri    = unit_tri3();
    // Ray from below hitting the back face.
    let origin = Vec3::new(0.2, 0.2, -5.0);
    let dir    = Vec3::new(0.0, 0.0, 1.0);
    // Without culling: should hit.
    assert!(tri.ray_intersect(origin, dir, false).is_some(), "No culling: back hit should register");
    // With culling: should miss (back face).
    assert!(tri.ray_intersect(origin, dir, true).is_none(), "Culling: back face hit should be rejected");
}

#[test]
fn triangle3_ray_hit_t_equals_geometric_distance() {
    let tri    = unit_tri3();
    let height = 3.0f32;
    let origin = Vec3::new(0.2, 0.2, height);
    let dir    = Vec3::new(0.0, 0.0, -1.0);

    let (t, _) = tri.ray_intersect(origin, dir, false).unwrap();
    // Triangle is in XY plane (z=0), origin is at z=height → t should equal height.
    assert!((t - height).abs() < 1e-4, "t={} expected {}", t, height);
}

// ─────────────────────────────────────────────────────────────────────────────
// Triangle3 — closest point
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn triangle3_closest_point_inside_projects_to_plane() {
    let tri = unit_tri3();
    let p   = Vec3::new(0.2, 0.2, 5.0);  // Above interior point.
    let (closest, bary) = tri.closest_point(p);
    // Closest point should be on the XY plane.
    assert!(approx(closest.z, 0.0), "Closest point z should be 0: {}", closest.z);
    // Bary should be interior.
    assert!(bary.is_inside_or_on_edge());
}

#[test]
fn triangle3_closest_point_to_vertex_a_is_vertex_a() {
    let tri = unit_tri3();
    // Point far beyond vertex A.
    let p   = Vec3::new(-10.0, -10.0, 0.0);
    let (closest, bary) = tri.closest_point(p);
    assert!(approx3(closest, Vec3::ZERO), "Closest to far-outside should be vertex A: {:?}", closest);
    assert!(approx(bary.u, 1.0), "Bary u should be 1 at vertex A: {:?}", bary);
}

#[test]
fn triangle3_closest_point_on_edge() {
    let tri = unit_tri3();
    // Point directly above midpoint of edge AB.
    let mid = Vec3::new(0.5, 0.0, 3.0);
    let (closest, bary) = tri.closest_point(mid);
    assert!(approx(closest.x, 0.5), "Closest x: {}", closest.x);
    assert!(approx(closest.y, 0.0), "Closest y: {}", closest.y);
    assert!(approx(bary.w, 0.0), "On edge AB: w should be 0, got {}", bary.w);
}

// ─────────────────────────────────────────────────────────────────────────────
// Triangle3 — plane
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn triangle3_plane_equation_correct() {
    let (n, d) = unit_tri3().plane_equation().unwrap();
    // Normal is +Z, offset d = -dot(Z, origin) = 0.
    assert!(approx(n.z, 1.0) && approx(n.x, 0.0) && approx(n.y, 0.0));
    assert!(approx(d, 0.0), "d for XY-plane triangle: {}", d);
}

#[test]
fn triangle3_plane_distance_on_plane_is_zero() {
    let tri = unit_tri3();
    let d   = tri.plane_distance(Vec3::new(0.3, 0.3, 0.0)).unwrap();
    assert!(approx(d, 0.0), "Point on plane should have distance 0, got {}", d);
}

#[test]
fn triangle3_plane_distance_above_is_positive() {
    let tri = unit_tri3();
    let d   = tri.plane_distance(Vec3::new(0.3, 0.3, 2.0)).unwrap();
    assert!(d > 0.0, "Point above XY plane should have positive distance, got {}", d);
}

#[test]
fn triangle3_plane_distance_below_is_negative() {
    let tri = unit_tri3();
    let d   = tri.plane_distance(Vec3::new(0.3, 0.3, -2.0)).unwrap();
    assert!(d < 0.0, "Point below XY plane should have negative distance, got {}", d);
}

// ─────────────────────────────────────────────────────────────────────────────
// Free functions
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn signed_area_2d_ccw_positive() {
    let a = Vec2::new(0.0, 0.0);
    let b = Vec2::new(1.0, 0.0);
    let c = Vec2::new(0.0, 1.0);
    let area = signed_area_2d(a, b, c);
    assert!(area > 0.0, "CCW signed area should be positive: {}", area);
    assert!(approx(area, 0.5));
}

#[test]
fn signed_area_2d_cw_negative() {
    let a = Vec2::new(0.0, 0.0);
    let b = Vec2::new(0.0, 1.0);
    let c = Vec2::new(1.0, 0.0);
    assert!(signed_area_2d(a, b, c) < 0.0, "CW signed area should be negative");
}

#[test]
fn triangle_area_3d_correct() {
    let a = Vec3::new(0.0, 0.0, 0.0);
    let b = Vec3::new(2.0, 0.0, 0.0);
    let c = Vec3::new(0.0, 2.0, 0.0);
    let area = triangle_area_3d(a, b, c);
    // Right triangle legs 2: area = 0.5 * 2 * 2 = 2.
    assert!(approx(area, 2.0), "area: {}", area);
}

#[test]
fn triangle_area_3d_degenerate_is_zero() {
    let a = Vec3::new(0.0, 0.0, 0.0);
    let b = Vec3::new(1.0, 0.0, 0.0);
    let c = Vec3::new(2.0, 0.0, 0.0);  // Collinear.
    let area = triangle_area_3d(a, b, c);
    assert!(approx(area, 0.0), "Degenerate triangle area: {}", area);
      }
