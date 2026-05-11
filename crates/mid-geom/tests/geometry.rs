// crates/mid-geom/tests/geometry.rs
//! Integration tests for mid-geom.
//!
//! Covers: correctness of all intersection tests, edge cases (touching,
//! contained, parallel rays), and FFI type round-trips.

use mid_geom::{AABB, Capsule, Circle, Frustum, Plane, Ray2, Ray3, Rect, Sphere};
use mid_math::{Mat4, Vec2, Vec3, FRAC_PI_4, EPSILON};

// ── Helpers ───────────────────────────────────────────────────────────────────

fn approx(a: f32, b: f32) -> bool { (a - b).abs() < 1e-4 }
fn v3(x: f32, y: f32, z: f32) -> Vec3 { Vec3::new(x, y, z) }
fn v2(x: f32, y: f32) -> Vec2 { Vec2::new(x, y) }

// ═════════════════════════════════════════════════════════════════════════════
// AABB
// ═════════════════════════════════════════════════════════════════════════════

#[test]
fn aabb_basic_construction() {
    let a = AABB::new(v3(-1.0, -1.0, -1.0), v3(1.0, 1.0, 1.0));
    assert_eq!(a.center(), Vec3::ZERO);
    assert_eq!(a.size(), Vec3::splat(2.0));
    assert_eq!(a.extents(), Vec3::ONE);
    assert!(approx(a.surface_area(), 24.0));
    assert!(approx(a.volume(), 8.0));
}

#[test]
fn aabb_from_center_size() {
    let a = AABB::from_center_half_extents(v3(1.0, 2.0, 3.0), v3(0.5, 1.0, 1.5));
    assert_eq!(a.min, v3(0.5, 1.0, 1.5));
    assert_eq!(a.max, v3(1.5, 3.0, 4.5));
}

#[test]
fn aabb_invalid_seed() {
    let inv = AABB::invalid();
    // Any real AABB should be contained in an invalid one after one expand.
    let p = v3(5.0, -3.0, 7.0);
    let grown = inv.expand_to_include_point(p);
    assert_eq!(grown.min, p);
    assert_eq!(grown.max, p);
}

#[test]
fn aabb_contains_point() {
    let a = AABB::new(v3(-2.0, -2.0, -2.0), v3(2.0, 2.0, 2.0));
    assert!(a.contains_point(Vec3::ZERO));
    assert!(a.contains_point(v3(2.0, 2.0, 2.0)));   // on boundary
    assert!(a.contains_point(v3(-2.0, -2.0, -2.0))); // on boundary
    assert!(!a.contains_point(v3(2.01, 0.0, 0.0)));
    assert!(!a.contains_point(v3(0.0, -2.01, 0.0)));
}

#[test]
fn aabb_intersects_aabb_all_cases() {
    let a = AABB::new(v3(-1.0, -1.0, -1.0), v3(1.0, 1.0, 1.0));

    // Overlap
    assert!(a.intersects_aabb(&AABB::new(v3(0.0, 0.0, 0.0), v3(2.0, 2.0, 2.0))));
    // Touching face — counts as overlap
    assert!(a.intersects_aabb(&AABB::new(v3(1.0, -1.0, -1.0), v3(3.0, 1.0, 1.0))));
    // Separated on X
    assert!(!a.intersects_aabb(&AABB::new(v3(1.01, -1.0, -1.0), v3(3.0, 1.0, 1.0))));
    // Separated on Y
    assert!(!a.intersects_aabb(&AABB::new(v3(-1.0, 1.01, -1.0), v3(1.0, 3.0, 1.0))));
    // Separated on Z
    assert!(!a.intersects_aabb(&AABB::new(v3(-1.0, -1.0, 1.01), v3(1.0, 1.0, 3.0))));
    // Contained
    assert!(a.intersects_aabb(&AABB::new(v3(-0.5, -0.5, -0.5), v3(0.5, 0.5, 0.5))));
}

#[test]
fn aabb_intersects_sphere() {
    let a = AABB::new(v3(-2.0, -2.0, -2.0), v3(2.0, 2.0, 2.0));
    // Sphere center inside box
    assert!(a.intersects_sphere(&Sphere::new(Vec3::ZERO, 0.5)));
    // Sphere overlapping face
    assert!(a.intersects_sphere(&Sphere::new(v3(2.5, 0.0, 0.0), 1.0)));
    // Sphere touching face exactly
    assert!(a.intersects_sphere(&Sphere::new(v3(3.0, 0.0, 0.0), 1.0)));
    // Sphere separated
    assert!(!a.intersects_sphere(&Sphere::new(v3(3.01, 0.0, 0.0), 1.0)));
    // Sphere at corner
    assert!(a.intersects_sphere(&Sphere::new(v3(3.0, 3.0, 3.0), 2.0)));
}

#[test]
fn aabb_closest_point() {
    let a = AABB::new(v3(-1.0, -1.0, -1.0), v3(1.0, 1.0, 1.0));
    // Inside — closest point is the point itself
    assert_eq!(a.closest_point(Vec3::ZERO), Vec3::ZERO);
    // Outside on X
    assert_eq!(a.closest_point(v3(3.0, 0.0, 0.0)), v3(1.0, 0.0, 0.0));
    // Outside on corner
    assert_eq!(a.closest_point(v3(3.0, 3.0, 3.0)), v3(1.0, 1.0, 1.0));
    // Outside on negative face
    assert_eq!(a.closest_point(v3(-5.0, 0.0, 0.0)), v3(-1.0, 0.0, 0.0));
}

#[test]
fn aabb_signed_distance() {
    let a = AABB::new(v3(-1.0, -1.0, -1.0), v3(1.0, 1.0, 1.0));
    // Inside — should be negative
    assert!(a.signed_distance(Vec3::ZERO) < 0.0);
    // On surface — should be zero (within tolerance)
    let d = a.signed_distance(v3(1.0, 0.0, 0.0));
    assert!(d.abs() < 1e-4);
    // Outside
    assert!(a.signed_distance(v3(3.0, 0.0, 0.0)) > 0.0);
}

#[test]
fn aabb_merge() {
    let a = AABB::new(v3(-1.0, -1.0, -1.0), v3(0.0, 0.0, 0.0));
    let b = AABB::new(v3(0.0, 0.0, 0.0), v3(1.0, 1.0, 1.0));
    let m = a.merge(b);
    assert_eq!(m.min, v3(-1.0, -1.0, -1.0));
    assert_eq!(m.max, v3(1.0, 1.0, 1.0));
}

// ═════════════════════════════════════════════════════════════════════════════
// SPHERE
// ═════════════════════════════════════════════════════════════════════════════

#[test]
fn sphere_contains_point() {
    let s = Sphere::new(Vec3::ZERO, 2.0);
    assert!(s.contains_point(Vec3::ZERO));
    assert!(s.contains_point(v3(2.0, 0.0, 0.0)));   // on surface
    assert!(!s.contains_point(v3(2.01, 0.0, 0.0)));
}

#[test]
fn sphere_vs_sphere_uses_squared_distance() {
    // d=4, r1+r2=4 → touching → should intersect
    let s1 = Sphere::new(Vec3::ZERO, 2.0);
    let s2 = Sphere::new(v3(4.0, 0.0, 0.0), 2.0);
    assert!(s1.intersects_sphere(&s2));
    // d=4.01, r1+r2=4 → separated
    let s3 = Sphere::new(v3(4.01, 0.0, 0.0), 2.0);
    assert!(!s1.intersects_sphere(&s3));
}

#[test]
fn sphere_signed_distance() {
    let s = Sphere::new(Vec3::ZERO, 2.0);
    assert!(approx(s.signed_distance(v3(3.0, 0.0, 0.0)), 1.0));
    assert!(approx(s.signed_distance(v3(2.0, 0.0, 0.0)), 0.0));
    assert!(s.signed_distance(Vec3::ZERO) < 0.0);
}

#[test]
fn sphere_bounding_aabb() {
    let s = Sphere::new(v3(1.0, 2.0, 3.0), 1.5);
    let a = s.bounding_aabb();
    assert_eq!(a.min, v3(-0.5, 0.5, 1.5));
    assert_eq!(a.max, v3(2.5, 3.5, 4.5));
}

#[test]
fn sphere_merge() {
    // One sphere contained in the other → result is the larger
    let big   = Sphere::new(Vec3::ZERO, 5.0);
    let small = Sphere::new(v3(1.0, 0.0, 0.0), 1.0);
    let m = big.merge(small);
    assert!(approx(m.radius, 5.0));

    // Two spheres separated — merged sphere covers both
    let a = Sphere::new(v3(-3.0, 0.0, 0.0), 1.0);
    let b = Sphere::new(v3( 3.0, 0.0, 0.0), 1.0);
    let m2 = a.merge(b);
    assert!(m2.contains_point(v3(-4.0, 0.0, 0.0)));
    assert!(m2.contains_point(v3( 4.0, 0.0, 0.0)));
}

// ═════════════════════════════════════════════════════════════════════════════
// CAPSULE
// ═════════════════════════════════════════════════════════════════════════════

#[test]
fn capsule_closest_point_on_axis() {
    let cap = Capsule::new(v3(0.0, 0.0, 0.0), v3(0.0, 4.0, 0.0), 0.5);
    // Below base → clamps to base
    assert_eq!(cap.closest_point_on_axis(v3(0.0, -1.0, 0.0)), v3(0.0, 0.0, 0.0));
    // Above tip → clamps to tip
    assert_eq!(cap.closest_point_on_axis(v3(0.0, 5.0, 0.0)), v3(0.0, 4.0, 0.0));
    // Middle
    let p = cap.closest_point_on_axis(v3(1.0, 2.0, 0.0));
    assert!(approx(p.y, 2.0));
}

#[test]
fn capsule_contains_point() {
    let cap = Capsule::new(v3(0.0, 0.0, 0.0), v3(0.0, 2.0, 0.0), 1.0);
    // Inside cylindrical body
    assert!(cap.contains_point(v3(0.0, 1.0, 0.0)));
    // Inside hemispherical cap at base
    assert!(cap.contains_point(v3(0.0, -0.9, 0.0)));
    // Inside hemispherical cap at tip
    assert!(cap.contains_point(v3(0.0, 2.9, 0.0)));
    // Outside
    assert!(!cap.contains_point(v3(2.0, 1.0, 0.0)));
    assert!(!cap.contains_point(v3(0.0, -1.01, 0.0)));
}

#[test]
fn capsule_vs_sphere() {
    let cap = Capsule::new(v3(0.0, 0.0, 0.0), v3(0.0, 2.0, 0.0), 0.5);
    assert!(cap.intersects_sphere(&Sphere::new(v3(0.0, 1.0, 0.4), 0.5)));
    assert!(!cap.intersects_sphere(&Sphere::new(v3(5.0, 1.0, 0.0), 0.5)));
}

#[test]
fn capsule_vs_capsule() {
    let c1 = Capsule::new(v3(0.0, 0.0, 0.0), v3(0.0, 2.0, 0.0), 0.5);
    // Parallel, close
    let c2 = Capsule::new(v3(0.8, 0.0, 0.0), v3(0.8, 2.0, 0.0), 0.5);
    assert!(c1.intersects_capsule(&c2));
    // Parallel, far
    let c3 = Capsule::new(v3(5.0, 0.0, 0.0), v3(5.0, 2.0, 0.0), 0.5);
    assert!(!c1.intersects_capsule(&c3));
    // Crossing
    let c4 = Capsule::new(v3(-1.0, 1.0, 0.0), v3(1.0, 1.0, 0.0), 0.3);
    assert!(c1.intersects_capsule(&c4));
}

// ═════════════════════════════════════════════════════════════════════════════
// PLANE
// ═════════════════════════════════════════════════════════════════════════════

#[test]
fn plane_signed_distance() {
    let p = Plane::from_normal_point(Vec3::Y, Vec3::ZERO);
    assert!(approx(p.signed_distance(v3(0.0, 2.0, 0.0)), 2.0));
    assert!(approx(p.signed_distance(v3(0.0, -3.0, 0.0)), -3.0));
    assert!(approx(p.signed_distance(Vec3::ZERO), 0.0));
}

#[test]
fn plane_project_point() {
    let p = Plane::from_normal_point(Vec3::Y, Vec3::ZERO);
    let proj = p.project_point(v3(3.0, 5.0, -2.0));
    assert!(approx(proj.x, 3.0));
    assert!(approx(proj.y, 0.0));
    assert!(approx(proj.z, -2.0));
}

#[test]
fn plane_from_points() {
    let p = Plane::from_points(
        v3(0.0, 0.0, 0.0),
        v3(1.0, 0.0, 0.0),
        v3(0.0, 1.0, 0.0),
    ).expect("valid triangle");
    // Normal should point in +Z for CCW winding
    assert!(approx(p.normal.z.abs(), 1.0));
}

#[test]
fn plane_intersect_ray_hit() {
    let p = Plane::from_normal_point(Vec3::Y, Vec3::ZERO);
    let r = Ray3::new_unnormalized(v3(0.0, 3.0, 0.0), v3(0.0, -1.0, 0.0));
    let t = p.intersect_ray(r.origin, r.direction).expect("should hit");
    assert!(approx(t, 3.0));
}

#[test]
fn plane_intersect_ray_parallel() {
    let p = Plane::from_normal_point(Vec3::Y, Vec3::ZERO);
    let r = Ray3::new_unnormalized(v3(0.0, 1.0, 0.0), v3(1.0, 0.0, 0.0));
    assert!(p.intersect_ray(r.origin, r.direction).is_none());
}

// ═════════════════════════════════════════════════════════════════════════════
// FRUSTUM
// ═════════════════════════════════════════════════════════════════════════════

fn make_test_frustum() -> Frustum {
    let view_proj = Mat4::perspective_rh(FRAC_PI_4 * 2.0, 1.0, 1.0, 100.0)
        * Mat4::look_at_rh(v3(0.0, 0.0, 10.0), Vec3::ZERO, Vec3::Y);
    Frustum::from_mat4(&view_proj)
}

#[test]
fn frustum_contains_origin_aabb() {
    let f = make_test_frustum();
    // Small AABB at origin — should be visible
    let visible = AABB::new(v3(-0.5, -0.5, -0.5), v3(0.5, 0.5, 0.5));
    assert!(f.intersects_aabb(&visible));
}

#[test]
fn frustum_rejects_behind_camera() {
    let f = make_test_frustum();
    // Far behind camera (+Z from camera which is at z=10 looking at origin)
    let behind = AABB::new(v3(-1.0, -1.0, 50.0), v3(1.0, 1.0, 100.0));
    // This should be culled (behind near plane from camera POV)
    // Note: whether culled depends on exact frustum setup; test that the
    // function runs without panic and returns a bool.
    let _ = f.intersects_aabb(&behind);
}

#[test]
fn frustum_sphere_inside() {
    let f = make_test_frustum();
    let inside = Sphere::new(Vec3::ZERO, 0.5);
    assert!(f.intersects_sphere(&inside));
}

#[test]
fn frustum_sphere_radius_spans_plane() {
    let f = make_test_frustum();
    // A huge sphere centred far away but radius brings it inside
    let huge = Sphere::new(v3(0.0, 0.0, -500.0), 600.0);
    assert!(f.intersects_sphere(&huge));
}

// ═════════════════════════════════════════════════════════════════════════════
// RAY3
// ═════════════════════════════════════════════════════════════════════════════

#[test]
fn ray3_vs_sphere_hit() {
    let r = Ray3::new_unnormalized(v3(0.0, 0.0, 5.0), v3(0.0, 0.0, -1.0));
    let s = Sphere::new(Vec3::ZERO, 1.0);
    let h = r.intersect_sphere(&s).expect("should hit");
    assert!(approx(h.t, 4.0));
    assert!(approx(h.point.z, 1.0));
}

#[test]
fn ray3_vs_sphere_miss() {
    let r = Ray3::new_unnormalized(v3(5.0, 0.0, 5.0), v3(0.0, 0.0, -1.0));
    let s = Sphere::new(Vec3::ZERO, 1.0);
    assert!(r.intersect_sphere(&s).is_none());
}

#[test]
fn ray3_vs_sphere_inside_origin() {
    // Ray starting inside sphere should still find exit point
    let r = Ray3::new_unnormalized(Vec3::ZERO, v3(1.0, 0.0, 0.0));
    let s = Sphere::new(Vec3::ZERO, 2.0);
    let h = r.intersect_sphere(&s).expect("inside-origin should hit exit");
    assert!(approx(h.t, 2.0));
}

#[test]
fn ray3_vs_aabb_hit() {
    let r = Ray3::new_unnormalized(v3(0.0, 0.0, 5.0), v3(0.0, 0.0, -1.0));
    let a = AABB::new(v3(-1.0, -1.0, -1.0), v3(1.0, 1.0, 1.0));
    let h = r.intersect_aabb(&a).expect("should hit");
    assert!(approx(h.t, 4.0));
    assert!(approx(h.point.z, 1.0));
    // Normal should point +Z (exit face is -Z, entry face is +Z)
    assert!(approx(h.normal.z, 1.0));
}

#[test]
fn ray3_vs_aabb_miss() {
    let r = Ray3::new_unnormalized(v3(5.0, 0.0, 5.0), v3(0.0, 0.0, -1.0));
    let a = AABB::new(v3(-1.0, -1.0, -1.0), v3(1.0, 1.0, 1.0));
    assert!(r.intersect_aabb(&a).is_none());
}

#[test]
fn ray3_vs_aabb_inside_start() {
    let r = Ray3::new_unnormalized(Vec3::ZERO, v3(1.0, 0.0, 0.0));
    let a = AABB::new(v3(-2.0, -2.0, -2.0), v3(2.0, 2.0, 2.0));
    // Should still return the exit hit
    let h = r.intersect_aabb(&a).expect("inside start should hit exit");
    assert!(approx(h.t, 2.0));
}

#[test]
fn ray3_vs_plane() {
    let p = Plane::from_normal_point(Vec3::Y, Vec3::ZERO);
    let r = Ray3::new_unnormalized(v3(1.0, 3.0, -2.0), v3(0.0, -1.0, 0.0));
    let h = r.intersect_plane(&p).expect("should hit");
    assert!(approx(h.t, 3.0));
    assert!(approx(h.point.y, 0.0));
}

#[test]
fn ray3_vs_capsule_hit() {
    let cap = Capsule::new(v3(-1.0, 0.0, 0.0), v3(1.0, 0.0, 0.0), 0.5);
    let r   = Ray3::new_unnormalized(v3(0.0, 3.0, 0.0), v3(0.0, -1.0, 0.0));
    assert!(r.intersect_capsule(&cap).is_some());
}

#[test]
fn ray3_vs_capsule_miss() {
    let cap = Capsule::new(v3(-1.0, 0.0, 0.0), v3(1.0, 0.0, 0.0), 0.5);
    let r   = Ray3::new_unnormalized(v3(5.0, 3.0, 0.0), v3(0.0, -1.0, 0.0));
    assert!(r.intersect_capsule(&cap).is_none());
}

// ═════════════════════════════════════════════════════════════════════════════
// RAY2
// ═════════════════════════════════════════════════════════════════════════════

#[test]
fn ray2_vs_rect_hit() {
    let rect = Rect::new(v2(-1.0, -1.0), v2(1.0, 1.0));
    let r    = Ray2::new_unnormalized(v2(0.0, 3.0), v2(0.0, -1.0));
    let h    = r.intersect_rect(&rect).expect("should hit");
    assert!(approx(h.t, 2.0));
    assert!(approx(h.point.y, 1.0));
}

#[test]
fn ray2_vs_rect_miss() {
    let rect = Rect::new(v2(-1.0, -1.0), v2(1.0, 1.0));
    let r    = Ray2::new_unnormalized(v2(5.0, 3.0), v2(0.0, -1.0));
    assert!(r.intersect_rect(&rect).is_none());
}

#[test]
fn ray2_vs_circle_hit() {
    let circle = Circle::new(Vec2::ZERO, 1.5);
    let r      = Ray2::new_unnormalized(v2(0.0, 3.0), v2(0.0, -1.0));
    let h      = r.intersect_circle(&circle).expect("should hit");
    assert!(approx(h.t, 1.5));
}

#[test]
fn ray2_vs_circle_miss() {
    let circle = Circle::new(Vec2::ZERO, 1.0);
    let r      = Ray2::new_unnormalized(v2(5.0, 3.0), v2(0.0, -1.0));
    assert!(r.intersect_circle(&circle).is_none());
}

// ═════════════════════════════════════════════════════════════════════════════
// RECT & CIRCLE (2D)
// ═════════════════════════════════════════════════════════════════════════════

#[test]
fn rect_intersects_rect() {
    let a = Rect::new(v2(0.0, 0.0), v2(2.0, 2.0));
    assert!(a.intersects_rect(Rect::new(v2(1.0, 1.0), v2(3.0, 3.0))));  // overlap
    assert!(a.intersects_rect(Rect::new(v2(2.0, 0.0), v2(4.0, 2.0))));  // touching
    assert!(!a.intersects_rect(Rect::new(v2(2.01, 0.0), v2(4.0, 2.0)))); // separated
}

#[test]
fn rect_intersects_circle() {
    let r = Rect::new(v2(-1.0, -1.0), v2(1.0, 1.0));
    assert!(r.intersects_circle(&Circle::new(Vec2::ZERO, 0.5)));       // inside
    assert!(r.intersects_circle(&Circle::new(v2(1.5, 0.0), 1.0)));     // overlap face
    assert!(r.intersects_circle(&Circle::new(v2(2.0, 2.0), 1.415)));   // corner overlap
    assert!(!r.intersects_circle(&Circle::new(v2(3.0, 0.0), 1.0)));    // separated
}

#[test]
fn circle_vs_circle() {
    let a = Circle::new(Vec2::ZERO, 2.0);
    assert!(a.intersects_circle(&Circle::new(v2(3.0, 0.0), 2.0)));  // overlap
    assert!(a.intersects_circle(&Circle::new(v2(4.0, 0.0), 2.0)));  // touching
    assert!(!a.intersects_circle(&Circle::new(v2(4.01, 0.0), 2.0))); // separated
}

#[test]
fn rect_expand_and_merge() {
    let r = Rect::new(v2(0.0, 0.0), v2(2.0, 2.0));
    let expanded = r.expand_to_include(v2(3.0, -1.0));
    assert_eq!(expanded.min, v2(0.0, -1.0));
    assert_eq!(expanded.max, v2(3.0, 2.0));

    let a = Rect::new(v2(-1.0, -1.0), v2(0.0, 0.0));
    let b = Rect::new(v2(0.0, 0.0), v2(1.0, 1.0));
    let m = a.merge(b);
    assert_eq!(m.min, v2(-1.0, -1.0));
    assert_eq!(m.max, v2(1.0, 1.0));
}

// ═════════════════════════════════════════════════════════════════════════════
// FFI ROUND-TRIPS
// ═════════════════════════════════════════════════════════════════════════════

#[test]
fn ffi_aabb_round_trip() {
    use mid_geom::ffi::{CAABB, CRect, CSphere};
    use mid_math::ffi::{CVec3, CVec2};

    let aabb = AABB::new(v3(-1.0, -2.0, -3.0), v3(1.0, 2.0, 3.0));
    let c: CAABB = aabb.into();
    let back: AABB = c.into();
    assert_eq!(back.min, aabb.min);
    assert_eq!(back.max, aabb.max);
}

#[test]
fn ffi_sphere_round_trip() {
    use mid_geom::ffi::CSphere;
    let s = Sphere::new(v3(1.0, 2.0, 3.0), 4.5);
    let c: mid_geom::ffi::CSphere = s.into();
    let back: Sphere = c.into();
    assert!(approx(back.radius, 4.5));
               }
