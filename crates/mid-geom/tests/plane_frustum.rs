// crates/mid-geom/tests/plane_frustum.rs
use mid_geom::{AABB, Frustum, Plane, Sphere};
use mid_math::{Mat4, Vec3, FRAC_PI_4};

fn v(x: f32, y: f32, z: f32) -> Vec3 { Vec3::new(x, y, z) }
fn approx(a: f32, b: f32) -> bool { (a - b).abs() < 1e-4 }

fn xz_plane() -> Plane { Plane::from_normal_point(Vec3::Y, Vec3::ZERO) }

fn test_frustum() -> Frustum {
    let vp = Mat4::perspective_rh(FRAC_PI_4 * 2.0, 1.0, 1.0, 100.0)
        * Mat4::look_at_rh(v(0.0,0.0,10.0), Vec3::ZERO, Vec3::Y);
    Frustum::from_mat4(&vp)
}

// ── Plane ─────────────────────────────────────────────────────────────────────

#[test] fn signed_distance_above() {
    assert!(approx(xz_plane().signed_distance(v(0.0, 2.0, 0.0)),  2.0));
}

#[test] fn signed_distance_below() {
    assert!(approx(xz_plane().signed_distance(v(0.0,-3.0, 0.0)), -3.0));
}

#[test] fn signed_distance_on_plane() {
    assert!(approx(xz_plane().signed_distance(Vec3::ZERO), 0.0));
}

#[test] fn project_point_onto_xz() {
    let pr = xz_plane().project_point(v(3.0, 5.0, -2.0));
    assert!(approx(pr.x,  3.0));
    assert!(approx(pr.y,  0.0));
    assert!(approx(pr.z, -2.0));
}

#[test] fn from_three_points_normal_direction() {
    let p = Plane::from_points(Vec3::ZERO, v(1.0,0.0,0.0), v(0.0,1.0,0.0))
        .expect("valid triangle");
    // CCW winding → normal in +Z
    assert!(p.normal.z.abs() > 0.99);
}

#[test] fn parallel_ray_returns_none() {
    let p = xz_plane();
    let result = p.intersect_ray(v(0.0,1.0,0.0), v(1.0,0.0,0.0));
    assert!(result.is_none());
}

#[test] fn ray_intersects_plane_at_correct_t() {
    let p = xz_plane();
    let t = p.intersect_ray(v(0.0,3.0,0.0), v(0.0,-1.0,0.0))
        .expect("should hit");
    assert!(approx(t, 3.0));
}

// ── Frustum ───────────────────────────────────────────────────────────────────

#[test] fn frustum_origin_aabb_visible() {
    let f = test_frustum();
    let visible = AABB::new(v(-0.5,-0.5,-0.5), v(0.5,0.5,0.5));
    assert!(f.intersects_aabb(&visible));
}

#[test] fn frustum_origin_sphere_visible() {
    let f = test_frustum();
    assert!(f.intersects_sphere(&Sphere::new(Vec3::ZERO, 0.5)));
}

#[test] fn frustum_huge_sphere_spans_frustum() {
    let f = test_frustum();
    let huge = Sphere::new(v(0.0,0.0,-500.0), 600.0);
    assert!(f.intersects_sphere(&huge));
}

#[test] fn frustum_contains_point_at_origin() {
    let f = test_frustum();
    assert!(f.contains_point(Vec3::ZERO));
}

#[test] fn frustum_does_not_panic_on_large_coords() {
    let f = test_frustum();
    let far_behind = AABB::new(v(-1.0,-1.0,999.0), v(1.0,1.0,1001.0));
    let _ = f.intersects_aabb(&far_behind); // must not panic
                                  }
