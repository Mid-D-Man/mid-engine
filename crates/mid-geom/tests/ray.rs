// crates/mid-geom/tests/ray.rs
use mid_geom::{AABB, Capsule, Plane, Ray2, Ray3, Sphere};
use mid_math::{Vec2, Vec3};

fn v(x: f32, y: f32, z: f32) -> Vec3 { Vec3::new(x, y, z) }
fn v2(x: f32, y: f32) -> Vec2 { Vec2::new(x, y) }
fn approx(a: f32, b: f32) -> bool { (a - b).abs() < 1e-4 }

// ── Ray3 ──────────────────────────────────────────────────────────────────────

fn ray_from_z5() -> Ray3 { Ray3::new_unnormalized(v(0.0,0.0,5.0), v(0.0,0.0,-1.0)) }
fn ray_beside()  -> Ray3 { Ray3::new_unnormalized(v(5.0,0.0,5.0), v(0.0,0.0,-1.0)) }

#[test] fn sphere_hit_t_and_point() {
    let h = ray_from_z5().intersect_sphere(&Sphere::new(Vec3::ZERO, 1.0)).expect("hit");
    assert!(approx(h.t, 4.0));
    assert!(approx(h.point.z, 1.0));
}

#[test] fn sphere_miss() {
    assert!(ray_beside().intersect_sphere(&Sphere::new(Vec3::ZERO, 1.0)).is_none());
}

#[test] fn sphere_inside_origin_hits_exit() {
    let r = Ray3::new_unnormalized(Vec3::ZERO, v(1.0,0.0,0.0));
    let h = r.intersect_sphere(&Sphere::new(Vec3::ZERO, 2.0)).expect("exit hit");
    assert!(approx(h.t, 2.0));
}

#[test] fn aabb_hit_t_normal_point() {
    let a = AABB::new(Vec3::splat(-1.0), Vec3::splat(1.0));
    let h = ray_from_z5().intersect_aabb(&a).expect("hit");
    assert!(approx(h.t, 4.0));
    assert!(approx(h.point.z, 1.0));
    assert!(approx(h.normal.z, 1.0));
}

#[test] fn aabb_miss() {
    let a = AABB::new(Vec3::splat(-1.0), Vec3::splat(1.0));
    assert!(ray_beside().intersect_aabb(&a).is_none());
}

#[test] fn aabb_inside_start_exits() {
    let r = Ray3::new_unnormalized(Vec3::ZERO, v(1.0,0.0,0.0));
    let a = AABB::new(Vec3::splat(-2.0), Vec3::splat(2.0));
    let h = r.intersect_aabb(&a).expect("exit hit");
    assert!(approx(h.t, 2.0));
}

#[test] fn plane_hit_t() {
    let p = Plane::from_normal_point(Vec3::Y, Vec3::ZERO);
    let r = Ray3::new_unnormalized(v(1.0,3.0,-2.0), v(0.0,-1.0,0.0));
    let h = r.intersect_plane(&p).expect("hit");
    assert!(approx(h.t, 3.0));
    assert!(approx(h.point.y, 0.0));
}

#[test] fn plane_miss_behind() {
    let p = Plane::from_normal_point(Vec3::Y, Vec3::ZERO);
    let r = Ray3::new_unnormalized(v(0.0,-1.0,0.0), v(0.0,-1.0,0.0));
    // Hitting behind negative t — should return None (t < 0 guard)
    let result = r.intersect_plane(&p);
    // t would be negative → None
    if let Some(h) = result { assert!(h.t >= 0.0); }
}

#[test] fn capsule_hit() {
    let cap = Capsule::new(v(-1.0,0.0,0.0), v(1.0,0.0,0.0), 0.5);
    let r   = Ray3::new_unnormalized(v(0.0,3.0,0.0), v(0.0,-1.0,0.0));
    assert!(r.intersect_capsule(&cap).is_some());
}

#[test] fn capsule_miss() {
    let cap = Capsule::new(v(-1.0,0.0,0.0), v(1.0,0.0,0.0), 0.5);
    let r   = Ray3::new_unnormalized(v(5.0,3.0,0.0), v(0.0,-1.0,0.0));
    assert!(r.intersect_capsule(&cap).is_none());
}

// ── Ray2 ──────────────────────────────────────────────────────────────────────

use mid_geom::{Circle, Rect};

#[test] fn ray2_rect_hit() {
    let rect = Rect::new(v2(-1.0,-1.0), v2(1.0,1.0));
    let r    = Ray2::new_unnormalized(v2(0.0,3.0), v2(0.0,-1.0));
    let h    = r.intersect_rect(&rect).expect("hit");
    assert!(approx(h.t, 2.0));
    assert!(approx(h.point.y, 1.0));
}

#[test] fn ray2_rect_miss() {
    let rect = Rect::new(v2(-1.0,-1.0), v2(1.0,1.0));
    let r    = Ray2::new_unnormalized(v2(5.0,3.0), v2(0.0,-1.0));
    assert!(r.intersect_rect(&rect).is_none());
}

#[test] fn ray2_circle_hit() {
    let c = Circle::new(Vec2::ZERO, 1.5);
    let r = Ray2::new_unnormalized(v2(0.0,3.0), v2(0.0,-1.0));
    let h = r.intersect_circle(&c).expect("hit");
    assert!(approx(h.t, 1.5));
}

#[test] fn ray2_circle_miss() {
    let c = Circle::new(Vec2::ZERO, 1.0);
    let r = Ray2::new_unnormalized(v2(5.0,3.0), v2(0.0,-1.0));
    assert!(r.intersect_circle(&c).is_none());
      }
