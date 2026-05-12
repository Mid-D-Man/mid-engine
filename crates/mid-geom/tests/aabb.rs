// crates/mid-geom/tests/aabb.rs
use mid_geom::{AABB, Sphere};
use mid_math::Vec3;

fn v(x: f32, y: f32, z: f32) -> Vec3 { Vec3::new(x, y, z) }
fn approx(a: f32, b: f32) -> bool { (a - b).abs() < 1e-4 }

#[test] fn construction() {
    let a = AABB::new(v(-1.0,-1.0,-1.0), v(1.0,1.0,1.0));
    assert_eq!(a.center(), Vec3::ZERO);
    assert_eq!(a.size(), Vec3::splat(2.0));
    assert_eq!(a.extents(), Vec3::ONE);
    assert!(approx(a.surface_area(), 24.0));
    assert!(approx(a.volume(), 8.0));
}

#[test] fn from_center_half_extents() {
    let a = AABB::from_center_half_extents(v(1.0,2.0,3.0), v(0.5,1.0,1.5));
    assert_eq!(a.min, v(0.5,1.0,1.5));
    assert_eq!(a.max, v(1.5,3.0,4.5));
}

#[test] fn invalid_seed_accumulates_correctly() {
    let inv = AABB::invalid();
    let p   = v(5.0,-3.0,7.0);
    let g   = inv.expand_to_include_point(p);
    assert_eq!(g.min, p);
    assert_eq!(g.max, p);
}

#[test] fn contains_point() {
    let a = AABB::new(v(-2.0,-2.0,-2.0), v(2.0,2.0,2.0));
    assert!(a.contains_point(Vec3::ZERO));
    assert!(a.contains_point(v(2.0,2.0,2.0)));    // boundary
    assert!(a.contains_point(v(-2.0,-2.0,-2.0))); // boundary
    assert!(!a.contains_point(v(2.01,0.0,0.0)));
    assert!(!a.contains_point(v(0.0,-2.01,0.0)));
}

#[test] fn intersects_aabb_overlap() {
    let a = AABB::new(v(-1.0,-1.0,-1.0), v(1.0,1.0,1.0));
    assert!(a.intersects_aabb(&AABB::new(v(0.0,0.0,0.0), v(2.0,2.0,2.0))));
}

#[test] fn intersects_aabb_touching_counts() {
    let a = AABB::new(v(-1.0,-1.0,-1.0), v(1.0,1.0,1.0));
    assert!(a.intersects_aabb(&AABB::new(v(1.0,-1.0,-1.0), v(3.0,1.0,1.0))));
}

#[test] fn intersects_aabb_separated_x() {
    let a = AABB::new(v(-1.0,-1.0,-1.0), v(1.0,1.0,1.0));
    assert!(!a.intersects_aabb(&AABB::new(v(1.01,-1.0,-1.0), v(3.0,1.0,1.0))));
}

#[test] fn intersects_aabb_separated_y() {
    let a = AABB::new(v(-1.0,-1.0,-1.0), v(1.0,1.0,1.0));
    assert!(!a.intersects_aabb(&AABB::new(v(-1.0,1.01,-1.0), v(1.0,3.0,1.0))));
}

#[test] fn intersects_aabb_separated_z() {
    let a = AABB::new(v(-1.0,-1.0,-1.0), v(1.0,1.0,1.0));
    assert!(!a.intersects_aabb(&AABB::new(v(-1.0,-1.0,1.01), v(1.0,1.0,3.0))));
}

#[test] fn intersects_aabb_contained() {
    let a = AABB::new(v(-1.0,-1.0,-1.0), v(1.0,1.0,1.0));
    assert!(a.intersects_aabb(&AABB::new(v(-0.5,-0.5,-0.5), v(0.5,0.5,0.5))));
}

#[test] fn intersects_sphere_inside() {
    let a = AABB::new(v(-2.0,-2.0,-2.0), v(2.0,2.0,2.0));
    assert!(a.intersects_sphere(&Sphere::new(Vec3::ZERO, 0.5)));
}

#[test] fn intersects_sphere_face_overlap() {
    let a = AABB::new(v(-2.0,-2.0,-2.0), v(2.0,2.0,2.0));
    assert!(a.intersects_sphere(&Sphere::new(v(2.5,0.0,0.0), 1.0)));
}

#[test] fn intersects_sphere_touching() {
    let a = AABB::new(v(-2.0,-2.0,-2.0), v(2.0,2.0,2.0));
    assert!(a.intersects_sphere(&Sphere::new(v(3.0,0.0,0.0), 1.0)));
}

#[test] fn intersects_sphere_separated() {
    let a = AABB::new(v(-2.0,-2.0,-2.0), v(2.0,2.0,2.0));
    assert!(!a.intersects_sphere(&Sphere::new(v(3.01,0.0,0.0), 1.0)));
}

#[test] fn closest_point_inside() {
    let a = AABB::new(v(-1.0,-1.0,-1.0), v(1.0,1.0,1.0));
    assert_eq!(a.closest_point(Vec3::ZERO), Vec3::ZERO);
}

#[test] fn closest_point_outside_x() {
    let a = AABB::new(v(-1.0,-1.0,-1.0), v(1.0,1.0,1.0));
    assert_eq!(a.closest_point(v(3.0,0.0,0.0)), v(1.0,0.0,0.0));
}

#[test] fn closest_point_outside_corner() {
    let a = AABB::new(v(-1.0,-1.0,-1.0), v(1.0,1.0,1.0));
    assert_eq!(a.closest_point(v(3.0,3.0,3.0)), v(1.0,1.0,1.0));
}

#[test] fn signed_distance_inside_negative() {
    let a = AABB::new(v(-1.0,-1.0,-1.0), v(1.0,1.0,1.0));
    assert!(a.signed_distance(Vec3::ZERO) < 0.0);
}

#[test] fn signed_distance_surface_zero() {
    let a = AABB::new(v(-1.0,-1.0,-1.0), v(1.0,1.0,1.0));
    assert!(a.signed_distance(v(1.0,0.0,0.0)).abs() < 1e-4);
}

#[test] fn signed_distance_outside_positive() {
    let a = AABB::new(v(-1.0,-1.0,-1.0), v(1.0,1.0,1.0));
    assert!(a.signed_distance(v(3.0,0.0,0.0)) > 0.0);
}

#[test] fn merge() {
    let a = AABB::new(v(-1.0,-1.0,-1.0), v(0.0,0.0,0.0));
    let b = AABB::new(v(0.0,0.0,0.0), v(1.0,1.0,1.0));
    let m = a.merge(b);
    assert_eq!(m.min, v(-1.0,-1.0,-1.0));
    assert_eq!(m.max, v(1.0,1.0,1.0));
  }
