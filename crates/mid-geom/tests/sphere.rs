// crates/mid-geom/tests/sphere.rs
use mid_geom::{AABB, Sphere};
use mid_math::Vec3;

fn v(x: f32, y: f32, z: f32) -> Vec3 { Vec3::new(x, y, z) }
fn approx(a: f32, b: f32) -> bool { (a - b).abs() < 1e-4 }

#[test] fn contains_point_inside() {
    let s = Sphere::new(Vec3::ZERO, 2.0);
    assert!(s.contains_point(Vec3::ZERO));
    assert!(s.contains_point(v(2.0,0.0,0.0))); // on surface
    assert!(!s.contains_point(v(2.01,0.0,0.0)));
}

#[test] fn sphere_vs_sphere_touching() {
    // d=4 == r1+r2=4 → touching → should intersect
    let s1 = Sphere::new(Vec3::ZERO, 2.0);
    let s2 = Sphere::new(v(4.0,0.0,0.0), 2.0);
    assert!(s1.intersects_sphere(&s2));
}

#[test] fn sphere_vs_sphere_separated() {
    let s1 = Sphere::new(Vec3::ZERO, 2.0);
    let s3 = Sphere::new(v(4.01,0.0,0.0), 2.0);
    assert!(!s1.intersects_sphere(&s3));
}

#[test] fn sphere_vs_sphere_no_sqrt() {
    // Verify by inspection: intersects_sphere uses length_sq, no sqrt call.
    // (d²=(4.01²)=16.0801) > (r1+r2)²=16 → miss
    let s1 = Sphere::new(Vec3::ZERO, 2.0);
    let s2 = Sphere::new(v(4.01,0.0,0.0), 2.0);
    assert!(!s1.intersects_sphere(&s2));
}

#[test] fn signed_distance() {
    let s = Sphere::new(Vec3::ZERO, 2.0);
    assert!(approx(s.signed_distance(v(3.0,0.0,0.0)), 1.0));
    assert!(approx(s.signed_distance(v(2.0,0.0,0.0)), 0.0));
    assert!(s.signed_distance(Vec3::ZERO) < 0.0);
}

#[test] fn bounding_aabb() {
    let s = Sphere::new(v(1.0,2.0,3.0), 1.5);
    let a = s.bounding_aabb();
    assert!(approx(a.min.x, -0.5));
    assert!(approx(a.min.y,  0.5));
    assert!(approx(a.min.z,  1.5));
    assert!(approx(a.max.x,  2.5));
    assert!(approx(a.max.y,  3.5));
    assert!(approx(a.max.z,  4.5));
}

#[test] fn merge_contained() {
    let big   = Sphere::new(Vec3::ZERO, 5.0);
    let small = Sphere::new(v(1.0,0.0,0.0), 1.0);
    let m = big.merge(small);
    assert!(approx(m.radius, 5.0));
}

#[test] fn merge_disjoint_covers_both() {
    let a  = Sphere::new(v(-3.0,0.0,0.0), 1.0);
    let b  = Sphere::new(v( 3.0,0.0,0.0), 1.0);
    let m  = a.merge(b);
    assert!(m.contains_point(v(-4.0,0.0,0.0)));
    assert!(m.contains_point(v( 4.0,0.0,0.0)));
}

#[test] fn sphere_vs_aabb_inside() {
    let a = AABB::new(Vec3::splat(-2.0), Vec3::splat(2.0));
    assert!(a.intersects_sphere(&Sphere::new(Vec3::ZERO, 0.5)));
}
