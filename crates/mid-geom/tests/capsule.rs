// crates/mid-geom/tests/capsule.rs
use mid_geom::{Capsule, Sphere};
use mid_math::Vec3;

fn v(x: f32, y: f32, z: f32) -> Vec3 { Vec3::new(x, y, z) }
fn approx(a: f32, b: f32) -> bool { (a - b).abs() < 1e-4 }

fn upright() -> Capsule {
    Capsule::new(Vec3::ZERO, v(0.0,2.0,0.0), 0.5)
}

#[test] fn closest_point_clamps_below_base() {
    let c = upright();
    assert_eq!(c.closest_point_on_axis(v(0.0,-1.0,0.0)), Vec3::ZERO);
}

#[test] fn closest_point_clamps_above_tip() {
    let c = upright();
    assert_eq!(c.closest_point_on_axis(v(0.0,3.0,0.0)), v(0.0,2.0,0.0));
}

#[test] fn closest_point_midpoint() {
    let c = upright();
    let p = c.closest_point_on_axis(v(1.0,1.0,0.0));
    assert!(approx(p.y, 1.0));
    assert!(approx(p.x, 0.0));
}

#[test] fn contains_point_in_body() {
    assert!(upright().contains_point(v(0.0,1.0,0.0)));
}

#[test] fn contains_point_in_lower_cap() {
    assert!(upright().contains_point(v(0.0,-0.4,0.0)));
}

#[test] fn contains_point_in_upper_cap() {
    assert!(upright().contains_point(v(0.0,2.4,0.0)));
}

#[test] fn contains_point_outside() {
    assert!(!upright().contains_point(v(2.0,1.0,0.0)));
    assert!(!upright().contains_point(v(0.0,-1.01,0.0)));
}

#[test] fn vs_sphere_hit() {
    let c = upright();
    assert!(c.intersects_sphere(&Sphere::new(v(0.0,1.0,0.4), 0.5)));
}

#[test] fn vs_sphere_miss() {
    let c = upright();
    assert!(!c.intersects_sphere(&Sphere::new(v(5.0,1.0,0.0), 0.5)));
}

#[test] fn vs_capsule_parallel_close() {
    let c1 = upright();
    let c2 = Capsule::new(v(0.8,0.0,0.0), v(0.8,2.0,0.0), 0.5);
    assert!(c1.intersects_capsule(&c2));
}

#[test] fn vs_capsule_parallel_far() {
    let c1 = upright();
    let c3 = Capsule::new(v(5.0,0.0,0.0), v(5.0,2.0,0.0), 0.5);
    assert!(!c1.intersects_capsule(&c3));
}

#[test] fn vs_capsule_crossing() {
    let c1 = upright();
    let c4 = Capsule::new(v(-1.0,1.0,0.0), v(1.0,1.0,0.0), 0.3);
    assert!(c1.intersects_capsule(&c4));
}
