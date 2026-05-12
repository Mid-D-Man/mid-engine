// crates/mid-geom/tests/ffi.rs
use mid_geom::{AABB, Capsule, Sphere, Plane};
use mid_geom::ffi::{CAABB, CCapsule, CSphere, CPlane};
use mid_math::Vec3;

fn v(x: f32, y: f32, z: f32) -> Vec3 { Vec3::new(x, y, z) }
fn approx(a: f32, b: f32) -> bool { (a - b).abs() < 1e-5 }

#[test] fn aabb_round_trip() {
    let src: AABB = AABB::new(v(-1.0,-2.0,-3.0), v(1.0,2.0,3.0));
    let c: CAABB  = src.into();
    let back: AABB = c.into();
    assert_eq!(back.min, src.min);
    assert_eq!(back.max, src.max);
}

#[test] fn sphere_round_trip() {
    let src = Sphere::new(v(1.0,2.0,3.0), 4.5);
    let c: CSphere  = src.into();
    let back: Sphere = c.into();
    assert!(approx(back.radius, 4.5));
    assert!(approx(back.center.x, 1.0));
    assert!(approx(back.center.y, 2.0));
    assert!(approx(back.center.z, 3.0));
}

#[test] fn capsule_round_trip() {
    let src = Capsule::new(v(0.0,0.0,0.0), v(0.0,2.0,0.0), 0.75);
    let c: CCapsule  = src.into();
    let back: Capsule = c.into();
    assert!(approx(back.radius, 0.75));
    assert!(approx(back.tip.y, 2.0));
}

#[test] fn plane_round_trip() {
    let src = Plane::from_normal_point(Vec3::Y, v(0.0,3.0,0.0));
    let c: CPlane  = src.into();
    let back: Plane = c.into();
    assert!(approx(back.normal.y, 1.0));
    assert!(approx(back.d, src.d));
}

#[test] fn ffi_no_mangle_aabb_new_correct() {
    use mid_geom::ffi::{mid_aabb_new, mid_aabb_center, mid_aabb_surface_area};
    use mid_math::ffi::CVec3;
    let min = CVec3::new(-1.0,-1.0,-1.0);
    let max = CVec3::new( 1.0, 1.0, 1.0);
    let a   = mid_aabb_new(min, max);
    let ctr = mid_aabb_center(a);
    assert!(approx(ctr.x, 0.0));
    assert!(approx(ctr.y, 0.0));
    assert!(approx(ctr.z, 0.0));
    assert!(approx(mid_aabb_surface_area(a), 24.0));
}

#[test] fn ffi_sphere_contains() {
    use mid_geom::ffi::{mid_sphere_new, mid_sphere_contains_point};
    use mid_math::ffi::CVec3;
    let s = mid_sphere_new(CVec3::new(0.0,0.0,0.0), 2.0);
    assert!(mid_sphere_contains_point(s, CVec3::new(1.0,0.0,0.0)));
    assert!(!mid_sphere_contains_point(s, CVec3::new(3.0,0.0,0.0)));
}
