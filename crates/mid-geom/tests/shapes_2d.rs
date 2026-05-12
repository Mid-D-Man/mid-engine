// crates/mid-geom/tests/shapes_2d.rs
use mid_geom::{Circle, Rect};
use mid_math::Vec2;

fn v(x: f32, y: f32) -> Vec2 { Vec2::new(x, y) }
fn approx(a: f32, b: f32) -> bool { (a - b).abs() < 1e-4 }

// ── Rect ──────────────────────────────────────────────────────────────────────

#[test] fn rect_overlap() {
    let a = Rect::new(v(0.0,0.0), v(2.0,2.0));
    assert!(a.intersects_rect(Rect::new(v(1.0,1.0), v(3.0,3.0))));
}

#[test] fn rect_touching_counts() {
    let a = Rect::new(v(0.0,0.0), v(2.0,2.0));
    assert!(a.intersects_rect(Rect::new(v(2.0,0.0), v(4.0,2.0))));
}

#[test] fn rect_separated() {
    let a = Rect::new(v(0.0,0.0), v(2.0,2.0));
    assert!(!a.intersects_rect(Rect::new(v(2.01,0.0), v(4.0,2.0))));
}

#[test] fn rect_contains_point_inside() {
    let r = Rect::new(v(-1.0,-1.0), v(1.0,1.0));
    assert!(r.contains_point(Vec2::ZERO));
    assert!(r.contains_point(v(1.0,1.0)));  // boundary
}

#[test] fn rect_contains_point_outside() {
    let r = Rect::new(v(-1.0,-1.0), v(1.0,1.0));
    assert!(!r.contains_point(v(1.01,0.0)));
}

#[test] fn rect_intersects_circle_inside() {
    let r = Rect::new(v(-1.0,-1.0), v(1.0,1.0));
    assert!(r.intersects_circle(&Circle::new(Vec2::ZERO, 0.5)));
}

#[test] fn rect_intersects_circle_face() {
    let r = Rect::new(v(-1.0,-1.0), v(1.0,1.0));
    assert!(r.intersects_circle(&Circle::new(v(1.5,0.0), 1.0)));
}

#[test] fn rect_intersects_circle_corner() {
    let r = Rect::new(v(-1.0,-1.0), v(1.0,1.0));
    // distance from corner (1,1) to center (2,2) = sqrt(2) ≈ 1.414
    assert!(r.intersects_circle(&Circle::new(v(2.0,2.0), 1.415)));
}

#[test] fn rect_misses_circle() {
    let r = Rect::new(v(-1.0,-1.0), v(1.0,1.0));
    assert!(!r.intersects_circle(&Circle::new(v(3.0,0.0), 1.0)));
}

#[test] fn rect_expand_to_include() {
    let r = Rect::new(v(0.0,0.0), v(2.0,2.0));
    let e = r.expand_to_include(v(3.0,-1.0));
    assert_eq!(e.min, v(0.0,-1.0));
    assert_eq!(e.max, v(3.0,2.0));
}

#[test] fn rect_merge() {
    let a = Rect::new(v(-1.0,-1.0), v(0.0,0.0));
    let b = Rect::new(v(0.0,0.0),   v(1.0,1.0));
    let m = a.merge(b);
    assert_eq!(m.min, v(-1.0,-1.0));
    assert_eq!(m.max, v(1.0,1.0));
}

#[test] fn rect_center() {
    let r = Rect::new(v(0.0,0.0), v(4.0,6.0));
    assert_eq!(r.center(), v(2.0,3.0));
}

// ── Circle ────────────────────────────────────────────────────────────────────

#[test] fn circle_overlap() {
    let a = Circle::new(Vec2::ZERO, 2.0);
    assert!(a.intersects_circle(&Circle::new(v(3.0,0.0), 2.0)));
}

#[test] fn circle_touching() {
    let a = Circle::new(Vec2::ZERO, 2.0);
    assert!(a.intersects_circle(&Circle::new(v(4.0,0.0), 2.0)));
}

#[test] fn circle_separated() {
    let a = Circle::new(Vec2::ZERO, 2.0);
    assert!(!a.intersects_circle(&Circle::new(v(4.01,0.0), 2.0)));
}

#[test] fn circle_contains_point() {
    let c = Circle::new(Vec2::ZERO, 3.0);
    assert!(c.contains_point(v(2.9,0.0)));
    assert!(!c.contains_point(v(3.01,0.0)));
}

#[test] fn circle_area_circumference() {
    let c = Circle::new(Vec2::ZERO, 1.0);
    assert!(approx(c.area(), core::f32::consts::PI));
    assert!(approx(c.circumference(), 2.0 * core::f32::consts::PI));
}

#[test] fn circle_bounding_rect() {
    let c = Circle::new(v(1.0,2.0), 1.5);
    let r = c.bounding_rect();
    assert!(approx(r.min.x, -0.5));
    assert!(approx(r.min.y,  0.5));
    assert!(approx(r.max.x,  2.5));
    assert!(approx(r.max.y,  3.5));
                      }
