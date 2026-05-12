// crates/mid-math/tests/curves_bezier.rs
//! Integration tests for QuadraticBezier and CubicBezier.

use mid_math::{Vec3, QuadraticBezier, CubicBezier};

const EPS: f32 = 1e-5;

fn approx(a: f32, b: f32) -> bool { (a - b).abs() < EPS }

fn v3eq(a: Vec3, b: Vec3) -> bool {
    approx(a.x, b.x) && approx(a.y, b.y) && approx(a.z, b.z)
}

// ─── QuadraticBezier ─────────────────────────────────────────────────────────

#[test]
fn quad_t0_returns_p0() {
    let b = QuadraticBezier::new(
        Vec3::new(1.0, 2.0, 3.0),
        Vec3::new(5.0, 5.0, 5.0),
        Vec3::new(9.0, 0.0, 0.0),
    );
    assert!(v3eq(b.evaluate(0.0), Vec3::new(1.0, 2.0, 3.0)));
}

#[test]
fn quad_t1_returns_p2() {
    let p2 = Vec3::new(9.0, 0.0, 0.0);
    let b = QuadraticBezier::new(Vec3::ZERO, Vec3::new(5.0, 5.0, 0.0), p2);
    assert!(v3eq(b.evaluate(1.0), p2));
}

#[test]
fn quad_midpoint_is_pulled_toward_handle() {
    // Symmetric: p0=(0,0,0), p1=(1,2,0), p2=(2,0,0)
    // At t=0.5: B(0.5) = 0.25*p0 + 0.5*p1 + 0.25*p2 = (1, 1, 0)
    let b = QuadraticBezier::new(
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 2.0, 0.0),
        Vec3::new(2.0, 0.0, 0.0),
    );
    let mid = b.evaluate(0.5);
    assert!(approx(mid.x, 1.0));
    assert!(approx(mid.y, 1.0)); // pulled toward handle but < 2.0
}

#[test]
fn quad_tangent_at_t0_direction() {
    // B'(0) = 2*(p1 - p0). With p0=(0,0,0), p1=(0,1,0): tangent = (0,2,0)
    let b = QuadraticBezier::new(
        Vec3::ZERO,
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(1.0, 1.0, 0.0),
    );
    let t = b.tangent(0.0);
    assert!(approx(t.x, 0.0));
    assert!(t.y > 0.0);
}

#[test]
fn quad_tangent_at_t1_direction() {
    // B'(1) = 2*(p2 - p1). With p1=(0,1,0), p2=(1,1,0): tangent = (2,0,0)
    let b = QuadraticBezier::new(
        Vec3::ZERO,
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(1.0, 1.0, 0.0),
    );
    let t = b.tangent(1.0);
    assert!(t.x > 0.0);
    assert!(approx(t.y, 0.0));
}

#[test]
fn quad_arc_length_exceeds_chord() {
    let b = QuadraticBezier::new(
        Vec3::ZERO,
        Vec3::new(1.0, 1.0, 0.0),
        Vec3::new(2.0, 0.0, 0.0),
    );
    let len = b.arc_length(64);
    // Chord = 2.0 (straight-line from p0 to p2); curved path must be longer.
    assert!(len > 2.0);
    assert!(len < 4.0); // sanity upper bound
}

#[test]
fn quad_sample_uniform_fills_and_matches_endpoints() {
    let p0 = Vec3::new(0.0, 0.0, 0.0);
    let p2 = Vec3::new(4.0, 0.0, 0.0);
    let b = QuadraticBezier::new(p0, Vec3::new(2.0, 2.0, 0.0), p2);
    let mut buf = [Vec3::ZERO; 5];
    b.sample_uniform(4, &mut buf);
    assert!(v3eq(buf[0], p0));
    assert!(v3eq(buf[4], p2));
}

#[test]
fn quad_f32_scalar_known_values() {
    // B(t) = (1-t)^2*0 + 2(1-t)t*2 + t^2*4
    // B(0)=0, B(0.5)=0+1+1=2... wait let me recalculate
    // B(0.5) = 0.25*0 + 0.5*2 + 0.25*4 = 0 + 1 + 1 = 2.0
    let b = QuadraticBezier::new(0.0f32, 2.0f32, 4.0f32);
    assert!(approx(b.evaluate(0.0), 0.0));
    assert!(approx(b.evaluate(1.0), 4.0));
    assert!(approx(b.evaluate(0.5), 2.0));
}

// ─── CubicBezier ─────────────────────────────────────────────────────────────

#[test]
fn cubic_t0_returns_p0() {
    let p0 = Vec3::new(1.0, 2.0, 3.0);
    let b = CubicBezier::new(p0, Vec3::X, Vec3::Y, Vec3::Z);
    assert!(v3eq(b.evaluate(0.0), p0));
}

#[test]
fn cubic_t1_returns_p3() {
    let p3 = Vec3::new(9.0, 8.0, 7.0);
    let b = CubicBezier::new(Vec3::ZERO, Vec3::X, Vec3::Y, p3);
    assert!(v3eq(b.evaluate(1.0), p3));
}

#[test]
fn cubic_tangent_at_t0_is_3_p1_minus_p0() {
    // B'(0) = 3*(p1 - p0)
    let p0 = Vec3::new(0.0, 0.0, 0.0);
    let p1 = Vec3::new(0.0, 1.0, 0.0);
    let b = CubicBezier::new(p0, p1, Vec3::new(1.0, 1.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
    let tan = b.tangent(0.0);
    assert!(approx(tan.x, 0.0));
    assert!(approx(tan.y, 3.0));
    assert!(approx(tan.z, 0.0));
}

#[test]
fn cubic_tangent_at_t1_is_3_p3_minus_p2() {
    // B'(1) = 3*(p3 - p2)
    let p2 = Vec3::new(1.0, 1.0, 0.0);
    let p3 = Vec3::new(2.0, 0.0, 0.0);
    let b = CubicBezier::new(Vec3::ZERO, Vec3::new(0.5, 0.5, 0.0), p2, p3);
    let tan = b.tangent(1.0);
    assert!(approx(tan.x, 3.0));
    assert!(approx(tan.y, -3.0));
}

#[test]
fn cubic_split_meets_at_split_point() {
    let b = CubicBezier::new(
        Vec3::ZERO,
        Vec3::new(1.0, 2.0, 0.0),
        Vec3::new(2.0, 2.0, 0.0),
        Vec3::new(3.0, 0.0, 0.0),
    );
    let split_t = 0.4;
    let original_at_t = b.evaluate(split_t);
    let (left, right) = b.split(split_t);
    assert!(v3eq(left.evaluate(1.0), original_at_t));
    assert!(v3eq(right.evaluate(0.0), original_at_t));
}

#[test]
fn cubic_split_preserves_outer_endpoints() {
    let p0 = Vec3::new(0.0, 0.0, 0.0);
    let p3 = Vec3::new(3.0, 0.0, 0.0);
    let b = CubicBezier::new(p0, Vec3::new(1.0, 1.0, 0.0), Vec3::new(2.0, 1.0, 0.0), p3);
    let (left, right) = b.split(0.5);
    assert!(v3eq(left.evaluate(0.0), p0));
    assert!(v3eq(right.evaluate(1.0), p3));
}

#[test]
fn cubic_second_derivative_is_finite() {
    let b = CubicBezier::new(
        Vec3::ZERO,
        Vec3::new(1.0, 1.0, 0.0),
        Vec3::new(2.0, 1.0, 0.0),
        Vec3::new(3.0, 0.0, 0.0),
    );
    for i in 0..=10 {
        let t = i as f32 / 10.0;
        let d2 = b.second_derivative(t);
        assert!(d2.x.is_finite() && d2.y.is_finite() && d2.z.is_finite());
    }
}

#[test]
fn cubic_arc_length_exceeds_chord() {
    let b = CubicBezier::new(
        Vec3::ZERO,
        Vec3::new(0.5, 1.5, 0.0),
        Vec3::new(1.5, 1.5, 0.0),
        Vec3::new(2.0, 0.0, 0.0),
    );
    let chord = 2.0f32;
    let len = b.arc_length(64);
    assert!(len > chord, "arc_length {len} should exceed chord {chord}");
}

#[test]
fn cubic_sample_uniform_monotone_x() {
    // Curve from x=0 to x=3 — x should be monotonically increasing.
    let b = CubicBezier::new(
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 1.0, 0.0),
        Vec3::new(2.0, 1.0, 0.0),
        Vec3::new(3.0, 0.0, 0.0),
    );
    let mut buf = [Vec3::ZERO; 11];
    b.sample_uniform(10, &mut buf);
    for w in buf.windows(2) {
        assert!(w[1].x >= w[0].x - EPS, "x should be non-decreasing");
    }
}
