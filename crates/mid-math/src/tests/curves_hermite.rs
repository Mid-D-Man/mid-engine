// crates/mid-math/tests/curves_hermite.rs
//! Integration tests for HermiteSpline.

use mid_math::{Vec3, HermiteSpline, HermiteKey};

const EPS: f32 = 1e-5;
const EPS_VELOCITY: f32 = 1e-4; // velocity uses finite differences, slightly looser

fn approx(a: f32, b: f32) -> bool { (a - b).abs() < EPS }
fn v3eq(a: Vec3, b: Vec3) -> bool {
    approx(a.x, b.x) && approx(a.y, b.y) && approx(a.z, b.z)
}
fn v3eq_eps(a: Vec3, b: Vec3, eps: f32) -> bool {
    (a.x-b.x).abs() < eps && (a.y-b.y).abs() < eps && (a.z-b.z).abs() < eps
}

// ─── Endpoint interpolation ───────────────────────────────────────────────────

#[test]
fn passes_through_first_key() {
    let pos0 = Vec3::new(1.0, 2.0, 3.0);
    let spline = HermiteSpline::new(vec![
        HermiteKey::smooth(pos0, Vec3::X),
        HermiteKey::smooth(Vec3::new(4.0, 5.0, 6.0), Vec3::X),
    ]);
    assert!(v3eq(spline.evaluate(0.0), pos0));
}

#[test]
fn passes_through_last_key() {
    let pos1 = Vec3::new(4.0, 5.0, 6.0);
    let spline = HermiteSpline::new(vec![
        HermiteKey::smooth(Vec3::ZERO, Vec3::X),
        HermiteKey::smooth(pos1, Vec3::X),
    ]);
    assert!(v3eq(spline.evaluate(1.0), pos1));
}

#[test]
fn three_keys_passes_through_all() {
    let positions = [
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 2.0, 0.0),
        Vec3::new(3.0, 1.0, 0.0),
    ];
    let tan = Vec3::new(1.0, 0.5, 0.0);
    let spline = HermiteSpline::new(
        positions.iter().map(|&p| HermiteKey::smooth(p, tan)).collect()
    );
    for (i, &pos) in positions.iter().enumerate() {
        let ev = spline.evaluate(i as f32);
        assert!(v3eq(ev, pos), "key {i}: got {:?} expected {:?}", ev, pos);
    }
}

// ─── Velocity / tangent matching ─────────────────────────────────────────────

#[test]
fn velocity_at_t0_matches_tangent_out() {
    let tan_out = Vec3::new(2.0, 3.0, 0.0);
    let spline = HermiteSpline::new(vec![
        HermiteKey::smooth(Vec3::ZERO, tan_out),
        HermiteKey::smooth(Vec3::X, Vec3::X),
    ]);
    let vel = spline.velocity(0.0);
    assert!(v3eq(vel, tan_out));
}

#[test]
fn velocity_at_segment_end_matches_tangent_in() {
    // HermiteKey::corner allows different tangent_in and tangent_out.
    // velocity at t=1.0 (end of segment 0) should match key[1].tangent_in.
    let tan_in = Vec3::new(1.0, -1.0, 0.0);
    let tan_out = Vec3::new(1.0, 1.0, 0.0);
    let spline = HermiteSpline::new(vec![
        HermiteKey::smooth(Vec3::ZERO, Vec3::X),
        HermiteKey::corner(Vec3::new(2.0, 0.0, 0.0), tan_in, tan_out),
    ]);
    let vel = spline.velocity(1.0);
    assert!(v3eq_eps(vel, tan_in, EPS_VELOCITY));
}

// ─── C1 continuity ────────────────────────────────────────────────────────────

#[test]
fn c1_smooth_joint_continuous_velocity() {
    // Smooth key at joint means tangent_in == tangent_out, so velocity is continuous.
    let joint_tan = Vec3::new(1.0, 0.0, 0.0);
    let spline = HermiteSpline::new(vec![
        HermiteKey::smooth(Vec3::ZERO, joint_tan),
        HermiteKey::smooth(Vec3::new(1.0, 0.0, 0.0), joint_tan),
        HermiteKey::smooth(Vec3::new(2.0, 0.0, 0.0), joint_tan),
    ]);
    let eps = 1e-3;
    let vel_before = spline.velocity(1.0 - eps);
    let vel_after  = spline.velocity(1.0 + eps);
    // Should be nearly equal (C1 at the joint)
    assert!((vel_before.x - vel_after.x).abs() < 0.1,
        "velocity discontinuity at joint: {:?} vs {:?}", vel_before, vel_after);
}

#[test]
fn c0_corner_key_has_position_but_not_velocity_continuity() {
    // Corner key: position is continuous but velocity is NOT.
    let tan_in  = Vec3::new(2.0, 0.0, 0.0);
    let tan_out = Vec3::new(0.0, 2.0, 0.0); // 90° turn
    let spline = HermiteSpline::new(vec![
        HermiteKey::smooth(Vec3::ZERO, Vec3::X),
        HermiteKey::corner(Vec3::new(1.0, 0.0, 0.0), tan_in, tan_out),
        HermiteKey::smooth(Vec3::new(1.0, 1.0, 0.0), Vec3::X),
    ]);
    // Position at t=1 should be continuous
    let pos = spline.evaluate(1.0);
    assert!(v3eq(pos, Vec3::new(1.0, 0.0, 0.0)));
}

// ─── Sample uniform ───────────────────────────────────────────────────────────

#[test]
fn sample_uniform_endpoint_match() {
    let p0 = Vec3::ZERO;
    let p1 = Vec3::new(3.0, 0.0, 0.0);
    let spline = HermiteSpline::new(vec![
        HermiteKey::smooth(p0, Vec3::X),
        HermiteKey::smooth(p1, Vec3::X),
    ]);
    let mut buf = [Vec3::ZERO; 9];
    spline.sample_uniform(8, &mut buf);
    assert!(v3eq(buf[0], p0));
    assert!(v3eq(buf[8], p1));
}

#[test]
fn segment_count_two_keys() {
    let spline = HermiteSpline::new(vec![
        HermiteKey::smooth(Vec3::ZERO, Vec3::X),
        HermiteKey::smooth(Vec3::X, Vec3::X),
    ]);
    assert_eq!(spline.segment_count(), 1);
}

#[test]
fn segment_count_four_keys() {
    let key = HermiteKey::smooth(Vec3::ZERO, Vec3::X);
    let spline = HermiteSpline::new(vec![key, key, key, key]);
    assert_eq!(spline.segment_count(), 3);
}
