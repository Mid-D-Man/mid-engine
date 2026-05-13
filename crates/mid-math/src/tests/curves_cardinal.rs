// crates/mid-math/tests/curves_cardinal.rs
//! Integration tests for CardinalSpline.

use mid_math::{Vec3, CardinalSpline};

const EPS: f32 = 1e-4;

fn approx(a: f32, b: f32) -> bool { (a - b).abs() < EPS }
fn v3eq(a: Vec3, b: Vec3) -> bool {
    approx(a.x, b.x) && approx(a.y, b.y) && approx(a.z, b.z)
}

fn pts() -> Vec<Vec3> {
    vec![
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 1.0, 0.0),
        Vec3::new(2.0, 0.0, 0.0),
        Vec3::new(3.0, 1.0, 0.0),
    ]
}

// ─── Interpolating property ───────────────────────────────────────────────────

#[test]
fn passes_through_first_point() {
    let sp = CardinalSpline::new(pts(), 0.0);
    assert!(v3eq(sp.evaluate(0.0), Vec3::new(0.0, 0.0, 0.0)));
}

#[test]
fn passes_through_last_point() {
    let p  = pts();
    let last = *p.last().unwrap();
    let sp = CardinalSpline::new(p, 0.0);
    let n  = sp.segment_count() as f32;
    assert!(v3eq(sp.evaluate(n), last));
}

#[test]
fn passes_through_all_intermediate_points() {
    let p  = pts();
    let sp = CardinalSpline::new(p.clone(), 0.0);
    for (i, &pos) in p.iter().enumerate() {
        let ev = sp.evaluate(i as f32);
        assert!(v3eq(ev, pos), "i={i}: got {:?} expected {:?}", ev, pos);
    }
}

// ─── Tension parameter ────────────────────────────────────────────────────────

#[test]
fn tension_0_is_catmull_rom() {
    // tension=0 → identical to catmull_rom() constructor
    let p  = pts();
    let t0 = CardinalSpline::new(p.clone(), 0.0);
    let cr = CardinalSpline::catmull_rom(p);
    // Both should pass through all control points
    for i in 0..=3 {
        let a = t0.evaluate(i as f32);
        let b = cr.evaluate(i as f32);
        assert!(v3eq(a, b), "mismatch at i={i}");
    }
}

#[test]
fn tension_1_midpoint_is_linear() {
    // tension=1 → zero tangents → piecewise linear → midpoint exactly at midpoint
    let p0 = Vec3::new(0.0, 0.0, 0.0);
    let p1 = Vec3::new(2.0, 0.0, 0.0);
    let sp = CardinalSpline::new(vec![p0, p1], 1.0);
    let mid = sp.evaluate(0.5);
    assert!(approx(mid.x, 1.0));
    assert!(approx(mid.y, 0.0));
}

#[test]
fn tension_clamps_stay_near_hull() {
    // High tension should bring curve tighter to the straight line between points.
    let p = pts();
    let tight  = CardinalSpline::new(p.clone(), 0.9);
    let loose  = CardinalSpline::new(p,         0.0);
    // At t=0.5 (midway through first segment) tight should deviate less from
    // the linear interpolation of p[0]→p[1] than loose.
    let linear_mid = Vec3::new(0.5, 0.5, 0.0);
    let tight_mid  = tight.evaluate(0.5);
    let loose_mid  = loose.evaluate(0.5);
    let tight_dev  = (tight_mid - linear_mid).length();
    let loose_dev  = (loose_mid - linear_mid).length();
    assert!(tight_dev <= loose_dev + EPS,
        "tight dev {tight_dev} should be <= loose dev {loose_dev}");
}

// ─── Segment count ────────────────────────────────────────────────────────────

#[test]
fn segment_count_four_points() {
    let sp = CardinalSpline::new(pts(), 0.0);
    assert_eq!(sp.segment_count(), 3);
}

#[test]
fn segment_count_two_points() {
    let sp = CardinalSpline::new(vec![Vec3::ZERO, Vec3::X], 0.0);
    assert_eq!(sp.segment_count(), 1);
}

// ─── Sample uniform ───────────────────────────────────────────────────────────

#[test]
fn sample_uniform_endpoints_match() {
    let p  = pts();
    let p0 = p[0];
    let pn = *p.last().unwrap();
    let sp = CardinalSpline::new(p, 0.0);
    let mut buf = [Vec3::ZERO; 11];
    sp.sample_uniform(10, &mut buf);
    assert!(v3eq(buf[0], p0));
    assert!(v3eq(buf[10], pn));
}

#[test]
fn sample_uniform_all_finite() {
    let sp = CardinalSpline::new(pts(), 0.5);
    let mut buf = [Vec3::ZERO; 21];
    sp.sample_uniform(20, &mut buf);
    for p in &buf {
        assert!(p.x.is_finite() && p.y.is_finite());
    }
                  }
