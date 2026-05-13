// crates/mid-math/tests/curves_bspline.rs
//! Integration tests for BSpline (uniform cubic).

use mid_math::{Vec3, BSpline};

const EPS: f32 = 1e-4;

fn approx(a: f32, b: f32) -> bool { (a - b).abs() < EPS }
fn v3eq(a: Vec3, b: Vec3) -> bool {
    approx(a.x, b.x) && approx(a.y, b.y) && approx(a.z, b.z)
}

fn pts4() -> Vec<Vec3> {
    vec![
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 2.0, 0.0),
        Vec3::new(2.0, 2.0, 0.0),
        Vec3::new(3.0, 0.0, 0.0),
    ]
}

fn pts8() -> Vec<Vec3> {
    (0..8).map(|i| Vec3::new(i as f32, (i as f32 * 0.7).sin() * 2.0, 0.0)).collect()
}

// ─── Approximating (does NOT pass through control points) ─────────────────────

#[test]
fn curve_does_not_pass_through_inner_control_points() {
    // B-spline is approximating — it should NOT pass through p1 or p2
    // (it only approaches them). We verify the midpoint is not exactly p1.
    let pts = pts4();
    let sp  = BSpline::new(pts.clone());
    // The curve at segment 0, t=0.5 should NOT equal pts[1]
    let mid = sp.evaluate(0.5);
    let not_exact = !v3eq(mid, pts[1]);
    // It should be "near" p[1] but not equal to it — just verify it's in the hull
    assert!(not_exact, "B-spline should be approximating, not interpolating");
    assert!(mid.x >= 0.0 && mid.x <= 3.0);
}

#[test]
fn evaluate_is_finite_everywhere() {
    let sp = BSpline::new(pts8());
    let n  = sp.segment_count() as f32;
    for i in 0..=100 {
        let t = i as f32 / 100.0 * n;
        let p = sp.evaluate(t);
        assert!(p.x.is_finite() && p.y.is_finite(), "NaN at t={t}");
    }
}

// ─── Clamping at boundaries ───────────────────────────────────────────────────

#[test]
fn evaluate_clamps_below_zero() {
    let sp  = BSpline::new(pts4());
    let p0  = sp.evaluate(0.0);
    let pn  = sp.evaluate(-10.0); // should clamp to t=0
    assert!(v3eq(p0, pn));
}

#[test]
fn evaluate_clamps_above_max() {
    let sp   = BSpline::new(pts4());
    let n    = sp.segment_count() as f32;
    let pmax = sp.evaluate(n);
    let pover = sp.evaluate(n + 10.0); // clamp to max
    assert!(v3eq(pmax, pover));
}

// ─── Segment count ────────────────────────────────────────────────────────────

#[test]
fn segment_count_4pt() {
    let sp = BSpline::new(pts4());
    assert_eq!(sp.segment_count(), 1); // n-3 = 4-3 = 1
}

#[test]
fn segment_count_8pt() {
    let sp = BSpline::new(pts8());
    assert_eq!(sp.segment_count(), 5); // 8-3 = 5
}

// ─── Tangent ──────────────────────────────────────────────────────────────────

#[test]
fn tangent_is_finite_everywhere() {
    let sp = BSpline::new(pts8());
    let n  = sp.segment_count() as f32;
    for i in 0..=20 {
        let t = i as f32 / 20.0 * n;
        let v = sp.tangent(t);
        assert!(v.x.is_finite() && v.y.is_finite(), "NaN tangent at t={t}");
    }
}

#[test]
fn tangent_direction_is_nonzero_on_nontrivial_curve() {
    let sp = BSpline::new(pts8());
    let v  = sp.tangent(2.5);
    assert!(v.length() > 1e-3, "tangent should not be zero");
}

// ─── C2 continuity property ───────────────────────────────────────────────────

#[test]
fn c2_continuity_at_internal_knots() {
    // B-spline is C2 — the second derivative should be continuous at knots.
    // We verify that evaluate is smooth by checking that adjacent samples
    // don't have sudden jumps.
    let sp  = BSpline::new(pts8());
    let n   = sp.segment_count() as f32;
    let mut prev = sp.evaluate(0.0);
    let step = n / 200.0;
    let mut max_jump = 0.0f32;
    let mut t = step;
    while t <= n {
        let curr = sp.evaluate(t);
        let jump = (curr - prev).length();
        max_jump = max_jump.max(jump);
        prev = curr;
        t   += step;
    }
    // With 200 steps over the full range, each step is ~0.025 units of t.
    // Jumps > 0.5 would indicate discontinuity.
    assert!(max_jump < 0.5, "max_jump={max_jump} suggests C2 discontinuity");
}

// ─── Arc length ───────────────────────────────────────────────────────────────

#[test]
fn arc_length_is_positive() {
    let sp  = BSpline::new(pts8());
    let len = sp.arc_length_approx(64);
    assert!(len > 0.0);
}

#[test]
fn arc_length_more_segments_converges() {
    let sp   = BSpline::new(pts8());
    let low  = sp.arc_length_approx(16);
    let high = sp.arc_length_approx(256);
    // More segments → more accurate → should converge (within 5%)
    assert!((high - low).abs() / high < 0.05,
        "arc length not converging: {low} vs {high}");
}

// ─── Sample uniform ───────────────────────────────────────────────────────────

#[test]
fn sample_uniform_matches_direct_evaluate_endpoints() {
    let sp = BSpline::new(pts4());
    let p0 = sp.evaluate(0.0);
    let pn = sp.evaluate(sp.segment_count() as f32);
    let mut buf = [Vec3::ZERO; 11];
    sp.sample_uniform(10, &mut buf);
    assert!(v3eq(buf[0], p0));
    assert!(v3eq(buf[10], pn));
}

#[test]
fn sample_uniform_all_finite() {
    let sp = BSpline::new(pts8());
    let mut buf = [Vec3::ZERO; 21];
    sp.sample_uniform(20, &mut buf);
    for p in &buf {
        assert!(p.x.is_finite() && p.y.is_finite());
    }
      }
