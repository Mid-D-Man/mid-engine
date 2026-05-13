// crates/mid-math/tests/curves_kochanek_bartels.rs
//! Integration tests for KochanekBartels (TCB) spline.

use mid_math::{Vec3, KochanekBartels, TcbKey};

const EPS: f32 = 1e-4;

fn approx(a: f32, b: f32) -> bool { (a - b).abs() < EPS }
fn v3eq(a: Vec3, b: Vec3) -> bool {
    approx(a.x, b.x) && approx(a.y, b.y) && approx(a.z, b.z)
}

fn default_keys(n: usize) -> Vec<TcbKey<Vec3>> {
    (0..n)
        .map(|i| TcbKey::new(Vec3::new(i as f32, (i as f32).sin(), 0.0)))
        .collect()
}

// ─── Interpolating property ───────────────────────────────────────────────────

#[test]
fn passes_through_first_key() {
    let keys = default_keys(4);
    let p0   = keys[0].position;
    let sp   = KochanekBartels::new(keys);
    assert!(v3eq(sp.evaluate(0.0), p0));
}

#[test]
fn passes_through_last_key() {
    let keys = default_keys(4);
    let last = keys.last().unwrap().position;
    let sp   = KochanekBartels::new(keys);
    let n    = sp.segment_count() as f32;
    assert!(v3eq(sp.evaluate(n), last));
}

#[test]
fn passes_through_intermediate_keys() {
    let keys = default_keys(5);
    let positions: Vec<Vec3> = keys.iter().map(|k| k.position).collect();
    let sp = KochanekBartels::new(keys);
    for (i, &pos) in positions.iter().enumerate() {
        let ev = sp.evaluate(i as f32);
        assert!(v3eq(ev, pos), "key {i}: got {:?} expected {:?}", ev, pos);
    }
}

// ─── TCB parameter effects ────────────────────────────────────────────────────

#[test]
fn zero_tcb_matches_catmull_rom_shape() {
    // TCB with T=C=B=0 should produce same shape as Catmull-Rom.
    // We can't directly compare but we verify it passes through all keys.
    let keys: Vec<TcbKey<Vec3>> = (0..4)
        .map(|i| TcbKey::with_tcb(
            Vec3::new(i as f32, 0.0, 0.0),
            0.0, 0.0, 0.0,
        ))
        .collect();
    let sp = KochanekBartels::new(keys.clone());
    for (i, k) in keys.iter().enumerate() {
        let ev = sp.evaluate(i as f32);
        assert!(v3eq(ev, k.position));
    }
}

#[test]
fn tension_1_is_near_linear() {
    // tension=1 means zero tangents everywhere → piecewise linear-ish
    // Between two keys the midpoint should be close to the linear midpoint.
    let p0 = Vec3::new(0.0, 0.0, 0.0);
    let p1 = Vec3::new(2.0, 0.0, 0.0);
    let keys = vec![
        TcbKey::with_tcb(p0, 1.0, 0.0, 0.0),
        TcbKey::with_tcb(p1, 1.0, 0.0, 0.0),
    ];
    let sp  = KochanekBartels::new(keys);
    let mid = sp.evaluate(0.5);
    // With zero tangents the hermite basis gives exactly the linear midpoint.
    assert!(approx(mid.x, 1.0));
    assert!(approx(mid.y, 0.0));
}

#[test]
fn negative_tension_overshoots() {
    // tension=-1 (loose) should overshoot the endpoints.
    let p0 = Vec3::new(0.0, 0.0, 0.0);
    let p1 = Vec3::new(1.0, 0.0, 0.0);
    let keys = vec![
        TcbKey::with_tcb(Vec3::new(-1.0, 0.0, 0.0), -0.8, 0.0, 0.0),
        TcbKey::with_tcb(p0,                          -0.8, 0.0, 0.0),
        TcbKey::with_tcb(p1,                          -0.8, 0.0, 0.0),
        TcbKey::with_tcb(Vec3::new(2.0, 0.0, 0.0),  -0.8, 0.0, 0.0),
    ];
    let sp  = KochanekBartels::new(keys);
    // With negative tension the curve overshoots; max x between p0 and p1
    // should exceed 1.0.
    let max_x = (0..=20)
        .map(|i| sp.evaluate(1.0 + i as f32 / 20.0).x)
        .fold(f32::NEG_INFINITY, f32::max);
    assert!(max_x > 1.0, "expected overshoot, got max_x={max_x}");
}

// ─── Velocity ─────────────────────────────────────────────────────────────────

#[test]
fn velocity_is_finite_everywhere() {
    let sp = KochanekBartels::new(default_keys(5));
    let n  = sp.segment_count() as f32;
    for i in 0..=40 {
        let t = i as f32 / 40.0 * n;
        let v = sp.velocity(t);
        assert!(v.x.is_finite() && v.y.is_finite() && v.z.is_finite(),
            "velocity NaN at t={t}");
    }
}

// ─── Segment count ────────────────────────────────────────────────────────────

#[test]
fn segment_count_four_keys() {
    let sp = KochanekBartels::new(default_keys(4));
    assert_eq!(sp.segment_count(), 3);
}

// ─── Sample uniform ───────────────────────────────────────────────────────────

#[test]
fn sample_uniform_endpoints() {
    let keys = default_keys(4);
    let p0   = keys[0].position;
    let pn   = keys.last().unwrap().position;
    let sp   = KochanekBartels::new(keys);
    let mut buf = [Vec3::ZERO; 11];
    sp.sample_uniform(10, &mut buf);
    assert!(v3eq(buf[0], p0));
    assert!(v3eq(buf[10], pn));
}

#[test]
fn sample_uniform_all_finite() {
    let sp = KochanekBartels::new(default_keys(6));
    let mut buf = [Vec3::ZERO; 21];
    sp.sample_uniform(20, &mut buf);
    for p in &buf {
        assert!(p.x.is_finite() && p.y.is_finite(), "NaN in sample");
    }
                                                   }
