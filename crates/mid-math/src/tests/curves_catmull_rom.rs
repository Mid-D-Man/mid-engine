// crates/mid-math/tests/curves_catmull_rom.rs
//! Integration tests for CatmullRom spline.

use mid_math::{Vec3, CatmullRom, CatmullRomAlpha};

const EPS: f32 = 1e-4;

fn approx(a: f32, b: f32) -> bool { (a - b).abs() < EPS }
fn v3eq(a: Vec3, b: Vec3) -> bool {
    approx(a.x, b.x) && approx(a.y, b.y) && approx(a.z, b.z)
}

fn pts4() -> Vec<Vec3> {
    vec![
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 1.0, 0.0),
        Vec3::new(2.0, 0.0, 0.0),
        Vec3::new(3.0, 1.0, 0.0),
    ]
}

// ─── Interpolating property ───────────────────────────────────────────────────

#[test]
fn passes_through_point_0() {
    let cr = CatmullRom::new(pts4());
    assert!(v3eq(cr.evaluate(0.0), Vec3::new(0.0, 0.0, 0.0)));
}

#[test]
fn passes_through_point_1() {
    let cr = CatmullRom::new(pts4());
    assert!(v3eq(cr.evaluate(1.0), Vec3::new(1.0, 1.0, 0.0)));
}

#[test]
fn passes_through_point_2() {
    let cr = CatmullRom::new(pts4());
    assert!(v3eq(cr.evaluate(2.0), Vec3::new(2.0, 0.0, 0.0)));
}

#[test]
fn passes_through_last_point() {
    let pts = pts4();
    let last = *pts.last().unwrap();
    let cr = CatmullRom::new(pts);
    let n = cr.segment_count() as f32;
    assert!(v3eq(cr.evaluate(n), last));
}

// ─── Metadata ─────────────────────────────────────────────────────────────────

#[test]
fn segment_count_correct() {
    let cr = CatmullRom::new(pts4());
    assert_eq!(cr.segment_count(), 3);
}

#[test]
fn two_point_segment_count() {
    let cr = CatmullRom::new(vec![Vec3::ZERO, Vec3::X]);
    assert_eq!(cr.segment_count(), 1);
}

// ─── Two-point edge case ──────────────────────────────────────────────────────

#[test]
fn two_point_passes_through_both() {
    let p0 = Vec3::ZERO;
    let p1 = Vec3::new(2.0, 0.0, 0.0);
    let cr = CatmullRom::new(vec![p0, p1]);
    assert!(v3eq(cr.evaluate(0.0), p0));
    assert!(v3eq(cr.evaluate(1.0), p1));
}

// ─── Alpha variants ───────────────────────────────────────────────────────────

#[test]
fn centripetal_still_interpolates() {
    let pts = pts4();
    let cr = CatmullRom::with_alpha(pts.clone(), CatmullRomAlpha::Centripetal);
    for (i, &p) in pts.iter().enumerate() {
        let ev = cr.evaluate(i as f32);
        assert!(v3eq(ev, p), "centripetal: failed at i={i}: got {:?}", ev);
    }
}

#[test]
fn chordal_still_interpolates() {
    let pts = pts4();
    let cr = CatmullRom::with_alpha(pts.clone(), CatmullRomAlpha::Chordal);
    for (i, &p) in pts.iter().enumerate() {
        let ev = cr.evaluate(i as f32);
        assert!(v3eq(ev, p), "chordal: failed at i={i}: got {:?}", ev);
    }
}

#[test]
fn centripetal_no_nan_non_uniform_spacing() {
    // Deliberately non-uniform spacing — centripetal should handle gracefully.
    let pts = vec![
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.001, 0.0, 0.0), // very close to previous
        Vec3::new(1.0, 1.0, 0.0),
        Vec3::new(2.0, 0.0, 0.0),
    ];
    let cr = CatmullRom::with_alpha(pts, CatmullRomAlpha::Centripetal);
    for i in 0..=20 {
        let t = i as f32 * cr.segment_count() as f32 / 20.0;
        let p = cr.evaluate(t);
        assert!(p.x.is_finite() && p.y.is_finite(), "NaN at t={t}");
    }
}

// ─── Tangent ──────────────────────────────────────────────────────────────────

#[test]
fn tangent_midway_is_finite() {
    let cr = CatmullRom::new(pts4());
    let t = cr.tangent(1.5);
    assert!(t.x.is_finite() && t.y.is_finite() && t.z.is_finite());
}

#[test]
fn tangent_at_start_points_forward() {
    // Points go in +x direction overall; tangent at t=0 should have positive x.
    let pts = vec![
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(2.0, 0.0, 0.0),
        Vec3::new(3.0, 0.0, 0.0),
    ];
    let cr = CatmullRom::new(pts);
    let t = cr.tangent(0.0);
    assert!(t.x > 0.0);
}

// ─── Sample uniform ───────────────────────────────────────────────────────────

#[test]
fn sample_uniform_endpoints_match() {
    let pts = pts4();
    let first = pts[0];
    let last  = *pts.last().unwrap();
    let cr = CatmullRom::new(pts);
    let mut buf = [Vec3::ZERO; 11];
    cr.sample_uniform(10, &mut buf);
    assert!(v3eq(buf[0], first));
    assert!(v3eq(buf[10], last));
}

#[test]
fn sample_uniform_all_finite() {
    let cr = CatmullRom::new(pts4());
    let mut buf = [Vec3::ZERO; 21];
    cr.sample_uniform(20, &mut buf);
    for p in &buf {
        assert!(p.x.is_finite() && p.y.is_finite() && p.z.is_finite());
    }
}

// ─── Scalar type ─────────────────────────────────────────────────────────────

#[test]
fn f32_passes_through_all_values() {
    let vals = vec![0.0f32, 1.0, 4.0, 9.0];
    let cr = CatmullRom::new(vals.clone());
    for (i, &v) in vals.iter().enumerate() {
        let ev = cr.evaluate(i as f32);
        assert!((ev - v).abs() < EPS, "f32: at i={i} got {ev}, expected {v}");
    }
  }
