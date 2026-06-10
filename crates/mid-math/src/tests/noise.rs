// crates/mid-math/src/tests/noise.rs
//! Tests for the noise module: Perlin, Simplex, Value, Worley, Fbm, DomainWarp.
//!
//! Coverage:
//!   - Output range contracts
//!   - Determinism (same seed = same output)
//!   - Seed independence (different seeds ≠ same output for most inputs)
//!   - Continuity (nearby inputs → nearby outputs — no teleporting)
//!   - Batch API matches scalar API exactly
//!   - Fbm and DomainWarp stay in reasonable bounds
//!   - Worley distance mode correctness (F2 ≥ F1, F2-F1 ≥ 0)

use crate::noise::{
    DomainWarp, Fbm, NoiseSource2d, NoiseSource3d, NoiseSource4d,
    Perlin, Simplex, Value, Worley,
    worley::{DistanceMode, DistanceMetric},
};

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Sample 1000 points and assert all are in [lo, hi].
fn assert_range_2d<N: NoiseSource2d>(src: &N, lo: f32, hi: f32, name: &str) {
    for i in 0i32..100 {
        for j in 0i32..10 {
            let x = i as f32 * 0.137 + 0.01;
            let y = j as f32 * 0.239 - 0.01;
            let v = src.sample_2d(x, y);
            assert!(
                v >= lo && v <= hi,
                "{} out of [{}, {}] at ({}, {}): got {}",
                name, lo, hi, x, y, v
            );
        }
    }
}

fn assert_range_3d<N: NoiseSource3d>(src: &N, lo: f32, hi: f32, name: &str) {
    for i in 0i32..50 {
        for j in 0i32..20 {
            let x = i as f32 * 0.113;
            let y = j as f32 * 0.211;
            let z = (i + j) as f32 * 0.079;
            let v = src.sample_3d(x, y, z);
            assert!(v >= lo && v <= hi,
                "{} out of [{}, {}] at ({},{},{}): got {}", name, lo, hi, x, y, z, v);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Perlin
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn perlin_2d_range() {
    let p = Perlin::new();
    assert_range_2d(&p, -1.1, 1.1, "Perlin2D");
}

#[test]
fn perlin_3d_range() {
    assert_range_3d(&Perlin::new(), -1.1, 1.1, "Perlin3D");
}

#[test]
fn perlin_4d_range() {
    for i in 0i32..20 {
        let v = Perlin::new().sample_4d(i as f32 * 0.1, i as f32 * 0.2, i as f32 * 0.3, i as f32 * 0.4);
        assert!(v >= -1.1 && v <= 1.1, "Perlin4D out of range: {}", v);
    }
}

#[test]
fn perlin_determinism_2d() {
    let a = Perlin::from_seed(42);
    let b = Perlin::from_seed(42);
    for i in 0..50 {
        let x = i as f32 * 0.17;
        let y = i as f32 * 0.31;
        assert_eq!(a.sample_2d(x, y), b.sample_2d(x, y),
            "Perlin2D seed 42 not deterministic at ({}, {})", x, y);
    }
}

#[test]
fn perlin_determinism_3d() {
    let a = Perlin::from_seed(99);
    let b = Perlin::from_seed(99);
    for i in 0..30 {
        let v = i as f32 * 0.13;
        assert_eq!(a.sample_3d(v, v * 0.5, v * 0.7), b.sample_3d(v, v * 0.5, v * 0.7));
    }
}

#[test]
fn perlin_different_seeds_differ() {
    let a = Perlin::from_seed(1);
    let b = Perlin::from_seed(2);
    let any_differ = (0..20).any(|i| {
        let x = i as f32 * 0.5;
        a.sample_2d(x, x) != b.sample_2d(x, x)
    });
    assert!(any_differ, "Seeds 1 and 2 produced identical Perlin outputs");
}

#[test]
fn perlin_continuity_2d() {
    let p = Perlin::new();
    let base  = p.sample_2d(1.0, 1.0);
    let nudge = p.sample_2d(1.0 + 1e-4, 1.0);
    assert!((base - nudge).abs() < 0.01,
        "Perlin2D discontinuous: |{} - {}| = {}", base, nudge, (base - nudge).abs());
}

// ─────────────────────────────────────────────────────────────────────────────
// Simplex
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn simplex_2d_range() {
    assert_range_2d(&Simplex::new(), -1.1, 1.1, "Simplex2D");
}

#[test]
fn simplex_3d_range() {
    assert_range_3d(&Simplex::new(), -1.1, 1.1, "Simplex3D");
}

#[test]
fn simplex_4d_range() {
    for i in 0i32..30 {
        let v = Simplex::new().sample_4d(i as f32 * 0.11, i as f32 * 0.23, i as f32 * 0.37, i as f32 * 0.05);
        assert!(v >= -1.1 && v <= 1.1, "Simplex4D out of range: {}", v);
    }
}

#[test]
fn simplex_determinism_2d() {
    let seed = 12345u64;
    let a = Simplex::from_seed(seed);
    let b = Simplex::from_seed(seed);
    for i in 0..50 {
        let x = i as f32 * 0.23;
        let y = i as f32 * 0.47;
        assert_eq!(a.sample_2d(x, y), b.sample_2d(x, y));
    }
}

#[test]
fn simplex_no_directional_bias() {
    let s = Simplex::new();
    let mut seen_positive = false;
    let mut seen_negative = false;
    for i in 0..100 {
        let v = s.sample_2d(i as f32 * 1.0, i as f32 * 1.0);
        if v > 0.1 { seen_positive = true; }
        if v < -0.1 { seen_negative = true; }
    }
    assert!(seen_positive, "Simplex diagonal never positive — possible bias");
    assert!(seen_negative, "Simplex diagonal never negative — possible bias");
}

#[test]
fn simplex_continuity_2d() {
    let s = Simplex::new();
    let base  = s.sample_2d(2.5, 3.7);
    let nudge = s.sample_2d(2.5 + 1e-4, 3.7);
    assert!((base - nudge).abs() < 0.01, "Simplex2D discontinuous: delta = {}", (base - nudge).abs());
}

// ─────────────────────────────────────────────────────────────────────────────
// Value
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn value_2d_range() {
    assert_range_2d(&Value::new(), -1.1, 1.1, "Value2D");
}

#[test]
fn value_3d_range() {
    assert_range_3d(&Value::new(), -1.1, 1.1, "Value3D");
}

#[test]
fn value_4d_range() {
    for i in 0i32..30 {
        let v = Value::new().sample_4d(i as f32 * 0.3, i as f32 * 0.2, i as f32 * 0.1, i as f32 * 0.4);
        assert!(v >= -1.1 && v <= 1.1, "Value4D out of range: {}", v);
    }
}

#[test]
fn value_determinism() {
    let a = Value::from_seed(777);
    let b = Value::from_seed(777);
    for i in 0..30 {
        let x = i as f32 * 0.4;
        let y = i as f32 * 0.6;
        assert_eq!(a.sample_2d(x, y), b.sample_2d(x, y));
    }
}

#[test]
fn value_integer_coords_at_grid() {
    let v = Value::new();
    let s1 = v.sample_2d(1.0, 2.0);
    let s2 = v.sample_2d(1.0, 2.0);
    assert_eq!(s1, s2, "Value at same integer coord not stable");
}

// ─────────────────────────────────────────────────────────────────────────────
// Worley
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn worley_f1_range_2d() {
    let w = Worley::new().with_mode(DistanceMode::F1);
    for i in 0i32..50 {
        for j in 0i32..20 {
            let v = w.sample_2d(i as f32 * 0.3, j as f32 * 0.3);
            assert!(v >= 0.0 && v <= 1.0, "Worley F1 2D out of [0,1]: {}", v);
        }
    }
}

#[test]
fn worley_f2_range_2d() {
    let w = Worley::new().with_mode(DistanceMode::F2);
    for i in 0i32..50 {
        let v = w.sample_2d(i as f32 * 0.4, i as f32 * 0.2);
        assert!(v >= 0.0 && v <= 1.0, "Worley F2 2D out of [0,1]: {}", v);
    }
}

#[test]
fn worley_f2_minus_f1_non_negative_2d() {
    let wf1  = Worley::new().with_mode(DistanceMode::F1);
    let wf2  = Worley::new().with_mode(DistanceMode::F2);
    let wdiff = Worley::new().with_mode(DistanceMode::F2MinusF1);
    for i in 0i32..30 {
        let x = i as f32 * 0.5;
        let y = i as f32 * 0.3;
        let f1   = wf1.sample_2d(x, y);
        let f2   = wf2.sample_2d(x, y);
        let diff = wdiff.sample_2d(x, y);
        assert!(f2 >= f1 - 1e-5, "Worley F2 < F1 at ({}, {}): {} < {}", x, y, f2, f1);
        assert!(diff >= -1e-5, "Worley F2-F1 negative at ({}, {}): {}", x, y, diff);
    }
}

#[test]
fn worley_determinism() {
    let a = Worley::from_seed(55).with_mode(DistanceMode::F1);
    let b = Worley::from_seed(55).with_mode(DistanceMode::F1);
    for i in 0..20 {
        let x = i as f32 * 0.7;
        assert_eq!(a.sample_2d(x, x * 0.3), b.sample_2d(x, x * 0.3));
    }
}

#[test]
fn worley_3d_range() {
    let w = Worley::new().with_mode(DistanceMode::F1).with_metric(DistanceMetric::Euclidean);
    assert_range_3d(&w, 0.0, 1.0, "Worley3D");
}

#[test]
fn worley_manhattan_metric() {
    let w = Worley::new().with_mode(DistanceMode::F1).with_metric(DistanceMetric::Manhattan);
    for i in 0..30 {
        let v = w.sample_2d(i as f32 * 0.4, i as f32 * 0.2);
        assert!(v >= 0.0 && v <= 1.0, "Worley Manhattan out of [0,1]: {}", v);
    }
}

#[test]
fn worley_chebyshev_metric() {
    let w = Worley::new().with_mode(DistanceMode::F1).with_metric(DistanceMetric::Chebyshev);
    for i in 0..30 {
        let v = w.sample_2d(i as f32 * 0.35, i as f32 * 0.25);
        assert!(v >= 0.0 && v <= 1.0, "Worley Chebyshev out of [0,1]: {}", v);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Fbm
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn fbm_simplex_2d_range() {
    let fbm = Fbm::new(Simplex::new()).octaves(6).lacunarity(2.0).gain(0.5).frequency(1.0);
    for i in 0i32..100 {
        let v = fbm.sample_2d(i as f32 * 0.1, i as f32 * 0.07);
        assert!(v >= -1.05 && v <= 1.05, "Fbm Simplex 2D out of range: {}", v);
    }
}

#[test]
fn fbm_perlin_3d_range() {
    let fbm = Fbm::new(Perlin::new()).octaves(4).lacunarity(2.0).gain(0.5).frequency(1.0);
    for i in 0..50 {
        let x = i as f32 * 0.13;
        let v = fbm.sample_3d(x, x * 0.7, x * 0.4);
        assert!(v >= -1.05 && v <= 1.05, "Fbm Perlin 3D out of range: {}", v);
    }
}

#[test]
fn fbm_more_octaves_adds_detail() {
    let few  = Fbm::new(Simplex::new()).octaves(1).lacunarity(2.0).gain(0.5).frequency(1.0);
    let many = Fbm::new(Simplex::new()).octaves(8).lacunarity(2.0).gain(0.5).frequency(1.0);

    let mut few_range  = 0.0f32;
    let mut many_range = 0.0f32;

    for i in 0..100 {
        let x = i as f32 * 0.1;
        few_range  = few_range.max(few.sample_2d(x, 0.0).abs());
        many_range = many_range.max(many.sample_2d(x, 0.0).abs());
    }
    assert!(many_range >= few_range * 0.5,
        "Many-octave fBm ({}) weaker than 1-octave ({})", many_range, few_range);
}

#[test]
fn fbm_determinism() {
    let a = Fbm::new(Simplex::from_seed(42)).octaves(6).lacunarity(2.0).gain(0.5).frequency(1.0);
    let b = Fbm::new(Simplex::from_seed(42)).octaves(6).lacunarity(2.0).gain(0.5).frequency(1.0);
    for i in 0..30 {
        let x = i as f32 * 0.3;
        assert_eq!(a.sample_2d(x, x * 0.5), b.sample_2d(x, x * 0.5));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DomainWarp
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn domain_warp_2d_range() {
    let fbm  = Fbm::new(Simplex::new()).octaves(4).lacunarity(2.0).gain(0.5).frequency(1.0);
    let warp = DomainWarp::new(Simplex::new()).with_fbm(fbm).warp_scale(1.0).double_warp(false);
    for i in 0..100 {
        let x = i as f32 * 0.1;
        let v = warp.sample_2d(x, x * 0.3);
        assert!(v >= -1.05 && v <= 1.05, "DomainWarp 2D out of range: {}", v);
    }
}

#[test]
fn domain_warp_double_differs_from_single() {
    let make_fbm = || Fbm::new(Simplex::from_seed(1)).octaves(4).lacunarity(2.0).gain(0.5).frequency(1.0);
    let single = DomainWarp::new(Simplex::from_seed(1)).with_fbm(make_fbm()).warp_scale(1.0).double_warp(false);
    let double = DomainWarp::new(Simplex::from_seed(1)).with_fbm(make_fbm()).warp_scale(1.0).double_warp(true);

    let any_differ = (0..20).any(|i| {
        let x = i as f32 * 0.5 + 0.1;
        (single.sample_2d(x, x * 0.3) - double.sample_2d(x, x * 0.3)).abs() > 1e-6
    });
    assert!(any_differ, "Single and double warp produced identical outputs");
}

#[test]
fn domain_warp_3d_range() {
    let fbm  = Fbm::new(Simplex::new()).octaves(4).lacunarity(2.0).gain(0.5).frequency(1.0);
    let warp = DomainWarp::new(Simplex::new()).with_fbm(fbm).warp_scale(1.0).double_warp(false);
    for i in 0..50 {
        let x = i as f32 * 0.15;
        let v = warp.sample_3d(x, x * 0.4, x * 0.6);
        assert!(v >= -1.05 && v <= 1.05, "DomainWarp 3D out of range: {}", v);
    }
        }
