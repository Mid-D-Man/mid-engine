// crates/mid-math/tests/prng.rs
//! Xorshift64 PRNG tests.
//!
//! Tests cover:
//!   - Non-zero initial state invariant
//!   - Determinism: same seed → same sequence
//!   - Period: state never revisits zero after many iterations
//!   - Float outputs in [0, 1)
//!   - Range functions: bounds are respected
//!   - Bool probability: converges to p
//!   - State save/restore
//!   - Statistical sanity (mean, no clumping)

use mid_math::Xorshift64;

// ── Construction ──────────────────────────────────────────────────────────────

#[test]
#[should_panic(expected = "seed must be non-zero")]
fn new_panics_on_zero_seed() {
    let _ = Xorshift64::new(0);
}

#[test]
fn new_safe_with_zero_seed_uses_one() {
    let mut rng = Xorshift64::new_safe(0);
    // Should not panic and should produce output
    let v = rng.next_u64();
    assert_ne!(v, 0, "first output of seed=1 should be non-zero");
}

#[test]
fn new_safe_with_nonzero_preserves_seed() {
    let mut a = Xorshift64::new_safe(12345);
    let mut b = Xorshift64::new(12345);
    // First 10 values must match
    for _ in 0..10 {
        assert_eq!(a.next_u64(), b.next_u64());
    }
}

// ── Determinism ───────────────────────────────────────────────────────────────

#[test]
fn same_seed_same_sequence() {
    let mut a = Xorshift64::new(0xDEAD_BEEF_CAFE_1234);
    let mut b = Xorshift64::new(0xDEAD_BEEF_CAFE_1234);

    for _ in 0..1000 {
        assert_eq!(a.next_u64(), b.next_u64());
    }
}

#[test]
fn different_seeds_different_sequences() {
    let mut a = Xorshift64::new(1);
    let mut b = Xorshift64::new(2);

    let va: Vec<u64> = (0..20).map(|_| a.next_u64()).collect();
    let vb: Vec<u64> = (0..20).map(|_| b.next_u64()).collect();
    // Very unlikely to match — if they do, there's a serious bug
    assert_ne!(va, vb);
}

// ── State never zero ──────────────────────────────────────────────────────────

#[test]
fn state_never_revisits_zero_in_10k_steps() {
    let mut rng = Xorshift64::new(42);
    for i in 0..10_000 {
        rng.next_u64();
        assert_ne!(rng.state(), 0, "state was zero after {} steps", i + 1);
    }
}

// ── f32 output ────────────────────────────────────────────────────────────────

#[test]
fn f32_is_in_unit_interval() {
    let mut rng = Xorshift64::new(99);
    for _ in 0..10_000 {
        let v = rng.f32();
        assert!(v >= 0.0, "f32 below 0: {}", v);
        assert!(v < 1.0,  "f32 at or above 1: {}", v);
    }
}

#[test]
fn f32_mean_is_approximately_half() {
    let mut rng = Xorshift64::new(12345);
    let n = 100_000;
    let sum: f64 = (0..n).map(|_| rng.f32() as f64).sum();
    let mean = sum / n as f64;
    // Mean of uniform [0,1) should be 0.5 ± 0.01 for 100k samples
    assert!(
        (mean - 0.5).abs() < 0.01,
        "mean {:.4} too far from 0.5", mean
    );
}

// ── f64 output ────────────────────────────────────────────────────────────────

#[test]
fn f64_is_in_unit_interval() {
    let mut rng = Xorshift64::new(7777);
    for _ in 0..10_000 {
        let v = rng.f64();
        assert!(v >= 0.0, "f64 below 0: {}", v);
        assert!(v < 1.0,  "f64 at or above 1: {}", v);
    }
}

// ── range_f32 ─────────────────────────────────────────────────────────────────

#[test]
fn range_f32_stays_in_bounds() {
    let mut rng = Xorshift64::new(555);
    for _ in 0..10_000 {
        let v = rng.range_f32(-5.0, 5.0);
        assert!(v >= -5.0 && v < 5.0, "out of range: {}", v);
    }
}

#[test]
fn range_f32_negative_range() {
    let mut rng = Xorshift64::new(111);
    for _ in 0..1000 {
        let v = rng.range_f32(-100.0, -1.0);
        assert!(v >= -100.0 && v < -1.0, "out of range: {}", v);
    }
}

// ── range_u32 ─────────────────────────────────────────────────────────────────

#[test]
fn range_u32_stays_in_bounds() {
    let mut rng = Xorshift64::new(999);
    for _ in 0..10_000 {
        let v = rng.range_u32(10, 20);
        assert!(v >= 10 && v < 20, "out of range: {}", v);
    }
}

#[test]
fn range_u32_covers_full_range() {
    let mut rng = Xorshift64::new(314);
    let mut seen = [false; 10];
    // With 10k samples over range [0,10) we expect all values seen
    for _ in 0..10_000 {
        seen[rng.range_u32(0, 10) as usize] = true;
    }
    for (i, &s) in seen.iter().enumerate() {
        assert!(s, "value {} never appeared in 10k samples", i);
    }
}

#[test]
#[should_panic(expected = "lo must be < hi")]
fn range_u32_panics_on_equal_bounds() {
    let mut rng = Xorshift64::new(1);
    rng.range_u32(5, 5);
}

#[test]
#[should_panic(expected = "lo must be < hi")]
fn range_u32_panics_when_lo_gt_hi() {
    let mut rng = Xorshift64::new(1);
    rng.range_u32(10, 5);
}

// ── range_u64 ─────────────────────────────────────────────────────────────────

#[test]
fn range_u64_stays_in_bounds() {
    let mut rng = Xorshift64::new(2048);
    let lo: u64 = 1_000_000_000_000;
    let hi: u64 = 2_000_000_000_000;
    for _ in 0..1000 {
        let v = rng.range_u64(lo, hi);
        assert!(v >= lo && v < hi, "out of range: {}", v);
    }
}

// ── bool_p ────────────────────────────────────────────────────────────────────

#[test]
fn bool_p_zero_always_false() {
    let mut rng = Xorshift64::new(1);
    for _ in 0..1000 {
        assert!(!rng.bool_p(0.0));
    }
}

#[test]
fn bool_p_one_always_true() {
    let mut rng = Xorshift64::new(1);
    for _ in 0..1000 {
        assert!(rng.bool_p(1.0));
    }
}

#[test]
fn bool_p_half_converges() {
    let mut rng = Xorshift64::new(0xABCDEF);
    let n = 100_000;
    let trues: u64 = (0..n).filter(|_| rng.bool_p(0.5)).count() as u64;
    let ratio = trues as f64 / n as f64;
    // Should be within 1% of 0.5
    assert!(
        (ratio - 0.5).abs() < 0.01,
        "bool_p(0.5) ratio {:.4} too far from 0.5", ratio
    );
}

#[test]
fn bool_p_clamps_above_one() {
    // p > 1.0 should clamp to always-true
    let mut rng = Xorshift64::new(1);
    for _ in 0..1000 {
        assert!(rng.bool_p(2.0));
    }
}

#[test]
fn bool_p_clamps_below_zero() {
    // p < 0.0 should clamp to always-false
    let mut rng = Xorshift64::new(1);
    for _ in 0..1000 {
        assert!(!rng.bool_p(-1.0));
    }
}

// ── State save/restore ────────────────────────────────────────────────────────

#[test]
fn state_save_and_restore() {
    let mut rng = Xorshift64::new(0x1234_5678_9ABC_DEF0);

    // Advance 100 steps
    for _ in 0..100 {
        rng.next_u64();
    }

    // Save state
    let saved = rng.state();
    let next_10: Vec<u64> = (0..10).map(|_| rng.next_u64()).collect();

    // Restore state and replay
    rng.set_state(saved);
    let replayed: Vec<u64> = (0..10).map(|_| rng.next_u64()).collect();

    assert_eq!(next_10, replayed, "restored sequence doesn't match");
}

#[test]
#[should_panic(expected = "state must be non-zero")]
fn set_state_panics_on_zero() {
    let mut rng = Xorshift64::new(1);
    rng.set_state(0);
}

// ── Debug format ─────────────────────────────────────────────────────────────

#[test]
fn debug_shows_hex_state() {
    let rng = Xorshift64::new(0xDEAD_BEEF);
    let dbg = format!("{:?}", rng);
    assert!(dbg.contains("Xorshift64"), "got: {}", dbg);
    assert!(dbg.contains("0x"), "no hex in: {}", dbg);
}

// ── Statistical: no long runs of identical u32 ────────────────────────────────

#[test]
fn no_long_run_of_identical_values() {
    let mut rng = Xorshift64::new(0xFEED_FACE);
    let mut prev = rng.next_u32();
    let mut run_len = 1usize;

    for _ in 0..10_000 {
        let cur = rng.next_u32();
        if cur == prev {
            run_len += 1;
            assert!(run_len < 5, "run of identical u32 values >= 5 — likely broken");
        } else {
            run_len = 1;
        }
        prev = cur;
    }
}

// ── next_u32 uses high bits ───────────────────────────────────────────────────

#[test]
fn next_u32_is_high_32_of_next_u64() {
    // Reset two rngs to same seed, verify the relationship
    let mut rng_a = Xorshift64::new(0xBEEF_CAFE);
    let mut rng_b = Xorshift64::new(0xBEEF_CAFE);

    for _ in 0..100 {
        let u64_val = rng_a.next_u64();
        let u32_val = rng_b.next_u32();
        assert_eq!(u32_val, (u64_val >> 32) as u32);
    }
  }
