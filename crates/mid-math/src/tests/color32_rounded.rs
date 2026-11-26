// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-math.md, section "tests/color32_rounded.rs"
// ============================================================================
//! Tests for the rounded `Color32` helpers: `div255_round`, `premultiply_rounded`,
//! `blend_over_rounded` and `scale_alpha_rounded`.

use crate::color::Color32;

/// Round-half-up of `x / 255` as an exact rational, the reference for every
/// rounded helper below.
fn exact_div255_round(x: u32) -> u32 {
    (2 * x + 255) / 510
}

#[test]
fn div255_round_matches_the_exact_rational_for_every_input() {
    for x in 0..=65025u32 {
        assert_eq!(Color32::div255_round(x), exact_div255_round(x), "x = {x}");
    }
}

#[test]
fn premultiply_rounded_matches_exact_rounding_for_every_channel_and_alpha() {
    for c in 0..=255u8 {
        for a in 0..=255u8 {
            let p = Color32::new(c, c, c, a).premultiply_rounded();
            let want = exact_div255_round(c as u32 * a as u32) as u8;
            assert_eq!((p.r, p.g, p.b, p.a), (want, want, want, a), "c = {c}, a = {a}");
        }
    }
}

#[test]
fn premultiply_rounded_is_never_below_truncating_and_at_most_one_above() {
    for c in 0..=255u8 {
        for a in 0..=255u8 {
            let t = Color32::new(c, c, c, a).premultiply();
            let r = Color32::new(c, c, c, a).premultiply_rounded();
            assert!(r.r >= t.r && r.r - t.r <= 1, "c = {c}, a = {a}: {} vs {}", r.r, t.r);
        }
    }
}

#[test]
fn premultiply_rounded_keeps_the_endpoints() {
    let opaque = Color32::new(12, 200, 99, 255);
    assert_eq!(opaque.premultiply_rounded(), opaque);
    let clear = Color32::new(12, 200, 99, 0);
    assert_eq!(clear.premultiply_rounded(), Color32::new(0, 0, 0, 0));
}

#[test]
fn scale_alpha_rounded_matches_exact_rounding_for_every_pair() {
    for a in 0..=255u8 {
        for f in 0..=255u8 {
            let got = Color32::new(1, 2, 3, a).scale_alpha_rounded(f);
            let want = exact_div255_round(a as u32 * f as u32) as u8;
            assert_eq!(got, Color32::new(1, 2, 3, want), "a = {a}, f = {f}");
        }
    }
}

#[test]
fn blend_over_rounded_matches_exact_rounding_on_a_dense_grid() {
    for a in 1..=254u8 {
        for s in (0..=255u8).step_by(5) {
            for d in (0..=255u8).step_by(5) {
                let dst = Color32::new_opaque(d, d, d);
                let src = Color32::new(s, s, s, a);
                let got = dst.blend_over_rounded(src);
                let want = exact_div255_round(s as u32 * a as u32 + d as u32 * (255 - a as u32)) as u8;
                assert_eq!((got.r, got.g, got.b), (want, want, want), "s = {s}, d = {d}, a = {a}");
            }
        }
    }
}

#[test]
fn blend_over_rounded_keeps_the_endpoints() {
    let dst = Color32::new_opaque(10, 20, 30);
    assert_eq!(dst.blend_over_rounded(Color32::WHITE), Color32::WHITE);
    assert_eq!(dst.blend_over_rounded(Color32::TRANSPARENT), dst);
}

#[test]
fn blend_over_rounded_onto_an_opaque_destination_stays_opaque() {
    for a in 1..=254u8 {
        let got = Color32::new_opaque(9, 9, 9).blend_over_rounded(Color32::new(200, 100, 50, a));
        assert_eq!(got.a, 255, "a = {a}");
    }
}

#[test]
fn blend_over_rounded_alpha_never_needs_the_saturating_add() {
    for sa in 1..=254u32 {
        for da in 0..=255u32 {
            let sum = sa + Color32::div255_round(da * (255 - sa));
            assert!(sum <= 255, "sa = {sa}, da = {da}");
        }
    }
}
