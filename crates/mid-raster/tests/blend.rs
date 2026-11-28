// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-raster.md, section "linear.rs"
// ============================================================================
//! Surfaces, the sRGB over, and the linear-light option with its tables.

use mid_math::Color32;
use mid_raster::linear::{decode, encode, over, over_channel};
use mid_raster::pixel::{over as blend, BlendSpace};
use mid_raster::tables::{DECODE, DECODE_FNV, THRESHOLDS, THRESHOLDS_FNV};
use mid_raster::{Surface, SurfaceMut};

fn fnv1a(bytes: impl Iterator<Item = u8>) -> u64 {
    let mut h = 0xCBF2_9CE4_8422_2325u64;
    for b in bytes {
        h = (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01B3);
    }
    h
}

// ── Surfaces ──────────────────────────────────────────────────────────────────

#[test]
fn a_surface_view_checks_its_stride_and_length() {
    let mut buf = vec![0u8; 4 * 3 * 2];
    assert!(SurfaceMut::new(&mut buf, 3, 2, 12).is_some());
    assert!(SurfaceMut::new(&mut buf, 3, 2, 11).is_none(), "stride below the row");
    assert!(SurfaceMut::new(&mut buf, 3, 3, 12).is_none(), "too short for the height");
    let mut padded = vec![0u8; 16 + 12];
    assert!(SurfaceMut::new(&mut padded, 3, 2, 16).is_some(), "the last row needs no padding");
    let mut tiny = vec![0u8; 15];
    assert!(SurfaceMut::new(&mut tiny, 3, 2, 16).is_none());
    assert!(SurfaceMut::new(&mut [], 0, 0, 0).is_some(), "an empty surface is fine");
    assert!(SurfaceMut::new(&mut [], 5, 0, 20).is_some(), "so is one with no rows");
}

#[test]
fn pixels_are_stored_r_g_b_a_and_the_stride_is_respected() {
    let mut buf = vec![7u8; 16 * 2];
    {
        let mut s = SurfaceMut::new(&mut buf, 3, 2, 16).unwrap();
        s.set(2, 1, Color32::new(1, 2, 3, 4));
        assert_eq!(s.pixel(2, 1), Color32::new(1, 2, 3, 4));
        assert_eq!((s.width(), s.height(), s.stride()), (3, 2, 16));
    }
    assert_eq!(&buf[16 + 8..16 + 12], &[1, 2, 3, 4]);
    assert_eq!(&buf[12..16], &[7, 7, 7, 7], "the padding is left alone");
    {
        let mut s = SurfaceMut::new(&mut buf, 3, 2, 16).unwrap();
        s.clear(Color32::new(9, 9, 9, 9));
    }
    assert_eq!(&buf[12..16], &[7, 7, 7, 7], "clear does not touch the padding either");
    assert_eq!(&buf[0..4], &[9, 9, 9, 9]);
}

#[test]
fn an_owned_surface_starts_transparent_and_is_drawn_through_a_view() {
    let mut s = Surface::new(4, 3);
    assert!(s.bytes().iter().all(|&b| b == 0));
    s.as_mut().set(3, 2, Color32::new(5, 6, 7, 8));
    assert_eq!(s.pixel(3, 2), Color32::new(5, 6, 7, 8));
    assert_eq!(s.bytes().len(), 4 * 3 * 4);
    assert_eq!((s.width(), s.height()), (4, 3));
}

// ── The sRGB over ─────────────────────────────────────────────────────────────

#[test]
fn the_srgb_over_is_the_rounded_helper_of_mid_math() {
    let dst = Color32::new(10, 100, 200, 255);
    for a in [0u8, 1, 64, 128, 200, 254, 255] {
        let src = Color32::new(250, 3, 77, a);
        assert_eq!(blend(dst, src, BlendSpace::Srgb), dst.blend_over_rounded(src));
    }
}

#[test]
fn half_white_over_opaque_black_is_128_in_srgb_and_stays_opaque() {
    let out = blend(Color32::new(0, 0, 0, 255), Color32::new(255, 255, 255, 128), BlendSpace::Srgb);
    assert_eq!(out, Color32::new(128, 128, 128, 255));
}

// ── The tables ────────────────────────────────────────────────────────────────

#[test]
fn the_decode_table_is_the_srgb_curve_at_its_known_points() {
    assert_eq!((decode(0), decode(255)), (0, 65535));
    assert_eq!((decode(1), decode(2), decode(128)), (20, 40, 14146));
    for v in 0..255u8 {
        assert!(decode(v) < decode(v + 1), "strictly increasing at {v}");
    }
}

#[test]
fn the_thresholds_are_the_midpoints_of_neighboring_decoded_levels() {
    for v in 0..255usize {
        assert_eq!(THRESHOLDS[v], ((DECODE[v] as u32 + DECODE[v + 1] as u32) / 2) as u16, "at {v}");
    }
}

#[test]
fn encoding_a_decoded_level_gives_the_level_back() {
    for v in 0..=255u8 {
        assert_eq!(encode(decode(v)), v);
    }
}

#[test]
fn encode_is_the_nearest_level_in_the_linear_domain_with_ties_to_the_lower() {
    for l in 0..=65535u32 {
        let mut best = 0usize;
        let mut best_d = u32::MAX;
        for (v, &dv) in DECODE.iter().enumerate() {
            let d = (dv as i64 - l as i64).unsigned_abs() as u32;
            if d < best_d {
                best_d = d;
                best = v;
            }
        }
        assert_eq!(encode(l as u16) as usize, best, "linear value {l}");
    }
}

#[test]
fn the_tables_still_match_their_generated_checksums() {
    let d = fnv1a(DECODE.iter().flat_map(|v| v.to_le_bytes()));
    let t = fnv1a(THRESHOLDS.iter().flat_map(|v| v.to_le_bytes()));
    assert_eq!(d, DECODE_FNV, "decode table edited by hand");
    assert_eq!(t, THRESHOLDS_FNV, "threshold table edited by hand");
}

// ── The linear-light over ─────────────────────────────────────────────────────

#[test]
fn half_white_over_black_is_188_in_linear_light() {
    let out = blend(Color32::new(0, 0, 0, 255), Color32::new(255, 255, 255, 128), BlendSpace::Linear);
    assert_eq!(out, Color32::new(188, 188, 188, 255));
}

#[test]
fn the_linear_over_keeps_the_endpoints_and_stays_between_its_inputs() {
    let dst = Color32::new(10, 100, 200, 255);
    assert_eq!(over(dst, Color32::new(1, 2, 3, 255)), Color32::new(1, 2, 3, 255));
    assert_eq!(over(dst, Color32::new(1, 2, 3, 0)), dst);
    for s in (0..=255u8).step_by(5) {
        for d in (0..=255u8).step_by(5) {
            for a in (0..=255u8).step_by(5) {
                let v = over_channel(s, d, a);
                assert!(v >= s.min(d) && v <= s.max(d), "s {s} d {d} a {a}: {v}");
            }
            assert_eq!(over_channel(s, d, 255), s);
            assert_eq!(over_channel(s, d, 0), d);
        }
    }
}

#[test]
fn linear_blending_is_brighter_than_srgb_for_a_light_source_over_dark() {
    for a in [32u8, 64, 128, 192] {
        let l = over_channel(255, 0, a);
        let s = blend(Color32::new(0, 0, 0, 255), Color32::new(255, 255, 255, a), BlendSpace::Srgb).r;
        assert!(l > s, "alpha {a}: {l} against {s}");
    }
}

#[test]
fn the_alpha_of_a_linear_over_accumulates_like_the_srgb_one() {
    for da in [0u8, 100, 255] {
        for a in [1u8, 90, 254] {
            let dst = Color32::new(5, 5, 5, da);
            let src = Color32::new(200, 100, 50, a);
            assert_eq!(over(dst, src).a, dst.blend_over_rounded(src).a, "da {da} a {a}");
        }
    }
}

#[test]
fn the_default_blend_space_is_srgb() {
    assert_eq!(BlendSpace::default(), BlendSpace::Srgb);
}
