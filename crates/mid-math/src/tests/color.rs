// crates/mid-math/tests/color.rs
//! Color type tests: sRGB, linear, hex parsing, tone mapping, Color32.

use mid_math::{Rgb, Rgba, Color32};

// ── sRGB roundtrip ────────────────────────────────────────────────────────────

#[test]
fn linear_to_srgb_midgray() {
    // Linear 0.5 → sRGB is approximately 0.735 (not 0.5)
    let linear = Rgb::new(0.5, 0.5, 0.5);
    let srgb = linear.to_srgb();
    assert!((srgb.r - 0.7353).abs() < 0.001, "got {}", srgb.r);
    assert_eq!(srgb.r, srgb.g);
    assert_eq!(srgb.g, srgb.b);
}

#[test]
fn srgb_to_linear_midgray() {
    // sRGB 0.5 → linear is approximately 0.214
    let srgb = Rgb::new(0.5, 0.5, 0.5);
    let linear = srgb.from_srgb();
    assert!((linear.r - 0.2140).abs() < 0.001, "got {}", linear.r);
}

#[test]
fn srgb_roundtrip_boundaries() {
    // Black and white are identical in both spaces
    let black = Rgb::BLACK;
    let white = Rgb::WHITE;
    assert!((black.to_srgb().r - 0.0).abs() < 1e-6);
    assert!((white.to_srgb().r - 1.0).abs() < 1e-6);
    assert!((black.from_srgb().r - 0.0).abs() < 1e-6);
    assert!((white.from_srgb().r - 1.0).abs() < 1e-6);
}

#[test]
fn srgb_linear_roundtrip_approx() {
    // For any value in [0,1], linear → sRGB → linear should round-trip within u8 precision
    let samples = [0.0f32, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0];
    for &v in &samples {
        let original = Rgb::new(v, v, v);
        let encoded  = original.to_srgb();
        let decoded  = encoded.from_srgb();
        assert!(
            (decoded.r - original.r).abs() < 1.0 / 255.0 + 1e-5,
            "roundtrip error at v={}: got {}", v, decoded.r
        );
    }
}

// ── Rgb::from_hex ─────────────────────────────────────────────────────────────

#[test]
fn hex_black() {
    let c = Rgb::from_hex("#000000").unwrap();
    assert!((c.r).abs() < 1e-6);
    assert!((c.g).abs() < 1e-6);
    assert!((c.b).abs() < 1e-6);
}

#[test]
fn hex_white() {
    let c = Rgb::from_hex("#FFFFFF").unwrap();
    assert!((c.r - 1.0).abs() < 1e-6);
    assert!((c.g - 1.0).abs() < 1e-6);
    assert!((c.b - 1.0).abs() < 1e-6);
}

#[test]
fn hex_red_linear() {
    // #FF0000 in sRGB → red in linear is still (1.0, 0.0, 0.0)
    let c = Rgb::from_hex("#FF0000").unwrap();
    assert!((c.r - 1.0).abs() < 1e-4);
    assert!(c.g.abs() < 1e-6);
    assert!(c.b.abs() < 1e-6);
}

#[test]
fn hex_case_insensitive_digits() {
    // 0-9 and A-F are both valid hex
    let upper = Rgb::from_hex("#AABBCC").unwrap();
    let lower = Rgb::from_hex("#aabbcc").unwrap();
    assert!((upper.r - lower.r).abs() < 1e-6);
    assert!((upper.g - lower.g).abs() < 1e-6);
    assert!((upper.b - lower.b).abs() < 1e-6);
}

#[test]
fn hex_invalid_returns_none() {
    assert!(Rgb::from_hex("FF0000").is_none());   // missing #
    assert!(Rgb::from_hex("#FF00").is_none());    // too short
    assert!(Rgb::from_hex("#GGGGGG").is_none());  // invalid hex digit
    assert!(Rgb::from_hex("").is_none());
}

// ── Rgba::from_hex ────────────────────────────────────────────────────────────

#[test]
fn rgba_hex_6_char_has_full_alpha() {
    let c = Rgba::from_hex("#FF0000").unwrap();
    assert!((c.a - 1.0).abs() < 1e-6);
}

#[test]
fn rgba_hex_8_char_parses_alpha() {
    // #FFFFFF80 = white at 50% alpha
    let c = Rgba::from_hex("#FFFFFF80").unwrap();
    assert!((c.r - 1.0).abs() < 1e-4);
    let expected_a = 0x80 as f32 / 255.0;
    assert!((c.a - expected_a).abs() < 1e-4, "got a={}", c.a);
}

#[test]
fn rgba_hex_alpha_is_linear_not_gamma() {
    // #80 in hex is 128/255 ≈ 0.502 — alpha should NOT be gamma-decoded
    let c = Rgba::from_hex("#FFFFFF80").unwrap();
    let expected = 0x80_u8 as f32 / 255.0;
    assert!((c.a - expected).abs() < 0.005, "alpha was gamma-decoded: got {}", c.a);
}

// ── Rgb operations ────────────────────────────────────────────────────────────

#[test]
fn rgb_lerp_midpoint() {
    let a = Rgb::BLACK;
    let b = Rgb::WHITE;
    let mid = a.lerp(b, 0.5);
    assert!((mid.r - 0.5).abs() < 1e-6);
    assert!((mid.g - 0.5).abs() < 1e-6);
    assert!((mid.b - 0.5).abs() < 1e-6);
}

#[test]
fn rgb_lerp_at_zero_returns_self() {
    let a = Rgb::RED;
    let b = Rgb::BLUE;
    let r = a.lerp(b, 0.0);
    assert!((r.r - 1.0).abs() < 1e-6);
    assert!(r.g.abs() < 1e-6);
    assert!(r.b.abs() < 1e-6);
}

#[test]
fn rgb_lerp_at_one_returns_rhs() {
    let a = Rgb::RED;
    let b = Rgb::BLUE;
    let r = a.lerp(b, 1.0);
    assert!(r.r.abs() < 1e-6);
    assert!(r.b.abs() - 1.0 < 1e-6);
}

#[test]
fn rgb_luminance_white() {
    assert!((Rgb::WHITE.luminance() - 1.0).abs() < 1e-6);
}

#[test]
fn rgb_luminance_black() {
    assert!(Rgb::BLACK.luminance().abs() < 1e-6);
}

#[test]
fn rgb_luminance_rec709_coefficients() {
    // Pure red: 0.2126, pure green: 0.7152, pure blue: 0.0722
    assert!((Rgb::RED.luminance()   - 0.2126).abs() < 1e-4);
    assert!((Rgb::GREEN.luminance() - 0.7152).abs() < 1e-4);
    assert!((Rgb::BLUE.luminance()  - 0.0722).abs() < 1e-4);
}

// ── Tone mapping ──────────────────────────────────────────────────────────────

#[test]
fn reinhard_maps_to_unit_interval() {
    let hdr = Rgb::new(2.0, 5.0, 10.0);
    let ldr = hdr.tone_map_reinhard();
    assert!(ldr.r >= 0.0 && ldr.r < 1.0);
    assert!(ldr.g >= 0.0 && ldr.g < 1.0);
    assert!(ldr.b >= 0.0 && ldr.b < 1.0);
}

#[test]
fn reinhard_dark_values_barely_change() {
    // For very small values, Reinhard ≈ identity
    let dim = Rgb::new(0.01, 0.01, 0.01);
    let tm  = dim.tone_map_reinhard();
    assert!((tm.r - dim.r).abs() < 0.001);
}

#[test]
fn aces_maps_to_unit_interval() {
    let hdr = Rgb::new(1.0, 2.0, 4.0);
    let ldr = hdr.tone_map_aces();
    assert!(ldr.r >= 0.0 && ldr.r <= 1.0, "r={}", ldr.r);
    assert!(ldr.g >= 0.0 && ldr.g <= 1.0, "g={}", ldr.g);
    assert!(ldr.b >= 0.0 && ldr.b <= 1.0, "b={}", ldr.b);
}

// ── Rgba alpha operations ─────────────────────────────────────────────────────

#[test]
fn premultiply_opaque_is_identity() {
    let c = Rgba::new(0.8, 0.4, 0.2, 1.0);
    let pm = c.premultiply_alpha();
    assert!((pm.r - 0.8).abs() < 1e-6);
    assert!((pm.g - 0.4).abs() < 1e-6);
    assert!((pm.b - 0.2).abs() < 1e-6);
}

#[test]
fn premultiply_half_alpha() {
    let c = Rgba::new(1.0, 1.0, 1.0, 0.5);
    let pm = c.premultiply_alpha();
    assert!((pm.r - 0.5).abs() < 1e-6);
    assert!((pm.a - 0.5).abs() < 1e-6);
}

#[test]
fn premultiply_unpremultiply_roundtrip() {
    let c = Rgba::new(0.6, 0.3, 0.9, 0.8);
    let pm   = c.premultiply_alpha();
    let back = pm.unpremultiply_alpha();
    assert!((back.r - c.r).abs() < 1e-4, "r: {} vs {}", back.r, c.r);
    assert!((back.g - c.g).abs() < 1e-4, "g: {} vs {}", back.g, c.g);
    assert!((back.b - c.b).abs() < 1e-4, "b: {} vs {}", back.b, c.b);
}

#[test]
fn unpremultiply_transparent_returns_transparent() {
    let t = Rgba::TRANSPARENT;
    let r = t.unpremultiply_alpha();
    assert_eq!(r, Rgba::TRANSPARENT);
}

// ── Color32 ───────────────────────────────────────────────────────────────────

#[test]
fn color32_from_hex_black() {
    let c = Color32::from_hex("#000000").unwrap();
    assert_eq!(c, Color32::new_opaque(0, 0, 0));
    assert_eq!(c.a, 255);
}

#[test]
fn color32_from_hex_white() {
    let c = Color32::from_hex("#FFFFFF").unwrap();
    assert_eq!(c, Color32::WHITE);
}

#[test]
fn color32_from_hex_with_alpha() {
    let c = Color32::from_hex("#FF0000FF").unwrap();
    assert_eq!(c.r, 255);
    assert_eq!(c.g, 0);
    assert_eq!(c.b, 0);
    assert_eq!(c.a, 255);
}

#[test]
fn color32_from_hex_half_alpha() {
    let c = Color32::from_hex("#FFFFFF80").unwrap();
    assert_eq!(c.r, 255);
    assert_eq!(c.a, 0x80);
}

#[test]
fn color32_invalid_hex() {
    assert!(Color32::from_hex("FFFFFF").is_none());
    assert!(Color32::from_hex("#FFFFF").is_none());
    assert!(Color32::from_hex("#ZZZZZZ").is_none());
}

#[test]
fn color32_to_u32_rgba_and_back() {
    let c = Color32::new(0xDE, 0xAD, 0xBE, 0xEF);
    let u = c.to_u32_rgba();
    let back = Color32::from_u32_rgba(u);
    assert_eq!(c, back);
}

// ── Rgba ↔ Color32 roundtrip ──────────────────────────────────────────────────

#[test]
fn rgba_to_color32_white() {
    let c = Rgba::WHITE.to_color32();
    assert_eq!(c.r, 255);
    assert_eq!(c.g, 255);
    assert_eq!(c.b, 255);
    assert_eq!(c.a, 255);
}

#[test]
fn rgba_to_color32_black() {
    let c = Rgba::BLACK.to_color32();
    assert_eq!(c, Color32::BLACK);
}

#[test]
fn color32_to_rgba_to_color32_roundtrip() {
    // For any sRGB u8 color, the roundtrip should reproduce the original within ±1 LSB.
    let original = Color32::new(100, 150, 200, 128);
    let linear   = Rgba::from_color32(original);
    let back     = linear.to_color32();
    // Allow ±1 for quantisation
    assert!((back.r as i32 - original.r as i32).abs() <= 1, "r: {} vs {}", back.r, original.r);
    assert!((back.g as i32 - original.g as i32).abs() <= 1, "g: {} vs {}", back.g, original.g);
    assert!((back.b as i32 - original.b as i32).abs() <= 1, "b: {} vs {}", back.b, original.b);
    assert!((back.a as i32 - original.a as i32).abs() <= 1, "a: {} vs {}", back.a, original.a);
}

// ── Color32 blend_over ────────────────────────────────────────────────────────

#[test]
fn blend_over_opaque_src_returns_src() {
    let dst = Color32::BLACK;
    let src = Color32::WHITE;
    assert_eq!(dst.blend_over(src), src);
}

#[test]
fn blend_over_transparent_src_returns_dst() {
    let dst = Color32::RED;
    let src = Color32::TRANSPARENT;
    assert_eq!(dst.blend_over(src), dst);
}

#[test]
fn blend_over_half_alpha() {
    let dst = Color32::new_opaque(0, 0, 0);
    let src = Color32::new(255, 255, 255, 128); // ~50% white
    let r = dst.blend_over(src);
    // Result should be approximately 50% grey
    assert!(r.r > 100 && r.r < 140, "expected ~128, got {}", r.r);
}

// ── Arithmetic operators ──────────────────────────────────────────────────────

#[test]
fn rgb_add() {
    let a = Rgb::new(0.2, 0.3, 0.4);
    let b = Rgb::new(0.1, 0.2, 0.3);
    let c = a + b;
    assert!((c.r - 0.3).abs() < 1e-6);
    assert!((c.g - 0.5).abs() < 1e-6);
    assert!((c.b - 0.7).abs() < 1e-6);
}

#[test]
fn rgb_scale() {
    let a = Rgb::WHITE;
    let b = a * 0.5;
    assert!((b.r - 0.5).abs() < 1e-6);
    assert!((b.g - 0.5).abs() < 1e-6);
    assert!((b.b - 0.5).abs() < 1e-6);
}

#[test]
fn rgb_modulate() {
    // Modulate (component-wise mul) — white modulated by red = red
    let white = Rgb::WHITE;
    let red   = Rgb::RED;
    let r = white * red;
    assert!((r.r - 1.0).abs() < 1e-6);
    assert!(r.g.abs() < 1e-6);
    assert!(r.b.abs() < 1e-6);
}
