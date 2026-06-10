// crates/mid-math/src/tests/fixed.rs
//! Fixed-point arithmetic tests.
//!
//! Tests cover: arithmetic correctness, floor/ceil, lerp,
//! overflow protection, f32 boundary, vec operations, cross product.

use crate::{Fixed16, Fixed8, Fixed12, Fixed16Vec2, Fixed16Vec3, Fixed};

// ── Arithmetic correctness ────────────────────────────────────────────────────

#[test]
fn integer_add_and_sub() {
    let a = Fixed16::from_i32(7);
    let b = Fixed16::from_i32(3);
    assert_eq!((a + b).to_i32_trunc(), 10);
    assert_eq!((a - b).to_i32_trunc(), 4);
}

#[test]
fn integer_mul_exact() {
    // 5 * 3 = 15
    let a = Fixed16::from_i32(5);
    let b = Fixed16::from_i32(3);
    let r = a * b;
    assert_eq!(r.to_i32_trunc(), 15);
}

#[test]
fn fractional_mul() {
    // 2.5 * 2.0 = 5.0
    let a = Fixed16::from_f32(2.5);
    let b = Fixed16::from_f32(2.0);
    let r = a * b;
    assert!((r.to_f32() - 5.0).abs() < 1e-4, "got {}", r.to_f32());
}

#[test]
fn fractional_mul_small() {
    // 0.5 * 0.5 = 0.25
    let a = Fixed16::from_f32(0.5);
    let b = Fixed16::from_f32(0.5);
    let r = a * b;
    assert!((r.to_f32() - 0.25).abs() < 1e-4, "got {}", r.to_f32());
}

#[test]
fn division_exact() {
    // 10 / 2 = 5
    let a = Fixed16::from_i32(10);
    let b = Fixed16::from_i32(2);
    let r = a / b;
    assert_eq!(r.to_i32_trunc(), 5);
}

#[test]
fn division_fractional() {
    // 1 / 4 = 0.25
    let a = Fixed16::from_i32(1);
    let b = Fixed16::from_i32(4);
    let r = a / b;
    assert!((r.to_f32() - 0.25).abs() < 1e-4, "got {}", r.to_f32());
}

#[test]
fn neg_and_abs() {
    let a = Fixed16::from_f32(3.75);
    let n = -a;
    assert!((n.to_f32() + 3.75).abs() < 1e-4);
    assert!((n.abs().to_f32() - 3.75).abs() < 1e-4);
}

// ── Ordering ──────────────────────────────────────────────────────────────────

#[test]
fn ordering() {
    let a = Fixed16::from_f32(1.0);
    let b = Fixed16::from_f32(2.0);
    let c = Fixed16::from_f32(-0.5);
    assert!(a < b);
    assert!(b > a);
    assert!(c < a);
    assert!(c < b);
}

#[test]
fn negative_ordering() {
    let a = Fixed16::from_f32(-1.0);
    let b = Fixed16::from_f32(-2.0);
    assert!(b < a);
    assert!(a > b);
}

// ── From/to i32 roundtrip ─────────────────────────────────────────────────────

#[test]
fn from_to_i32_roundtrip() {
    for n in [-1000i32, -1, 0, 1, 42, 1000] {
        let f = Fixed16::from_i32(n);
        assert_eq!(f.to_i32_trunc(), n, "failed for n={}", n);
    }
}

// ── From/to f32 boundary ──────────────────────────────────────────────────────

#[test]
fn from_f32_precision() {
    // Fixed16 has 1/65536 resolution ≈ 0.0000153
    let values = [0.0f32, 0.5, 1.0, -1.0, 3.14159, -2.71828, 100.0];
    for &v in &values {
        let f = Fixed16::from_f32(v);
        let back = f.to_f32();
        assert!((back - v).abs() < 2.0 / 65536.0,
            "from_f32({}) roundtrip error: {}", v, (back - v).abs());
    }
}

// ── Floor and ceil ────────────────────────────────────────────────────────────

#[test]
fn floor_positive() {
    let f = Fixed16::from_f32(1.75);
    assert_eq!(f.floor().to_i32_trunc(), 1);
}

#[test]
fn floor_negative() {
    let f = Fixed16::from_f32(-1.75);
    assert_eq!(f.floor().to_i32_trunc(), -2);
}

#[test]
fn floor_exact_integer() {
    let f = Fixed16::from_f32(3.0);
    assert_eq!(f.floor().to_i32_trunc(), 3);
}

#[test]
fn ceil_positive() {
    let f = Fixed16::from_f32(1.25);
    assert_eq!(f.ceil().to_i32_trunc(), 2);
}

#[test]
fn ceil_negative() {
    // ceil(-1.75) = -1
    let f = Fixed16::from_f32(-1.75);
    assert_eq!(f.ceil().to_i32_trunc(), -1);
}

#[test]
fn ceil_exact_integer() {
    let f = Fixed16::from_f32(3.0);
    assert_eq!(f.ceil().to_i32_trunc(), 3);
}

// ── Lerp ─────────────────────────────────────────────────────────────────────

#[test]
fn lerp_midpoint() {
    let a = Fixed16::from_f32(0.0);
    let b = Fixed16::from_f32(10.0);
    let t = Fixed16::from_f32(0.5);
    let r = a.lerp(b, t);
    assert!((r.to_f32() - 5.0).abs() < 0.01, "got {}", r.to_f32());
}

#[test]
fn lerp_at_zero() {
    let a = Fixed16::from_f32(3.0);
    let b = Fixed16::from_f32(7.0);
    let r = a.lerp(b, Fixed16::ZERO);
    assert!((r.to_f32() - 3.0).abs() < 0.001);
}

#[test]
fn lerp_at_one() {
    let a = Fixed16::from_f32(3.0);
    let b = Fixed16::from_f32(7.0);
    let r = a.lerp(b, Fixed16::one());
    assert!((r.to_f32() - 7.0).abs() < 0.001);
}

// ── Saturation ────────────────────────────────────────────────────────────────

#[test]
fn saturating_add_no_overflow() {
    let a = Fixed16::from_i32(100);
    let b = Fixed16::from_i32(50);
    assert_eq!((a.saturating_add(b)).to_i32_trunc(), 150);
}

#[test]
fn saturating_add_clamps_at_max() {
    let a = Fixed16::MAX;
    let b = Fixed16::from_i32(1);
    assert_eq!(a.saturating_add(b), Fixed16::MAX);
}

#[test]
fn saturating_sub_clamps_at_min() {
    let a = Fixed16::MIN;
    let b = Fixed16::from_i32(1);
    assert_eq!(a.saturating_sub(b), Fixed16::MIN);
}

#[test]
fn checked_mul_overflow_returns_none() {
    let a = Fixed16::MAX;
    let b = Fixed16::from_i32(2);
    assert!(a.checked_mul(b).is_none());
}

#[test]
fn checked_mul_normal_returns_some() {
    let a = Fixed16::from_i32(3);
    let b = Fixed16::from_i32(4);
    let r = a.checked_mul(b).unwrap();
    assert_eq!(r.to_i32_trunc(), 12);
}

// ── Different FRAC values ─────────────────────────────────────────────────────

#[test]
fn fixed8_resolution() {
    // Fixed8 has 1/256 = ~0.004 resolution
    let a = Fixed8::from_f32(0.5);
    assert!((a.to_f32() - 0.5).abs() < 1.0 / 256.0 + 1e-6);
}

#[test]
fn fixed12_mul_precision() {
    let a = Fixed12::from_f32(1.5);
    let b = Fixed12::from_f32(1.5);
    let r = a * b;
    assert!((r.to_f32() - 2.25).abs() < 0.001, "got {}", r.to_f32());
}

// ── Determinism ───────────────────────────────────────────────────────────────

#[test]
fn same_raw_same_value() {
    // Fixed-point is just integers — deterministic by construction.
    let a = Fixed16::from_raw(123456);
    let b = Fixed16::from_raw(123456);
    assert_eq!(a, b);
    assert_eq!(a.to_raw(), b.to_raw());
}

#[test]
fn add_commutativity() {
    let a = Fixed16::from_f32(1.5);
    let b = Fixed16::from_f32(2.5);
    assert_eq!(a + b, b + a);
}

// ── FixedVec2 ─────────────────────────────────────────────────────────────────

#[test]
fn vec2_add_and_sub() {
    let a = Fixed16Vec2::from_i32(3, 4);
    let b = Fixed16Vec2::from_i32(1, 2);
    let sum = a + b;
    assert_eq!(sum.x.to_i32_trunc(), 4);
    assert_eq!(sum.y.to_i32_trunc(), 6);
    let diff = a - b;
    assert_eq!(diff.x.to_i32_trunc(), 2);
    assert_eq!(diff.y.to_i32_trunc(), 2);
}

#[test]
fn vec2_dot_product() {
    // (3, 4) · (3, 4) = 9 + 16 = 25
    let a = Fixed16Vec2::from_i32(3, 4);
    let dot = a.dot(a);
    assert_eq!(dot.to_i32_trunc(), 25);
}

#[test]
fn vec2_dot_orthogonal() {
    let x = Fixed16Vec2::from_i32(1, 0);
    let y = Fixed16Vec2::from_i32(0, 1);
    assert_eq!(x.dot(y), Fixed16::ZERO);
}

#[test]
fn vec2_length_sq() {
    // (3, 4) length_sq = 25
    let a = Fixed16Vec2::from_i32(3, 4);
    assert_eq!(a.length_sq().to_i32_trunc(), 25);
}

#[test]
fn vec2_scale() {
    let a = Fixed16Vec2::from_i32(2, 3);
    let s = Fixed16::from_i32(5);
    let r = a.scale(s);
    assert_eq!(r.x.to_i32_trunc(), 10);
    assert_eq!(r.y.to_i32_trunc(), 15);
}

#[test]
fn vec2_neg() {
    let a = Fixed16Vec2::from_i32(3, -4);
    let n = -a;
    assert_eq!(n.x.to_i32_trunc(), -3);
    assert_eq!(n.y.to_i32_trunc(), 4);
}

#[test]
fn vec2_perp_dot() {
    // (1, 0) perp_dot (0, 1) = 1 (CCW = positive)
    let x = Fixed16Vec2::from_i32(1, 0);
    let y = Fixed16Vec2::from_i32(0, 1);
    assert_eq!(x.perp_dot(y).to_i32_trunc(), 1);
    assert_eq!(y.perp_dot(x).to_i32_trunc(), -1);
}

#[test]
fn vec2_lerp() {
    let a = Fixed16Vec2::from_i32(0, 0);
    let b = Fixed16Vec2::from_i32(10, 20);
    let t = Fixed16::from_f32(0.5);
    let r = a.lerp(b, t);
    assert!((r.x.to_f32() - 5.0).abs() < 0.01);
    assert!((r.y.to_f32() - 10.0).abs() < 0.01);
}

#[test]
fn vec2_to_from_vec2_boundary() {
    let fv = Fixed16Vec2::from_f32(1.5, -2.5);
    let v2 = fv.to_vec2();
    assert!((v2.x - 1.5).abs() < 0.001);
    assert!((v2.y - (-2.5)).abs() < 0.001);
    let back = Fixed16Vec2::from_vec2(v2);
    assert!((back.x.to_f32() - 1.5).abs() < 0.001);
}

// ── FixedVec3 ─────────────────────────────────────────────────────────────────

#[test]
fn vec3_dot_self() {
    // (1, 2, 2) · (1, 2, 2) = 1 + 4 + 4 = 9
    let a = Fixed16Vec3::from_i32(1, 2, 2);
    assert_eq!(a.dot(a).to_i32_trunc(), 9);
}

#[test]
fn vec3_cross_basic() {
    // X cross Y = Z
    let x = Fixed16Vec3::from_i32(1, 0, 0);
    let y = Fixed16Vec3::from_i32(0, 1, 0);
    let z = x.cross(y);
    assert_eq!(z.x.to_i32_trunc(), 0);
    assert_eq!(z.y.to_i32_trunc(), 0);
    assert_eq!(z.z.to_i32_trunc(), 1);
}

#[test]
fn vec3_cross_anticommutativity() {
    // a cross b = -(b cross a)
    let a = Fixed16Vec3::from_i32(1, 2, 3);
    let b = Fixed16Vec3::from_i32(4, 5, 6);
    let ab = a.cross(b);
    let ba = b.cross(a);
    assert_eq!(ab.x, -ba.x);
    assert_eq!(ab.y, -ba.y);
    assert_eq!(ab.z, -ba.z);
}

#[test]
fn vec3_cross_parallel_is_zero() {
    // Parallel vectors have zero cross product
    let a = Fixed16Vec3::from_i32(2, 0, 0);
    let b = Fixed16Vec3::from_i32(5, 0, 0);
    let r = a.cross(b);
    assert_eq!(r.x, Fixed16::ZERO);
    assert_eq!(r.y, Fixed16::ZERO);
    assert_eq!(r.z, Fixed16::ZERO);
}

#[test]
fn vec3_to_from_vec3_boundary() {
    let fv = Fixed16Vec3::from_f32(1.0, 2.5, -3.0);
    let v3 = fv.to_vec3();
    assert!((v3.x - 1.0).abs() < 0.001);
    assert!((v3.y - 2.5).abs() < 0.001);
    assert!((v3.z - (-3.0)).abs() < 0.001);
}
