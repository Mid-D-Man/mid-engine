// crates/mid-math/src/tests/wide_tests.rs
//! Correctness tests for all wide SIMD types — integer and float.
//!
//! Tests are written against the public API only — the scalar and SSE2
//! implementations share the same interface so both paths are covered
//! on their respective platforms by the same test binary.

#[cfg(test)]
mod tests {
    // ── Integer wide ──────────────────────────────────────────────────────────
    use crate::{
        IMask4, IMask8, IMask16,
        i32x4, u32x4,
        i16x8, u16x8,
        i8x16, u8x16,
    };

    // ── Float wide ────────────────────────────────────────────────────────────
    use crate::{
        Mask4, f32x4, Vec3x4, QuatX4,
        Vec3, Quat, Mat4,
        to_radians, approx_eq, EPSILON,
    };

    // ─────────────────────────────────────────────────────────────────────────
    // Helpers
    // ─────────────────────────────────────────────────────────────────────────

    fn approx4(a: [f32; 4], b: [f32; 4], eps: f32) -> bool {
        a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < eps)
    }

    fn vec3_approx(a: Vec3, b: Vec3) -> bool {
        (a.x - b.x).abs() < 1e-5
            && (a.y - b.y).abs() < 1e-5
            && (a.z - b.z).abs() < 1e-5
    }

    // ═════════════════════════════════════════════════════════════════════════
    //  IMask4
    // ═════════════════════════════════════════════════════════════════════════

    #[test]
    fn imask4_false_all_zero() {
        let m = IMask4::FALSE;
        assert!(!m.any());
        assert!(!m.all());
        assert!(m.none());
        assert_eq!(m.bitmask(), 0b0000);
    }

    #[test]
    fn imask4_true_all_set() {
        let m = IMask4::TRUE;
        assert!(m.any());
        assert!(m.all());
        assert!(!m.none());
        assert_eq!(m.bitmask(), 0b1111);
    }

    #[test]
    fn imask4_bitops() {
        let a = i32x4::new(1, 0, 1, 0).cmpeq(i32x4::new(1, 0, 0, 1));
        // lanes 0,1 equal; lanes 2,3 not → 0b0011
        assert_eq!(a.bitmask(), 0b0011);
        let b = !a;
        assert_eq!(b.bitmask(), 0b1100);
        let c = a | b;
        assert!(c.all());
        let d = a & b;
        assert!(d.none());
    }

    // ═════════════════════════════════════════════════════════════════════════
    //  i32x4
    // ═════════════════════════════════════════════════════════════════════════

    #[test]
    fn i32x4_size_align() {
        assert_eq!(core::mem::size_of::<i32x4>(), 16);
        assert_eq!(core::mem::align_of::<i32x4>(), 16);
    }

    #[test]
    fn i32x4_splat_and_get() {
        let v = i32x4::splat(42);
        assert_eq!(v.to_array(), [42; 4]);
        assert_eq!(v.get(2), 42);
    }

    #[test]
    fn i32x4_new_roundtrip() {
        let v = i32x4::new(1, -2, 3, -4);
        assert_eq!(v.to_array(), [1, -2, 3, -4]);
    }

    #[test]
    fn i32x4_from_array_roundtrip() {
        let arr = [10, 20, -30, 40];
        assert_eq!(i32x4::from_array(arr).to_array(), arr);
    }

    #[test]
    fn i32x4_add() {
        let a = i32x4::new(1, 2, 3, 4);
        let b = i32x4::new(10, 20, 30, 40);
        assert_eq!((a + b).to_array(), [11, 22, 33, 44]);
    }

    #[test]
    fn i32x4_sub() {
        let a = i32x4::new(10, 20, 30, 40);
        let b = i32x4::new(1, 2, 3, 4);
        assert_eq!((a - b).to_array(), [9, 18, 27, 36]);
    }

    #[test]
    fn i32x4_mul() {
        let a = i32x4::new(2, 3, 4, 5);
        let b = i32x4::new(3, 4, 5, 6);
        assert_eq!((a * b).to_array(), [6, 12, 20, 30]);
    }

    #[test]
    fn i32x4_neg() {
        let v = i32x4::new(1, -2, 3, -4);
        assert_eq!((-v).to_array(), [-1, 2, -3, 4]);
    }

    #[test]
    fn i32x4_abs() {
        let v = i32x4::new(-5, 3, -7, 0);
        assert_eq!(v.abs().to_array(), [5, 3, 7, 0]);
    }

    #[test]
    fn i32x4_min_max() {
        let a = i32x4::new(1, 5, 3, 9);
        let b = i32x4::new(4, 2, 3, 7);
        assert_eq!(a.min(b).to_array(), [1, 2, 3, 7]);
        assert_eq!(a.max(b).to_array(), [4, 5, 3, 9]);
    }

    #[test]
    fn i32x4_clamp() {
        let v  = i32x4::new(-5, 3, 15, 7);
        let lo = i32x4::splat(0);
        let hi = i32x4::splat(10);
        assert_eq!(v.clamp(lo, hi).to_array(), [0, 3, 10, 7]);
    }

    #[test]
    fn i32x4_element_sum() {
        assert_eq!(i32x4::new(1, 2, 3, 4).element_sum(), 10);
    }

    #[test]
    fn i32x4_min_max_element() {
        let v = i32x4::new(-3, 7, 2, -1);
        assert_eq!(v.min_element(), -3);
        assert_eq!(v.max_element(), 7);
    }

    #[test]
    fn i32x4_shifts() {
        let v = i32x4::new(1, 2, 4, 8);
        assert_eq!(v.shl(2).to_array(), [4, 8, 16, 32]);
        assert_eq!(v.shl(2).shr_arithmetic(2).to_array(), [1, 2, 4, 8]);
        // Logical shift: sign bits become 0
        let neg = i32x4::new(-1, -2, -4, -8);
        let logical = neg.shr_logical(1);
        for lane in logical.to_array() {
            assert!(lane > 0, "logical shr must zero-fill MSB");
        }
    }

    #[test]
    fn i32x4_cmp_and_blend() {
        let a = i32x4::new(1, 5, 3, 9);
        let b = i32x4::new(4, 2, 3, 7);
        let eq = a.cmpeq(b);
        // Only lane 2 equal
        assert_eq!(eq.bitmask(), 0b0100);
        let gt = a.cmpgt(b);
        // lanes 1,3
        assert_eq!(gt.bitmask(), 0b1010);
        let blended = i32x4::blend(gt, a, b);
        assert_eq!(blended.to_array(), [4, 5, 3, 9]);
    }

    #[test]
    fn i32x4_wrapping() {
        let max = i32x4::splat(i32::MAX);
        let one = i32x4::splat(1);
        assert_eq!(max.wrapping_add(one).to_array(), [i32::MIN; 4]);
        let min = i32x4::splat(i32::MIN);
        assert_eq!(min.wrapping_sub(one).to_array(), [i32::MAX; 4]);
    }

    #[test]
    fn i32x4_saturating() {
        let max = i32x4::splat(i32::MAX);
        let one = i32x4::splat(1);
        assert_eq!(max.saturating_add(one).to_array(), [i32::MAX; 4]);
        let min = i32x4::splat(i32::MIN);
        assert_eq!(min.saturating_sub(one).to_array(), [i32::MIN; 4]);
    }

    #[test]
    fn i32x4_bitwise() {
        let a = i32x4::splat(0b1100);
        let b = i32x4::splat(0b1010);
        assert_eq!((a & b).to_array(), [0b1000; 4]);
        assert_eq!((a | b).to_array(), [0b1110; 4]);
        assert_eq!((a ^ b).to_array(), [0b0110; 4]);
        assert_eq!((!i32x4::splat(0)).to_array(), [-1; 4]);
    }

    #[test]
    #[should_panic]
    fn i32x4_get_oob_panics() { let _ = i32x4::splat(0).get(4); }

    // ═════════════════════════════════════════════════════════════════════════
    //  u32x4
    // ═════════════════════════════════════════════════════════════════════════

    #[test]
    fn u32x4_unsigned_comparison() {
        // 0xFFFFFFFF > 1 unsigned — would be < 0 signed
        let a = u32x4::new(u32::MAX, 2, 0, 4);
        let b = u32x4::new(1,        5, 0, 3);
        let gt = a.cmpgt(b);
        // lane 0: MAX > 1 ✓, lane 1: 2 > 5 ✗, lane 2: equal, lane 3: 4 > 3 ✓
        assert_eq!(gt.bitmask(), 0b1001);
    }

    #[test]
    fn u32x4_saturating_add_no_overflow() {
        let max = u32x4::splat(u32::MAX);
        let one = u32x4::splat(1);
        assert_eq!(max.saturating_add(one).to_array(), [u32::MAX; 4]);
    }

    #[test]
    fn u32x4_saturating_sub_no_underflow() {
        let zero = u32x4::splat(0);
        let one  = u32x4::splat(1);
        assert_eq!(zero.saturating_sub(one).to_array(), [0u32; 4]);
    }

    #[test]
    fn u32x4_min_max() {
        // Ensure unsigned semantics: large values don't become negative
        let a = u32x4::new(u32::MAX, 10, 0, 100);
        let b = u32x4::new(0, 20, u32::MAX, 50);
        assert_eq!(a.min(b).to_array(), [0, 10, 0, 50]);
        assert_eq!(a.max(b).to_array(), [u32::MAX, 20, u32::MAX, 100]);
    }

    // ═════════════════════════════════════════════════════════════════════════
    //  i16x8
    // ═════════════════════════════════════════════════════════════════════════

    #[test]
    fn i16x8_size_align() {
        assert_eq!(core::mem::size_of::<i16x8>(), 16);
        assert_eq!(core::mem::align_of::<i16x8>(), 16);
    }

    #[test]
    fn i16x8_splat_and_to_array() {
        let v = i16x8::splat(7);
        assert_eq!(v.to_array(), [7i16; 8]);
    }

    #[test]
    fn i16x8_add_sub() {
        let a = i16x8::from_array([1, 2, 3, 4, 5, 6, 7, 8]);
        let b = i16x8::from_array([8, 7, 6, 5, 4, 3, 2, 1]);
        let sum = (a + b).to_array();
        assert!(sum.iter().all(|&x| x == 9));
        assert_eq!((a - b).to_array(), [-7, -5, -3, -1, 1, 3, 5, 7]);
    }

    #[test]
    fn i16x8_saturating_add() {
        let max = i16x8::splat(i16::MAX);
        let one = i16x8::splat(1);
        assert_eq!(max.saturating_add(one).to_array(), [i16::MAX; 8]);
    }

    #[test]
    fn i16x8_saturating_sub() {
        let min = i16x8::splat(i16::MIN);
        let one = i16x8::splat(1);
        assert_eq!(min.saturating_sub(one).to_array(), [i16::MIN; 8]);
    }

    #[test]
    fn i16x8_mul_lo() {
        // 100 * 200 = 20000 — fits in i16
        let a = i16x8::splat(100);
        let b = i16x8::splat(200);
        assert_eq!(a.mul_lo(b).to_array(), [20000i16; 8]);
    }

    #[test]
    fn i16x8_widen_roundtrip() {
        let v = i16x8::from_array([1, -2, 3, -4, 5, -6, 7, -8]);
        let lo = v.as_i32x4_lo().to_array();
        let hi = v.as_i32x4_hi().to_array();
        assert_eq!(lo, [1, -2, 3, -4]);
        assert_eq!(hi, [5, -6, 7, -8]);
        // Pack back
        let packed = i16x8::pack_i32x4(
            i32x4::from_array(lo),
            i32x4::from_array(hi),
        );
        assert_eq!(packed.to_array(), v.to_array());
    }

    #[test]
    fn i16x8_min_max() {
        let a = i16x8::from_array([1, 5, 3, -1, 0, 9, 2, 4]);
        let b = i16x8::from_array([4, 2, 3,  7, 0, 1, 8, 4]);
        assert_eq!(a.min(b).to_array(), [1, 2, 3, -1, 0, 1, 2, 4]);
        assert_eq!(a.max(b).to_array(), [4, 5, 3,  7, 0, 9, 8, 4]);
    }

    #[test]
    fn i16x8_element_sum() {
        let v = i16x8::from_array([1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(v.element_sum(), 36);
    }

    #[test]
    fn i16x8_cmpeq_bitmask() {
        let a = i16x8::from_array([1, 2, 3, 4, 5, 6, 7, 8]);
        let b = i16x8::from_array([1, 0, 3, 0, 5, 0, 7, 0]);
        // lanes 0,2,4,6 equal
        let m = a.cmpeq(b);
        assert_eq!(m.bitmask(), 0b01010101);
    }

    // ═════════════════════════════════════════════════════════════════════════
    //  u16x8
    // ═════════════════════════════════════════════════════════════════════════

    #[test]
    fn u16x8_saturating_add() {
        let max = u16x8::splat(u16::MAX);
        assert_eq!(max.saturating_add(u16x8::splat(1)).to_array(), [u16::MAX; 8]);
    }

    #[test]
    fn u16x8_saturating_sub() {
        let zero = u16x8::splat(0);
        assert_eq!(zero.saturating_sub(u16x8::splat(1)).to_array(), [0u16; 8]);
    }

    #[test]
    fn u16x8_unsigned_min_max() {
        // 60000 > 1000 unsigned — ensure unsigned comparison used
        let a = u16x8::splat(60000);
        let b = u16x8::splat(1000);
        assert_eq!(a.min(b).to_array(), [1000u16; 8]);
        assert_eq!(a.max(b).to_array(), [60000u16; 8]);
    }

    // ═════════════════════════════════════════════════════════════════════════
    //  i8x16
    // ═════════════════════════════════════════════════════════════════════════

    #[test]
    fn i8x16_size_align() {
        assert_eq!(core::mem::size_of::<i8x16>(), 16);
        assert_eq!(core::mem::align_of::<i8x16>(), 16);
    }

    #[test]
    fn i8x16_splat_and_to_array() {
        let v = i8x16::splat(3);
        assert_eq!(v.to_array(), [3i8; 16]);
    }

    #[test]
    fn i8x16_saturating() {
        let max = i8x16::splat(i8::MAX);
        let one = i8x16::splat(1);
        assert_eq!(max.saturating_add(one).to_array(), [i8::MAX; 16]);
        let min = i8x16::splat(i8::MIN);
        assert_eq!(min.saturating_sub(one).to_array(), [i8::MIN; 16]);
    }

    #[test]
    fn i8x16_cmpeq_and_count() {
        let a = i8x16::from_bytes([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]);
        let needle = i8x16::splat(7);
        let count = a.count_eq(needle);
        assert_eq!(count, 1);
        assert!(a.contains(7i8));
        assert!(!a.contains(99i8));
    }

    #[test]
    fn i8x16_abs() {
        let v = i8x16::from_array([-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12,-13,-14,-15,-16]);
        let a = v.abs().to_array();
        for (i, &x) in a.iter().enumerate() {
            assert_eq!(x, (i as i8) + 1);
        }
    }

    // ═════════════════════════════════════════════════════════════════════════
    //  u8x16
    // ═════════════════════════════════════════════════════════════════════════

    #[test]
    fn u8x16_element_sum_correct() {
        // 16 bytes all = 1 → sum = 16
        let v = u8x16::splat(1);
        assert_eq!(v.element_sum(), 16);
        // 16 bytes all = 10 → sum = 160
        let v2 = u8x16::splat(10);
        assert_eq!(v2.element_sum(), 160);
    }

    #[test]
    fn u8x16_saturating_add() {
        let max = u8x16::splat(u8::MAX);
        assert_eq!(max.saturating_add(u8x16::splat(1)).to_array(), [u8::MAX; 16]);
    }

    #[test]
    fn u8x16_saturating_sub() {
        let zero = u8x16::splat(0);
        assert_eq!(zero.saturating_sub(u8x16::splat(1)).to_array(), [0u8; 16]);
    }

    #[test]
    fn u8x16_min_max_unsigned() {
        let a = u8x16::splat(200);
        let b = u8x16::splat(50);
        assert_eq!(a.min(b).to_array(), [50u8; 16]);
        assert_eq!(a.max(b).to_array(), [200u8; 16]);
    }

    #[test]
    fn u8x16_contains_and_count() {
        let v = u8x16::from_array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,5]);
        // value 5 appears at indices 4 and 15
        assert_eq!(v.count_eq(u8x16::splat(5)), 2);
        assert!(v.contains(5));
        assert!(!v.contains(99));
    }

    // ═════════════════════════════════════════════════════════════════════════
    //  IMask8 / IMask16
    // ═════════════════════════════════════════════════════════════════════════

    #[test]
    fn imask8_from_i16x8_cmpeq() {
        let a = i16x8::from_array([1,2,3,4,1,2,3,4]);
        let b = i16x8::from_array([1,0,3,0,1,0,3,0]);
        let m = a.cmpeq(b); // lanes 0,2,4,6
        assert!(m.any());
        assert!(!m.all());
        assert_eq!(m.bitmask(), 0b01010101);
    }

    #[test]
    fn imask16_from_i8x16_cmpeq() {
        let a = i8x16::splat(7);
        let b = i8x16::splat(7);
        let m = a.cmpeq(b);
        assert!(m.all());
        assert_eq!(m.bitmask(), 0xFFFF);
        let m2 = i8x16::splat(1).cmpeq(i8x16::splat(2));
        assert!(m2.none());
        assert_eq!(m2.bitmask(), 0x0000);
    }

    // ═════════════════════════════════════════════════════════════════════════
    //  Mask4 (float)
    // ═════════════════════════════════════════════════════════════════════════

    #[test]
    fn mask4_false_true_constants() {
        assert!(Mask4::TRUE.all());
        assert!(!Mask4::FALSE.any());
        assert_eq!(Mask4::TRUE.bitmask(),  0b1111);
        assert_eq!(Mask4::FALSE.bitmask(), 0b0000);
    }

    #[test]
    fn mask4_from_f32x4_comparison() {
        let a = f32x4::new(1.0, 2.0, 3.0, 4.0);
        let b = f32x4::new(2.0, 2.0, 2.0, 2.0);
        let lt = a.cmplt(b); // lanes 0,1 (<= 2.0 → lane 0 only: 1<2)
        // 1<2 ✓, 2<2 ✗, 3<2 ✗, 4<2 ✗ → 0b0001
        assert_eq!(lt.bitmask(), 0b0001);
        let gt = a.cmpgt(b);
        // 1>2 ✗, 2>2 ✗, 3>2 ✓, 4>2 ✓ → 0b1100
        assert_eq!(gt.bitmask(), 0b1100);
        let eq = a.cmpeq(b);
        // only lane 1 (2==2) → 0b0010
        assert_eq!(eq.bitmask(), 0b0010);
    }

    #[test]
    fn mask4_bitops() {
        let a = f32x4::new(1.0, 0.0, 1.0, 0.0).cmpgt(f32x4::splat(0.5)); // 0b0101
        let b = f32x4::new(0.0, 1.0, 1.0, 0.0).cmpgt(f32x4::splat(0.5)); // 0b0110
        assert_eq!((a & b).bitmask(), 0b0100);
        assert_eq!((a | b).bitmask(), 0b0111);
        assert_eq!((a ^ b).bitmask(), 0b0011);
        assert_eq!((!a).bitmask(),    0b1010);
    }

    // ═════════════════════════════════════════════════════════════════════════
    //  f32x4
    // ═════════════════════════════════════════════════════════════════════════

    #[test]
    fn f32x4_size_align() {
        assert_eq!(core::mem::size_of::<f32x4>(), 16);
        assert_eq!(core::mem::align_of::<f32x4>(), 16);
    }

    #[test]
    fn f32x4_splat_and_get() {
        let v = f32x4::splat(3.14);
        assert!(approx4(v.to_array(), [3.14; 4], 1e-6));
        assert!((v.get(0) - 3.14).abs() < 1e-6);
    }

    #[test]
    fn f32x4_new_and_array_roundtrip() {
        let v = f32x4::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(v.to_array(), [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn f32x4_arithmetic() {
        let a = f32x4::new(1.0, 2.0, 3.0, 4.0);
        let b = f32x4::new(4.0, 3.0, 2.0, 1.0);
        assert!(approx4((a + b).to_array(), [5.0; 4], 1e-6));
        assert!(approx4((a - b).to_array(), [-3.0, -1.0, 1.0, 3.0], 1e-6));
        assert!(approx4((a * b).to_array(), [4.0, 6.0, 6.0, 4.0], 1e-6));
        let d = (a / b).to_array();
        assert!((d[0] - 0.25).abs() < 1e-5);
    }

    #[test]
    fn f32x4_sqrt() {
        let v = f32x4::new(1.0, 4.0, 9.0, 16.0);
        assert!(approx4(v.sqrt().to_array(), [1.0, 2.0, 3.0, 4.0], 1e-6));
    }

    #[test]
    fn f32x4_recip_sqrt_accuracy() {
        // rsqrt + Newton-Raphson should hit ~23-bit accuracy
        let v = f32x4::new(4.0, 9.0, 16.0, 25.0);
        let r = v.recip_sqrt().to_array();
        let expected = [0.5f32, 1.0/3.0, 0.25, 0.2];
        for (got, exp) in r.iter().zip(expected.iter()) {
            let err = (got - exp).abs() / exp.abs();
            assert!(err < 1e-6, "recip_sqrt error {} for {}", err, exp);
        }
    }

    #[test]
    fn f32x4_recip_accuracy() {
        let v = f32x4::new(2.0, 4.0, 8.0, 10.0);
        let r = v.recip().to_array();
        let expected = [0.5f32, 0.25, 0.125, 0.1];
        for (got, exp) in r.iter().zip(expected.iter()) {
            let err = (got - exp).abs() / exp.abs();
            assert!(err < 1e-6, "recip error {} for {}", err, exp);
        }
    }

    #[test]
    fn f32x4_min_max_abs() {
        let a = f32x4::new(-1.0, 2.0, -3.0, 4.0);
        let b = f32x4::new(1.0, -2.0, 3.0, -4.0);
        assert_eq!(a.min(b).to_array(), [-1.0, -2.0, -3.0, -4.0]);
        assert_eq!(a.max(b).to_array(), [1.0,  2.0,  3.0,  4.0]);
        assert_eq!(a.abs().to_array(), [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn f32x4_blend_branchless() {
        let a   = f32x4::new(1.0, 2.0, 3.0, 4.0);
        let b   = f32x4::new(10.0, 20.0, 30.0, 40.0);
        // Choose a where lane > 2, else b
        let gt  = a.cmpgt(f32x4::splat(2.0)); // 0b1100
        let out = f32x4::blend(gt, a, b);
        assert_eq!(out.to_array(), [10.0, 20.0, 3.0, 4.0]);
    }

    #[test]
    fn f32x4_is_finite() {
        assert!(f32x4::splat(1.0).is_finite());
        assert!(!f32x4::new(1.0, f32::INFINITY, 1.0, 1.0).is_finite());
        assert!(!f32x4::new(1.0, f32::NAN, 1.0, 1.0).is_finite());
    }

    // ═════════════════════════════════════════════════════════════════════════
    //  Vec3x4
    // ═════════════════════════════════════════════════════════════════════════

    #[test]
    fn vec3x4_size_align() {
        assert_eq!(core::mem::size_of::<Vec3x4>(), 48);
        assert_eq!(core::mem::align_of::<Vec3x4>(), 16);
    }

    #[test]
    fn vec3x4_aos_soa_roundtrip() {
        let vecs = [
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(4.0, 5.0, 6.0),
            Vec3::new(7.0, 8.0, 9.0),
            Vec3::new(10.0, 11.0, 12.0),
        ];
        let wide = Vec3x4::from_slice(&vecs);
        let back = wide.to_array();
        for (a, b) in vecs.iter().zip(back.iter()) {
            assert!(vec3_approx(*a, *b), "roundtrip mismatch: {:?} vs {:?}", a, b);
        }
    }

    #[test]
    fn vec3x4_splat() {
        let v = Vec3::new(3.0, 4.0, 5.0);
        let wide = Vec3x4::splat(v);
        for lane in wide.to_array() {
            assert!(vec3_approx(lane, v));
        }
    }

    #[test]
    fn vec3x4_add() {
        let a = Vec3x4::splat(Vec3::new(1.0, 2.0, 3.0));
        let b = Vec3x4::splat(Vec3::new(4.0, 5.0, 6.0));
        let expected = Vec3::new(5.0, 7.0, 9.0);
        for lane in (a + b).to_array() {
            assert!(vec3_approx(lane, expected));
        }
    }

    #[test]
    fn vec3x4_sub() {
        let a = Vec3x4::splat(Vec3::new(5.0, 7.0, 9.0));
        let b = Vec3x4::splat(Vec3::new(1.0, 2.0, 3.0));
        let expected = Vec3::new(4.0, 5.0, 6.0);
        for lane in (a - b).to_array() {
            assert!(vec3_approx(lane, expected));
        }
    }

    #[test]
    fn vec3x4_dot_4_simultaneous() {
        // X · Y = 0 for all 4 pairs
        let a = Vec3x4::splat(Vec3::X);
        let b = Vec3x4::splat(Vec3::Y);
        let dots = a.dot(b).to_array();
        for d in dots { assert!(d.abs() < 1e-6, "X·Y should be 0, got {}", d); }

        // X · X = 1 for all 4 pairs
        let dots2 = a.dot(a).to_array();
        for d in dots2 { assert!((d - 1.0).abs() < 1e-6, "X·X should be 1, got {}", d); }
    }

    #[test]
    fn vec3x4_cross_basis() {
        // X × Y = Z for all 4 lanes simultaneously
        let a = Vec3x4::splat(Vec3::X);
        let b = Vec3x4::splat(Vec3::Y);
        let result = a.cross(b);
        for lane in result.to_array() {
            assert!(vec3_approx(lane, Vec3::Z), "X×Y should be Z, got {:?}", lane);
        }
    }

    #[test]
    fn vec3x4_cross_anticommutative() {
        let vecs_a = [Vec3::new(1.0,2.0,3.0); 4];
        let vecs_b = [Vec3::new(4.0,5.0,6.0); 4];
        let a = Vec3x4::from_slice(&vecs_a);
        let b = Vec3x4::from_slice(&vecs_b);
        // a×b + b×a = 0
        let sum = a.cross(b) + b.cross(a);
        for lane in sum.to_array() {
            assert!(vec3_approx(lane, Vec3::ZERO),
                "a×b + b×a should be zero, got {:?}", lane);
        }
    }

    #[test]
    fn vec3x4_normalize_unit_length() {
        let vecs = [
            Vec3::new(3.0, 4.0, 0.0),
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(0.0, 0.0, 5.0),
            Vec3::new(2.0, 0.0, 0.0),
        ];
        let wide = Vec3x4::from_slice(&vecs).normalize();
        for lane in wide.to_array() {
            let len = (lane.x*lane.x + lane.y*lane.y + lane.z*lane.z).sqrt();
            assert!((len - 1.0).abs() < 1e-5, "normalized length should be 1, got {}", len);
        }
    }

    #[test]
    fn vec3x4_normalize_degenerate_lane_safe() {
        // One zero vector — should not produce NaN/Inf
        let vecs = [
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::ZERO,  // degenerate
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ];
        let wide = Vec3x4::from_slice(&vecs).normalize();
        for lane in wide.to_array() {
            assert!(lane.is_finite(),
                "normalize should not produce NaN/Inf for degenerate lane");
        }
    }

    #[test]
    fn vec3x4_lerp_midpoint() {
        let a = Vec3x4::splat(Vec3::ZERO);
        let b = Vec3x4::splat(Vec3::new(2.0, 4.0, 6.0));
        let t = f32x4::splat(0.5);
        let mid = a.lerp(b, t);
        let expected = Vec3::new(1.0, 2.0, 3.0);
        for lane in mid.to_array() {
            assert!(vec3_approx(lane, expected));
        }
    }

    #[test]
    fn vec3x4_lerp_per_lane_t() {
        let a = Vec3x4::splat(Vec3::ZERO);
        let b = Vec3x4::splat(Vec3::new(10.0, 10.0, 10.0));
        // Different t per lane
        let t = f32x4::new(0.0, 0.25, 0.5, 1.0);
        let result = a.lerp(b, t).to_array();
        let expected = [0.0f32, 2.5, 5.0, 10.0];
        for (lane, exp) in result.iter().zip(expected.iter()) {
            assert!((lane.x - exp).abs() < 1e-5,
                "lane x {} vs expected {}", lane.x, exp);
        }
    }

    #[test]
    fn vec3x4_select_branchless() {
        let zero_lanes = Vec3x4::splat(Vec3::ZERO);
        let one_lanes  = Vec3x4::splat(Vec3::ONE);
        // Select ones where lane index < 2 (first 2 lanes)
        // Mask: lane 0 and 1 = true (all ones), lane 2 and 3 = false
        let vals = f32x4::new(1.0, 1.0, 0.0, 0.0);
        let mask = vals.cmpgt(f32x4::splat(0.5)); // lanes 0,1 true
        let sel = Vec3x4::select(mask, one_lanes, zero_lanes).to_array();
        assert!(vec3_approx(sel[0], Vec3::ONE));
        assert!(vec3_approx(sel[1], Vec3::ONE));
        assert!(vec3_approx(sel[2], Vec3::ZERO));
        assert!(vec3_approx(sel[3], Vec3::ZERO));
    }

    #[test]
    fn vec3x4_matches_scalar_transform() {
        // The critical regression test: Vec3x4 ops must match their scalar equivalents
        let vecs = [
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(1.0, 1.0, 1.0),
        ];
        let other = Vec3::new(2.0, 3.0, 4.0);
        let other_wide = Vec3x4::splat(other);
        let vecs_wide  = Vec3x4::from_slice(&vecs);

        // Dot product
        let wide_dots  = vecs_wide.dot(other_wide).to_array();
        let scalar_dots: Vec<f32> = vecs.iter().map(|v| v.dot(other)).collect();
        for (w, s) in wide_dots.iter().zip(scalar_dots.iter()) {
            assert!((w - s).abs() < 1e-5, "dot mismatch: {} vs {}", w, s);
        }

        // Cross product
        let wide_cross  = vecs_wide.cross(other_wide).to_array();
        let scalar_cross: Vec<Vec3> = vecs.iter().map(|v| v.cross(other)).collect();
        for (w, s) in wide_cross.iter().zip(scalar_cross.iter()) {
            assert!(vec3_approx(*w, *s), "cross mismatch: {:?} vs {:?}", w, s);
        }
    }

    #[test]
    fn vec3x4_mat4_transform_matches_scalar() {
        // transform_vec3x4 must match 4× scalar transform_point
        let m = Mat4::from_trs(
            Vec3::new(1.0, 2.0, 3.0),
            Quat::from_axis_angle(Vec3::Y, to_radians(45.0)),
            Vec3::new(2.0, 2.0, 2.0),
        );
        let positions = [
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(1.0, 1.0, 1.0),
        ];
        let wide_in  = Vec3x4::from_slice(&positions);
        let wide_out = m.transform_vec3x4(wide_in).to_array();
        let scalar_out: Vec<Vec3> = positions.iter()
            .map(|&p| m.transform_point(p))
            .collect();

        for (w, s) in wide_out.iter().zip(scalar_out.iter()) {
            assert!(vec3_approx(*w, *s),
                "transform_vec3x4 mismatch: {:?} vs {:?}", w, s);
        }
    }

    // ═════════════════════════════════════════════════════════════════════════
    //  QuatX4
    // ═════════════════════════════════════════════════════════════════════════

    #[test]
    fn quatx4_size_align() {
        assert_eq!(core::mem::size_of::<QuatX4>(), 64);
        assert_eq!(core::mem::align_of::<QuatX4>(), 16);
    }

    #[test]
    fn quatx4_identity_default() {
        let q = QuatX4::IDENTITY;
        for lane in q.to_array() {
            assert!((lane.x).abs() < 1e-6);
            assert!((lane.y).abs() < 1e-6);
            assert!((lane.z).abs() < 1e-6);
            assert!((lane.w - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn quatx4_aos_soa_roundtrip() {
        let quats = [
            Quat::from_axis_angle(Vec3::Y, to_radians(45.0)),
            Quat::from_axis_angle(Vec3::X, to_radians(90.0)),
            Quat::from_axis_angle(Vec3::Z, to_radians(30.0)),
            Quat::IDENTITY,
        ];
        let wide = QuatX4::from_slice(&quats);
        let back = wide.to_array();
        for (a, b) in quats.iter().zip(back.iter()) {
            assert!((a.x - b.x).abs() < 1e-5);
            assert!((a.y - b.y).abs() < 1e-5);
            assert!((a.z - b.z).abs() < 1e-5);
            assert!((a.w - b.w).abs() < 1e-5);
        }
    }

    #[test]
    fn quatx4_dot_self_is_one_for_unit_quats() {
        let q = Quat::from_axis_angle(Vec3::Y, to_radians(45.0));
        let wide = QuatX4::splat(q);
        let dots = wide.dot(wide).to_array();
        for d in dots {
            assert!((d - 1.0).abs() < 1e-5, "unit quat |q|² should be 1, got {}", d);
        }
    }

    #[test]
    fn quatx4_normalize_unit_length() {
        // Construct un-normalised quaternions by splat + small perturbation via arithmetic
        let q = Quat::from_axis_angle(Vec3::Y, to_radians(45.0));
        let wide = QuatX4::splat(q).normalize();
        for lane in wide.to_array() {
            let len_sq = lane.x*lane.x + lane.y*lane.y + lane.z*lane.z + lane.w*lane.w;
            assert!((len_sq - 1.0).abs() < 1e-5, "length² after normalize: {}", len_sq);
        }
    }

    #[test]
    fn quatx4_mul_matches_scalar() {
        let q1 = Quat::from_axis_angle(Vec3::Y, to_radians(30.0));
        let q2 = Quat::from_axis_angle(Vec3::X, to_radians(45.0));
        let scalar = q1 * q2;

        let w1 = QuatX4::splat(q1);
        let w2 = QuatX4::splat(q2);
        let wide = (w1 * w2).to_array();

        for lane in wide {
            assert!((lane.x - scalar.x).abs() < 1e-5, "x mismatch");
            assert!((lane.y - scalar.y).abs() < 1e-5, "y mismatch");
            assert!((lane.z - scalar.z).abs() < 1e-5, "z mismatch");
            assert!((lane.w - scalar.w).abs() < 1e-5, "w mismatch");
        }
    }

    #[test]
    fn quatx4_conjugate_is_inverse_for_unit() {
        let q = Quat::from_axis_angle(Vec3::new(1.0,1.0,0.0).normalize(), to_radians(37.0));
        let w  = QuatX4::splat(q);
        let wc = w.conjugate();
        // w * conj(w) should be identity (x=y=z≈0, w≈1)
        let product = (w * wc).normalize().to_array();
        for lane in product {
            assert!(lane.x.abs() < 1e-4, "x should be ~0 after w*conj(w)");
            assert!(lane.y.abs() < 1e-4, "y should be ~0 after w*conj(w)");
            assert!(lane.z.abs() < 1e-4, "z should be ~0 after w*conj(w)");
            assert!((lane.w - 1.0).abs() < 1e-4, "w should be ~1 after w*conj(w)");
        }
    }

    #[test]
    fn quatx4_nlerp_stays_normalised() {
        let a = Quat::from_axis_angle(Vec3::Y, to_radians(0.0));
        let b = Quat::from_axis_angle(Vec3::Y, to_radians(90.0));
        let wa = QuatX4::splat(a);
        let wb = QuatX4::splat(b);
        let result = wa.nlerp(wb, f32x4::splat(0.3)).to_array();
        for lane in result {
            let len_sq = lane.x*lane.x + lane.y*lane.y + lane.z*lane.z + lane.w*lane.w;
            assert!((len_sq - 1.0).abs() < 1e-5, "nlerp result not unit: {}", len_sq);
        }
    }

    #[test]
    fn quatx4_rotate_matches_scalar() {
        let q = Quat::from_axis_angle(Vec3::Y, to_radians(90.0));
        let v = Vec3::X;
        let scalar_result = q.rotate(v);

        let wq = QuatX4::splat(q);
        let wv = Vec3x4::splat(v);
        let wide_result = wq.rotate(wv).to_array();

        for lane in wide_result {
            assert!(vec3_approx(lane, scalar_result),
                "QuatX4::rotate mismatch: {:?} vs {:?}", lane, scalar_result);
        }
    }

    #[test]
    fn quatx4_rotate_x_by_90y_is_neg_z() {
        // 90° rotation around Y maps X → -Z
        let q = Quat::from_axis_angle(Vec3::Y, to_radians(90.0));
        let wq = QuatX4::splat(q);
        let wv = Vec3x4::splat(Vec3::X);
        for lane in wq.rotate(wv).to_array() {
            assert!(vec3_approx(lane, Vec3::NEG_Z),
                "Expected NEG_Z, got {:?}", lane);
        }
    }

    #[test]
    fn quatx4_4_different_rotations_simultaneously() {
        // Each lane rotates by a different angle — verify against individual scalar results
        let angles = [0.0f32, 45.0, 90.0, 135.0];
        let quats: [Quat; 4] = core::array::from_fn(|i| {
            Quat::from_axis_angle(Vec3::Y, to_radians(angles[i]))
        });
        let v = Vec3::X;
        let scalar_results: [Vec3; 4] = core::array::from_fn(|i| quats[i].rotate(v));

        let wq = QuatX4::from_slice(&quats);
        let wv = Vec3x4::splat(v);
        let wide_results = wq.rotate(wv).to_array();

        for (i, (w, s)) in wide_results.iter().zip(scalar_results.iter()).enumerate() {
            assert!(vec3_approx(*w, *s),
                "Lane {} mismatch: wide={:?} scalar={:?}", i, w, s);
        }
    }
               }
