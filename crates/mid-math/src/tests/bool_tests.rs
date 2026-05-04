// crates/mid-math/src/tests/bool_tests.rs
#[cfg(test)]
mod tests {
    use crate::{BVec2, BVec3, BVec4};

    // ── BVec2 ─────────────────────────────────────────────────────────────────

    #[test]
    fn bvec2_size_align() {
        assert_eq!(core::mem::size_of::<BVec2>(), 2);
        assert_eq!(core::mem::align_of::<BVec2>(), 1);
    }

    #[test]
    fn bvec2_constants() {
        assert!(!BVec2::FALSE.any());
        assert!(!BVec2::FALSE.all());
        assert!( BVec2::TRUE.any());
        assert!( BVec2::TRUE.all());
    }

    #[test]
    fn bvec2_any_all() {
        assert!( BVec2::new(true, false).any());
        assert!(!BVec2::new(true, false).all());
        assert!(!BVec2::new(false, false).any());
        assert!( BVec2::new(true, true).all());
    }

    #[test]
    fn bvec2_bitmask() {
        assert_eq!(BVec2::new(false, false).bitmask(), 0b00);
        assert_eq!(BVec2::new(true,  false).bitmask(), 0b01);
        assert_eq!(BVec2::new(false, true ).bitmask(), 0b10);
        assert_eq!(BVec2::new(true,  true ).bitmask(), 0b11);
    }

    #[test]
    fn bvec2_test_indexed() {
        let v = BVec2::new(true, false);
        assert!( v.test(0));
        assert!(!v.test(1));
    }

    #[test]
    #[should_panic]
    fn bvec2_test_oob_panics() {
        let _ = BVec2::FALSE.test(2);
    }

    #[test]
    fn bvec2_bitops() {
        let a = BVec2::new(true, false);
        let b = BVec2::new(false, true);
        assert_eq!(a & b, BVec2::FALSE);
        assert_eq!(a | b, BVec2::TRUE);
        assert_eq!(a ^ b, BVec2::TRUE);
        assert_eq!(!a, b);
        assert_eq!(!b, a);
        assert_eq!(a ^ a, BVec2::FALSE);
    }

    #[test]
    fn bvec2_assign_ops() {
        let mut v = BVec2::new(true, false);
        v &= BVec2::TRUE;
        assert_eq!(v, BVec2::new(true, false));
        v |= BVec2::new(false, true);
        assert_eq!(v, BVec2::TRUE);
        v ^= BVec2::TRUE;
        assert_eq!(v, BVec2::FALSE);
    }

    #[test]
    fn bvec2_splat() {
        assert_eq!(BVec2::splat(true),  BVec2::TRUE);
        assert_eq!(BVec2::splat(false), BVec2::FALSE);
    }

    #[test]
    fn bvec2_from_array_roundtrip() {
        let arr = [true, false];
        let v = BVec2::from(arr);
        assert_eq!(v.x, arr[0]);
        assert_eq!(v.y, arr[1]);
        let back: [bool; 2] = v.into();
        assert_eq!(arr, back);
    }

    #[test]
    fn bvec2_from_tuple_roundtrip() {
        let v = BVec2::from((true, false));
        let t: (bool, bool) = v.into();
        assert_eq!(t, (true, false));
    }

    // ── BVec3 ─────────────────────────────────────────────────────────────────

    #[test]
    fn bvec3_size_align() {
        assert_eq!(core::mem::size_of::<BVec3>(), 3);
        assert_eq!(core::mem::align_of::<BVec3>(), 1);
    }

    #[test]
    fn bvec3_any_all() {
        assert!(!BVec3::FALSE.any());
        assert!( BVec3::TRUE.all());
        assert!( BVec3::new(false, false, true).any());
        assert!(!BVec3::new(true,  true,  false).all());
    }

    #[test]
    fn bvec3_bitmask() {
        assert_eq!(BVec3::new(true, false, false).bitmask(), 0b001);
        assert_eq!(BVec3::new(false, true, false).bitmask(), 0b010);
        assert_eq!(BVec3::new(false, false, true).bitmask(), 0b100);
        assert_eq!(BVec3::new(true,  false, true).bitmask(), 0b101);
        assert_eq!(BVec3::TRUE.bitmask(),                    0b111);
        assert_eq!(BVec3::FALSE.bitmask(),                   0b000);
    }

    #[test]
    fn bvec3_bitops() {
        let a = BVec3::new(true, false, true);
        let b = BVec3::new(false, true, true);
        assert_eq!(a & b, BVec3::new(false, false, true));
        assert_eq!(a | b, BVec3::TRUE);
        assert_eq!(a ^ b, BVec3::new(true, true, false));
        assert_eq!(!BVec3::TRUE, BVec3::FALSE);
        assert_eq!(!BVec3::FALSE, BVec3::TRUE);
    }

    #[test]
    fn bvec3_test_indexed() {
        let v = BVec3::new(false, true, false);
        assert!(!v.test(0));
        assert!( v.test(1));
        assert!(!v.test(2));
    }

    #[test]
    #[should_panic]
    fn bvec3_test_oob_panics() {
        let _ = BVec3::FALSE.test(3);
    }

    #[test]
    fn bvec3_from_array_roundtrip() {
        let arr = [true, false, true];
        let v = BVec3::from(arr);
        let back: [bool; 3] = v.into();
        assert_eq!(arr, back);
    }

    // ── BVec4 ─────────────────────────────────────────────────────────────────

    #[test]
    fn bvec4_size_align() {
        assert_eq!(core::mem::size_of::<BVec4>(), 4);
        assert_eq!(core::mem::align_of::<BVec4>(), 1);
    }

    #[test]
    fn bvec4_any_all() {
        assert!(!BVec4::FALSE.any());
        assert!( BVec4::TRUE.all());
        assert!( BVec4::new(false, false, false, true).any());
        assert!(!BVec4::new(true, true, true, false).all());
    }

    #[test]
    fn bvec4_bitmask() {
        assert_eq!(BVec4::new(true, false, false, false).bitmask(), 0b0001);
        assert_eq!(BVec4::new(false, true, false, false).bitmask(), 0b0010);
        assert_eq!(BVec4::new(false, false, true, false).bitmask(), 0b0100);
        assert_eq!(BVec4::new(false, false, false, true).bitmask(), 0b1000);
        assert_eq!(BVec4::new(true, false, true, false).bitmask(),  0b0101);
        assert_eq!(BVec4::new(false, true, false, true).bitmask(),  0b1010);
        assert_eq!(BVec4::TRUE.bitmask(),                           0b1111);
    }

    #[test]
    fn bvec4_bitops() {
        let a = BVec4::new(true, false, true, false);
        let b = BVec4::new(true, true, false, false);
        assert_eq!(a & b, BVec4::new(true,  false, false, false));
        assert_eq!(a | b, BVec4::new(true,  true,  true,  false));
        assert_eq!(a ^ b, BVec4::new(false, true,  true,  false));
        assert_eq!(!a,    BVec4::new(false, true,  false, true));
    }

    #[test]
    fn bvec4_test_indexed() {
        let v = BVec4::new(true, false, true, false);
        assert!( v.test(0));
        assert!(!v.test(1));
        assert!( v.test(2));
        assert!(!v.test(3));
    }

    #[test]
    #[should_panic]
    fn bvec4_test_oob_panics() {
        let _ = BVec4::FALSE.test(4);
    }

    #[test]
    fn bvec4_splat() {
        assert_eq!(BVec4::splat(true),  BVec4::TRUE);
        assert_eq!(BVec4::splat(false), BVec4::FALSE);
    }

    #[test]
    fn bvec4_from_array_roundtrip() {
        let arr = [true, false, true, false];
        let v = BVec4::from(arr);
        let back: [bool; 4] = v.into();
        assert_eq!(arr, back);
    }

    #[test]
    fn bvec4_assign_ops() {
        let mut v = BVec4::TRUE;
        v &= BVec4::new(true, false, true, false);
        assert_eq!(v, BVec4::new(true, false, true, false));
        v |= BVec4::new(false, true, false, true);
        assert_eq!(v, BVec4::TRUE);
    }
  }
