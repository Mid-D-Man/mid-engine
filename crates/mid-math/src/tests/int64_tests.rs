// crates/mid-math/src/tests/int64_tests.rs
//! Correctness tests for i64/u64 integer vector types.
//!
//! Mirrors int32 tests exactly — same coverage pattern, larger element size.
//! Expected: all size/align, arithmetic, bitwise, overflow, cast, FFI checks.

#[cfg(test)]
mod tests {
    use crate::{
        BVec2, BVec3, BVec4,
        I64Vec2, I64Vec3, I64Vec4,
        U64Vec2, U64Vec3, U64Vec4,
    };

    // ═══════════════════════════════════════════════════════════════════════
    //  I64Vec2
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn i64vec2_size_align() {
        assert_eq!(std::mem::size_of::<I64Vec2>(),  16);
        assert_eq!(std::mem::align_of::<I64Vec2>(),  8);
    }

    #[test]
    fn i64vec2_constants() {
        assert_eq!(I64Vec2::ZERO, I64Vec2::new(0, 0));
        assert_eq!(I64Vec2::ONE,  I64Vec2::new(1, 1));
        assert_eq!(I64Vec2::X,    I64Vec2::new(1, 0));
        assert_eq!(I64Vec2::Y,    I64Vec2::new(0, 1));
        assert_eq!(I64Vec2::MIN,  I64Vec2::splat(i64::MIN));
        assert_eq!(I64Vec2::MAX,  I64Vec2::splat(i64::MAX));
    }

    #[test]
    fn i64vec2_add_sub() {
        let a = I64Vec2::new(10, 20);
        let b = I64Vec2::new(3, 7);
        assert_eq!(a + b, I64Vec2::new(13, 27));
        assert_eq!(a - b, I64Vec2::new(7, 13));
    }

    #[test]
    fn i64vec2_mul_scale() {
        let a = I64Vec2::new(3, 4);
        let b = I64Vec2::new(2, 5);
        assert_eq!(a * b, I64Vec2::new(6, 20));
        assert_eq!(a * 3i64, I64Vec2::new(9, 12));
        assert_eq!(3i64 * a, I64Vec2::new(9, 12));
    }

    #[test]
    fn i64vec2_neg() {
        assert_eq!(-I64Vec2::new(3, -4), I64Vec2::new(-3, 4));
    }

    #[test]
    fn i64vec2_dot() {
        let a = I64Vec2::new(1, 2);
        let b = I64Vec2::new(3, 4);
        assert_eq!(a.dot(b), 11);
    }

    #[test]
    fn i64vec2_abs() {
        assert_eq!(I64Vec2::new(-5, 3).abs(), I64Vec2::new(5, 3));
        assert_eq!(I64Vec2::new(0, -100).abs(), I64Vec2::new(0, 100));
    }

    #[test]
    fn i64vec2_signum() {
        assert_eq!(I64Vec2::new(-7, 3).signum(), I64Vec2::new(-1, 1));
        assert_eq!(I64Vec2::new(0, 0).signum(), I64Vec2::new(0, 0));
    }

    #[test]
    fn i64vec2_min_max_clamp() {
        let a = I64Vec2::new(1, 8);
        let b = I64Vec2::new(5, 3);
        assert_eq!(a.min(b), I64Vec2::new(1, 3));
        assert_eq!(a.max(b), I64Vec2::new(5, 8));
        let lo = I64Vec2::new(2, 2);
        let hi = I64Vec2::new(6, 6);
        assert_eq!(I64Vec2::new(0, 10).clamp(lo, hi), I64Vec2::new(2, 6));
    }

    #[test]
    fn i64vec2_element_ops() {
        let v = I64Vec2::new(3, 7);
        assert_eq!(v.min_element(), 3);
        assert_eq!(v.max_element(), 7);
        assert_eq!(v.element_sum(), 10);
        assert_eq!(v.element_product(), 21);
    }

    #[test]
    fn i64vec2_perp_and_perp_dot() {
        let v = I64Vec2::new(3, 0);
        assert_eq!(v.perp(), I64Vec2::new(0, 3));
        let a = I64Vec2::new(1, 0);
        let b = I64Vec2::new(0, 1);
        assert_eq!(a.perp_dot(b), 1);
    }

    #[test]
    fn i64vec2_length_sq() {
        assert_eq!(I64Vec2::new(3, 4).length_sq(), 25);
        assert_eq!(I64Vec2::new(0, 0).length_sq(), 0);
    }

    #[test]
    fn i64vec2_distance_sq() {
        let a = I64Vec2::new(0, 0);
        let b = I64Vec2::new(3, 4);
        assert_eq!(a.distance_sq(b), 25);
    }

    #[test]
    fn i64vec2_manhattan_distance() {
        let a = I64Vec2::new(0, 0);
        let b = I64Vec2::new(3, 4);
        assert_eq!(a.manhattan_distance(b), 7);
    }

    #[test]
    fn i64vec2_checked_manhattan_overflow() {
        let a = I64Vec2::new(i64::MAX, 0);
        let b = I64Vec2::new(i64::MIN, 0);
        // abs_diff saturates to u64::MAX at overflow, checked_manhattan_distance
        // may return None when sum overflows u64
        let _ = a.manhattan_distance(b); // must not panic
        let _ = a.checked_manhattan_distance(b); // None is acceptable
    }

    #[test]
    fn i64vec2_wrapping() {
        let a = I64Vec2::splat(i64::MAX);
        let b = I64Vec2::splat(1);
        assert_eq!(a.wrapping_add(b), I64Vec2::splat(i64::MIN));
        let c = I64Vec2::splat(i64::MIN);
        assert_eq!(c.wrapping_sub(b), I64Vec2::splat(i64::MAX));
    }

    #[test]
    fn i64vec2_saturating() {
        let a = I64Vec2::splat(i64::MAX);
        let b = I64Vec2::splat(1);
        assert_eq!(a.saturating_add(b), I64Vec2::splat(i64::MAX));
        let c = I64Vec2::splat(i64::MIN);
        assert_eq!(c.saturating_sub(b), I64Vec2::splat(i64::MIN));
    }

    #[test]
    fn i64vec2_checked() {
        let a = I64Vec2::splat(i64::MAX);
        let b = I64Vec2::splat(1);
        assert!(a.checked_add(b).is_none());
        assert!(I64Vec2::new(2, 3).checked_add(I64Vec2::new(4, 5)).is_some());
    }

    #[test]
    fn i64vec2_cmp_ops() {
        let a = I64Vec2::new(1, 5);
        let b = I64Vec2::new(3, 5);
        assert_eq!(a.cmpeq(b), BVec2::new(false, true));
        assert_eq!(a.cmplt(b), BVec2::new(true,  false));
        assert_eq!(a.cmpgt(b), BVec2::new(false, false));
        assert_eq!(a.cmple(b), BVec2::new(true,  true));
    }

    #[test]
    fn i64vec2_select() {
        let t = I64Vec2::new(10, 20);
        let f = I64Vec2::new(1,  2);
        let r = I64Vec2::select(BVec2::new(true, false), t, f);
        assert_eq!(r, I64Vec2::new(10, 2));
    }

    #[test]
    fn i64vec2_extend() {
        assert_eq!(I64Vec2::new(1, 2).extend(3), I64Vec3::new(1, 2, 3));
    }

    #[test]
    fn i64vec2_index() {
        let v = I64Vec2::new(10, 20);
        assert_eq!(v[0], 10);
        assert_eq!(v[1], 20);
    }

    #[test]
    #[should_panic]
    fn i64vec2_index_oob_panics() {
        let v = I64Vec2::new(1, 2);
        let _ = v[2];
    }

    #[test]
    fn i64vec2_from_array_roundtrip() {
        let a = [3i64, 7i64];
        let v = I64Vec2::from(a);
        let b: [i64; 2] = v.into();
        assert_eq!(a, b);
    }

    #[test]
    fn i64vec2_casts() {
        let v = I64Vec2::new(1, 2);
        let _f  = v.as_vec2();
        let _d  = v.as_dvec2();
        let _i  = v.as_ivec2();
        let _u  = v.as_uvec2();
        let _u6 = v.as_u64vec2();
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  I64Vec3
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn i64vec3_size_align() {
        // 24 bytes, no padding — matches DVec3 philosophy
        assert_eq!(std::mem::size_of::<I64Vec3>(),  24);
        assert_eq!(std::mem::align_of::<I64Vec3>(),  8);
    }

    #[test]
    fn i64vec3_constants() {
        assert_eq!(I64Vec3::ZERO,  I64Vec3::new(0, 0, 0));
        assert_eq!(I64Vec3::X,     I64Vec3::new(1, 0, 0));
        assert_eq!(I64Vec3::NEG_Z, I64Vec3::new(0, 0, -1));
    }

    #[test]
    fn i64vec3_add_sub_neg() {
        let a = I64Vec3::new(1, 2, 3);
        let b = I64Vec3::new(4, 5, 6);
        assert_eq!(a + b, I64Vec3::new(5, 7, 9));
        assert_eq!(b - a, I64Vec3::new(3, 3, 3));
        assert_eq!(-a,    I64Vec3::new(-1, -2, -3));
    }

    #[test]
    fn i64vec3_dot() {
        let a = I64Vec3::new(1, 0, 0);
        let b = I64Vec3::new(0, 1, 0);
        assert_eq!(a.dot(b), 0);
        assert_eq!(a.dot(a), 1);
        assert_eq!(I64Vec3::new(1, 2, 3).dot(I64Vec3::new(4, 5, 6)), 32);
    }

    #[test]
    fn i64vec3_cross_basis() {
        let z = I64Vec3::X.cross(I64Vec3::Y);
        assert_eq!(z, I64Vec3::Z);
    }

    #[test]
    fn i64vec3_cross_anticommutative() {
        let a = I64Vec3::new(1, 2, 3);
        let b = I64Vec3::new(4, 5, 6);
        let sum = a.cross(b) + b.cross(a);
        assert_eq!(sum, I64Vec3::ZERO);
    }

    #[test]
    fn i64vec3_abs() {
        assert_eq!(I64Vec3::new(-1, 2, -3).abs(), I64Vec3::new(1, 2, 3));
    }

    #[test]
    fn i64vec3_length_sq() {
        assert_eq!(I64Vec3::new(1, 2, 2).length_sq(), 9);
    }

    #[test]
    fn i64vec3_min_max_clamp() {
        let a = I64Vec3::new(1, 5, 3);
        let b = I64Vec3::new(4, 2, 3);
        assert_eq!(a.min(b), I64Vec3::new(1, 2, 3));
        assert_eq!(a.max(b), I64Vec3::new(4, 5, 3));
        let lo = I64Vec3::splat(2);
        let hi = I64Vec3::splat(4);
        assert_eq!(I64Vec3::new(0, 3, 9).clamp(lo, hi), I64Vec3::new(2, 3, 4));
    }

    #[test]
    fn i64vec3_element_ops() {
        let v = I64Vec3::new(1, 5, 3);
        assert_eq!(v.min_element(), 1);
        assert_eq!(v.max_element(), 5);
        assert_eq!(v.element_sum(), 9);
    }

    #[test]
    fn i64vec3_manhattan_distance() {
        let a = I64Vec3::ZERO;
        let b = I64Vec3::new(1, 2, 3);
        assert_eq!(a.manhattan_distance(b), 6);
    }

    #[test]
    fn i64vec3_wrapping_saturating() {
        let max = I64Vec3::splat(i64::MAX);
        let one = I64Vec3::splat(1);
        assert_eq!(max.wrapping_add(one), I64Vec3::splat(i64::MIN));
        assert_eq!(max.saturating_add(one), max);
    }

    #[test]
    fn i64vec3_cmp_ops() {
        let a = I64Vec3::new(1, 2, 3);
        let b = I64Vec3::new(1, 3, 2);
        assert_eq!(a.cmpeq(b), BVec3::new(true, false, false));
        assert_eq!(a.cmplt(b), BVec3::new(false, true, false));
        assert_eq!(a.cmpgt(b), BVec3::new(false, false, true));
    }

    #[test]
    fn i64vec3_select() {
        let t = I64Vec3::new(10, 20, 30);
        let f = I64Vec3::new(1,  2,  3);
        let r = I64Vec3::select(BVec3::new(true, false, true), t, f);
        assert_eq!(r, I64Vec3::new(10, 2, 30));
    }

    #[test]
    fn i64vec3_extend_truncate() {
        let v = I64Vec3::new(1, 2, 3);
        assert_eq!(v.extend(4), I64Vec4::new(1, 2, 3, 4));
        assert_eq!(v.truncate(), I64Vec2::new(1, 2));
    }

    #[test]
    fn i64vec3_index() {
        let v = I64Vec3::new(10, 20, 30);
        assert_eq!(v[0], 10);
        assert_eq!(v[1], 20);
        assert_eq!(v[2], 30);
    }

    #[test]
    #[should_panic]
    fn i64vec3_index_oob_panics() {
        let _ = I64Vec3::ZERO[3];
    }

    #[test]
    fn i64vec3_casts() {
        let v = I64Vec3::new(1, 2, 3);
        let _f  = v.as_vec3();
        let _d  = v.as_dvec3();
        let _i  = v.as_ivec3();
        let _u  = v.as_uvec3();
        let _u6 = v.as_u64vec3();
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  I64Vec4
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn i64vec4_size_align() {
        assert_eq!(std::mem::size_of::<I64Vec4>(),  32);
        assert_eq!(std::mem::align_of::<I64Vec4>(),  8);
    }

    #[test]
    fn i64vec4_add_sub_neg() {
        let a = I64Vec4::new(1, 2, 3, 4);
        let b = I64Vec4::new(5, 6, 7, 8);
        assert_eq!(a + b, I64Vec4::new(6, 8, 10, 12));
        assert_eq!(b - a, I64Vec4::new(4, 4, 4, 4));
        assert_eq!(-a,    I64Vec4::new(-1, -2, -3, -4));
    }

    #[test]
    fn i64vec4_dot() {
        let a = I64Vec4::new(1, 2, 3, 4);
        let b = I64Vec4::new(5, 6, 7, 8);
        assert_eq!(a.dot(b), 70);
    }

    #[test]
    fn i64vec4_abs() {
        assert_eq!(I64Vec4::new(-1, 2, -3, 4).abs(), I64Vec4::new(1, 2, 3, 4));
    }

    #[test]
    fn i64vec4_length_sq() {
        assert_eq!(I64Vec4::new(1, 2, 3, 4).length_sq(), 30);
    }

    #[test]
    fn i64vec4_min_max_clamp() {
        let lo = I64Vec4::splat(2);
        let hi = I64Vec4::splat(5);
        let v  = I64Vec4::new(0, 3, 7, 4);
        assert_eq!(v.clamp(lo, hi), I64Vec4::new(2, 3, 5, 4));
    }

    #[test]
    fn i64vec4_element_ops() {
        let v = I64Vec4::new(1, 5, 3, 2);
        assert_eq!(v.min_element(), 1);
        assert_eq!(v.max_element(), 5);
        assert_eq!(v.element_sum(), 11);
    }

    #[test]
    fn i64vec4_manhattan_distance() {
        let a = I64Vec4::ZERO;
        let b = I64Vec4::new(1, 2, 3, 4);
        assert_eq!(a.manhattan_distance(b), 10);
    }

    #[test]
    fn i64vec4_wrapping_saturating() {
        let max = I64Vec4::splat(i64::MAX);
        let one = I64Vec4::splat(1);
        assert_eq!(max.wrapping_add(one), I64Vec4::splat(i64::MIN));
        assert_eq!(max.saturating_add(one), max);
    }

    #[test]
    fn i64vec4_cmp_ops() {
        let a = I64Vec4::new(1, 2, 3, 4);
        let b = I64Vec4::new(1, 3, 2, 4);
        assert_eq!(a.cmpeq(b), BVec4::new(true, false, false, true));
        assert_eq!(a.cmplt(b), BVec4::new(false, true, false, false));
    }

    #[test]
    fn i64vec4_select() {
        let t = I64Vec4::new(10, 20, 30, 40);
        let f = I64Vec4::new(1,  2,  3,  4);
        let r = I64Vec4::select(BVec4::new(true, false, true, false), t, f);
        assert_eq!(r, I64Vec4::new(10, 2, 30, 4));
    }

    #[test]
    fn i64vec4_truncate_and_xy() {
        let v = I64Vec4::new(1, 2, 3, 4);
        assert_eq!(v.truncate(), I64Vec3::new(1, 2, 3));
        assert_eq!(v.xy(),      I64Vec2::new(1, 2));
    }

    #[test]
    fn i64vec4_index() {
        let v = I64Vec4::new(10, 20, 30, 40);
        assert_eq!(v[0], 10);
        assert_eq!(v[3], 40);
    }

    #[test]
    #[should_panic]
    fn i64vec4_index_oob_panics() {
        let _ = I64Vec4::ZERO[4];
    }

    #[test]
    fn i64vec4_casts() {
        let v = I64Vec4::new(1, 2, 3, 4);
        let _f  = v.as_vec4();
        let _d  = v.as_dvec4();
        let _i  = v.as_ivec4();
        let _u  = v.as_uvec4();
        let _u6 = v.as_u64vec4();
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  U64Vec2
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn u64vec2_size_align() {
        assert_eq!(std::mem::size_of::<U64Vec2>(),  16);
        assert_eq!(std::mem::align_of::<U64Vec2>(),  8);
    }

    #[test]
    fn u64vec2_add_sub_mul() {
        let a = U64Vec2::new(10, 20);
        let b = U64Vec2::new(3, 7);
        assert_eq!(a + b, U64Vec2::new(13, 27));
        assert_eq!(a - b, U64Vec2::new(7, 13));
        assert_eq!(a * b, U64Vec2::new(30, 140));
        assert_eq!(a * 2u64, U64Vec2::new(20, 40));
    }

    #[test]
    fn u64vec2_dot() {
        assert_eq!(U64Vec2::new(2, 3).dot(U64Vec2::new(4, 5)), 23);
    }

    #[test]
    fn u64vec2_min_max_clamp() {
        let a = U64Vec2::new(1, 8);
        let b = U64Vec2::new(5, 3);
        assert_eq!(a.min(b), U64Vec2::new(1, 3));
        assert_eq!(a.max(b), U64Vec2::new(5, 8));
        let lo = U64Vec2::new(2, 2);
        let hi = U64Vec2::new(6, 6);
        assert_eq!(U64Vec2::new(0, 10).clamp(lo, hi), U64Vec2::new(2, 6));
    }

    #[test]
    fn u64vec2_element_ops() {
        let v = U64Vec2::new(3, 7);
        assert_eq!(v.min_element(), 3);
        assert_eq!(v.max_element(), 7);
        assert_eq!(v.element_sum(), 10);
        assert_eq!(v.element_product(), 21);
    }

    #[test]
    fn u64vec2_length_sq() {
        assert_eq!(U64Vec2::new(3, 4).length_sq(), 25);
    }

    #[test]
    fn u64vec2_wrapping() {
        let a = U64Vec2::splat(u64::MAX);
        let b = U64Vec2::splat(1);
        assert_eq!(a.wrapping_add(b), U64Vec2::splat(0));
        assert_eq!(U64Vec2::splat(0).wrapping_sub(b), U64Vec2::splat(u64::MAX));
    }

    #[test]
    fn u64vec2_saturating() {
        let max = U64Vec2::splat(u64::MAX);
        let one = U64Vec2::splat(1);
        assert_eq!(max.saturating_add(one), max);
        assert_eq!(U64Vec2::ZERO.saturating_sub(one), U64Vec2::ZERO);
    }

    #[test]
    fn u64vec2_cmp_ops() {
        let a = U64Vec2::new(1, 5);
        let b = U64Vec2::new(3, 5);
        assert_eq!(a.cmpeq(b), BVec2::new(false, true));
        assert_eq!(a.cmplt(b), BVec2::new(true,  false));
        assert_eq!(a.cmpge(b), BVec2::new(false, true));
    }

    #[test]
    fn u64vec2_select() {
        let t = U64Vec2::new(100, 200);
        let f = U64Vec2::new(1,   2);
        let r = U64Vec2::select(BVec2::new(false, true), t, f);
        assert_eq!(r, U64Vec2::new(1, 200));
    }

    #[test]
    fn u64vec2_extend() {
        assert_eq!(U64Vec2::new(1, 2).extend(3), U64Vec3::new(1, 2, 3));
    }

    #[test]
    fn u64vec2_casts() {
        let v = U64Vec2::new(1, 2);
        let _f  = v.as_vec2();
        let _d  = v.as_dvec2();
        let _i  = v.as_ivec2();
        let _u  = v.as_uvec2();
        let _i6 = v.as_i64vec2();
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  U64Vec3
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn u64vec3_size_align() {
        assert_eq!(std::mem::size_of::<U64Vec3>(),  24);
        assert_eq!(std::mem::align_of::<U64Vec3>(),  8);
    }

    #[test]
    fn u64vec3_cross_basis() {
        // U64Vec3 cross uses wrapping arithmetic — check X×Y≈Z for small values
        let z = U64Vec3::X.cross(U64Vec3::Y);
        assert_eq!(z, U64Vec3::Z);
    }

    #[test]
    fn u64vec3_dot() {
        let a = U64Vec3::new(1, 2, 3);
        let b = U64Vec3::new(4, 5, 6);
        assert_eq!(a.dot(b), 32);
    }

    #[test]
    fn u64vec3_min_max_clamp() {
        let a = U64Vec3::new(1, 5, 3);
        let b = U64Vec3::new(4, 2, 3);
        assert_eq!(a.min(b), U64Vec3::new(1, 2, 3));
        assert_eq!(a.max(b), U64Vec3::new(4, 5, 3));
    }

    #[test]
    fn u64vec3_element_ops() {
        let v = U64Vec3::new(2, 7, 4);
        assert_eq!(v.min_element(), 2);
        assert_eq!(v.max_element(), 7);
        assert_eq!(v.element_sum(), 13);
    }

    #[test]
    fn u64vec3_length_sq() {
        assert_eq!(U64Vec3::new(1, 2, 2).length_sq(), 9);
    }

    #[test]
    fn u64vec3_wrapping_saturating() {
        let max = U64Vec3::splat(u64::MAX);
        let one = U64Vec3::splat(1);
        assert_eq!(max.wrapping_add(one), U64Vec3::ZERO);
        assert_eq!(max.saturating_add(one), max);
    }

    #[test]
    fn u64vec3_cmp_ops() {
        let a = U64Vec3::new(1, 2, 3);
        let b = U64Vec3::new(1, 3, 2);
        assert_eq!(a.cmpeq(b), BVec3::new(true, false, false));
        assert_eq!(a.cmplt(b), BVec3::new(false, true, false));
    }

    #[test]
    fn u64vec3_extend_truncate() {
        let v = U64Vec3::new(1, 2, 3);
        assert_eq!(v.extend(4), U64Vec4::new(1, 2, 3, 4));
        assert_eq!(v.truncate(), U64Vec2::new(1, 2));
    }

    #[test]
    fn u64vec3_index() {
        let v = U64Vec3::new(10, 20, 30);
        assert_eq!(v[0], 10);
        assert_eq!(v[2], 30);
    }

    #[test]
    #[should_panic]
    fn u64vec3_index_oob_panics() {
        let _ = U64Vec3::ZERO[3];
    }

    #[test]
    fn u64vec3_casts() {
        let v = U64Vec3::new(1, 2, 3);
        let _f  = v.as_vec3();
        let _d  = v.as_dvec3();
        let _i  = v.as_ivec3();
        let _u  = v.as_uvec3();
        let _i6 = v.as_i64vec3();
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  U64Vec4
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn u64vec4_size_align() {
        assert_eq!(std::mem::size_of::<U64Vec4>(),  32);
        assert_eq!(std::mem::align_of::<U64Vec4>(),  8);
    }

    #[test]
    fn u64vec4_add_mul_scale() {
        let a = U64Vec4::new(1, 2, 3, 4);
        let b = U64Vec4::new(5, 6, 7, 8);
        assert_eq!(a + b, U64Vec4::new(6, 8, 10, 12));
        assert_eq!(a * b, U64Vec4::new(5, 12, 21, 32));
        assert_eq!(a * 2u64, U64Vec4::new(2, 4, 6, 8));
    }

    #[test]
    fn u64vec4_dot() {
        let a = U64Vec4::new(1, 2, 3, 4);
        let b = U64Vec4::new(5, 6, 7, 8);
        assert_eq!(a.dot(b), 70);
    }

    #[test]
    fn u64vec4_length_sq() {
        assert_eq!(U64Vec4::new(1, 2, 3, 4).length_sq(), 30);
    }

    #[test]
    fn u64vec4_min_max_clamp() {
        let lo = U64Vec4::splat(2);
        let hi = U64Vec4::splat(5);
        assert_eq!(U64Vec4::new(0, 3, 7, 4).clamp(lo, hi), U64Vec4::new(2, 3, 5, 4));
    }

    #[test]
    fn u64vec4_element_ops() {
        let v = U64Vec4::new(1, 5, 3, 2);
        assert_eq!(v.min_element(), 1);
        assert_eq!(v.max_element(), 5);
        assert_eq!(v.element_sum(), 11);
    }

    #[test]
    fn u64vec4_wrapping_saturating() {
        let max = U64Vec4::splat(u64::MAX);
        let one = U64Vec4::splat(1);
        assert_eq!(max.wrapping_add(one), U64Vec4::ZERO);
        assert_eq!(max.saturating_add(one), max);
    }

    #[test]
    fn u64vec4_cmp_ops() {
        let a = U64Vec4::new(1, 2, 3, 4);
        let b = U64Vec4::new(1, 3, 2, 4);
        assert_eq!(a.cmpeq(b), BVec4::new(true, false, false, true));
    }

    #[test]
    fn u64vec4_select() {
        let t = U64Vec4::new(10, 20, 30, 40);
        let f = U64Vec4::new(1,  2,  3,  4);
        let r = U64Vec4::select(BVec4::new(false, true, false, true), t, f);
        assert_eq!(r, U64Vec4::new(1, 20, 3, 40));
    }

    #[test]
    fn u64vec4_truncate_and_xy() {
        let v = U64Vec4::new(1, 2, 3, 4);
        assert_eq!(v.truncate(), U64Vec3::new(1, 2, 3));
        assert_eq!(v.xy(),      U64Vec2::new(1, 2));
    }

    #[test]
    fn u64vec4_index() {
        let v = U64Vec4::new(10, 20, 30, 40);
        assert_eq!(v[0], 10);
        assert_eq!(v[3], 40);
    }

    #[test]
    #[should_panic]
    fn u64vec4_index_oob_panics() {
        let _ = U64Vec4::ZERO[4];
    }

    #[test]
    fn u64vec4_casts() {
        let v = U64Vec4::new(1, 2, 3, 4);
        let _f  = v.as_vec4();
        let _d  = v.as_dvec4();
        let _i  = v.as_ivec4();
        let _u  = v.as_uvec4();
        let _i6 = v.as_i64vec4();
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  FFI round-trips
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn ffi_ci64vec2_roundtrip_and_layout() {
        use crate::ffi::CI64Vec2;
        assert_eq!(std::mem::size_of::<CI64Vec2>(),  16);
        assert_eq!(std::mem::align_of::<CI64Vec2>(),  8);
        let v  = I64Vec2::new(i64::MIN, i64::MAX);
        let cv = CI64Vec2::from(v);
        let v2 = I64Vec2::from(cv);
        assert_eq!(v, v2);
    }

    #[test]
    fn ffi_ci64vec3_roundtrip_and_layout() {
        use crate::ffi::CI64Vec3;
        // 24 bytes, align 8 — no padding (same as I64Vec3)
        assert_eq!(std::mem::size_of::<CI64Vec3>(),  24);
        assert_eq!(std::mem::align_of::<CI64Vec3>(),  8);
        let v  = I64Vec3::new(-1, 0, i64::MAX);
        let cv = CI64Vec3::from(v);
        let v2 = I64Vec3::from(cv);
        assert_eq!(v, v2);
    }

    #[test]
    fn ffi_ci64vec4_roundtrip_and_layout() {
        use crate::ffi::CI64Vec4;
        assert_eq!(std::mem::size_of::<CI64Vec4>(),  32);
        assert_eq!(std::mem::align_of::<CI64Vec4>(),  8);
        let v  = I64Vec4::new(1, -2, 3, -4);
        let cv = CI64Vec4::from(v);
        let v2 = I64Vec4::from(cv);
        assert_eq!(v, v2);
    }

    #[test]
    fn ffi_cu64vec2_roundtrip_and_layout() {
        use crate::ffi::CU64Vec2;
        assert_eq!(std::mem::size_of::<CU64Vec2>(),  16);
        assert_eq!(std::mem::align_of::<CU64Vec2>(),  8);
        let v  = U64Vec2::new(0, u64::MAX);
        let cv = CU64Vec2::from(v);
        let v2 = U64Vec2::from(cv);
        assert_eq!(v, v2);
    }

    #[test]
    fn ffi_cu64vec3_roundtrip_and_layout() {
        use crate::ffi::CU64Vec3;
        assert_eq!(std::mem::size_of::<CU64Vec3>(),  24);
        assert_eq!(std::mem::align_of::<CU64Vec3>(),  8);
        let v  = U64Vec3::new(1, 2, u64::MAX);
        let cv = CU64Vec3::from(v);
        let v2 = U64Vec3::from(cv);
        assert_eq!(v, v2);
    }

    #[test]
    fn ffi_cu64vec4_roundtrip_and_layout() {
        use crate::ffi::CU64Vec4;
        assert_eq!(std::mem::size_of::<CU64Vec4>(),  32);
        assert_eq!(std::mem::align_of::<CU64Vec4>(),  8);
        let v  = U64Vec4::new(u64::MAX, 0, 1, 2);
        let cv = CU64Vec4::from(v);
        let v2 = U64Vec4::from(cv);
        assert_eq!(v, v2);
    }
      }
