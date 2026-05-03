// crates/mid-math/src/tests/f64_tests.rs
//! Correctness tests for all f64 types.
//!
//! Mirrors the structure of mat_tests.rs and quat_tests.rs but uses
//! DEPSILON-based tolerances appropriate for f64 (1e-10 for most checks).

#[cfg(test)]
mod tests {
    use crate::{
        DAffine3, DMat2, DMat3, DMat4, DQuat, DVec2, DVec3, DVec4, DEPSILON,
    };
    use core::f64::consts::{FRAC_PI_2, PI};

    // ── helpers ───────────────────────────────────────────────────────────────

    fn approx(a: f64, b: f64) -> bool { (a - b).abs() < 1e-10 }
    fn approx_loose(a: f64, b: f64) -> bool { (a - b).abs() < 1e-6 }

    fn vec3_approx(a: DVec3, b: DVec3) -> bool {
        approx(a.x, b.x) && approx(a.y, b.y) && approx(a.z, b.z)
    }
    fn vec3_approx_loose(a: DVec3, b: DVec3) -> bool {
        approx_loose(a.x, b.x) && approx_loose(a.y, b.y) && approx_loose(a.z, b.z)
    }

    fn mat4_approx(a: DMat4, b: DMat4) -> bool {
        for c in 0..4 { for r in 0..4 {
            if !approx(a.cols[c][r], b.cols[c][r]) { return false; }
        }}
        true
    }
    fn mat4_approx_loose(a: DMat4, b: DMat4) -> bool {
        for c in 0..4 { for r in 0..4 {
            if !approx_loose(a.cols[c][r], b.cols[c][r]) { return false; }
        }}
        true
    }

    // ── DVec2 ─────────────────────────────────────────────────────────────────

    #[test]
    fn dvec2_size_align() {
        assert_eq!(std::mem::size_of::<DVec2>(),  16);
        assert_eq!(std::mem::align_of::<DVec2>(), 16);
    }

    #[test]
    fn dvec2_add_sub() {
        let a = DVec2::new(1.0, 2.0);
        let b = DVec2::new(3.0, 4.0);
        assert_eq!(a + b, DVec2::new(4.0, 6.0));
        assert_eq!(b - a, DVec2::new(2.0, 2.0));
    }

    #[test]
    fn dvec2_dot() {
        let a = DVec2::new(1.0, 0.0);
        let b = DVec2::new(0.0, 1.0);
        assert!(approx(a.dot(b), 0.0));
        assert!(approx(a.dot(a), 1.0));
    }

    #[test]
    fn dvec2_normalize_unit_length() {
        let v = DVec2::new(3.0, 4.0).normalize();
        assert!(approx(v.length(), 1.0), "|n| = {}", v.length());
    }

    #[test]
    fn dvec2_zero_normalize_returns_zero() {
        let v = DVec2::ZERO.normalize();
        assert_eq!(v, DVec2::ZERO);
    }

    #[test]
    fn dvec2_perp_is_orthogonal() {
        let v = DVec2::new(1.0, 0.5).normalize();
        let p = v.perp();
        assert!(approx(v.dot(p), 0.0), "v·perp(v) = {}", v.dot(p));
    }

    #[test]
    fn dvec2_lerp_midpoint() {
        let m = DVec2::ZERO.lerp(DVec2::new(2.0, 4.0), 0.5);
        assert_eq!(m, DVec2::new(1.0, 2.0));
    }

    #[test]
    fn dvec2_angle_roundtrip() {
        for deg in [0.0f64, 30.0, 45.0, 90.0, 135.0, 180.0] {
            let angle = deg.to_radians();
            let v = DVec2::from_angle(angle);
            assert!(approx(v.to_angle(), angle), "deg={} angle={}", deg, v.to_angle());
        }
    }

    // ── DVec3 ─────────────────────────────────────────────────────────────────

    #[test]
    fn dvec3_size_align() {
        assert_eq!(std::mem::size_of::<DVec3>(),  32);
        assert_eq!(std::mem::align_of::<DVec3>(), 32);
    }

    #[test]
    fn dvec3_cross_basis() {
        let z = DVec3::X.cross(DVec3::Y);
        assert!(vec3_approx(z, DVec3::Z), "X×Y = {:?}", z);
    }

    #[test]
    fn dvec3_cross_anticommutative() {
        let a = DVec3::new(1.0, 2.0, 3.0);
        let b = DVec3::new(4.0, 5.0, 6.0);
        let sum = a.cross(b) + b.cross(a);
        assert!(vec3_approx(sum, DVec3::ZERO), "a×b + b×a = {:?}", sum);
    }

    #[test]
    fn dvec3_normalize_unit_length() {
        let n = DVec3::new(1.0, 2.0, 3.0).normalize();
        assert!(approx(n.length(), 1.0), "|n| = {}", n.length());
    }

    #[test]
    fn dvec3_reflect() {
        // Vector hitting Y-plane should flip Y component
        let v = DVec3::new(1.0, -1.0, 0.0).normalize();
        let r = v.reflect(DVec3::Y);
        assert!(approx(r.y, -v.y), "reflect y: {} vs {}", r.y, -v.y);
        assert!(approx(r.x, v.x));
    }

    #[test]
    fn dvec3_angle_between_orthogonal() {
        let angle = DVec3::X.angle_between(DVec3::Y);
        assert!(approx(angle, FRAC_PI_2), "angle = {}", angle);
    }

    #[test]
    fn dvec3_angle_between_parallel() {
        let angle = DVec3::X.angle_between(DVec3::X);
        assert!(approx(angle, 0.0), "angle = {}", angle);
    }

    #[test]
    fn dvec3_equality_ignores_pad() {
        let a = DVec3::new(1.0, 2.0, 3.0);
        let b = DVec3::new(1.0, 2.0, 3.0);
        assert_eq!(a, b);
    }

    // ── DVec4 ─────────────────────────────────────────────────────────────────

    #[test]
    fn dvec4_size_align() {
        assert_eq!(std::mem::size_of::<DVec4>(),  32);
        assert_eq!(std::mem::align_of::<DVec4>(), 32);
    }

    #[test]
    fn dvec4_dot_with_self_is_length_sq() {
        let v = DVec4::new(1.0, 2.0, 3.0, 4.0);
        assert!(approx(v.dot(v), v.length_sq()));
    }

    #[test]
    fn dvec4_normalize_unit_length() {
        let n = DVec4::new(1.0, 2.0, 3.0, 4.0).normalize();
        assert!(approx(n.length(), 1.0), "|n| = {}", n.length());
    }

    #[test]
    fn dvec3_extend_truncate_roundtrip() {
        let v3 = DVec3::new(1.0, 2.0, 3.0);
        let v4 = v3.extend(4.0);
        let back = v4.truncate();
        assert!(vec3_approx(back, v3));
    }

    // ── DQuat ─────────────────────────────────────────────────────────────────

    #[test]
    fn dquat_identity_does_not_rotate() {
        let v = DVec3::new(1.0, 2.0, 3.0);
        let r = DQuat::IDENTITY.rotate(v);
        assert!(vec3_approx(r, v), "IDENTITY.rotate = {:?}", r);
    }

    #[test]
    fn dquat_90deg_y_rotates_x_to_neg_z() {
        let q = DQuat::from_axis_angle(DVec3::Y, FRAC_PI_2);
        let r = q.rotate(DVec3::X);
        assert!(vec3_approx_loose(r, DVec3::NEG_Z), "expected NEG_Z, got {:?}", r);
    }

    #[test]
    fn dquat_180deg_y_rotates_x_to_neg_x() {
        let q = DQuat::from_axis_angle(DVec3::Y, PI);
        let r = q.rotate(DVec3::X);
        assert!(vec3_approx_loose(r, DVec3::NEG_X), "expected NEG_X, got {:?}", r);
    }

    #[test]
    fn dquat_compose_90deg_twice_is_180deg() {
        let q = DQuat::from_axis_angle(DVec3::Y, FRAC_PI_2);
        let r = (q * q).normalize().rotate(DVec3::X);
        assert!(vec3_approx_loose(r, DVec3::NEG_X), "expected NEG_X, got {:?}", r);
    }

    #[test]
    fn dquat_conjugate_inverse_for_unit() {
        let q = DQuat::from_axis_angle(
            DVec3::new(1.0, 1.0, 0.0).normalize(), PI / 4.0,
        );
        let c = (q * q.conjugate()).normalize();
        assert!(approx(c.w, 1.0) && approx(c.x, 0.0),
            "q*q.conj = ({},{},{},{})", c.x, c.y, c.z, c.w);
    }

    #[test]
    fn dquat_euler_roundtrip() {
        let cases = [
            (0.3f64, 0.5f64, 1.2f64),
            (0.0,    0.0,    0.0),
            (1.0,   -0.5,    2.0),
        ];
        for (roll0, pitch0, yaw0) in cases {
            let q = DQuat::from_euler(roll0, pitch0, yaw0);
            let (r1, p1, y1) = q.to_euler();
            assert!(approx_loose(roll0, r1),  "roll  {} vs {}",  roll0, r1);
            assert!(approx_loose(pitch0, p1), "pitch {} vs {}", pitch0, p1);
            assert!(approx_loose(yaw0, y1),   "yaw   {} vs {}",   yaw0, y1);
        }
    }

    #[test]
    fn dquat_nlerp_stays_normalized() {
        let a = DQuat::from_axis_angle(DVec3::Y, 0.0);
        let b = DQuat::from_axis_angle(DVec3::Y, FRAC_PI_2);
        let n = a.nlerp(b, 0.3);
        assert!(approx(n.length(), 1.0), "|nlerp| = {}", n.length());
    }

    #[test]
    fn dquat_slerp_midpoint_in_first_quadrant() {
        let a   = DQuat::from_axis_angle(DVec3::Y, 0.0);
        let b   = DQuat::from_axis_angle(DVec3::Y, FRAC_PI_2);
        let mid = a.slerp(b, 0.5).rotate(DVec3::X);
        assert!(mid.x > 0.0 && mid.z < 0.0,
            "expected 1st quadrant, got {:?}", mid);
    }

    #[test]
    fn dquat_to_mat4_identity_roundtrip() {
        let m = DQuat::IDENTITY.to_mat4();
        for c in 0..4 { for r in 0..4 {
            let exp = if c == r { 1.0 } else { 0.0 };
            assert!((m.cols[c][r] - exp).abs() < 1e-10,
                "col={} row={}: {} != {}", c, r, m.cols[c][r], exp);
        }}
    }

    #[test]
    fn dquat_mul_matches_scalar_reference() {
        let q1 = DQuat::from_axis_angle(DVec3::Y, 30.0f64.to_radians());
        let q2 = DQuat::from_axis_angle(
            DVec3::new(1.0, 1.0, 0.0).normalize(), 45.0f64.to_radians(),
        );
        let result = q1 * q2;
        // Scalar reference
        let (ax, ay, az, aw) = (q1.x, q1.y, q1.z, q1.w);
        let (bx, by, bz, bw) = (q2.x, q2.y, q2.z, q2.w);
        let sx = aw*bx + ax*bw + ay*bz - az*by;
        let sy = aw*by - ax*bz + ay*bw + az*bx;
        let sz = aw*bz + ax*by - ay*bx + az*bw;
        let sw = aw*bw - ax*bx - ay*by - az*bz;
        assert!(approx(result.x, sx));
        assert!(approx(result.y, sy));
        assert!(approx(result.z, sz));
        assert!(approx(result.w, sw));
    }

    // ── DMat2 ─────────────────────────────────────────────────────────────────

    #[test]
    fn dmat2_identity_times_vec_is_vec() {
        let v = DVec2::new(3.0, 7.0);
        let r = DMat2::IDENTITY.mul_vec2(v);
        assert_eq!(r, v);
    }

    #[test]
    fn dmat2_inverse_identity_is_identity() {
        let inv = DMat2::IDENTITY.inverse().unwrap();
        assert_eq!(inv, DMat2::IDENTITY);
    }

    #[test]
    fn dmat2_rotation_inverse_roundtrip() {
        let m   = DMat2::from_angle(0.7);
        let inv = m.inverse().expect("rotation matrix is invertible");
        let p   = m * inv;
        assert!(approx(p.x_axis.x, 1.0));
        assert!(approx(p.x_axis.y, 0.0));
        assert!(approx(p.y_axis.x, 0.0));
        assert!(approx(p.y_axis.y, 1.0));
    }

    #[test]
    fn dmat2_singular_inverse_returns_none() {
        let m = DMat2::ZERO;
        assert!(m.inverse().is_none());
    }

    // ── DMat3 ─────────────────────────────────────────────────────────────────

    #[test]
    fn dmat3_identity_times_vec_is_vec() {
        let v = DVec3::new(1.0, 2.0, 3.0);
        assert!(vec3_approx(DMat3::IDENTITY.transform(v), v));
    }

    #[test]
    fn dmat3_inverse_roundtrip() {
        let m = DMat3::from_cols(
            [2.0, 0.5, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 4.0],
        );
        let inv = m.inverse().expect("diagonal matrix is invertible");
        let p = m * inv;
        for c in 0..3 { for r in 0..3 {
            let exp = if c == r { 1.0 } else { 0.0 };
            assert!((p.cols[c][r] - exp).abs() < 1e-10,
                "m*inv[{}][{}] = {}", c, r, p.cols[c][r]);
        }}
    }

    #[test]
    fn dmat3_singular_inverse_returns_none() {
        assert!(DMat3::ZERO.inverse().is_none());
    }

    // ── DMat4 ─────────────────────────────────────────────────────────────────

    #[test]
    fn dmat4_size_align() {
        assert_eq!(std::mem::size_of::<DMat4>(),  128);
        assert_eq!(std::mem::align_of::<DMat4>(),  32);
    }

    #[test]
    fn dmat4_identity_transform_point_unchanged() {
        let p = DVec3::new(1.0, 2.0, 3.0);
        assert!(vec3_approx(DMat4::IDENTITY.transform_point(p), p));
    }

    #[test]
    fn dmat4_translation_moves_point() {
        let r = DMat4::from_translation(DVec3::new(10.0, 20.0, 30.0))
                    .transform_point(DVec3::ONE);
        assert!(vec3_approx(r, DVec3::new(11.0, 21.0, 31.0)), "got {:?}", r);
    }

    #[test]
    fn dmat4_translation_does_not_affect_vectors() {
        let m = DMat4::from_translation(DVec3::new(99.0, 99.0, 99.0));
        assert!(vec3_approx(m.transform_vector(DVec3::X), DVec3::X));
    }

    #[test]
    fn dmat4_scale_scales_point() {
        let r = DMat4::from_scale(DVec3::new(2.0, 3.0, 4.0))
                    .transform_point(DVec3::ONE);
        assert!(vec3_approx(r, DVec3::new(2.0, 3.0, 4.0)), "got {:?}", r);
    }

    #[test]
    fn dmat4_multiply_identity_is_identity() {
        assert_eq!(DMat4::IDENTITY * DMat4::IDENTITY, DMat4::IDENTITY);
    }

    #[test]
    fn dmat4_inverse_of_identity_is_identity() {
        assert_eq!(DMat4::IDENTITY.inverse().unwrap(), DMat4::IDENTITY);
    }

    #[test]
    fn dmat4_inverse_roundtrip() {
        let m = DMat4::from_trs(
            DVec3::new(1.0, 2.0, 3.0),
            DQuat::from_axis_angle(DVec3::Y, 45.0f64.to_radians()),
            DVec3::new(2.0, 2.0, 2.0),
        );
        let inv = m.inverse().expect("TRS matrix is invertible");
        let eye = m * inv;
        for c in 0..4 { for r in 0..4 {
            let exp = if c == r { 1.0 } else { 0.0 };
            assert!((eye.cols[c][r] - exp).abs() < 1e-9,
                "m*inv[{}][{}] = {:.10}", c, r, eye.cols[c][r]);
        }}
    }

    #[test]
    fn dmat4_singular_inverse_returns_none() {
        assert!(DMat4::ZERO.inverse().is_none());
    }

    #[test]
    fn dmat4_inverse_trs_identity() {
        assert!(mat4_approx(DMat4::IDENTITY.inverse_trs(), DMat4::IDENTITY));
    }

    #[test]
    fn dmat4_inverse_trs_translation_only() {
        let t   = DVec3::new(5.0, -3.0, 7.0);
        let m   = DMat4::from_translation(t);
        let inv = m.inverse_trs();
        let p   = inv.transform_point(DVec3::ZERO);
        assert!(vec3_approx(p, -t), "expected {:?} got {:?}", -t, p);
    }

    #[test]
    fn dmat4_inverse_trs_roundtrip_matches_general_inverse() {
        let m = DMat4::from_trs(
            DVec3::new(3.0, -1.0, 5.0),
            DQuat::from_axis_angle(
                DVec3::new(1.0, 1.0, 0.0).normalize(), 37.0f64.to_radians()),
            DVec3::new(2.0, 0.5, 3.0),
        );
        let inv_g = m.inverse().expect("invertible");
        let inv_t = m.inverse_trs();
        for c in 0..4 { for r in 0..4 {
            let diff = (inv_g.cols[c][r] - inv_t.cols[c][r]).abs();
            assert!(diff < 1e-8,
                "col={} row={}: general={:.10} trs={:.10}",
                c, r, inv_g.cols[c][r], inv_t.cols[c][r]);
        }}
    }

    #[test]
    fn dmat4_perspective_has_neg_one_at_col2_row3() {
        let m = DMat4::perspective_rh(60.0f64.to_radians(), 16.0/9.0, 0.1, 1000.0);
        assert!(approx(m.cols[2][3], -1.0), "cols[2][3] = {}", m.cols[2][3]);
    }

    #[test]
    fn dmat4_look_at_target_on_neg_z() {
        let view = DMat4::look_at_rh(
            DVec3::new(0.0, 0.0, 5.0), DVec3::ZERO, DVec3::Y);
        let t = view.transform_point(DVec3::ZERO);
        assert!(t.z < 0.0, "target should be on -Z in view space, got z={}", t.z);
    }

    // ── DAffine3 ──────────────────────────────────────────────────────────────

    #[test]
    fn daffine3_size_align() {
        assert_eq!(std::mem::size_of::<DAffine3>(),  128);
        assert_eq!(std::mem::align_of::<DAffine3>(),  32);
    }

    #[test]
    fn daffine3_identity_is_default() {
        assert_eq!(DAffine3::default(), DAffine3::IDENTITY);
    }

    #[test]
    fn daffine3_identity_transform_point_unchanged() {
        let p = DVec3::new(1.0, 2.0, 3.0);
        assert!(vec3_approx(DAffine3::IDENTITY.transform_point(p), p));
    }

    #[test]
    fn daffine3_translation_moves_point() {
        let t = DVec3::new(10.0, 20.0, 30.0);
        let a = DAffine3::from_translation(t);
        let r = a.transform_point(DVec3::ONE);
        assert!(vec3_approx(r, DVec3::new(11.0, 21.0, 31.0)), "got {:?}", r);
    }

    #[test]
    fn daffine3_translation_does_not_affect_vectors() {
        let a = DAffine3::from_translation(DVec3::new(99.0, 99.0, 99.0));
        assert!(vec3_approx(a.transform_vector(DVec3::X), DVec3::X));
    }

    #[test]
    fn daffine3_scale_scales_point() {
        let a = DAffine3::from_scale(DVec3::new(2.0, 3.0, 4.0));
        let r = a.transform_point(DVec3::ONE);
        assert!(vec3_approx(r, DVec3::new(2.0, 3.0, 4.0)), "got {:?}", r);
    }

    #[test]
    fn daffine3_from_trs_matches_dmat4_from_trs() {
        let t = DVec3::new(1.0, 2.0, 3.0);
        let r = DQuat::from_axis_angle(DVec3::Y, 45.0f64.to_radians());
        let s = DVec3::new(2.0, 0.5, 3.0);

        let a = DAffine3::from_trs(t, r, s);
        let m = DMat4::from_trs(t, r, s);
        let p = DVec3::new(1.5, -0.5, 2.0);

        let pa = a.transform_point(p);
        let pm = m.transform_point(p);
        assert!(vec3_approx_loose(pa, pm),
            "DAffine3 {:?} vs DMat4 {:?}", pa, pm);
    }

    #[test]
    fn daffine3_inverse_identity() {
        let inv = DAffine3::IDENTITY.inverse();
        assert_eq!(inv, DAffine3::IDENTITY);
    }

    #[test]
    fn daffine3_inverse_translation_only() {
        let t   = DVec3::new(5.0, -3.0, 7.0);
        let a   = DAffine3::from_translation(t);
        let inv = a.inverse();
        let p   = inv.transform_point(DVec3::ZERO);
        assert!(vec3_approx(p, -t), "expected {:?} got {:?}", -t, p);
    }

    #[test]
    fn daffine3_inverse_scale_only() {
        let a   = DAffine3::from_scale(DVec3::new(2.0, 4.0, 0.5));
        let inv = a.inverse();
        let p   = inv.transform_point(DVec3::new(2.0, 4.0, 0.5));
        assert!(vec3_approx(p, DVec3::ONE), "expected ONE got {:?}", p);
    }

    #[test]
    fn daffine3_inverse_trs_roundtrip() {
        let a = DAffine3::from_trs(
            DVec3::new(3.0, -1.0, 5.0),
            DQuat::from_axis_angle(
                DVec3::new(1.0, 1.0, 0.0).normalize(), 37.0f64.to_radians()),
            DVec3::new(2.0, 0.5, 3.0),
        );
        let composed = a * a.inverse();
        let p = DVec3::new(7.0, -2.0, 4.0);
        let result = composed.transform_point(p);
        assert!(vec3_approx_loose(result, p),
            "a * a.inverse() should be identity, got {:?}", result);
    }

    #[test]
    fn daffine3_inverse_matches_dmat4_inverse_trs() {
        let t = DVec3::new(3.0, -1.0, 5.0);
        let r = DQuat::from_axis_angle(
            DVec3::new(1.0, 1.0, 0.0).normalize(), 37.0f64.to_radians());
        let s = DVec3::new(2.0, 0.5, 3.0);

        let a    = DAffine3::from_trs(t, r, s);
        let m    = DMat4::from_trs(t, r, s);
        let ainv = a.inverse().to_mat4();
        let minv = m.inverse_trs();

        for c in 0..4 { for row in 0..4 {
            let d = (ainv.cols[c][row] - minv.cols[c][row]).abs();
            assert!(d < 1e-8,
                "col={} row={}: affine3={:.10} mat4_trs={:.10}",
                c, row, ainv.cols[c][row], minv.cols[c][row]);
        }}
    }

    #[test]
    fn daffine3_compose_matches_dmat4_mul() {
        let a = DAffine3::from_trs(
            DVec3::new(1.0, 0.0, 0.0),
            DQuat::from_axis_angle(DVec3::Y, 45.0f64.to_radians()),
            DVec3::new(2.0, 2.0, 2.0),
        );
        let b = DAffine3::from_trs(
            DVec3::new(0.0, 1.0, 0.0),
            DQuat::from_axis_angle(DVec3::X, 30.0f64.to_radians()),
            DVec3::ONE,
        );
        let p = DVec3::new(1.0, 2.0, 3.0);

        let r_affine = (a * b).transform_point(p);
        let r_mat4   = (a.to_mat4() * b.to_mat4()).transform_point(p);

        assert!(vec3_approx_loose(r_affine, r_mat4),
            "DAffine3 {:?} vs DMat4 {:?}", r_affine, r_mat4);
    }

    #[test]
    fn daffine3_inverse_zero_scale_does_not_panic() {
        let a = DAffine3::from_scale(DVec3::new(0.0, 1.0, 1.0));
        let _ = a.inverse();
    }

    #[test]
    fn daffine3_from_mat4_roundtrip() {
        let m = DMat4::from_trs(
            DVec3::new(3.0, -1.0, 5.0),
            DQuat::from_axis_angle(DVec3::new(1.0,1.0,0.0).normalize(), 37.0f64.to_radians()),
            DVec3::new(2.0, 0.5, 3.0),
        );
        let a  = DAffine3::from_mat4(m);
        let m2 = a.to_mat4();
        for c in 0..4 { for row in 0..4 {
            let d = (m.cols[c][row] - m2.cols[c][row]).abs();
            assert!(d < 1e-10,
                "col={} row={}: {:.12} vs {:.12}", c, row, m.cols[c][row], m2.cols[c][row]);
        }}
    }

    // ── FFI round-trips ───────────────────────────────────────────────────────
    // Verify that converting Rust→C→Rust preserves values exactly.

    #[test]
    fn ffi_dvec3_roundtrip() {
        use crate::ffi::types::{CDVec3};
        let v  = DVec3::new(1.5, -2.5, 3.5);
        let cv = CDVec3::from(v);
        let v2 = DVec3::from(cv);
        assert!(vec3_approx(v, v2));
        // _pad must be zero in C type
        assert_eq!(cv._pad, 0.0);
    }

    #[test]
    fn ffi_dquat_roundtrip() {
        use crate::ffi::types::CDQuat;
        let q  = DQuat::from_axis_angle(DVec3::Y, 1.0);
        let cq = CDQuat::from(q);
        let q2 = DQuat::from(cq);
        assert!(approx(q.x, q2.x) && approx(q.y, q2.y)
             && approx(q.z, q2.z) && approx(q.w, q2.w));
    }

    #[test]
    fn ffi_dmat4_roundtrip() {
        use crate::ffi::types::CDMat4;
        let m  = DMat4::from_trs(
            DVec3::new(1.0, 2.0, 3.0),
            DQuat::from_axis_angle(DVec3::Y, 0.5),
            DVec3::new(1.5, 1.5, 1.5),
        );
        let cm = CDMat4::from(m);
        let m2 = DMat4::from(cm);
        assert!(mat4_approx(m, m2));
    }

    #[test]
    fn ffi_daffine3_roundtrip() {
        use crate::ffi::types::CDAffine3;
        let a  = DAffine3::from_trs(
            DVec3::new(1.0, 2.0, 3.0),
            DQuat::from_axis_angle(DVec3::Y, 0.5),
            DVec3::new(2.0, 2.0, 2.0),
        );
        let ca = CDAffine3::from(a);
        let a2 = DAffine3::from(ca);
        assert_eq!(a, a2);
    }
}
