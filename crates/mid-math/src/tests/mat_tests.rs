// crates/mid-math/src/tests/mat_tests.rs
#[cfg(test)]
mod tests {
    use crate::{Mat3, Mat4, Vec3, Quat, approx_eq, to_radians};

    // ── Mat3 ─────────────────────────────────────────────────────────────────

    #[test]
    fn mat3_identity_times_vec_is_vec() {
        let v = Vec3::new(1.0,2.0,3.0);
        assert!(Mat3::IDENTITY.transform(v).approx_eq(v));
    }

    #[test]
    fn mat3_multiply_identity_is_identity() {
        assert_eq!(Mat3::IDENTITY * Mat3::IDENTITY, Mat3::IDENTITY);
    }

    #[test]
    fn mat3_inverse_of_identity_is_identity() {
        assert_eq!(Mat3::IDENTITY.inverse().unwrap(), Mat3::IDENTITY);
    }

    #[test]
    fn mat3_inverse_roundtrip() {
        let m   = Mat3::from_cols([2.0,0.0,0.0],[0.0,3.0,0.0],[0.0,0.0,4.0]);
        let inv = m.inverse().expect("diagonal matrix is invertible");
        let p   = m * inv;
        for c in 0..3 { for r in 0..3 {
            let exp = if c==r { 1.0 } else { 0.0 };
            assert!((p.cols[c][r]-exp).abs()<1e-5,
                "m*inv[{}][{}] = {}", c, r, p.cols[c][r]);
        }}
    }

    // ── Mat4 ─────────────────────────────────────────────────────────────────

    #[test]
    fn mat4_size_is_64_bytes() {
        assert_eq!(std::mem::size_of::<Mat4>(), 64);
    }

    #[test]
    fn mat4_identity_transform_point_unchanged() {
        let p = Vec3::new(1.0,2.0,3.0);
        assert!(Mat4::IDENTITY.transform_point(p).approx_eq(p));
    }

    #[test]
    fn mat4_translation_moves_point() {
        let r = Mat4::from_translation(Vec3::new(10.0,20.0,30.0))
                    .transform_point(Vec3::ONE);
        assert!(r.approx_eq(Vec3::new(11.0,21.0,31.0)),
            "got {:?}", r);
    }

    #[test]
    fn mat4_translation_does_not_affect_vectors() {
        let m = Mat4::from_translation(Vec3::new(99.0,99.0,99.0));
        assert!(m.transform_vector(Vec3::X).approx_eq(Vec3::X));
    }

    #[test]
    fn mat4_scale_scales_point() {
        let r = Mat4::from_scale(Vec3::new(2.0,3.0,4.0))
                    .transform_point(Vec3::ONE);
        assert!(r.approx_eq(Vec3::new(2.0,3.0,4.0)), "got {:?}", r);
    }

    #[test]
    fn mat4_multiply_identity_is_identity() {
        assert_eq!(Mat4::IDENTITY * Mat4::IDENTITY, Mat4::IDENTITY);
    }

    #[test]
    fn mat4_inverse_of_identity_is_identity() {
        assert_eq!(Mat4::IDENTITY.inverse().unwrap(), Mat4::IDENTITY);
    }

    #[test]
    fn mat4_inverse_roundtrip() {
        let m = Mat4::from_trs(
            Vec3::new(1.0,2.0,3.0),
            Quat::from_axis_angle(Vec3::Y, to_radians(45.0)),
            Vec3::new(2.0,2.0,2.0),
        );
        let inv = m.inverse().expect("TRS matrix is invertible");
        let eye = m * inv;
        for c in 0..4 { for r in 0..4 {
            let exp = if c==r { 1.0 } else { 0.0 };
            assert!((eye.cols[c][r]-exp).abs()<1e-4,
                "m*inv[{}][{}] = {:.6}", c, r, eye.cols[c][r]);
        }}
    }

    #[test]
    fn mat4_singular_inverse_returns_none() {
        assert!(Mat4::ZERO.inverse().is_none());
    }

    #[test]
    fn mat4_perspective_has_negative_one_at_col3_row2() {
        let m = Mat4::perspective_rh(to_radians(60.0), 16.0/9.0, 0.1, 1000.0);
        assert!(approx_eq(m.cols[2][3], -1.0), "cols[2][3] = {}", m.cols[2][3]);
    }

    #[test]
    fn mat4_look_at_z_axis_points_toward_target() {
        let view = Mat4::look_at_rh(
            Vec3::new(0.0,0.0,5.0), Vec3::ZERO, Vec3::Y);
        let t = view.transform_point(Vec3::ZERO);
        assert!(t.z < 0.0, "target should be on -Z in view space, got z={}", t.z);
    }

    // ── TRS inverse ───────────────────────────────────────────────────────────

    #[test]
    fn mat4_inverse_trs_identity() {
        assert_eq!(Mat4::IDENTITY.inverse_trs(), Mat4::IDENTITY);
    }

    #[test]
    fn mat4_inverse_trs_translation_only() {
        let t   = Vec3::new(5.0, -3.0, 7.0);
        let m   = Mat4::from_translation(t);
        let inv = m.inverse_trs();
        let p   = inv.transform_point(Vec3::ZERO);
        assert!(p.approx_eq(-t), "expected {:?} got {:?}", -t, p);
    }

    #[test]
    fn mat4_inverse_trs_scale_only() {
        let m   = Mat4::from_scale(Vec3::new(2.0,4.0,0.5));
        let inv = m.inverse_trs();
        let p   = inv.transform_point(Vec3::new(2.0,4.0,0.5));
        assert!(p.approx_eq(Vec3::ONE), "expected ONE got {:?}", p);
    }

    #[test]
    fn mat4_inverse_trs_roundtrip_matches_general_inverse() {
        let m = Mat4::from_trs(
            Vec3::new(3.0,-1.0,5.0),
            Quat::from_axis_angle(
                Vec3::new(1.0,1.0,0.0).normalize(), to_radians(37.0)),
            Vec3::new(2.0,0.5,3.0),
        );
        let inv_g = m.inverse().expect("invertible");
        let inv_t = m.inverse_trs();
        for c in 0..4 { for r in 0..4 {
            let diff = (inv_g.cols[c][r] - inv_t.cols[c][r]).abs();
            assert!(diff < 1e-4,
                "col={} row={}: general={:.6} trs={:.6}",
                c, r, inv_g.cols[c][r], inv_t.cols[c][r]);
        }}
    }

    #[test]
    fn mat4_inverse_trs_zero_scale_does_not_panic() {
        let m = Mat4::from_scale(Vec3::new(0.0,1.0,1.0));
        let _ = m.inverse_trs();
    }

    // ── SSE2 correctness ──────────────────────────────────────────────────────

    #[test]
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn mat4_inverse_general_sse2_matches_scalar() {
        let cases: &[Mat4] = &[
            Mat4::IDENTITY,
            Mat4::from_translation(Vec3::new(5.0,-3.0,7.0)),
            Mat4::from_scale(Vec3::new(2.0,0.5,4.0)),
            Mat4::from_rotation(Quat::from_axis_angle(
                Vec3::new(1.0,1.0,0.0).normalize(), to_radians(37.0))),
            Mat4::from_trs(
                Vec3::new(3.0,-1.0,5.0),
                Quat::from_axis_angle(
                    Vec3::new(1.0,1.0,0.0).normalize(), to_radians(37.0)),
                Vec3::new(2.0,0.5,3.0),
            ),
            Mat4::from_trs(
                Vec3::new(-10.0,0.5,3.3),
                Quat::from_axis_angle(Vec3::Z, to_radians(180.0)),
                Vec3::new(0.1,5.0,2.0),
            ),
        ];
        for (i, m) in cases.iter().enumerate() {
            let sse2   = m.inverse();
            let scalar = m.inverse_scalar();
            match (sse2, scalar) {
                (None, None) => {}
                (Some(s2), Some(sc)) => {
                    for c in 0..4 { for r in 0..4 {
                        let d = (s2.cols[c][r] - sc.cols[c][r]).abs();
                        assert!(d < 1e-4,
                            "case {} col={} row={}: sse2={:.6} scalar={:.6}",
                            i, c, r, s2.cols[c][r], sc.cols[c][r]);
                    }}
                }
                _ => panic!("case {}: SSE2 and scalar disagree on singularity", i),
            }
        }
    }

    #[test]
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn mat4_inverse_trs_sse2_matches_scalar() {
        let cases: &[(Vec3, Quat, Vec3)] = &[
            (Vec3::ZERO, Quat::IDENTITY, Vec3::ONE),
            (Vec3::new(1.0,2.0,3.0), Quat::IDENTITY, Vec3::ONE),
            (Vec3::ZERO, Quat::from_axis_angle(Vec3::Y, to_radians(90.0)), Vec3::ONE),
            (Vec3::ZERO, Quat::IDENTITY, Vec3::new(2.0,3.0,4.0)),
            (
                Vec3::new(5.0,-2.0,7.0),
                Quat::from_axis_angle(
                    Vec3::new(1.0,1.0,0.0).normalize(), to_radians(37.0)),
                Vec3::new(2.0,0.5,3.0),
            ),
            (
                Vec3::new(-10.0,0.5,3.3),
                Quat::from_axis_angle(Vec3::Z, to_radians(180.0)),
                Vec3::new(0.1,5.0,2.0),
            ),
        ];
        for (i, &(t,r,s)) in cases.iter().enumerate() {
            let m      = Mat4::from_trs(t, r, s);
            let sse2   = m.inverse_trs();
            let scalar = m.inverse_trs_scalar();
            for c in 0..4 { for row in 0..4 {
                let d = (sse2.cols[c][row] - scalar.cols[c][row]).abs();
                assert!(d < 1e-5,
                    "case {} col={} row={}: sse2={:.7} scalar={:.7}",
                    i, c, row, sse2.cols[c][row], scalar.cols[c][row]);
            }}
        }
    }
}
