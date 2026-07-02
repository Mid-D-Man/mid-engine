// crates/mid-math/src/tests/mat_tests.rs
#[cfg(test)]
mod tests {
    use crate::{Affine3, Mat3, Mat4, Quat, Vec3, approx_eq, to_radians};

    /// `Mat4` stores columns as four named `Vec4` fields (x_axis/y_axis/
    /// z_axis/w_axis), not an indexable `cols` array like `Mat3` does.
    /// This helper gives tests the `m[c][r]` access pattern they want
    /// without adding an indexing API to the hot-path Mat4 type itself.
    fn m4(m: &Mat4, c: usize, r: usize) -> f32 {
        let col = match c {
            0 => m.x_axis, 1 => m.y_axis, 2 => m.z_axis, 3 => m.w_axis,
            _ => panic!("column index {c} out of range"),
        };
        match r {
            0 => col.x, 1 => col.y, 2 => col.z, 3 => col.w,
            _ => panic!("row index {r} out of range"),
        }
    }

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
        let m   = Mat3::from_cols(
            Vec3::new(2.0,0.0,0.0), Vec3::new(0.0,3.0,0.0), Vec3::new(0.0,0.0,4.0));
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
        assert!(r.approx_eq(Vec3::new(11.0,21.0,31.0)), "got {:?}", r);
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
            assert!((m4(&eye,c,r)-exp).abs()<1e-4,
                "m*inv[{}][{}] = {:.6}", c, r, m4(&eye,c,r));
        }}
    }

    #[test]
    fn mat4_singular_inverse_returns_none() {
        assert!(Mat4::ZERO.inverse().is_none());
    }

    #[test]
    fn mat4_perspective_has_negative_one_at_col3_row2() {
        let m = Mat4::perspective_rh(to_radians(60.0), 16.0/9.0, 0.1, 1000.0);
        assert!(approx_eq(m4(&m,2,3), -1.0), "cols[2][3] = {}", m4(&m,2,3));
    }

    #[test]
    fn mat4_look_at_z_axis_points_toward_target() {
        let view = Mat4::look_at_rh(
            Vec3::new(0.0,0.0,5.0), Vec3::ZERO, Vec3::Y);
        let t = view.transform_point(Vec3::ZERO);
        assert!(t.z < 0.0, "target should be on -Z in view space, got z={}", t.z);
    }

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
            let diff = (m4(&inv_g,c,r) - m4(&inv_t,c,r)).abs();
            assert!(diff < 1e-4,
                "col={} row={}: general={:.6} trs={:.6}",
                c, r, m4(&inv_g,c,r), m4(&inv_t,c,r));
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
        ];
        for (i, m) in cases.iter().enumerate() {
            let sse2   = m.inverse();
            let scalar = m.inverse_scalar();
            match (sse2, scalar) {
                (None, None) => {}
                (Some(s2), Some(sc)) => {
                    for c in 0..4 { for r in 0..4 {
                        let d = (m4(&s2,c,r) - m4(&sc,c,r)).abs();
                        assert!(d < 1e-4,
                            "case {} col={} row={}: sse2={:.6} scalar={:.6}",
                            i, c, r, m4(&s2,c,r), m4(&sc,c,r));
                    }}
                }
                _ => panic!("case {}: SSE2 and scalar disagree on singularity", i),
            }
        }
    }

    // ── Affine3 ───────────────────────────────────────────────────────────────

    #[test]
    fn affine3_size_is_64_bytes() {
        assert_eq!(std::mem::size_of::<Affine3>(), 64);
    }

    #[test]
    fn affine3_identity_is_default() {
        assert_eq!(Affine3::default(), Affine3::IDENTITY);
    }

    #[test]
    fn affine3_identity_transform_point_unchanged() {
        let p = Vec3::new(1.0, 2.0, 3.0);
        assert!(Affine3::IDENTITY.transform_point(p).approx_eq(p));
    }

    #[test]
    fn affine3_identity_transform_vector_unchanged() {
        let v = Vec3::new(1.0, 0.0, 0.0);
        assert!(Affine3::IDENTITY.transform_vector(v).approx_eq(v));
    }

    #[test]
    fn affine3_translation_moves_point() {
        let t = Vec3::new(10.0, 20.0, 30.0);
        let a = Affine3::from_translation(t);
        let r = a.transform_point(Vec3::ONE);
        assert!(r.approx_eq(Vec3::new(11.0, 21.0, 31.0)), "got {:?}", r);
    }

    #[test]
    fn affine3_translation_does_not_affect_vectors() {
        let a = Affine3::from_translation(Vec3::new(99.0, 99.0, 99.0));
        assert!(a.transform_vector(Vec3::X).approx_eq(Vec3::X));
    }

    #[test]
    fn affine3_scale_scales_point() {
        let a = Affine3::from_scale(Vec3::new(2.0, 3.0, 4.0));
        let r = a.transform_point(Vec3::ONE);
        assert!(r.approx_eq(Vec3::new(2.0, 3.0, 4.0)), "got {:?}", r);
    }

    #[test]
    fn affine3_from_trs_matches_mat4_from_trs() {
        let t = Vec3::new(1.0, 2.0, 3.0);
        let r = Quat::from_axis_angle(Vec3::Y, to_radians(45.0));
        let s = Vec3::new(2.0, 0.5, 3.0);

        let a = Affine3::from_trs(t, r, s);
        let m = Mat4::from_trs(t, r, s);
        let p = Vec3::new(1.5, -0.5, 2.0);

        let pa = a.transform_point(p);
        let pm = m.transform_point(p);
        assert!(pa.approx_eq(pm),
            "Affine3 {:?} vs Mat4 {:?}", pa, pm);
    }

    #[test]
    fn affine3_from_mat4_roundtrip() {
        let m = Mat4::from_trs(
            Vec3::new(3.0, -1.0, 5.0),
            Quat::from_axis_angle(Vec3::new(1.0,1.0,0.0).normalize(), to_radians(37.0)),
            Vec3::new(2.0, 0.5, 3.0),
        );
        let a = Affine3::from_mat4(m);
        let m2 = a.to_mat4();
        for c in 0..4 { for row in 0..4 {
            let d = (m4(&m,c,row) - m4(&m2,c,row)).abs();
            assert!(d < 1e-5, "col={} row={}: {:.7} vs {:.7}", c, row, m4(&m,c,row), m4(&m2,c,row));
        }}
    }

    #[test]
    fn affine3_inverse_identity() {
        let inv = Affine3::IDENTITY.inverse();
        assert_eq!(inv, Affine3::IDENTITY);
    }

    #[test]
    fn affine3_inverse_translation_only() {
        let t   = Vec3::new(5.0, -3.0, 7.0);
        let a   = Affine3::from_translation(t);
        let inv = a.inverse();
        let p   = inv.transform_point(Vec3::ZERO);
        assert!(p.approx_eq(-t), "expected {:?} got {:?}", -t, p);
    }

    #[test]
    fn affine3_inverse_scale_only() {
        let a   = Affine3::from_scale(Vec3::new(2.0, 4.0, 0.5));
        let inv = a.inverse();
        let p   = inv.transform_point(Vec3::new(2.0, 4.0, 0.5));
        assert!(p.approx_eq(Vec3::ONE), "expected ONE got {:?}", p);
    }

    #[test]
    fn affine3_inverse_trs_roundtrip() {
        let a = Affine3::from_trs(
            Vec3::new(3.0, -1.0, 5.0),
            Quat::from_axis_angle(
                Vec3::new(1.0, 1.0, 0.0).normalize(), to_radians(37.0)),
            Vec3::new(2.0, 0.5, 3.0),
        );
        let composed = a * a.inverse();
        let p = Vec3::new(7.0, -2.0, 4.0);
        let result = composed.transform_point(p);
        assert!(result.approx_eq(p),
            "a * a.inverse() should be identity, got {:?} for {:?}", result, p);
    }

    #[test]
    fn affine3_inverse_matches_mat4_inverse_trs() {
        let t = Vec3::new(3.0, -1.0, 5.0);
        let r = Quat::from_axis_angle(
            Vec3::new(1.0, 1.0, 0.0).normalize(), to_radians(37.0));
        let s = Vec3::new(2.0, 0.5, 3.0);

        let a    = Affine3::from_trs(t, r, s);
        let m    = Mat4::from_trs(t, r, s);
        let ainv = a.inverse().to_mat4();
        let minv = m.inverse_trs();

        for c in 0..4 { for row in 0..4 {
            let d = (m4(&ainv,c,row) - m4(&minv,c,row)).abs();
            assert!(d < 1e-4,
                "col={} row={}: affine3={:.6} mat4_trs={:.6}",
                c, row, m4(&ainv,c,row), m4(&minv,c,row));
        }}
    }

    #[test]
    fn affine3_compose_matches_mat4_mul() {
        let a = Affine3::from_trs(
            Vec3::new(1.0, 0.0, 0.0),
            Quat::from_axis_angle(Vec3::Y, to_radians(45.0)),
            Vec3::new(2.0, 2.0, 2.0),
        );
        let b = Affine3::from_trs(
            Vec3::new(0.0, 1.0, 0.0),
            Quat::from_axis_angle(Vec3::X, to_radians(30.0)),
            Vec3::new(1.0, 1.0, 1.0),
        );
        let p = Vec3::new(1.0, 2.0, 3.0);

        let r_affine = (a * b).transform_point(p);
        let r_mat4   = (a.to_mat4() * b.to_mat4()).transform_point(p);

        assert!(r_affine.approx_eq(r_mat4),
            "compose: Affine3 {:?} vs Mat4 {:?}", r_affine, r_mat4);
    }

    #[test]
    fn affine3_inverse_zero_scale_does_not_panic() {
        let a = Affine3::from_scale(Vec3::new(0.0, 1.0, 1.0));
        let _ = a.inverse();
    }
        }
