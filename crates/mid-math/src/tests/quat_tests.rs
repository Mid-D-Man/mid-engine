// crates/mid-math/src/tests/quat_tests.rs
#[cfg(test)]
mod tests {
    use crate::{Quat, Vec3, Mat4, approx_eq, to_radians, EPSILON};

    #[test]
    fn quat_identity_does_not_rotate_vector() {
        let v = Vec3::new(1.0,2.0,3.0);
        let r = Quat::IDENTITY.rotate(v);
        assert!(r.approx_eq(v), "IDENTITY.rotate = {:?}", r);
    }

    #[test]
    fn quat_90deg_around_y_rotates_x_to_neg_z() {
        let q = Quat::from_axis_angle(Vec3::Y, to_radians(90.0));
        let r = q.rotate(Vec3::X);
        assert!(r.approx_eq(Vec3::NEG_Z), "expected NEG_Z, got {:?}", r);
    }

    #[test]
    fn quat_multiply_composes_rotations() {
        let q = Quat::from_axis_angle(Vec3::Y, to_radians(90.0));
        let r = (q * q).normalize().rotate(Vec3::X);
        assert!(r.approx_eq(Vec3::NEG_X), "expected NEG_X, got {:?}", r);
    }

    #[test]
    fn quat_conjugate_is_inverse_for_unit() {
        let q = Quat::from_axis_angle(
            Vec3::new(1.0,1.0,0.0).normalize(), to_radians(45.0));
        let c = (q * q.conjugate()).normalize();
        assert!(approx_eq(c.w, 1.0) && approx_eq(c.x, 0.0),
            "q*q.conj = ({},{},{},{})", c.x,c.y,c.z,c.w);
    }

    #[test]
    fn quat_euler_round_trip() {
        let cases = [
            (0.3_f32,  0.5_f32, 1.2_f32),
            (0.0,      0.0,     0.0),
            (1.0,     -0.5,     2.0),
            (-0.7,     0.3,    -1.5),
        ];
        for (roll0, pitch0, yaw0) in cases {
            let q = Quat::from_euler(roll0, pitch0, yaw0);
            let (r1, p1, y1) = q.to_euler();
            assert!(approx_eq(roll0, r1),  "roll  {} vs {}", roll0, r1);
            assert!(approx_eq(pitch0, p1), "pitch {} vs {}", pitch0, p1);
            assert!(approx_eq(yaw0, y1),   "yaw   {} vs {}", yaw0, y1);
        }
    }

    #[test]
    fn quat_slerp_midpoint_in_first_quadrant() {
        let a   = Quat::from_axis_angle(Vec3::Y, to_radians(0.0));
        let b   = Quat::from_axis_angle(Vec3::Y, to_radians(90.0));
        let mid = a.slerp(b, 0.5).rotate(Vec3::X);
        assert!(mid.x > 0.0 && mid.z < 0.0,
            "expected first quadrant, got {:?}", mid);
    }

    #[test]
    fn quat_nlerp_stays_normalized() {
        let a = Quat::from_axis_angle(Vec3::Y, to_radians(0.0));
        let b = Quat::from_axis_angle(Vec3::Y, to_radians(90.0));
        let n = a.nlerp(b, 0.3);
        assert!(approx_eq(n.length(), 1.0), "|nlerp| = {}", n.length());
    }

    #[test]
    fn quat_to_mat4_identity_roundtrip() {
        let m = Quat::IDENTITY.to_mat4();
        for c in 0..4 {
            for r in 0..4 {
                let exp = if c == r { 1.0 } else { 0.0 };
                assert!((m.cols[c][r] - exp).abs() < 1e-5,
                    "col={} row={}: {} != {}", c, r, m.cols[c][r], exp);
            }
        }
    }

    #[test]
    fn quat_inverse_undoes_rotation() {
        let q = Quat::from_axis_angle(Vec3::Y, to_radians(45.0));
        let v = Vec3::new(1.0, 0.0, 0.5).normalize();
        let rotated  = q.rotate(v);
        let restored = q.inverse().rotate(rotated);
        assert!(restored.approx_eq(v),
            "restore failed: {:?} vs {:?}", restored, v);
    }

    // ── SSE2 path vs scalar ───────────────────────────────────────────────────
    // The SIMD mul_quat must match the scalar reference.

    #[test]
    fn quat_mul_matches_scalar_reference() {
        let q1 = Quat::from_axis_angle(Vec3::Y, to_radians(30.0));
        let q2 = Quat::from_axis_angle(Vec3::new(1.0,1.0,0.0).normalize(), to_radians(45.0));
        let simd   = q1 * q2;
        // Scalar reference via components
        let (ax,ay,az,aw) = (q1.x,q1.y,q1.z,q1.w);
        let (bx,by,bz,bw) = (q2.x,q2.y,q2.z,q2.w);
        let sx = aw*bx + ax*bw + ay*bz - az*by;
        let sy = aw*by - ax*bz + ay*bw + az*bx;
        let sz = aw*bz + ax*by - ay*bx + az*bw;
        let sw = aw*bw - ax*bx - ay*by - az*bz;
        assert!(approx_eq(simd.x, sx), "x: {} vs {}", simd.x, sx);
        assert!(approx_eq(simd.y, sy), "y: {} vs {}", simd.y, sy);
        assert!(approx_eq(simd.z, sz), "z: {} vs {}", simd.z, sz);
        assert!(approx_eq(simd.w, sw), "w: {} vs {}", simd.w, sw);
    }
}
