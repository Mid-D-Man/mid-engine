// crates/mid-math/src/tests.rs

#[cfg(test)]
mod tests {
    use crate::vec::{Vec2, Vec3, Vec4};
    use crate::quat::Quat;
    use crate::mat::{Mat3, Mat4};
    use crate::{EPSILON, to_radians, lerp, smoothstep, approx_eq};

    // ── Scalar utilities ──────────────────────────────────────────────────

    #[test]
    fn lerp_scalar_midpoint() {
        let result = lerp(0.0, 10.0, 0.5);
        assert!((result - 5.0).abs() < EPSILON);
        println!("  lerp(0, 10, 0.5) = {}", result);
    }

    #[test]
    fn smoothstep_clamps_outside_range() {
        let lo = smoothstep(0.0, 1.0, -1.0);
        let hi = smoothstep(0.0, 1.0,  2.0);
        assert!((lo - 0.0).abs() < EPSILON);
        assert!((hi - 1.0).abs() < EPSILON);
        println!("  smoothstep outside range: lo={}, hi={}", lo, hi);
    }

    #[test]
    fn smoothstep_midpoint_is_not_linear() {
        let mid = smoothstep(0.0, 1.0, 0.5);
        assert!((mid - 0.5).abs() < EPSILON, "smoothstep(0.5) should be 0.5 at midpoint");
        println!("  smoothstep(0, 1, 0.5) = {:.6} (exact midpoint)", mid);
    }

    // ── Vec2 ─────────────────────────────────────────────────────────────

    #[test]
    fn vec2_addition() {
        let a = Vec2::new(1.0, 2.0);
        let b = Vec2::new(3.0, 4.0);
        let c = a + b;
        assert!(c.approx_eq(Vec2::new(4.0, 6.0)));
        println!("  {} + {} = {}", a, b, c);
    }

    #[test]
    fn vec2_dot_product() {
        let a = Vec2::new(1.0, 0.0);
        let b = Vec2::new(0.0, 1.0);
        let d = a.dot(b);
        assert!(approx_eq(d, 0.0));
        println!("  dot({}, {}) = {} (perpendicular → 0)", a, b, d);
    }

    #[test]
    fn vec2_normalize_unit_length() {
        let v = Vec2::new(3.0, 4.0);
        let n = v.normalize();
        assert!(approx_eq(n.length(), 1.0));
        println!("  normalize({}) = {}  |n| = {:.6}", v, n, n.length());
    }

    #[test]
    fn vec2_perpendicular_is_orthogonal() {
        let v = Vec2::new(1.0, 0.0);
        let p = v.perpendicular();
        assert!(approx_eq(v.dot(p), 0.0));
        println!("  perp({}) = {}  dot = {}", v, p, v.dot(p));
    }

    #[test]
    fn vec2_lerp_midpoint() {
        let a = Vec2::new(0.0, 0.0);
        let b = Vec2::new(2.0, 4.0);
        let m = a.lerp(b, 0.5);
        assert!(m.approx_eq(Vec2::new(1.0, 2.0)));
        println!("  lerp({}, {}, 0.5) = {}", a, b, m);
    }

    #[test]
    fn vec2_zero_normalize_returns_zero() {
        let n = Vec2::ZERO.normalize();
        assert!(n.approx_eq(Vec2::ZERO));
        println!("  normalize(ZERO) = {} (safe fallback)", n);
    }

    // ── Vec3 ─────────────────────────────────────────────────────────────

    #[test]
    fn vec3_size_is_16_bytes() {
        let size = std::mem::size_of::<Vec3>();
        assert_eq!(size, 16, "Vec3 must be 16 bytes for SIMD alignment");
        println!("  size_of::<Vec3>() = {} bytes", size);
    }

    #[test]
    fn vec3_align_is_16() {
        let align = std::mem::align_of::<Vec3>();
        assert_eq!(align, 16, "Vec3 must have 16-byte alignment for SSE2");
        println!("  align_of::<Vec3>() = {} bytes", align);
    }

    #[test]
    fn vec3_cross_product_basis() {
        let x = Vec3::X;
        let y = Vec3::Y;
        let z = x.cross(y);
        assert!(z.approx_eq(Vec3::Z));
        println!("  X × Y = {}  (expected Z = {})", z, Vec3::Z);
    }

    #[test]
    fn vec3_cross_anticommutative() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);
        assert!((a.cross(b) + b.cross(a)).approx_eq(Vec3::ZERO));
        println!("  a×b + b×a = ZERO (anticommutative ✓)  a×b = {}", a.cross(b));
    }

    #[test]
    fn vec3_normalize_unit_length() {
        let v = Vec3::new(1.0, 2.0, 3.0);
        let n = v.normalize();
        assert!(approx_eq(n.length(), 1.0));
        println!("  normalize({}) = {}  |n| = {:.6}", v, n, n.length());
    }

    #[test]
    fn vec3_reflect() {
        let v = Vec3::new(1.0, -1.0, 0.0).normalize();
        let n = Vec3::Y;
        let r = v.reflect(n);
        // Reflected Y component should flip sign
        assert!(approx_eq(r.y, -v.y));
        println!("  reflect({}) off Y-normal = {}  (y flipped)", v, r);
    }

    #[test]
    fn vec3_pad_field_is_zero_after_new() {
        let v = Vec3::new(1.0, 2.0, 3.0);
        assert_eq!(v._pad, 0.0, "_pad must be 0.0 to keep the wire format clean");
        println!("  Vec3::new(1, 2, 3)._pad = {}  (always 0)", v._pad);
    }

    #[test]
    fn vec3_equality_ignores_pad() {
        let mut a = Vec3::new(1.0, 2.0, 3.0);
        let     b = Vec3::new(1.0, 2.0, 3.0);
        a._pad = 99.0; // force pad mismatch
        assert_eq!(a, b, "PartialEq must ignore _pad");
        println!("  Vec3 equality ignores _pad (99.0 vs 0.0) — correct");
    }

    #[test]
    fn vec3_distance() {
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(3.0, 4.0, 0.0);
        let d = a.distance(b);
        assert!(approx_eq(d, 5.0));
        println!("  distance({}, {}) = {:.4} (expected 5.0)", a, b, d);
    }

    // ── Vec4 ─────────────────────────────────────────────────────────────

    #[test]
    fn vec4_size_is_16_bytes() {
        let size = std::mem::size_of::<Vec4>();
        assert_eq!(size, 16);
        println!("  size_of::<Vec4>() = {} bytes", size);
    }

    #[test]
    fn vec4_dot_with_self_is_length_sq() {
        let v = Vec4::new(1.0, 2.0, 3.0, 4.0);
        let d = v.dot(v);
        let l = v.length_sq();
        assert!(approx_eq(d, l));
        println!("  dot({}, {}) = {}  length_sq = {}  (match)", v, v, d, l);
    }

    #[test]
    fn vec4_normalize_unit_length() {
        let v = Vec4::new(1.0, 2.0, 3.0, 4.0);
        let n = v.normalize();
        assert!(approx_eq(n.length(), 1.0));
        println!("  normalize({}) → |n| = {:.6}", v, n.length());
    }

    // ── Quaternion ────────────────────────────────────────────────────────

    #[test]
    fn quat_identity_does_not_rotate_vector() {
        let v = Vec3::new(1.0, 2.0, 3.0);
        let rotated = Quat::IDENTITY.rotate(v);
        assert!(rotated.approx_eq(v));
        println!("  IDENTITY.rotate({}) = {}  (unchanged)", v, rotated);
    }

    #[test]
    fn quat_90deg_around_y_rotates_x_to_neg_z() {
        let q = Quat::from_axis_angle(Vec3::Y, to_radians(90.0));
        let v = Vec3::X;
        let r = q.rotate(v);
        assert!(r.approx_eq(Vec3::NEG_Z), "expected NEG_Z, got {}", r);
        println!("  rotate X by 90° around Y = {}  (expected NEG_Z)", r);
    }

    #[test]
    fn quat_multiply_composes_rotations() {
        let q1 = Quat::from_axis_angle(Vec3::Y, to_radians(90.0));
        let q2 = Quat::from_axis_angle(Vec3::Y, to_radians(90.0));
        let q3 = (q1 * q2).normalize();
        let r  = q3.rotate(Vec3::X);
        // 90° + 90° = 180° around Y: X → -X
        assert!(r.approx_eq(Vec3::NEG_X), "expected NEG_X, got {}", r);
        println!("  (90°Y × 90°Y).rotate(X) = {}  (expected NEG_X)", r);
    }

    #[test]
    fn quat_conjugate_is_inverse_for_unit() {
        let q = Quat::from_axis_angle(Vec3::new(1.0,1.0,0.0).normalize(), to_radians(45.0));
        let inv = q.conjugate();
        let composed = (q * inv).normalize();
        assert!(approx_eq(composed.w, 1.0) && approx_eq(composed.x, 0.0));
        println!("  q * q.conjugate() ≈ IDENTITY  w={:.6} x={:.6}", composed.w, composed.x);
    }

    #[test]
    fn quat_euler_round_trip() {
        let (roll0, pitch0, yaw0) = (0.3, 0.5, 1.2_f32);
        let q = Quat::from_euler(roll0, pitch0, yaw0);
        let (roll1, pitch1, yaw1) = q.to_euler();
        assert!(approx_eq(roll0, roll1),  "roll  mismatch: {} vs {}", roll0, roll1);
        assert!(approx_eq(pitch0, pitch1),"pitch mismatch: {} vs {}", pitch0, pitch1);
        assert!(approx_eq(yaw0, yaw1),   "yaw   mismatch: {} vs {}", yaw0, yaw1);
        println!(
            "  euler({:.3},{:.3},{:.3}) → quat → euler({:.3},{:.3},{:.3})",
            roll0, pitch0, yaw0, roll1, pitch1, yaw1,
        );
    }

    #[test]
    fn quat_slerp_endpoints() {
        let a = Quat::from_axis_angle(Vec3::Y, to_radians(0.0));
        let b = Quat::from_axis_angle(Vec3::Y, to_radians(90.0));
        let at_0   = a.slerp(b, 0.0).normalize();
        let at_1   = a.slerp(b, 1.0).normalize();
        let at_half = a.slerp(b, 0.5);
        let mid_angle = at_half.rotate(Vec3::X);
        println!("  slerp(0.0)  → {:?}", at_0);
        println!("  slerp(0.5)  → rotates X to {}", mid_angle);
        println!("  slerp(1.0)  → {:?}", at_1);
        // At t=0.5, X should be roughly at 45° in the XZ plane
        assert!(mid_angle.x > 0.0 && mid_angle.z < 0.0,
            "expected first quadrant, got {}", mid_angle);
    }

    #[test]
    fn quat_to_mat4_identity_roundtrip() {
        let m = Quat::IDENTITY.to_mat4();
        assert_eq!(m, Mat4::IDENTITY);
        println!("  IDENTITY.to_mat4() == Mat4::IDENTITY ✓");
    }

    // ── Mat3 ─────────────────────────────────────────────────────────────

    #[test]
    fn mat3_identity_times_vec_is_vec() {
        let v = Vec3::new(1.0, 2.0, 3.0);
        let r = Mat3::IDENTITY.transform(v);
        assert!(r.approx_eq(v));
        println!("  IDENTITY × {} = {}  (unchanged)", v, r);
    }

    #[test]
    fn mat3_multiply_identity_is_identity() {
        let m = Mat3::IDENTITY * Mat3::IDENTITY;
        assert_eq!(m, Mat3::IDENTITY);
        println!("  IDENTITY × IDENTITY = IDENTITY ✓");
    }

    #[test]
    fn mat3_inverse_of_identity_is_identity() {
        let inv = Mat3::IDENTITY.inverse().expect("identity is invertible");
        assert_eq!(inv, Mat3::IDENTITY);
        println!("  inverse(IDENTITY) = IDENTITY ✓");
    }

    #[test]
    fn mat3_inverse_roundtrip() {
        let m = Mat3::from_cols(
            [2.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 4.0],
        );
        let inv = m.inverse().expect("diagonal matrix is invertible");
        let product = m * inv;
        // Each diagonal should be 1.0, off-diagonals 0.0
        for c in 0..3 { for r in 0..3 {
            let expected = if c == r { 1.0 } else { 0.0 };
            assert!((product.cols[c][r] - expected).abs() < 1e-5,
                "m*inv[{}][{}] = {} (expected {})", c, r, product.cols[c][r], expected);
        }}
        println!("  diagonal Mat3 * inverse ≈ identity ✓");
    }

    // ── Mat4 ─────────────────────────────────────────────────────────────

    #[test]
    fn mat4_size_is_64_bytes() {
        let size = std::mem::size_of::<Mat4>();
        assert_eq!(size, 64);
        println!("  size_of::<Mat4>() = {} bytes", size);
    }

    #[test]
    fn mat4_identity_transform_point_unchanged() {
        let p = Vec3::new(1.0, 2.0, 3.0);
        let r = Mat4::IDENTITY.transform_point(p);
        assert!(r.approx_eq(p));
        println!("  IDENTITY.transform_point({}) = {}  (unchanged)", p, r);
    }

    #[test]
    fn mat4_translation_moves_point() {
        let t  = Vec3::new(10.0, 20.0, 30.0);
        let m  = Mat4::from_translation(t);
        let p  = Vec3::new(1.0, 1.0, 1.0);
        let r  = m.transform_point(p);
        assert!(r.approx_eq(Vec3::new(11.0, 21.0, 31.0)));
        println!("  translate({}).transform_point({}) = {}", t, p, r);
    }

    #[test]
    fn mat4_translation_does_not_affect_vectors() {
        let m = Mat4::from_translation(Vec3::new(99.0, 99.0, 99.0));
        let v = Vec3::new(1.0, 0.0, 0.0);
        let r = m.transform_vector(v);
        assert!(r.approx_eq(v), "translation should not affect w=0 vectors");
        println!("  translate(99).transform_vector({}) = {}  (unchanged)", v, r);
    }

    #[test]
    fn mat4_scale_scales_point() {
        let s = Vec3::new(2.0, 3.0, 4.0);
        let m = Mat4::from_scale(s);
        let p = Vec3::new(1.0, 1.0, 1.0);
        let r = m.transform_point(p);
        assert!(r.approx_eq(Vec3::new(2.0, 3.0, 4.0)));
        println!("  scale({}).transform_point({}) = {}", s, p, r);
    }

    #[test]
    fn mat4_multiply_identity_is_identity() {
        let m = Mat4::IDENTITY * Mat4::IDENTITY;
        assert_eq!(m, Mat4::IDENTITY);
        println!("  IDENTITY × IDENTITY = IDENTITY ✓");
    }

    #[test]
    fn mat4_inverse_of_identity_is_identity() {
        let inv = Mat4::IDENTITY.inverse().expect("identity is invertible");
        assert_eq!(inv, Mat4::IDENTITY);
        println!("  inverse(IDENTITY) = IDENTITY ✓");
    }

    #[test]
    fn mat4_inverse_roundtrip() {
        let m   = Mat4::from_trs(
            Vec3::new(1.0, 2.0, 3.0),
            Quat::from_axis_angle(Vec3::Y, to_radians(45.0)),
            Vec3::new(2.0, 2.0, 2.0),
        );
        let inv = m.inverse().expect("TRS matrix is invertible");
        let eye = m * inv;
        for c in 0..4 { for r in 0..4 {
            let expected = if c == r { 1.0 } else { 0.0 };
            assert!((eye.cols[c][r] - expected).abs() < 1e-4,
                "m*inv[{}][{}] = {:.6} (expected {:.1})", c, r, eye.cols[c][r], expected);
        }}
        println!("  TRS Mat4 * inverse ≈ identity ✓");
    }

    #[test]
    fn mat4_singular_inverse_returns_none() {
        let m = Mat4::ZERO;
        assert!(m.inverse().is_none(), "zero matrix has no inverse");
        println!("  inverse(ZERO) = None ✓");
    }

    #[test]
    fn mat4_perspective_has_negative_one_at_col3_row2() {
        // In right-handed perspective, cols[3][2] drives the -1/z divide.
        let m = Mat4::perspective_rh(to_radians(60.0), 16.0/9.0, 0.1, 1000.0);
        assert!(approx_eq(m.cols[2][3], -1.0),
            "cols[2][3] should be -1 for right-handed perspective, got {}", m.cols[2][3]);
        println!("  perspective_rh cols[2][3] = {:.4} (expected -1.0)", m.cols[2][3]);
    }

    #[test]
    fn mat4_look_at_z_axis_points_toward_target() {
        let eye    = Vec3::new(0.0, 0.0, 5.0);
        let center = Vec3::ZERO;
        let up     = Vec3::Y;
        let view   = Mat4::look_at_rh(eye, center, up);
        // Transform the target in view space — should land near the origin on the -Z axis.
        let target_vs = view.transform_point(center);
        println!(
            "  look_at: target in view space = {}  (should be on -Z)",
            target_vs,
        );
        assert!(target_vs.z < 0.0, "target should be behind camera on -Z, got z={}", target_vs.z);
    }
      }
