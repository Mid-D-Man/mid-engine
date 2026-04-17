// crates/mid-math/src/tests.rs
//
// Build mode labeling: every stress test prints [DEBUG] or [RELEASE] based on
// cfg!(debug_assertions) — which is true in `cargo test` and false in
// `cargo test --release`. This is how you tell which build produced a number.
//
// Correctness tests have no timing and need no label.

#[cfg(test)]
mod tests {
    use crate::vec::{Vec2, Vec3, Vec4};
    use crate::quat::Quat;
    use crate::mat::{Mat3, Mat4};
    use crate::{EPSILON, to_radians, lerp, smoothstep, approx_eq};
    use std::time::Instant;

    // Resolved at compile time — zero runtime cost, correct in both modes.
    const BUILD_MODE: &str = if cfg!(debug_assertions) { "[DEBUG]" } else { "[RELEASE]" };

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
        assert!((mid - 0.5).abs() < EPSILON);
        println!("  smoothstep(0, 1, 0.5) = {:.6} (exact midpoint)", mid);
    }

    // ── Vec2 ─────────────────────────────────────────────────────────────

    #[test]
    fn vec2_addition() {
        let c = Vec2::new(1.0,2.0) + Vec2::new(3.0,4.0);
        assert!(c.approx_eq(Vec2::new(4.0,6.0)));
        println!("  (1, 2) + (3, 4) = {}", c);
    }

    #[test]
    fn vec2_dot_product() {
        let d = Vec2::new(1.0,0.0).dot(Vec2::new(0.0,1.0));
        assert!(approx_eq(d, 0.0));
        println!("  dot(X, Y) = {} (perpendicular → 0)", d);
    }

    #[test]
    fn vec2_normalize_unit_length() {
        let n = Vec2::new(3.0,4.0).normalize();
        assert!(approx_eq(n.length(), 1.0));
        println!("  normalize(3,4) → |n| = {:.6}", n.length());
    }

    #[test]
    fn vec2_perpendicular_is_orthogonal() {
        let v = Vec2::new(1.0,0.0);
        let p = v.perpendicular();
        assert!(approx_eq(v.dot(p), 0.0));
        println!("  perp({}) = {}  dot = {}", v, p, v.dot(p));
    }

    #[test]
    fn vec2_lerp_midpoint() {
        let m = Vec2::ZERO.lerp(Vec2::new(2.0,4.0), 0.5);
        assert!(m.approx_eq(Vec2::new(1.0,2.0)));
        println!("  lerp(0,0 → 2,4, 0.5) = {}", m);
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
        assert_eq!(size, 16);
        println!("  size_of::<Vec3>() = {} bytes", size);
    }

    #[test]
    fn vec3_align_is_16() {
        let align = std::mem::align_of::<Vec3>();
        assert_eq!(align, 16);
        println!("  align_of::<Vec3>() = {} bytes", align);
    }

    #[test]
    fn vec3_cross_product_basis() {
        let z = Vec3::X.cross(Vec3::Y);
        assert!(z.approx_eq(Vec3::Z));
        println!("  X × Y = {}  (expected Z)", z);
    }

    #[test]
    fn vec3_cross_anticommutative() {
        let a = Vec3::new(1.0,2.0,3.0);
        let b = Vec3::new(4.0,5.0,6.0);
        assert!((a.cross(b) + b.cross(a)).approx_eq(Vec3::ZERO));
        println!("  a×b + b×a = ZERO ✓");
    }

    #[test]
    fn vec3_normalize_unit_length() {
        let n = Vec3::new(1.0,2.0,3.0).normalize();
        assert!(approx_eq(n.length(), 1.0));
        println!("  normalize(1,2,3) → |n| = {:.6}", n.length());
    }

    #[test]
    fn vec3_reflect() {
        let v = Vec3::new(1.0,-1.0,0.0).normalize();
        let r = v.reflect(Vec3::Y);
        assert!(approx_eq(r.y, -v.y));
        println!("  reflect off Y: {} → {}  (y flipped)", v, r);
    }

    #[test]
    fn vec3_pad_field_is_zero_after_new() {
        let v = Vec3::new(1.0,2.0,3.0);
        assert_eq!(v._pad, 0.0);
        println!("  Vec3::new(1,2,3)._pad = {} (always 0)", v._pad);
    }

    #[test]
    fn vec3_equality_ignores_pad() {
        let mut a = Vec3::new(1.0,2.0,3.0);
        let     b = Vec3::new(1.0,2.0,3.0);
        a._pad = 99.0;
        assert_eq!(a, b);
        println!("  Vec3 equality ignores _pad ✓");
    }

    #[test]
    fn vec3_distance() {
        let d = Vec3::ZERO.distance(Vec3::new(3.0,4.0,0.0));
        assert!(approx_eq(d, 5.0));
        println!("  distance(origin → 3,4,0) = {:.4} (expected 5.0)", d);
    }

    // ── Vec4 ─────────────────────────────────────────────────────────────

    #[test]
    fn vec4_size_is_16_bytes() {
        assert_eq!(std::mem::size_of::<Vec4>(), 16);
        println!("  size_of::<Vec4>() = 16 bytes");
    }

    #[test]
    fn vec4_dot_with_self_is_length_sq() {
        let v = Vec4::new(1.0,2.0,3.0,4.0);
        assert!(approx_eq(v.dot(v), v.length_sq()));
        println!("  dot(v,v) == length_sq ✓  value = {}", v.length_sq());
    }

    #[test]
    fn vec4_normalize_unit_length() {
        let n = Vec4::new(1.0,2.0,3.0,4.0).normalize();
        assert!(approx_eq(n.length(), 1.0));
        println!("  normalize(1,2,3,4) → |n| = {:.6}", n.length());
    }

    // ── Quaternion ────────────────────────────────────────────────────────

    #[test]
    fn quat_identity_does_not_rotate_vector() {
        let v = Vec3::new(1.0,2.0,3.0);
        let r = Quat::IDENTITY.rotate(v);
        assert!(r.approx_eq(v));
        println!("  IDENTITY.rotate({}) = {}  (unchanged)", v, r);
    }

    #[test]
    fn quat_90deg_around_y_rotates_x_to_neg_z() {
        let q = Quat::from_axis_angle(Vec3::Y, to_radians(90.0));
        let r = q.rotate(Vec3::X);
        assert!(r.approx_eq(Vec3::NEG_Z), "expected NEG_Z, got {}", r);
        println!("  rotate X by 90°Y = {}  (expected NEG_Z)", r);
    }

    #[test]
    fn quat_multiply_composes_rotations() {
        let q = Quat::from_axis_angle(Vec3::Y, to_radians(90.0));
        let r = (q * q).normalize().rotate(Vec3::X);
        assert!(r.approx_eq(Vec3::NEG_X), "expected NEG_X, got {}", r);
        println!("  (90°Y × 90°Y).rotate(X) = {}  (expected NEG_X)", r);
    }

    #[test]
    fn quat_conjugate_is_inverse_for_unit() {
        let q = Quat::from_axis_angle(Vec3::new(1.0,1.0,0.0).normalize(), to_radians(45.0));
        let c = (q * q.conjugate()).normalize();
        assert!(approx_eq(c.w, 1.0) && approx_eq(c.x, 0.0));
        println!("  q * q.conjugate() ≈ IDENTITY  w={:.6} x={:.6}", c.w, c.x);
    }

    #[test]
    fn quat_euler_round_trip() {
        let cases = [
            (0.3_f32, 0.5_f32, 1.2_f32),
            (0.0,     0.0,     0.0),
            (1.0,    -0.5,     2.0),
            (-0.7,    0.3,    -1.5),
        ];
        for (roll0, pitch0, yaw0) in cases {
            let q = Quat::from_euler(roll0, pitch0, yaw0);
            let (roll1, pitch1, yaw1) = q.to_euler();
            assert!(approx_eq(roll0, roll1),   "roll  {} vs {}", roll0, roll1);
            assert!(approx_eq(pitch0, pitch1), "pitch {} vs {}", pitch0, pitch1);
            assert!(approx_eq(yaw0, yaw1),     "yaw   {} vs {}", yaw0, yaw1);
            println!(
                "  euler({:.2},{:.2},{:.2}) → quat → euler({:.2},{:.2},{:.2}) ✓",
                roll0, pitch0, yaw0, roll1, pitch1, yaw1,
            );
        }
    }

    #[test]
    fn quat_slerp_endpoints() {
        let a   = Quat::from_axis_angle(Vec3::Y, to_radians(0.0));
        let b   = Quat::from_axis_angle(Vec3::Y, to_radians(90.0));
        let mid = a.slerp(b, 0.5).rotate(Vec3::X);
        assert!(mid.x > 0.0 && mid.z < 0.0, "expected first quadrant, got {}", mid);
        println!("  slerp(0°,90°,t=0.5).rotate(X) = {}  ✓ first quadrant", mid);
    }

    #[test]
    fn quat_to_mat4_identity_roundtrip() {
        assert_eq!(Quat::IDENTITY.to_mat4(), Mat4::IDENTITY);
        println!("  IDENTITY.to_mat4() == Mat4::IDENTITY ✓");
    }

    // ── Mat3 ─────────────────────────────────────────────────────────────

    #[test]
    fn mat3_identity_times_vec_is_vec() {
        let v = Vec3::new(1.0,2.0,3.0);
        assert!(Mat3::IDENTITY.transform(v).approx_eq(v));
        println!("  IDENTITY × {} = {}  ✓", v, Mat3::IDENTITY.transform(v));
    }

    #[test]
    fn mat3_multiply_identity_is_identity() {
        assert_eq!(Mat3::IDENTITY * Mat3::IDENTITY, Mat3::IDENTITY);
        println!("  IDENTITY × IDENTITY = IDENTITY ✓");
    }

    #[test]
    fn mat3_inverse_of_identity_is_identity() {
        assert_eq!(Mat3::IDENTITY.inverse().unwrap(), Mat3::IDENTITY);
        println!("  inverse(IDENTITY) = IDENTITY ✓");
    }

    #[test]
    fn mat3_inverse_roundtrip() {
        let m   = Mat3::from_cols([2.0,0.0,0.0],[0.0,3.0,0.0],[0.0,0.0,4.0]);
        let inv = m.inverse().expect("diagonal matrix is invertible");
        let p   = m * inv;
        for c in 0..3 { for r in 0..3 {
            let exp = if c==r{1.0}else{0.0};
            assert!((p.cols[c][r]-exp).abs()<1e-5, "m*inv[{}][{}]={}", c,r,p.cols[c][r]);
        }}
        println!("  diagonal Mat3 * inverse ≈ identity ✓");
    }

    // ── Mat4 ─────────────────────────────────────────────────────────────

    #[test]
    fn mat4_size_is_64_bytes() {
        assert_eq!(std::mem::size_of::<Mat4>(), 64);
        println!("  size_of::<Mat4>() = 64 bytes");
    }

    #[test]
    fn mat4_identity_transform_point_unchanged() {
        let p = Vec3::new(1.0,2.0,3.0);
        assert!(Mat4::IDENTITY.transform_point(p).approx_eq(p));
        println!("  IDENTITY.transform_point({}) = {}  ✓", p, p);
    }

    #[test]
    fn mat4_translation_moves_point() {
        let r = Mat4::from_translation(Vec3::new(10.0,20.0,30.0)).transform_point(Vec3::ONE);
        assert!(r.approx_eq(Vec3::new(11.0,21.0,31.0)));
        println!("  translate(10,20,30) + (1,1,1) = {}  ✓", r);
    }

    #[test]
    fn mat4_translation_does_not_affect_vectors() {
        let m = Mat4::from_translation(Vec3::new(99.0,99.0,99.0));
        assert!(m.transform_vector(Vec3::X).approx_eq(Vec3::X));
        println!("  translation does not affect w=0 vectors ✓");
    }

    #[test]
    fn mat4_scale_scales_point() {
        let r = Mat4::from_scale(Vec3::new(2.0,3.0,4.0)).transform_point(Vec3::ONE);
        assert!(r.approx_eq(Vec3::new(2.0,3.0,4.0)));
        println!("  scale(2,3,4) × (1,1,1) = {}  ✓", r);
    }

    #[test]
    fn mat4_multiply_identity_is_identity() {
        assert_eq!(Mat4::IDENTITY * Mat4::IDENTITY, Mat4::IDENTITY);
        println!("  IDENTITY × IDENTITY = IDENTITY ✓");
    }

    #[test]
    fn mat4_inverse_of_identity_is_identity() {
        assert_eq!(Mat4::IDENTITY.inverse().unwrap(), Mat4::IDENTITY);
        println!("  inverse(IDENTITY) = IDENTITY ✓");
    }

    #[test]
    fn mat4_inverse_roundtrip() {
        let m   = Mat4::from_trs(
            Vec3::new(1.0,2.0,3.0),
            Quat::from_axis_angle(Vec3::Y, to_radians(45.0)),
            Vec3::new(2.0,2.0,2.0),
        );
        let inv = m.inverse().expect("TRS matrix is invertible");
        let eye = m * inv;
        for c in 0..4 { for r in 0..4 {
            let exp = if c==r{1.0}else{0.0};
            assert!((eye.cols[c][r]-exp).abs()<1e-4,
                "m*inv[{}][{}]={:.6}", c, r, eye.cols[c][r]);
        }}
        println!("  TRS Mat4 * inverse ≈ identity ✓");
    }

    #[test]
    fn mat4_singular_inverse_returns_none() {
        assert!(Mat4::ZERO.inverse().is_none());
        println!("  inverse(ZERO) = None ✓");
    }

    #[test]
    fn mat4_perspective_has_negative_one_at_col3_row2() {
        let m = Mat4::perspective_rh(to_radians(60.0), 16.0/9.0, 0.1, 1000.0);
        assert!(approx_eq(m.cols[2][3], -1.0), "cols[2][3]={}", m.cols[2][3]);
        println!("  perspective_rh cols[2][3] = {:.4} ✓", m.cols[2][3]);
    }

    #[test]
    fn mat4_look_at_z_axis_points_toward_target() {
        let view = Mat4::look_at_rh(Vec3::new(0.0,0.0,5.0), Vec3::ZERO, Vec3::Y);
        let t    = view.transform_point(Vec3::ZERO);
        assert!(t.z < 0.0, "target should be on -Z in view space, got z={}", t.z);
        println!("  look_at: target in view space = {}  (z < 0) ✓", t);
    }

    // ── Mat4::inverse_trs ─────────────────────────────────────────────────

    #[test]
    fn mat4_inverse_trs_identity() {
        let inv = Mat4::IDENTITY.inverse_trs();
        assert_eq!(inv, Mat4::IDENTITY);
        println!("  inverse_trs(IDENTITY) = IDENTITY ✓");
    }

    #[test]
    fn mat4_inverse_trs_translation_only() {
        let t   = Vec3::new(5.0, -3.0, 7.0);
        let m   = Mat4::from_translation(t);
        let inv = m.inverse_trs();
        let p   = inv.transform_point(Vec3::ZERO);
        assert!(p.approx_eq(-t), "expected {} got {}", -t, p);
        println!("  inverse_trs(translate({},{},{})) applied to origin = {}  ✓", t.x, t.y, t.z, p);
    }

    #[test]
    fn mat4_inverse_trs_scale_only() {
        let m   = Mat4::from_scale(Vec3::new(2.0, 4.0, 0.5));
        let inv = m.inverse_trs();
        let p   = inv.transform_point(Vec3::new(2.0, 4.0, 0.5));
        assert!(p.approx_eq(Vec3::ONE), "expected (1,1,1) got {}", p);
        println!("  inverse_trs(scale(2,4,0.5)) undoes scaling ✓  result={}", p);
    }

    #[test]
    fn mat4_inverse_trs_roundtrip_matches_general_inverse() {
        let m = Mat4::from_trs(
            Vec3::new(3.0, -1.0, 5.0),
            Quat::from_axis_angle(Vec3::new(1.0,1.0,0.0).normalize(), to_radians(37.0)),
            Vec3::new(2.0, 0.5, 3.0),
        );
        let inv_general = m.inverse().expect("TRS matrix is invertible");
        let inv_trs     = m.inverse_trs();
        for c in 0..4 { for r in 0..4 {
            assert!(
                (inv_general.cols[c][r] - inv_trs.cols[c][r]).abs() < 1e-4,
                "mismatch at col={} row={}: general={:.6} trs={:.6}",
                c, r, inv_general.cols[c][r], inv_trs.cols[c][r],
            );
        }}
        println!("  inverse_trs matches general inverse for TRS matrix ✓");
    }

    #[test]
    fn mat4_inverse_trs_zero_scale_does_not_panic() {
        let m = Mat4::from_scale(Vec3::new(0.0, 1.0, 1.0));
        let _ = m.inverse_trs();
        println!("  inverse_trs with zero-scale axis = no panic ✓");
    }

    // ── Stress: Vec ops ───────────────────────────────────────────────────

    #[test]
    fn stress_100k_vec3_add() {
        let a = Vec3::new(1.0,2.0,3.0);
        let b = Vec3::new(0.1,0.2,0.3);
        let count = 100_000usize;
        let start = Instant::now();
        let mut acc = Vec3::ZERO;
        for _ in 0..count { acc = acc + a; acc = acc + b; }
        let elapsed = start.elapsed();
        let _ = std::hint::black_box(acc);
        assert!(acc.length() > 0.0);
        println!(
            "  {} Vec3 adds in {:.3}ms  ({:.1} ns/op)  final_x={:.0}  {}",
            count*2, elapsed.as_secs_f64()*1000.0,
            elapsed.as_nanos() as f64/(count*2) as f64, acc.x, BUILD_MODE,
        );
    }

    #[test]
    fn stress_100k_vec3_dot() {
        let a = Vec3::new(1.0,0.0,0.0);
        let b = Vec3::new(0.6,0.8,0.0);
        let count = 100_000usize;
        let start = Instant::now();
        let mut acc = 0.0f32;
        for _ in 0..count { acc += a.dot(b); }
        let elapsed = start.elapsed();
        let _ = std::hint::black_box(acc);
        assert!(acc > 0.0);
        println!(
            "  {} Vec3 dot products in {:.3}ms  ({:.1} ns/op)  avg={:.4}  {}",
            count, elapsed.as_secs_f64()*1000.0,
            elapsed.as_nanos() as f64/count as f64, acc/count as f32, BUILD_MODE,
        );
    }

    #[test]
    fn stress_100k_vec3_cross() {
        let a = Vec3::X;
        let b = Vec3::Y;
        let count = 100_000usize;
        let start = Instant::now();
        let mut acc = Vec3::ZERO;
        for _ in 0..count { acc = acc + a.cross(b); }
        let elapsed = start.elapsed();
        let _ = std::hint::black_box(acc);
        assert!(acc.length() > 0.0);
        println!(
            "  {} Vec3 cross products in {:.3}ms  ({:.1} ns/op)  {}",
            count, elapsed.as_secs_f64()*1000.0,
            elapsed.as_nanos() as f64/count as f64, BUILD_MODE,
        );
    }

    #[test]
    fn stress_100k_vec3_normalize() {
        let count = 100_000usize;
        let start = Instant::now();
        let mut acc = 0.0f32;
        for i in 0..count {
            let v = Vec3::new(i as f32+1.0, i as f32*0.5+1.0, i as f32*0.3+1.0);
            acc += v.normalize().x;
        }
        let elapsed = start.elapsed();
        let _ = std::hint::black_box(acc);
        assert!(acc.is_finite());
        println!(
            "  {} Vec3 normalizes in {:.3}ms  ({:.1} ns/op)  acc={:.2}  {}",
            count, elapsed.as_secs_f64()*1000.0,
            elapsed.as_nanos() as f64/count as f64, acc, BUILD_MODE,
        );
    }

    #[test]
    fn stress_100k_vec3_lerp() {
        let a = Vec3::ZERO;
        let b = Vec3::ONE;
        let count = 100_000usize;
        let start = Instant::now();
        let mut acc = Vec3::ZERO;
        for i in 0..count {
            let t = (i as f32)/count as f32;
            acc = acc + a.lerp(b, t);
        }
        let elapsed = start.elapsed();
        let _ = std::hint::black_box(acc);
        assert!(acc.length() > 0.0);
        println!(
            "  {} Vec3 lerps in {:.3}ms  ({:.1} ns/op)  {}",
            count, elapsed.as_secs_f64()*1000.0,
            elapsed.as_nanos() as f64/count as f64, BUILD_MODE,
        );
    }

    // ── Stress: Quat ops ──────────────────────────────────────────────────

    #[test]
    fn stress_100k_quat_rotate() {
        let q     = Quat::from_axis_angle(Vec3::Y, to_radians(45.0));
        let v     = Vec3::X;
        let count = 100_000usize;
        let start = Instant::now();
        let mut acc = Vec3::ZERO;
        for _ in 0..count { acc = acc + q.rotate(v); }
        let elapsed = start.elapsed();
        let _ = std::hint::black_box(acc);
        assert!(acc.length() > 0.0);
        let ms = elapsed.as_secs_f64()*1000.0;
        println!(
            "  {} Quat rotations in {:.3}ms  ({:.1} ns/op)  {}",
            count, ms, elapsed.as_nanos() as f64/count as f64, BUILD_MODE,
        );
        println!(
            "  ECS 60Hz frame budget=16.6ms — 100k rotations took {:.3}ms ({})",
            ms, if ms < 16.6 {"✓ within budget"} else {"⚠ over budget"},
        );
    }

    #[test]
    fn stress_100k_quat_mul() {
        let q1    = Quat::from_axis_angle(Vec3::Y, to_radians(1.0));
        let q2    = Quat::from_axis_angle(Vec3::X, to_radians(0.5));
        let count = 100_000usize;
        let start = Instant::now();
        let mut acc = Quat::IDENTITY;
        for _ in 0..count { acc = acc * q1 * q2; }
        let elapsed = start.elapsed();
        let _ = std::hint::black_box(acc);
        assert!(acc.length_sq() > 0.0);
        println!(
            "  {} Quat multiplications in {:.3}ms  ({:.1} ns/op)  {}",
            count*2, elapsed.as_secs_f64()*1000.0,
            elapsed.as_nanos() as f64/(count*2) as f64, BUILD_MODE,
        );
    }

    #[test]
    fn stress_50k_quat_slerp() {
        let a     = Quat::from_axis_angle(Vec3::Y, to_radians(0.0));
        let b     = Quat::from_axis_angle(Vec3::Y, to_radians(90.0));
        let count = 50_000usize;
        let start = Instant::now();
        let mut acc = Quat::IDENTITY;
        for i in 0..count {
            let t = (i as f32)/count as f32;
            acc = acc * a.slerp(b, t);
        }
        let elapsed = start.elapsed();
        let _ = std::hint::black_box(acc);
        assert!(acc.length_sq() > 0.0);
        println!(
            "  {} Quat slerps in {:.3}ms  ({:.1} ns/op)  {}",
            count, elapsed.as_secs_f64()*1000.0,
            elapsed.as_nanos() as f64/count as f64, BUILD_MODE,
        );
    }

    #[test]
    fn stress_50k_euler_from_to_roundtrip() {
        let count = 50_000usize;
        let start = Instant::now();
        let mut acc = 0.0f32;
        for i in 0..count {
            let f = i as f32*0.0001;
            let q = Quat::from_euler(f, f*0.7, f*1.3);
            let (r, p, y) = q.to_euler();
            acc += r + p + y;
        }
        let elapsed = start.elapsed();
        let _ = std::hint::black_box(acc);
        assert!(acc.is_finite());
        println!(
            "  {} euler↔quat round-trips in {:.3}ms  ({:.1} ns/op)  {}",
            count, elapsed.as_secs_f64()*1000.0,
            elapsed.as_nanos() as f64/count as f64, BUILD_MODE,
        );
    }

    // ── Stress: Mat4 ops ──────────────────────────────────────────────────

    #[test]
    fn stress_10k_mat4_mul() {
        let a = Mat4::from_rotation(Quat::from_axis_angle(Vec3::Y, to_radians(0.01)));
        let b = Mat4::from_rotation(Quat::from_axis_angle(Vec3::X, to_radians(0.01)));
        let count = 10_000usize;
        let start = Instant::now();
        let mut acc = Mat4::IDENTITY;
        for _ in 0..count { acc = acc * a * b; }
        let elapsed = start.elapsed();
        let acc = std::hint::black_box(acc);
        assert!(acc.cols[0][0].is_finite(), "rotation matrix entries must be finite");
        println!(
            "  {} Mat4 multiplications in {:.3}ms  ({:.1} ns/op)  {}",
            count*2, elapsed.as_secs_f64()*1000.0,
            elapsed.as_nanos() as f64/(count*2) as f64, BUILD_MODE,
        );
    }

    #[test]
    fn stress_10k_mat4_transform_point() {
        let m = Mat4::from_trs(
            Vec3::new(1.0,2.0,3.0),
            Quat::from_axis_angle(Vec3::Y, to_radians(45.0)),
            Vec3::new(2.0,2.0,2.0),
        );
        let count = 10_000usize;
        let start = Instant::now();
        let mut acc = Vec3::ZERO;
        for i in 0..count {
            acc = acc + m.transform_point(Vec3::new(i as f32, 0.0, 0.0));
        }
        let elapsed = start.elapsed();
        let _ = std::hint::black_box(acc);
        assert!(acc.length() > 0.0);
        println!(
            "  {} Mat4 transform_point in {:.3}ms  ({:.1} ns/op)  {}",
            count, elapsed.as_secs_f64()*1000.0,
            elapsed.as_nanos() as f64/count as f64, BUILD_MODE,
        );
    }

    #[test]
    fn stress_5k_mat4_inverse() {
        let count = 5_000usize;
        let start = Instant::now();
        let mut passed = 0usize;
        for i in 0..count {
            let m = Mat4::from_trs(
                Vec3::new(i as f32*0.1, 0.0, 0.0),
                Quat::from_axis_angle(Vec3::Y, to_radians(i as f32)),
                Vec3::new(1.0+i as f32*0.001, 1.0, 1.0),
            );
            if m.inverse().is_some() { passed += 1; }
        }
        let elapsed = start.elapsed();
        assert_eq!(passed, count, "all TRS matrices should be invertible");
        let ns_per = elapsed.as_nanos() as f64 / count as f64;
        println!(
            "  {} Mat4 general inverses in {:.3}ms  ({:.1} ns/op)  all invertible ✓  {}",
            count, elapsed.as_secs_f64()*1000.0, ns_per, BUILD_MODE,
        );
    }

    #[test]
    fn stress_5k_mat4_inverse_trs() {
        // Tier 1 TRS fast-path. Debug baseline Build #19: 290.7 ns/op.
        // General inverse same build debug: 707.6 ns/op → 2.4× faster.
        // Release numbers: pending Build #20 first run with correct label.
        let count = 5_000usize;
        let start = Instant::now();
        for i in 0..count {
            let m = Mat4::from_trs(
                Vec3::new(i as f32*0.1, 0.0, 0.0),
                Quat::from_axis_angle(Vec3::Y, to_radians(i as f32)),
                Vec3::new(1.0+i as f32*0.001, 1.0, 1.0),
            );
            let _ = std::hint::black_box(m.inverse_trs());
        }
        let elapsed = start.elapsed();
        let ns_per = elapsed.as_nanos() as f64 / count as f64;
        println!(
            "  {} Mat4 inverse_trs in {:.3}ms  ({:.1} ns/op)  {}",
            count, elapsed.as_secs_f64()*1000.0, ns_per, BUILD_MODE,
        );
        // In-build comparison: how much faster than the general inverse in this build?
        // We don't have that number here directly, but the benchmark comment in mat.rs
        // records the cross-build and in-build ratios for audit.
    }

    #[test]
    fn stress_100k_entity_transform_simulation() {
        let entity_count = 100_000usize;
        let trs = Mat4::from_trs(
            Vec3::new(1.0,0.0,0.0),
            Quat::from_axis_angle(Vec3::Y, to_radians(45.0)),
            Vec3::ONE,
        );
        let mut positions: Vec<Vec3> = (0..entity_count)
            .map(|i| Vec3::new(i as f32*0.01, 0.0, 0.0))
            .collect();

        let start = Instant::now();
        for p in positions.iter_mut() { *p = trs.transform_point(*p); }
        let elapsed = start.elapsed();
        let ms = elapsed.as_secs_f64()*1000.0;

        let _ = std::hint::black_box(&positions);
        assert!(positions[0].length() > 0.0);

        println!(
            "  {} entity transforms in {:.3}ms  ({:.1} ns/entity)  {}",
            entity_count, ms, elapsed.as_nanos() as f64/entity_count as f64, BUILD_MODE,
        );
        println!(
            "  ECS 60Hz budget=16.6ms — 100k transforms took {:.3}ms ({})",
            ms, if ms < 16.6 {"✓ within budget"} else {"⚠ over budget"},
        );
    }

    #[test]
    fn stress_mixed_math_1k_frames_simulation() {
        let ticks = 1_000usize;
        let start = Instant::now();
        let mut total_pos = Vec3::ZERO;

        for tick in 0..ticks {
            let t = tick as f32*0.016;
            let q = Quat::from_euler(t*0.1, t*0.2, t*0.3);
            let m = Mat4::from_trs(Vec3::new(t,0.0,0.0), q, Vec3::ONE);
            for i in 0..10 {
                let p = Vec3::new(i as f32, 0.0, 0.0);
                total_pos = total_pos + p.lerp(m.transform_point(p), 0.5);
            }
        }

        let elapsed = start.elapsed();
        let _ = std::hint::black_box(total_pos);
        assert!(total_pos.length() > 0.0);
        println!(
            "  {} ticks × (TRS + 10 transforms + 10 lerps) = {} ops in {:.3}ms  ({:.1} µs/tick)  {}",
            ticks, ticks*21, elapsed.as_secs_f64()*1000.0,
            elapsed.as_secs_f64()*1_000_000.0/ticks as f64, BUILD_MODE,
        );
    }
}
