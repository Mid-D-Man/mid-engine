// crates/mid-math/src/tests/vec_tests.rs
#[cfg(test)]
mod tests {
    use crate::{Vec2, Vec3, Vec4, approx_eq, EPSILON};

    // ── Vec2 ──────────────────────────────────────────────────────────────────

    #[test]
    fn vec2_addition() {
        let c = Vec2::new(1.0,2.0) + Vec2::new(3.0,4.0);
        assert!(c.approx_eq(Vec2::new(4.0,6.0)));
    }

    #[test]
    fn vec2_dot_product() {
        let d = Vec2::new(1.0,0.0).dot(Vec2::new(0.0,1.0));
        assert!(approx_eq(d, 0.0));
    }

    #[test]
    fn vec2_normalize_unit_length() {
        let n = Vec2::new(3.0,4.0).normalize();
        assert!(approx_eq(n.length(), 1.0));
    }

    #[test]
    fn vec2_perpendicular_is_orthogonal() {
        let v = Vec2::new(1.0,0.0);
        let p = v.perpendicular();
        assert!(approx_eq(v.dot(p), 0.0));
    }

    #[test]
    fn vec2_lerp_midpoint() {
        let m = Vec2::ZERO.lerp(Vec2::new(2.0,4.0), 0.5);
        assert!(m.approx_eq(Vec2::new(1.0,2.0)));
    }

    #[test]
    fn vec2_zero_normalize_returns_zero() {
        let n = Vec2::ZERO.normalize();
        assert!(n.approx_eq(Vec2::ZERO));
    }

    // ── Vec3 ──────────────────────────────────────────────────────────────────

    #[test]
    fn vec3_size_is_16_bytes() {
        assert_eq!(std::mem::size_of::<Vec3>(), 16);
    }

    #[test]
    fn vec3_align_is_16() {
        assert_eq!(std::mem::align_of::<Vec3>(), 16);
    }

    #[test]
    fn vec3_cross_product_basis() {
        let z = Vec3::X.cross(Vec3::Y);
        assert!(z.approx_eq(Vec3::Z), "X × Y = {:?}, expected Z", z);
    }

    #[test]
    fn vec3_cross_anticommutative() {
        let a = Vec3::new(1.0,2.0,3.0);
        let b = Vec3::new(4.0,5.0,6.0);
        let sum = a.cross(b) + b.cross(a);
        assert!(sum.approx_eq(Vec3::ZERO), "a×b + b×a = {:?}", sum);
    }

    #[test]
    fn vec3_normalize_unit_length() {
        let n = Vec3::new(1.0,2.0,3.0).normalize();
        assert!(approx_eq(n.length(), 1.0), "|n| = {}", n.length());
    }

    #[test]
    fn vec3_reflect() {
        let v = Vec3::new(1.0,-1.0,0.0).normalize();
        let r = v.reflect(Vec3::Y);
        assert!(approx_eq(r.y, -v.y), "reflect y: {} vs {}", r.y, -v.y);
    }

    #[test]
    fn vec3_distance() {
        let d = Vec3::ZERO.distance(Vec3::new(3.0,4.0,0.0));
        assert!(approx_eq(d, 5.0), "distance = {}", d);
    }

    #[test]
    fn vec3_equality_ignores_padding_lane() {
        // Both represent the same (x,y,z) — PartialEq must ignore lane 3.
        let a = Vec3::new(1.0,2.0,3.0);
        let b = Vec3::new(1.0,2.0,3.0);
        assert_eq!(a, b);
    }

    // ── Vec4 ──────────────────────────────────────────────────────────────────

    #[test]
    fn vec4_size_is_16_bytes() {
        assert_eq!(std::mem::size_of::<Vec4>(), 16);
    }

    #[test]
    fn vec4_dot_with_self_is_length_sq() {
        let v = Vec4::new(1.0,2.0,3.0,4.0);
        assert!(approx_eq(v.dot(v), v.length_sq()));
    }

    #[test]
    fn vec4_normalize_unit_length() {
        let n = Vec4::new(1.0,2.0,3.0,4.0).normalize();
        assert!(approx_eq(n.length(), 1.0), "|n| = {}", n.length());
    }

    #[test]
    fn vec3_extend_and_truncate_roundtrip() {
        let v3 = Vec3::new(1.0, 2.0, 3.0);
        let v4 = v3.extend(4.0);
        let back = v4.truncate();
        assert!(back.approx_eq(v3));
    }
}
