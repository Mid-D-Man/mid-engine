// crates/mid-math/src/tests/stress_tests.rs
#[cfg(test)]
mod tests {
    use std::time::Instant;
    use crate::{Vec3, Vec4, Quat, Mat4, to_radians, lerp, smoothstep, approx_eq, EPSILON};

    const BUILD_MODE: &str = if cfg!(debug_assertions) { "[DEBUG]" } else { "[RELEASE]" };

    // ── Scalar utilities ──────────────────────────────────────────────────────

    #[test]
    fn lerp_scalar_midpoint() {
        assert!((lerp(0.0, 10.0, 0.5) - 5.0).abs() < EPSILON);
    }

    #[test]
    fn smoothstep_clamps_outside_range() {
        assert!((smoothstep(0.0,1.0,-1.0) - 0.0).abs() < EPSILON);
        assert!((smoothstep(0.0,1.0, 2.0) - 1.0).abs() < EPSILON);
    }

    #[test]
    fn smoothstep_midpoint_is_half() {
        assert!((smoothstep(0.0,1.0,0.5) - 0.5).abs() < EPSILON);
    }

    // ── Vec3 stress ───────────────────────────────────────────────────────────

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
        println!("  {} Vec3 adds in {:.3}ms  ({:.1} ns/op)  final_x={:.0}  {}",
            count*2, elapsed.as_secs_f64()*1000.0,
            elapsed.as_nanos() as f64/(count*2) as f64, acc.x, BUILD_MODE);
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
        println!("  {} Vec3 dot products in {:.3}ms  ({:.1} ns/op)  {}",
            count, elapsed.as_secs_f64()*1000.0,
            elapsed.as_nanos() as f64/count as f64, BUILD_MODE);
    }

    #[test]
    fn stress_100k_vec3_cross() {
        let count = 100_000usize;
        let start = Instant::now();
        let mut acc = Vec3::ZERO;
        for _ in 0..count { acc = acc + Vec3::X.cross(Vec3::Y); }
        let elapsed = start.elapsed();
        let _ = std::hint::black_box(acc);
        println!("  {} Vec3 cross in {:.3}ms  ({:.1} ns/op)  {}",
            count, elapsed.as_secs_f64()*1000.0,
            elapsed.as_nanos() as f64/count as f64, BUILD_MODE);
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
        println!("  {} Vec3 normalize in {:.3}ms  ({:.1} ns/op)  {}",
            count, elapsed.as_secs_f64()*1000.0,
            elapsed.as_nanos() as f64/count as f64, BUILD_MODE);
    }

    #[test]
    fn stress_100k_vec3_lerp() {
        let count = 100_000usize;
        let start = Instant::now();
        let mut acc = Vec3::ZERO;
        for i in 0..count {
            let t = i as f32 / count as f32;
            acc = acc + Vec3::ZERO.lerp(Vec3::ONE, t);
        }
        let elapsed = start.elapsed();
        let _ = std::hint::black_box(acc);
        println!("  {} Vec3 lerp in {:.3}ms  ({:.1} ns/op)  {}",
            count, elapsed.as_secs_f64()*1000.0,
            elapsed.as_nanos() as f64/count as f64, BUILD_MODE);
    }

    // ── Quat stress ───────────────────────────────────────────────────────────

    #[test]
    fn stress_100k_quat_rotate() {
        let q     = Quat::from_axis_angle(Vec3::Y, to_radians(45.0));
        let count = 100_000usize;
        let start = Instant::now();
        let mut acc = Vec3::ZERO;
        for _ in 0..count { acc = acc + q.rotate(Vec3::X); }
        let elapsed = start.elapsed();
        let _ = std::hint::black_box(acc);
        let ms = elapsed.as_secs_f64()*1000.0;
        println!("  {} Quat rotations in {:.3}ms  ({:.1} ns/op)  {}",
            count, ms, elapsed.as_nanos() as f64/count as f64, BUILD_MODE);
        println!("  ECS 60Hz budget=16.6ms — 100k rotations took {:.3}ms ({})",
            ms, if ms < 16.6 {"✓ within budget"} else {"⚠ over budget"});
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
        println!("  {} Quat mul in {:.3}ms  ({:.1} ns/op)  {}",
            count*2, elapsed.as_secs_f64()*1000.0,
            elapsed.as_nanos() as f64/(count*2) as f64, BUILD_MODE);
    }

    #[test]
    fn stress_50k_quat_slerp() {
        let a     = Quat::from_axis_angle(Vec3::Y, to_radians(0.0));
        let b     = Quat::from_axis_angle(Vec3::Y, to_radians(90.0));
        let count = 50_000usize;
        let start = Instant::now();
        let mut acc = Quat::IDENTITY;
        for i in 0..count {
            let t = i as f32 / count as f32;
            acc = acc * a.slerp(b, t);
        }
        let elapsed = start.elapsed();
        let _ = std::hint::black_box(acc);
        println!("  {} Quat slerp in {:.3}ms  ({:.1} ns/op)  {}",
            count, elapsed.as_secs_f64()*1000.0,
            elapsed.as_nanos() as f64/count as f64, BUILD_MODE);
    }

    #[test]
    fn stress_50k_euler_from_to_roundtrip() {
        let count = 50_000usize;
        let start = Instant::now();
        let mut acc = 0.0f32;
        for i in 0..count {
            let f = i as f32 * 0.0001;
            let q = Quat::from_euler(f, f*0.7, f*1.3);
            let (r,p,y) = q.to_euler();
            acc += r + p + y;
        }
        let elapsed = start.elapsed();
        let _ = std::hint::black_box(acc);
        assert!(acc.is_finite());
        println!("  {} euler round-trips in {:.3}ms  ({:.1} ns/op)  {}",
            count, elapsed.as_secs_f64()*1000.0,
            elapsed.as_nanos() as f64/count as f64, BUILD_MODE);
    }

    // ── Mat4 stress ───────────────────────────────────────────────────────────

    #[test]
    fn stress_10k_mat4_mul() {
        let a     = Mat4::from_rotation(Quat::from_axis_angle(Vec3::Y, to_radians(0.01)));
        let b     = Mat4::from_rotation(Quat::from_axis_angle(Vec3::X, to_radians(0.01)));
        let count = 10_000usize;
        let start = Instant::now();
        let mut acc = Mat4::IDENTITY;
        for _ in 0..count { acc = acc * a * b; }
        let elapsed = start.elapsed();
        let acc = std::hint::black_box(acc);
        assert!(acc.cols[0][0].is_finite());
        println!("  {} Mat4 mul in {:.3}ms  ({:.1} ns/op)  {}",
            count*2, elapsed.as_secs_f64()*1000.0,
            elapsed.as_nanos() as f64/(count*2) as f64, BUILD_MODE);
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
        println!("  {} Mat4 transform_point in {:.3}ms  ({:.1} ns/op)  {}",
            count, elapsed.as_secs_f64()*1000.0,
            elapsed.as_nanos() as f64/count as f64, BUILD_MODE);
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
        assert_eq!(passed, count);
        let ns = elapsed.as_nanos() as f64 / count as f64;
        println!("  {} Mat4 general inverse in {:.3}ms  ({:.1} ns/op)  {}",
            count, elapsed.as_secs_f64()*1000.0, ns, BUILD_MODE);
        if !cfg!(debug_assertions) {
            println!("  Scalar baseline 117.1 ns/op — speedup: {:.1}×", 117.1/ns);
        }
    }

    #[test]
    fn stress_5k_mat4_inverse_trs() {
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
        let ns = elapsed.as_nanos() as f64 / count as f64;
        println!("  {} Mat4 inverse_trs in {:.3}ms  ({:.1} ns/op)  {}",
            count, elapsed.as_secs_f64()*1000.0, ns, BUILD_MODE);
        if !cfg!(debug_assertions) {
            println!("  Scalar baseline 81.8 ns/op — speedup: {:.1}×", 81.8/ns);
        }
    }

    #[test]
    fn stress_100k_entity_transform_simulation() {
        let trs = Mat4::from_trs(
            Vec3::new(1.0,0.0,0.0),
            Quat::from_axis_angle(Vec3::Y, to_radians(45.0)),
            Vec3::ONE,
        );
        let mut positions: Vec<Vec3> = (0..100_000)
            .map(|i| Vec3::new(i as f32*0.01, 0.0, 0.0))
            .collect();
        let start = Instant::now();
        for p in positions.iter_mut() { *p = trs.transform_point(*p); }
        let elapsed = start.elapsed();
        let ms = elapsed.as_secs_f64()*1000.0;
        let _ = std::hint::black_box(&positions);
        println!("  100k entity transforms in {:.3}ms  ({:.1} ns/entity)  {}",
            ms, elapsed.as_nanos() as f64/100_000.0, BUILD_MODE);
        println!("  ECS 60Hz budget=16.6ms — ({})",
            if ms < 16.6 {"✓ within budget"} else {"⚠ over budget"});
    }

    #[test]
    fn stress_mixed_math_1k_frames_simulation() {
        let ticks = 1_000usize;
        let start = Instant::now();
        let mut total_pos = Vec3::ZERO;
        for tick in 0..ticks {
            let t = tick as f32 * 0.016;
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
        println!("  {} ticks × (TRS + 10 transforms + 10 lerps) in {:.3}ms  ({:.1} µs/tick)  {}",
            ticks, elapsed.as_secs_f64()*1000.0,
            elapsed.as_secs_f64()*1_000_000.0/ticks as f64, BUILD_MODE);
    }
}
