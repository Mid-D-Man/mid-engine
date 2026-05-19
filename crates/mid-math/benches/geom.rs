// crates/mid-math/benches/geom.rs
//! Benchmarks for geometric primitives — barycentric, Triangle2, Triangle3.
//!
//! Groups:
//!   barycentric_ops       — interpolation at single-call latency
//!   triangle2_ops         — 2D triangle tests (contains, area, circumcircle)
//!   triangle3_ops         — 3D triangle tests (ray intersect, closest point)
//!   ray_batch_100k        — sustained mesh-picking throughput (100k triangles)
//!   aabb_generation_100k  — building per-triangle AABBs for BVH input
//!
//! Run: cargo bench --bench geom -p mid-math
//! HTML: target/criterion/report/index.html

use criterion::{
    black_box, criterion_group, criterion_main,
    BatchSize, Criterion, Throughput,
};

use mid_math::{Vec2, Vec3};
use mid_math::geom::barycentric::{
    BarycentricCoords, Triangle2, Triangle3,
    signed_area_2d, triangle_area_3d,
};

// ─────────────────────────────────────────────────────────────────────────────
// Shared geometry
// ─────────────────────────────────────────────────────────────────────────────

fn unit_tri2() -> Triangle2 {
    Triangle2::new(Vec2::new(0.0, 0.0), Vec2::new(1.0, 0.0), Vec2::new(0.0, 1.0))
}

fn unit_tri3() -> Triangle3 {
    Triangle3::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0))
}

fn tilted_tri3() -> Triangle3 {
    // More realistic: tilted in 3D space.
    Triangle3::new(
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(2.0, 0.5, 0.0),
        Vec3::new(0.5, 2.0, 1.0),
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// Group 1: BarycentricCoords operations
// ─────────────────────────────────────────────────────────────────────────────

fn bench_barycentric_ops(c: &mut Criterion) {
    let mut g = c.benchmark_group("barycentric_ops");

    let bary = BarycentricCoords::new(0.25, 0.35, 0.40);
    let va   = Vec3::new(1.0, 0.0, 0.0);
    let vb   = Vec3::new(0.0, 1.0, 0.0);
    let vc   = Vec3::new(0.0, 0.0, 1.0);
    let uv_a = Vec2::new(0.0, 0.0);
    let uv_b = Vec2::new(1.0, 0.0);
    let uv_c = Vec2::new(0.0, 1.0);

    g.bench_function("interpolate_f32",  |b| b.iter(|| bary.interpolate_f32(black_box(1.0), black_box(2.0), black_box(3.0))));
    g.bench_function("interpolate_vec2", |b| b.iter(|| bary.interpolate_vec2(black_box(uv_a), black_box(uv_b), black_box(uv_c))));
    g.bench_function("interpolate_vec3", |b| b.iter(|| bary.interpolate_vec3(black_box(va), black_box(vb), black_box(vc))));
    g.bench_function("is_inside",        |b| b.iter(|| bary.is_inside()));
    g.bench_function("is_inside_or_on_edge", |b| b.iter(|| bary.is_inside_or_on_edge()));
    g.bench_function("is_valid",         |b| b.iter(|| bary.is_valid()));

    g.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Group 2: Triangle2 operations
// ─────────────────────────────────────────────────────────────────────────────

fn bench_triangle2_ops(c: &mut Criterion) {
    let mut g = c.benchmark_group("triangle2_ops");

    let t = unit_tri2();
    let interior = Vec2::new(0.2, 0.2);
    let exterior = Vec2::new(2.0, 2.0);

    g.bench_function("barycentric_interior",  |b| b.iter(|| t.barycentric(black_box(interior))));
    g.bench_function("barycentric_exterior",  |b| b.iter(|| t.barycentric(black_box(exterior))));
    g.bench_function("contains_interior",     |b| b.iter(|| t.contains(black_box(interior))));
    g.bench_function("contains_exterior",     |b| b.iter(|| t.contains(black_box(exterior))));
    g.bench_function("signed_area",           |b| b.iter(|| t.signed_area()));
    g.bench_function("area",                  |b| b.iter(|| t.area()));
    g.bench_function("centroid",              |b| b.iter(|| t.centroid()));
    g.bench_function("circumcircle",          |b| b.iter(|| t.circumcircle()));
    g.bench_function("circumcircle_contains", |b| b.iter(|| t.circumcircle_contains(black_box(interior))));

    // signed_area_2d free function — used in Delaunay orientation tests.
    g.bench_function("signed_area_2d_free", |b| {
        b.iter(|| signed_area_2d(black_box(t.a), black_box(t.b), black_box(t.c)))
    });

    g.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Group 3: Triangle3 operations
// ─────────────────────────────────────────────────────────────────────────────

fn bench_triangle3_ops(c: &mut Criterion) {
    let mut g = c.benchmark_group("triangle3_ops");

    let t_flat   = unit_tri3();
    let t_tilted = tilted_tri3();
    let interior = Vec3::new(0.2, 0.2, 0.0);
    let origin   = Vec3::new(0.2, 0.2, 5.0);
    let dir      = Vec3::new(0.0, 0.0, -1.0);
    let dir_miss = Vec3::new(0.0, 0.0,  1.0);  // Away from triangle.
    let above    = Vec3::new(0.3, 0.3, 3.0);

    // Normal and area.
    g.bench_function("normal",     |b| b.iter(|| t_flat.normal()));
    g.bench_function("area",       |b| b.iter(|| t_flat.area()));
    g.bench_function("centroid",   |b| b.iter(|| t_flat.centroid()));

    // Barycentric.
    g.bench_function("barycentric_flat",   |b| b.iter(|| t_flat.barycentric(black_box(interior))));
    g.bench_function("barycentric_tilted", |b| b.iter(|| t_tilted.barycentric(black_box(above))));

    // Ray intersection — hit case.
    g.bench_function("ray_intersect_hit_no_cull",  |b| {
        b.iter(|| t_flat.ray_intersect(black_box(origin), black_box(dir), false))
    });
    g.bench_function("ray_intersect_hit_cull",     |b| {
        b.iter(|| t_flat.ray_intersect(black_box(origin), black_box(dir), true))
    });
    // Ray intersection — miss case (early out path).
    g.bench_function("ray_intersect_miss",         |b| {
        b.iter(|| t_flat.ray_intersect(black_box(origin), black_box(dir_miss), false))
    });
    // Tilted triangle — exercises dominant-axis projection in barycentric.
    g.bench_function("ray_intersect_tilted",       |b| {
        let tilted_origin = Vec3::new(0.5, 0.5, 5.0);
        let tilted_dir    = Vec3::new(0.0, 0.0, -1.0);
        b.iter(|| t_tilted.ray_intersect(black_box(tilted_origin), black_box(tilted_dir), false))
    });

    // Closest point.
    g.bench_function("closest_point_interior", |b| {
        b.iter(|| t_flat.closest_point(black_box(above)))
    });
    g.bench_function("closest_point_exterior", |b| {
        let far = Vec3::new(10.0, 10.0, 5.0);
        b.iter(|| t_flat.closest_point(black_box(far)))
    });

    // Plane.
    g.bench_function("plane_equation",         |b| b.iter(|| t_flat.plane_equation()));
    g.bench_function("plane_distance",         |b| b.iter(|| t_flat.plane_distance(black_box(above))));

    // Free function.
    g.bench_function("triangle_area_3d_free", |b| {
        b.iter(|| triangle_area_3d(black_box(t_flat.a), black_box(t_flat.b), black_box(t_flat.c)))
    });

    g.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Group 4: 100k triangle ray batch — mesh picking hot path
//
// The engine scenario: given a mesh of 100k triangles, find the closest
// intersection for a single picking ray. This benchmark measures raw
// throughput and is the target for BVH acceleration in mid-geom.
// ─────────────────────────────────────────────────────────────────────────────

fn bench_ray_batch_100k(c: &mut Criterion) {
    const N: usize = 100_000;
    let mut g = c.benchmark_group("ray_batch_100k");
    g.throughput(Throughput::Elements(N as u64));

    // Generate N triangles in a flat grid — realistic mesh scenario.
    let triangles: Vec<Triangle3> = (0..N)
        .map(|i| {
            let row = (i / 100) as f32;
            let col = (i % 100) as f32;
            let x = col * 0.1;
            let z = row * 0.1;
            Triangle3::new(
                Vec3::new(x,        0.0, z      ),
                Vec3::new(x + 0.1, 0.0, z      ),
                Vec3::new(x,        0.0, z + 0.1),
            )
        })
        .collect();

    let origin = Vec3::new(5.0, 5.0, 5.0);  // Ray from above.
    let dir    = Vec3::new(0.0, -1.0, 0.0); // Pointing down — hits many triangles.

    // Brute-force ray vs all triangles — no BVH, no culling.
    // This is the baseline that BVH in mid-geom will beat.
    g.bench_function("brute_force_no_cull", |b| {
        b.iter_batched(
            || triangles.clone(),
            |tris| {
                let mut best_t = f32::MAX;
                let mut hits   = 0u32;
                for tri in &tris {
                    if let Some((t, _)) = tri.ray_intersect(
                        black_box(origin), black_box(dir), false,
                    ) {
                        if t < best_t { best_t = t; }
                        hits += 1;
                    }
                }
                black_box((hits, best_t))
            },
            BatchSize::LargeInput,
        )
    });

    // With back-face culling — realistic for opaque meshes.
    g.bench_function("brute_force_with_cull", |b| {
        b.iter_batched(
            || triangles.clone(),
            |tris| {
                let mut best_t = f32::MAX;
                let mut hits   = 0u32;
                for tri in &tris {
                    if let Some((t, _)) = tri.ray_intersect(
                        black_box(origin), black_box(dir), true,
                    ) {
                        if t < best_t { best_t = t; }
                        hits += 1;
                    }
                }
                black_box((hits, best_t))
            },
            BatchSize::LargeInput,
        )
    });

    g.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Group 5: per-triangle AABB generation — BVH build input
//
// Building a BVH requires an AABB per primitive. This measures how fast
// we can compute min/max bounds for 100k triangles — BVH build cost baseline.
// ─────────────────────────────────────────────────────────────────────────────

fn bench_aabb_generation_100k(c: &mut Criterion) {
    const N: usize = 100_000;
    let mut g = c.benchmark_group("aabb_generation_100k");
    g.throughput(Throughput::Elements(N as u64));

    let triangles: Vec<Triangle3> = (0..N)
        .map(|i| {
            let row = (i / 100) as f32;
            let col = (i % 100) as f32;
            let x = col * 0.1;
            let z = row * 0.1;
            Triangle3::new(
                Vec3::new(x,       0.0, z      ),
                Vec3::new(x + 0.1, 0.0, z      ),
                Vec3::new(x,       0.0, z + 0.1),
            )
        })
        .collect();

    g.bench_function("min_max_per_triangle", |b| {
        b.iter_batched(
            || triangles.clone(),
            |tris| {
                let aabbs: Vec<(Vec3, Vec3)> = tris.iter().map(|t| {
                    let mn = t.a.min(t.b).min(t.c);
                    let mx = t.a.max(t.b).max(t.c);
                    (mn, mx)
                }).collect();
                black_box(aabbs)
            },
            BatchSize::LargeInput,
        )
    });

    g.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Group 6: Delaunay circumcircle batch — 10k triangles
//
// Bowyer-Watson Delaunay triangulation iterates circumcircle_contains
// for every existing triangle when inserting a new point.
// Worst case per-insert is O(n). This benchmark measures the raw predicate cost.
// ─────────────────────────────────────────────────────────────────────────────

fn bench_delaunay_circumcircle_10k(c: &mut Criterion) {
    const N: usize = 10_000;
    let mut g = c.benchmark_group("delaunay_circumcircle_10k");
    g.throughput(Throughput::Elements(N as u64));

    let triangles: Vec<Triangle2> = (0..N)
        .map(|i| {
            let x = (i % 100) as f32 * 0.1;
            let y = (i / 100) as f32 * 0.1;
            Triangle2::new(
                Vec2::new(x, y),
                Vec2::new(x + 0.1, y),
                Vec2::new(x, y + 0.1),
            )
        })
        .collect();

    let query = Vec2::new(0.05, 0.05);

    g.bench_function("circumcircle_contains_batch", |b| {
        b.iter_batched(
            || triangles.clone(),
            |tris| {
                let mut count = 0u32;
                for t in &tris {
                    if t.circumcircle_contains(black_box(query)) { count += 1; }
                }
                black_box(count)
            },
            BatchSize::LargeInput,
        )
    });

    g.finish();
}

// ─────────────────────────────────────────────────────────────────────────────

criterion_group!(
    benches,
    bench_barycentric_ops,
    bench_triangle2_ops,
    bench_triangle3_ops,
    bench_ray_batch_100k,
    bench_aabb_generation_100k,
    bench_delaunay_circumcircle_10k,
);
criterion_main!(benches);
