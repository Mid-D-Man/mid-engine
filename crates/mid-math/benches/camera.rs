// crates/mid-math/benches/camera.rs
//! Benchmarks for camera math utilities — Frustum, projections, unproject, CSM.
//!
//! Groups:
//!   frustum_tests        — sphere/AABB/point culling latency
//!   frustum_batch_100k   — batch culling throughput (engine-critical path)
//!   projection_ops       — decompose, resize, infinite, reversed-Z
//!   unproject            — screen → world, picking ray construction
//!   csm                  — split depth generation + sub-frustum corners
//!
//! Run: cargo bench --bench camera -p mid-math
//! HTML: target/criterion/report/index.html

use criterion::{
    black_box, criterion_group, criterion_main,
    BatchSize, Criterion, Throughput,
};

use mid_math::{Mat4, Vec3, Vec4, to_radians};
use mid_math::camera::{
    Frustum,
    projection::{
        csm_split_depths, perspective_decompose, perspective_infinite_rh,
        perspective_resize, perspective_reversed_z_rh, picking_ray,
        sub_frustum_corners, unproject,
    },
};

// ─────────────────────────────────────────────────────────────────────────────
// Shared setup
// ─────────────────────────────────────────────────────────────────────────────

fn make_vp() -> Mat4 {
    let view = Mat4::look_at_rh(Vec3::new(0.0, 5.0, 10.0), Vec3::ZERO, Vec3::Y);
    let proj = Mat4::perspective_rh(to_radians(60.0), 16.0 / 9.0, 0.1, 500.0);
    proj * view
}

fn make_frustum() -> Frustum {
    Frustum::from_view_proj(make_vp())
}

// ─────────────────────────────────────────────────────────────────────────────
// Group 1: frustum single-call tests
// ─────────────────────────────────────────────────────────────────────────────

fn bench_frustum_tests(c: &mut Criterion) {
    let mut g = c.benchmark_group("frustum_tests");
    let f = make_frustum();

    let point  = Vec3::new(1.0, 1.0, 0.0);
    let center = Vec3::new(0.0, 0.0, 0.0);
    let radius = 2.0f32;
    let aabb_min = Vec3::new(-1.0, -1.0, -1.0);
    let aabb_max = Vec3::new( 1.0,  1.0,  1.0);

    g.bench_function("test_point",              |b| b.iter(|| f.test_point(black_box(point))));
    g.bench_function("test_sphere",             |b| b.iter(|| f.test_sphere(black_box(center), black_box(radius))));
    g.bench_function("test_sphere_visibility",  |b| b.iter(|| f.test_sphere_visibility(black_box(center), black_box(radius))));
    g.bench_function("test_aabb",               |b| b.iter(|| f.test_aabb(black_box(aabb_min), black_box(aabb_max))));
    g.bench_function("test_aabb_visibility",    |b| b.iter(|| f.test_aabb_visibility(black_box(aabb_min), black_box(aabb_max))));
    g.bench_function("from_view_proj",          |b| b.iter(|| Frustum::from_view_proj(black_box(make_vp()))));

    g.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Group 2: batch frustum culling — 100k entities
//
// This is the engine-critical path. Each entity has an AABB.
// We measure ns/entity for both AABB and sphere culling.
// Phase 2 target: vectorise with Vec3x4 to test 4 AABBs per SSE2 op.
// ─────────────────────────────────────────────────────────────────────────────

fn bench_frustum_batch_100k(c: &mut Criterion) {
    const N: usize = 100_000;
    let mut g = c.benchmark_group("frustum_batch_100k");
    g.throughput(Throughput::Elements(N as u64));

    let f = make_frustum();

    // Generate N AABBs scattered around the scene — roughly half visible.
    let aabbs: Vec<(Vec3, Vec3)> = (0..N)
        .map(|i| {
            let x = ((i % 100) as f32 - 50.0) * 3.0;
            let z = ((i / 100) as f32 - 50.0) * 3.0;
            let half = 0.5f32;
            (Vec3::new(x - half, -half, z - half),
             Vec3::new(x + half,  half, z + half))
        })
        .collect();

    // Spheres: same centres, radius 1.
    let spheres: Vec<(Vec3, f32)> = aabbs.iter()
        .map(|(mn, mx)| ((*mn + *mx) * 0.5, 1.0f32))
        .collect();

    // AABB batch.
    g.bench_function("aabb_cull", |b| {
        b.iter_batched(
            || aabbs.clone(),
            |boxes| {
                let mut visible = 0u32;
                for (mn, mx) in boxes {
                    if f.test_aabb(black_box(mn), black_box(mx)) { visible += 1; }
                }
                black_box(visible)
            },
            BatchSize::LargeInput,
        )
    });

    // Sphere batch.
    g.bench_function("sphere_cull", |b| {
        b.iter_batched(
            || spheres.clone(),
            |sph| {
                let mut visible = 0u32;
                for (center, r) in sph {
                    if f.test_sphere(black_box(center), black_box(r)) { visible += 1; }
                }
                black_box(visible)
            },
            BatchSize::LargeInput,
        )
    });

    g.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Group 3: projection matrix operations
// ─────────────────────────────────────────────────────────────────────────────

fn bench_projection_ops(c: &mut Criterion) {
    let mut g = c.benchmark_group("projection_ops");

    let fov_y  = to_radians(60.0);
    let aspect = 16.0f32 / 9.0;
    let near   = 0.1f32;
    let far    = 500.0f32;

    let standard_proj = Mat4::perspective_rh(fov_y, aspect, near, far);

    g.bench_function("perspective_decompose", |b| {
        b.iter(|| perspective_decompose(black_box(standard_proj)))
    });

    g.bench_function("perspective_resize", |b| {
        b.iter(|| {
            let mut p = standard_proj;
            perspective_resize(&mut p, black_box(4.0f32 / 3.0));
            black_box(p)
        })
    });

    g.bench_function("perspective_infinite_rh", |b| {
        b.iter(|| perspective_infinite_rh(black_box(fov_y), black_box(aspect), black_box(near)))
    });

    g.bench_function("perspective_reversed_z_rh_finite", |b| {
        b.iter(|| perspective_reversed_z_rh(black_box(fov_y), black_box(aspect), black_box(near), black_box(far)))
    });

    g.bench_function("perspective_reversed_z_rh_infinite", |b| {
        b.iter(|| perspective_reversed_z_rh(black_box(fov_y), black_box(aspect), black_box(near), f32::INFINITY))
    });

    g.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Group 4: unproject and picking ray
// ─────────────────────────────────────────────────────────────────────────────

fn bench_unproject(c: &mut Criterion) {
    let mut g = c.benchmark_group("unproject");

    let view = Mat4::look_at_rh(Vec3::new(0.0, 0.0, 10.0), Vec3::ZERO, Vec3::Y);
    let proj = Mat4::perspective_rh(to_radians(60.0), 16.0 / 9.0, 0.1, 500.0);
    let inv_vp   = (proj * view).inverse().unwrap();
    let viewport = Vec4::new(0.0, 0.0, 1920.0, 1080.0);
    let window   = Vec3::new(960.0, 540.0, 0.5);

    g.bench_function("unproject_single", |b| {
        b.iter(|| unproject(black_box(window), black_box(inv_vp), black_box(viewport)))
    });

    g.bench_function("picking_ray", |b| {
        b.iter(|| picking_ray(black_box(960.0f32), black_box(540.0f32), black_box(inv_vp), black_box(viewport)))
    });

    // Cost of the full pipeline: build VP, invert, unproject.
    // This is what an editor does on every mouse event.
    g.bench_function("full_unproject_pipeline", |b| {
        b.iter(|| {
            let vp  = proj * view;
            let inv = vp.inverse().unwrap();
            unproject(black_box(window), black_box(inv), black_box(viewport))
        })
    });

    g.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Group 5: CSM utilities
// ─────────────────────────────────────────────────────────────────────────────

fn bench_csm(c: &mut Criterion) {
    let mut g = c.benchmark_group("csm");

    let view = Mat4::look_at_rh(Vec3::new(0.0, 0.0, 10.0), Vec3::ZERO, Vec3::Y);
    let proj = Mat4::perspective_rh(to_radians(60.0), 16.0 / 9.0, 0.1, 500.0);

    // Split depth generation at various cascade counts.
    for count in [2u32, 4, 8] {
        g.bench_function(format!("csm_split_depths_{}_cascades", count), |b| {
            b.iter(|| csm_split_depths(black_box(0.1f32), black_box(500.0f32), black_box(count as usize), black_box(0.5f32)))
        });
    }

    // Sub-frustum corner extraction for one cascade.
    g.bench_function("sub_frustum_corners", |b| {
        b.iter(|| sub_frustum_corners(black_box(view), black_box(proj), black_box(0.1f32), black_box(25.0f32)))
    });

    // Full CSM setup: generate 4 cascade splits + 4 sub-frustum corner sets.
    // This runs once per frame for shadow maps.
    g.bench_function("full_csm_4_cascades", |b| {
        b.iter(|| {
            let splits = csm_split_depths(0.1, 500.0, 4, 0.5);
            let mut total_corners = 0usize;
            for i in 0..4 {
                if let Some(c) = sub_frustum_corners(
                    black_box(view), black_box(proj),
                    black_box(splits[i]), black_box(splits[i + 1]),
                ) {
                    total_corners += c.len();
                }
            }
            black_box(total_corners)
        })
    });

    g.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Group 6: frustum world geometry
// ─────────────────────────────────────────────────────────────────────────────

fn bench_frustum_world_geometry(c: &mut Criterion) {
    let mut g = c.benchmark_group("frustum_world_geometry");

    let vp     = make_vp();
    let inv_vp = vp.inverse().unwrap();

    g.bench_function("world_corners", |b| {
        b.iter(|| Frustum::world_corners(black_box(inv_vp)))
    });

    g.bench_function("world_center", |b| {
        b.iter(|| Frustum::world_center(black_box(inv_vp)))
    });

    g.bench_function("world_aabb", |b| {
        b.iter(|| Frustum::world_aabb(black_box(inv_vp)))
    });

    g.finish();
}

// ─────────────────────────────────────────────────────────────────────────────

criterion_group!(
    benches,
    bench_frustum_tests,
    bench_frustum_batch_100k,
    bench_projection_ops,
    bench_unproject,
    bench_csm,
    bench_frustum_world_geometry,
);
criterion_main!(benches);
