// crates/mid-math/src/ffi/noise.rs
//! C-ABI exports for coherent noise generators.
//!
//! Design: stateless pure functions — no heap, no opaque handles.
//! Configuration passed as plain C scalars.
//!
//! noise_type (u32): 0=Perlin  1=Simplex  2=Value  3=Worley
//!
//! DistanceMode (u32, Worley only):
//!   0=F1  1=F2  2=F2MinusF1  3=F1PlusF2
//!
//! DistanceMetric (u32, Worley only):
//!   0=Euclidean  1=Manhattan  2=Chebyshev  3=Minkowski
//!
//! # Safety contract (all pointer arguments)
//!   Non-null, valid for stated element count, caller owns memory.

use crate::noise::{
    DomainWarp, Fbm, 
    Perlin, Simplex, Value, Worley,
    worley::{DistanceMode, DistanceMetric},
};

// ── helpers ───────────────────────────────────────────────────────────────────

#[inline(always)]
fn dist_mode(v: u32) -> DistanceMode {
    match v {
        1 => DistanceMode::F2,
        2 => DistanceMode::F2MinusF1,
        3 => DistanceMode::F1PlusF2,
        _ => DistanceMode::F1,
    }
}

#[inline(always)]
fn dist_metric(v: u32) -> DistanceMetric {
    match v {
        1 => DistanceMetric::Manhattan,
        2 => DistanceMetric::Chebyshev,
        3 => DistanceMetric::Minkowski,
        _ => DistanceMetric::Euclidean,
    }
}

#[inline(always)]
fn perlin(seed: u64) -> Perlin {
    if seed == 0 { Perlin::new() } else { Perlin::from_seed(seed) }
}
#[inline(always)]
fn simplex(seed: u64) -> Simplex {
    if seed == 0 { Simplex::new() } else { Simplex::from_seed(seed) }
}
#[inline(always)]
fn value(seed: u64) -> Value {
    if seed == 0 { Value::new() } else { Value::from_seed(seed) }
}
#[inline(always)]
fn worley_configured(seed: u64, mode: u32, metric: u32) -> Worley {
    let w = if seed == 0 { Worley::new() } else { Worley::from_seed(seed) };
    w.with_mode(dist_mode(mode)).with_metric(dist_metric(metric))
}

// ═══════════════════════════════════════════════════════════════════════════
//  Perlin
// ═══════════════════════════════════════════════════════════════════════════

#[no_mangle] pub extern "C" fn mid_perlin_sample_2d(x: f32, y: f32) -> f32 { Perlin::new().sample_2d(x, y) }
#[no_mangle] pub extern "C" fn mid_perlin_seeded_sample_2d(seed: u64, x: f32, y: f32) -> f32 { perlin(seed).sample_2d(x, y) }
#[no_mangle] pub extern "C" fn mid_perlin_sample_3d(x: f32, y: f32, z: f32) -> f32 { Perlin::new().sample_3d(x, y, z) }
#[no_mangle] pub extern "C" fn mid_perlin_seeded_sample_3d(seed: u64, x: f32, y: f32, z: f32) -> f32 { perlin(seed).sample_3d(x, y, z) }
#[no_mangle] pub extern "C" fn mid_perlin_sample_4d(x: f32, y: f32, z: f32, w: f32) -> f32 { Perlin::new().sample_4d(x, y, z, w) }
#[no_mangle] pub extern "C" fn mid_perlin_seeded_sample_4d(seed: u64, x: f32, y: f32, z: f32, w: f32) -> f32 { perlin(seed).sample_4d(x, y, z, w) }

/// Batch 2D Perlin into `out[0..count]`.
/// # Safety — xs, ys, out non-null, valid for count elements.
#[no_mangle]
pub unsafe extern "C" fn mid_perlin_sample_2d_batch(
    seed: u64, xs: *const f32, ys: *const f32, out: *mut f32, count: u32,
) {
    let n = count as usize;
    let g = perlin(seed);
    let xs  = core::slice::from_raw_parts(xs, n);
    let ys  = core::slice::from_raw_parts(ys, n);
    let out = core::slice::from_raw_parts_mut(out, n);
    for i in 0..n { out[i] = g.sample_2d(xs[i], ys[i]); }
}

/// Batch 3D Perlin into `out[0..count]`.
/// # Safety — all pointers non-null, valid for count elements.
#[no_mangle]
pub unsafe extern "C" fn mid_perlin_sample_3d_batch(
    seed: u64, xs: *const f32, ys: *const f32, zs: *const f32,
    out: *mut f32, count: u32,
) {
    let n = count as usize;
    let g = perlin(seed);
    let xs  = core::slice::from_raw_parts(xs, n);
    let ys  = core::slice::from_raw_parts(ys, n);
    let zs  = core::slice::from_raw_parts(zs, n);
    let out = core::slice::from_raw_parts_mut(out, n);
    for i in 0..n { out[i] = g.sample_3d(xs[i], ys[i], zs[i]); }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Simplex
// ═══════════════════════════════════════════════════════════════════════════

#[no_mangle] pub extern "C" fn mid_simplex_sample_2d(x: f32, y: f32) -> f32 { Simplex::new().sample_2d(x, y) }
#[no_mangle] pub extern "C" fn mid_simplex_seeded_sample_2d(seed: u64, x: f32, y: f32) -> f32 { simplex(seed).sample_2d(x, y) }
#[no_mangle] pub extern "C" fn mid_simplex_sample_3d(x: f32, y: f32, z: f32) -> f32 { Simplex::new().sample_3d(x, y, z) }
#[no_mangle] pub extern "C" fn mid_simplex_seeded_sample_3d(seed: u64, x: f32, y: f32, z: f32) -> f32 { simplex(seed).sample_3d(x, y, z) }
#[no_mangle] pub extern "C" fn mid_simplex_sample_4d(x: f32, y: f32, z: f32, w: f32) -> f32 { Simplex::new().sample_4d(x, y, z, w) }
#[no_mangle] pub extern "C" fn mid_simplex_seeded_sample_4d(seed: u64, x: f32, y: f32, z: f32, w: f32) -> f32 { simplex(seed).sample_4d(x, y, z, w) }

/// Batch 2D Simplex.
/// # Safety — all pointers non-null, valid for count.
#[no_mangle]
pub unsafe extern "C" fn mid_simplex_sample_2d_batch(
    seed: u64, xs: *const f32, ys: *const f32, out: *mut f32, count: u32,
) {
    let n = count as usize;
    let g = simplex(seed);
    let xs  = core::slice::from_raw_parts(xs, n);
    let ys  = core::slice::from_raw_parts(ys, n);
    let out = core::slice::from_raw_parts_mut(out, n);
    for i in 0..n { out[i] = g.sample_2d(xs[i], ys[i]); }
}

/// Batch 3D Simplex.
/// # Safety — all pointers non-null, valid for count.
#[no_mangle]
pub unsafe extern "C" fn mid_simplex_sample_3d_batch(
    seed: u64, xs: *const f32, ys: *const f32, zs: *const f32,
    out: *mut f32, count: u32,
) {
    let n = count as usize;
    let g = simplex(seed);
    let xs  = core::slice::from_raw_parts(xs, n);
    let ys  = core::slice::from_raw_parts(ys, n);
    let zs  = core::slice::from_raw_parts(zs, n);
    let out = core::slice::from_raw_parts_mut(out, n);
    for i in 0..n { out[i] = g.sample_3d(xs[i], ys[i], zs[i]); }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Value
// ═══════════════════════════════════════════════════════════════════════════

#[no_mangle] pub extern "C" fn mid_value_noise_sample_2d(x: f32, y: f32) -> f32 { Value::new().sample_2d(x, y) }
#[no_mangle] pub extern "C" fn mid_value_noise_seeded_sample_2d(seed: u64, x: f32, y: f32) -> f32 { value(seed).sample_2d(x, y) }
#[no_mangle] pub extern "C" fn mid_value_noise_sample_3d(x: f32, y: f32, z: f32) -> f32 { Value::new().sample_3d(x, y, z) }
#[no_mangle] pub extern "C" fn mid_value_noise_seeded_sample_3d(seed: u64, x: f32, y: f32, z: f32) -> f32 { value(seed).sample_3d(x, y, z) }
#[no_mangle] pub extern "C" fn mid_value_noise_sample_4d(x: f32, y: f32, z: f32, w: f32) -> f32 { Value::new().sample_4d(x, y, z, w) }
#[no_mangle] pub extern "C" fn mid_value_noise_seeded_sample_4d(seed: u64, x: f32, y: f32, z: f32, w: f32) -> f32 { value(seed).sample_4d(x, y, z, w) }

// ═══════════════════════════════════════════════════════════════════════════
//  Worley
// ═══════════════════════════════════════════════════════════════════════════

/// Sample 2D Worley. mode: 0=F1 1=F2 2=F2-F1 3=F1+F2. metric: 0=Euclid 1=Manhattan 2=Chebyshev 3=Minkowski.
#[no_mangle]
pub extern "C" fn mid_worley_sample_2d(seed: u64, mode: u32, metric: u32, x: f32, y: f32) -> f32 {
    worley_configured(seed, mode, metric).sample_2d(x, y)
}

/// Sample 3D Worley.
#[no_mangle]
pub extern "C" fn mid_worley_sample_3d(seed: u64, mode: u32, metric: u32, x: f32, y: f32, z: f32) -> f32 {
    worley_configured(seed, mode, metric).sample_3d(x, y, z)
}

/// Batch 2D Worley.
/// # Safety — all pointers non-null, valid for count.
#[no_mangle]
pub unsafe extern "C" fn mid_worley_sample_2d_batch(
    seed: u64, mode: u32, metric: u32,
    xs: *const f32, ys: *const f32, out: *mut f32, count: u32,
) {
    let n = count as usize;
    let g = worley_configured(seed, mode, metric);
    let xs  = core::slice::from_raw_parts(xs, n);
    let ys  = core::slice::from_raw_parts(ys, n);
    let out = core::slice::from_raw_parts_mut(out, n);
    for i in 0..n { out[i] = g.sample_2d(xs[i], ys[i]); }
}

/// Batch 3D Worley.
/// # Safety — all pointers non-null, valid for count.
#[no_mangle]
pub unsafe extern "C" fn mid_worley_sample_3d_batch(
    seed: u64, mode: u32, metric: u32,
    xs: *const f32, ys: *const f32, zs: *const f32,
    out: *mut f32, count: u32,
) {
    let n = count as usize;
    let g = worley_configured(seed, mode, metric);
    let xs  = core::slice::from_raw_parts(xs, n);
    let ys  = core::slice::from_raw_parts(ys, n);
    let zs  = core::slice::from_raw_parts(zs, n);
    let out = core::slice::from_raw_parts_mut(out, n);
    for i in 0..n { out[i] = g.sample_3d(xs[i], ys[i], zs[i]); }
}

// ═══════════════════════════════════════════════════════════════════════════
//  fBm — Fractional Brownian Motion
// ═══════════════════════════════════════════════════════════════════════════

/// Sample 2D fBm (Simplex base). Output ≈ [-1, 1].
#[no_mangle]
pub extern "C" fn mid_fbm_simplex_sample_2d(
    seed: u64, octaves: u32, lacunarity: f32, gain: f32, frequency: f32,
    x: f32, y: f32,
) -> f32 {
    Fbm::new(simplex(seed))
        .octaves(octaves).lacunarity(lacunarity).gain(gain).frequency(frequency)
        .sample_2d(x, y)
}

/// Sample 3D fBm (Simplex base).
#[no_mangle]
pub extern "C" fn mid_fbm_simplex_sample_3d(
    seed: u64, octaves: u32, lacunarity: f32, gain: f32, frequency: f32,
    x: f32, y: f32, z: f32,
) -> f32 {
    Fbm::new(simplex(seed))
        .octaves(octaves).lacunarity(lacunarity).gain(gain).frequency(frequency)
        .sample_3d(x, y, z)
}

/// Sample 2D fBm (Perlin base).
#[no_mangle]
pub extern "C" fn mid_fbm_perlin_sample_2d(
    seed: u64, octaves: u32, lacunarity: f32, gain: f32, frequency: f32,
    x: f32, y: f32,
) -> f32 {
    Fbm::new(perlin(seed))
        .octaves(octaves).lacunarity(lacunarity).gain(gain).frequency(frequency)
        .sample_2d(x, y)
}

/// Sample 3D fBm (Perlin base).
#[no_mangle]
pub extern "C" fn mid_fbm_perlin_sample_3d(
    seed: u64, octaves: u32, lacunarity: f32, gain: f32, frequency: f32,
    x: f32, y: f32, z: f32,
) -> f32 {
    Fbm::new(perlin(seed))
        .octaves(octaves).lacunarity(lacunarity).gain(gain).frequency(frequency)
        .sample_3d(x, y, z)
}

/// Batch 2D fBm (Simplex base).
/// # Safety — all pointers non-null, valid for count.
#[no_mangle]
pub unsafe extern "C" fn mid_fbm_simplex_sample_2d_batch(
    seed: u64, octaves: u32, lacunarity: f32, gain: f32, frequency: f32,
    xs: *const f32, ys: *const f32, out: *mut f32, count: u32,
) {
    let n = count as usize;
    let g = Fbm::new(simplex(seed))
        .octaves(octaves).lacunarity(lacunarity).gain(gain).frequency(frequency);
    let xs  = core::slice::from_raw_parts(xs, n);
    let ys  = core::slice::from_raw_parts(ys, n);
    let out = core::slice::from_raw_parts_mut(out, n);
    for i in 0..n { out[i] = g.sample_2d(xs[i], ys[i]); }
}

/// Batch 3D fBm (Simplex base).
/// # Safety — all pointers non-null, valid for count.
#[no_mangle]
pub unsafe extern "C" fn mid_fbm_simplex_sample_3d_batch(
    seed: u64, octaves: u32, lacunarity: f32, gain: f32, frequency: f32,
    xs: *const f32, ys: *const f32, zs: *const f32,
    out: *mut f32, count: u32,
) {
    let n = count as usize;
    let g = Fbm::new(simplex(seed))
        .octaves(octaves).lacunarity(lacunarity).gain(gain).frequency(frequency);
    let xs  = core::slice::from_raw_parts(xs, n);
    let ys  = core::slice::from_raw_parts(ys, n);
    let zs  = core::slice::from_raw_parts(zs, n);
    let out = core::slice::from_raw_parts_mut(out, n);
    for i in 0..n { out[i] = g.sample_3d(xs[i], ys[i], zs[i]); }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Domain Warp
// ═══════════════════════════════════════════════════════════════════════════

/// Sample 2D domain-warped noise (Simplex base).
/// `double_warp`: 0 = single pass, 1 = double pass (richer, slower).
#[no_mangle]
pub extern "C" fn mid_domain_warp_simplex_sample_2d(
    seed: u64, octaves: u32, lacunarity: f32, gain: f32, frequency: f32,
    warp_scale: f32, double_warp: u32,
    x: f32, y: f32,
) -> f32 {
    let fbm = Fbm::new(simplex(seed))
        .octaves(octaves).lacunarity(lacunarity).gain(gain).frequency(frequency);
    DomainWarp::new(simplex(seed))
        .with_fbm(fbm)
        .warp_scale(warp_scale)
        .double_warp(double_warp != 0)
        .sample_2d(x, y)
}

/// Sample 3D domain-warped noise (Simplex base).
#[no_mangle]
pub extern "C" fn mid_domain_warp_simplex_sample_3d(
    seed: u64, octaves: u32, lacunarity: f32, gain: f32, frequency: f32,
    warp_scale: f32, double_warp: u32,
    x: f32, y: f32, z: f32,
) -> f32 {
    let fbm = Fbm::new(simplex(seed))
        .octaves(octaves).lacunarity(lacunarity).gain(gain).frequency(frequency);
    DomainWarp::new(simplex(seed))
        .with_fbm(fbm)
        .warp_scale(warp_scale)
        .double_warp(double_warp != 0)
        .sample_3d(x, y, z)
}

/// Batch 2D domain warp (Simplex base).
/// # Safety — all pointers non-null, valid for count.
#[no_mangle]
pub unsafe extern "C" fn mid_domain_warp_simplex_sample_2d_batch(
    seed: u64, octaves: u32, lacunarity: f32, gain: f32, frequency: f32,
    warp_scale: f32, double_warp: u32,
    xs: *const f32, ys: *const f32, out: *mut f32, count: u32,
) {
    let n   = count as usize;
    let fbm = Fbm::new(simplex(seed))
        .octaves(octaves).lacunarity(lacunarity).gain(gain).frequency(frequency);
    let warp = DomainWarp::new(simplex(seed))
        .with_fbm(fbm).warp_scale(warp_scale).double_warp(double_warp != 0);
    let xs  = core::slice::from_raw_parts(xs, n);
    let ys  = core::slice::from_raw_parts(ys, n);
    let out = core::slice::from_raw_parts_mut(out, n);
    for i in 0..n { out[i] = warp.sample_2d(xs[i], ys[i]); }
  }
