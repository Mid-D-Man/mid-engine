// crates/mid-math/src/ffi/curves.rs
//! C-ABI exports for curve and spline evaluation.
//!
//! Design: stateless pure functions — no heap allocation at the API level,
//! no opaque handles. Control points are passed as pointer + count.
//!
//! All functions operate on Vec3 (CVec3) or f32 scalars.
//! The caller owns all memory — no allocations are retained after return.
//!
//! # CatmullRom alpha modes
//!   0 = Uniform      — fastest, risk of cusps on uneven spacing
//!   1 = Centripetal  — recommended, no cusps or self-intersections
//!   2 = Chordal      — maximally arc-length-proportional
//!   other            — treated as Centripetal
//!
//! # Safety contract (applies to all pointer arguments)
//!   - Must be non-null
//!   - Must point to `count` (or `n_segments + 1`) valid, initialised values
//!   - Must be aligned to the type's alignment (CVec3 is #[repr(C)], align 4)
//!   - The memory must remain valid for the duration of the call

use crate::ffi::float32::CVec3;
use crate::{
    BSpline, CardinalSpline, CatmullRom, CatmullRomAlpha, CubicBezier,
    HermiteKey, HermiteSpline, QuadraticBezier, Vec3,
};

// ── Internal helpers ──────────────────────────────────────────────────────────

#[inline(always)]
fn alpha_from_u32(mode: u32) -> CatmullRomAlpha {
    match mode {
        0 => CatmullRomAlpha::Uniform,
        2 => CatmullRomAlpha::Chordal,
        _ => CatmullRomAlpha::Centripetal,
    }
}

/// Read `count` CVec3 values from a raw C pointer into a `Vec<Vec3>`.
///
/// # Safety
/// `ptr` must be non-null and valid for `count` initialised CVec3 reads.
#[inline]
unsafe fn read_vec3s(ptr: *const CVec3, count: u32) -> Vec<Vec3> {
    core::slice::from_raw_parts(ptr, count as usize)
        .iter()
        .map(|&v| Vec3::from(v))
        .collect()
}

/// Write a `Vec<Vec3>` buffer into a raw C output pointer.
///
/// # Safety
/// `out` must be non-null and valid for `buf.len()` CVec3 writes.
#[inline]
unsafe fn write_vec3s(buf: Vec<Vec3>, out: *mut CVec3) {
    let out_slice = core::slice::from_raw_parts_mut(out, buf.len());
    for (i, v) in buf.into_iter().enumerate() {
        out_slice[i] = v.into();
    }
}

// ── Quadratic Bézier ──────────────────────────────────────────────────────────

/// Evaluate a quadratic Bézier at `t ∈ [0, 1]`.
///
/// The curve passes through `p0` (t=0) and `p2` (t=1). `p1` is the control handle.
#[no_mangle]
pub extern "C" fn mid_quadratic_bezier_vec3_evaluate(
    p0: CVec3, p1: CVec3, p2: CVec3,
    t: f32,
) -> CVec3 {
    QuadraticBezier::new(Vec3::from(p0), Vec3::from(p1), Vec3::from(p2))
        .evaluate(t)
        .into()
}

/// Tangent direction of a quadratic Bézier at `t ∈ [0, 1]`.
#[no_mangle]
pub extern "C" fn mid_quadratic_bezier_vec3_tangent(
    p0: CVec3, p1: CVec3, p2: CVec3,
    t: f32,
) -> CVec3 {
    QuadraticBezier::new(Vec3::from(p0), Vec3::from(p1), Vec3::from(p2))
        .tangent(t)
        .into()
}

/// Evaluate a quadratic Bézier scalar at `t ∈ [0, 1]`.
#[no_mangle]
pub extern "C" fn mid_quadratic_bezier_f32_evaluate(
    p0: f32, p1: f32, p2: f32,
    t: f32,
) -> f32 {
    QuadraticBezier::new(p0, p1, p2).evaluate(t)
}

// ── Cubic Bézier ──────────────────────────────────────────────────────────────

/// Evaluate a cubic Bézier at `t ∈ [0, 1]`.
///
/// The curve passes through `p0` (t=0) and `p3` (t=1). `p1` and `p2` are handles.
#[no_mangle]
pub extern "C" fn mid_cubic_bezier_vec3_evaluate(
    p0: CVec3, p1: CVec3, p2: CVec3, p3: CVec3,
    t: f32,
) -> CVec3 {
    CubicBezier::new(Vec3::from(p0), Vec3::from(p1), Vec3::from(p2), Vec3::from(p3))
        .evaluate(t)
        .into()
}

/// Tangent direction of a cubic Bézier at `t ∈ [0, 1]`.
#[no_mangle]
pub extern "C" fn mid_cubic_bezier_vec3_tangent(
    p0: CVec3, p1: CVec3, p2: CVec3, p3: CVec3,
    t: f32,
) -> CVec3 {
    CubicBezier::new(Vec3::from(p0), Vec3::from(p1), Vec3::from(p2), Vec3::from(p3))
        .tangent(t)
        .into()
}

/// Evaluate a cubic Bézier scalar at `t ∈ [0, 1]`.
#[no_mangle]
pub extern "C" fn mid_cubic_bezier_f32_evaluate(
    p0: f32, p1: f32, p2: f32, p3: f32,
    t: f32,
) -> f32 {
    CubicBezier::new(p0, p1, p2, p3).evaluate(t)
}

/// Approximate arc length of a cubic Bézier via `segments` linear samples.
#[no_mangle]
pub extern "C" fn mid_cubic_bezier_vec3_arc_length(
    p0: CVec3, p1: CVec3, p2: CVec3, p3: CVec3,
    segments: u32,
) -> f32 {
    CubicBezier::new(Vec3::from(p0), Vec3::from(p1), Vec3::from(p2), Vec3::from(p3))
        .arc_length(segments as usize)
}

/// Sample `n_segments + 1` uniformly-spaced points from a cubic Bézier.
///
/// Writes exactly `n_segments + 1` CVec3 values into `out`.
/// Caller must allocate `(n_segments + 1) * sizeof(CVec3)` bytes.
///
/// # Safety
/// `out` must be non-null and valid for `n_segments + 1` CVec3 writes.
#[no_mangle]
pub unsafe extern "C" fn mid_cubic_bezier_vec3_sample(
    p0: CVec3, p1: CVec3, p2: CVec3, p3: CVec3,
    out: *mut CVec3,
    n_segments: u32,
) {
    let n = n_segments as usize;
    let mut buf = vec![Vec3::ZERO; n + 1];
    CubicBezier::new(Vec3::from(p0), Vec3::from(p1), Vec3::from(p2), Vec3::from(p3))
        .sample_uniform(n, &mut buf);
    write_vec3s(buf, out);
}

// ── Catmull-Rom ───────────────────────────────────────────────────────────────

/// Evaluate a Catmull-Rom spline at parameter `t`.
///
/// `t` ranges over `[0, count - 1]` — each integer lands on a control point.
/// Requires `count >= 2`.
///
/// `alpha`: 0=Uniform, 1=Centripetal (recommended), 2=Chordal.
///
/// # Safety
/// `points` must be non-null and valid for `count` CVec3 reads. `count >= 2`.
#[no_mangle]
pub unsafe extern "C" fn mid_catmull_rom_vec3_evaluate(
    points: *const CVec3,
    count: u32,
    t: f32,
    alpha: u32,
) -> CVec3 {
    let pts = read_vec3s(points, count);
    CatmullRom::with_alpha(pts, alpha_from_u32(alpha))
        .evaluate(t)
        .into()
}

/// Sample `n_segments + 1` uniformly-spaced points from a Catmull-Rom spline.
///
/// Writes exactly `n_segments + 1` CVec3 values into `out`.
/// Caller must allocate `(n_segments + 1) * sizeof(CVec3)` bytes.
///
/// # Safety
/// `points` valid for `count`. `out` valid for `n_segments + 1`. `count >= 2`.
#[no_mangle]
pub unsafe extern "C" fn mid_catmull_rom_vec3_sample(
    points: *const CVec3,
    count: u32,
    alpha: u32,
    out: *mut CVec3,
    n_segments: u32,
) {
    let pts = read_vec3s(points, count);
    let cr  = CatmullRom::with_alpha(pts, alpha_from_u32(alpha));
    let n   = n_segments as usize;
    let mut buf = vec![Vec3::ZERO; n + 1];
    cr.sample_uniform(n, &mut buf);
    write_vec3s(buf, out);
}

/// Number of segments for a Catmull-Rom spline with `count` points: `max(0, count - 1)`.
#[no_mangle]
pub extern "C" fn mid_catmull_rom_segment_count(count: u32) -> u32 {
    count.saturating_sub(1)
}

// ── Hermite ───────────────────────────────────────────────────────────────────

/// Evaluate a Hermite spline at parameter `t ∈ [0, count - 1]`.
///
/// Three parallel arrays describe the keyframes:
///   `positions`    — world-space position of each keyframe
///   `tangents_out` — outgoing tangent (velocity leaving this key)
///   `tangents_in`  — incoming tangent (velocity arriving at this key)
///
/// For smooth C¹ keys: `tangents_in[i] == tangents_out[i]`.
/// For corner keys: set them independently.
/// Requires `count >= 2`.
///
/// # Safety
/// All three arrays must be non-null and valid for `count` CVec3 reads.
#[no_mangle]
pub unsafe extern "C" fn mid_hermite_vec3_evaluate(
    positions:    *const CVec3,
    tangents_out: *const CVec3,
    tangents_in:  *const CVec3,
    count: u32,
    t: f32,
) -> CVec3 {
    let n = count as usize;
    let pos  = core::slice::from_raw_parts(positions,    n);
    let tout = core::slice::from_raw_parts(tangents_out, n);
    let tin  = core::slice::from_raw_parts(tangents_in,  n);
    let keys: Vec<HermiteKey<Vec3>> = (0..n)
        .map(|i| HermiteKey {
            position:    Vec3::from(pos[i]),
            tangent_out: Vec3::from(tout[i]),
            tangent_in:  Vec3::from(tin[i]),
        })
        .collect();
    HermiteSpline::new(keys).evaluate(t).into()
}

/// Sample `n_segments + 1` uniformly-spaced points from a Hermite spline.
///
/// Writes exactly `n_segments + 1` CVec3 values into `out`.
///
/// # Safety
/// All input arrays valid for `count`. `out` valid for `n_segments + 1`. `count >= 2`.
#[no_mangle]
pub unsafe extern "C" fn mid_hermite_vec3_sample(
    positions:    *const CVec3,
    tangents_out: *const CVec3,
    tangents_in:  *const CVec3,
    count: u32,
    out: *mut CVec3,
    n_segments: u32,
) {
    let n = count as usize;
    let pos  = core::slice::from_raw_parts(positions,    n);
    let tout = core::slice::from_raw_parts(tangents_out, n);
    let tin  = core::slice::from_raw_parts(tangents_in,  n);
    let keys: Vec<HermiteKey<Vec3>> = (0..n)
        .map(|i| HermiteKey {
            position:    Vec3::from(pos[i]),
            tangent_out: Vec3::from(tout[i]),
            tangent_in:  Vec3::from(tin[i]),
        })
        .collect();
    let spline = HermiteSpline::new(keys);
    let ns = n_segments as usize;
    let mut buf = vec![Vec3::ZERO; ns + 1];
    spline.sample_uniform(ns, &mut buf);
    write_vec3s(buf, out);
}

// ── B-Spline ──────────────────────────────────────────────────────────────────

/// Evaluate a uniform cubic B-spline at parameter `t ∈ [0, segment_count]`.
///
/// The curve does NOT pass through control points.
/// `segment_count = count - 3`. Requires `count >= 4`.
///
/// # Safety
/// `points` must be non-null and valid for `count` CVec3 reads. `count >= 4`.
#[no_mangle]
pub unsafe extern "C" fn mid_bspline_vec3_evaluate(
    points: *const CVec3,
    count: u32,
    t: f32,
) -> CVec3 {
    let pts = read_vec3s(points, count);
    BSpline::new(pts).evaluate(t).into()
}

/// Sample `n_segments + 1` uniformly-spaced points from a B-spline.
///
/// # Safety
/// `points` valid for `count`. `out` valid for `n_segments + 1`. `count >= 4`.
#[no_mangle]
pub unsafe extern "C" fn mid_bspline_vec3_sample(
    points: *const CVec3,
    count: u32,
    out: *mut CVec3,
    n_segments: u32,
) {
    let pts    = read_vec3s(points, count);
    let spline = BSpline::new(pts);
    let ns     = n_segments as usize;
    let mut buf = vec![Vec3::ZERO; ns + 1];
    spline.sample_uniform(ns, &mut buf);
    write_vec3s(buf, out);
}

/// Number of segments for a B-spline with `count` control points: `max(0, count - 3)`.
#[no_mangle]
pub extern "C" fn mid_bspline_segment_count(count: u32) -> u32 {
    count.saturating_sub(3)
}

// ── Cardinal spline ───────────────────────────────────────────────────────────

/// Evaluate a Cardinal spline at parameter `t ∈ [0, count - 1]`.
///
/// `tension = 0.0` is identical to Catmull-Rom (centripetal-equivalent tangents).
/// `tension = 1.0` produces piecewise-linear interpolation.
/// Values outside `[0, 1]` produce overshoot.
/// Requires `count >= 2`.
///
/// # Safety
/// `points` must be non-null and valid for `count` CVec3 reads. `count >= 2`.
#[no_mangle]
pub unsafe extern "C" fn mid_cardinal_vec3_evaluate(
    points: *const CVec3,
    count: u32,
    tension: f32,
    t: f32,
) -> CVec3 {
    let pts = read_vec3s(points, count);
    CardinalSpline::new(pts, tension).evaluate(t).into()
}

/// Sample `n_segments + 1` uniformly-spaced points from a Cardinal spline.
///
/// # Safety
/// `points` valid for `count`. `out` valid for `n_segments + 1`. `count >= 2`.
#[no_mangle]
pub unsafe extern "C" fn mid_cardinal_vec3_sample(
    points: *const CVec3,
    count: u32,
    tension: f32,
    out: *mut CVec3,
    n_segments: u32,
) {
    let pts    = read_vec3s(points, count);
    let spline = CardinalSpline::new(pts, tension);
    let ns     = n_segments as usize;
    let mut buf = vec![Vec3::ZERO; ns + 1];
    spline.sample_uniform(ns, &mut buf);
    write_vec3s(buf, out);
}
