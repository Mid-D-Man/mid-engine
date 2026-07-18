// crates/mid-math/src/camera/csm.rs
//! Cascaded Shadow Map (CSM/PSSM) utilities (not chainsaw man ) — cascade split depths and
//! per-cascade world-space frustum corners.

use crate::{Mat4, MidVec, Vec3, Vec4};
use super::projection::perspective_decompose;

/// Inline capacity for `csm_split_depths`'s return value. Generous upper
/// bound on shadow cascade count for real-time rendering — most engines use
/// 3-4, few go past 6-8. `count` itself is NOT capped at this value (see
/// below); this only controls when the result spills to the heap.
pub const CSM_N: usize = 8;

/// Generate `count` cascade split depths for Cascaded Shadow Maps (CSM/PSSM).
///
/// Uses the **practical split scheme** (logarithmic + linear blend):
/// ```text
/// split_log    = near * (far/near) ^ (i/count)
/// split_linear = near + (far-near) * (i/count)
/// split_i      = lambda * split_log + (1-lambda) * split_linear
/// ```
///
/// - `lambda = 0.0` → fully linear
/// - `lambda = 1.0` → fully logarithmic
/// - `lambda = 0.5` → NVIDIA recommended default
///
/// Returns `count + 1` values: `[near, split_1, ..., split_{count-1}, far]`.
///
/// `count` is caller-controlled with no compile-time upper bound (this
/// function is exposed over FFI — see `ffi::camera::mid_camera_csm_split_depths`
/// — so an unexpectedly large `count` must degrade gracefully, not panic).
/// The result stays inline for the first `CSM_N` cascades and transparently
/// spills to the heap past that; callers never need to know which happened.
pub fn csm_split_depths(near: f32, far: f32, count: usize, lambda: f32) -> MidVec<f32, CSM_N> {
    assert!(count >= 1, "CSM requires at least 1 cascade");
    assert!(far > near, "CSM: far must be > near");
    // The logarithmic term divides by `near`; only require near > 0 when
    // that term is actually used. A pure linear split (lambda == 0) has no
    // such dependency, so near == 0 is valid input in that case.
    assert!(lambda <= 0.0 || near > 0.0,
        "CSM: near must be > 0 when lambda > 0 (logarithmic term divides by near)");

    let mut splits = MidVec::with_capacity(count + 1);
    splits.push(near);

    for i in 1..count {
        let p     = i as f32 / count as f32;
        let c_lin = near + (far - near) * p;
        // Skip the log term entirely when lambda <= 0: besides being wasted
        // work, `near * (far/near).powf(p)` is NaN at near == 0 (0 * inf),
        // and `lambda * NaN` stays NaN even when lambda is 0.0.
        let split = if lambda <= 0.0 {
            c_lin
        } else {
            let c_log = near * (far / near).powf(p);
            lambda * c_log + (1.0 - lambda) * c_lin
        };
        splits.push(split);
    }

    splits.push(far);
    splits
}

/// Compute the 8 world-space corners of a sub-frustum between `near` and `far`.
///
/// Used to build tight per-cascade bounding boxes for shadow map projection.
/// Returns `None` if `proj` is not a valid RH perspective matrix or singular.
pub fn sub_frustum_corners(
    view: Mat4,
    proj: Mat4,
    near: f32,
    far: f32,
) -> Option<[Vec3; 8]> {
    let p = perspective_decompose(proj)?;

    let sub_proj = Mat4::perspective_rh(p.fov_y, p.aspect, near, far);
    let inv_vp = (sub_proj * view).inverse()?;

    // z ∈ [0,1] (ZO — matches Mat4::perspective_rh, which sub_proj uses).
    let ndc: [Vec4; 8] = [
        // near face: z = 0
        Vec4::new(-1., -1., 0., 1.), Vec4::new( 1., -1., 0., 1.),
        Vec4::new( 1.,  1., 0., 1.), Vec4::new(-1.,  1., 0., 1.),
        // far face: z = 1
        Vec4::new(-1., -1., 1., 1.), Vec4::new( 1., -1., 1., 1.),
        Vec4::new( 1.,  1., 1., 1.), Vec4::new(-1.,  1., 1., 1.),
    ];

    let mut corners = [Vec3::ZERO; 8];
    for (i, &c) in ndc.iter().enumerate() {
        let w  = inv_vp * c;
        let iw = 1.0 / w.w;
        corners[i] = Vec3::new(w.x * iw, w.y * iw, w.z * iw);
    }
    Some(corners)
                  }
