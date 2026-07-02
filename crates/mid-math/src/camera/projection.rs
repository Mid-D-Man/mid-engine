// crates/mid-math/src/camera/projection.rs
//! Extended projection matrix utilities.
//!
//! Supplements `Mat4`'s built-in projection constructors with:
//!   - `unproject`                  — window coords → world position
//!   - `picking_ray`                — click → world-space ray
//!   - `perspective_infinite_rh`    — infinite far plane (RH)
//!   - `perspective_reversed_z_rh`  — reversed depth (RH, Vulkan/DX12)
//!   - `perspective_infinite_lh`    — infinite far plane (LH)
//!   - `perspective_reversed_z_lh`  — reversed depth (LH, DX12/Metal)
//!   - `perspective_decompose`      — read back fov / aspect / near / far
//!   - `perspective_resize`         — cheaply update aspect ratio
//!   - `csm_split_depths`           — logarithmic+linear CSM splits
//!   - `sub_frustum_corners`        — world-space corners for one CSM cascade

use crate::{Mat4, Vec3, Vec4};

// ── Decomposed parameters ─────────────────────────────────────────────────────

/// Parameters reconstructed from a perspective projection matrix.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PerspectiveParams {
    /// Vertical field of view in radians.
    pub fov_y:  f32,
    /// Aspect ratio (width / height).
    pub aspect: f32,
    /// Near clipping distance (positive, in world units).
    pub near:   f32,
    /// Far clipping distance. `f32::INFINITY` indicates an infinite projection.
    pub far:    f32,
}

// ── Unproject ─────────────────────────────────────────────────────────────────

/// Unproject a window-space position back to world space.
///
/// # Parameters
/// - `window_pos` — `(x, y, depth)`.
///   - `x`, `y`: pixel coordinates relative to the viewport's top-left origin.
///   - `depth`:  value read from the depth buffer, in `[0, 1]`.
/// - `inv_view_proj` — the inverse of `(proj * view)`.
/// - `viewport` — `Vec4(x, y, width, height)` of the viewport rectangle.
pub fn unproject(window_pos: Vec3, inv_view_proj: Mat4, viewport: Vec4) -> Vec3 {
    let ndc_x = 2.0 * (window_pos.x - viewport.x) / viewport.z - 1.0;
    let ndc_y = 2.0 * (window_pos.y - viewport.y) / viewport.w - 1.0;
    let ndc_z = 2.0 *  window_pos.z - 1.0;

    let clip  = Vec4::new(ndc_x, ndc_y, ndc_z, 1.0);
    let world = inv_view_proj * clip;

    let iw = 1.0 / world.w;
    Vec3::new(world.x * iw, world.y * iw, world.z * iw)
}

/// Convenience wrapper: unproject using separate view and projection matrices.
///
/// Returns `None` if the view-projection matrix is singular.
pub fn unproject_separate(
    window_pos: Vec3,
    view: Mat4,
    proj: Mat4,
    viewport: Vec4,
) -> Option<Vec3> {
    let vp  = proj * view;
    let inv = vp.inverse()?;
    Some(unproject(window_pos, inv, viewport))
}

/// Build a world-space picking ray from a window-space mouse position.
///
/// Returns `(ray_origin, ray_direction)` where direction is NOT normalised.
pub fn picking_ray(
    mouse_x: f32,
    mouse_y: f32,
    inv_view_proj: Mat4,
    viewport: Vec4,
) -> (Vec3, Vec3) {
    let near_pt = unproject(Vec3::new(mouse_x, mouse_y, 0.0), inv_view_proj, viewport);
    let far_pt  = unproject(Vec3::new(mouse_x, mouse_y, 1.0), inv_view_proj, viewport);
    (near_pt, far_pt - near_pt)
}

// ── Right-handed infinite / reversed-Z ───────────────────────────────────────

/// Right-hand perspective with an **infinite far plane**.
///
/// Clip space: `[-1, 1]` (OpenGL / mid-math default).
pub fn perspective_infinite_rh(fov_y: f32, aspect: f32, near: f32) -> Mat4 {
    let f = 1.0 / (fov_y * 0.5).tan();
    Mat4::from_cols(
        [f / aspect, 0.0,  0.0,          0.0],
        [0.0,        f,    0.0,          0.0],
        [0.0,        0.0, -1.0,         -1.0],
        [0.0,        0.0, -2.0 * near,   0.0],
    )
}

/// Right-hand **reversed-Z** perspective projection.
///
/// Maps near → depth `1.0` and far → depth `0.0`.
/// Requires a reversed depth test (`GREATER` or `GREATER_OR_EQUAL`).
/// Pass `far = f32::INFINITY` for maximum precision.
pub fn perspective_reversed_z_rh(fov_y: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
    let f = 1.0 / (fov_y * 0.5).tan();

    if far.is_infinite() {
        // Infinite reversed-Z RH: near → 1, far → 0
        Mat4::from_cols(
            [f / aspect, 0.0,  0.0,   0.0],
            [0.0,        f,    0.0,   0.0],
            [0.0,        0.0,  0.0,  -1.0],
            [0.0,        0.0,  near,  0.0],
        )
    } else {
        // Finite reversed-Z RH
        let z = far - near;
        Mat4::from_cols(
            [f / aspect, 0.0,  0.0,                  0.0],
            [0.0,        f,    0.0,                  0.0],
            [0.0,        0.0,  near / z,             -1.0],
            [0.0,        0.0,  near * far / z,        0.0],
        )
    }
}

// ── Left-handed infinite / reversed-Z ────────────────────────────────────────

/// Left-hand perspective with an **infinite far plane**.
///
/// Suitable for DirectX 12, Metal, and left-handed Vulkan configurations.
/// Depth range `[0, 1]`. Maps near → 0, far → 1 (approaching from infinity).
///
/// ```text
/// col[2][2] = 1.0   (limit of far/(far-near) as far → ∞)
/// col[3][2] = -near (limit of -near·far/(far-near) as far → ∞)
/// col[2][3] = +1.0  (LH w-divide drives positive z forward)
/// ```
pub fn perspective_infinite_lh(fov_y: f32, aspect: f32, near: f32) -> Mat4 {
    let f = 1.0 / (fov_y * 0.5).tan();
    Mat4::from_cols(
        [f / aspect, 0.0,  0.0,    0.0],
        [0.0,        f,    0.0,    0.0],
        [0.0,        0.0,  1.0,    1.0],
        [0.0,        0.0, -near,   0.0],
    )
}

/// Left-hand **reversed-Z** perspective projection.
///
/// Maps near → depth `1.0` and far → depth `0.0`.
/// Requires a reversed depth test (`GREATER` or `GREATER_OR_EQUAL`).
/// Pass `far = f32::INFINITY` for an infinite reversed-Z LH projection
/// (best precision for large open worlds on DirectX 12 / Metal).
///
/// # Derivation
/// Standard LH maps near→0, far→1. Reversed-Z swaps the endpoints:
/// ```text
/// A = near / (near - far)       (col[2][2])
/// B = near · far / (far - near) (col[3][2])
/// col[2][3] = +1.0              (LH perspective divide)
/// ```
/// Verify: z_ndc(near) = A + B/near = 1.0 ✓   z_ndc(far) = A + B/far = 0.0 ✓
pub fn perspective_reversed_z_lh(fov_y: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
    let f = 1.0 / (fov_y * 0.5).tan();

    if far.is_infinite() {
        // Infinite reversed-Z LH: near → 1, far → 0
        // A = 0, B = near  (limits as far → ∞)
        Mat4::from_cols(
            [f / aspect, 0.0, 0.0,   0.0],
            [0.0,        f,   0.0,   0.0],
            [0.0,        0.0, 0.0,   1.0],
            [0.0,        0.0, near,  0.0],
        )
    } else {
        // Finite reversed-Z LH
        Mat4::from_cols(
            [f / aspect, 0.0, 0.0,                         0.0],
            [0.0,        f,   0.0,                         0.0],
            [0.0,        0.0, near / (near - far),          1.0],
            [0.0,        0.0, near * far / (far - near),    0.0],
        )
    }
}

// ── Decompose ─────────────────────────────────────────────────────────────────

/// Decompose a right-hand perspective projection matrix (clip `[-1, 1]`)
/// back into its constituent parameters.
///
/// Returns `None` if `proj` is not a valid RH perspective matrix.
///
/// Field mapping (Build 8 Vec4-field layout):
///   col 0 = x_axis, col 1 = y_axis, col 2 = z_axis, col 3 = w_axis
///   row 0 = .x,     row 1 = .y,     row 2 = .z,     row 3 = .w
pub fn perspective_decompose(proj: Mat4) -> Option<PerspectiveParams> {
    // RH perspective: z_axis.w must be -1.0 (the perspective divide sentinel).
    // Previously: proj.cols[2][3]  →  col 2, row 3  →  z_axis.w
    if (proj.z_axis.w + 1.0).abs() > 1e-4 { return None; }

    // proj.cols[1][1]  →  col 1, row 1  →  y_axis.y  (vertical scale = f)
    let f: f32 = proj.y_axis.y;
    if f < 1e-6 { return None; }

    // proj.cols[0][0]  →  col 0, row 0  →  x_axis.x  (horizontal scale = f/aspect)
    let aspect = f / proj.x_axis.x;
    let fov_y  = 2.0_f32 * (1.0_f32 / f).atan();

    // proj.cols[2][2]  →  col 2, row 2  →  z_axis.z
    // proj.cols[3][2]  →  col 3, row 2  →  w_axis.z
    let a = proj.z_axis.z;
    let b = proj.w_axis.z;

    let near = b / (a - 1.0);
    let far  = b / (a + 1.0);

    if near <= 0.0 || far <= near { return None; }

    Some(PerspectiveParams { fov_y, aspect, near, far })
}

// ── Resize ────────────────────────────────────────────────────────────────────

/// Update only the aspect ratio of an existing perspective projection matrix.
///
/// Works for both RH and LH projections — only the horizontal scale changes.
///
/// Field mapping (Build 8 Vec4-field layout):
///   x_axis.x = col 0, row 0  (horizontal scale = f/aspect)
///   y_axis.y = col 1, row 1  (vertical scale   = f)
#[inline]
pub fn perspective_resize(proj: &mut Mat4, new_aspect: f32) {
    if proj.x_axis.x == 0.0 || new_aspect == 0.0 { return; }
    proj.x_axis.x = proj.y_axis.y / new_aspect;
}

// ── Cascaded Shadow Maps ──────────────────────────────────────────────────────

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
pub fn csm_split_depths(near: f32, far: f32, count: usize, lambda: f32) -> Vec<f32> {
    assert!(count >= 1, "CSM requires at least 1 cascade");
    assert!(far > near, "CSM: far must be > near");
    // The logarithmic term divides by `near`; only require near > 0 when
    // that term is actually used. A pure linear split (lambda == 0) has no
    // such dependency, so near == 0 is valid input in that case.
    assert!(lambda <= 0.0 || near > 0.0,
        "CSM: near must be > 0 when lambda > 0 (logarithmic term divides by near)");

    let mut splits = Vec::with_capacity(count + 1);
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

    let ndc: [Vec4; 8] = [
        Vec4::new(-1., -1., -1., 1.), Vec4::new( 1., -1., -1., 1.),
        Vec4::new( 1.,  1., -1., 1.), Vec4::new(-1.,  1., -1., 1.),
        Vec4::new(-1., -1.,  1., 1.), Vec4::new( 1., -1.,  1., 1.),
        Vec4::new( 1.,  1.,  1., 1.), Vec4::new(-1.,  1.,  1., 1.),
    ];

    let mut corners = [Vec3::ZERO; 8];
    for (i, &c) in ndc.iter().enumerate() {
        let w  = inv_vp * c;
        let iw = 1.0 / w.w;
        corners[i] = Vec3::new(w.x * iw, w.y * iw, w.z * iw);
    }
    Some(corners)
        }
