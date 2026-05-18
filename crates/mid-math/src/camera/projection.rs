// crates/mid-math/src/camera/projection.rs
//! Extended projection matrix utilities.
//!
//! Supplements `Mat4`'s built-in projection constructors with:
//!   - `unproject`                 — window coords → world position (picking)
//!   - `picking_ray`               — click → world-space ray
//!   - `perspective_infinite_rh`   — infinite far plane
//!   - `perspective_reversed_z_rh` — reversed depth for Vulkan / DX12
//!   - `perspective_decompose`     — read back fov / aspect / near / far
//!   - `perspective_resize`        — cheaply update aspect ratio
//!   - `csm_split_depths`          — logarithmic+linear CSM splits
//!   - `sub_frustum_corners`       — world-space corners for one CSM cascade

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
///
/// # Returns
/// The reconstructed world-space position corresponding to the screen pixel.
pub fn unproject(window_pos: Vec3, inv_view_proj: Mat4, viewport: Vec4) -> Vec3 {
    // Map window coords to NDC [-1, 1] on each axis.
    let ndc_x = 2.0 * (window_pos.x - viewport.x) / viewport.z - 1.0;
    let ndc_y = 2.0 * (window_pos.y - viewport.y) / viewport.w - 1.0;
    let ndc_z = 2.0 *  window_pos.z - 1.0;   // depth [0,1] → [-1,1]

    let clip  = Vec4::new(ndc_x, ndc_y, ndc_z, 1.0);
    let world = inv_view_proj * clip;          // Mat4 * Vec4

    // Perspective divide.
    let iw = 1.0 / world.w;
    Vec3::new(world.x * iw, world.y * iw, world.z * iw)
}

/// Convenience wrapper: unproject using separate view and projection matrices.
///
/// Internally computes `(proj * view)⁻¹`. **Cache the inverse** if calling
/// per-frame from a tight loop — this internally calls `Mat4::inverse()`.
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
/// Returns `(ray_origin, ray_direction)` where:
///   - `ray_origin`    is the reconstructed near-plane world position.
///   - `ray_direction` points from near to far (NOT normalised — full length).
///
/// To get a unit direction: `ray_direction.normalize()`.
///
/// # Usage for click-on-mesh
/// ```text
/// let (origin, dir) = picking_ray(mouse_x, mouse_y, inv_vp, vp_rect);
/// let dir = dir.normalize();
/// // now ray-cast against scene geometry
/// ```
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

// ── Infinite perspective ───────────────────────────────────────────────────────

/// Right-hand perspective with an **infinite far plane**.
///
/// Useful for sky boxes, outer-space scenes, and any geometry that should
/// never be clipped. Derived by taking `far → ∞` in `perspective_rh`:
///
/// ```text
/// P[2][2] = -1  (column 2, row 2)
/// P[3][2] = -2·near (column 3, row 2)
/// P[2][3] = -1  (column 2, row 3)
/// ```
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

// ── Reversed-Z ────────────────────────────────────────────────────────────────

/// Right-hand **reversed-Z** perspective projection.
///
/// Maps the near plane → depth `1.0` and far plane → depth `0.0`. This
/// distributes floating-point precision more evenly across the depth range,
/// virtually eliminating z-fighting for distant geometry.
///
/// **Requires** a reversed depth test (`GREATER` or `GREATER_OR_EQUAL`).
///
/// Pass `far = f32::INFINITY` to combine with an infinite far plane — the
/// most precise configuration for large open worlds.
pub fn perspective_reversed_z_rh(fov_y: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
    let f = 1.0 / (fov_y * 0.5).tan();

    if far.is_infinite() {
        // Infinite reversed-Z: far → 0, near → 1.
        Mat4::from_cols(
            [f / aspect, 0.0,  0.0,   0.0],
            [0.0,        f,    0.0,   0.0],
            [0.0,        0.0,  0.0,  -1.0],
            [0.0,        0.0,  near,  0.0],
        )
    } else {
        // Finite reversed-Z.
        let z = far - near;
        Mat4::from_cols(
            [f / aspect, 0.0,  0.0,                 0.0],
            [0.0,        f,    0.0,                 0.0],
            [0.0,        0.0,  near / z,            -1.0],
            [0.0,        0.0,  near * far / z,       0.0],
        )
    }
}

// ── Decompose ─────────────────────────────────────────────────────────────────

/// Decompose a right-hand perspective projection matrix (clip `[-1, 1]`)
/// back into its constituent parameters.
///
/// This is the inverse of `Mat4::perspective_rh`. Useful for serialisation,
/// shader parameter extraction, and sub-frustum construction.
///
/// Returns `None` if `proj` does not appear to be a valid RH perspective
/// matrix (e.g. it is orthographic, infinite, reversed-Z, or malformed).
///
/// # Derivation (column-major `cols[c][r]`)
/// ```text
/// cols[1][1]  = f = 1/tan(fov_y/2)     → fov_y  = 2·atan(1/f)
/// cols[0][0]  = f/aspect               → aspect = f / cols[0][0]
/// cols[2][2]  = (far+near)/(near-far)  ← call this A
/// cols[3][2]  = 2·far·near/(near-far)  ← call this B
///   → near = B/(A-1),  far = B/(A+1)
/// cols[2][3]  = -1  (the w-divide sign)
/// ```
pub fn perspective_decompose(proj: Mat4) -> Option<PerspectiveParams> {
    // cols[2][3] is M[3][2] in row-major notation — the -1 that drives
    // the perspective divide. Anything else means this isn't RH perspective.
    if (proj.cols[2][3] + 1.0).abs() > 1e-4 { return None; }

    let f      = proj.cols[1][1];       // = 1/tan(fov_y/2)
    if f < 1e-6 { return None; }

    let aspect = f / proj.cols[0][0];   // = (f) / (f/aspect)
    let fov_y  = 2.0 * (1.0 / f).atan();

    // A = M[2][2] = cols[2][2],  B = M[2][3] = cols[3][2]
    let a = proj.cols[2][2];            // (far+near)/(near-far)
    let b = proj.cols[3][2];            // 2·far·near/(near-far)

    let near = b / (a - 1.0);          // derived above
    let far  = b / (a + 1.0);

    if near <= 0.0 || far <= near { return None; }

    Some(PerspectiveParams { fov_y, aspect, near, far })
}

// ── Resize ────────────────────────────────────────────────────────────────────

/// Update only the aspect ratio of an existing perspective projection matrix.
///
/// Much cheaper than rebuilding the matrix from scratch — changes only a
/// single element. Call this when the window is resized.
///
/// No-ops silently if `cols[0][0]` is zero (degenerate matrix).
#[inline]
pub fn perspective_resize(proj: &mut Mat4, new_aspect: f32) {
    if proj.cols[0][0] == 0.0 || new_aspect == 0.0 { return; }
    // cols[0][0] = f/old_aspect → new = f/new_aspect = cols[1][1]/new_aspect
    proj.cols[0][0] = proj.cols[1][1] / new_aspect;
}

// ── Cascaded Shadow Maps ──────────────────────────────────────────────────────

/// Generate `count` cascade split depths for Cascaded Shadow Maps (CSM/PSSM).
///
/// Uses the **practical split scheme** (logarithmic + linear blend) from the
/// NVIDIA CSM whitepaper:
/// ```text
/// split_log    = near * (far/near) ^ (i/count)
/// split_linear = near + (far-near) * (i/count)
/// split_i      = lambda * split_log + (1-lambda) * split_linear
/// ```
///
/// - `lambda = 0.0` → fully linear (even distribution, poor far-range quality)
/// - `lambda = 1.0` → fully logarithmic (best near quality, harsh far transitions)
/// - `lambda = 0.5` → NVIDIA's recommended default
///
/// Returns `count + 1` values: `[near, split_1, ..., split_{count-1}, far]`.
pub fn csm_split_depths(near: f32, far: f32, count: usize, lambda: f32) -> Vec<f32> {
    assert!(count >= 1, "CSM requires at least 1 cascade");
    assert!(far > near && near > 0.0, "CSM: far must be > near > 0");

    let mut splits = Vec::with_capacity(count + 1);
    splits.push(near);

    for i in 1..count {
        let p      = i as f32 / count as f32;
        let c_log  = near * (far / near).powf(p);
        let c_lin  = near + (far - near) * p;
        splits.push(lambda * c_log + (1.0 - lambda) * c_lin);
    }

    splits.push(far);
    splits
}

/// Compute the 8 world-space corners of a sub-frustum between `near` and `far`.
///
/// Used to build tight per-cascade bounding boxes for shadow map projection.
///
/// Extracts `fov_y` and `aspect` from the provided projection matrix
/// (via `perspective_decompose`), rebuilds a sub-frustum projection for
/// `[near, far]`, then transforms its NDC corners to world space.
///
/// Returns `None` if `proj` is not a valid RH perspective matrix or if the
/// resulting matrix is singular.
pub fn sub_frustum_corners(
    view: Mat4,
    proj: Mat4,
    near: f32,
    far: f32,
) -> Option<[Vec3; 8]> {
    let p = perspective_decompose(proj)?;

    // Rebuild projection only for the [near, far] sub-range.
    let sub_proj = Mat4::perspective_rh(p.fov_y, p.aspect, near, far);

    // Compute inverse of (sub_proj * view).
    let inv_vp = (sub_proj * view).inverse()?;

    // NDC cube corners, w=1.
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
