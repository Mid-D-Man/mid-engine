// crates/mid-math/src/camera/projection.rs
//! Extended perspective projection utilities — infinite far plane,
//! reversed-Z, decomposition, and cheap aspect-ratio resize.
//!
//! Supplements `Mat4`'s built-in projection constructors
//! (`perspective_rh/lh`, `ortho_rh/lh`, `frustum_rh/lh`, all in
//! `f32/mat4_projection.rs`) with the variants that don't fit a plain
//! `near`/`far` pair:
//!   - `perspective_infinite_rh`    — infinite far plane, z ∈ [0,1] (RH)
//!   - `perspective_infinite_rh_gl` — infinite far plane, z ∈ [-1,1] (RH, legacy GL)
//!   - `perspective_reversed_z_rh`  — reversed depth, z ∈ [0,1] (RH, Vulkan/DX12)
//!   - `perspective_infinite_lh`    — infinite far plane, z ∈ [0,1] (LH)
//!   - `perspective_reversed_z_lh`  — reversed depth, z ∈ [0,1] (LH, DX12/Metal)
//!   - `perspective_decompose`      — read back fov / aspect / near / far
//!   - `perspective_resize`         — cheaply update aspect ratio
//!
//! See `camera::ray` for unprojection/picking and `camera::csm` for
//! Cascaded Shadow Map utilities — both used to live in this file.

use crate::Mat4;

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

// ── Right-handed infinite / reversed-Z ───────────────────────────────────────

/// Right-hand perspective with an **infinite far plane**, `z ∈ [0,1]`
/// (Vulkan / D3D12 / Metal / WebGPU — matches `Mat4::perspective_rh`).
pub fn perspective_infinite_rh(fov_y: f32, aspect: f32, near: f32) -> Mat4 {
    let f = 1.0 / (fov_y * 0.5).tan();
    Mat4::from_cols(
        [f / aspect, 0.0,  0.0,   0.0],
        [0.0,        f,    0.0,   0.0],
        [0.0,        0.0, -1.0,  -1.0],
        [0.0,        0.0, -near,  0.0],
    )
}

/// Right-hand perspective with an **infinite far plane**, `z ∈ [-1,1]`
/// (legacy OpenGL).
pub fn perspective_infinite_rh_gl(fov_y: f32, aspect: f32, near: f32) -> Mat4 {
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

/// Left-hand perspective with an **infinite far plane**, `z ∈ [0,1]`
/// (Vulkan / D3D12 / Metal / WebGPU — matches `Mat4::perspective_lh`).
///
/// Maps near → 0, far → 1 (approaching from infinity).
///
/// ```text
/// z_axis.z = 1.0     (limit of far/(far-near) as far → ∞)
/// w_axis.z = -near   (limit of -near·far/(far-near) as far → ∞)
/// z_axis.w = +1.0    (LH w-divide drives positive z forward)
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

/// Decompose a right-hand perspective projection matrix (`z ∈ [0,1]`, ZO —
/// matches `Mat4::perspective_rh`) back into its constituent parameters.
///
/// Returns `None` if `proj` is not a valid RH ZO perspective matrix.
pub fn perspective_decompose(proj: Mat4) -> Option<PerspectiveParams> {
    // RH perspective: z_axis.w must be -1.0 (the perspective divide sentinel).
    if (proj.z_axis.w + 1.0).abs() > 1e-4 { return None; }

    // y_axis.y = vertical scale = f
    let f: f32 = proj.y_axis.y;
    if f < 1e-6 { return None; }

    // x_axis.x = horizontal scale = f/aspect
    let aspect = f / proj.x_axis.x;
    let fov_y  = 2.0_f32 * (1.0_f32 / f).atan();

    // z ∈ [0,1] (ZO, matches Mat4::perspective_rh):
    //   z_axis.z = a = far/(near-far)
    //   w_axis.z = b = a*near
    // Solving: near = b/a, far = b/(1+a)
    let a = proj.z_axis.z;
    let b = proj.w_axis.z;
    if a.abs() < 1e-8 { return None; }

    let near = b / a;
    let far  = b / (1.0 + a);

    if near <= 0.0 || far <= near { return None; }

    Some(PerspectiveParams { fov_y, aspect, near, far })
}

// ── Resize ────────────────────────────────────────────────────────────────────

/// Update only the aspect ratio of an existing perspective projection matrix.
///
/// Works for both RH and LH projections — only the horizontal scale changes.
#[inline]
pub fn perspective_resize(proj: &mut Mat4, new_aspect: f32) {
    if proj.x_axis.x == 0.0 || new_aspect == 0.0 { return; }
    proj.x_axis.x = proj.y_axis.y / new_aspect;
            }
