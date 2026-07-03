// crates/mid-math/src/camera/ray.rs
//! Screen-space ↔ world-space ray utilities — unprojection and mouse picking.
//!
//! Split out of the old `projection.rs` (which used to hold everything from
//! matrix decomposition to CSM splits in one file): this file's concern is
//! specifically "turn a screen coordinate into a world-space ray," distinct
//! from *constructing* projection matrices.

use crate::{Mat4, Vec3, Vec4};

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
