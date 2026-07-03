// crates/mid-math/src/ffi/camera.rs
//! C-ABI exports for camera math utilities.
//!
//! C types:
//!   CFrustum          — 6 planes as Vec4 (nx, ny, nz, d)
//!   CVisibility (u32) — 0=Outside  1=Inside  2=Intersect
//!   CPerspectiveParams — fov_y, aspect, near, far (f32 × 4)
//!
//! Functions follow the mid_camera_* prefix.
//! All pointer output parameters are written only on success.

use crate::{
    Mat4, Vec3, Vec4,
    camera::{
        Frustum, Visibility,
        csm_split_depths, perspective_decompose, perspective_infinite_rh,
        perspective_resize, perspective_reversed_z_rh, picking_ray,
        sub_frustum_corners, unproject,
    },
};
use crate::ffi::float32::{CMat4, CVec3, CVec4};

// ═══════════════════════════════════════════════════════════════════════════
//  C types
// ═══════════════════════════════════════════════════════════════════════════

/// View frustum — 6 normalised half-space planes.
/// Each `CVec4(nx, ny, nz, d)`: point P is inside when nx·Px+ny·Py+nz·Pz+d ≥ 0.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct CFrustum {
    pub planes: [CVec4; 6],
}

/// Visibility classification result.
/// 0 = Outside  1 = Inside  2 = Intersect
pub type CVisibility = u32;

/// Decomposed perspective projection parameters.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct CPerspectiveParams {
    /// Vertical field of view in radians.
    pub fov_y:  f32,
    /// Aspect ratio (width / height).
    pub aspect: f32,
    /// Near plane distance.
    pub near:   f32,
    /// Far plane distance (`f32::INFINITY` for infinite projections).
    pub far:    f32,
}

// ── conversions ───────────────────────────────────────────────────────────────

#[inline(always)]
fn frustum_to_c(f: Frustum) -> CFrustum {
    CFrustum {
        planes: [
            CVec4 { x: f.planes[0].x, y: f.planes[0].y, z: f.planes[0].z, w: f.planes[0].w },
            CVec4 { x: f.planes[1].x, y: f.planes[1].y, z: f.planes[1].z, w: f.planes[1].w },
            CVec4 { x: f.planes[2].x, y: f.planes[2].y, z: f.planes[2].z, w: f.planes[2].w },
            CVec4 { x: f.planes[3].x, y: f.planes[3].y, z: f.planes[3].z, w: f.planes[3].w },
            CVec4 { x: f.planes[4].x, y: f.planes[4].y, z: f.planes[4].z, w: f.planes[4].w },
            CVec4 { x: f.planes[5].x, y: f.planes[5].y, z: f.planes[5].z, w: f.planes[5].w },
        ],
    }
}

#[inline(always)]
fn frustum_from_c(c: CFrustum) -> Frustum {
    Frustum {
        planes: [
            Vec4::new(c.planes[0].x, c.planes[0].y, c.planes[0].z, c.planes[0].w),
            Vec4::new(c.planes[1].x, c.planes[1].y, c.planes[1].z, c.planes[1].w),
            Vec4::new(c.planes[2].x, c.planes[2].y, c.planes[2].z, c.planes[2].w),
            Vec4::new(c.planes[3].x, c.planes[3].y, c.planes[3].z, c.planes[3].w),
            Vec4::new(c.planes[4].x, c.planes[4].y, c.planes[4].z, c.planes[4].w),
            Vec4::new(c.planes[5].x, c.planes[5].y, c.planes[5].z, c.planes[5].w),
        ],
    }
}

#[inline(always)]
fn vis_to_u32(v: Visibility) -> u32 {
    match v {
        Visibility::Outside   => 0,
        Visibility::Inside    => 1,
        Visibility::Intersect => 2,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Frustum — construction
// ═══════════════════════════════════════════════════════════════════════════

/// Extract frustum planes from a combined view-projection matrix.
/// Uses Gribb-Hartmann (2001). Planes are normalised on output.
#[no_mangle]
pub extern "C" fn mid_frustum_from_view_proj(m: CMat4) -> CFrustum {
    frustum_to_c(Frustum::from_view_proj(Mat4::from(m)))
}

/// Build a frustum from 6 pre-normalised planes (in order: L R B T Near Far).
/// Normalises each plane internally.
///
/// # Safety — planes must be non-null, valid for 6 CVec4 reads.
#[no_mangle]
pub unsafe extern "C" fn mid_frustum_from_planes(planes: *const CVec4) -> CFrustum {
    let ps = core::slice::from_raw_parts(planes, 6);
    let arr = [
        Vec4::new(ps[0].x, ps[0].y, ps[0].z, ps[0].w),
        Vec4::new(ps[1].x, ps[1].y, ps[1].z, ps[1].w),
        Vec4::new(ps[2].x, ps[2].y, ps[2].z, ps[2].w),
        Vec4::new(ps[3].x, ps[3].y, ps[3].z, ps[3].w),
        Vec4::new(ps[4].x, ps[4].y, ps[4].z, ps[4].w),
        Vec4::new(ps[5].x, ps[5].y, ps[5].z, ps[5].w),
    ];
    frustum_to_c(Frustum::from_planes(arr))
}

// ═══════════════════════════════════════════════════════════════════════════
//  Frustum — visibility tests
// ═══════════════════════════════════════════════════════════════════════════

/// Returns `true` if point `p` is inside or on the frustum boundary.
#[no_mangle]
pub extern "C" fn mid_frustum_test_point(f: CFrustum, p: CVec3) -> bool {
    frustum_from_c(f).test_point(Vec3::from(p))
}

/// Conservative sphere test. Returns `false` only when definitely outside.
#[no_mangle]
pub extern "C" fn mid_frustum_test_sphere(f: CFrustum, center: CVec3, radius: f32) -> bool {
    frustum_from_c(f).test_sphere(Vec3::from(center), radius)
}

/// Precise sphere visibility. Returns 0=Outside 1=Inside 2=Intersect.
#[no_mangle]
pub extern "C" fn mid_frustum_test_sphere_visibility(
    f: CFrustum, center: CVec3, radius: f32,
) -> CVisibility {
    vis_to_u32(frustum_from_c(f).test_sphere_visibility(Vec3::from(center), radius))
}

/// Conservative AABB test. Returns `false` only when definitely outside.
#[no_mangle]
pub extern "C" fn mid_frustum_test_aabb(f: CFrustum, min: CVec3, max: CVec3) -> bool {
    frustum_from_c(f).test_aabb(Vec3::from(min), Vec3::from(max))
}

/// Precise AABB visibility. Returns 0=Outside 1=Inside 2=Intersect.
#[no_mangle]
pub extern "C" fn mid_frustum_test_aabb_visibility(
    f: CFrustum, min: CVec3, max: CVec3,
) -> CVisibility {
    vis_to_u32(frustum_from_c(f).test_aabb_visibility(Vec3::from(min), Vec3::from(max)))
}

/// Signed distance from point `p` to frustum plane `plane_index` (0–5).
/// Positive = inside that half-space.
#[no_mangle]
pub extern "C" fn mid_frustum_plane_distance(f: CFrustum, plane_index: u32, p: CVec3) -> f32 {
    let fi = frustum_from_c(f);
    let idx = (plane_index as usize).min(5);
    Frustum::plane_distance(fi.planes[idx], Vec3::from(p))
}

// ═══════════════════════════════════════════════════════════════════════════
//  Frustum — world-space geometry
// ═══════════════════════════════════════════════════════════════════════════

/// Extract the 8 world-space frustum corners from the INVERSE view-projection matrix.
/// Corner order: [LBN, RBN, RTN, LTN, LBF, RBF, RTF, LTF].
/// `out` must be valid for 8 CVec3 writes.
///
/// # Safety — out non-null, valid for 8 CVec3.
#[no_mangle]
pub unsafe extern "C" fn mid_frustum_world_corners(inv_view_proj: CMat4, out: *mut CVec3) {
    let corners = Frustum::world_corners(Mat4::from(inv_view_proj));
    let out = core::slice::from_raw_parts_mut(out, 8);
    for (i, &c) in corners.iter().enumerate() {
        out[i] = CVec3::new(c.x, c.y, c.z);
    }
}

/// Centroid of the frustum in world space. Requires inverse view-projection.
#[no_mangle]
pub extern "C" fn mid_frustum_world_center(inv_view_proj: CMat4) -> CVec3 {
    let c = Frustum::world_center(Mat4::from(inv_view_proj));
    CVec3::new(c.x, c.y, c.z)
}

/// AABB (min, max) enclosing the frustum in world space.
/// Writes min to `out_min` and max to `out_max`.
///
/// # Safety — out_min, out_max non-null.
#[no_mangle]
pub unsafe extern "C" fn mid_frustum_world_aabb(
    inv_view_proj: CMat4,
    out_min: *mut CVec3,
    out_max: *mut CVec3,
) {
    let (mn, mx) = Frustum::world_aabb(Mat4::from(inv_view_proj));
    *out_min = CVec3::new(mn.x, mn.y, mn.z);
    *out_max = CVec3::new(mx.x, mx.y, mx.z);
}

// ═══════════════════════════════════════════════════════════════════════════
//  Batch frustum tests — engine-critical path
// ═══════════════════════════════════════════════════════════════════════════

/// Batch AABB culling. Writes 1 to `out[i]` if AABB i is visible, 0 if culled.
/// `mins` and `maxs` are parallel arrays of `count` world-space AABBs.
///
/// # Safety — mins, maxs, out non-null, valid for count.
#[no_mangle]
pub unsafe extern "C" fn mid_frustum_test_aabb_batch(
    f: CFrustum,
    mins: *const CVec3,
    maxs: *const CVec3,
    out: *mut u8,
    count: u32,
) {
    let n = count as usize;
    let fr   = frustum_from_c(f);
    let mins = core::slice::from_raw_parts(mins, n);
    let maxs = core::slice::from_raw_parts(maxs, n);
    let out  = core::slice::from_raw_parts_mut(out, n);
    for i in 0..n {
        out[i] = u8::from(fr.test_aabb(Vec3::from(mins[i]), Vec3::from(maxs[i])));
    }
}

/// Batch sphere culling. Writes 1 to `out[i]` if sphere i is visible, 0 if culled.
/// `centers` and `radii` are parallel arrays of `count` spheres.
///
/// # Safety — centers, radii, out non-null, valid for count.
#[no_mangle]
pub unsafe extern "C" fn mid_frustum_test_sphere_batch(
    f: CFrustum,
    centers: *const CVec3,
    radii:   *const f32,
    out: *mut u8,
    count: u32,
) {
    let n = count as usize;
    let fr      = frustum_from_c(f);
    let centers = core::slice::from_raw_parts(centers, n);
    let radii   = core::slice::from_raw_parts(radii, n);
    let out     = core::slice::from_raw_parts_mut(out, n);
    for i in 0..n {
        out[i] = u8::from(fr.test_sphere(Vec3::from(centers[i]), radii[i]));
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Projection utilities
// ═══════════════════════════════════════════════════════════════════════════

/// Unproject a window-space position to world space.
/// `window_pos`: (x, y, depth∈[0,1]).
/// `viewport`:   Vec4(x, y, width, height).
#[no_mangle]
pub extern "C" fn mid_camera_unproject(
    window_pos: CVec3,
    inv_view_proj: CMat4,
    viewport: CVec4,
) -> CVec3 {
    let r = unproject(
        Vec3::from(window_pos),
        Mat4::from(inv_view_proj),
        Vec4::new(viewport.x, viewport.y, viewport.z, viewport.w),
    );
    CVec3::new(r.x, r.y, r.z)
}

/// Build a world-space picking ray from a mouse position.
/// Writes `ray_origin` and `ray_direction` (not normalised — full near-to-far length).
/// Returns 1 on success, 0 if view-projection is singular.
///
/// # Safety — out_origin, out_dir non-null.
#[no_mangle]
pub unsafe extern "C" fn mid_camera_picking_ray(
    mouse_x: f32,
    mouse_y: f32,
    inv_view_proj: CMat4,
    viewport: CVec4,
    out_origin: *mut CVec3,
    out_dir:    *mut CVec3,
) {
    let vp = Vec4::new(viewport.x, viewport.y, viewport.z, viewport.w);
    let (origin, dir) = picking_ray(mouse_x, mouse_y, Mat4::from(inv_view_proj), vp);
    *out_origin = CVec3::new(origin.x, origin.y, origin.z);
    *out_dir    = CVec3::new(dir.x,    dir.y,    dir.z);
}

/// Right-hand infinite-far perspective projection.
#[no_mangle]
pub extern "C" fn mid_camera_perspective_infinite_rh(fov_y: f32, aspect: f32, near: f32) -> CMat4 {
    perspective_infinite_rh(fov_y, aspect, near).into()
}

/// Right-hand reversed-Z perspective projection.
/// Pass `far = f32::INFINITY` to combine with infinite far plane.
#[no_mangle]
pub extern "C" fn mid_camera_perspective_reversed_z_rh(
    fov_y: f32, aspect: f32, near: f32, far: f32,
) -> CMat4 {
    perspective_reversed_z_rh(fov_y, aspect, near, far).into()
}

/// Decompose a perspective projection matrix back into its parameters.
/// Returns 1 on success, 0 if `proj` is not a valid RH perspective matrix.
///
/// # Safety — out non-null on success path.
#[no_mangle]
pub unsafe extern "C" fn mid_camera_perspective_decompose(
    proj: CMat4,
    out: *mut CPerspectiveParams,
) -> u32 {
    match perspective_decompose(Mat4::from(proj)) {
        Some(p) => {
            *out = CPerspectiveParams { fov_y: p.fov_y, aspect: p.aspect, near: p.near, far: p.far };
            1
        }
        None => 0,
    }
}

/// Update only the aspect ratio of an existing perspective projection.
/// Cheaper than rebuilding the full matrix.
///
/// # Safety — proj non-null, valid CMat4.
#[no_mangle]
pub unsafe extern "C" fn mid_camera_perspective_resize(proj: *mut CMat4, new_aspect: f32) {
    let mut m = Mat4::from(*proj);
    perspective_resize(&mut m, new_aspect);
    *proj = m.into();
}

/// Generate CSM (Cascaded Shadow Map) split depths.
/// Returns `count + 1` values written to `out`: [near, split_1..split_{n-1}, far].
/// `lambda`: 0.0=linear, 1.0=logarithmic, 0.5=NVIDIA recommended.
///
/// # Safety — out non-null, valid for count+1 f32 writes.
#[no_mangle]
pub unsafe extern "C" fn mid_camera_csm_split_depths(
    near: f32, far: f32, count: u32, lambda: f32,
    out: *mut f32,
) {
    let splits = csm_split_depths(near, far, count as usize, lambda);
    let out = core::slice::from_raw_parts_mut(out, splits.len());
    out.copy_from_slice(&splits);
}

/// Compute the 8 world-space corners of a sub-frustum for one CSM cascade.
/// Returns 1 on success, 0 if decompose fails or matrix is singular.
/// Writes 8 CVec3 corners to `out` on success.
///
/// # Safety — out non-null, valid for 8 CVec3 writes.
#[no_mangle]
pub unsafe extern "C" fn mid_camera_sub_frustum_corners(
    view: CMat4,
    proj: CMat4,
    near: f32,
    far: f32,
    out: *mut CVec3,
) -> u32 {
    match sub_frustum_corners(Mat4::from(view), Mat4::from(proj), near, far) {
        Some(corners) => {
            let out = core::slice::from_raw_parts_mut(out, 8);
            for (i, &c) in corners.iter().enumerate() {
                out[i] = CVec3::new(c.x, c.y, c.z);
            }
            1
        }
        None => 0,
    }
    }
