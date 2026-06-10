// crates/mid-math/src/tests/camera.rs
//! Tests for camera math utilities — Frustum, projections, unproject, CSM.

use crate::mid_math::{
    Mat4, Quat, Vec3, Vec4,
    camera::{
        Frustum, Visibility,
        projection::{
            PerspectiveParams,
            csm_split_depths, perspective_decompose, perspective_infinite_rh,
            perspective_resize, perspective_reversed_z_rh, picking_ray,
            sub_frustum_corners, unproject,
        },
    },
    to_radians,
};

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Build a standard test view-projection matrix looking down -Z.
fn test_view_proj() -> Mat4 {
    let view = Mat4::look_at_rh(
        Vec3::new(0.0, 0.0, 5.0),
        Vec3::ZERO,
        Vec3::Y,
    );
    let proj = Mat4::perspective_rh(to_radians(60.0), 16.0 / 9.0, 0.1, 100.0);
    proj * view
}

/// Assert two f32 values are within `eps` of each other.
fn assert_approx(a: f32, b: f32, eps: f32, label: &str) {
    assert!((a - b).abs() < eps, "{}: expected ≈{}, got {} (diff={})", label, b, a, (a - b).abs());
}

// ─────────────────────────────────────────────────────────────────────────────
// Frustum — construction
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn frustum_from_view_proj_has_six_planes() {
    let vp = test_view_proj();
    let f  = Frustum::from_view_proj(vp);
    // Each plane normal should be unit length (they are normalised).
    for (i, plane) in f.planes.iter().enumerate() {
        let len = (plane.x * plane.x + plane.y * plane.y + plane.z * plane.z).sqrt();
        assert!((len - 1.0).abs() < 1e-4, "Plane {} not normalised: len={}", i, len);
    }
}

#[test]
fn frustum_origin_is_inside() {
    // Camera at (0,0,5) looking at origin — origin should be visible.
    let vp = test_view_proj();
    let f  = Frustum::from_view_proj(vp);
    assert!(f.test_point(Vec3::ZERO), "Origin should be inside frustum");
}

#[test]
fn frustum_behind_camera_is_outside() {
    let vp = test_view_proj();
    let f  = Frustum::from_view_proj(vp);
    // Point behind camera eye (eye at Z=5 looking toward -Z origin).
    let behind = Vec3::new(0.0, 0.0, 6.0);
    assert!(!f.test_point(behind), "Point behind camera should be outside frustum");
}

#[test]
fn frustum_far_behind_target_is_outside() {
    let vp = test_view_proj();
    let f  = Frustum::from_view_proj(vp);
    // Very far point along -Z.
    assert!(!f.test_point(Vec3::new(0.0, 0.0, -200.0)), "Far point should be outside far plane");
}

#[test]
fn frustum_test_sphere_large_sphere_visible() {
    let vp = test_view_proj();
    let f  = Frustum::from_view_proj(vp);
    // Giant sphere centred near origin — definitely visible.
    assert!(f.test_sphere(Vec3::ZERO, 50.0), "Huge sphere at origin should be visible");
}

#[test]
fn frustum_test_sphere_tiny_far_sphere_invisible() {
    let vp = test_view_proj();
    let f  = Frustum::from_view_proj(vp);
    // Tiny sphere far behind camera.
    assert!(!f.test_sphere(Vec3::new(0.0, 0.0, 10.0), 0.01));
}

#[test]
fn frustum_sphere_visibility_origin() {
    let vp = test_view_proj();
    let f  = Frustum::from_view_proj(vp);
    let vis = f.test_sphere_visibility(Vec3::ZERO, 0.1);
    // Small sphere at origin — inside or intersect, never outside.
    assert_ne!(vis, Visibility::Outside, "Small sphere at origin should not be Outside");
}

#[test]
fn frustum_sphere_visibility_outside() {
    let vp = test_view_proj();
    let f  = Frustum::from_view_proj(vp);
    let vis = f.test_sphere_visibility(Vec3::new(0.0, 0.0, 20.0), 0.01);
    assert_eq!(vis, Visibility::Outside);
}

#[test]
fn frustum_test_aabb_origin_box_visible() {
    let vp  = test_view_proj();
    let f   = Frustum::from_view_proj(vp);
    let min = Vec3::new(-1.0, -1.0, -1.0);
    let max = Vec3::new( 1.0,  1.0,  1.0);
    assert!(f.test_aabb(min, max), "Unit AABB at origin should be visible");
}

#[test]
fn frustum_test_aabb_far_box_invisible() {
    let vp  = test_view_proj();
    let f   = Frustum::from_view_proj(vp);
    let min = Vec3::new(-1.0, -1.0, 500.0);
    let max = Vec3::new( 1.0,  1.0, 502.0);
    assert!(!f.test_aabb(min, max), "AABB behind far plane should be invisible");
}

#[test]
fn frustum_aabb_visibility_classifications() {
    let vp  = test_view_proj();
    let f   = Frustum::from_view_proj(vp);
    // Huge box that contains the whole frustum → Inside.
    let big_min = Vec3::new(-1000.0, -1000.0, -1000.0);
    let big_max = Vec3::new( 1000.0,  1000.0,  1000.0);
    let vis = f.test_aabb_visibility(big_min, big_max);
    assert_ne!(vis, Visibility::Outside, "Huge AABB should not be Outside");
}

// ─────────────────────────────────────────────────────────────────────────────
// Frustum — world-space geometry
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn frustum_world_corners_count_and_finite() {
    let vp     = test_view_proj();
    let inv_vp = vp.inverse().expect("test VP should be invertible");
    let corners = Frustum::world_corners(inv_vp);
    assert_eq!(corners.len(), 8);
    for (i, c) in corners.iter().enumerate() {
        assert!(c.is_finite(), "Corner {} not finite: {:?}", i, c);
    }
}

#[test]
fn frustum_world_center_near_camera_target() {
    let vp     = test_view_proj();
    let inv_vp = vp.inverse().expect("VP invertible");
    let center = Frustum::world_center(inv_vp);
    // Center should be between near plane and far plane along view axis.
    assert!(center.is_finite(), "World center not finite");
    // Should be somewhere between camera eye (Z=5) and target (Z=0) toward far.
    assert!(center.z < 5.1, "Center should be in front of camera");
    assert!(center.z > -105.0, "Center should be before far plane");
}

#[test]
fn frustum_world_aabb_contains_center() {
    let vp     = test_view_proj();
    let inv_vp = vp.inverse().expect("VP invertible");
    let (mn, mx) = Frustum::world_aabb(inv_vp);
    let center   = Frustum::world_center(inv_vp);
    assert!(center.x >= mn.x - 0.01 && center.x <= mx.x + 0.01);
    assert!(center.y >= mn.y - 0.01 && center.y <= mx.y + 0.01);
    assert!(center.z >= mn.z - 0.01 && center.z <= mx.z + 0.01);
}

// ─────────────────────────────────────────────────────────────────────────────
// Unproject / picking ray
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn unproject_near_depth_is_near_plane() {
    let view = Mat4::look_at_rh(Vec3::new(0.0, 0.0, 5.0), Vec3::ZERO, Vec3::Y);
    let proj = Mat4::perspective_rh(to_radians(60.0), 1.0, 0.1, 100.0);
    let vp   = proj * view;
    let inv  = vp.inverse().expect("invertible");
    let viewport = Vec4::new(0.0, 0.0, 800.0, 600.0);

    // Screen center at depth 0 should land near the near plane.
    let world = unproject(Vec3::new(400.0, 300.0, 0.0), inv, viewport);
    assert!(world.is_finite(), "Unproject result not finite: {:?}", world);
    // Should be somewhere in front of camera, not behind.
    assert!(world.z < 5.1, "Unprojected point should be in front of eye (z=5)");
}

#[test]
fn unproject_far_depth_is_far_from_eye() {
    let view = Mat4::look_at_rh(Vec3::new(0.0, 0.0, 5.0), Vec3::ZERO, Vec3::Y);
    let proj = Mat4::perspective_rh(to_radians(60.0), 1.0, 0.1, 100.0);
    let vp   = proj * view;
    let inv  = vp.inverse().expect("invertible");
    let vp2d = Vec4::new(0.0, 0.0, 800.0, 600.0);

    let near_pt = unproject(Vec3::new(400.0, 300.0, 0.0), inv, vp2d);
    let far_pt  = unproject(Vec3::new(400.0, 300.0, 1.0), inv, vp2d);

    // Far point should be significantly further from eye than near point.
    let eye = Vec3::new(0.0, 0.0, 5.0);
    let d_near = (near_pt - eye).length();
    let d_far  = (far_pt  - eye).length();
    assert!(d_far > d_near * 2.0,
        "Far depth ({}) should be further than near depth ({})", d_far, d_near);
}

#[test]
fn picking_ray_direction_not_zero() {
    let view = Mat4::look_at_rh(Vec3::new(0.0, 0.0, 5.0), Vec3::ZERO, Vec3::Y);
    let proj = Mat4::perspective_rh(to_radians(60.0), 1.0, 0.1, 100.0);
    let vp   = (proj * view).inverse().expect("invertible");
    let port = Vec4::new(0.0, 0.0, 800.0, 600.0);

    let (origin, dir) = picking_ray(400.0, 300.0, vp, port);
    assert!(origin.is_finite());
    assert!(dir.is_finite());
    assert!(dir.length() > 0.01, "Picking ray direction is near-zero");
}

// ─────────────────────────────────────────────────────────────────────────────
// Perspective decompose / resize
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn perspective_decompose_roundtrip() {
    let fov_y  = to_radians(60.0);
    let aspect = 16.0 / 9.0;
    let near   = 0.1f32;
    let far    = 100.0f32;

    let proj = Mat4::perspective_rh(fov_y, aspect, near, far);
    let p    = perspective_decompose(proj).expect("should decompose standard RH perspective");

    assert_approx(p.fov_y,  fov_y,  1e-4, "fov_y");
    assert_approx(p.aspect, aspect, 1e-4, "aspect");
    assert_approx(p.near,   near,   1e-4, "near");
    assert_approx(p.far,    far,    0.1,  "far");
}

#[test]
fn perspective_decompose_rejects_orthographic() {
    let ortho = Mat4::ortho_rh(-1.0, 1.0, -1.0, 1.0, 0.1, 100.0);
    assert!(perspective_decompose(ortho).is_none(),
        "Should not decompose an orthographic matrix");
}

#[test]
fn perspective_decompose_rejects_identity() {
    assert!(perspective_decompose(Mat4::IDENTITY).is_none());
}

#[test]
fn perspective_resize_changes_only_aspect() {
    let fov_y = to_radians(60.0);
    let proj  = Mat4::perspective_rh(fov_y, 16.0 / 9.0, 0.1, 100.0);

    let mut resized = proj;
    perspective_resize(&mut resized, 4.0 / 3.0);

    // cols[0][0] = f/aspect — must change.
    assert!(
        (resized.cols[0][0] - proj.cols[0][0]).abs() > 1e-5,
        "cols[0][0] should change after resize"
    );
    // cols[1][1] = f — must NOT change.
    assert_approx(resized.cols[1][1], proj.cols[1][1], 1e-6, "f (cols[1][1])");
    // cols[2][2] and cols[3][2] — must NOT change (depth).
    assert_approx(resized.cols[2][2], proj.cols[2][2], 1e-6, "depth (cols[2][2])");
    assert_approx(resized.cols[3][2], proj.cols[3][2], 1e-6, "depth (cols[3][2])");
}

#[test]
fn perspective_infinite_rh_no_far_clip() {
    let proj = perspective_infinite_rh(to_radians(60.0), 1.0, 0.1);
    // cols[2][2] should be -1 for infinite far.
    assert_approx(proj.cols[2][2], -1.0, 1e-5, "infinite proj cols[2][2]");
    assert!(proj.cols[3][2].abs() > 0.0, "Near plane term should be non-zero");
}

#[test]
fn perspective_reversed_z_far_maps_to_zero() {
    // For reversed-Z: far plane → depth 0, near plane → depth 1.
    // We just check the matrix is finite and non-trivial.
    let proj = perspective_reversed_z_rh(to_radians(60.0), 1.0, 0.1, 100.0);
    for col in &proj.cols {
        for &v in col {
            assert!(v.is_finite(), "Reversed-Z projection contains non-finite value");
        }
    }
    // cols[2][2] should be positive for reversed-Z (opposite sign from standard).
    assert!(proj.cols[2][2] > 0.0, "Reversed-Z cols[2][2] should be positive");
}

// ─────────────────────────────────────────────────────────────────────────────
// CSM split depths
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn csm_split_depths_count() {
    let splits = csm_split_depths(0.1, 100.0, 4, 0.5);
    assert_eq!(splits.len(), 5, "4 cascades → 5 split values");
}

#[test]
fn csm_split_depths_boundaries() {
    let near = 0.1f32;
    let far  = 100.0f32;
    let splits = csm_split_depths(near, far, 4, 0.5);
    assert_approx(*splits.first().unwrap(), near, 1e-5, "first split = near");
    assert_approx(*splits.last().unwrap(),  far,  1e-5, "last split = far");
}

#[test]
fn csm_split_depths_sorted_ascending() {
    let splits = csm_split_depths(0.1, 100.0, 6, 0.5);
    for i in 0..splits.len() - 1 {
        assert!(splits[i] < splits[i + 1],
            "CSM splits not sorted: splits[{}]={} ≥ splits[{}]={}", i, splits[i], i+1, splits[i+1]);
    }
}

#[test]
fn csm_split_depths_lambda_0_is_linear() {
    // λ=0 → linear distribution.
    let splits = csm_split_depths(0.0, 100.0, 4, 0.0);
    // Linear: split[i] = near + (far-near)*i/count = 25, 50, 75.
    assert_approx(splits[1], 25.0,  1.0, "linear split 1");
    assert_approx(splits[2], 50.0,  1.0, "linear split 2");
    assert_approx(splits[3], 75.0,  1.0, "linear split 3");
}

#[test]
fn csm_split_depths_one_cascade() {
    let splits = csm_split_depths(0.1, 50.0, 1, 0.5);
    assert_eq!(splits.len(), 2);
    assert_approx(splits[0], 0.1,  1e-5, "near");
    assert_approx(splits[1], 50.0, 1e-5, "far");
}

// ─────────────────────────────────────────────────────────────────────────────
// sub_frustum_corners
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn sub_frustum_corners_returns_eight_finite_points() {
    let view = Mat4::look_at_rh(Vec3::new(0.0, 0.0, 5.0), Vec3::ZERO, Vec3::Y);
    let proj = Mat4::perspective_rh(to_radians(60.0), 16.0 / 9.0, 0.1, 100.0);

    let corners = sub_frustum_corners(view, proj, 0.1, 10.0)
        .expect("Valid sub-frustum should succeed");
    assert_eq!(corners.len(), 8);
    for (i, c) in corners.iter().enumerate() {
        assert!(c.is_finite(), "Sub-frustum corner {} not finite: {:?}", i, c);
    }
}

#[test]
fn sub_frustum_corners_fails_for_non_perspective() {
    let view = Mat4::look_at_rh(Vec3::new(0.0, 0.0, 5.0), Vec3::ZERO, Vec3::Y);
    let ortho = Mat4::ortho_rh(-1.0, 1.0, -1.0, 1.0, 0.1, 100.0);
    assert!(
        sub_frustum_corners(view, ortho, 0.1, 10.0).is_none(),
        "sub_frustum_corners should fail for orthographic proj"
    );
      }
