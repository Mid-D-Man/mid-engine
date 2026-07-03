// crates/mid-math/src/f32/mat4_projection.rs
//! `Mat4` projection & view matrix builders — perspective, orthographic,
//! look-at/look-to, and general (asymmetric) frustum construction.
//!
//! ## Why this file exists
//! These builders do no SIMD-specific work — they're plain arithmetic that
//! assigns into `Mat4`'s four public `Vec4` columns. Before this file, that
//! meant the exact same formula was hand-copied into `sse2/mat4.rs`,
//! `neon/mat4.rs`, `wasm/mat4.rs`, `coresimd/mat4.rs`, *and* `scalar/mat4.rs`
//! — five independent copies of the same math, with nothing keeping them in
//! sync. They drifted: `scalar::ortho_rh` computed the ZO (Vulkan/D3D/Metal,
//! `z ∈ [0,1]`) formula while every SIMD backend's `ortho_rh` computed NO
//! (OpenGL, `z ∈ [-1,1]`) under the identical name — same function, silently
//! different math depending on target arch. `scalar::perspective_rh` also
//! carried a doc comment claiming ZO while its body was byte-for-byte the
//! NO formula.
//!
//! Written once here against `crate::Mat4`/`crate::Vec4` (the per-target
//! dispatched aliases), this resolves to whichever concrete backend type is
//! active and needs no `#[cfg]` at all — there is now exactly one
//! implementation per function, for every target, permanently.
//!
//! ## Naming convention
//! Plain name = **ZO** (`z ∈ [0,1]`) — the modern default (Vulkan, D3D12,
//! Metal, WebGPU). `_gl` suffix = **NO** (`z ∈ [-1,1]`) — legacy OpenGL only.
//! This matches glam's convention. There is no separate `_vk`/`_wgpu_dx`
//! split the way ultraviolet does it — Vulkan/D3D/Metal/WebGPU all use
//! identical ZO math, so a per-API split would just be duplicate functions
//! with duplicate formulas for zero mathematical difference.

use crate::{Mat4, Vec3, Vec4};
use crate::f32::math;

impl Mat4 {
    // ── Perspective ─────────────────────────────────────────────────────

    /// RH perspective, `z ∈ [0,1]` (Vulkan / D3D12 / Metal / WebGPU).
    #[inline]
    pub fn perspective_rh(fov_y: f32, aspect: f32, near: f32, far: f32) -> Self {
        let (sin_fov, cos_fov) = math::sin_cos(fov_y * 0.5);
        let f = cos_fov / sin_fov;
        let r = far / (near - far);
        Self {
            x_axis: Vec4::new(f / aspect, 0.0, 0.0, 0.0),
            y_axis: Vec4::new(0.0, f, 0.0, 0.0),
            z_axis: Vec4::new(0.0, 0.0, r, -1.0),
            w_axis: Vec4::new(0.0, 0.0, r * near, 0.0),
        }
    }

    /// LH perspective, `z ∈ [0,1]` (Vulkan / D3D12 / Metal / WebGPU).
    #[inline]
    pub fn perspective_lh(fov_y: f32, aspect: f32, near: f32, far: f32) -> Self {
        let (sin_fov, cos_fov) = math::sin_cos(fov_y * 0.5);
        let f = cos_fov / sin_fov;
        let r = far / (far - near);
        Self {
            x_axis: Vec4::new(f / aspect, 0.0, 0.0, 0.0),
            y_axis: Vec4::new(0.0, f, 0.0, 0.0),
            z_axis: Vec4::new(0.0, 0.0, r, 1.0),
            w_axis: Vec4::new(0.0, 0.0, -r * near, 0.0),
        }
    }

    /// RH perspective, `z ∈ [-1,1]` (legacy OpenGL).
    #[inline]
    pub fn perspective_rh_gl(fov_y: f32, aspect: f32, near: f32, far: f32) -> Self {
        let (sin_fov, cos_fov) = math::sin_cos(fov_y * 0.5);
        let f = cos_fov / sin_fov;
        let z = near - far;
        Self {
            x_axis: Vec4::new(f / aspect, 0.0, 0.0, 0.0),
            y_axis: Vec4::new(0.0, f, 0.0, 0.0),
            z_axis: Vec4::new(0.0, 0.0, (far + near) / z, -1.0),
            w_axis: Vec4::new(0.0, 0.0, (2.0 * far * near) / z, 0.0),
        }
    }

    /// LH perspective, `z ∈ [-1,1]` (legacy OpenGL).
    #[inline]
    pub fn perspective_lh_gl(fov_y: f32, aspect: f32, near: f32, far: f32) -> Self {
        let (sin_fov, cos_fov) = math::sin_cos(fov_y * 0.5);
        let f = cos_fov / sin_fov;
        let z = far - near;
        Self {
            x_axis: Vec4::new(f / aspect, 0.0, 0.0, 0.0),
            y_axis: Vec4::new(0.0, f, 0.0, 0.0),
            z_axis: Vec4::new(0.0, 0.0, far / z, 1.0),
            w_axis: Vec4::new(0.0, 0.0, -(far * near) / z, 0.0),
        }
    }

    // ── Orthographic ────────────────────────────────────────────────────

    /// RH orthographic, `z ∈ [0,1]` (Vulkan / D3D12 / Metal / WebGPU).
    #[inline]
    pub fn ortho_rh(left: f32, right: f32, bottom: f32, top: f32, near: f32, far: f32) -> Self {
        let rl = right - left; let tb = top - bottom; let nf = far - near;
        Self {
            x_axis: Vec4::new(2.0 / rl, 0.0, 0.0, 0.0),
            y_axis: Vec4::new(0.0, 2.0 / tb, 0.0, 0.0),
            z_axis: Vec4::new(0.0, 0.0, -1.0 / nf, 0.0),
            w_axis: Vec4::new(
                -(right + left) / rl, -(top + bottom) / tb, -near / nf, 1.0,
            ),
        }
    }

    /// LH orthographic, `z ∈ [0,1]` (Vulkan / D3D12 / Metal / WebGPU).
    #[inline]
    pub fn ortho_lh(left: f32, right: f32, bottom: f32, top: f32, near: f32, far: f32) -> Self {
        let rl = right - left; let tb = top - bottom; let nf = far - near;
        Self {
            x_axis: Vec4::new(2.0 / rl, 0.0, 0.0, 0.0),
            y_axis: Vec4::new(0.0, 2.0 / tb, 0.0, 0.0),
            z_axis: Vec4::new(0.0, 0.0, 1.0 / nf, 0.0),
            w_axis: Vec4::new(
                -(right + left) / rl, -(top + bottom) / tb, -near / nf, 1.0,
            ),
        }
    }

    /// RH orthographic, `z ∈ [-1,1]` (legacy OpenGL).
    #[inline]
    pub fn ortho_rh_gl(left: f32, right: f32, bottom: f32, top: f32, near: f32, far: f32) -> Self {
        let rl = right - left; let tb = top - bottom; let nf = far - near;
        Self {
            x_axis: Vec4::new(2.0 / rl, 0.0, 0.0, 0.0),
            y_axis: Vec4::new(0.0, 2.0 / tb, 0.0, 0.0),
            z_axis: Vec4::new(0.0, 0.0, -2.0 / nf, 0.0),
            w_axis: Vec4::new(
                -(right + left) / rl, -(top + bottom) / tb, -(far + near) / nf, 1.0,
            ),
        }
    }

    /// LH orthographic, `z ∈ [-1,1]` (legacy OpenGL).
    #[inline]
    pub fn ortho_lh_gl(left: f32, right: f32, bottom: f32, top: f32, near: f32, far: f32) -> Self {
        let rl = right - left; let tb = top - bottom; let nf = far - near;
        Self {
            x_axis: Vec4::new(2.0 / rl, 0.0, 0.0, 0.0),
            y_axis: Vec4::new(0.0, 2.0 / tb, 0.0, 0.0),
            z_axis: Vec4::new(0.0, 0.0, 2.0 / nf, 0.0),
            w_axis: Vec4::new(
                -(right + left) / rl, -(top + bottom) / tb, -(far + near) / nf, 1.0,
            ),
        }
    }

    // ── General (asymmetric) frustum ───────────────────────────────────
    // Formulas carried over verbatim from the pre-existing scalar
    // implementation (the RH case independently cross-checks exact-match
    // against cglm's glm_frustum_rh_zo) rather than re-derived, to avoid
    // introducing a new sign-convention bug in the LH case.

    /// RH perspective frustum from explicit clip-plane bounds, `z ∈ [0,1]`.
    /// Unlike `perspective_rh` (symmetric FOV), this allows off-axis
    /// projections — VR (per-eye asymmetric FOV), tiled/portal rendering,
    /// shearing for projector edge-blending.
    #[inline]
    pub fn frustum_rh(left: f32, right: f32, bottom: f32, top: f32, near: f32, far: f32) -> Self {
        let rl = right - left; let tb = top - bottom; let nf = near - far;
        Self {
            x_axis: Vec4::new(2.0 * near / rl, 0.0, 0.0, 0.0),
            y_axis: Vec4::new(0.0, 2.0 * near / tb, 0.0, 0.0),
            z_axis: Vec4::new((right + left) / rl, (top + bottom) / tb, far / nf, -1.0),
            w_axis: Vec4::new(0.0, 0.0, (far * near) / nf, 0.0),
        }
    }

    /// LH perspective frustum from explicit clip-plane bounds, `z ∈ [0,1]`.
    #[inline]
    pub fn frustum_lh(left: f32, right: f32, bottom: f32, top: f32, near: f32, far: f32) -> Self {
        let rl = right - left; let tb = top - bottom; let fnz = far - near;
        Self {
            x_axis: Vec4::new(2.0 * near / rl, 0.0, 0.0, 0.0),
            y_axis: Vec4::new(0.0, 2.0 * near / tb, 0.0, 0.0),
            z_axis: Vec4::new(-(right + left) / rl, -(top + bottom) / tb, far / fnz, 1.0),
            w_axis: Vec4::new(0.0, 0.0, -(far * near) / fnz, 0.0),
        }
    }

    /// RH perspective frustum, `z ∈ [-1,1]` (legacy OpenGL).
    #[inline]
    pub fn frustum_rh_gl(left: f32, right: f32, bottom: f32, top: f32, near: f32, far: f32) -> Self {
        let rl = right - left; let tb = top - bottom; let nf = near - far;
        Self {
            x_axis: Vec4::new(2.0 * near / rl, 0.0, 0.0, 0.0),
            y_axis: Vec4::new(0.0, 2.0 * near / tb, 0.0, 0.0),
            z_axis: Vec4::new((right + left) / rl, (top + bottom) / tb, (far + near) / nf, -1.0),
            w_axis: Vec4::new(0.0, 0.0, (2.0 * far * near) / nf, 0.0),
        }
    }

    /// LH perspective frustum, `z ∈ [-1,1]` (legacy OpenGL).
    #[inline]
    pub fn frustum_lh_gl(left: f32, right: f32, bottom: f32, top: f32, near: f32, far: f32) -> Self {
        let rl = right - left; let tb = top - bottom; let fnz = far - near;
        Self {
            x_axis: Vec4::new(2.0 * near / rl, 0.0, 0.0, 0.0),
            y_axis: Vec4::new(0.0, 2.0 * near / tb, 0.0, 0.0),
            z_axis: Vec4::new(-(right + left) / rl, -(top + bottom) / tb, (far + near) / fnz, 1.0),
            w_axis: Vec4::new(0.0, 0.0, -(2.0 * far * near) / fnz, 0.0),
        }
    }

    // ── View (direction-based, complements the existing look_at) ──────

    /// RH view matrix from an eye position and a forward *direction*
    /// (already normalized), rather than a target point. Complements
    /// `look_at_rh` — use `look_to` when you're driving the camera by
    /// forward vector (FPS controllers, physics-attached cameras) and don't
    /// want to synthesize a fake target point just to call `look_at`.
    #[inline]
    pub fn look_to_rh(eye: Vec3, dir: Vec3, up: Vec3) -> Self {
        let f = dir.normalize();
        let s = f.cross(up).normalize();
        let u = s.cross(f);
        Self {
            x_axis: Vec4::new(s.x, u.x, -f.x, 0.0),
            y_axis: Vec4::new(s.y, u.y, -f.y, 0.0),
            z_axis: Vec4::new(s.z, u.z, -f.z, 0.0),
            w_axis: Vec4::new(-s.dot(eye), -u.dot(eye), f.dot(eye), 1.0),
        }
    }

    /// LH view matrix from an eye position and a forward *direction*.
    /// See [`Mat4::look_to_rh`].
    #[inline]
    pub fn look_to_lh(eye: Vec3, dir: Vec3, up: Vec3) -> Self {
        Self::look_to_rh(eye, -dir, up)
    }
}
