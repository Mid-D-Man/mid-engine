// crates/mid-math/src/camera/mod.rs
//! Camera math utilities — frustum culling, projection decomposition,
//! unprojection, and Cascaded Shadow Map helpers.
//!
//! The core view/projection matrices live on `Mat4` directly, in
//! `f32/mat4_projection.rs` (one implementation shared across every SIMD
//! backend — see that file's doc comment for why):
//!   - `Mat4::look_at_rh/lh`, `Mat4::look_to_rh/lh`      — view matrix
//!   - `Mat4::perspective_rh/lh` (+ `_gl`)                — perspective projection
//!   - `Mat4::ortho_rh/lh` (+ `_gl`)                      — orthographic projection
//!   - `Mat4::frustum_rh/lh` (+ `_gl`)                    — asymmetric/off-axis frustum
//!
//! Plain name = `z ∈ [0,1]` (ZO — Vulkan/D3D12/Metal/WebGPU, the default).
//! `_gl` suffix = `z ∈ [-1,1]` (NO — legacy OpenGL only).
//!
//! This module adds utilities that operate *on top* of those matrices, split
//! by concern (mirrors the shared-root + focused-submodule shape of `f32/`,
//! not a per-arch backend split — there's nothing here that's SIMD-specific):
//!
//! | Submodule    | Concern                                              |
//! |--------------|-------------------------------------------------------|
//! | `frustum`    | Visibility culling (sphere, AABB, point)              |
//! | `ray`        | Screen-space → world-space rays (unproject, picking)  |
//! | `projection` | Infinite/reversed-Z perspective, decompose, resize    |
//! | `csm`        | Cascaded Shadow Map split depths + per-cascade corners |

pub mod frustum;
pub mod projection;
pub mod ray;
pub mod csm;

pub use frustum::{
    Frustum, Visibility,
    FRUSTUM_LEFT, FRUSTUM_RIGHT, FRUSTUM_BOTTOM,
    FRUSTUM_TOP,  FRUSTUM_NEAR,  FRUSTUM_FAR,
};

pub use projection::{
    PerspectiveParams,
    // Right-handed
    perspective_infinite_rh,
    perspective_infinite_rh_gl,
    perspective_reversed_z_rh,
    // Left-handed
    perspective_infinite_lh,
    perspective_reversed_z_lh,
    // Decompose / resize
    perspective_decompose,
    perspective_resize,
};

pub use ray::{unproject, unproject_separate, picking_ray};

pub use csm::{csm_split_depths, sub_frustum_corners, CSM_N};
