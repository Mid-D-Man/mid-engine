// crates/mid-math/src/camera/mod.rs
//! Camera math utilities — frustum culling, projection decomposition,
//! unprojection, and Cascaded Shadow Map helpers.
//!
//! The core view/projection matrices live on `Mat4` directly:
//!   - `Mat4::look_at_rh` / `Mat4::look_at_lh`        — view matrix
//!   - `Mat4::perspective_rh` / `Mat4::perspective_lh` — perspective projection
//!   - `Mat4::ortho_rh` / `Mat4::ortho_lh`             — orthographic projection
//!
//! This module adds utilities that operate *on top* of those matrices:
//!
//! | Utility                       | Use case                                  |
//! |-------------------------------|-------------------------------------------|
//! | `Frustum`                     | Visibility culling (sphere, AABB, point)  |
//! | `unproject`                   | Mouse picking, ray casting                |
//! | `picking_ray`                 | Click-to-world-space ray                  |
//! | `perspective_infinite_rh/lh`  | Sky / outer-space rendering               |
//! | `perspective_reversed_z_rh/lh`| Improved depth precision (Vulkan/DX12)    |
//! | `perspective_decompose`       | Read back fov/near/far from a matrix      |
//! | `perspective_resize`          | Update aspect ratio cheaply               |
//! | `csm_split_depths`            | Cascaded Shadow Map depth splits          |
//! | `sub_frustum_corners`         | Per-cascade world-space bounding boxes    |

pub mod frustum;
pub mod projection;

pub use frustum::{
    Frustum, Visibility,
    FRUSTUM_LEFT, FRUSTUM_RIGHT, FRUSTUM_BOTTOM,
    FRUSTUM_TOP,  FRUSTUM_NEAR,  FRUSTUM_FAR,
};

pub use projection::{
    PerspectiveParams,
    unproject,
    unproject_separate,
    picking_ray,
    // Right-handed
    perspective_infinite_rh,
    perspective_reversed_z_rh,
    // Left-handed
    perspective_infinite_lh,
    perspective_reversed_z_lh,
    // Decompose / resize / CSM
    perspective_decompose,
    perspective_resize,
    csm_split_depths,
    sub_frustum_corners,
};
