// crates/mid-math/src/ffi/geometry.rs
//! C-ABI types and exports for geometry primitives.
//!
//! Phase 3C stub — empty until geometry/ module is implemented.
//!
//! Planned types:
//!   CTransform  — { position: CVec3, rotation: CQuat, scale: CVec3 }
//!   CAABB       — { min: CVec3, max: CVec3 }
//!   CSphere     — { center: CVec3, radius: f32 }
//!   CPlane      — { normal: CVec3, d: f32 }
//!   CRay3       — { origin: CVec3, direction: CVec3 }
