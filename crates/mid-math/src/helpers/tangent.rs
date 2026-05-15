// crates/mid-math/src/tangent.rs
//! Tangent-space frame (TBN) for per-pixel normal mapping.
//!
//! A TangentFrame defines a local coordinate system at each vertex/fragment:
//!   N = Normal    — points away from surface
//!   T = Tangent   — points in the U direction of the UV map
//!   B = Bitangent — points in the V direction of the UV map (B = N × T)
//!
//! Normal maps store offsets in tangent space. The TBN matrix transforms them
//! to world space: `world_normal = TBN * tangent_space_normal`.
//!
//! Handedness: B = sign * (N × T). Stored as T.w to save space in GPU buffers.

use core::fmt;
use crate::{Mat3, Vec2, Vec3};

/// Tangent-space frame. Normal, Tangent, Bitangent — all unit vectors.
///
/// Always orthogonalise after construction if the source data may have
/// precision issues (`orthogonalise()` method).
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct TangentFrame {
    /// Surface normal — points away from the surface.
    pub normal:    Vec3,
    /// Tangent — along UV.x (U direction).
    pub tangent:   Vec3,
    /// Bitangent — along UV.y (V direction). Equal to `sign * (normal × tangent)`.
    pub bitangent: Vec3,
}

/// Tangent stored as Vec4 for GPU packing: (tx, ty, tz, handedness).
///
/// `w = +1.0` or `w = -1.0` (handedness / chirality).
/// Bitangent = w * cross(normal, tangent.xyz).
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct PackedTangent {
    pub tangent:    Vec3,
    /// `+1.0` or `-1.0`.
    pub handedness: f32,
}

impl TangentFrame {
    // ── Constructors ──────────────────────────────────────────────────────────

    /// Build from explicit components. Does NOT normalise — caller must ensure unit vectors.
    #[inline(always)]
    pub fn new(normal: Vec3, tangent: Vec3, bitangent: Vec3) -> Self {
        Self { normal, tangent, bitangent }
    }

    /// Build from normal + tangent, computing bitangent automatically.
    ///
    /// `handedness`: `+1.0` for right-handed, `-1.0` for left-handed UVs.
    #[inline]
    pub fn from_normal_tangent(normal: Vec3, tangent: Vec3, handedness: f32) -> Self {
        let bitangent = normal.cross(tangent) * handedness;
        Self { normal: normal.normalize(), tangent: tangent.normalize(), bitangent }
    }

    /// Compute from a triangle (p0/p1/p2) and its UV coordinates.
    ///
    /// Returns `None` if the triangle is degenerate or UVs are degenerate.
    pub fn from_triangle(
        p0: Vec3, p1: Vec3, p2: Vec3,
        uv0: Vec2, uv1: Vec2, uv2: Vec2,
        normal: Vec3,
    ) -> Option<Self> {
        let edge1 = p1 - p0;
        let edge2 = p2 - p0;
        let delta_uv1 = uv1 - uv0;
        let delta_uv2 = uv2 - uv0;

        let det = delta_uv1.x * delta_uv2.y - delta_uv1.y * delta_uv2.x;
        if det.abs() < 1e-8 { return None; }
        let inv = 1.0 / det;

        let tangent = (edge1 * delta_uv2.y - edge2 * delta_uv1.y) * inv;
        let bitangent = (edge2 * delta_uv1.x - edge1 * delta_uv2.x) * inv;

        let tangent   = tangent.normalize();
        let bitangent = bitangent.normalize();
        let normal    = normal.normalize();

        if !tangent.is_finite() || !bitangent.is_finite() { return None; }
        Some(Self { normal, tangent, bitangent })
    }

    // ── Operations ────────────────────────────────────────────────────────────

    /// Convert tangent-space normal to world space.
    ///
    /// Input `n` is typically read from a normal map: `(nx, ny, nz) ∈ [-1, 1]`.
    /// Output is a world-space unit normal.
    #[inline]
    pub fn transform_normal(self, n: Vec3) -> Vec3 {
        (self.tangent   * n.x
       + self.bitangent * n.y
       + self.normal    * n.z).normalize()
    }

    /// Convert world-space normal to tangent space (for baking, projection).
    #[inline]
    pub fn to_tangent_space(self, world_n: Vec3) -> Vec3 {
        Vec3::new(
            world_n.dot(self.tangent),
            world_n.dot(self.bitangent),
            world_n.dot(self.normal),
        )
    }

    /// Re-orthogonalise using Gram-Schmidt. Call if source data has precision issues.
    pub fn orthogonalise(self) -> Self {
        let n = self.normal.normalize();
        // Project tangent onto plane perpendicular to N
        let t = (self.tangent - n * n.dot(self.tangent)).normalize();
        // Recompute bitangent to guarantee orthogonality and handedness
        let det = n.cross(self.tangent).dot(self.bitangent);
        let sign = if det >= 0.0 { 1.0 } else { -1.0 };
        let b = n.cross(t) * sign;
        Self { normal: n, tangent: t, bitangent: b }
    }

    /// Handedness: `+1.0` if right-handed, `-1.0` if left-handed.
    #[inline]
    pub fn handedness(self) -> f32 {
        if self.normal.cross(self.tangent).dot(self.bitangent) >= 0.0 { 1.0 } else { -1.0 }
    }

    /// Pack to GPU-friendly `Vec3 + f32` (tangent.xyz + handedness).
    #[inline]
    pub fn pack(self) -> PackedTangent {
        PackedTangent { tangent: self.tangent, handedness: self.handedness() }
    }

    /// Unpack from GPU format given a surface normal.
    #[inline]
    pub fn unpack(p: PackedTangent, normal: Vec3) -> Self {
        let bitangent = normal.cross(p.tangent) * p.handedness;
        Self { normal, tangent: p.tangent, bitangent }
    }

    /// Build the 3×3 TBN matrix: columns are T, B, N.
    #[inline]
    pub fn to_mat3(self) -> Mat3 {
        Mat3::from_vecs(self.tangent, self.bitangent, self.normal)
    }
}

impl fmt::Display for TangentFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TBN(n={}, t={}, b={})", self.normal, self.tangent, self.bitangent)
    }
}
