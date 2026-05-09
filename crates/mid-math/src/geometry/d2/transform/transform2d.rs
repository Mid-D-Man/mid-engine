// crates/mid-math/src/geometry/d2/transform/transform2d.rs
//! 2D transform — position, rotation (radians), scale.

use crate::{Vec2, Mat2, EPSILON};

/// 2D transform: position + rotation (radians) + scale.
///
/// 20 bytes. Stored decomposed for easy manipulation.
/// Convert to `Mat3` for rendering or physics pipeline use.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct Transform2D {
    pub position: Vec2,
    /// Counter-clockwise rotation in radians.
    pub rotation: f32,
    pub scale:    Vec2,
}

impl Transform2D {
    pub const IDENTITY: Self = Self {
        position: Vec2::ZERO,
        rotation: 0.0,
        scale:    Vec2::ONE,
    };

    #[inline(always)]
    pub fn from_position(position: Vec2) -> Self {
        Self { position, ..Self::IDENTITY }
    }

    #[inline(always)]
    pub fn from_rotation(rotation: f32) -> Self {
        Self { rotation, ..Self::IDENTITY }
    }

    #[inline(always)]
    pub fn from_scale(scale: Vec2) -> Self {
        Self { scale, ..Self::IDENTITY }
    }

    #[inline(always)]
    pub fn new(position: Vec2, rotation: f32, scale: Vec2) -> Self {
        Self { position, rotation, scale }
    }

    // ── Transform helpers ─────────────────────────────────────────────────────

    /// Transform a point: scale → rotate → translate.
    #[inline]
    pub fn transform_point(self, p: Vec2) -> Vec2 {
        let scaled   = Vec2::new(p.x * self.scale.x, p.y * self.scale.y);
        let rotated  = self.rotation_matrix() * scaled;
        rotated + self.position
    }

    /// Transform a direction vector: scale → rotate (no translation).
    #[inline]
    pub fn transform_vector(self, v: Vec2) -> Vec2 {
        let scaled = Vec2::new(v.x * self.scale.x, v.y * self.scale.y);
        self.rotation_matrix() * scaled
    }

    /// Inverse-transform a point (undo translate → rotate → scale).
    #[inline]
    pub fn inverse_transform_point(self, p: Vec2) -> Vec2 {
        let translated = p - self.position;
        let unrotated  = self.rotation_matrix_inv() * translated;
        Vec2::new(
            if self.scale.x.abs() > EPSILON { unrotated.x / self.scale.x } else { 0.0 },
            if self.scale.y.abs() > EPSILON { unrotated.y / self.scale.y } else { 0.0 },
        )
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    #[inline]
    fn rotation_matrix(self) -> Mat2 { Mat2::from_angle(self.rotation) }

    #[inline]
    fn rotation_matrix_inv(self) -> Mat2 { Mat2::from_angle(-self.rotation) }

    // ── Composition ───────────────────────────────────────────────────────────

    /// Compose: `self * rhs` applies `rhs` first, then `self`.
    #[inline]
    pub fn compose(self, rhs: Self) -> Self {
        Self {
            position: self.transform_point(rhs.position),
            rotation: self.rotation + rhs.rotation,
            scale:    Vec2::new(self.scale.x * rhs.scale.x, self.scale.y * rhs.scale.y),
        }
    }
}

impl Default for Transform2D { fn default() -> Self { Self::IDENTITY } }
