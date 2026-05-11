// crates/mid-geom/src/d3/transform/transform3d.rs
//! 3D decomposed TRS transform — position, rotation, scale.

use mid_math::{Affine3, Mat4, Quat, Vec3, EPSILON};

/// 3D transform: position + rotation (quaternion) + scale.
///
/// 40 bytes. Stored decomposed — easy to interpolate, safe to compose.
/// Convert to `Mat4` or `Affine3` for transform math.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct Transform {
    pub position: Vec3,
    pub rotation: Quat,
    pub scale:    Vec3,
}

impl Transform {
    pub const IDENTITY: Self = Self {
        position: Vec3::ZERO,
        rotation: Quat::IDENTITY,
        scale:    Vec3::ONE,
    };

    // ── Constructors ──────────────────────────────────────────────────────────

    #[inline(always)] pub fn from_position(p: Vec3) -> Self { Self { position: p, ..Self::IDENTITY } }
    #[inline(always)] pub fn from_rotation(r: Quat) -> Self { Self { rotation: r, ..Self::IDENTITY } }
    #[inline(always)] pub fn from_scale(s: Vec3)    -> Self { Self { scale: s,    ..Self::IDENTITY } }
    #[inline(always)] pub fn from_trs(position: Vec3, rotation: Quat, scale: Vec3) -> Self {
        Self { position, rotation, scale }
    }

    // ── Conversion ────────────────────────────────────────────────────────────

    /// Build a `Mat4` from this transform (scale → rotate → translate).
    #[inline]
    pub fn to_mat4(self) -> Mat4 {
        Mat4::from_trs(self.position, self.rotation, self.scale)
    }

    /// Build an `Affine3` from this transform (more efficient than Mat4 for TRS).
    #[inline]
    pub fn to_affine3(self) -> Affine3 {
        Affine3::from_trs(self.position, self.rotation, self.scale)
    }

    // ── Transform helpers ─────────────────────────────────────────────────────

    /// Transform a point: scale → rotate → translate.
    #[inline]
    pub fn transform_point(self, p: Vec3) -> Vec3 {
        let scaled = Vec3::new(
            p.x * self.scale.x,
            p.y * self.scale.y,
            p.z * self.scale.z,
        );
        self.rotation.rotate(scaled) + self.position
    }

    /// Transform a direction: scale → rotate (no translation).
    #[inline]
    pub fn transform_vector(self, v: Vec3) -> Vec3 {
        let scaled = Vec3::new(
            v.x * self.scale.x,
            v.y * self.scale.y,
            v.z * self.scale.z,
        );
        self.rotation.rotate(scaled)
    }

    /// Transform a direction by rotation only (no scale, no translation).
    #[inline(always)]
    pub fn transform_direction(self, v: Vec3) -> Vec3 { self.rotation.rotate(v) }

    /// Inverse-transform a point: undo translate → rotate → scale.
    ///
    /// Valid for any non-zero scale. Zero-scale axes produce zero output.
    #[inline]
    pub fn inverse_transform_point(self, p: Vec3) -> Vec3 {
        let translated = p - self.position;
        let unrotated  = self.rotation.conjugate().rotate(translated);
        Vec3::new(
            if self.scale.x.abs() > EPSILON { unrotated.x / self.scale.x } else { 0.0 },
            if self.scale.y.abs() > EPSILON { unrotated.y / self.scale.y } else { 0.0 },
            if self.scale.z.abs() > EPSILON { unrotated.z / self.scale.z } else { 0.0 },
        )
    }

    /// Inverse-transform a direction: undo scale → rotate.
    #[inline]
    pub fn inverse_transform_vector(self, v: Vec3) -> Vec3 {
        let unrotated = self.rotation.conjugate().rotate(v);
        Vec3::new(
            if self.scale.x.abs() > EPSILON { unrotated.x / self.scale.x } else { 0.0 },
            if self.scale.y.abs() > EPSILON { unrotated.y / self.scale.y } else { 0.0 },
            if self.scale.z.abs() > EPSILON { unrotated.z / self.scale.z } else { 0.0 },
        )
    }

    // ── Composition ───────────────────────────────────────────────────────────

    /// Compose: `self * child` applies `child` in `self`'s local space.
    #[inline]
    pub fn compose(self, child: Self) -> Self {
        Self {
            position: self.transform_point(child.position),
            rotation: (self.rotation * child.rotation).normalize(),
            scale:    Vec3::new(
                self.scale.x * child.scale.x,
                self.scale.y * child.scale.y,
                self.scale.z * child.scale.z,
            ),
        }
    }

    // ── Interpolation ─────────────────────────────────────────────────────────

    /// Linear interpolation between two transforms.
    ///
    /// Position and scale are lerped; rotation uses slerp for correct arc.
    #[inline]
    pub fn lerp(self, rhs: Self, t: f32) -> Self {
        Self {
            position: self.position.lerp(rhs.position, t),
            rotation: self.rotation.slerp(rhs.rotation, t),
            scale:    self.scale.lerp(rhs.scale, t),
        }
    }

    // ── Predicates ────────────────────────────────────────────────────────────

    #[inline]
    pub fn is_finite(self) -> bool {
        self.position.is_finite() && self.rotation.is_finite() && self.scale.is_finite()
    }
}

impl Default for Transform { fn default() -> Self { Self::IDENTITY } }

impl From<Mat4> for Transform {
    /// Extract approximate TRS from a Mat4.
    /// Assumes the matrix was built from TRS — shear is not handled.
    fn from(m: Mat4) -> Self {
        let position = Vec3::new(m.cols[3][0], m.cols[3][1], m.cols[3][2]);
        let sx = Vec3::new(m.cols[0][0], m.cols[0][1], m.cols[0][2]).length();
        let sy = Vec3::new(m.cols[1][0], m.cols[1][1], m.cols[1][2]).length();
        let sz = Vec3::new(m.cols[2][0], m.cols[2][1], m.cols[2][2]).length();
        // Placeholder rotation — full extraction via Affine3 is the correct path
        // for accurate results when starting from a Mat4.
        let rotation = Quat::from_axis_angle(Vec3::Y, 0.0);
        Self { position, rotation, scale: Vec3::new(sx, sy, sz) }
    }
  }
