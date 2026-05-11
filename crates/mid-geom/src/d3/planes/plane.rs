// crates/mid-geom/src/d3/planes/plane.rs
//! 3D plane — infinite surface defined by normal + offset.

use mid_math::{Vec3, EPSILON};

/// Which side of a plane a point lies on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlaneSide { Front, On, Behind }

/// 3D plane. 16 bytes.
///
/// Convention: `signed_distance(p) = dot(normal, p) + d`
/// - Positive → in front (normal side)
/// - Negative → behind
/// - Zero     → on the plane
///
/// `normal` must be unit length. Constructors normalise automatically.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct Plane {
    pub normal: Vec3,
    pub d:      f32,
}

impl Plane {
    /// Create from a unit normal and a point on the plane.
    /// `normal` is normalised internally.
    #[inline]
    pub fn from_normal_point(normal: Vec3, point: Vec3) -> Self {
        let n = normal.normalize();
        Self { normal: n, d: -n.dot(point) }
    }

    /// Create from three non-collinear points (CCW winding = outward normal).
    #[inline]
    pub fn from_points(a: Vec3, b: Vec3, c: Vec3) -> Option<Self> {
        let n = (b - a).cross(c - a);
        if n.length_sq() < EPSILON * EPSILON { return None; }
        Some(Self::from_normal_point(n, a))
    }

    /// Create from unnormalised coefficients (ax + by + cz + d = 0).
    /// Used internally by `Frustum::from_mat4`.
    #[inline]
    pub(crate) fn from_coefficients(a: f32, b: f32, c: f32, d: f32) -> Self {
        let len = (a * a + b * b + c * c).sqrt();
        if len < EPSILON {
            return Self { normal: Vec3::Y, d: 0.0 };
        }
        let inv = 1.0 / len;
        Self { normal: Vec3::new(a * inv, b * inv, c * inv), d: d * inv }
    }

    // ── Queries ───────────────────────────────────────────────────────────────

    /// Signed distance from `p` to this plane.
    #[inline(always)]
    pub fn signed_distance(self, p: Vec3) -> f32 { self.normal.dot(p) + self.d }

    /// Which side of the plane `p` lies on (with a thin tolerance band).
    #[inline]
    pub fn classify(self, p: Vec3) -> PlaneSide {
        let d = self.signed_distance(p);
        if d > EPSILON        { PlaneSide::Front  }
        else if d < -EPSILON  { PlaneSide::Behind }
        else                  { PlaneSide::On     }
    }

    /// Project `p` onto the plane surface.
    #[inline]
    pub fn project_point(self, p: Vec3) -> Vec3 { p - self.normal * self.signed_distance(p) }

    /// Reflect `p` through this plane.
    #[inline]
    pub fn reflect_point(self, p: Vec3) -> Vec3 { p - self.normal * (2.0 * self.signed_distance(p)) }

    /// Reflect a direction vector `v` through this plane.
    #[inline]
    pub fn reflect_vector(self, v: Vec3) -> Vec3 { v - self.normal * (2.0 * self.normal.dot(v)) }

    /// Ray-plane intersection — returns `t` such that `origin + t * direction` lies on the plane.
    /// Returns `None` if the ray is parallel to the plane.
    #[inline]
    pub fn intersect_ray(self, origin: Vec3, direction: Vec3) -> Option<f32> {
        let denom = self.normal.dot(direction);
        if denom.abs() < EPSILON { return None; }
        Some(-self.signed_distance(origin) / denom)
    }
}

impl Default for Plane {
    fn default() -> Self { Self { normal: Vec3::Y, d: 0.0 } }
                                    }
