// crates/mid-math/src/geometry/d3/shapes/sphere.rs
//! 3D sphere.

use crate::{Vec3, EPSILON};
use super::aabb::AABB;

/// 3D sphere. 16 bytes.
///
/// Good BVH leaf for round/compact geometry. Prefer AABB for elongated shapes.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct Sphere {
    pub center: Vec3,
    pub radius: f32,
}

impl Sphere {
    #[inline(always)]
    pub fn new(center: Vec3, radius: f32) -> Self { Self { center, radius } }
    #[inline]
    pub fn unit() -> Self { Self::new(Vec3::ZERO, 1.0) }

    #[inline] pub fn is_valid(self) -> bool { self.radius >= 0.0 }
    #[inline] pub fn surface_area(self) -> f32 { 4.0 * core::f32::consts::PI * self.radius * self.radius }
    #[inline] pub fn volume(self) -> f32 { (4.0 / 3.0) * core::f32::consts::PI * self.radius * self.radius * self.radius }

    /// True if `p` is inside or on the boundary.
    #[inline]
    pub fn contains_point(self, p: Vec3) -> bool {
        (p - self.center).length_sq() <= self.radius * self.radius
    }

    /// True if this sphere overlaps `other`.
    #[inline]
    pub fn intersects_sphere(self, other: &Self) -> bool {
        let combined = self.radius + other.radius;
        (self.center - other.center).length_sq() <= combined * combined
    }

    /// True if this sphere overlaps an AABB.
    #[inline]
    pub fn intersects_aabb(self, aabb: &AABB) -> bool {
        aabb.intersects_sphere(self)
    }

    /// Signed distance from `p` to the sphere surface (negative = inside).
    #[inline]
    pub fn signed_distance(self, p: Vec3) -> f32 {
        (p - self.center).length() - self.radius
    }

    /// Bounding AABB that exactly contains this sphere.
    #[inline]
    pub fn bounding_aabb(self) -> AABB {
        let r = Vec3::splat(self.radius);
        AABB::new(self.center - r, self.center + r)
    }

    /// Merge two spheres into a sphere that contains both (approximate).
    pub fn merge(self, other: Self) -> Self {
        let d = other.center - self.center;
        let dist = d.length();
        if dist + other.radius <= self.radius { return self; }
        if dist + self.radius <= other.radius { return other; }
        let new_radius = (dist + self.radius + other.radius) * 0.5;
        let new_center = if dist > EPSILON {
            self.center + d * ((new_radius - self.radius) / dist)
        } else {
            self.center
        };
        Self::new(new_center, new_radius)
    }
}

impl Default for Sphere { fn default() -> Self { Self::new(Vec3::ZERO, 0.0) } }
