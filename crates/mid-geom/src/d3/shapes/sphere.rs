// crates/mid-geom/src/d3/shapes/sphere.rs
//! 3D sphere.

use mid_math::Vec3;
use super::aabb::AABB;

/// 3D sphere. 16 bytes.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct Sphere {
    pub center: Vec3,
    pub radius: f32,
}

impl Sphere {
    #[inline(always)] pub fn new(center: Vec3, radius: f32) -> Self { Self { center, radius } }
    #[inline] pub fn unit() -> Self { Self::new(Vec3::ZERO, 1.0) }
    #[inline] pub fn is_valid(self) -> bool { self.radius >= 0.0 }
    #[inline] pub fn surface_area(self) -> f32 { 4.0 * core::f32::consts::PI * self.radius * self.radius }
    #[inline] pub fn volume(self) -> f32 {
        (4.0 / 3.0) * core::f32::consts::PI * self.radius * self.radius * self.radius
    }

    #[inline]
    pub fn contains_point(self, p: Vec3) -> bool {
        (p - self.center).length_sq() <= self.radius * self.radius
    }

    #[inline]
    pub fn intersects_sphere(self, other: &Self) -> bool {
        let combined = self.radius + other.radius;
        (self.center - other.center).length_sq() <= combined * combined
    }

    #[inline]
    pub fn intersects_aabb(self, aabb: &AABB) -> bool {
        aabb.intersects_sphere(&self)
    }

    #[inline]
    pub fn signed_distance(self, p: Vec3) -> f32 {
        (p - self.center).length() - self.radius
    }

    #[inline]
    pub fn bounding_aabb(self) -> AABB {
        let r = Vec3::splat(self.radius);
        AABB::new(self.center - r, self.center + r)
    }

    pub fn merge(self, other: Self) -> Self {
        let d = other.center - self.center;
        let dist = d.length();
        if dist + other.radius <= self.radius { return self; }
        if dist + self.radius <= other.radius { return other; }
        let new_radius = (dist + self.radius + other.radius) * 0.5;
        let new_center = if dist > 1e-6 {
            self.center + d * ((new_radius - self.radius) / dist)
        } else {
            self.center
        };
        Self::new(new_center, new_radius)
    }
}

impl Default for Sphere { fn default() -> Self { Self::new(Vec3::ZERO, 0.0) } }
