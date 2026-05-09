// crates/mid-math/src/geometry/d3/shapes/capsule.rs
//! 3D capsule — cylinder with hemispherical caps.

use crate::{Vec3, EPSILON};
use super::sphere::Sphere;
use super::aabb::AABB;

/// 3D capsule defined by a line segment (base→tip) and a radius.
///
/// Useful for character controllers, swept collision tests.
/// 28 bytes.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct Capsule {
    /// Bottom center of the cylindrical body.
    pub base:   Vec3,
    /// Top center of the cylindrical body.
    pub tip:    Vec3,
    pub radius: f32,
}

impl Capsule {
    #[inline(always)]
    pub fn new(base: Vec3, tip: Vec3, radius: f32) -> Self { Self { base, tip, radius } }

    #[inline] pub fn center(self) -> Vec3  { (self.base + self.tip) * 0.5 }
    #[inline] pub fn height(self) -> f32   { (self.tip - self.base).length() + 2.0 * self.radius }
    #[inline] pub fn is_valid(self) -> bool { self.radius >= 0.0 }

    /// Closest point on the capsule axis segment to `p`.
    #[inline]
    pub fn closest_point_on_axis(self, p: Vec3) -> Vec3 {
        let ab  = self.tip - self.base;
        let len_sq = ab.dot(ab);
        if len_sq < EPSILON { return self.base; }
        let t = ((p - self.base).dot(ab) / len_sq).clamp(0.0, 1.0);
        self.base + ab * t
    }

    /// True if `p` lies inside the capsule.
    #[inline]
    pub fn contains_point(self, p: Vec3) -> bool {
        let closest = self.closest_point_on_axis(p);
        (p - closest).length_sq() <= self.radius * self.radius
    }

    /// True if this capsule overlaps `sphere`.
    #[inline]
    pub fn intersects_sphere(self, sphere: &Sphere) -> bool {
        let closest = self.closest_point_on_axis(sphere.center);
        let combined = self.radius + sphere.radius;
        (sphere.center - closest).length_sq() <= combined * combined
    }

    /// True if this capsule overlaps `other` capsule.
    pub fn intersects_capsule(self, other: &Capsule) -> bool {
        // Distance between the two line segments (Ericson, Real-Time Collision Detection §5.1.9)
        let (c1, c2) = closest_points_on_segments(
            self.base, self.tip,
            other.base, other.tip,
        );
        let combined = self.radius + other.radius;
        (c1 - c2).length_sq() <= combined * combined
    }

    /// Conservative bounding AABB.
    #[inline]
    pub fn bounding_aabb(self) -> AABB {
        let r = Vec3::splat(self.radius);
        let base_box = AABB::new(self.base - r, self.base + r);
        let tip_box  = AABB::new(self.tip  - r, self.tip  + r);
        base_box.merge(tip_box)
    }
}

// ── Private helpers ───────────────────────────────────────────────────────────

/// Closest points between two line segments.
/// Returns `(point_on_seg1, point_on_seg2)`.
/// Ericson — Real-Time Collision Detection §5.1.9.
fn closest_points_on_segments(p1: Vec3, q1: Vec3, p2: Vec3, q2: Vec3) -> (Vec3, Vec3) {
    let d1 = q1 - p1;
    let d2 = q2 - p2;
    let r  = p1 - p2;
    let a  = d1.dot(d1);
    let e  = d2.dot(d2);
    let f  = d2.dot(r);

    if a <= EPSILON && e <= EPSILON { return (p1, p2); }

    let (mut s, mut t);

    if a <= EPSILON {
        s = 0.0;
        t = (f / e).clamp(0.0, 1.0);
    } else {
        let c = d1.dot(r);
        if e <= EPSILON {
            t = 0.0;
            s = (-c / a).clamp(0.0, 1.0);
        } else {
            let b     = d1.dot(d2);
            let denom = a * e - b * b;
            s = if denom.abs() > EPSILON {
                ((b * f - c * e) / denom).clamp(0.0, 1.0)
            } else {
                0.0
            };
            t = (b * s + f) / e;
            if t < 0.0 {
                t = 0.0;
                s = (-c / a).clamp(0.0, 1.0);
            } else if t > 1.0 {
                t = 1.0;
                s = ((b - c) / a).clamp(0.0, 1.0);
            }
        }
    }

    (p1 + d1 * s, p2 + d2 * t)
}

impl Default for Capsule {
    fn default() -> Self { Self::new(Vec3::NEG_Y, Vec3::Y, 0.5) }
  }
