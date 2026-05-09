// crates/mid-math/src/geometry/d2/raycast/ray2.rs
//! 2D ray — origin + direction.

use crate::{Vec2, EPSILON};
use super::super::shapes::{Circle, Rect};

/// 2D ray. 16 bytes.
///
/// `direction` should be normalised. Use `Ray2::new` to auto-normalise,
/// or `Ray2::new_unnormalized` if you control the direction yourself.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct Ray2 {
    pub origin:    Vec2,
    pub direction: Vec2,
}

/// Result of a 2D ray intersection test.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hit2D {
    /// Distance along the ray to the hit point.
    pub t: f32,
    /// Hit point in world space.
    pub point: Vec2,
}

impl Ray2 {
    /// Create a new ray, normalising `direction`. Returns `None` if direction is zero.
    #[inline]
    pub fn new(origin: Vec2, direction: Vec2) -> Option<Self> {
        let n = direction.normalize();
        if n.length_sq() < EPSILON {
            None
        } else {
            Some(Self { origin, direction: n })
        }
    }

    /// Create without normalising. Use when direction is already unit length.
    #[inline(always)]
    pub fn new_unnormalized(origin: Vec2, direction: Vec2) -> Self {
        Self { origin, direction }
    }

    /// Point along the ray at parameter `t`.
    #[inline(always)]
    pub fn at(self, t: f32) -> Vec2 { self.origin + self.direction * t }

    /// Intersect against an axis-aligned rect — slab method.
    ///
    /// Returns the nearest `t` (entry point). `t` may be negative if origin is inside.
    pub fn intersect_rect(self, rect: &Rect) -> Option<Hit2D> {
        let inv_dx = if self.direction.x.abs() > EPSILON { 1.0 / self.direction.x } else { f32::INFINITY };
        let inv_dy = if self.direction.y.abs() > EPSILON { 1.0 / self.direction.y } else { f32::INFINITY };

        let tx1 = (rect.min.x - self.origin.x) * inv_dx;
        let tx2 = (rect.max.x - self.origin.x) * inv_dx;
        let ty1 = (rect.min.y - self.origin.y) * inv_dy;
        let ty2 = (rect.max.y - self.origin.y) * inv_dy;

        let t_min = tx1.min(tx2).max(ty1.min(ty2));
        let t_max = tx1.max(tx2).min(ty1.max(ty2));

        if t_max < 0.0 || t_min > t_max { return None; }
        let t = if t_min < 0.0 { t_max } else { t_min };
        Some(Hit2D { t, point: self.at(t) })
    }

    /// Intersect against a circle — quadratic formula.
    ///
    /// Returns the nearest positive `t` (entry). Returns `None` if no hit
    /// or hit is behind the ray.
    pub fn intersect_circle(self, circle: &Circle) -> Option<Hit2D> {
        let oc = self.origin - circle.center;
        // a = dot(dir, dir) = 1 for normalised rays
        let a = self.direction.dot(self.direction);
        let b = 2.0 * oc.dot(self.direction);
        let c = oc.dot(oc) - circle.radius * circle.radius;
        let disc = b * b - 4.0 * a * c;
        if disc < 0.0 { return None; }
        let sq = disc.sqrt();
        let t0 = (-b - sq) / (2.0 * a);
        let t1 = (-b + sq) / (2.0 * a);
        let t = if t0 >= 0.0 { t0 } else if t1 >= 0.0 { t1 } else { return None; };
        Some(Hit2D { t, point: self.at(t) })
    }
}

impl Default for Ray2 {
    fn default() -> Self { Self { origin: Vec2::ZERO, direction: Vec2::X } }
  }
