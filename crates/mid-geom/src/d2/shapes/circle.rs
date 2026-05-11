// crates/mid-geom/src/d2/shapes/circle.rs
//! 2D circle.

use mid_math::Vec2;
use super::rect::Rect;

/// 2D circle defined by a center and radius. 12 bytes.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct Circle {
    pub center: Vec2,
    pub radius: f32,
}

impl Circle {
    #[inline(always)]
    pub fn new(center: Vec2, radius: f32) -> Self { Self { center, radius } }

    #[inline] pub fn area(self) -> f32 { core::f32::consts::PI * self.radius * self.radius }
    #[inline] pub fn circumference(self) -> f32 { 2.0 * core::f32::consts::PI * self.radius }
    #[inline] pub fn is_valid(self) -> bool { self.radius >= 0.0 }

    #[inline]
    pub fn contains_point(self, p: Vec2) -> bool {
        (p - self.center).length_sq() <= self.radius * self.radius
    }

    #[inline]
    pub fn intersects_circle(self, other: &Circle) -> bool {
        let combined = self.radius + other.radius;
        (self.center - other.center).length_sq() <= combined * combined
    }

    #[inline]
    pub fn intersects_rect(self, rect: &Rect) -> bool {
        rect.intersects_circle(&self)
    }

    #[inline]
    pub fn bounding_rect(self) -> Rect {
        let r = Vec2::splat(self.radius);
        Rect::new(self.center - r, self.center + r)
    }
}

impl Default for Circle {
    fn default() -> Self { Self::new(Vec2::ZERO, 0.0) }
}
