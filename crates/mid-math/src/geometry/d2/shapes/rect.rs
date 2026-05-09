// crates/mid-math/src/geometry/d2/shapes/rect.rs
//! Axis-aligned 2D rectangle.

use crate::{Vec2, EPSILON};
use super::circle::Circle;

/// Axis-aligned 2D rectangle defined by its minimum and maximum corners.
///
/// 16 bytes. Always axis-aligned — use `Transform2D` for oriented boxes.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct Rect {
    pub min: Vec2,
    pub max: Vec2,
}

impl Rect {
    pub const ZERO: Self = Self { min: Vec2::ZERO, max: Vec2::ZERO };
    pub const UNIT: Self = Self { min: Vec2::ZERO, max: Vec2::ONE  };

    /// Create from explicit min/max. Callers are responsible for min ≤ max.
    #[inline(always)]
    pub fn new(min: Vec2, max: Vec2) -> Self { Self { min, max } }

    /// Create from center and half-extents.
    #[inline]
    pub fn from_center_half_extents(center: Vec2, half: Vec2) -> Self {
        Self { min: center - half, max: center + half }
    }

    /// Create from center and full size.
    #[inline]
    pub fn from_center_size(center: Vec2, size: Vec2) -> Self {
        Self::from_center_half_extents(center, size * 0.5)
    }

    // ── Queries ───────────────────────────────────────────────────────────────

    #[inline] pub fn center(self) -> Vec2  { (self.min + self.max) * 0.5 }
    #[inline] pub fn size(self) -> Vec2    { self.max - self.min }
    #[inline] pub fn half(self) -> Vec2    { self.size() * 0.5 }
    #[inline] pub fn width(self) -> f32    { self.max.x - self.min.x }
    #[inline] pub fn height(self) -> f32   { self.max.y - self.min.y }
    #[inline] pub fn area(self) -> f32     { self.width() * self.height() }
    #[inline] pub fn perimeter(self) -> f32 { 2.0 * (self.width() + self.height()) }
    #[inline] pub fn is_valid(self) -> bool { self.min.x <= self.max.x && self.min.y <= self.max.y }

    /// True if `point` lies inside or on the boundary.
    #[inline]
    pub fn contains_point(self, p: Vec2) -> bool {
        p.x >= self.min.x && p.x <= self.max.x
            && p.y >= self.min.y && p.y <= self.max.y
    }

    /// True if this rect overlaps `other` (touching counts as overlap).
    #[inline]
    pub fn intersects_rect(self, other: Self) -> bool {
        self.min.x <= other.max.x && self.max.x >= other.min.x
            && self.min.y <= other.max.y && self.max.y >= other.min.y
    }

    /// True if this rect overlaps a circle (nearest-point method).
    #[inline]
    pub fn intersects_circle(self, c: &Circle) -> bool {
        let nearest = Vec2::new(
            c.center.x.clamp(self.min.x, self.max.x),
            c.center.y.clamp(self.min.y, self.max.y),
        );
        (c.center - nearest).length_sq() <= c.radius * c.radius
    }

    /// Closest point on (or inside) this rect to `p`.
    #[inline]
    pub fn closest_point(self, p: Vec2) -> Vec2 {
        Vec2::new(
            p.x.clamp(self.min.x, self.max.x),
            p.y.clamp(self.min.y, self.max.y),
        )
    }

    // ── Mutation ──────────────────────────────────────────────────────────────

    /// Expand this rect to include `point`.
    #[inline]
    pub fn expand_to_include(self, p: Vec2) -> Self {
        Self {
            min: self.min.min(p),
            max: self.max.max(p),
        }
    }

    /// Merge two rects into their union.
    #[inline]
    pub fn merge(self, other: Self) -> Self {
        Self {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }

    /// Grow by `amount` in all directions.
    #[inline]
    pub fn expand(self, amount: f32) -> Self {
        let v = Vec2::splat(amount);
        Self { min: self.min - v, max: self.max + v }
    }

    /// Intersection (overlap) of two rects. Returns `None` if they don't overlap.
    #[inline]
    pub fn intersection(self, other: Self) -> Option<Self> {
        let r = Self {
            min: self.min.max(other.min),
            max: self.max.min(other.max),
        };
        if r.is_valid() { Some(r) } else { None }
    }
}

impl Default for Rect { fn default() -> Self { Self::ZERO } }
