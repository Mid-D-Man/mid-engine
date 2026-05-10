// crates/mid-math/src/geometry/d3/shapes/aabb.rs
//! Axis-aligned bounding box (3D).

use crate::Vec3;
use super::sphere::Sphere;

/// Axis-aligned bounding box. 24 bytes (2 × Vec3 without padding).
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct AABB {
    pub min: Vec3,
    pub max: Vec3,
}

impl AABB {
    pub const ZERO: Self = Self { min: Vec3::ZERO, max: Vec3::ZERO };

    /// An invalid (inside-out) AABB useful as an accumulator seed.
    ///
    /// Not a `const` because `Vec3::new` calls SSE2 intrinsics on x86 which
    /// are not available in const context. Use `AABB::invalid()` instead.
    #[inline(always)]
    pub fn invalid() -> Self {
        Self {
            min: Vec3::new( f32::INFINITY,  f32::INFINITY,  f32::INFINITY),
            max: Vec3::new(-f32::INFINITY, -f32::INFINITY, -f32::INFINITY),
        }
    }

    #[inline(always)] pub fn new(min: Vec3, max: Vec3) -> Self { Self { min, max } }

    #[inline]
    pub fn from_center_half_extents(center: Vec3, half: Vec3) -> Self {
        Self { min: center - half, max: center + half }
    }

    #[inline]
    pub fn from_center_size(center: Vec3, size: Vec3) -> Self {
        Self::from_center_half_extents(center, size * 0.5)
    }

    #[inline] pub fn center(self) -> Vec3   { (self.min + self.max) * 0.5 }
    #[inline] pub fn extents(self) -> Vec3  { (self.max - self.min) * 0.5 }
    #[inline] pub fn size(self) -> Vec3     { self.max - self.min }
    #[inline] pub fn is_valid(self) -> bool { self.min.x <= self.max.x && self.min.y <= self.max.y && self.min.z <= self.max.z }

    #[inline]
    pub fn surface_area(self) -> f32 {
        let s = self.size();
        2.0 * (s.x*s.y + s.y*s.z + s.z*s.x)
    }

    #[inline]
    pub fn volume(self) -> f32 { let s=self.size(); s.x*s.y*s.z }

    #[inline]
    pub fn contains_point(self, p: Vec3) -> bool {
        p.x >= self.min.x && p.x <= self.max.x
            && p.y >= self.min.y && p.y <= self.max.y
            && p.z >= self.min.z && p.z <= self.max.z
    }

    #[inline]
    pub fn contains_aabb(self, other: Self) -> bool {
        self.contains_point(other.min) && self.contains_point(other.max)
    }

    #[inline]
    pub fn intersects_aabb(self, other: &Self) -> bool {
        self.min.x <= other.max.x && self.max.x >= other.min.x
            && self.min.y <= other.max.y && self.max.y >= other.min.y
            && self.min.z <= other.max.z && self.max.z >= other.min.z
    }

    #[inline]
    pub fn intersects_sphere(self, sphere: &Sphere) -> bool {
        let nearest = self.closest_point(sphere.center);
        (sphere.center - nearest).length_sq() <= sphere.radius * sphere.radius
    }

    #[inline]
    pub fn closest_point(self, p: Vec3) -> Vec3 {
        Vec3::new(
            p.x.clamp(self.min.x, self.max.x),
            p.y.clamp(self.min.y, self.max.y),
            p.z.clamp(self.min.z, self.max.z),
        )
    }

    #[inline]
    pub fn signed_distance(self, p: Vec3) -> f32 {
        let c = self.center();
        let h = self.extents();
        let q = (p - c).abs() - h;
        let outside = Vec3::new(q.x.max(0.0), q.y.max(0.0), q.z.max(0.0)).length();
        let inside  = q.x.max(q.y).max(q.z).min(0.0);
        outside + inside
    }

    #[inline]
    pub fn expand_to_include_point(self, p: Vec3) -> Self {
        Self { min: self.min.min(p), max: self.max.max(p) }
    }

    #[inline]
    pub fn merge(self, other: Self) -> Self {
        Self { min: self.min.min(other.min), max: self.max.max(other.max) }
    }

    #[inline]
    pub fn expand(self, amount: f32) -> Self {
        let v = Vec3::splat(amount);
        Self { min: self.min - v, max: self.max + v }
    }

    #[inline]
    pub fn transform(self, m: &crate::Mat4) -> Self {
        let t = Vec3::new(m.cols[3][0], m.cols[3][1], m.cols[3][2]);
        let mut out_min = t;
        let mut out_max = t;
        for col in 0..3 {
            for row in 0..3 {
                let a = m.cols[col][row] * match col { 0=>self.min.x, 1=>self.min.y, _=>self.min.z };
                let b = m.cols[col][row] * match col { 0=>self.max.x, 1=>self.max.y, _=>self.max.z };
                let (lo, hi) = if a < b { (a,b) } else { (b,a) };
                match row {
                    0 => { out_min=Vec3::new(out_min.x+lo,out_min.y,out_min.z); out_max=Vec3::new(out_max.x+hi,out_max.y,out_max.z); }
                    1 => { out_min=Vec3::new(out_min.x,out_min.y+lo,out_min.z); out_max=Vec3::new(out_max.x,out_max.y+hi,out_max.z); }
                    _ => { out_min=Vec3::new(out_min.x,out_min.y,out_min.z+lo); out_max=Vec3::new(out_max.x,out_max.y,out_max.z+hi); }
                }
            }
        }
        Self::new(out_min, out_max)
    }
}

impl Default for AABB { fn default() -> Self { Self::ZERO } }
