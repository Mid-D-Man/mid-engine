// crates/mid-math/src/geometry/d3/raycast/ray3.rs
//! 3D ray — origin + direction.

use crate::{Vec3, EPSILON};
use super::super::shapes::{AABB, Sphere, Capsule};
use super::super::planes::Plane;

/// 3D ray. 24 bytes.
///
/// `direction` should be unit length. Use `Ray3::new` to auto-normalise.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct Ray3 {
    pub origin:    Vec3,
    pub direction: Vec3,
}

/// Result of a 3D ray intersection test.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hit3D {
    /// Distance along the ray to the nearest hit.
    pub t:      f32,
    /// World-space hit position.
    pub point:  Vec3,
    /// Surface normal at the hit point (outward-facing).
    pub normal: Vec3,
}

impl Ray3 {
    /// Create and normalise direction. Returns `None` if direction is zero.
    #[inline]
    pub fn new(origin: Vec3, direction: Vec3) -> Option<Self> {
        let n = direction.normalize();
        if n.length_sq() < EPSILON {
            None
        } else {
            Some(Self { origin, direction: n })
        }
    }

    /// Create without normalising. Use when direction is already unit length.
    #[inline(always)]
    pub fn new_unnormalized(origin: Vec3, direction: Vec3) -> Self {
        Self { origin, direction }
    }

    /// Point along the ray at parameter `t`.
    #[inline(always)]
    pub fn at(self, t: f32) -> Vec3 { self.origin + self.direction * t }

    // ── Plane ─────────────────────────────────────────────────────────────────

    /// Intersect against a plane.
    ///
    /// Returns `None` if ray is parallel to the plane.
    #[inline]
    pub fn intersect_plane(self, plane: &Plane) -> Option<Hit3D> {
        let t = plane.intersect_ray(self.origin, self.direction)?;
        if t < 0.0 { return None; }
        Some(Hit3D { t, point: self.at(t), normal: plane.normal })
    }

    // ── AABB ──────────────────────────────────────────────────────────────────

    /// Intersect against an AABB — slab method.
    ///
    /// Returns the nearest non-negative `t`. If the origin is inside the AABB,
    /// returns the exit `t` (positive).
    pub fn intersect_aabb(self, aabb: &AABB) -> Option<Hit3D> {
        let inv = |d: f32| if d.abs() > EPSILON { 1.0 / d } else { f32::INFINITY };
        let idx = inv(self.direction.x);
        let idy = inv(self.direction.y);
        let idz = inv(self.direction.z);

        let tx1 = (aabb.min.x - self.origin.x) * idx;
        let tx2 = (aabb.max.x - self.origin.x) * idx;
        let ty1 = (aabb.min.y - self.origin.y) * idy;
        let ty2 = (aabb.max.y - self.origin.y) * idy;
        let tz1 = (aabb.min.z - self.origin.z) * idz;
        let tz2 = (aabb.max.z - self.origin.z) * idz;

        let t_min = tx1.min(tx2).max(ty1.min(ty2)).max(tz1.min(tz2));
        let t_max = tx1.max(tx2).min(ty1.max(ty2)).min(tz1.max(tz2));

        if t_max < 0.0 || t_min > t_max { return None; }

        let t = if t_min < 0.0 { t_max } else { t_min };
        let p = self.at(t);

        // Compute face normal from which face was hit
        let c = aabb.center();
        let h = aabb.extents();
        let local = (p - c) / h; // [-1, 1] per axis
        let ax = local.x.abs();
        let ay = local.y.abs();
        let az = local.z.abs();
        let normal = if ax >= ay && ax >= az {
            Vec3::new(local.x.signum(), 0.0, 0.0)
        } else if ay >= az {
            Vec3::new(0.0, local.y.signum(), 0.0)
        } else {
            Vec3::new(0.0, 0.0, local.z.signum())
        };

        Some(Hit3D { t, point: p, normal })
    }

    // ── Sphere ────────────────────────────────────────────────────────────────

    /// Intersect against a sphere — quadratic formula.
    ///
    /// Returns nearest positive `t`. Returns `None` if no hit or hit is behind.
    pub fn intersect_sphere(self, sphere: &Sphere) -> Option<Hit3D> {
        let oc = self.origin - sphere.center;
        let a  = self.direction.dot(self.direction); // 1.0 if normalised
        let b  = 2.0 * oc.dot(self.direction);
        let c  = oc.dot(oc) - sphere.radius * sphere.radius;
        let disc = b * b - 4.0 * a * c;
        if disc < 0.0 { return None; }
        let sq = disc.sqrt();
        let t0 = (-b - sq) / (2.0 * a);
        let t1 = (-b + sq) / (2.0 * a);
        let t = if t0 >= 0.0 { t0 } else if t1 >= 0.0 { t1 } else { return None; };
        let p = self.at(t);
        let n = (p - sphere.center) * (1.0 / sphere.radius);
        Some(Hit3D { t, point: p, normal: n })
    }

    // ── Capsule ───────────────────────────────────────────────────────────────

    /// Intersect against a capsule.
    ///
    /// Tests the infinite cylinder body and both hemispherical caps,
    /// returns the nearest positive `t`.
    pub fn intersect_capsule(self, capsule: &Capsule) -> Option<Hit3D> {
        let ab  = capsule.tip - capsule.base;
        let ao  = self.origin - capsule.base;
        let r   = capsule.radius;

        let ab_d  = ab.dot(self.direction);
        let ab_ab = ab.dot(ab);
        let ab_ao = ab.dot(ao);

        // ── Cylinder body ──────────────────────────────────────────────────
        let a = ab_ab - ab_d * ab_d;
        let b = ab_ab * ao.dot(self.direction) - ab_ao * ab_d;
        let c = ab_ab * ao.dot(ao) - ab_ao * ab_ao - r * r * ab_ab;

        let mut best: Option<Hit3D> = None;
        let mut update = |h: Hit3D| {
            if best.map_or(true, |b: Hit3D| h.t < b.t) { best = Some(h); }
        };

        if a.abs() > EPSILON {
            let disc = b * b - a * c;
            if disc >= 0.0 {
                let sq = disc.sqrt();
                for &t in &[(-b - sq) / a, (-b + sq) / a] {
                    if t >= 0.0 {
                        let p = self.at(t);
                        let proj = (p - capsule.base).dot(ab) / ab_ab;
                        if proj >= 0.0 && proj <= 1.0 {
                            // Normal is perpendicular to the cylinder axis
                            let on_axis = capsule.base + ab * proj;
                            let n = (p - on_axis) * (1.0 / r);
                            update(Hit3D { t, point: p, normal: n });
                        }
                    }
                }
            }
        }

        // ── Caps ──────────────────────────────────────────────────────────
        for &cap_center in &[capsule.base, capsule.tip] {
            let cap = Sphere::new(cap_center, r);
            if let Some(h) = self.intersect_sphere(&cap) {
                // Only accept if on the hemisphere facing away from the cylinder
                let dir = if core::ptr::eq(&cap_center, &capsule.base) {
                    -ab // base cap points downward
                } else {
                    ab  // tip cap points upward
                };
                if h.normal.dot(dir) <= 0.0 {
                    update(h);
                }
            }
        }

        best
    }
}

impl Default for Ray3 {
    fn default() -> Self { Self { origin: Vec3::ZERO, direction: Vec3::Z } }
               }
