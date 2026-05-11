// crates/mid-geom/src/ffi.rs
//! C-ABI types and #[no_mangle] exports for geometry primitives.
//!
//! C types mirror the Rust types exactly in memory layout.
//! All exports follow the `mid_<type>_<op>` naming convention.

use mid_math::ffi::{CVec2, CVec3, CQuat, CMat4};
use mid_math::{Vec2, Vec3, Quat, Mat4};

use crate::{
    AABB, Capsule, Circle, Frustum, Plane, Ray2, Ray3, Rect, Sphere, Transform, Transform2D,
};

// ═══════════════════════════════════════════════════════════════════════════
//  C types
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy)] #[repr(C)]
pub struct CRect     { pub min: CVec2, pub max: CVec2 }

#[derive(Debug, Clone, Copy)] #[repr(C)]
pub struct CCircle   { pub center: CVec2, pub radius: f32 }

#[derive(Debug, Clone, Copy)] #[repr(C)]
pub struct CAABB     { pub min: CVec3, pub max: CVec3 }

#[derive(Debug, Clone, Copy)] #[repr(C)]
pub struct CSphere   { pub center: CVec3, pub radius: f32 }

#[derive(Debug, Clone, Copy)] #[repr(C)]
pub struct CCapsule  { pub base: CVec3, pub tip: CVec3, pub radius: f32 }

#[derive(Debug, Clone, Copy)] #[repr(C)]
pub struct CPlane    { pub normal: CVec3, pub d: f32 }

#[derive(Debug, Clone, Copy)] #[repr(C)]
pub struct CRay2     { pub origin: CVec2, pub direction: CVec2 }

#[derive(Debug, Clone, Copy)] #[repr(C)]
pub struct CRay3     { pub origin: CVec3, pub direction: CVec3 }

#[derive(Debug, Clone, Copy)] #[repr(C)]
pub struct CTransform2D { pub position: CVec2, pub rotation: f32, pub scale: CVec2 }

#[derive(Debug, Clone, Copy)] #[repr(C)]
pub struct CTransform {
    pub position: CVec3,
    pub rotation: CQuat,
    pub scale:    CVec3,
}

// ── Conversions ───────────────────────────────────────────────────────────────

impl From<Rect>    for CRect    { fn from(r: Rect)    -> Self { Self { min: r.min.into(), max: r.max.into() } } }
impl From<CRect>   for Rect     { fn from(r: CRect)   -> Self { Self::new(r.min.into(), r.max.into()) } }

impl From<Circle>  for CCircle  { fn from(c: Circle)  -> Self { Self { center: c.center.into(), radius: c.radius } } }
impl From<CCircle> for Circle   { fn from(c: CCircle) -> Self { Self::new(c.center.into(), c.radius) } }

impl From<AABB>    for CAABB    { fn from(a: AABB)    -> Self { Self { min: a.min.into(), max: a.max.into() } } }
impl From<CAABB>   for AABB     { fn from(a: CAABB)   -> Self { Self::new(a.min.into(), a.max.into()) } }

impl From<Sphere>  for CSphere  { fn from(s: Sphere)  -> Self { Self { center: s.center.into(), radius: s.radius } } }
impl From<CSphere> for Sphere   { fn from(s: CSphere) -> Self { Self::new(s.center.into(), s.radius) } }

impl From<Capsule>  for CCapsule { fn from(c: Capsule)  -> Self { Self { base: c.base.into(), tip: c.tip.into(), radius: c.radius } } }
impl From<CCapsule> for Capsule  { fn from(c: CCapsule) -> Self { Self::new(c.base.into(), c.tip.into(), c.radius) } }

impl From<Plane>   for CPlane   { fn from(p: Plane)   -> Self { Self { normal: p.normal.into(), d: p.d } } }
impl From<CPlane>  for Plane    { fn from(p: CPlane)  -> Self { Self { normal: p.normal.into(), d: p.d } } }

impl From<Ray3>    for CRay3    { fn from(r: Ray3)    -> Self { Self { origin: r.origin.into(), direction: r.direction.into() } } }
impl From<CRay3>   for Ray3     { fn from(r: CRay3)   -> Self { Self::new_unnormalized(r.origin.into(), r.direction.into()) } }

impl From<Ray2>    for CRay2    { fn from(r: Ray2)    -> Self { Self { origin: r.origin.into(), direction: r.direction.into() } } }
impl From<CRay2>   for Ray2     { fn from(r: CRay2)   -> Self { Ray2::new_unnormalized(r.origin.into(), r.direction.into()) } }

impl From<Transform2D> for CTransform2D {
    fn from(t: Transform2D) -> Self {
        Self { position: t.position.into(), rotation: t.rotation, scale: t.scale.into() }
    }
}
impl From<CTransform2D> for Transform2D {
    fn from(t: CTransform2D) -> Self {
        Self::new(t.position.into(), t.rotation, t.scale.into())
    }
}

impl From<Transform> for CTransform {
    fn from(t: Transform) -> Self {
        Self {
            position: t.position.into(),
            rotation: Quat::new(t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w).into(),
            scale:    t.scale.into(),
        }
    }
}
impl From<CTransform> for Transform {
    fn from(t: CTransform) -> Self {
        Self::from_trs(
            t.position.into(),
            Quat::new(t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w),
            t.scale.into(),
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Exports — Rect
// ═══════════════════════════════════════════════════════════════════════════

#[no_mangle] pub extern "C" fn mid_rect_new(min: CVec2, max: CVec2) -> CRect {
    Rect::new(min.into(), max.into()).into()
}
#[no_mangle] pub extern "C" fn mid_rect_center(r: CRect) -> CVec2 {
    Rect::from(r).center().into()
}
#[no_mangle] pub extern "C" fn mid_rect_size(r: CRect) -> CVec2 {
    Rect::from(r).size().into()
}
#[no_mangle] pub extern "C" fn mid_rect_contains_point(r: CRect, p: CVec2) -> bool {
    Rect::from(r).contains_point(p.into())
}
#[no_mangle] pub extern "C" fn mid_rect_intersects_rect(a: CRect, b: CRect) -> bool {
    Rect::from(a).intersects_rect(Rect::from(b))
}
#[no_mangle] pub extern "C" fn mid_rect_expand_to_include(r: CRect, p: CVec2) -> CRect {
    Rect::from(r).expand_to_include(p.into()).into()
}
#[no_mangle] pub extern "C" fn mid_rect_merge(a: CRect, b: CRect) -> CRect {
    Rect::from(a).merge(Rect::from(b)).into()
}

// ═══════════════════════════════════════════════════════════════════════════
//  Exports — Circle
// ═══════════════════════════════════════════════════════════════════════════

#[no_mangle] pub extern "C" fn mid_circle_new(center: CVec2, radius: f32) -> CCircle {
    Circle::new(center.into(), radius).into()
}
#[no_mangle] pub extern "C" fn mid_circle_contains_point(c: CCircle, p: CVec2) -> bool {
    Circle::from(c).contains_point(p.into())
}
#[no_mangle] pub extern "C" fn mid_circle_intersects_circle(a: CCircle, b: CCircle) -> bool {
    Circle::from(a).intersects_circle(&Circle::from(b))
}
#[no_mangle] pub extern "C" fn mid_circle_intersects_rect(c: CCircle, r: CRect) -> bool {
    Circle::from(c).intersects_rect(&Rect::from(r))
}

// ═══════════════════════════════════════════════════════════════════════════
//  Exports — AABB
// ═══════════════════════════════════════════════════════════════════════════

#[no_mangle] pub extern "C" fn mid_aabb_new(min: CVec3, max: CVec3) -> CAABB {
    AABB::new(min.into(), max.into()).into()
}
#[no_mangle] pub extern "C" fn mid_aabb_center(a: CAABB) -> CVec3 {
    AABB::from(a).center().into()
}
#[no_mangle] pub extern "C" fn mid_aabb_size(a: CAABB) -> CVec3 {
    AABB::from(a).size().into()
}
#[no_mangle] pub extern "C" fn mid_aabb_surface_area(a: CAABB) -> f32 {
    AABB::from(a).surface_area()
}
#[no_mangle] pub extern "C" fn mid_aabb_contains_point(a: CAABB, p: CVec3) -> bool {
    AABB::from(a).contains_point(p.into())
}
#[no_mangle] pub extern "C" fn mid_aabb_intersects_aabb(a: CAABB, b: CAABB) -> bool {
    AABB::from(a).intersects_aabb(&AABB::from(b))
}
#[no_mangle] pub extern "C" fn mid_aabb_intersects_sphere(a: CAABB, s: CSphere) -> bool {
    AABB::from(a).intersects_sphere(&Sphere::from(s))
}
#[no_mangle] pub extern "C" fn mid_aabb_merge(a: CAABB, b: CAABB) -> CAABB {
    AABB::from(a).merge(AABB::from(b)).into()
}
#[no_mangle] pub extern "C" fn mid_aabb_expand(a: CAABB, amount: f32) -> CAABB {
    AABB::from(a).expand(amount).into()
}
#[no_mangle] pub extern "C" fn mid_aabb_closest_point(a: CAABB, p: CVec3) -> CVec3 {
    AABB::from(a).closest_point(p.into()).into()
}

// ═══════════════════════════════════════════════════════════════════════════
//  Exports — Sphere
// ═══════════════════════════════════════════════════════════════════════════

#[no_mangle] pub extern "C" fn mid_sphere_new(center: CVec3, radius: f32) -> CSphere {
    Sphere::new(center.into(), radius).into()
}
#[no_mangle] pub extern "C" fn mid_sphere_contains_point(s: CSphere, p: CVec3) -> bool {
    Sphere::from(s).contains_point(p.into())
}
#[no_mangle] pub extern "C" fn mid_sphere_intersects_sphere(a: CSphere, b: CSphere) -> bool {
    Sphere::from(a).intersects_sphere(&Sphere::from(b))
}
#[no_mangle] pub extern "C" fn mid_sphere_intersects_aabb(s: CSphere, a: CAABB) -> bool {
    Sphere::from(s).intersects_aabb(&AABB::from(a))
}
#[no_mangle] pub extern "C" fn mid_sphere_signed_distance(s: CSphere, p: CVec3) -> f32 {
    Sphere::from(s).signed_distance(p.into())
}
#[no_mangle] pub extern "C" fn mid_sphere_bounding_aabb(s: CSphere) -> CAABB {
    Sphere::from(s).bounding_aabb().into()
}

// ═══════════════════════════════════════════════════════════════════════════
//  Exports — Capsule
// ═══════════════════════════════════════════════════════════════════════════

#[no_mangle] pub extern "C" fn mid_capsule_new(base: CVec3, tip: CVec3, radius: f32) -> CCapsule {
    Capsule::new(base.into(), tip.into(), radius).into()
}
#[no_mangle] pub extern "C" fn mid_capsule_contains_point(c: CCapsule, p: CVec3) -> bool {
    Capsule::from(c).contains_point(p.into())
}
#[no_mangle] pub extern "C" fn mid_capsule_intersects_sphere(c: CCapsule, s: CSphere) -> bool {
    Capsule::from(c).intersects_sphere(&Sphere::from(s))
}
#[no_mangle] pub extern "C" fn mid_capsule_intersects_capsule(a: CCapsule, b: CCapsule) -> bool {
    Capsule::from(a).intersects_capsule(&Capsule::from(b))
}

// ═══════════════════════════════════════════════════════════════════════════
//  Exports — Plane
// ═══════════════════════════════════════════════════════════════════════════

#[no_mangle] pub extern "C" fn mid_plane_from_normal_point(n: CVec3, p: CVec3) -> CPlane {
    Plane::from_normal_point(n.into(), p.into()).into()
}
#[no_mangle] pub extern "C" fn mid_plane_signed_distance(pl: CPlane, p: CVec3) -> f32 {
    Plane::from(pl).signed_distance(p.into())
}
#[no_mangle] pub extern "C" fn mid_plane_project_point(pl: CPlane, p: CVec3) -> CVec3 {
    Plane::from(pl).project_point(p.into()).into()
}

// ═══════════════════════════════════════════════════════════════════════════
//  Exports — Ray3
// ═══════════════════════════════════════════════════════════════════════════

#[no_mangle] pub extern "C" fn mid_ray3_at(r: CRay3, t: f32) -> CVec3 {
    Ray3::from(r).at(t).into()
}
#[no_mangle] pub extern "C" fn mid_ray3_intersects_sphere(r: CRay3, s: CSphere) -> f32 {
    Ray3::from(r).intersect_sphere(&Sphere::from(s)).map_or(-1.0, |h| h.t)
}
#[no_mangle] pub extern "C" fn mid_ray3_intersects_aabb(r: CRay3, a: CAABB) -> f32 {
    Ray3::from(r).intersect_aabb(&AABB::from(a)).map_or(-1.0, |h| h.t)
}
#[no_mangle] pub extern "C" fn mid_ray3_intersects_plane(r: CRay3, pl: CPlane) -> f32 {
    Ray3::from(r).intersect_plane(&Plane::from(pl)).map_or(-1.0, |h| h.t)
}

// ═══════════════════════════════════════════════════════════════════════════
//  Exports — Transform
// ═══════════════════════════════════════════════════════════════════════════

#[no_mangle] pub extern "C" fn mid_transform_identity() -> CTransform {
    Transform::IDENTITY.into()
}
#[no_mangle] pub extern "C" fn mid_transform_transform_point(t: CTransform, p: CVec3) -> CVec3 {
    Transform::from(t).transform_point(p.into()).into()
}
#[no_mangle] pub extern "C" fn mid_transform_transform_vector(t: CTransform, v: CVec3) -> CVec3 {
    Transform::from(t).transform_vector(v.into()).into()
}
#[no_mangle] pub extern "C" fn mid_transform_inverse_transform_point(t: CTransform, p: CVec3) -> CVec3 {
    Transform::from(t).inverse_transform_point(p.into()).into()
}
#[no_mangle] pub extern "C" fn mid_transform_to_mat4(t: CTransform) -> CMat4 {
    Transform::from(t).to_mat4().into()
}
#[no_mangle] pub extern "C" fn mid_transform_compose(parent: CTransform, child: CTransform) -> CTransform {
    Transform::from(parent).compose(Transform::from(child)).into()
  }
