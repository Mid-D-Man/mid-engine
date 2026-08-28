//! `GlobalTransform`/`GlobalTransformLWC` — the Archetype Core's hottest,
//! most-iterated component family.
//!
//! Implements the design locked in by `docs/mid-ecs.md`'s "Large World
//! Coordinates: GlobalTransform" section — read that section first if
//! you're wondering *why* this is two types instead of one. Short version:
//! `f32` `GlobalTransform` is the default (cheap, cache-friendly, what
//! most entities use), `f64` `GlobalTransformLWC` is opt-in for entities
//! that actually travel far enough from world origin to need it. Two
//! distinct component types — two distinct Archetype Core families — not
//! one type with a runtime-branching representation, because a tagged
//! union inside one column would break the homogeneous-`Vec<T>`-column
//! assumption every other system reading a column already depends on.
//!
//! Both are thin `#[repr(transparent)]` wrappers over `mid-math`'s
//! existing `Affine3`/`DAffine3` — no affine-transform math gets
//! re-derived here, this module only adds the ECS-facing identity.
//!
//! # A real discrepancy found while implementing this, not silently
//! carried forward
//!
//! `docs/mid-ecs.md` states "`DAffine3` is 96 bytes; `Affine3` is 48" as
//! the justification for `f32` being the cheaper default. Checked against
//! `Affine3`'s own doc comment (`crates/mid-math/src/f32/affine3.rs`):
//! it's actually "64 bytes, 16-byte aligned" (SSE2-backed `__m128` fields
//! on x86/x86_64), not 48. `DAffine3`'s own doc comment confirms 96 bytes
//! is correct. The 48-vs-64 discrepancy doesn't change the actual
//! decision — `GlobalTransformLWC` is still 1.5x `GlobalTransform`'s
//! size, not 2x, but still meaningfully larger for the hottest column in
//! the engine — so the opt-in-`f64` conclusion stands. Flagged here and
//! in `docs/mid-ecs.md` rather than left as a quiet inconsistency between
//! a doc and the code it describes.
//!
//! # What this module does NOT do yet
//!
//! No FFI-span registration (`register_ffi_static_component`) support —
//! that needs either `zerocopy` derives added to `mid-math`'s
//! `Affine3`/`DAffine3` directly, or a separate FFI-safe wrapper, and is
//! a real, separate follow-up, not attempted here. No `LocalTransform`/
//! hierarchy composition, and no `mid-camera` integration — both
//! explicitly called out as "not yet decided"/"not started" in
//! `docs/mid-ecs.md`.

use mid_math::{Affine3, DAffine3, DVec3};

/// The default world transform. `f32`, 64 bytes, backed by `mid-math`'s
/// `Affine3`. Lives in the Archetype Core via `World::insert_static`,
/// same as any other static, every-frame-touched component.
///
/// `#[repr(transparent)]`: identical layout to a bare `Affine3`, so this
/// wrapper costs nothing over storing `Affine3` directly in a column.
#[derive(Clone, Copy, PartialEq, Debug)]
#[repr(transparent)]
pub struct GlobalTransform(pub Affine3);

impl GlobalTransform {
    /// No rotation, no scale, no translation.
    pub const IDENTITY: Self = Self(Affine3::IDENTITY);
}

impl Default for GlobalTransform {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl From<Affine3> for GlobalTransform {
    fn from(affine: Affine3) -> Self {
        Self(affine)
    }
}

impl From<GlobalTransform> for Affine3 {
    fn from(gt: GlobalTransform) -> Self {
        gt.0
    }
}

// Ergonomic access to the underlying Affine3's fields/methods (x_axis,
// y_axis, z_axis, translation, transform_point, ...) without a `.0`
// everywhere it's used — the same newtype-Deref pattern, not a new one
// invented for this type.
impl core::ops::Deref for GlobalTransform {
    type Target = Affine3;
    fn deref(&self) -> &Affine3 {
        &self.0
    }
}

impl core::ops::DerefMut for GlobalTransform {
    fn deref_mut(&mut self) -> &mut Affine3 {
        &mut self.0
    }
}

/// The Large World Coordinates opt-in world transform. `f64`, backed by
/// `mid-math`'s `DAffine3`. For entities that actually travel far enough
/// from world origin to need `f64` precision (open-world terrain, distant
/// structures) — most entities should use [`GlobalTransform`] instead.
///
/// A genuinely distinct Archetype Core family from `GlobalTransform` (see
/// this module's own top doc comment for why) — an entity has one or the
/// other, never both as the "same" transform.
#[derive(Clone, Copy, PartialEq, Debug)]
#[repr(transparent)]
pub struct GlobalTransformLWC(pub DAffine3);

impl GlobalTransformLWC {
    /// No rotation, no scale, no translation.
    pub const IDENTITY: Self = Self(DAffine3::IDENTITY);

    /// The one place `GlobalTransform` and `GlobalTransformLWC` actually
    /// have to know about each other — see `docs/mid-ecs.md`'s "The
    /// pipeline this feeds into" paragraph. Composes the camera-relative
    /// shift and the `f64` → `f32` truncation in one step
    /// (`DAffine3::to_view_relative`, see `docs/mid-math.md`), safe
    /// regardless of how far `self` is from world origin, because only
    /// the already-shifted (small) translation gets truncated, never the
    /// raw world-magnitude one.
    pub fn to_view_relative(self, camera_origin: DVec3) -> GlobalTransform {
        GlobalTransform(self.0.to_view_relative(camera_origin))
    }
}

impl Default for GlobalTransformLWC {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl From<DAffine3> for GlobalTransformLWC {
    fn from(affine: DAffine3) -> Self {
        Self(affine)
    }
}

impl From<GlobalTransformLWC> for DAffine3 {
    fn from(gt: GlobalTransformLWC) -> Self {
        gt.0
    }
}

impl core::ops::Deref for GlobalTransformLWC {
    type Target = DAffine3;
    fn deref(&self) -> &DAffine3 {
        &self.0
    }
}

impl core::ops::DerefMut for GlobalTransformLWC {
    fn deref_mut(&mut self) -> &mut DAffine3 {
        &mut self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::{Entity, World};

    // ── GlobalTransform (f32) ───────────────────────────────────────────

    #[test]
    fn global_transform_identity_matches_affine3_identity() {
        assert!(GlobalTransform::IDENTITY.0 == Affine3::IDENTITY);
    }

    #[test]
    fn global_transform_default_is_identity() {
        assert!(GlobalTransform::default() == GlobalTransform::IDENTITY);
    }

    #[test]
    fn global_transform_from_affine3_round_trips() {
        let a = Affine3::from_translation(mid_math::Vec3::new(1.0, 2.0, 3.0));
        let gt: GlobalTransform = a.into();
        let back: Affine3 = gt.into();
        assert!(back == a);
    }

    #[test]
    fn global_transform_derefs_to_underlying_affine3() {
        let a = Affine3::from_translation(mid_math::Vec3::new(4.0, 5.0, 6.0));
        let gt = GlobalTransform(a);
        // Field access through Deref, not gt.0.translation -- the actual
        // point of implementing Deref at all.
        assert_eq!(gt.translation.x, 4.0);
        assert_eq!(gt.translation.y, 5.0);
        assert_eq!(gt.translation.z, 6.0);
    }

    // ── GlobalTransformLWC (f64) ────────────────────────────────────────

    #[test]
    fn global_transform_lwc_identity_matches_daffine3_identity() {
        assert!(GlobalTransformLWC::IDENTITY.0 == DAffine3::IDENTITY);
    }

    #[test]
    fn global_transform_lwc_default_is_identity() {
        assert!(GlobalTransformLWC::default() == GlobalTransformLWC::IDENTITY);
    }

    #[test]
    fn global_transform_lwc_from_daffine3_round_trips() {
        let a = DAffine3::from_translation(DVec3::new(100_000.0, 0.0, 0.0));
        let gt: GlobalTransformLWC = a.into();
        let back: DAffine3 = gt.into();
        assert!(back == a);
    }

    #[test]
    fn global_transform_lwc_to_view_relative_preserves_the_small_offset() {
        // Mirrors mid-math's own dvec3_to_view_relative_is_the_actual_fix_for_the_jitter_this_exists_for
        // at the component level, not just the primitive level -- proves
        // the wiring through GlobalTransformLWC doesn't lose the fix.
        let far_translation = DVec3::new(100_000.2, 0.0, 0.0);
        let camera = DVec3::new(100_000.0, 0.0, 0.0);
        let world_transform = GlobalTransformLWC(DAffine3::from_translation(far_translation));

        let relative = world_transform.to_view_relative(camera);

        assert!(
            (relative.translation.x - 0.2).abs() < 0.0001,
            "to_view_relative must preserve the small offset from camera, got {}",
            relative.translation.x
        );
    }

    #[test]
    fn global_transform_lwc_to_view_relative_at_origin_matches_a_plain_cast() {
        let small = DVec3::new(1.5, 2.5, 3.5);
        let world_transform = GlobalTransformLWC(DAffine3::from_translation(small));

        let relative = world_transform.to_view_relative(DVec3::ZERO);

        assert_eq!(relative.translation.x, 1.5);
        assert_eq!(relative.translation.y, 2.5);
        assert_eq!(relative.translation.z, 3.5);
    }

    // ── Real World integration: Archetype Core storage ─────────────────

    #[test]
    fn world_can_insert_and_query_static_global_transform() {
        let mut w = World::new();
        let e = w.spawn();
        assert!(w.insert_static(e, GlobalTransform::IDENTITY));

        let found: Vec<Entity> = w
            .query_static::<GlobalTransform>()
            .map(|(e, _)| e)
            .collect();
        assert_eq!(found, vec![e]);
    }

    #[test]
    fn world_can_insert_and_query_static_global_transform_lwc() {
        let mut w = World::new();
        let e = w.spawn();
        assert!(w.insert_static(e, GlobalTransformLWC::IDENTITY));

        let found: Vec<Entity> = w
            .query_static::<GlobalTransformLWC>()
            .map(|(e, _)| e)
            .collect();
        assert_eq!(found, vec![e]);
    }

    #[test]
    fn global_transform_and_global_transform_lwc_are_genuinely_distinct_archetype_families() {
        // The actual point of the two-component-type design: an entity
        // with GlobalTransform must not show up in a GlobalTransformLWC
        // query and vice versa -- proving the split is real, not just
        // documented intent.
        let mut w = World::new();
        let f32_entity = w.spawn();
        let f64_entity = w.spawn();

        assert!(w.insert_static(f32_entity, GlobalTransform::IDENTITY));
        assert!(w.insert_static(f64_entity, GlobalTransformLWC::IDENTITY));

        let f32_found: Vec<Entity> = w
            .query_static::<GlobalTransform>()
            .map(|(e, _)| e)
            .collect();
        let f64_found: Vec<Entity> = w
            .query_static::<GlobalTransformLWC>()
            .map(|(e, _)| e)
            .collect();

        assert_eq!(f32_found, vec![f32_entity]);
        assert_eq!(f64_found, vec![f64_entity]);
    }

    #[test]
    fn global_transform_get_static_after_insert_returns_the_real_value() {
        let mut w = World::new();
        let e = w.spawn();
        let t = GlobalTransform(Affine3::from_translation(mid_math::Vec3::new(
            7.0, 8.0, 9.0,
        )));
        assert!(w.insert_static(e, t));

        let got = w.get_static::<GlobalTransform>(e).expect("just inserted");
        assert!(*got == t);
    }
}
