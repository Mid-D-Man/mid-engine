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
//! # `LocalTransform` and composition
//!
//! [`LocalTransform`] is the parent-relative pose (translation, rotation,
//! scale, always `f32`), and [`World::propagate_transforms`] composes it down
//! the hierarchy into whichever of [`GlobalTransform`] and
//! [`GlobalTransformLWC`] an entity has. The decisions (what a node with no
//! `LocalTransform` does, how the two precisions mix, why a global is only
//! written when it changed) are in `docs/mid-ecs-transform.md`.
//!
//! # What this module does NOT do yet
//!
//! No FFI-span registration (`register_ffi_static_component`) support --
//! that needs either `zerocopy` derives added to `mid-math`'s
//! `Affine3`/`DAffine3` directly, or a separate FFI-safe wrapper, and is
//! a real, separate follow-up, not attempted here -- so no C counterpart of
//! `propagate_transforms` either. No `mid-camera` integration.
//!
//! NOTICE: design, decisions and verification for the composition are in
//! `docs/mid-ecs-transform.md`; the two-types decision is in
//! `docs/mid-ecs.md`, "Large World Coordinates: GlobalTransform".

use crate::world::{Entity, World};
use mid_math::{Affine3, DAffine3, DVec3, Quat, Vec3};

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

/// A pose relative to the parent: translation, rotation, scale. Always `f32`,
/// whatever precision the entity's global transform has, because a pose
/// relative to a parent stays small.
///
/// [`World::propagate_transforms`] composes it with the parent's global
/// transform into the entity's [`GlobalTransform`] and/or
/// [`GlobalTransformLWC`]. Lives in the Archetype Core via
/// `World::insert_static`.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct LocalTransform {
    /// Offset from the parent, in the parent's space.
    pub translation: Vec3,
    /// Orientation relative to the parent.
    pub rotation: Quat,
    /// Per-axis scale, applied before the rotation.
    pub scale: Vec3,
}

impl LocalTransform {
    /// No translation, no rotation, scale one.
    pub const IDENTITY: Self = Self {
        translation: Vec3::ZERO,
        rotation: Quat::IDENTITY,
        scale: Vec3::ONE,
    };

    /// A pose from its three parts.
    pub fn new(translation: Vec3, rotation: Quat, scale: Vec3) -> Self {
        Self {
            translation,
            rotation,
            scale,
        }
    }

    /// Only a translation.
    pub fn from_translation(translation: Vec3) -> Self {
        Self {
            translation,
            ..Self::IDENTITY
        }
    }

    /// Only a rotation.
    pub fn from_rotation(rotation: Quat) -> Self {
        Self {
            rotation,
            ..Self::IDENTITY
        }
    }

    /// Only a scale.
    pub fn from_scale(scale: Vec3) -> Self {
        Self {
            scale,
            ..Self::IDENTITY
        }
    }

    /// The same pose as an affine transform: scale, then rotation, then
    /// translation.
    pub fn to_affine(self) -> Affine3 {
        Affine3::from_scale_rotation_translation(self.scale, self.rotation, self.translation)
    }
}

impl Default for LocalTransform {
    fn default() -> Self {
        Self::IDENTITY
    }
}

/// What the transforms above an entity add up to, in the precision they are
/// being added in.
#[derive(Clone, Copy)]
enum Frame {
    /// Nothing above has a transform.
    Identity,
    F32(Affine3),
    F64(DAffine3),
}

impl Frame {
    fn to_f32(self) -> Affine3 {
        match self {
            Frame::Identity => Affine3::IDENTITY,
            Frame::F32(a) => a,
            Frame::F64(d) => d.as_affine3(),
        }
    }

    fn to_f64(self) -> DAffine3 {
        match self {
            Frame::Identity => DAffine3::IDENTITY,
            Frame::F32(a) => promote(a),
            Frame::F64(d) => d,
        }
    }
}

/// `f32` to `f64`, axis by axis. Exact.
fn promote(a: Affine3) -> DAffine3 {
    let wide = |v: Vec3| DVec3::new(v.x as f64, v.y as f64, v.z as f64);
    DAffine3 {
        x_axis: wide(a.x_axis),
        y_axis: wide(a.y_axis),
        z_axis: wide(a.z_axis),
        translation: wide(a.translation),
    }
}

impl World {
    /// Composes every [`LocalTransform`] with the transforms above it and
    /// writes the result into the entity's [`GlobalTransform`] and/or
    /// [`GlobalTransformLWC`], top down from the roots of the hierarchy.
    ///
    /// - An entity with a `LocalTransform` gets `parent * local` written to
    ///   each global it has (neither is inserted for it).
    /// - An entity with no `LocalTransform` and a global of its own (set by
    ///   hand, by physics) is left alone, and its children compose onto that
    ///   value. With both globals, the `GlobalTransformLWC` is the one used.
    /// - An entity with no transform at all passes its parent's transform
    ///   on, as if it were the identity.
    /// - Composition is in `f64` below any entity with a
    ///   `GlobalTransformLWC` and in `f32` otherwise; an `f32` global below
    ///   an `f64` one gets the truncated result, and what is below it still
    ///   composes in `f64`.
    /// - A global is written only when its value changed, so
    ///   `Changed<GlobalTransform>` finds the entities that moved.
    ///
    /// Enables the hierarchy (see [`World::enable_hierarchy`]) if it is not
    /// yet, because it walks `Children`. See `docs/mid-ecs-transform.md`.
    ///
    /// ```
    /// use mid_ecs::{GlobalTransform, LocalTransform, World};
    /// use mid_math::Vec3;
    ///
    /// let mut world = World::new();
    /// let ship = world.spawn();
    /// world.insert_static(ship, LocalTransform::from_translation(Vec3::new(10.0, 0.0, 0.0)));
    /// world.insert_static(ship, GlobalTransform::IDENTITY);
    /// let gun = world.spawn_child(ship).unwrap();
    /// world.insert_static(gun, LocalTransform::from_translation(Vec3::new(0.0, 1.0, 0.0)));
    /// world.insert_static(gun, GlobalTransform::IDENTITY);
    ///
    /// world.propagate_transforms();
    /// let at = world.get_static::<GlobalTransform>(gun).unwrap().translation;
    /// assert_eq!((at.x, at.y), (10.0, 1.0));
    /// ```
    pub fn propagate_transforms(&mut self) {
        self.enable_hierarchy();
        let mut roots: Vec<Entity> = Vec::new();
        roots.extend(
            self.query_static::<LocalTransform>()
                .map(|(e, _)| e)
                .filter(|e| self.parent(*e).is_none()),
        );
        roots.extend(
            self.query_static::<crate::hierarchy::Children>()
                .map(|(e, _)| e)
                .filter(|e| self.parent(*e).is_none()),
        );
        roots.sort_unstable_by_key(|e| e.index());
        roots.dedup();

        // An explicit stack: a chain can be far deeper than the call stack.
        let mut stack: Vec<(Entity, Frame)> =
            roots.into_iter().map(|e| (e, Frame::Identity)).collect();
        while let Some((entity, above)) = stack.pop() {
            let here = self.compose_one(entity, above);
            stack.extend(self.children(entity).iter().map(|c| (*c, here)));
        }
    }

    /// The frame `entity` hands to its children, writing its globals.
    fn compose_one(&mut self, entity: Entity, above: Frame) -> Frame {
        let has_f32 = self.has_static::<GlobalTransform>(entity);
        let has_lwc = self.has_static::<GlobalTransformLWC>(entity);
        let Some(local) = self
            .get_static::<LocalTransform>(entity)
            .map(|l| l.to_affine())
        else {
            return if has_lwc {
                self.get_static::<GlobalTransformLWC>(entity)
                    .map_or(above, |g| Frame::F64(g.0))
            } else if has_f32 {
                self.get_static::<GlobalTransform>(entity)
                    .map_or(above, |g| Frame::F32(g.0))
            } else {
                above
            };
        };
        let here = if has_lwc || matches!(above, Frame::F64(_)) {
            Frame::F64(above.to_f64() * promote(local))
        } else {
            Frame::F32(above.to_f32() * local)
        };
        if has_f32 {
            let value = here.to_f32();
            if self.get_static::<GlobalTransform>(entity).map(|g| g.0) != Some(value) {
                if let Some(g) = self.get_static_mut::<GlobalTransform>(entity) {
                    g.0 = value;
                }
            }
        }
        if has_lwc {
            let value = here.to_f64();
            if self.get_static::<GlobalTransformLWC>(entity).map(|g| g.0) != Some(value) {
                if let Some(g) = self.get_static_mut::<GlobalTransformLWC>(entity) {
                    g.0 = value;
                }
            }
        }
        here
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

    // ── LocalTransform and propagate_transforms ──────────────────────────

    use crate::{ChangeTracker, Changed};
    use std::f32::consts::FRAC_PI_2;

    fn at(x: f32, y: f32, z: f32) -> LocalTransform {
        LocalTransform::from_translation(Vec3::new(x, y, z))
    }

    /// An entity (a child of `parent` if given) with the parts asked for.
    fn node(
        w: &mut World,
        parent: Option<Entity>,
        local: Option<LocalTransform>,
        f32_global: bool,
        lwc_global: bool,
    ) -> Entity {
        let e = match parent {
            Some(p) => w.spawn_child(p).unwrap(),
            None => w.spawn(),
        };
        if let Some(l) = local {
            w.insert_static(e, l);
        }
        if f32_global {
            w.insert_static(e, GlobalTransform::IDENTITY);
        }
        if lwc_global {
            w.insert_static(e, GlobalTransformLWC::IDENTITY);
        }
        e
    }

    /// A transformed entity with an `f32` global.
    fn tn(w: &mut World, parent: Option<Entity>, local: LocalTransform) -> Entity {
        node(w, parent, Some(local), true, false)
    }

    fn g(w: &World, e: Entity) -> Affine3 {
        w.get_static::<GlobalTransform>(e).unwrap().0
    }

    fn gl(w: &World, e: Entity) -> DAffine3 {
        w.get_static::<GlobalTransformLWC>(e).unwrap().0
    }

    fn close(a: f32, b: f32) {
        assert!((a - b).abs() < 1e-5, "{a} is not {b}");
    }

    fn changed_globals(w: &World, tracker: &ChangeTracker) -> Vec<Entity> {
        let mut v: Vec<Entity> = w
            .query_data_since::<(Entity, Changed<GlobalTransform>)>(tracker)
            .filter(|(_, changed)| *changed)
            .map(|(e, _)| e)
            .collect();
        v.sort_by_key(|e| e.index());
        v
    }

    #[test]
    fn local_transform_identity_is_the_identity_affine() {
        assert!(LocalTransform::IDENTITY.to_affine() == Affine3::IDENTITY);
        assert!(LocalTransform::default() == LocalTransform::IDENTITY);
    }

    #[test]
    fn local_transform_constructors_set_only_their_part() {
        let t = LocalTransform::from_translation(Vec3::new(1.0, 2.0, 3.0));
        assert!(t.rotation == Quat::IDENTITY && t.scale == Vec3::ONE);
        assert_eq!(t.translation.y, 2.0);
        let r = LocalTransform::from_rotation(Quat::from_axis_angle(Vec3::Z, 1.0));
        assert!(r.translation == Vec3::ZERO && r.scale == Vec3::ONE);
        let s = LocalTransform::from_scale(Vec3::new(2.0, 3.0, 4.0));
        assert!(s.translation == Vec3::ZERO && s.rotation == Quat::IDENTITY);
        assert_eq!(s.scale.z, 4.0);
        let all = LocalTransform::new(t.translation, r.rotation, s.scale);
        assert!(
            all.translation == t.translation && all.rotation == r.rotation && all.scale == s.scale
        );
    }

    #[test]
    fn local_transform_to_affine_scales_then_rotates_then_translates() {
        let t = LocalTransform::new(
            Vec3::new(10.0, 0.0, 0.0),
            Quat::from_axis_angle(Vec3::Z, FRAC_PI_2),
            Vec3::new(2.0, 2.0, 2.0),
        );
        let p = t.to_affine().transform_point(Vec3::new(1.0, 0.0, 0.0));
        close(p.x, 10.0);
        close(p.y, 2.0);
        close(p.z, 0.0);
    }

    #[test]
    fn a_root_gets_its_local_as_its_global() {
        let mut w = World::new();
        let e = tn(&mut w, None, at(1.0, 2.0, 3.0));
        w.propagate_transforms();
        assert!(g(&w, e) == at(1.0, 2.0, 3.0).to_affine());
    }

    #[test]
    fn a_child_adds_its_translation_to_its_parents() {
        let mut w = World::new();
        let root = tn(&mut w, None, at(1.0, 0.0, 0.0));
        let kid = tn(&mut w, Some(root), at(0.0, 2.0, 0.0));
        w.propagate_transforms();
        assert_eq!(g(&w, kid).translation.x, 1.0);
        assert_eq!(g(&w, kid).translation.y, 2.0);
    }

    #[test]
    fn a_parents_rotation_turns_its_childs_offset() {
        let mut w = World::new();
        let root = tn(
            &mut w,
            None,
            LocalTransform::from_rotation(Quat::from_axis_angle(Vec3::Z, FRAC_PI_2)),
        );
        let kid = tn(&mut w, Some(root), at(1.0, 0.0, 0.0));
        w.propagate_transforms();
        close(g(&w, kid).translation.x, 0.0);
        close(g(&w, kid).translation.y, 1.0);
    }

    #[test]
    fn a_parents_scale_scales_its_childs_offset_and_multiplies_with_its_scale() {
        let mut w = World::new();
        let root = tn(
            &mut w,
            None,
            LocalTransform::from_scale(Vec3::new(2.0, 2.0, 2.0)),
        );
        let mut child_local = at(1.0, 0.0, 0.0);
        child_local.scale = Vec3::new(3.0, 3.0, 3.0);
        let kid = tn(&mut w, Some(root), child_local);
        w.propagate_transforms();
        assert_eq!(g(&w, kid).translation.x, 2.0);
        assert_eq!(g(&w, kid).x_axis.x, 6.0);
    }

    #[test]
    fn a_chain_is_the_product_of_its_locals_root_first() {
        let a = LocalTransform::new(
            Vec3::new(1.0, -2.0, 0.5),
            Quat::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), 0.7),
            Vec3::new(1.5, 1.0, 0.5),
        );
        let b = LocalTransform::new(
            Vec3::new(-3.0, 0.25, 2.0),
            Quat::from_axis_angle(Vec3::new(1.0, 0.0, 0.0), -0.4),
            Vec3::new(0.5, 2.0, 1.0),
        );
        let c = LocalTransform::new(
            Vec3::new(0.1, 0.2, 0.3),
            Quat::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), 1.9),
            Vec3::new(2.0, 2.0, 2.0),
        );
        let mut w = World::new();
        let ea = tn(&mut w, None, a);
        let eb = tn(&mut w, Some(ea), b);
        let ec = tn(&mut w, Some(eb), c);
        w.propagate_transforms();
        assert!(g(&w, ea) == Affine3::IDENTITY * a.to_affine());
        assert!(g(&w, eb) == Affine3::IDENTITY * a.to_affine() * b.to_affine());
        assert!(g(&w, ec) == Affine3::IDENTITY * a.to_affine() * b.to_affine() * c.to_affine());
    }

    #[test]
    fn running_it_again_with_nothing_edited_changes_nothing() {
        let mut w = World::new();
        let root = tn(&mut w, None, at(1.0, 0.0, 0.0));
        tn(&mut w, Some(root), at(0.0, 1.0, 0.0));
        w.propagate_transforms();
        let mut tracker = ChangeTracker::new();
        tracker.update(&w);
        w.increment_change_tick();
        w.propagate_transforms();
        assert!(changed_globals(&w, &tracker).is_empty());
    }

    #[test]
    fn only_the_entities_whose_global_moved_are_marked_changed() {
        let mut w = World::new();
        let a = tn(&mut w, None, at(1.0, 0.0, 0.0));
        let b = tn(&mut w, Some(a), at(0.0, 1.0, 0.0));
        let c = tn(&mut w, Some(a), at(0.0, 0.0, 1.0));
        let d = tn(&mut w, Some(b), at(5.0, 0.0, 0.0));
        w.propagate_transforms();
        let mut tracker = ChangeTracker::new();
        tracker.update(&w);
        w.increment_change_tick();
        w.get_static_mut::<LocalTransform>(b).unwrap().translation = Vec3::new(0.0, 9.0, 0.0);
        w.propagate_transforms();
        let mut want = vec![b, d];
        want.sort_by_key(|e| e.index());
        assert_eq!(changed_globals(&w, &tracker), want);
        assert_eq!(g(&w, d).translation.y, 9.0);
        assert_eq!(g(&w, c).translation.y, 0.0);
    }

    #[test]
    fn only_the_lwc_globals_whose_value_moved_are_marked_changed() {
        let mut w = World::new();
        let a = node(&mut w, None, Some(at(1.0, 0.0, 0.0)), false, true);
        let b = node(&mut w, Some(a), Some(at(0.0, 1.0, 0.0)), false, true);
        let c = node(&mut w, Some(a), Some(at(0.0, 0.0, 1.0)), false, true);
        w.propagate_transforms();
        let changed = |w: &World, tracker: &ChangeTracker| {
            let mut v: Vec<Entity> = w
                .query_data_since::<(Entity, Changed<GlobalTransformLWC>)>(tracker)
                .filter(|(_, changed)| *changed)
                .map(|(e, _)| e)
                .collect();
            v.sort_by_key(|e| e.index());
            v
        };
        let mut tracker = ChangeTracker::new();
        tracker.update(&w);
        w.increment_change_tick();
        w.propagate_transforms();
        assert!(changed(&w, &tracker).is_empty(), "nothing edited");
        w.get_static_mut::<LocalTransform>(b).unwrap().translation = Vec3::new(0.0, 9.0, 0.0);
        w.propagate_transforms();
        assert_eq!(changed(&w, &tracker), vec![b]);
        assert_eq!(gl(&w, c).translation.z, 1.0);
    }

    #[test]
    fn an_entity_with_no_transform_passes_its_parents_on() {
        let mut w = World::new();
        let root = tn(&mut w, None, at(1.0, 0.0, 0.0));
        let group = node(&mut w, Some(root), None, false, false);
        let leaf = tn(&mut w, Some(group), at(0.0, 1.0, 0.0));
        w.propagate_transforms();
        assert_eq!(g(&w, leaf).translation.x, 1.0);
        assert_eq!(g(&w, leaf).translation.y, 1.0);
        assert!(!w.has_static::<GlobalTransform>(group));
    }

    #[test]
    fn a_global_set_by_hand_with_no_local_is_kept_and_its_children_follow_it() {
        let mut w = World::new();
        let root = tn(&mut w, None, at(100.0, 0.0, 0.0));
        let driven = node(&mut w, Some(root), None, true, false);
        *w.get_static_mut::<GlobalTransform>(driven).unwrap() =
            GlobalTransform(Affine3::from_translation(Vec3::new(5.0, 0.0, 0.0)));
        let kid = tn(&mut w, Some(driven), at(1.0, 0.0, 0.0));
        w.propagate_transforms();
        assert_eq!(
            g(&w, driven).translation.x,
            5.0,
            "not overwritten by its parent"
        );
        assert_eq!(g(&w, kid).translation.x, 6.0);
    }

    #[test]
    fn with_both_globals_and_no_local_the_lwc_one_is_the_source() {
        let mut w = World::new();
        let driven = node(&mut w, None, None, true, true);
        *w.get_static_mut::<GlobalTransform>(driven).unwrap() =
            GlobalTransform(Affine3::from_translation(Vec3::new(1.0, 0.0, 0.0)));
        *w.get_static_mut::<GlobalTransformLWC>(driven).unwrap() =
            GlobalTransformLWC(DAffine3::from_translation(DVec3::new(2.0, 0.0, 0.0)));
        let kid = tn(&mut w, Some(driven), at(0.5, 0.0, 0.0));
        w.propagate_transforms();
        assert_eq!(g(&w, kid).translation.x, 2.5);
    }

    #[test]
    fn a_local_with_no_global_writes_nothing_and_still_carries_its_children() {
        let mut w = World::new();
        let bare = node(&mut w, None, Some(at(5.0, 0.0, 0.0)), false, false);
        let kid = tn(&mut w, Some(bare), at(1.0, 0.0, 0.0));
        w.propagate_transforms();
        assert!(!w.has_static::<GlobalTransform>(bare));
        assert!(!w.has_static::<GlobalTransformLWC>(bare));
        assert_eq!(g(&w, kid).translation.x, 6.0);
    }

    #[test]
    fn an_lwc_entity_below_a_far_parent_composes_in_f64() {
        let mut w = World::new();
        let far = node(&mut w, None, None, false, true);
        *w.get_static_mut::<GlobalTransformLWC>(far).unwrap() = GlobalTransformLWC(
            DAffine3::from_translation(DVec3::new(100_000_000.5, 0.0, 0.0)),
        );
        let kid = node(&mut w, Some(far), Some(at(0.25, 0.0, 0.0)), false, true);
        w.propagate_transforms();
        assert_eq!(gl(&w, kid).translation.x, 100_000_000.75);
    }

    #[test]
    fn an_f32_global_below_an_lwc_one_is_the_truncated_result() {
        let mut w = World::new();
        let far = node(&mut w, None, None, false, true);
        *w.get_static_mut::<GlobalTransformLWC>(far).unwrap() = GlobalTransformLWC(
            DAffine3::from_translation(DVec3::new(100_000_000.5, 0.0, 0.0)),
        );
        let kid = tn(&mut w, Some(far), at(0.25, 0.0, 0.0));
        w.propagate_transforms();
        assert_eq!(
            g(&w, kid).translation.x,
            100_000_000.0,
            "f32 spacing there is 8"
        );
    }

    #[test]
    fn what_is_below_an_lwc_ancestor_keeps_composing_in_f64_through_an_f32_node() {
        let mut w = World::new();
        let far = node(&mut w, None, None, false, true);
        *w.get_static_mut::<GlobalTransformLWC>(far).unwrap() = GlobalTransformLWC(
            DAffine3::from_translation(DVec3::new(100_000_000.5, 0.0, 0.0)),
        );
        let mid = tn(&mut w, Some(far), at(0.25, 0.0, 0.0));
        let grand = node(&mut w, Some(mid), Some(at(0.125, 0.0, 0.0)), false, true);
        w.propagate_transforms();
        assert_eq!(
            gl(&w, grand).translation.x,
            100_000_000.875,
            "not built on the truncated middle"
        );
    }

    #[test]
    fn an_lwc_entity_below_an_f32_parent_promotes_it() {
        let mut w = World::new();
        let root = tn(&mut w, None, at(1.0, 2.0, 3.0));
        let kid = node(&mut w, Some(root), Some(at(0.5, 0.0, 0.0)), false, true);
        w.propagate_transforms();
        let t = gl(&w, kid).translation;
        assert_eq!((t.x, t.y, t.z), (1.5, 2.0, 3.0));
    }

    #[test]
    fn an_lwc_global_has_the_axes_of_the_f32_pose_widened() {
        // Checked axis by axis against the f32 pose, not through `promote`,
        // so a mistake in the widening cannot cancel out.
        let local = LocalTransform::new(
            Vec3::new(1.5, -2.5, 3.5),
            Quat::from_axis_angle(Vec3::new(1.0, 2.0, 3.0), 0.9),
            Vec3::new(1.5, 2.0, 0.5),
        );
        let mut w = World::new();
        let e = node(&mut w, None, Some(local), false, true);
        w.propagate_transforms();
        let want = Affine3::IDENTITY * local.to_affine();
        let got = gl(&w, e);
        for (g, f) in [
            (got.x_axis, want.x_axis),
            (got.y_axis, want.y_axis),
            (got.z_axis, want.z_axis),
            (got.translation, want.translation),
        ] {
            assert_eq!((g.x, g.y, g.z), (f.x as f64, f.y as f64, f.z as f64));
        }
    }

    #[test]
    fn an_entity_with_both_globals_gets_both_written() {
        let mut w = World::new();
        let e = node(&mut w, None, Some(at(1.0, 2.0, 3.0)), true, true);
        w.propagate_transforms();
        assert_eq!(g(&w, e).translation.y, 2.0);
        assert_eq!(gl(&w, e).translation.y, 2.0);
    }

    #[test]
    fn a_deep_chain_is_walked_by_a_loop_not_by_recursion() {
        let mut w = World::new();
        let mut at_end = tn(&mut w, None, at(1.0, 0.0, 0.0));
        for _ in 0..5_000 {
            at_end = tn(&mut w, Some(at_end), at(1.0, 0.0, 0.0));
        }
        w.propagate_transforms();
        assert_eq!(g(&w, at_end).translation.x, 5_001.0);
    }

    #[test]
    fn a_wide_fan_out_is_all_composed() {
        let mut w = World::new();
        let root = tn(&mut w, None, at(10.0, 0.0, 0.0));
        let kids: Vec<Entity> = (0..1_000)
            .map(|i| tn(&mut w, Some(root), at(i as f32, 0.0, 0.0)))
            .collect();
        w.propagate_transforms();
        for (i, k) in kids.iter().enumerate() {
            assert_eq!(g(&w, *k).translation.x, 10.0 + i as f32);
        }
    }

    #[test]
    fn moving_an_entity_to_another_parent_moves_its_global() {
        let mut w = World::new();
        let a = tn(&mut w, None, at(10.0, 0.0, 0.0));
        let b = tn(&mut w, None, at(20.0, 0.0, 0.0));
        let c = tn(&mut w, Some(a), at(1.0, 0.0, 0.0));
        w.propagate_transforms();
        assert_eq!(g(&w, c).translation.x, 11.0);
        w.set_parent(c, b).unwrap();
        w.propagate_transforms();
        assert_eq!(g(&w, c).translation.x, 21.0);
        w.remove_parent(c);
        w.propagate_transforms();
        assert_eq!(g(&w, c).translation.x, 1.0, "a root again");
    }

    #[test]
    fn it_enables_the_hierarchy_and_an_empty_world_is_fine() {
        let mut w = World::new();
        assert!(!w.hierarchy_enabled);
        w.propagate_transforms();
        assert!(w.hierarchy_enabled);
    }

    #[test]
    fn separate_trees_do_not_see_each_other() {
        let mut w = World::new();
        let a = tn(&mut w, None, at(1.0, 0.0, 0.0));
        let b = tn(&mut w, None, at(0.0, 7.0, 0.0));
        let ak = tn(&mut w, Some(a), at(1.0, 0.0, 0.0));
        let bk = tn(&mut w, Some(b), at(0.0, 1.0, 0.0));
        w.propagate_transforms();
        assert_eq!(
            (g(&w, ak).translation.x, g(&w, ak).translation.y),
            (2.0, 0.0)
        );
        assert_eq!(
            (g(&w, bk).translation.x, g(&w, bk).translation.y),
            (0.0, 8.0)
        );
    }

    #[test]
    fn an_entity_made_before_its_parent_is_still_composed_onto_it() {
        // The child has the lower index and has children of its own, so
        // a walk that treated it as a root would reach it last.
        let mut w = World::new();
        let kid = tn(&mut w, None, at(1.0, 0.0, 0.0));
        let grand = tn(&mut w, Some(kid), at(0.0, 1.0, 0.0));
        let parent = tn(&mut w, None, at(10.0, 0.0, 0.0));
        assert!(kid.index() < parent.index());
        w.set_parent(kid, parent).unwrap();
        w.propagate_transforms();
        assert_eq!(g(&w, kid).translation.x, 11.0);
        assert_eq!(g(&w, grand).translation.x, 11.0);
        assert_eq!(g(&w, grand).translation.y, 1.0);
    }

    #[test]
    fn propagation_matches_a_model_on_random_forests_of_every_kind_of_node() {
        struct Lcg(u64);
        impl Lcg {
            fn below(&mut self, n: usize) -> usize {
                self.0 = self
                    .0
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((self.0 >> 33) as usize) % n
            }
            fn unit(&mut self) -> f32 {
                self.below(2001) as f32 / 1000.0 - 1.0
            }
        }
        #[derive(Clone, Copy)]
        enum M {
            Id,
            F32(Affine3),
            F64(DAffine3),
        }
        fn m32(m: M) -> Affine3 {
            match m {
                M::Id => Affine3::IDENTITY,
                M::F32(a) => a,
                M::F64(d) => d.as_affine3(),
            }
        }
        fn m64(m: M) -> DAffine3 {
            match m {
                M::Id => DAffine3::IDENTITY,
                M::F32(a) => promote(a),
                M::F64(d) => d,
            }
        }

        let (mut roots, mut deep, mut lwc_under_lwc, mut f32_under_lwc) = (0, 0, 0, 0);
        for seed in 0..10u64 {
            let mut rng = Lcg(seed * 104729 + 7);
            let mut w = World::new();
            let n = 80;
            // kinds: 0 f32 transform, 1 lwc transform, 2 both, 3 grouping,
            // 4 local only, 5 driven f32, 6 driven lwc
            let mut kind = vec![0usize; n];
            let mut parent: Vec<Option<usize>> = vec![None; n];
            let mut local = vec![LocalTransform::IDENTITY; n];
            let mut ents: Vec<Entity> = Vec::new();
            let mut depth = vec![0usize; n];
            let mut driven32 = vec![Affine3::IDENTITY; n];
            let mut driven64 = vec![DAffine3::IDENTITY; n];
            for i in 0..n {
                kind[i] = rng.below(7);
                if i > 0 && rng.below(6) != 0 {
                    let p = rng.below(i);
                    parent[i] = Some(p);
                    depth[i] = depth[p] + 1;
                }
                local[i] = LocalTransform::new(
                    Vec3::new(rng.unit() * 4.0, rng.unit() * 4.0, rng.unit() * 4.0),
                    Quat::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), rng.unit() * 3.0),
                    Vec3::new(
                        1.0 + rng.unit() * 0.5,
                        1.0 + rng.unit() * 0.5,
                        1.0 + rng.unit() * 0.5,
                    ),
                );
                let (has_local, f32g, lwcg) = match kind[i] {
                    0 => (true, true, false),
                    1 => (true, false, true),
                    2 => (true, true, true),
                    3 => (false, false, false),
                    4 => (true, false, false),
                    5 => (false, true, false),
                    _ => (false, false, true),
                };
                let e = node(
                    &mut w,
                    parent[i].map(|p| ents[p]),
                    has_local.then_some(local[i]),
                    f32g,
                    lwcg,
                );
                if kind[i] == 5 {
                    driven32[i] = Affine3::from_translation(Vec3::new(
                        rng.unit() * 50.0,
                        rng.unit() * 50.0,
                        rng.unit() * 50.0,
                    ));
                    *w.get_static_mut::<GlobalTransform>(e).unwrap() = GlobalTransform(driven32[i]);
                }
                if kind[i] == 6 {
                    driven64[i] = DAffine3::from_translation(DVec3::new(
                        1.0e8 + rng.unit() as f64 * 100.0,
                        rng.unit() as f64 * 100.0,
                        rng.unit() as f64 * 100.0,
                    ));
                    *w.get_static_mut::<GlobalTransformLWC>(e).unwrap() =
                        GlobalTransformLWC(driven64[i]);
                }
                ents.push(e);
            }
            w.propagate_transforms();

            // The model: parents come before children, so one pass in order.
            let mut frames: Vec<M> = Vec::new();
            for i in 0..n {
                let above = parent[i].map_or(M::Id, |p| frames[p]);
                let lwc = matches!(kind[i], 1 | 2 | 6);
                let frame = match kind[i] {
                    3 => above,
                    5 => M::F32(driven32[i]),
                    6 => M::F64(driven64[i]),
                    _ => {
                        if lwc || matches!(above, M::F64(_)) {
                            M::F64(m64(above) * promote(local[i].to_affine()))
                        } else {
                            M::F32(m32(above) * local[i].to_affine())
                        }
                    }
                };
                frames.push(frame);
                match kind[i] {
                    0 | 2 => assert!(
                        g(&w, ents[i]) == m32(frame),
                        "seed {seed}, node {i}, f32 global"
                    ),
                    _ => {}
                }
                match kind[i] {
                    1 | 2 => assert!(
                        gl(&w, ents[i]) == m64(frame),
                        "seed {seed}, node {i}, lwc global"
                    ),
                    _ => {}
                }
                if kind[i] == 4 {
                    assert!(!w.has_static::<GlobalTransform>(ents[i]));
                }
                if parent[i].is_none() {
                    roots += 1;
                }
                if depth[i] >= 4 {
                    deep += 1;
                }
                if let (1 | 2, Some(p)) = (kind[i], parent[i]) {
                    if matches!(frames[p], M::F64(_)) {
                        lwc_under_lwc += 1;
                    }
                }
                if let (0 | 2, Some(p)) = (kind[i], parent[i]) {
                    if matches!(frames[p], M::F64(_)) {
                        f32_under_lwc += 1;
                    }
                }
            }
        }
        assert!(roots >= 80, "too few roots: {roots}");
        assert!(deep >= 150, "too few deep nodes: {deep}");
        assert!(lwc_under_lwc >= 60, "too few f64 over f64: {lwc_under_lwc}");
        assert!(
            f32_under_lwc >= 60,
            "too few f32 under f64: {f32_under_lwc}"
        );
    }
}
