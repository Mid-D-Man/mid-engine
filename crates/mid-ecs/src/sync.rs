//! Network sync — marks components for `mid-net` replication.
//!
//! This is the Multiplayer-First mandate in practice: networking is
//! baked into the ECS, not bolted on later. See `docs/mid-ecs.md`'s own
//! "Network Sync" section for the full vision — `@net`-style
//! attribute flagging via DixScript, automatic serialization, a SIMD
//! pass over contiguous Archetype Core memory to detect deltas,
//! MBFA-lite compression, encryption, shipped over `mid-net`. None of
//! that exists yet, and this file doesn't attempt it — DixScript
//! codegen, delta detection, and compression/encryption are each a
//! real, separate, later undertaking, not something to bolt on here
//! as a side effect of a marking registry.
//!
//! What this file *does* build: the one small, real piece every one of
//! those later systems needs as its first input regardless of how the
//! rest gets designed — a way to say "this component type is
//! network-relevant" at all, and to ask that back later. Deliberately
//! keyed by `TypeId`, not `ComponentId`: a marked type doesn't have to
//! have been registered with either storage system yet (see
//! `World::mark_for_sync`'s own doc comment for why that matters), and
//! `ComponentId` would also force picking one storage system's
//! numbering space over the other for no real reason at this layer —
//! whether a component is network-relevant has nothing to do with
//! which storage system it happens to live in.

use std::any::TypeId;
use std::collections::HashSet;

use crate::world::World;

/// A plain set of marked `TypeId`s. Its own type, not just a bare
/// `HashSet` field on `World` directly, so this file's own doc comment
/// has somewhere to live and so a future delta-detection pass has a
/// real, named thing to hold whatever per-type metadata it eventually
/// needs (last-synced snapshot, dirty flags, replication priority) —
/// today, a plain marked/not-marked set is all there's a real,
/// grounded need for.
#[derive(Debug, Default)]
pub struct SyncRegistry {
    marked: HashSet<TypeId>,
}

impl SyncRegistry {
    pub(crate) fn new() -> Self {
        Self::default()
    }
}

impl World {
    /// Marks `T` as network-relevant. Idempotent — marking an
    /// already-marked type is a safe no-op, not an error, matching
    /// this crate's own established `register_ffi`/`register_ffi_static`
    /// convention for "declaring intent" calls.
    ///
    /// Deliberately does **not** require `T` to already be registered
    /// with either `SparseShell` or `Archetypes` — a real, deliberate
    /// choice, not an oversight: which storage system a type ends up
    /// in is an unrelated decision (see `docs/mid-ecs.md`'s "Hybrid
    /// ECS Architecture" section — volatile vs. structural, not
    /// synced vs. not-synced), and a caller marking intent to
    /// eventually replicate a type shouldn't be blocked on, or forced
    /// to think about, which storage system it'll live in.
    pub fn mark_for_sync<T: 'static>(&mut self) {
        self.sync.marked.insert(TypeId::of::<T>());
    }

    /// Whether `T` was ever marked via [`Self::mark_for_sync`].
    pub fn is_marked_for_sync<T: 'static>(&self) -> bool {
        self.sync.marked.contains(&TypeId::of::<T>())
    }

    /// Unmarks `T`. A safe no-op if `T` was never marked.
    pub fn unmark_for_sync<T: 'static>(&mut self) {
        self.sync.marked.remove(&TypeId::of::<T>());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Position {
        #[allow(dead_code)]
        x: f32,
    }
    struct Velocity {
        #[allow(dead_code)]
        dx: f32,
    }

    #[test]
    fn a_type_starts_unmarked() {
        let w = World::new();
        assert!(!w.is_marked_for_sync::<Position>());
    }

    #[test]
    fn mark_then_is_marked_round_trips() {
        let mut w = World::new();
        w.mark_for_sync::<Position>();
        assert!(w.is_marked_for_sync::<Position>());
    }

    #[test]
    fn marking_one_type_does_not_mark_another() {
        let mut w = World::new();
        w.mark_for_sync::<Position>();
        assert!(!w.is_marked_for_sync::<Velocity>());
    }

    #[test]
    fn marking_is_idempotent() {
        let mut w = World::new();
        w.mark_for_sync::<Position>();
        w.mark_for_sync::<Position>(); // real, safe no-op, not a panic
        assert!(w.is_marked_for_sync::<Position>());
    }

    #[test]
    fn unmark_removes_the_mark() {
        let mut w = World::new();
        w.mark_for_sync::<Position>();
        w.unmark_for_sync::<Position>();
        assert!(!w.is_marked_for_sync::<Position>());
    }

    #[test]
    fn unmark_on_a_never_marked_type_is_a_safe_no_op() {
        let mut w = World::new();
        w.unmark_for_sync::<Position>(); // never marked -- must not panic
        assert!(!w.is_marked_for_sync::<Position>());
    }

    #[test]
    fn marking_does_not_require_prior_ffi_or_storage_registration() {
        // The real point of keying this off TypeId, not ComponentId:
        // Position has never touched SparseShell or Archetypes at all
        // here, and marking it still works.
        let mut w = World::new();
        w.mark_for_sync::<Position>();
        assert!(w.is_marked_for_sync::<Position>());
    }
}
