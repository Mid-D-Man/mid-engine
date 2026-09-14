// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-ecs.md, section "scratch.rs"
// ============================================================================
//! Transient, type-erased scratch storage for structural changes.
//!
//! `archetype.rs`'s own doc comment already names the cost this exists
//! to address: each migrated component value is briefly boxed as
//! `Box<dyn Any>` (`Column::swap_remove_and_forget` / `push_any`) —
//! "one heap allocation per moved component per structural change."
//! `spawn_insert_bundle` and `structural_churn` are both flagged in
//! `benches/archetype_core.rs` as real, measured gaps against
//! `bevy_ecs` (~2.9-5x) with **no root-cause pass done yet** — this
//! module doesn't claim boxing IS that gap, only that it's the one
//! named, understood cost in the current design, and the obvious first
//! thing to make swappable before spending a session profiling to
//! confirm or rule it out.
//!
//! # What's here, and what isn't yet
//!
//! [`ScratchArena`] is the abstraction: "give me type-erased, owned
//! storage for a value, get it back downcast later, same contract
//! `Box<dyn Any>` already has." Two real implementations:
//!
//! - [`HeapScratch`] — today's actual behavior, just moved behind the
//!   trait: one real heap allocation per `alloc`, freed whenever the
//!   returned value is dropped or consumed. Zero new dependencies,
//!   default, always available.
//! - `BumpaloScratch` (behind the `scratch-arena` feature) — backed by
//!   `bumpalo::Bump` plus its own `boxed::Box<'a, dyn Any>`, which
//!   gives the same owned-with-correct-Drop contract `Box<dyn Any>`
//!   has, but reclaims memory in one `reset()` per structural change
//!   instead of one `free` per moved component.
//!
//! **Neither is wired into `Column::swap_remove_and_forget`/`push_any`
//! yet.** That's the deliberate next step, not this one — this module
//! is standalone and tested on its own, so the swap (bumpalo today, a
//! future `mid-arena`-backed arena once that's been benched against
//! bumpalo for this exact access pattern, not just `vs_arena_crates`'
//! general survey) is a matter of satisfying [`ScratchArena`], not
//! rewriting `archetype.rs`'s migration path a second time.
//!
//! # Why a trait instead of picking bumpalo directly
//!
//! `mid-arena` already exists in this workspace, with its own
//! `BumpArena<T>` — but that's a single-typed, chunk-linked arena for
//! many values of *one* `T` (see `docs/mid-arena.md`), not a match for
//! what a structural change actually needs: one arena holding several
//! *different* component types at once, for the width of one entity's
//! migration, then thrown away. `bumpalo::Bump` already solves exactly
//! that shape (it's what `bevy_ecs`'s own `BundleScratch` uses, real
//! source read directly), so it's the right backing *today* — the
//! trait exists so that if `mid-arena` grows a matching
//! heterogeneous-scratch type later, or this project ends up wanting
//! its own for dependency reasons, swapping is changing which type
//! gets named at the one call site that will eventually exist in
//! `archetype.rs`, not redesigning `Column` a second time.

use std::any::Any;

/// A type-erased, owned value handle a [`ScratchArena`] hands back.
/// Same contract `Box<dyn Any>` already has: dropping one without
/// downcasting it still runs the original value's destructor exactly
/// once. `downcast` mirrors `Box<dyn Any>::downcast` deliberately — on
/// the wrong type it returns the handle unchanged in `Err`, rather than
/// losing or dropping the value.
pub(crate) trait ErasedValue: Sized {
    fn downcast<T: 'static>(self) -> Result<T, Self>;
}

/// Provides scratch storage for the type-erased values a structural
/// change moves between columns. See this file's module doc comment
/// for what's real here today and what's still a future step.
///
/// `alloc` takes `&'a self` (not `&mut self`), matching
/// `bumpalo::Bump::alloc`'s own shape — lets every component being
/// migrated for one entity allocate from the same arena without
/// fighting the borrow checker over a single exclusive borrow. `reset`
/// takes `&mut self` specifically so it can't compile at a call site
/// where any `Self::Erased` this arena produced is still alive — for
/// the bumpalo-backed implementation this is enforced by the borrow
/// checker itself, not just documented; see this file's tests.
pub(crate) trait ScratchArena<'a> {
    type Erased: ErasedValue;

    fn alloc<T: 'static>(&'a self, value: T) -> Self::Erased;

    fn reset(&mut self);
}

// ── HeapScratch: today's real behavior, named and testable on its own ──

/// The default, dependency-free [`ScratchArena`]: `Column`'s current
/// actual behavior (one heap allocation per `alloc`), just given a name
/// and a place to be swapped out from.
pub(crate) struct HeapScratch;

impl ErasedValue for Box<dyn Any> {
    fn downcast<T: 'static>(self) -> Result<T, Self> {
        Box::<dyn Any>::downcast::<T>(self).map(|boxed| *boxed)
    }
}

impl<'a> ScratchArena<'a> for HeapScratch {
    type Erased = Box<dyn Any>;

    fn alloc<T: 'static>(&'a self, value: T) -> Self::Erased {
        Box::new(value)
    }

    fn reset(&mut self) {
        // Nothing to reclaim in bulk -- each value already frees itself
        // independently, the same way `Column`'s current `Box<dyn Any>`
        // usage already does.
    }
}

// ── BumpaloScratch: the actual arena-backed implementation ─────────────
// A whole submodule, gated once, rather than `#[cfg(...)]` on every
// item -- `bumpalo` is an optional dependency, so any unconditional
// `use bumpalo::...` at file scope would be a hard compile error with
// the feature off, not just a warning.
#[cfg(feature = "scratch-arena")]
mod bumpalo_backed {
    use super::{ErasedValue, ScratchArena};
    use std::any::Any;

    /// Bump-allocated [`ScratchArena`]. Backed by `bumpalo::Bump`
    /// directly (not `mid-arena`'s own `BumpArena<T>` — see this file's
    /// module doc comment for why that's a different shape than this
    /// needs) and `bumpalo`'s own `boxed` feature, which gives a
    /// `Box<'a, dyn Any>` with the same owned-and-correctly-dropped
    /// contract `alloc::boxed::Box<dyn Any>` has (real source read
    /// directly, `bumpalo` 3.20.3's `src/boxed.rs`: its `Drop` impl
    /// calls `drop_in_place` and nothing else — the arena, not the
    /// `Box`, owns reclaiming the memory itself, exactly the property
    /// this module exists to use).
    pub(crate) struct BumpaloScratch(bumpalo::Bump);

    impl BumpaloScratch {
        pub(crate) fn new() -> Self {
            Self(bumpalo::Bump::new())
        }
    }

    impl Default for BumpaloScratch {
        fn default() -> Self {
            Self::new()
        }
    }

    pub(crate) struct BumpaloErased<'a>(bumpalo::boxed::Box<'a, dyn Any>);

    impl<'a> ErasedValue for BumpaloErased<'a> {
        fn downcast<T: 'static>(self) -> Result<T, Self> {
            match self.0.downcast::<T>() {
                Ok(boxed) => Ok(bumpalo::boxed::Box::into_inner(boxed)),
                Err(erased) => Err(BumpaloErased(erased)),
            }
        }
    }

    impl<'a> ScratchArena<'a> for BumpaloScratch {
        type Erased = BumpaloErased<'a>;

        fn alloc<T: 'static>(&'a self, value: T) -> Self::Erased {
            let boxed = bumpalo::boxed::Box::new_in(value, &self.0);
            let raw: *mut T = bumpalo::boxed::Box::into_raw(boxed);
            // SAFETY: `raw` was produced by `Box::into_raw` on the line
            // above, so it's a valid, uniquely-owned, non-null
            // allocation from this same arena with `T`'s own layout --
            // exactly what `Box::from_raw` requires. The `as *mut dyn
            // Any` widening only builds a fat pointer (vtable + data);
            // it doesn't read through the pointer, so it's an ordinary
            // safe unsize coercion, not part of what makes this block
            // unsafe. This is `bumpalo`'s own documented pattern for
            // building a type-erased `Box` (`boxed.rs`'s "Manually
            // create a `Box`" doc example), not an invented technique.
            let erased: bumpalo::boxed::Box<'a, dyn Any> =
                unsafe { bumpalo::boxed::Box::from_raw(raw as *mut dyn Any) };
            BumpaloErased(erased)
        }

        fn reset(&mut self) {
            self.0.reset();
        }
    }
}

#[cfg(feature = "scratch-arena")]
pub(crate) use bumpalo_backed::BumpaloScratch;

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::rc::Rc;

    struct Position {
        x: f32,
        y: f32,
    }

    /// Increments a shared counter on drop -- lets a test prove the
    /// destructor ran exactly once, whether the value was reclaimed via
    /// `downcast` or dropped un-downcast.
    struct DropCounter(Rc<Cell<u32>>);
    impl Drop for DropCounter {
        fn drop(&mut self) {
            self.0.set(self.0.get() + 1);
        }
    }

    fn roundtrip<'a, A: ScratchArena<'a>>(arena: &'a A) {
        let erased = arena.alloc(Position { x: 1.0, y: 2.0 });
        let pos = erased
            .downcast::<Position>()
            .unwrap_or_else(|_| panic!("downcast to the type just allocated should succeed"));
        assert_eq!(pos.x, 1.0);
        assert_eq!(pos.y, 2.0);
    }

    fn wrong_type_roundtrip<'a, A: ScratchArena<'a>>(arena: &'a A) {
        let erased = arena.alloc(Position { x: 1.0, y: 2.0 });
        // Downcasting to the wrong type must fail *and* hand the value
        // back unharmed, same as `Box<dyn Any>::downcast` -- not drop
        // it, not panic.
        let erased = match erased.downcast::<u64>() {
            Ok(_) => panic!("downcast to the wrong type must not succeed"),
            Err(erased) => erased,
        };
        let pos = erased.downcast::<Position>().unwrap_or_else(|_| {
            panic!("the value must still be recoverable after a failed downcast")
        });
        assert_eq!(pos.x, 1.0);
    }

    fn drop_without_downcast_runs_once<'a, A: ScratchArena<'a>>(arena: &'a A) {
        let counter = Rc::new(Cell::new(0u32));
        let erased = arena.alloc(DropCounter(counter.clone()));
        assert_eq!(counter.get(), 0);
        drop(erased);
        assert_eq!(
            counter.get(),
            1,
            "dropping an un-downcast value must still run its destructor exactly once"
        );
    }

    fn multiple_live_allocations_stay_independent<'a, A: ScratchArena<'a>>(arena: &'a A) {
        let a = arena.alloc(1u32);
        let b = arena.alloc("two");
        let c = arena.alloc(Position { x: 3.0, y: 4.0 });
        assert_eq!(a.downcast::<u32>().ok(), Some(1));
        assert_eq!(b.downcast::<&'static str>().ok(), Some("two"));
        let pos = c.downcast::<Position>().ok().expect("Position roundtrip");
        assert_eq!((pos.x, pos.y), (3.0, 4.0));
    }

    #[test]
    fn heap_scratch_roundtrip() {
        roundtrip(&HeapScratch);
    }

    #[test]
    fn heap_scratch_wrong_type() {
        wrong_type_roundtrip(&HeapScratch);
    }

    #[test]
    fn heap_scratch_drop_without_downcast() {
        drop_without_downcast_runs_once(&HeapScratch);
    }

    #[test]
    fn heap_scratch_multiple_live_allocations() {
        multiple_live_allocations_stay_independent(&HeapScratch);
    }

    #[cfg(feature = "scratch-arena")]
    #[test]
    fn bumpalo_scratch_roundtrip() {
        roundtrip(&BumpaloScratch::new());
    }

    #[cfg(feature = "scratch-arena")]
    #[test]
    fn bumpalo_scratch_wrong_type() {
        wrong_type_roundtrip(&BumpaloScratch::new());
    }

    #[cfg(feature = "scratch-arena")]
    #[test]
    fn bumpalo_scratch_drop_without_downcast() {
        drop_without_downcast_runs_once(&BumpaloScratch::new());
    }

    #[cfg(feature = "scratch-arena")]
    #[test]
    fn bumpalo_scratch_multiple_live_allocations() {
        multiple_live_allocations_stay_independent(&BumpaloScratch::new());
    }

    #[cfg(feature = "scratch-arena")]
    #[test]
    fn bumpalo_scratch_reset_reclaims_after_all_values_consumed() {
        let mut arena = BumpaloScratch::new();
        {
            let erased = arena.alloc(Position { x: 5.0, y: 6.0 });
            let _ = erased
                .downcast::<Position>()
                .unwrap_or_else(|_| panic!("downcast to the type just allocated should succeed"));
        }
        // Only reachable once every `Erased` this arena produced is out
        // of scope -- `reset` takes `&mut self`, and `alloc` borrowed
        // `&'a self`, so the borrow checker is what actually enforces
        // this, not just the doc comment above.
        arena.reset();
        // Arena is still usable after reset.
        let erased = arena.alloc(7u32);
        assert_eq!(erased.downcast::<u32>().ok(), Some(7));
    }
}
