//! Archetype query iterators.
//!
//! Split out of `archetype.rs` as a child module -- `Table::columns`,
//! `Table::entities`, the `Column` trait and `Archetype` are all
//! private to `archetype`, and a child module sees its parent's
//! private items without any visibility being widened.
//!
//! # Revision history, stated plainly rather than silently overwritten
//!
//! The first version of this module (this crate's own history has the
//! diff) restructured `Iter1`/`Iter2::next` into a `#[cold]
//! #[inline(never)] fn advance` cold/hot split, added `#[inline]` to
//! `next` itself, and added `fold`/`size_hint` overrides -- on the
//! hypothesis that `Iter2`'s `Item` being 24 bytes (`(Entity, &A, &B)`,
//! MEMORY-class under SysV AMD64) rather than 16
//! (`Iter1`'s `(Entity, &T)`, RAX:RDX-class) was the dominant cost.
//!
//! Real CI (Archetype Core builds #29/#30, immediately after that
//! version shipped) refuted it directly, and did so by breaking
//! something that was previously fine: `query_static_single_component`
//! -- previously at parity with `bevy_ecs` (9.42µs vs 9.35µs, build
//! #22) -- collapsed to the same ~4x-over-floor regime
//! `query2_static_two_components` has always been in, while
//! `query2_static_two_components` itself and `raw_slice_ceiling` both
//! stayed exactly where they'd always been. The project's own
//! regression-guard script caught this live and flagged it correctly
//! ("query_static_single_component itself is running 4.3×
//! raw_slice_ceiling's floor this run... a low ratio here likely means
//! the baseline got worse, not that query2_static got better").
//!
//! That result should have been read against `docs/mid-ecs.md`'s own
//! closing sections (builds #19/#20 and "Iter1 Isolated" #1/#2) --
//! read in full only after the fact, which is the real process failure
//! here, not the hypothesis itself being tried. Those sections had
//! *already run* effectively this same comparison (1-column item vs
//! 2-column item, LTO held fixed) and found the two within ~5% of each
//! other under `bench-nolto` -- item size is not the dominant factor.
//! The actual, upstream-corroborated (rust-lang/rust#106609, #146497)
//! mechanism they identify instead: **LTO's whole-program inliner has
//! a finite budget, and it stops fully resolving `next()`'s call sites
//! once the surrounding compilation unit gets large enough** --
//! confirmed by `benches/iter1-isolated`, where the real, unmodified
//! `Iter1` returns to parity with LTO and to the same ~4x everything
//! else shows without it, with nothing about `Iter1`'s own source
//! changed between those two runs.
//!
//! Under that mechanism, the cold-split version here is the textbook
//! way to trigger the regression, not fix it: it made `next()`'s own
//! estimated inline cost bigger (a real, un-inlined call to `advance`
//! on the cold branch; a `fold` override; an added attribute) inside a
//! bench binary (`archetype_core.rs`) already right at the edge of
//! that budget, per the same closing sections. This revision undoes
//! all of it -- `Iter1`/`Iter2` below are byte-for-byte the same
//! bodies that measured at parity/4x respectively before any of this
//! module existed. The investigation itself was explicitly closed in
//! `docs/mid-ecs.md` ("further digging is diminishing returns"); this
//! module doesn't reopen it, it just stops actively fighting its
//! conclusion.
//!
//! # What's still here, and why
//!
//! `Iter1Ref`/`Iter2Ref` (16-byte item, no `Entity`) stay -- they're
//! the real, apples-to-apples shape against `bevy_ecs`'s
//! `Query<&A>`/`Query<(&A, &B)>` regardless of which mechanism turns
//! out to explain the gap, and dropping them would throw away a
//! legitimate API improvement over an unrelated regression. They're
//! built with the exact same structure as `Iter1`/`Iter2` -- same
//! inline cold path, no attribute, no overrides -- so the only
//! remaining difference between `Iter1`/`Iter1Ref` and between
//! `Iter2`/`Iter2Ref` is genuinely just the returned item.
//!
//! Whether that remaining difference matters is now an open question
//! again, not a closed one -- and the right way to ask it, per this
//! project's own established method, is `benches/iter1-isolated`'s
//! approach (the real method, in a minimal real compilation unit, real
//! `mid-ecs` dependency, nothing else in the binary), not another
//! bundled change to the file that's already known to be tight on
//! budget. See `benches/query2-ref-isolated` for that test.

use super::{ArchetypeId, Archetypes, Column};
use crate::component::ComponentId;
use crate::world::Entity;

// =====================================================================
// One component
// =====================================================================

/// The real `Iterator` behind [`Archetypes::iter`] -- see that method's
/// own doc comment and [`Iter2`]'s doc comment for the full
/// real-numbers writeup of why this is a hand-written state machine,
/// not `flat_map`/`zip` adaptors. Same shape as [`Iter2`], one column
/// instead of two.
///
/// Body unchanged from before this module existed -- see this module's
/// own header for why that's now a deliberate constraint, not an
/// oversight.
pub(crate) struct Iter1<'a, T> {
    archetypes: &'a Archetypes,
    id: Option<ComponentId>,
    matched: std::vec::IntoIter<ArchetypeId>,
    entities: &'a [Entity],
    column: &'a [T],
    row: usize,
    len: usize,
}

impl<'a, T: 'static> Iter1<'a, T> {
    #[inline]
    pub(crate) fn new(
        archetypes: &'a Archetypes,
        id: Option<ComponentId>,
        matched: Vec<ArchetypeId>,
    ) -> Self {
        Self {
            archetypes,
            id,
            matched: matched.into_iter(),
            entities: &[],
            column: &[],
            row: 0,
            len: 0,
        }
    }
}

impl<'a, T: 'static> Iterator for Iter1<'a, T> {
    type Item = (Entity, &'a T);

    // Deliberately NOT #[inline(always)] -- tried it (reasoning: bevy's
    // own `QueryIterationCursor::next` has it explicitly, and this
    // method's sibling `Iter2::next` regressed ~4x on real CI's rustc
    // 1.98.0 specifically), and it made this method measurably WORSE
    // on this sandbox's rustc 1.91.1 -- a real, reproducible ~4x
    // regression (28.6µs vs 6.6-7.2µs at N=10,000, confirmed across
    // repeated runs, not noise), the opposite of the intended fix.
    // Reverted rather than shipped on an unproven, one-toolchain-tested
    // hypothesis. Also NOT split into a separate cold `advance` -- that
    // was tried too, real CI (Archetype Core builds #29/#30), and it
    // reproduced this exact regression for the first time on this
    // function specifically. See this module's own header.
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.row < self.len {
                let item = (self.entities[self.row], &self.column[self.row]);
                self.row += 1;
                return Some(item);
            }
            let id = self.id?;
            let archetype_id = self.matched.next()?;
            let archetype = self.archetypes.archetypes.get(archetype_id).expect(
                "iter's precomputed matched list only ever contains real, currently-existing archetype ids",
            );
            let entities: &[Entity] = &archetype.table.entities;
            let column: &[T] = match archetype.table.columns.get(id) {
                Some(column) => column
                    .as_any()
                    .downcast_ref::<Vec<T>>()
                    .expect("column type must match component_id's T")
                    .as_slice(),
                None => &[],
            };
            self.len = entities.len().min(column.len());
            self.entities = entities;
            self.column = column;
            self.row = 0;
        }
    }
}

/// Entity-free counterpart to [`Iter1`]: yields `&T` alone. Same
/// fields (including the unused `entities` slice -- kept only for
/// structural parity with [`Iter1`], not because this type needs it,
/// so the two types differ in exactly one place: `Item`), same body
/// shape, same lack of any inline attribute or cold-path split. The
/// counterpart to `bevy_ecs`'s `Query<&T>`, which yields exactly this.
pub(crate) struct Iter1Ref<'a, T> {
    archetypes: &'a Archetypes,
    id: Option<ComponentId>,
    matched: std::vec::IntoIter<ArchetypeId>,
    entities: &'a [Entity],
    column: &'a [T],
    row: usize,
    len: usize,
}

impl<'a, T: 'static> Iter1Ref<'a, T> {
    #[inline]
    pub(crate) fn new(
        archetypes: &'a Archetypes,
        id: Option<ComponentId>,
        matched: Vec<ArchetypeId>,
    ) -> Self {
        Self {
            archetypes,
            id,
            matched: matched.into_iter(),
            entities: &[],
            column: &[],
            row: 0,
            len: 0,
        }
    }
}

impl<'a, T: 'static> Iterator for Iter1Ref<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.row < self.len {
                let item = &self.column[self.row];
                self.row += 1;
                return Some(item);
            }
            let id = self.id?;
            let archetype_id = self.matched.next()?;
            let archetype = self.archetypes.archetypes.get(archetype_id).expect(
                "iter's precomputed matched list only ever contains real, currently-existing archetype ids",
            );
            let entities: &[Entity] = &archetype.table.entities;
            let column: &[T] = match archetype.table.columns.get(id) {
                Some(column) => column
                    .as_any()
                    .downcast_ref::<Vec<T>>()
                    .expect("column type must match component_id's T")
                    .as_slice(),
                None => &[],
            };
            self.len = entities.len().min(column.len());
            self.entities = entities;
            self.column = column;
            self.row = 0;
        }
    }
}

// =====================================================================
// Two components
// =====================================================================

/// flat state machine, not a chain of `flat_map`/`filter`/`zip`
/// adaptors. Shape (precompute matching storages once, then a single
/// `if current_row == current_len { advance } else { fetch, ++ }`
/// loop) copied deliberately from `bevy_ecs`'s own
/// `QueryIterationCursor::next` (`query/iter.rs`, real source read
/// directly -- see this module's doc comment) after a controlled bench
/// showed it mattered.
///
/// **What the earlier combinator-based `iter2` still got wrong,** even
/// after the per-entity-lookup fix (see git history / the previous
/// version of this doc comment for that writeup): `for` loops in Rust
/// always drive an iterator through repeated `Iterator::next()` calls
/// (this is worth stating precisely -- an earlier internal writeup of
/// this exact investigation incorrectly assumed `for` loops could
/// dispatch to a custom `fold` override; they can't, `for` only ever
/// calls `next()`). The real difference is what `next()` *is*: `bevy_ecs`
/// writes `QueryIterationCursor::next` as a single, self-contained,
/// `#[inline(always)]`-adjacent function -- one `loop`, `get_unchecked`
/// (no bounds check), no nested generic adaptor types. The old `iter2`
/// composed `Option::into_iter().flat_map(|_| ...filter(...).flat_map(|_|
/// ...zip...))` -- each layer is its *own* `Iterator` impl with its own
/// `next()`, and driving the outer one means threading through all of
/// them, even once "locked onto" one archetype's `zip`. Controlled,
/// same-machine bench (`crates/mid-ecs/benches/archetype_core.rs`,
/// N=10,000, single archetype, `--sample-size 30`): the combinator
/// version measured 152.34µs; the flat hand-written loop below, with
/// the exact same bounds-checked (safe, no `unsafe`) indexing,
/// measured 9.1605µs -- a ~94% reduction, landing within noise of
/// `bevy_ecs`'s own real CI number for the same N=10,000 workload
/// (9.3882µs, `benches/ecs-vs-bevy-ecs`'s real run). The per-entity
/// redundant-lookup fix (the version this replaced) was real and
/// worth keeping, but it was never the dominant cost -- the generic
/// `flat_map`/`filter`/`zip` adaptor stack itself was, and removing
/// it closed nearly the entire gap against `bevy_ecs`, with zero
/// `unsafe`.
///
/// `entities`/`a_col`/`b_col` are always the *same* length by
/// construction (`next` clamps `len` to the min of all three on every
/// archetype advance -- belt-and-suspenders on top of the "no column
/// implies no rows" invariant [`Archetypes::iter`] already documents
/// and relies on), so `row < len` alone is what makes every
/// `entities[row]`/`a_col[row]`/`b_col[row]` access below provably
/// in-bounds -- safe, ordinary Rust indexing, not `unsafe`. Revisit
/// only if a *further* real bench shows the three redundant bounds
/// checks this still pays (one per slice, per item) are themselves
/// the next real cost -- `docs/mid-ecs.md`'s "zero unsafe by choice,
/// revisit only against a real profile" policy applies exactly the
/// same way here as it always has.
///
/// Body unchanged from before this module existed, same as [`Iter1`] --
/// see this module's own header.
pub(crate) struct Iter2<'a, A, B> {
    archetypes: &'a Archetypes,
    ids: Option<(ComponentId, ComponentId)>,
    matched: std::vec::IntoIter<ArchetypeId>,
    entities: &'a [Entity],
    a_col: &'a [A],
    b_col: &'a [B],
    row: usize,
    len: usize,
}

impl<'a, A: 'static, B: 'static> Iter2<'a, A, B> {
    #[inline]
    pub(crate) fn new(
        archetypes: &'a Archetypes,
        ids: Option<(ComponentId, ComponentId)>,
        matched: Vec<ArchetypeId>,
    ) -> Self {
        Self {
            archetypes,
            ids,
            matched: matched.into_iter(),
            entities: &[],
            a_col: &[],
            b_col: &[],
            row: 0,
            len: 0,
        }
    }
}

impl<'a, A: 'static, B: 'static> Iterator for Iter2<'a, A, B> {
    type Item = (Entity, &'a A, &'a B);

    // Deliberately NOT #[inline(always)] -- see Iter1::next's doc
    // comment above for the full story: tried it here specifically
    // (this method has the real, confirmed CI regression it was meant
    // to fix -- 26.605µs/9.1913µs on this sandbox's rustc 1.91.1 vs a
    // genuine ~4x against bevy_ecs on real CI's rustc 1.98.0), and it
    // made this method measurably WORSE on this sandbox too (~29µs,
    // matching Iter1's own regression exactly). Reverted. The real
    // rustc-1.98.0-specific regression this was meant to explain is
    // now understood -- see this module's own header -- and isn't
    // fixable by any attribute or structural change to this function
    // alone; also NOT split into a separate cold `advance`, same
    // reasoning as `Iter1::next`.
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.row < self.len {
                let item = (
                    self.entities[self.row],
                    &self.a_col[self.row],
                    &self.b_col[self.row],
                );
                self.row += 1;
                return Some(item);
            }
            let (a_id, b_id) = self.ids?;
            let archetype_id = self.matched.next()?;
            let archetype = self.archetypes.archetypes.get(archetype_id).expect(
                "iter2's precomputed matched list only ever contains real, currently-existing archetype ids",
            );
            let entities: &[Entity] = &archetype.table.entities;
            let a_col: &[A] = match archetype.table.columns.get(a_id) {
                Some(column) => column
                    .as_any()
                    .downcast_ref::<Vec<A>>()
                    .expect("column type must match component_id's T")
                    .as_slice(),
                None => &[],
            };
            let b_col: &[B] = match archetype.table.columns.get(b_id) {
                Some(column) => column
                    .as_any()
                    .downcast_ref::<Vec<B>>()
                    .expect("column type must match component_id's T")
                    .as_slice(),
                None => &[],
            };
            self.len = entities.len().min(a_col.len()).min(b_col.len());
            self.entities = entities;
            self.a_col = a_col;
            self.b_col = b_col;
            self.row = 0;
        }
    }
}

/// Entity-free counterpart to [`Iter2`]: yields `(&A, &B)`. Same
/// fields as [`Iter2`] (including the unused `entities` slice), same
/// body shape, same lack of any inline attribute or cold-path split --
/// the only difference anywhere in this type from [`Iter2`] is `Item`.
///
/// The apples-to-apples shape against `bevy_ecs`'s `Query<(&A, &B)>`.
/// `benches/ecs-vs-bevy-ecs`'s `dense_query_iteration` has never
/// actually compared the same thing on both sides: it puts `Iter2`'s
/// three-element item against bevy's two-element one, discarding the
/// entity with `_` on the mid-ecs side only. Whether that gap matters
/// once measured cleanly is now open again -- see
/// `benches/query2-ref-isolated`, not this doc comment, for an answer.
pub(crate) struct Iter2Ref<'a, A, B> {
    archetypes: &'a Archetypes,
    ids: Option<(ComponentId, ComponentId)>,
    matched: std::vec::IntoIter<ArchetypeId>,
    entities: &'a [Entity],
    a_col: &'a [A],
    b_col: &'a [B],
    row: usize,
    len: usize,
}

impl<'a, A: 'static, B: 'static> Iter2Ref<'a, A, B> {
    #[inline]
    pub(crate) fn new(
        archetypes: &'a Archetypes,
        ids: Option<(ComponentId, ComponentId)>,
        matched: Vec<ArchetypeId>,
    ) -> Self {
        Self {
            archetypes,
            ids,
            matched: matched.into_iter(),
            entities: &[],
            a_col: &[],
            b_col: &[],
            row: 0,
            len: 0,
        }
    }
}

impl<'a, A: 'static, B: 'static> Iterator for Iter2Ref<'a, A, B> {
    type Item = (&'a A, &'a B);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.row < self.len {
                let item = (&self.a_col[self.row], &self.b_col[self.row]);
                self.row += 1;
                return Some(item);
            }
            let (a_id, b_id) = self.ids?;
            let archetype_id = self.matched.next()?;
            let archetype = self.archetypes.archetypes.get(archetype_id).expect(
                "iter2's precomputed matched list only ever contains real, currently-existing archetype ids",
            );
            let entities: &[Entity] = &archetype.table.entities;
            let a_col: &[A] = match archetype.table.columns.get(a_id) {
                Some(column) => column
                    .as_any()
                    .downcast_ref::<Vec<A>>()
                    .expect("column type must match component_id's T")
                    .as_slice(),
                None => &[],
            };
            let b_col: &[B] = match archetype.table.columns.get(b_id) {
                Some(column) => column
                    .as_any()
                    .downcast_ref::<Vec<B>>()
                    .expect("column type must match component_id's T")
                    .as_slice(),
                None => &[],
            };
            self.len = entities.len().min(a_col.len()).min(b_col.len());
            self.entities = entities;
            self.a_col = a_col;
            self.b_col = b_col;
            self.row = 0;
        }
    }
}

// =====================================================================
// Constructors reached from `archetype.rs`
// =====================================================================

/// `Iter2UncheckedAlways` (this crate's own diagnostic history) already
/// tested `get_unchecked` + `#[inline(always)]` together — but only
/// ever inside `archetype_core.rs`, whose own size is exactly what the
/// closed investigation found LTO's inliner sensitive to. `Iter1`'s
/// real fix turned out to be isolation itself (`benches/iter1-isolated`),
/// not any attribute — so the one combination never actually tried is
/// unsafe + forced inlining *and* true isolation, together, on the
/// entity-free 16-byte item specifically. `benches/query2-ref-isolated`
/// is where that gets tested; this type is the same body as
/// [`Iter2Ref`], nothing else changed, so a result isolates exactly
/// this one variable the same way every other diagnostic in this
/// crate's history has.
pub(crate) struct Iter2RefUncheckedAlways<'a, A, B> {
    archetypes: &'a Archetypes,
    ids: Option<(ComponentId, ComponentId)>,
    matched: std::vec::IntoIter<ArchetypeId>,
    entities: &'a [Entity],
    a_col: &'a [A],
    b_col: &'a [B],
    row: usize,
    len: usize,
}

impl<'a, A: 'static, B: 'static> Iter2RefUncheckedAlways<'a, A, B> {
    #[inline]
    pub(crate) fn new(
        archetypes: &'a Archetypes,
        ids: Option<(ComponentId, ComponentId)>,
        matched: Vec<ArchetypeId>,
    ) -> Self {
        Self {
            archetypes,
            ids,
            matched: matched.into_iter(),
            entities: &[],
            a_col: &[],
            b_col: &[],
            row: 0,
            len: 0,
        }
    }
}

impl<'a, A: 'static, B: 'static> Iterator for Iter2RefUncheckedAlways<'a, A, B> {
    type Item = (&'a A, &'a B);

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.row < self.len {
                // SAFETY: `self.len` is set to `entities.len().min(a_col.len())
                // .min(b_col.len())` on every archetype advance below and
                // never grown afterward, so `self.row < self.len` already
                // proves `self.row` is in bounds for both `a_col` and
                // `b_col` — same invariant `Iter2Unchecked`/
                // `Iter2UncheckedAlways` already rely on.
                let item = unsafe {
                    (
                        self.a_col.get_unchecked(self.row),
                        self.b_col.get_unchecked(self.row),
                    )
                };
                self.row += 1;
                return Some(item);
            }
            let (a_id, b_id) = self.ids?;
            let archetype_id = self.matched.next()?;
            let archetype = self.archetypes.archetypes.get(archetype_id).expect(
                "iter2's precomputed matched list only ever contains real, currently-existing archetype ids",
            );
            let entities: &[Entity] = &archetype.table.entities;
            let a_col: &[A] = match archetype.table.columns.get(a_id) {
                Some(column) => column
                    .as_any()
                    .downcast_ref::<Vec<A>>()
                    .expect("column type must match component_id's T")
                    .as_slice(),
                None => &[],
            };
            let b_col: &[B] = match archetype.table.columns.get(b_id) {
                Some(column) => column
                    .as_any()
                    .downcast_ref::<Vec<B>>()
                    .expect("column type must match component_id's T")
                    .as_slice(),
                None => &[],
            };
            self.len = entities.len().min(a_col.len()).min(b_col.len());
            self.entities = entities;
            self.a_col = a_col;
            self.b_col = b_col;
            self.row = 0;
        }
    }
}

/// Verbatim copies of [`Archetypes::iter`]/[`Archetypes::iter2`]'s own
/// matched-archetype-list logic -- see those methods' doc comments in
/// `archetype.rs` for the real explanation. Duplicated here rather than
/// shared, deliberately: if the two ever diverge, an entity-free query
/// would silently visit a different archetype set than its
/// entity-carrying counterpart, which is a correctness bug, not a
/// performance one. Keep them in step by inspection, not by sharing.
impl Archetypes {
    pub(crate) fn iter_ref<T: 'static>(&self) -> Iter1Ref<'_, T> {
        let id = self.existing_component_id::<T>();
        let matched: Vec<ArchetypeId> = match id {
            Some(id) => self.archetypes_with(id).collect(),
            None => Vec::new(),
        };
        Iter1Ref::new(self, id, matched)
    }

    pub(crate) fn iter2_ref<A: 'static, B: 'static>(&self) -> Iter2Ref<'_, A, B> {
        let ids = self
            .existing_component_id::<A>()
            .zip(self.existing_component_id::<B>());
        let matched: Vec<ArchetypeId> = match ids {
            Some((a_id, b_id)) => self
                .archetypes_with(a_id)
                .filter(|&archetype_id| {
                    self.archetypes
                        .get(archetype_id)
                        .is_some_and(|archetype| archetype.component_ids.contains(&b_id))
                })
                .collect(),
            None => Vec::new(),
        };
        Iter2Ref::new(self, ids, matched)
    }

    pub(crate) fn iter2_ref_unchecked_always<A: 'static, B: 'static>(
        &self,
    ) -> Iter2RefUncheckedAlways<'_, A, B> {
        let ids = self
            .existing_component_id::<A>()
            .zip(self.existing_component_id::<B>());
        let matched: Vec<ArchetypeId> = match ids {
            Some((a_id, b_id)) => self
                .archetypes_with(a_id)
                .filter(|&archetype_id| {
                    self.archetypes
                        .get(archetype_id)
                        .is_some_and(|archetype| archetype.component_ids.contains(&b_id))
                })
                .collect(),
            None => Vec::new(),
        };
        Iter2RefUncheckedAlways::new(self, ids, matched)
    }
}
