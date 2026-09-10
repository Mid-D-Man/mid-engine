//! TEMPORARY DIAGNOSTIC MODULE — delete once the investigation this
//! exists for concludes, not meant to become permanent API surface.
//!
//! Real CI (rustc 1.98.0/1.98.1) shows `query2_static_two_components`
//! (backed by `archetype.rs`'s `Iter2`) running ~4x slower than
//! `bevy_ecs` on an equivalent workload, in both `benches/ecs-vs-bevy-ecs`
//! and this crate's own `archetype_core.rs` — while this sandbox's
//! rustc 1.91.1 measures the same code within noise of `bevy_ecs`. Tried
//! adding `#[inline(always)]` to `Iter2::next` (reasoning: `bevy_ecs`'s
//! own `QueryIterationCursor::next` has it explicitly — confirmed
//! directly against `Mid-D-Man/bevy`'s real `query/iter.rs` source) —
//! that made things measurably ~4x *worse* on this sandbox's rustc
//! 1.91.1, the opposite of the intended fix, and was reverted (see
//! `Iter2::next`'s own doc comment in `archetype.rs`). That revert
//! decision was made from sandbox data only — this module exists to
//! gather the same comparison from the actual affected toolchain
//! instead, since the sandbox has never once correctly predicted which
//! way one of these variants goes on real CI.
//!
//! Originally: three copies of `Iter2`'s exact logic, differing only in
//! the inline attribute on `next`: `Default` (no attribute, what's
//! currently shipped), `Never` (`#[inline(never)]`), `Always`
//! (`#[inline(always)]`, the reverted sandbox attempt). Real CI has now
//! answered the question this was built to ask, across two runs
//! (Archetype Core builds #11, #13) — see `docs/mid-ecs.md`'s
//! "diag_inline.rs" section for the full numbers, but the short version:
//! `Default` tracks `Never` almost exactly in both runs (0.2-1.1% apart
//! at N=100,000), not `Always` — so the compiler is **not** silently
//! auto-inlining `Iter2::next` by default on either toolchain. That
//! specific theory is ruled out. What's *not* settled: build #13 showed
//! `Always` measurably faster than `Default`/`Never` (~11-17% at
//! N≥10,000) — the opposite of the sandbox's own finding — while build
//! #11 showed all three within noise of each other. One run showing an
//! effect and one run showing nothing isn't a confirmed result either
//! way yet.
//!
//! **Extended this pass to `Iter1` too** — `query_static_single_component`
//! (backed by `Iter1`, real production code, *not* a diagnostic) has its
//! own real anomaly worth chasing directly: 10.693µs → 26.185µs at
//! N=10,000 between builds #11 and #13 (a ~2.45x regression, zero source
//! change), while its own already-existing unchecked diagnostic twin
//! (`query_static_diag_unchecked`, `Iter1Unchecked` in
//! `diag_query2_unchecked.rs`) moved the *opposite* direction in the
//! same span (10.593µs → 6.571µs, ~1.6x faster) — along with every other
//! benchmark in the same binary except this one. `Iter1` has never been
//! run through this Never/Always/Default comparison the way `Iter2` has
//! — until now. If `Iter1`'s own Default/Never/Always split lines up
//! with `Iter2`'s (Default≈Never, Always sometimes faster), that's a
//! second, independent confirmation pointing at the same lever. If it
//! doesn't, `query_static_single_component`'s build #13 anomaly has a
//! different cause than whatever this module is measuring, and that's
//! worth knowing too.
//!
//! Exposed via `World::query_static_diag_never`/`_always` and
//! `World::query2_static_diag_never`/`_always` — real public methods
//! only so `benches/archetype_core.rs` (a separate, external-to-the-crate
//! binary) can reach them; not meant to be used for anything else, and
//! should leave with this module.

use crate::archetype::{ArchetypeId, Archetypes};
use crate::component::ComponentId;
use crate::world::Entity;

pub(crate) struct Iter1Never<'a, T> {
    archetypes: &'a Archetypes,
    id: Option<ComponentId>,
    matched: std::vec::IntoIter<ArchetypeId>,
    entities: &'a [Entity],
    column: &'a [T],
    row: usize,
    len: usize,
}

pub(crate) struct Iter1Always<'a, T> {
    archetypes: &'a Archetypes,
    id: Option<ComponentId>,
    matched: std::vec::IntoIter<ArchetypeId>,
    entities: &'a [Entity],
    column: &'a [T],
    row: usize,
    len: usize,
}

macro_rules! impl_diag_iter1 {
    ($name:ident, $inline_attr:meta) => {
        impl<'a, T: 'static> Iterator for $name<'a, T> {
            type Item = (Entity, &'a T);

            #[$inline_attr]
            fn next(&mut self) -> Option<Self::Item> {
                loop {
                    if self.row < self.len {
                        let item = (self.entities[self.row], &self.column[self.row]);
                        self.row += 1;
                        return Some(item);
                    }
                    let id = self.id?;
                    let archetype_id = self.matched.next()?;
                    let (entities, column) =
                        self.archetypes.diag_entities_and_column::<T>(archetype_id, id);
                    self.len = entities.len().min(column.len());
                    self.entities = entities;
                    self.column = column;
                    self.row = 0;
                }
            }
        }
    };
}

impl_diag_iter1!(Iter1Never, inline(never));
impl_diag_iter1!(Iter1Always, inline(always));

pub(crate) struct Iter2Never<'a, A, B> {
    archetypes: &'a Archetypes,
    ids: Option<(ComponentId, ComponentId)>,
    matched: std::vec::IntoIter<ArchetypeId>,
    entities: &'a [Entity],
    a_col: &'a [A],
    b_col: &'a [B],
    row: usize,
    len: usize,
}

pub(crate) struct Iter2Always<'a, A, B> {
    archetypes: &'a Archetypes,
    ids: Option<(ComponentId, ComponentId)>,
    matched: std::vec::IntoIter<ArchetypeId>,
    entities: &'a [Entity],
    a_col: &'a [A],
    b_col: &'a [B],
    row: usize,
    len: usize,
}

macro_rules! impl_diag_iter2 {
    ($name:ident, $inline_attr:meta) => {
        impl<'a, A: 'static, B: 'static> Iterator for $name<'a, A, B> {
            type Item = (Entity, &'a A, &'a B);

            #[$inline_attr]
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
                    let (entities, a_col, b_col) = self
                        .archetypes
                        .diag_entities_and_columns::<A, B>(archetype_id, a_id, b_id);
                    self.len = entities.len().min(a_col.len()).min(b_col.len());
                    self.entities = entities;
                    self.a_col = a_col;
                    self.b_col = b_col;
                    self.row = 0;
                }
            }
        }
    };
}

impl_diag_iter2!(Iter2Never, inline(never));
impl_diag_iter2!(Iter2Always, inline(always));

impl Archetypes {
    pub(crate) fn iter_diag_never<T: 'static>(&self) -> Iter1Never<'_, T> {
        let (id, matched) = self.diag_matched_and_id::<T>();
        Iter1Never {
            archetypes: self,
            id,
            matched: matched.into_iter(),
            entities: &[],
            column: &[],
            row: 0,
            len: 0,
        }
    }

    pub(crate) fn iter_diag_always<T: 'static>(&self) -> Iter1Always<'_, T> {
        let (id, matched) = self.diag_matched_and_id::<T>();
        Iter1Always {
            archetypes: self,
            id,
            matched: matched.into_iter(),
            entities: &[],
            column: &[],
            row: 0,
            len: 0,
        }
    }

    pub(crate) fn iter2_diag_never<A: 'static, B: 'static>(&self) -> Iter2Never<'_, A, B> {
        let (ids, matched) = self.diag_matched_and_ids::<A, B>();
        Iter2Never {
            archetypes: self,
            ids,
            matched: matched.into_iter(),
            entities: &[],
            a_col: &[],
            b_col: &[],
            row: 0,
            len: 0,
        }
    }

    pub(crate) fn iter2_diag_always<A: 'static, B: 'static>(&self) -> Iter2Always<'_, A, B> {
        let (ids, matched) = self.diag_matched_and_ids::<A, B>();
        Iter2Always {
            archetypes: self,
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
