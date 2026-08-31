//! TEMPORARY DIAGNOSTIC MODULE — delete once the investigation this
//! exists for concludes, not meant to become permanent API surface.
//!
//! Real CI (rustc 1.98.0) shows `query2_static_two_components` (backed
//! by `archetype.rs`'s `Iter2`) running ~4x slower than `bevy_ecs` on
//! an equivalent workload, in both `benches/ecs-vs-bevy-ecs` and this
//! crate's own `archetype_core.rs` — while this sandbox's rustc 1.91.1
//! measures the same code within noise of `bevy_ecs`. Tried adding
//! `#[inline(always)]` to `Iter2::next` (reasoning: `bevy_ecs`'s own
//! `QueryIterationCursor::next` has it explicitly) — that made things
//! measurably ~4x *worse* on this sandbox's rustc 1.91.1, the opposite
//! of the intended fix, and was reverted (see `Iter2::next`'s own doc
//! comment in `archetype.rs`). Since this sandbox can't reproduce the
//! real-CI regression to begin with, guessing again from here isn't
//! useful — this module exists to gather real data *from* the actual
//! affected toolchain instead.
//!
//! Three copies of the exact same `Iter2` logic, differing only in the
//! inline attribute on `next`: `Default` (no attribute, what's
//! currently shipped), `Never` (`#[inline(never)]`, forces a real
//! function call every item), `Always` (`#[inline(always)]`, the
//! reverted attempt). Whichever of `Never`/`Always` ends up closest to
//! `Default`'s own real-CI number tells us what rustc 1.98.0 is
//! already doing by default — if `Never` matches `Default`, the
//! compiler isn't inlining this by default on that toolchain at all
//! (a real, fixable-but-differently finding); if `Always` matches
//! `Default`, it already is, and the regression's cause is something
//! else entirely, not inlining.
//!
//! Exposed via `World::query2_static_diag_never`/`_always` — real
//! public methods only so `benches/archetype_core.rs` (a separate,
//! external-to-the-crate binary) can reach them; not meant to be used
//! for anything else, and should leave with this module.

use crate::archetype::{ArchetypeId, Archetypes};
use crate::component::ComponentId;
use crate::world::Entity;

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
