// NOTICE: See docs/mid-ecs.md, section "diag_query2_unchecked.rs", for
// why this module exists, the real numbers behind it, and its current
// status. TEMPORARY — delete once the investigation concludes.

use crate::archetype::{ArchetypeId, Archetypes};
use crate::component::ComponentId;
use crate::world::Entity;

pub(crate) struct Iter1Unchecked<'a, T> {
    archetypes: &'a Archetypes,
    id: Option<ComponentId>,
    matched: std::vec::IntoIter<ArchetypeId>,
    entities: &'a [Entity],
    column: &'a [T],
    row: usize,
    len: usize,
}

impl<'a, T: 'static> Iterator for Iter1Unchecked<'a, T> {
    type Item = (Entity, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.row < self.len {
                // SAFETY: `len` is set to `entities.len().min(column.len())`
                // on every archetype advance below, and this branch only
                // runs while `row < len`, so `row` is in bounds for both
                // slices.
                let item = unsafe {
                    (
                        *self.entities.get_unchecked(self.row),
                        self.column.get_unchecked(self.row),
                    )
                };
                self.row += 1;
                return Some(item);
            }
            let id = self.id?;
            let archetype_id = self.matched.next()?;
            let (entities, column) = self
                .archetypes
                .diag_entities_and_column::<T>(archetype_id, id);
            self.len = entities.len().min(column.len());
            self.entities = entities;
            self.column = column;
            self.row = 0;
        }
    }
}

pub(crate) struct Iter2Unchecked<'a, A, B> {
    archetypes: &'a Archetypes,
    ids: Option<(ComponentId, ComponentId)>,
    matched: std::vec::IntoIter<ArchetypeId>,
    entities: &'a [Entity],
    a_col: &'a [A],
    b_col: &'a [B],
    row: usize,
    len: usize,
}

impl<'a, A: 'static, B: 'static> Iterator for Iter2Unchecked<'a, A, B> {
    type Item = (Entity, &'a A, &'a B);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.row < self.len {
                // SAFETY: same invariant as Iter1Unchecked::next, extended
                // to three slices — `len` is the min of all three lengths
                // on every archetype advance below.
                let item = unsafe {
                    (
                        *self.entities.get_unchecked(self.row),
                        self.a_col.get_unchecked(self.row),
                        self.b_col.get_unchecked(self.row),
                    )
                };
                self.row += 1;
                return Some(item);
            }
            let (a_id, b_id) = self.ids?;
            let archetype_id = self.matched.next()?;
            let (entities, a_col, b_col) =
                self.archetypes
                    .diag_entities_and_columns::<A, B>(archetype_id, a_id, b_id);
            self.len = entities.len().min(a_col.len()).min(b_col.len());
            self.entities = entities;
            self.a_col = a_col;
            self.b_col = b_col;
            self.row = 0;
        }
    }
}

/// Reads both `a_col` and `b_col` per item like the real `Iter2`, but
/// combines them into one derived value before returning, so `Item` is
/// a 2-tuple instead of a 3-tuple. Isolates the tuple/Item-shape
/// question from the two-slice-fields question below.
pub(crate) struct Iter2TwoTupleItem<'a, A, B> {
    archetypes: &'a Archetypes,
    ids: Option<(ComponentId, ComponentId)>,
    matched: std::vec::IntoIter<ArchetypeId>,
    entities: &'a [Entity],
    a_col: &'a [A],
    b_col: &'a [B],
    row: usize,
    len: usize,
    combine: fn(&A, &B) -> A,
}

impl<'a, A: 'static + Clone, B: 'static> Iterator for Iter2TwoTupleItem<'a, A, B> {
    type Item = (Entity, A);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.row < self.len {
                let item = (
                    self.entities[self.row],
                    (self.combine)(&self.a_col[self.row], &self.b_col[self.row]),
                );
                self.row += 1;
                return Some(item);
            }
            let (a_id, b_id) = self.ids?;
            let archetype_id = self.matched.next()?;
            let (entities, a_col, b_col) =
                self.archetypes
                    .diag_entities_and_columns::<A, B>(archetype_id, a_id, b_id);
            self.len = entities.len().min(a_col.len()).min(b_col.len());
            self.entities = entities;
            self.a_col = a_col;
            self.b_col = b_col;
            self.row = 0;
        }
    }
}

/// Same struct shape as the real `Iter2` (two slice fields, both kept
/// current on every archetype advance), but the hot loop only ever
/// reads `entities`/`a_col` — `b_col` is tracked for `len` only, never
/// indexed in the per-item path. Isolates whether merely carrying the
/// extra field matters, independent of whether it's read.
pub(crate) struct Iter2UnusedBCol<'a, A, B> {
    archetypes: &'a Archetypes,
    ids: Option<(ComponentId, ComponentId)>,
    matched: std::vec::IntoIter<ArchetypeId>,
    entities: &'a [Entity],
    a_col: &'a [A],
    b_col: &'a [B],
    row: usize,
    len: usize,
}

impl<'a, A: 'static, B: 'static> Iterator for Iter2UnusedBCol<'a, A, B> {
    type Item = (Entity, &'a A);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.row < self.len {
                let item = (self.entities[self.row], &self.a_col[self.row]);
                self.row += 1;
                return Some(item);
            }
            let (a_id, b_id) = self.ids?;
            let archetype_id = self.matched.next()?;
            let (entities, a_col, b_col) =
                self.archetypes
                    .diag_entities_and_columns::<A, B>(archetype_id, a_id, b_id);
            self.len = entities.len().min(a_col.len()).min(b_col.len());
            self.entities = entities;
            self.a_col = a_col;
            self.b_col = b_col;
            self.row = 0;
        }
    }
}

impl Archetypes {
    pub(crate) fn iter_diag_unchecked<T: 'static>(&self) -> Iter1Unchecked<'_, T> {
        let (id, matched) = self.diag_matched_and_id::<T>();
        Iter1Unchecked {
            archetypes: self,
            id,
            matched: matched.into_iter(),
            entities: &[],
            column: &[],
            row: 0,
            len: 0,
        }
    }

    pub(crate) fn iter2_diag_unchecked<A: 'static, B: 'static>(&self) -> Iter2Unchecked<'_, A, B> {
        let (ids, matched) = self.diag_matched_and_ids::<A, B>();
        Iter2Unchecked {
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

    pub(crate) fn iter2_diag_two_tuple_item<A: 'static + Clone, B: 'static>(
        &self,
        combine: fn(&A, &B) -> A,
    ) -> Iter2TwoTupleItem<'_, A, B> {
        let (ids, matched) = self.diag_matched_and_ids::<A, B>();
        Iter2TwoTupleItem {
            archetypes: self,
            ids,
            matched: matched.into_iter(),
            entities: &[],
            a_col: &[],
            b_col: &[],
            row: 0,
            len: 0,
            combine,
        }
    }

    pub(crate) fn iter2_diag_unused_b_col<A: 'static, B: 'static>(
        &self,
    ) -> Iter2UnusedBCol<'_, A, B> {
        let (ids, matched) = self.diag_matched_and_ids::<A, B>();
        Iter2UnusedBCol {
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
