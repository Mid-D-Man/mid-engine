// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-ecs.md, section "Mutable queries"
// ============================================================================
//! The Archetype Core's bulk mutable iteration: the engine behind
//! `World::query_static_mut` and its two-component forms.
//!
//! A separate file from `iter.rs` on purpose. `Iter1`/`Iter2` and their
//! `Ref` forms have `next()` bodies held byte-for-byte fixed for the LTO
//! inline budget (see `iter.rs`'s header and `docs/mid-ecs.md`), so nothing
//! mutable is layered onto them.
//!
//! **Hand-written iterators, not `flat_map`.** The first version of this
//! file was `flat_map` over per-archetype `zip` chains: tidy, and 6 Ir/row
//! through `for_each`, but 63 Ir/row through a `for` loop (callgrind,
//! sandbox), because a `for` loop calls `next()` and `FlatMap::next` is the
//! slow path. These structs hold the current archetype's slice iterators
//! and implement `next()` directly, and override `fold` so `for_each`,
//! `sum` and friends keep the by-value `zip` fast path. See `docs/mid-ecs.md`,
//! "Mutable queries", for the numbers.
//!
//! **Zero `unsafe`.** Two safe pieces make that possible: `SparseSet::
//! iter_mut` hands out one `&mut Archetype` at a time, and inside one
//! archetype a `Table`'s three fields (`columns`, `ticks`, `entities`) are
//! disjoint, so destructuring `&mut Table` gives independent borrows of the
//! value column, its tick column and the entity list. Two columns of one
//! archetype come from `SparseSet::get_disjoint_mut`, which is
//! `split_at_mut` underneath.
//!
//! Archetype selection reuses `matched_filtered`, then keeps the matching
//! archetypes in the dense order `SparseSet::iter_mut` walks them in. That
//! is the same order `iter()` enumerates, so the result visits the same
//! entities in the same order as `Archetypes::iter_filtered`. Both are
//! asserted by tests, because the merge below depends on it.

use std::any::TypeId;
use std::iter::{Copied, Zip};
use std::slice::{Iter, IterMut};

use super::{Archetype, ArchetypeId, Archetypes, Column, Table};
use crate::component::ComponentId;
use crate::filter::QueryFilter;
use crate::tick::{ComponentTicks, Mut, Tick};
use crate::world::Entity;

/// The matching archetypes, mutably, in dense order. `matched` must be in
/// dense order (as `matched_filtered` produces it); each archetype is
/// taken when its id is the next one wanted.
fn select<'a>(
    archetypes: &'a mut mid_collections::SparseSet<ArchetypeId, Archetype>,
    matched: Vec<ArchetypeId>,
) -> impl Iterator<Item = &'a mut Archetype> + 'a {
    let mut want = matched.into_iter().peekable();
    archetypes.iter_mut().filter_map(move |(id, archetype)| {
        if want.peek() == Some(&id) {
            want.next();
            Some(archetype)
        } else {
            None
        }
    })
}

fn values_mut<'a, T: 'static>(column: Option<&'a mut Box<dyn Column>>) -> &'a mut [T] {
    match column {
        Some(column) => column
            .as_any_mut()
            .downcast_mut::<Vec<T>>()
            .expect("column type must match component_id's T")
            .as_mut_slice(),
        None => &mut [],
    }
}

fn values_ref<'a, T: 'static>(column: Option<&'a mut Box<dyn Column>>) -> &'a [T] {
    match column {
        Some(column) => column
            .as_any()
            .downcast_ref::<Vec<T>>()
            .expect("column type must match component_id's T")
            .as_slice(),
        None => &[],
    }
}

fn ticks_mut(ticks: Option<&mut Vec<ComponentTicks>>) -> &mut [ComponentTicks] {
    match ticks {
        Some(ticks) => ticks.as_mut_slice(),
        None => &mut [],
    }
}

// ── Per-archetype row cursors ───────────────────────────────────────
//
// One nested `Zip` of slice iterators per archetype, not one cursor field
// per column. With every side a slice iterator, `Zip` specializes to a
// single shared row index, so the state a `for` loop carries between
// calls is one counter however many columns the query has. (The first
// hand-written version kept one cursor per column: 19 Ir/row for one
// column, 41 for four, because `advance(&mut self)` being out of line
// keeps the struct in memory and every cursor is loaded and stored each
// row. Measured; see `docs/mid-ecs.md`, "Mutable queries".) A column
// shorter than the entity list truncates the zip, so a violated invariant
// skips rows instead of misaligning them.

type Rows1<'a, T> = Zip<Zip<Copied<Iter<'a, Entity>>, IterMut<'a, T>>, IterMut<'a, ComponentTicks>>;

type Rows2<'a, A, B> = Zip<
    Zip<
        Zip<Zip<Copied<Iter<'a, Entity>>, IterMut<'a, A>>, IterMut<'a, B>>,
        IterMut<'a, ComponentTicks>,
    >,
    IterMut<'a, ComponentTicks>,
>;

type Rows2Ref<'a, A, B> = Zip<
    Zip<Zip<Copied<Iter<'a, Entity>>, IterMut<'a, A>>, Iter<'a, B>>,
    IterMut<'a, ComponentTicks>,
>;

fn no_entities<'a>() -> Copied<Iter<'a, Entity>> {
    let none: &'a [Entity] = &[];
    none.iter().copied()
}

fn no_values<'a, T>() -> IterMut<'a, T> {
    let none: &'a mut [T] = &mut [];
    none.iter_mut()
}

fn no_ticks<'a>() -> IterMut<'a, ComponentTicks> {
    no_values()
}

fn no_shared<'a, T>() -> Iter<'a, T> {
    let none: &'a [T] = &[];
    none.iter()
}

fn empty_rows1<'a, T>() -> Rows1<'a, T> {
    no_entities().zip(no_values()).zip(no_ticks())
}

fn empty_rows2<'a, A, B>() -> Rows2<'a, A, B> {
    no_entities()
        .zip(no_values())
        .zip(no_values())
        .zip(no_ticks())
        .zip(no_ticks())
}

fn empty_rows2_ref<'a, A, B>() -> Rows2Ref<'a, A, B> {
    no_entities()
        .zip(no_values())
        .zip(no_shared())
        .zip(no_ticks())
}

// ── One mutable component ───────────────────────────────────────────

/// `(Entity, Mut<T>)`. `I` is the matching-archetype walk from `select`;
/// it is a type parameter only because that walk's type cannot be named.
pub(crate) struct IterMut1<'a, T, I> {
    archetypes: I,
    id: Option<ComponentId>,
    this_run: Tick,
    rows: Rows1<'a, T>,
}

impl<'a, T: 'static, I: Iterator<Item = &'a mut Archetype>> IterMut1<'a, T, I> {
    fn load(&mut self, archetype: &'a mut Archetype) {
        let id = self
            .id
            .expect("an archetype is only ever yielded when the component id exists");
        let Table {
            columns,
            ticks,
            entities,
        } = &mut archetype.table;
        self.rows = entities
            .iter()
            .copied()
            .zip(values_mut::<T>(columns.get_mut(id)).iter_mut())
            .zip(ticks_mut(ticks.get_mut(id)).iter_mut());
    }

    /// Moves on to the next matching archetype that has a row. Kept out of
    /// line and cold so `next()` stays small enough to inline into the
    /// caller's loop.
    #[cold]
    #[inline(never)]
    fn advance(&mut self) -> Option<(Entity, Mut<'a, T>)> {
        loop {
            let archetype = self.archetypes.next()?;
            self.load(archetype);
            if let Some(((entity, value), ticks)) = self.rows.next() {
                return Some((entity, Mut::new(value, ticks, self.this_run)));
            }
        }
    }
}

impl<'a, T: 'static, I: Iterator<Item = &'a mut Archetype>> Iterator for IterMut1<'a, T, I> {
    type Item = (Entity, Mut<'a, T>);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if let Some(((entity, value), ticks)) = self.rows.next() {
            return Some((entity, Mut::new(value, ticks, self.this_run)));
        }
        self.advance()
    }

    fn fold<B, F: FnMut(B, Self::Item) -> B>(mut self, init: B, mut f: F) -> B {
        let this_run = self.this_run;
        let mut acc = init;
        loop {
            let rows = std::mem::replace(&mut self.rows, empty_rows1());
            acc = rows.fold(acc, |acc, ((entity, value), ticks)| {
                f(acc, (entity, Mut::new(value, ticks, this_run)))
            });
            match self.archetypes.next() {
                Some(archetype) => self.load(archetype),
                None => return acc,
            }
        }
    }
}

// ── Two mutable components ──────────────────────────────────────────

/// `(Entity, Mut<A>, Mut<B>)`.
pub(crate) struct IterMut2<'a, A, B, I> {
    archetypes: I,
    ids: Option<(ComponentId, ComponentId)>,
    this_run: Tick,
    rows: Rows2<'a, A, B>,
}

impl<'a, A: 'static, B: 'static, I: Iterator<Item = &'a mut Archetype>> IterMut2<'a, A, B, I> {
    fn load(&mut self, archetype: &'a mut Archetype) {
        let (a_id, b_id) = self
            .ids
            .expect("an archetype is only ever yielded when both component ids exist");
        let Table {
            columns,
            ticks,
            entities,
        } = &mut archetype.table;
        let (col_a, col_b) = columns.get_disjoint_mut(a_id, b_id);
        let (ticks_a, ticks_b) = ticks.get_disjoint_mut(a_id, b_id);
        self.rows = entities
            .iter()
            .copied()
            .zip(values_mut::<A>(col_a).iter_mut())
            .zip(values_mut::<B>(col_b).iter_mut())
            .zip(ticks_mut(ticks_a).iter_mut())
            .zip(ticks_mut(ticks_b).iter_mut());
    }

    /// See [`IterMut1::advance`].
    #[cold]
    #[inline(never)]
    fn advance(&mut self) -> Option<(Entity, Mut<'a, A>, Mut<'a, B>)> {
        loop {
            let archetype = self.archetypes.next()?;
            self.load(archetype);
            if let Some(((((entity, a), b), ta), tb)) = self.rows.next() {
                return Some((
                    entity,
                    Mut::new(a, ta, self.this_run),
                    Mut::new(b, tb, self.this_run),
                ));
            }
        }
    }
}

impl<'a, A: 'static, B: 'static, I: Iterator<Item = &'a mut Archetype>> Iterator
    for IterMut2<'a, A, B, I>
{
    type Item = (Entity, Mut<'a, A>, Mut<'a, B>);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if let Some(((((entity, a), b), ta), tb)) = self.rows.next() {
            return Some((
                entity,
                Mut::new(a, ta, self.this_run),
                Mut::new(b, tb, self.this_run),
            ));
        }
        self.advance()
    }

    fn fold<Acc, F: FnMut(Acc, Self::Item) -> Acc>(mut self, init: Acc, mut f: F) -> Acc {
        let this_run = self.this_run;
        let mut acc = init;
        loop {
            let rows = std::mem::replace(&mut self.rows, empty_rows2());
            acc = rows.fold(acc, |acc, ((((entity, a), b), ta), tb)| {
                f(
                    acc,
                    (entity, Mut::new(a, ta, this_run), Mut::new(b, tb, this_run)),
                )
            });
            match self.archetypes.next() {
                Some(archetype) => self.load(archetype),
                None => return acc,
            }
        }
    }
}

// ── One mutable, one shared ─────────────────────────────────────────

/// `(Entity, Mut<A>, &B)`.
pub(crate) struct IterMut2Ref<'a, A, B, I> {
    archetypes: I,
    ids: Option<(ComponentId, ComponentId)>,
    this_run: Tick,
    rows: Rows2Ref<'a, A, B>,
}

impl<'a, A: 'static, B: 'static, I: Iterator<Item = &'a mut Archetype>> IterMut2Ref<'a, A, B, I> {
    fn load(&mut self, archetype: &'a mut Archetype) {
        let (a_id, b_id) = self
            .ids
            .expect("an archetype is only ever yielded when both component ids exist");
        let Table {
            columns,
            ticks,
            entities,
        } = &mut archetype.table;
        let (col_a, col_b) = columns.get_disjoint_mut(a_id, b_id);
        self.rows = entities
            .iter()
            .copied()
            .zip(values_mut::<A>(col_a).iter_mut())
            .zip(values_ref::<B>(col_b).iter())
            .zip(ticks_mut(ticks.get_mut(a_id)).iter_mut());
    }

    /// See [`IterMut1::advance`].
    #[cold]
    #[inline(never)]
    fn advance(&mut self) -> Option<(Entity, Mut<'a, A>, &'a B)> {
        loop {
            let archetype = self.archetypes.next()?;
            self.load(archetype);
            if let Some((((entity, a), b), ta)) = self.rows.next() {
                return Some((entity, Mut::new(a, ta, self.this_run), b));
            }
        }
    }
}

impl<'a, A: 'static, B: 'static, I: Iterator<Item = &'a mut Archetype>> Iterator
    for IterMut2Ref<'a, A, B, I>
{
    type Item = (Entity, Mut<'a, A>, &'a B);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if let Some((((entity, a), b), ta)) = self.rows.next() {
            return Some((entity, Mut::new(a, ta, self.this_run), b));
        }
        self.advance()
    }

    fn fold<Acc, F: FnMut(Acc, Self::Item) -> Acc>(mut self, init: Acc, mut f: F) -> Acc {
        let this_run = self.this_run;
        let mut acc = init;
        loop {
            let rows = std::mem::replace(&mut self.rows, empty_rows2_ref());
            acc = rows.fold(acc, |acc, (((entity, a), b), ta)| {
                f(acc, (entity, Mut::new(a, ta, this_run), b))
            });
            match self.archetypes.next() {
                Some(archetype) => self.load(archetype),
                None => return acc,
            }
        }
    }
}

impl Archetypes {
    /// `(Entity, Mut<T>)` over every archetype satisfying `F` that holds
    /// `T`. A zero-row archetype whose signature has `T` but no column yet
    /// (the intermediates `insert_bundle` leaves) yields nothing, like
    /// every other iterator here.
    pub(crate) fn iter_mut_filtered<T: 'static, F: QueryFilter>(
        &mut self,
        this_run: Tick,
    ) -> impl Iterator<Item = (Entity, Mut<'_, T>)> + '_ {
        let id = self.existing_component_id::<T>();
        let matched = match id {
            Some(id) => self.matched_filtered::<F>(&[id]),
            None => Vec::new(),
        };
        IterMut1 {
            archetypes: select(&mut self.archetypes, matched),
            id,
            this_run,
            rows: empty_rows1(),
        }
    }

    /// `(Entity, Mut<A>, Mut<B>)`: both mutable. Panics if `A` and `B` are
    /// the same type: one value cannot be borrowed mutably twice.
    pub(crate) fn iter2_mut_filtered<A: 'static, B: 'static, F: QueryFilter>(
        &mut self,
        this_run: Tick,
    ) -> impl Iterator<Item = (Entity, Mut<'_, A>, Mut<'_, B>)> + '_ {
        assert_ne!(
            TypeId::of::<A>(),
            TypeId::of::<B>(),
            "a two-component mutable query needs two distinct component types"
        );
        let ids = self
            .existing_component_id::<A>()
            .zip(self.existing_component_id::<B>());
        let matched = match ids {
            Some((a, b)) => self.matched_filtered::<F>(&[a, b]),
            None => Vec::new(),
        };
        IterMut2 {
            archetypes: select(&mut self.archetypes, matched),
            ids,
            this_run,
            rows: empty_rows2(),
        }
    }

    /// `(Entity, Mut<A>, &B)`: `A` mutable, `B` shared, the
    /// `(&mut Position, &Velocity)` shape. Panics if `A` and `B` are the
    /// same type.
    pub(crate) fn iter2_mut_ref_filtered<A: 'static, B: 'static, F: QueryFilter>(
        &mut self,
        this_run: Tick,
    ) -> impl Iterator<Item = (Entity, Mut<'_, A>, &B)> + '_ {
        assert_ne!(
            TypeId::of::<A>(),
            TypeId::of::<B>(),
            "a two-component mutable query needs two distinct component types"
        );
        let ids = self
            .existing_component_id::<A>()
            .zip(self.existing_component_id::<B>());
        let matched = match ids {
            Some((a, b)) => self.matched_filtered::<F>(&[a, b]),
            None => Vec::new(),
        };
        IterMut2Ref {
            archetypes: select(&mut self.archetypes, matched),
            ids,
            this_run,
            rows: empty_rows2_ref(),
        }
    }
}
