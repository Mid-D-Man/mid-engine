//! Archetype query iterators.
//!
//! Split out of `archetype.rs` as a **child module** rather than a
//! sibling, deliberately: `Table::columns`, `Table::entities`, the
//! `Column` trait and the `Archetype` struct are all private to
//! `archetype`, and a child module can see its parent's private items
//! while a sibling cannot. No visibility had to be widened to move this
//! code here, which matters -- `Column` staying module-private is what
//! keeps the `Vec<T>`-backed column representation an implementation
//! detail rather than API surface.
//!
//! # Why these iterators look the way they do
//!
//! Two prior rewrites are already recorded in `docs/mid-ecs.md` and
//! should not be re-litigated here:
//!
//! 1. The original `flat_map`/`filter`/`zip` combinator chains measured
//!    145-152µs at N=10,000. Replacing them with flat hand-written state
//!    machines took that to ~7-9µs. That finding stands and this module
//!    keeps the flat shape.
//! 2. `unsafe`/`get_unchecked` alone, `#[inline(always)]` alone, raw
//!    pointer columns, composed per-component fetches, and splitting the
//!    cold archetype-advance into its own `#[inline(never)]` function
//!    were each tried against the remaining ~4x two-column gap and each
//!    came back negative. Those are closed questions.
//!
//! **What this module changes is the one thing none of those touched:
//! the size of what `next()` returns.**
//!
//! `Entity` is 8 bytes (a `mid-collections` `GenerationalIndex`: two
//! `u32`s). Every reference is 8 bytes. So:
//!
//! - `Iter1::Item` = `(Entity, &T)` -> `Option<Item>` is 16 bytes, which
//!   SysV AMD64 returns in the RAX:RDX register pair.
//! - `Iter2::Item` = `(Entity, &A, &B)` -> `Option<Item>` is 24 bytes.
//!   SysV AMD64 §3.2.3 classifies any aggregate over two eightbytes as
//!   MEMORY: the caller allocates stack space, the callee stores the
//!   result into it, the caller loads it back. Every single item.
//!
//! That threshold sits exactly between `Iter1` and `Iter2` and nowhere
//! else, which is why `query_static_single_component` is at parity with
//! `bevy_ecs` (9.42µs vs 9.35µs) while `dense_query_iteration` is 3.99x.
//! `bevy_ecs` does not cross it: its `Query<(&A, &B)>` yields `(&A, &B)`
//! -- 16 bytes -- because `Entity` there is opt-in query data, not a
//! mandatory first tuple element. The gap is an API-shape difference,
//! not a storage-architecture one.
//!
//! **This is a hypothesis with a cheap decisive test, not a conclusion.**
//! Run `benches/abi-return-size` -- which links against nothing and so
//! cannot be tipped by this crate's compilation-unit layout the way
//! `archetype_core.rs`'s own controls were in builds #15-#18 -- before
//! treating any of the below as the explanation. Its `ret16_ref_ref`
//! group is the control that separates "item size" from "reads a second
//! column", which every diagnostic so far has had confounded.
//!
//! # The two fixes, and why both ship
//!
//! **Narrowed items** (`Iter1Ref`, `Iter2Ref`): drop `Entity` from the
//! tuple. `Iter2Ref::Item` is `(&A, &B)`, 16 bytes, register-returned.
//! This is the only fix that helps a plain `for` loop, because a `for`
//! loop can only ever drive an iterator through repeated `next()`.
//!
//! **A `fold` override** (on all four): `for_each`, `sum`, `fold` and
//! `collect` do *not* go through `next()` -- they go through `fold`,
//! where the per-archetype run is one flat contiguous loop and no
//! `Option<Item>` crosses a call boundary at all. This is what
//! `bevy_ecs` does (`QueryIter::fold` delegating to
//! `fold_over_table_range`, `query/iter.rs`, read directly from the
//! `Mid-D-Man/bevy` checkout). It keeps `Entity` available at full speed
//! for anything willing to write `.for_each(..)` instead of `for ..`.
//!
//! `try_fold` is deliberately *not* overridden: `std::ops::Try` is
//! unstable to implement against, so `find`/`any`/`position` still route
//! through `next()`. They are not hot paths here. Revisit only if a real
//! profile says otherwise.
//!
//! # Safety
//!
//! Every `get_unchecked` below is guarded by the same single invariant:
//! `len` is recomputed on every archetype advance as the minimum of all
//! participating slice lengths, and every access is gated on
//! `row < len`. The `advance` functions are the only writers of `len`,
//! and they set `entities`/columns in the same statement sequence, so
//! the three (or two) slices and `len` can never be out of step. This
//! replaces the previous bounds-checked indexing; per `docs/mid-ecs.md`
//! that swap alone was measured to be worth ~nothing, and it is kept
//! here only because the `fold` inner loops want the bounds check gone
//! to vectorise, not because it was ever the gap.

use super::{Archetype, ArchetypeId, Archetypes, Column};
use crate::component::ComponentId;
use crate::world::Entity;

/// Resolves one archetype's column for `id` as a typed slice, or an
/// empty slice if no column exists.
///
/// "No column" is not a broken invariant and must not panic:
/// `insert_bundle` chains one `edge_for_insert` per bundle element, so a
/// two-element bundle creates an *intermediate* archetype that is
/// correctly registered in `component_ids` but that no entity is ever
/// moved into. Such an archetype has zero rows, so "no column" and "an
/// empty column" are observationally identical from here -- both
/// contribute zero items. This was a real panic once (`archetypes_with
/// guarantees component_id is in this archetype's own signature`), hit
/// by exactly the bulk `insert_bundle` calls the bench harness makes.
///
/// Called once per archetype, never per item -- it is the expensive part
/// (a `SparseSet` probe, a `dyn Column` vtable dispatch through
/// `as_any`, and a `TypeId` comparison in `downcast_ref`) and lives
/// entirely on the cold path below.
#[inline]
fn column_slice<'a, T: 'static>(archetype: &'a Archetype, id: ComponentId) -> &'a [T] {
    match archetype.table.columns.get(id) {
        Some(column) => column
            .as_any()
            .downcast_ref::<Vec<T>>()
            .expect("column type must match component_id's T")
            .as_slice(),
        None => &[],
    }
}

/// Looks up an archetype that the precomputed `matched` list promised
/// exists. `matched` is built inside `Archetypes::iter`/`iter2` from
/// `archetypes_with`, and nothing can remove an archetype while the
/// returned iterator borrows `&self`, so this is infallible in practice.
#[inline]
fn archetype_of(archetypes: &Archetypes, id: ArchetypeId) -> &Archetype {
    archetypes
        .archetypes
        .get(id)
        .expect("a query's precomputed matched list only ever contains real, currently-existing archetype ids")
}

// =====================================================================
// One component
// =====================================================================

/// Yields `(Entity, &T)`. `Option<Item>` is 16 bytes -> register return.
/// This one was already at the `raw_slice_ceiling` floor and at parity
/// with `bevy_ecs`; it is moved here unchanged in behaviour, gaining
/// only the `fold` override and the cold-path split.
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

    /// Moves to the next matching archetype that actually has rows.
    /// Returns `false` once the matched list is exhausted.
    ///
    /// `#[cold]` + `#[inline(never)]`: this runs once per archetype, the
    /// hot path runs once per entity. Note that splitting the cold path
    /// out was already tested on its own (`Iter2ColdSplit`,
    /// `docs/mid-ecs.md` builds #17/#18) and came back negative -- it is
    /// kept here because it is the right shape, not because it is
    /// expected to move the number by itself.
    #[cold]
    #[inline(never)]
    fn advance(&mut self) -> bool {
        let Some(id) = self.id else {
            return false;
        };
        loop {
            let Some(archetype_id) = self.matched.next() else {
                return false;
            };
            let archetype = archetype_of(self.archetypes, archetype_id);
            let entities: &[Entity] = &archetype.table.entities;
            let column: &[T] = column_slice(archetype, id);
            let len = entities.len().min(column.len());
            if len == 0 {
                continue;
            }
            self.entities = entities;
            self.column = column;
            self.len = len;
            self.row = 0;
            return true;
        }
    }
}

impl<'a, T: 'static> Iterator for Iter1<'a, T> {
    type Item = (Entity, &'a T);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.row < self.len {
                let row = self.row;
                self.row = row + 1;
                // SAFETY: `row < len`, and `len` is the min of both slice
                // lengths as of the last `advance`.
                return Some(unsafe {
                    (
                        *self.entities.get_unchecked(row),
                        self.column.get_unchecked(row),
                    )
                });
            }
            if !self.advance() {
                return None;
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len - self.row, None)
    }

    #[inline]
    fn fold<Acc, F>(mut self, init: Acc, mut f: F) -> Acc
    where
        F: FnMut(Acc, Self::Item) -> Acc,
    {
        let mut acc = init;
        loop {
            let (entities, column) = (self.entities, self.column);
            for row in self.row..self.len {
                // SAFETY: `row < len`, as above.
                let item =
                    unsafe { (*entities.get_unchecked(row), column.get_unchecked(row)) };
                acc = f(acc, item);
            }
            self.row = self.len;
            if !self.advance() {
                return acc;
            }
        }
    }
}

/// Yields `&T`. `Option<Item>` is 8 bytes. The counterpart to
/// `bevy_ecs`'s `Query<&T>`, which yields exactly this.
pub(crate) struct Iter1Ref<'a, T> {
    inner: Iter1<'a, T>,
}

impl<'a, T: 'static> Iter1Ref<'a, T> {
    #[inline]
    pub(crate) fn new(
        archetypes: &'a Archetypes,
        id: Option<ComponentId>,
        matched: Vec<ArchetypeId>,
    ) -> Self {
        Self {
            inner: Iter1::new(archetypes, id, matched),
        }
    }
}

impl<'a, T: 'static> Iterator for Iter1Ref<'a, T> {
    type Item = &'a T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.inner.row < self.inner.len {
                let row = self.inner.row;
                self.inner.row = row + 1;
                // SAFETY: `row < len`, as in `Iter1::next`.
                return Some(unsafe { self.inner.column.get_unchecked(row) });
            }
            if !self.inner.advance() {
                return None;
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.inner.len - self.inner.row, None)
    }

    #[inline]
    fn fold<Acc, F>(mut self, init: Acc, mut f: F) -> Acc
    where
        F: FnMut(Acc, Self::Item) -> Acc,
    {
        let mut acc = init;
        loop {
            let column = self.inner.column;
            for row in self.inner.row..self.inner.len {
                // SAFETY: `row < len`, as above.
                acc = f(acc, unsafe { column.get_unchecked(row) });
            }
            self.inner.row = self.inner.len;
            if !self.inner.advance() {
                return acc;
            }
        }
    }
}

// =====================================================================
// Two components
// =====================================================================

/// Yields `(Entity, &A, &B)`. `Option<Item>` is **24 bytes** -- MEMORY
/// class under SysV AMD64, returned through a hidden pointer with a
/// stack round-trip per item.
///
/// Kept, because dropping `Entity` from a two-component query is an API
/// break and because `Entity` is genuinely needed by plenty of callers.
/// But `next()` here is the shape currently suspected of the whole 4x
/// gap, so: **prefer `Iter2Ref` when the entity is not needed, and
/// prefer `.for_each(..)`/`.fold(..)` over `for ..` when it is.** The
/// `fold` override below sidesteps the return ABI entirely; the `for`
/// loop cannot.
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

    #[cold]
    #[inline(never)]
    fn advance(&mut self) -> bool {
        let Some((a_id, b_id)) = self.ids else {
            return false;
        };
        loop {
            let Some(archetype_id) = self.matched.next() else {
                return false;
            };
            let archetype = archetype_of(self.archetypes, archetype_id);
            let entities: &[Entity] = &archetype.table.entities;
            let a_col: &[A] = column_slice(archetype, a_id);
            let b_col: &[B] = column_slice(archetype, b_id);
            let len = entities.len().min(a_col.len()).min(b_col.len());
            if len == 0 {
                continue;
            }
            self.entities = entities;
            self.a_col = a_col;
            self.b_col = b_col;
            self.len = len;
            self.row = 0;
            return true;
        }
    }
}

impl<'a, A: 'static, B: 'static> Iterator for Iter2<'a, A, B> {
    type Item = (Entity, &'a A, &'a B);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.row < self.len {
                let row = self.row;
                self.row = row + 1;
                // SAFETY: `row < len`, and `len` is the min of all three
                // slice lengths as of the last `advance`.
                return Some(unsafe {
                    (
                        *self.entities.get_unchecked(row),
                        self.a_col.get_unchecked(row),
                        self.b_col.get_unchecked(row),
                    )
                });
            }
            if !self.advance() {
                return None;
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len - self.row, None)
    }

    /// The escape hatch from the 24-byte return. One flat contiguous
    /// loop per archetype, no `Option`, nothing crossing a call
    /// boundary. Structurally the same as `bevy_ecs`'s
    /// `fold_over_table_range`.
    #[inline]
    fn fold<Acc, F>(mut self, init: Acc, mut f: F) -> Acc
    where
        F: FnMut(Acc, Self::Item) -> Acc,
    {
        let mut acc = init;
        loop {
            let (entities, a_col, b_col) = (self.entities, self.a_col, self.b_col);
            for row in self.row..self.len {
                // SAFETY: `row < len`, as above.
                let item = unsafe {
                    (
                        *entities.get_unchecked(row),
                        a_col.get_unchecked(row),
                        b_col.get_unchecked(row),
                    )
                };
                acc = f(acc, item);
            }
            self.row = self.len;
            if !self.advance() {
                return acc;
            }
        }
    }
}

/// Yields `(&A, &B)`. `Option<Item>` is **16 bytes** -> RAX:RDX.
///
/// The direct counterpart to `bevy_ecs`'s `Query<(&A, &B)>`, and the
/// only two-component shape that is apples-to-apples against it. If the
/// ABI hypothesis holds, this is the variant that closes
/// `dense_query_iteration`'s 3.99x on its own, without touching storage,
/// `unsafe`-ness anywhere else, or any inline attribute.
pub(crate) struct Iter2Ref<'a, A, B> {
    inner: Iter2<'a, A, B>,
}

impl<'a, A: 'static, B: 'static> Iter2Ref<'a, A, B> {
    #[inline]
    pub(crate) fn new(
        archetypes: &'a Archetypes,
        ids: Option<(ComponentId, ComponentId)>,
        matched: Vec<ArchetypeId>,
    ) -> Self {
        Self {
            inner: Iter2::new(archetypes, ids, matched),
        }
    }
}

impl<'a, A: 'static, B: 'static> Iterator for Iter2Ref<'a, A, B> {
    type Item = (&'a A, &'a B);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.inner.row < self.inner.len {
                let row = self.inner.row;
                self.inner.row = row + 1;
                // SAFETY: `row < len`, as in `Iter2::next`. `entities` is
                // still tracked and still clamps `len` -- it is simply
                // not returned.
                return Some(unsafe {
                    (
                        self.inner.a_col.get_unchecked(row),
                        self.inner.b_col.get_unchecked(row),
                    )
                });
            }
            if !self.inner.advance() {
                return None;
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.inner.len - self.inner.row, None)
    }

    #[inline]
    fn fold<Acc, F>(mut self, init: Acc, mut f: F) -> Acc
    where
        F: FnMut(Acc, Self::Item) -> Acc,
    {
        let mut acc = init;
        loop {
            let (a_col, b_col) = (self.inner.a_col, self.inner.b_col);
            for row in self.inner.row..self.inner.len {
                // SAFETY: `row < len`, as above.
                let item =
                    unsafe { (a_col.get_unchecked(row), b_col.get_unchecked(row)) };
                acc = f(acc, item);
            }
            self.inner.row = self.inner.len;
            if !self.inner.advance() {
                return acc;
            }
        }
    }
}

// =====================================================================
// Constructors for the entity-free variants
// =====================================================================

/// These sit here rather than next to `iter`/`iter2` in `archetype.rs`
/// purely so the whole entity-free path is one reviewable file. Being a
/// child module, this can reach `Archetypes`' private `archetypes` field
/// and `Archetype`'s private `component_ids` exactly as the parent can —
/// the matching logic below is a verbatim copy of `iter`/`iter2`'s own,
/// and deliberately so: if the two ever diverge, the entity-free query
/// would silently visit a different archetype set than its counterpart,
/// which is a correctness bug, not a performance one. Keep them in step.
impl Archetypes {
    /// Entity-free counterpart to [`Archetypes::iter`]. Same matched
    /// set, same order, same rows — `Iter1Ref` simply does not return
    /// the entity.
    pub(crate) fn iter_ref<T: 'static>(&self) -> Iter1Ref<'_, T> {
        let id = self.existing_component_id::<T>();
        let matched: Vec<ArchetypeId> = match id {
            Some(id) => self.archetypes_with(id).collect(),
            None => Vec::new(),
        };
        Iter1Ref::new(self, id, matched)
    }

    /// Entity-free counterpart to [`Archetypes::iter2`], and the
    /// apples-to-apples shape against `bevy_ecs`'s `Query<(&A, &B)>`.
    /// See `World::query2_static_ref`'s doc comment for why the item
    /// shape is the whole point.
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
}
