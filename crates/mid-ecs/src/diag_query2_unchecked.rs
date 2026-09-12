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

/// Structural test, not a safety/inlining test like the three above.
/// bevy_ecs's own tuple `QueryData::fetch` (`fetch.rs`'s
/// `impl_tuple_query_data!` macro, read directly from `Mid-D-Man/bevy`)
/// composes a 2-component fetch as
/// `Some((A::fetch(...)?, B::fetch(...)?))` — two independent,
/// single-component fetch calls glued together with `?`, not one
/// hand-written block that reads both columns inline. `Iter2`,
/// `Iter2Unchecked`, `Iter2TwoTupleItem`, and `Iter2UnusedBCol` above all
/// still do the latter. This is the one variant built to mirror the
/// former, with everything else (archetype advance, per-archetype
/// downcast, row/len bookkeeping, unsafe unchecked access) identical to
/// `Iter2Unchecked`.
pub(crate) struct Iter2Composed<'a, A, B> {
    archetypes: &'a Archetypes,
    ids: Option<(ComponentId, ComponentId)>,
    matched: std::vec::IntoIter<ArchetypeId>,
    entities: &'a [Entity],
    a_col: &'a [A],
    b_col: &'a [B],
    row: usize,
    len: usize,
}

/// One small function per component, mirroring bevy's per-component
/// `WorldQuery::fetch` — each does exactly one unchecked slice read and
/// wraps it in `Some`, same shape bevy's own `&T` read-only fetch has.
#[inline(always)]
unsafe fn fetch_one<T>(col: &[T], row: usize) -> Option<&T> {
    // SAFETY: caller (Iter2Composed::next) only calls this with
    // row < self.len, and len is set to the min of entities/a_col/b_col
    // lengths on every archetype advance below.
    Some(unsafe { col.get_unchecked(row) })
}

impl<'a, A: 'static, B: 'static> Iterator for Iter2Composed<'a, A, B> {
    type Item = (Entity, &'a A, &'a B);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.row < self.len {
                let row = self.row;
                self.row += 1;
                // SAFETY: row < self.len, same invariant as the two
                // fetch_one calls below.
                let entity = unsafe { *self.entities.get_unchecked(row) };
                // Composed like bevy's tuple fetch: Some((A::fetch(...)?, B::fetch(...)?)).
                return Some((
                    entity,
                    unsafe { fetch_one(self.a_col, row) }?,
                    unsafe { fetch_one(self.b_col, row) }?,
                ));
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

/// Tests a different candidate than the four variants above:
/// `Iter2UnusedBCol`'s own real-CI result (see docs/mid-ecs.md) showed
/// that merely *holding* a second, unused, differently-typed slice
/// field (`b_col: &'a [B]`) reproduces the full regression even though
/// its own `Item` is a single reference, identical in shape to the
/// fast `Iter1Unchecked`. `Iter2Composed` showed the fetch-composition
/// shape doesn't matter either. What's left: holding two independently
/// typed `&'a [_]` slice references live in the same struct, at all,
/// might be the actual cost, not what gets returned. Real unsafe-heavy
/// ECS implementations don't store columns as typed slice references
/// internally for exactly this class of reason — bevy's own table
/// storage holds raw/`NonNull` pointers, materializing a `&T` only at
/// the point of return. This is the same idea: `a_col`/`b_col` here
/// are `*const A`/`*const B`, not `&'a [A]`/`&'a [B]`, converted to a
/// reference only when the returned item is constructed.
pub(crate) struct Iter2RawPtr<'a, A, B> {
    archetypes: &'a Archetypes,
    ids: Option<(ComponentId, ComponentId)>,
    matched: std::vec::IntoIter<ArchetypeId>,
    entities: &'a [Entity],
    a_col: *const A,
    b_col: *const B,
    row: usize,
    len: usize,
    _marker: std::marker::PhantomData<(&'a [A], &'a [B])>,
}

impl<'a, A: 'static, B: 'static> Iterator for Iter2RawPtr<'a, A, B> {
    type Item = (Entity, &'a A, &'a B);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.row < self.len {
                let row = self.row;
                self.row += 1;
                // SAFETY: row < self.len, and len is the min of
                // entities/a_col/b_col lengths as of the most recent
                // archetype advance below, so both offsets are in
                // bounds of their respective (still-borrowed, per the
                // 'a in PhantomData) allocations.
                let item = unsafe {
                    (
                        *self.entities.get_unchecked(row),
                        &*self.a_col.add(row),
                        &*self.b_col.add(row),
                    )
                };
                return Some(item);
            }
            let (a_id, b_id) = self.ids?;
            let archetype_id = self.matched.next()?;
            let (entities, a_col, b_col) =
                self.archetypes
                    .diag_entities_and_columns::<A, B>(archetype_id, a_id, b_id);
            self.len = entities.len().min(a_col.len()).min(b_col.len());
            self.entities = entities;
            self.a_col = a_col.as_ptr();
            self.b_col = b_col.as_ptr();
            self.row = 0;
        }
    }
}

/// Required by `Iter2OwnedDirect` below. A monomorphized trait method,
/// not a stored `fn` pointer like `Iter2TwoTupleItem`'s `combine`
/// field. `Iter2TwoTupleItem` was built to isolate item-shape (2-tuple
/// vs 3-tuple) from the two-slice-fields question, and its real-CI
/// result (see docs/mid-ecs.md) was the only fast variant among six —
/// but its own `combine: fn(&A, &B) -> A` field is a genuine indirect
/// call, which is its own kind of optimization barrier LLVM usually
/// can't inline through. Every other variant tested (`Unchecked`,
/// `UnusedBCol`, `Composed`, `RawPtr`) called nothing indirectly and
/// stayed slow. `TwoTupleItem` called something indirectly and was
/// fast. Before concluding "owned return" is what matters, "opaque
/// call in the loop" needs to be ruled out as the actual reason,
/// since they've been confounded together in every test so far. This
/// trait call is fully known at compile time and monomorphized, so
/// the compiler is free to inline it, unlike a stored `fn` pointer.
/// `pub`, not `pub(crate)`: `benches/archetype_core.rs` is a separate
/// crate linking against `mid-ecs` as a library, and needs to
/// implement this for its own `Position`/`Velocity` types to call
/// `World::query2_static_diag_owned_direct`. Hidden from docs, same as
/// every other `#[doc(hidden)]` diagnostic method in this file.
#[doc(hidden)]
pub trait DiagCombine<B> {
    #[doc(hidden)]
    fn diag_combine(a: &Self, b: &B) -> Self;
}

pub(crate) struct Iter2OwnedDirect<'a, A, B> {
    archetypes: &'a Archetypes,
    ids: Option<(ComponentId, ComponentId)>,
    matched: std::vec::IntoIter<ArchetypeId>,
    entities: &'a [Entity],
    a_col: &'a [A],
    b_col: &'a [B],
    row: usize,
    len: usize,
}

impl<'a, A: 'static + Copy + DiagCombine<B>, B: 'static> Iterator for Iter2OwnedDirect<'a, A, B> {
    type Item = (Entity, A);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.row < self.len {
                let row = self.row;
                self.row += 1;
                // SAFETY: row < self.len, same invariant as every other
                // variant's archetype-advance logic below.
                let item = unsafe {
                    (
                        *self.entities.get_unchecked(row),
                        A::diag_combine(self.a_col.get_unchecked(row), self.b_col.get_unchecked(row)),
                    )
                };
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

// Real Iter1's outer-#[inline(never)]-wrapper test (RealQuery1/
// RealQuery2 in archetype_core.rs's bench_query2_static_diag_unchecked)
// is structurally byte-identical for both arities -- same
// `#[inline(never)] fn run(&mut self) -> f32`, same for-loop shape,
// only the summed expression differs (`pos.x` vs `pos.x + vel.dx`) --
// yet real CI (Archetype Core builds #12/#13, reconfirmed #15/#16)
// consistently shows the 1-column wrapper fast and the 2-column
// wrapper just as slow as unwrapped `Iter2`. An outer wrapper around
// the *whole consuming loop* isn't sufficient for 2 columns, whatever
// it does for 1 -- so the difference has to be inside `Iter2::next`
// itself, not in how its caller is (or isn't) walled off.
//
// `Iter2::next`'s own doc comment already establishes the shape: a
// tiny, per-entity hot path (`if row < len {...}`) and a much larger,
// per-archetype-only cold path (id lookup, column resolution, `if let`
// unwrapping) that only runs once per archetype, not once per entity
// -- see `archetype.rs`. Untested until now: does the hot path being
// small enough to inline cleanly actually matter on its own, decoupled
// from the cold path's own size? This variant is `Iter2`'s exact logic
// with the cold path physically moved into its own `#[inline(never)]`
// function, so the hot path -- the part that runs N times, not once
// per archetype -- is the only thing left for the compiler to inline
// at the call site, regardless of how big or small the (now
// irrelevant, walled off) archetype-resolution code is.
pub(crate) struct Iter2ColdSplit<'a, A, B> {
    archetypes: &'a Archetypes,
    ids: Option<(ComponentId, ComponentId)>,
    matched: std::vec::IntoIter<ArchetypeId>,
    entities: &'a [Entity],
    a_col: &'a [A],
    b_col: &'a [B],
    row: usize,
    len: usize,
}

impl<'a, A: 'static, B: 'static> Iter2ColdSplit<'a, A, B> {
    #[inline(never)]
    fn advance(&mut self) -> Option<(Entity, &'a A, &'a B)> {
        loop {
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
            if self.row < self.len {
                let item = (
                    self.entities[self.row],
                    &self.a_col[self.row],
                    &self.b_col[self.row],
                );
                self.row += 1;
                return Some(item);
            }
        }
    }
}

impl<'a, A: 'static, B: 'static> Iterator for Iter2ColdSplit<'a, A, B> {
    type Item = (Entity, &'a A, &'a B);

    fn next(&mut self) -> Option<Self::Item> {
        if self.row < self.len {
            let item = (
                self.entities[self.row],
                &self.a_col[self.row],
                &self.b_col[self.row],
            );
            self.row += 1;
            return Some(item);
        }
        self.advance()
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

    pub(crate) fn iter2_diag_composed<A: 'static, B: 'static>(&self) -> Iter2Composed<'_, A, B> {
        let (ids, matched) = self.diag_matched_and_ids::<A, B>();
        Iter2Composed {
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

    pub(crate) fn iter2_diag_raw_ptr<A: 'static, B: 'static>(&self) -> Iter2RawPtr<'_, A, B> {
        let (ids, matched) = self.diag_matched_and_ids::<A, B>();
        Iter2RawPtr {
            archetypes: self,
            ids,
            matched: matched.into_iter(),
            entities: &[],
            a_col: std::ptr::null(),
            b_col: std::ptr::null(),
            row: 0,
            len: 0,
            _marker: std::marker::PhantomData,
        }
    }

    pub(crate) fn iter2_diag_owned_direct<A: 'static + Copy + DiagCombine<B>, B: 'static>(
        &self,
    ) -> Iter2OwnedDirect<'_, A, B> {
        let (ids, matched) = self.diag_matched_and_ids::<A, B>();
        Iter2OwnedDirect {
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

    pub(crate) fn iter2_diag_cold_split<A: 'static, B: 'static>(
        &self,
    ) -> Iter2ColdSplit<'_, A, B> {
        let (ids, matched) = self.diag_matched_and_ids::<A, B>();
        Iter2ColdSplit {
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
