//! Decisive, standalone test for the one hypothesis that explains the
//! 4x `query2_static` gap: **`Iter2::Item` crosses the System V AMD64
//! 16-byte register-return threshold and `Iter1::Item` does not.**
//!
//! # Why this bench exists and why it is deliberately NOT in mid-ecs
//!
//! `docs/mid-ecs.md` (builds #15-#18) records that `archetype_core.rs`'s
//! own numbers became untrustworthy: adding code to `mid-ecs`'s
//! compilation unit tipped `query_static_single_component` AND
//! `query_static_unchecked_1col` -- the two most reliable "stays fast"
//! reference points in the whole matrix -- from ~65-106µs into the
//! ~375µs slow cluster, with `archetype.rs` itself unchanged. Every
//! variant result measured in that era (`Iter2UnusedBCol` slow,
//! `Iter2TwoTupleItem` fast) is confounded by that, because the known-fast
//! control moved too.
//!
//! This crate links against NOTHING. No `mid-ecs`, no `mid-collections`,
//! no `bevy_ecs`. Its types are local copies with identical layout. It
//! therefore cannot be tipped by mid-ecs's compilation-unit layout, and
//! its result is valid regardless of what `archetype_core.rs` is doing
//! that week. Run it first; rewrite `archetype.rs` only after it answers.
//!
//! # The hypothesis, precisely
//!
//! SysV AMD64 §3.2.3: an aggregate larger than two eightbytes (16 bytes)
//! is classified MEMORY and returned through a hidden pointer -- the
//! caller allocates stack space, the callee stores the result there, the
//! caller loads it back. At or under 16 bytes it is returned in RAX:RDX.
//!
//! With `Entity` = 8 bytes (two `u32`s -- `mid-collections`'
//! `GenerationalIndex`) and every reference 8 bytes:
//!
//! | Iterator                   | `Item`                | `Option<Item>` | Class    |
//! |----------------------------|-----------------------|----------------|----------|
//! | `Iter1` (mid-ecs)          | `(Entity, &A)`        | 16 B (niche)   | REGISTER |
//! | `Iter2` (mid-ecs)          | `(Entity, &A, &B)`    | 24 B (niche)   | MEMORY   |
//! | `bevy` `Query<&A>`         | `&A`                  | 8 B            | REGISTER |
//! | `bevy` `Query<(&A, &B)>`   | `(&A, &B)`            | 16 B (niche)   | REGISTER |
//!
//! That boundary falls *exactly* where the measured gap falls, and
//! nowhere else. `query_static_single_component` is at parity with
//! `bevy_ecs` (9.42µs vs 9.35µs, build #22) because both sides return in
//! registers. `dense_query_iteration` is 3.99x because mid-ecs crossed
//! into MEMORY and bevy did not. mid-ecs is not losing an architecture
//! argument there -- it is paying for putting `Entity` in every item
//! while bevy makes `Entity` opt-in query data.
//!
//! # What each group isolates
//!
//! `ret16_entity_ref` vs `ret24_entity_ref_ref` reproduces the real
//! `Iter1`/`Iter2` pair. But that pair confounds two variables at once
//! (item size AND how many columns are read), which is exactly the
//! confound every previous diagnostic in this investigation inherited.
//!
//! **`ret16_ref_ref` is the control that breaks the confound.** It reads
//! *both* columns, does *identical* per-item work to `ret24_entity_ref_ref`,
//! and differs in one respect only: `Entity` is not in the returned
//! tuple, so `Option<Item>` is 16 bytes instead of 24.
//!
//! - If `ret16_ref_ref` lands with `ret16_entity_ref` and
//!   `raw_two_field`, and `ret24_entity_ref_ref` sits ~4x above all
//!   three: **hypothesis confirmed.** The fix is the return type, and
//!   no amount of `unsafe`, `#[inline(always)]`, raw pointers or cold-path
//!   splitting can reach it -- which is precisely why all four of those
//!   came back negative.
//! - If `ret16_ref_ref` lands *with* `ret24_entity_ref_ref` instead:
//!   **hypothesis falsified.** Reading a second column is the cost, item
//!   size is irrelevant, and this whole line of attack is dead. That is
//!   a genuinely useful negative -- it would be the first result in this
//!   investigation that rules out the return path rather than one more
//!   intervention on it.
//!
//! `ret24_via_fold` tests the second, independent fix: same 24-byte
//! `Item`, but consumed through `Iterator::fold`, whose override runs a
//! flat contiguous inner loop and never materialises an `Option<Item>`
//! across a call boundary at all. This is what `bevy_ecs`'s own
//! `QueryIter::fold` / `fold_over_table_range` do (`query/iter.rs`, read
//! directly from `Mid-D-Man/bevy`). If this lands fast while
//! `ret24_entity_ref_ref` is slow, `Entity` can stay in the item for
//! internal iteration (`for_each`/`sum`/`collect`) and only `for` loops
//! need the narrowed item.
//!
//! `two_archetype` repeats the decisive pair against two matching
//! archetypes rather than one, so the result cannot be an artifact of
//! the single-archetype shape `populated_world` happens to produce.
//!
//! # Reading it
//!
//! The absolute numbers do not matter and should not be compared against
//! `archetype_core.rs`'s. Only the ratios within this file's own groups
//! matter, and they are all measured in one binary in one run.
//!
//! Sizes are asserted at compile time below, so a layout assumption that
//! is wrong on some target fails the build instead of silently
//! invalidating the run.

use std::mem::size_of;

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};

const N: usize = 100_000;

/// Layout-identical to `mid-ecs`'s `Entity` (a `mid-collections`
/// `GenerationalIndex`: two `u32`s, 8 bytes, no niche).
#[derive(Clone, Copy)]
struct Entity {
    index: u32,
    generation: u32,
}

/// Layout-identical to `archetype_core.rs`'s own bench component.
#[derive(Clone, Copy)]
struct Position {
    x: f32,
    y: f32,
    z: f32,
}

#[derive(Clone, Copy)]
struct Velocity {
    dx: f32,
    dy: f32,
    dz: f32,
}

// The entire premise of this bench, enforced at compile time rather than
// assumed. If any of these fail, the hypothesis is not merely unproven --
// it is not even well-formed on this target, and the run means nothing.
const _: () = {
    assert!(size_of::<Entity>() == 8);
    // The fast side: fits in RAX:RDX.
    assert!(size_of::<Option<(Entity, &Position)>>() == 16);
    assert!(size_of::<Option<(&Position, &Velocity)>>() == 16);
    // The slow side: MEMORY class, returned through a hidden pointer.
    assert!(size_of::<Option<(Entity, &Position, &Velocity)>>() == 24);
};

/// One archetype's worth of columns, in the same SoA shape
/// `archetype.rs`'s `Table` holds them.
struct Chunk {
    entities: Vec<Entity>,
    positions: Vec<Position>,
    velocities: Vec<Velocity>,
}

fn chunk(n: usize) -> Chunk {
    Chunk {
        entities: (0..n)
            .map(|i| Entity {
                index: i as u32,
                generation: 1,
            })
            .collect(),
        positions: (0..n)
            .map(|_| Position {
                x: 1.0,
                y: 2.0,
                z: 3.0,
            })
            .collect(),
        velocities: (0..n)
            .map(|_| Velocity {
                dx: 0.1,
                dy: 0.2,
                dz: 0.3,
            })
            .collect(),
    }
}

// ---------------------------------------------------------------------
// The iterators. Structurally identical on purpose: same fields, same
// `row`/`len` bookkeeping, same `get_unchecked` access, same multi-chunk
// advance shape as the real `Iter1`/`Iter2`. The ONLY thing that varies
// between them is the `Item` type -- and therefore the return ABI.
// ---------------------------------------------------------------------

/// Mirrors `mid-ecs`'s real `Iter1`. `Option<Item>` = 16 B -> RAX:RDX.
struct Ret16EntityRef<'a> {
    chunks: &'a [Chunk],
    chunk: usize,
    entities: &'a [Entity],
    a: &'a [Position],
    row: usize,
    len: usize,
}

impl<'a> Ret16EntityRef<'a> {
    fn new(chunks: &'a [Chunk]) -> Self {
        Self {
            chunks,
            chunk: 0,
            entities: &[],
            a: &[],
            row: 0,
            len: 0,
        }
    }

    #[cold]
    #[inline(never)]
    fn advance(&mut self) -> bool {
        while self.chunk < self.chunks.len() {
            let c = &self.chunks[self.chunk];
            self.chunk += 1;
            let len = c.entities.len().min(c.positions.len());
            if len == 0 {
                continue;
            }
            self.entities = &c.entities;
            self.a = &c.positions;
            self.len = len;
            self.row = 0;
            return true;
        }
        false
    }
}

impl<'a> Iterator for Ret16EntityRef<'a> {
    type Item = (Entity, &'a Position);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.row < self.len {
                let row = self.row;
                self.row = row + 1;
                // SAFETY: `row < len`, and `len` is clamped to the min of
                // both slice lengths on every `advance`.
                return Some(unsafe {
                    (*self.entities.get_unchecked(row), self.a.get_unchecked(row))
                });
            }
            if !self.advance() {
                return None;
            }
        }
    }
}

/// Mirrors `mid-ecs`'s real `Iter2`. `Option<Item>` = 24 B -> MEMORY.
/// This is the shape currently shipping, and the one under suspicion.
struct Ret24EntityRefRef<'a> {
    chunks: &'a [Chunk],
    chunk: usize,
    entities: &'a [Entity],
    a: &'a [Position],
    b: &'a [Velocity],
    row: usize,
    len: usize,
}

impl<'a> Ret24EntityRefRef<'a> {
    fn new(chunks: &'a [Chunk]) -> Self {
        Self {
            chunks,
            chunk: 0,
            entities: &[],
            a: &[],
            b: &[],
            row: 0,
            len: 0,
        }
    }

    #[cold]
    #[inline(never)]
    fn advance(&mut self) -> bool {
        while self.chunk < self.chunks.len() {
            let c = &self.chunks[self.chunk];
            self.chunk += 1;
            let len = c
                .entities
                .len()
                .min(c.positions.len())
                .min(c.velocities.len());
            if len == 0 {
                continue;
            }
            self.entities = &c.entities;
            self.a = &c.positions;
            self.b = &c.velocities;
            self.len = len;
            self.row = 0;
            return true;
        }
        false
    }
}

impl<'a> Iterator for Ret24EntityRefRef<'a> {
    type Item = (Entity, &'a Position, &'a Velocity);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.row < self.len {
                let row = self.row;
                self.row = row + 1;
                // SAFETY: `row < len`, clamped to the min of all three.
                return Some(unsafe {
                    (
                        *self.entities.get_unchecked(row),
                        self.a.get_unchecked(row),
                        self.b.get_unchecked(row),
                    )
                });
            }
            if !self.advance() {
                return None;
            }
        }
    }

    /// The `bevy_ecs` technique, ported. A `for` loop can only ever drive
    /// an iterator through repeated `next()` -- `docs/mid-ecs.md` already
    /// establishes that correctly -- but `for_each`, `sum`, `fold` and
    /// `collect` all route through *this* instead, where the per-chunk
    /// run is one flat contiguous loop and no `Option<Item>` ever crosses
    /// a call boundary. Same structure as `bevy_ecs`'s
    /// `QueryIter::fold` / `fold_over_table_range`.
    #[inline]
    fn fold<Acc, F>(mut self, init: Acc, mut f: F) -> Acc
    where
        F: FnMut(Acc, Self::Item) -> Acc,
    {
        let mut acc = init;
        loop {
            let (entities, a, b) = (self.entities, self.a, self.b);
            for row in self.row..self.len {
                // SAFETY: `row < len`, clamped to the min of all three.
                let item = unsafe {
                    (
                        *entities.get_unchecked(row),
                        a.get_unchecked(row),
                        b.get_unchecked(row),
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

/// **The control that breaks the confound.** Reads both columns, exactly
/// like `Ret24EntityRefRef`. Identical per-item work. The only
/// difference in the entire type is that `Entity` is not in the tuple,
/// so `Option<Item>` is 16 B and comes back in RAX:RDX.
struct Ret16RefRef<'a> {
    chunks: &'a [Chunk],
    chunk: usize,
    entities: &'a [Entity],
    a: &'a [Position],
    b: &'a [Velocity],
    row: usize,
    len: usize,
}

impl<'a> Ret16RefRef<'a> {
    fn new(chunks: &'a [Chunk]) -> Self {
        Self {
            chunks,
            chunk: 0,
            entities: &[],
            a: &[],
            b: &[],
            row: 0,
            len: 0,
        }
    }

    #[cold]
    #[inline(never)]
    fn advance(&mut self) -> bool {
        while self.chunk < self.chunks.len() {
            let c = &self.chunks[self.chunk];
            self.chunk += 1;
            let len = c
                .entities
                .len()
                .min(c.positions.len())
                .min(c.velocities.len());
            if len == 0 {
                continue;
            }
            self.entities = &c.entities;
            self.a = &c.positions;
            self.b = &c.velocities;
            self.len = len;
            self.row = 0;
            return true;
        }
        false
    }
}

impl<'a> Iterator for Ret16RefRef<'a> {
    type Item = (&'a Position, &'a Velocity);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.row < self.len {
                let row = self.row;
                self.row = row + 1;
                // SAFETY: `row < len`, clamped to the min of all three.
                // `entities` is still tracked and still clamps `len`,
                // exactly as in `Ret24EntityRefRef` -- it is simply not
                // returned. Keeping it live is deliberate: it holds the
                // struct's field set identical so that struct size is not
                // a second uncontrolled variable.
                return Some(unsafe { (self.a.get_unchecked(row), self.b.get_unchecked(row)) });
            }
            if !self.advance() {
                return None;
            }
        }
    }
}

// ---------------------------------------------------------------------
// Benches
// ---------------------------------------------------------------------

fn bench_item_size(c: &mut Criterion) {
    let chunks = vec![chunk(N)];
    let chunks = black_box(chunks);

    let mut g = c.benchmark_group("item_size_one_archetype");
    g.throughput(Throughput::Elements(N as u64));

    // Reads one column. 16-byte item. Mirrors the real `Iter1`.
    g.bench_function("ret16_entity_ref", |b| {
        b.iter(|| {
            let mut sum = 0.0f32;
            for (_, pos) in Ret16EntityRef::new(&chunks) {
                sum += pos.x;
            }
            black_box(sum)
        });
    });

    // Reads two columns. 24-byte item. Mirrors the real `Iter2`.
    g.bench_function("ret24_entity_ref_ref", |b| {
        b.iter(|| {
            let mut sum = 0.0f32;
            for (_, pos, vel) in Ret24EntityRefRef::new(&chunks) {
                sum += pos.x + vel.dx;
            }
            black_box(sum)
        });
    });

    // THE DECISIVE ONE. Reads two columns, same as above. 16-byte item.
    g.bench_function("ret16_ref_ref", |b| {
        b.iter(|| {
            let mut sum = 0.0f32;
            for (pos, vel) in Ret16RefRef::new(&chunks) {
                sum += pos.x + vel.dx;
            }
            black_box(sum)
        });
    });

    // 24-byte item, but internal iteration -- the `bevy_ecs` `fold` path.
    g.bench_function("ret24_via_fold", |b| {
        b.iter(|| {
            let sum = Ret24EntityRefRef::new(&chunks)
                .fold(0.0f32, |acc, (_, pos, vel)| acc + pos.x + vel.dx);
            black_box(sum)
        });
    });

    // Floor. Zero iterator abstraction, both columns.
    g.bench_function("raw_two_field", |b| {
        b.iter(|| {
            let c = &chunks[0];
            let mut sum = 0.0f32;
            for i in 0..c.positions.len() {
                sum += c.positions[i].x + c.velocities[i].dx;
            }
            black_box(sum)
        });
    });

    g.finish();
}

/// Repeats the decisive pair across two matching archetypes, so the
/// result cannot be an artifact of the single-archetype world
/// `populated_world` happens to build. Same total element count.
fn bench_two_archetypes(c: &mut Criterion) {
    let chunks = vec![chunk(N / 2), chunk(N / 2)];
    let chunks = black_box(chunks);

    let mut g = c.benchmark_group("item_size_two_archetypes");
    g.throughput(Throughput::Elements(N as u64));

    g.bench_function("ret24_entity_ref_ref", |b| {
        b.iter(|| {
            let mut sum = 0.0f32;
            for (_, pos, vel) in Ret24EntityRefRef::new(&chunks) {
                sum += pos.x + vel.dx;
            }
            black_box(sum)
        });
    });

    g.bench_function("ret16_ref_ref", |b| {
        b.iter(|| {
            let mut sum = 0.0f32;
            for (pos, vel) in Ret16RefRef::new(&chunks) {
                sum += pos.x + vel.dx;
            }
            black_box(sum)
        });
    });

    g.bench_function("ret24_via_fold", |b| {
        b.iter(|| {
            let sum = Ret24EntityRefRef::new(&chunks)
                .fold(0.0f32, |acc, (_, pos, vel)| acc + pos.x + vel.dx);
            black_box(sum)
        });
    });

    g.finish();
}

criterion_group!(benches, bench_item_size, bench_two_archetypes);
criterion_main!(benches);
