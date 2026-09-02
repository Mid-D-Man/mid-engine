//! Criterion benchmarks: `mid_collections::SparseSet` vs `std::HashMap`.
//!
//! `HashMap<u32, T>` is the natural baseline — it's what anyone reaching
//! for "map an integer key to a value" in Rust would use without a
//! reason not to, so it's the honest comparison, not a strawman. The
//! point of `SparseSet` is that it should win decisively on `get` and
//! `iterate` (direct array indexing / contiguous scan vs hashing +
//! probing), while `insert`/`remove` should be closer, and could even
//! lose at small sizes where the sparse array's lazy growth
//! (`docs/mid-collections.md`'s own documented no-paging trade-off) has
//! to repeatedly `Vec::resize` before `HashMap`'s hashing overhead would
//! start to matter.
//!
//! Run: `cargo bench -p mid-collections --bench sparse_set`
//! Report: `target/criterion/report/index.html` (via the `html_reports`
//! feature — same as mid-math's own bench setup).
//!
//! Sizes chosen to bracket Mid Engine's own stated target
//! (`docs/architecture.md`: "100,000+ entities per core"): 100 and 1,000
//! as small/mid reference points, 10,000 and 100,000 to actually land on
//! and past the real target scale.

use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput};
use mid_collections::SparseSet;
use std::collections::HashMap;
use std::hint::black_box;

const SIZES: [u32; 4] = [100, 1_000, 10_000, 100_000];

fn populated_sparse_set(n: u32) -> SparseSet<u32, u32> {
    let mut s = SparseSet::with_capacity(n as usize);
    for i in 0..n {
        s.insert(i, i);
    }
    s
}

fn populated_hash_map(n: u32) -> HashMap<u32, u32> {
    let mut m = HashMap::with_capacity(n as usize);
    for i in 0..n {
        m.insert(i, i);
    }
    m
}

fn bench_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert_sequential");
    for &n in &SIZES {
        group.throughput(Throughput::Elements(n as u64));

        group.bench_with_input(BenchmarkId::new("SparseSet", n), &n, |b, &n| {
            b.iter(|| {
                let mut s: SparseSet<u32, u32> = SparseSet::with_capacity(n as usize);
                for i in 0..n {
                    s.insert(i, i);
                }
                black_box(&s);
            });
        });

        group.bench_with_input(BenchmarkId::new("HashMap", n), &n, |b, &n| {
            b.iter(|| {
                let mut m: HashMap<u32, u32> = HashMap::with_capacity(n as usize);
                for i in 0..n {
                    m.insert(i, i);
                }
                black_box(&m);
            });
        });
    }
    group.finish();
}

fn bench_get_existing(c: &mut Criterion) {
    let mut group = c.benchmark_group("get_existing");
    for &n in &SIZES {
        group.throughput(Throughput::Elements(n as u64));

        let s = populated_sparse_set(n);
        group.bench_with_input(BenchmarkId::new("SparseSet", n), &n, |b, &n| {
            b.iter(|| {
                let mut sum = 0u64;
                for i in 0..n {
                    sum = sum.wrapping_add(*s.get(i).unwrap() as u64);
                }
                black_box(sum);
            });
        });

        let m = populated_hash_map(n);
        group.bench_with_input(BenchmarkId::new("HashMap", n), &n, |b, &n| {
            b.iter(|| {
                let mut sum = 0u64;
                for i in 0..n {
                    sum = sum.wrapping_add(*m.get(&i).unwrap() as u64);
                }
                black_box(sum);
            });
        });
    }
    group.finish();
}

fn bench_remove_all(c: &mut Criterion) {
    // Removal drains the structure, so each measured iteration needs a
    // freshly-populated one — iter_batched's setup closure handles that
    // without timing the (re-)population itself, only the removals.
    let mut group = c.benchmark_group("remove_all");
    for &n in &SIZES {
        group.throughput(Throughput::Elements(n as u64));

        group.bench_with_input(BenchmarkId::new("SparseSet", n), &n, |b, &n| {
            b.iter_batched(
                || populated_sparse_set(n),
                |mut s| {
                    for i in 0..n {
                        black_box(s.remove(i));
                    }
                },
                BatchSize::LargeInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("HashMap", n), &n, |b, &n| {
            b.iter_batched(
                || populated_hash_map(n),
                |mut m| {
                    for i in 0..n {
                        black_box(m.remove(&i));
                    }
                },
                BatchSize::LargeInput,
            );
        });
    }
    group.finish();
}

fn bench_iterate_values(c: &mut Criterion) {
    let mut group = c.benchmark_group("iterate_values");
    for &n in &SIZES {
        group.throughput(Throughput::Elements(n as u64));

        let s = populated_sparse_set(n);
        group.bench_with_input(BenchmarkId::new("SparseSet", n), &n, |b, _| {
            b.iter(|| {
                let sum: u64 = s.values().map(|&v| v as u64).sum();
                black_box(sum);
            });
        });

        let m = populated_hash_map(n);
        group.bench_with_input(BenchmarkId::new("HashMap", n), &n, |b, _| {
            b.iter(|| {
                let sum: u64 = m.values().map(|&v| v as u64).sum();
                black_box(sum);
            });
        });
    }
    group.finish();
}

/// Real motivation: `mid-ecs`'s `Archetypes::take_two`/`give_back_two`
/// (crates/mid-ecs/src/archetype.rs) needs two `&mut Archetype`s at once
/// for every structural change (insert/remove component), and currently
/// gets them by `remove`-ing both out of the `SparseSet<ArchetypeId,
/// Archetype>` and `insert`-ing them back afterward -- two swap-removes
/// plus two dense-array pushes, every single call, purely to satisfy the
/// borrow checker, not because the archetypes actually need to move.
/// `bevy_ecs`'s own equivalent (`Archetypes::get_maybe_disjoint_mut`,
/// read directly from `Mid-D-Man/bevy`) avoids this entirely -- a plain
/// `Vec` plus a disjoint-index split, zero data movement. This group
/// measures that exact difference: `remove`+`remove`+`insert`+`insert`
/// vs the new `get_disjoint_mut` (`split_at_mut`-based, zero new
/// `unsafe`), repeatedly fetching the SAME two keys (the real access
/// pattern -- a game hammering the same two archetypes during
/// structural churn, not a random pair each time) out of a set sized
/// like a realistic archetype count, not this file's other groups'
/// component-count scale.
const ARCHETYPE_COUNTS: [u32; 3] = [16, 64, 256];

/// Stand-in for the real `Archetype` struct's own size (216 bytes,
/// measured directly via `std::mem::size_of::<Archetype>()` in
/// mid-ecs -- `Archetype` itself isn't `pub`, so this bench can't name
/// it directly; matching its byte size is what matters for a realistic
/// swap-remove/push cost, not the exact field layout). A bare `u32`
/// value (the other groups' choice) would understate the real
/// move/copy cost of `remove`+`insert` shuffling something this size
/// around, so this group repeats the same comparison at the size that
/// actually matters for the `take_two`/`give_back_two` call site.
#[derive(Clone, Copy)]
struct ArchetypeSized {
    _bytes: [u8; 216],
}

impl ArchetypeSized {
    fn new(seed: u8) -> Self {
        Self {
            _bytes: [seed; 216],
        }
    }
    // Touches one byte so the compiler can't fold the whole 216-byte
    // value away as dead weight -- same reasoning as the `u32` group's
    // `+= 1`.
    fn touch(&mut self) {
        self._bytes[0] = self._bytes[0].wrapping_add(1);
    }
}

fn populated_archetype_sized_set(n: u32) -> SparseSet<u32, ArchetypeSized> {
    let mut s = SparseSet::with_capacity(n as usize);
    for i in 0..n {
        s.insert(i, ArchetypeSized::new(i as u8));
    }
    s
}

fn bench_two_at_once_archetype_sized(c: &mut Criterion) {
    let mut group = c.benchmark_group("two_at_once_archetype_sized");
    for &n in &ARCHETYPE_COUNTS {
        group.throughput(Throughput::Elements(1));
        let key_a = n / 4;
        let key_b = (n * 3) / 4;

        let mut s_remove_reinsert = populated_archetype_sized_set(n);
        group.bench_with_input(BenchmarkId::new("remove_then_reinsert", n), &n, |b, _| {
            b.iter(|| {
                let mut val_a = s_remove_reinsert.remove(key_a).unwrap();
                let mut val_b = s_remove_reinsert.remove(key_b).unwrap();
                val_a.touch();
                val_b.touch();
                s_remove_reinsert.insert(key_a, val_a);
                s_remove_reinsert.insert(key_b, val_b);
            });
        });

        let mut s_disjoint = populated_archetype_sized_set(n);
        group.bench_with_input(BenchmarkId::new("get_disjoint_mut", n), &n, |b, _| {
            b.iter(|| {
                let (a, b) = s_disjoint.get_disjoint_mut(key_a, key_b);
                a.unwrap().touch();
                b.unwrap().touch();
            });
        });
    }
    group.finish();
}

fn bench_two_at_once(c: &mut Criterion) {
    let mut group = c.benchmark_group("two_at_once");
    for &n in &ARCHETYPE_COUNTS {
        group.throughput(Throughput::Elements(1));
        // Two keys roughly a quarter and three-quarters of the way
        // through the set -- neither at an edge, matching how real
        // archetype ids for a churning entity aren't reliably first or
        // last.
        let key_a = n / 4;
        let key_b = (n * 3) / 4;

        // Persistent set, mutated in place across every sample -- NOT
        // `iter_batched` with fresh O(n) setup per iteration, which
        // would let the setup cost dominate a measurement this small
        // and defeat the point. `remove_then_reinsert` restores the
        // same two keys every cycle (positions of OTHER elements drift
        // as swap-remove reshuffles them, same as the real workload
        // this models -- structural churn that keeps evolving, not a
        // fixed snapshot replayed identically), so reusing one set
        // across all samples is both cheaper and more representative,
        // not just a shortcut.
        let mut s_remove_reinsert = populated_sparse_set(n);
        group.bench_with_input(BenchmarkId::new("remove_then_reinsert", n), &n, |b, _| {
            b.iter(|| {
                let val_a = s_remove_reinsert.remove(key_a).unwrap();
                let val_b = s_remove_reinsert.remove(key_b).unwrap();
                // Real call sites mutate the two values here
                // (migrating columns) -- represented as a trivial
                // touch so the compiler can't elide the whole
                // thing, without adding unrelated cost.
                let val_a = black_box(val_a) + 1;
                let val_b = black_box(val_b) + 1;
                s_remove_reinsert.insert(key_a, val_a);
                s_remove_reinsert.insert(key_b, val_b);
            });
        });

        let mut s_disjoint = populated_sparse_set(n);
        group.bench_with_input(BenchmarkId::new("get_disjoint_mut", n), &n, |b, _| {
            b.iter(|| {
                let (a, b) = s_disjoint.get_disjoint_mut(key_a, key_b);
                *a.unwrap() += 1;
                *b.unwrap() += 1;
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_insert,
    bench_get_existing,
    bench_remove_all,
    bench_iterate_values,
    bench_two_at_once,
    bench_two_at_once_archetype_sized
);
criterion_main!(benches);
