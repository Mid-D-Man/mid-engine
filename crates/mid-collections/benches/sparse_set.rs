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

criterion_group!(
    benches,
    bench_insert,
    bench_get_existing,
    bench_remove_all,
    bench_iterate_values
);
criterion_main!(benches);
