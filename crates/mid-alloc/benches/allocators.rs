//! Criterion benchmarks for every allocator strategy `mid-alloc` has
//! built so far. Structured the same way
//! `mid-arena/benches/vs_arena_crates.rs` does: one `Criterion` group
//! per operation, with feature-gated `#[cfg(feature = "...")]` entries
//! *inside* each group rather than whole conditional groups -- so
//! `cargo bench -p mid-alloc` still runs cleanly with any subset of
//! features enabled, it just has fewer bars in whichever group needed
//! the missing one. Run the full suite with `--all-features`.
//!
//! Every entry constructs its own fresh allocator state inside the
//! timed closure and measures construct+use together -- the same real
//! convention `mid-arena/benches/vs_arena_crates.rs` already uses (a
//! fresh `b.iter(|| { ... })` per entry, not `iter_batched`), kept
//! consistent rather than introducing a second pattern in the same
//! workspace. Construction cost is real and shared identically across
//! every entry in a given group, so relative comparisons stay fair
//! even though the absolute numbers include it.
//!
//! What's compared against what, and why each is a fair baseline, not
//! a strawman:
//!
//! - **`StackAllocator` vs `bumpalo::Bump`**: both are untyped, raw
//!   bump allocators over the global heap -- the same operation
//!   (`alloc_raw`/`Bump::alloc_layout`), the natural apples-to-apples
//!   comparison, same reasoning `mid-arena`'s own `BumpArena`-vs-
//!   `bumpalo` bench already uses for the typed case.
//! - **`PoolAllocator<T>` vs repeated `Box::new`/drop**: `Box` is what
//!   anyone reaching for "heap-allocate one value, free it later" in
//!   Rust would use without a specific reason not to -- the honest
//!   baseline `PoolAllocator` exists to beat on repeated create/destroy
//!   churn, not a strawman.
//! - **Combinator overhead (`FallbackAllocator`, `Segregator`,
//!   `Tracked`, `SyncAlloc`) vs the plain `HeapAlloc` they wrap**: the
//!   real question for a wrapper type isn't "is it fast" in isolation,
//!   it's "what does wrapping something already-fast actually cost" --
//!   each combinator here is built once outside the timed closure
//!   (construction is cheap and not what this group measures), then
//!   the same N alloc/dealloc pairs run through it as through the bare
//!   baseline.
//! - **`BumpVec<T, HeapAlloc>` vs `std::vec::Vec<T>`**: same real
//!   comparison `mid-collections`' own `sparse_set.rs` bench uses for
//!   its own collection type -- what everyone already reaches for,
//!   not a purpose-built loser.
//!
//! **Honest verification note, same shape as `bench-mid-collections-
//! sparse-set.yml`'s own header comment:** this crate hits the same
//! criterion-needs-edition2024 wall as every other bench in this
//! workspace (root `Cargo.toml`'s comments) -- this sandbox's rustc
//! 1.75 cannot compile a crate that depends on criterion at all, and
//! no newer toolchain is reachable here (checked directly: `apt-cache
//! policy rustc` offers nothing past 1.75, `rustup`'s own install
//! domain is not in this sandbox's allowed network list). Every
//! constructor and method signature used below was cross-checked
//! against this crate's own real source directly (not memory) --
//! `PoolAllocator::destroy`'s real signature
//! (`unsafe fn destroy(&self, item: &mut T)`) caught this file's first
//! draft using it wrong (assumed `&mut self`, assumed safe) -- but the
//! file as a whole has never actually compiled anywhere. Its first
//! real CI trigger is what actually proves it, the same way that
//! workflow's own bench file states for itself.
//!
//! Run: `cargo bench -p mid-alloc --all-features --bench allocators`
//! Report: `target/criterion/report/index.html` (`html_reports`
//! feature, same as every other bench in this workspace).

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use mid_alloc::{HeapAlloc, RawAlloc, StackAllocator};

#[cfg(feature = "backed")]
use mid_alloc::BackedStack;
#[cfg(feature = "bump_vec")]
use mid_alloc::BumpVec;
#[cfg(feature = "fallback")]
use mid_alloc::FallbackAllocator;
#[cfg(feature = "pool")]
use mid_alloc::PoolAllocator;
#[cfg(feature = "segregator")]
use mid_alloc::Segregator;
#[cfg(feature = "sync")]
use mid_alloc::SyncAlloc;
#[cfg(feature = "tracking")]
use mid_alloc::Tracked;

const SIZES: [u32; 3] = [100, 1_000, 10_000];

/// `StackAllocator` vs `bumpalo::Bump`: N sequential raw 16-byte
/// allocations, each entry constructing its own fresh, generously
/// pre-sized allocator so the run never actually exhausts either one.
fn bench_raw_alloc(c: &mut Criterion) {
    let mut group = c.benchmark_group("raw_alloc_sequential");
    for &n in &SIZES {
        group.throughput(Throughput::Elements(n as u64));

        group.bench_with_input(BenchmarkId::new("StackAllocator", n), &n, |b, &n| {
            b.iter(|| {
                let s = StackAllocator::with_capacity(n as usize * 16);
                for _ in 0..n {
                    black_box(s.alloc_raw(16, 8));
                }
            });
        });

        group.bench_with_input(BenchmarkId::new("bumpalo::Bump", n), &n, |b, &n| {
            b.iter(|| {
                let bump = bumpalo::Bump::with_capacity(n as usize * 16);
                let layout = std::alloc::Layout::from_size_align(16, 8).unwrap();
                for _ in 0..n {
                    black_box(bump.alloc_layout(layout));
                }
            });
        });
    }
    group.finish();
}

/// `PoolAllocator<[u64; 4]>` vs `Box<[u64; 4]>`: N create+destroy (or
/// new+drop) cycles, one at a time -- the churn pattern both are meant
/// to handle, not a bulk-allocate-then-bulk-free shape neither
/// specifically optimizes for.
fn bench_create_destroy_churn(c: &mut Criterion) {
    let mut group = c.benchmark_group("create_destroy_churn");
    for &n in &SIZES {
        group.throughput(Throughput::Elements(n as u64));

        #[cfg(feature = "pool")]
        group.bench_with_input(BenchmarkId::new("PoolAllocator", n), &n, |b, &n| {
            b.iter(|| {
                let pool = PoolAllocator::<[u64; 4]>::new();
                for i in 0..n {
                    let v = pool.create([i as u64; 4]).unwrap();
                    // SAFETY: `v` came from this same `pool`'s own
                    // `create` call immediately above and has not been
                    // destroyed yet.
                    unsafe {
                        pool.destroy(v);
                    }
                }
            });
        });

        group.bench_with_input(BenchmarkId::new("Box", n), &n, |b, &n| {
            b.iter(|| {
                for i in 0..n {
                    let v = Box::new([i as u64; 4]);
                    black_box(&v);
                    drop(v);
                }
            });
        });
    }
    group.finish();
}

/// Overhead of each combinator's dispatch versus calling `HeapAlloc`
/// directly: N single allocations, all comfortably within whatever
/// primary/small side is configured, so this measures dispatch cost,
/// not growth or fallback-path cost. Each allocator here is built once
/// outside the timed closure -- construction is cheap and is
/// deliberately not what this group measures.
fn bench_combinator_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("combinator_dispatch_overhead");
    let n = 10_000u32;
    group.throughput(Throughput::Elements(n as u64));

    group.bench_function("HeapAlloc (baseline)", |b| {
        let a = HeapAlloc;
        b.iter(|| {
            for _ in 0..n {
                let p = a.try_alloc_raw(16, 8).unwrap();
                // SAFETY: `p` came from `a.try_alloc_raw` with these
                // exact size/align, immediately above.
                unsafe {
                    a.try_dealloc_raw(p, 16, 8);
                }
            }
        });
    });

    #[cfg(feature = "fallback")]
    group.bench_function("FallbackAllocator<Heap, Heap>", |b| {
        let a = FallbackAllocator::new(HeapAlloc, HeapAlloc);
        b.iter(|| {
            for _ in 0..n {
                let p = a.try_alloc_raw(16, 8).unwrap();
                unsafe {
                    a.try_dealloc_raw(p, 16, 8);
                }
            }
        });
    });

    #[cfg(feature = "segregator")]
    group.bench_function("Segregator<Heap, Heap>", |b| {
        let a = Segregator::new(64, HeapAlloc, HeapAlloc);
        b.iter(|| {
            for _ in 0..n {
                let p = a.try_alloc_raw(16, 8).unwrap();
                unsafe {
                    a.try_dealloc_raw(p, 16, 8);
                }
            }
        });
    });

    #[cfg(feature = "tracking")]
    group.bench_function("Tracked<Heap>", |b| {
        let a = Tracked::new(HeapAlloc);
        b.iter(|| {
            for _ in 0..n {
                let p = a.try_alloc_raw(16, 8).unwrap();
                unsafe {
                    a.try_dealloc_raw(p, 16, 8);
                }
            }
        });
    });

    #[cfg(feature = "sync")]
    group.bench_function("SyncAlloc<Heap> (uncontended)", |b| {
        let a = SyncAlloc::new(HeapAlloc);
        b.iter(|| {
            for _ in 0..n {
                let p = a.try_alloc_raw(16, 8).unwrap();
                unsafe {
                    a.try_dealloc_raw(p, 16, 8);
                }
            }
        });
    });

    group.finish();
}

/// `BackedStack` (backed by `HeapAlloc`) vs a plain `StackAllocator`
/// directly, for the same N sequential allocations -- checks what the
/// extra indirection through a parent `RawAlloc` costs versus an owned
/// `Vec<u8>` buffer.
#[cfg(feature = "backed")]
fn bench_backed_vs_direct(c: &mut Criterion) {
    let mut group = c.benchmark_group("backed_stack_vs_direct");
    let heap = HeapAlloc;
    for &n in &SIZES {
        group.throughput(Throughput::Elements(n as u64));

        group.bench_with_input(BenchmarkId::new("StackAllocator", n), &n, |b, &n| {
            b.iter(|| {
                let s = StackAllocator::with_capacity(n as usize * 16);
                for _ in 0..n {
                    black_box(s.alloc_raw(16, 8));
                }
            });
        });

        group.bench_with_input(BenchmarkId::new("BackedStack<Heap>", n), &n, |b, &n| {
            b.iter(|| {
                let s = BackedStack::new(&heap, n as usize * 16, 8).unwrap();
                for _ in 0..n {
                    black_box(s.alloc_raw(16, 8));
                }
            });
        });
    }
    group.finish();
}

/// `BumpVec<u64, HeapAlloc>` vs `std::vec::Vec<u64>`: N sequential
/// pushes from empty, no `with_capacity`/`new_in`-plus-capacity head
/// start on either side, so this measures real growth-cycle cost, not
/// just steady-state writes.
#[cfg(feature = "bump_vec")]
fn bench_bump_vec_vs_std_vec(c: &mut Criterion) {
    let mut group = c.benchmark_group("push_sequential");
    let heap = HeapAlloc;
    for &n in &SIZES {
        group.throughput(Throughput::Elements(n as u64));

        group.bench_with_input(BenchmarkId::new("BumpVec", n), &n, |b, &n| {
            b.iter(|| {
                let mut v: BumpVec<u64, _> = BumpVec::new_in(&heap);
                for i in 0..n {
                    v.push(i as u64);
                }
                black_box(&v);
            });
        });

        group.bench_with_input(BenchmarkId::new("std::Vec", n), &n, |b, &n| {
            b.iter(|| {
                let mut v: Vec<u64> = Vec::new();
                for i in 0..n {
                    v.push(i as u64);
                }
                black_box(&v);
            });
        });
    }
    group.finish();
}

#[cfg(all(feature = "backed", feature = "bump_vec"))]
criterion_group!(
    benches,
    bench_raw_alloc,
    bench_create_destroy_churn,
    bench_combinator_overhead,
    bench_backed_vs_direct,
    bench_bump_vec_vs_std_vec
);
#[cfg(all(feature = "backed", not(feature = "bump_vec")))]
criterion_group!(
    benches,
    bench_raw_alloc,
    bench_create_destroy_churn,
    bench_combinator_overhead,
    bench_backed_vs_direct
);
#[cfg(all(not(feature = "backed"), feature = "bump_vec"))]
criterion_group!(
    benches,
    bench_raw_alloc,
    bench_create_destroy_churn,
    bench_combinator_overhead,
    bench_bump_vec_vs_std_vec
);
#[cfg(all(not(feature = "backed"), not(feature = "bump_vec")))]
criterion_group!(
    benches,
    bench_raw_alloc,
    bench_create_destroy_churn,
    bench_combinator_overhead
);
criterion_main!(benches);
