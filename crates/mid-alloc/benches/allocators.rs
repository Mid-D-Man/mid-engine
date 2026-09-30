// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-alloc.md, section "Benches"
// ============================================================================
//! Criterion benchmarks for `mid-alloc`, one group per comparison.
//!
//! - `raw_alloc_sequential`: `StackAllocator` vs `bumpalo::Bump`, N raw
//!   16-byte allocations, allocator built inside every iteration.
//! - `raw_alloc_reset`: the same comparison with each allocator built
//!   once and reset per iteration, so only the bump path is timed.
//! - `create_destroy_churn`: `PoolAllocator` vs `Box`, N create/destroy
//!   (new/drop) cycles, one at a time.
//! - `combinator_dispatch_overhead`: `FallbackAllocator`, `Segregator`,
//!   `Tracked` and `SyncAlloc` vs the plain `HeapAlloc` they wrap.
//! - `backed_stack_vs_direct`: `BackedStack<HeapAlloc>` vs
//!   `StackAllocator`.
//! - `push_sequential`: `BumpVec<u64, HeapAlloc>` vs `std::vec::Vec<u64>`,
//!   N pushes from empty.
//! - `push_in_arena`: `BumpVec<u64, &StackAllocator>` vs
//!   `bumpalo::collections::Vec<u64>` vs `std::vec::Vec<u64>`, N pushes
//!   from empty, arena built outside the timed region.
//!
//! Optional-feature entries are gated with `#[cfg(feature = "...")]`
//! inside each group, so `cargo bench` builds under any feature subset
//! and just shows fewer bars. Each entry builds its own allocator inside
//! the timed closure unless noted (`raw_alloc_reset` and `push_in_arena`
//! build outside it), so the other groups include construction cost.
//! Results go through `black_box` so the compiler cannot remove the work
//! being timed.
//!
//! Run: `cargo bench -p mid-alloc --all-features --bench allocators`

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
/// allocations, in two groups.
///
/// - `raw_alloc_sequential` builds a fresh, generously pre-sized
///   allocator inside every iteration. `StackAllocator::with_capacity`
///   zero-fills its buffer and `bumpalo` does not, so this group
///   includes a cost only one side pays.
/// - `raw_alloc_reset` builds each allocator once, outside the timed
///   region, and resets it at the start of every iteration, so it
///   measures the bump path alone. The arena has 64 bytes of slack so
///   no allocation can run out of room.
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

    let mut group = c.benchmark_group("raw_alloc_reset");
    for &n in &SIZES {
        group.throughput(Throughput::Elements(n as u64));
        let arena_bytes = n as usize * 16 + 64;

        group.bench_with_input(BenchmarkId::new("StackAllocator", n), &n, |b, &n| {
            let mut stack = StackAllocator::with_capacity(arena_bytes);
            b.iter(|| {
                stack.reset();
                for _ in 0..n {
                    black_box(stack.alloc_raw(16, 8));
                }
            });
        });

        group.bench_with_input(BenchmarkId::new("bumpalo::Bump", n), &n, |b, &n| {
            let mut bump = bumpalo::Bump::with_capacity(arena_bytes);
            let layout = std::alloc::Layout::from_size_align(16, 8).unwrap();
            b.iter(|| {
                bump.reset();
                for _ in 0..n {
                    black_box(bump.alloc_layout(layout));
                }
            });
        });
    }
    group.finish();
}

/// `PoolAllocator<[u64; 4]>` vs `Box<[u64; 4]>`: N create+destroy (or
/// new+drop) cycles, one at a time, the churn pattern both are meant to
/// handle. Both sides pass the live value through `black_box` by value
/// (the `&mut T` for the pool, the `Box` itself for `Box`), so neither
/// allocation can be elided.
fn bench_create_destroy_churn(c: &mut Criterion) {
    let mut group = c.benchmark_group("create_destroy_churn");
    for &n in &SIZES {
        group.throughput(Throughput::Elements(n as u64));

        #[cfg(feature = "pool")]
        group.bench_with_input(BenchmarkId::new("PoolAllocator", n), &n, |b, &n| {
            b.iter(|| {
                let pool = PoolAllocator::<[u64; 4]>::new();
                for i in 0..n {
                    let v = black_box(pool.create([i as u64; 4]).unwrap());
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
                    let v = black_box(Box::new([i as u64; 4]));
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
                let p = black_box(a.try_alloc_raw(16, 8).unwrap());
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
                let p = black_box(a.try_alloc_raw(16, 8).unwrap());
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
                let p = black_box(a.try_alloc_raw(16, 8).unwrap());
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
                let p = black_box(a.try_alloc_raw(16, 8).unwrap());
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
                let p = black_box(a.try_alloc_raw(16, 8).unwrap());
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

/// Two groups of N sequential pushes from empty, no head-start capacity
/// on any side, so both measure real growth-cycle cost.
///
/// - `push_sequential`: `BumpVec<u64, HeapAlloc>` vs `std::vec::Vec<u64>`.
///   This measures `BumpVec`'s growth path over the global allocator, not
///   an arena.
/// - `push_in_arena`: `BumpVec<u64, &StackAllocator>` vs
///   `bumpalo::collections::Vec<u64>` vs `std::vec::Vec<u64>`. Each arena
///   is built once, outside the timed region, and reset at the start of
///   every iteration, so the group does not include the `StackAllocator`
///   constructor's zero-fill. The arena holds the final doubled capacity,
///   so no push can run out of room.
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

    let mut group = c.benchmark_group("push_in_arena");
    for &n in &SIZES {
        group.throughput(Throughput::Elements(n as u64));
        let final_cap = (n as usize).next_power_of_two().max(4);
        let arena_bytes = final_cap * core::mem::size_of::<u64>() + 64;

        group.bench_with_input(
            BenchmarkId::new("BumpVec<StackAllocator>", n),
            &n,
            |b, &n| {
                let mut stack = StackAllocator::with_capacity(arena_bytes);
                b.iter(|| {
                    stack.reset();
                    let mut v: BumpVec<u64, _> = BumpVec::new_in(&stack);
                    for i in 0..n {
                        v.push(i as u64);
                    }
                    black_box(&v);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("bumpalo::collections::Vec", n),
            &n,
            |b, &n| {
                let mut bump = bumpalo::Bump::with_capacity(arena_bytes);
                b.iter(|| {
                    bump.reset();
                    let mut v = bumpalo::collections::Vec::new_in(&bump);
                    for i in 0..n {
                        v.push(i as u64);
                    }
                    black_box(&v);
                });
            },
        );

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
