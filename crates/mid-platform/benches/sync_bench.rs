// crates/mid-platform/benches/sync_bench.rs
//! `mid_platform::sync::Mutex` and `mid_platform::sync::RwLock` against the
//! two real, honest baselines this crate's own no_std fallback algorithms
//! were grounded in (`spin`, read fresh from its real 0.10.0 source when
//! each module was written -- see `sync/mutex.rs`, `sync/rwlock.rs`'s own
//! doc comments) and against `std::sync`'s equivalents. Comparison crates
//! are `[dev-dependencies]` only (`crates/mid-platform/Cargo.toml`'s own
//! comment), never promoted to real dependencies -- same pattern
//! `mid-arena/benches/vs_arena_crates.rs` and `mid-alloc/benches/allocators.rs`
//! already use.
//!
//! What this measures, and what it deliberately doesn't
//! ------------------------------------------------------
//! Every group here is single-threaded, repeated-acquire-and-release
//! throughput -- the uncontended cost of `lock()`/`try_lock()`/`read()`/
//! `write()` themselves, not scaling under real multi-threaded contention.
//! Real multi-threaded correctness (no lost updates, no torn reads) is
//! already covered by each type's own test suite
//! (`many_real_threads_racing_*` in `sync/mutex.rs` and `sync/rwlock.rs`) --
//! not duplicated here as a criterion group, since spawning real OS threads
//! inside a `b.iter()` closure is a different, more elaborate benchmark
//! shape this workspace hasn't established a pattern for anywhere yet, and
//! the uncontended number is the one that actually answers "did the
//! hand-rolled fallback leave obvious performance on the table."
//!
//! `cell::{SyncCell, SyncUnsafeCell}` are intentionally not benched here --
//! zero-cost wrappers with no runtime logic, a benchmark would just measure
//! noise (`docs/mid-platform.md`'s own benchmarking-task note agrees). Once/
//! OnceLock/LazyLock/Barrier are also not separately benched: `OnceLock`'s
//! own fast path is one `Acquire` load, not meaningfully distinct from what
//! `mutex_lock_unlock` already shows for the underlying lock's acquire cost
//! on its slow path; `Barrier` is `docs/mid-platform.md`'s own
//! lowest-priority, rarely-used primitive of this group.
//!
//! `mid-platform`'s `std`/`alloc` cargo features change what
//! `mid_platform::sync::{Mutex, RwLock}` actually compile to (see
//! `sync/mutex.rs` and `sync/rwlock.rs`'s own top doc comments), so this
//! file is meant to run twice, matching `.github/workflows/mid-platform-test.yml`'s
//! own established two-configuration pattern:
//!
//! - **Default features** (`std` on): `mid_platform::sync::Mutex`/`RwLock`
//!   are a direct `std::sync` passthrough, so this run is mostly a sanity
//!   check -- it should track the explicit `std::sync::*` baseline group
//!   near-identically, confirming the passthrough really is zero-cost, not
//!   the run that answers anything new about this crate's own code.
//! - **`--no-default-features`**: `mid_platform::sync::Mutex`/`RwLock` are
//!   this crate's own hand-rolled spin-based fallback -- the run that
//!   actually exercises what this file exists to measure.
//!
//! Run
//! ---
//!   cargo bench -p mid-platform --bench sync_bench
//!   cargo bench -p mid-platform --no-default-features --bench sync_bench

use criterion::{black_box, criterion_group, criterion_main, Criterion};

const N: u64 = 10_000;

// ── mutex: lock/unlock throughput ─────────────────────────────────────────

fn bench_mutex_lock_unlock(c: &mut Criterion) {
    let mut g = c.benchmark_group("mutex_lock_unlock");

    g.bench_function("mid-platform::sync::Mutex", |b| {
        let m = mid_platform::sync::Mutex::new(0u64);
        b.iter(|| {
            for _ in 0..N {
                *m.lock().unwrap() += 1;
            }
            black_box(*m.lock().unwrap())
        })
    });

    g.bench_function("spin::Mutex", |b| {
        let m = spin::Mutex::new(0u64);
        b.iter(|| {
            for _ in 0..N {
                *m.lock() += 1;
            }
            black_box(*m.lock())
        })
    });

    g.bench_function("std::sync::Mutex", |b| {
        let m = std::sync::Mutex::new(0u64);
        b.iter(|| {
            for _ in 0..N {
                *m.lock().unwrap() += 1;
            }
            black_box(*m.lock().unwrap())
        })
    });

    g.finish();
}

// ── mutex: try_lock throughput (always succeeds -- no real contention) ────

fn bench_mutex_try_lock(c: &mut Criterion) {
    let mut g = c.benchmark_group("mutex_try_lock");

    g.bench_function("mid-platform::sync::Mutex", |b| {
        let m = mid_platform::sync::Mutex::new(0u64);
        b.iter(|| {
            for _ in 0..N {
                *m.try_lock().unwrap() += 1;
            }
            black_box(*m.try_lock().unwrap())
        })
    });

    g.bench_function("spin::Mutex", |b| {
        let m = spin::Mutex::new(0u64);
        b.iter(|| {
            for _ in 0..N {
                *m.try_lock().unwrap() += 1;
            }
            black_box(*m.try_lock().unwrap())
        })
    });

    g.bench_function("std::sync::Mutex", |b| {
        let m = std::sync::Mutex::new(0u64);
        b.iter(|| {
            for _ in 0..N {
                *m.try_lock().unwrap() += 1;
            }
            black_box(*m.try_lock().unwrap())
        })
    });

    g.finish();
}

// ── rwlock: read-heavy throughput ──────────────────────────────────────────

fn bench_rwlock_read(c: &mut Criterion) {
    let mut g = c.benchmark_group("rwlock_read");

    g.bench_function("mid-platform::sync::RwLock", |b| {
        let l = mid_platform::sync::RwLock::new(0u64);
        b.iter(|| {
            let mut sum = 0u64;
            for _ in 0..N {
                sum = sum.wrapping_add(*l.read().unwrap());
            }
            black_box(sum)
        })
    });

    g.bench_function("spin::RwLock", |b| {
        let l = spin::RwLock::new(0u64);
        b.iter(|| {
            let mut sum = 0u64;
            for _ in 0..N {
                sum = sum.wrapping_add(*l.read());
            }
            black_box(sum)
        })
    });

    g.bench_function("std::sync::RwLock", |b| {
        let l = std::sync::RwLock::new(0u64);
        b.iter(|| {
            let mut sum = 0u64;
            for _ in 0..N {
                sum = sum.wrapping_add(*l.read().unwrap());
            }
            black_box(sum)
        })
    });

    g.finish();
}

// ── rwlock: write-heavy throughput ─────────────────────────────────────────

fn bench_rwlock_write(c: &mut Criterion) {
    let mut g = c.benchmark_group("rwlock_write");

    g.bench_function("mid-platform::sync::RwLock", |b| {
        let l = mid_platform::sync::RwLock::new(0u64);
        b.iter(|| {
            for _ in 0..N {
                *l.write().unwrap() += 1;
            }
            black_box(*l.write().unwrap())
        })
    });

    g.bench_function("spin::RwLock", |b| {
        let l = spin::RwLock::new(0u64);
        b.iter(|| {
            for _ in 0..N {
                *l.write() += 1;
            }
            black_box(*l.write())
        })
    });

    g.bench_function("std::sync::RwLock", |b| {
        let l = std::sync::RwLock::new(0u64);
        b.iter(|| {
            for _ in 0..N {
                *l.write().unwrap() += 1;
            }
            black_box(*l.write().unwrap())
        })
    });

    g.finish();
}

criterion_group!(
    benches,
    bench_mutex_lock_unlock,
    bench_mutex_try_lock,
    bench_rwlock_read,
    bench_rwlock_write
);
criterion_main!(benches);
