// crates/mid-platform/benches/hash_bench.rs
//! `mid_platform::hash::FixedHasher` against the two honest baselines for
//! what it is: `rustc-hash` 1.1.0 (the published crate whose algorithm
//! `hash::FxHasher` is -- verified bit-for-bit on 64-bit targets by
//! `src/hash.rs`'s own parity test, so any real gap here is codegen, not
//! algorithm) and `std`'s default `DefaultHasher` (SipHash 1-3, the
//! DoS-resistant hasher this module deliberately is *not*, see `hash.rs`'s
//! own top doc comment -- included so the "why bother" number is on the
//! page, not assumed). `rustc-hash` is a `[dev-dependencies]` entry only
//! (`crates/mid-platform/Cargo.toml`'s own comment), same pattern
//! `sync_bench.rs` uses for `spin`.
//!
//! Unlike `sync_bench.rs`, there is no `std`/`no_std` feature split to run
//! twice: `hash.rs` is pure `core` arithmetic with one code path in both
//! configurations, so this file runs once, under default features.
//!
//! What this measures, and what it deliberately doesn't
//! ------------------------------------------------------
//! - `hash_u64` / `hash_str_16b` / `hash_bytes_256b`: raw hashing cost for
//!   three key shapes (an integer, a short string, a long byte slice --
//!   different branches of `FxHasher::write`).
//! - `hashmap_insert_lookup_u64`: the same hashers used the way they
//!   actually will be -- as the `BuildHasher` of a `std::collections::
//!   HashMap`. Includes `RandomState` (std's default) as the baseline a
//!   real consumer would otherwise get.
//!
//! Not measured: collision quality/distribution (this module makes no
//! quality claim beyond "the same algorithm rustc itself uses"), and
//! adversarial-input behavior (explicitly out of scope, see `hash.rs`).
//!
//! Run
//! ---
//!   cargo bench -p mid-platform --bench hash_bench

use std::collections::HashMap;
use std::hash::{BuildHasher, BuildHasherDefault};

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use mid_platform::hash::FixedHasher;

const N: u64 = 10_000;

type RustcFx = BuildHasherDefault<rustc_hash::FxHasher>;

// ── raw hashing: u64 keys ─────────────────────────────────────────────────

fn bench_hash_u64(c: &mut Criterion) {
    let mut g = c.benchmark_group("hash_u64");

    g.bench_function("mid-platform::hash::FixedHasher", |b| {
        b.iter(|| {
            let mut acc = 0u64;
            for i in 0..N {
                acc ^= FixedHasher.hash_one(black_box(i));
            }
            black_box(acc)
        })
    });

    g.bench_function("rustc-hash 1.1.0 FxHasher", |b| {
        let s = RustcFx::default();
        b.iter(|| {
            let mut acc = 0u64;
            for i in 0..N {
                acc ^= s.hash_one(black_box(i));
            }
            black_box(acc)
        })
    });

    g.bench_function("std RandomState (SipHash 1-3)", |b| {
        let s = std::collections::hash_map::RandomState::new();
        b.iter(|| {
            let mut acc = 0u64;
            for i in 0..N {
                acc ^= s.hash_one(black_box(i));
            }
            black_box(acc)
        })
    });

    g.finish();
}

// ── raw hashing: short string keys ────────────────────────────────────────

fn bench_hash_str_16b(c: &mut Criterion) {
    let keys: Vec<String> = (0..1024).map(|i| format!("entity_{i:09}")).collect();
    assert!(keys.iter().all(|k| k.len() == 16));

    let mut g = c.benchmark_group("hash_str_16b");

    g.bench_function("mid-platform::hash::FixedHasher", |b| {
        b.iter(|| {
            let mut acc = 0u64;
            for k in &keys {
                acc ^= FixedHasher.hash_one(black_box(k.as_str()));
            }
            black_box(acc)
        })
    });

    g.bench_function("rustc-hash 1.1.0 FxHasher", |b| {
        let s = RustcFx::default();
        b.iter(|| {
            let mut acc = 0u64;
            for k in &keys {
                acc ^= s.hash_one(black_box(k.as_str()));
            }
            black_box(acc)
        })
    });

    g.bench_function("std RandomState (SipHash 1-3)", |b| {
        let s = std::collections::hash_map::RandomState::new();
        b.iter(|| {
            let mut acc = 0u64;
            for k in &keys {
                acc ^= s.hash_one(black_box(k.as_str()));
            }
            black_box(acc)
        })
    });

    g.finish();
}

// ── raw hashing: long byte slice ──────────────────────────────────────────

fn bench_hash_bytes_256b(c: &mut Criterion) {
    let blob: Vec<u8> = (0..256u32).map(|i| (i.wrapping_mul(31) ^ 0x5A) as u8).collect();

    let mut g = c.benchmark_group("hash_bytes_256b");

    g.bench_function("mid-platform::hash::FixedHasher", |b| {
        b.iter(|| black_box(FixedHasher.hash_one(black_box(blob.as_slice()))))
    });

    g.bench_function("rustc-hash 1.1.0 FxHasher", |b| {
        let s = RustcFx::default();
        b.iter(|| black_box(s.hash_one(black_box(blob.as_slice()))))
    });

    g.bench_function("std RandomState (SipHash 1-3)", |b| {
        let s = std::collections::hash_map::RandomState::new();
        b.iter(|| black_box(s.hash_one(black_box(blob.as_slice()))))
    });

    g.finish();
}

// ── as the BuildHasher of a real HashMap ──────────────────────────────────

fn bench_hashmap_insert_lookup(c: &mut Criterion) {
    let mut g = c.benchmark_group("hashmap_insert_lookup_u64");

    g.bench_function("HashMap<_, _, mid-platform FixedHasher>", |b| {
        b.iter(|| {
            let mut m: HashMap<u64, u64, FixedHasher> = HashMap::with_hasher(FixedHasher);
            for i in 0..N {
                m.insert(i, i);
            }
            let mut sum = 0u64;
            for i in 0..N {
                sum = sum.wrapping_add(*m.get(&i).unwrap());
            }
            black_box(sum)
        })
    });

    g.bench_function("HashMap<_, _, rustc-hash 1.1.0>", |b| {
        b.iter(|| {
            let mut m: HashMap<u64, u64, RustcFx> = HashMap::default();
            for i in 0..N {
                m.insert(i, i);
            }
            let mut sum = 0u64;
            for i in 0..N {
                sum = sum.wrapping_add(*m.get(&i).unwrap());
            }
            black_box(sum)
        })
    });

    g.bench_function("HashMap<_, _, std RandomState>", |b| {
        b.iter(|| {
            let mut m: HashMap<u64, u64> = HashMap::new();
            for i in 0..N {
                m.insert(i, i);
            }
            let mut sum = 0u64;
            for i in 0..N {
                sum = sum.wrapping_add(*m.get(&i).unwrap());
            }
            black_box(sum)
        })
    });

    g.finish();
}

criterion_group!(
    benches,
    bench_hash_u64,
    bench_hash_str_16b,
    bench_hash_bytes_256b,
    bench_hashmap_insert_lookup
);
criterion_main!(benches);
