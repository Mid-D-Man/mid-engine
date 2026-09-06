// crates/mid-arena/benches/vs_arena_crates.rs
//! mid-arena's own `SlotArena<T>` (and `BumpArena<T>`, behind the `bump`
//! feature — run with `--features bump` to include it) against the 10
//! real Rust arena crates actually surveyed and benched for
//! docs/mid-arena.md — same N, same payload shape, same operations,
//! converted from the `std::time::Instant` version that produced that
//! doc's recorded SlotArena numbers. `BumpArena`'s entries here were
//! never run through the Instant-based version first; they're written
//! directly against the same real, unit-tested API (25/25 tests passing,
//! docs/mid-arena.md's "What's built") rather than ported from a
//! sandbox pass that doesn't exist for this one.
//!
//! Run
//! ---
//!   cargo bench --bench vs_arena_crates -p mid-arena --features bump

use criterion::{black_box, criterion_group, criterion_main, Criterion};

const N: usize = 100_000;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct Payload {
    a: u64,
    b: u64,
}

fn payload(i: usize) -> Payload {
    Payload {
        a: i as u64,
        b: (i as u64).wrapping_mul(2654435761),
    }
}

// ── insert ──────────────────────────────────────────────────────────────

fn bench_insert(c: &mut Criterion) {
    let mut g = c.benchmark_group("insert");

    g.bench_function("mid-arena/SlotArena", |b| {
        b.iter(|| {
            let mut a = mid_arena::SlotArena::with_capacity(N);
            for i in 0..N {
                black_box(a.insert(payload(i)));
            }
            a
        })
    });

    #[cfg(feature = "compact")]
    g.bench_function("mid-arena/CompactSlotArena", |b| {
        b.iter(|| {
            let mut a = mid_arena::CompactSlotArena::with_capacity(N);
            for i in 0..N {
                black_box(a.insert(payload(i)));
            }
            a
        })
    });

    g.bench_function("slab", |b| {
        b.iter(|| {
            let mut s: slab::Slab<Payload> = slab::Slab::with_capacity(N);
            for i in 0..N {
                black_box(s.insert(payload(i)));
            }
            s
        })
    });

    g.bench_function("slotmap", |b| {
        b.iter(|| {
            let mut sm: slotmap::SlotMap<slotmap::DefaultKey, Payload> =
                slotmap::SlotMap::with_capacity(N);
            for i in 0..N {
                black_box(sm.insert(payload(i)));
            }
            sm
        })
    });

    g.bench_function("generational-arena", |b| {
        b.iter(|| {
            let mut a: generational_arena::Arena<Payload> =
                generational_arena::Arena::with_capacity(N);
            for i in 0..N {
                black_box(a.insert(payload(i)));
            }
            a
        })
    });

    g.bench_function("typed-generational-arena", |b| {
        b.iter(|| {
            let mut a: typed_generational_arena::StandardArena<Payload> =
                typed_generational_arena::StandardArena::with_capacity(N);
            for i in 0..N {
                black_box(a.insert(payload(i)));
            }
            a
        })
    });

    g.bench_function("atomic-arena", |b| {
        b.iter(|| {
            let mut a: atomic_arena::Arena<Payload> = atomic_arena::Arena::new(N);
            for i in 0..N {
                black_box(a.insert(payload(i)).expect("arena sized exactly for N inserts"));
            }
            a
        })
    });

    g.bench_function("id-arena", |b| {
        b.iter(|| {
            let mut a: id_arena::Arena<Payload> = id_arena::Arena::with_capacity(N);
            for i in 0..N {
                black_box(a.alloc(payload(i)));
            }
            a
        })
    });

    g.bench_function("thunderdome", |b| {
        b.iter(|| {
            let mut a: thunderdome::Arena<Payload> = thunderdome::Arena::with_capacity(N);
            for i in 0..N {
                black_box(a.insert(payload(i)));
            }
            a
        })
    });

    g.bench_function("sharded-slab", |b| {
        b.iter(|| {
            let s: sharded_slab::Slab<Payload> = sharded_slab::Slab::new();
            for i in 0..N {
                black_box(s.insert(payload(i)).expect("shard not full"));
            }
            s
        })
    });

    g.bench_function("bumpalo", |b| {
        b.iter(|| {
            let bump = bumpalo::Bump::with_capacity(N * std::mem::size_of::<Payload>());
            for i in 0..N {
                black_box(bump.alloc(payload(i)));
            }
            bump
        })
    });

    g.bench_function("typed-arena", |b| {
        b.iter(|| {
            let arena: typed_arena::Arena<Payload> = typed_arena::Arena::with_capacity(N);
            for i in 0..N {
                black_box(arena.alloc(payload(i)));
            }
            arena
        })
    });

    #[cfg(feature = "bump")]
    g.bench_function("mid-arena/BumpArena", |b| {
        b.iter(|| {
            let arena: mid_arena::BumpArena<Payload> = mid_arena::BumpArena::with_capacity(N);
            for i in 0..N {
                black_box(arena.alloc(payload(i)));
            }
            arena
        })
    });

    g.bench_function("internment/ArenaIntern (unique)", |b| {
        b.iter(|| {
            let arena: internment::Arena<Payload> = internment::Arena::new();
            for i in 0..N {
                black_box(arena.intern(payload(i)));
            }
            arena
        })
    });

    g.finish();
}

// ── get ─────────────────────────────────────────────────────────────────

fn bench_get(c: &mut Criterion) {
    let mut g = c.benchmark_group("get");

    {
        let mut a = mid_arena::SlotArena::with_capacity(N);
        let keys: Vec<_> = (0..N).map(|i| a.insert(payload(i))).collect();
        g.bench_function("mid-arena/SlotArena", |b| {
            b.iter(|| {
                let mut sum = 0u64;
                for &k in &keys {
                    sum = sum.wrapping_add(a.get(k).unwrap().a);
                }
                black_box(sum)
            })
        });
    }

    #[cfg(feature = "compact")]
    {
        let mut a = mid_arena::CompactSlotArena::with_capacity(N);
        let keys: Vec<_> = (0..N).map(|i| a.insert(payload(i))).collect();
        g.bench_function("mid-arena/CompactSlotArena", |b| {
            b.iter(|| {
                let mut sum = 0u64;
                for &k in &keys {
                    sum = sum.wrapping_add(a.get(k).unwrap().a);
                }
                black_box(sum)
            })
        });
    }

    {
        let mut s: slab::Slab<Payload> = slab::Slab::with_capacity(N);
        let keys: Vec<_> = (0..N).map(|i| s.insert(payload(i))).collect();
        g.bench_function("slab", |b| {
            b.iter(|| {
                let mut sum = 0u64;
                for &k in &keys {
                    sum = sum.wrapping_add(s[k].a);
                }
                black_box(sum)
            })
        });
    }

    {
        let mut sm: slotmap::SlotMap<slotmap::DefaultKey, Payload> =
            slotmap::SlotMap::with_capacity(N);
        let keys: Vec<_> = (0..N).map(|i| sm.insert(payload(i))).collect();
        g.bench_function("slotmap", |b| {
            b.iter(|| {
                let mut sum = 0u64;
                for &k in &keys {
                    sum = sum.wrapping_add(sm[k].a);
                }
                black_box(sum)
            })
        });
    }

    {
        let mut arena: generational_arena::Arena<Payload> =
            generational_arena::Arena::with_capacity(N);
        let idxs: Vec<_> = (0..N).map(|i| arena.insert(payload(i))).collect();
        g.bench_function("generational-arena", |b| {
            b.iter(|| {
                let mut sum = 0u64;
                for &idx in &idxs {
                    sum = sum.wrapping_add(arena[idx].a);
                }
                black_box(sum)
            })
        });
    }

    {
        let mut arena: typed_generational_arena::StandardArena<Payload> =
            typed_generational_arena::StandardArena::with_capacity(N);
        let idxs: Vec<_> = (0..N).map(|i| arena.insert(payload(i))).collect();
        g.bench_function("typed-generational-arena", |b| {
            b.iter(|| {
                let mut sum = 0u64;
                for &idx in &idxs {
                    sum = sum.wrapping_add(arena[idx].a);
                }
                black_box(sum)
            })
        });
    }

    {
        let mut arena: atomic_arena::Arena<Payload> = atomic_arena::Arena::new(N);
        let keys: Vec<_> = (0..N)
            .map(|i| arena.insert(payload(i)).expect("arena sized exactly for N inserts"))
            .collect();
        g.bench_function("atomic-arena", |b| {
            b.iter(|| {
                let mut sum = 0u64;
                for &k in &keys {
                    sum = sum.wrapping_add(arena.get(k).unwrap().a);
                }
                black_box(sum)
            })
        });
    }

    {
        let mut arena: id_arena::Arena<Payload> = id_arena::Arena::with_capacity(N);
        let ids: Vec<_> = (0..N).map(|i| arena.alloc(payload(i))).collect();
        g.bench_function("id-arena", |b| {
            b.iter(|| {
                let mut sum = 0u64;
                for &id in &ids {
                    sum = sum.wrapping_add(arena[id].a);
                }
                black_box(sum)
            })
        });
    }

    {
        let mut arena: thunderdome::Arena<Payload> = thunderdome::Arena::with_capacity(N);
        let idxs: Vec<_> = (0..N).map(|i| arena.insert(payload(i))).collect();
        g.bench_function("thunderdome", |b| {
            b.iter(|| {
                let mut sum = 0u64;
                for &idx in &idxs {
                    sum = sum.wrapping_add(arena[idx].a);
                }
                black_box(sum)
            })
        });
    }

    {
        let s: sharded_slab::Slab<Payload> = sharded_slab::Slab::new();
        let keys: Vec<_> = (0..N)
            .map(|i| s.insert(payload(i)).expect("shard not full"))
            .collect();
        g.bench_function("sharded-slab", |b| {
            b.iter(|| {
                let mut sum = 0u64;
                for &k in &keys {
                    sum = sum.wrapping_add(s.get(k).unwrap().a);
                }
                black_box(sum)
            })
        });
    }

    {
        let bump = bumpalo::Bump::with_capacity(N * std::mem::size_of::<Payload>());
        let refs: Vec<&mut Payload> = (0..N).map(|i| bump.alloc(payload(i))).collect();
        g.bench_function("bumpalo", |b| {
            b.iter(|| {
                let mut sum = 0u64;
                for r in &refs {
                    sum = sum.wrapping_add(r.a);
                }
                black_box(sum)
            })
        });
    }

    {
        let arena: typed_arena::Arena<Payload> = typed_arena::Arena::with_capacity(N);
        let refs: Vec<&mut Payload> = (0..N).map(|i| arena.alloc(payload(i))).collect();
        g.bench_function("typed-arena", |b| {
            b.iter(|| {
                let mut sum = 0u64;
                for r in &refs {
                    sum = sum.wrapping_add(r.a);
                }
                black_box(sum)
            })
        });
    }

    #[cfg(feature = "bump")]
    {
        let arena: mid_arena::BumpArena<Payload> = mid_arena::BumpArena::with_capacity(N);
        let refs: Vec<&mut Payload> = (0..N).map(|i| arena.alloc(payload(i))).collect();
        g.bench_function("mid-arena/BumpArena", |b| {
            b.iter(|| {
                let mut sum = 0u64;
                for r in &refs {
                    sum = sum.wrapping_add(r.a);
                }
                black_box(sum)
            })
        });
    }

    g.finish();
}

// ── remove_half / reinsert_half (arenas that support reuse only) ──────────

fn bench_churn(c: &mut Criterion) {
    let mut g = c.benchmark_group("remove_half_then_reinsert_half");

    g.bench_function("mid-arena/SlotArena", |b| {
        b.iter(|| {
            let mut a = mid_arena::SlotArena::with_capacity(N);
            let keys: Vec<_> = (0..N).map(|i| a.insert(payload(i))).collect();
            for &k in keys.iter().step_by(2) {
                a.remove(k);
            }
            for i in 0..N / 2 {
                black_box(a.insert(payload(i)));
            }
            a
        })
    });

    #[cfg(feature = "compact")]
    g.bench_function("mid-arena/CompactSlotArena", |b| {
        b.iter(|| {
            let mut a = mid_arena::CompactSlotArena::with_capacity(N);
            let keys: Vec<_> = (0..N).map(|i| a.insert(payload(i))).collect();
            for &k in keys.iter().step_by(2) {
                a.remove(k);
            }
            for i in 0..N / 2 {
                black_box(a.insert(payload(i)));
            }
            a
        })
    });

    g.bench_function("slab", |b| {
        b.iter(|| {
            let mut s: slab::Slab<Payload> = slab::Slab::with_capacity(N);
            let keys: Vec<_> = (0..N).map(|i| s.insert(payload(i))).collect();
            for &k in keys.iter().step_by(2) {
                s.remove(k);
            }
            for i in 0..N / 2 {
                black_box(s.insert(payload(i)));
            }
            s
        })
    });

    g.bench_function("slotmap", |b| {
        b.iter(|| {
            let mut sm: slotmap::SlotMap<slotmap::DefaultKey, Payload> =
                slotmap::SlotMap::with_capacity(N);
            let keys: Vec<_> = (0..N).map(|i| sm.insert(payload(i))).collect();
            for &k in keys.iter().step_by(2) {
                sm.remove(k);
            }
            for i in 0..N / 2 {
                black_box(sm.insert(payload(i)));
            }
            sm
        })
    });

    g.bench_function("generational-arena", |b| {
        b.iter(|| {
            let mut a: generational_arena::Arena<Payload> =
                generational_arena::Arena::with_capacity(N);
            let idxs: Vec<_> = (0..N).map(|i| a.insert(payload(i))).collect();
            for &idx in idxs.iter().step_by(2) {
                a.remove(idx);
            }
            for i in 0..N / 2 {
                black_box(a.insert(payload(i)));
            }
            a
        })
    });

    g.bench_function("typed-generational-arena", |b| {
        b.iter(|| {
            let mut a: typed_generational_arena::StandardArena<Payload> =
                typed_generational_arena::StandardArena::with_capacity(N);
            let idxs: Vec<_> = (0..N).map(|i| a.insert(payload(i))).collect();
            for &idx in idxs.iter().step_by(2) {
                a.remove(idx);
            }
            for i in 0..N / 2 {
                black_box(a.insert(payload(i)));
            }
            a
        })
    });

    g.bench_function("atomic-arena", |b| {
        b.iter(|| {
            let mut a: atomic_arena::Arena<Payload> = atomic_arena::Arena::new(N);
            let keys: Vec<_> = (0..N)
                .map(|i| a.insert(payload(i)).expect("arena sized exactly for N inserts"))
                .collect();
            for &k in keys.iter().step_by(2) {
                a.remove(k);
            }
            for i in 0..N / 2 {
                black_box(a.insert(payload(i)).expect("removed half leaves room for N/2 more"));
            }
            a
        })
    });

    g.bench_function("thunderdome", |b| {
        b.iter(|| {
            let mut a: thunderdome::Arena<Payload> = thunderdome::Arena::with_capacity(N);
            let idxs: Vec<_> = (0..N).map(|i| a.insert(payload(i))).collect();
            for &idx in idxs.iter().step_by(2) {
                a.remove(idx);
            }
            for i in 0..N / 2 {
                black_box(a.insert(payload(i)));
            }
            a
        })
    });

    g.bench_function("sharded-slab (remove only, no reinsert measured)", |b| {
        b.iter(|| {
            let s: sharded_slab::Slab<Payload> = sharded_slab::Slab::new();
            let keys: Vec<_> = (0..N)
                .map(|i| s.insert(payload(i)).expect("shard not full"))
                .collect();
            for &k in keys.iter().step_by(2) {
                s.remove(k);
            }
            s
        })
    });

    // bumpalo, typed-arena, id-arena: no per-item remove/reuse API — see
    // docs/mid-arena.md's comparison table ("Reuse Mem" column) rather than
    // benching an operation these crates don't offer.

    g.finish();
}

// ── gc: alloc + force_collect (doesn't share the remove/reinsert shape) ───

mod gc_bench {
    use super::{black_box, Criterion, N};
    use gc::{Finalize, Gc, Trace};

    #[derive(Trace, Finalize, Clone)]
    struct GcPayload {
        a: u64,
        b: u64,
    }

    pub fn run(c: &mut Criterion) {
        let mut g = c.benchmark_group("gc");

        g.bench_function("gc/alloc", |b| {
            b.iter(|| {
                let mut refs = Vec::with_capacity(N);
                for i in 0..N {
                    refs.push(Gc::new(GcPayload {
                        a: i as u64,
                        b: (i as u64).wrapping_mul(2654435761),
                    }));
                }
                black_box(refs)
            })
        });

        let refs: Vec<_> = (0..N)
            .map(|i| {
                Gc::new(GcPayload {
                    a: i as u64,
                    b: (i as u64).wrapping_mul(2654435761),
                })
            })
            .collect();
        g.bench_function("gc/force_collect (all live)", |b| {
            b.iter(|| {
                gc::force_collect();
                black_box(&refs);
            })
        });

        g.finish();
    }
}

criterion_group!(benches, bench_insert, bench_get, bench_churn, gc_bench::run);
criterion_main!(benches);
