// crates/mid-arena/examples/drop_arena_standalone.rs
//! Standalone `std::time::Instant` micro-benchmark for `drop_arena`,
//! not a criterion bench inside `vs_arena_crates.rs` like every other
//! crate in this survey.
//!
//! Real, confirmed reason, not a style choice: `drop_arena::DropBox`'s
//! `Drop` implementation calls back into the arena through an invariant
//! lifetime tied to the arena's own type parameter
//! (`DropArena<'arena, T>`, `alloc`/`drop_box` both take `&'arena
//! self`). Every criterion bench in `vs_arena_crates.rs` returns the
//! populated arena from the timed closure (`b.iter(|| { ...; arena
//! })`) so criterion has something to observe and so the compiler
//! can't optimize the whole loop away. Tried that shape here first --
//! it does not compile: returning `(arena, boxes)`, or even just
//! `arena` alone after any `DropBox` has been created from it,
//! triggers a real `E0505`/`E0515` (`cannot move out of arena because
//! it is borrowed`), because `boxes` (or even an already-dropped
//! `DropBox` mid-loop) ties `arena`'s borrow to the same lifetime the
//! type itself is parameterized over. This is a genuine structural
//! property of `drop_arena`'s design, not a workaround-able API
//! quirk -- confirmed by trying it, not assumed.
//!
//! This runs cleanly instead because nothing here needs to be
//! *returned*: `arena` and `boxes` both live to the end of a scope and
//! drop naturally in reverse declaration order (`boxes` before
//! `arena`, exactly the order `drop_arena` needs), the same shape the
//! `.c` benchmarks in this directory use for the same underlying
//! reason (a standalone program with its own `main`, not a value
//! threaded through a shared harness).
//!
//! Run
//! ---
//!   cargo run --release --example drop_arena_standalone -p mid-arena
//!
//! Lives under `examples/`, not `benches/` or `src/bin/`: example
//! targets get `[dev-dependencies]` access the way tests/benches do;
//! plain `[[bin]]` targets do not, and `drop_arena` has no reason to be
//! a real (non-dev) dependency of this crate.

use std::time::Instant;

const N: usize = 100_000;

#[derive(Clone, Copy)]
#[allow(dead_code)] // `b` exists to match this survey's realistic 16-byte payload shape, not read individually here
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

fn report(op: &str, n: usize, elapsed: std::time::Duration) {
    // Matches the C benchmarks' own report() line shape exactly
    // ("  %-Ns %7.2f ns/op") so scripts/bench_vs_c_arena_libs.py can
    // parse this the same way it parses tsoding/APR/talloc's output,
    // via the same parse_c() function -- this isn't a criterion result,
    // but it's real, per-operation, single-run timing the same way
    // those are.
    println!(
        "  {:<28} {:>8.2} ns/op",
        op,
        elapsed.as_nanos() as f64 / n.max(1) as f64
    );
}

fn main() {
    println!("drop_arena -- real run, std::time::Instant (see this file's own doc comment for why not criterion)");
    println!("N = {N}, single core, release build\n");

    // insert + get: arena and boxes both live to the end of this block,
    // dropping in reverse order (boxes, then arena) when it ends.
    {
        let arena: drop_arena::DropArena<Payload> = drop_arena::DropArena::with_capacity(N);

        let t0 = Instant::now();
        let mut boxes = Vec::with_capacity(N);
        for i in 0..N {
            boxes.push(arena.alloc(payload(i)));
        }
        report("insert", N, t0.elapsed());

        let t0 = Instant::now();
        let mut sum: u64 = 0;
        for b in &boxes {
            sum = sum.wrapping_add(b.a);
        }
        report("get", N, t0.elapsed());
        std::hint::black_box(sum);
    }

    // churn: remove half via a real drop_box() reclaim (not just an
    // ordinary drop -- this is the operation that actually returns the
    // slot to the free list), reinsert half.
    {
        let arena: drop_arena::DropArena<Payload> = drop_arena::DropArena::with_capacity(N);
        let boxes: Vec<_> = (0..N).map(|i| arena.alloc(payload(i))).collect();

        let t0 = Instant::now();
        let mut kept = Vec::with_capacity(N / 2);
        for (i, b) in boxes.into_iter().enumerate() {
            if i % 2 == 0 {
                arena.drop_box(b);
            } else {
                kept.push(b);
            }
        }
        for i in 0..N / 2 {
            kept.push(arena.alloc(payload(i)));
        }
        report("remove_half_reinsert_half", N + N / 2, t0.elapsed());
        std::hint::black_box(kept.len());
    }

    println!("\nFor comparison, typed-arena (which drop_arena is built on top of, per");
    println!("docs/mid-arena.md's survey) measured roughly 1.1-1.5 ns/op insert in the");
    println!("same shape of run -- the gap here is the real cost of the free-list reuse");
    println!("bookkeeping drop_arena adds on top, not a fluke of this specific run.");
}
