# Design Mandates

Four rules apply to every crate in this workspace, no exceptions:

## Multiplayer-first

Network sync is baked into the ECS from day one, rather than being bolted on
later. Multiplayer isn't a feature added after the fact — it shapes how
state is structured from the start.

## FFI-ready

Every crate exposes a strict `#[repr(C)]` FFI boundary to act as a
cross-language "Middle Man." In practice: `rlib` + `cdylib` + `staticlib`
crate types, and a real `extern "C"` surface with a C header, not just a
Rust API that happens to be `#[repr(C)]`-annotated internally.

## Zero hidden abstractions

If you need to understand the memory layout, you can read the code
directly. No macro-generated indirection that obscures what's actually
stored where.

## Profile before optimize

Every performance claim cites a real, benchmarked `[RELEASE]` build number —
not a theoretical estimate. See [Benchmarks](/benchmarks/) for the numbers
behind any performance claim made elsewhere in this book.

## Zero-to-minimal external dependencies

Every core crate, no exceptions. This project has turned down its own
published `dixscript` crate as a core dependency over its transitive
dependency count — the bar is real, not aspirational. When a crate needs a
capability an external crate provides (a fast hash algorithm, a spin-based
mutex for `no_std`), the default is to check whether it can be hand-rolled
first, and to defer the decision with a named trigger condition when it
can't be decided cleanly yet. `mid-platform`'s own page is the clearest
example of this in practice.
