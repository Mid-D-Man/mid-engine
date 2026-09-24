# mid-collections

`no_std` + `alloc` collection types: a generational-index allocator, a
sparse set, and `FfiSpan` — a type-erased, C-safe view into a Rust array for
crossing the FFI boundary safely (built on `zerocopy`).

`FfiSpan` specifically is why `mid-platform`'s own Phase 1 doesn't need to
solve the "type-erased pointer for FFI" problem again — see that crate's
own page and `docs/roadmap.md`'s Decision 3 for the full reasoning.

**Status:** in progress.
