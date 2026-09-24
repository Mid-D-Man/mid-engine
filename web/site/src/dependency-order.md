# Crate Dependency Order

```text
mid-math        (no engine deps — pure math foundation)
mid-ptr         (no engine deps — type-erased pointer wrappers)
mid-platform    (no engine deps — Mutex/Arc/atomics/cells)
mid-collections (no engine deps — SparseSet, generational index, FFI span)
mid-arena       (no engine deps — bump/slot/compact arenas)
mid-alloc       (no engine deps — custom allocators, SpinLock)
mid-common      (uses mid-math — shared traits and error types)
mid-log         (uses mid-common)
mid-trace       (uses mid-common)
mid-geom        (uses mid-math — geometric algorithms)
mid-ecs         (uses mid-math, mid-common, mid-collections)
mid-net         (uses mid-math, mid-common)
mid-physics     (uses mid-math, mid-geom)
mid-anim        (uses mid-math, mid-ecs)
```

The crates with no engine-internal dependency at all (`mid-math`, `mid-ptr`,
`mid-platform`, `mid-collections`, `mid-arena`, `mid-alloc`) are the ones
safest to work on in parallel — nothing else in the workspace has to land
first for one of them to be usable, and nothing about them can be blocked by
another crate's own unfinished state.

See each crate's own page for what it actually depends on today versus this
list's target shape — a couple of these (`mid-physics`, `mid-anim`,
`mid-app`, `mid-time`) currently exist as early v0 stubs rather than
finished crates.
