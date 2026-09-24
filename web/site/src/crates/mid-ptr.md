# mid-ptr

Type-erased raw pointer wrappers, ported from Bevy's `bevy_ptr` crate
(MIT/Apache-2.0) and adapted to this workspace's own conventions. `no_std`,
zero dependencies.

## What it gives you

- **`Ptr`, `PtrMut`, `OwningPtr`** — type-erased pointers mirroring `&T`,
  `&mut T`, and `Box<T>` respectively, minus the compile-time type
  information. Useful for storing heterogeneous data (different component
  types in one column, for instance) without generics at the storage layer.
- **`MovingPtr`** — moves a value to a new location without ever passing it
  by value, and can be deconstructed into per-field `MovingPtr`s via the
  `deconstruct_moving_ptr!` macro. Useful for migrating a value's bytes
  between two locations (an archetype table column, say) without an extra
  copy.
- **`ThinSlicePtr`** — a `&[T]` with the length stripped out, for callers
  that already track the length separately.
- **`Aligned`/`Unaligned`** — marker types threaded through all of the
  above, so a pointer's alignment guarantee (or lack of one) is visible in
  its type rather than only in a doc comment.

## Full port, including `MovingPtr`

This is a close port of the real, current upstream source, not a redesign —
including the newer `MovingPtr` + field-deconstruction machinery, which is
genuinely the more novel, riskier-to-hand-verify half of the crate. Built on
direct instruction, after surfacing that the project's own roadmap had
originally deferred this crate.

## FFI

Not yet exposed over a C boundary — a known, tracked gap against this
project's own FFI-ready mandate, not an oversight left undocumented. See the
repository's `docs/mid-ptr.md` for the current status.

## Status

See the [Tests](/tests/) page for the latest CI run, or run
`mid-ptr — Tests` from the Actions tab yourself.
