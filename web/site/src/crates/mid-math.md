# mid-math

SIMD-dispatched vector and matrix math library: `Vec2`/`Vec3`/`Vec4`,
`Quat`, `Mat3`/`Mat4`, strictly 16-byte-aligned `#[repr(C)]` primitives for
FFI safety. Includes a hand-rolled `MidVec<T, N>` small-vector container
(union + `MaybeUninit`) used by curve types and cascaded shadow maps.

Zero external dependencies, comprehensive benchmarking infrastructure
(see [Benchmarks](/benchmarks/)). The second crate in the workspace (after
`mid-math` itself, chronologically first) to opt into
`[lints] workspace = true` and take on real `unsafe` for its SIMD
intrinsics.

**Status:** practically done, second optimization pass planned.
