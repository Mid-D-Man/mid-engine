# mid-math

SIMD-optimised math library for Mid Engine. Feature-complete for engine v1.

No required external runtime dependencies (one opt-in exception: `mint`, behind the
`mint` feature — see below). `no_std`-compatible core. C-ABI exports for every type.

---

## What's in here

### Float types (f32)
`Vec2` `Vec3` `Vec4` `Quat` `Mat2` `Mat3` `Mat4` `Affine3`

SSE2 on x86/x86_64. NEON stubs on aarch64 (optimisation pass pending). Scalar fallback everywhere.

### Float types (f64)
`DVec2` `DVec3` `DVec4` `DQuat` `DMat2` `DMat3` `DMat4` `DAffine3`

Scalar only. 32-byte alignment reserved for future AVX2 fast path.

### Swizzle
`.xy()` `.xzy()` `.wzyx()` ...-style component-permutation getters, plus
`.with_xy(rhs)` `.with_xyz(rhs)` ...-style same-or-narrower-width replacement
setters — `src/swizzle/`, `Vec2Swizzles` / `Vec3Swizzles` / `Vec4Swizzles`.
`engine.rs` holds the shared trait/macro definitions (numeric-family-agnostic);
one file per numeric family (`f32.rs`, `f64.rs`, ...) just invokes those macros
once per concrete type in that family.

So far: **f32** (`Vec2`, `Vec3` × every backend, `Vec4` × every backend),
**f64** (`DVec3`, `DVec2` × every backend, `DVec4` × every backend), all 8
narrow int families (**i8/u8/i16/u16/i32/u32/i64/u64**, all always-scalar and
canonical, no backend split), and the wide SIMD types — **axis-swizzle**
(`Vec3AxisSwizzle`, same-width-only) for `Vec3x4`/`Vec3x8`, and
**lane-shuffle** (`LaneShuffle4`/`8`/`16`/`32`) for the opaque single-register
wide types (`f32x4`, `i32x4`/`u32x4`/`i16x8`/`u16x8`/`i8x16`/`u8x16`, and
AVX2's wider additions). No `QuatX4` — matches this crate's existing scope,
`Quat`/`DQuat` never got swizzle either. See mint below for the interop layer.

### Boolean masks
`BVec2` `BVec3` `BVec4`

### Integer vectors (i32 / u32)
`IVec2` `IVec3` `IVec4` `UVec2` `UVec3` `UVec4`

### Integer vectors (i64 / u64)
`I64Vec2` `I64Vec3` `I64Vec4` `U64Vec2` `U64Vec3` `U64Vec4`

### Wide SIMD (f32)
`f32x4` `Vec3x4` `QuatX4` `Mask4`
`Vec3x8` (AVX2 only, gated behind `target_feature = "avx2"`)

### Wide SIMD (integer)
`i32x4` `u32x4` `i16x8` `u16x8` `i8x16` `u8x16`
`IMask4` `IMask8` `IMask16`

### Curves and splines
`CatmullRom` (uniform / centripetal / chordal)
`CubicBezier` `QuadraticBezier`
`HermiteSpline` `KochanekBartels` (TCB)
`BSpline` `CardinalSpline`

All operate on any type implementing the `Interpolate` trait: `Vec2`, `Vec3`, `f32`, `f64`, `Quat`.

### Fixed-point (deterministic simulation)
`Fixed<FRAC>` generic scalar — integer-only arithmetic, platform-identical results.
Aliases: `Fixed8` `Fixed12` `Fixed16`
Vectors: `Fixed8Vec2/3` `Fixed12Vec2/3` `Fixed16Vec2/3`

### Color
| Type | Space | When to use |
|---|---|---|
| `Color32` | sRGB u8 | GPU upload, PNG I/O |
| `Rgb` | Linear f32 | All lighting math |
| `Rgba` | Linear f32 + alpha | Compositing |
| `Hsv` `Hsl` | sRGB f32 | Color pickers |
| `Rgbe` | Linear HDR | Environment maps |
| `LogLuv32` | Perceptual HDR | Physics lighting |
| `YCbCr` | sRGB chroma | Video encode/decode |

### Coherent noise
`Perlin` `Simplex` `Value` `Worley` — all 2D/3D/4D with seed control.
`Fbm` — fractional Brownian motion over any base noise type.
`DomainWarp` — single and double-pass coordinate warping.

### Camera math
`Frustum` — Gribb-Hartmann plane extraction, point/sphere/AABB visibility tests.
`unproject` `picking_ray` — screen space → world space.
`perspective_decompose` `perspective_resize` — matrix introspection.
`perspective_infinite_rh` `perspective_reversed_z_rh` — Vulkan/DX12 projection variants.
`csm_split_depths` `sub_frustum_corners` — Cascaded Shadow Map helpers.

### Geometry primitives
`BarycentricCoords` — interpolation of f32, Vec2, Vec3, [f32;4] across triangles.
`Triangle2` — barycentric, contains, area, circumcircle, Delaunay predicate.
`Triangle3` — barycentric, Möller–Trumbore ray intersection, closest point, plane.

### Helpers
`Radians` `Degrees` — type-safe angles, no unit-mismatch bugs.
`DualQuat` — rigid body skinning with dual-linear blending (`blend2`, `blend4`).
`Rotor3` — Geometric Algebra rotation, isomorphic to quaternion.
`SpatialVelocity` `SpatialForce` `SpatialInertia` — Featherstone V6 spatial vectors.
`TangentFrame` `PackedTangent` — TBN construction and GPU packing.

### Random number generators
`Xorshift64` — 1 ns/call, good for hot loops.
`Pcg32` — statistically excellent, multiple independent streams.

### Utilities
`StringId` — compile-time FNV-1a hash. `sid!("Position")` is zero runtime cost.
`lerp` `smoothstep` `remap` `saturate` `approx_eq` — scalar helpers.

---

## mint (optional — `mint` feature)
`Into`/`From` conversions to and from the real [`mint`](https://crates.io/crates/mint)
crate — not a vendored lookalike. mint's entire value is being a *shared* type other
crates (nalgebra, ultraviolet, glam, most glTF/asset-loading crates) already depend
on, so a caller holding a real `mint::Vector3<f32>` from any of those can hand it
straight to mid-math and back — a local reimplementation would be a different Rust
type and satisfy nobody's `Into<mint::X>` bound. Confirmed genuinely tiny before
adding it (318 lines, `#![no_std]`, only optional dep is `serde`) — negligible cost
either way.

Covers `Point2`/`Point3` + `Vector2`/`Vector3`/`Vector4` for every numeric family
(f32/f64 + all 8 narrow int families), plus `Quaternion` and both `Row`- and
`Column`-major `Matrix2`/`3`/`4` for f32/f64 only — no integer quaternion or matrix
conversions, matching what the real `mint` crate itself and glam's own mint support
both actually cover. `src/features/mint_conversions/` — `vectors.rs` (one shared
macro, all 10 families) + `f32.rs`/`f64.rs` (hand-written — `Mat2`/`Mat3`/`Mat4`
don't share one internal shape closely enough for one macro to fit all three; see
that file's own doc comment for the specifics).

## C FFI

Every type above has a corresponding `C*` type and `mid_*` exported function in `src/ffi/`.
The ABI is `extern "C"` with `#[repr(C)]` layouts throughout.

```c
// Example: C consumer
CVec3 a = mid_vec3_new(1.0f, 2.0f, 3.0f);
CVec3 b = mid_vec3_new(4.0f, 5.0f, 6.0f);
CVec3 c = mid_vec3_cross(a, b);

CFrustum f = mid_frustum_from_view_proj(view_proj_matrix);
bool visible = mid_frustum_test_aabb(f, aabb_min, aabb_max);

float height = mid_fbm_simplex_sample_2d(seed, 6, 2.0f, 0.5f, 1.0f, x, z);
```

---

## Performance

All numbers from `cargo bench --release` on x86_64 with `RUSTFLAGS="-C target-cpu=native"`.

| Operation | ns/op | Notes |
|---|---|---|
| Vec3 add | ~1.5 | SSE2, parity with glam |
| Vec3 normalize | ~3.8 | SSE2 |
| Quat rotate | ~0.9 | SSE2 |
| Mat4 mul | ~7.1 | SSE2, ~2× gap vs glam (Phase 2 target) |
| Mat4 inverse | ~117 | Phase 2 target: SSE2 shuffle approach |
| 100k entity transforms | ~1.9 ns/entity | SSE2 |
| Simplex 2D | baseline | Phase 2: f32x4 vectorisation pending |
| Frustum AABB test | baseline | Phase 2: Vec3x4 batch pending |

---

## Running tests and benchmarks

```bash
# All tests
cargo test -p mid-math

# Release tests (for correctness at optimisation level)
cargo test -p mid-math --release

# All benchmarks
cargo bench -p mid-math

# Specific bench groups
cargo bench --bench noise   -p mid-math
cargo bench --bench camera  -p mid-math
cargo bench --bench geom    -p mid-math
cargo bench --bench vs_all  -p mid-math
```

---

## Phase 2 optimisation targets (upcoming)

1. NEON `float32x4_t` for Vec3/Vec4/Quat/Mat4 on aarch64
2. WASM `v128` for Vec3/Vec4/Quat/Mat4 on wasm32
3. SSE2 shuffle-based Mat4 general inverse (~6× gap vs glam)
4. `f32x4` vectorised noise batch sampling
5. `Vec3x4` frustum AABB batch culling (4 AABBs per SSE2 instruction)
6. Swizzle getters currently always go through `<Output>::new(...)`; SSE2/NEON/WASM/coresimd
   could do same-width swizzles in one shuffle instruction instead (glam's approach) — needs each
   backend's shuffle-immediate encoding and `Vec3`'s padding-lane behaviour verified with a real
   compiler before it's safe to generate across ~500+ call sites per backend
7. `LaneShuffle4/8/16/32::shuffle()` takes a runtime `[usize; N]` (checked via `to_array()` +
   indexing, not a single shuffle instruction) for the same no-compiler reason as #6 — a
   const-generic-per-lane version (mirroring how `_mm_shuffle_ps`/`i32x4_shuffle` themselves take
   compile-time lane indices) would let the compiler fold `shuffle()` into one instruction for a
   fixed permutation known at compile time, instead of an array build + indirect indexing
