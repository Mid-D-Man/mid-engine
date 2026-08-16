# Mid Engine Architecture

Modular Anti-Engine — each crate is independently publishable.

## Crates

| Crate | Role | Status |
|---|---|---|
| mid-math | SIMD numerics, f32/f64, zero external deps | f32 optimization complete, f64 optimization complete (this pass fixed `DMat4::inverse`, `DVec4`/`DQuat` normalize, NEON scalar-fallback for normalize). Queued: int vectors, wide vector types |
| mid-common | Shared types, traits, string/FFI utilities | Partial — `string/`, `ffi/` substantial (300-500+ real lines each); `error.rs`/`traits.rs`/`types.rs` still thin (2-3 lines) |
| mid-geom | 2D/3D shapes, intersection tests, raycasting, frustum culling | Substantial — real AABB/sphere/capsule/rect/circle shapes with AABB-AABB, AABB-sphere, sphere-sphere, capsule-sphere, capsule-capsule intersection tests, ray-vs-plane/AABB/sphere/capsule raycasting, plane/frustum culling. Known gaps: no OBB, no capsule-vs-AABB, no broadphase/BVH — fill these reactively once mid-physics defines what it actually needs, not speculatively |
| **mid-net** | Reliable UDP-class transport over QUIC (native `quinn`, browser WebTransport), pluggable `Transport` trait, hand-rolled wire codec | **In progress.** Subfolder crates (`mid-net-wire`, `mid-net-transport`, `mid-net-reliable`, `mid-net-transport-quinn`, `mid-net-transport-wasm`, facade `mid-net` — see `docs/mid-net.md` "Crate Structure"), mirroring naia's own `socket/{client,server,shared}` split. `connection.rs`'s `Connection<T: Transport>` is the real composed send/poll API a game loop calls; `ffi.rs` is built and verified against a real compiled C program, not just Rust-side tests (see `docs/mid-net.md` "FFI"). 60 tests real and passing on real CI (rustc 1.97.1, build #30) — dashboard-tracked count, separate from the two Transport backends below which aren't wired into that dashboard yet. `mid-net-reliable`'s retransmit buffer isn't used by `Connection` — `Transport` already guarantees delivery — kept as what a future raw-UDP backend would need internally; see `docs/mid-net.md`. `mid-net-transport-quinn` (native): written, and its first real compile on real CI **succeeded cold** — 6 tests passing there for real, though those are the pure `framing` module only, the async/quinn-integration code has only ever type-checked, never been exercised over a real connection (a real integration test exists, delivered separately, not yet confirmed merged). `mid-net-transport-wasm` (browser): also written this pass, but verified *less* than the quinn crate — no wasm32 target or JS runtime available anywhere in this project's tooling so far means nothing in its actual `WasmTransport` code has been compiled, type-checked, or run by anything, ever; only its platform-agnostic `framing`/`queue` modules have real test coverage (10/10, confirmed on real rustc 1.75). See `docs/mid-net.md` "Transport" for the precise verification status of each |
| mid-ecs | Data-oriented ECS (SoA), network sync baked in from day one | **In progress.** `World`/`Archetype`/`Query`/`sync`/`ffi` still skeleton — the real work so far is `mid-collections` below, which `mid-ecs` will build its Sparse Shell on once the entity allocator (generational arena) exists to key it with |
| **mid-collections** | Sparse set, generational arena, lock-free ring buffer, intrusive lists, hierarchical bitset, `std::hive`, FFI wrapper | **Started.** Real crate now (`crates/mid-collections`), zero dependencies, `#![no_std]` + `alloc`. `SparseSet<I: SparseSetIndex, T>` built and tested — 18/18 real tests passing on real rustc 1.75 (no MSRV wall at all, first thing in the whole `mid-ecs` orbit that's true for). Generic over a `SparseSetIndex` trait rather than any concrete `Entity` type, deliberately — the generational-arena piece that will define what an entity handle actually looks like hasn't been built yet. See `docs/mid-collections.md` for the design reasoning (EnTT/Bevy research, why no paging, why a sentinel instead of `Option<u32>`, why `swap_and_pop` only). Everything else in this crate stays design-doc-only, pulled in piece-by-piece exactly as `mid-ecs` needs it, per that doc |
| mid-physics | Rigid body dynamics, collision response | Not started — crate doesn't exist yet |
| mid-anim | Animation | Not started |
| mid-nodes | Scene graph / node system | Not started |

## Build order (reassessed, this pass)

math → common → geom → **net → ecs → physics** → anim → nodes

Net before ecs, deliberately reordered from an earlier plan that had it the other way: `docs/mid-net.md`'s sync module is explicitly meant to integrate with mid-net's replication model "from day one, not bolted on later." Building net's transport first means ecs's sync module gets designed against a real API instead of a speculative one — the alternative risks designing replication hooks that don't match what net actually needs once it exists, which is exactly the kind of rework this reordering is meant to avoid.

`mid-collections` isn't a step in this list — its pieces (sparse set, generational arena, ring buffer, intrusive lists, bitset, `std::hive`, FFI wrapper) get built piecemeal *during* the ecs phase, each one when ecs actually needs it, not as a blocking phase before ecs starts. See `docs/mid-collections.md`.

Geometry gap-filling isn't a dedicated phase — it's mostly done, and the remaining gaps (OBB, capsule-vs-AABB, broadphase) are best driven by mid-physics's actual requirements rather than built speculatively.

Physics goes after ecs specifically so rigid bodies can be ECS components from day one, rather than a hand-rolled container that gets retrofitted into ECS later.

*(Note: this file previously had a "Priority" column with mid-net=2, mid-ecs=3, mid-math=4, mid-common=0, and didn't list mid-geom or mid-physics at all. That table predates the current 8-crate plan and is superseded by the table and build order above.)*

## Technical Mandates

- **No exceptions** — everything fast and memory-safe
- **Zero-copy** — minimize RAM-to-RAM movement
- **Zero-to-minimal external dependencies, every core crate, no exceptions
  — including DixScript.** Decided this pass, generalized from the
  mid-net-specific call: `dixscript` (1.0.0, published 2026-07-27) is not
  a dependency of any core crate (mid-math, mid-common, mid-geom,
  mid-net, mid-ecs, mid-physics), full stop — not just "for now." Checked
  its own manifest: 23 mandatory transitive crates even with
  `default-features = false` (serde, regex, chrono, aes-gcm,
  chacha20poly1305, argon2, uuid, phf, ...). That's the right budget for
  a general-purpose config/data-interchange format; it isn't for
  hot-path systems code held to the same standard mid-math's SIMD work
  set.
  **Where DixScript actually belongs:** as *the engine's own convenient
  data format* — one layer up from these core crates, for things like
  save data, level/scene data, editor project files, and general
  config, wherever built-in encryption/compression and human-authored
  syntax are worth the dependency weight and there's no 128 Hz hot path
  involved. `tools/mdix-compiler` (a separate binary, not linked into
  any core crate) is the concrete home for this today — its manifest
  now depends on `dixscript` for real, since compiling `.mdix` files is
  exactly its job. The `.mdix` files under `mid-net/packets/` stay as
  human-authored reference schema only; nothing parses them at build or
  run time, so they don't pull the dependency into mid-net itself.
- **Multiplayer-first** — net sync baked into ECS from day one
- **FFI-ready** — every crate exposes a C-compatible API
- **Works anywhere** — no platform-specific runtime requirement (no
  io_uring-only paths, no eBPF/XDP, nothing Linux-only, and no
  `std::net`/`std::time::Instant` assumptions that break on `wasm32`) in
  any crate meant to run on the client. mid-net's `reliable.rs` is the
  first place this got load-bearing: it takes time as a caller-supplied
  `Timestamp(u64)` rather than calling a clock itself, specifically so
  the same protocol code runs unchanged on native (UDP) and in-browser
  (WebTransport datagrams — baseline-available across browsers as of
  March 2026) without an `Instant`-doesn't-exist-on-wasm32 rewrite later.

## Performance Targets

| System | Frequency | Budget |
|---|---|---|
| Network tick | 128 Hz | 7.8 ms |
| Physics | 60 Hz | 16.6 ms |
| Max entities | 100 000+ per core | — |
