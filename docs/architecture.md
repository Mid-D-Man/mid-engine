# Mid Engine Architecture

Modular Anti-Engine — each crate is independently publishable.

## Crates

| Crate | Role | Status |
|---|---|---|
| mid-math | SIMD numerics, f32/f64, zero external deps | f32 optimization complete, f64 optimization complete (this pass fixed `DMat4::inverse`, `DVec4`/`DQuat` normalize, NEON scalar-fallback for normalize). Queued: int vectors, wide vector types |
| mid-common | Shared types, traits, string/FFI utilities | Partial — `string/`, `ffi/` substantial (300-500+ real lines each); `error.rs`/`traits.rs`/`types.rs` still thin (2-3 lines) |
| mid-geom | 2D/3D shapes, intersection tests, raycasting, frustum culling | Substantial — real AABB/sphere/capsule/rect/circle shapes with AABB-AABB, AABB-sphere, sphere-sphere, capsule-sphere, capsule-capsule intersection tests, ray-vs-plane/AABB/sphere/capsule raycasting, plane/frustum culling. Known gaps: no OBB, no capsule-vs-AABB, no broadphase/BVH — fill these reactively once mid-physics defines what it actually needs, not speculatively |
| **mid-net** | Reliable UDP transport, two-channel (reliable/unreliable), DixScript packet definitions | **Starting now.** Currently skeleton only (all files 2-18 lines) |
| mid-ecs | Data-oriented ECS (SoA), network sync baked in from day one | Next after mid-net. Currently skeleton only (all files 2-15 lines) |
| mid-physics | Rigid body dynamics, collision response | Not started — crate doesn't exist yet |
| mid-anim | Animation | Not started |
| mid-nodes | Scene graph / node system | Not started |

## Build order (reassessed, this pass)

math → common → geom → **net → ecs → physics** → anim → nodes

Net before ecs, deliberately reordered from an earlier plan that had it the other way: `docs/mid-net.md`'s sync module is explicitly meant to integrate with mid-net's replication model "from day one, not bolted on later." Building net's transport first means ecs's sync module gets designed against a real API instead of a speculative one — the alternative risks designing replication hooks that don't match what net actually needs once it exists, which is exactly the kind of rework this reordering is meant to avoid.

Geometry gap-filling isn't a dedicated phase — it's mostly done, and the remaining gaps (OBB, capsule-vs-AABB, broadphase) are best driven by mid-physics's actual requirements rather than built speculatively.

Physics goes after ecs specifically so rigid bodies can be ECS components from day one, rather than a hand-rolled container that gets retrofitted into ECS later.

*(Note: this file previously had a "Priority" column with mid-net=2, mid-ecs=3, mid-math=4, mid-common=0, and didn't list mid-geom or mid-physics at all. That table predates the current 8-crate plan and is superseded by the table and build order above.)*

## Technical Mandates

- **No exceptions** — everything fast and memory-safe
- **Zero-copy** — minimize RAM-to-RAM movement
- **Zero-to-minimal external dependencies** — same standard mid-math holds itself to; applies to mid-net too (see `docs/mid-net.md`'s dependency philosophy section)
- **Multiplayer-first** — net sync baked into ECS from day one
- **FFI-ready** — every crate exposes a C-compatible API
- **Works anywhere** — no platform-specific runtime requirement (no io_uring-only paths, no eBPF/XDP, nothing Linux-only) in any crate meant to run on the client

## Performance Targets

| System | Frequency | Budget |
|---|---|---|
| Network tick | 128 Hz | 7.8 ms |
| Physics | 60 Hz | 16.6 ms |
| Max entities | 100 000+ per core | — |
