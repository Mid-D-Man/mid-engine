# Mid Engine

> The Middle Man — Modular Anti-Engine

Unity is a black box. Unreal needs unlimited hardware.
Mid is for the Mad Scientists who want a modular, high-speed toolkit they can actually control.

---

## Crates

| Crate | Role | Status |
|---|---|---|
| `mid-common` | Shared types and traits | 🟡 In progress |
| `mid-log` | Non-blocking tiered logger | 🟡 In progress |
| `mid-math` | SIMD-optimised numerics | ✅ Feature complete |
| `mid-net` | Reliable UDP + DixScript networking | 🟡 In progress |
| `mid-ecs` | Data-oriented Entity Component System | 🔵 Planned |
| `mid-geom` | BVH, Delaunay, convex hull, mesh ops | 🔵 Planned |
| `mid-trace` | Distributed tracing | 🔵 Planned |

## Performance Targets

| System | Frequency | Budget |
|---|---|---|
| Network tick | 128 Hz | 7.8 ms |
| Physics | 60 Hz | 16.6 ms |
| Max entities | 100 000+ per core | — |

## Getting Started

```bash
cargo build
cargo test
cargo test --release
cargo run --example headless-server
```

## Design Mandates

- **Multiplayer-first** — network sync is baked into ECS from day one, not bolted on
- **FFI-ready** — every crate exposes a C-ABI layer for cross-language consumers
- **Zero black boxes** — if you need to understand it, you can read it
- **Profile before optimise** — every performance claim cites a `[RELEASE]` build number

## Crate Dependency Ordermid-math        (no engine deps — pure math foundation)
mid-common      (uses mid-math — shared traits and error types)
mid-log         (uses mid-common)
mid-trace       (uses mid-common)
mid-geom        (uses mid-math — geometric algorithms)
mid-ecs         (uses mid-math, mid-common)
mid-net         (uses mid-math, mid-common)
mid-physics     (uses mid-math, mid-geom)
mid-anim        (uses mid-math, mid-ecs)
See `docs/` for architecture details and `packets/` for DixScript packet definitions.
