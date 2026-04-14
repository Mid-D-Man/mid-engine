# Mid Engine Architecture

Modular Anti-Engine — each crate is independently publishable.

## Crates

| Crate | Role | Priority |
|---|---|---|
| mid-log | Non-blocking tiered logger | 1 |
| mid-net | Reliable UDP + DixScript | 2 |
| mid-ecs | Data-oriented ECS (SoA) | 3 |
| mid-math | SIMD numerics | 4 |
| mid-common | Shared types and traits | 0 |

## Technical Mandates

- **No exceptions** — everything fast and memory-safe
- **Zero-copy** — minimize RAM-to-RAM movement
- **Multiplayer-first** — net sync baked into ECS from day one
- **FFI-ready** — every crate exposes a C-compatible API

## Performance Targets

| System | Frequency | Budget |
|---|---|---|
| Network tick | 128 Hz | 7.8 ms |
| Physics | 60 Hz | 16.6 ms |
| Max entities | 100 000+ per core | — |
