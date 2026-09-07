# Mid Engine

> The Middle Man — Modular Engine Toolkit

Traditional monolithic engines hide their internals behind layers of abstraction and bloated UI, while others demand unlimited hardware just to render a basic scene. Mid Engine is built for unhinged engineering—providing a modular, high-speed toolkit for developers who want complete control over their hardware stack.

It serves as a true "Middle Man," providing an ecosystem where every crate is FFI-compatible out of the box and built with as few external dependencies as possible. Whether you are assembling a sovereign engine or injecting core modules into an existing C++/C# environment via raw pointers, Mid Engine is designed to get out of your way and let the hardware eat.

---

## 📦 Crates

| Crate | Role | Status |
|---|---|---|
| `mid-common` | Shared types and traits | 🟡 In progress |
| `mid-alloc` / `mid-arena` | Arena allocation and custom memory layout management | 🟡 In progress |
| `mid-log` | Non-blocking tiered logger utilizing a lock-free ring buffer | 🟢 Done (Needs optimization) |
| `mid-math` | SIMD-optimized numerics featuring strictly 16-byte aligned `#[repr(C)]` primitives | 🟢 Done (Needs second pass) |
| `mid-net` | Reliable UDP + DixScript transport core | 🟡 In progress (Awaiting ECS integration) |
| `mid-ecs` | Data-oriented Entity Component System utilizing Archetype and Sparse Set layouts | 🔵 Planned |
| `mid-geom` | BVH, Delaunay, convex hull, mesh ops | 🔵 Planned |
| `mid-trace` | Distributed tracing | 🔵 Planned |
| `mdix-compiler`| DixScript (`.mdix`) schema compiler handling encryption and asset embedding | 🟡 In progress |

## ⚡ Performance Targets

| System | Frequency | Budget |
|---|---|---|
| Network tick | 128 Hz | 7.8 ms |
| Physics | 60 Hz | 16.6 ms |
| Max entities | 100 000+ per core | — |

## 🚀 Getting Started

```bash
cargo build
cargo test
cargo test --release
cargo run --example headless-server

🛠️ Design Mandates
 * Multiplayer-first — Network sync is baked into the ECS from day one, rather than being bolted on later.
 * FFI-ready — Every crate exposes a strict #[repr(C)] FFI boundary to act as a cross-language "Middle Man".
 * Zero hidden abstractions — If you need to understand the memory layout, you can read the code directly.
 * Profile before optimize — Every performance claim cites a [RELEASE] build number, benchmarked to run natively even on legacy target hardware.
🏗️ Crate Dependency Order
mid-math        (no engine deps — pure math foundation)
mid-common      (uses mid-math — shared traits and error types)
mid-log         (uses mid-common)
mid-trace       (uses mid-common)
mid-geom        (uses mid-math — geometric algorithms)
mid-ecs         (uses mid-math, mid-common)
mid-net         (uses mid-math, mid-common)
mid-physics     (uses mid-math, mid-geom)
mid-anim        (uses mid-math, mid-ecs)

See docs/ for architecture details and packets/ for DixScript packet definitions.

