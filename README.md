# MidManStudio: Mid Engine

> The Middle Man — Modular Anti-Engine & Toolkit

Unity is a black box. Unreal needs unlimited hardware. Mid is for the Mad Scientists who want a modular, high-speed toolkit they can actually control[span_0](start_span)[span_0](end_span). 

Mid Engine isn't just a monolith; it is a highly modular, high-speed toolkit[span_1](start_span)[span_1](end_span). We embrace a "Middle Man" architecture: every core crate is built in native Rust with minimal external dependencies and designed with strict `#[repr(C)]` FFI boundaries from day one[span_2](start_span)[span_2](end_span)[span_3](start_span)[span_3](end_span). Whether you want to snap these modules together to build a sovereign ecosystem, or inject our high-performance SIMD math and networking stacks straight into an existing C++/C# environment via raw pointers, Mid Engine is built to let the hardware eat[span_4](start_span)[span_4](end_span)[span_5](start_span)[span_5](end_span).

---

## 🚀 Design Mandates

* **Multiplayer-First:** Networking is baked into the ECS from day one, not bolted on later[span_6](start_span)[span_6](end_span). 
* **FFI-Ready:** Every crate exposes a C-ABI layer for cross-language consumers, acting as the ultimate "Middle Man[span_7](start_span)"[span_7](end_span).
* **Minimal Dependencies:** Built in-house using native Rust to avoid bloat and maintain total control over the memory layout.
* **Zero Black Boxes:** If you need to understand it, you can read it.
* **Profile Before Optimise:** Every performance claim cites a `[RELEASE]` build number, benchmarked to run flawlessly even on a 2010 MacBook Pro[span_8](start_span)[span_8](end_span)[span_9](start_span)[span_9](end_span).

---

## ⚡ Performance Targets

We enforce a strict "No Exceptions" performance mandate[span_10](start_span)[span_10](end_span). 

| System | Target Frequency | Hardware Budget |
|---|---|---|
| **Network tick** | 128 Hz[span_11](start_span)[span_11](end_span) | 7.8 ms |
| **Physics** | 60 Hz[span_12](start_span)[span_12](end_span) | 16.6 ms |
| **Data Throughput** | 100,000+ entities per core[span_13](start_span)[span_13](end_span) | ~5.3 ms total[span_14](start_span)[span_14](end_span) |

---

## 📦 The Crate Arsenal

Every module is a standalone, dependency-light weapon[span_15](start_span)[span_15](end_span). 

| Crate | Role | The Madness |
|---|---|---|
| `mid-math` | SIMD-Optimized Numerics | Pure math foundation featuring strictly 16-byte aligned `#[repr(C)]` primitives optimized with SSE2/AVX/Neon[span_16](start_span)[span_16](end_span)[span_17](start_span)[span_17](end_span)[span_18](start_span)[span_18](end_span). |
| `mid-common` | Shared Types & Traits | Core engine-wide traits and custom memory allocation strategies like bump arenas[span_19](start_span)[span_19](end_span)[span_20](start_span)[span_20](end_span). |
| `mid-log` | Non-blocking Tiered Logger | Lock-free ring buffer architecture with background thread processing for zero main-thread stalls[span_21](start_span)[span_21](end_span)[span_22](start_span)[span_22](end_span). |
| `mid-ecs` | Hybrid Entity Component System | A Data-Oriented powerhouse utilizing Archetype (SoA layout) and Sparse Sets[span_23](start_span)[span_23](end_span)[span_24](start_span)[span_24](end_span). Parallelized via rayon to handle 100,000+ entities[span_25](start_span)[span_25](end_span). |
| `mid-net` | Reliable UDP Sync | Integrated natively with DixScript[span_26](start_span)[span_26](end_span). Built to dodge TCP Head-of-Line blocking and sync 128Hz ticks efficiently[span_27](start_span)[span_27](end_span). |
| `mdix-compiler` | DixScript Tooling | Compiles `.mdix` schemas, handling compile-time logic, AES-256 encryption, and asset embedding[span_28](start_span)[span_28](end_span). |
| `mid-geom` | Geometry Algorithms | Handles BVH construction, Delaunay triangulation, convex hulls, and mesh ops[span_29](start_span)[span_29](end_span). *(Planned)* |

---

## 🏗️ Crate Dependency Order

The ecosystem is layered to prevent circular dependencies while maximizing modularity:

```text
mid-math       (no engine deps — pure math foundation)
mid-common     (uses mid-math — shared traits and error types)
mid-log        (uses mid-common)
mid-trace      (uses mid-common)
mid-geom       (uses mid-math — geometric algorithms)
mid-ecs        (uses mid-math, mid-common)
mid-net        (uses mid-math, mid-common)
mid-physics    (uses mid-math, mid-geom)
mid-anim       (uses mid-math, mid-ecs)
mid-nodes      (uses mid-math, mid-geom, mid-anim)
