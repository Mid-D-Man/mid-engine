# Performance Targets

| System | Frequency | Budget | Status |
|---|---|---|---|
| Network tick (`mid-net`) | 128 Hz | 7.8 ms / tick | Not started |
| Physics (`mid-ecs`) | 60 Hz | 16.6 ms / tick | Not started |
| Max entities (`mid-ecs`) | 100,000+ per core | — | Not started |
| Log hot path (`mid-log`) | 0 µs | zero frame impact | Implemented |
| Math primitives (`mid-math`) | SSE2 | 16-byte aligned Vec3/4/Quat | In progress |

Per the [Design Mandates](design-mandates.md), every claim here should
eventually link to a real benchmark build number on the
[Benchmarks](/benchmarks/) page rather than stand as an unverified target.
Until a row does, treat it as a target, not a measured result.
