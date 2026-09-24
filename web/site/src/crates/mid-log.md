# mid-log

Non-blocking tiered logger built on a lock-free SPSC ring buffer — zero
frame-time impact on the hot path. Ships a C header for FFI callers
(`headers/mid_log.h`) and has its own C-side smoke test.

**Status:** done, needs a second optimization pass.
