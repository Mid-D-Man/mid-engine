# mid-alloc

Custom memory allocators and layout management, including `SpinLock<T>` — a
hand-rolled spinlock validated under real multi-threaded stress tests (8
real OS threads, thousands of increments each, zero lost updates). That
same proven algorithm is the basis for `mid-platform::sync::Mutex`'s own
`no_std` fallback (an independent copy, not a dependency — see that crate's
page for why).

**Status:** in progress; paused pending `mid-arena` updates.
