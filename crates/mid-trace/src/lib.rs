// crates/mid-trace/src/lib.rs

//! Frame profiler and scope tracer for Mid Engine.
//!
//! ## Why this is not inside mid-log
//!
//! | mid-log                         | mid-trace                              |
//! |---------------------------------|----------------------------------------|
//! | Sparse events, human text       | Dense timing, nanosecond resolution    |
//! | Async IO to file / stderr       | Timeline to Tracy / flamegraph         |
//! | Always-on in production         | Zero-cost unless `profile` feature on  |
//! | Level filter (TRACE..FATAL)     | Scope hierarchy, frame boundaries      |
//!
//! ## Feature flags
//!
//! | Flag      | Effect                                                     |
//! |-----------|------------------------------------------------------------|
//! | (none)    | All macros compile to nothing — truly zero overhead        |
//! | `profile` | Scope timing, frame accumulator, `SpanRecord` collection  |
//! | `tracy`   | Tracy profiler integration (implies `profile`)             |
//! | `perf`    | Linux `perf_event_open` hardware counters (implies `profile`) |
//!
//! ## Quick start
//!
//! ```rust,no_run
//! // Cargo.toml: mid-trace = { features = ["profile"] }
//!
//! fn game_loop() {
//!     loop {
//!         mid_trace::frame_mark();   // frame boundary
//!         physics_tick();
//!         render_tick();
//!     }
//! }
//!
//! fn physics_tick() {
//!     mid_trace::mid_span!("physics_tick");
//!     // ... work ...
//! } // timing recorded on drop
//! ```

// ── Re-exports ────────────────────────────────────────────────────────────────

pub use span::Span;
pub use frame::frame_mark;

#[cfg(feature = "profile")]
pub use frame::{flush_frame, SpanRecord};

#[cfg(all(feature = "perf", target_os = "linux"))]
pub mod perf;

mod span;
mod frame;

// ── Macros ────────────────────────────────────────────────────────────────────

/// Profile the current scope. Zero-cost no-op unless `profile` feature is on.
///
/// Records elapsed time from the macro site to the end of the enclosing scope.
/// When `tracy` is enabled, the span appears in the Tracy timeline.
///
/// # Example
/// ```rust,no_run
/// fn update_ai(entities: &mut [Entity]) {
///     mid_trace::mid_span!("update_ai");
///     for e in entities { e.think(); }
/// } // "update_ai" timing recorded here
/// ```
#[cfg(feature = "profile")]
#[macro_export]
macro_rules! mid_span {
    ($name:literal) => {
        let _mid_span = $crate::Span::enter($name, file!(), line!());
    };
}

/// Zero-cost compile-time no-op when `profile` is not enabled.
#[cfg(not(feature = "profile"))]
#[macro_export]
macro_rules! mid_span {
    ($($tt:tt)*) => {};
}

// ── span.rs ───────────────────────────────────────────────────────────────────

mod span {
    /// RAII profiling scope. Zero-size and zero-cost without `profile` feature.
    ///
    /// Use via `mid_span!()` — do not construct directly.
    #[cfg(feature = "profile")]
    pub struct Span {
        name:  &'static str,
        file:  &'static str,
        line:  u32,
        start: std::time::Instant,
        #[cfg(feature = "tracy")]
        _tracy: tracy_client::Span,
    }

    #[cfg(feature = "profile")]
    impl Span {
        #[inline]
        pub fn enter(name: &'static str, file: &'static str, line: u32) -> Self {
            #[cfg(feature = "tracy")]
            let _tracy = tracy_client::Span::new(name, "", file, line, 100);
            Span {
                name,
                file,
                line,
                start: std::time::Instant::now(),
                #[cfg(feature = "tracy")]
                _tracy,
            }
        }
    }

    #[cfg(feature = "profile")]
    impl Drop for Span {
        #[inline]
        fn drop(&mut self) {
            let elapsed_ns = self.start.elapsed().as_nanos() as u64;
            crate::frame::record_span(self.name, self.file, self.line, elapsed_ns);
        }
    }

    /// Zero-size stub — compiles entirely away when profiling is disabled.
    #[cfg(not(feature = "profile"))]
    pub struct Span;

    #[cfg(not(feature = "profile"))]
    impl Span {
        #[inline(always)]
        pub fn enter(_name: &'static str, _file: &'static str, _line: u32) -> Self {
            Span
        }
    }
}

// ── frame.rs ──────────────────────────────────────────────────────────────────

mod frame {
    /// One timing record inside a frame.
    #[cfg(feature = "profile")]
    #[derive(Debug, Clone)]
    pub struct SpanRecord {
        pub name:       &'static str,
        pub file:       &'static str,
        pub line:       u32,
        /// Wall-clock nanoseconds from scope entry to drop.
        pub elapsed_ns: u64,
    }

    #[cfg(feature = "profile")]
    static FRAME_SPANS: std::sync::Mutex<Vec<SpanRecord>> =
        std::sync::Mutex::new(Vec::new());

    /// Record a completed span. Called from `Span::drop()`.
    #[cfg(feature = "profile")]
    pub(crate) fn record_span(
        name:       &'static str,
        file:       &'static str,
        line:       u32,
        elapsed_ns: u64,
    ) {
        if let Ok(mut g) = FRAME_SPANS.lock() {
            g.push(SpanRecord { name, file, line, elapsed_ns });
        }
    }

    #[cfg(not(feature = "profile"))]
    pub(crate) fn record_span(_: &'static str, _: &'static str, _: u32, _: u64) {}

    /// Mark the end of one frame / start of the next.
    ///
    /// - With `tracy`: advances the Tracy frame timeline.
    /// - With `profile`: flushes the span accumulator (see `flush_frame()`).
    /// - Without either: compiles to nothing.
    ///
    /// Call once per game tick, typically at the top of the loop.
    #[inline]
    pub fn frame_mark() {
        #[cfg(feature = "tracy")]
        tracy_client::frame_mark();

        #[cfg(all(feature = "profile", not(feature = "tracy")))]
        { let _ = flush_frame(); }
    }

    /// Drain the current frame's span records and reset for the next frame.
    ///
    /// Returns collected `SpanRecord`s in chronological order.
    /// Useful for in-game overlays or flamegraph exporters.
    ///
    /// ```rust,no_run
    /// // render overlay:
    /// for record in mid_trace::flush_frame() {
    ///     println!("{}: {:.3} ms", record.name, record.elapsed_ns as f64 / 1e6);
    /// }
    /// ```
    #[cfg(feature = "profile")]
    pub fn flush_frame() -> Vec<SpanRecord> {
        FRAME_SPANS.lock()
            .map(|mut g| std::mem::take(&mut *g))
            .unwrap_or_default()
    }

    #[cfg(not(feature = "profile"))]
    pub fn flush_frame() {}
}

// ── perf.rs (Linux only, "perf" feature) ─────────────────────────────────────

#[cfg(all(feature = "perf", target_os = "linux"))]
pub mod perf {
    //! Linux `perf_event_open` hardware performance counters.
    //!
    //! This is the userspace equivalent of ftrace for hot code paths.
    //! Measure CPU cycles, instructions, cache misses, and branch
    //! mispredictions with nanosecond granularity and near-zero overhead.
    //!
    //! ## Usage
    //! ```rust,no_run
    //! #[cfg(all(feature = "perf", target_os = "linux"))]
    //! {
    //!     use mid_trace::perf::{PerfCounter, HwEvent};
    //!
    //!     let mut cycles = PerfCounter::open(HwEvent::CpuCycles).unwrap();
    //!     cycles.reset();
    //!     expensive_function();
    //!     println!("CPU cycles: {}", cycles.read());
    //!
    //!     let mut misses = PerfCounter::open(HwEvent::CacheMisses).unwrap();
    //!     misses.reset();
    //!     cache_heavy_function();
    //!     println!("Cache misses: {}", misses.read());
    //! }
    //! ```

    use std::fs::File;
    use std::io::Read;
    use std::os::unix::io::{FromRawFd, RawFd};

    /// Hardware performance event to measure.
    #[repr(u64)]
    #[derive(Debug, Clone, Copy)]
    pub enum HwEvent {
        CpuCycles         = 0,
        Instructions      = 1,
        CacheReferences   = 2,
        CacheMisses       = 3,
        BranchInstructions= 4,
        BranchMisses      = 5,
    }

    // perf_event_attr — must be exactly 128 bytes on x86_64 Linux.
    // We zero-initialize and set only the fields we need.
    #[repr(C)]
    struct PerfEventAttr {
        type_:   u32,  // PERF_TYPE_HARDWARE = 0
        size:    u32,
        config:  u64,  // HwEvent value
        _rest:   [u64; 14], // remaining fields, all zero
    }

    const PERF_TYPE_HARDWARE:   u32 = 0;
    const PERF_FLAG_FD_CLOEXEC: u64 = 8;

    /// A hardware performance counter backed by one `perf_event_open` fd.
    pub struct PerfCounter {
        file: File,
    }

    impl PerfCounter {
        /// Open a hardware performance counter for the current thread.
        ///
        /// Returns `Err` if the kernel does not support `perf_event_open`
        /// (very rare on modern Linux) or if you lack permission
        /// (`/proc/sys/kernel/perf_event_paranoid` may need lowering).
        pub fn open(event: HwEvent) -> std::io::Result<Self> {
            let attr = PerfEventAttr {
                type_:  PERF_TYPE_HARDWARE,
                size:   std::mem::size_of::<PerfEventAttr>() as u32,
                config: event as u64,
                _rest:  [0u64; 14],
            };

            // perf_event_open(attr, pid=0=this thread, cpu=-1=any, group=-1, flags)
            let fd = unsafe {
                libc_perf_event_open(
                    &attr as *const PerfEventAttr as *const _,
                    0,   // pid = current thread
                    -1,  // cpu = any
                    -1,  // group_fd = standalone
                    PERF_FLAG_FD_CLOEXEC,
                )
            };

            if fd < 0 {
                return Err(std::io::Error::last_os_error());
            }

            Ok(PerfCounter {
                file: unsafe { File::from_raw_fd(fd as RawFd) },
            })
        }

        /// Read the current counter value.
        pub fn read(&mut self) -> u64 {
            let mut buf = [0u8; 8];
            // Rewind to start of file (counters are read from offset 0).
            use std::io::Seek;
            let _ = self.file.seek(std::io::SeekFrom::Start(0));
            self.file.read_exact(&mut buf).unwrap_or(());
            u64::from_ne_bytes(buf)
        }

        /// Reset the counter to zero.
        pub fn reset(&mut self) {
            const PERF_EVENT_IOC_RESET: u64 = 0x2403;
            unsafe {
                libc_ioctl(
                    std::os::unix::io::AsRawFd::as_raw_fd(&self.file),
                    PERF_EVENT_IOC_RESET,
                    0,
                );
            }
        }

        /// Enable counting (starts paused by default after open on some kernels).
        pub fn enable(&mut self) {
            const PERF_EVENT_IOC_ENABLE: u64 = 0x2400;
            unsafe {
                libc_ioctl(
                    std::os::unix::io::AsRawFd::as_raw_fd(&self.file),
                    PERF_EVENT_IOC_ENABLE,
                    0,
                );
            }
        }
    }

    // ── Raw syscall shims (no libc dependency) ────────────────────────────────

    unsafe fn libc_perf_event_open(
        attr:     *const std::ffi::c_void,
        pid:      i32,
        cpu:      i32,
        group_fd: i32,
        flags:    u64,
    ) -> i64 {
        // SYS_perf_event_open = 298 on x86_64 Linux.
        // This is the ONLY use of asm! in mid-trace — syscall ABI,
        // not a math operation, so it falls outside the Tier 3 prohibition.
        let ret: i64;
        std::arch::asm!(
            "syscall",
            inout("rax") 298i64 => ret,
            in("rdi")    attr as u64,
            in("rsi")    pid   as i64,
            in("rdx")    cpu   as i64,
            in("r10")    group_fd as i64,
            in("r8")     flags,
            options(nostack, preserves_flags),
        );
        ret
    }

    unsafe fn libc_ioctl(fd: RawFd, request: u64, arg: u64) -> i64 {
        // SYS_ioctl = 16 on x86_64 Linux.
        let ret: i64;
        std::arch::asm!(
            "syscall",
            inout("rax") 16i64 => ret,
            in("rdi")    fd      as i64,
            in("rsi")    request,
            in("rdx")    arg,
            options(nostack, preserves_flags),
        );
        ret
    }
}
