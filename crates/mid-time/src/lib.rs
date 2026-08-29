// crates/mid-time/src/lib.rs
//! mid-time — portable clock + fixed-timestep accumulator for Mid Engine.
//!
//! New this pass (`docs/roadmap.md`, Decision 2): standalone, not folded
//! into `mid-common`, because — like Bevy's own `bevy_time` — this isn't
//! a passive data type. `FixedTimestep` in particular has real per-tick
//! logic other crates (`mid-physics` at the 60 Hz target in
//! `docs/architecture.md`) are meant to drive their step rate from.
//!
//! Zero Cargo dependencies, on every target this project currently
//! supports:
//!
//! - **Native** (anything not `wasm32`): `std::time::Instant` and
//!   `std::time::Duration` — part of the standard library, not a crate
//!   you add to `Cargo.toml`. Nothing to hand-roll here at all.
//! - **`wasm32-unknown-unknown`**: `std::time::Instant::now()` panics on
//!   this target (no monotonic clock syscall without JS help) — the one
//!   real platform split this crate has to make. Solved with a single
//!   hand-rolled `extern "C"` import (see the `wasm` module below), not
//!   `wasm-bindgen`/`web-sys`/`js-sys`/`web-time`. That's the actual
//!   point: `bevy_platform` reaches for exactly that dependency tree to
//!   solve this same problem (`docs/bevy-comparison.md` §5) — this
//!   project's zero-to-minimal-external-deps mandate says no to it by
//!   default, and the problem is small enough (one `f64` in, one `f64`
//!   out) that hand-rolling it costs one imported function and one line
//!   of host-side JS glue instead.
//!
//! **Not built yet, deliberately** (`docs/roadmap.md`'s "What can be
//! built in parallel" section scopes this crate to exactly this): the
//! `mid-app`/`mid-ecs` System wrapper that calls [`Clock::tick`] once a
//! frame. That's a few lines once `mid-app` exists to register a system
//! against — bolting it on doesn't change anything below, so it isn't
//! worth blocking this crate on `mid-app` landing first.
//!
//! **Unverified on wasm32**, same honest caveat this workspace already
//! uses for `mid-net-transport-wasm` (`docs/mid-net.md`): no wasm32
//! target or JS runtime exists anywhere in this project's tooling yet.
//! The `wasm` module below compiles by inspection against the
//! `wasm32-unknown-unknown` import-linking rules, not against a real
//! build.

pub use core::time::Duration;

#[cfg(not(target_arch = "wasm32"))]
mod platform {
    use std::time::Instant as StdInstant;

    /// Opaque monotonic time point — meaningless in isolation, exactly
    /// like `std::time::Instant`'s own contract. Only differences
    /// between two `Instant`s (via [`duration_since`](Instant::duration_since))
    /// are well-defined.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    pub struct Instant(StdInstant);

    impl Instant {
        #[inline]
        pub fn now() -> Self {
            Self(StdInstant::now())
        }

        #[inline]
        pub fn duration_since(&self, earlier: Self) -> super::Duration {
            self.0.duration_since(earlier.0)
        }

        #[inline]
        pub fn elapsed(&self) -> super::Duration {
            self.0.elapsed()
        }
    }
}

#[cfg(target_arch = "wasm32")]
mod platform {
    use super::Duration;

    // Hand-rolled, not wasm-bindgen/web-sys — see module doc above.
    //
    // Whoever instantiates the `.wasm` module (the JS-side loader) must
    // supply this import, e.g.:
    //
    //   const imports = {
    //     env: {
    //       mid_time_now_ms: () => performance.now(),
    //     },
    //   };
    //
    // `performance.now()` returns milliseconds since the page's time
    // origin as an `f64` with sub-millisecond precision — monotonic
    // within a single page load, which is all a frame clock needs.
    #[link(wasm_import_module = "env")]
    extern "C" {
        fn mid_time_now_ms() -> f64;
    }

    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct Instant(f64);

    impl Instant {
        #[inline]
        pub fn now() -> Self {
            // Safety: `mid_time_now_ms` is a JS-provided import with no
            // preconditions beyond existing (documented above, required
            // of the loader) — it returns a plain `f64` with no way to
            // violate a Rust-side invariant.
            Self(unsafe { mid_time_now_ms() })
        }

        #[inline]
        pub fn duration_since(&self, earlier: Self) -> Duration {
            let ms = (self.0 - earlier.0).max(0.0);
            Duration::from_secs_f64(ms / 1000.0)
        }

        #[inline]
        pub fn elapsed(&self) -> Duration {
            Self::now().duration_since(*self)
        }
    }
}

pub use platform::Instant;

/// Per-frame clock: real elapsed time since the previous [`tick`](Clock::tick),
/// plus a running total. Reads the platform clock internally by default —
/// [`tick_with`](Clock::tick_with) is the escape hatch for tests, replay,
/// and deterministic lockstep netcode, matching `mid-net::reliable`'s own
/// caller-supplied-time precedent (`docs/mid-net.md`, "Platform Design
/// Principles").
#[derive(Debug, Clone, Copy)]
pub struct Clock {
    last: Instant,
    delta: Duration,
    elapsed: Duration,
}

impl Clock {
    pub fn new() -> Self {
        Self {
            last: Instant::now(),
            delta: Duration::ZERO,
            elapsed: Duration::ZERO,
        }
    }

    /// Reads the platform clock and updates `delta`/`elapsed`. Call once
    /// per frame.
    pub fn tick(&mut self) {
        let now = Instant::now();
        self.delta = now.duration_since(self.last);
        self.elapsed += self.delta;
        self.last = now;
    }

    /// Advances by a caller-supplied delta instead of reading the
    /// platform clock — for tests, replay, and deterministic lockstep,
    /// where the real wall clock must not be consulted.
    pub fn tick_with(&mut self, delta: Duration) {
        self.delta = delta;
        self.elapsed += delta;
    }

    #[inline]
    pub fn delta(&self) -> Duration {
        self.delta
    }

    #[inline]
    pub fn delta_secs(&self) -> f32 {
        self.delta.as_secs_f32()
    }

    #[inline]
    pub fn elapsed(&self) -> Duration {
        self.elapsed
    }
}

impl Default for Clock {
    fn default() -> Self {
        Self::new()
    }
}

/// Fixed-timestep accumulator — decouples simulation rate (e.g.
/// `mid-physics`'s 60 Hz target, `docs/architecture.md` "Performance
/// Targets") from however fast frames actually render. Feed it a
/// frame's [`Clock::delta`]; it tells you how many fixed steps are now
/// ready to run.
#[derive(Debug, Clone, Copy)]
pub struct FixedTimestep {
    step: Duration,
    accumulator: Duration,
    max_steps_per_tick: u32,
}

impl FixedTimestep {
    /// `hz`: simulation rate, e.g. `60.0` to match the physics budget in
    /// `docs/architecture.md`. `max_steps_per_tick` caps how many fixed
    /// steps one [`accumulate`](Self::accumulate) call can drain —
    /// protects against the classic "spiral of death" after a long
    /// stall (a debugger breakpoint, a stutter) by dropping the excess
    /// instead of trying to catch up all at once.
    pub fn new(hz: f32, max_steps_per_tick: u32) -> Self {
        Self {
            step: Duration::from_secs_f32(1.0 / hz),
            accumulator: Duration::ZERO,
            max_steps_per_tick,
        }
    }

    /// Feeds one frame's delta in; returns how many fixed steps are now
    /// ready. Caller runs its simulation step exactly that many times:
    ///
    /// ```ignore
    /// let n = fixed.accumulate(clock.delta());
    /// for _ in 0..n {
    ///     physics_step(fixed.step_secs());
    /// }
    /// ```
    pub fn accumulate(&mut self, delta: Duration) -> u32 {
        self.accumulator += delta;
        let mut steps = 0;
        while self.accumulator >= self.step && steps < self.max_steps_per_tick {
            self.accumulator -= self.step;
            steps += 1;
        }
        if steps == self.max_steps_per_tick {
            // Spiral-of-death guard: drop whatever's left rather than
            // trying to catch up all at once on the next call.
            self.accumulator = Duration::ZERO;
        }
        steps
    }

    #[inline]
    pub fn step_duration(&self) -> Duration {
        self.step
    }

    #[inline]
    pub fn step_secs(&self) -> f32 {
        self.step.as_secs_f32()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clock_tick_with_accumulates_deterministically() {
        let mut clock = Clock::new();
        clock.tick_with(Duration::from_millis(16));
        clock.tick_with(Duration::from_millis(16));
        clock.tick_with(Duration::from_millis(16));
        assert_eq!(clock.delta(), Duration::from_millis(16));
        assert_eq!(clock.elapsed(), Duration::from_millis(48));
    }

    #[test]
    fn fixed_timestep_drains_expected_steps_at_60hz() {
        let mut fixed = FixedTimestep::new(60.0, 8);
        // One frame's worth of time at 60Hz should drain to exactly 1 step.
        let steps = fixed.accumulate(Duration::from_secs_f32(1.0 / 60.0));
        assert_eq!(steps, 1);

        // A big stall (half a second) should drain several steps, not
        // just one — but never more than max_steps_per_tick.
        let steps = fixed.accumulate(Duration::from_millis(500));
        assert!(steps <= 8);
        assert!(steps > 1);
    }

    #[test]
    fn fixed_timestep_caps_at_max_steps_per_tick() {
        let mut fixed = FixedTimestep::new(60.0, 4);
        // Ten seconds of accumulated time at 60Hz would naively be 600
        // steps — must be capped at 4, with the remainder dropped, not
        // carried into the next call.
        let steps = fixed.accumulate(Duration::from_secs(10));
        assert_eq!(steps, 4);

        let steps = fixed.accumulate(Duration::ZERO);
        assert_eq!(steps, 0);
    }
}
