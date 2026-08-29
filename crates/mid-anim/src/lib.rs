// crates/mid-anim/src/lib.rs
//! mid-anim — animation clip sampling for Mid Engine.
//!
//! New this pass (`docs/roadmap.md`, "What can be built in parallel"):
//! scoped to sampling only. Applying a sampled value to a skeleton joint
//! or scene-graph node needs hierarchy, which doesn't exist yet —
//! `docs/mid-ecs.md`'s "GlobalTransform" section calls the hierarchy-
//! composition work "not yet started" (that's `mid-nodes`'s job, per
//! `docs/roadmap.md`). This crate stops at [`Track::sample`]; wiring its
//! output into a transform is later work, once `mid-nodes` exists.
//!
//! ## What's real in this pass
//!
//! [`Track<T>`] — a sequence of `(time, value)` keyframes sampled with
//! Catmull-Rom interpolation (`mid_math::curves::CatmullRom`, already
//! built) between them. `CatmullRom::evaluate` itself parameterizes by
//! control-point index (`t ∈ [0, segment_count()]`, assuming even
//! spacing) — this crate's actual new content is [`Track::sample`]'s
//! wall-time-to-that-parameterization mapping, since real keyframes are
//! rarely evenly spaced in time.
//!
//! Works for any `T: mid_math::curves::Interpolate`, which today means
//! `f32`, `f64`, `Vec2`, `Vec3`, `Quat`, `DVec2`, `DVec3`, `DQuat` — so
//! e.g. `Track<Vec3>` for a position track, `Track<Quat>` for a rotation
//! track (interpolated via `slerp`, not raw `lerp` — `Quat`'s own
//! `Interpolate` impl already handles that correctly).
//!
//! ## What's not built yet, on purpose
//!
//! - **Multi-track clips** (grouping a position + rotation + scale track
//!   under one named clip) and **blending** between clips — real
//!   features, deferred rather than guessed at ahead of `mid-nodes`
//!   existing to actually consume them.
//! - **Looping/ping-pong playback modes** — [`Track::sample`] clamps to
//!   `[start_time, end_time]` with no wraparound; a caller wanting a
//!   loop wraps `time` itself before calling (`time % track.duration()
//!   + track.start_time()`).

use mid_math::curves::{CatmullRom, Interpolate};

/// A single animation track: a sequence of `(time, value)` keyframes,
/// sampled with Catmull-Rom interpolation between them. Keyframe times
/// need not be evenly spaced.
pub struct Track<T: Interpolate + Clone> {
    times: Vec<f32>,
    curve: CatmullRom<T>,
}

impl<T: Interpolate + Clone> Track<T> {
    /// `keyframes` must be sorted by time, ascending, with at least 2
    /// entries — both requirements are asserted, not silently handled,
    /// since a malformed track is an authoring bug, not a runtime state
    /// to recover from.
    pub fn new(keyframes: Vec<(f32, T)>) -> Self {
        assert!(keyframes.len() >= 2, "Track needs at least 2 keyframes");
        assert!(
            keyframes.windows(2).all(|w| w[0].0 <= w[1].0),
            "Track keyframes must be sorted by time, ascending"
        );
        let (times, values): (Vec<f32>, Vec<T>) = keyframes.into_iter().unzip();
        Self {
            curve: CatmullRom::new(values),
            times,
        }
    }

    #[inline]
    pub fn start_time(&self) -> f32 {
        self.times[0]
    }

    #[inline]
    pub fn end_time(&self) -> f32 {
        *self.times.last().unwrap()
    }

    #[inline]
    pub fn duration(&self) -> f32 {
        self.end_time() - self.start_time()
    }

    /// Samples the track at wall time `time`, clamped to
    /// `[start_time(), end_time()]` — no extrapolation or looping here,
    /// see the module doc.
    pub fn sample(&self, time: f32) -> T {
        let time = time.clamp(self.start_time(), self.end_time());

        // Find the segment containing `time`: the last index `i` with
        // `times[i] <= time`. Linear scan — tracks are small (a handful
        // to a few dozen keyframes), not worth a binary search yet.
        let mut seg = 0;
        for i in 0..self.times.len() - 1 {
            if time >= self.times[i] {
                seg = i;
            }
        }

        let seg_start = self.times[seg];
        let seg_end = self.times[seg + 1];
        let local = if seg_end > seg_start {
            (time - seg_start) / (seg_end - seg_start)
        } else {
            0.0 // degenerate: two keyframes at the same time
        };

        self.curve.evaluate(seg as f32 + local)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn track_samples_exactly_at_keyframe_times() {
        let track = Track::new(vec![(0.0, 10.0_f32), (1.0, 20.0), (2.5, 5.0)]);

        assert!((track.sample(0.0) - 10.0).abs() < 1e-4);
        assert!((track.sample(1.0) - 20.0).abs() < 1e-4);
        assert!((track.sample(2.5) - 5.0).abs() < 1e-4);
    }

    #[test]
    fn track_clamps_before_start_and_after_end() {
        let track = Track::new(vec![(1.0, 10.0_f32), (2.0, 20.0)]);

        assert!((track.sample(-5.0) - track.sample(1.0)).abs() < 1e-4);
        assert!((track.sample(100.0) - track.sample(2.0)).abs() < 1e-4);
    }

    #[test]
    fn track_reports_start_end_and_duration() {
        let track = Track::new(vec![(1.0, 0.0_f32), (4.0, 1.0)]);
        assert_eq!(track.start_time(), 1.0);
        assert_eq!(track.end_time(), 4.0);
        assert_eq!(track.duration(), 3.0);
    }

    #[test]
    #[should_panic(expected = "at least 2 keyframes")]
    fn new_rejects_single_keyframe() {
        let _ = Track::new(vec![(0.0, 1.0_f32)]);
    }
}
