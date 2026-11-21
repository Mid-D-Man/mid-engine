// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-editor.md, section "stats.rs"
// ============================================================================
//! Frame statistics for the diagnostics overlay.

const WINDOW: usize = 60;

/// A rolling window over the last 60 frame times, plus the core's own cost.
#[derive(Clone, Debug)]
pub struct Stats {
    frames: [f32; WINDOW],
    next: usize,
    filled: usize,
    core_ms: f32,
}

impl Default for Stats {
    fn default() -> Self {
        Self {
            frames: [0.0; WINDOW],
            next: 0,
            filled: 0,
            core_ms: 0.0,
        }
    }
}

impl Stats {
    /// Records one frame time in seconds. Negative and non-finite values are
    /// ignored; a pause (a frame over a second) is recorded as a second.
    pub fn push(&mut self, dt: f32) {
        if !dt.is_finite() || dt < 0.0 {
            return;
        }
        self.frames[self.next] = dt.min(1.0);
        self.next = (self.next + 1) % WINDOW;
        self.filled = (self.filled + 1).min(WINDOW);
    }

    /// Records how long the core spent on its own work in the last frame.
    pub fn set_core_ms(&mut self, ms: f32) {
        if ms.is_finite() && ms >= 0.0 {
            // A moving average, so one slow frame does not flicker the display.
            self.core_ms = if self.core_ms == 0.0 {
                ms
            } else {
                self.core_ms * 0.9 + ms * 0.1
            };
        }
    }

    /// The average frame time in milliseconds, or 0 with no frames.
    pub fn avg_ms(&self) -> f32 {
        if self.filled == 0 {
            return 0.0;
        }
        self.frames[..self.filled.min(WINDOW)].iter().sum::<f32>() / self.filled as f32 * 1000.0
    }

    /// The slowest frame in the window, in milliseconds.
    pub fn worst_ms(&self) -> f32 {
        self.frames[..self.filled]
            .iter()
            .fold(0.0_f32, |m, v| m.max(*v))
            * 1000.0
    }

    /// Frames per second from the average, or 0 with no frames.
    pub fn fps(&self) -> f32 {
        let ms = self.avg_ms();
        if ms > 0.0 {
            1000.0 / ms
        } else {
            0.0
        }
    }

    /// The core's moving-average cost per frame in milliseconds.
    pub fn core_ms(&self) -> f32 {
        self.core_ms
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_empty_window_reports_zero() {
        let s = Stats::default();
        assert_eq!((s.avg_ms(), s.worst_ms(), s.fps()), (0.0, 0.0, 0.0));
    }

    #[test]
    fn average_worst_and_fps_follow_the_frames() {
        let mut s = Stats::default();
        for dt in [0.010, 0.020, 0.030] {
            s.push(dt);
        }
        assert!((s.avg_ms() - 20.0).abs() < 1e-3);
        assert!((s.worst_ms() - 30.0).abs() < 1e-3);
        assert!((s.fps() - 50.0).abs() < 1e-2);
    }

    #[test]
    fn the_window_forgets_old_frames() {
        let mut s = Stats::default();
        s.push(0.5);
        for _ in 0..WINDOW {
            s.push(0.010);
        }
        assert!(
            (s.worst_ms() - 10.0).abs() < 1e-3,
            "the slow frame left the window"
        );
    }

    #[test]
    fn bad_values_are_ignored_and_pauses_are_capped() {
        let mut s = Stats::default();
        s.push(f32::NAN);
        s.push(-1.0);
        s.push(f32::INFINITY);
        assert_eq!(s.avg_ms(), 0.0);
        s.push(30.0);
        assert!((s.worst_ms() - 1000.0).abs() < 1e-3);
    }

    #[test]
    fn core_cost_is_a_moving_average() {
        let mut s = Stats::default();
        s.set_core_ms(2.0);
        assert_eq!(s.core_ms(), 2.0);
        s.set_core_ms(12.0);
        assert!((s.core_ms() - 3.0).abs() < 1e-5);
        s.set_core_ms(f32::NAN);
        s.set_core_ms(-5.0);
        assert!((s.core_ms() - 3.0).abs() < 1e-5);
    }
}
