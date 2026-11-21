// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-editor.md, section "input.rs"
// ============================================================================
//! Pointer tracking: raw touch events in, taps, drags and pinches out.

/// The phase of one pointer event.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    /// A pointer touched down.
    Down = 0,
    /// A pointer that is down moved.
    Move = 1,
    /// A pointer lifted.
    Up = 2,
    /// The system took the pointer away.
    Cancel = 3,
}

impl Phase {
    /// The phase for the number a C host passes, or `None` if it is not 0 to 3.
    pub fn from_i32(v: i32) -> Option<Phase> {
        match v {
            0 => Some(Phase::Down),
            1 => Some(Phase::Move),
            2 => Some(Phase::Up),
            3 => Some(Phase::Cancel),
            _ => None,
        }
    }
}

/// What the tracker recognised.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Gesture {
    /// A quick touch that did not move.
    Tap {
        /// Where it lifted.
        x: f32,
        /// Where it lifted.
        y: f32,
    },
    /// One finger moved past the slop distance.
    Drag {
        /// Current position.
        x: f32,
        /// Current position.
        y: f32,
        /// Movement since the last event.
        dx: f32,
        /// Movement since the last event.
        dy: f32,
        /// Where the finger first touched.
        start_x: f32,
        /// Where the finger first touched.
        start_y: f32,
    },
    /// Two fingers moved.
    Pinch {
        /// New distance between the fingers over the old one.
        ratio: f32,
        /// Midpoint of the fingers.
        x: f32,
        /// Midpoint of the fingers.
        y: f32,
        /// Movement of the midpoint since the last event.
        dx: f32,
        /// Movement of the midpoint since the last event.
        dy: f32,
    },
    /// A dragging finger lifted or was cancelled.
    DragEnd,
}

const MAX_POINTERS: usize = 5;
/// Longest touch, in seconds, that still counts as a tap.
pub const TAP_SECONDS: f32 = 0.6;

#[derive(Clone, Copy, Debug)]
struct Pointer {
    id: i32,
    start: [f32; 2],
    pos: [f32; 2],
    down_at: f32,
    moved: bool,
}

/// Follows the pointers that are down and turns their events into gestures.
#[derive(Clone, Debug)]
pub struct Tracker {
    pointers: Vec<Pointer>,
    /// How far a finger must travel, in pixels, before a touch becomes a drag.
    pub slop: f32,
    pinch_dist: f32,
    pinch_mid: [f32; 2],
}

fn dist(a: [f32; 2], b: [f32; 2]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

impl Tracker {
    /// A tracker with a slop of `slop` pixels.
    pub fn new(slop: f32) -> Self {
        Self {
            pointers: Vec::with_capacity(MAX_POINTERS),
            slop,
            pinch_dist: 0.0,
            pinch_mid: [0.0; 2],
        }
    }

    /// Forgets every pointer, for when the host loses track of the touches
    /// (the app paused, or an `Up` never arrived).
    pub fn clear(&mut self) {
        self.pointers.clear();
        self.pinch_dist = 0.0;
    }

    /// Number of pointers down.
    pub fn down_count(&self) -> usize {
        self.pointers.len()
    }

    fn index_of(&self, id: i32) -> Option<usize> {
        self.pointers.iter().position(|p| p.id == id)
    }

    fn pair(&self) -> Option<([f32; 2], [f32; 2])> {
        match self.pointers.as_slice() {
            [a, b, ..] => Some((a.pos, b.pos)),
            _ => None,
        }
    }

    fn reset_pinch(&mut self) {
        if let Some((a, b)) = self.pair() {
            self.pinch_dist = dist(a, b);
            self.pinch_mid = [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5];
        }
    }

    /// Feeds one event. `now` is the time in seconds. Gestures are appended to `out`.
    pub fn event(
        &mut self,
        phase: Phase,
        id: i32,
        x: f32,
        y: f32,
        now: f32,
        out: &mut Vec<Gesture>,
    ) {
        if !x.is_finite() || !y.is_finite() {
            return;
        }
        match phase {
            Phase::Down => {
                if let Some(i) = self.index_of(id) {
                    self.pointers.remove(i);
                }
                if self.pointers.len() >= MAX_POINTERS {
                    return;
                }
                self.pointers.push(Pointer {
                    id,
                    start: [x, y],
                    pos: [x, y],
                    down_at: now,
                    moved: false,
                });
                if self.pointers.len() >= 2 {
                    for p in &mut self.pointers {
                        p.moved = true;
                    }
                    self.reset_pinch();
                }
            }
            Phase::Move => {
                let Some(i) = self.index_of(id) else { return };
                let prev = self.pointers[i].pos;
                self.pointers[i].pos = [x, y];
                if self.pointers.len() == 1 {
                    let p = &mut self.pointers[0];
                    if !p.moved && dist(p.start, p.pos) > self.slop {
                        p.moved = true;
                    }
                    if p.moved {
                        out.push(Gesture::Drag {
                            x,
                            y,
                            dx: x - prev[0],
                            dy: y - prev[1],
                            start_x: p.start[0],
                            start_y: p.start[1],
                        });
                    }
                } else if let Some((a, b)) = self.pair() {
                    let d = dist(a, b);
                    let mid = [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5];
                    if self.pinch_dist > 1.0 && d > 1.0 {
                        out.push(Gesture::Pinch {
                            ratio: d / self.pinch_dist,
                            x: mid[0],
                            y: mid[1],
                            dx: mid[0] - self.pinch_mid[0],
                            dy: mid[1] - self.pinch_mid[1],
                        });
                    }
                    self.pinch_dist = d;
                    self.pinch_mid = mid;
                }
            }
            Phase::Up | Phase::Cancel => {
                let Some(i) = self.index_of(id) else { return };
                let p = self.pointers.remove(i);
                if self.pointers.is_empty() {
                    if p.moved {
                        out.push(Gesture::DragEnd);
                    } else if phase == Phase::Up && now - p.down_at <= TAP_SECONDS {
                        out.push(Gesture::Tap { x, y });
                    }
                } else {
                    // Fingers are left down: they must not turn into a tap later.
                    for q in &mut self.pointers {
                        q.moved = true;
                    }
                    self.reset_pinch();
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(t: &mut Tracker, events: &[(Phase, i32, f32, f32, f32)]) -> Vec<Gesture> {
        let mut out = Vec::new();
        for (ph, id, x, y, now) in events {
            t.event(*ph, *id, *x, *y, *now, &mut out);
        }
        out
    }

    #[test]
    fn phases_convert_from_the_numbers_a_c_host_sends() {
        assert_eq!(Phase::from_i32(0), Some(Phase::Down));
        assert_eq!(Phase::from_i32(3), Some(Phase::Cancel));
        assert_eq!(Phase::from_i32(4), None);
        assert_eq!(Phase::from_i32(-1), None);
    }

    #[test]
    fn a_short_still_touch_is_a_tap_where_it_lifted() {
        let mut t = Tracker::new(10.0);
        let g = run(
            &mut t,
            &[
                (Phase::Down, 0, 50.0, 60.0, 1.0),
                (Phase::Up, 0, 51.0, 61.0, 1.2),
            ],
        );
        assert_eq!(g, vec![Gesture::Tap { x: 51.0, y: 61.0 }]);
        assert_eq!(t.down_count(), 0);
    }

    #[test]
    fn a_long_press_is_not_a_tap() {
        let mut t = Tracker::new(10.0);
        let g = run(
            &mut t,
            &[
                (Phase::Down, 0, 5.0, 5.0, 1.0),
                (Phase::Up, 0, 5.0, 5.0, 1.0 + TAP_SECONDS + 0.1),
            ],
        );
        assert!(g.is_empty());
    }

    #[test]
    fn small_movement_stays_a_tap_and_large_movement_becomes_a_drag() {
        let mut t = Tracker::new(10.0);
        let g = run(
            &mut t,
            &[
                (Phase::Down, 0, 0.0, 0.0, 0.0),
                (Phase::Move, 0, 5.0, 0.0, 0.1),
                (Phase::Up, 0, 5.0, 0.0, 0.2),
            ],
        );
        assert_eq!(g, vec![Gesture::Tap { x: 5.0, y: 0.0 }]);

        let g = run(
            &mut t,
            &[
                (Phase::Down, 0, 0.0, 0.0, 1.0),
                (Phase::Move, 0, 20.0, 0.0, 1.1),
                (Phase::Move, 0, 30.0, 5.0, 1.2),
                (Phase::Up, 0, 30.0, 5.0, 1.3),
            ],
        );
        assert_eq!(g.len(), 3);
        assert!(matches!(g[0], Gesture::Drag { dx, start_x, .. } if dx == 20.0 && start_x == 0.0));
        assert!(matches!(g[1], Gesture::Drag { dx, dy, .. } if dx == 10.0 && dy == 5.0));
        assert_eq!(g[2], Gesture::DragEnd);
    }

    #[test]
    fn two_fingers_pinch_with_a_ratio_and_a_midpoint() {
        let mut t = Tracker::new(10.0);
        let g = run(
            &mut t,
            &[
                (Phase::Down, 0, 100.0, 100.0, 0.0),
                (Phase::Down, 1, 200.0, 100.0, 0.0),
                (Phase::Move, 1, 300.0, 100.0, 0.1),
            ],
        );
        assert_eq!(g.len(), 1);
        match g[0] {
            Gesture::Pinch {
                ratio,
                x,
                y,
                dx,
                dy,
            } => {
                assert!((ratio - 2.0).abs() < 1e-5, "{ratio}");
                assert_eq!((x, y), (200.0, 100.0));
                assert_eq!((dx, dy), (50.0, 0.0));
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn lifting_one_of_two_fingers_never_becomes_a_tap_or_a_drag_start() {
        let mut t = Tracker::new(10.0);
        let g = run(
            &mut t,
            &[
                (Phase::Down, 0, 0.0, 0.0, 0.0),
                (Phase::Down, 1, 50.0, 0.0, 0.0),
                (Phase::Up, 1, 50.0, 0.0, 0.1),
                (Phase::Up, 0, 0.0, 0.0, 0.2),
            ],
        );
        assert_eq!(
            g,
            vec![Gesture::DragEnd],
            "the last finger ends a gesture, it is not a tap"
        );
    }

    #[test]
    fn a_cancel_is_never_a_tap() {
        let mut t = Tracker::new(10.0);
        let g = run(
            &mut t,
            &[
                (Phase::Down, 0, 0.0, 0.0, 0.0),
                (Phase::Cancel, 0, 0.0, 0.0, 0.1),
            ],
        );
        assert!(g.is_empty());
        assert_eq!(t.down_count(), 0);
    }

    #[test]
    fn unknown_pointers_and_non_finite_positions_are_ignored() {
        let mut t = Tracker::new(10.0);
        let g = run(
            &mut t,
            &[
                (Phase::Move, 7, 1.0, 1.0, 0.0),
                (Phase::Up, 7, 1.0, 1.0, 0.0),
                (Phase::Down, 0, f32::NAN, 1.0, 0.0),
                (Phase::Down, 0, 1.0, f32::INFINITY, 0.0),
            ],
        );
        assert!(g.is_empty());
        assert_eq!(t.down_count(), 0);
    }

    #[test]
    fn clear_forgets_every_pointer_so_the_next_touch_can_tap() {
        let mut t = Tracker::new(10.0);
        let mut out = Vec::new();
        t.event(Phase::Down, 0, 0.0, 0.0, 0.0, &mut out);
        t.event(Phase::Down, 1, 9.0, 0.0, 0.0, &mut out);
        t.clear();
        assert_eq!(t.down_count(), 0);
        let g = run(
            &mut t,
            &[
                (Phase::Down, 0, 5.0, 5.0, 1.0),
                (Phase::Up, 0, 5.0, 5.0, 1.1),
            ],
        );
        assert_eq!(g, vec![Gesture::Tap { x: 5.0, y: 5.0 }]);
    }

    #[test]
    fn at_most_five_pointers_are_tracked_and_a_repeated_down_replaces() {
        let mut t = Tracker::new(10.0);
        let mut out = Vec::new();
        for id in 0..8 {
            t.event(Phase::Down, id, id as f32, 0.0, 0.0, &mut out);
        }
        assert_eq!(t.down_count(), 5);
        t.event(Phase::Down, 0, 99.0, 99.0, 0.0, &mut out);
        assert_eq!(t.down_count(), 5);
    }
}
