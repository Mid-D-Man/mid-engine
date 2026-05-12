// crates/mid-math/src/curves/kochanek_bartels.rs
//! Kochanek-Bartels (TCB) spline — Hermite-based with tension, continuity, bias.
//!
//! The TCB parameters modify the tangent vectors at each keyframe:
//!
//!   Source tangent (incoming):
//!     s_in  = ((1-t)(1-c)(1+b)/2) * (P[i] - P[i-1])
//!            + ((1-t)(1+c)(1-b)/2) * (P[i+1] - P[i])
//!
//!   Destination tangent (outgoing):
//!     s_out = ((1-t)(1+c)(1+b)/2) * (P[i] - P[i-1])
//!            + ((1-t)(1-c)(1-b)/2) * (P[i+1] - P[i])
//!
//! Parameters:
//!   tension    ∈ [-1, 1] — tightness of the curve. 0 = Catmull-Rom tangent.
//!                          +1 = sharp corner (zero tangent), -1 = loose overshoot.
//!   continuity ∈ [-1, 1] — smoothness at the keyframe.
//!                          0 = smooth, ±1 = corner (tangent discontinuity).
//!   bias       ∈ [-1, 1] — direction of influence.
//!                          0 = symmetric, +1 = biased toward previous segment,
//!                          -1 = biased toward next segment.
//!
//! When T=C=B=0 the curve reduces to a standard Catmull-Rom spline.
//!
//! Game use: professional cinematic tools, Blender NLA editor, game cutscenes.
//!           Lets animators dial in "anticipation" (negative bias) and
//!           "follow-through" (positive bias) without manual tangent editing.

use super::interpolate::Interpolate;
use super::hermite::{evaluate_segment, derivative_segment};

/// One TCB keyframe.
#[derive(Clone, Copy, Debug)]
pub struct TcbKey<T> {
    pub position:   T,
    pub tension:    f32,
    pub continuity: f32,
    pub bias:       f32,
}

impl<T: Clone> TcbKey<T> {
    /// Key with default TCB (0, 0, 0) — behaves like Catmull-Rom.
    #[inline]
    pub fn new(position: T) -> Self {
        Self { position, tension: 0.0, continuity: 0.0, bias: 0.0 }
    }

    #[inline]
    pub fn with_tcb(position: T, tension: f32, continuity: f32, bias: f32) -> Self {
        Self { position, tension, continuity, bias }
    }
}

/// Kochanek-Bartels (TCB) spline.
///
/// Requires at least 2 keys. The first and last segments use reflected
/// phantom points for the out-of-range neighbours.
#[derive(Clone, Debug)]
pub struct KochanekBartels<T> {
    pub keys: Vec<TcbKey<T>>,
}

impl<T: Interpolate + Clone> KochanekBartels<T> {
    #[inline]
    pub fn new(keys: Vec<TcbKey<T>>) -> Self {
        assert!(keys.len() >= 2, "KochanekBartels requires at least 2 keys");
        Self { keys }
    }

    /// Number of segments.
    #[inline]
    pub fn segment_count(&self) -> usize { self.keys.len() - 1 }

    /// Compute the incoming and outgoing tangents for key `i`.
    fn tangents(&self, i: usize) -> (T, T) {
        let n = self.keys.len();
        let ki = &self.keys[i];

        // Neighbour positions with phantom point reflection at boundaries.
        let prev = if i == 0 {
            ki.position.scale(2.0).sub(self.keys[1].position.clone())
        } else {
            self.keys[i - 1].position.clone()
        };

        let next = if i + 1 >= n {
            ki.position.scale(2.0).sub(self.keys[n - 2].position.clone())
        } else {
            self.keys[i + 1].position.clone()
        };

        let t = ki.tension;
        let c = ki.continuity;
        let b = ki.bias;

        let d_prev = ki.position.sub(prev);       // P[i] - P[i-1]
        let d_next = next.sub(ki.position.clone()); // P[i+1] - P[i]

        // Source (incoming) tangent
        let s_in_a = (1.0 - t) * (1.0 - c) * (1.0 + b) * 0.5;
        let s_in_b = (1.0 - t) * (1.0 + c) * (1.0 - b) * 0.5;
        let tangent_in = d_prev.scale(s_in_a).add(d_next.scale(s_in_b));

        // Destination (outgoing) tangent
        let s_out_a = (1.0 - t) * (1.0 + c) * (1.0 + b) * 0.5;
        let s_out_b = (1.0 - t) * (1.0 - c) * (1.0 - b) * 0.5;
        let tangent_out = d_prev.scale(s_out_a).add(d_next.scale(s_out_b));

        (tangent_in, tangent_out)
    }

    /// Evaluate at `t ∈ [0, segment_count]`.
    pub fn evaluate(&self, t: f32) -> T {
        let n_segs = self.segment_count() as f32;
        let t      = t.clamp(0.0, n_segs);
        let seg    = (t as usize).min(self.segment_count() - 1);
        let local  = t - seg as f32;

        let p0 = self.keys[seg].position.clone();
        let p1 = self.keys[seg + 1].position.clone();

        let (_, m0) = self.tangents(seg);
        let (m1, _) = self.tangents(seg + 1);

        evaluate_segment(p0, m0, p1, m1, local)
    }

    /// Velocity (first derivative) at `t`.
    pub fn velocity(&self, t: f32) -> T {
        let n_segs = self.segment_count() as f32;
        let t      = t.clamp(0.0, n_segs);
        let seg    = (t as usize).min(self.segment_count() - 1);
        let local  = t - seg as f32;

        let p0 = self.keys[seg].position.clone();
        let p1 = self.keys[seg + 1].position.clone();
        let (_, m0) = self.tangents(seg);
        let (m1, _) = self.tangents(seg + 1);

        derivative_segment(p0, m0, p1, m1, local)
    }

    /// Sample `n+1` uniformly-spaced points.
    pub fn sample_uniform(&self, n: usize, out: &mut [T]) {
        let total = self.segment_count() as f32;
        for i in 0..=n {
            out[i] = self.evaluate(i as f32 / n as f32 * total);
        }
    }
          }
