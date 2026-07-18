// crates/mid-math/src/curves/cardinal.rs
//! Cardinal spline — interpolating with a tension parameter.
//!
//! A cardinal spline is a Catmull-Rom spline with a global tension
//! parameter `c` that scales all tangent vectors uniformly:
//!
//!   Tangent at P[i] = (1 - c) * 0.5 * (P[i+1] - P[i-1])
//!
//! When `c = 0` this is exactly Catmull-Rom.
//! When `c = 1` all tangents are zero → piecewise linear interpolation.
//! When `c < 0` the curve overshoots (looping behaviour).
//!
//! Game use: tension control for smooth vs tight camera paths,
//!           fluid simulation particle trails, rope physics visualisation.

use super::interpolate::Interpolate;
use super::hermite::evaluate_segment;
use super::CURVE_N;
use crate::MidVec;

/// Cardinal spline through a list of control points.
///
/// `tension ∈ [0, 1]` — 0 = Catmull-Rom, 1 = linear interpolation.
/// Values outside [0,1] produce overshoot (intentional in some animation styles).
#[derive(Clone, Debug)]
pub struct CardinalSpline<T> {
    pub points:  MidVec<T, CURVE_N>,
    /// Tension parameter. 0 = Catmull-Rom tangents. 1 = straight lines.
    pub tension: f32,
}

impl<T: Interpolate + Clone> CardinalSpline<T> {
    /// Create with given tension.
    #[inline]
    pub fn new(points: Vec<T>, tension: f32) -> Self {
        assert!(points.len() >= 2, "CardinalSpline requires at least 2 points");
        Self { points: MidVec::from_vec_or_inline(points), tension }
    }

    /// Create with tension = 0 (equivalent to uniform Catmull-Rom).
    #[inline]
    pub fn catmull_rom(points: Vec<T>) -> Self { Self::new(points, 0.0) }

    /// Number of segments.
    #[inline]
    pub fn segment_count(&self) -> usize { self.points.len() - 1 }

    /// Tangent at index `i`.
    #[inline]
    fn tangent_at(&self, i: usize) -> T {
        let n = self.points.len();
        let scale = (1.0 - self.tension) * 0.5;

        let prev = if i == 0 {
            // Reflect: 2*p0 - p1
            self.points[0].scale(2.0).sub(self.points[1].clone())
        } else {
            self.points[i - 1].clone()
        };

        let next = if i + 1 >= n {
            // Reflect: 2*p_{n-1} - p_{n-2}
            self.points[n - 1].scale(2.0).sub(self.points[n - 2].clone())
        } else {
            self.points[i + 1].clone()
        };

        next.sub(prev).scale(scale)
    }

    /// Evaluate at `t ∈ [0, segment_count]`.
    pub fn evaluate(&self, t: f32) -> T {
        let n_segs = self.segment_count() as f32;
        let t      = t.clamp(0.0, n_segs);
        let seg    = (t as usize).min(self.segment_count() - 1);
        let local  = t - seg as f32;

        let p0 = self.points[seg].clone();
        let p1 = self.points[seg + 1].clone();
        let m0 = self.tangent_at(seg);
        let m1 = self.tangent_at(seg + 1);

        evaluate_segment(p0, m0, p1, m1, local)
    }

    /// Sample `n+1` uniformly-spaced points.
    pub fn sample_uniform(&self, n: usize, out: &mut [T]) {
        let total = self.segment_count() as f32;
        for i in 0..=n {
            out[i] = self.evaluate(i as f32 / n as f32 * total);
        }
    }
}
