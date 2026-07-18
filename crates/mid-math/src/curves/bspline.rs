// crates/mid-math/src/curves/bspline.rs
//! B-spline — approximating, C² continuous, local control.
//!
//! Formula (de Boor's algorithm):
//!   S(t) = Σ N_{i,k}(t) · P_i
//!
//! where N_{i,k} are the B-spline basis functions of degree k,
//! defined recursively via the Cox-de Boor recursion:
//!   N_{i,1}(t) = 1 if t_{i} ≤ t < t_{i+1}, else 0
//!   N_{i,k}(t) = (t-t_i)/(t_{i+k-1}-t_i) · N_{i,k-1}(t)
//!              + (t_{i+k}-t)/(t_{i+k}-t_{i+1}) · N_{i+1,k-1}(t)
//!
//! Properties:
//!   - Does NOT pass through control points (approximating).
//!   - C^(k-2) continuous for degree k. Cubic (k=4) gives C².
//!   - Local control — moving one point affects only k adjacent segments.
//!   - Uniform knot vector → equally weighted basis functions.
//!
//! Implemented here: uniform cubic B-spline (degree 3, order 4, k=4).
//! This is the most common variant in game engines — same as the
//! "box spline" used in subdivision surface limit curves.
//!
//! Game use: cloth simulation, smooth path approximation when exact
//!           interpolation is not required, hair/ribbon rendering.

use super::interpolate::Interpolate;
use super::CURVE_N;
use crate::MidVec;

/// Uniform cubic B-spline. Degree 3, C² continuous.
///
/// For `n` control points the curve has `n - 3` segments.
/// Minimum 4 control points required.
///
/// The curve does NOT pass through the first or last control points —
/// it starts and ends near (but not at) them. Use `CatmullRom` if
/// you need the curve to interpolate all points.
#[derive(Clone, Debug)]
pub struct BSpline<T> {
    pub control_points: MidVec<T, CURVE_N>,
}

impl<T: Interpolate + Clone> BSpline<T> {
    #[inline]
    pub fn new(control_points: Vec<T>) -> Self {
        assert!(
            control_points.len() >= 4,
            "BSpline (cubic) requires at least 4 control points"
        );
        Self { control_points: MidVec::from_vec_or_inline(control_points) }
    }

    /// Number of curve segments.
    #[inline]
    pub fn segment_count(&self) -> usize { self.control_points.len() - 3 }

    /// Evaluate the spline at `t ∈ [0, segment_count]`.
    ///
    /// Uses the closed-form uniform cubic B-spline basis matrix:
    ///
    /// M_bs = (1/6) * [ -1  3  -3  1 ]
    ///                [  3 -6   3  0 ]
    ///                [ -3  0   3  0 ]
    ///                [  1  4   1  0 ]
    ///
    /// P(t) = [t³ t² t 1] · M_bs · [P_i, P_{i+1}, P_{i+2}, P_{i+3}]ᵀ
    pub fn evaluate(&self, t: f32) -> T {
        let n_segs = self.segment_count() as f32;
        let t      = t.clamp(0.0, n_segs);
        let seg    = (t as usize).min(self.segment_count() - 1);
        let local  = t - seg as f32;

        let p0 = self.control_points[seg].clone();
        let p1 = self.control_points[seg + 1].clone();
        let p2 = self.control_points[seg + 2].clone();
        let p3 = self.control_points[seg + 3].clone();

        cubic_bspline_segment(p0, p1, p2, p3, local)
    }

    /// Tangent (first derivative) at `t`.
    pub fn tangent(&self, t: f32) -> T {
        let n_segs = self.segment_count() as f32;
        let t      = t.clamp(0.0, n_segs);
        let seg    = (t as usize).min(self.segment_count() - 1);
        let local  = t - seg as f32;

        let p0 = self.control_points[seg].clone();
        let p1 = self.control_points[seg + 1].clone();
        let p2 = self.control_points[seg + 2].clone();
        let p3 = self.control_points[seg + 3].clone();

        cubic_bspline_derivative(p0, p1, p2, p3, local)
    }

    /// Sample `n+1` uniformly-spaced points.
    pub fn sample_uniform(&self, n: usize, out: &mut [T]) {
        let total = self.segment_count() as f32;
        for i in 0..=n {
            out[i] = self.evaluate(i as f32 / n as f32 * total);
        }
    }

    /// Approximate arc length by sampling `segments` linear pieces.
    pub fn arc_length_approx(&self, segments: usize) -> f32
    where T: Into<crate::Vec3> + Copy
    {
        let total  = self.segment_count() as f32;
        let mut len = 0.0f32;
        let mut prev: crate::Vec3 = self.evaluate(0.0).into();
        for i in 1..=segments {
            let curr: crate::Vec3 = self.evaluate(i as f32 / segments as f32 * total).into();
            len += (curr - prev).length();
            prev = curr;
        }
        len
    }
}

// ── Basis functions ───────────────────────────────────────────────────────────

/// Evaluate one uniform cubic B-spline segment.
///
/// Matrix form:
///   P(t) = (1/6) * ((-t³+3t²-3t+1)P0 + (3t³-6t²+4)P1 + (-3t³+3t²+3t+1)P2 + t³P3)
fn cubic_bspline_segment<T: Interpolate>(p0: T, p1: T, p2: T, p3: T, t: f32) -> T {
    let t2 = t * t;
    let t3 = t2 * t;
    let inv6 = 1.0 / 6.0;

    let c0 = inv6 * (-t3 + 3.0*t2 - 3.0*t + 1.0);
    let c1 = inv6 * ( 3.0*t3 - 6.0*t2 + 4.0);
    let c2 = inv6 * (-3.0*t3 + 3.0*t2 + 3.0*t + 1.0);
    let c3 = inv6 *  t3;

    p0.scale(c0)
        .add(p1.scale(c1))
        .add(p2.scale(c2))
        .add(p3.scale(c3))
}

/// Derivative of the cubic B-spline segment.
///
/// dP/dt = (1/6) * ((-3t²+6t-3)P0 + (9t²-12t)P1 + (-9t²+6t+3)P2 + 3t²P3)
fn cubic_bspline_derivative<T: Interpolate>(p0: T, p1: T, p2: T, p3: T, t: f32) -> T {
    let t2 = t * t;
    let inv6 = 1.0 / 6.0;

    let c0 = inv6 * (-3.0*t2 + 6.0*t - 3.0);
    let c1 = inv6 * ( 9.0*t2 - 12.0*t);
    let c2 = inv6 * (-9.0*t2 +  6.0*t + 3.0);
    let c3 = inv6 *  3.0*t2;

    p0.scale(c0)
        .add(p1.scale(c1))
        .add(p2.scale(c2))
        .add(p3.scale(c3))
}
