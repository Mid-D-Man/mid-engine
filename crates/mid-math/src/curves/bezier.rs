// crates/mid-math/src/curves/bezier.rs
//! Bézier curves — quadratic and cubic.
//!
//! Formula (cubic, de Casteljau expanded):
//!   B(t) = (1-t)³·P0 + 3(1-t)²t·P1 + 3(1-t)t²·P2 + t³·P3
//!
//! Game use: UI animation, simple camera arcs, font rendering, projectile arcs.
//! Approximating — the curve does NOT pass through P1 and P2 (control points).
//!
//! For a curve that passes through all points, use `CatmullRom` instead.

use crate::Vec3;
use super::interpolate::Interpolate;

// ── Quadratic Bézier ──────────────────────────────────────────────────────────

/// Quadratic Bézier curve — 3 control points.
///
/// B(t) = (1-t)²·P0 + 2(1-t)t·P1 + t²·P2
///
/// The curve passes through P0 (t=0) and P2 (t=1); P1 is the control handle.
#[derive(Clone, Copy, Debug)]
pub struct QuadraticBezier<T> {
    pub p0: T,
    /// Control handle — curve is attracted toward this point.
    pub p1: T,
    pub p2: T,
}

impl<T: Interpolate> QuadraticBezier<T> {
    #[inline(always)]
    pub fn new(p0: T, p1: T, p2: T) -> Self { Self { p0, p1, p2 } }

    /// Evaluate the curve at parameter `t ∈ [0, 1]`.
    ///
    /// Uses de Casteljau's algorithm — numerically stable, no polynomial expansion.
    #[inline]
    pub fn evaluate(&self, t: f32) -> T {
        // Level 1: linear interpolation between adjacent control points
        let q0 = self.p0.lerp(self.p1, t);
        let q1 = self.p1.lerp(self.p2, t);
        // Level 2: interpolate between level-1 results
        q0.lerp(q1, t)
    }

    /// Tangent (first derivative) at `t`.
    ///
    /// B'(t) = 2[(1-t)(P1-P0) + t(P2-P1)]
    #[inline]
    pub fn tangent(&self, t: f32) -> T {
        let d0 = self.p1.sub(self.p0).scale(2.0);
        let d1 = self.p2.sub(self.p1).scale(2.0);
        d0.lerp(d1, t)
    }

    /// Sample `n+1` evenly-spaced points into `out`.
    /// `out` must have length ≥ `n + 1`.
    pub fn sample_uniform(&self, n: usize, out: &mut [T]) {
        for i in 0..=n {
            out[i] = self.evaluate(i as f32 / n as f32);
        }
    }

    /// Approximate arc length by sampling `segments` line segments.
    pub fn arc_length(&self, segments: usize) -> f32
    where T: Into<Vec3> + Copy
    {
        let mut length = 0.0f32;
        let mut prev: Vec3 = self.evaluate(0.0).into();
        for i in 1..=segments {
            let curr: Vec3 = self.evaluate(i as f32 / segments as f32).into();
            length += (curr - prev).length();
            prev = curr;
        }
        length
    }
}

// ── Cubic Bézier ──────────────────────────────────────────────────────────────

/// Cubic Bézier curve — 4 control points.
///
/// B(t) = (1-t)³·P0 + 3(1-t)²t·P1 + 3(1-t)t²·P2 + t³·P3
///
/// The curve passes through P0 (t=0) and P3 (t=1).
/// P1 and P2 are tangent handles — the curve is "pulled toward" them
/// but does not pass through them.
///
/// To join two segments smoothly (C¹), ensure the tangent handles are
/// collinear across the join: the last handle of segment A and the first
/// handle of segment B should mirror each other through the join point.
#[derive(Clone, Copy, Debug)]
pub struct CubicBezier<T> {
    pub p0: T,
    /// First tangent handle.
    pub p1: T,
    /// Second tangent handle.
    pub p2: T,
    pub p3: T,
}

impl<T: Interpolate> CubicBezier<T> {
    #[inline(always)]
    pub fn new(p0: T, p1: T, p2: T, p3: T) -> Self { Self { p0, p1, p2, p3 } }

    /// Evaluate at `t ∈ [0, 1]` using de Casteljau's algorithm.
    ///
    /// 3 rounds of linear interpolation — numerically identical to the
    /// Bernstein polynomial form but significantly more stable near t=0 and t=1.
    #[inline]
    pub fn evaluate(&self, t: f32) -> T {
        // Round 1
        let q0 = self.p0.lerp(self.p1, t);
        let q1 = self.p1.lerp(self.p2, t);
        let q2 = self.p2.lerp(self.p3, t);
        // Round 2
        let r0 = q0.lerp(q1, t);
        let r1 = q1.lerp(q2, t);
        // Round 3
        r0.lerp(r1, t)
    }

    /// Tangent (first derivative) at `t`.
    ///
    /// B'(t) = 3[(1-t)²(P1-P0) + 2(1-t)t(P2-P1) + t²(P3-P2)]
    #[inline]
    pub fn tangent(&self, t: f32) -> T {
        let u = 1.0 - t;
        let d0 = self.p1.sub(self.p0).scale(3.0 * u * u);
        let d1 = self.p2.sub(self.p1).scale(6.0 * u * t);
        let d2 = self.p3.sub(self.p2).scale(3.0 * t * t);
        d0.add(d1).add(d2)
    }

    /// Second derivative (curvature direction) at `t`.
    ///
    /// B''(t) = 6[(1-t)(P2-2P1+P0) + t(P3-2P2+P1)]
    #[inline]
    pub fn second_derivative(&self, t: f32) -> T {
        let u = 1.0 - t;
        // (P2 - 2P1 + P0)
        let a = self.p2.sub(self.p1.scale(2.0)).add(self.p0).scale(6.0 * u);
        // (P3 - 2P2 + P1)
        let b = self.p3.sub(self.p2.scale(2.0)).add(self.p1).scale(6.0 * t);
        a.add(b)
    }

    /// Split the curve at `t` into two cubic Béziers.
    /// Returns `(left, right)` where `left.p3 == right.p0`.
    pub fn split(&self, t: f32) -> (CubicBezier<T>, CubicBezier<T>) {
        let q0 = self.p0.lerp(self.p1, t);
        let q1 = self.p1.lerp(self.p2, t);
        let q2 = self.p2.lerp(self.p3, t);
        let r0 = q0.lerp(q1, t);
        let r1 = q1.lerp(q2, t);
        let m  = r0.lerp(r1, t);
        (
            CubicBezier::new(self.p0, q0, r0, m),
            CubicBezier::new(m, r1, q2, self.p3),
        )
    }

    /// Approximate arc length by sampling `segments` line segments.
    pub fn arc_length(&self, segments: usize) -> f32
    where T: Into<Vec3> + Copy
    {
        let mut length = 0.0f32;
        let mut prev: Vec3 = self.evaluate(0.0).into();
        for i in 1..=segments {
            let curr: Vec3 = self.evaluate(i as f32 / segments as f32).into();
            length += (curr - prev).length();
            prev = curr;
        }
        length
    }

    /// Sample `n+1` points uniformly into `out`.
    pub fn sample_uniform(&self, n: usize, out: &mut [T]) {
        for i in 0..=n {
            out[i] = self.evaluate(i as f32 / n as f32);
        }
    }
  }
