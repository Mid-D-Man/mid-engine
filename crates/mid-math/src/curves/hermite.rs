// crates/mid-math/src/curves/hermite.rs
//! Hermite spline — defined by positions AND tangent vectors at each point.
//!
//! Formula (one segment, positions P0/P1, tangents M0/M1):
//!   P(t) = h00(t)·P0 + h10(t)·M0 + h01(t)·P1 + h11(t)·M1
//!
//! where the Hermite basis functions are:
//!   h00(t) =  2t³ - 3t² + 1   (blend from P0)
//!   h10(t) =   t³ - 2t² + t   (scale M0 tangent)
//!   h01(t) = -2t³ + 3t²       (blend to P1)
//!   h11(t) =   t³ -  t²       (scale M1 tangent)
//!
//! Properties:
//!   - C¹ continuous — matches both position AND velocity at endpoints.
//!   - The tangent vectors are explicit (not computed from neighbours).
//!   - Direct tangent control makes this ideal for animation curve editors.
//!
//! Game use: Unity/Unreal animation windows use Hermite internally.
//!           Physics velocity integration. Procedural walk cycle blending.

use super::interpolate::Interpolate;
use super::CURVE_N;
use crate::MidVec;

/// A single keyframe — position + incoming and outgoing tangent.
///
/// For C¹ continuity across keyframes, the outgoing tangent of one
/// keyframe must equal the incoming tangent of the next.
/// Set them independently for C⁰ (corner) transitions.
#[derive(Clone, Copy, Debug)]
pub struct HermiteKey<T> {
    pub position: T,
    /// Tangent leaving this keyframe (outgoing velocity).
    pub tangent_out: T,
    /// Tangent arriving at this keyframe (incoming velocity).
    /// Equal to `tangent_out` for smooth (C¹) behaviour.
    pub tangent_in:  T,
}

impl<T: Interpolate + Clone> HermiteKey<T> {
    /// Smooth key — both tangents are the same (C¹ at this keyframe).
    #[inline]
    pub fn smooth(position: T, tangent: T) -> Self {
        Self { position, tangent_out: tangent.clone(), tangent_in: tangent }
    }

    /// Corner key — tangents differ (C⁰ at this keyframe, velocity discontinuity).
    #[inline]
    pub fn corner(position: T, tangent_in: T, tangent_out: T) -> Self {
        Self { position, tangent_in, tangent_out }
    }
}

impl<T: Default> Default for HermiteKey<T> {
    /// All-zero key (tangents default to zero, matching `Vec3::default()`).
    /// Not used by the crate's own curve logic — added so this type can
    /// satisfy `tinyvec::Array::Item: Default`, which `vs_mid_vec`'s
    /// benchmarks need for `TinyVec`/`ArrayVec` comparisons.
    #[inline]
    fn default() -> Self {
        Self { position: T::default(), tangent_out: T::default(), tangent_in: T::default() }
    }
}

/// Hermite spline through a sequence of `HermiteKey` values.
///
/// Each segment is defined by two consecutive keys.
/// Minimum 2 keys required.
#[derive(Clone, Debug)]
pub struct HermiteSpline<T> {
    pub keys: MidVec<HermiteKey<T>, CURVE_N>,
}

impl<T: Interpolate + Clone> HermiteSpline<T> {
    #[inline]
    pub fn new(keys: Vec<HermiteKey<T>>) -> Self {
        assert!(keys.len() >= 2, "HermiteSpline requires at least 2 keys");
        Self { keys: MidVec::from_vec_or_inline(keys) }
    }

    /// Number of segments.
    #[inline]
    pub fn segment_count(&self) -> usize { self.keys.len() - 1 }

    /// Evaluate the spline at `t ∈ [0, segment_count]`.
    pub fn evaluate(&self, t: f32) -> T {
        let n_segs = self.segment_count() as f32;
        let t = t.clamp(0.0, n_segs);
        let seg = (t as usize).min(self.segment_count() - 1);
        let local = t - seg as f32;

        let k0 = &self.keys[seg];
        let k1 = &self.keys[seg + 1];

        evaluate_segment(
            k0.position.clone(),
            k0.tangent_out.clone(),
            k1.position.clone(),
            k1.tangent_in.clone(),
            local,
        )
    }

    /// First derivative (velocity) at `t`.
    pub fn velocity(&self, t: f32) -> T {
        let n_segs = self.segment_count() as f32;
        let t = t.clamp(0.0, n_segs);
        let seg = (t as usize).min(self.segment_count() - 1);
        let local = t - seg as f32;

        let k0 = &self.keys[seg];
        let k1 = &self.keys[seg + 1];

        derivative_segment(
            k0.position.clone(),
            k0.tangent_out.clone(),
            k1.position.clone(),
            k1.tangent_in.clone(),
            local,
        )
    }

    /// Sample `n+1` uniformly-spaced points.
    pub fn sample_uniform(&self, n: usize, out: &mut [T]) {
        let total = self.segment_count() as f32;
        for i in 0..=n {
            out[i] = self.evaluate(i as f32 / n as f32 * total);
        }
    }
}

// ── Segment evaluation ────────────────────────────────────────────────────────

/// Hermite basis functions evaluated at `t`.
///
/// Returns `(h00, h10, h01, h11)`.
#[inline(always)]
fn hermite_basis(t: f32) -> (f32, f32, f32, f32) {
    let t2 = t * t;
    let t3 = t2 * t;
    let h00 =  2.0 * t3 - 3.0 * t2 + 1.0;
    let h10 =        t3 - 2.0 * t2 + t;
    let h01 = -2.0 * t3 + 3.0 * t2;
    let h11 =        t3 -       t2;
    (h00, h10, h01, h11)
}

/// Derivatives of the Hermite basis functions.
#[inline(always)]
fn hermite_basis_derivative(t: f32) -> (f32, f32, f32, f32) {
    let t2 = t * t;
    let dh00 =  6.0 * t2 - 6.0 * t;
    let dh10 =  3.0 * t2 - 4.0 * t + 1.0;
    let dh01 = -6.0 * t2 + 6.0 * t;
    let dh11 =  3.0 * t2 - 2.0 * t;
    (dh00, dh10, dh01, dh11)
}

pub(super) fn evaluate_segment<T: Interpolate>(
    p0: T, m0: T, p1: T, m1: T, t: f32,
) -> T {
    let (h00, h10, h01, h11) = hermite_basis(t);
    p0.scale(h00)
        .add(m0.scale(h10))
        .add(p1.scale(h01))
        .add(m1.scale(h11))
}

pub(super) fn derivative_segment<T: Interpolate>(
    p0: T, m0: T, p1: T, m1: T, t: f32,
) -> T {
    let (dh00, dh10, dh01, dh11) = hermite_basis_derivative(t);
    p0.scale(dh00)
        .add(m0.scale(dh10))
        .add(p1.scale(dh01))
        .add(m1.scale(dh11))
    }
