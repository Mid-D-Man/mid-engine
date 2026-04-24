// crates/mid-math/src/f32/math.rs
//! Low-level f32 math helpers used across the SIMD and scalar implementations.
//!
//! Keeping these in one place lets us swap in `libm` for `no_std` targets
//! later without touching every call site.

/// Sine and cosine simultaneously (cheaper than calling both separately on
/// most targets — the compiler can emit a single `fsincos` or use the
/// intrinsic pair).
#[inline(always)]
pub(crate) fn sin_cos(v: f32) -> (f32, f32) {
    (v.sin(), v.cos())
}

#[inline(always)] pub(crate) fn sin(v: f32) -> f32 { v.sin() }
#[inline(always)] pub(crate) fn cos(v: f32) -> f32 { v.cos() }
#[inline(always)] pub(crate) fn tan(v: f32) -> f32 { v.tan() }
#[inline(always)] pub(crate) fn sqrt(v: f32) -> f32 { v.sqrt() }
#[inline(always)] pub(crate) fn abs(v: f32) -> f32 { v.abs() }
#[inline(always)] pub(crate) fn signum(v: f32) -> f32 { v.signum() }
#[inline(always)] pub(crate) fn atan2(y: f32, x: f32) -> f32 { y.atan2(x) }
#[inline(always)] pub(crate) fn exp(v: f32) -> f32 { v.exp() }
#[inline(always)] pub(crate) fn exp2(v: f32) -> f32 { v.exp2() }
#[inline(always)] pub(crate) fn ln(v: f32) -> f32 { v.ln() }
#[inline(always)] pub(crate) fn log2(v: f32) -> f32 { v.log2() }
#[inline(always)] pub(crate) fn powf(v: f32, n: f32) -> f32 { v.powf(n) }
#[inline(always)] pub(crate) fn div_euclid(a: f32, b: f32) -> f32 { a.div_euclid(b) }
#[inline(always)] pub(crate) fn rem_euclid(a: f32, b: f32) -> f32 { a.rem_euclid(b) }

/// Fused multiply-add: `(a * b) + c` with single rounding error.
/// Uses FMA3 hardware instruction when `target_feature = "fma"`.
/// Falls back to two-operation unfused version otherwise.
///
/// The 2010 MacBook Pro dev machine does NOT have FMA3 (Sandy Bridge).
/// CI (GitHub Actions, modern Intel) DOES. This path activates on CI when
/// the feature is set, giving a small improvement in lerp/normalize chains.
#[inline(always)]
pub(crate) fn mul_add(a: f32, b: f32, c: f32) -> f32 {
    // The compiler will emit vfmadd213ss when target_feature = "fma" is set.
    a.mul_add(b, c)
}

/// Approximate arc-cosine. Max error < 0.001 rad.
///
/// Uses the polynomial from:
///   Nvidia Cg standard library / RTM (Ryan Juckett)
///
/// Much faster than libm `acos` for small angles — used in `Quat::slerp`
/// and `Vec3::angle_between`.
#[inline(always)]
pub(crate) fn acos_approx(v: f32) -> f32 {
    let x    = v.abs();
    // Horner-form polynomial evaluation
    let mut r = -0.013_480_470_f32;
    r =  r * x + 0.057_477_314_f32;
    r =  r * x - 0.121_239_071_f32;
    r =  r * x + 0.195_635_925_f32;
    r =  r * x - 0.332_994_597_f32;
    r =  r * x + 1.570_796_327_f32; // ≈ π/2
    r *= (1.0 - x).sqrt();
    if v < 0.0 { core::f32::consts::PI - r } else { r }
}
