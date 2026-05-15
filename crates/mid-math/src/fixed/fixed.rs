// crates/mid-math/src/fixed/fixed.rs
//! Fixed-point scalar — deterministic on all platforms.
//!
//! Bit layout: i64 where the integer 1 << FRAC represents 1.0.
//! All arithmetic is integer-only. Never use in simulation code with f32 intermediates.
//!
//! FRAC must be in [1, 62]. Values outside this range produce UB in shifts.

use core::fmt;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// Fixed-point scalar with `FRAC` fractional bits stored as `i64`.
///
/// Scaling factor: integer `1 << FRAC` represents 1.0.
/// Deterministic on all platforms — no floating-point in arithmetic paths.
///
/// Aliases: `Fixed8` (1/256), `Fixed12` (1/4096), `Fixed16` (1/65536).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[repr(transparent)]
pub struct Fixed<const FRAC: u32>(pub(crate) i64);

impl<const FRAC: u32> Fixed<FRAC> {
    // ── Constants ─────────────────────────────────────────────────────────────

    pub const ZERO: Self = Fixed(0);
    pub const MAX:  Self = Fixed(i64::MAX);
    pub const MIN:  Self = Fixed(i64::MIN);

    // ── Constructors ──────────────────────────────────────────────────────────

    /// Wrap a raw `i64` without scaling — use when deserialising from network/disk.
    #[inline(always)] pub const fn from_raw(raw: i64) -> Self { Fixed(raw) }

    /// The raw underlying integer — use when serialising to network/disk.
    #[inline(always)] pub const fn to_raw(self) -> i64 { self.0 }

    /// Fixed-point 1.0 = `1 << FRAC`.
    #[inline(always)] pub fn one() -> Self { Fixed(1_i64 << FRAC) }

    /// Convert from `i32`. Exact — no precision loss.
    #[inline] pub fn from_i32(n: i32) -> Self { Fixed((n as i64) << FRAC) }

    /// Truncate to `i32` (floor for positive, round toward −∞ for negative).
    #[inline] pub fn to_i32_trunc(self) -> i32 { (self.0 >> FRAC) as i32 }

    /// Convert from `f32` — **lossy, boundary use only**.
    /// Never call inside the deterministic simulation loop.
    #[inline] pub fn from_f32(f: f32) -> Self { Fixed((f * (1_u64 << FRAC) as f32) as i64) }

    /// Convert to `f32` — **lossy, boundary use only**.
    #[inline] pub fn to_f32(self) -> f32 { self.0 as f32 / (1_u64 << FRAC) as f32 }

    /// Convert from `f64` — **lossy, boundary use only**.
    #[inline] pub fn from_f64(f: f64) -> Self { Fixed((f * (1_u64 << FRAC) as f64) as i64) }

    /// Convert to `f64` — **lossy, boundary use only**.
    #[inline] pub fn to_f64(self) -> f64 { self.0 as f64 / (1_u64 << FRAC) as f64 }

    // ── Core fixed-point arithmetic ───────────────────────────────────────────

    /// Fixed-point multiply via `i128` intermediate. Wraps on extreme overflow.
    #[inline]
    pub fn fixed_mul(self, rhs: Self) -> Self {
        Fixed(((self.0 as i128 * rhs.0 as i128) >> FRAC) as i64)
    }

    /// Fixed-point divide via `i128` intermediate. Panics on division by zero.
    #[inline]
    pub fn fixed_div(self, rhs: Self) -> Self {
        Fixed((((self.0 as i128) << FRAC) / rhs.0 as i128) as i64)
    }

    /// Checked multiply — `None` if result overflows `i64`.
    #[inline]
    pub fn checked_mul(self, rhs: Self) -> Option<Self> {
        let r = (self.0 as i128 * rhs.0 as i128) >> FRAC;
        if r > i64::MAX as i128 || r < i64::MIN as i128 { None } else { Some(Fixed(r as i64)) }
    }

    /// Checked divide — `None` on division by zero or overflow.
    #[inline]
    pub fn checked_div(self, rhs: Self) -> Option<Self> {
        if rhs.0 == 0 { return None; }
        let r = ((self.0 as i128) << FRAC) / rhs.0 as i128;
        if r > i64::MAX as i128 || r < i64::MIN as i128 { None } else { Some(Fixed(r as i64)) }
    }

    #[inline] pub fn saturating_add(self, rhs: Self) -> Self { Fixed(self.0.saturating_add(rhs.0)) }
    #[inline] pub fn saturating_sub(self, rhs: Self) -> Self { Fixed(self.0.saturating_sub(rhs.0)) }

    #[inline]
    pub fn saturating_mul(self, rhs: Self) -> Self {
        let r = (self.0 as i128 * rhs.0 as i128) >> FRAC;
        Fixed(r.clamp(i64::MIN as i128, i64::MAX as i128) as i64)
    }

    // ── Component math ────────────────────────────────────────────────────────

    /// Absolute value. Wraps on `Fixed::MIN` (identical to `i64::wrapping_abs`).
    #[inline] pub fn abs(self) -> Self { Fixed(self.0.wrapping_abs()) }

    /// Signum: `one()`, `ZERO`, or `-one()`.
    #[inline]
    pub fn signum(self) -> Self {
        match self.0.cmp(&0) {
            core::cmp::Ordering::Greater => Self::one(),
            core::cmp::Ordering::Equal   => Self::ZERO,
            core::cmp::Ordering::Less    => -Self::one(),
        }
    }

    /// Floor — round toward negative infinity.
    ///
    /// Arithmetic right-shift + left-shift clears fractional bits and
    /// correctly rounds toward −∞ for both positive and negative values.
    #[inline] pub fn floor(self) -> Self { Fixed((self.0 >> FRAC) << FRAC) }

    /// Ceil — round toward positive infinity.
    #[inline]
    pub fn ceil(self) -> Self {
        let scale = 1_i64 << FRAC;
        let mask  = scale - 1;
        if self.0 & mask != 0 {
            Fixed(self.0.saturating_add(mask) & !mask)
        } else {
            self
        }
    }

    /// Fractional part. Always non-negative for positive inputs.
    #[inline] pub fn fract(self) -> Self { Fixed(self.0 & ((1_i64 << FRAC) - 1)) }

    /// Clamp between `lo` and `hi`.
    #[inline] pub fn clamp(self, lo: Self, hi: Self) -> Self { Fixed(self.0.clamp(lo.0, hi.0)) }

    /// Component min.
    #[inline] pub fn min(self, rhs: Self) -> Self { if self.0 <= rhs.0 { self } else { rhs } }

    /// Component max.
    #[inline] pub fn max(self, rhs: Self) -> Self { if self.0 >= rhs.0 { self } else { rhs } }

    /// Linear interpolation: `self + (rhs - self) * t`.
    ///
    /// Uses `i128` for the multiply. `t` should be in `[ZERO, one()]`.
    #[inline]
    pub fn lerp(self, rhs: Self, t: Self) -> Self {
        let diff   = rhs.0 as i128 - self.0 as i128;
        let scaled = (diff * t.0 as i128) >> FRAC;
        Fixed((self.0 as i128 + scaled) as i64)
    }
}

// ── Operators ─────────────────────────────────────────────────────────────────

impl<const FRAC: u32> Add for Fixed<FRAC> {
    type Output = Self;
    #[inline] fn add(self, r: Self) -> Self { Fixed(self.0.wrapping_add(r.0)) }
}
impl<const FRAC: u32> AddAssign for Fixed<FRAC> {
    #[inline] fn add_assign(&mut self, r: Self) { *self = *self + r; }
}
impl<const FRAC: u32> Sub for Fixed<FRAC> {
    type Output = Self;
    #[inline] fn sub(self, r: Self) -> Self { Fixed(self.0.wrapping_sub(r.0)) }
}
impl<const FRAC: u32> SubAssign for Fixed<FRAC> {
    #[inline] fn sub_assign(&mut self, r: Self) { *self = *self - r; }
}
/// Fixed-point multiply — uses `i128` intermediate. Wraps on extreme overflow.
impl<const FRAC: u32> Mul for Fixed<FRAC> {
    type Output = Self;
    #[inline] fn mul(self, r: Self) -> Self { self.fixed_mul(r) }
}
impl<const FRAC: u32> MulAssign for Fixed<FRAC> {
    #[inline] fn mul_assign(&mut self, r: Self) { *self = self.fixed_mul(r); }
}
/// Fixed-point divide — panics on division by zero.
impl<const FRAC: u32> Div for Fixed<FRAC> {
    type Output = Self;
    #[inline] fn div(self, r: Self) -> Self { self.fixed_div(r) }
}
impl<const FRAC: u32> DivAssign for Fixed<FRAC> {
    #[inline] fn div_assign(&mut self, r: Self) { *self = self.fixed_div(r); }
}
impl<const FRAC: u32> Neg for Fixed<FRAC> {
    type Output = Self;
    #[inline] fn neg(self) -> Self { Fixed(self.0.wrapping_neg()) }
}

impl<const FRAC: u32> fmt::Debug for Fixed<FRAC> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Fixed<{}>({:.6})", FRAC, self.to_f64())
    }
}
impl<const FRAC: u32> fmt::Display for Fixed<FRAC> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.6}", self.to_f64())
    }
      }
