// crates/mid-math/src/storage/bf16.rs
//! BFloat16 (brain float 16) — the easiest low-precision type to implement.
//!
//! ## Why bf16 is almost free
//!
//! An f32 has 32 bits: `[s:1][e:8][m:23]`.
//! A bf16 has 16 bits: `[s:1][e:8][m:7]`.
//!
//! They share the **same sign and exponent layout** — only the mantissa
//! is truncated.  This means:
//!
//! * **bf16 → f32**: zero-extend the u16 to u32 and reinterpret as f32.
//!   Zero CPU cycles — one instruction.
//! * **f32 → bf16**: shift the bits right by 16 (with rounding) and store
//!   the upper half.  Two to four instructions.
//!
//! ## bf16 vs f16 — which to use
//!
//! | Property          | `bf16`                      | `f16`                         |
//! |-------------------|-----------------------------|-------------------------------|
//! | Exponent bits     | 8 (same range as f32)       | 5 (limited: ±65504 max)       |
//! | Mantissa bits     | 7 (~2 significant decimals) | 10 (~3 significant decimals)  |
//! | f32 overflow risk | None                        | Yes — values >65504 blow up   |
//! | ML training use   | ✅ Preferred (PyTorch, JAX)  | Less common for training      |
//! | GPU texture use   | ❌ Not a texture format      | ✅ Native GPU texture format  |
//! | Animation data    | ❌ Range fine, precision low | ✅ Better precision            |
//!
//! **Rule of thumb:** use `bf16` when you need the same *range* as f32 with
//! less RAM (ML models, large embeddings).  Use `f16` when you need GPU
//! texture compatibility or slightly better precision (animation, normals).
//!
//! ## f4 note
//! When f4 is added, it will follow the same truncation philosophy:
//! take the top bits, round, store.  The complexity grows with the mantissa
//! shrinkage, not the sign/exponent logic.

use core::{
    fmt,
    hash::{Hash, Hasher},
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

// ═════════════════════════════════════════════════════════════════════════════
// Type definition
// ═════════════════════════════════════════════════════════════════════════════

/// BFloat16 — upper 16 bits of an IEEE 754 f32.
///
/// **Storage only — call `.to_f32()` before arithmetic.**
///
/// * `size_of::<bf16>() == 2` always.
/// * `bf16 → f32` is free (zero-extend in a register).
/// * Same dynamic range as f32 (no overflow risk when converting from f32).
/// * Less precision than `f16` (7 mantissa bits vs 10).
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Default)]
#[repr(transparent)]
pub struct bf16(u16);

// ═════════════════════════════════════════════════════════════════════════════
// Constants
// ═════════════════════════════════════════════════════════════════════════════

impl bf16 {
    /// +0.0
    pub const ZERO:         Self = Self(0x0000);
    /// -0.0
    pub const NEG_ZERO:     Self = Self(0x8000);
    /// 1.0
    pub const ONE:          Self = Self(0x3F80);
    /// -1.0
    pub const NEG_ONE:      Self = Self(0xBF80);
    /// +∞  (same exponent encoding as f32 +∞, just top 16 bits)
    pub const INFINITY:     Self = Self(0x7F80);
    /// -∞
    pub const NEG_INFINITY: Self = Self(0xFF80);
    /// Canonical quiet NaN
    pub const NAN:          Self = Self(0x7FC0);
    /// Maximum finite ≈ 3.39e38 (same as f32::MAX, just less precise)
    pub const MAX:          Self = Self(0x7F7F);
    /// Minimum finite ≈ -3.39e38
    pub const MIN:          Self = Self(0xFF7F);
    /// Smallest positive normal ≈ 1.175e-38 (same as f32 MIN_POSITIVE)
    pub const MIN_POSITIVE: Self = Self(0x0080);
    /// Machine epsilon = 2^-7 ≈ 0.0078125
    pub const EPSILON:      Self = Self(0x3C00);
    /// π ≈ 3.140625 (nearest bf16)
    pub const PI:           Self = Self(0x4049);
    /// e ≈ 2.71875
    pub const E:            Self = Self(0x402E);
}

// ═════════════════════════════════════════════════════════════════════════════
// Construction & extraction
// ═════════════════════════════════════════════════════════════════════════════

impl bf16 {
    /// Construct from raw bf16 bits (upper 16 bits of the f32 layout).
    #[inline(always)]
    pub const fn from_bits(bits: u16) -> Self { Self(bits) }

    /// Return raw bits.
    #[inline(always)]
    pub const fn to_bits(self) -> u16 { self.0 }

    /// Convert `f32` → `bf16` (lossy — rounds to nearest even).
    ///
    /// Unlike `f16`, this never overflows — bf16 has the same exponent range.
    #[inline]
    pub fn from_f32(v: f32) -> Self {
        let bits = v.to_bits();
        // NaN: preserve the quiet bit to ensure the result is still NaN
        if v.is_nan() {
            return Self(((bits >> 16) | 0x0040) as u16);
        }
        // Round to nearest even using bits 0–15 as the rounding region.
        // round_bit = bit 15 of the lower half (first dropped bit).
        // sticky   = bits 14:0 of the lower half.
        let lower      = bits & 0x0000_FFFFu32;
        let round_bit  = (bits >> 15) & 1;            // bit 15 of lower half
        let sticky     = bits & 0x0000_7FFFu32;       // bits 14:0
        let mut upper  = (bits >> 16) as u16;
        // Round up if: round_bit set AND (sticky nonzero OR LSB of result is 1)
        if round_bit != 0 && (sticky != 0 || upper & 1 != 0) {
            upper = upper.wrapping_add(1);
        }
        let _ = lower; // used structurally above
        // A round-up can carry out of the mantissa into the exponent field,
        // which for values near f32::MAX pushes the exponent to all-ones —
        // i.e. rounding just turned a finite input into bf16 infinity. bf16
        // has the same exponent range as f32, so no *finite* f32 should ever
        // become non-finite here; saturate to the largest finite magnitude
        // instead (same convention used by PyTorch/TF bf16 casts).
        if v.is_finite() && (upper & 0x7F80) == 0x7F80 {
            upper = (upper & 0x8000) | 0x7F7F;
        }
        Self(upper)
    }

    /// Convert `bf16` → `f32`. **One instruction** — zero-extend the u16.
    #[inline(always)]
    pub fn to_f32(self) -> f32 {
        f32::from_bits((self.0 as u32) << 16)
    }

    /// Convert `f64` → `bf16` (lossy).
    #[inline]
    pub fn from_f64(v: f64) -> Self { Self::from_f32(v as f32) }

    /// Convert `bf16` → `f64`.
    #[inline]
    pub fn to_f64(self) -> f64 { self.to_f32() as f64 }
}

// ═════════════════════════════════════════════════════════════════════════════
// Classification
// ═════════════════════════════════════════════════════════════════════════════

impl bf16 {
    #[inline] pub fn is_nan(self)           -> bool { self.0 & 0x7FFF > 0x7F80 }
    #[inline] pub fn is_infinite(self)      -> bool { self.0 & 0x7FFF == 0x7F80 }
    #[inline] pub fn is_finite(self)        -> bool { self.0 & 0x7F80 != 0x7F80 }
    #[inline] pub fn is_subnormal(self)     -> bool { self.0 & 0x7F80 == 0 && self.0 & 0x007F != 0 }
    #[inline] pub fn is_normal(self)        -> bool { let e = self.0 & 0x7F80; e != 0 && e != 0x7F80 }
    #[inline] pub fn is_zero(self)          -> bool { self.0 & 0x7FFF == 0 }
    #[inline] pub fn is_sign_positive(self) -> bool { self.0 & 0x8000 == 0 }
    #[inline] pub fn is_sign_negative(self) -> bool { self.0 & 0x8000 != 0 }
    #[inline] pub fn abs(self)              -> Self { Self(self.0 & 0x7FFF) }
    #[inline] pub fn copysign(self, sign_src: Self) -> Self {
        Self((self.0 & 0x7FFF) | (sign_src.0 & 0x8000))
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Arithmetic (f32 round-trip)
// ═════════════════════════════════════════════════════════════════════════════

impl Neg for bf16 {
    type Output = Self;
    #[inline] fn neg(self) -> Self { Self(self.0 ^ 0x8000) }
}

macro_rules! impl_bf16_arith {
    ($Trait:ident, $fn:ident, $Assign:ident, $afn:ident, $op:tt) => {
        impl $Trait for bf16 {
            type Output = Self;
            #[inline] fn $fn(self, rhs: Self) -> Self {
                Self::from_f32(self.to_f32() $op rhs.to_f32())
            }
        }
        impl $Assign for bf16 {
            #[inline] fn $afn(&mut self, rhs: Self) { *self = *self $op rhs; }
        }
        impl $Trait<f32> for bf16 {
            type Output = f32;
            #[inline] fn $fn(self, rhs: f32) -> f32 { self.to_f32() $op rhs }
        }
    };
}
impl_bf16_arith!(Add, add, AddAssign, add_assign, +);
impl_bf16_arith!(Sub, sub, SubAssign, sub_assign, -);
impl_bf16_arith!(Mul, mul, MulAssign, mul_assign, *);
impl_bf16_arith!(Div, div, DivAssign, div_assign, /);

// ═════════════════════════════════════════════════════════════════════════════
// Batch helpers
// ═════════════════════════════════════════════════════════════════════════════

/// Convert 4 × f32 → 4 × bf16.
/// On modern x86 the compiler will use two `VPSRLD` + `VMOVDQU` instructions.
#[inline]
pub fn f32x4_to_bf16x4(v: [f32; 4]) -> [bf16; 4] { v.map(bf16::from_f32) }

/// Convert 4 × bf16 → 4 × f32.  Near-zero cost — just zero-extend each u16.
#[inline]
pub fn bf16x4_to_f32x4(v: [bf16; 4]) -> [f32; 4] { v.map(bf16::to_f32) }

/// Convert 8 × f32 → 8 × bf16.
#[inline]
pub fn f32x8_to_bf16x8(v: [f32; 8]) -> [bf16; 8] { v.map(bf16::from_f32) }

/// Convert 8 × bf16 → 8 × f32.
#[inline]
pub fn bf16x8_to_f32x8(v: [bf16; 8]) -> [f32; 8] { v.map(bf16::to_f32) }

/// Batch-convert a slice of `f32` → `bf16`. Panics if lengths differ.
pub fn f32_slice_to_bf16(src: &[f32], dst: &mut [bf16]) {
    assert_eq!(src.len(), dst.len(), "f32_slice_to_bf16: length mismatch");
    for (d, &s) in dst.iter_mut().zip(src.iter()) { *d = bf16::from_f32(s); }
}

/// Batch-convert a slice of `bf16` → `f32`. Panics if lengths differ.
pub fn bf16_slice_to_f32(src: &[bf16], dst: &mut [f32]) {
    assert_eq!(src.len(), dst.len(), "bf16_slice_to_f32: length mismatch");
    for (d, &s) in dst.iter_mut().zip(src.iter()) { *d = s.to_f32(); }
}

// ═════════════════════════════════════════════════════════════════════════════
// From / Into
// ═════════════════════════════════════════════════════════════════════════════

impl From<f32>   for bf16 { #[inline] fn from(v: f32)  -> Self { Self::from_f32(v) } }
impl From<f64>   for bf16 { #[inline] fn from(v: f64)  -> Self { Self::from_f64(v) } }
impl From<bf16>  for f32  { #[inline] fn from(v: bf16) -> f32  { v.to_f32() } }
impl From<bf16>  for f64  { #[inline] fn from(v: bf16) -> f64  { v.to_f64() } }

// ═════════════════════════════════════════════════════════════════════════════
// Comparison & hashing
// ═════════════════════════════════════════════════════════════════════════════

impl PartialEq for bf16 {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        if self.is_nan() || other.is_nan() { return false; }
        (self.0 == other.0) || ((self.0 | other.0) & 0x7FFF == 0)
    }
}

impl PartialOrd for bf16 {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        if self.is_nan() || other.is_nan() { return None; }
        self.to_f32().partial_cmp(&other.to_f32())
    }
}

impl Hash for bf16 {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        let bits = if self.0 & 0x7FFF == 0 { 0u16 } else { self.0 };
        bits.hash(state);
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Formatting
// ═════════════════════════════════════════════════════════════════════════════

impl fmt::Debug   for bf16 { fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "bf16({:?})", self.to_f32()) } }
impl fmt::Display for bf16 { fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { fmt::Display::fmt(&self.to_f32(), f) } }
impl fmt::Binary  for bf16 { fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "bf16({:01b} {:08b} {:07b})", self.0>>15, (self.0>>7)&0xFF, self.0&0x7F) } }

// ═════════════════════════════════════════════════════════════════════════════
// Tests
// ═════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test] fn size() { assert_eq!(core::mem::size_of::<bf16>(), 2); }

    #[test]
    fn to_f32_is_free() {
        // bf16(0x3F80) is 1.0 (same bit pattern as f32's exponent for 1.0)
        assert_eq!(bf16::ONE.to_f32(), 1.0f32);
        assert_eq!(bf16::ZERO.to_f32(), 0.0f32);
        assert_eq!(bf16::NEG_ONE.to_f32(), -1.0f32);
    }

    #[test]
    fn no_overflow_from_large_f32() {
        // Unlike f16, bf16 can represent any f32 magnitude.
        let big = f32::MAX;
        let h   = bf16::from_f32(big);
        assert!(h.is_finite(), "bf16 must NOT overflow for f32::MAX");
    }

    #[test]
    fn infinity_roundtrip() {
        assert!(bf16::from_f32(f32::INFINITY).is_infinite());
        assert!(bf16::from_f32(f32::NEG_INFINITY).is_infinite());
        assert!(bf16::INFINITY.to_f32().is_infinite());
    }

    #[test]
    fn nan_roundtrip() {
        let h = bf16::from_f32(f32::NAN);
        assert!(h.is_nan());
        assert!(h.to_f32().is_nan());
    }

    #[test]
    fn roundtrip_precision() {
        // bf16 has ~2 significant decimal digits of precision
        let cases = [0.0f32, 1.0, -1.0, 0.5, 3.14159, 1e10, -1e-10];
        for &v in &cases {
            let rt = bf16::from_f32(v).to_f32();
            if v == 0.0 {
                assert_eq!(rt, 0.0);
            } else {
                let rel = (rt - v).abs() / v.abs();
                assert!(rel < 0.02, "bf16 roundtrip: {v} → {rt}, rel_err = {rel:.4}");
            }
        }
    }

    #[test]
    fn neg_flip() {
        assert_eq!((-bf16::ONE).to_bits(), bf16::NEG_ONE.to_bits());
    }

    #[test]
    fn partial_eq_nan() {
        assert_ne!(bf16::NAN, bf16::NAN);
    }

    #[test]
    fn batch_x4() {
        let src = [1.0f32, -2.0, 0.5, 100.0];
        let rt  = bf16x4_to_f32x4(f32x4_to_bf16x4(src));
        for i in 0..4 {
            let rel = (rt[i] - src[i]).abs() / src[i].abs().max(f32::EPSILON);
            assert!(rel < 0.02, "bf16 x4 batch [{i}]: {s} → {r}", s=src[i], r=rt[i]);
        }
    }
    }
