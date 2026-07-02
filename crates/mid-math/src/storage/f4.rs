// crates/mid-math/src/storage/f4.rs
//! 4-bit float storage types — the most compressed useful floating-point formats.
//!
//! ## Why 4-bit floats
//!
//! F4 is the frontier of practical ML weight compression.  A fully-connected
//! layer with 1 billion parameters stored in f4 costs 500 MB instead of 4 GB.
//! Modern hardware (NVIDIA Blackwell B200, OCP MX spec) has native F4 support.
//!
//! ## Sub-byte storage
//!
//! F4 is 4 bits — half a byte.  You cannot store one value per byte efficiently.
//! Mid-Engine solves this with the **pair type**:
//!
//! * `F4E2M1Pair` — two `F4E2M1` values packed into one byte
//! * `F4E3M0Pair` — two `F4E3M0` values packed into one byte
//!
//! Layout: `[high nibble = first value][low nibble = second value]`
//!
//! ```text
//! byte:  7 6 5 4 | 3 2 1 0
//!        ───────   ───────
//!        val[0]    val[1]
//! ```
//!
//! ## Format comparison
//!
//! | Type      | Layout     | Values (positive)                  | Best for               |
//! |-----------|------------|------------------------------------|------------------------|
//! | `F4E2M1`  | 1s+2e+1m   | 0, 0.5, 1, 1.5, 2, 3, 4, 6        | ML weights (MXFP4)     |
//! | `F4E3M0`  | 1s+3e+0m   | 0, 0.25, 0.5, 1, 2, 4, 8, 16      | Ultra-low ML inference |
//!
//! ## API pattern
//!
//! ```no_run
//! # use mid_math::storage::{F4E2M1, F4E2M1Pair, f32x8_to_f4e2m1x4pairs, f4e2m1x4pairs_to_f32x8};
//! // Pack 8 f32 weights into 4 bytes (8 × F4E2M1, 2 per byte)
//! let weights: [f32; 8] = [0.5, -1.0, 1.5, 0.0, -3.0, 2.0, 0.5, -0.5];
//! let packed: [F4E2M1Pair; 4] = f32x8_to_f4e2m1x4pairs(weights);
//! // 4 bytes on wire / in VRAM instead of 32 bytes!
//!
//! // Unpack for computation
//! let unpacked: [f32; 8] = f4e2m1x4pairs_to_f32x8(packed);
//! ```
//!
//! ## Block quantization (future — mid-quant crate)
//!
//! Shared-scale block quantization (à la ggml's Q4_0) stores N F4 values
//! together with one f16 scale.  That logic belongs in `mid-quant`, which
//! imports these primitive types.  These types are the raw nibble layer only.

use core::fmt;

// ═════════════════════════════════════════════════════════════════════════════
//  F4E2M1 — MXFP4, OCP MX format
// ═════════════════════════════════════════════════════════════════════════════
//
// Bit layout (stored in low nibble of a u8): [s:1][e:2][m:1]
// Exponent bias = 1.
// No infinity.  No NaN.
//
// All 16 representable values (positive × sign):
//   0b0_00_0 = 0.0
//   0b0_00_1 = 0.5    (subnormal: 2^(1-1) × 0.5)
//   0b0_01_0 = 1.0    (normal:    2^(1-1) × 1.0)
//   0b0_01_1 = 1.5
//   0b0_10_0 = 2.0    (2^(2-1) × 1.0)
//   0b0_10_1 = 3.0    (2^(2-1) × 1.5)
//   0b0_11_0 = 4.0    (2^(3-1) × 1.0)
//   0b0_11_1 = 6.0    (2^(3-1) × 1.5)   ← max
//   1b_xxx   = negatives of above
//
// Reference: OCP MX Floating-Point Specification v1.0, Table 2.

/// 4-bit float: 1 sign, 2 exponent, 1 mantissa bit.
///
/// Stored in the **low nibble** of a `u8` (bits 3:0).  Bits 7:4 are always 0.
/// Use `F4E2M1Pair` to pack two values into a single byte for storage.
///
/// This is the MXFP4 format used by NVIDIA Blackwell and the OCP MX spec.
///
/// **Storage only.** Convert to `f32` for all arithmetic.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(transparent)]
pub struct F4E2M1(u8); // only bits [3:0] are meaningful

impl F4E2M1 {
    /// +0.0
    pub const ZERO:     Self = Self(0b0000);
    /// -0.0
    pub const NEG_ZERO: Self = Self(0b1000);
    /// +1.0
    pub const ONE:      Self = Self(0b0010);
    /// -1.0
    pub const NEG_ONE:  Self = Self(0b1010);
    /// +0.5
    pub const HALF:     Self = Self(0b0001);
    /// Maximum finite: +6.0
    pub const MAX:      Self = Self(0b0111);
    /// Minimum finite: -6.0
    pub const MIN:      Self = Self(0b1111);
    /// Smallest positive subnormal: 0.5
    pub const MIN_POSITIVE: Self = Self(0b0001);

    // Lookup table: positive values indexed by 3-bit mantissa+exponent field.
    const POS: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

    /// Construct from raw nibble (bits 3:0 used, bits 7:4 ignored).
    #[inline(always)]
    pub const fn from_bits(bits: u8) -> Self { Self(bits & 0x0F) }

    /// Raw nibble value (bits 3:0).
    #[inline(always)]
    pub const fn to_bits(self) -> u8 { self.0 & 0x0F }

    /// Convert `f32` → `F4E2M1` (lossy — rounds to nearest representable value).
    /// Values beyond ±6.0 clamp to ±6.0.
    #[inline]
    pub fn from_f32(value: f32) -> Self {
        if value.is_nan() { return Self::ZERO; } // no NaN in F4E2M1
        let sign: u8 = if value < 0.0 { 0x8 } else { 0x0 };
        let abs  = value.abs();

        // Clamp to max before searching
        let abs = abs.min(6.0);

        // Nearest-neighbour search through 8 positive values.
        // Midpoints between adjacent representable values:
        //   0↔0.5: 0.25  |  0.5↔1: 0.75  |  1↔1.5: 1.25  |  1.5↔2: 1.75
        //   2↔3:   2.5   |  3↔4:   3.5   |  4↔6:   5.0
        let idx: u8 = if abs < 0.25 { 0 }
            else if abs < 0.75 { 1 }
            else if abs < 1.25 { 2 }
            else if abs < 1.75 { 3 }
            else if abs < 2.5  { 4 }
            else if abs < 3.5  { 5 }
            else if abs < 5.0  { 6 }
            else               { 7 };

        Self(sign | idx)
    }

    /// Convert `F4E2M1` → `f32`. Uses a 16-entry lookup — one table load.
    #[inline]
    pub fn to_f32(self) -> f32 {
        let mag = Self::POS[(self.0 & 0x07) as usize];
        if self.0 & 0x08 != 0 && mag != 0.0 { -mag } else { mag }
    }

    /// Convert `f64` → `F4E2M1` (lossy).
    #[inline] pub fn from_f64(v: f64) -> Self { Self::from_f32(v as f32) }
    /// Convert `F4E2M1` → `f64`.
    #[inline] pub fn to_f64(self) -> f64 { self.to_f32() as f64 }

    #[inline] pub fn is_zero(self)          -> bool { self.0 & 0x07 == 0 }
    #[inline] pub fn is_sign_negative(self) -> bool { self.0 & 0x08 != 0 && !self.is_zero() }
    #[inline] pub fn is_sign_positive(self) -> bool { !self.is_sign_negative() }
    #[inline] pub fn abs(self)              -> Self { Self(self.0 & 0x07) }
    #[inline] pub fn neg(self)              -> Self { Self(self.0 ^ 0x08) }
}

impl From<f32>    for F4E2M1 { #[inline] fn from(v: f32)  -> Self { Self::from_f32(v) } }
impl From<F4E2M1> for f32    { #[inline] fn from(v: F4E2M1) -> f32 { v.to_f32() } }

impl fmt::Debug for F4E2M1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "F4E2M1({:04b} = {})", self.0 & 0x0F, self.to_f32())
    }
}
impl fmt::Display for F4E2M1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.to_f32(), f)
    }
}

// ═════════════════════════════════════════════════════════════════════════════
//  F4E3M0 — pure power-of-2 format
// ═════════════════════════════════════════════════════════════════════════════
//
// Bit layout (stored in low nibble): [s:1][e:3]
// Exponent bias = 3.
// No mantissa bits — every value is an exact power of 2 (or zero).
// No infinity.  No NaN.
//
// All 16 representable values:
//   0b0_000 = 0.0
//   0b0_001 = 0.25   (2^(1-3))
//   0b0_010 = 0.5    (2^(2-3))
//   0b0_011 = 1.0    (2^(3-3))
//   0b0_100 = 2.0    (2^(4-3))
//   0b0_101 = 4.0
//   0b0_110 = 8.0
//   0b0_111 = 16.0   ← max
//   1b_xxx  = negatives of above

/// 4-bit float: 1 sign, 3 exponent, 0 mantissa bits — exact powers of 2.
///
/// Every positive value is a power of 2 in `{0, 0.25, 0.5, 1, 2, 4, 8, 16}`.
/// This makes multiply/divide exact and hardware-friendly (just add/subtract
/// exponents with no mantissa alignment).
///
/// Useful for neural-network inference where weights are first quantised to
/// this format and then the compute engine uses integer shifters rather than
/// a full FPU.
///
/// **Storage only.** Convert to `f32` for arithmetic.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(transparent)]
pub struct F4E3M0(u8); // only bits [3:0] meaningful

impl F4E3M0 {
    /// +0.0
    pub const ZERO:     Self = Self(0b0000);
    /// -0.0
    pub const NEG_ZERO: Self = Self(0b1000);
    /// +1.0
    pub const ONE:      Self = Self(0b0011);
    /// -1.0
    pub const NEG_ONE:  Self = Self(0b1011);
    /// Maximum finite: +16.0
    pub const MAX:      Self = Self(0b0111);
    /// Minimum finite: -16.0
    pub const MIN:      Self = Self(0b1111);
    /// Smallest positive value: 0.25
    pub const MIN_POSITIVE: Self = Self(0b0001);

    const POS: [f32; 8] = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0];

    /// Construct from raw nibble (bits 3:0 used).
    #[inline(always)]
    pub const fn from_bits(bits: u8) -> Self { Self(bits & 0x0F) }

    /// Raw nibble value.
    #[inline(always)]
    pub const fn to_bits(self) -> u8 { self.0 & 0x0F }

    /// Convert `f32` → `F4E3M0` (lossy — rounds to nearest power-of-2).
    /// Values beyond ±16.0 clamp to ±16.0.
    #[inline]
    pub fn from_f32(value: f32) -> Self {
        if value.is_nan() { return Self::ZERO; }
        let sign: u8 = if value < 0.0 { 0x8 } else { 0x0 };
        let abs  = value.abs().min(16.0);

        // Midpoints between adjacent log-spaced values:
        //   0↔0.25: geometric midpt = sqrt(0×0.25) = 0 → use arithmetic: 0.125
        //   0.25↔0.5: 0.3536  |  0.5↔1: 0.7071  |  1↔2: 1.4142
        //   2↔4: 2.8284  |  4↔8: 5.6569  |  8↔16: 11.3137
        let idx: u8 = if abs < 0.125     { 0 }
            else if abs < 0.3536 { 1 }
            else if abs < 0.7071 { 2 }
            else if abs < 1.4142 { 3 }
            else if abs < 2.8284 { 4 }
            else if abs < 5.6569 { 5 }
            else if abs < 11.314 { 6 }
            else                 { 7 };

        Self(sign | idx)
    }

    /// Convert `F4E3M0` → `f32`. One table load.
    #[inline]
    pub fn to_f32(self) -> f32 {
        let mag = Self::POS[(self.0 & 0x07) as usize];
        if self.0 & 0x08 != 0 && mag != 0.0 { -mag } else { mag }
    }

    #[inline] pub fn from_f64(v: f64) -> Self { Self::from_f32(v as f32) }
    #[inline] pub fn to_f64(self)     -> f64  { self.to_f32() as f64 }

    #[inline] pub fn is_zero(self)          -> bool { self.0 & 0x07 == 0 }
    #[inline] pub fn is_sign_negative(self) -> bool { self.0 & 0x08 != 0 && !self.is_zero() }
    #[inline] pub fn is_sign_positive(self) -> bool { !self.is_sign_negative() }
    #[inline] pub fn abs(self)              -> Self { Self(self.0 & 0x07) }
    #[inline] pub fn neg(self)              -> Self { Self(self.0 ^ 0x08) }

    /// True for exact power-of-2 inputs.  All F4E3M0 normals are exact.
    #[inline] pub fn is_exact_f32(v: f32) -> bool {
        // v is representable iff it equals one of the 8 positive values
        let abs = v.abs();
        Self::POS.iter().any(|&p| p == abs)
    }
}

impl From<f32>    for F4E3M0 { #[inline] fn from(v: f32)  -> Self { Self::from_f32(v) } }
impl From<F4E3M0> for f32    { #[inline] fn from(v: F4E3M0) -> f32 { v.to_f32() } }

impl fmt::Debug for F4E3M0 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "F4E3M0({:04b} = {})", self.0 & 0x0F, self.to_f32())
    }
}
impl fmt::Display for F4E3M0 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.to_f32(), f)
    }
}

// ═════════════════════════════════════════════════════════════════════════════
//  Packed pair types — two F4 values in one byte
// ═════════════════════════════════════════════════════════════════════════════
//
// Layout: byte = [val_a (bits 7:4)] [val_b (bits 3:0)]
//
// This is the actual unit of storage.  An array of N F4 values is stored as
// N/2 pairs, costing N/2 bytes (vs N bytes for F8, or 4N bytes for f32).

/// Two `F4E2M1` values packed into one byte.
///
/// Bit layout: `[val_a: bits 7:4][val_b: bits 3:0]`
///
/// This is the natural storage unit — never store an unpacked `F4E2M1` array.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(transparent)]
pub struct F4E2M1Pair(pub u8);

impl F4E2M1Pair {
    /// Pack two `F4E2M1` values: `a` → high nibble, `b` → low nibble.
    #[inline]
    pub fn new(a: F4E2M1, b: F4E2M1) -> Self {
        Self((a.to_bits() << 4) | b.to_bits())
    }

    /// Raw byte.
    #[inline(always)] pub const fn to_bits(self) -> u8 { self.0 }
    #[inline(always)] pub const fn from_bits(b: u8) -> Self { Self(b) }

    /// Decode the first (high-nibble) value.
    #[inline] pub fn first(self)  -> F4E2M1 { F4E2M1::from_bits(self.0 >> 4) }
    /// Decode the second (low-nibble) value.
    #[inline] pub fn second(self) -> F4E2M1 { F4E2M1::from_bits(self.0 & 0x0F) }

    /// Unpack both values as `f32`.
    #[inline]
    pub fn to_f32x2(self) -> [f32; 2] {
        [self.first().to_f32(), self.second().to_f32()]
    }
}

impl fmt::Debug for F4E2M1Pair {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "F4E2M1Pair({:08b} = [{}, {}])",
            self.0, self.first().to_f32(), self.second().to_f32())
    }
}

/// Two `F4E3M0` values packed into one byte.
///
/// Bit layout: `[val_a: bits 7:4][val_b: bits 3:0]`
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(transparent)]
pub struct F4E3M0Pair(pub u8);

impl F4E3M0Pair {
    /// Pack two `F4E3M0` values: `a` → high nibble, `b` → low nibble.
    #[inline]
    pub fn new(a: F4E3M0, b: F4E3M0) -> Self {
        Self((a.to_bits() << 4) | b.to_bits())
    }

    #[inline(always)] pub const fn to_bits(self) -> u8 { self.0 }
    #[inline(always)] pub const fn from_bits(b: u8) -> Self { Self(b) }

    /// Decode the first (high-nibble) value.
    #[inline] pub fn first(self)  -> F4E3M0 { F4E3M0::from_bits(self.0 >> 4) }
    /// Decode the second (low-nibble) value.
    #[inline] pub fn second(self) -> F4E3M0 { F4E3M0::from_bits(self.0 & 0x0F) }

    /// Unpack both values as `f32`.
    #[inline]
    pub fn to_f32x2(self) -> [f32; 2] {
        [self.first().to_f32(), self.second().to_f32()]
    }
}

impl fmt::Debug for F4E3M0Pair {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "F4E3M0Pair({:08b} = [{}, {}])",
            self.0, self.first().to_f32(), self.second().to_f32())
    }
}

// ═════════════════════════════════════════════════════════════════════════════
//  Batch conversion — the primary high-performance interface
// ═════════════════════════════════════════════════════════════════════════════

/// Pack 8 × f32 → 4 × `F4E2M1Pair` (8 nibbles = 4 bytes).
/// This is the core operation for loading ML weights.
#[inline]
pub fn f32x8_to_f4e2m1x4pairs(v: [f32; 8]) -> [F4E2M1Pair; 4] {
    [
        F4E2M1Pair::new(F4E2M1::from_f32(v[0]), F4E2M1::from_f32(v[1])),
        F4E2M1Pair::new(F4E2M1::from_f32(v[2]), F4E2M1::from_f32(v[3])),
        F4E2M1Pair::new(F4E2M1::from_f32(v[4]), F4E2M1::from_f32(v[5])),
        F4E2M1Pair::new(F4E2M1::from_f32(v[6]), F4E2M1::from_f32(v[7])),
    ]
}

/// Unpack 4 × `F4E2M1Pair` → 8 × f32.
#[inline]
pub fn f4e2m1x4pairs_to_f32x8(v: [F4E2M1Pair; 4]) -> [f32; 8] {
    [
        v[0].first().to_f32(), v[0].second().to_f32(),
        v[1].first().to_f32(), v[1].second().to_f32(),
        v[2].first().to_f32(), v[2].second().to_f32(),
        v[3].first().to_f32(), v[3].second().to_f32(),
    ]
}

/// Pack 8 × f32 → 4 × `F4E3M0Pair`.
#[inline]
pub fn f32x8_to_f4e3m0x4pairs(v: [f32; 8]) -> [F4E3M0Pair; 4] {
    [
        F4E3M0Pair::new(F4E3M0::from_f32(v[0]), F4E3M0::from_f32(v[1])),
        F4E3M0Pair::new(F4E3M0::from_f32(v[2]), F4E3M0::from_f32(v[3])),
        F4E3M0Pair::new(F4E3M0::from_f32(v[4]), F4E3M0::from_f32(v[5])),
        F4E3M0Pair::new(F4E3M0::from_f32(v[6]), F4E3M0::from_f32(v[7])),
    ]
}

/// Unpack 4 × `F4E3M0Pair` → 8 × f32.
#[inline]
pub fn f4e3m0x4pairs_to_f32x8(v: [F4E3M0Pair; 4]) -> [f32; 8] {
    [
        v[0].first().to_f32(), v[0].second().to_f32(),
        v[1].first().to_f32(), v[1].second().to_f32(),
        v[2].first().to_f32(), v[2].second().to_f32(),
        v[3].first().to_f32(), v[3].second().to_f32(),
    ]
}

/// Pack a slice of f32 into a tightly-packed byte slice of F4E2M1 pairs.
///
/// `src.len()` must be even and `dst.len()` must equal `src.len() / 2`.
///
/// # Panics
/// If lengths are wrong.
pub fn f32_slice_to_f4e2m1_packed(src: &[f32], dst: &mut [u8]) {
    assert!(src.len() % 2 == 0, "f32_slice_to_f4e2m1_packed: src length must be even");
    assert_eq!(dst.len(), src.len() / 2, "f32_slice_to_f4e2m1_packed: dst must be src.len()/2");
    for (i, chunk) in src.chunks_exact(2).enumerate() {
        let a = F4E2M1::from_f32(chunk[0]);
        let b = F4E2M1::from_f32(chunk[1]);
        dst[i] = F4E2M1Pair::new(a, b).to_bits();
    }
}

/// Unpack a tightly-packed byte slice of F4E2M1 pairs into f32.
///
/// `dst.len()` must equal `src.len() * 2`.
///
/// # Panics
/// If lengths are wrong.
pub fn f4e2m1_packed_to_f32_slice(src: &[u8], dst: &mut [f32]) {
    assert_eq!(dst.len(), src.len() * 2, "f4e2m1_packed_to_f32_slice: dst must be src.len()*2");
    for (i, &byte) in src.iter().enumerate() {
        let pair = F4E2M1Pair::from_bits(byte);
        dst[i * 2]     = pair.first().to_f32();
        dst[i * 2 + 1] = pair.second().to_f32();
    }
}

// ═════════════════════════════════════════════════════════════════════════════
//  Tests
// ═════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── Size checks ───────────────────────────────────────────────────────────

    #[test]
    fn sizes() {
        assert_eq!(core::mem::size_of::<F4E2M1>(),     1); // stored in low nibble of u8
        assert_eq!(core::mem::size_of::<F4E3M0>(),     1);
        assert_eq!(core::mem::size_of::<F4E2M1Pair>(), 1); // two values in ONE byte
        assert_eq!(core::mem::size_of::<F4E3M0Pair>(), 1);
    }

    // ── F4E2M1 exact values ───────────────────────────────────────────────────

    #[test]
    fn e2m1_all_positive_values() {
        let expected = [0.0f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
        for (i, &exp) in expected.iter().enumerate() {
            let bits = i as u8;
            let val  = F4E2M1::from_bits(bits).to_f32();
            assert_eq!(val, exp, "F4E2M1 bits {bits:04b} should be {exp}");
        }
    }

    #[test]
    fn e2m1_all_negative_values() {
        let expected = [0.0f32, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0];
        for (i, &exp) in expected.iter().enumerate() {
            let bits = (0x8 | i) as u8;
            let val  = F4E2M1::from_bits(bits).to_f32();
            assert_eq!(val, exp, "F4E2M1 bits {bits:04b} should be {exp}");
        }
    }

    #[test]
    fn e2m1_exact_roundtrip() {
        // Values that ARE exactly representable should survive perfectly.
        let exact = [0.0f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                     -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0];
        for &v in &exact {
            let rt = F4E2M1::from_f32(v).to_f32();
            assert_eq!(rt, v, "F4E2M1 exact roundtrip failed for {v}");
        }
    }

    #[test]
    fn e2m1_rounding() {
        // 0.8 is between 0.5 and 1.0; midpoint is 0.75, and 0.8 > 0.75 → rounds to 1.0
        let h = F4E2M1::from_f32(0.8);
        assert_eq!(h.to_f32(), 1.0, "0.8 should round to 1.0");

        // 0.6 < 0.75 → rounds to 0.5
        let h = F4E2M1::from_f32(0.6);
        assert_eq!(h.to_f32(), 0.5, "0.6 should round to 0.5");
    }

    #[test]
    fn e2m1_overflow_clamps() {
        assert_eq!(F4E2M1::from_f32(100.0).to_f32(), 6.0);
        assert_eq!(F4E2M1::from_f32(-100.0).to_f32(), -6.0);
    }

    #[test]
    fn e2m1_zero() {
        assert!(F4E2M1::from_f32(0.0).is_zero());
        assert!(F4E2M1::from_f32(-0.0).is_zero());
    }

    #[test]
    fn e2m1_no_nan_output() {
        // F4E2M1 has no NaN — NaN input maps to 0
        assert_eq!(F4E2M1::from_f32(f32::NAN).to_f32(), 0.0);
    }

    #[test]
    fn e2m1_sign() {
        let h = F4E2M1::from_f32(-3.0);
        assert!(h.is_sign_negative());
        assert_eq!(h.abs().to_f32(), 3.0);
        assert_eq!(h.neg().to_f32(), 3.0);
    }

    // ── F4E3M0 exact values ───────────────────────────────────────────────────

    #[test]
    fn e3m0_all_positive_values() {
        let expected = [0.0f32, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0];
        for (i, &exp) in expected.iter().enumerate() {
            let val = F4E3M0::from_bits(i as u8).to_f32();
            assert_eq!(val, exp, "F4E3M0 bits {i:04b} should be {exp}");
        }
    }

    #[test]
    fn e3m0_exact_roundtrip() {
        let exact = [0.0f32, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0,
                     -0.25, -0.5, -1.0, -2.0, -4.0, -8.0, -16.0];
        for &v in &exact {
            let rt = F4E3M0::from_f32(v).to_f32();
            assert_eq!(rt, v, "F4E3M0 exact roundtrip failed for {v}");
        }
    }

    #[test]
    fn e3m0_rounding_geometric() {
        // 0.3 is between 0.25 and 0.5; geometric midpoint ≈ 0.354
        // 0.3 < 0.354 → rounds to 0.25
        assert_eq!(F4E3M0::from_f32(0.3).to_f32(), 0.25);
        // 0.4 > 0.354 → rounds to 0.5
        assert_eq!(F4E3M0::from_f32(0.4).to_f32(), 0.5);
    }

    #[test]
    fn e3m0_overflow_clamps() {
        assert_eq!(F4E3M0::from_f32(1000.0).to_f32(), 16.0);
        assert_eq!(F4E3M0::from_f32(-1000.0).to_f32(), -16.0);
    }

    // ── Pair packing ──────────────────────────────────────────────────────────

    #[test]
    fn e2m1_pair_packs_two_nibbles() {
        let a = F4E2M1::from_f32(1.5);  // bits = 0b0011 = 3
        let b = F4E2M1::from_f32(3.0);  // bits = 0b0101 = 5
        let pair = F4E2M1Pair::new(a, b);
        assert_eq!(pair.to_bits(), 0b0011_0101, "pair byte should be 0x35");
        assert_eq!(pair.first().to_f32(),  1.5);
        assert_eq!(pair.second().to_f32(), 3.0);
    }

    #[test]
    fn e2m1_pair_zero_pair_is_one_byte() {
        let pair = F4E2M1Pair::new(F4E2M1::ZERO, F4E2M1::ZERO);
        assert_eq!(pair.to_bits(), 0x00);
        assert_eq!(core::mem::size_of_val(&pair), 1);
    }

    #[test]
    fn e3m0_pair_roundtrip() {
        let a = F4E3M0::from_f32(2.0);
        let b = F4E3M0::from_f32(-1.0);
        let pair = F4E3M0Pair::new(a, b);
        assert_eq!(pair.first().to_f32(),  2.0);
        assert_eq!(pair.second().to_f32(), -1.0);
    }

    // ── Batch conversion ──────────────────────────────────────────────────────

    #[test]
    fn batch_8_to_4_pairs() {
        let weights = [0.5f32, -1.0, 1.5, 0.0, -3.0, 2.0, 1.0, -0.5];
        let packed  = f32x8_to_f4e2m1x4pairs(weights);
        let out     = f4e2m1x4pairs_to_f32x8(packed);

        assert_eq!(core::mem::size_of_val(&packed), 4,
            "8 F4E2M1 values must pack into exactly 4 bytes");

        for i in 0..8 {
            let expected = F4E2M1::from_f32(weights[i]).to_f32();
            assert_eq!(out[i], expected,
                "batch[{i}]: f32 {w} → f4 → f32 {r} (expected {e})",
                w=weights[i], r=out[i], e=expected);
        }
    }

    #[test]
    fn batch_e3m0_8_to_4_pairs() {
        let vals   = [1.0f32, -2.0, 0.5, 4.0, -0.25, 8.0, 0.0, -1.0];
        let packed = f32x8_to_f4e3m0x4pairs(vals);
        let out    = f4e3m0x4pairs_to_f32x8(packed);

        assert_eq!(core::mem::size_of_val(&packed), 4);
        for i in 0..8 {
            let expected = F4E3M0::from_f32(vals[i]).to_f32();
            assert_eq!(out[i], expected, "E3M0 batch[{i}] mismatch");
        }
    }

    #[test]
    fn slice_pack_unpack_roundtrip() {
        let src: [f32; 8] = [1.0, -1.0, 2.0, 0.5, 3.0, -3.0, 0.0, 6.0];
        let mut packed = [0u8; 4];
        let mut dst    = [0.0f32; 8];

        f32_slice_to_f4e2m1_packed(&src, &mut packed);
        f4e2m1_packed_to_f32_slice(&packed, &mut dst);

        for i in 0..8 {
            let expected = F4E2M1::from_f32(src[i]).to_f32();
            assert_eq!(dst[i], expected, "slice round-trip [{i}] failed");
        }
    }

    #[test]
    fn memory_density() {
        // Key guarantee: 8 F4 values fit in 4 bytes (8x denser than f32)
        let pairs: [F4E2M1Pair; 4] = f32x8_to_f4e2m1x4pairs([1.0; 8]);
        assert_eq!(core::mem::size_of_val(&pairs), 4);
        // vs 8 f32s = 32 bytes → 8x compression
    }
}
