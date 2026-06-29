// crates/mid-math/src/storage/f8.rs
//! 8-bit float storage types — three distinct formats for three distinct jobs.
//!
//! ## Format reference
//!
//! | Type       | Layout        | Range      | Inf  | NaN     | Best use                      |
//! |------------|---------------|------------|------|---------|-------------------------------|
//! | `F8Norm`   | u8 unorm      | [0.0, 1.0] | —    | —       | Colors, alpha, blend weights  |
//! | `F8E4M3`   | 1s + 4e + 3m  | ±448.0     | none | ±0x7F   | ML weights / activations      |
//! | `F8E5M2`   | 1s + 5e + 2m  | ±57344.0   | ±Inf | ±0x7D–7F| ML gradients (wider range)    |
//!
//! ## All three are storage-only
//! Pack to 8-bit for storage and transport. Unpack to `f32` for every
//! arithmetic operation.
//!
//! ## Future: f4
//! 4-bit formats (`F4E2M1`, `F4E3M0`) will live in `storage/f4.rs` following
//! the exact same pattern. Two nibbles packed per byte, explicit pack/unpack
//! helpers, and `f4x8` batch types. The `BitMask*` family works with f4
//! arrays unchanged — a mask is always just bits.

use core::fmt;

// ═════════════════════════════════════════════════════════════════════════════
//  F8Norm — unsigned normalised [0.0, 1.0]
// ═════════════════════════════════════════════════════════════════════════════

/// 8-bit unsigned normalised value — maps `[0, 255]` linearly to `[0.0, 1.0]`.
///
/// Matches the GPU `R8_UNORM` texture format exactly.
/// `size_of::<F8Norm>() == 1`.
///
/// **Storage only.** Convert to `f32` before arithmetic.
///
/// Arithmetic operators are deliberately not implemented. The correct
/// pattern is:
/// ```no_run
/// # use mid_math::storage::F8Norm;
/// # let a = F8Norm::ZERO; let b = F8Norm::ONE;
/// let result = F8Norm::from_f32((a.to_f32() + b.to_f32()).clamp(0.0, 1.0));
/// ```
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[repr(transparent)]
pub struct F8Norm(u8);

impl F8Norm {
    /// 0.0
    pub const ZERO: Self = Self(0);
    /// 1.0
    pub const ONE:  Self = Self(255);
    /// ≈ 0.502 (128/255)
    pub const HALF: Self = Self(128);

    /// Raw byte (0 = 0.0, 255 = 1.0).
    #[inline(always)] pub const fn from_bits(bits: u8) -> Self { Self(bits) }
    #[inline(always)] pub const fn to_bits(self) -> u8 { self.0 }

    /// `f32` → `F8Norm`.  Input clamped to `[0.0, 1.0]` before conversion.
    #[inline]
    pub fn from_f32(value: f32) -> Self {
        Self((value.clamp(0.0, 1.0) * 255.0 + 0.5) as u8)
    }

    /// `F8Norm` → `f32` in `[0.0, 1.0]`.
    #[inline]
    pub fn to_f32(self) -> f32 { self.0 as f32 / 255.0 }

    /// Linear interpolation: `a*(1-t) + b*t` where `t` is also `F8Norm`.
    #[inline]
    pub fn lerp(a: Self, b: Self, t: Self) -> Self {
        let tt = t.to_f32();
        Self::from_f32(a.to_f32() + (b.to_f32() - a.to_f32()) * tt)
    }
}

impl From<f32>    for F8Norm { #[inline] fn from(v: f32)  -> Self { Self::from_f32(v) } }
impl From<F8Norm> for f32    { #[inline] fn from(v: F8Norm) -> f32 { v.to_f32() } }
impl From<u8>     for F8Norm { #[inline] fn from(v: u8)   -> Self { Self(v) } }
impl From<F8Norm> for u8     { #[inline] fn from(v: F8Norm) -> u8 { v.0 } }

impl fmt::Debug for F8Norm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "F8Norm({} ≈ {:.4})", self.0, self.to_f32())
    }
}
impl fmt::Display for F8Norm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.4}", self.to_f32())
    }
}

// ═════════════════════════════════════════════════════════════════════════════
//  F8E4M3 — IEEE FP8 E4M3FN
// ═════════════════════════════════════════════════════════════════════════════
//
// Bit layout:  [s:1][e:4][m:3]
// Exponent bias = 7.
//
// Special values (E4M3FN — "Finite, No infinity"):
//   Zero:         S.0000.000
//   Subnormal:    S.0000.MMM  (M ≠ 000)  value = 2^(-6) * (M/8)
//   Normal:       S.EEEE.MMM  (E ∈ 1..15, except S.1111.111)
//   NaN:          S.1111.111  (only two NaN patterns: 0x7F, 0xFF)
//   No infinity!
//
// Max finite: 0x7E = 0 1111 110 = 2^(15-7) * (1+6/8) = 256 * 1.75 = 448.0
//
// NVIDIA H100 native format for weights/activations.

/// IEEE FP8 E4M3FN — 1 sign, 4 exponent, 3 mantissa bits.
///
/// Used to compress neural-network weights and activations for ML accelerators.
/// No infinities; only two NaN patterns (0x7F / 0xFF).
///
/// **Storage only.** `size_of::<F8E4M3>() == 1`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(transparent)]
pub struct F8E4M3(u8);

impl F8E4M3 {
    /// +0.0
    pub const ZERO:     Self = Self(0x00);
    /// -0.0
    pub const NEG_ZERO: Self = Self(0x80);
    /// +1.0  (e=7, m=0 → 2^(7-7)*(1+0) = 1.0)
    pub const ONE:      Self = Self(0x38);
    /// -1.0
    pub const NEG_ONE:  Self = Self(0xB8);
    /// Maximum finite: +448.0  (e=15, m=6)
    pub const MAX:      Self = Self(0x7E);
    /// Minimum finite: -448.0
    pub const MIN:      Self = Self(0xFE);
    /// Smallest positive normal: 2^(1-7) = 2^-6 ≈ 0.01563
    pub const MIN_POSITIVE: Self = Self(0x08);
    /// Smallest positive subnormal: 2^-9 ≈ 0.00195
    pub const MIN_POSITIVE_SUBNORMAL: Self = Self(0x01);
    /// Canonical NaN (positive)
    pub const NAN:      Self = Self(0x7F);

    #[inline(always)] pub const fn from_bits(bits: u8) -> Self { Self(bits) }
    #[inline(always)] pub const fn to_bits(self) -> u8 { self.0 }

    /// Convert `f32` → `F8E4M3` (lossy, rounds to nearest even).
    /// Overflow → ±448.0.  ±Inf → ±448.0.
    #[inline] pub fn from_f32(value: f32) -> Self { Self(e4m3::f32_to_e4m3(value)) }
    /// Convert `F8E4M3` → `f32`. No loss — f32 has far more precision.
    #[inline] pub fn to_f32(self) -> f32  { e4m3::e4m3_to_f32(self.0) }
    /// Convert `f64` → `F8E4M3` (lossy).
    #[inline] pub fn from_f64(value: f64) -> Self { Self::from_f32(value as f32) }
    /// Convert `F8E4M3` → `f64`.
    #[inline] pub fn to_f64(self) -> f64  { self.to_f32() as f64 }

    #[inline] pub fn is_nan(self)           -> bool { self.0 & 0x7F == 0x7F }
    #[inline] pub fn is_zero(self)          -> bool { self.0 & 0x7F == 0 }
    #[inline] pub fn is_finite(self)        -> bool { !self.is_nan() }
    #[inline] pub fn is_normal(self)        -> bool { let e=(self.0>>3)&0xF; e!=0 && !self.is_nan() }
    #[inline] pub fn is_subnormal(self)     -> bool { (self.0>>3)&0xF==0 && self.0&0x07!=0 }
    #[inline] pub fn is_sign_positive(self) -> bool { self.0 & 0x80 == 0 }
    #[inline] pub fn is_sign_negative(self) -> bool { self.0 & 0x80 != 0 }
    #[inline] pub fn abs(self)              -> Self { Self(self.0 & 0x7F) }
    #[inline] pub fn neg(self)              -> Self { Self(self.0 ^ 0x80) }
}

impl From<f32>    for F8E4M3 { #[inline] fn from(v: f32)  -> Self { Self::from_f32(v) } }
impl From<F8E4M3> for f32    { #[inline] fn from(v: F8E4M3) -> f32 { v.to_f32() } }
impl From<f64>    for F8E4M3 { #[inline] fn from(v: f64)  -> Self { Self::from_f64(v) } }
impl From<F8E4M3> for f64    { #[inline] fn from(v: F8E4M3) -> f64 { v.to_f64() } }

impl fmt::Debug for F8E4M3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_nan() { write!(f, "F8E4M3(NaN)") }
        else { write!(f, "F8E4M3({:?})", self.to_f32()) }
    }
}
impl fmt::Display for F8E4M3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_nan() { write!(f, "NaN") } else { fmt::Display::fmt(&self.to_f32(), f) }
    }
}
impl fmt::Binary for F8E4M3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:01b}_{:04b}_{:03b}", self.0>>7, (self.0>>3)&0xF, self.0&0x7)
    }
}

// ── E4M3 conversion ───────────────────────────────────────────────────────────
mod e4m3 {
    // ── f32 → E4M3FN ─────────────────────────────────────────────────────────
    //
    // Rounds to nearest even.
    //
    // Special handling required:
    //   - f32 NaN   → 0x7F (canonical E4M3 NaN)
    //   - f32 ±Inf  → ±0x7E (max finite ±448.0; E4M3 has no infinity)
    //   - Overflow  → ±0x7E (clamp to max finite)
    //   - Rounding up into 0x7F → clamp to 0x7E  (avoid creating NaN)
    pub fn f32_to_e4m3(value: f32) -> u8 {
        let x    = value.to_bits();
        let sign = ((x >> 31) & 1) as u8;  // 0 or 1
        let exp  = x & 0x7F80_0000u32;     // biased exponent field
        let man  = x & 0x007F_FFFFu32;     // 23-bit mantissa

        // ── Special cases ─────────────────────────────────────────────────────
        if exp == 0x7F80_0000 && man != 0 { return 0x7F; }          // NaN → NaN
        if exp == 0x7F80_0000             { return (sign<<7)|0x7E; } // ±Inf → max finite
        if exp == 0 && man == 0           { return sign << 7; }      // ±Zero
        // Subnormal f32 (exp==0, man≠0): value < 2^(-126), far below E4M3 range → flush
        if exp == 0                       { return sign << 7; }

        // ── Normal f32 ────────────────────────────────────────────────────────
        let unbiased: i32 = (exp >> 23) as i32 - 127;
        let fp8_exp:  i32 = unbiased + 7;  // rebias for E4M3 (bias = 7)

        // Overflow → max finite ±448.0
        if fp8_exp >= 16 { return (sign << 7) | 0x7E; }

        // Subnormal E4M3 (fp8_exp in {-3, -2, -1, 0})
        if fp8_exp <= 0 {
            if fp8_exp <= -4 { return sign << 7; } // too small → flush to zero
            // Include hidden '1' bit, then shift down to 3-bit mantissa.
            // Formula derivation:  E4M3 subnormal = 2^(-6) * (man3/8) = man3 * 2^(-9)
            // Highest bit of man_h at position (2 - leading). Shift man_h left by
            // (21 - fp8_exp) to put hidden-1 at bit 23, then mask gives 3-bit result.
            // Wait — for subnormal we actually want the raw 3-bit mantissa, not normalised.
            // shift = 21 - fp8_exp places the hidden-1 to yield the subnormal man3.
            let man_h = 0x0080_0000_u32 | man;
            let shift = (21 - fp8_exp) as u32;  // fp8_exp ≤ 0 → shift ≥ 21
            let half_m = man_h >> shift;
            // Round to nearest even
            let rb     = (man_h >> (shift - 1)) & 1;
            let sticky = man_h & ((1u32 << (shift - 1)).wrapping_sub(1));
            let rup    = rb != 0 && (sticky != 0 || (half_m & 1) != 0);
            return (sign << 7) | ((half_m + rup as u32) as u8 & 0x7F);
        }

        // ── Normal E4M3 ───────────────────────────────────────────────────────
        let fp8_e  = fp8_exp as u8;
        // Take the top 3 bits of the 23-bit mantissa
        let half_m = (man >> 20) as u8 & 0x07;
        let rb     = (man >> 19) & 1;
        let sticky = man & 0x0007_FFFF;
        let rup    = rb != 0 && (sticky != 0 || (half_m & 1) != 0);

        let packed = (sign << 7) | (fp8_e << 3) | half_m;
        let result = packed.wrapping_add(rup as u8);

        // If rounding produced 0x7F (NaN), clamp back to max finite 0x7E
        if result & 0x7F == 0x7F { (result & 0x80) | 0x7E } else { result }
    }

    // ── E4M3FN → f32 ─────────────────────────────────────────────────────────
    //
    // Conversion formulae for subnormals derived from:
    //   E4M3 subnormal value = man3 * 2^(-9)
    //
    // The highest set bit of man3 is at position k = 2 - leading_zeros_3bit.
    // Normalised:  (1.frac) * 2^(k - 9)
    //
    // f32 biased exponent: k - 9 + 127 = k + 118
    //   With k = 2 - leading: biased_exp = (2 - leading) + 118 = 120 - leading
    //
    // f32 mantissa: shift man3 left by (21 + leading) to put the hidden-1 at
    //   bit 23, then AND 0x7FFFFF removes it, leaving the fractional bits.
    pub fn e4m3_to_f32(bits: u8) -> f32 {
        let sign = (bits >> 7) as u32;
        let exp  = (bits >> 3) & 0x0F;   // 4-bit exponent field
        let man  = (bits & 0x07) as u32; // 3-bit mantissa

        // NaN
        if exp == 0x0F && man == 7 { return f32::NAN; }

        let f32_sign = sign << 31;

        // ±Zero
        if exp == 0 && man == 0 { return f32::from_bits(f32_sign); }

        // Subnormal E4M3 → normal f32
        if exp == 0 {
            // man is 3-bit (positions 2:0 in u32).
            // leading = zeros above the 3-bit field in a u32.
            let leading = man.leading_zeros() - 29; // 29 = 32 - 3
            let f32_biased_exp = 120u32.wrapping_sub(leading); // = k + 118 where k = 2 - leading
            let f32_exp = f32_biased_exp << 23;
            // Shift man left so hidden-1 lands at bit 23; &0x7FFFFF strips it.
            let f32_man = (man << (21 + leading)) & 0x007F_FFFFu32;
            return f32::from_bits(f32_sign | f32_exp | f32_man);
        }

        // Normal E4M3
        let unbiased = (exp as i32) - 7;
        let f32_exp  = ((unbiased + 127) as u32) << 23;
        // 3-bit mantissa → top of f32's 23-bit mantissa field (left-align to bit 22)
        let f32_man  = man << 20;
        f32::from_bits(f32_sign | f32_exp | f32_man)
    }
}

// ═════════════════════════════════════════════════════════════════════════════
//  F8E5M2 — IEEE FP8 E5M2
// ═════════════════════════════════════════════════════════════════════════════
//
// Bit layout:  [s:1][e:5][m:2]
// Exponent bias = 15.
//
// Special values (IEEE-like, has infinities):
//   Zero:         S.00000.00
//   Subnormal:    S.00000.MM  (M ≠ 00)  value = 2^(-14) * (M/4)
//   Normal:       S.EEEEE.MM  (E ∈ 1..30)
//   ±Infinity:    S.11111.00  (0x7C / 0xFC)
//   NaN:          S.11111.MM  (M ≠ 00, values 0x7D–0x7F / 0xFD–0xFF)
//
// Max finite: 0x7B = 0 11110 11 = 2^(30-15) * (1+3/4) = 32768 * 1.75 = 57344.0
//
// NVIDIA H100 native format for gradients (wider dynamic range than E4M3).

/// IEEE FP8 E5M2 — 1 sign, 5 exponent, 2 mantissa bits.
///
/// Wider dynamic range than `F8E4M3`. Used for gradient storage in mixed-
/// precision ML training.  Has ±infinity and NaN (IEEE-like).
///
/// **Storage only.** `size_of::<F8E5M2>() == 1`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(transparent)]
pub struct F8E5M2(u8);

impl F8E5M2 {
    /// +0.0
    pub const ZERO:         Self = Self(0x00);
    /// -0.0
    pub const NEG_ZERO:     Self = Self(0x80);
    /// +1.0  (e=15, m=0 → 2^(15-15)*(1+0) = 1.0)
    pub const ONE:          Self = Self(0x3C);
    /// -1.0
    pub const NEG_ONE:      Self = Self(0xBC);
    /// Maximum finite: +57344.0
    pub const MAX:          Self = Self(0x7B);
    /// Minimum finite: -57344.0
    pub const MIN:          Self = Self(0xFB);
    /// Smallest positive normal: 2^(1-15) = 2^-14 ≈ 6.1e-5
    pub const MIN_POSITIVE: Self = Self(0x04);
    /// Smallest positive subnormal: 2^-16 ≈ 1.5e-5
    pub const MIN_POSITIVE_SUBNORMAL: Self = Self(0x01);
    /// +∞
    pub const INFINITY:     Self = Self(0x7C);
    /// -∞
    pub const NEG_INFINITY: Self = Self(0xFC);
    /// Canonical NaN
    pub const NAN:          Self = Self(0x7F);

    #[inline(always)] pub const fn from_bits(bits: u8) -> Self { Self(bits) }
    #[inline(always)] pub const fn to_bits(self) -> u8 { self.0 }

    /// Convert `f32` → `F8E5M2` (lossy). Overflow → ±∞.
    #[inline] pub fn from_f32(value: f32) -> Self { Self(e5m2::f32_to_e5m2(value)) }
    /// Convert `F8E5M2` → `f32`.
    #[inline] pub fn to_f32(self) -> f32  { e5m2::e5m2_to_f32(self.0) }
    /// Convert `f64` → `F8E5M2` (lossy).
    #[inline] pub fn from_f64(value: f64) -> Self { Self::from_f32(value as f32) }
    /// Convert `F8E5M2` → `f64`.
    #[inline] pub fn to_f64(self) -> f64  { self.to_f32() as f64 }

    #[inline] pub fn is_nan(self)           -> bool { self.0 & 0x7F > 0x7C }
    #[inline] pub fn is_infinite(self)      -> bool { self.0 & 0x7F == 0x7C }
    #[inline] pub fn is_finite(self)        -> bool { self.0 & 0x7C != 0x7C }
    #[inline] pub fn is_zero(self)          -> bool { self.0 & 0x7F == 0 }
    #[inline] pub fn is_sign_positive(self) -> bool { self.0 & 0x80 == 0 }
    #[inline] pub fn is_sign_negative(self) -> bool { self.0 & 0x80 != 0 }
    #[inline] pub fn is_normal(self) -> bool {
        let e = (self.0 >> 2) & 0x1F;
        e != 0 && e != 0x1F
    }
    #[inline] pub fn is_subnormal(self) -> bool {
        (self.0 >> 2) & 0x1F == 0 && self.0 & 0x03 != 0
    }
    #[inline] pub fn abs(self) -> Self { Self(self.0 & 0x7F) }
    #[inline] pub fn neg(self) -> Self { Self(self.0 ^ 0x80) }
}

impl From<f32>    for F8E5M2 { #[inline] fn from(v: f32)  -> Self { Self::from_f32(v) } }
impl From<F8E5M2> for f32    { #[inline] fn from(v: F8E5M2) -> f32 { v.to_f32() } }
impl From<f64>    for F8E5M2 { #[inline] fn from(v: f64)  -> Self { Self::from_f64(v) } }
impl From<F8E5M2> for f64    { #[inline] fn from(v: F8E5M2) -> f64 { v.to_f64() } }

impl fmt::Debug for F8E5M2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_nan() {
            write!(f, "F8E5M2(NaN)")
        } else if self.is_infinite() {
            write!(f, "F8E5M2({}inf)", if self.is_sign_negative() { "-" } else { "+" })
        } else {
            write!(f, "F8E5M2({:?})", self.to_f32())
        }
    }
}
impl fmt::Display for F8E5M2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_nan() { write!(f, "NaN") }
        else if self.is_infinite() {
            write!(f, "{}inf", if self.is_sign_negative() { "-" } else { "" })
        } else {
            fmt::Display::fmt(&self.to_f32(), f)
        }
    }
}
impl fmt::Binary for F8E5M2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:01b}_{:05b}_{:02b}", self.0>>7, (self.0>>2)&0x1F, self.0&0x3)
    }
}

// ── E5M2 conversion ───────────────────────────────────────────────────────────
mod e5m2 {
    // ── f32 → E5M2 ───────────────────────────────────────────────────────────
    pub fn f32_to_e5m2(value: f32) -> u8 {
        let x    = value.to_bits();
        let sign = ((x >> 31) & 1) as u8;
        let exp  = x & 0x7F80_0000u32;
        let man  = x & 0x007F_FFFFu32;

        // ── Special cases ─────────────────────────────────────────────────────
        if exp == 0x7F80_0000 && man != 0 { return (sign<<7)|0x7F; } // NaN
        if exp == 0x7F80_0000             { return (sign<<7)|0x7C; } // ±Inf → ±Inf
        if exp == 0 && man == 0           { return sign << 7; }       // ±Zero
        if exp == 0                       { return sign << 7; }       // subnormal f32 → flush

        // ── Normal f32 ────────────────────────────────────────────────────────
        let unbiased: i32 = (exp >> 23) as i32 - 127;
        let fp8_exp:  i32 = unbiased + 15;  // rebias for E5M2 (bias = 15)

        // Overflow → ±Inf (E5M2 has infinity, so this is valid)
        if fp8_exp >= 32 { return (sign << 7) | 0x7C; }

        // Subnormal E5M2 (fp8_exp in {-2, -1, 0})
        if fp8_exp <= 0 {
            if fp8_exp <= -3 { return sign << 7; } // too small → flush to zero
            // E5M2 subnormal = 2^(-14) * (man2/4) = man2 * 2^(-16)
            // shift = 22 - fp8_exp; fp8_exp ∈ {-2,-1,0} → shift ∈ {24,23,22}
            let man_h = 0x0080_0000_u32 | man;
            let shift = (22 - fp8_exp) as u32;
            let half_m = man_h >> shift;
            let rb     = (man_h >> (shift - 1)) & 1;
            let sticky = man_h & ((1u32 << (shift - 1)).wrapping_sub(1));
            let rup    = rb != 0 && (sticky != 0 || (half_m & 1) != 0);
            return (sign << 7) | ((half_m + rup as u32) as u8 & 0x7F);
        }

        // ── Normal E5M2 ───────────────────────────────────────────────────────
        let fp8_e  = fp8_exp as u8;
        // Top 2 bits of the 23-bit mantissa
        let half_m = (man >> 21) as u8 & 0x03;
        let rb     = (man >> 20) & 1;
        let sticky = man & 0x000F_FFFF;
        let rup    = rb != 0 && (sticky != 0 || (half_m & 1) != 0);

        let packed = (sign << 7) | (fp8_e << 2) | half_m;
        // Adding 1 may roll into infinity (0x7C) — that is correct for E5M2.
        packed.wrapping_add(rup as u8)
    }

    // ── E5M2 → f32 ───────────────────────────────────────────────────────────
    //
    // Subnormal formula:
    //   E5M2 subnormal value = man2 * 2^(-16)
    //
    //   Highest set bit of man2 at position k = 1 - leading_zeros_2bit.
    //   f32 biased exponent = k + 111   (= k - 16 + 127)
    //   With k = 1 - leading: biased_exp = 112 - leading
    //
    //   f32 mantissa: shift man2 left by (22 + leading) so hidden-1 lands at
    //   bit 23, then AND 0x7FFFFF removes it.
    pub fn e5m2_to_f32(bits: u8) -> f32 {
        let sign = (bits >> 7) as u32;
        let exp  = (bits >> 2) & 0x1F;   // 5-bit exponent field
        let man  = (bits & 0x03) as u32; // 2-bit mantissa

        // NaN (E=11111, M≠00)
        if exp == 0x1F && man != 0 { return f32::NAN; }
        // ±Infinity
        if exp == 0x1F && man == 0 {
            return f32::from_bits((sign << 31) | 0x7F80_0000u32);
        }

        let f32_sign = sign << 31;

        // ±Zero
        if exp == 0 && man == 0 { return f32::from_bits(f32_sign); }

        // Subnormal E5M2 → normal f32
        if exp == 0 {
            // man is 2-bit (positions 1:0 in u32).
            let leading = man.leading_zeros() - 30; // 30 = 32 - 2
            let f32_biased_exp = 112u32.wrapping_sub(leading); // = k + 111 where k = 1 - leading
            let f32_exp = f32_biased_exp << 23;
            // Shift man left so hidden-1 lands at bit 23; &0x7FFFFF strips it.
            let f32_man = (man << (22 + leading)) & 0x007F_FFFFu32;
            return f32::from_bits(f32_sign | f32_exp | f32_man);
        }

        // Normal E5M2
        let unbiased = (exp as i32) - 15;
        let f32_exp  = ((unbiased + 127) as u32) << 23;
        // 2-bit mantissa → top of f32's 23-bit mantissa (left-align to bit 22)
        let f32_man  = man << 21;
        f32::from_bits(f32_sign | f32_exp | f32_man)
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Batch helpers
// ═════════════════════════════════════════════════════════════════════════════

/// Batch-convert 4 × f32 → 4 × `F8E4M3`.
#[inline] pub fn f32x4_to_f8e4m3x4(v: [f32; 4]) -> [F8E4M3; 4] { v.map(F8E4M3::from_f32) }
/// Batch-convert 4 × `F8E4M3` → 4 × f32.
#[inline] pub fn f8e4m3x4_to_f32x4(v: [F8E4M3; 4]) -> [f32; 4] { v.map(F8E4M3::to_f32) }
/// Batch-convert 4 × f32 → 4 × `F8E5M2`.
#[inline] pub fn f32x4_to_f8e5m2x4(v: [f32; 4]) -> [F8E5M2; 4] { v.map(F8E5M2::from_f32) }
/// Batch-convert 4 × `F8E5M2` → 4 × f32.
#[inline] pub fn f8e5m2x4_to_f32x4(v: [F8E5M2; 4]) -> [f32; 4] { v.map(F8E5M2::to_f32) }

// ═════════════════════════════════════════════════════════════════════════════
// Tests
// ═════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── F8Norm ────────────────────────────────────────────────────────────────

    #[test] fn f8norm_size()       { assert_eq!(core::mem::size_of::<F8Norm>(), 1); }
    #[test] fn f8norm_zero_one()   { assert_eq!(F8Norm::ZERO.to_f32(), 0.0); assert_eq!(F8Norm::ONE.to_f32(), 1.0); }
    #[test] fn f8norm_clamp()      { assert_eq!(F8Norm::from_f32(-1.0).to_bits(), 0); assert_eq!(F8Norm::from_f32(2.0).to_bits(), 255); }

    #[test]
    fn f8norm_roundtrip_all() {
        for i in 0u8..=255 {
            let v  = F8Norm::from_bits(i);
            let rt = F8Norm::from_f32(v.to_f32());
            assert!((rt.to_bits() as i16 - i as i16).abs() <= 1,
                "F8Norm roundtrip failed at byte {i}");
        }
    }

    // ── F8E4M3 ───────────────────────────────────────────────────────────────

    #[test] fn e4m3_size()    { assert_eq!(core::mem::size_of::<F8E4M3>(), 1); }
    #[test] fn e4m3_nan()     { assert!(F8E4M3::from_f32(f32::NAN).is_nan()); }
    #[test] fn e4m3_zero()    { assert!(F8E4M3::from_f32(0.0).is_zero()); }
    #[test] fn e4m3_negative(){ let h = F8E4M3::from_f32(-1.0); assert!(h.is_sign_negative()); assert!((h.to_f32()+1.0).abs() < 0.03); }

    #[test]
    fn e4m3_no_infinity() {
        // E4M3FN has no infinity — both ±Inf clamp to max finite
        let pi = F8E4M3::from_f32(f32::INFINITY);
        let ni = F8E4M3::from_f32(f32::NEG_INFINITY);
        assert!(!pi.is_nan() && pi.to_bits() & 0x7F == 0x7E, "pos inf should clamp to 0x7E");
        assert!(!ni.is_nan() && ni.to_bits() & 0x7F == 0x7E, "neg inf should clamp to 0x7E");
    }

    #[test]
    fn e4m3_overflow_clamps() {
        let h = F8E4M3::from_f32(1000.0);
        assert_eq!(h.to_bits() & 0x7F, 0x7E, "overflow should clamp to max finite 0x7E");
    }

    #[test]
    fn e4m3_one_roundtrip() {
        let h = F8E4M3::from_f32(1.0);
        assert_eq!(h.to_bits(), F8E4M3::ONE.to_bits());
        assert!((h.to_f32() - 1.0).abs() < 0.01);
    }

    #[test]
    fn e4m3_representative_values() {
        // E4M3 has 3 mantissa bits → up to 12.5% relative error per ULP
        let cases = [0.5f32, -0.5, 2.0, -2.0, 16.0, 128.0];
        for &v in &cases {
            let rt = F8E4M3::from_f32(v).to_f32();
            assert!((rt - v).abs() / v.abs() < 0.15,
                "E4M3 {v} → {rt}: relative error too large");
        }
    }

    #[test]
    fn e4m3_subnormal_roundtrip() {
        // Smallest subnormal: 2^(-9) ≈ 0.00195
        let min_sub = F8E4M3::MIN_POSITIVE_SUBNORMAL;
        assert!(min_sub.is_subnormal());
        let v = min_sub.to_f32();
        // Should be close to 2^(-9)
        let expected = 2.0f32.powi(-9);
        assert!((v - expected).abs() / expected < 0.01,
            "Smallest E4M3 subnormal: got {v}, expected {expected}");
    }

    #[test]
    fn e4m3_no_rounding_into_nan() {
        // Values close to 448.0 must not produce NaN (0x7F) after rounding
        let near_max = F8E4M3::from_f32(447.9);
        assert!(!near_max.is_nan(), "Rounding near max must not produce NaN");
        let at_max = F8E4M3::from_f32(448.0);
        assert!(!at_max.is_nan());
        let over_max = F8E4M3::from_f32(449.0);
        assert!(!over_max.is_nan(), "Values just above max must clamp, not NaN");
    }

    // ── F8E5M2 ───────────────────────────────────────────────────────────────

    #[test] fn e5m2_size()   { assert_eq!(core::mem::size_of::<F8E5M2>(), 1); }
    #[test] fn e5m2_nan()    { assert!(F8E5M2::from_f32(f32::NAN).is_nan()); }
    #[test] fn e5m2_zero()   { assert!(F8E5M2::from_f32(0.0).is_zero()); }

    #[test]
    fn e5m2_infinity_roundtrip() {
        let pi = F8E5M2::from_f32(f32::INFINITY);
        let ni = F8E5M2::from_f32(f32::NEG_INFINITY);
        assert!(pi.is_infinite() && pi.is_sign_positive());
        assert!(ni.is_infinite() && ni.is_sign_negative());
        assert!(pi.to_f32().is_infinite());
        assert!(ni.to_f32().is_infinite());
    }

    #[test]
    fn e5m2_overflow_to_inf() {
        assert!(F8E5M2::from_f32(1e8).is_infinite());
    }

    #[test]
    fn e5m2_one_roundtrip() {
        let h = F8E5M2::from_f32(1.0);
        assert_eq!(h.to_bits(), F8E5M2::ONE.to_bits());
        assert!((h.to_f32() - 1.0).abs() < 0.01);
    }

    #[test]
    fn e5m2_wider_range_than_e4m3() {
        let big = 10_000.0f32;
        let e4  = F8E4M3::from_f32(big);
        let e5  = F8E5M2::from_f32(big);
        assert_eq!(e4.to_bits() & 0x7F, 0x7E, "E4M3 should clamp 10000 to max");
        assert!(!e5.is_infinite(),             "E5M2 should represent 10000 without overflow");
    }

    #[test]
    fn e5m2_subnormal_roundtrip() {
        // Smallest E5M2 subnormal: 2^(-16) ≈ 1.526e-5
        let min_sub = F8E5M2::MIN_POSITIVE_SUBNORMAL;
        assert!(min_sub.is_subnormal());
        let v = min_sub.to_f32();
        let expected = 2.0f32.powi(-16);
        assert!((v - expected).abs() / expected < 0.01,
            "Smallest E5M2 subnormal: got {v}, expected {expected}");
    }

    #[test]
    fn batch_f32x4_e4m3() {
        let src = [0.5f32, -0.5, 1.0, 0.0];
        let packed = f32x4_to_f8e4m3x4(src);
        let out    = f8e4m3x4_to_f32x4(packed);
        for i in 0..4 {
            assert!((out[i] - src[i]).abs() <= src[i].abs() * 0.15 + 0.01,
                "E4M3 batch[{i}]: {src} → {out}", src=src[i], out=out[i]);
        }
    }

    #[test]
    fn batch_f32x4_e5m2() {
        let src = [1.0f32, -1.0, 100.0, -100.0];
        let packed = f32x4_to_f8e5m2x4(src);
        let out    = f8e5m2x4_to_f32x4(packed);
        for i in 0..4 {
            assert!((out[i] - src[i]).abs() <= src[i].abs() * 0.30 + 0.01,
                "E5M2 batch[{i}]: {src} → {out}", src=src[i], out=out[i]);
        }
    }
         }
