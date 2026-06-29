// crates/mid-math/src/storage/f16.rs
//! IEEE 754-2008 binary16 ("half precision") — a **storage-only** boundary type.
//!
//! ## The golden rule
//! ```text
//! f16 → store it, upload it to the GPU, pack it into a network packet
//! f32 → do all arithmetic
//! ```
//!
//! ## Storage guarantee
//! `size_of::<f16>() == 2` — always exactly 2 bytes regardless of what
//! conversion functions are called.  The `f32` that appears in
//! `from_f32` / `to_f32` is a **transient register value**, never stored.
//!
//! ## Platform dispatch (compile-time, zero runtime overhead)
//! | Target                              | Backend                              |
//! |-------------------------------------|--------------------------------------|
//! | x86/x86_64 + `target_feature=f16c` | `VCVTPS2PH` / `VCVTPH2PS` (SSE/AVX) |
//! | aarch64 + `target_feature=fp16`    | `FCVT` scalar, compiler vectorises   |
//! | everything else                    | Pure-Rust IEEE 754 bit manipulation  |
//!
//! ## Game-engine use cases
//! * **GPU uploads** — normals, tangents, G-buffer channels, HDR colours.
//!   Packing to f16 doubles effective VRAM bandwidth.
//! * **Animation clips** — 250-bone skeletons at 60 fps in f32 burns RAM.
//!   f16 bone transforms cut that budget in half with near-zero visual loss.
//! * **Network sync** — pack world-positions into f16 to stay under the UDP
//!   MTU without hand-rolling custom bit fields.
//! * **ML inference pipeline** — the conversion bridge between `F8E4M3` /
//!   `F8E5M2` storage and the f32 compute path.

use core::{
    fmt,
    hash::{Hash, Hasher},
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

// ═════════════════════════════════════════════════════════════════════════════
// Type definition
// ═════════════════════════════════════════════════════════════════════════════

/// IEEE 754-2008 binary16 half-precision float.
///
/// **Storage only — call `.to_f32()` before any arithmetic.**
/// `size_of::<f16>() == 2` always.
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Default)]
#[repr(transparent)]
pub struct f16(u16);

// ═════════════════════════════════════════════════════════════════════════════
// Constants
// ═════════════════════════════════════════════════════════════════════════════

impl f16 {
    /// +0.0
    pub const ZERO:               Self = Self(0x0000);
    /// -0.0
    pub const NEG_ZERO:           Self = Self(0x8000);
    /// 1.0
    pub const ONE:                Self = Self(0x3C00);
    /// -1.0
    pub const NEG_ONE:            Self = Self(0xBC00);
    /// +∞
    pub const INFINITY:           Self = Self(0x7C00);
    /// -∞
    pub const NEG_INFINITY:       Self = Self(0xFC00);
    /// Canonical quiet NaN
    pub const NAN:                Self = Self(0x7E00);
    /// Maximum finite value ≈ 65504.0
    pub const MAX:                Self = Self(0x7BFF);
    /// Minimum finite value ≈ -65504.0
    pub const MIN:                Self = Self(0xFBFF);
    /// Smallest positive normal ≈ 6.104e-5  (2^−14)
    pub const MIN_POSITIVE:       Self = Self(0x0400);
    /// Smallest positive subnormal ≈ 5.96e-8  (2^−24)
    pub const MIN_POSITIVE_SUBNORMAL: Self = Self(0x0001);
    /// Machine epsilon = 2^−10 ≈ 9.766e-4
    pub const EPSILON:            Self = Self(0x1400);
    /// π ≈ 3.140625
    pub const PI:                 Self = Self(0x4248);
    /// τ = 2π ≈ 6.28125
    pub const TAU:                Self = Self(0x4A48);
    /// e ≈ 2.71875
    pub const E:                  Self = Self(0x4170);
    /// √2 ≈ 1.41406
    pub const SQRT_2:             Self = Self(0x3DA8);
    /// ln(2) ≈ 0.6931
    pub const LN_2:               Self = Self(0x398C);
}

// ═════════════════════════════════════════════════════════════════════════════
// Construction & extraction
// ═════════════════════════════════════════════════════════════════════════════

impl f16 {
    /// Construct from raw IEEE 754 binary16 bits.
    #[inline(always)]
    pub const fn from_bits(bits: u16) -> Self { Self(bits) }

    /// Return raw IEEE 754 binary16 bits.
    #[inline(always)]
    pub const fn to_bits(self) -> u16 { self.0 }

    /// Convert `f32` → `f16` (lossy — rounds to nearest even).
    ///
    /// * Overflow → ±∞
    /// * Very small f32 subnormals may underflow → ±0
    #[inline]
    pub fn from_f32(value: f32) -> Self { Self(dispatch::f32_to_f16(value)) }

    /// Convert `f16` → `f32`. **Lossless** — every f16 is exactly representable in f32.
    #[inline]
    pub fn to_f32(self) -> f32 { dispatch::f16_to_f32(self.0) }

    /// Convert `f64` → `f16` (lossy).
    #[inline]
    pub fn from_f64(value: f64) -> Self { Self(dispatch::f64_to_f16(value)) }

    /// Convert `f16` → `f64`.
    #[inline]
    pub fn to_f64(self) -> f64 { dispatch::f16_to_f32(self.0) as f64 }
}

// ═════════════════════════════════════════════════════════════════════════════
// Classification
// ═════════════════════════════════════════════════════════════════════════════

impl f16 {
    /// True if this is any NaN value.
    #[inline] pub fn is_nan(self)              -> bool { self.0 & 0x7FFF >  0x7C00 }
    /// True if ±∞.
    #[inline] pub fn is_infinite(self)         -> bool { self.0 & 0x7FFF == 0x7C00 }
    /// True if neither NaN nor infinite.
    #[inline] pub fn is_finite(self)           -> bool { self.0 & 0x7C00 != 0x7C00 }
    /// True if subnormal (tiny non-zero, exponent bits all zero).
    #[inline] pub fn is_subnormal(self)        -> bool { self.0 & 0x7C00 == 0 && self.0 & 0x03FF != 0 }
    /// True if normal (not zero, subnormal, infinite, or NaN).
    #[inline] pub fn is_normal(self)           -> bool { let e = self.0 & 0x7C00; e != 0 && e != 0x7C00 }
    /// True if ±0.
    #[inline] pub fn is_zero(self)             -> bool { self.0 & 0x7FFF == 0 }
    /// True if sign bit is clear (positive, +0, or positive NaN).
    #[inline] pub fn is_sign_positive(self)    -> bool { self.0 & 0x8000 == 0 }
    /// True if sign bit is set.
    #[inline] pub fn is_sign_negative(self)    -> bool { self.0 & 0x8000 != 0 }
    /// Absolute value — clears sign bit, no conversion required.
    #[inline] pub fn abs(self)                 -> Self { Self(self.0 & 0x7FFF) }
    /// Copy sign from `sign_src` into `self`'s magnitude.
    #[inline] pub fn copysign(self, sign_src: Self) -> Self {
        Self((self.0 & 0x7FFF) | (sign_src.0 & 0x8000))
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Batch conversion helpers
// ═════════════════════════════════════════════════════════════════════════════

/// Convert 4 × f32 → 4 × f16.
/// On x86 with F16C this issues a single `VCVTPS2PH` instruction.
#[inline]
pub fn f32x4_to_f16x4(v: [f32; 4]) -> [f16; 4] {
    let b = dispatch::f32x4_to_f16x4(v);
    [f16(b[0]), f16(b[1]), f16(b[2]), f16(b[3])]
}

/// Convert 4 × f16 → 4 × f32.
#[inline]
pub fn f16x4_to_f32x4(v: [f16; 4]) -> [f32; 4] {
    dispatch::f16x4_to_f32x4([v[0].0, v[1].0, v[2].0, v[3].0])
}

/// Convert 8 × f32 → 8 × f16.
#[inline]
pub fn f32x8_to_f16x8(v: [f32; 8]) -> [f16; 8] {
    let b = dispatch::f32x8_to_f16x8(v);
    [
        f16(b[0]), f16(b[1]), f16(b[2]), f16(b[3]),
        f16(b[4]), f16(b[5]), f16(b[6]), f16(b[7]),
    ]
}

/// Convert 8 × f16 → 8 × f32.
#[inline]
pub fn f16x8_to_f32x8(v: [f16; 8]) -> [f32; 8] {
    dispatch::f16x8_to_f32x8([
        v[0].0, v[1].0, v[2].0, v[3].0,
        v[4].0, v[5].0, v[6].0, v[7].0,
    ])
}

/// Batch-convert a slice of `f32` into a pre-allocated slice of `f16`.
/// Processes in chunks of 8 to maximise SIMD utilisation.
///
/// # Panics
/// If `src.len() != dst.len()`.
pub fn f32_slice_to_f16(src: &[f32], dst: &mut [f16]) {
    assert_eq!(src.len(), dst.len(), "f32_slice_to_f16: length mismatch");
    let mut i = 0;
    while i + 8 <= src.len() {
        let chunk: [f32; 8] = src[i..i + 8].try_into().unwrap();
        let out = f32x8_to_f16x8(chunk);
        dst[i..i + 8].copy_from_slice(&out);
        i += 8;
    }
    while i < src.len() {
        dst[i] = f16::from_f32(src[i]);
        i += 1;
    }
}

/// Batch-convert a slice of `f16` into a pre-allocated slice of `f32`.
///
/// # Panics
/// If `src.len() != dst.len()`.
pub fn f16_slice_to_f32(src: &[f16], dst: &mut [f32]) {
    assert_eq!(src.len(), dst.len(), "f16_slice_to_f32: length mismatch");
    let mut i = 0;
    while i + 8 <= src.len() {
        let chunk: [f16; 8] = src[i..i + 8].try_into().unwrap();
        let out = f16x8_to_f32x8(chunk);
        dst[i..i + 8].copy_from_slice(&out);
        i += 8;
    }
    while i < src.len() {
        dst[i] = src[i].to_f32();
        i += 1;
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Arithmetic (f32 round-trip — f16 is storage only)
// ═════════════════════════════════════════════════════════════════════════════

impl Neg for f16 {
    type Output = Self;
    /// Flip sign bit — no conversion needed.
    #[inline] fn neg(self) -> Self { Self(self.0 ^ 0x8000) }
}

macro_rules! impl_f16_arith {
    ($Trait:ident, $fn:ident, $Assign:ident, $afn:ident, $op:tt) => {
        impl $Trait for f16 {
            type Output = Self;
            #[inline]
            fn $fn(self, rhs: Self) -> Self {
                Self::from_f32(self.to_f32() $op rhs.to_f32())
            }
        }
        impl $Assign for f16 {
            #[inline] fn $afn(&mut self, rhs: Self) { *self = *self $op rhs; }
        }
        /// Compute in f32 directly (avoids a conversion back to f16).
        impl $Trait<f32> for f16 {
            type Output = f32;
            #[inline] fn $fn(self, rhs: f32) -> f32 { self.to_f32() $op rhs }
        }
    };
}
impl_f16_arith!(Add, add, AddAssign, add_assign, +);
impl_f16_arith!(Sub, sub, SubAssign, sub_assign, -);
impl_f16_arith!(Mul, mul, MulAssign, mul_assign, *);
impl_f16_arith!(Div, div, DivAssign, div_assign, /);

// ═════════════════════════════════════════════════════════════════════════════
// From / Into
// ═════════════════════════════════════════════════════════════════════════════

impl From<f32>  for f16 { #[inline] fn from(v: f32) -> Self { Self::from_f32(v) } }
impl From<f64>  for f16 { #[inline] fn from(v: f64) -> Self { Self::from_f64(v) } }
impl From<f16>  for f32 { #[inline] fn from(v: f16) -> f32  { v.to_f32() } }
impl From<f16>  for f64 { #[inline] fn from(v: f16) -> f64  { v.to_f64() } }

// ═════════════════════════════════════════════════════════════════════════════
// Comparison & hashing
// ═════════════════════════════════════════════════════════════════════════════

impl PartialEq for f16 {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        // NaN ≠ NaN; +0 == -0
        if self.is_nan() || other.is_nan() { return false; }
        (self.0 == other.0) || ((self.0 | other.0) & 0x7FFF == 0)
    }
}

impl PartialOrd for f16 {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        if self.is_nan() || other.is_nan() { return None; }
        self.to_f32().partial_cmp(&other.to_f32())
    }
}

impl Hash for f16 {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Canonicalise -0 → +0 for hashing
        let bits = if self.0 & 0x7FFF == 0 { 0u16 } else { self.0 };
        bits.hash(state);
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Formatting
// ═════════════════════════════════════════════════════════════════════════════

impl fmt::Debug for f16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "f16({:?})", self.to_f32())
    }
}
impl fmt::Display for f16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.to_f32(), f)
    }
}
impl fmt::LowerHex for f16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "f16(0x{:04x})", self.0)
    }
}
/// Displays individual field bits: `f16(s eeee mmmmmmmmmm)`.
impl fmt::Binary for f16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "f16({:01b} {:05b} {:010b})",
            self.0 >> 15, (self.0 >> 10) & 0x1F, self.0 & 0x3FF)
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Platform dispatch
// ═════════════════════════════════════════════════════════════════════════════

mod dispatch {

    // ─────────────────────────────────────────────────────────────────────────
    // x86 / x86_64 with F16C
    // ─────────────────────────────────────────────────────────────────────────
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "f16c"))]
    mod hw {
        #[cfg(target_arch = "x86")]
        use core::arch::x86::{__m128, __m128i, _mm_cvtph_ps, _mm_cvtps_ph, _MM_FROUND_TO_NEAREST_INT};
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::{__m128, __m128i, _mm_cvtph_ps, _mm_cvtps_ph, _MM_FROUND_TO_NEAREST_INT};

        /// f32 → f16 via F16C `VCVTPS2PH`.
        #[target_feature(enable = "f16c")]
        #[inline]
        pub unsafe fn f32_to_f16(v: f32) -> u16 {
            // Broadcast to 128-bit; result packed into low u16 of returned __m128i.
            let vec: __m128 = core::mem::transmute([v, 0.0f32, 0.0f32, 0.0f32]);
            let out: __m128i = _mm_cvtps_ph(vec, _MM_FROUND_TO_NEAREST_INT);
            let out: [u16; 8] = core::mem::transmute(out);
            out[0]
        }

        /// f16 → f32 via F16C `VCVTPH2PS`.
        #[target_feature(enable = "f16c")]
        #[inline]
        pub unsafe fn f16_to_f32(bits: u16) -> f32 {
            let vec: __m128i = core::mem::transmute(
                [bits, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16]
            );
            let out: __m128 = _mm_cvtph_ps(vec);
            let out: [f32; 4] = core::mem::transmute(out);
            out[0]
        }

        /// 4 × f32 → 4 × f16 in one `VCVTPS2PH`.
        #[target_feature(enable = "f16c")]
        #[inline]
        pub unsafe fn f32x4_to_f16x4(v: [f32; 4]) -> [u16; 4] {
            let vec: __m128 = core::mem::transmute(v);
            let out: __m128i = _mm_cvtps_ph(vec, _MM_FROUND_TO_NEAREST_INT);
            let out: [u16; 8] = core::mem::transmute(out);
            [out[0], out[1], out[2], out[3]]
        }

        /// 4 × f16 → 4 × f32 in one `VCVTPH2PS`.
        #[target_feature(enable = "f16c")]
        #[inline]
        pub unsafe fn f16x4_to_f32x4(bits: [u16; 4]) -> [f32; 4] {
            let vec: __m128i = core::mem::transmute(
                [bits[0], bits[1], bits[2], bits[3], 0u16, 0u16, 0u16, 0u16]
            );
            let out: __m128 = _mm_cvtph_ps(vec);
            core::mem::transmute(out)
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // aarch64 with fp16 extension
    // ─────────────────────────────────────────────────────────────────────────
    // We use scalar FCVT instructions here and rely on LLVM to auto-vectorise
    // the batch functions into FCVTN/FCVTL when fp16 is enabled.
    // Attempting to use inline asm on mixed-width NEON types (float32x4_t vs
    // uint16x4_t) is fragile due to 128-bit vs 64-bit register constraint
    // differences — scalar+autovec is safer and equally fast in practice.
    #[cfg(all(target_arch = "aarch64", target_feature = "fp16"))]
    mod hw {
        /// f32 → f16 via scalar FCVT.
        #[target_feature(enable = "fp16")]
        #[inline]
        pub unsafe fn f32_to_f16(v: f32) -> u16 {
            let result: u16;
            core::arch::asm!(
                "fcvt {0:h}, {1:s}",
                out(vreg) result,
                in(vreg)  v,
                options(pure, nomem, nostack, preserves_flags)
            );
            result
        }

        /// f16 → f32 via scalar FCVT.
        #[target_feature(enable = "fp16")]
        #[inline]
        pub unsafe fn f16_to_f32(bits: u16) -> f32 {
            let result: f32;
            core::arch::asm!(
                "fcvt {0:s}, {1:h}",
                out(vreg) result,
                in(vreg)  bits,
                options(pure, nomem, nostack, preserves_flags)
            );
            result
        }

        /// 4 × f32 → 4 × f16 — scalar FCVT × 4, LLVM will vectorise to FCVTN.
        #[target_feature(enable = "fp16")]
        #[inline]
        pub unsafe fn f32x4_to_f16x4(v: [f32; 4]) -> [u16; 4] {
            [f32_to_f16(v[0]), f32_to_f16(v[1]), f32_to_f16(v[2]), f32_to_f16(v[3])]
        }

        /// 4 × f16 → 4 × f32 — scalar FCVT × 4, LLVM will vectorise to FCVTL.
        #[target_feature(enable = "fp16")]
        #[inline]
        pub unsafe fn f16x4_to_f32x4(bits: [u16; 4]) -> [f32; 4] {
            [f16_to_f32(bits[0]), f16_to_f32(bits[1]), f16_to_f32(bits[2]), f16_to_f32(bits[3])]
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Public dispatch API
    //
    // Each function appears at most once per compilation because the cfg guards
    // are mutually exclusive (a target cannot be both x86 and aarch64).
    // ─────────────────────────────────────────────────────────────────────────

    // f32 → u16 ─────────────────────────────────────────────────────────────
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "f16c"))]
    #[inline] pub fn f32_to_f16(v: f32) -> u16 { unsafe { hw::f32_to_f16(v) } }

    #[cfg(all(target_arch = "aarch64", target_feature = "fp16"))]
    #[inline] pub fn f32_to_f16(v: f32) -> u16 { unsafe { hw::f32_to_f16(v) } }

    #[cfg(not(any(
        all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "f16c"),
        all(target_arch = "aarch64", target_feature = "fp16"),
    )))]
    #[inline] pub fn f32_to_f16(v: f32) -> u16 { soft::f32_to_f16_soft(v) }

    // u16 → f32 ─────────────────────────────────────────────────────────────
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "f16c"))]
    #[inline] pub fn f16_to_f32(bits: u16) -> f32 { unsafe { hw::f16_to_f32(bits) } }

    #[cfg(all(target_arch = "aarch64", target_feature = "fp16"))]
    #[inline] pub fn f16_to_f32(bits: u16) -> f32 { unsafe { hw::f16_to_f32(bits) } }

    #[cfg(not(any(
        all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "f16c"),
        all(target_arch = "aarch64", target_feature = "fp16"),
    )))]
    #[inline] pub fn f16_to_f32(bits: u16) -> f32 { soft::f16_to_f32_soft(bits) }

    // f64 → u16 (always via software — no hardware path worth the complexity)
    #[inline] pub fn f64_to_f16(v: f64) -> u16 { soft::f64_to_f16_soft(v) }

    // x4 batches ─────────────────────────────────────────────────────────────
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "f16c"))]
    #[inline] pub fn f32x4_to_f16x4(v: [f32; 4]) -> [u16; 4] { unsafe { hw::f32x4_to_f16x4(v) } }

    #[cfg(all(target_arch = "aarch64", target_feature = "fp16"))]
    #[inline] pub fn f32x4_to_f16x4(v: [f32; 4]) -> [u16; 4] { unsafe { hw::f32x4_to_f16x4(v) } }

    #[cfg(not(any(
        all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "f16c"),
        all(target_arch = "aarch64", target_feature = "fp16"),
    )))]
    #[inline] pub fn f32x4_to_f16x4(v: [f32; 4]) -> [u16; 4] {
        [soft::f32_to_f16_soft(v[0]), soft::f32_to_f16_soft(v[1]),
         soft::f32_to_f16_soft(v[2]), soft::f32_to_f16_soft(v[3])]
    }

    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "f16c"))]
    #[inline] pub fn f16x4_to_f32x4(bits: [u16; 4]) -> [f32; 4] { unsafe { hw::f16x4_to_f32x4(bits) } }

    #[cfg(all(target_arch = "aarch64", target_feature = "fp16"))]
    #[inline] pub fn f16x4_to_f32x4(bits: [u16; 4]) -> [f32; 4] { unsafe { hw::f16x4_to_f32x4(bits) } }

    #[cfg(not(any(
        all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "f16c"),
        all(target_arch = "aarch64", target_feature = "fp16"),
    )))]
    #[inline] pub fn f16x4_to_f32x4(bits: [u16; 4]) -> [f32; 4] {
        [soft::f16_to_f32_soft(bits[0]), soft::f16_to_f32_soft(bits[1]),
         soft::f16_to_f32_soft(bits[2]), soft::f16_to_f32_soft(bits[3])]
    }

    // x8 batches (two x4 calls; compiler fuses into AVX VCVTPS2PH when possible)
    #[inline] pub fn f32x8_to_f16x8(v: [f32; 8]) -> [u16; 8] {
        let lo = f32x4_to_f16x4([v[0], v[1], v[2], v[3]]);
        let hi = f32x4_to_f16x4([v[4], v[5], v[6], v[7]]);
        [lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3]]
    }

    #[inline] pub fn f16x8_to_f32x8(bits: [u16; 8]) -> [f32; 8] {
        let lo = f16x4_to_f32x4([bits[0], bits[1], bits[2], bits[3]]);
        let hi = f16x4_to_f32x4([bits[4], bits[5], bits[6], bits[7]]);
        [lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3]]
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Pure-Rust IEEE 754 bit-manipulation fallback
    //
    // Algorithm derived from the `half` crate (MIT / Apache-2.0):
    //   https://github.com/VoidStarKat/half-rs
    // Ported to be dependency-free: uses `f32::to_bits()` / `f32::from_bits()`
    // instead of `mem::transmute`, and `u16::leading_zeros()` instead of the
    // `crunchy` crate for the SPIRV-free path.
    // ─────────────────────────────────────────────────────────────────────────
    mod soft {

        // ── f32 → binary16 ──────────────────────────────────────────────────
        //
        // Rounds to nearest, ties to even.
        // round_bit  = first bit that gets dropped.
        // Round up when: round_bit is set AND (any sticky bit is set OR the LSB
        // of the result is 1, i.e. round-to-even applies).
        // Combined: (man & round_bit) != 0 && (man & (3*round_bit - 1)) != 0
        pub const fn f32_to_f16_soft(value: f32) -> u16 {
            let x    = value.to_bits();
            let sign = x & 0x8000_0000u32;
            let exp  = x & 0x7F80_0000u32;
            let man  = x & 0x007F_FFFFu32;

            // All-ones exponent → Infinity or NaN
            if exp == 0x7F80_0000u32 {
                let nan_bit = if man == 0 { 0u32 } else { 0x0200u32 };
                return ((sign >> 16) | 0x7C00u32 | nan_bit | (man >> 13)) as u16;
            }

            let half_sign = sign >> 16;
            let unbiased  = (exp >> 23) as i32 - 127;
            let half_exp  = unbiased + 15;

            // Overflow → ±∞
            if half_exp >= 0x1F {
                return (half_sign | 0x7C00u32) as u16;
            }

            // Underflow → subnormal or zero
            if half_exp <= 0 {
                if 14 - half_exp > 24 {
                    return half_sign as u16; // too small, flush to zero
                }
                let man_h        = man | 0x0080_0000u32;
                let mut half_man = man_h >> (14 - half_exp);
                let round_bit    = 1u32 << (13 - half_exp);
                if (man_h & round_bit) != 0 && (man_h & (3 * round_bit - 1)) != 0 {
                    half_man += 1;
                }
                return (half_sign | half_man) as u16;
            }

            // Normal
            let half_exp_bits = (half_exp as u32) << 10;
            let half_man      = man >> 13;
            let round_bit     = 0x0000_1000u32;
            if (man & round_bit) != 0 && (man & (3 * round_bit - 1)) != 0 {
                ((half_sign | half_exp_bits | half_man) + 1) as u16
            } else {
                (half_sign | half_exp_bits | half_man) as u16
            }
        }

        // ── binary16 → f32 ──────────────────────────────────────────────────
        pub const fn f16_to_f32_soft(i: u16) -> f32 {
            // Signed zero short-circuit
            if i & 0x7FFF == 0 {
                return f32::from_bits((i as u32) << 16);
            }

            let half_sign = (i & 0x8000u16) as u32;
            let half_exp  = (i & 0x7C00u16) as u32;
            let half_man  = (i & 0x03FFu16) as u32;

            // Infinity or NaN
            if half_exp == 0x7C00u32 {
                return if half_man == 0 {
                    f32::from_bits((half_sign << 16) | 0x7F80_0000u32)
                } else {
                    // Set MSB of mantissa → quiet NaN
                    f32::from_bits((half_sign << 16) | 0x7FC0_0000u32 | (half_man << 13))
                };
            }

            let sign = half_sign << 16;

            // Subnormal f16 → normalised f32
            if half_exp == 0 {
                // leading zeros within the 10-bit mantissa field
                let e         = (half_man as u16).leading_zeros() - 6;
                let exp_bits  = (127u32 - 15u32 - e) << 23;
                let man_bits  = (half_man << (14 + e)) & 0x007F_FFFFu32;
                return f32::from_bits(sign | exp_bits | man_bits);
            }

            // Normal f16 → normal f32
            let unbiased = ((half_exp as i32) >> 10) - 15;
            let exp_bits = ((unbiased + 127) as u32) << 23;
            let man_bits = (half_man & 0x03FFu32) << 13;
            f32::from_bits(sign | exp_bits | man_bits)
        }

        // ── f64 → binary16 ──────────────────────────────────────────────────
        // Uses the upper 32 bits of the f64 mantissa — the lower 32 bits are
        // completely lost in an f64→f16 conversion anyway (f16 has only 10 bits).
        pub const fn f64_to_f16_soft(value: f64) -> u16 {
            let val: u64 = value.to_bits();
            let x        = (val >> 32) as u32;

            let sign = x & 0x8000_0000u32;
            let exp  = x & 0x7FF0_0000u32;
            let man  = x & 0x000F_FFFFu32;

            if exp == 0x7FF0_0000u32 {
                let nan_bit = if man == 0 && val as u32 == 0 { 0u32 } else { 0x0200u32 };
                return ((sign >> 16) | 0x7C00u32 | nan_bit | (man >> 10)) as u16;
            }

            let half_sign = sign >> 16;
            let unbiased  = ((exp >> 20) as i64) - 1023;
            let half_exp  = unbiased + 15;

            if half_exp >= 0x1F { return (half_sign | 0x7C00u32) as u16; }
            if half_exp <= 0 {
                if 10 - half_exp > 21 { return half_sign as u16; }
                let man_h     = man | 0x0010_0000u32;
                let mut hm    = man_h >> (11 - half_exp) as u32;
                let rb        = 1u32 << (10 - half_exp) as u32;
                if (man_h & rb) != 0 && (man_h & (3 * rb - 1)) != 0 { hm += 1; }
                return (half_sign | hm) as u16;
            }

            let half_exp_bits = (half_exp as u32) << 10;
            let half_man      = man >> 10;
            let round_bit     = 0x0000_0200u32;
            if (man & round_bit) != 0 && (man & (3 * round_bit - 1)) != 0 {
                ((half_sign | half_exp_bits | half_man) + 1) as u16
            } else {
                (half_sign | half_exp_bits | half_man) as u16
            }
        }
    } // mod soft
} // mod dispatch

// ═════════════════════════════════════════════════════════════════════════════
// Tests
// ═════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn storage_is_two_bytes() {
        assert_eq!(core::mem::size_of::<f16>(), 2,
            "f16 must always be exactly 2 bytes — the f32 in conversions is NOT stored");
    }

    #[test] fn zero() { assert_eq!(f16::from_f32(0.0).to_f32(), 0.0); }
    #[test] fn neg_zero() { assert_eq!(f16::from_f32(-0.0).to_bits(), f16::NEG_ZERO.to_bits()); }
    #[test] fn one()  { assert_eq!(f16::from_f32(1.0).to_bits(), f16::ONE.to_bits()); }
    #[test] fn neg_one() { assert_eq!(f16::from_f32(-1.0).to_bits(), f16::NEG_ONE.to_bits()); }
    #[test] fn neg_flip() { assert_eq!((-f16::ONE).to_bits(), f16::NEG_ONE.to_bits()); }

    #[test]
    fn pi_approx() {
        let h  = f16::from_f32(core::f32::consts::PI);
        let rt = h.to_f32();
        assert!((rt - core::f32::consts::PI).abs() < 0.002,
            "pi roundtrip: got {rt}");
    }

    #[test]
    fn infinity_and_nan() {
        assert!(f16::INFINITY.is_infinite());
        assert!(f16::NEG_INFINITY.is_infinite());
        assert!(f16::NAN.is_nan());
        assert!(f16::MAX.is_finite());
        assert!(!f16::MAX.is_nan());
    }

    #[test]
    fn overflow_to_infinity() {
        assert!(f16::from_f32(1e10).is_infinite(),
            "values beyond MAX must overflow to infinity");
    }

    #[test]
    fn negative_roundtrip() {
        for &v in &[-1.0f32, -0.5, -100.0, -65504.0] {
            let rt = f16::from_f32(v).to_f32();
            assert!((rt - v).abs() < v.abs() * 0.002,
                "negative roundtrip failed for {v}: got {rt}");
        }
    }

    #[test]
    fn subnormal_positive() {
        let h = f16::MIN_POSITIVE_SUBNORMAL;
        assert!(h.is_subnormal());
        assert!(h.to_f32() > 0.0);
    }

    #[test]
    fn batch_x4() {
        let src = [1.0f32, 2.0, 0.5, -3.5];
        let rt  = f16x4_to_f32x4(f32x4_to_f16x4(src));
        for i in 0..4 {
            assert!((rt[i] - src[i]).abs() < src[i].abs().max(1.0) * 0.002,
                "x4 batch [{i}]: {s} → {r}", s = src[i], r = rt[i]);
        }
    }

    #[test]
    fn batch_x8() {
        let src = [1.0f32, -1.0, 0.5, 0.25, 100.0, -100.0, 0.0, 65504.0];
        let rt  = f16x8_to_f32x8(f32x8_to_f16x8(src));
        for i in 0..8 {
            assert!((rt[i] - src[i]).abs() < src[i].abs().max(1.0) * 0.002,
                "x8 batch [{i}]: {s} → {r}", s = src[i], r = rt[i]);
        }
    }

    #[test]
    fn slice_conversion() {
        let src: Vec<f32>  = (0..100).map(|i| i as f32 * 0.1).collect();
        let mut h: Vec<f16> = vec![f16::ZERO; 100];
        let mut rt: Vec<f32> = vec![0.0; 100];
        f32_slice_to_f16(&src, &mut h);
        f16_slice_to_f32(&h, &mut rt);
        for i in 0..100 {
            assert!((rt[i] - src[i]).abs() < src[i].abs().max(1.0) * 0.002,
                "slice roundtrip [{i}]: {s} → {r}", s = src[i], r = rt[i]);
        }
    }

    #[test]
    fn partial_eq_nan() {
        assert_ne!(f16::NAN, f16::NAN, "NaN must not equal itself");
    }

    #[test]
    fn abs_and_copysign() {
        assert_eq!((-f16::ONE).abs().to_bits(), f16::ONE.to_bits());
        let a = f16::ONE.copysign(f16::NEG_ONE);
        assert_eq!(a.to_bits(), f16::NEG_ONE.to_bits());
    }

    #[test]
    fn from_f64() {
        let h = f16::from_f64(3.14159265358979);
        assert!((h.to_f64() - 3.14159).abs() < 0.01);
    }
}
