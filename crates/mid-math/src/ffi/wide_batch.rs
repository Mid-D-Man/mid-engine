// crates/mid-math/src/ffi/wide_batch.rs
//! C-ABI "width-hiding" batch functions -- "FFI option 3" from the AVX2
//! rework's design pass, alongside wide_int_avx2.rs/wide_float_avx2.rs's
//! "option 1" (direct dispatched access to a specific width).
//!
//! Unlike option 1, these are NOT architecture-gated -- that's the
//! whole point. A C caller passes flat scalar arrays + a count and
//! never picks, or even knows about, a lane width at all:
//!
//!   void mid_i32_batch_add(const int32_t* a, const int32_t* b,
//!                          int32_t* out, uint32_t count);
//!
//! Internally, each function chunks the range in three passes:
//!   1. AVX2-only wide types (i32x8 and friends, f32x8) -- x86/x86_64
//!      only, since that's all they exist on. Each already dispatches
//!      to a real AVX2 instruction or a portable two-half fallback
//!      internally at runtime (see wide/int/avx2/*.rs's own doc
//!      comments) -- this file doesn't duplicate that check, it's
//!      just a consumer of it.
//!   2. The always-available narrow types (i32x4 and friends, f32x4)
//!      -- SSE2/NEON/scalar/wasm-tier, present on every architecture
//!      -- for the remainder below one AVX2-width chunk, and for the
//!      ENTIRE range on non-x86 architectures where step 1 doesn't
//!      compile at all.
//!   3. A plain per-element scalar tail below one narrow-width chunk.
//!
//! No existing precedent in this crate to mirror for the chunking
//! itself -- this crate's other "_batch" FFI functions (ffi/noise.rs)
//! are plain per-element scalar loops with no internal SIMD chunking,
//! so this is new design, not a copy of an established pattern.
//!
//! Scalar-tail semantics deliberately match the wide types' own
//! arithmetic, not Rust's default integer semantics: i32x4's Add/Sub/
//! Mul (crates/mid-math/src/wide/int/sse2/i32x4.rs) compile to raw
//! SIMD intrinsics, which wrap on overflow with no panic either way --
//! confirmed directly against source before choosing this. Using plain
//! `+`/`-`/`*` for the scalar tail would panic on overflow in debug
//! builds while the SIMD-chunked portion of the very same call
//! silently wraps, so the tail uses `wrapping_add`/`wrapping_sub`/
//! `wrapping_mul` instead, making behavior identical regardless of
//! where the count happens to fall relative to a chunk boundary.
//! min/max/f32 arithmetic have no such discrepancy (min/max never
//! overflow; float ops don't panic), so no equivalent adjustment
//! needed there.
//!
//! `mul` is omitted for i8/u8 -- i8x32/u8x32 have no `Mul` impl,
//! matching i8x16/u8x16's own omission (see wide_int.rs's header).
//!
//! # Safety contract (all pointer arguments)
//!   Non-null, valid for stated element count, caller owns memory --
//!   same contract as this crate's other slice-based batch FFI
//!   functions (ffi/noise.rs, ffi/camera.rs). Overlapping/aliased
//!   `a`/`b`/`out` buffers are not a documented, supported case here
//!   (same as those files' own silence on the topic) -- pass distinct
//!   buffers.

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use crate::{i32x8, u32x8, i16x16, u16x16, i8x32, u8x32, f32x8};
use crate::{i32x4, u32x4, i16x8, u16x8, i8x16, u8x16, f32x4};

// =============================================================================
//  Shared chunking machinery
// =============================================================================

/// Generates one `#[no_mangle]` batch binary-op function. Each of
/// `$wide_op`/`$narrow_op`/`$scalar_op` is a non-capturing closure
/// (coerced to a plain fn pointer -- these types implement the
/// arithmetic op or expose `.min()`/`.max()` as inherent methods, so
/// no trait bound machinery is needed here).
macro_rules! def_batch_binop {
    (
        $fn_name:ident, $scalar:ty,
        $narrow:ty, $narrow_n:literal, $narrow_op:expr,
        $wide:ty, $wide_n:literal, $wide_op:expr,
        $scalar_op:expr
    ) => {
        #[no_mangle]
        pub unsafe extern "C" fn $fn_name(
            a: *const $scalar, b: *const $scalar, out: *mut $scalar, count: u32,
        ) {
            let n = count as usize;
            let a = core::slice::from_raw_parts(a, n);
            let b = core::slice::from_raw_parts(b, n);
            let out = core::slice::from_raw_parts_mut(out, n);
            let mut i = 0usize;

            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                let op: fn($wide, $wide) -> $wide = $wide_op;
                while i + $wide_n <= n {
                    let wa = <$wide>::from_array(a[i..i + $wide_n].try_into().unwrap());
                    let wb = <$wide>::from_array(b[i..i + $wide_n].try_into().unwrap());
                    out[i..i + $wide_n].copy_from_slice(&op(wa, wb).to_array());
                    i += $wide_n;
                }
            }

            let op: fn($narrow, $narrow) -> $narrow = $narrow_op;
            while i + $narrow_n <= n {
                let na = <$narrow>::from_array(a[i..i + $narrow_n].try_into().unwrap());
                let nb = <$narrow>::from_array(b[i..i + $narrow_n].try_into().unwrap());
                out[i..i + $narrow_n].copy_from_slice(&op(na, nb).to_array());
                i += $narrow_n;
            }

            let op: fn($scalar, $scalar) -> $scalar = $scalar_op;
            while i < n {
                out[i] = op(a[i], b[i]);
                i += 1;
            }
        }
    };
}

// =============================================================================
//  i32
// =============================================================================

def_batch_binop!(mid_i32_batch_add, i32, i32x4, 4, |a: i32x4, b: i32x4| a + b, i32x8, 8, |a: i32x8, b: i32x8| a + b, |a: i32, b: i32| a.wrapping_add(b));
def_batch_binop!(mid_i32_batch_sub, i32, i32x4, 4, |a: i32x4, b: i32x4| a - b, i32x8, 8, |a: i32x8, b: i32x8| a - b, |a: i32, b: i32| a.wrapping_sub(b));
def_batch_binop!(mid_i32_batch_mul, i32, i32x4, 4, |a: i32x4, b: i32x4| a * b, i32x8, 8, |a: i32x8, b: i32x8| a * b, |a: i32, b: i32| a.wrapping_mul(b));
def_batch_binop!(mid_i32_batch_min, i32, i32x4, 4, |a: i32x4, b: i32x4| a.min(b), i32x8, 8, |a: i32x8, b: i32x8| a.min(b), |a: i32, b: i32| a.min(b));
def_batch_binop!(mid_i32_batch_max, i32, i32x4, 4, |a: i32x4, b: i32x4| a.max(b), i32x8, 8, |a: i32x8, b: i32x8| a.max(b), |a: i32, b: i32| a.max(b));

// =============================================================================
//  u32
// =============================================================================

def_batch_binop!(mid_u32_batch_add, u32, u32x4, 4, |a: u32x4, b: u32x4| a + b, u32x8, 8, |a: u32x8, b: u32x8| a + b, |a: u32, b: u32| a.wrapping_add(b));
def_batch_binop!(mid_u32_batch_sub, u32, u32x4, 4, |a: u32x4, b: u32x4| a - b, u32x8, 8, |a: u32x8, b: u32x8| a - b, |a: u32, b: u32| a.wrapping_sub(b));
def_batch_binop!(mid_u32_batch_mul, u32, u32x4, 4, |a: u32x4, b: u32x4| a * b, u32x8, 8, |a: u32x8, b: u32x8| a * b, |a: u32, b: u32| a.wrapping_mul(b));
def_batch_binop!(mid_u32_batch_min, u32, u32x4, 4, |a: u32x4, b: u32x4| a.min(b), u32x8, 8, |a: u32x8, b: u32x8| a.min(b), |a: u32, b: u32| a.min(b));
def_batch_binop!(mid_u32_batch_max, u32, u32x4, 4, |a: u32x4, b: u32x4| a.max(b), u32x8, 8, |a: u32x8, b: u32x8| a.max(b), |a: u32, b: u32| a.max(b));

// =============================================================================
//  i16
// =============================================================================

def_batch_binop!(mid_i16_batch_add, i16, i16x8, 8, |a: i16x8, b: i16x8| a + b, i16x16, 16, |a: i16x16, b: i16x16| a + b, |a: i16, b: i16| a.wrapping_add(b));
def_batch_binop!(mid_i16_batch_sub, i16, i16x8, 8, |a: i16x8, b: i16x8| a - b, i16x16, 16, |a: i16x16, b: i16x16| a - b, |a: i16, b: i16| a.wrapping_sub(b));
// mul: both i16x8 and i16x16's Mul impls yield the low 16 bits of the
// product (mul_lo) -- matches i16::wrapping_mul's own semantics exactly.
def_batch_binop!(mid_i16_batch_mul, i16, i16x8, 8, |a: i16x8, b: i16x8| a * b, i16x16, 16, |a: i16x16, b: i16x16| a * b, |a: i16, b: i16| a.wrapping_mul(b));
def_batch_binop!(mid_i16_batch_min, i16, i16x8, 8, |a: i16x8, b: i16x8| a.min(b), i16x16, 16, |a: i16x16, b: i16x16| a.min(b), |a: i16, b: i16| a.min(b));
def_batch_binop!(mid_i16_batch_max, i16, i16x8, 8, |a: i16x8, b: i16x8| a.max(b), i16x16, 16, |a: i16x16, b: i16x16| a.max(b), |a: i16, b: i16| a.max(b));

// =============================================================================
//  u16
// =============================================================================

def_batch_binop!(mid_u16_batch_add, u16, u16x8, 8, |a: u16x8, b: u16x8| a + b, u16x16, 16, |a: u16x16, b: u16x16| a + b, |a: u16, b: u16| a.wrapping_add(b));
def_batch_binop!(mid_u16_batch_sub, u16, u16x8, 8, |a: u16x8, b: u16x8| a - b, u16x16, 16, |a: u16x16, b: u16x16| a - b, |a: u16, b: u16| a.wrapping_sub(b));
def_batch_binop!(mid_u16_batch_mul, u16, u16x8, 8, |a: u16x8, b: u16x8| a * b, u16x16, 16, |a: u16x16, b: u16x16| a * b, |a: u16, b: u16| a.wrapping_mul(b));
def_batch_binop!(mid_u16_batch_min, u16, u16x8, 8, |a: u16x8, b: u16x8| a.min(b), u16x16, 16, |a: u16x16, b: u16x16| a.min(b), |a: u16, b: u16| a.min(b));
def_batch_binop!(mid_u16_batch_max, u16, u16x8, 8, |a: u16x8, b: u16x8| a.max(b), u16x16, 16, |a: u16x16, b: u16x16| a.max(b), |a: u16, b: u16| a.max(b));

// =============================================================================
//  i8 -- no mul: i8x16/i8x32 have no Mul impl (see wide_int.rs's header)
// =============================================================================

def_batch_binop!(mid_i8_batch_add, i8, i8x16, 16, |a: i8x16, b: i8x16| a + b, i8x32, 32, |a: i8x32, b: i8x32| a + b, |a: i8, b: i8| a.wrapping_add(b));
def_batch_binop!(mid_i8_batch_sub, i8, i8x16, 16, |a: i8x16, b: i8x16| a - b, i8x32, 32, |a: i8x32, b: i8x32| a - b, |a: i8, b: i8| a.wrapping_sub(b));
def_batch_binop!(mid_i8_batch_min, i8, i8x16, 16, |a: i8x16, b: i8x16| a.min(b), i8x32, 32, |a: i8x32, b: i8x32| a.min(b), |a: i8, b: i8| a.min(b));
def_batch_binop!(mid_i8_batch_max, i8, i8x16, 16, |a: i8x16, b: i8x16| a.max(b), i8x32, 32, |a: i8x32, b: i8x32| a.max(b), |a: i8, b: i8| a.max(b));

// =============================================================================
//  u8 -- no mul, same reasoning as i8 above
// =============================================================================

def_batch_binop!(mid_u8_batch_add, u8, u8x16, 16, |a: u8x16, b: u8x16| a + b, u8x32, 32, |a: u8x32, b: u8x32| a + b, |a: u8, b: u8| a.wrapping_add(b));
def_batch_binop!(mid_u8_batch_sub, u8, u8x16, 16, |a: u8x16, b: u8x16| a - b, u8x32, 32, |a: u8x32, b: u8x32| a - b, |a: u8, b: u8| a.wrapping_sub(b));
def_batch_binop!(mid_u8_batch_min, u8, u8x16, 16, |a: u8x16, b: u8x16| a.min(b), u8x32, 32, |a: u8x32, b: u8x32| a.min(b), |a: u8, b: u8| a.min(b));
def_batch_binop!(mid_u8_batch_max, u8, u8x16, 16, |a: u8x16, b: u8x16| a.max(b), u8x32, 32, |a: u8x32, b: u8x32| a.max(b), |a: u8, b: u8| a.max(b));

// =============================================================================
//  f32 -- no wrapping concept; plain arithmetic throughout. min/max use
//  the wide types' own (hardware-min/max-backed) semantics for the
//  chunked portion and f32::min/f32::max (IEEE-754 minNum/maxNum,
//  non-NaN-propagating) for the scalar tail -- not bit-identical to
//  every possible hardware min/max instruction's NaN edge case, but
//  consistent with how the rest of this crate treats float min/max
//  (see wide_float.rs, which makes the same choice for f32x4).
// =============================================================================

def_batch_binop!(mid_f32_batch_add, f32, f32x4, 4, |a: f32x4, b: f32x4| a + b, f32x8, 8, |a: f32x8, b: f32x8| a + b, |a: f32, b: f32| a + b);
def_batch_binop!(mid_f32_batch_sub, f32, f32x4, 4, |a: f32x4, b: f32x4| a - b, f32x8, 8, |a: f32x8, b: f32x8| a - b, |a: f32, b: f32| a - b);
def_batch_binop!(mid_f32_batch_mul, f32, f32x4, 4, |a: f32x4, b: f32x4| a * b, f32x8, 8, |a: f32x8, b: f32x8| a * b, |a: f32, b: f32| a * b);
def_batch_binop!(mid_f32_batch_min, f32, f32x4, 4, |a: f32x4, b: f32x4| a.min(b), f32x8, 8, |a: f32x8, b: f32x8| a.min(b), |a: f32, b: f32| a.min(b));
def_batch_binop!(mid_f32_batch_max, f32, f32x4, 4, |a: f32x4, b: f32x4| a.max(b), f32x8, 8, |a: f32x8, b: f32x8| a.max(b), |a: f32, b: f32| a.max(b));
