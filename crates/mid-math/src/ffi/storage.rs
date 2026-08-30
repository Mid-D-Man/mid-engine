// crates/mid-math/src/ffi/storage.rs
//! C-ABI exports for the storage/ compressed-format types.
//!
//! f16/bf16/F8Norm/F8E4M3/F8E5M2 are exposed as their raw bit pattern
//! (u16 or u8) directly -- no C wrapper struct needed, since these
//! Rust types are themselves plain single-field tuple structs over
//! u16/u8 (confirmed directly against source: `pub struct f16(u16)`,
//! `pub struct F8Norm(u8)`, etc.) and the module's own documented
//! philosophy is "store compressed, unpack to f32 only to compute" --
//! so no arithmetic (add/sub/mul) is exposed here, matching that these
//! types don't implement it in Rust either. Conversion (from_f32/to_f32)
//! plus the handful of bit-level predicates (is_nan/is_finite/is_zero/
//! abs/copysign) that DO exist directly on the type are what's exposed.
//!
//! BitMask8/16/32/64 are `#[repr(transparent)]` over u8/u16/u32/u64
//! (confirmed against source) -- exposed as that same raw integer type.
//! BitMask128/256 are `#[repr(C)]` over `[u64;2]`/`[u64;4]` already,
//! so their existing layout IS the C layout -- exposed via a thin
//! `[u64;2]`/`[u64;4]`-based C type reusing that representation, not a
//! new one.
//!
//! Deliberately excludes F4E2M1/F4E3M0/F4E2M1Pair/F4E3M0Pair (4-bit,
//! packed two-per-byte) -- that packing scheme needs its own C-side
//! representation decision (a single loose 4-bit value has no natural
//! standalone C type), not decided here, same deferral pattern as the
//! AVX2-only wide types in wide_int.rs/wide_float.rs.

use crate::{f16, bf16, F8Norm, F8E4M3, F8E5M2, BitMask8, BitMask16, BitMask32, BitMask64, BitMask128, BitMask256};

// ═══════════════════════════════════════════════════════════════════════════
//  Exports — f16 (raw u16 bit pattern)
// ═══════════════════════════════════════════════════════════════════════════

#[no_mangle] pub extern "C" fn mid_f16_from_f32(v:f32)->u16{f16::from_f32(v).to_bits()}
#[no_mangle] pub extern "C" fn mid_f16_to_f32(bits:u16)->f32{f16::from_bits(bits).to_f32()}
#[no_mangle] pub extern "C" fn mid_f16_abs(bits:u16)->u16{f16::from_bits(bits).abs().to_bits()}
#[no_mangle] pub extern "C" fn mid_f16_copysign(bits:u16,sign_src:u16)->u16{f16::from_bits(bits).copysign(f16::from_bits(sign_src)).to_bits()}
#[no_mangle] pub extern "C" fn mid_f16_is_nan(bits:u16)->bool{f16::from_bits(bits).is_nan()}
#[no_mangle] pub extern "C" fn mid_f16_is_finite(bits:u16)->bool{f16::from_bits(bits).is_finite()}
#[no_mangle] pub extern "C" fn mid_f16_is_zero(bits:u16)->bool{f16::from_bits(bits).is_zero()}

// ═══════════════════════════════════════════════════════════════════════════
//  Exports — bf16 (raw u16 bit pattern)
// ═══════════════════════════════════════════════════════════════════════════

#[no_mangle] pub extern "C" fn mid_bf16_from_f32(v:f32)->u16{bf16::from_f32(v).to_bits()}
#[no_mangle] pub extern "C" fn mid_bf16_to_f32(bits:u16)->f32{bf16::from_bits(bits).to_f32()}
#[no_mangle] pub extern "C" fn mid_bf16_abs(bits:u16)->u16{bf16::from_bits(bits).abs().to_bits()}
#[no_mangle] pub extern "C" fn mid_bf16_copysign(bits:u16,sign_src:u16)->u16{bf16::from_bits(bits).copysign(bf16::from_bits(sign_src)).to_bits()}
#[no_mangle] pub extern "C" fn mid_bf16_is_nan(bits:u16)->bool{bf16::from_bits(bits).is_nan()}
#[no_mangle] pub extern "C" fn mid_bf16_is_finite(bits:u16)->bool{bf16::from_bits(bits).is_finite()}
#[no_mangle] pub extern "C" fn mid_bf16_is_zero(bits:u16)->bool{bf16::from_bits(bits).is_zero()}

// ═══════════════════════════════════════════════════════════════════════════
//  Exports — F8Norm (raw u8 bit pattern) -- [0.0, 1.0], colors/alpha/blend weights
// ═══════════════════════════════════════════════════════════════════════════

#[no_mangle] pub extern "C" fn mid_f8norm_from_f32(v:f32)->u8{F8Norm::from_f32(v).to_bits()}
#[no_mangle] pub extern "C" fn mid_f8norm_to_f32(bits:u8)->f32{F8Norm::from_bits(bits).to_f32()}
#[no_mangle] pub extern "C" fn mid_f8norm_lerp(a:u8,b:u8,t:u8)->u8{F8Norm::lerp(F8Norm::from_bits(a),F8Norm::from_bits(b),F8Norm::from_bits(t)).to_bits()}

// ═══════════════════════════════════════════════════════════════════════════
//  Exports — F8E4M3 (raw u8 bit pattern) -- ±448.0, ML weights/activations
// ═══════════════════════════════════════════════════════════════════════════

#[no_mangle] pub extern "C" fn mid_f8e4m3_from_f32(v:f32)->u8{F8E4M3::from_f32(v).to_bits()}
#[no_mangle] pub extern "C" fn mid_f8e4m3_to_f32(bits:u8)->f32{F8E4M3::from_bits(bits).to_f32()}

// ═══════════════════════════════════════════════════════════════════════════
//  Exports — F8E5M2 (raw u8 bit pattern) -- ±57344.0, ML gradients
// ═══════════════════════════════════════════════════════════════════════════

#[no_mangle] pub extern "C" fn mid_f8e5m2_from_f32(v:f32)->u8{F8E5M2::from_f32(v).to_bits()}
#[no_mangle] pub extern "C" fn mid_f8e5m2_to_f32(bits:u8)->f32{F8E5M2::from_bits(bits).to_f32()}

// ═══════════════════════════════════════════════════════════════════════════
//  Exports — BitMask8/16/32/64 (raw u8/u16/u32/u64, #[repr(transparent)] already)
// ═══════════════════════════════════════════════════════════════════════════

#[no_mangle] pub extern "C" fn mid_bitmask8_get(m:u8,index:u32)->bool{BitMask8::from_bits(m).get(index as usize)}
#[no_mangle] pub extern "C" fn mid_bitmask8_set(m:u8,index:u32)->u8{let mut v=BitMask8::from_bits(m);v.set(index as usize);v.to_bits()}
#[no_mangle] pub extern "C" fn mid_bitmask8_clear(m:u8,index:u32)->u8{let mut v=BitMask8::from_bits(m);v.clear(index as usize);v.to_bits()}
#[no_mangle] pub extern "C" fn mid_bitmask8_toggle(m:u8,index:u32)->u8{let mut v=BitMask8::from_bits(m);v.toggle(index as usize);v.to_bits()}
#[no_mangle] pub extern "C" fn mid_bitmask8_any(m:u8)->bool{BitMask8::from_bits(m).any()}
#[no_mangle] pub extern "C" fn mid_bitmask8_all(m:u8)->bool{BitMask8::from_bits(m).all()}
#[no_mangle] pub extern "C" fn mid_bitmask8_none(m:u8)->bool{BitMask8::from_bits(m).none()}
#[no_mangle] pub extern "C" fn mid_bitmask8_count_ones(m:u8)->u32{BitMask8::from_bits(m).count_ones()}

#[no_mangle] pub extern "C" fn mid_bitmask16_get(m:u16,index:u32)->bool{BitMask16::from_bits(m).get(index as usize)}
#[no_mangle] pub extern "C" fn mid_bitmask16_set(m:u16,index:u32)->u16{let mut v=BitMask16::from_bits(m);v.set(index as usize);v.to_bits()}
#[no_mangle] pub extern "C" fn mid_bitmask16_clear(m:u16,index:u32)->u16{let mut v=BitMask16::from_bits(m);v.clear(index as usize);v.to_bits()}
#[no_mangle] pub extern "C" fn mid_bitmask16_toggle(m:u16,index:u32)->u16{let mut v=BitMask16::from_bits(m);v.toggle(index as usize);v.to_bits()}
#[no_mangle] pub extern "C" fn mid_bitmask16_any(m:u16)->bool{BitMask16::from_bits(m).any()}
#[no_mangle] pub extern "C" fn mid_bitmask16_all(m:u16)->bool{BitMask16::from_bits(m).all()}
#[no_mangle] pub extern "C" fn mid_bitmask16_none(m:u16)->bool{BitMask16::from_bits(m).none()}
#[no_mangle] pub extern "C" fn mid_bitmask16_count_ones(m:u16)->u32{BitMask16::from_bits(m).count_ones()}

#[no_mangle] pub extern "C" fn mid_bitmask32_get(m:u32,index:u32)->bool{BitMask32::from_bits(m).get(index as usize)}
#[no_mangle] pub extern "C" fn mid_bitmask32_set(m:u32,index:u32)->u32{let mut v=BitMask32::from_bits(m);v.set(index as usize);v.to_bits()}
#[no_mangle] pub extern "C" fn mid_bitmask32_clear(m:u32,index:u32)->u32{let mut v=BitMask32::from_bits(m);v.clear(index as usize);v.to_bits()}
#[no_mangle] pub extern "C" fn mid_bitmask32_toggle(m:u32,index:u32)->u32{let mut v=BitMask32::from_bits(m);v.toggle(index as usize);v.to_bits()}
#[no_mangle] pub extern "C" fn mid_bitmask32_any(m:u32)->bool{BitMask32::from_bits(m).any()}
#[no_mangle] pub extern "C" fn mid_bitmask32_all(m:u32)->bool{BitMask32::from_bits(m).all()}
#[no_mangle] pub extern "C" fn mid_bitmask32_none(m:u32)->bool{BitMask32::from_bits(m).none()}
#[no_mangle] pub extern "C" fn mid_bitmask32_count_ones(m:u32)->u32{BitMask32::from_bits(m).count_ones()}

#[no_mangle] pub extern "C" fn mid_bitmask64_get(m:u64,index:u32)->bool{BitMask64::from_bits(m).get(index as usize)}
#[no_mangle] pub extern "C" fn mid_bitmask64_set(m:u64,index:u32)->u64{let mut v=BitMask64::from_bits(m);v.set(index as usize);v.to_bits()}
#[no_mangle] pub extern "C" fn mid_bitmask64_clear(m:u64,index:u32)->u64{let mut v=BitMask64::from_bits(m);v.clear(index as usize);v.to_bits()}
#[no_mangle] pub extern "C" fn mid_bitmask64_toggle(m:u64,index:u32)->u64{let mut v=BitMask64::from_bits(m);v.toggle(index as usize);v.to_bits()}
#[no_mangle] pub extern "C" fn mid_bitmask64_any(m:u64)->bool{BitMask64::from_bits(m).any()}
#[no_mangle] pub extern "C" fn mid_bitmask64_all(m:u64)->bool{BitMask64::from_bits(m).all()}
#[no_mangle] pub extern "C" fn mid_bitmask64_none(m:u64)->bool{BitMask64::from_bits(m).none()}
#[no_mangle] pub extern "C" fn mid_bitmask64_count_ones(m:u64)->u32{BitMask64::from_bits(m).count_ones()}

// ═══════════════════════════════════════════════════════════════════════════
//  Exports — BitMask128/256 ([u64;2] / [u64;4], #[repr(C)] already)
// ═══════════════════════════════════════════════════════════════════════════

#[no_mangle] pub extern "C" fn mid_bitmask128_get(m:[u64;2],index:u32)->bool{BitMask128::from_words(m).get(index as usize)}
#[no_mangle] pub extern "C" fn mid_bitmask128_set(m:[u64;2],index:u32)->[u64;2]{let mut v=BitMask128::from_words(m);v.set(index as usize);v.to_words()}
#[no_mangle] pub extern "C" fn mid_bitmask128_clear(m:[u64;2],index:u32)->[u64;2]{let mut v=BitMask128::from_words(m);v.clear(index as usize);v.to_words()}
#[no_mangle] pub extern "C" fn mid_bitmask128_toggle(m:[u64;2],index:u32)->[u64;2]{let mut v=BitMask128::from_words(m);v.toggle(index as usize);v.to_words()}
#[no_mangle] pub extern "C" fn mid_bitmask128_any(m:[u64;2])->bool{BitMask128::from_words(m).any()}
#[no_mangle] pub extern "C" fn mid_bitmask128_all(m:[u64;2])->bool{BitMask128::from_words(m).all()}
#[no_mangle] pub extern "C" fn mid_bitmask128_none(m:[u64;2])->bool{BitMask128::from_words(m).none()}
#[no_mangle] pub extern "C" fn mid_bitmask128_count_ones(m:[u64;2])->u32{BitMask128::from_words(m).count_ones()}

#[no_mangle] pub extern "C" fn mid_bitmask256_get(m:[u64;4],index:u32)->bool{BitMask256::from_words(m).get(index as usize)}
#[no_mangle] pub extern "C" fn mid_bitmask256_set(m:[u64;4],index:u32)->[u64;4]{let mut v=BitMask256::from_words(m);v.set(index as usize);v.to_words()}
#[no_mangle] pub extern "C" fn mid_bitmask256_clear(m:[u64;4],index:u32)->[u64;4]{let mut v=BitMask256::from_words(m);v.clear(index as usize);v.to_words()}
#[no_mangle] pub extern "C" fn mid_bitmask256_toggle(m:[u64;4],index:u32)->[u64;4]{let mut v=BitMask256::from_words(m);v.toggle(index as usize);v.to_words()}
#[no_mangle] pub extern "C" fn mid_bitmask256_any(m:[u64;4])->bool{BitMask256::from_words(m).any()}
#[no_mangle] pub extern "C" fn mid_bitmask256_all(m:[u64;4])->bool{BitMask256::from_words(m).all()}
#[no_mangle] pub extern "C" fn mid_bitmask256_none(m:[u64;4])->bool{BitMask256::from_words(m).none()}
#[no_mangle] pub extern "C" fn mid_bitmask256_count_ones(m:[u64;4])->u32{BitMask256::from_words(m).count_ones()}
