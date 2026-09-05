// crates/mid-math/src/swizzle/wide_int.rs
//! Lane-shuffle impls for the wide int family (via `LaneShuffle4`/`8`/`16`,
//! see `wide_lane_engine.rs`). No axis-swizzle here — these are all opaque
//! single-register types with no x/y/z/w fields (`i32x4(pub(crate) __m128i)`),
//! unlike `Vec3x4`. Each `#[cfg(...)]` here is copied from that backend's own
//! module declaration in `wide/int/mod.rs`. AVX2's wider types (`i32x8` etc.)
//! are additive alongside the always-available ones, not a replacement — same
//! relationship `wide/int/mod.rs`'s own doc comment describes.

// ── x86 / x86_64 (SSE2) ──
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
crate::impl_lane_shuffle4!(crate::wide::int::sse2::i32x4::i32x4);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
crate::impl_lane_shuffle4!(crate::wide::int::sse2::u32x4::u32x4);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
crate::impl_lane_shuffle8!(crate::wide::int::sse2::i16x8::i16x8);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
crate::impl_lane_shuffle8!(crate::wide::int::sse2::u16x8::u16x8);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
crate::impl_lane_shuffle16!(crate::wide::int::sse2::i8x16::i8x16);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
crate::impl_lane_shuffle16!(crate::wide::int::sse2::u8x16::u8x16);

// ── aarch64 (NEON) ──
#[cfg(target_arch = "aarch64")]
crate::impl_lane_shuffle4!(crate::wide::int::neon::i32x4::i32x4);
#[cfg(target_arch = "aarch64")]
crate::impl_lane_shuffle4!(crate::wide::int::neon::u32x4::u32x4);
#[cfg(target_arch = "aarch64")]
crate::impl_lane_shuffle8!(crate::wide::int::neon::i16x8::i16x8);
#[cfg(target_arch = "aarch64")]
crate::impl_lane_shuffle8!(crate::wide::int::neon::u16x8::u16x8);
#[cfg(target_arch = "aarch64")]
crate::impl_lane_shuffle16!(crate::wide::int::neon::i8x16::i8x16);
#[cfg(target_arch = "aarch64")]
crate::impl_lane_shuffle16!(crate::wide::int::neon::u8x16::u8x16);

// ── wasm32 / wasm64 + simd128 ──
#[cfg(all(any(target_arch = "wasm32", target_arch = "wasm64"), target_feature = "simd128"))]
crate::impl_lane_shuffle4!(crate::wide::int::wasm::i32x4::i32x4);
#[cfg(all(any(target_arch = "wasm32", target_arch = "wasm64"), target_feature = "simd128"))]
crate::impl_lane_shuffle4!(crate::wide::int::wasm::u32x4::u32x4);
#[cfg(all(any(target_arch = "wasm32", target_arch = "wasm64"), target_feature = "simd128"))]
crate::impl_lane_shuffle8!(crate::wide::int::wasm::i16x8::i16x8);
#[cfg(all(any(target_arch = "wasm32", target_arch = "wasm64"), target_feature = "simd128"))]
crate::impl_lane_shuffle8!(crate::wide::int::wasm::u16x8::u16x8);
#[cfg(all(any(target_arch = "wasm32", target_arch = "wasm64"), target_feature = "simd128"))]
crate::impl_lane_shuffle16!(crate::wide::int::wasm::i8x16::i8x16);
#[cfg(all(any(target_arch = "wasm32", target_arch = "wasm64"), target_feature = "simd128"))]
crate::impl_lane_shuffle16!(crate::wide::int::wasm::u8x16::u8x16);

// ── Scalar fallback (unconditional) ──
crate::impl_lane_shuffle4!(crate::wide::int::scalar::i32x4::i32x4);
crate::impl_lane_shuffle4!(crate::wide::int::scalar::u32x4::u32x4);
crate::impl_lane_shuffle8!(crate::wide::int::scalar::i16x8::i16x8);
crate::impl_lane_shuffle8!(crate::wide::int::scalar::u16x8::u16x8);
crate::impl_lane_shuffle16!(crate::wide::int::scalar::i8x16::i8x16);
crate::impl_lane_shuffle16!(crate::wide::int::scalar::u8x16::u8x16);

// ── AVX2 (additive) ──
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
crate::impl_lane_shuffle8!(crate::wide::int::avx2::i32x8::i32x8);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
crate::impl_lane_shuffle8!(crate::wide::int::avx2::u32x8::u32x8);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
crate::impl_lane_shuffle16!(crate::wide::int::avx2::i16x16::i16x16);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
crate::impl_lane_shuffle16!(crate::wide::int::avx2::u16x16::u16x16);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
crate::impl_lane_shuffle32!(crate::wide::int::avx2::i8x32::i8x32);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
crate::impl_lane_shuffle32!(crate::wide::int::avx2::u8x32::u8x32);
