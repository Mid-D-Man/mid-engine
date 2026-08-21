// crates/mid-math/src/wide/int/avx2/mod.rs
//! AVX2-backed integer wide types — x86 / x86_64 with `target_feature = "avx2"`.
//!
//! Additive alongside the SSE2/NEON i32x4-family in `wide/int/mod.rs` —
//! same relationship AVX2's `Vec3x8` has to `Vec3x4` in `wide/float/`.
//! Not a replacement for i32x4 etc.; these are new, wider types.
//!
//! ## Register mapping
//!
//! | Rust type | AVX2 register | Mask type    |
//! |-----------|----------------|--------------|
//! | i32x8     | `__m256i`      | IMask32x8    |
//! | u32x8     | `__m256i`      | IMask32x8    |
//! | i16x16    | `__m256i`      | IMask16x16   |
//! | u16x16    | `__m256i`      | IMask16x16   |
//! | i8x32     | `__m256i`      | IMask8x32    |
//! | u8x32     | `__m256i`      | IMask8x32    |
//!
//! ## AVX2 advantages over SSE2 (native ops SSE2 has to emulate)
//!
//! | Operation                | SSE2                          | AVX2                    |
//! |---------------------------|--------------------------------|--------------------------|
//! | i32 min/max               | cmplt/cmpgt + blend (3+ instr) | `_mm256_min/max_epi32` (1) |
//! | u32 min/max               | XOR-flip cmpgt + blend         | `_mm256_min/max_epu32` (1) |
//! | u16 min/max               | XOR-flip cmpgt + blend         | `_mm256_min/max_epu16` (1) |
//! | i32 multiply              | shuffle/unpack chain (5+ instr)| `_mm256_mullo_epi32` (1)   |
//! | i8/i16/i32 abs            | cmplt + sub + blend (3 instr)  | `_mm256_abs_epi8/16/32` (1)|
//!
//! `cmpgt` for unsigned lanes still needs the XOR-sign-flip trick at every
//! width — AVX2 adds no unsigned compare instruction.
//!
//! ## Known omissions (this pass)
//!
//! `_mm256_shuffle_epi8`/`_mm256_unpacklo_epi8`/`_mm256_unpackhi_epi8` (and
//! their 16-bit equivalents) operate PER 128-BIT LANE, not across the full
//! 256 bits. Porting sse2/i8x16.rs's `shuffle_bytes`/`as_i16x8_lo`/`as_i16x8_hi`
//! (and u8x16's/i16x8's equivalents) naively would silently produce wrong
//! results — indices/splits would be scoped to their own 16-byte half
//! instead of the full 32/16-lane width. Each new file's header documents
//! this specifically; fixing it needs an explicit
//! `_mm256_permute4x64_epi64`/`_mm256_permute2x128_si256` lane-fixup that
//! wasn't worth shipping unverified without a local `cargo check`.
//!
//! Cross-compilation check:
//!   RUSTFLAGS="-C target-feature=+avx2" cargo test -p mid-math --release
//!   (or --target with an explicit avx2-capable -C target-cpu)

pub mod imask32x8;
pub mod imask16x16;
pub mod imask8x32;
pub mod i32x8;
pub mod u32x8;
pub mod i16x16;
pub mod u16x16;
pub mod i8x32;
pub mod u8x32;

pub use imask32x8::IMask32x8;
pub use imask16x16::IMask16x16;
pub use imask8x32::IMask8x32;
// Lowercase types resolved by parent via full sub-module path: avx2::i32x8::i32x8 etc.
