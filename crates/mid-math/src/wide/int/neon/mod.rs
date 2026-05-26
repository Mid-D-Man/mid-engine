// crates/mid-math/src/wide/int/neon/mod.rs
//! NEON-backed integer wide types — aarch64.
//!
//! ## Register mapping
//!
//! | Rust type  | NEON register  | Mask type    | NEON mask      |
//! |------------|----------------|--------------|----------------|
//! | i32x4      | int32x4_t      | IMask4       | uint32x4_t     |
//! | u32x4      | uint32x4_t     | IMask4       | uint32x4_t     |
//! | i16x8      | int16x8_t      | IMask8       | uint16x8_t     |
//! | u16x8      | uint16x8_t     | IMask8       | uint16x8_t     |
//! | i8x16      | int8x16_t      | IMask16      | uint8x16_t     |
//! | u8x16      | uint8x16_t     | IMask16      | uint8x16_t     |
//!
//! ## NEON integer advantages over SSE2
//!
//! | Operation              | SSE2                         | NEON                      |
//! |------------------------|------------------------------|---------------------------|
//! | Horizontal min/max     | shuffle chain (4+ instr)     | vminvq_s32 (1 instr)      |
//! | Horizontal sum         | shuffle + add chain          | vaddvq_s32 (1 instr)      |
//! | Saturating add i32     | scalar (no SSE2 sat i32)     | vqaddq_s32 (1 instr)      |
//! | Saturating sub i32     | scalar fallback              | vqsubq_s32 (1 instr)      |
//! | Abs (signed)           | cmplt + blend (3 instr)      | vabsq_s32 (1 instr)       |
//! | Blend/select           | AND + ANDNOT + OR (3 instr)  | vbslq_s32 (1 instr)       |
//! | Neg (signed)           | sub from zero                | vnegq_s32 (1 instr)       |
//! | Variable shift         | _mm_sll + cvtsi (2 instr)    | vshlq_s32 with vdupq (2)  |
//!
//! Cross-compilation check:
//!   cross test -p mid-math --target aarch64-unknown-linux-gnu --release

pub mod imask4;
pub mod imask8;
pub mod imask16;
pub mod i32x4;
pub mod u32x4;
pub mod i16x8;
pub mod u16x8;
pub mod i8x16;
pub mod u8x16;

pub use imask4::IMask4;
pub use imask8::IMask8;
pub use imask16::IMask16;
// Lowercase types resolved by parent via full sub-module path: neon::i32x4::i32x4 etc.
