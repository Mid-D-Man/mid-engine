// crates/mid-math/src/wide/int/mod.rs
//! Integer wide types — platform dispatch.
//!
//! ## Platform matrix
//!
//! | Backend | Target                        | Types                                                    |
//! |---------|-------------------------------|-----------------------------------------------------------|
//! | SSE2    | x86 / x86_64                  | i32x4, u32x4, i16x8, u16x8, i8x16, u8x16, IMask4/8/16     |
//! | NEON    | aarch64                       | i32x4, u32x4, i16x8, u16x8, i8x16, u8x16, IMask4/8/16     |
//! | AVX2    | x86 / x86_64 + avx2 feature   | i32x8, u32x8, i16x16, u16x16, i8x32, u8x32, IMask32x8/16x16/8x32 (additional) |
//! | WASM    | wasm32/64 + simd128 feature   | i32x4, u32x4, i16x8, u16x8, i8x16, u8x16 (next pass)      |
//! | Scalar  | all others                    | i32x4, u32x4, i16x8, u16x8, i8x16, u8x16, IMask4/8/16     |
//!
//! Mirrors `wide/float/mod.rs`'s dispatch shape.
//!
//! ## Fixes / additions this pass
//!
//! `wide/int/neon/` was fully implemented — i32x4, u32x4, i16x8, u16x8,
//! i8x16, u8x16, IMask4/8/16, 1747 lines, file-for-file parity with
//! `wide/int/sse2/` — but was never declared/dispatched below. aarch64
//! targets were silently falling back to scalar despite the NEON code
//! sitting right there compiled-and-ready. Fixed by adding the same
//! `#[cfg(target_arch = "aarch64")]` mod declaration + `pub use` pair
//! that `wide/float/mod.rs` already uses for its NEON branch, and
//! narrowing the scalar fallback's `not(any(...))` to also exclude
//! aarch64 (previously only excluded x86/x86_64, so scalar and — once
//! wired — neon would have both matched on aarch64 simultaneously).
//!
//! AVX2 adds i32x8/u32x8/i16x16/u16x16/i8x32/u8x32 (additive like
//! Vec3x8, not gated by force-scalar) — see `wide/int/avx2/mod.rs` for
//! the per-op advantage table and this pass's known omissions.
//!
//! WASM (same widths as SSE2/NEON, mirrors `wide/float/wasm/`) is
//! landing in a follow-up pass — see docs/platform-optimization.md §9.

// ── Scalar fallback — always compiled ────────────────────────────────────────
pub(crate) mod scalar;

// ── SSE2 — x86 / x86_64 ──────────────────────────────────────────────────────

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub(crate) mod sse2;

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), not(feature = "force-scalar")))]
pub use sse2::{IMask4, IMask8, IMask16};

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), not(feature = "force-scalar")))]
#[allow(non_camel_case_types)]
pub use sse2::{
    i32x4::i32x4, u32x4::u32x4,
    i16x8::i16x8, u16x8::u16x8,
    i8x16::i8x16, u8x16::u8x16,
};

// ── NEON — aarch64 ────────────────────────────────────────────────────────────
//
// int32x4_t/int16x8_t/int8x16_t are mandatory on all AArch64 targets —
// no runtime check needed, same reasoning wide/float/mod.rs uses for its
// NEON branch. See wide/int/neon/mod.rs's own doc comment for the
// per-op NEON-vs-SSE2 advantage table (horizontal min/max, saturating
// add/sub i32, abs, blend, neg all drop to a single instruction on NEON
// where SSE2 needs a shuffle/compare chain).

#[cfg(target_arch = "aarch64")]
pub(crate) mod neon;

#[cfg(all(target_arch = "aarch64", not(feature = "force-scalar")))]
pub use neon::{IMask4, IMask8, IMask16};

#[cfg(all(target_arch = "aarch64", not(feature = "force-scalar")))]
#[allow(non_camel_case_types)]
pub use neon::{
    i32x4::i32x4, u32x4::u32x4,
    i16x8::i16x8, u16x8::u16x8,
    i8x16::i8x16, u8x16::u8x16,
};

// ── AVX2 — x86 / x86_64 + avx2 ────────────────────────────────────────────────
//
// Adds WIDER types (i32x8/u32x8/i16x16/u16x16/i8x32/u8x32 via __m256i) —
// additive alongside the SSE2/NEON i32x4-family above, not a replacement,
// same relationship AVX2's Vec3x8 has to Vec3x4 in wide/float/mod.rs. Not
// gated by force-scalar, same as Vec3x8. See wide/int/avx2/mod.rs for the
// per-op AVX2-vs-SSE2 advantage table and this pass's known omissions
// (shuffle_bytes / cross-lane widen, deferred pending a compile check).

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    target_feature = "avx2",
))]
pub(crate) mod avx2;

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    target_feature = "avx2",
))]
pub use avx2::{IMask32x8, IMask16x16, IMask8x32};

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    target_feature = "avx2",
))]
#[allow(non_camel_case_types)]
pub use avx2::{
    i32x8::i32x8, u32x8::u32x8,
    i16x16::i16x16, u16x16::u16x16,
    i8x32::i8x32, u8x32::u8x32,
};

// ── WASM SIMD128 (next pass) ───────────────────────────────────────────────────
//
// Same i32x4/u32x4/i16x8/u16x8/i8x16/u8x16 widths as SSE2/NEON above,
// mirrors wide/float/wasm/'s approach.

// ── Scalar fallback ───────────────────────────────────────────────────────────
//
// Active when no SIMD backend applies. Narrowed this pass to also
// exclude aarch64 now that NEON is wired above — previously this only
// excluded x86/x86_64, so scalar and neon would both have matched on
// aarch64 simultaneously (a duplicate-definition compile error) the
// moment the neon mod/pub-use pair above was added.

#[cfg(any(
    feature = "force-scalar",
    not(any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "aarch64",
    )),
))]
pub use scalar::{IMask4, IMask8, IMask16};

#[cfg(any(
    feature = "force-scalar",
    not(any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "aarch64",
    )),
))]
#[allow(non_camel_case_types)]
pub use scalar::{
    i32x4::i32x4, u32x4::u32x4,
    i16x8::i16x8, u16x8::u16x8,
    i8x16::i8x16, u8x16::u8x16,
};
