// crates/mid-math/src/wide/int/mod.rs
//! Integer wide types — platform dispatch.
//!
//! ## Platform matrix
//!
//! | Backend | Target                        | Types                                                    |
//! |---------|-------------------------------|-----------------------------------------------------------|
//! | SSE2    | x86 / x86_64                  | i32x4, u32x4, i16x8, u16x8, i8x16, u8x16, IMask4/8/16     |
//! | NEON    | aarch64                       | i32x4, u32x4, i16x8, u16x8, i8x16, u8x16, IMask4/8/16     |
//! | AVX2    | x86 / x86_64                  | i32x8, u32x8, i16x16, u16x16, i8x32, u8x32, IMask32x8/16x16/8x32 (additional) |
//! | WASM    | wasm32/64 + simd128 feature   | i32x4, u32x4, i16x8, u16x8, i8x16, u8x16, IMask4/8/16     |
//! | Scalar  | all others                    | i32x4, u32x4, i16x8, u16x8, i8x16, u8x16, IMask4/8/16     |
//!
//! Mirrors `wide/float/mod.rs`'s dispatch shape.
//!
//! AVX2's row changed from "x86/x86_64 + avx2 feature" to plain
//! "x86/x86_64": these 6 types are always compiled now, storage is two
//! portable halves of the matching always-available width (e.g. `i32x8`
//! is two `i32x4`), and each arithmetic method checks
//! `crate::wide::avx2_available()` at runtime, calling a
//! `#[target_feature(enable = "avx2")]`-gated fast path when true and the
//! portable halves' own methods otherwise — see `wide/int/avx2/i32x8.rs`'s
//! doc comment for the full reasoning (this was previously a compile-time
//! `target_feature = "avx2"` gate on the whole module, which meant a C
//! caller linking a non-AVX2 build would get a link error referencing
//! these types at all, not just a slower runtime path).
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
//! the per-op advantage table and known omissions/fixes.
//!
//! WASM (same widths as SSE2/NEON) is wired in this pass — see
//! `wide/int/wasm/mod.rs` for the per-op WASM-vs-SSE2 comparison. The
//! scalar fallback's exclusion list needed the SAME fix NEON needed:
//! narrowed to also exclude `wasm32`/`wasm64` — but conditionally, only
//! when `target_feature = "simd128"` is actually enabled, since a wasm
//! target *without* simd128 still needs scalar as a real fallback (this
//! is why it's `all(wasm-arch, simd128)` inside the `not(any(...))`
//! below, not a bare arch check like aarch64's).

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

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub(crate) mod avx2;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub use avx2::{IMask32x8, IMask16x16, IMask8x32};

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(non_camel_case_types)]
pub use avx2::{
    i32x8::i32x8, u32x8::u32x8,
    i16x16::i16x16, u16x16::u16x16,
    i8x32::i8x32, u8x32::u8x32,
};

// ── WASM SIMD128 — wasm32 / wasm64 + simd128 ──────────────────────────────────
//
// Same widths as SSE2/NEON above — see wide/int/wasm/mod.rs for the
// per-op WASM-vs-SSE2 comparison table and the (SSE2-matching, hardware-
// inherent) gaps: no i32/u32 saturating add/sub, no i8/u8 multiply.

#[cfg(all(
    any(target_arch = "wasm32", target_arch = "wasm64"),
    target_feature = "simd128",
))]
pub(crate) mod wasm;

#[cfg(all(
    any(target_arch = "wasm32", target_arch = "wasm64"),
    target_feature = "simd128",
    not(feature = "force-scalar"),
))]
pub use wasm::{IMask4, IMask8, IMask16};

#[cfg(all(
    any(target_arch = "wasm32", target_arch = "wasm64"),
    target_feature = "simd128",
    not(feature = "force-scalar"),
))]
#[allow(non_camel_case_types)]
pub use wasm::{
    i32x4::i32x4, u32x4::u32x4,
    i16x8::i16x8, u16x8::u16x8,
    i8x16::i8x16, u8x16::u8x16,
};

// ── Scalar fallback ───────────────────────────────────────────────────────────
//
// Active when no SIMD backend applies. Narrowed this pass to also
// exclude aarch64 now that NEON is wired above — previously this only
// excluded x86/x86_64, so scalar and neon would both have matched on
// aarch64 simultaneously (a duplicate-definition compile error) the
// moment the neon mod/pub-use pair above was added.

// Scalar fallback — narrowed each time a new backend gets wired in
// above (aarch64 excluded when NEON landed; wasm32/wasm64+simd128
// excluded now that WASM has too). The wasm exclusion is conditional
// on simd128 specifically, not the bare arch — a wasm32 target WITHOUT
// simd128 enabled still needs to fall through to scalar correctly.

#[cfg(any(
    feature = "force-scalar",
    not(any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "aarch64",
        all(any(target_arch = "wasm32", target_arch = "wasm64"), target_feature = "simd128"),
    )),
))]
pub use scalar::{IMask4, IMask8, IMask16};

#[cfg(any(
    feature = "force-scalar",
    not(any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "aarch64",
        all(any(target_arch = "wasm32", target_arch = "wasm64"), target_feature = "simd128"),
    )),
))]
#[allow(non_camel_case_types)]
pub use scalar::{
    i32x4::i32x4, u32x4::u32x4,
    i16x8::i16x8, u16x8::u16x8,
    i8x16::i8x16, u8x16::u8x16,
};
