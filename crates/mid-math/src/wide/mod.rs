// crates/mid-math/src/wide/mod.rs
//! Wide SIMD types — vertical operations on N values simultaneously.

pub mod float;
pub mod int;

// ── Runtime AVX2 detection (shared by wide::int::avx2 and wide::float::avx2) ──
//
// The AVX2-only wide types (i32x8 and friends, Vec3x8) are always compiled
// on x86/x86_64 now, not gated on the crate's own `avx2` target-feature
// baseline -- see wide/int/avx2/i32x8.rs's doc comment for the full
// reasoning. Every one of their arithmetic methods checks this once per
// call (cached after the first check) before deciding between a real AVX2
// fast path and a portable fallback built from two width-4/8/16 halves.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
pub(crate) fn avx2_available() -> bool {
    use std::sync::OnceLock;
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| std::is_x86_feature_detected!("avx2"))
}

// ── Integer wide re-exports ───────────────────────────────────────────────────

pub use int::{IMask4, IMask8, IMask16};

#[allow(non_camel_case_types)]
pub use int::{i32x4, u32x4, i16x8, u16x8, i8x16, u8x16};

// ── Float wide re-exports ─────────────────────────────────────────────────────

pub use float::{Mask4, Mask4LaneIter};

#[allow(non_camel_case_types)]
pub use float::f32x4;

pub use float::Vec3x4;
pub use float::QuatX4;

// Vec3x8 is only available when targeting AVX2
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    target_feature = "avx2",
))]
pub use float::Vec3x8;
