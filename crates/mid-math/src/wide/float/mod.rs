// crates/mid-math/src/wide/float/mod.rs
pub(crate) mod scalar;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub(crate) mod sse2;

// ── AVX2-gated 8-wide float types ─────────────────────────────────────────────
// Only compiled when the compiler sees target_feature = "avx2" (i.e. the user
// passed RUSTFLAGS="-C target-feature=+avx2" or equivalent).
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    target_feature = "avx2",
))]
pub(crate) mod avx2;

// ── Platform dispatch — SSE2 ──────────────────────────────────────────────────
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub use sse2::{Mask4, f32x4, Vec3x4, QuatX4};

// ── Platform dispatch — scalar fallback ───────────────────────────────────────
#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
pub use scalar::{Mask4, f32x4, Vec3x4, QuatX4};

// ── AVX2 8-wide types — conditionally exported ────────────────────────────────
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    target_feature = "avx2",
))]
pub use avx2::Vec3x8;
