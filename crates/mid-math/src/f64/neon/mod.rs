// crates/mid-math/src/f64/neon/mod.rs
//! AArch64 NEON f64 types — float64x2_t on aarch64.
//!
//! `float64x2_t` holds 2× f64 (128 bits, 16-byte aligned).
//! FMA (`vfmaq_f64`) is mandatory on AArch64 — no feature gate needed.
//! `vaddvq_f64` gives a direct single-instruction horizontal add.
//!
//! | Type  | Storage               | Size  | Align | Notes               |
//! |-------|-----------------------|-------|-------|---------------------|
//! | DVec2 | 1× float64x2_t        | 16 B  | 16 B  | Perfect 2-lane fit  |
//! | DVec4 | 2× float64x2_t lo+hi  | 32 B  | 32 B  | lo=[x,y] hi=[z,w]   |
//! | DQuat | 2× float64x2_t lo+hi  | 32 B  | 32 B  | lo=[x,y] hi=[z,w]   |
//!
//! Cross-compile check:
//!   cross test -p mid-math --target aarch64-unknown-linux-gnu --release

pub mod dvec2;
pub mod dvec4;
pub mod dquat;

pub use dvec2::DVec2;
pub use dvec4::DVec4;
pub use dquat::DQuat;
