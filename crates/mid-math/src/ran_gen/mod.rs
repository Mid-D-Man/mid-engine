// crates/mid-math/src/ran_gen/mod.rs
//! Deterministic pseudo-random number generators.
//!
//! Two generators — pick based on use case:
//!
//! | Type        | Algorithm   | Speed   | Quality  | Use when                              |
//! |-------------|-------------|---------|----------|---------------------------------------|
//! | `Xorshift64`| Xorshift64  | ~1 ns   | Good     | Hot inner loops, particle systems     |
//! | `Pcg32`     | PCG-XSH-RR  | ~1-2 ns | Excellent| AI, loot, proc-gen, multiple streams  |
//!
//! Both are deterministic: same seed → same sequence on all platforms.
//! Neither is cryptographically secure.

pub mod prng;
pub mod pcg;

pub use prng::Xorshift64;
pub use pcg::Pcg32;
