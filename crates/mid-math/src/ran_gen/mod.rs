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
//!
//! Both have `new_from_hardware_entropy[_or]` constructors that seed from
//! RDSEED/RDRAND (x86_64 only, see [`hw_entropy`]) instead of a seed you
//! provide yourself — useful for "different sequence every run" without
//! reaching for a system clock or an external crate.

pub mod prng;
pub mod pcg;
pub mod hw_entropy;

pub use prng::Xorshift64;
pub use pcg::Pcg32;
pub use hw_entropy::hardware_seed_u64;
