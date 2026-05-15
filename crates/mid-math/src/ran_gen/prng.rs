// crates/mid-math/src/prng.rs
//! Deterministic pseudo-random number generator — Xorshift64.
//!
//! Algorithm: Xorshift64 (George Marsaglia, 2003).
//!   x ^= x << 13
//!   x ^= x >> 7
//!   x ^= x << 17
//!
//! Properties:
//!   - Period: 2^64 - 1 (all non-zero u64 values visited exactly once)
//!   - Passes BigCrush and Diehard test suites
//!   - 3 XOR + 3 shift = ~1ns per call on modern hardware
//!   - Fully deterministic: same seed → same sequence on ALL platforms
//!   - No heap allocation, no_std compatible
//!   - NOT cryptographically secure — use for simulation only
//!
//! Engine uses: physics jitter, particle variation, AI randomisation,
//! procedural generation, deterministic network replay, bench data generation.

use core::fmt;

/// Xorshift64 pseudo-random number generator.
///
/// Seed must be non-zero — the algorithm never visits state 0.
/// The same seed produces the exact same sequence on all platforms (x86, ARM, WASM).
///
/// # Example
/// ```rust
/// use mid_math::Xorshift64;
///
/// let mut rng = Xorshift64::new(12345);
/// let x: f32 = rng.f32();           // [0, 1)
/// let y: f32 = rng.range_f32(-1.0, 1.0);
/// let n: u32 = rng.range_u32(0, 100);
/// ```
#[derive(Clone)]
pub struct Xorshift64(u64);

impl Xorshift64 {
    /// Create a new RNG with `seed`. Panics if `seed == 0`.
    #[inline]
    pub fn new(seed: u64) -> Self {
        assert!(seed != 0, "Xorshift64: seed must be non-zero");
        Self(seed)
    }

    /// Create from a seed — if seed is 0, uses 1 instead (never panics).
    #[inline]
    pub fn new_safe(seed: u64) -> Self {
        Self(if seed == 0 { 1 } else { seed })
    }

    /// Return the next raw u64.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// Return the next u32 (upper 32 bits of next_u64).
    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }

    /// Uniform f32 in `[0, 1)`.
    ///
    /// Uses the top 24 bits (mantissa of f32) for full precision.
    #[inline]
    pub fn f32(&mut self) -> f32 {
        // Top 24 bits → integer in [0, 2^24); divide by 2^24 → [0, 1)
        (self.next_u64() >> 40) as f32 * (1.0 / 16_777_216.0)
    }

    /// Uniform f64 in `[0, 1)`.
    ///
    /// Uses the top 53 bits (mantissa of f64) for full precision.
    #[inline]
    pub fn f64(&mut self) -> f64 {
        // Top 53 bits → integer in [0, 2^53); divide by 2^53 → [0, 1)
        (self.next_u64() >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
    }

    /// Uniform f32 in `[lo, hi)`.
    #[inline]
    pub fn range_f32(&mut self, lo: f32, hi: f32) -> f32 {
        lo + self.f32() * (hi - lo)
    }

    /// Uniform f64 in `[lo, hi)`.
    #[inline]
    pub fn range_f64(&mut self, lo: f64, hi: f64) -> f64 {
        lo + self.f64() * (hi - lo)
    }

    /// Uniform u32 in `[lo, hi)`. Panics if `lo >= hi`.
    #[inline]
    pub fn range_u32(&mut self, lo: u32, hi: u32) -> u32 {
        assert!(lo < hi, "Xorshift64::range_u32: lo must be < hi");
        lo + (self.next_u64() % (hi - lo) as u64) as u32
    }

    /// Uniform u64 in `[lo, hi)`. Panics if `lo >= hi`.
    #[inline]
    pub fn range_u64(&mut self, lo: u64, hi: u64) -> u64 {
        assert!(lo < hi, "Xorshift64::range_u64: lo must be < hi");
        lo + self.next_u64() % (hi - lo)
    }

    /// True with probability `p` (clamped to `[0, 1]`).
    #[inline]
    pub fn bool_p(&mut self, p: f32) -> bool {
        self.f32() < p.clamp(0.0, 1.0)
    }

    /// Return the current internal state (for serialisation / reproducibility).
    #[inline(always)]
    pub fn state(&self) -> u64 { self.0 }

    /// Restore a previously saved state. Panics if state is 0.
    #[inline]
    pub fn set_state(&mut self, state: u64) {
        assert!(state != 0, "Xorshift64: state must be non-zero");
        self.0 = state;
    }
}

impl fmt::Debug for Xorshift64 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Xorshift64(state={:#018x})", self.0)
    }
}
