// crates/mid-math/src/pcg.rs
//! PCG32 — Permuted Congruential Generator.
//!
//! PCG is the current gold standard for game PRNGs:
//!   - 1 multiply + 1 XOR-shift + 1 rotate = ~1-2 ns/call
//!   - Better statistical quality than Xorshift (passes PractRand, TestU01)
//!   - Multiple independent streams via the `seq` parameter
//!   - Period: 2^64 per stream, 2^63 streams
//!
//! When to use PCG32 vs Xorshift64:
//!   PCG32:      AI decisions, loot tables, proc-gen, anything that must pass
//!               statistical tests or produce uncorrelated streams
//!   Xorshift64: Performance-critical inner loops where raw speed wins and
//!               correlation between nearby values is acceptable
//!
//! Reference: O'Neill (2014) "PCG: A Family of Simple Fast Space-Efficient
//!            Statistically Good Algorithms for Random Number Generation"

use core::fmt;

/// PCG32 generator. Two u64 state values: `state` (position) and `inc` (stream selector).
///
/// Different `seq` values produce independent, non-overlapping sequences.
/// Same `seed` + `seq` always produces identical output on all platforms.
#[derive(Clone)]
pub struct Pcg32 {
    state: u64,
    inc:   u64,   // Must always be odd. inc = (seq << 1) | 1.
}

impl Pcg32 {
    // ── Construction ─────────────────────────────────────────────────────────

    /// Create a new generator.
    ///
    /// `seed`  — initial state. Any value is valid.
    /// `seq`   — stream selector (0..2^63). Different values = independent streams.
    pub fn new(seed: u64, seq: u64) -> Self {
        let inc = (seq << 1) | 1;
        let mut rng = Self { state: 0, inc };
        // Warmup: advance once before seeding to avoid trivially weak first output
        rng.state = rng.state.wrapping_add(seed);
        rng.next_u32(); // mix seed into state
        rng
    }

    /// Convenience: single-stream generator. Equivalent to `new(seed, 1)`.
    #[inline] pub fn new_single_stream(seed: u64) -> Self { Self::new(seed, 1) }

    // ── Core generation ───────────────────────────────────────────────────────

    /// Generate next u32. PCG-XSH-RR output permutation.
    ///
    /// One multiply + XOR-shift + rotate. ~1 ns/call.
    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        let old_state = self.state;
        // LCG advance
        self.state = old_state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(self.inc);
        // XSH-RR output permutation
        let xsh = (((old_state >> 18) ^ old_state) >> 27) as u32;
        let rot = (old_state >> 59) as u32;
        xsh.rotate_right(rot)
    }

    /// Generate u64 from two u32 outputs.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let lo = self.next_u32() as u64;
        let hi = self.next_u32() as u64;
        lo | (hi << 32)
    }

    // ── Float generation ──────────────────────────────────────────────────────

    /// Uniform f32 in `[0, 1)`. Uses top 24 bits (full f32 mantissa precision).
    #[inline]
    pub fn f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 * (1.0 / 16_777_216.0)
    }

    /// Uniform f64 in `[0, 1)`. Uses two u32s for 53-bit mantissa precision.
    #[inline]
    pub fn f64(&mut self) -> f64 {
        let v = ((self.next_u32() as u64) << 21) | (self.next_u32() as u64 >> 11);
        v as f64 * (1.0 / 9_007_199_254_740_992.0)
    }

    // ── Range functions ───────────────────────────────────────────────────────

    /// Uniform f32 in `[lo, hi)`.
    #[inline] pub fn range_f32(&mut self, lo: f32, hi: f32) -> f32 { lo + self.f32() * (hi - lo) }

    /// Uniform f64 in `[lo, hi)`.
    #[inline] pub fn range_f64(&mut self, lo: f64, hi: f64) -> f64 { lo + self.f64() * (hi - lo) }

    /// Uniform u32 in `[lo, hi)`. Uses Lemire's fast bounded algorithm — no modulo bias.
    ///
    /// Panics if `lo >= hi`.
    pub fn range_u32(&mut self, lo: u32, hi: u32) -> u32 {
        assert!(lo < hi, "Pcg32::range_u32: lo must be < hi");
        let range = (hi - lo) as u64;
        let mut r = self.next_u32() as u64 * range;
        if (r as u32) < range as u32 {
            let threshold = range.wrapping_neg() % range;
            while (r as u32) < threshold as u32 {
                r = self.next_u32() as u64 * range;
            }
        }
        lo + (r >> 32) as u32
    }

    /// Uniform u64 in `[lo, hi)`. Panics if `lo >= hi`.
    #[inline]
    pub fn range_u64(&mut self, lo: u64, hi: u64) -> u64 {
        assert!(lo < hi, "Pcg32::range_u64: lo must be < hi");
        lo + self.next_u64() % (hi - lo)
    }

    /// True with probability `p` (clamped to [0, 1]).
    #[inline] pub fn bool_p(&mut self, p: f32) -> bool { self.f32() < p.clamp(0.0, 1.0) }

    // ── State management ─────────────────────────────────────────────────────

    /// Current state pair for serialisation.
    #[inline] pub fn state(&self) -> (u64, u64) { (self.state, self.inc) }

    /// Restore saved state. `inc` must be odd — panics otherwise.
    pub fn set_state(&mut self, state: u64, inc: u64) {
        assert!(inc & 1 == 1, "Pcg32: inc must be odd");
        self.state = state;
        self.inc   = inc;
    }

    /// Advance the generator by `delta` steps in O(log n). Useful for parallelism.
    pub fn advance(&mut self, delta: u64) {
        let mut acc_mul = 1u64;
        let mut acc_add = 0u64;
        let mut cur_mul = 6_364_136_223_846_793_005u64;
        let mut cur_add = self.inc;
        let mut d = delta;
        while d > 0 {
            if d & 1 != 0 {
                acc_mul = acc_mul.wrapping_mul(cur_mul);
                acc_add = acc_add.wrapping_mul(cur_mul).wrapping_add(cur_add);
            }
            cur_add = cur_mul.wrapping_add(1).wrapping_mul(cur_add);
            cur_mul = cur_mul.wrapping_mul(cur_mul);
            d >>= 1;
        }
        self.state = acc_mul.wrapping_mul(self.state).wrapping_add(acc_add);
    }
}

impl fmt::Debug for Pcg32 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Pcg32(state={:#018x}, inc={:#018x})", self.state, self.inc)
    }
                   }
