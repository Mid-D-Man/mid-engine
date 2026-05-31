// crates/mid-math/src/ran_gen/pcg.rs
//! PCG32 — Permuted Congruential Generator.
//!
//! Better statistical quality than Xorshift. Supports multiple independent
//! streams via the `seq` parameter. Period: 2^64. ~1-2 ns/call.
//!
//! Reference: O'Neill (2014) "PCG: A Family of Simple Fast Space-Efficient
//! Statistically Good Algorithms for Random Number Generation"

use core::fmt;

/// PCG32 generator. 16 bytes.
#[derive(Clone)]
pub struct Pcg32 {
    pub state: u64,
    pub inc:   u64,
}

impl Pcg32 {
    // ── Construction ─────────────────────────────────────────────────────────

    /// Create with explicit `seed` and stream `seq`.
    ///
    /// Different `seq` values produce completely independent streams —
    /// use this to give each subsystem its own RNG without seed management.
    pub fn new(seed: u64, seq: u64) -> Self {
        let inc = (seq << 1) | 1;
        let mut rng = Self { state: 0, inc };
        rng.state = rng.state.wrapping_add(seed);
        rng.next_u32();
        rng
    }

    /// Single-stream convenience constructor.
    #[inline]
    pub fn new_single_stream(seed: u64) -> Self { Self::new(seed, 1) }

    // ── Core generation ───────────────────────────────────────────────────────

    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        let old_state = self.state;
        self.state = old_state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(self.inc);
        let xsh = (((old_state >> 18) ^ old_state) >> 27) as u32;
        let rot = (old_state >> 59) as u32;
        xsh.rotate_right(rot)
    }

    /// Generate a u64 by combining two u32 outputs.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let lo = self.next_u32() as u64;
        let hi = self.next_u32() as u64;
        lo | (hi << 32)
    }

    // ── Float generation ──────────────────────────────────────────────────────

    /// Uniform f32 in `[0, 1)`. Uses top 24 bits of next_u32.
    #[inline]
    pub fn f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 * (1.0 / 16_777_216.0)
    }

    /// Uniform f64 in `[0, 1)`. Uses two u32 calls for 53-bit precision.
    #[inline]
    pub fn f64(&mut self) -> f64 {
        // Combine two 32-bit outputs into 53 significant bits.
        let hi = (self.next_u32() as u64) << 21;
        let lo =  self.next_u32() as u64 >> 11;
        (hi | lo) as f64 * (1.0 / 9_007_199_254_740_992.0)
    }

    // ── Range helpers ─────────────────────────────────────────────────────────

    #[inline]
    pub fn range_f32(&mut self, lo: f32, hi: f32) -> f32 {
        lo + self.f32() * (hi - lo)
    }

    #[inline]
    pub fn range_f64(&mut self, lo: f64, hi: f64) -> f64 {
        lo + self.f64() * (hi - lo)
    }

    /// Unbiased u32 in `[lo, hi)` using Lemire's debiased algorithm.
    pub fn range_u32(&mut self, lo: u32, hi: u32) -> u32 {
        assert!(lo < hi, "Pcg32::range_u32: lo must be < hi");
        let range  = (hi - lo) as u64;
        let mut r  = self.next_u32() as u64 * range;
        if (r as u32) < range as u32 {
            let threshold = range.wrapping_neg() % range;
            while (r as u32) < threshold as u32 {
                r = self.next_u32() as u64 * range;
            }
        }
        lo + (r >> 32) as u32
    }

    #[inline]
    pub fn range_u64(&mut self, lo: u64, hi: u64) -> u64 {
        assert!(lo < hi, "Pcg32::range_u64: lo must be < hi");
        lo + self.next_u64() % (hi - lo)
    }

    #[inline]
    pub fn bool_p(&mut self, p: f32) -> bool {
        self.f32() < p.clamp(0.0, 1.0)
    }

    // ── Unbiased integer distributions ────────────────────────────────────────

    /// Uniform bool — exactly 50/50.
    #[inline]
    pub fn next_bool(&mut self) -> bool {
        self.next_u32() & 1 != 0
    }

    /// Unbiased u32 in `[lo, hi)` — alias for `range_u32` (already uses Lemire).
    #[inline]
    pub fn next_u32_range(&mut self, lo: u32, hi: u32) -> u32 {
        self.range_u32(lo, hi)
    }

    /// Uniform i32 in `[lo, hi)`.
    #[inline]
    pub fn next_i32_range(&mut self, lo: i32, hi: i32) -> i32 {
        assert!(hi > lo, "Pcg32::next_i32_range: lo must be < hi");
        lo + self.next_u32_range(0, (hi - lo) as u32) as i32
    }

    // ── Geometric distributions ───────────────────────────────────────────────

    /// Uniform point ON the unit circle (|v| = 1). Rejection sampling.
    #[inline]
    pub fn next_vec2_unit_circle(&mut self) -> crate::Vec2 {
        loop {
            let x = self.range_f32(-1.0, 1.0);
            let y = self.range_f32(-1.0, 1.0);
            let sq = x * x + y * y;
            if sq > 0.0 && sq <= 1.0 {
                let inv = 1.0 / sq.sqrt();
                return crate::Vec2::new(x * inv, y * inv);
            }
        }
    }

    /// Uniform point IN the unit disk (|v| ≤ 1). Rejection sampling.
    #[inline]
    pub fn next_vec2_in_unit_disk(&mut self) -> crate::Vec2 {
        loop {
            let x = self.range_f32(-1.0, 1.0);
            let y = self.range_f32(-1.0, 1.0);
            if x * x + y * y <= 1.0 {
                return crate::Vec2::new(x, y);
            }
        }
    }

    /// Uniform point ON the unit sphere (|v| = 1). Marsaglia rejection.
    ///
    /// Unbiased — unlike the (θ,φ) approach which clusters at poles.
    /// Average 1.27 iterations per call.
    #[inline]
    pub fn next_vec3_unit_sphere(&mut self) -> crate::Vec3 {
        loop {
            let x = self.range_f32(-1.0, 1.0);
            let y = self.range_f32(-1.0, 1.0);
            let z = self.range_f32(-1.0, 1.0);
            let sq = x * x + y * y + z * z;
            if sq > 0.0 && sq <= 1.0 {
                let inv = 1.0 / sq.sqrt();
                return crate::Vec3::new(x * inv, y * inv, z * inv);
            }
        }
    }

    /// Uniform point IN the unit ball (|v| ≤ 1). Marsaglia rejection.
    #[inline]
    pub fn next_vec3_in_unit_sphere(&mut self) -> crate::Vec3 {
        loop {
            let x = self.range_f32(-1.0, 1.0);
            let y = self.range_f32(-1.0, 1.0);
            let z = self.range_f32(-1.0, 1.0);
            if x * x + y * y + z * z <= 1.0 {
                return crate::Vec3::new(x, y, z);
            }
        }
    }

    /// Uniform point on the hemisphere oriented around `normal`.
    #[inline]
    pub fn next_vec3_in_hemisphere(&mut self, normal: crate::Vec3) -> crate::Vec3 {
        let v = self.next_vec3_unit_sphere();
        if v.dot(normal) < 0.0 { -v } else { v }
    }

    /// Random opaque RGB color. Alpha is always 255.
    #[inline]
    pub fn next_color32_opaque(&mut self) -> crate::Color32 {
        let r = self.next_u32();
        crate::Color32::new(
            (r        & 0xFF) as u8,
            ((r >> 8)  & 0xFF) as u8,
            ((r >> 16) & 0xFF) as u8,
            255,
        )
    }

    /// Random RGBA color including random alpha.
    #[inline]
    pub fn next_color32(&mut self) -> crate::Color32 {
        let r = self.next_u64();
        crate::Color32::new(
            (r        & 0xFF) as u8,
            ((r >> 8)  & 0xFF) as u8,
            ((r >> 16) & 0xFF) as u8,
            ((r >> 24) & 0xFF) as u8,
        )
    }

    // ── State management ─────────────────────────────────────────────────────

    #[inline]
    pub fn state(&self) -> (u64, u64) { (self.state, self.inc) }

    pub fn set_state(&mut self, state: u64, inc: u64) {
        assert!(inc & 1 == 1, "Pcg32: inc must be odd");
        self.state = state;
        self.inc   = inc;
    }

    /// Advance the generator by `delta` steps in O(log n) — no output generated.
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
