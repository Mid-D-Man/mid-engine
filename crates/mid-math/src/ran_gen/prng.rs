// crates/mid-math/src/ran_gen/prng.rs
//! Xorshift64 — deterministic pseudo-random number generator.
//!
//! Algorithm: Xorshift64 (George Marsaglia, 2003).
//! Period: 2^64 - 1.  ~1 ns/call.  NOT cryptographically secure.

use core::fmt;

/// Xorshift64 PRNG. Seed must be non-zero.
#[derive(Clone)]
pub struct Xorshift64(u64);

impl Xorshift64 {
    /// Create a new RNG with `seed`. Panics if `seed == 0`.
    #[inline]
    pub fn new(seed: u64) -> Self {
        assert!(seed != 0, "Xorshift64: seed must be non-zero");
        Self(seed)
    }

    /// Create from a seed — substitutes 1 if seed is 0. Never panics.
    #[inline]
    pub fn new_safe(seed: u64) -> Self {
        Self(if seed == 0 { 1 } else { seed })
    }

    /// Seed from hardware entropy (RDSEED/RDRAND, x86_64 only) if available,
    /// falling back to `fallback_seed` otherwise — see
    /// [`crate::ran_gen::hardware_seed_u64`] for exactly when that happens.
    #[inline]
    pub fn new_from_hardware_entropy_or(fallback_seed: u64) -> Self {
        Self::new_safe(crate::ran_gen::hardware_seed_u64().unwrap_or(fallback_seed))
    }

    /// Seed from hardware entropy. `None` if unavailable (non-x86_64 target,
    /// build without `rdrand`/`rdseed` target features, or both transiently
    /// exhausted their retry budget) — caller decides the fallback.
    #[inline]
    pub fn new_from_hardware_entropy() -> Option<Self> {
        crate::ran_gen::hardware_seed_u64().map(Self::new_safe)
    }

    // ── Core generation ───────────────────────────────────────────────────────

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }

    // ── Float generation ──────────────────────────────────────────────────────

    /// Uniform f32 in `[0, 1)`. Uses top 24 bits.
    #[inline]
    pub fn f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 * (1.0 / 16_777_216.0)
    }

    /// Uniform f64 in `[0, 1)`. Uses top 53 bits.
    #[inline]
    pub fn f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
    }

    // ── Range helpers ─────────────────────────────────────────────────────────

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

    /// Uniform u32 in `[lo, hi)`. Uses modulo — fast, very slight bias for
    /// non-power-of-2 ranges. Use `next_u32_range` for unbiased output.
    #[inline]
    pub fn range_u32(&mut self, lo: u32, hi: u32) -> u32 {
        assert!(lo < hi, "Xorshift64::range_u32: lo must be < hi");
        lo + (self.next_u64() % (hi - lo) as u64) as u32
    }

    /// Uniform u64 in `[lo, hi)`.
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

    // ── Unbiased integer distributions ────────────────────────────────────────

    /// Uniform bool — exactly 50/50, no bias.
    #[inline]
    pub fn next_bool(&mut self) -> bool {
        self.next_u64() & 1 != 0
    }

    /// Unbiased uniform u32 in `[lo, hi)` using Lemire's debiased algorithm.
    ///
    /// Avoids the modulo bias that `range_u32` has for non-power-of-2 ranges.
    /// Slightly slower than `range_u32` due to the rejection loop (rare in practice).
    #[inline]
    pub fn next_u32_range(&mut self, lo: u32, hi: u32) -> u32 {
        assert!(hi > lo, "Xorshift64::next_u32_range: lo must be < hi");
        let range = (hi - lo) as u64;
        // Threshold for rejection: avoids bias in the lower bits.
        let threshold = range.wrapping_neg() % range;
        loop {
            let r = self.next_u32() as u64;
            if r >= threshold {
                return lo + (r % range) as u32;
            }
        }
    }

    /// Uniform i32 in `[lo, hi)`.
    #[inline]
    pub fn next_i32_range(&mut self, lo: i32, hi: i32) -> i32 {
        assert!(hi > lo, "Xorshift64::next_i32_range: lo must be < hi");
        lo + self.next_u32_range(0, (hi - lo) as u32) as i32
    }

    // ── Geometric distributions ───────────────────────────────────────────────

    /// Uniform point ON the unit circle (|v| = 1). Rejection sampling.
    ///
    /// Average 1.27 iterations. Use for random 2D directions.
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
    ///
    /// Unbiased; avoids the clustering-at-centre flaw of angle+radius methods.
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

    /// Uniform point ON the unit sphere (|v| = 1). Marsaglia rejection sampling.
    ///
    /// Unbiased — unlike the (θ,φ) approach which clusters at poles.
    /// Average 1.27 iterations.
    ///
    /// Use for random 3D directions, random normals, ambient occlusion rays.
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
    ///
    /// Use for random offsets within a sphere, explosion/spawn volumes.
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
    ///
    /// Useful for diffuse lighting Monte Carlo integration (cosine-weighted
    /// sampling would be better for that, but this is the uniform version).
    #[inline]
    pub fn next_vec3_in_hemisphere(&mut self, normal: crate::Vec3) -> crate::Vec3 {
        let v = self.next_vec3_unit_sphere();
        // Flip to the same side as normal if needed.
        if v.dot(normal) < 0.0 { -v } else { v }
    }

    /// Random opaque RGB color. Alpha is always 255.
    ///
    /// Uniform across the full 24-bit RGB cube.
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

    #[inline(always)]
    pub fn state(&self) -> u64 { self.0 }

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
