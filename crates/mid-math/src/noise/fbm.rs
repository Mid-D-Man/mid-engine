// crates/mid-math/src/noise/fbm.rs
//! Fractional Brownian Motion (fBm) and Domain Warping.
//!
//! fBm layers multiple octaves of a base noise type, halving amplitude and
//! doubling frequency at each octave. This produces natural-looking fractal
//! detail at all scales — the standard technique for:
//!   - Terrain heightmaps
//!   - Cloud / fog volumes
//!   - Procedural textures
//!
//! Domain Warping feeds the output of one fBm into another as a coordinate
//! offset, producing highly organic, swirling patterns. Ingo Quilez (2002).
//!
//! # Example — terrain heightmap
//! ```rust
//! use mid_math::noise::{Simplex, Fbm};
//!
//! let fbm = Fbm::new(Simplex::new())
//!     .octaves(8)
//!     .lacunarity(2.0)
//!     .gain(0.5)
//!     .frequency(1.0);
//!
//! let height = fbm.sample_2d(x * 0.01, z * 0.01);
//! ```

use super::{NoiseSource2d, NoiseSource3d};

// ── fBm ───────────────────────────────────────────────────────────────────────

/// Fractional Brownian Motion — layered octaves of base noise.
///
/// Generic over any type implementing `NoiseSource2d` and/or `NoiseSource3d`.
#[derive(Clone)]
pub struct Fbm<N> {
    source:     N,
    /// Number of octaves to sum. More = finer detail, higher cost.
    octaves:    u32,
    /// Frequency multiplier per octave (typically 2.0).
    lacunarity: f32,
    /// Amplitude multiplier per octave (typically 0.5 = persistence).
    gain:       f32,
    /// Starting frequency scale.
    frequency:  f32,
    /// Starting amplitude.
    amplitude:  f32,
}

impl<N> Fbm<N> {
    pub fn new(source: N) -> Self {
        Self {
            source,
            octaves:    6,
            lacunarity: 2.0,
            gain:       0.5,
            frequency:  1.0,
            amplitude:  1.0,
        }
    }

    pub fn octaves(mut self, v: u32)    -> Self { self.octaves    = v; self }
    pub fn lacunarity(mut self, v: f32) -> Self { self.lacunarity = v; self }
    /// Set `gain` (persistence). Values < 0.5 smooth out, > 0.5 amplify detail.
    pub fn gain(mut self, v: f32)       -> Self { self.gain       = v; self }
    pub fn frequency(mut self, v: f32)  -> Self { self.frequency  = v; self }
    pub fn amplitude(mut self, v: f32)  -> Self { self.amplitude  = v; self }

    /// Max possible absolute value — useful for normalising the output.
    pub fn max_value(&self) -> f32 {
        let mut a = self.amplitude;
        let mut total = 0.0;
        for _ in 0..self.octaves {
            total += a;
            a *= self.gain;
        }
        total
    }
}

impl<N: NoiseSource2d> Fbm<N> {
    /// Sample 2D fBm. Output ≈ `[-1, 1]`.
    pub fn sample_2d(&self, x: f32, y: f32) -> f32 {
        let mut value = 0.0f32;
        let mut freq  = self.frequency;
        let mut amp   = self.amplitude;
        let mut max   = 0.0f32;

        for _ in 0..self.octaves {
            value += self.source.sample_2d(x * freq, y * freq) * amp;
            max   += amp;
            freq  *= self.lacunarity;
            amp   *= self.gain;
        }

        // Normalise so the output stays in [-1, 1]
        if max > 0.0 { value / max } else { 0.0 }
    }
}

impl<N: NoiseSource3d> Fbm<N> {
    /// Sample 3D fBm. Output ≈ `[-1, 1]`.
    pub fn sample_3d(&self, x: f32, y: f32, z: f32) -> f32 {
        let mut value = 0.0f32;
        let mut freq  = self.frequency;
        let mut amp   = self.amplitude;
        let mut max   = 0.0f32;

        for _ in 0..self.octaves {
            value += self.source.sample_3d(x * freq, y * freq, z * freq) * amp;
            max   += amp;
            freq  *= self.lacunarity;
            amp   *= self.gain;
        }

        if max > 0.0 { value / max } else { 0.0 }
    }
}

impl<N: NoiseSource2d> NoiseSource2d for Fbm<N> {
    fn sample_2d(&self, x: f32, y: f32) -> f32 { self.sample_2d(x, y) }
}
impl<N: NoiseSource3d> NoiseSource3d for Fbm<N> {
    fn sample_3d(&self, x: f32, y: f32, z: f32) -> f32 { self.sample_3d(x, y, z) }
}

// ── Domain Warp ───────────────────────────────────────────────────────────────

/// Domain Warping — feeds fBm into its own coordinate space.
///
/// At each sample, the coordinate is offset by scaled fBm values before
/// the final noise lookup. Produces highly organic, swirling patterns.
///
/// Two passes of warping are available: first-order (`warp`) and second-order
/// (`warp2`). Each pass increases visual complexity at the cost of 2 extra
/// noise evaluations per dimension.
///
/// # Ingo Quilez pattern
/// ```text
/// p' = p + warp_scale * fbm(p)
/// result = fbm(p')          // first-order warp
///
/// p'' = p' + warp_scale * fbm(p')
/// result = fbm(p'')         // second-order warp
/// ```
#[derive(Clone)]
pub struct DomainWarp<N> {
    fbm:        Fbm<N>,
    /// How far to displace the coordinates (in noise units). Typical: 0.5–2.0.
    warp_scale: f32,
    /// Apply two rounds of warping for more complex output.
    double_warp: bool,
}

impl<N: Clone> DomainWarp<N> {
    pub fn new(source: N) -> Self {
        Self {
            fbm: Fbm::new(source),
            warp_scale:  1.0,
            double_warp: false,
        }
    }

    pub fn with_fbm(mut self, fbm: Fbm<N>) -> Self { self.fbm = fbm; self }

    /// Scale of the coordinate displacement. Larger = more distortion.
    pub fn warp_scale(mut self, v: f32) -> Self { self.warp_scale = v; self }

    /// Enable second-order warping for richer, more complex patterns.
    pub fn double_warp(mut self, v: bool) -> Self { self.double_warp = v; self }
}

impl<N: NoiseSource2d + Clone> DomainWarp<N> {
    /// Sample 2D domain-warped noise. Output ≈ `[-1, 1]`.
    pub fn sample_2d(&self, x: f32, y: f32) -> f32 {
        // First warp: offset coordinates by scaled noise.
        let ox1 = self.fbm.sample_2d(x + 0.0,   y + 0.0);
        let oy1 = self.fbm.sample_2d(x + 5.2,   y + 1.3);  // phase offset

        let wx1 = x + self.warp_scale * ox1;
        let wy1 = y + self.warp_scale * oy1;

        if !self.double_warp {
            return self.fbm.sample_2d(wx1, wy1);
        }

        // Second warp: warp again through the already-warped coordinates.
        let ox2 = self.fbm.sample_2d(wx1 + 1.7,  wy1 + 9.2);
        let oy2 = self.fbm.sample_2d(wx1 + 8.3,  wy1 + 2.8);

        self.fbm.sample_2d(
            x + self.warp_scale * ox2,
            y + self.warp_scale * oy2,
        )
    }
}

impl<N: NoiseSource3d + Clone> DomainWarp<N> {
    /// Sample 3D domain-warped noise. Output ≈ `[-1, 1]`.
    pub fn sample_3d(&self, x: f32, y: f32, z: f32) -> f32 {
        let ox1 = self.fbm.sample_3d(x + 0.0, y + 0.0, z + 0.0);
        let oy1 = self.fbm.sample_3d(x + 5.2, y + 1.3, z + 2.8);
        let oz1 = self.fbm.sample_3d(x + 1.7, y + 9.2, z + 4.1);

        let wx1 = x + self.warp_scale * ox1;
        let wy1 = y + self.warp_scale * oy1;
        let wz1 = z + self.warp_scale * oz1;

        if !self.double_warp {
            return self.fbm.sample_3d(wx1, wy1, wz1);
        }

        let ox2 = self.fbm.sample_3d(wx1 + 8.3, wy1 + 2.8, wz1 + 0.4);
        let oy2 = self.fbm.sample_3d(wx1 + 3.1, wy1 + 6.7, wz1 + 1.9);
        let oz2 = self.fbm.sample_3d(wx1 + 7.4, wy1 + 0.1, wz1 + 5.5);

        self.fbm.sample_3d(
            x + self.warp_scale * ox2,
            y + self.warp_scale * oy2,
            z + self.warp_scale * oz2,
        )
    }
}

impl<N: NoiseSource2d + Clone> NoiseSource2d for DomainWarp<N> {
    fn sample_2d(&self, x: f32, y: f32) -> f32 { self.sample_2d(x, y) }
}
impl<N: NoiseSource3d + Clone> NoiseSource3d for DomainWarp<N> {
    fn sample_3d(&self, x: f32, y: f32, z: f32) -> f32 { self.sample_3d(x, y, z) }
}
