// crates/mid-math/src/noise/worley.rs
//! Worley / Cellular noise — Steven Worley (1996).
//!
//! Divides space into random cells (Voronoi diagram) and returns the
//! distance to the nearest feature point. Common visual uses:
//!   - `F1`        : cracked stone, skin cells, water surface
//!   - `F2 - F1`   : foam bubbles, soap film, organic membranes
//!   - `F1 + F2`   : leather, rocky surfaces
//!
//! All generators tile properly and are seed-controllable.
//! Output is normalised to ≈ `[0, 1]` by default; use `DistanceMode` to
//! control which distance combination is returned.

use super::build_perm;
use super::{NoiseSource2d, NoiseSource3d};

/// Which distance combination the Worley sampler returns.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DistanceMode {
    /// Nearest feature point. Gives cell / cracked-rock look. Output ≈ [0, 1].
    F1,
    /// Second-nearest feature point. Output ≈ [0, 1].
    F2,
    /// `F2 − F1`. Gives bubble / foam look. Output ≈ [0, 1].
    F2MinusF1,
    /// `F1 + F2`. Gives leather / rocky look. Output ≈ [0, 1].
    F1PlusF2,
}

/// Distance metric for feature-point distances.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DistanceMetric {
    /// Standard Euclidean distance. Smooth, rounded cells.
    Euclidean,
    /// Manhattan (L1) distance. Diamond-shaped cells.
    Manhattan,
    /// Chebyshev (L∞) distance. Square cells.
    Chebyshev,
    /// Minkowski with p = 0.5 — more irregular, spiky cells.
    Minkowski,
}

/// Worley / Cellular noise generator.
#[derive(Clone)]
pub struct Worley {
    perm:   [u8; 512],
    mode:   DistanceMode,
    metric: DistanceMetric,
    /// Maximum distance expected — used for normalisation. Default = 1.0.
    scale:  f32,
}

impl Worley {
    pub fn new() -> Self {
        Self { perm: build_perm(None), mode: DistanceMode::F1,
               metric: DistanceMetric::Euclidean, scale: 1.0 }
    }
    pub fn from_seed(seed: u64) -> Self {
        Self { perm: build_perm(Some(seed)), mode: DistanceMode::F1,
               metric: DistanceMetric::Euclidean, scale: 1.0 }
    }
    pub fn with_mode(mut self, mode: DistanceMode) -> Self { self.mode = mode; self }
    pub fn with_metric(mut self, metric: DistanceMetric) -> Self { self.metric = metric; self }

    // ── Internals ─────────────────────────────────────────────────────────────

    #[inline(always)]
    fn p(&self, i: usize) -> usize { self.perm[i & 511] as usize }

    /// Pseudo-random feature point offset ∈ [0, 1) for cell (cx, cy).
    #[inline]
    fn feature_2d(&self, cx: i32, cy: i32) -> (f32, f32) {
        let h1 = self.p(self.p((cx.wrapping_add(1024)) as usize & 255) +
                              (cy.wrapping_add(1024)) as usize & 255);
        let h2 = self.p(self.p(h1 + (cx.wrapping_mul(7) as usize & 255)) + 37);
        (h1 as f32 / 255.0, h2 as f32 / 255.0)
    }

    #[inline]
    fn feature_3d(&self, cx: i32, cy: i32, cz: i32) -> (f32, f32, f32) {
        let h1 = self.p(self.p(self.p(
            (cx.wrapping_add(1024)) as usize & 255) +
            (cy.wrapping_add(1024)) as usize & 255) +
            (cz.wrapping_add(1024)) as usize & 255);
        let h2 = self.p(self.p(h1 + 13) + 7);
        let h3 = self.p(self.p(h2 + 23) + 41);
        (h1 as f32 / 255.0, h2 as f32 / 255.0, h3 as f32 / 255.0)
    }

    #[inline]
    fn dist2(&self, dx: f32, dy: f32) -> f32 {
        match self.metric {
            DistanceMetric::Euclidean  => (dx*dx + dy*dy).sqrt(),
            DistanceMetric::Manhattan  => dx.abs() + dy.abs(),
            DistanceMetric::Chebyshev  => dx.abs().max(dy.abs()),
            DistanceMetric::Minkowski  => {
                let s = dx.abs().sqrt() + dy.abs().sqrt();
                s * s
            }
        }
    }

    #[inline]
    fn dist3(&self, dx: f32, dy: f32, dz: f32) -> f32 {
        match self.metric {
            DistanceMetric::Euclidean  => (dx*dx + dy*dy + dz*dz).sqrt(),
            DistanceMetric::Manhattan  => dx.abs() + dy.abs() + dz.abs(),
            DistanceMetric::Chebyshev  => dx.abs().max(dy.abs()).max(dz.abs()),
            DistanceMetric::Minkowski  => {
                let s = dx.abs().sqrt() + dy.abs().sqrt() + dz.abs().sqrt();
                s * s
            }
        }
    }

    fn combine(&self, f1: f32, f2: f32) -> f32 {
        let v = match self.mode {
            DistanceMode::F1        => f1,
            DistanceMode::F2        => f2,
            DistanceMode::F2MinusF1 => f2 - f1,
            DistanceMode::F1PlusF2  => f1 + f2,
        };
        (v / self.scale).clamp(0.0, 1.0)
    }

    // ── Public API ────────────────────────────────────────────────────────────

    /// Sample 2D Worley noise. Output ≈ `[0, 1]`.
    pub fn sample_2d(&self, x: f32, y: f32) -> f32 {
        let cx = x.floor() as i32;
        let cy = y.floor() as i32;
        let mut f1 = f32::MAX;
        let mut f2 = f32::MAX;

        for dy in -2i32..=2 {
            for dx in -2i32..=2 {
                let (fx, fy) = self.feature_2d(cx + dx, cy + dy);
                let px = (cx + dx) as f32 + fx;
                let py = (cy + dy) as f32 + fy;
                let d  = self.dist2(x - px, y - py);
                if d < f1 { f2 = f1; f1 = d; }
                else if d < f2 { f2 = d; }
            }
        }
        self.combine(f1, f2)
    }

    /// Sample 3D Worley noise. Output ≈ `[0, 1]`.
    pub fn sample_3d(&self, x: f32, y: f32, z: f32) -> f32 {
        let cx = x.floor() as i32;
        let cy = y.floor() as i32;
        let cz = z.floor() as i32;
        let mut f1 = f32::MAX;
        let mut f2 = f32::MAX;

        for dz in -2i32..=2 {
            for dy in -2i32..=2 {
                for dx in -2i32..=2 {
                    let (fx, fy, fz) = self.feature_3d(cx+dx, cy+dy, cz+dz);
                    let px = (cx + dx) as f32 + fx;
                    let py = (cy + dy) as f32 + fy;
                    let pz = (cz + dz) as f32 + fz;
                    let d  = self.dist3(x - px, y - py, z - pz);
                    if d < f1 { f2 = f1; f1 = d; }
                    else if d < f2 { f2 = d; }
                }
            }
        }
        self.combine(f1, f2)
    }
}

impl Default for Worley { fn default() -> Self { Self::new() } }

impl NoiseSource2d for Worley {
    #[inline]
    fn sample_2d(&self, x: f32, y: f32) -> f32 { self.sample_2d(x, y) }
}

impl NoiseSource3d for Worley {
    #[inline]
    fn sample_3d(&self, x: f32, y: f32, z: f32) -> f32 { self.sample_3d(x, y, z) }
        }
