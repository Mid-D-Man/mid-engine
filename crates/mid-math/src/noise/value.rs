// crates/mid-math/src/noise/value.rs
//! Value noise — interpolated random values on a lattice.
//!
//! Simpler and faster than gradient noise. Produces a "blocky" look at low
//! frequencies that can be tamed by using fBm layering (see `fbm.rs`).
//!
//! Output range ≈ `[-1, 1]`.

use super::{build_perm, NoiseSource2d, NoiseSource3d, NoiseSource4d};

/// Value noise generator. Deterministic and seed-controllable.
#[derive(Clone)]
pub struct Value {
    perm: [u8; 512],
}

impl Value {
    pub fn new() -> Self { Self { perm: build_perm(None) } }
    pub fn from_seed(seed: u64) -> Self { Self { perm: build_perm(Some(seed)) } }

    #[inline(always)]
    fn p(&self, i: usize) -> usize { self.perm[i & 511] as usize }

    /// Random value in `[0, 1)` from a hash.
    #[inline(always)]
    fn val(hash: usize) -> f32 { hash as f32 / 255.0 }

    /// Sample 2D value noise. Output ≈ `[-1, 1]`.
    pub fn sample_2d(&self, x: f32, y: f32) -> f32 {
        let xi = x.floor() as i32 as usize & 255;
        let yi = y.floor() as i32 as usize & 255;
        let xf = x - x.floor();
        let yf = y - y.floor();
        let u = fade(xf);
        let v = fade(yf);

        let v00 = Self::val(self.p(self.p(xi    ) + yi    ));
        let v10 = Self::val(self.p(self.p(xi + 1) + yi    ));
        let v01 = Self::val(self.p(self.p(xi    ) + yi + 1));
        let v11 = Self::val(self.p(self.p(xi + 1) + yi + 1));

        let x1 = lerp(u, v00, v10);
        let x2 = lerp(u, v01, v11);
        // Remap [0,1] → [-1,1]
        lerp(v, x1, x2) * 2.0 - 1.0
    }

    /// Sample 3D value noise. Output ≈ `[-1, 1]`.
    pub fn sample_3d(&self, x: f32, y: f32, z: f32) -> f32 {
        let xi = x.floor() as i32 as usize & 255;
        let yi = y.floor() as i32 as usize & 255;
        let zi = z.floor() as i32 as usize & 255;
        let xf = x - x.floor();
        let yf = y - y.floor();
        let zf = z - z.floor();
        let u = fade(xf); let v = fade(yf); let w = fade(zf);

        let hash = |a: usize, b: usize, c: usize| {
            Self::val(self.p(self.p(self.p(a) + b) + c))
        };

        let x1 = lerp(u, hash(xi, yi,   zi), hash(xi+1, yi,   zi));
        // Full trilinear interpolation across 8 corners
        let c000 = Self::val(self.p(self.p(self.p(xi    ) + yi    ) + zi    ));
        let c100 = Self::val(self.p(self.p(self.p(xi + 1) + yi    ) + zi    ));
        let c010 = Self::val(self.p(self.p(self.p(xi    ) + yi + 1) + zi    ));
        let c110 = Self::val(self.p(self.p(self.p(xi + 1) + yi + 1) + zi    ));
        let c001 = Self::val(self.p(self.p(self.p(xi    ) + yi    ) + zi + 1));
        let c101 = Self::val(self.p(self.p(self.p(xi + 1) + yi    ) + zi + 1));
        let c011 = Self::val(self.p(self.p(self.p(xi    ) + yi + 1) + zi + 1));
        let c111 = Self::val(self.p(self.p(self.p(xi + 1) + yi + 1) + zi + 1));

        let a1 = lerp(u, c000, c100);
        let a2 = lerp(u, c010, c110);
        let a3 = lerp(u, c001, c101);
        let a4 = lerp(u, c011, c111);
        let b1 = lerp(v, a1, a2);
        let b2 = lerp(v, a3, a4);
        let _ = x1; // suppress unused warning from early draft
        lerp(w, b1, b2) * 2.0 - 1.0
    }

    /// Sample 4D value noise. Output ≈ `[-1, 1]`.
    pub fn sample_4d(&self, x: f32, y: f32, z: f32, ww: f32) -> f32 {
        let xi = x.floor()  as i32 as usize & 255;
        let yi = y.floor()  as i32 as usize & 255;
        let zi = z.floor()  as i32 as usize & 255;
        let wi = ww.floor() as i32 as usize & 255;
        let xf = x  - x.floor();
        let yf = y  - y.floor();
        let zf = z  - z.floor();
        let wf = ww - ww.floor();
        let u = fade(xf); let v = fade(yf); let s = fade(zf); let t = fade(wf);

        let h = |a: usize, b: usize, c: usize, d: usize| -> f32 {
            Self::val(self.p(self.p(self.p(self.p(a) + b) + c) + d))
        };

        // 16 corners of 4D hypercube
        let mut vals = [[0.0f32; 2]; 8];
        for xo in 0..2usize { for yo in 0..2 { for zo in 0..2 {
            vals[xo*4+yo*2+zo][0] = h(xi+xo, yi+yo, zi+zo, wi    );
            vals[xo*4+yo*2+zo][1] = h(xi+xo, yi+yo, zi+zo, wi + 1);
        }}}

        // Quadrilinear interpolation
        let mut r = [0.0f32; 8];
        for i in 0..8 { r[i] = lerp(t, vals[i][0], vals[i][1]); }
        let mut r2 = [0.0f32; 4];
        for i in 0..4 { r2[i] = lerp(s, r[i], r[i+4]); }
        let mut r3 = [0.0f32; 2];
        for i in 0..2 { r3[i] = lerp(v, r2[i], r2[i+2]); }
        lerp(u, r3[0], r3[1]) * 2.0 - 1.0
    }
}

impl Default for Value { fn default() -> Self { Self::new() } }

impl NoiseSource2d for Value {
    fn sample_2d(&self, x: f32, y: f32) -> f32 { self.sample_2d(x, y) }
}
impl NoiseSource3d for Value {
    fn sample_3d(&self, x: f32, y: f32, z: f32) -> f32 { self.sample_3d(x, y, z) }
}
impl NoiseSource4d for Value {
    fn sample_4d(&self, x: f32, y: f32, z: f32, w: f32) -> f32 { self.sample_4d(x, y, z, w) }
}

#[inline(always)] fn fade(t: f32) -> f32 { t * t * t * (t * (t * 6.0 - 15.0) + 10.0) }
#[inline(always)] fn lerp(t: f32, a: f32, b: f32) -> f32 { a + t * (b - a) }
