// crates/mid-math/src/noise/perlin.rs
//! Classic improved Perlin gradient noise (2D / 3D / 4D).
//!
//! Uses the improved fade curve `6t⁵ − 15t⁴ + 10t³` and a doubled 512-element
//! permutation table so nested indexing `perm[perm[x]+y]` never overflows.
//!
//! Output range ≈ `[-1, 1]` for all dimensions.

use super::{build_perm, NoiseSource2d, NoiseSource3d, NoiseSource4d};

/// Classic improved Perlin noise. Deterministic and seed-controllable.
#[derive(Clone)]
pub struct Perlin {
    perm: [u8; 512],
}

impl Perlin {
    /// Construct with the canonical Ken Perlin permutation table.
    pub fn new() -> Self { Self { perm: build_perm(None) } }

    /// Construct with a shuffled table derived from `seed`.
    pub fn from_seed(seed: u64) -> Self { Self { perm: build_perm(Some(seed)) } }

    // ── Lookup helper ─────────────────────────────────────────────────────────

    #[inline(always)]
    fn p(&self, i: usize) -> usize { self.perm[i] as usize }

    // ── Public sampling API ───────────────────────────────────────────────────

    /// Sample 2D Perlin noise. Output ≈ `[-1, 1]`.
    pub fn sample_2d(&self, x: f32, y: f32) -> f32 {
        let xi = x.floor() as i32 as usize & 255;
        let yi = y.floor() as i32 as usize & 255;
        let xf = x - x.floor();
        let yf = y - y.floor();
        let u = fade(xf);
        let v = fade(yf);

        let aa = self.p(self.p(xi    ) + yi    );
        let ab = self.p(self.p(xi    ) + yi + 1);
        let ba = self.p(self.p(xi + 1) + yi    );
        let bb = self.p(self.p(xi + 1) + yi + 1);

        let x1 = lerp(u, grad2(aa, xf,       yf      ), grad2(ba, xf - 1.0, yf      ));
        let x2 = lerp(u, grad2(ab, xf,       yf - 1.0), grad2(bb, xf - 1.0, yf - 1.0));
        lerp(v, x1, x2)
    }

    /// Sample 3D Perlin noise. Output ≈ `[-1, 1]`.
    pub fn sample_3d(&self, x: f32, y: f32, z: f32) -> f32 {
        let xi = x.floor() as i32 as usize & 255;
        let yi = y.floor() as i32 as usize & 255;
        let zi = z.floor() as i32 as usize & 255;
        let xf = x - x.floor();
        let yf = y - y.floor();
        let zf = z - z.floor();
        let u = fade(xf); let v = fade(yf); let w = fade(zf);

        let aaa = self.p(self.p(self.p(xi    ) + yi    ) + zi    );
        let aba = self.p(self.p(self.p(xi    ) + yi + 1) + zi    );
        let aab = self.p(self.p(self.p(xi    ) + yi    ) + zi + 1);
        let abb = self.p(self.p(self.p(xi    ) + yi + 1) + zi + 1);
        let baa = self.p(self.p(self.p(xi + 1) + yi    ) + zi    );
        let bba = self.p(self.p(self.p(xi + 1) + yi + 1) + zi    );
        let bab = self.p(self.p(self.p(xi + 1) + yi    ) + zi + 1);
        let bbb = self.p(self.p(self.p(xi + 1) + yi + 1) + zi + 1);

        let x1 = lerp(u, grad3(aaa, xf,       yf,       zf      ), grad3(baa, xf-1., yf,   zf   ));
        let x2 = lerp(u, grad3(aba, xf,       yf - 1.0, zf      ), grad3(bba, xf-1., yf-1.,zf   ));
        let x3 = lerp(u, grad3(aab, xf,       yf,       zf - 1.0), grad3(bab, xf-1., yf,   zf-1.));
        let x4 = lerp(u, grad3(abb, xf,       yf - 1.0, zf - 1.0), grad3(bbb, xf-1., yf-1.,zf-1.));
        lerp(w, lerp(v, x1, x2), lerp(v, x3, x4))
    }

    /// Sample 4D Perlin noise. Output ≈ `[-1, 1]`.
    pub fn sample_4d(&self, x: f32, y: f32, z: f32, w: f32) -> f32 {
        let xi = x.floor() as i32 as usize & 255;
        let yi = y.floor() as i32 as usize & 255;
        let zi = z.floor() as i32 as usize & 255;
        let wi = w.floor() as i32 as usize & 255;
        let xf = x - x.floor(); let yf = y - y.floor();
        let zf = z - z.floor(); let wf = w - w.floor();
        // u=x fade, v=y fade, s=z fade, t=w fade
        let u = fade(xf); let v = fade(yf); let s = fade(zf); let t = fade(wf);

        // 16 corners of the 4D hypercube — index[w][z][y][x]
        let h = |a: usize, b: usize, c: usize, d: usize| {
            self.p(self.p(self.p(self.p(a) + b) + c) + d)
        };
        let mut corners = [0.0f32; 16];
        for bits in 0usize..16 {
            let xo = bits & 1;
            let yo = (bits >> 1) & 1;
            let zo = (bits >> 2) & 1;
            let wo = (bits >> 3) & 1;
            corners[bits] = grad4d(
                h(xi + xo, yi + yo, zi + zo, wi + wo),
                xf - xo as f32,
                yf - yo as f32,
                zf - zo as f32,
                wf - wo as f32,
            );
        }

        // Quadrilinear interpolation across all 16 corners.
        // index layout: wo*8 + zo*4 + yo*2 + xo
        let q = |xo: usize, yo: usize, zo: usize, wo: usize| -> f32 {
            corners[wo * 8 + zo * 4 + yo * 2 + xo]
        };

        // Innermost: x (u), then y (v), then z (s), then w (t).
        lerp(t,
            lerp(s,
                lerp(v,
                    lerp(u, q(0,0,0,0), q(1,0,0,0)),
                    lerp(u, q(0,1,0,0), q(1,1,0,0))),
                lerp(v,
                    lerp(u, q(0,0,1,0), q(1,0,1,0)),
                    lerp(u, q(0,1,1,0), q(1,1,1,0)))),
            lerp(s,
                lerp(v,
                    lerp(u, q(0,0,0,1), q(1,0,0,1)),
                    lerp(u, q(0,1,0,1), q(1,1,0,1))),
                lerp(v,
                    lerp(u, q(0,0,1,1), q(1,0,1,1)),
                    lerp(u, q(0,1,1,1), q(1,1,1,1)))))
    }
}

impl Default for Perlin {
    fn default() -> Self { Self::new() }
}

impl NoiseSource2d for Perlin {
    fn sample_2d(&self, x: f32, y: f32) -> f32 { self.sample_2d(x, y) }
}
impl NoiseSource3d for Perlin {
    fn sample_3d(&self, x: f32, y: f32, z: f32) -> f32 { self.sample_3d(x, y, z) }
}
impl NoiseSource4d for Perlin {
    fn sample_4d(&self, x: f32, y: f32, z: f32, w: f32) -> f32 { self.sample_4d(x, y, z, w) }
}

// ── Private helpers ───────────────────────────────────────────────────────────

/// Improved fade: `6t⁵ − 15t⁴ + 10t³` (C² continuity).
#[inline(always)]
fn fade(t: f32) -> f32 { t * t * t * (t * (t * 6.0 - 15.0) + 10.0) }

#[inline(always)]
fn lerp(t: f32, a: f32, b: f32) -> f32 { a + t * (b - a) }

/// 2D gradient: project `(x, y)` onto one of 4 gradient directions.
fn grad2(hash: usize, x: f32, y: f32) -> f32 {
    match hash & 3 {
        0 =>  x + y,
        1 => -x + y,
        2 =>  x - y,
        _ => -x - y,
    }
}

/// 3D gradient: project onto one of 12 mid-edge unit vectors.
fn grad3(hash: usize, x: f32, y: f32, z: f32) -> f32 {
    match hash & 15 {
        0 | 12 =>  x + y,
        1 | 14 => -x + y,
        2      =>  x - y,
        3      => -x - y,
        4      =>  x + z,
        5      => -x + z,
        6      =>  x - z,
        7      => -x - z,
        8      =>  y + z,
        9      => -y + z,
        10     =>  y - z,
        _      => -y - z,
    }
}

/// 4D gradient: project onto one of 32 unit vectors.
fn grad4d(hash: usize, x: f32, y: f32, z: f32, w: f32) -> f32 {
    let h = hash & 31;
    let a = if h < 24 { x } else { y };
    let b = if h < 16 { y } else { z };
    let c = if h < 8  { z } else { w };
    let sa = if h & 1 != 0 { -1.0 } else { 1.0 };
    let sb = if h & 2 != 0 { -1.0 } else { 1.0 };
    let sc = if h & 4 != 0 { -1.0 } else { 1.0 };
    sa * a + sb * b + sc * c
    }
