// crates/mid-math/src/noise/simplex.rs
//! Simplex noise — Ken Perlin (2001), Stefan Gustavson reference (2005).
//!
//! Advantages over classic Perlin:
//!   - No directional grid artifacts
//!   - O(n²) complexity vs O(2ⁿ) for n-dimensional Perlin
//!   - Slightly smoother visual quality
//!
//! Implements 2D, 3D, and 4D variants.
//! Output range ≈ `[-1, 1]`.

use super::{build_perm, NoiseSource2d, NoiseSource3d, NoiseSource4d};

// 12 gradient vectors for 3D (mid-edges of a cube); 2D uses only x,y components.
const GRAD3: [[f32; 3]; 12] = [
    [ 1.0,  1.0,  0.0], [-1.0,  1.0,  0.0], [ 1.0, -1.0,  0.0], [-1.0, -1.0,  0.0],
    [ 1.0,  0.0,  1.0], [-1.0,  0.0,  1.0], [ 1.0,  0.0, -1.0], [-1.0,  0.0, -1.0],
    [ 0.0,  1.0,  1.0], [ 0.0, -1.0,  1.0], [ 0.0,  1.0, -1.0], [ 0.0, -1.0, -1.0],
];

// 32 gradient vectors for 4D.
const GRAD4: [[f32; 4]; 32] = [
    [ 0., 1., 1., 1.], [ 0., 1., 1.,-1.], [ 0., 1.,-1., 1.], [ 0., 1.,-1.,-1.],
    [ 0.,-1., 1., 1.], [ 0.,-1., 1.,-1.], [ 0.,-1.,-1., 1.], [ 0.,-1.,-1.,-1.],
    [ 1., 0., 1., 1.], [ 1., 0., 1.,-1.], [ 1., 0.,-1., 1.], [ 1., 0.,-1.,-1.],
    [-1., 0., 1., 1.], [-1., 0., 1.,-1.], [-1., 0.,-1., 1.], [-1., 0.,-1.,-1.],
    [ 1., 1., 0., 1.], [ 1., 1., 0.,-1.], [ 1.,-1., 0., 1.], [ 1.,-1., 0.,-1.],
    [-1., 1., 0., 1.], [-1., 1., 0.,-1.], [-1.,-1., 0., 1.], [-1.,-1., 0.,-1.],
    [ 1., 1., 1., 0.], [ 1., 1.,-1., 0.], [ 1.,-1., 1., 0.], [ 1.,-1.,-1., 0.],
    [-1., 1., 1., 0.], [-1., 1.,-1., 0.], [-1.,-1., 1., 0.], [-1.,-1.,-1., 0.],
];

/// Simplex noise generator. Deterministic and seed-controllable.
#[derive(Clone)]
pub struct Simplex {
    perm:      [u8; 512],
    perm_mod12: [u8; 512],  // perm[i] % 12, precomputed
    perm_mod32: [u8; 512],  // perm[i] % 32, precomputed for 4D
}

impl Simplex {
    pub fn new() -> Self { Self::from_raw(build_perm(None)) }
    pub fn from_seed(seed: u64) -> Self { Self::from_raw(build_perm(Some(seed))) }

    fn from_raw(perm: [u8; 512]) -> Self {
        let mut perm_mod12 = [0u8; 512];
        let mut perm_mod32 = [0u8; 512];
        for i in 0..512 {
            perm_mod12[i] = perm[i] % 12;
            perm_mod32[i] = perm[i] % 32;
        }
        Self { perm, perm_mod12, perm_mod32 }
    }

    // ── 2D ───────────────────────────────────────────────────────────────────

    /// Sample 2D simplex noise. Output ≈ `[-1, 1]`.
    pub fn sample_2d(&self, xin: f32, yin: f32) -> f32 {
        const F2: f32 = 0.366025403784439;   // 0.5*(sqrt(3)-1)
        const G2: f32 = 0.211324865405187;   // (3-sqrt(3))/6

        let s = (xin + yin) * F2;
        let i = fast_floor(xin + s);
        let j = fast_floor(yin + s);
        let t = (i + j) as f32 * G2;

        let x0 = xin - (i as f32 - t);
        let y0 = yin - (j as f32 - t);

        let (i1, j1) = if x0 > y0 { (1, 0) } else { (0, 1) };

        let x1 = x0 - i1 as f32 + G2;
        let y1 = y0 - j1 as f32 + G2;
        let x2 = x0 - 1.0 + 2.0 * G2;
        let y2 = y0 - 1.0 + 2.0 * G2;

        let ii = (i & 255) as usize;
        let jj = (j & 255) as usize;

        let gi0 = self.perm_mod12[ii +     self.perm[jj          ] as usize] as usize;
        let gi1 = self.perm_mod12[ii + i1 + self.perm[jj + j1     ] as usize] as usize;
        let gi2 = self.perm_mod12[ii + 1  + self.perm[jj + 1      ] as usize] as usize;

        let n0 = corner2(x0, y0, &GRAD3[gi0]);
        let n1 = corner2(x1, y1, &GRAD3[gi1]);
        let n2 = corner2(x2, y2, &GRAD3[gi2]);

        70.0 * (n0 + n1 + n2)
    }

    // ── 3D ───────────────────────────────────────────────────────────────────

    /// Sample 3D simplex noise. Output ≈ `[-1, 1]`.
    pub fn sample_3d(&self, xin: f32, yin: f32, zin: f32) -> f32 {
        const F3: f32 = 1.0 / 3.0;
        const G3: f32 = 1.0 / 6.0;

        let s = (xin + yin + zin) * F3;
        let i = fast_floor(xin + s);
        let j = fast_floor(yin + s);
        let k = fast_floor(zin + s);
        let t = (i + j + k) as f32 * G3;

        let x0 = xin - (i as f32 - t);
        let y0 = yin - (j as f32 - t);
        let z0 = zin - (k as f32 - t);

        // Determine which simplex (6 possible tetrahedra in a cube).
        let (i1, j1, k1, i2, j2, k2) = if x0 >= y0 {
            if      y0 >= z0 { (1,0,0, 1,1,0) }
            else if x0 >= z0 { (1,0,0, 1,0,1) }
            else             { (0,0,1, 1,0,1) }
        } else {
            if      y0 <  z0 { (0,0,1, 0,1,1) }
            else if x0 <  z0 { (0,1,0, 0,1,1) }
            else             { (0,1,0, 1,1,0) }
        };

        let x1 = x0 - i1 as f32 + G3;
        let y1 = y0 - j1 as f32 + G3;
        let z1 = z0 - k1 as f32 + G3;
        let x2 = x0 - i2 as f32 + 2.0 * G3;
        let y2 = y0 - j2 as f32 + 2.0 * G3;
        let z2 = z0 - k2 as f32 + 2.0 * G3;
        let x3 = x0 - 1.0 + 3.0 * G3;
        let y3 = y0 - 1.0 + 3.0 * G3;
        let z3 = z0 - 1.0 + 3.0 * G3;

        let ii = (i & 255) as usize;
        let jj = (j & 255) as usize;
        let kk = (k & 255) as usize;

        let gi0 = self.perm_mod12[ii +     self.perm[jj +     self.perm[kk          ] as usize] as usize] as usize;
        let gi1 = self.perm_mod12[ii + i1 + self.perm[jj + j1 + self.perm[kk + k1    ] as usize] as usize] as usize;
        let gi2 = self.perm_mod12[ii + i2 + self.perm[jj + j2 + self.perm[kk + k2    ] as usize] as usize] as usize;
        let gi3 = self.perm_mod12[ii + 1  + self.perm[jj + 1  + self.perm[kk + 1     ] as usize] as usize] as usize;

        let n0 = corner3(x0, y0, z0, &GRAD3[gi0]);
        let n1 = corner3(x1, y1, z1, &GRAD3[gi1]);
        let n2 = corner3(x2, y2, z2, &GRAD3[gi2]);
        let n3 = corner3(x3, y3, z3, &GRAD3[gi3]);

        32.0 * (n0 + n1 + n2 + n3)
    }

    // ── 4D ───────────────────────────────────────────────────────────────────

    /// Sample 4D simplex noise. Output ≈ `[-1, 1]`.
    pub fn sample_4d(&self, xin: f32, yin: f32, zin: f32, win: f32) -> f32 {
        const F4: f32 = 0.309016994374947; // (sqrt(5)-1)/4
        const G4: f32 = 0.138196601125011; // (5-sqrt(5))/20
        const G4_2: f32 = G4 * 2.0;
        const G4_3: f32 = G4 * 3.0;
        const G4_4: f32 = G4 * 4.0 - 1.0;

        let s = (xin + yin + zin + win) * F4;
        let i = fast_floor(xin + s);
        let j = fast_floor(yin + s);
        let k = fast_floor(zin + s);
        let l = fast_floor(win + s);
        let t = (i + j + k + l) as f32 * G4;

        let x0 = xin - (i as f32 - t);
        let y0 = yin - (j as f32 - t);
        let z0 = zin - (k as f32 - t);
        let w0 = win - (l as f32 - t);

        // Rank-based simplex determination.
        let mut rank = [0u32; 4]; // [x, y, z, w]
        if x0 > y0 { rank[0] += 1; } else { rank[1] += 1; }
        if x0 > z0 { rank[0] += 1; } else { rank[2] += 1; }
        if x0 > w0 { rank[0] += 1; } else { rank[3] += 1; }
        if y0 > z0 { rank[1] += 1; } else { rank[2] += 1; }
        if y0 > w0 { rank[1] += 1; } else { rank[3] += 1; }
        if z0 > w0 { rank[2] += 1; } else { rank[3] += 1; }

        let i1 = usize::from(rank[0] >= 3); let j1 = usize::from(rank[1] >= 3);
        let k1 = usize::from(rank[2] >= 3); let l1 = usize::from(rank[3] >= 3);
        let i2 = usize::from(rank[0] >= 2); let j2 = usize::from(rank[1] >= 2);
        let k2 = usize::from(rank[2] >= 2); let l2 = usize::from(rank[3] >= 2);
        let i3 = usize::from(rank[0] >= 1); let j3 = usize::from(rank[1] >= 1);
        let k3 = usize::from(rank[2] >= 1); let l3 = usize::from(rank[3] >= 1);

        let x1 = x0 - i1 as f32 + G4;   let y1 = y0 - j1 as f32 + G4;
        let z1 = z0 - k1 as f32 + G4;   let w1 = w0 - l1 as f32 + G4;
        let x2 = x0 - i2 as f32 + G4_2; let y2 = y0 - j2 as f32 + G4_2;
        let z2 = z0 - k2 as f32 + G4_2; let w2 = w0 - l2 as f32 + G4_2;
        let x3 = x0 - i3 as f32 + G4_3; let y3 = y0 - j3 as f32 + G4_3;
        let z3 = z0 - k3 as f32 + G4_3; let w3 = w0 - l3 as f32 + G4_3;
        let x4 = x0 + G4_4;             let y4 = y0 + G4_4;
        let z4 = z0 + G4_4;             let w4 = w0 + G4_4;

        let ii = (i & 255) as usize;
        let jj = (j & 255) as usize;
        let kk = (k & 255) as usize;
        let ll = (l & 255) as usize;

        let lookup = |a: usize, b: usize, c: usize, d: usize| -> usize {
            self.perm_mod32[a + self.perm[b + self.perm[c + self.perm[d] as usize] as usize] as usize] as usize
        };

        let gi0 = lookup(ii,     jj,     kk,     ll    );
        let gi1 = lookup(ii + i1, jj + j1, kk + k1, ll + l1);
        let gi2 = lookup(ii + i2, jj + j2, kk + k2, ll + l2);
        let gi3 = lookup(ii + i3, jj + j3, kk + k3, ll + l3);
        let gi4 = lookup(ii + 1,  jj + 1,  kk + 1,  ll + 1 );

        let n0 = corner4(x0, y0, z0, w0, &GRAD4[gi0]);
        let n1 = corner4(x1, y1, z1, w1, &GRAD4[gi1]);
        let n2 = corner4(x2, y2, z2, w2, &GRAD4[gi2]);
        let n3 = corner4(x3, y3, z3, w3, &GRAD4[gi3]);
        let n4 = corner4(x4, y4, z4, w4, &GRAD4[gi4]);

        27.0 * (n0 + n1 + n2 + n3 + n4)
    }
}

impl Default for Simplex { fn default() -> Self { Self::new() } }

impl NoiseSource2d for Simplex {
    fn sample_2d(&self, x: f32, y: f32) -> f32 { self.sample_2d(x, y) }
}
impl NoiseSource3d for Simplex {
    fn sample_3d(&self, x: f32, y: f32, z: f32) -> f32 { self.sample_3d(x, y, z) }
}
impl NoiseSource4d for Simplex {
    fn sample_4d(&self, x: f32, y: f32, z: f32, w: f32) -> f32 { self.sample_4d(x, y, z, w) }
}

// ── Private helpers ───────────────────────────────────────────────────────────

#[inline(always)]
fn fast_floor(x: f32) -> i32 { if x >= 0.0 { x as i32 } else { x as i32 - 1 } }

/// Contribution from one 2D simplex corner.
#[inline(always)]
fn corner2(x: f32, y: f32, g: &[f32; 3]) -> f32 {
    let t = 0.5 - x * x - y * y;
    if t < 0.0 { 0.0 } else { let t2 = t * t; t2 * t2 * (g[0] * x + g[1] * y) }
}

/// Contribution from one 3D simplex corner.
#[inline(always)]
fn corner3(x: f32, y: f32, z: f32, g: &[f32; 3]) -> f32 {
    let t = 0.6 - x * x - y * y - z * z;
    if t < 0.0 { 0.0 } else { let t2 = t * t; t2 * t2 * (g[0] * x + g[1] * y + g[2] * z) }
}

/// Contribution from one 4D simplex corner.
#[inline(always)]
fn corner4(x: f32, y: f32, z: f32, w: f32, g: &[f32; 4]) -> f32 {
    let t = 0.6 - x * x - y * y - z * z - w * w;
    if t < 0.0 { 0.0 } else {
        let t2 = t * t;
        t2 * t2 * (g[0] * x + g[1] * y + g[2] * z + g[3] * w)
    }
  }
