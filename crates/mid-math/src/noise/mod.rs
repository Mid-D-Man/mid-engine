// crates/mid-math/src/noise/mod.rs
//! Coherent noise primitives for Mid Engine.
//!
//! | Type         | Style         | Best for                              |
//! |--------------|---------------|---------------------------------------|
//! | `Perlin`     | Gradient      | Classic terrain, fog, animation       |
//! | `Simplex`    | Gradient      | No directional artifacts, faster 3D+  |
//! | `Value`      | Interpolated  | Simple blocky look, fast              |
//! | `Worley`     | Cellular      | Stone, cells, cracked surfaces        |
//! | `Fbm`        | Layered       | Mountains, clouds, organic detail     |
//! | `DomainWarp` | Warped        | Swirling, highly organic shapes       |
//!
//! All generators are deterministic and seed-controllable.
//! Output is in the approximate range `[-1, 1]` unless noted otherwise.

// ── Shared permutation table ──────────────────────────────────────────────────

/// Ken Perlin's reference 256-entry permutation table.
/// Doubled to 512 in each struct so indices like `perm[perm[x]+y]` are safe.
pub(crate) const DEFAULT_PERM: [u8; 256] = [
    151, 160, 137,  91,  90,  15, 131,  13, 201,  95,  96,  53, 194, 233,   7, 225,
    140,  36, 103,  30,  69, 142,   8,  99,  37, 240,  21,  10,  23, 190,   6, 148,
    247, 120, 234,  75,   0,  26, 197,  62,  94, 252, 219, 203, 117,  35,  11,  32,
     57, 177,  33,  88, 237, 149,  56,  87, 174,  20, 125, 136, 171, 168,  68, 175,
     74, 165,  71, 134, 139,  48,  27, 166,  77, 146, 158, 231,  83, 111, 229, 122,
     60, 211, 133, 230, 220, 105,  92,  41,  55,  46, 245,  40, 244, 102, 143,  54,
     65,  25,  63, 161,   1, 216,  80,  73, 209,  76, 132, 187, 208,  89,  18, 169,
    200, 196, 135, 130, 116, 188, 159,  86, 164, 100, 109, 198, 173, 186,   3,  64,
     52, 217, 226, 250, 124, 123,   5, 202,  38, 147, 118, 126, 255,  82,  85, 212,
    207, 206,  59, 227,  47,  16,  58,  17, 182, 189,  28,  42, 223, 183, 170, 213,
    119, 248, 152,   2,  44, 154, 163,  70, 221, 153, 101, 155, 167,  43, 172,   9,
    129,  22,  39, 253,  19,  98, 108, 110,  79, 113, 224, 232, 178, 185, 112, 104,
    218, 246,  97, 228, 251,  34, 242, 193, 238, 210, 144,  12, 191, 179, 162, 241,
     81,  51, 145, 235, 249,  14, 239, 107,  49, 192, 214,  31, 181, 199, 106, 157,
    184,  84, 204, 176, 115, 121,  50,  45, 127,   4, 150, 254, 138, 236, 205,  93,
    222, 114,  67,  29,  24,  72, 243, 141, 128, 195,  78,  66, 215,  61, 156, 180,
];

/// Build a doubled 512-element permutation table.
/// `seed = None` uses the default table; `seed = Some(s)` Fisher-Yates shuffles it.
pub(crate) fn build_perm(seed: Option<u64>) -> [u8; 512] {
    let mut table = DEFAULT_PERM;
    if let Some(s) = seed {
        let mut rng = if s == 0 { 1u64 } else { s };
        for i in (1usize..256).rev() {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let j = (rng as usize) % (i + 1);
            table.swap(i, j);
        }
    }
    let mut perm = [0u8; 512];
    for i in 0..256 {
        perm[i]       = table[i];
        perm[i + 256] = table[i];
    }
    perm
}

// ── Sampling traits ───────────────────────────────────────────────────────────

/// 2D coherent noise sampler. Output ≈ `[-1, 1]`.
pub trait NoiseSource2d {
    fn sample_2d(&self, x: f32, y: f32) -> f32;
}

/// 3D coherent noise sampler. Output ≈ `[-1, 1]`.
pub trait NoiseSource3d {
    fn sample_3d(&self, x: f32, y: f32, z: f32) -> f32;
}

/// 4D coherent noise sampler. Output ≈ `[-1, 1]`.
pub trait NoiseSource4d {
    fn sample_4d(&self, x: f32, y: f32, z: f32, w: f32) -> f32;
}

// ── Submodules ────────────────────────────────────────────────────────────────

pub mod perlin;
pub mod simplex;
pub mod value;
pub mod worley;
pub mod fbm;

pub use perlin::Perlin;
pub use simplex::Simplex;
pub use value::Value;
pub use worley::Worley;
pub use fbm::{Fbm, DomainWarp};
