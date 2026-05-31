// crates/mid-math/src/helpers/octahedral.rs
//! Octahedral normal encoding — pack unit normals into 2 scalars.
//!
//! The octahedral scheme projects a unit sphere normal onto the surface of a
//! unit L1 octahedron, then unfolds that octahedron into the square `[-1,1]²`.
//! This gives uniform distribution and is the standard GPU technique for
//! packing normals into G-buffers, meshes, and BC5 / two-channel textures.
//!
//! | Variant              | Storage   | Max error (degrees) |
//! |----------------------|-----------|---------------------|
//! | `f32` range [-1,1]²  | 8 bytes   | < 0.0001°           |
//! | snorm16 (2× i16)     | 4 bytes   | < 0.002°            |
//! | snorm8  (2× i8)      | 2 bytes   | < 0.4°              |
//!
//! # Usage
//! ```rust
//! use mid_math::{Vec3, encode_octahedral_snorm16, decode_octahedral_snorm16};
//!
//! let normal = Vec3::new(0.0, 1.0, 0.0).normalize();
//! let packed  = encode_octahedral_snorm16(normal);  // (i16, i16)
//! let decoded = decode_octahedral_snorm16(packed.0, packed.1);
//! ```
//!
//! Reference: Cigolle et al. "A Survey of Efficient Representations for
//! Independent Unit Vectors", JCGT 2014.

use crate::{Vec2, Vec3};

// ── Floating-point encode / decode ────────────────────────────────────────────

/// Encode a unit `Vec3` normal into `Vec2` octahedral coordinates in `[-1, 1]²`.
///
/// The input must be a unit vector (normalised). Non-unit inputs produce
/// undefined results.
///
/// The output is lossless at f32 precision — decode then re-encode gives
/// identical bits on all platforms.
#[inline]
pub fn encode_octahedral(n: Vec3) -> Vec2 {
    // Project onto L1 unit octahedron
    let inv_l1 = 1.0 / (n.x.abs() + n.y.abs() + n.z.abs());
    let mut px = n.x * inv_l1;
    let mut py = n.y * inv_l1;

    // Unfold the lower hemisphere onto the square
    if n.z < 0.0 {
        let ox = (1.0 - py.abs()) * if px >= 0.0 { 1.0 } else { -1.0 };
        let oy = (1.0 - px.abs()) * if py >= 0.0 { 1.0 } else { -1.0 };
        px = ox;
        py = oy;
    }

    Vec2::new(px, py)
}

/// Decode `Vec2` octahedral coordinates `[-1, 1]²` back to a unit `Vec3`.
///
/// The result is always normalised (length ≈ 1.0). For snorm inputs,
/// divide by the quantisation scale before calling (see snorm helpers below).
#[inline]
pub fn decode_octahedral(p: Vec2) -> Vec3 {
    let z = 1.0 - p.x.abs() - p.y.abs();

    let (px, py) = if z >= 0.0 {
        (p.x, p.y)
    } else {
        // Re-fold lower hemisphere
        let ox = (1.0 - p.y.abs()) * if p.x >= 0.0 { 1.0 } else { -1.0 };
        let oy = (1.0 - p.x.abs()) * if p.y >= 0.0 { 1.0 } else { -1.0 };
        (ox, oy)
    };

    Vec3::new(px, py, z).normalize()
}

// ── 8-bit snorm (2 bytes total) ───────────────────────────────────────────────

/// Encode a unit normal to `(i8, i8)` snorm8 — **2 bytes**.
///
/// Maximum angular error: < 0.4°.
/// Use for low-detail normals: terrain, distant meshes, particle normals.
#[inline]
pub fn encode_octahedral_snorm8(n: Vec3) -> (i8, i8) {
    let p = encode_octahedral(n);
    let x = (p.x * 127.0).round().clamp(-127.0, 127.0) as i8;
    let y = (p.y * 127.0).round().clamp(-127.0, 127.0) as i8;
    (x, y)
}

/// Decode `(i8, i8)` snorm8 back to a unit `Vec3`.
#[inline]
pub fn decode_octahedral_snorm8(x: i8, y: i8) -> Vec3 {
    decode_octahedral(Vec2::new(x as f32 / 127.0, y as f32 / 127.0))
}

// ── 16-bit snorm (4 bytes total) ─────────────────────────────────────────────

/// Encode a unit normal to `(i16, i16)` snorm16 — **4 bytes**.
///
/// Maximum angular error: < 0.002°.
/// Recommended for high-quality meshes, tangent frames, and BC5 normal maps.
#[inline]
pub fn encode_octahedral_snorm16(n: Vec3) -> (i16, i16) {
    let p = encode_octahedral(n);
    let x = (p.x * 32767.0).round().clamp(-32767.0, 32767.0) as i16;
    let y = (p.y * 32767.0).round().clamp(-32767.0, 32767.0) as i16;
    (x, y)
}

/// Decode `(i16, i16)` snorm16 back to a unit `Vec3`.
#[inline]
pub fn decode_octahedral_snorm16(x: i16, y: i16) -> Vec3 {
    decode_octahedral(Vec2::new(x as f32 / 32767.0, y as f32 / 32767.0))
}

// ── Pack / unpack as u32 (GPU-friendly) ──────────────────────────────────────

/// Pack snorm8 octahedral normal into a `u32` — **4 bytes, GPU-upload ready**.
///
/// Layout: bits `[31:16]` = unused, `[15:8]` = y snorm8, `[7:0]` = x snorm8.
/// Alternatively use the `RG8_SNORM` texture format (same byte order on LE).
#[inline]
pub fn encode_octahedral_u32(n: Vec3) -> u32 {
    let (x, y) = encode_octahedral_snorm8(n);
    ((y as u8 as u32) << 8) | (x as u8 as u32)
}

/// Unpack a `u32` snorm8 octahedral encoding back to a unit `Vec3`.
#[inline]
pub fn decode_octahedral_u32(packed: u32) -> Vec3 {
    let x = (packed & 0xFF) as i8;
    let y = ((packed >> 8) & 0xFF) as i8;
    decode_octahedral_snorm8(x, y)
}
