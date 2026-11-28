// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-raster.md, section "linear.rs"
// ============================================================================
//! The linear-light blend option of `docs/mid-ui-contracts.md`, section 1.3.
//!
//! Surfaces keep sRGB-encoded bytes. Only the arithmetic of the over changes:
//! both colors are decoded to 16-bit linear light through a table, mixed, and
//! encoded back through a second table. Everything is integer, so the result
//! is the same on every target. Destinations are treated as opaque.

use mid_math::Color32;

use crate::tables::{DECODE, THRESHOLDS};

/// sRGB level to linear light in `0..=65535`.
#[inline]
pub fn decode(v: u8) -> u16 {
    DECODE[v as usize]
}

/// The sRGB level nearest to `l` in the linear domain, ties to the lower level.
/// A binary search of eight comparisons over the 255 thresholds.
pub fn encode(l: u16) -> u8 {
    let (mut lo, mut hi) = (0usize, 255usize);
    while lo < hi {
        let mid = (lo + hi) / 2;
        if l > THRESHOLDS[mid] {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo as u8
}

/// One color channel of `src` with alpha `a` over an opaque `dst`, in linear
/// light: `E((L(s) * a + L(d) * (255 - a) + 127) / 255)`.
#[inline]
pub fn over_channel(s: u8, d: u8, a: u8) -> u8 {
    let a = a as u32;
    let l = (decode(s) as u32 * a + decode(d) as u32 * (255 - a) + 127) / 255;
    encode(l as u16)
}

/// `src` over `dst` in linear light. Like [`Color32::blend_over_rounded`], the
/// color channels treat `dst` as opaque and the alpha accumulates as
/// `a + div255(dst.a * (255 - a))`.
pub fn over(dst: Color32, src: Color32) -> Color32 {
    if src.a == 255 {
        return src;
    }
    if src.a == 0 {
        return dst;
    }
    let ia = 255 - src.a as u32;
    Color32 {
        r: over_channel(src.r, dst.r, src.a),
        g: over_channel(src.g, dst.g, src.a),
        b: over_channel(src.b, dst.b, src.a),
        a: src.a.saturating_add(Color32::div255_round(dst.a as u32 * ia) as u8),
    }
}
