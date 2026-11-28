// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-raster.md, section "canvas.rs"
// ============================================================================
//! The golden scene, the random-path set and their pinned hashes. One source for
//! the native test (`tests/golden.rs`) and for the wasm32 module of
//! `tools/wasm-hash`, so both hash exactly the same drawing.
//!
//! The scene uses literal coordinates only, no trigonometry, so the input is the
//! same on every target too.

// `is_multiple_of` is newer than the toolchains the tests should still build on.
#![allow(clippy::manual_is_multiple_of)]
// Each user of this file needs a different part of it.
#![allow(dead_code)]

use mid_math::Color32;
use mid_raster::cover::rasterize;
use mid_raster::shapes::KAPPA;
use mid_raster::{BlendSpace, Canvas, FillRule, IRect, Path, Surface};

pub fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h = 0xCBF2_9CE4_8422_2325u64;
    for &b in bytes {
        h = (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01B3);
    }
    h
}

fn circle(p: &mut Path, cx: f32, cy: f32, r: f32, clockwise: bool) {
    let k = r * KAPPA;
    if clockwise {
        p.move_to(cx + r, cy)
            .cubic_to(cx + r, cy + k, cx + k, cy + r, cx, cy + r)
            .cubic_to(cx - k, cy + r, cx - r, cy + k, cx - r, cy)
            .cubic_to(cx - r, cy - k, cx - k, cy - r, cx, cy - r)
            .cubic_to(cx + k, cy - r, cx + r, cy - k, cx + r, cy)
            .close();
    } else {
        p.move_to(cx + r, cy)
            .cubic_to(cx + r, cy - k, cx + k, cy - r, cx, cy - r)
            .cubic_to(cx - k, cy - r, cx - r, cy - k, cx - r, cy)
            .cubic_to(cx - r, cy + k, cx - k, cy + r, cx, cy + r)
            .cubic_to(cx + k, cy + r, cx + r, cy + k, cx + r, cy)
            .close();
    }
}

pub fn scene(space: BlendSpace) -> Surface {
    let mut s = Surface::new(64, 64);
    s.as_mut().clear(Color32::new(16, 32, 48, 255));
    let mut c = Canvas::new(s.as_mut()).with_blend_space(space);
    c.fill_rect(3.3, 4.7, 20.5, 12.25, Color32::new(200, 60, 40, 255));
    c.fill_rect(10.0, 10.0, 30.0, 20.0, Color32::new(40, 160, 220, 160));
    c.fill_rounded_rect(8.5, 30.25, 40.0, 24.0, 7.5, Color32::new(250, 230, 40, 200));
    c.stroke_line(2.0, 60.0, 62.0, 40.0, 3.5, Color32::new(255, 255, 255, 220));
    c.push_clip(30.0, 2.0, 30.0, 26.0);
    let mut star = Path::new();
    star.move_to(45.0, 3.0)
        .line_to(52.5, 24.5)
        .line_to(33.5, 11.5)
        .line_to(56.5, 11.5)
        .line_to(37.5, 24.5)
        .close();
    c.fill_path(&star, FillRule::EvenOdd, Color32::new(240, 240, 240, 230));
    c.push_clip(40.0, 8.0, 12.0, 12.0);
    c.fill_rect(0.0, 0.0, 64.0, 64.0, Color32::new(0, 0, 0, 128));
    c.pop_clip();
    c.pop_clip();
    let mut ring = Path::new();
    circle(&mut ring, 20.0, 48.0, 9.0, true);
    circle(&mut ring, 20.0, 48.0, 4.5, false);
    c.fill_path(&ring, FillRule::NonZero, Color32::new(60, 220, 90, 170));
    let stats = c.finish();
    assert_eq!((stats.ignored_pops, stats.closed_clips, stats.skipped_shapes), (0, 0, 0));
    s
}

/// The hashes of the scene in sRGB and in linear light. Set from a run on
/// x86_64 and compared on every other target in CI.
pub const SRGB_HASH: u64 = 0x2a10_5a8f_3771_5748;
pub const LINEAR_HASH: u64 = 0xcbc2_2468_1834_7507;


struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn unit(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (self.next() >> 40) as f32 / (1u64 << 24) as f32 * (hi - lo)
    }
}

/// The hash of 300 random polygons with quadratic and cubic pieces, rasterized
/// under both fill rules, every mask byte included. Pinned from x86_64; the same
/// value came out of a wasm32 build run in V8.
pub const RANDOM_HASH: u64 = 0xf856_c062_fb26_5d6f;

pub fn random_hash() -> u64 {
    let mut rng = Rng(0xD1B5_4A32_D192_ED03);
    let region = IRect::new(-4, -4, 26, 26);
    let mut all: Vec<u8> = Vec::new();
    for _ in 0..300 {
        let mut p = Path::new();
        p.move_to(rng.unit(-6.0, 28.0), rng.unit(-6.0, 28.0));
        for _ in 0..(3 + rng.next() % 5) {
            if rng.next() % 3 == 0 {
                p.quad_to(rng.unit(-6.0, 28.0), rng.unit(-6.0, 28.0), rng.unit(-6.0, 28.0), rng.unit(-6.0, 28.0));
            } else if rng.next() % 4 == 0 {
                p.cubic_to(
                    rng.unit(-6.0, 28.0),
                    rng.unit(-6.0, 28.0),
                    rng.unit(-6.0, 28.0),
                    rng.unit(-6.0, 28.0),
                    rng.unit(-6.0, 28.0),
                    rng.unit(-6.0, 28.0),
                );
            } else {
                p.line_to(rng.unit(-6.0, 28.0), rng.unit(-6.0, 28.0));
            }
        }
        p.close();
        let segs = p.flatten().unwrap();
        for rule in [FillRule::NonZero, FillRule::EvenOdd] {
            all.extend_from_slice(&rasterize(&segs, region, rule).data);
        }
    }
    fnv1a(&all)
}
