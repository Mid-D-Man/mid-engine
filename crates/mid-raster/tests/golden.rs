// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-raster.md, section "canvas.rs"
// ============================================================================
//! Golden images: one scene drawn in both blend spaces, and 300 random curved
//! paths, hashed. The same hashes must come out on every target, which is the
//! integer-exactness the contract asks for (`docs/mid-ui-contracts.md`, section
//! 1.2). `tools/wasm-hash` checks them on wasm32.

use mid_raster::BlendSpace;

#[path = "golden/scenes.rs"]
mod scenes;

use scenes::{fnv1a, random_hash, scene, LINEAR_HASH, RANDOM_HASH, SRGB_HASH};

#[test]
fn the_scene_hashes_are_the_same_on_every_target() {
    let a = fnv1a(scene(BlendSpace::Srgb).bytes());
    let b = fnv1a(scene(BlendSpace::Linear).bytes());
    eprintln!("golden srgb {a:#018x} linear {b:#018x}");
    assert_ne!(a, b, "the two blend spaces draw differently");
    assert_eq!(a, SRGB_HASH, "sRGB scene changed");
    assert_eq!(b, LINEAR_HASH, "linear scene changed");
}

#[test]
fn drawing_the_scene_twice_gives_the_same_bytes() {
    assert_eq!(scene(BlendSpace::Srgb), scene(BlendSpace::Srgb));
}

#[test]
fn random_curved_paths_hash_the_same_on_every_target() {
    assert_eq!(random_hash(), RANDOM_HASH);
}
