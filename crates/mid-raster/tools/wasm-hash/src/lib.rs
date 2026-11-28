//! The golden hashes of `mid-raster`, exported for a wasm32 host.

#[path = "../../../tests/golden/scenes.rs"]
mod scenes;

use mid_raster::BlendSpace;

#[no_mangle]
pub extern "C" fn srgb() -> u64 {
    scenes::fnv1a(scenes::scene(BlendSpace::Srgb).bytes())
}

#[no_mangle]
pub extern "C" fn linear() -> u64 {
    scenes::fnv1a(scenes::scene(BlendSpace::Linear).bytes())
}

#[no_mangle]
pub extern "C" fn random() -> u64 {
    scenes::random_hash()
}

#[no_mangle]
pub extern "C" fn expected_srgb() -> u64 {
    scenes::SRGB_HASH
}

#[no_mangle]
pub extern "C" fn expected_linear() -> u64 {
    scenes::LINEAR_HASH
}

#[no_mangle]
pub extern "C" fn expected_random() -> u64 {
    scenes::RANDOM_HASH
}
