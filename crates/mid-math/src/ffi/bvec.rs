// crates/mid-math/src/ffi/bvec.rs
//! C-ABI exports for boolean vector masks (BVec2/BVec3/BVec4).
//!
//! No C wrapper struct needed here -- BVec2/3/4 are already
//! `#[repr(C)]` with plain `bool` fields (confirmed directly against
//! source), so they're FFI-safe as-is. `test`'s underlying Rust method
//! panics on an out-of-range index -- not guarded here, matching the
//! rest of this FFI layer's existing convention of not adding bounds-
//! checking beyond what the wrapped method already does.

use crate::{BVec2, BVec3, BVec4};

// ═══════════════════════════════════════════════════════════════════════════
//  Exports — BVec2
// ═══════════════════════════════════════════════════════════════════════════

#[no_mangle] pub extern "C" fn mid_bvec2_new(x:bool,y:bool)->BVec2{BVec2::new(x,y)}
#[no_mangle] pub extern "C" fn mid_bvec2_splat(v:bool)->BVec2{BVec2::splat(v)}
#[no_mangle] pub extern "C" fn mid_bvec2_any(v:BVec2)->bool{v.any()}
#[no_mangle] pub extern "C" fn mid_bvec2_all(v:BVec2)->bool{v.all()}
#[no_mangle] pub extern "C" fn mid_bvec2_bitmask(v:BVec2)->u32{v.bitmask()}
#[no_mangle] pub extern "C" fn mid_bvec2_test(v:BVec2,index:u32)->bool{v.test(index as usize)}
#[no_mangle] pub extern "C" fn mid_bvec2_and(a:BVec2,b:BVec2)->BVec2{a&b}
#[no_mangle] pub extern "C" fn mid_bvec2_or(a:BVec2,b:BVec2)->BVec2{a|b}
#[no_mangle] pub extern "C" fn mid_bvec2_xor(a:BVec2,b:BVec2)->BVec2{a^b}
#[no_mangle] pub extern "C" fn mid_bvec2_not(v:BVec2)->BVec2{!v}

// ═══════════════════════════════════════════════════════════════════════════
//  Exports — BVec3
// ═══════════════════════════════════════════════════════════════════════════

#[no_mangle] pub extern "C" fn mid_bvec3_new(x:bool,y:bool,z:bool)->BVec3{BVec3::new(x,y,z)}
#[no_mangle] pub extern "C" fn mid_bvec3_splat(v:bool)->BVec3{BVec3::splat(v)}
#[no_mangle] pub extern "C" fn mid_bvec3_any(v:BVec3)->bool{v.any()}
#[no_mangle] pub extern "C" fn mid_bvec3_all(v:BVec3)->bool{v.all()}
#[no_mangle] pub extern "C" fn mid_bvec3_bitmask(v:BVec3)->u32{v.bitmask()}
#[no_mangle] pub extern "C" fn mid_bvec3_test(v:BVec3,index:u32)->bool{v.test(index as usize)}
#[no_mangle] pub extern "C" fn mid_bvec3_and(a:BVec3,b:BVec3)->BVec3{a&b}
#[no_mangle] pub extern "C" fn mid_bvec3_or(a:BVec3,b:BVec3)->BVec3{a|b}
#[no_mangle] pub extern "C" fn mid_bvec3_xor(a:BVec3,b:BVec3)->BVec3{a^b}
#[no_mangle] pub extern "C" fn mid_bvec3_not(v:BVec3)->BVec3{!v}

// ═══════════════════════════════════════════════════════════════════════════
//  Exports — BVec4
// ═══════════════════════════════════════════════════════════════════════════

#[no_mangle] pub extern "C" fn mid_bvec4_new(x:bool,y:bool,z:bool,w:bool)->BVec4{BVec4::new(x,y,z,w)}
#[no_mangle] pub extern "C" fn mid_bvec4_splat(v:bool)->BVec4{BVec4::splat(v)}
#[no_mangle] pub extern "C" fn mid_bvec4_any(v:BVec4)->bool{v.any()}
#[no_mangle] pub extern "C" fn mid_bvec4_all(v:BVec4)->bool{v.all()}
#[no_mangle] pub extern "C" fn mid_bvec4_bitmask(v:BVec4)->u32{v.bitmask()}
#[no_mangle] pub extern "C" fn mid_bvec4_test(v:BVec4,index:u32)->bool{v.test(index as usize)}
#[no_mangle] pub extern "C" fn mid_bvec4_and(a:BVec4,b:BVec4)->BVec4{a&b}
#[no_mangle] pub extern "C" fn mid_bvec4_or(a:BVec4,b:BVec4)->BVec4{a|b}
#[no_mangle] pub extern "C" fn mid_bvec4_xor(a:BVec4,b:BVec4)->BVec4{a^b}
#[no_mangle] pub extern "C" fn mid_bvec4_not(v:BVec4)->BVec4{!v}
