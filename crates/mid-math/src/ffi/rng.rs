// crates/mid-math/src/ffi/rng.rs
//! C-ABI exports for deterministic PRNGs.
//!
//! Pattern: pass state by mutable pointer. Each call advances the state
//! and returns a value. The state structs are plain POD — safe to memcpy
//! for save/restore.

use crate::{Pcg32, Xorshift64};

// ═══════════════════════════════════════════════════════════════════════════
//  C state types
// ═══════════════════════════════════════════════════════════════════════════

/// Xorshift64 generator state. 8 bytes. Must be non-zero.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct CXorshift64State {
    pub state: u64,
}

/// PCG32 generator state. 16 bytes. `inc` must be odd (set via `mid_pcg32_create`).
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct CPcg32State {
    pub state: u64,
    pub inc:   u64,
}

// ── Internal helpers ──────────────────────────────────────────────────────────

#[inline(always)]
unsafe fn xs64(s: *const CXorshift64State) -> Xorshift64 {
    Xorshift64::new((*s).state)
}

#[inline(always)]
unsafe fn write_xs64(s: *mut CXorshift64State, rng: &Xorshift64) {
    (*s).state = rng.state();
}

#[inline(always)]
unsafe fn pcg32(s: *const CPcg32State) -> Pcg32 {
    let mut r = Pcg32 { state: 0, inc: 0 };
    // Use the raw state/inc directly — bypass the warmup done in Pcg32::new.
    // We restore a previously valid state, not creating fresh.
    let inc = (*s).inc | 1; // guarantee odd
    r.state = (*s).state;
    // Reconstruct correctly: use internal field access via struct literal trick.
    // Since Pcg32 fields are private, use the public API workaround:
    // create a new generator and force state.
    let mut r2 = Pcg32::new((*s).state, (inc >> 1) as u64);
    // Adjust state: the constructor does a warmup step, so we can't use it
    // to restore a mid-stream state. Expose the state as raw via set_state.
    r2.set_state((*s).state, inc);
    r2
}

#[inline(always)]
unsafe fn write_pcg32(s: *mut CPcg32State, rng: &Pcg32) {
    let (st, inc) = rng.state();
    (*s).state = st;
    (*s).inc   = inc;
}

// ═══════════════════════════════════════════════════════════════════════════
//  Xorshift64 exports
// ═══════════════════════════════════════════════════════════════════════════

/// Create Xorshift64 state from `seed`. If seed is 0, uses 1.
#[no_mangle]
pub extern "C" fn mid_xorshift64_create(seed: u64) -> CXorshift64State {
    CXorshift64State { state: if seed == 0 { 1 } else { seed } }
}

/// Advance state and return next u64.
#[no_mangle]
pub unsafe extern "C" fn mid_xorshift64_next_u64(s: *mut CXorshift64State) -> u64 {
    let mut rng = xs64(s);
    let v = rng.next_u64();
    write_xs64(s, &rng);
    v
}

/// Advance state and return next u32 (upper 32 bits of next_u64).
#[no_mangle]
pub unsafe extern "C" fn mid_xorshift64_next_u32(s: *mut CXorshift64State) -> u32 {
    let mut rng = xs64(s);
    let v = rng.next_u32();
    write_xs64(s, &rng);
    v
}

/// Advance state and return uniform f32 in [0, 1).
#[no_mangle]
pub unsafe extern "C" fn mid_xorshift64_f32(s: *mut CXorshift64State) -> f32 {
    let mut rng = xs64(s);
    let v = rng.f32();
    write_xs64(s, &rng);
    v
}

/// Advance state and return uniform f64 in [0, 1).
#[no_mangle]
pub unsafe extern "C" fn mid_xorshift64_f64(s: *mut CXorshift64State) -> f64 {
    let mut rng = xs64(s);
    let v = rng.f64();
    write_xs64(s, &rng);
    v
}

/// Advance state and return uniform f32 in [lo, hi).
#[no_mangle]
pub unsafe extern "C" fn mid_xorshift64_range_f32(
    s: *mut CXorshift64State, lo: f32, hi: f32,
) -> f32 {
    let mut rng = xs64(s);
    let v = rng.range_f32(lo, hi);
    write_xs64(s, &rng);
    v
}

/// Advance state and return uniform u32 in [lo, hi). lo must be < hi.
#[no_mangle]
pub unsafe extern "C" fn mid_xorshift64_range_u32(
    s: *mut CXorshift64State, lo: u32, hi: u32,
) -> u32 {
    let mut rng = xs64(s);
    let v = rng.range_u32(lo, hi);
    write_xs64(s, &rng);
    v
}

/// Advance state and return true with probability p (clamped to [0, 1]).
#[no_mangle]
pub unsafe extern "C" fn mid_xorshift64_bool_p(s: *mut CXorshift64State, p: f32) -> bool {
    let mut rng = xs64(s);
    let v = rng.bool_p(p);
    write_xs64(s, &rng);
    v
}

// ═══════════════════════════════════════════════════════════════════════════
//  PCG32 exports
// ═══════════════════════════════════════════════════════════════════════════

/// Create PCG32 state from seed and seq (stream selector).
/// Different seq values produce independent, non-overlapping sequences.
#[no_mangle]
pub extern "C" fn mid_pcg32_create(seed: u64, seq: u64) -> CPcg32State {
    let rng = Pcg32::new(seed, seq);
    let (state, inc) = rng.state();
    CPcg32State { state, inc }
}

/// Create PCG32 with default stream (seq = 1).
#[no_mangle]
pub extern "C" fn mid_pcg32_create_single_stream(seed: u64) -> CPcg32State {
    mid_pcg32_create(seed, 1)
}

/// Advance state and return next u32.
#[no_mangle]
pub unsafe extern "C" fn mid_pcg32_next_u32(s: *mut CPcg32State) -> u32 {
    let mut rng = pcg32(s);
    let v = rng.next_u32();
    write_pcg32(s, &rng);
    v
}

/// Advance state and return next u64 (two u32 calls).
#[no_mangle]
pub unsafe extern "C" fn mid_pcg32_next_u64(s: *mut CPcg32State) -> u64 {
    let mut rng = pcg32(s);
    let v = rng.next_u64();
    write_pcg32(s, &rng);
    v
}

/// Advance state and return uniform f32 in [0, 1).
#[no_mangle]
pub unsafe extern "C" fn mid_pcg32_f32(s: *mut CPcg32State) -> f32 {
    let mut rng = pcg32(s);
    let v = rng.f32();
    write_pcg32(s, &rng);
    v
}

/// Advance state and return uniform f64 in [0, 1).
#[no_mangle]
pub unsafe extern "C" fn mid_pcg32_f64(s: *mut CPcg32State) -> f64 {
    let mut rng = pcg32(s);
    let v = rng.f64();
    write_pcg32(s, &rng);
    v
}

/// Advance state and return uniform f32 in [lo, hi).
#[no_mangle]
pub unsafe extern "C" fn mid_pcg32_range_f32(s: *mut CPcg32State, lo: f32, hi: f32) -> f32 {
    let mut rng = pcg32(s);
    let v = rng.range_f32(lo, hi);
    write_pcg32(s, &rng);
    v
}

/// Advance state and return uniform u32 in [lo, hi). lo must be < hi.
#[no_mangle]
pub unsafe extern "C" fn mid_pcg32_range_u32(s: *mut CPcg32State, lo: u32, hi: u32) -> u32 {
    let mut rng = pcg32(s);
    let v = rng.range_u32(lo, hi);
    write_pcg32(s, &rng);
    v
}

/// Advance state and return true with probability p.
#[no_mangle]
pub unsafe extern "C" fn mid_pcg32_bool_p(s: *mut CPcg32State, p: f32) -> bool {
    let mut rng = pcg32(s);
    let v = rng.bool_p(p);
    write_pcg32(s, &rng);
    v
}

/// Advance the PCG32 generator by `delta` steps in O(log n).
/// Useful for splitting work across threads.
#[no_mangle]
pub unsafe extern "C" fn mid_pcg32_advance(s: *mut CPcg32State, delta: u64) {
    let mut rng = pcg32(s);
    rng.advance(delta);
    write_pcg32(s, &rng);
}
