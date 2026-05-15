// crates/mid-math/src/fixed/mod.rs
//! Deterministic fixed-point math for network-synchronised simulation.
//!
//! All arithmetic is pure integer — no floating-point on any platform.
//! This guarantees bit-identical results on x86_64, aarch64, and wasm32,
//! eliminating the desync that floating-point non-determinism causes in
//! authoritative multiplayer simulations.
//!
//! # Usage pattern
//! ```
//! // Simulation: fixed-point only
//! let pos = FixedVec3_16::from_i32(10, 0, 0);
//! let vel = FixedVec3_16::from_f32(1.5, 0.0, 0.0); // f32 only at boundary
//! let new_pos = pos + vel.scale(Fixed16::from_f32(dt));
//!
//! // Rendering boundary: convert once
//! let render_pos = new_pos.to_vec3();
//! ```

mod fixed;
mod vec2;
mod vec3;

pub use fixed::Fixed;
pub use vec2::FixedVec2;
pub use vec3::FixedVec3;

// ── Type aliases ──────────────────────────────────────────────────────────────

/// 8 fractional bits — 1/256 resolution, range ≈ ±36 billion units.
/// Good for: large world coordinates where sub-unit precision is enough.
pub type Fixed8 = Fixed<8>;

/// 12 fractional bits — 1/4096 resolution, range ≈ ±2.25 billion units.
/// Good for: physics velocities, angular values.
pub type Fixed12 = Fixed<12>;

/// 16 fractional bits — 1/65536 resolution, range ≈ ±140 thousand units.
/// Good for: sub-millimetre precision in bounded worlds.
pub type Fixed16 = Fixed<16>;

/// 2D fixed-point vector, 8 fractional bits.
pub type Fixed8Vec2  = FixedVec2<8>;
/// 2D fixed-point vector, 12 fractional bits.
pub type Fixed12Vec2 = FixedVec2<12>;
/// 2D fixed-point vector, 16 fractional bits.
pub type Fixed16Vec2 = FixedVec2<16>;

/// 3D fixed-point vector, 8 fractional bits.
pub type Fixed8Vec3  = FixedVec3<8>;
/// 3D fixed-point vector, 12 fractional bits.
pub type Fixed12Vec3 = FixedVec3<12>;
/// 3D fixed-point vector, 16 fractional bits.
pub type Fixed16Vec3 = FixedVec3<16>;

// Legacy shorter names — kept for ergonomics in test/bench code.
pub type FixedVec2_8  = FixedVec2<8>;
pub type FixedVec2_12 = FixedVec2<12>;
pub type FixedVec2_16 = FixedVec2<16>;
pub type FixedVec3_8  = FixedVec3<8>;
pub type FixedVec3_12 = FixedVec3<12>;
pub type FixedVec3_16 = FixedVec3<16>;
