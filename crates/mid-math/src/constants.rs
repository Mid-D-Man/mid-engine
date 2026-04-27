// crates/mid-math/src/constants.rs
//! Global f32 constants for mid-math.
//!
//! Separated from lib.rs so they can be imported by internal modules
//! without circular dependencies, and exported cleanly to users.

pub const PI:            f32 = core::f32::consts::PI;
pub const TAU:           f32 = core::f32::consts::TAU;
pub const FRAC_PI_2:     f32 = core::f32::consts::FRAC_PI_2;
/// π/3 = 60° in radians. Common camera FOV, hexagonal geometry.
pub const FRAC_PI_3:     f32 = core::f32::consts::FRAC_PI_3;
/// π/4 = 45° in radians. Common default FOV, isometric angle.
pub const FRAC_PI_4:     f32 = core::f32::consts::FRAC_PI_4;
/// π/6 = 30° in radians. Common in hexagonal grids and lighting rigs.
pub const FRAC_PI_6:     f32 = core::f32::consts::FRAC_PI_6;
/// 1/π — used in lighting normalisation (Lambert, Blinn-Phong).
pub const FRAC_1_PI:     f32 = core::f32::consts::FRAC_1_PI;
/// √2 ≈ 1.4142. Length of a unit-square diagonal.
pub const SQRT_2:        f32 = core::f32::consts::SQRT_2;
/// 1/√2 ≈ 0.7071. Normalised 2D diagonal, common in trig.
pub const FRAC_1_SQRT_2: f32 = core::f32::consts::FRAC_1_SQRT_2;
pub const DEG2RAD:       f32 = PI / 180.0;
pub const RAD2DEG:       f32 = 180.0 / PI;
/// Epsilon for approximate float comparisons.
pub const EPSILON:       f32 = 1e-6;
