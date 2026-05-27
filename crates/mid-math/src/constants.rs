// crates/mid-math/src/constants.rs
//! Global f32 constants for mid-math.
//!
//! Separated from lib.rs so they can be imported by internal modules
//! without circular dependencies, and exported cleanly to users.

pub const PI:             f32 = core::f32::consts::PI;
pub const TAU:            f32 = core::f32::consts::TAU;
pub const FRAC_PI_2:      f32 = core::f32::consts::FRAC_PI_2;
/// π/3 = 60° in radians. Common camera FOV, hexagonal geometry.
pub const FRAC_PI_3:      f32 = core::f32::consts::FRAC_PI_3;
/// π/4 = 45° in radians. Common default FOV, isometric angle.
pub const FRAC_PI_4:      f32 = core::f32::consts::FRAC_PI_4;
/// π/6 = 30° in radians. Common in hexagonal grids and lighting rigs.
pub const FRAC_PI_6:      f32 = core::f32::consts::FRAC_PI_6;
/// π/8 = 22.5° in radians.
pub const FRAC_PI_8:      f32 = core::f32::consts::FRAC_PI_8;
/// 1/π — used in lighting normalisation (Lambert, Blinn-Phong).
pub const FRAC_1_PI:      f32 = core::f32::consts::FRAC_1_PI;
/// √2 ≈ 1.4142. Length of a unit-square diagonal.
pub const SQRT_2:         f32 = core::f32::consts::SQRT_2;
/// 1/√2 ≈ 0.7071. Normalised 2D diagonal, common in trig.
pub const FRAC_1_SQRT_2:  f32 = core::f32::consts::FRAC_1_SQRT_2;
/// √3 ≈ 1.7321. Length of a unit-cube space diagonal divided by √3;
/// appears in hexagonal grid math and 3D diagonals.
pub const SQRT_3:         f32 = 1.732_050_8_f32;
/// 1/√3 ≈ 0.5774. Normalised 3D diagonal direction component.
pub const FRAC_1_SQRT_3:  f32 = 0.577_350_27_f32;
/// Euler's number e ≈ 2.7183.
pub const E:              f32 = core::f32::consts::E;
/// ln(2) ≈ 0.6931. Used in exponential falloff / octave scaling.
pub const LN_2:           f32 = core::f32::consts::LN_2;
/// log₂(e) ≈ 1.4427. Conversion factor between natural and binary log.
pub const LOG2_E:         f32 = core::f32::consts::LOG2_E;
/// Golden ratio φ ≈ 1.6180. Useful for Fibonacci sphere point distributions.
pub const GOLDEN_RATIO:   f32 = 1.618_033_9_f32;
pub const DEG2RAD:        f32 = PI / 180.0;
pub const RAD2DEG:        f32 = 180.0 / PI;
/// Epsilon for approximate float comparisons.
pub const EPSILON:        f32 = 1e-6;
/// Positive infinity. Useful as an initial "closest distance" sentinel.
pub const INF:            f32 = f32::INFINITY;
/// Negative infinity.
pub const NEG_INF:        f32 = f32::NEG_INFINITY;
/// Not-a-Number. Use `v.is_nan()` to test; NaN != NaN by definition.
pub const NAN:            f32 = f32::NAN;
