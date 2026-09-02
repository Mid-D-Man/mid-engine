// crates/mid-math/src/swizzle/mod.rs
//! Component-permutation traits (`.xy()`, `.xzy()`, `.wzyx()`, `.with_xy(rhs)`,
//! ...) for this crate's Vec2/Vec3/Vec4-family types across every numeric
//! family, plus axis-swizzle and lane-shuffle for the wide SIMD types.
//!
//! - `engine.rs` — the shared `Vec2Swizzles`/`Vec3Swizzles`/`Vec4Swizzles`
//!   traits + macros (numeric-family-agnostic).
//! - `f32.rs` / `f64.rs` / `int8.rs` / `int16.rs` / `int32.rs` / `int64.rs` —
//!   one file per numeric family, each just invoking `engine.rs`'s macros
//!   once per concrete type in that family.
//! - `wide_axis_engine.rs` — `Vec3AxisSwizzle` (same-width-only, for the SoA
//!   `Vec3x4`/`Vec3x8` types — no `Vec2x4`/`Vec4x4` exists to narrow/widen
//!   into, so this is a genuinely separate, smaller trait, not a reuse of
//!   `Vec3Swizzles`). No `QuatX4` — matches this crate's scalar scope
//!   (`Quat`/`DQuat` never got `Vec4Swizzles` either).
//! - `wide_lane_engine.rs` — `LaneShuffle4`/`8`/`16`/`32` (lane permutation
//!   for the opaque single-register wide types — `f32x4`, `i32x4`, etc. —
//!   which have no x/y/z/w fields at all, so this isn't "swizzle" in the
//!   axis sense, it's reordering which lane holds what value).
//! - `wide_float.rs` / `wide_int.rs` — invocations for the two engines above,
//!   across every wide backend (sse2/neon/wasm/scalar/avx2).
//!
//! Every `mod ...;` below is private on purpose: they only exist to run
//! macro invocations for `impl SomeTrait for ... { ... }` blocks, which
//! become visible everywhere in the crate the moment they're compiled
//! (that's how trait impls work in Rust) regardless of which module
//! physically contains the invocation — there's nothing else in any of them
//! worth reaching by path.
//!
//! (The shared-engine file is named `engine`, not `core` — this crate uses
//! `core::fmt`/`core::ops` extensively elsewhere, so a local module named
//! `core` isn't worth the risk of colliding with the extern-prelude crate
//! for zero actual benefit.)

pub mod engine;
pub mod wide_axis_engine;
pub mod wide_lane_engine;

mod f32;
mod f64;
mod int8;
mod int16;
mod int32;
mod int64;
mod wide_float;
mod wide_int;

pub use engine::{Vec2Swizzles, Vec3Swizzles, Vec4Swizzles};
pub use wide_axis_engine::Vec3AxisSwizzle;
pub use wide_lane_engine::{LaneShuffle4, LaneShuffle8, LaneShuffle16, LaneShuffle32};
