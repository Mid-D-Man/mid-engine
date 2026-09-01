// crates/mid-math/src/swizzle/mod.rs
//! Component-permutation traits (`.xy()`, `.xzy()`, `.wzyx()`, `.with_xy(rhs)`,
//! ...) for this crate's Vec2/Vec3/Vec4-family types across every numeric
//! family. See `engine.rs` for the trait/macro definitions themselves — that
//! file is shared and numeric-family-agnostic. Everything else here is one
//! file per numeric family, each just invoking `engine.rs`'s macros once per
//! concrete type in that family:
//!
//! - `f32.rs` — `Vec2`/`Vec3`/`Vec4` (Vec2 canonical, Vec3/Vec4 per backend)
//! - `f64.rs` — `DVec2`/`DVec3`/`DVec4` (DVec3 canonical, DVec2/DVec4 per backend)
//! - `int8.rs`/`int16.rs`/`int32.rs`/`int64.rs` — all always-scalar and
//!   canonical (no backend split, no `#[cfg(...)]` needed anywhere in them)
//! - wide/int + wide/float axis-shuffles: not yet added — queued, see
//!   `crates/mid-math/README.md`.
//!
//! `mod f32;` / `mod f64;` / etc. are private on purpose: they only exist to
//! run the macro invocations for their `impl Vec3Swizzles for ... { ... }`
//! blocks, which become visible everywhere in the crate the moment they're
//! compiled (that's how trait impls work in Rust) regardless of which
//! module physically contains the invocation — there's nothing else in any
//! of them worth reaching by path.
//!
//! (Named `engine`, not `core` — this crate uses `core::fmt`/`core::ops`
//! extensively elsewhere, so a local module named `core` isn't worth the
//! risk of colliding with the extern-prelude crate for zero actual benefit.)

pub mod engine;

mod f32;
mod f64;
mod int8;
mod int16;
mod int32;
mod int64;

pub use engine::{Vec2Swizzles, Vec3Swizzles, Vec4Swizzles};

