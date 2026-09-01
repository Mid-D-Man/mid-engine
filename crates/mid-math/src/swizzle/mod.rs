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
//! - narrow int (i8/u8/i16/u16/i32/u32/i64/u64) and wide/int + wide/float:
//!   not yet added — queued, see `crates/mid-math/README.md`.
//!
//! `mod f32;` / `mod f64;` are private on purpose: they only exist to run
//! the macro invocations for their `impl Vec3Swizzles for ... { ... }`
//! blocks, which become visible everywhere in the crate the moment they're
//! compiled (that's how trait impls work in Rust) regardless of which
//! module physically contains the invocation — there's nothing else in
//! either file worth reaching by path.
//!
//! (Named `engine`, not `core` — this crate uses `core::fmt`/`core::ops`
//! extensively elsewhere, so a local module named `core` isn't worth the
//! risk of colliding with the extern-prelude crate for zero actual benefit.)

pub mod engine;

mod f32;
mod f64;

pub use engine::{Vec2Swizzles, Vec3Swizzles, Vec4Swizzles};
