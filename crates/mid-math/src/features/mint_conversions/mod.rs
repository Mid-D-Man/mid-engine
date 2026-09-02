// crates/mid-math/src/features/mint_conversions/mod.rs
//! mint (math interoperability types) conversions, enabled by the `mint`
//! Cargo feature. See `vectors.rs` (every Vec2/3/4-family type, all 10
//! numeric families) and `f32.rs`/`f64.rs` (Quat + Mat2/3/4, float only —
//! matches the real `mint`/glam scope: mint has no integer quaternion or
//! matrix types to convert to/from).
//!
//! Both submodules are private — same reasoning as `swizzle/`'s per-family
//! files: they only exist to bring `impl From<...> for ...`/`impl IntoMint`
//! blocks into the crate, which become visible everywhere the moment
//! they're compiled regardless of which module physically contains them.

mod vectors;
mod f32;
mod f64;
