//! mid-ecs — Data-oriented ECS (Structure of Arrays layout)
//!
//! Target: 100 000+ entities per core at 60 Hz physics frequency.
//! Queries parallelised via rayon.
//!
//! Multiplayer-first: the sync module marks components for
//! mid-net replication from day one.

pub mod world;
pub mod archetype;
pub mod query;
pub mod sync;
pub mod ffi;

// `pub use world::World;` intentionally NOT here yet -- world.rs is still
// an empty auto-generated stub (see its own doc comment), no `World`
// type exists to re-export. This line was here pointing at nothing,
// which is what broke every build depending on this crate (including
// `examples/headless-server`, since before this fix) -- confirmed via a
// real build log, not assumed. Add it back once `world.rs` actually
// defines `World`, during the real mid-ecs build pass.
