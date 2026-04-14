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

pub use world::World;
