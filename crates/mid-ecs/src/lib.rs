//! mid-ecs — Data-oriented ECS (Structure of Arrays layout)
//!
//! Target: 100 000+ entities per core at 60 Hz physics frequency.
//! Queries parallelised via rayon.
//!
//! Multiplayer-first: the sync module marks components for
//! mid-net replication from day one.

pub mod archetype;
pub mod component;
pub mod ffi;
pub mod query;
pub mod sync;
pub mod transform;
pub mod world;

// TEMPORARY — real-CI inlining-regression diagnostic, see this
// module's own doc comment. Delete this line together with
// src/diag_inline.rs once the investigation concludes.
mod diag_inline;

// Restored: world.rs now actually defines World (and Entity), as of the
// generational-arena/entity-allocation pass -- see world.rs's own doc
// comment. The comment that used to be here explained why this line was
// deliberately absent (it pointed at nothing, and that broke every build
// depending on this crate, including examples/headless-server, before
// that was caught); kept as a note for anyone re-deriving this crate's
// history, not because the bug risk still applies now that the type is
// real.
pub use archetype::ArchetypeId;
pub use component::{ComponentId, SparseShell};
pub use transform::{GlobalTransform, GlobalTransformLWC};
pub use world::{Entity, World};
