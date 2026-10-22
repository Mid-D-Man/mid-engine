//! mid-ecs — Data-oriented ECS (Structure of Arrays layout)
//!
//! Target: 100 000+ entities per core at 60 Hz physics frequency.
//! Queries parallelised via rayon (`Query::par_iter`, `QueryBatch`); on
//! wasm32 the same batches run on the calling thread.
//!
//! Multiplayer-first: the sync module marks components for
//! mid-net replication from day one.

pub mod access;
pub mod archetype;
pub mod batching;
pub mod commands;
pub mod component;
pub mod data_filter;
pub mod ffi;
pub mod filter;
mod hash;
mod interior;
pub mod query;
pub mod query_data;
mod removal;
pub mod resource;
mod scratch;
pub mod sync;
pub mod system;
pub mod system_param;
pub mod tick;
pub mod transform;
pub mod world;
pub mod world_cell;

// Restored: world.rs now actually defines World (and Entity), as of the
// generational-arena/entity-allocation pass -- see world.rs's own doc
// comment. The comment that used to be here explained why this line was
// deliberately absent (it pointed at nothing, and that broke every build
// depending on this crate, including examples/headless-server, before
// that was caught); kept as a note for anyone re-deriving this crate's
// history, not because the bug risk still applies now that the type is
// real.
pub use access::{Access, AccessConflicts, FilteredAccess, FilteredAccessSet};
pub use archetype::{ArchetypeId, ComponentFfiError, FfiSpanMut, QueryBatch};
pub use batching::BatchingStrategy;
pub use commands::{CommandQueue, Commands, EntityCommands};
pub use component::{ComponentId, SparseShell};
pub use data_filter::DataFilter;
pub use filter::{Or, QueryFilter, With, Without};
pub use query_data::{Added, Changed, Has, QueryData, ReadOnlyQueryData};
pub use resource::{ResourceFfiError, ResourceId};
pub use system::{
    ExclusiveFunctionSystem, FunctionSystem, IntoSystem, System, SystemParamFunction,
};
pub use system_param::{
    Local, Query, QueryParIter, RemovedComponents, Res, ResMut, SystemMeta, SystemParam,
    SystemParamItem, SystemState,
};
pub use tick::{ChangeTracker, Mut, Ref, Tick};
pub use transform::{GlobalTransform, GlobalTransformLWC};
pub use world::{Entity, World};
pub use world_cell::WorldCell;
