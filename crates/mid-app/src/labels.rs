// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-app.md, section "labels.rs"
// ============================================================================
//! The schedules an [`App`](crate::App) is made of, as labels.

/// The schedule that runs one frame: the stages below, in order. Every
/// `App` has it; its one system is the driver that runs them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Main;

/// Runs first in a frame: input, anything the rest of the frame reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct PreUpdate;

/// Runs second: game logic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Update;

/// Runs third: the network.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Net;

/// Runs fourth, on the fixed timestep: zero or more times a frame, each
/// run one step of `FixedTimestep::step_secs` long.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Physics;

/// Runs last in a frame: whatever follows from all of the above.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct PostUpdate;
