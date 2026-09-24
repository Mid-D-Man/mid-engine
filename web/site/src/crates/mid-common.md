# mid-common

Shared types and traits used across the workspace — `EntityId`, `TickId`,
network-sync-related types. Depends on `mid-math`.

Kept deliberately thin: helper crates (`mid-app`, `mid-time`, and others)
live as their own workspace members rather than being folded in here, so
this crate doesn't accumulate every cross-cutting concern by default.

**Status:** in progress.
