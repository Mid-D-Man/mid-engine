// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-platform.md, section "collections/"
// ============================================================================
//! `HashMap` and `HashSet` with [`FixedHasher`] as the default hasher.
//!
//! * With the `std` feature (on by default), [`HashMap`] and [`HashSet`] are
//!   aliases for `std::collections`' own types. `std`'s map is hashbrown
//!   underneath, so this is hashbrown's speed and quality with no dependency
//!   and no unsafe code written here.
//! * Without `std` (but with `alloc`), they are aliases for
//!   [`OpenHashMap`] / [`OpenHashSet`], this crate's own safe-code tables.
//!
//! The two implementations share one source-compatible API, so code written
//! against `collections::HashMap` builds in either configuration. That API
//! is the subset of `std`'s map both have, with `std`'s signatures. Two
//! consequences of sharing `std`'s shape:
//!
//! * There is no `new()`. `std` only provides it for its default
//!   `RandomState`, and the default here is [`FixedHasher`]. Construct with
//!   `HashMap::default()` (the type must be known from context or an
//!   annotation: `let m: HashMap<K, V> = HashMap::default();`),
//!   `HashMap::with_hasher(FixedHasher)`, or
//!   `HashMap::with_capacity_and_hasher(n, FixedHasher)`.
//! * `FixedHasher` is deterministic and not DoS-resistant. Keys an attacker
//!   chooses belong in `std::collections::HashMap<K, V, RandomState>`
//!   instead; see `hash`.
//!
//! [`OpenHashMap`] and [`OpenHashSet`] stay reachable under their own names
//! in every configuration, so they can be tested and measured against
//! `std`'s map with `std` on.

pub mod open_map;
pub mod open_set;
mod raw;

#[cfg(test)]
mod tests;

pub use open_map::OpenHashMap;
pub use open_set::OpenHashSet;

use crate::hash::FixedHasher;

/// A hash map with [`FixedHasher`] as its default hasher: `std`'s map when
/// the `std` feature is on, [`OpenHashMap`] otherwise.
#[cfg(feature = "std")]
pub type HashMap<K, V, S = FixedHasher> = std::collections::HashMap<K, V, S>;

/// A hash set with [`FixedHasher`] as its default hasher: `std`'s set when
/// the `std` feature is on, [`OpenHashSet`] otherwise.
#[cfg(feature = "std")]
pub type HashSet<T, S = FixedHasher> = std::collections::HashSet<T, S>;

/// A hash map with [`FixedHasher`] as its default hasher: `std`'s map when
/// the `std` feature is on, [`OpenHashMap`] otherwise.
#[cfg(not(feature = "std"))]
pub type HashMap<K, V, S = FixedHasher> = OpenHashMap<K, V, S>;

/// A hash set with [`FixedHasher`] as its default hasher: `std`'s set when
/// the `std` feature is on, [`OpenHashSet`] otherwise.
#[cfg(not(feature = "std"))]
pub type HashSet<T, S = FixedHasher> = OpenHashSet<T, S>;

/// The entry types of whichever map [`HashMap`] currently is.
pub mod hash_map {
    #[cfg(not(feature = "std"))]
    pub use super::open_map::{Entry, OccupiedEntry, VacantEntry};
    #[cfg(feature = "std")]
    pub use std::collections::hash_map::{Entry, OccupiedEntry, VacantEntry};
}
