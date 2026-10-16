// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-platform.md, section "collections/"
// ============================================================================
//! [`OpenHashSet`]: a set over [`OpenHashMap`] with `()` values.
//!
//! What `collections::HashSet` is when the `std` feature is off. Same
//! arrangement as the map: available under its own name with `std` on too.

use core::borrow::Borrow;
use core::fmt;
use core::hash::{BuildHasher, Hash};
use core::iter::FusedIterator;

use super::open_map::{self, OpenHashMap};
use crate::hash::FixedHasher;

/// A hash set using open addressing with linear probing, written entirely in
/// safe code. See [`OpenHashMap`] for the construction and iteration-order
/// notes; they apply here unchanged.
///
/// # Example
///
/// ```
/// use mid_platform::collections::OpenHashSet;
///
/// let mut set: OpenHashSet<u32> = OpenHashSet::default();
/// assert!(set.insert(7));
/// assert!(!set.insert(7));
/// assert!(set.contains(&7));
/// assert!(set.remove(&7));
/// assert!(set.is_empty());
/// ```
pub struct OpenHashSet<T, S = FixedHasher> {
    map: OpenHashMap<T, (), S>,
}

impl<T, S> OpenHashSet<T, S> {
    /// An empty set that uses `hash_builder` to hash elements. Allocates
    /// nothing until the first insert.
    #[inline]
    pub const fn with_hasher(hash_builder: S) -> Self {
        Self {
            map: OpenHashMap::with_hasher(hash_builder),
        }
    }

    /// An empty set that can hold at least `capacity` elements before it
    /// reallocates.
    pub fn with_capacity_and_hasher(capacity: usize, hash_builder: S) -> Self {
        Self {
            map: OpenHashMap::with_capacity_and_hasher(capacity, hash_builder),
        }
    }

    /// The set's hasher builder.
    pub fn hasher(&self) -> &S {
        self.map.hasher()
    }

    /// Number of elements.
    #[inline]
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Whether the set has no elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Elements the set can hold without growing.
    pub fn capacity(&self) -> usize {
        self.map.capacity()
    }

    /// Removes every element, keeping the allocation.
    pub fn clear(&mut self) {
        self.map.clear();
    }

    /// Iterator over the elements, in unspecified order.
    pub fn iter(&self) -> Iter<'_, T> {
        Iter {
            inner: self.map.keys(),
        }
    }

    /// Keeps only the elements `keep` returns `true` for.
    pub fn retain<F: FnMut(&T) -> bool>(&mut self, mut keep: F) {
        self.map.retain(|element, _| keep(element));
    }
}

impl<T, S> OpenHashSet<T, S>
where
    T: Eq + Hash,
    S: BuildHasher,
{
    /// Makes room for at least `additional` more elements.
    pub fn reserve(&mut self, additional: usize) {
        self.map.reserve(additional);
    }

    /// Adds `value`. Returns `true` if it was not already present. An
    /// existing equal element is left as it is.
    pub fn insert(&mut self, value: T) -> bool {
        match self.map.entry(value) {
            open_map::Entry::Occupied(_) => false,
            open_map::Entry::Vacant(entry) => {
                entry.insert(());
                true
            }
        }
    }

    /// Whether `value` is in the set.
    pub fn contains<Q>(&self, value: &Q) -> bool
    where
        T: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.map.contains_key(value)
    }

    /// The stored element equal to `value`, if any.
    pub fn get<Q>(&self, value: &Q) -> Option<&T>
    where
        T: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.map.get_key_value(value).map(|(element, _)| element)
    }

    /// Removes `value`. Returns `true` if it was present.
    pub fn remove<Q>(&mut self, value: &Q) -> bool
    where
        T: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.map.remove(value).is_some()
    }

    /// Removes and returns the stored element equal to `value`, if any.
    pub fn take<Q>(&mut self, value: &Q) -> Option<T>
    where
        T: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.map.remove_entry(value).map(|(element, _)| element)
    }

    /// Elements in `self` that are not in `other`.
    pub fn difference<'a>(&'a self, other: &'a Self) -> impl Iterator<Item = &'a T> + 'a {
        self.iter().filter(move |element| !other.contains(*element))
    }

    /// Elements in both `self` and `other`.
    pub fn intersection<'a>(&'a self, other: &'a Self) -> impl Iterator<Item = &'a T> + 'a {
        self.iter().filter(move |element| other.contains(*element))
    }

    /// Elements in `self`, `other`, or both, each once.
    pub fn union<'a>(&'a self, other: &'a Self) -> impl Iterator<Item = &'a T> + 'a {
        self.iter().chain(other.difference(self))
    }

    /// Whether the two sets have no element in common.
    pub fn is_disjoint(&self, other: &Self) -> bool {
        self.iter().all(|element| !other.contains(element))
    }

    /// Whether every element of `self` is in `other`.
    pub fn is_subset(&self, other: &Self) -> bool {
        self.len() <= other.len() && self.iter().all(|element| other.contains(element))
    }

    /// Whether every element of `other` is in `self`.
    pub fn is_superset(&self, other: &Self) -> bool {
        other.is_subset(self)
    }
}

/// Iterator over set elements. Built by [`OpenHashSet::iter`].
pub struct Iter<'a, T> {
    inner: open_map::Keys<'a, T, ()>,
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;

    #[inline]
    fn next(&mut self) -> Option<&'a T> {
        self.inner.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<T> ExactSizeIterator for Iter<'_, T> {}
impl<T> FusedIterator for Iter<'_, T> {}

impl<T> Clone for Iter<'_, T> {
    fn clone(&self) -> Self {
        Iter {
            inner: self.inner.clone(),
        }
    }
}

/// Owning iterator over set elements. Built by `OpenHashSet::into_iter`.
pub struct IntoIter<T> {
    inner: open_map::IntoKeys<T, ()>,
}

impl<T> Iterator for IntoIter<T> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        self.inner.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<T> ExactSizeIterator for IntoIter<T> {}
impl<T> FusedIterator for IntoIter<T> {}

impl<T, S> IntoIterator for OpenHashSet<T, S> {
    type Item = T;
    type IntoIter = IntoIter<T>;

    fn into_iter(self) -> IntoIter<T> {
        IntoIter {
            inner: self.map.into_keys(),
        }
    }
}

impl<'a, T, S> IntoIterator for &'a OpenHashSet<T, S> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Iter<'a, T> {
        self.iter()
    }
}

impl<T, S: Default> Default for OpenHashSet<T, S> {
    fn default() -> Self {
        Self::with_hasher(S::default())
    }
}

impl<T: Clone, S: Clone> Clone for OpenHashSet<T, S> {
    fn clone(&self) -> Self {
        Self {
            map: self.map.clone(),
        }
    }
}

impl<T: fmt::Debug, S> fmt::Debug for OpenHashSet<T, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_set().entries(self.iter()).finish()
    }
}

impl<T, S> PartialEq for OpenHashSet<T, S>
where
    T: Eq + Hash,
    S: BuildHasher,
{
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len() && self.iter().all(|element| other.contains(element))
    }
}

impl<T, S> Eq for OpenHashSet<T, S>
where
    T: Eq + Hash,
    S: BuildHasher,
{
}

impl<T, S> Extend<T> for OpenHashSet<T, S>
where
    T: Eq + Hash,
    S: BuildHasher,
{
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        self.map
            .extend(iter.into_iter().map(|element| (element, ())));
    }
}

impl<T, S> FromIterator<T> for OpenHashSet<T, S>
where
    T: Eq + Hash,
    S: BuildHasher + Default,
{
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut set = Self::default();
        set.extend(iter);
        set
    }
}
