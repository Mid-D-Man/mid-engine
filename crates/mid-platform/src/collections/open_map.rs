// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-platform.md, section "collections/"
// ============================================================================
//! [`OpenHashMap`]: a hash map in safe code, linear probing, no tombstones.
//!
//! This is what `collections::HashMap` is when the `std` feature is off. With
//! `std` on, `collections::HashMap` is `std`'s own map instead and this type
//! is still available under its own name, so it is tested, benchmarked, and
//! usable in both configurations.
//!
//! The method set is the subset of `std::collections::HashMap` that both
//! have, with `std`'s signatures, so code written against
//! `collections::HashMap` compiles unchanged in either configuration.

use core::borrow::Borrow;
use core::fmt;
use core::hash::{BuildHasher, Hash};
use core::iter::FusedIterator;
use core::mem;
use core::ops::Index;

use super::raw::{Bucket, RawTable, Slot};
use crate::hash::FixedHasher;

/// A hash map using open addressing with linear probing, written entirely in
/// safe code.
///
/// Iteration order is unspecified and changes when the table grows. Like
/// `std`'s map, there is no `new()`: the default hasher is
/// [`FixedHasher`], not a random one, so build one with
/// [`Default::default`], [`with_hasher`](Self::with_hasher), or
/// [`with_capacity_and_hasher`](Self::with_capacity_and_hasher).
///
/// # Example
///
/// ```
/// use mid_platform::collections::OpenHashMap;
///
/// let mut map: OpenHashMap<&str, u32> = OpenHashMap::default();
/// map.insert("a", 1);
/// *map.entry("a").or_insert(0) += 10;
/// assert_eq!(map.get("a"), Some(&11));
/// assert_eq!(map.remove("a"), Some(11));
/// assert!(map.is_empty());
/// ```
pub struct OpenHashMap<K, V, S = FixedHasher> {
    table: RawTable<K, V>,
    hash_builder: S,
}

impl<K, V, S> OpenHashMap<K, V, S> {
    /// An empty map that uses `hash_builder` to hash keys. Allocates nothing
    /// until the first insert.
    #[inline]
    pub const fn with_hasher(hash_builder: S) -> Self {
        Self {
            table: RawTable::new(),
            hash_builder,
        }
    }

    /// An empty map that can hold at least `capacity` entries before it
    /// reallocates.
    pub fn with_capacity_and_hasher(capacity: usize, hash_builder: S) -> Self {
        Self {
            table: RawTable::with_capacity(capacity),
            hash_builder,
        }
    }

    /// The map's hasher builder.
    pub fn hasher(&self) -> &S {
        &self.hash_builder
    }

    /// Number of entries.
    #[inline]
    pub fn len(&self) -> usize {
        self.table.len()
    }

    /// Whether the map has no entries.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.table.len() == 0
    }

    /// Entries the map can hold without growing.
    pub fn capacity(&self) -> usize {
        self.table.capacity()
    }

    /// Removes every entry, keeping the allocation.
    pub fn clear(&mut self) {
        self.table.clear();
    }

    /// Iterator over `(&K, &V)`, in unspecified order.
    pub fn iter(&self) -> Iter<'_, K, V> {
        Iter {
            inner: self.table.slots().iter(),
            remaining: self.table.len(),
        }
    }

    /// Iterator over `(&K, &mut V)`, in unspecified order.
    pub fn iter_mut(&mut self) -> IterMut<'_, K, V> {
        let remaining = self.table.len();
        IterMut {
            inner: self.table.slots_mut().iter_mut(),
            remaining,
        }
    }

    /// Iterator over the keys.
    pub fn keys(&self) -> Keys<'_, K, V> {
        Keys { inner: self.iter() }
    }

    /// Iterator over the values.
    pub fn values(&self) -> Values<'_, K, V> {
        Values { inner: self.iter() }
    }

    /// Iterator over mutable references to the values.
    pub fn values_mut(&mut self) -> ValuesMut<'_, K, V> {
        ValuesMut {
            inner: self.iter_mut(),
        }
    }

    /// Consuming iterator over the keys.
    pub fn into_keys(self) -> IntoKeys<K, V> {
        IntoKeys {
            inner: self.into_iter(),
        }
    }

    /// Consuming iterator over the values.
    pub fn into_values(self) -> IntoValues<K, V> {
        IntoValues {
            inner: self.into_iter(),
        }
    }

    /// Keeps only the entries `keep` returns `true` for. Visits entries in
    /// unspecified order and may rebuild the table at its current size.
    pub fn retain<F: FnMut(&K, &mut V) -> bool>(&mut self, keep: F) {
        self.table.retain(keep);
    }

    #[cfg(test)]
    pub(crate) fn longest_cluster(&self) -> usize {
        self.table.longest_cluster()
    }
}

impl<K, V, S> OpenHashMap<K, V, S>
where
    K: Eq + Hash,
    S: BuildHasher,
{
    #[inline]
    fn hash_of<Q: Hash + ?Sized>(&self, key: &Q) -> u64 {
        self.hash_builder.hash_one(key)
    }

    #[inline]
    fn find_index<Q>(&self, key: &Q) -> Option<usize>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        if self.table.len() == 0 {
            return None;
        }
        let hash = self.hash_of(key);
        self.table.find(hash, |candidate: &K| {
            let candidate: &Q = candidate.borrow();
            candidate == key
        })
    }

    /// Makes room for at least `additional` more entries.
    pub fn reserve(&mut self, additional: usize) {
        self.table.reserve(additional);
    }

    /// Inserts a key-value pair. If the key was present, its value is
    /// replaced and the old value returned; the stored key is kept.
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        let hash = self.hash_of(&key);
        match self.table.probe(hash, |candidate: &K| *candidate == key) {
            Slot::Found(index) => {
                Some(mem::replace(&mut self.table.bucket_mut(index).value, value))
            }
            Slot::Empty(empty) => {
                self.table.place(hash, key, value, empty);
                None
            }
        }
    }

    /// The value for `key`, if present.
    pub fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let index = self.find_index(key)?;
        Some(&self.table.bucket(index).value)
    }

    /// The stored key and value for `key`, if present.
    pub fn get_key_value<Q>(&self, key: &Q) -> Option<(&K, &V)>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let index = self.find_index(key)?;
        let bucket = self.table.bucket(index);
        Some((&bucket.key, &bucket.value))
    }

    /// A mutable reference to the value for `key`, if present.
    pub fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let index = self.find_index(key)?;
        Some(&mut self.table.bucket_mut(index).value)
    }

    /// Whether `key` is present.
    pub fn contains_key<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.find_index(key).is_some()
    }

    /// Removes `key`, returning its value if it was present.
    pub fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.remove_entry(key).map(|(_, value)| value)
    }

    /// Removes `key`, returning the stored key and its value if it was
    /// present.
    pub fn remove_entry<Q>(&mut self, key: &Q) -> Option<(K, V)>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let index = self.find_index(key)?;
        let bucket = self.table.remove_at(index);
        Some((bucket.key, bucket.value))
    }

    /// The entry for `key`, for in-place get-or-insert.
    pub fn entry(&mut self, key: K) -> Entry<'_, K, V> {
        let hash = self.hash_of(&key);
        match self.table.probe(hash, |candidate: &K| *candidate == key) {
            Slot::Found(index) => Entry::Occupied(OccupiedEntry {
                table: &mut self.table,
                index,
            }),
            Slot::Empty(empty) => Entry::Vacant(VacantEntry {
                table: &mut self.table,
                hash,
                key,
                empty,
            }),
        }
    }
}

// ---------------------------------------------------------------------------
// Entry API
// ---------------------------------------------------------------------------

/// A view into one entry of an [`OpenHashMap`], present or absent. Built by
/// [`OpenHashMap::entry`].
pub enum Entry<'a, K, V> {
    /// The key is present.
    Occupied(OccupiedEntry<'a, K, V>),
    /// The key is absent.
    Vacant(VacantEntry<'a, K, V>),
}

/// An entry whose key is present. Part of [`Entry`].
pub struct OccupiedEntry<'a, K, V> {
    table: &'a mut RawTable<K, V>,
    index: usize,
}

/// An entry whose key is absent. Part of [`Entry`].
pub struct VacantEntry<'a, K, V> {
    table: &'a mut RawTable<K, V>,
    hash: u64,
    key: K,
    empty: usize,
}

impl<'a, K, V> Entry<'a, K, V> {
    /// The entry's value, inserting `default` first if it was absent.
    pub fn or_insert(self, default: V) -> &'a mut V {
        match self {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => entry.insert(default),
        }
    }

    /// The entry's value, inserting the result of `default` first if it was
    /// absent.
    pub fn or_insert_with<F: FnOnce() -> V>(self, default: F) -> &'a mut V {
        match self {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => entry.insert(default()),
        }
    }

    /// Like [`or_insert_with`](Self::or_insert_with), passing the key to
    /// `default`.
    pub fn or_insert_with_key<F: FnOnce(&K) -> V>(self, default: F) -> &'a mut V {
        match self {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => {
                let value = default(entry.key());
                entry.insert(value)
            }
        }
    }

    /// The entry's key.
    pub fn key(&self) -> &K {
        match self {
            Entry::Occupied(entry) => entry.key(),
            Entry::Vacant(entry) => entry.key(),
        }
    }

    /// Runs `f` on the value if the entry is occupied, then returns the
    /// entry.
    #[must_use]
    pub fn and_modify<F: FnOnce(&mut V)>(self, f: F) -> Self {
        match self {
            Entry::Occupied(mut entry) => {
                f(entry.get_mut());
                Entry::Occupied(entry)
            }
            Entry::Vacant(entry) => Entry::Vacant(entry),
        }
    }
}

impl<'a, K, V: Default> Entry<'a, K, V> {
    /// The entry's value, inserting `V::default()` first if it was absent.
    pub fn or_default(self) -> &'a mut V {
        match self {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => entry.insert(V::default()),
        }
    }
}

impl<'a, K, V> OccupiedEntry<'a, K, V> {
    /// The stored key.
    pub fn key(&self) -> &K {
        &self.table.bucket(self.index).key
    }

    /// The value.
    pub fn get(&self) -> &V {
        &self.table.bucket(self.index).value
    }

    /// The value, mutably.
    pub fn get_mut(&mut self) -> &mut V {
        &mut self.table.bucket_mut(self.index).value
    }

    /// The value, mutably, for as long as the map is borrowed.
    pub fn into_mut(self) -> &'a mut V {
        &mut self.table.bucket_mut(self.index).value
    }

    /// Replaces the value, returning the old one.
    pub fn insert(&mut self, value: V) -> V {
        mem::replace(self.get_mut(), value)
    }

    /// Removes the entry, returning its value.
    pub fn remove(self) -> V {
        self.remove_entry().1
    }

    /// Removes the entry, returning the stored key and value.
    pub fn remove_entry(self) -> (K, V) {
        let bucket = self.table.remove_at(self.index);
        (bucket.key, bucket.value)
    }
}

impl<'a, K, V> VacantEntry<'a, K, V> {
    /// The key that would be inserted.
    pub fn key(&self) -> &K {
        &self.key
    }

    /// Takes the key back without inserting.
    pub fn into_key(self) -> K {
        self.key
    }

    /// Inserts `value` and returns a mutable reference to it.
    pub fn insert(self, value: V) -> &'a mut V {
        let table = self.table;
        let index = table.place(self.hash, self.key, value, self.empty);
        &mut table.bucket_mut(index).value
    }
}

// ---------------------------------------------------------------------------
// Iterators
// ---------------------------------------------------------------------------

/// Iterator over `(&K, &V)`. Built by [`OpenHashMap::iter`].
pub struct Iter<'a, K, V> {
    inner: core::slice::Iter<'a, Option<Bucket<K, V>>>,
    remaining: usize,
}

impl<'a, K, V> Iterator for Iter<'a, K, V> {
    type Item = (&'a K, &'a V);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let bucket = self.inner.by_ref().flatten().next()?;
        self.remaining -= 1;
        Some((&bucket.key, &bucket.value))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<K, V> ExactSizeIterator for Iter<'_, K, V> {}
impl<K, V> FusedIterator for Iter<'_, K, V> {}

impl<K, V> Clone for Iter<'_, K, V> {
    fn clone(&self) -> Self {
        Iter {
            inner: self.inner.clone(),
            remaining: self.remaining,
        }
    }
}

/// Iterator over `(&K, &mut V)`. Built by [`OpenHashMap::iter_mut`].
pub struct IterMut<'a, K, V> {
    inner: core::slice::IterMut<'a, Option<Bucket<K, V>>>,
    remaining: usize,
}

impl<'a, K, V> Iterator for IterMut<'a, K, V> {
    type Item = (&'a K, &'a mut V);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let bucket = self.inner.by_ref().flatten().next()?;
        self.remaining -= 1;
        Some((&bucket.key, &mut bucket.value))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<K, V> ExactSizeIterator for IterMut<'_, K, V> {}
impl<K, V> FusedIterator for IterMut<'_, K, V> {}

/// Owning iterator over `(K, V)`. Built by `OpenHashMap::into_iter`.
pub struct IntoIter<K, V> {
    inner: alloc::vec::IntoIter<Option<Bucket<K, V>>>,
    remaining: usize,
}

impl<K, V> Iterator for IntoIter<K, V> {
    type Item = (K, V);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let bucket = self.inner.by_ref().flatten().next()?;
        self.remaining -= 1;
        Some((bucket.key, bucket.value))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<K, V> ExactSizeIterator for IntoIter<K, V> {}
impl<K, V> FusedIterator for IntoIter<K, V> {}

/// Iterator over keys. Built by [`OpenHashMap::keys`].
pub struct Keys<'a, K, V> {
    inner: Iter<'a, K, V>,
}

impl<'a, K, V> Iterator for Keys<'a, K, V> {
    type Item = &'a K;

    #[inline]
    fn next(&mut self) -> Option<&'a K> {
        self.inner.next().map(|(key, _)| key)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<K, V> ExactSizeIterator for Keys<'_, K, V> {}
impl<K, V> FusedIterator for Keys<'_, K, V> {}

impl<K, V> Clone for Keys<'_, K, V> {
    fn clone(&self) -> Self {
        Keys {
            inner: self.inner.clone(),
        }
    }
}

/// Iterator over values. Built by [`OpenHashMap::values`].
pub struct Values<'a, K, V> {
    inner: Iter<'a, K, V>,
}

impl<'a, K, V> Iterator for Values<'a, K, V> {
    type Item = &'a V;

    #[inline]
    fn next(&mut self) -> Option<&'a V> {
        self.inner.next().map(|(_, value)| value)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<K, V> ExactSizeIterator for Values<'_, K, V> {}
impl<K, V> FusedIterator for Values<'_, K, V> {}

impl<K, V> Clone for Values<'_, K, V> {
    fn clone(&self) -> Self {
        Values {
            inner: self.inner.clone(),
        }
    }
}

/// Iterator over mutable values. Built by [`OpenHashMap::values_mut`].
pub struct ValuesMut<'a, K, V> {
    inner: IterMut<'a, K, V>,
}

impl<'a, K, V> Iterator for ValuesMut<'a, K, V> {
    type Item = &'a mut V;

    #[inline]
    fn next(&mut self) -> Option<&'a mut V> {
        self.inner.next().map(|(_, value)| value)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<K, V> ExactSizeIterator for ValuesMut<'_, K, V> {}
impl<K, V> FusedIterator for ValuesMut<'_, K, V> {}

/// Owning iterator over keys. Built by [`OpenHashMap::into_keys`].
pub struct IntoKeys<K, V> {
    inner: IntoIter<K, V>,
}

impl<K, V> Iterator for IntoKeys<K, V> {
    type Item = K;

    #[inline]
    fn next(&mut self) -> Option<K> {
        self.inner.next().map(|(key, _)| key)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<K, V> ExactSizeIterator for IntoKeys<K, V> {}
impl<K, V> FusedIterator for IntoKeys<K, V> {}

/// Owning iterator over values. Built by [`OpenHashMap::into_values`].
pub struct IntoValues<K, V> {
    inner: IntoIter<K, V>,
}

impl<K, V> Iterator for IntoValues<K, V> {
    type Item = V;

    #[inline]
    fn next(&mut self) -> Option<V> {
        self.inner.next().map(|(_, value)| value)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<K, V> ExactSizeIterator for IntoValues<K, V> {}
impl<K, V> FusedIterator for IntoValues<K, V> {}

impl<K, V, S> IntoIterator for OpenHashMap<K, V, S> {
    type Item = (K, V);
    type IntoIter = IntoIter<K, V>;

    fn into_iter(self) -> IntoIter<K, V> {
        let remaining = self.table.len();
        IntoIter {
            inner: self.table.into_slots().into_iter(),
            remaining,
        }
    }
}

impl<'a, K, V, S> IntoIterator for &'a OpenHashMap<K, V, S> {
    type Item = (&'a K, &'a V);
    type IntoIter = Iter<'a, K, V>;

    fn into_iter(self) -> Iter<'a, K, V> {
        self.iter()
    }
}

impl<'a, K, V, S> IntoIterator for &'a mut OpenHashMap<K, V, S> {
    type Item = (&'a K, &'a mut V);
    type IntoIter = IterMut<'a, K, V>;

    fn into_iter(self) -> IterMut<'a, K, V> {
        self.iter_mut()
    }
}

// ---------------------------------------------------------------------------
// Trait impls
// ---------------------------------------------------------------------------

impl<K, V, S: Default> Default for OpenHashMap<K, V, S> {
    fn default() -> Self {
        Self::with_hasher(S::default())
    }
}

impl<K: Clone, V: Clone, S: Clone> Clone for OpenHashMap<K, V, S> {
    fn clone(&self) -> Self {
        Self {
            table: self.table.clone(),
            hash_builder: self.hash_builder.clone(),
        }
    }
}

impl<K: fmt::Debug, V: fmt::Debug, S> fmt::Debug for OpenHashMap<K, V, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}

impl<K, V, S> PartialEq for OpenHashMap<K, V, S>
where
    K: Eq + Hash,
    V: PartialEq,
    S: BuildHasher,
{
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len()
            && self
                .iter()
                .all(|(key, value)| other.get(key).map_or(false, |v| *value == *v))
    }
}

impl<K, V, S> Eq for OpenHashMap<K, V, S>
where
    K: Eq + Hash,
    V: Eq,
    S: BuildHasher,
{
}

impl<K, Q, V, S> Index<&Q> for OpenHashMap<K, V, S>
where
    K: Eq + Hash + Borrow<Q>,
    Q: Eq + Hash + ?Sized,
    S: BuildHasher,
{
    type Output = V;

    /// # Panics
    ///
    /// If `key` is not present.
    fn index(&self, key: &Q) -> &V {
        self.get(key).expect("no entry found for key")
    }
}

impl<K, V, S> Extend<(K, V)> for OpenHashMap<K, V, S>
where
    K: Eq + Hash,
    S: BuildHasher,
{
    fn extend<I: IntoIterator<Item = (K, V)>>(&mut self, iter: I) {
        let iter = iter.into_iter();
        let (lower, _) = iter.size_hint();
        // Same guess `std`'s map makes: duplicates are likely once the map
        // already has entries.
        self.reserve(if self.is_empty() {
            lower
        } else {
            (lower + 1) / 2
        });
        for (key, value) in iter {
            self.insert(key, value);
        }
    }
}

impl<K, V, S> FromIterator<(K, V)> for OpenHashMap<K, V, S>
where
    K: Eq + Hash,
    S: BuildHasher + Default,
{
    fn from_iter<I: IntoIterator<Item = (K, V)>>(iter: I) -> Self {
        let mut map = Self::default();
        map.extend(iter);
        map
    }
}
