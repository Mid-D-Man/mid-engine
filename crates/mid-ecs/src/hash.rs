// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-ecs.md, section "hash.rs: fast hashers for the Archetype
// Core's hot-path maps"
// ============================================================================
//! Cheap, non-cryptographic hashers for the Archetype Core's own
//! `TypeId -> ComponentId` map and per-archetype `ComponentId ->
//! ArchetypeId` edge maps.
//!
//! Both key types are generated locally (`TypeId::of::<T>()`, a counter)
//! and never attacker-supplied, so std's default SipHash HashDoS
//! resistance buys nothing here. `pub(crate)` on purpose: nothing about
//! these belongs in `mid-ecs`'s public surface.

use std::any::TypeId;
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

const FIB: u64 = 0x9E37_79B9_7F4A_7C15;

/// `TypeId` is already a well-mixed fingerprint, so this passes the
/// value `TypeId`'s own `Hash` impl feeds it straight through.
/// Correctness never depends on which `write_*` method `TypeId` calls
/// (equal keys always produce the same write sequence, hence the same
/// hash); the fallbacks only keep quality acceptable if that impl
/// changes shape in a future toolchain.
#[derive(Default)]
pub(crate) struct TypeIdHasher {
    hash: u64,
}

impl Hasher for TypeIdHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.hash
    }

    #[inline]
    fn write_u64(&mut self, v: u64) {
        self.hash = v;
    }

    #[inline]
    fn write_u128(&mut self, v: u128) {
        self.hash = (v as u64) ^ ((v >> 64) as u64);
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.hash = (self.hash.rotate_left(5) ^ u64::from(b)).wrapping_mul(FIB);
        }
    }
}

/// Multiplicative (Fibonacci) hash for small, dense integer ids.
/// Multiplying by an odd constant is a bijection on the low bits (so
/// consecutive ids never collide in a bucket index) while the high
/// bits hashbrown uses for its control bytes come out well mixed.
#[derive(Default)]
pub(crate) struct DenseIdHasher {
    hash: u64,
}

impl Hasher for DenseIdHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.hash
    }

    #[inline]
    fn write_u32(&mut self, v: u32) {
        self.hash = u64::from(v).wrapping_mul(FIB);
    }

    #[inline]
    fn write_u64(&mut self, v: u64) {
        self.hash = v.wrapping_mul(FIB);
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.hash = (self.hash.rotate_left(5) ^ u64::from(b)).wrapping_mul(FIB);
        }
    }
}

pub(crate) type TypeIdMap<V> = HashMap<TypeId, V, BuildHasherDefault<TypeIdHasher>>;
pub(crate) type DenseIdMap<K, V> = HashMap<K, V, BuildHasherDefault<DenseIdHasher>>;

#[cfg(test)]
mod tests {
    use super::*;
    use std::hash::{BuildHasher, Hash};

    struct A;
    struct B;

    #[test]
    fn type_id_map_roundtrips_distinct_types() {
        let mut m: TypeIdMap<u32> = TypeIdMap::default();
        m.insert(TypeId::of::<A>(), 1);
        m.insert(TypeId::of::<B>(), 2);
        m.insert(TypeId::of::<u64>(), 3);
        assert_eq!(m.get(&TypeId::of::<A>()), Some(&1));
        assert_eq!(m.get(&TypeId::of::<B>()), Some(&2));
        assert_eq!(m.get(&TypeId::of::<u64>()), Some(&3));
        assert_eq!(m.get(&TypeId::of::<u8>()), None);
        assert_eq!(m.len(), 3);
    }

    #[test]
    fn type_id_hash_is_deterministic_and_type_sensitive() {
        let s: BuildHasherDefault<TypeIdHasher> = BuildHasherDefault::default();
        let h = |t: TypeId| s.hash_one(t);
        assert_eq!(h(TypeId::of::<A>()), h(TypeId::of::<A>()));
        assert_ne!(h(TypeId::of::<A>()), h(TypeId::of::<B>()));
    }

    #[test]
    fn type_id_hasher_fallback_paths_are_deterministic() {
        let mut a = TypeIdHasher::default();
        let mut b = TypeIdHasher::default();
        a.write(&[1, 2, 3, 4]);
        b.write(&[1, 2, 3, 4]);
        assert_eq!(a.finish(), b.finish());
        let mut c = TypeIdHasher::default();
        c.write(&[1, 2, 3, 5]);
        assert_ne!(a.finish(), c.finish());
        let mut d = TypeIdHasher::default();
        d.write_u128(0x1234_5678_9abc_def0_1111_2222_3333_4444);
        let mut e = TypeIdHasher::default();
        e.write_u128(0x1234_5678_9abc_def0_1111_2222_3333_4444);
        assert_eq!(d.finish(), e.finish());
    }

    #[test]
    fn dense_id_hash_has_no_low_bit_collisions_for_consecutive_ids() {
        // Bijective on the low bits: 1024 consecutive ids must land in
        // 1024 distinct buckets of a 1024-slot table.
        let s: BuildHasherDefault<DenseIdHasher> = BuildHasherDefault::default();
        let mut seen = vec![false; 1024];
        for id in 0u32..1024 {
            let bucket = (s.hash_one(id) & 1023) as usize;
            assert!(!seen[bucket], "id {id} collided in the low 10 bits");
            seen[bucket] = true;
        }
    }

    #[test]
    fn dense_id_map_roundtrips() {
        let mut m: DenseIdMap<u32, u32> = DenseIdMap::default();
        for i in 0..200u32 {
            m.insert(i, i * 3);
        }
        for i in 0..200u32 {
            assert_eq!(m.get(&i), Some(&(i * 3)));
        }
        assert_eq!(m.get(&200), None);
    }

    #[test]
    fn hash_impl_for_derived_newtype_reaches_write_u32() {
        // ComponentId is `#[derive(Hash)]` over a u32; make sure a
        // newtype like that lands on the fast path, not the byte
        // fallback.
        #[derive(Hash)]
        struct Id(u32);
        let s: BuildHasherDefault<DenseIdHasher> = BuildHasherDefault::default();
        let mut h = s.build_hasher();
        Id(7).hash(&mut h);
        assert_eq!(h.finish(), u64::from(7u32).wrapping_mul(FIB));
    }
}
