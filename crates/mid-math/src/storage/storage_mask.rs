// crates/mid-math/src/storage/storage_mask.rs
//! Bit-packed boolean storage — the "storage mask" half of Mid-Engine's two-mask system.
//!
//! ## The two-mask rule
//!
//! | Property           | SIMD mask (`Mask4`, `IMask4`, …)     | Storage mask (`BitMask*`)            |
//! |--------------------|--------------------------------------|--------------------------------------|
//! | Bits per boolean   | 32 (full SIMD lane)                  | 1                                    |
//! | `true` encoding    | `0xFFFF_FFFF`                        | single bit `1`                       |
//! | Purpose            | Branchless blend/select in math      | Compact flag storage & bulk AND/OR   |
//! | Home               | `wide/float/`, `wide/int/`           | `storage/` (here)                    |
//! | Memory for 64 bits | 256 bytes                            | 8 bytes                              |
//!
//! **Never use a SIMD mask for storage. Never use a storage mask for SIMD blending.**
//!
//! ## ECS usage
//! ```text
//! // Component IDs:  Transform=0  Velocity=1  Player=2  Enemy=3
//! let needs       = BitMask64::from_indices(&[0, 1]);        // query: want Transform + Velocity
//! let entity_mask = BitMask64::from_indices(&[0, 1, 2]);     // entity has all three
//!
//! assert!(entity_mask.matches(needs));    // entity satisfies the query
//! assert!(!needs.matches(entity_mask));   // query doesn't have Player bit
//! ```
//!
//! ## Works with every storage type
//! Whether your data array holds `f32`, `f16`, `F8E4M3`, or the upcoming `f4`,
//! a `BitMask*` over that array means the same thing: which slots are active.
//! The mask is format-agnostic.

use core::{
    fmt,
    ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not},
};

// ─────────────────────────────────────────────────────────────────────────────
// Shared iterator (upcasts to u64 — works for all scalar mask sizes)
// ─────────────────────────────────────────────────────────────────────────────

/// Iterator over the indices of all SET bits in a storage mask, ascending.
///
/// Returned by `BitMask8::iter_ones()`, `BitMask16::iter_ones()`, etc.
pub struct IterOnes {
    bits: u64,
}

impl Iterator for IterOnes {
    type Item = usize;
    #[inline]
    fn next(&mut self) -> Option<usize> {
        if self.bits == 0 {
            return None;
        }
        let index = self.bits.trailing_zeros() as usize;
        self.bits &= self.bits - 1; // Kernighan's trick: clear lowest set bit
        Some(index)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let count = self.bits.count_ones() as usize;
        (count, Some(count))
    }
}

impl core::iter::FusedIterator for IterOnes {}

// ─────────────────────────────────────────────────────────────────────────────
// Wide iterator (multi-word masks: BitMask128, BitMask256)
// ─────────────────────────────────────────────────────────────────────────────

/// Iterator over set bit indices in a multi-word storage mask.
pub struct WideIterOnes<const N: usize> {
    words:   [u64; N],
    /// Index of the word whose bits are currently loaded into `bit`.
    word:    usize,
    /// Remaining bits from `words[word]` still to be yielded.
    bit:     u64,
}

impl<const N: usize> Iterator for WideIterOnes<N> {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<usize> {
        // Skip exhausted words until we find one with a set bit or run out.
        while self.bit == 0 {
            self.word += 1;
            if self.word >= N {
                return None;
            }
            self.bit = self.words[self.word];
        }
        let local = self.bit.trailing_zeros() as usize;
        self.bit &= self.bit - 1; // clear lowest set bit
        Some(self.word * 64 + local)
    }
}

impl<const N: usize> core::iter::FusedIterator for WideIterOnes<N> {}

// ─────────────────────────────────────────────────────────────────────────────
// Macro — generates BitMask8 / 16 / 32 / 64 from a single integer primitive
// ─────────────────────────────────────────────────────────────────────────────

macro_rules! impl_bitmask_scalar {
    ($Name:ident, $Inner:ty, $bits:expr, $doc:literal) => {
        #[doc = $doc]
        #[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
        #[repr(transparent)]
        pub struct $Name($Inner);

        impl $Name {
            /// All bits clear — every slot is `false`.
            pub const NONE: Self = Self(0);
            /// All bits set — every slot is `true`.
            pub const ALL:  Self = Self(!0);
            /// Total number of boolean slots.
            pub const CAPACITY: usize = $bits;

            /// Construct from raw bits.
            #[inline(always)]
            pub const fn from_bits(bits: $Inner) -> Self { Self(bits) }

            /// Return the raw bits.
            #[inline(always)]
            pub const fn to_bits(self) -> $Inner { self.0 }

            /// Build a mask with exactly the listed bit positions set (0-based).
            /// Indices ≥ `CAPACITY` are silently ignored.
            pub fn from_indices(indices: &[usize]) -> Self {
                let mut v: $Inner = 0;
                for &i in indices {
                    if i < $bits { v |= (1 as $Inner) << i; }
                }
                Self(v)
            }

            // ── Single-bit operations ─────────────────────────────────────────

            /// Returns the boolean at `index`. Panics if `index >= CAPACITY`.
            #[inline]
            pub fn get(self, index: usize) -> bool {
                assert!(index < $bits, "BitMask index {} out of range (capacity {})", index, $bits);
                (self.0 >> index) & 1 != 0
            }

            /// Sets bit at `index` to `true`. Panics if `index >= CAPACITY`.
            #[inline]
            pub fn set(&mut self, index: usize) {
                assert!(index < $bits, "BitMask index {} out of range (capacity {})", index, $bits);
                self.0 |= (1 as $Inner) << index;
            }

            /// Clears bit at `index` to `false`. Panics if `index >= CAPACITY`.
            #[inline]
            pub fn clear(&mut self, index: usize) {
                assert!(index < $bits, "BitMask index {} out of range (capacity {})", index, $bits);
                self.0 &= !((1 as $Inner) << index);
            }

            /// Flips bit at `index`. Panics if `index >= CAPACITY`.
            #[inline]
            pub fn toggle(&mut self, index: usize) {
                assert!(index < $bits, "BitMask index {} out of range (capacity {})", index, $bits);
                self.0 ^= (1 as $Inner) << index;
            }

            // ── Aggregate queries ─────────────────────────────────────────────

            /// `true` if at least one bit is set.
            #[inline] pub fn any(self)         -> bool { self.0 != 0 }
            /// `true` if every bit is set.
            #[inline] pub fn all(self)         -> bool { self.0 == !0 }
            /// `true` if no bit is set.
            #[inline] pub fn none(self)        -> bool { self.0 == 0 }
            /// Number of set bits (`popcount`).
            #[inline] pub fn count_ones(self)  -> u32  { self.0.count_ones() }
            /// Number of clear bits.
            #[inline] pub fn count_zeros(self) -> u32  { self.0.count_zeros() }

            // ── Set algebra ───────────────────────────────────────────────────

            /// Bits present in **both** `self` and `other` (AND).
            #[inline] pub fn intersection(self, other: Self) -> Self { Self(self.0 & other.0) }
            /// Bits present in **either** `self` or `other` (OR).
            #[inline] pub fn union(self, other: Self)        -> Self { Self(self.0 | other.0) }
            /// Bits in `self` but NOT in `other` (AND NOT).
            #[inline] pub fn difference(self, other: Self)   -> Self { Self(self.0 & !other.0) }
            /// Bits in exactly one of the two masks (XOR).
            #[inline] pub fn symmetric_difference(self, other: Self) -> Self { Self(self.0 ^ other.0) }

            /// **ECS archetype check.** Returns `true` when ALL bits set in
            /// `required` are also set in `self`.
            ///
            /// ```text
            /// (self & required) == required
            /// ```
            #[inline]
            pub fn matches(self, required: Self) -> bool {
                (self.0 & required.0) == required.0
            }

            /// `true` when `self` and `other` share no set bits.
            #[inline]
            pub fn is_disjoint(self, other: Self) -> bool {
                (self.0 & other.0) == 0
            }

            /// `true` when every set bit in `self` is also set in `other`.
            #[inline]
            pub fn is_subset_of(self, other: Self) -> bool {
                other.matches(self)
            }

            /// Iterate over indices of all SET bits in ascending order.
            #[inline]
            pub fn iter_ones(self) -> IterOnes {
                IterOnes { bits: self.0 as u64 }
            }
        }

        // ── Operator overloads ────────────────────────────────────────────────

        impl BitAnd for $Name {
            type Output = Self;
            #[inline] fn bitand(self, r: Self) -> Self { Self(self.0 & r.0) }
        }
        impl BitAndAssign for $Name {
            #[inline] fn bitand_assign(&mut self, r: Self) { self.0 &= r.0; }
        }
        impl BitOr for $Name {
            type Output = Self;
            #[inline] fn bitor(self, r: Self) -> Self { Self(self.0 | r.0) }
        }
        impl BitOrAssign for $Name {
            #[inline] fn bitor_assign(&mut self, r: Self) { self.0 |= r.0; }
        }
        impl BitXor for $Name {
            type Output = Self;
            #[inline] fn bitxor(self, r: Self) -> Self { Self(self.0 ^ r.0) }
        }
        impl BitXorAssign for $Name {
            #[inline] fn bitxor_assign(&mut self, r: Self) { self.0 ^= r.0; }
        }
        impl Not for $Name {
            type Output = Self;
            #[inline] fn not(self) -> Self { Self(!self.0) }
        }

        // ── Formatting ────────────────────────────────────────────────────────

        impl fmt::Debug for $Name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}(0b{:0width$b})", stringify!($Name), self.0, width = $bits)
            }
        }
        impl fmt::Display for $Name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "0b{:0width$b}", self.0, width = $bits)
            }
        }
        impl fmt::LowerHex for $Name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "0x{:x}", self.0)
            }
        }
        impl fmt::Binary for $Name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{:0width$b}", self.0, width = $bits)
            }
        }
    };
}

// ── Scalar mask types ─────────────────────────────────────────────────────────

impl_bitmask_scalar!(BitMask8,  u8,  8,
    "8-bit storage mask — 8 boolean slots in 1 byte. Useful for per-vertex or small-batch flags.");
impl_bitmask_scalar!(BitMask16, u16, 16,
    "16-bit storage mask — 16 boolean slots in 2 bytes.");
impl_bitmask_scalar!(BitMask32, u32, 32,
    "32-bit storage mask — 32 boolean slots in 4 bytes. Good for layer masks, render flags.");
impl_bitmask_scalar!(BitMask64, u64, 64,
    "64-bit storage mask — 64 boolean slots in 8 bytes. The standard ECS component mask size.");

// ─────────────────────────────────────────────────────────────────────────────
// Multi-word masks: BitMask128 and BitMask256
// ─────────────────────────────────────────────────────────────────────────────

macro_rules! impl_bitmask_wide {
    ($Name:ident, $WORDS:expr, $bits:expr, $doc:literal) => {
        #[doc = $doc]
        #[derive(Clone, Copy, PartialEq, Eq, Hash)]
        #[repr(C)]
        pub struct $Name([u64; $WORDS]);

        impl Default for $Name {
            #[inline] fn default() -> Self { Self([0u64; $WORDS]) }
        }

        impl $Name {
            /// All bits clear.
            pub const NONE: Self = Self([0u64; $WORDS]);
            /// All bits set.
            pub const ALL:  Self = Self([!0u64; $WORDS]);
            /// Total boolean slots.
            pub const CAPACITY: usize = $bits;

            #[inline(always)]
            pub const fn from_words(words: [u64; $WORDS]) -> Self { Self(words) }
            #[inline(always)]
            pub const fn to_words(self) -> [u64; $WORDS] { self.0 }

            /// Build from a list of set bit indices (0-based). Ignores indices ≥ CAPACITY.
            pub fn from_indices(indices: &[usize]) -> Self {
                let mut w = [0u64; $WORDS];
                for &i in indices {
                    if i < $bits { w[i / 64] |= 1u64 << (i % 64); }
                }
                Self(w)
            }

            #[inline]
            pub fn get(self, index: usize) -> bool {
                assert!(index < $bits, "BitMask index {} out of range (capacity {})", index, $bits);
                (self.0[index / 64] >> (index % 64)) & 1 != 0
            }
            #[inline]
            pub fn set(&mut self, index: usize) {
                assert!(index < $bits, "BitMask index {} out of range (capacity {})", index, $bits);
                self.0[index / 64] |= 1u64 << (index % 64);
            }
            #[inline]
            pub fn clear(&mut self, index: usize) {
                assert!(index < $bits, "BitMask index {} out of range (capacity {})", index, $bits);
                self.0[index / 64] &= !(1u64 << (index % 64));
            }
            #[inline]
            pub fn toggle(&mut self, index: usize) {
                assert!(index < $bits, "BitMask index {} out of range (capacity {})", index, $bits);
                self.0[index / 64] ^= 1u64 << (index % 64);
            }

            #[inline] pub fn any(self)         -> bool { self.0.iter().any(|&w| w != 0) }
            #[inline] pub fn all(self)         -> bool { self.0.iter().all(|&w| w == !0) }
            #[inline] pub fn none(self)        -> bool { !self.any() }
            #[inline] pub fn count_ones(self)  -> u32  { self.0.iter().map(|w| w.count_ones()).sum() }
            #[inline] pub fn count_zeros(self) -> u32  { ($bits as u32) - self.count_ones() }

            #[inline]
            pub fn intersection(self, other: Self) -> Self {
                let mut r = [0u64; $WORDS];
                for i in 0..$WORDS { r[i] = self.0[i] & other.0[i]; }
                Self(r)
            }
            #[inline]
            pub fn union(self, other: Self) -> Self {
                let mut r = [0u64; $WORDS];
                for i in 0..$WORDS { r[i] = self.0[i] | other.0[i]; }
                Self(r)
            }
            #[inline]
            pub fn difference(self, other: Self) -> Self {
                let mut r = [0u64; $WORDS];
                for i in 0..$WORDS { r[i] = self.0[i] & !other.0[i]; }
                Self(r)
            }
            #[inline]
            pub fn symmetric_difference(self, other: Self) -> Self {
                let mut r = [0u64; $WORDS];
                for i in 0..$WORDS { r[i] = self.0[i] ^ other.0[i]; }
                Self(r)
            }

            /// ECS archetype check — all bits in `required` must be set in `self`.
            #[inline]
            pub fn matches(self, required: Self) -> bool {
                (0..$WORDS).all(|i| (self.0[i] & required.0[i]) == required.0[i])
            }

            #[inline]
            pub fn is_disjoint(self, other: Self) -> bool {
                (0..$WORDS).all(|i| (self.0[i] & other.0[i]) == 0)
            }

            #[inline]
            pub fn is_subset_of(self, other: Self) -> bool { other.matches(self) }

            /// Iterate over indices of all SET bits in ascending order.
            ///
            /// Starts loading `words[0]` immediately — word 0 is never skipped.
            #[inline]
            pub fn iter_ones(self) -> WideIterOnes<$WORDS> {
                WideIterOnes {
                    words: self.0,
                    word: 0,
                    // Preload word 0 so the iterator doesn't skip it.
                    bit: self.0[0],
                }
            }
        }

        impl BitAnd for $Name {
            type Output = Self;
            #[inline] fn bitand(self, r: Self) -> Self { self.intersection(r) }
        }
        impl BitAndAssign for $Name {
            #[inline] fn bitand_assign(&mut self, r: Self) { *self = self.intersection(r); }
        }
        impl BitOr for $Name {
            type Output = Self;
            #[inline] fn bitor(self, r: Self) -> Self { self.union(r) }
        }
        impl BitOrAssign for $Name {
            #[inline] fn bitor_assign(&mut self, r: Self) { *self = self.union(r); }
        }
        impl BitXor for $Name {
            type Output = Self;
            #[inline] fn bitxor(self, r: Self) -> Self { self.symmetric_difference(r) }
        }
        impl BitXorAssign for $Name {
            #[inline] fn bitxor_assign(&mut self, r: Self) { *self = self.symmetric_difference(r); }
        }
        impl Not for $Name {
            type Output = Self;
            #[inline]
            fn not(self) -> Self {
                let mut r = [0u64; $WORDS];
                for i in 0..$WORDS { r[i] = !self.0[i]; }
                Self(r)
            }
        }

        impl fmt::Debug for $Name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}([", stringify!($Name))?;
                for (i, &w) in self.0.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "0x{:016x}", w)?;
                }
                write!(f, "])")
            }
        }
    };
}

impl_bitmask_wide!(BitMask128, 2, 128,
    "128-bit storage mask — 128 boolean slots (2×u64). Handles up to 128 ECS component types.");
impl_bitmask_wide!(BitMask256, 4, 256,
    "256-bit storage mask — 256 boolean slots (4×u64). Handles up to 256 ECS component types.");

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes() {
        assert_eq!(core::mem::size_of::<BitMask8>(),   1);
        assert_eq!(core::mem::size_of::<BitMask16>(),  2);
        assert_eq!(core::mem::size_of::<BitMask32>(),  4);
        assert_eq!(core::mem::size_of::<BitMask64>(),  8);
        assert_eq!(core::mem::size_of::<BitMask128>(), 16);
        assert_eq!(core::mem::size_of::<BitMask256>(), 32);
    }

    #[test]
    fn set_get_clear() {
        let mut m = BitMask64::NONE;
        assert!(!m.get(0));
        m.set(3);
        assert!(m.get(3));
        assert!(!m.get(2));
        m.clear(3);
        assert!(!m.get(3));
        m.toggle(7);
        assert!(m.get(7));
        m.toggle(7);
        assert!(!m.get(7));
    }

    #[test]
    fn from_indices() {
        let m = BitMask64::from_indices(&[0, 1, 5, 63]);
        assert!(m.get(0) && m.get(1) && m.get(5) && m.get(63));
        assert!(!m.get(2) && !m.get(62));
        assert_eq!(m.count_ones(), 4);
    }

    #[test]
    fn ecs_archetype_matches() {
        // Entity has Transform(0), Velocity(1), Player(2)
        let entity = BitMask64::from_indices(&[0, 1, 2]);
        // Query requires only Transform + Velocity
        let query  = BitMask64::from_indices(&[0, 1]);

        assert!(entity.matches(query),   "entity should satisfy the query");
        assert!(!query.matches(entity),  "query should NOT satisfy entity (missing Player)");
        assert!(!entity.matches(BitMask64::from_indices(&[0, 1, 5])),
                "entity missing component 5 should fail");
    }

    #[test]
    fn disjoint_and_subset() {
        let a = BitMask64::from_indices(&[0, 2, 4]);
        let b = BitMask64::from_indices(&[1, 3, 5]);
        let c = BitMask64::from_indices(&[0, 2]);
        assert!( a.is_disjoint(b), "a and b share no bits");
        assert!(!a.is_disjoint(c), "a and c share bits 0, 2");
        assert!( c.is_subset_of(a), "c ⊆ a");
        assert!(!a.is_subset_of(c), "a ⊄ c (has bit 4)");
    }

    #[test]
    fn bitwise_ops() {
        let a = BitMask32::from_bits(0b1010);
        let b = BitMask32::from_bits(0b1100);
        assert_eq!((a & b).to_bits(), 0b1000);
        assert_eq!((a | b).to_bits(), 0b1110);
        assert_eq!((a ^ b).to_bits(), 0b0110);
        assert_eq!((!BitMask8::from_bits(0b0000_1111)).to_bits(), 0b1111_0000);
    }

    #[test]
    fn iter_ones_scalar_empty() {
        let m = BitMask64::NONE;
        assert_eq!(m.iter_ones().next(), None);
    }

    #[test]
    fn iter_ones_scalar_order() {
        let m = BitMask64::from_indices(&[0, 3, 7, 63]);
        let mut it = m.iter_ones();
        assert_eq!(it.next(), Some(0));
        assert_eq!(it.next(), Some(3));
        assert_eq!(it.next(), Some(7));
        assert_eq!(it.next(), Some(63));
        assert_eq!(it.next(), None);
    }

    #[test]
    fn iter_ones_scalar_bit0_not_skipped() {
        // Regression test: ensure bit 0 is yielded (not skipped by initialization)
        let m = BitMask8::from_bits(0b0000_0001);
        let mut it = m.iter_ones();
        assert_eq!(it.next(), Some(0));
        assert_eq!(it.next(), None);
    }

    #[test]
    fn iter_ones_wide_word0_not_skipped() {
        // Regression test: WideIterOnes must NOT skip word 0.
        let m = BitMask128::from_indices(&[0]);  // bit 0 is in word 0
        let mut it = m.iter_ones();
        assert_eq!(it.next(), Some(0), "bit 0 (word 0) must not be skipped");
        assert_eq!(it.next(), None);
    }

    #[test]
    fn iter_ones_wide_across_words() {
        let m = BitMask128::from_indices(&[0, 63, 64, 127]);
        let mut it = m.iter_ones();
        assert_eq!(it.next(), Some(0));
        assert_eq!(it.next(), Some(63));
        assert_eq!(it.next(), Some(64));
        assert_eq!(it.next(), Some(127));
        assert_eq!(it.next(), None);
    }

    #[test]
    fn iter_ones_wide_sparse() {
        // Only bits in word 1 (indices 64-127) — word 0 is empty.
        let m = BitMask128::from_indices(&[65, 100]);
        let mut it = m.iter_ones();
        assert_eq!(it.next(), Some(65));
        assert_eq!(it.next(), Some(100));
        assert_eq!(it.next(), None);
    }

    #[test]
    fn wide_matches() {
        let entity = BitMask256::from_indices(&[0, 1, 200, 255]);
        let query  = BitMask256::from_indices(&[0, 200]);
        assert!(entity.matches(query));
        assert!(!query.matches(entity));
    }

    #[test]
    fn wide_count() {
        let m = BitMask256::from_indices(&[0, 64, 128, 192, 255]);
        assert_eq!(m.count_ones(), 5);
        assert_eq!(m.count_zeros(), 256 - 5);
    }

    #[test]
    fn wide_iter_256() {
        let m = BitMask256::from_indices(&[0, 127, 128, 255]);
        let mut it = m.iter_ones();
        assert_eq!(it.next(), Some(0));
        assert_eq!(it.next(), Some(127));
        assert_eq!(it.next(), Some(128));
        assert_eq!(it.next(), Some(255));
        assert_eq!(it.next(), None);
    }

    #[test]
    fn all_and_none() {
        assert!(BitMask32::ALL.all());
        assert!(BitMask32::ALL.any());
        assert!(!BitMask32::ALL.none());
        assert!(BitMask32::NONE.none());
        assert!(!BitMask32::NONE.any());
        assert_eq!(BitMask32::ALL.count_ones(), 32);
        assert_eq!(BitMask32::NONE.count_ones(), 0);
    }

    #[test]
    fn difference_op() {
        let a = BitMask64::from_bits(0b1111);
        let b = BitMask64::from_bits(0b1010);
        // a - b = bits in a but not b = 0b0101
        assert_eq!(a.difference(b).to_bits(), 0b0101);
    }
  }
