// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-platform.md, section "hash.rs"
// ============================================================================

//! A fast, non-cryptographic hasher (`FxHasher`) and a deterministic
//! `BuildHasher` for it (`FixedHasher`) — this crate's Phase 3 fast hasher.
//!
//! Real `bevy_platform::hash` wraps the `foldhash` crate's own
//! `FoldHasher`/`FixedState`/`RandomState` (`docs/mid-platform.md`'s own
//! dependency verdict already covers `foldhash` — hand-roll, deferred to
//! this phase). Not wrapped here either: a fold/multiply-based hash is
//! well-trodden ground, and this crate already has a proven, working
//! example of exactly that shape one workspace member away — see "Why a
//! second, independent `FxHasher`" in `docs/mid-platform.md` for why this
//! is a fresh hand-written copy of `mid-collections`'s own `FxHasher`
//! rather than a dependency on that crate (same reasoning `sync::mutex`'s
//! own doc comment already gives for not depending on `mid-alloc`).
//!
//! No `std`/`no_std` split needed anywhere in this file, unlike everything
//! else in this crate — the algorithm is pure integer arithmetic over
//! `core::hash::Hasher`, nothing here needs an OS or an allocator either
//! way.
//!
//! **Why `FixedHasher` does not hand out a bare [`FxHasher`].** A plain
//! fold/multiply hash ends in a multiplication, so the *low* bits of its
//! output depend only on the low bits of the key, and `std`'s map (hashbrown)
//! picks a bucket from the low bits. Keys whose entropy sits higher up
//! (`i << 12`, entity ids with a generation in the high half, `f64` bits)
//! collide almost completely: measured in this crate's own benches, 8x to
//! 17x slower than `std`'s default SipHash map. `FixedHasher` therefore
//! builds a [`FoldedFxHasher`]: the same Fx accumulation, then one 64x64 to
//! 128-bit multiply with the two halves XORed together on `finish` (the same
//! idea as `foldhash`'s). [`FxHasher`] itself is unchanged and keeps its
//! bit-for-bit parity with `rustc-hash` 1.1.0; use it directly only where
//! the consumer already spreads the hash (like `OpenHashMap` does).
//!
//! **Deliberately no `RandomState`/HashDoS-resistant equivalent.** Real
//! `foldhash::fast::RandomState` seeds itself from OS randomness so an
//! attacker who doesn't know the process's seed can't precompute a
//! collision attack ahead of time. Two separate reasons this isn't
//! provided here:
//! 1. FxHash's own diffusion is weak by design — the whole point of a
//!    fold/multiply hash is speed over bit-mixing quality. Randomizing
//!    the seed raises the bar for a *precomputed* attack but does nothing
//!    for an attacker who can just observe hash outputs (or timing) at
//!    runtime and adapt; it's a false sense of security for a hasher this
//!    algorithm was never trying to make DoS-resistant in the first place
//!    (`mid-collections`'s own `fx_hash.rs` doc comment states the same
//!    "no adversarial-input threat model" reasoning for its own use case).
//! 2. Getting real OS entropy with zero external dependencies, in a way
//!    that also has to work under this crate's `no_std` fallback (no OS to
//!    ask on a genuinely bare target), is a separate, harder problem not
//!    worth solving to randomize a hasher that wouldn't meaningfully
//!    benefit from it anyway.
//!
//! A caller that genuinely needs HashDoS resistance should reach for
//! `std::collections::HashMap`'s own default `RandomState` (real SipHash,
//! real OS-seeded randomness) instead of this module — this hasher is for
//! trusted, engine-internal keys (spatial-hash cells, entity ids, and
//! similar), the same category `mid-collections`'s own copy already
//! targets.

use core::hash::{BuildHasher, Hasher};

/// Multiplier used by [`FxHasher::add`] — identical to `mid-collections`'s
/// own `FxHasher`, the same value rustc itself uses internally for this
/// algorithm.
const MULTIPLIER: u64 = 0x51_7c_c1_b7_27_22_0a_95;

/// Fixed, non-zero starting state for [`FixedHasher`]. The 64-bit
/// "golden ratio" constant (`2^64 / φ`, rounded to an odd integer) widely
/// used for exactly this purpose — seeding fold/multiply-style hashers
/// and Fibonacci hashing — not a value this implementation depends on for
/// any specific property beyond "a well-distributed, non-zero, publicly
/// unremarkable odd constant."
const FIXED_SEED: u64 = 0x9E37_79B9_7F4A_7C15;

/// A fast, non-cryptographic hasher — the same fold/multiply algorithm
/// rustc uses internally for its own compiler data structures
/// (originally from Firefox). Not DoS-hardened like `std`'s default
/// SipHash — deliberately, see this module's own top doc comment.
///
/// An independent copy of `mid_collections::FxHasher`'s own algorithm,
/// including its choice to sign-extend the narrow signed integer writes
/// (`write_i8`/`write_i16`/`write_i32`) — see "Why a second, independent
/// `FxHasher`" in `docs/mid-platform.md`. The one real difference from
/// that copy: this version can start from an arbitrary caller-supplied
/// seed via [`FxHasher::with_seed`] (needed for [`FixedHasher`] below),
/// where `mid-collections`'s own version only ever starts from zero.
///
/// Matches the published `rustc-hash` 1.1.0 crate bit-for-bit on 64-bit
/// targets from a zero seed for everything except negative `i8`/`i16`/
/// `i32` values: `rustc-hash` zero-extends those through the unsigned
/// writes, this sign-extends them. Both are fine hash inputs; the
/// difference is inherited from `mid-collections`, not introduced here.
/// Unlike `rustc-hash` (a `usize`-sized state, different constant on
/// 32-bit), this uses a `u64` state everywhere, so its output is the
/// same on 32-bit and 64-bit targets (wasm32 vs native).
#[derive(Clone, Debug, Default)]
pub struct FxHasher {
    hash: u64,
}

impl FxHasher {
    /// Starts a fresh hasher seeded with `seed` instead of zero. Lets
    /// [`FixedHasher`] mix in a fixed, non-zero starting state without a
    /// second, near-duplicate `Hasher` implementation.
    #[inline]
    #[must_use]
    pub const fn with_seed(seed: u64) -> Self {
        Self { hash: seed }
    }

    #[inline]
    fn add(&mut self, w: u64) {
        self.hash = (self.hash.rotate_left(5) ^ w).wrapping_mul(MULTIPLIER);
    }
}

impl Hasher for FxHasher {
    #[inline]
    fn write(&mut self, mut bytes: &[u8]) {
        while bytes.len() >= 8 {
            self.add(u64::from_ne_bytes(bytes[..8].try_into().unwrap()));
            bytes = &bytes[8..];
        }
        if bytes.len() >= 4 {
            self.add(u32::from_ne_bytes(bytes[..4].try_into().unwrap()) as u64);
            bytes = &bytes[4..];
        }
        if bytes.len() >= 2 {
            self.add(u16::from_ne_bytes(bytes[..2].try_into().unwrap()) as u64);
            bytes = &bytes[2..];
        }
        if let Some(&b) = bytes.first() {
            self.add(b as u64);
        }
    }

    #[inline] fn write_u8(&mut self, i: u8)       { self.add(i as u64); }
    #[inline] fn write_u16(&mut self, i: u16)     { self.add(i as u64); }
    #[inline] fn write_u32(&mut self, i: u32)     { self.add(i as u64); }
    #[inline] fn write_u64(&mut self, i: u64)     { self.add(i); }
    #[inline] fn write_usize(&mut self, i: usize) { self.add(i as u64); }
    #[inline] fn write_i8(&mut self, i: i8)       { self.add(i as u64); }
    #[inline] fn write_i16(&mut self, i: i16)     { self.add(i as u64); }
    #[inline] fn write_i32(&mut self, i: i32)     { self.add(i as u64); }
    #[inline] fn write_i64(&mut self, i: i64)     { self.add(i as u64); }
    #[inline] fn write_isize(&mut self, i: isize) { self.add(i as u64); }

    #[inline]
    fn finish(&self) -> u64 { self.hash }
}

/// Multiplier of the folded multiply in [`FoldedFxHasher::finish`]. The same
/// odd 64-bit "golden ratio" constant as [`FIXED_SEED`], in a different role;
/// it is the value the measurements in `docs/mid-platform.md` used.
const FOLD_MULTIPLIER: u64 = 0x9E37_79B9_7F4A_7C15;

/// 64x64 to 128-bit multiply, then the high and low halves XORed. Every
/// input bit reaches the low half of the result, which is what a table that
/// indexes by low bits needs. A single `mul` on x86-64; on `wasm32` LLVM
/// expands the 128-bit product into a few 64-bit multiplies (cost there not
/// measured).
#[inline(always)]
const fn folded_multiply(x: u64) -> u64 {
    let full = (x as u128).wrapping_mul(FOLD_MULTIPLIER as u128);
    (full as u64) ^ ((full >> 64) as u64)
}

/// [`FxHasher`]'s accumulation with a folded-multiply finish: what
/// [`FixedHasher`] builds. See this module's top doc comment for why the
/// finish step exists.
///
/// Every `write_*` is exactly [`FxHasher`]'s (sign extension of narrow
/// signed integers included); only [`finish`](Hasher::finish) differs.
#[derive(Clone, Debug, Default)]
pub struct FoldedFxHasher {
    inner: FxHasher,
}

impl FoldedFxHasher {
    /// Starts a fresh hasher seeded with `seed` instead of zero, like
    /// [`FxHasher::with_seed`].
    #[inline]
    #[must_use]
    pub const fn with_seed(seed: u64) -> Self {
        Self { inner: FxHasher::with_seed(seed) }
    }
}

impl Hasher for FoldedFxHasher {
    #[inline] fn write(&mut self, bytes: &[u8])   { self.inner.write(bytes); }
    #[inline] fn write_u8(&mut self, i: u8)       { self.inner.write_u8(i); }
    #[inline] fn write_u16(&mut self, i: u16)     { self.inner.write_u16(i); }
    #[inline] fn write_u32(&mut self, i: u32)     { self.inner.write_u32(i); }
    #[inline] fn write_u64(&mut self, i: u64)     { self.inner.write_u64(i); }
    #[inline] fn write_usize(&mut self, i: usize) { self.inner.write_usize(i); }
    #[inline] fn write_i8(&mut self, i: i8)       { self.inner.write_i8(i); }
    #[inline] fn write_i16(&mut self, i: i16)     { self.inner.write_i16(i); }
    #[inline] fn write_i32(&mut self, i: i32)     { self.inner.write_i32(i); }
    #[inline] fn write_i64(&mut self, i: i64)     { self.inner.write_i64(i); }
    #[inline] fn write_isize(&mut self, i: isize) { self.inner.write_isize(i); }

    #[inline]
    fn finish(&self) -> u64 { folded_multiply(self.inner.finish()) }
}

/// Deterministic [`BuildHasher`] — the same fixed seed every run, every
/// process, matching real `bevy_platform::hash::FixedHasher`'s own name and
/// role (hand-rolled instead of wrapping `foldhash::fast::FixedState` — see
/// this module's own top doc comment). Builds a [`FoldedFxHasher`].
///
/// `HashMap<K, V, FixedHasher>::default()` to use it. No `RandomState`
/// equivalent — see this module's own top doc comment for why.
#[derive(Copy, Clone, Default, Debug)]
pub struct FixedHasher;

impl BuildHasher for FixedHasher {
    type Hasher = FoldedFxHasher;

    #[inline]
    fn build_hasher(&self) -> FoldedFxHasher {
        FoldedFxHasher::with_seed(FIXED_SEED)
    }
}

/// Hashes one value with the deterministic [`FixedHasher`]. Matches real
/// `bevy_platform::hash::fixed_hash_one`'s own name and signature.
#[inline]
pub fn fixed_hash_one(x: impl core::hash::Hash) -> u64 {
    FixedHasher.hash_one(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_input_hashes_the_same_every_time() {
        let a = FixedHasher.hash_one("mid-engine");
        let b = FixedHasher.hash_one("mid-engine");
        assert_eq!(a, b);
    }

    #[test]
    fn different_input_usually_hashes_differently() {
        // Not a formal collision-resistance claim -- just a basic sanity
        // check that this isn't accidentally a constant function.
        assert_ne!(FixedHasher.hash_one(1u64), FixedHasher.hash_one(2u64));
        assert_ne!(FixedHasher.hash_one("abc"), FixedHasher.hash_one("abd"));
    }

    #[test]
    fn with_seed_changes_the_result() {
        let default_seed = FxHasher::default().hash_one_value(42u64);
        let custom_seed = FxHasher::with_seed(FIXED_SEED).hash_one_value(42u64);
        assert_ne!(default_seed, custom_seed);
    }

    // Real parity check against the actual published crate, not a
    // reimplementation of ourselves: `rustc-hash` 1.1.0 (read directly from
    // crates.io source) is this exact algorithm on 64-bit targets -- same
    // multiplier, same rotate-5/xor/wrapping-mul step, same 8/4/2/1-byte
    // chunking in `write`. Zero-seeded `FxHasher::default()` here must
    // match it bit-for-bit. 64-bit only: `rustc-hash` uses a `usize` state
    // and a different constant on 32-bit targets, so its output there
    // legitimately differs from ours -- ours is `u64` state everywhere, so
    // it's deterministic across 32-bit/64-bit (wasm32 vs native) where
    // `rustc-hash`'s isn't. Negative i8/i16/i32 are the one known
    // divergence (see the next test). `rustc-hash` is a dev-dependency
    // only.
    #[cfg(target_pointer_width = "64")]
    #[test]
    fn matches_rustc_hash_1_1_0_bit_for_bit_from_a_zero_seed() {
        use core::hash::Hash;

        fn ours<T: Hash + ?Sized>(v: &T) -> u64 {
            let mut h = FxHasher::default();
            v.hash(&mut h);
            h.finish()
        }
        fn theirs<T: Hash + ?Sized>(v: &T) -> u64 {
            let mut h = rustc_hash::FxHasher::default();
            v.hash(&mut h);
            h.finish()
        }

        assert_eq!(ours(&0xDEAD_BEEF_u64), theirs(&0xDEAD_BEEF_u64));
        assert_eq!(ours(&7u8), theirs(&7u8));
        assert_eq!(ours(&1234u16), theirs(&1234u16));
        assert_eq!(ours(&0xCAFE_F00D_u32), theirs(&0xCAFE_F00D_u32));
        assert_eq!(ours(&-5i64), theirs(&-5i64));
        assert_eq!(ours(&(1u32, 2u64, 3u8)), theirs(&(1u32, 2u64, 3u8)));
        // Every `write` tail length (0..=15 bytes), so the 8/4/2/1-byte
        // chunking branches all get exercised.
        let bytes: [u8; 15] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        for len in 0..=bytes.len() {
            let mut a = FxHasher::default();
            a.write(&bytes[..len]);
            let mut b = rustc_hash::FxHasher::default();
            b.write(&bytes[..len]);
            assert_eq!(a.finish(), b.finish(), "write() mismatch at len {len}");
        }
    }

    // The one known, deliberate divergence from `rustc-hash` 1.1.0: narrow
    // signed integers. `rustc-hash` leaves `write_i32` etc. at the trait
    // default (reinterpret as the unsigned type, zero-extend); this copy
    // sign-extends instead, matching `mid-collections`'s own `FxHasher`.
    // Pinned here so a future "simplify by deleting the signed overrides"
    // change is a conscious decision, not an accident.
    #[cfg(target_pointer_width = "64")]
    #[test]
    fn negative_narrow_signed_ints_sign_extend_unlike_rustc_hash() {
        use core::hash::Hash;

        let mut ours = FxHasher::default();
        (-1i32).hash(&mut ours);
        let mut theirs = rustc_hash::FxHasher::default();
        (-1i32).hash(&mut theirs);
        assert_ne!(ours.finish(), theirs.finish());

        // Non-negative values are unaffected -- sign-extension and
        // zero-extension agree.
        let mut ours = FxHasher::default();
        5i32.hash(&mut ours);
        let mut theirs = rustc_hash::FxHasher::default();
        5i32.hash(&mut theirs);
        assert_eq!(ours.finish(), theirs.finish());
    }

    // --- FoldedFxHasher / FixedHasher ----------------------------------

    /// Independent restatement of the fold, so a bug in `folded_multiply`
    /// cannot hide by sitting on both sides of the comparison.
    fn fold_reference(x: u64) -> u64 {
        let p = u128::from(x) * u128::from(0x9E37_79B9_7F4A_7C15u64);
        (p as u64) ^ ((p >> 64) as u64)
    }

    /// Feeds `feed` to a `FoldedFxHasher` and to a plain `FxHasher` with the
    /// same seed; returns the folded hasher's `finish` and the reference
    /// fold of the Fx state.
    fn folded_and_reference(seed: u64, feed: impl Fn(&mut dyn Hasher)) -> (u64, u64) {
        let mut folded = FoldedFxHasher::with_seed(seed);
        feed(&mut folded);
        let mut fx = FxHasher::with_seed(seed);
        feed(&mut fx);
        (folded.finish(), fold_reference(fx.finish()))
    }

    #[test]
    fn folded_hasher_is_fx_accumulation_then_the_fold() {
        macro_rules! agrees {
            ($seed:expr, $feed:expr) => {{
                let (folded, reference) = folded_and_reference($seed, $feed);
                assert_eq!(folded, reference);
            }};
        }
        for seed in [0u64, FIXED_SEED, u64::MAX] {
            agrees!(seed, |h| h.write_u8(200));
            agrees!(seed, |h| h.write_u16(60_000));
            agrees!(seed, |h| h.write_u32(4_000_000_000));
            agrees!(seed, |h| h.write_u64(u64::MAX - 7));
            agrees!(seed, |h| h.write_usize(123_456));
            // Negative values: sign extension must carry through, not
            // zero extension.
            agrees!(seed, |h| h.write_i8(-3));
            agrees!(seed, |h| h.write_i16(-3));
            agrees!(seed, |h| h.write_i32(-3));
            agrees!(seed, |h| h.write_i64(-3));
            agrees!(seed, |h| h.write_isize(-3));
            let bytes: [u8; 20] = core::array::from_fn(|i| (i as u8).wrapping_mul(37).wrapping_add(11));
            for len in 0..=bytes.len() {
                agrees!(seed, |h| h.write(&bytes[..len]));
            }
            // Several writes in a row, the way a tuple or struct key hashes.
            agrees!(seed, |h| {
                h.write_u32(7);
                h.write_u32(9);
                h.write_u8(1);
            });
        }
    }

    #[test]
    fn fixed_hasher_starts_from_the_fixed_seed_and_folds() {
        // No writes at all: just the seed, folded.
        assert_eq!(FixedHasher.build_hasher().finish(), fold_reference(FIXED_SEED));
        // One u64: the Fx step from the fixed seed, then the fold.
        let fx = (FIXED_SEED.rotate_left(5) ^ 42).wrapping_mul(MULTIPLIER);
        assert_eq!(FixedHasher.hash_one(42u64), fold_reference(fx));
        assert_eq!(fixed_hash_one(42u64), fold_reference(fx));
    }

    #[test]
    fn fx_hasher_finish_is_still_the_raw_state() {
        // FoldedFxHasher folds `FxHasher::finish()`; if FxHasher ever grew a
        // finalizer of its own, it would be applied twice and its parity
        // with rustc-hash 1.1.0 would be gone.
        let mut fx = FxHasher::with_seed(FIXED_SEED);
        fx.write_u64(42);
        assert_eq!(fx.finish(), (FIXED_SEED.rotate_left(5) ^ 42).wrapping_mul(MULTIPLIER));
    }

    /// How many of 65,536 low-16-bit values, and how many of the 128
    /// top-7-bit values (hashbrown's tag), `keys` reach under `hash`.
    fn distinct_low16_and_top7(keys: impl Iterator<Item = u64>, hash: impl Fn(u64) -> u64) -> (usize, usize) {
        let mut low = [false; 1 << 16];
        let mut top = [false; 128];
        for key in keys {
            let h = hash(key);
            low[(h & 0xFFFF) as usize] = true;
            top[(h >> 57) as usize] = true;
        }
        let count = |seen: &[bool]| seen.iter().filter(|hit| **hit).count();
        (count(&low), count(&top))
    }

    #[test]
    fn fixed_hasher_spreads_keys_that_share_their_low_bits() {
        // 10,000 keys into 65,536 slots: an ideal hash occupies about 9,270.
        // On these shapes the folded hash measured between roughly 7,400 and
        // 10,000 when this was written; a hash that ignores the high key
        // bits lands in the tens (a bare `FxHasher` reaches 16 on
        // `i << 12`, see the next test). The bar sits between the two.
        let shapes: [(&str, fn(u64) -> u64); 7] = [
            ("sequential", |i| i),
            ("i << 12", |i| i << 12),
            ("i << 32", |i| i << 32),
            ("i << 48", |i| i << 48),
            ("i * 1000", |i| i * 1000),
            ("f64 bits", |i| (i as f64).to_bits()),
            ("i * 65536 + 7", |i| i * 65_536 + 7),
        ];
        for (name, key) in shapes {
            let (low, top) = distinct_low16_and_top7((0..10_000u64).map(key), fixed_hash_one);
            assert!(low > 5_000, "{name}: only {low} distinct low-16-bit values");
            assert!(top >= 120, "{name}: only {top} of 128 top-7-bit values");
        }
    }

    #[test]
    fn a_bare_fx_hasher_does_not_spread_shifted_keys() {
        // The reason FoldedFxHasher exists, pinned: Fx's low output bits
        // depend only on the key's low bits, so keys differing only above
        // bit 12 reach at most 16 distinct low-16-bit values.
        let (low, _) = distinct_low16_and_top7(
            (0..10_000u64).map(|i| i << 12),
            |k| {
                let mut h = FxHasher::with_seed(FIXED_SEED);
                h.write_u64(k);
                h.finish()
            },
        );
        assert!(low <= 16, "unexpectedly spread: {low}");
    }

    trait HashOneValue: Hasher + Sized {
        fn hash_one_value(mut self, v: u64) -> u64 {
            self.write_u64(v);
            self.finish()
        }
    }
    impl<H: Hasher> HashOneValue for H {}
}
