// crates/chemistry_core/src/fx_hash.rs
//! Minimal FxHash — the same algorithm rustc uses internally for its own
//! compiler data structures (originally from Firefox). Not cryptographic,
//! not DoS-hardened like std's default SipHash — deliberately: this hasher
//! is only ever used for SpatialHash's cell keys, which are (i32,i32,i32)
//! derived from atom positions this sim computes itself, not attacker-
//! supplied strings/integers arriving over a network. There's no
//! adversarial-input threat model for a local physics sim to defend
//! against, so SipHash's HashDoS resistance is pure overhead here — same
//! reasoning rustc applies to its own internal maps.
//!
//! Written by hand instead of depending on the `rustc-hash` crate, per the
//! "no external dependency" rule for this crate.

use std::hash::{BuildHasherDefault, Hasher};

const SEED: u64 = 0x51_7c_c1_b7_27_22_0a_95;

#[derive(Default)]
pub struct FxHasher {
    hash: u64,
}

impl FxHasher {
    #[inline]
    fn add(&mut self, w: u64) {
        self.hash = (self.hash.rotate_left(5) ^ w).wrapping_mul(SEED);
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

/// `HashMap<K, V, FxBuildHasher>::default()` to use it.
pub type FxBuildHasher = BuildHasherDefault<FxHasher>;
