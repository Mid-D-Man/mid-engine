// crates/mid-math/src/string_id.rs
//! Compile-time FNV-1a string hashing.
//!
//! StringId is a 64-bit hash of a string produced at compile time via const fn.
//! Zero runtime cost for constant IDs. Safe to use in HashMap/HashSet.
//!
//! Engine uses: component type registration, system names, asset keys,
//! packet type dispatch in DixScript.
//!
//! Algorithm: FNV-1a 64-bit (Fowler–Noll–Vo).
//!   hash = 14695981039346656037  (FNV offset basis)
//!   for each byte: hash ^= byte; hash *= 1099511628211  (FNV prime)
//!
//! Collisions are astronomically rare for typical identifier strings.
//! If a collision is detected at startup, use a different string.

use core::fmt;

/// Compile-time or runtime FNV-1a 64-bit string hash.
///
/// Two StringIds are equal if and only if their hashes are equal.
/// Construct at compile time with [`sid!`] or at runtime with [`StringId::new`].
///
/// # Example
/// ```rust
/// use mid_math::{StringId, sid};
///
/// // Compile-time — zero cost at runtime
/// const POSITION: StringId = sid!("Position");
/// const VELOCITY: StringId = sid!("Velocity");
///
/// assert_ne!(POSITION, VELOCITY);
/// ```
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct StringId(pub u64);

const FNV_OFFSET: u64 = 0xcbf29ce484222325;
const FNV_PRIME:  u64 = 0x00000100000001b3;

impl StringId {
    /// Compute the FNV-1a hash of `s` at compile time.
    ///
    /// This is a `const fn` — if called with a string literal or const `&str`,
    /// the hash is computed entirely at compile time with zero runtime cost.
    #[inline]
    pub const fn new(s: &str) -> Self {
        let bytes = s.as_bytes();
        let mut hash = FNV_OFFSET;
        let mut i    = 0usize;
        while i < bytes.len() {
            hash ^= bytes[i] as u64;
            hash  = hash.wrapping_mul(FNV_PRIME);
            i    += 1;
        }
        Self(hash)
    }

    /// Return the raw 64-bit hash value.
    #[inline(always)]
    pub const fn raw(self) -> u64 { self.0 }

    /// True if this ID was constructed from the empty string `""`.
    #[inline(always)]
    pub const fn is_empty_string(self) -> bool { self.0 == FNV_OFFSET }
}

/// Shorthand macro for compile-time StringId construction.
///
/// ```rust
/// use mid_math::sid;
/// const TRANSFORM: mid_math::StringId = sid!("Transform");
/// ```
#[macro_export]
macro_rules! sid {
    ($s:expr) => {
        $crate::StringId::new($s)
    };
}

impl fmt::Debug for StringId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "StringId({:#018x})", self.0)
    }
}

impl fmt::Display for StringId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:#018x}", self.0)
    }
}

impl From<&str> for StringId {
    #[inline]
    fn from(s: &str) -> Self { Self::new(s) }
}

impl From<StringId> for u64 {
    #[inline]
    fn from(id: StringId) -> u64 { id.0 }
  }
