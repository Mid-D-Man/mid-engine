// crates/mid-common/src/string/fixed_str.rs
//! Stack-allocated, null-terminated fixed-capacity string.
//!
//! `FixedStr<N>` holds up to `N-1` UTF-8 bytes plus a null terminator.
//! No heap allocation, no `std`, FFI safe.
//!
//! Engine uses: entity names, component type names, asset keys,
//! log category labels, DixScript identifier buffers.
//!
//! Layout: `[u8; N]` buffer + `usize` length. The buffer is always
//! null-terminated at `buf[len]`, so `buf[N-1]` is always available
//! for the null terminator when the string is at max capacity.
//!
//! Inspired by Blender's C-style char arrays + BLI_string operations,
//! but type-safe and with explicit capacity checking.

use core::fmt;
use core::ffi::CStr;
use crate::string::{NulStr, StrRef};

// ─────────────────────────────────────────────────────────────────────────────
// FixedStr<N>
// ─────────────────────────────────────────────────────────────────────────────

/// Stack-allocated string with capacity for `N-1` bytes + null terminator.
///
/// Always null-terminated at `buf[len]`. FFI safe via `as_ptr()`.
/// Attempting to write past capacity silently truncates (no panic).
///
/// ```rust
/// use mid_common::string::FixedStr;
///
/// let mut name: FixedStr<64> = FixedStr::new();
/// name.push_str("Player");
/// name.push_str(".001");
/// assert_eq!(name.as_str(), "Player.001");
/// assert_eq!(name.len(), 10);
/// ```
#[derive(Clone, Copy)]
pub struct FixedStr<const N: usize> {
    buf: [u8; N],
    len: usize,
}

impl<const N: usize> FixedStr<N> {
    const _ASSERT: () = assert!(N >= 1, "FixedStr<N>: N must be at least 1 for null terminator");

    /// Maximum number of content bytes (excludes null terminator).
    pub const CAPACITY: usize = N - 1;

    // ── Constructors ──────────────────────────────────────────────────────────

    /// Create an empty `FixedStr`.
    #[inline]
    pub const fn new() -> Self {
        let _ = Self::_ASSERT;
        let mut buf = [0u8; N];
        buf[0] = 0;
        Self { buf, len: 0 }
    }

    /// Create from a string slice. Truncates silently if `s` exceeds capacity.
    /// Ensures no partial UTF-8 codepoint at the truncation boundary.
    pub fn from_str(s: &str) -> Self {
        let mut out = Self::new();
        out.push_str(s);
        out
    }

    /// Create from a byte slice. Bytes must be valid UTF-8.
    /// Returns `None` if bytes contain invalid UTF-8 or a null byte.
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        let s = core::str::from_utf8(bytes).ok()?;
        if s.contains('\0') { return None; }
        Some(Self::from_str(s))
    }

    // ── Write ─────────────────────────────────────────────────────────────────

    /// Append bytes from `s`, truncating at capacity.
    /// Returns the number of bytes actually written.
    pub fn push_str(&mut self, s: &str) -> usize {
        let remaining = Self::CAPACITY - self.len;
        if remaining == 0 { return 0; }

        // Find the largest valid UTF-8 prefix of `s` that fits
        let bytes = s.as_bytes();
        let copy_len = bytes.len().min(remaining);
        // Walk back from copy_len to find a valid char boundary
        let copy_len = floor_char_boundary(s, copy_len);

        self.buf[self.len..self.len + copy_len].copy_from_slice(&bytes[..copy_len]);
        self.len += copy_len;
        self.buf[self.len] = 0; // maintain null terminator
        copy_len
    }

    /// Append a single ASCII character. Ignores if at capacity or non-ASCII.
    #[inline]
    pub fn push_ascii(&mut self, c: u8) -> bool {
        if self.len >= Self::CAPACITY || c == 0 { return false; }
        self.buf[self.len] = c;
        self.len += 1;
        self.buf[self.len] = 0;
        true
    }

    /// Clear content, reset to empty string.
    #[inline]
    pub fn clear(&mut self) {
        self.len = 0;
        self.buf[0] = 0;
    }

    /// Truncate to `new_len` bytes. Must fall on a UTF-8 char boundary.
    /// Clamped to current length if `new_len > len`.
    pub fn truncate(&mut self, new_len: usize) {
        let new_len = new_len.min(self.len);
        // Ensure char boundary
        let new_len = floor_char_boundary(self.as_str(), new_len);
        self.len = new_len;
        self.buf[self.len] = 0;
    }

    // ── Query ─────────────────────────────────────────────────────────────────

    #[inline]
    pub fn len(&self) -> usize { self.len }

    #[inline]
    pub fn is_empty(&self) -> bool { self.len == 0 }

    #[inline]
    pub fn is_full(&self) -> bool { self.len >= Self::CAPACITY }

    #[inline]
    pub fn remaining_capacity(&self) -> usize { Self::CAPACITY - self.len }

    // ── Views ─────────────────────────────────────────────────────────────────

    /// View as `&str`.
    #[inline]
    pub fn as_str(&self) -> &str {
        // SAFETY: we only ever write valid UTF-8 (enforced by push_str/from_str)
        unsafe { core::str::from_utf8_unchecked(&self.buf[..self.len]) }
    }

    /// View as a `StrRef`.
    #[inline]
    pub fn as_str_ref(&self) -> StrRef<'_> {
        StrRef::new(self.as_str())
    }

    /// View as a `NulStr`. The buffer is always null-terminated.
    #[inline]
    pub fn as_nul_str(&self) -> NulStr<'_> {
        // SAFETY: buf[len] is always 0; no interior nulls (ensured by push_str)
        let cs = unsafe { CStr::from_ptr(self.buf.as_ptr() as *const core::ffi::c_char) };
        NulStr::from_cstr(cs)
    }

    /// Raw null-terminated pointer. FFI safe.
    #[inline]
    pub fn as_ptr(&self) -> *const u8 { self.buf.as_ptr() }

    /// The full backing buffer including null terminator.
    #[inline]
    pub fn as_bytes_with_nul(&self) -> &[u8] { &self.buf[..self.len + 1] }

    // ── Numeric suffix utilities (from Blender's BLI_string_split_name_number) ─

    /// Split `"Bone.001"` into `("Bone", 1, '.')`. Returns `(self, 0, '\0')` if no suffix.
    pub fn split_name_number(&self, delim: char) -> (StrRef<'_>, u32) {
        StrRef::new(self.as_str()).split_name_number(delim)
    }

    /// Set content from a `&str`. Equivalent to `clear()` + `push_str(s)`.
    #[inline]
    pub fn set(&mut self, s: &str) -> usize {
        self.clear();
        self.push_str(s)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Find the largest byte index ≤ `index` that is a valid UTF-8 char boundary in `s`.
#[inline]
fn floor_char_boundary(s: &str, index: usize) -> usize {
    if index >= s.len() { return s.len(); }
    // Walk back until we hit a char boundary
    let bytes = s.as_bytes();
    let mut i = index;
    // UTF-8 continuation bytes are 10xxxxxx (0x80..=0xBF)
    while i > 0 && (bytes[i] & 0xC0) == 0x80 {
        i -= 1;
    }
    i
}

// ─────────────────────────────────────────────────────────────────────────────
// Trait impls
// ─────────────────────────────────────────────────────────────────────────────

impl<const N: usize> Default for FixedStr<N> {
    #[inline]
    fn default() -> Self { Self::new() }
}

impl<const N: usize> PartialEq for FixedStr<N> {
    fn eq(&self, other: &Self) -> bool { self.as_str() == other.as_str() }
}

impl<const N: usize> Eq for FixedStr<N> {}

impl<const N: usize> PartialEq<str> for FixedStr<N> {
    fn eq(&self, other: &str) -> bool { self.as_str() == other }
}

impl<const N: usize> PartialEq<&str> for FixedStr<N> {
    fn eq(&self, other: &&str) -> bool { self.as_str() == *other }
}

impl<const N: usize> core::hash::Hash for FixedStr<N> {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.as_str().hash(state);
    }
}

impl<const N: usize> core::ops::Deref for FixedStr<N> {
    type Target = str;
    #[inline]
    fn deref(&self) -> &str { self.as_str() }
}

impl<const N: usize> fmt::Debug for FixedStr<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FixedStr<{}>({:?})", N, self.as_str())
    }
}

impl<const N: usize> fmt::Display for FixedStr<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl<const N: usize> From<&str> for FixedStr<N> {
    #[inline]
    fn from(s: &str) -> Self { Self::from_str(s) }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_push() {
        let mut s: FixedStr<8> = FixedStr::new(); // capacity 7
        assert_eq!(s.push_str("hello"), 5);
        assert_eq!(s.as_str(), "hello");
        assert_eq!(s.len(), 5);
        assert!(!s.is_full());
    }

    #[test]
    fn truncate_at_capacity() {
        let mut s: FixedStr<8> = FixedStr::new(); // capacity 7
        let written = s.push_str("12345678"); // 8 chars, only 7 fit
        assert_eq!(written, 7);
        assert_eq!(s.as_str(), "1234567");
        assert!(s.is_full());
    }

    #[test]
    fn utf8_truncation_safe() {
        let mut s: FixedStr<5> = FixedStr::new(); // capacity 4
        // '°' is 2 bytes (0xc2 0xb0). "ab°c" = 5 bytes: 'a'(1) + 'b'(1) +
        // '°'(2) = 4 bytes fits exactly in the 4-byte capacity without
        // splitting the 2-byte char; 'c' is the one that doesn't fit.
        // (Previously asserted 2/"ab" here -- verified independently
        // that's wrong: 'a'+'b'+'°' is 4 bytes total, which fits exactly,
        // it doesn't overflow into '°' the way the old comment assumed.)
        let written = s.push_str("ab°c");
        assert_eq!(written, 4);
        assert_eq!(s.as_str(), "ab°");

        // The actual split-avoidance case the test's name promises --
        // the above no longer exercises it now that the byte math is
        // fixed, so this covers what that one used to claim to: 'a'+'b'+
        // 'c' = 3 bytes, then '°' would need bytes 3..5 but only byte 3
        // is available before hitting the 4-byte capacity, so it must be
        // excluded whole rather than half-copied into invalid UTF-8.
        let mut s2: FixedStr<5> = FixedStr::new();
        let written2 = s2.push_str("abc°d");
        assert_eq!(written2, 3);
        assert_eq!(s2.as_str(), "abc");
    }

    #[test]
    fn null_termination() {
        let s: FixedStr<16> = FixedStr::from_str("hello");
        let ptr = s.as_ptr();
        // Byte after content must be null
        assert_eq!(unsafe { *ptr.add(5) }, 0);
    }

    #[test]
    fn nul_str_view() {
        let s: FixedStr<16> = FixedStr::from_str("engine");
        let ns = s.as_nul_str();
        assert_eq!(ns.len(), 6);
        assert_eq!(ns.to_str().unwrap(), "engine");
    }

    #[test]
    fn split_name_number() {
        let s: FixedStr<32> = FixedStr::from_str("Bone.001");
        let (name, num) = s.split_name_number('.');
        assert_eq!(name.as_str(), "Bone");
        assert_eq!(num, 1);
    }

    #[test]
    fn set_and_clear() {
        let mut s: FixedStr<16> = FixedStr::from_str("first");
        s.set("second");
        assert_eq!(s.as_str(), "second");
        s.clear();
        assert!(s.is_empty());
        assert_eq!(s.as_str(), "");
    }
  }
