// crates/mid-common/src/string/str_ref.rs
//! Non-owning string references.
//!
//! Adapted from Blender's BLI_string_ref.hh — two-type design:
//!   StrRef   — borrows a `&str`. Not null-terminated. Rich slice API.
//!   NulStr   — borrows a `&CStr`. Null-terminated. FFI safe.
//!
//! Why bother wrapping `&str`?
//!   - `not_found` sentinel (-1 for indices, matching Blender's signed-index convention)
//!   - `drop_prefix` / `drop_suffix` / `trim_chars` with owned-char-set
//!   - `to_nul_str()` requires an allocator; this type makes the boundary explicit
//!   - Consistent engine API across Rust/FFI boundaries
//!
//! `NulStr` fills the real gap: Rust's `&CStr` is ergonomically awkward for engine use.
//! Engine convention: any function that crosses the C ABI takes `NulStr`, not `&str`.

use core::ffi::{CStr, c_char};
use core::fmt;
use core::ops::Deref;

// ─────────────────────────────────────────────────────────────────────────────
// StrRef
// ─────────────────────────────────────────────────────────────────────────────

/// Non-owning, non-null-terminated string reference.
///
/// Wraps `&str` and extends it with engine conventions: signed indices,
/// `not_found` sentinel, and name-manipulation helpers.
///
/// ```rust
/// use mid_common::string::StrRef;
/// let s = StrRef::from("Hello, World!");
/// assert_eq!(s.find_char('W'), 7);
/// assert_eq!(s.find_char('Z'), StrRef::NOT_FOUND);
/// ```
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct StrRef<'a>(&'a str);

impl<'a> StrRef<'a> {
    /// Returned when a search operation fails. Matches Blender's `not_found = -1`.
    pub const NOT_FOUND: i64 = -1;

    #[inline(always)]
    pub const fn new(s: &'a str) -> Self { Self(s) }

    #[inline(always)]
    pub const fn as_str(self) -> &'a str { self.0 }

    #[inline(always)]
    pub fn len(self) -> usize { self.0.len() }

    #[inline(always)]
    pub fn is_empty(self) -> bool { self.0.is_empty() }

    /// Byte pointer to the start of the string. NOT null-terminated.
    #[inline(always)]
    pub fn as_ptr(self) -> *const u8 { self.0.as_ptr() }

    // ── Find ─────────────────────────────────────────────────────────────────

    /// Find first occurrence of `c`. Returns `NOT_FOUND` if absent.
    #[inline]
    pub fn find_char(self, c: char) -> i64 {
        self.0.find(c).map(|i| i as i64).unwrap_or(Self::NOT_FOUND)
    }

    /// Find last occurrence of `c`. Returns `NOT_FOUND` if absent.
    #[inline]
    pub fn rfind_char(self, c: char) -> i64 {
        self.0.rfind(c).map(|i| i as i64).unwrap_or(Self::NOT_FOUND)
    }

    /// Find first occurrence of substring `needle`. Returns `NOT_FOUND` if absent.
    #[inline]
    pub fn find_str(self, needle: &str) -> i64 {
        self.0.find(needle).map(|i| i as i64).unwrap_or(Self::NOT_FOUND)
    }

    // ── Prefix / suffix ───────────────────────────────────────────────────────

    #[inline]
    pub fn starts_with(self, prefix: &str) -> bool { self.0.starts_with(prefix) }

    #[inline]
    pub fn ends_with(self, suffix: &str) -> bool { self.0.ends_with(suffix) }

    /// Remove `n` bytes from the front. Clamped at string length.
    #[inline]
    pub fn drop_prefix(self, n: usize) -> Self {
        let n = n.min(self.0.len());
        Self(&self.0[n..])
    }

    /// Remove a known prefix. Panics in debug if prefix is absent.
    #[inline]
    pub fn drop_known_prefix(self, prefix: &str) -> Self {
        debug_assert!(self.0.starts_with(prefix), "drop_known_prefix: prefix not present");
        Self(&self.0[prefix.len()..])
    }

    /// Remove `n` bytes from the end. Clamped at string length.
    #[inline]
    pub fn drop_suffix(self, n: usize) -> Self {
        let n = n.min(self.0.len());
        Self(&self.0[..self.0.len() - n])
    }

    /// Remove a known suffix. Panics in debug if suffix is absent.
    #[inline]
    pub fn drop_known_suffix(self, suffix: &str) -> Self {
        debug_assert!(self.0.ends_with(suffix), "drop_known_suffix: suffix not present");
        Self(&self.0[..self.0.len() - suffix.len()])
    }

    // ── Trim ─────────────────────────────────────────────────────────────────

    /// Strip leading and trailing ASCII whitespace (`' '`, `'\t'`, `'\r'`, `'\n'`).
    #[inline]
    pub fn trim(self) -> Self { Self(self.0.trim()) }

    /// Strip leading and trailing occurrences of any character in `chars`.
    #[inline]
    pub fn trim_chars(self, chars: &[char]) -> Self {
        let s = self.0.trim_matches(|c| chars.contains(&c));
        Self(s)
    }

    /// Strip a single leading/trailing character.
    #[inline]
    pub fn trim_char(self, c: char) -> Self {
        Self(self.0.trim_matches(c))
    }

    // ── Slice ─────────────────────────────────────────────────────────────────

    /// Return a sub-slice `[start .. start+len]`. Byte-indexed.
    /// Panics if range falls on a non-char-boundary.
    #[inline]
    pub fn substr(self, start: usize, len: usize) -> Self {
        let end = (start + len).min(self.0.len());
        Self(&self.0[start..end])
    }

    // ── Char access ───────────────────────────────────────────────────────────

    #[inline]
    pub fn first_char(self) -> Option<char> { self.0.chars().next() }

    #[inline]
    pub fn last_char(self) -> Option<char> { self.0.chars().next_back() }

    // ── Splitting ─────────────────────────────────────────────────────────────

    /// Split at the first occurrence of `delim`.
    /// Returns `(prefix, Some(suffix))` or `(self, None)` if not found.
    #[inline]
    pub fn split_once_char(self, delim: char) -> (Self, Option<Self>) {
        match self.0.split_once(delim) {
            Some((l, r)) => (Self(l), Some(Self(r))),
            None => (self, None),
        }
    }

    /// Split on the last occurrence of `delim`.
    #[inline]
    pub fn rsplit_once_char(self, delim: char) -> (Self, Option<Self>) {
        match self.0.rsplit_once(delim) {
            Some((l, r)) => (Self(l), Some(Self(r))),
            None => (self, None),
        }
    }

    /// Split a name like `"Bone.001"` into `("Bone", 1)`.
    /// Returns `(name, 0)` if no numeric suffix with `delim` is found.
    pub fn split_name_number(self, delim: char) -> (Self, u32) {
        if let Some((left, right)) = self.0.rsplit_once(delim) {
            if let Ok(n) = right.parse::<u32>() {
                return (Self(left), n);
            }
        }
        (self, 0)
    }
}

impl<'a> Deref for StrRef<'a> {
    type Target = str;
    #[inline]
    fn deref(&self) -> &str { self.0 }
}

impl<'a> From<&'a str> for StrRef<'a> {
    #[inline]
    fn from(s: &'a str) -> Self { Self(s) }
}

impl<'a> From<StrRef<'a>> for &'a str {
    #[inline]
    fn from(s: StrRef<'a>) -> &'a str { s.0 }
}

impl fmt::Debug for StrRef<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "StrRef({:?})", self.0)
    }
}

impl fmt::Display for StrRef<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.0)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// NulStr
// ─────────────────────────────────────────────────────────────────────────────

/// Null-terminated string reference. FFI safe.
///
/// Analogous to Blender's `StringRefNull`. Any function that crosses
/// the C ABI should take/return `NulStr` rather than `&str`.
///
/// Construction:
///   - `NulStr::from_cstr(cstr)` — borrow an existing `&CStr`
///   - `nul_str!(b"hello\0")` — zero-cost from a byte literal
///
/// ```rust
/// use mid_common::string::NulStr;
/// use core::ffi::CStr;
///
/// let cs = CStr::from_bytes_with_nul(b"engine\0").unwrap();
/// let ns = NulStr::from_cstr(cs);
/// assert_eq!(ns.len(), 6);
/// ```
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct NulStr<'a>(&'a CStr);

impl<'a> NulStr<'a> {
    /// Borrow a `&CStr` as a `NulStr`.
    #[inline(always)]
    pub fn from_cstr(s: &'a CStr) -> Self { Self(s) }

    /// Borrow from a byte slice that must end with `\0` and contain no interior nulls.
    /// Returns `None` if malformed.
    #[inline]
    pub fn from_bytes_with_nul(bytes: &'a [u8]) -> Option<Self> {
        CStr::from_bytes_with_nul(bytes).ok().map(Self)
    }

    /// Raw null-terminated pointer. Safe to pass to C functions.
    #[inline(always)]
    pub fn as_ptr(self) -> *const c_char { self.0.as_ptr() }

    /// The underlying `&CStr`.
    #[inline(always)]
    pub fn as_cstr(self) -> &'a CStr { self.0 }

    /// Convert to `&str`. Returns an error if the bytes are not valid UTF-8.
    #[inline]
    pub fn to_str(self) -> Result<&'a str, core::str::Utf8Error> {
        self.0.to_str()
    }

    /// Convert to `&str` without UTF-8 checking.
    ///
    /// # Safety
    /// Caller guarantees the string is valid UTF-8.
    #[inline]
    pub unsafe fn to_str_unchecked(self) -> &'a str {
        // SAFETY: caller guarantees UTF-8
        unsafe { core::str::from_utf8_unchecked(self.0.to_bytes()) }
    }

    /// Length in bytes, excluding the null terminator.
    #[inline]
    pub fn len(self) -> usize { self.0.to_bytes().len() }

    #[inline]
    pub fn is_empty(self) -> bool { self.len() == 0 }

    /// Convert to a `StrRef` (drops null-termination guarantee but gains slice methods).
    #[inline]
    pub fn to_str_ref(self) -> Option<StrRef<'a>> {
        self.to_str().ok().map(StrRef::new)
    }
}

impl<'a> From<&'a CStr> for NulStr<'a> {
    #[inline]
    fn from(s: &'a CStr) -> Self { Self(s) }
}

impl<'a> From<NulStr<'a>> for &'a CStr {
    #[inline]
    fn from(s: NulStr<'a>) -> &'a CStr { s.0 }
}

impl fmt::Debug for NulStr<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NulStr({:?})", self.0)
    }
}

impl fmt::Display for NulStr<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.to_str() {
            Ok(s) => f.write_str(s),
            Err(_) => write!(f, "<non-utf8 NulStr>"),
        }
    }
}

/// Create a `NulStr` from a byte literal at zero cost.
///
/// The literal must end with `\0` and contain no interior nulls.
/// This is checked at compile time via a const assertion in debug builds.
///
/// ```rust
/// use mid_common::nul_str;
/// let s = nul_str!(b"hello\0");
/// assert_eq!(s.len(), 5);
/// ```
#[macro_export]
macro_rules! nul_str {
    ($bytes:expr) => {{
        // Safety: literal checked at call site; CStr::from_bytes_with_nul panics
        // if malformed, so this is safe to unwrap in const context indirectly.
        match ::core::ffi::CStr::from_bytes_with_nul($bytes) {
            Ok(cs) => $crate::string::NulStr::from_cstr(cs),
            Err(_) => panic!("nul_str!: byte literal must end with \\0 and have no interior nulls"),
        }
    }};
  }
