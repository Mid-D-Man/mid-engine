// crates/mid-common/src/ffi/string.rs
//! C-ABI exports for mid-common string utilities.
//!
//! Exports:
//!   CFixedStr32 / CFixedStr64 / CFixedStr256  — concrete FixedStr<N> variants
//!   MidStringSearch (opaque)                   — StringSearch<u32> with raw-pointer API
//!   mid_flip_side_name                         — L/R name mirroring
//!   mid_uniquename                             — unique name generation via callback
//!   mid_split_name_number                      — "Bone.001" → ("Bone", 1)
//!   mid_damerau_levenshtein_distance           — edit distance
//!   mid_fuzzy_match_score                      — fuzzy match with score output
//!
//! All string input/output uses null-terminated `*const c_char` / `*mut c_char`.
//! Buffer-writing functions always null-terminate and return bytes written
//! (excluding the null), matching the Blender BLI_strncpy convention.

use core::ffi::{c_char, c_void, CStr};
use alloc::boxed::Box;

use crate::string::{
    FixedStr,
    StringSearch,
    damerau_levenshtein_distance,
    fuzzy_match_score,
    flip_side_name,
    utils::split_name_number,
};

// ═══════════════════════════════════════════════════════════════════════════
//  C types — FixedStr concrete sizes
// ═══════════════════════════════════════════════════════════════════════════

/// `FixedStr<32>` — 32-byte buffer, max 31 content bytes. Entity/bone names.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct CFixedStr32 {
    pub buf: [u8; 32],
    pub len: usize,
}

/// `FixedStr<64>` — 64-byte buffer, max 63 content bytes. Asset keys, system names.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct CFixedStr64 {
    pub buf: [u8; 64],
    pub len: usize,
}

/// `FixedStr<256>` — 256-byte buffer, max 255 content bytes. File paths, descriptions.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct CFixedStr256 {
    pub buf: [u8; 256],
    pub len: usize,
}

// ── Conversion helpers ────────────────────────────────────────────────────────

#[inline(always)]
fn fixed32_to_c(f: &FixedStr<32>) -> CFixedStr32 {
    let mut out = CFixedStr32 { buf: [0u8; 32], len: f.len() };
    out.buf[..f.len()].copy_from_slice(f.as_str().as_bytes());
    out
}

#[inline(always)]
fn fixed32_from_c(c: &CFixedStr32) -> FixedStr<32> {
    let s = core::str::from_utf8(&c.buf[..c.len.min(31)]).unwrap_or("");
    FixedStr::from_str(s)
}

#[inline(always)]
fn fixed64_to_c(f: &FixedStr<64>) -> CFixedStr64 {
    let mut out = CFixedStr64 { buf: [0u8; 64], len: f.len() };
    out.buf[..f.len()].copy_from_slice(f.as_str().as_bytes());
    out
}

#[inline(always)]
fn fixed64_from_c(c: &CFixedStr64) -> FixedStr<64> {
    let s = core::str::from_utf8(&c.buf[..c.len.min(63)]).unwrap_or("");
    FixedStr::from_str(s)
}

#[inline(always)]
fn fixed256_to_c(f: &FixedStr<256>) -> CFixedStr256 {
    let mut out = CFixedStr256 { buf: [0u8; 256], len: f.len() };
    out.buf[..f.len()].copy_from_slice(f.as_str().as_bytes());
    out
}

#[inline(always)]
fn fixed256_from_c(c: &CFixedStr256) -> FixedStr<256> {
    let s = core::str::from_utf8(&c.buf[..c.len.min(255)]).unwrap_or("");
    FixedStr::from_str(s)
}

/// Write `s` into `out[..out_len]`, always null-terminate, return bytes written.
/// Safe to call with out_len == 0 (writes nothing, returns 0).
#[inline]
unsafe fn write_cstr(s: &str, out: *mut c_char, out_len: usize) -> usize {
    if out.is_null() || out_len == 0 { return 0; }
    let bytes = s.as_bytes();
    let cap = out_len - 1; // reserve space for null
    let write = bytes.len().min(cap);
    // Safety: caller guarantees out points to out_len valid writable bytes
    core::ptr::copy_nonoverlapping(bytes.as_ptr(), out as *mut u8, write);
    *out.add(write) = 0;
    write
}

/// Read a null-terminated C string into a &str. Returns "" on null or invalid UTF-8.
#[inline]
unsafe fn read_cstr<'a>(ptr: *const c_char) -> &'a str {
    if ptr.is_null() { return ""; }
    CStr::from_ptr(ptr).to_str().unwrap_or("")
}

// ═══════════════════════════════════════════════════════════════════════════
//  FixedStr32 exports
// ═══════════════════════════════════════════════════════════════════════════

#[no_mangle]
pub extern "C" fn mid_fixed_str32_new() -> CFixedStr32 {
    fixed32_to_c(&FixedStr::new())
}

/// Build from a null-terminated C string. Truncates silently if too long.
#[no_mangle]
pub unsafe extern "C" fn mid_fixed_str32_from_cstr(s: *const c_char) -> CFixedStr32 {
    fixed32_to_c(&FixedStr::from_str(read_cstr(s)))
}

/// Append `s` to `str`. Returns bytes written. Silently truncates at capacity.
#[no_mangle]
pub unsafe extern "C" fn mid_fixed_str32_push_str(
    str: *mut CFixedStr32,
    s: *const c_char,
) -> usize {
    let c = &mut *str;
    let mut f = fixed32_from_c(c);
    let written = f.push_str(read_cstr(s));
    *c = fixed32_to_c(&f);
    written
}

/// Set content to `s`, replacing previous value. Returns bytes written.
#[no_mangle]
pub unsafe extern "C" fn mid_fixed_str32_set(
    str: *mut CFixedStr32,
    s: *const c_char,
) -> usize {
    let c = &mut *str;
    let mut f = fixed32_from_c(c);
    let written = f.set(read_cstr(s));
    *c = fixed32_to_c(&f);
    written
}

/// Clear to empty string.
#[no_mangle]
pub unsafe extern "C" fn mid_fixed_str32_clear(str: *mut CFixedStr32) {
    let c = &mut *str;
    let mut f = fixed32_from_c(c);
    f.clear();
    *c = fixed32_to_c(&f);
}

/// Null-terminated pointer to string content. Valid until next mutation.
#[no_mangle]
pub unsafe extern "C" fn mid_fixed_str32_as_ptr(str: *const CFixedStr32) -> *const c_char {
    (*str).buf.as_ptr() as *const c_char
}

#[no_mangle]
pub unsafe extern "C" fn mid_fixed_str32_len(str: *const CFixedStr32) -> usize {
    (*str).len
}

#[no_mangle]
pub unsafe extern "C" fn mid_fixed_str32_is_empty(str: *const CFixedStr32) -> bool {
    (*str).len == 0
}

#[no_mangle]
pub unsafe extern "C" fn mid_fixed_str32_is_full(str: *const CFixedStr32) -> bool {
    (*str).len >= 31
}

// ═══════════════════════════════════════════════════════════════════════════
//  FixedStr64 exports
// ═══════════════════════════════════════════════════════════════════════════

#[no_mangle] pub extern "C" fn mid_fixed_str64_new() -> CFixedStr64 {
    fixed64_to_c(&FixedStr::new())
}

#[no_mangle] pub unsafe extern "C" fn mid_fixed_str64_from_cstr(s: *const c_char) -> CFixedStr64 {
    fixed64_to_c(&FixedStr::from_str(read_cstr(s)))
}

#[no_mangle] pub unsafe extern "C" fn mid_fixed_str64_push_str(
    str: *mut CFixedStr64, s: *const c_char,
) -> usize {
    let c = &mut *str;
    let mut f = fixed64_from_c(c);
    let w = f.push_str(read_cstr(s));
    *c = fixed64_to_c(&f);
    w
}

#[no_mangle] pub unsafe extern "C" fn mid_fixed_str64_set(
    str: *mut CFixedStr64, s: *const c_char,
) -> usize {
    let c = &mut *str;
    let mut f = fixed64_from_c(c);
    let w = f.set(read_cstr(s));
    *c = fixed64_to_c(&f);
    w
}

#[no_mangle] pub unsafe extern "C" fn mid_fixed_str64_clear(str: *mut CFixedStr64) {
    let c = &mut *str;
    let mut f = fixed64_from_c(c);
    f.clear();
    *c = fixed64_to_c(&f);
}

#[no_mangle] pub unsafe extern "C" fn mid_fixed_str64_as_ptr(str: *const CFixedStr64) -> *const c_char {
    (*str).buf.as_ptr() as *const c_char
}

#[no_mangle] pub unsafe extern "C" fn mid_fixed_str64_len(str: *const CFixedStr64) -> usize { (*str).len }
#[no_mangle] pub unsafe extern "C" fn mid_fixed_str64_is_empty(str: *const CFixedStr64) -> bool { (*str).len == 0 }
#[no_mangle] pub unsafe extern "C" fn mid_fixed_str64_is_full(str: *const CFixedStr64) -> bool { (*str).len >= 63 }

// ═══════════════════════════════════════════════════════════════════════════
//  FixedStr256 exports
// ═══════════════════════════════════════════════════════════════════════════

#[no_mangle] pub extern "C" fn mid_fixed_str256_new() -> CFixedStr256 {
    fixed256_to_c(&FixedStr::new())
}

#[no_mangle] pub unsafe extern "C" fn mid_fixed_str256_from_cstr(s: *const c_char) -> CFixedStr256 {
    fixed256_to_c(&FixedStr::from_str(read_cstr(s)))
}

#[no_mangle] pub unsafe extern "C" fn mid_fixed_str256_push_str(
    str: *mut CFixedStr256, s: *const c_char,
) -> usize {
    let c = &mut *str;
    let mut f = fixed256_from_c(c);
    let w = f.push_str(read_cstr(s));
    *c = fixed256_to_c(&f);
    w
}

#[no_mangle] pub unsafe extern "C" fn mid_fixed_str256_set(
    str: *mut CFixedStr256, s: *const c_char,
) -> usize {
    let c = &mut *str;
    let mut f = fixed256_from_c(c);
    let w = f.set(read_cstr(s));
    *c = fixed256_to_c(&f);
    w
}

#[no_mangle] pub unsafe extern "C" fn mid_fixed_str256_clear(str: *mut CFixedStr256) {
    let c = &mut *str;
    let mut f = fixed256_from_c(c);
    f.clear();
    *c = fixed256_to_c(&f);
}

#[no_mangle] pub unsafe extern "C" fn mid_fixed_str256_as_ptr(str: *const CFixedStr256) -> *const c_char {
    (*str).buf.as_ptr() as *const c_char
}

#[no_mangle] pub unsafe extern "C" fn mid_fixed_str256_len(str: *const CFixedStr256) -> usize { (*str).len }
#[no_mangle] pub unsafe extern "C" fn mid_fixed_str256_is_empty(str: *const CFixedStr256) -> bool { (*str).len == 0 }
#[no_mangle] pub unsafe extern "C" fn mid_fixed_str256_is_full(str: *const CFixedStr256) -> bool { (*str).len >= 255 }

// ═══════════════════════════════════════════════════════════════════════════
//  String utility exports
// ═══════════════════════════════════════════════════════════════════════════

/// Flip L/R side in `name`, write result to `out[..out_len]`.
/// Always null-terminates. Returns bytes written (excluding null).
///
/// ```c
/// char result[64];
/// mid_flip_side_name("Arm.L", result, sizeof(result));
/// // result == "Arm.R"
/// ```
#[no_mangle]
pub unsafe extern "C" fn mid_flip_side_name(
    name:    *const c_char,
    out:     *mut c_char,
    out_len: usize,
) -> usize {
    let flipped = flip_side_name(read_cstr(name));
    write_cstr(&flipped, out, out_len)
}

/// Split `"Bone.001"` into base name and numeric suffix.
///
/// Writes the base name into `out_base[..out_base_len]` (null-terminated).
/// Returns the numeric value (0 if no suffix).
/// `delim` is the delimiter character (e.g. `.` = 0x2E, `_` = 0x5F).
///
/// ```c
/// char base[64];
/// uint32_t n = mid_split_name_number("Bone.001", '.', base, sizeof(base));
/// // base == "Bone", n == 1
/// ```
#[no_mangle]
pub unsafe extern "C" fn mid_split_name_number(
    name:         *const c_char,
    delim:        c_char,
    out_base:     *mut c_char,
    out_base_len: usize,
) -> u32 {
    let s = read_cstr(name);
    let d = char::from(delim as u8);
    let (base, num) = split_name_number(s, d);
    write_cstr(base, out_base, out_base_len);
    num
}

/// Make `name` unique using a caller-supplied "is this name taken?" callback.
///
/// Writes the unique name into `out[..out_len]`. Returns bytes written.
/// Tries `name`, then `name.001`, `name.002`, … up to `name.9999`.
/// `delim` is the separator character between base and number.
///
/// ```c
/// bool my_is_taken(const char* name, void* ctx) {
///     MyNameSet* set = (MyNameSet*)ctx;
///     return name_set_contains(set, name);
/// }
///
/// char result[64];
/// mid_uniquename("Entity", '.', my_is_taken, &my_set, result, sizeof(result));
/// ```
#[no_mangle]
pub unsafe extern "C" fn mid_uniquename(
    name:     *const c_char,
    delim:    c_char,
    is_taken: unsafe extern "C" fn(name: *const c_char, userdata: *mut c_void) -> bool,
    userdata: *mut c_void,
    out:      *mut c_char,
    out_len:  usize,
) -> usize {
    let s = read_cstr(name);
    let d = char::from(delim as u8);

    let result = crate::string::uniquename(s, d, |candidate| {
        // Convert candidate to a temporary null-terminated buffer for the callback.
        // Max name length is bounded by our uniquename impl (base + ".9999").
        let mut buf = [0u8; 512];
        let bytes = candidate.as_bytes();
        let copy = bytes.len().min(511);
        buf[..copy].copy_from_slice(&bytes[..copy]);
        // SAFETY: caller guarantees the callback is valid and doesn't alias out
        is_taken(buf.as_ptr() as *const c_char, userdata)
    });

    write_cstr(&result, out, out_len)
}

/// Damerau-Levenshtein edit distance between two null-terminated strings.
/// Operates at Unicode codepoint level.
///
/// ```c
/// uint64_t dist = mid_damerau_levenshtein_distance("kitten", "sitting");
/// // dist == 3
/// ```
#[no_mangle]
pub unsafe extern "C" fn mid_damerau_levenshtein_distance(
    a: *const c_char,
    b: *const c_char,
) -> u64 {
    damerau_levenshtein_distance(read_cstr(a), read_cstr(b)) as u64
}

/// Fuzzy match `query` against `full`. Writes match score to `*out_score`.
///
/// Returns `true` if there is a match (score written to `*out_score`).
/// Returns `false` if no reasonable match (score is undefined).
/// Score 0 = perfect. Higher = worse.
///
/// ```c
/// uint64_t score;
/// bool matched = mid_fuzzy_match_score("pos", "position", &score);
/// // matched == true, score == 0
/// ```
#[no_mangle]
pub unsafe extern "C" fn mid_fuzzy_match_score(
    query:     *const c_char,
    full:      *const c_char,
    out_score: *mut u64,
) -> bool {
    match fuzzy_match_score(read_cstr(query), read_cstr(full)) {
        Some(score) => {
            if !out_score.is_null() { *out_score = score as u64; }
            true
        }
        None => false,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  StringSearch — opaque pointer API
//
//  Concrete over u32 indices. Caller manages the actual data array;
//  the search stores (name, index, weight) and returns sorted indices.
// ═══════════════════════════════════════════════════════════════════════════

/// Opaque handle to a `StringSearch<u32>`.
pub struct MidStringSearch(StringSearch<u32>);

/// Result entry from `mid_string_search_query`.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct CSearchResult {
    /// User index passed to `mid_string_search_add`.
    pub index: u32,
    /// Match quality. 0 = perfect. Higher = worse.
    pub score: u32,
}

/// Allocate a new empty `StringSearch`. Must be freed with `mid_string_search_destroy`.
#[no_mangle]
pub extern "C" fn mid_string_search_create() -> *mut MidStringSearch {
    Box::into_raw(Box::new(MidStringSearch(StringSearch::new())))
}

/// Free a `StringSearch` created by `mid_string_search_create`.
/// No-op on null pointer.
#[no_mangle]
pub unsafe extern "C" fn mid_string_search_destroy(handle: *mut MidStringSearch) {
    if !handle.is_null() {
        drop(Box::from_raw(handle));
    }
}

/// Add an item to the search set.
///
/// `name`   — display name, null-terminated.
/// `index`  — caller-defined index (returned in query results).
/// `weight` — priority on tie. Higher = sorted first when scores are equal.
#[no_mangle]
pub unsafe extern "C" fn mid_string_search_add(
    handle: *mut MidStringSearch,
    name:   *const c_char,
    index:  u32,
    weight: f32,
) {
    if handle.is_null() { return; }
    (*handle).0.add(read_cstr(name), index, weight);
}

/// Mark an item as recently used (boosts it in results on equal score).
/// `name` must exactly match the name used in `mid_string_search_add`.
#[no_mangle]
pub unsafe extern "C" fn mid_string_search_mark_recent(
    handle: *mut MidStringSearch,
    name:   *const c_char,
) {
    if handle.is_null() { return; }
    (*handle).0.mark_recent(read_cstr(name));
}

/// Query the search set.
///
/// Writes up to `max_results` entries into `out_results`.
/// Returns the number of results written.
/// Results are sorted best-first (lowest score first).
///
/// Pass `query = ""` (or null) to get all items sorted by weight + recency.
///
/// ```c
/// CSearchResult results[32];
/// uint64_t count = mid_string_search_query(search, "arm", results, 32);
/// for (uint64_t i = 0; i < count; i++) {
///     printf("index=%u score=%u\n", results[i].index, results[i].score);
/// }
/// ```
#[no_mangle]
pub unsafe extern "C" fn mid_string_search_query(
    handle:      *const MidStringSearch,
    query:       *const c_char,
    out_results: *mut CSearchResult,
    max_results: usize,
) -> usize {
    if handle.is_null() || out_results.is_null() || max_results == 0 {
        return 0;
    }

    let q = if query.is_null() { "" } else { read_cstr(query) };
    let results = (*handle).0.query(q);

    let write = results.len().min(max_results);
    for (i, r) in results.iter().take(write).enumerate() {
        *out_results.add(i) = CSearchResult {
            index: *r.data(),
            score: r.score as u32,
        };
    }
    write
}

/// Remove all items from the search set.
#[no_mangle]
pub unsafe extern "C" fn mid_string_search_clear(handle: *mut MidStringSearch) {
    if !handle.is_null() {
        (*handle).0.clear();
    }
}

/// Return the number of items in the search set.
#[no_mangle]
pub unsafe extern "C" fn mid_string_search_len(handle: *const MidStringSearch) -> usize {
    if handle.is_null() { return 0; }
    (*handle).0.len()
  }
