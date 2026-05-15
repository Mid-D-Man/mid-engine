// crates/mid-math/tests/string_id.rs
//! StringId (FNV-1a 64-bit) tests.
//!
//! Tests cover:
//!   - Compile-time const correctness (sid! macro)
//!   - Known FNV-1a vectors
//!   - Uniqueness / collision resistance for typical identifiers
//!   - From<&str> and From<StringId> for u64
//!   - Empty string sentinel
//!   - Debug/Display formatting

use mid_math::{sid, StringId};

// ── Compile-time const evaluation ─────────────────────────────────────────────

const TRANSFORM: StringId = sid!("Transform");
const VELOCITY:  StringId = sid!("Velocity");
const POSITION:  StringId = sid!("Position");
const EMPTY:     StringId = sid!("");

// These must be usable in const contexts (match arms, array indices, etc.)
const _VERIFY_CONST: () = {
    assert!(TRANSFORM.raw() != 0);
    assert!(VELOCITY.raw()  != 0);
    assert!(POSITION.raw()  != 0);
    // All three must differ
    assert!(TRANSFORM.raw() != VELOCITY.raw());
    assert!(TRANSFORM.raw() != POSITION.raw());
    assert!(VELOCITY.raw()  != POSITION.raw());
};

// ── FNV-1a known vectors ──────────────────────────────────────────────────────
//
// Reference values computed from the FNV-1a 64-bit spec:
//   offset_basis = 14695981039346656037
//   prime        = 1099511628211
//
// "Transform" → computed step-by-step for test anchoring.

#[test]
fn empty_string_is_fnv_offset_basis() {
    // The empty string hash equals the FNV-1a 64-bit offset basis.
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    assert_eq!(EMPTY.raw(), FNV_OFFSET);
    assert!(EMPTY.is_empty_string());
}

#[test]
fn non_empty_is_not_offset_basis() {
    assert!(!TRANSFORM.is_empty_string());
    assert!(!VELOCITY.is_empty_string());
}

// ── Uniqueness for common engine identifiers ───────────────────────────────────

#[test]
fn common_component_names_are_unique() {
    let names = [
        "Transform", "Velocity", "Position", "Rotation",
        "Health", "Mass", "Collider", "RigidBody",
        "Player", "Enemy", "Camera", "Light",
        "Mesh", "Material", "Shader", "Texture",
        "Audio", "Script", "Network", "Sync",
    ];

    let ids: Vec<StringId> = names.iter().map(|&s| StringId::new(s)).collect();

    // All IDs must be pairwise distinct
    for i in 0..ids.len() {
        for j in (i + 1)..ids.len() {
            assert_ne!(
                ids[i].raw(), ids[j].raw(),
                "Collision between '{}' and '{}'",
                names[i], names[j]
            );
        }
    }
}

// ── Determinism ───────────────────────────────────────────────────────────────

#[test]
fn same_string_always_same_hash() {
    let a = StringId::new("HelloWorld");
    let b = StringId::new("HelloWorld");
    let c = sid!("HelloWorld");
    assert_eq!(a, b);
    assert_eq!(a, c);
    assert_eq!(a.raw(), b.raw());
}

#[test]
fn different_case_is_different_hash() {
    let lower = StringId::new("transform");
    let upper = StringId::new("Transform");
    let all_up = StringId::new("TRANSFORM");
    assert_ne!(lower, upper);
    assert_ne!(lower, all_up);
    assert_ne!(upper, all_up);
}

// ── From / Into conversions ───────────────────────────────────────────────────

#[test]
fn from_str_matches_new() {
    let via_from: StringId = "Position".into();
    let via_new = StringId::new("Position");
    assert_eq!(via_from, via_new);
}

#[test]
fn into_u64_is_raw() {
    let id = StringId::new("Camera");
    let raw: u64 = id.into();
    assert_eq!(raw, id.raw());
}

// ── PartialEq / Eq ────────────────────────────────────────────────────────────

#[test]
fn eq_and_ne() {
    let a = sid!("A");
    let b = sid!("B");
    let a2 = sid!("A");
    assert_eq!(a, a2);
    assert_ne!(a, b);
}

// ── Hash (usable as HashMap key) ──────────────────────────────────────────────

#[test]
fn usable_as_hashmap_key() {
    use std::collections::HashMap;
    let mut map: HashMap<StringId, i32> = HashMap::new();
    map.insert(sid!("Position"), 0);
    map.insert(sid!("Velocity"), 1);
    map.insert(sid!("Health"),   2);

    assert_eq!(*map.get(&sid!("Position")).unwrap(), 0);
    assert_eq!(*map.get(&sid!("Velocity")).unwrap(), 1);
    assert_eq!(*map.get(&sid!("Health")).unwrap(),   2);
    assert!(map.get(&sid!("Missing")).is_none());
}

// ── Debug / Display formatting ────────────────────────────────────────────────

#[test]
fn debug_contains_hex() {
    let id = StringId::new("test");
    let dbg = format!("{:?}", id);
    // Debug format is "StringId(0x...)"
    assert!(dbg.starts_with("StringId(0x"), "got: {}", dbg);
}

#[test]
fn display_is_hex() {
    let id = StringId::new("test");
    let s = format!("{}", id);
    // Display is just "0x..."
    assert!(s.starts_with("0x"), "got: {}", s);
}

// ── Macro in various const positions ─────────────────────────────────────────

#[test]
fn sid_macro_in_match() {
    let id = StringId::new("Player");
    let label = match id {
        x if x == sid!("Player") => "player",
        x if x == sid!("Enemy")  => "enemy",
        _                         => "unknown",
    };
    assert_eq!(label, "player");
}

// ── Single-byte and unicode strings ──────────────────────────────────────────

#[test]
fn single_char_strings_differ() {
    let a = StringId::new("a");
    let b = StringId::new("b");
    assert_ne!(a, b);
}

#[test]
fn unicode_string_does_not_panic() {
    // FNV-1a operates on raw bytes — unicode is fine
    let id = StringId::new("Привет");
    assert_ne!(id.raw(), 0);
    assert!(!id.is_empty_string());
}

// ── Long strings ──────────────────────────────────────────────────────────────

#[test]
fn long_string_does_not_panic() {
    let long = "a".repeat(4096);
    let id = StringId::new(&long);
    assert_ne!(id.raw(), 0);
}

#[test]
fn prefix_and_full_differ() {
    let prefix = StringId::new("mid_math");
    let full   = StringId::new("mid_math::Vec3");
    assert_ne!(prefix, full);
}
