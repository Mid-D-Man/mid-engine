// crates/mid-common/src/string/utils.rs
//! String name utilities.
//!
//! Ported from Blender's BLI_string_utils.hh:
//!   - `uniquename`     — ensure a name is unique by appending/incrementing `.NNN`
//!   - `flip_side_name` — mirror L/R suffixes and prefixes (for skeletal animation)
//!
//! `flip_side_name` is essential for mirroring bone/socket names in animation
//! rigs: "Arm.L" ↔ "Arm.R", "Left_Knee" ↔ "Right_Knee", "L_Hand" ↔ "R_Hand".
//!
//! `uniquename` is used anywhere you create named entities and need to avoid
//! collisions: ECS entity names, animation tracks, audio buses, network channels.

use alloc::string::{String, ToString};
use alloc::format;

// ─────────────────────────────────────────────────────────────────────────────
// Side / handedness
// ─────────────────────────────────────────────────────────────────────────────

/// The side detected in or flipped from a name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SideChar {
    Left,
    Right,
}

// ─────────────────────────────────────────────────────────────────────────────
// flip_side_name
// ─────────────────────────────────────────────────────────────────────────────

/// Flip the left/right side indicator in a name.
///
/// Checks suffixes first (most common in Blender/game rigs), then prefixes.
/// If no pattern is found, returns the name unchanged.
///
/// Suffix patterns (case-sensitive as written, checked with delimiter `.` `_` `-`):
///
/// | Input       | Output      |
/// |-------------|-------------|
/// | `"Arm.L"`   | `"Arm.R"`   |
/// | `"Arm.R"`   | `"Arm.L"`   |
/// | `"Arm_L"`   | `"Arm_R"`   |
/// | `"Arm-L"`   | `"Arm-R"`   |
/// | `"ArmLeft"` | `"ArmRight"`|
/// | `"ArmRight"`| `"ArmLeft"` |
///
/// Prefix patterns:
///
/// | Input       | Output      |
/// |-------------|-------------|
/// | `"L_Arm"`   | `"R_Arm"`   |
/// | `"Left_Arm"`| `"Right_Arm"`|
///
/// ```rust
/// use mid_common::string::flip_side_name;
/// assert_eq!(flip_side_name("Bone.L"), "Bone.R");
/// assert_eq!(flip_side_name("Right_Knee"), "Left_Knee");
/// assert_eq!(flip_side_name("NoPair"), "NoPair");
/// ```
pub fn flip_side_name(name: &str) -> String {
    // ── Suffix patterns ───────────────────────────────────────────────────────
    // Ordered: longer patterns first to avoid partial matches.
    // Each entry: (left_suffix, right_suffix)
    const SUFFIX_PAIRS: &[(&str, &str)] = &[
        // Word suffixes with delimiter
        (".Left",  ".Right"),
        (".left",  ".right"),
        (".LEFT",  ".RIGHT"),
        ("_Left",  "_Right"),
        ("_left",  "_right"),
        ("_LEFT",  "_RIGHT"),
        ("-Left",  "-Right"),
        ("-left",  "-right"),
        ("-LEFT",  "-RIGHT"),
        (" Left",  " Right"),
        (" left",  " right"),
        (" LEFT",  " RIGHT"),
        // Single-char suffixes with delimiter
        (".L",     ".R"),
        (".l",     ".r"),
        ("_L",     "_R"),
        ("_l",     "_r"),
        ("-L",     "-R"),
        ("-l",     "-r"),
        // Word bare suffixes (no delimiter — check last since they can false-positive)
        ("Left",   "Right"),
        ("left",   "right"),
        ("LEFT",   "RIGHT"),
    ];

    for &(left, right) in SUFFIX_PAIRS {
        if let Some(base) = name.strip_suffix(left) {
            return format!("{}{}", base, right);
        }
        if let Some(base) = name.strip_suffix(right) {
            return format!("{}{}", base, left);
        }
    }

    // ── Prefix patterns ───────────────────────────────────────────────────────
    const PREFIX_PAIRS: &[(&str, &str)] = &[
        ("Left_",  "Right_"),
        ("left_",  "right_"),
        ("LEFT_",  "RIGHT_"),
        ("L_",     "R_"),
        ("l_",     "r_"),
    ];

    for &(left, right) in PREFIX_PAIRS {
        if let Some(rest) = name.strip_prefix(left) {
            return format!("{}{}", right, rest);
        }
        if let Some(rest) = name.strip_prefix(right) {
            return format!("{}{}", left, rest);
        }
    }

    // No flip pattern found
    name.to_string()
}

/// Detect which side a name belongs to, without flipping.
///
/// Returns `None` if no side pattern is found.
pub fn detect_side(name: &str) -> Option<SideChar> {
    // Suffix check (L/Left = Left side)
    let left_suffixes  = [".L", ".l", "_L", "_l", "-L", "-l",
                          ".Left", ".left", "_Left", "_left",
                          "-Left", "-left", " Left", " left",
                          "Left", "left"];
    let right_suffixes = [".R", ".r", "_R", "_r", "-R", "-r",
                          ".Right", ".right", "_Right", "_right",
                          "-Right", "-right", " Right", " right",
                          "Right", "right"];

    for s in &left_suffixes {
        if name.ends_with(s) { return Some(SideChar::Left); }
    }
    for s in &right_suffixes {
        if name.ends_with(s) { return Some(SideChar::Right); }
    }

    // Prefix check
    let left_prefixes  = ["L_", "l_", "Left_", "left_"];
    let right_prefixes = ["R_", "r_", "Right_", "right_"];

    for p in &left_prefixes {
        if name.starts_with(p) { return Some(SideChar::Left); }
    }
    for p in &right_prefixes {
        if name.starts_with(p) { return Some(SideChar::Right); }
    }

    None
}

// ─────────────────────────────────────────────────────────────────────────────
// uniquename
// ─────────────────────────────────────────────────────────────────────────────

/// Make `name` unique by incrementing a numeric suffix separated by `delim`.
///
/// The `is_taken` closure returns `true` if a candidate name is already used.
/// Iterates `name`, `name.001`, `name.002`, … until a free slot is found.
///
/// Ported from Blender's `BLI_uniquename_cb`.
///
/// ```rust
/// use mid_common::string::uniquename;
///
/// let taken = ["Entity", "Entity.001", "Entity.002"];
/// let result = uniquename("Entity", '.', |n| taken.contains(&n));
/// assert_eq!(result, "Entity.003");
/// ```
pub fn uniquename<F>(name: &str, delim: char, mut is_taken: F) -> String
where
    F: FnMut(&str) -> bool,
{
    // Strip any existing numeric suffix first
    let (base, _) = split_name_number(name, delim);

    // Try the bare base name first (no suffix)
    if !is_taken(base) {
        return base.to_string();
    }

    // Then try base + incrementing suffix
    for n in 1u32..=9999 {
        let candidate = format!("{}{}{:03}", base, delim, n);
        if !is_taken(&candidate) {
            return candidate;
        }
    }

    // Absolute fallback — shouldn't happen in practice
    format!("{}{}9999", base, delim)
}

/// Increment the numeric suffix of `name` by one, preserving padding.
///
/// `"Bone.001"` → `"Bone.002"`. `"Bone"` → `"Bone.001"`.
/// Useful when you know the name is already taken and want the next slot.
pub fn increment_name(name: &str, delim: char) -> String {
    let (base, n) = split_name_number(name, delim);
    format!("{}{}{:03}", base, delim, n + 1)
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Split `"Bone.001"` into `("Bone", 1)`.
/// Returns `(name, 0)` if no numeric suffix with `delim` exists.
pub fn split_name_number(name: &str, delim: char) -> (&str, u32) {
    if let Some(pos) = name.rfind(delim) {
        let suffix = &name[pos + delim.len_utf8()..];
        if !suffix.is_empty() && suffix.chars().all(|c| c.is_ascii_digit()) {
            if let Ok(n) = suffix.parse::<u32>() {
                return (&name[..pos], n);
            }
        }
    }
    (name, 0)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── flip_side_name ────────────────────────────────────────────────────────

    #[test]
    fn flip_dot_suffix() {
        assert_eq!(flip_side_name("Arm.L"),  "Arm.R");
        assert_eq!(flip_side_name("Arm.R"),  "Arm.L");
        assert_eq!(flip_side_name("Arm.l"),  "Arm.r");
    }

    #[test]
    fn flip_underscore_suffix() {
        assert_eq!(flip_side_name("Knee_L"), "Knee_R");
        assert_eq!(flip_side_name("Knee_R"), "Knee_L");
    }

    #[test]
    fn flip_word_suffix() {
        assert_eq!(flip_side_name("ArmLeft"),   "ArmRight");
        assert_eq!(flip_side_name("ArmRight"),  "ArmLeft");
        assert_eq!(flip_side_name("Arm Left"),  "Arm Right");
        assert_eq!(flip_side_name("Arm_Left"),  "Arm_Right");
        assert_eq!(flip_side_name("Arm_Right"), "Arm_Left");
    }

    #[test]
    fn flip_prefix() {
        assert_eq!(flip_side_name("L_Shoulder"),    "R_Shoulder");
        assert_eq!(flip_side_name("R_Shoulder"),    "L_Shoulder");
        assert_eq!(flip_side_name("Left_Knee"),     "Right_Knee");
        assert_eq!(flip_side_name("Right_Knee"),    "Left_Knee");
    }

    #[test]
    fn flip_no_pattern() {
        assert_eq!(flip_side_name("Spine"),  "Spine");
        assert_eq!(flip_side_name(""),       "");
        assert_eq!(flip_side_name("Center"), "Center");
    }

    #[test]
    fn flip_round_trip() {
        let names = ["Bone.L", "R_Hand", "Left_Knee", "ArmRight", "leg_l"];
        for n in &names {
            assert_eq!(flip_side_name(&flip_side_name(n)), *n,
                "round-trip failed for {n}");
        }
    }

    // ── detect_side ───────────────────────────────────────────────────────────

    #[test]
    fn detect_sides() {
        assert_eq!(detect_side("Arm.L"),   Some(SideChar::Left));
        assert_eq!(detect_side("Arm.R"),   Some(SideChar::Right));
        assert_eq!(detect_side("L_Hand"),  Some(SideChar::Left));
        assert_eq!(detect_side("Spine"),   None);
    }

    // ── uniquename ────────────────────────────────────────────────────────────

    #[test]
    fn unique_free_immediately() {
        let result = uniquename("Entity", '.', |_| false);
        assert_eq!(result, "Entity");
    }

    #[test]
    fn unique_increments() {
        let taken = ["Entity", "Entity.001", "Entity.002"];
        let result = uniquename("Entity", '.', |n| taken.contains(&n));
        assert_eq!(result, "Entity.003");
    }

    #[test]
    fn unique_strips_existing_suffix() {
        // "Entity.001" is taken → should strip to "Entity" and re-increment
        let taken = ["Entity", "Entity.001"];
        let result = uniquename("Entity.001", '.', |n| taken.contains(&n));
        assert_eq!(result, "Entity.002");
    }

    #[test]
    fn unique_underscore_delim() {
        let taken = ["Bone", "Bone_001"];
        let result = uniquename("Bone", '_', |n| taken.contains(&n));
        assert_eq!(result, "Bone_002");
    }

    // ── split_name_number ─────────────────────────────────────────────────────

    #[test]
    fn split_number() {
        assert_eq!(split_name_number("Bone.001", '.'), ("Bone", 1));
        assert_eq!(split_name_number("Bone",     '.'), ("Bone", 0));
        assert_eq!(split_name_number("Bone.abc", '.'), ("Bone.abc", 0));
        assert_eq!(split_name_number("A.B.003",  '.'), ("A.B", 3));
    }
}
