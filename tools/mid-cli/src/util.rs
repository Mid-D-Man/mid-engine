// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-cli.md, section "util.rs"
// ============================================================================
//! Small helpers: hashing, log files, durations.

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// FNV-1a over bytes, as 16 hex digits. For naming state files, not security.
pub fn fnv_hex(text: &str) -> String {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in text.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{hash:016x}")
}

/// Writes `text` to a new file in `<temp>/mid-cli/logs/` and returns its
/// path, or `None` if it could not be written.
pub fn write_log(temp: &Path, name: &str, text: &str) -> Option<PathBuf> {
    let dir = temp.join("mid-cli").join("logs");
    std::fs::create_dir_all(&dir).ok()?;
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let safe: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect();
    let path = dir.join(format!("{stamp}-{safe}.log"));
    std::fs::write(&path, text).ok()?;
    Some(path)
}

/// The path as text, with the lossy conversion made explicit.
pub fn path_text(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fnv_is_stable_and_distinguishes() {
        assert_eq!(fnv_hex(""), "cbf29ce484222325");
        assert_eq!(fnv_hex("a"), "af63dc4c8601ec8c");
        assert_ne!(fnv_hex("ab"), fnv_hex("ba"));
        assert_eq!(fnv_hex("x").len(), 16);
    }
}
