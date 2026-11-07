// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-ui-mxui.md, section "bind.rs"
// ============================================================================
//! Data-binding keys.

/// Integer id for a binding key named in a `.mxui` file, such as
/// `player_health`. It is the 32-bit FNV-1a hash of the name's UTF-8 bytes, so
/// the same name gives the same id in every build and on every platform.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BindKey(pub u32);

impl BindKey {
    /// Hashes `name`. Usable in `const` contexts, so a game can declare
    /// `const HEALTH: BindKey = BindKey::of("player_health");`.
    pub const fn of(name: &str) -> Self {
        let bytes = name.as_bytes();
        let mut hash: u32 = 0x811c_9dc5;
        let mut i = 0;
        while i < bytes.len() {
            hash ^= bytes[i] as u32;
            hash = hash.wrapping_mul(0x0100_0193);
            i += 1;
        }
        BindKey(hash)
    }
}
