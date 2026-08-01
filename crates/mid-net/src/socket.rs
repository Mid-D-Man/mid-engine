//! Raw UDP socket abstraction.
//!
//! Not implemented yet. `MidSocket` below is a placeholder that exists
//! only so the crate compiles and `lib.rs`'s `pub use socket::MidSocket`
//! resolves — real transport work hasn't started (see docs/mid-net.md
//! status). The actual design question here — native UDP vs. browser
//! WebTransport datagrams, `cfg`-gated per target (see docs/mid-net.md
//! "Platform & FFI") — is real work still to do, not filled in
//! speculatively just to make this placeholder feel more finished.

/// Placeholder only. Constructing this does nothing useful yet — no
/// fields, no methods, no behavior. Exists purely so `mid_net::MidSocket`
/// resolves and the crate (and its CI) can build while `packet.rs`,
/// `sequence.rs`, and `reliable.rs` are exercised for real.
#[derive(Debug)]
pub struct MidSocket;
