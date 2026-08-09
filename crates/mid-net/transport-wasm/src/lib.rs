//! Browser `Transport` backend over WebTransport, via `web-transport-wasm`.
//!
//! **Verification status, read this first:** the `framing` and `queue`
//! modules are plain Rust with zero wasm-specific dependencies — they
//! compile and their tests actually run on any host, including the
//! sandbox this crate was written in (verified: `cargo test` there,
//! real pass, not asserted). `transport.rs` (the actual `WasmTransport`)
//! is a different story — it's gated to `--target wasm32-unknown-unknown`
//! because `web-sys`'s WebTransport bindings don't exist anywhere else,
//! and even building for that target here wouldn't help, since there's
//! no browser or JS runtime in this sandbox to actually run a
//! WebTransport session against. Its syntax was checked — but precisely,
//! not overstated: `#[cfg(...)] mod transport;` (an external-file module
//! reference, which is what's actually below) turns out to skip opening
//! the file at all when the cfg doesn't match the host — proven directly
//! with a deliberately-broken standalone test file before trusting this
//! either way, not assumed. What actually caught anything was wrapping
//! this file's real contents as an *inline* `#[cfg(...)] mod transport {
//! .. }` block in a throwaway harness instead — inline cfg'd-out code
//! does get parsed (confirmed the same way, with a deliberately unclosed
//! brace), and running that against this file's real contents came back
//! clean: no unclosed delimiters, no malformed tokens, valid item
//! syntax. That's still only syntax, not more — cfg-disabled code
//! doesn't get import/name resolution even when parsed this way either
//! (also confirmed directly, not assumed: the same kind of harness with
//! nonexistent crate imports and no crate declarations for them came
//! back with zero errors), so this proves less than the
//! "missing-crate-errors-only" check `mid-net-transport-quinn` got on
//! its own unverified code. Every API call below is still cited against
//! the real, downloaded `web-transport-wasm` 0.5.10 source — that
//! citation discipline is what's actually carrying this file's
//! correctness, not the syntax check. Needs real CI with the wasm32
//! target, and ideally a `wasm-bindgen-test` run in a real browser,
//! before it's trusted the way `mid-net-transport-quinn` now can be
//! (that one's own first real compile, on real CI, succeeded cold — this
//! one hasn't had that moment yet).
//!
//! **Required build flag**, not optional: `web-sys` gates its
//! WebTransport bindings behind `--cfg=web_sys_unstable_apis`, which
//! `web-transport-wasm` cannot enable on a consumer's behalf (confirmed
//! from that crate's own `lib.rs` — it has a `compile_error!` pointing at
//! this exact fix if the flag is missing). The workspace's root
//! `.cargo/config.toml` sets this, scoped to the wasm32 target only —
//! deliberately not placed inside this crate's own directory, since
//! Cargo's config discovery walks up from wherever it's invoked, not
//! down into workspace members, and every build in this project runs
//! from the workspace root. If `RUSTFLAGS` is ever set by whatever CI
//! step builds this crate, it overrides that file entirely rather than
//! merging with it (see the root config's own comment on this).
//!
//! **Wire format is deliberately byte-identical to
//! `mid-net-transport-quinn`'s own** — see `framing.rs`'s doc comment for
//! why: native and browser peers need to be able to talk to each other
//! over the same protocol.

pub mod framing;
pub mod queue;

#[cfg(target_arch = "wasm32")]
mod transport;

#[cfg(target_arch = "wasm32")]
pub use transport::{WasmTransport, WasmTransportError};
