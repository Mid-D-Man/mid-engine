//! Raw UDP socket abstraction.
//!
//! Not implemented yet. The pluggable-transport *boundary* itself lives
//! in `transport.rs` (the `Transport` trait + `LoopbackTransport`) —
//! this file is where concrete backends implementing that trait will
//! land: native (`quinn`/`web-transport-quinn`) and browser
//! (`web-transport-wasm`), per docs/mid-net.md "Transport". Neither is
//! built yet — quinn's dependency tree needs a newer Rust than this
//! sandbox can compile (edition2024 in a transitive dep), confirmed by
//! trying to resolve it directly, so that part stays static-analysis-only
//! until it's written against a real toolchain.
//!
//! No placeholder type here on purpose this time — `transport.rs`'s
//! `Transport` trait plus `LoopbackTransport` already give `lib.rs`
//! something real to export and the crate something real to compile
//! against, so there's no unresolved-import gap left for a placeholder
//! to paper over.
