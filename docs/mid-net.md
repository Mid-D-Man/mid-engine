# mid-net

Reliable UDP-class netcode over QUIC (native: `quinn`, browser:
WebTransport), hand-rolled wire format, pluggable transport boundary.
Packet shapes are documented as `.mdix` reference schema, but DixScript
itself is not a dependency of this crate — see "Dependency philosophy"
below and `docs/architecture.md`.

**Status:** in progress (build order: math → common → geom → **net** → ecs → physics). Restructured into subfolder crates this pass — see "Crate Structure" below. `mid-net-wire` (packet codec, 12 tests + sequence/ack arithmetic, 13 tests), `mid-net-reliable` (frame headers, RTT estimator, retransmit buffer, 17 tests), `mid-net-transport` (the `Transport` trait + `LoopbackTransport`, 4 tests), and `connection.rs` in the facade crate (`Connection<T: Transport>` — the actual `send_player_state`/`send_player_event`/`poll` API, 5 tests) all have real implementations — 51 tests passing total, verified as a real multi-crate Cargo workspace before delivery each time, not just reorganized files. Still not built: the real `quinn`/`web-transport-wasm`-backed `Transport` impls, planned as sibling subfolder crates (`mid-net-transport-quinn`, `mid-net-transport-wasm`). MSRV for that dependency tree is ~1.85, past what this sandbox can compile-verify, so that work is static-analysis-verified only until it lands in real CI. `ffi.rs` is still a skeleton.

## Connection

`connection.rs` (in the facade crate) is what a game loop actually calls
— `Connection::new(transport)`, then `send_player_state`/`send_player_event`
to send and `poll()` once a tick to get back a `Vec<ConnectionEvent>`.
Two things worth knowing about what it deliberately does *not* do:

- **Doesn't use `mid-net-reliable`'s `RetransmitBuffer`/`RttEstimator` at
  all.** `Transport::send_reliable`/`poll_reliable` already guarantee
  real delivery — there's no `has_native_reliability()` escape hatch in
  the real trait, every implementation is required to actually deliver
  reliably. So there's nothing for a retransmit buffer to do at this
  layer. Those pieces aren't dead code — they're exactly what a future
  raw-UDP `Transport` impl would need internally to satisfy that
  guarantee itself, one layer down — but nothing in the current call
  graph reaches them, and that's worth saying plainly rather than
  leaving quiet.
- **Only frames the unreliable channel, not the reliable one.**
  `PlayerState` still gets `mid-net-reliable`'s kind+sequence framing
  (`encode_unreliable_frame`) because staleness detection is still this
  layer's job even though delivery isn't. `PlayerEvent` gets just a
  one-byte kind tag — no sequence number, since ordering is the
  transport's guarantee now, not something to redundantly re-derive.

Tested against `LoopbackTransport` pairs, including a staleness case
that needed injecting frames with an explicit out-of-order sequence
directly (`LoopbackTransport`'s FIFO pump can't produce real reordering
on its own) to actually exercise that path rather than just assert it
works.

## Crate Structure

Restructured this pass from one flat crate into subfolder crates —
checked naia's actual layout first (`socket/{client,server,shared}`,
each its own crate under a subfolder) rather than inventing a
convention, since that's exactly this situation: naia splits out
`naia-socket-shared` (protocol-adjacent, no transport-specific deps)
from `naia-client-socket`/`naia-server-socket` (the crates that
actually carry platform-specific transport dependencies), for the same
reason we're doing it here — so a heavy, platform-specific transport
backend's dependency tree never contaminates the zero-dependency
protocol layer.

```
crates/mid-net/
  Cargo.toml, src/lib.rs, src/ffi.rs   — facade: re-exports everything, owns the FFI surface
  wire/                                — mid-net-wire: packet.rs + sequence.rs. Zero deps.
  transport/                           — mid-net-transport: the Transport trait + LoopbackTransport. Zero deps.
  reliable/                            — mid-net-reliable: frame headers, RTT, retransmit buffer. Depends on mid-net-wire only.
  transport-quinn/  (planned)          — native Transport impl, depends on quinn. Not built.
  transport-wasm/   (planned)          — browser Transport impl, depends on wasm-bindgen/web-sys. Not built.
```

Dependency graph is a clean DAG, verified by actually building it as a
real Cargo workspace (`cargo build --workspace` / `cargo test --workspace`
against a local 4-member workspace mirroring this layout), not just
asserted from the file split: `mid-net-wire` and `mid-net-transport` have
zero path dependencies on each other or on anything else in the family;
`mid-net-reliable` depends on `mid-net-wire` only (needs `PacketKind` for
frame headers, `Sequence`/`is_acked` for the retransmit buffer); the
facade depends on all three. `mid-net-transport` deliberately has zero
dependency on `mid-net-wire` — it only ever moves `&[u8]`/`Vec<u8>`, it
has no idea `PlayerState` or `PacketKind` exist, which is exactly what
lets it be swapped or reused independently of the wire format.

Old flat-file layout retired this pass:
`crates/mid-net/src/{packet,sequence}.rs` → `crates/mid-net/wire/src/`,
`crates/mid-net/src/reliable.rs` → `crates/mid-net/reliable/src/lib.rs`,
`crates/mid-net/src/transport.rs` → `crates/mid-net/transport/src/lib.rs`,
`crates/mid-net/src/socket.rs` → retired entirely (its role — "where
concrete Transport backends land" — is now the planned
`transport-quinn`/`transport-wasm` subfolder crates instead of a file).
Root workspace `Cargo.toml` updated to list the three new members.

## Dependency philosophy

Same mandate as mid-math: **zero external dependencies where at all possible, minimal where not, works on every target with no platform-specific runtime requirement.** No assumption of a particular OS network stack beyond standard UDP sockets — no io_uring-only paths, no eBPF/XDP, nothing that only runs on Linux. This rules out reaching for existing Rust netcode crates (laminar, renet, etc.) as dependencies; they're useful as *reference reading*, not as things to pull in.

**Resolved:** "wire encoder uses bincode" meant the literal `bincode` crate (confirmed against the actual stub doc comments in `lib.rs`/`packet.rs`, which said so explicitly). That contradicted the zero-dependency mandate, so it's replaced with a hand-rolled encoder — explicit little-endian, fixed layout for `PlayerState`, length-prefixed fields for `PlayerEvent`'s strings. `bincode` is removed from `Cargo.toml`. Same spirit as mid-math's SIMD work: hand-rolled over dependency, in this case with an extra motivation — Ubel Stratum's LOW tier (manual memory, FFI) is a plausible future consumer of these bytes, and a Rust-only reflection-based format like bincode gives a non-Rust caller nothing to bind against, where a flat byte layout does.

**Resolved:** `tokio`/`bytes` sitting unconditionally in the old single `Cargo.toml` (the "separate, still-open" question from earlier this pass) is superseded by the transport decision below, not answered in isolation, and now further resolved by the restructuring above — `tokio` will only ever appear in the future `mid-net-transport-quinn` crate's manifest, never in `mid-net-wire`/`mid-net-transport`/`mid-net-reliable`/the facade. `quinn` (verified against its real, published Cargo.toml, not a summary) declares `tokio = { workspace = true }` **unconditionally** — every runtime feature choice (`runtime-tokio`/`runtime-async-std`/`runtime-smol`) only gates which of tokio's *own* features (`time`/`rt`/`net`) get turned on, not whether the crate itself is present. So "zero tokio" isn't achievable while using `quinn` on native, and that's fine — it's normal, well-supported, native-only weight, nothing like the `"full"` feature set the old flat `Cargo.toml` had, and now it's contained to exactly one sub-crate instead of leaking into the whole family. The wasm side is unaffected either way: `web-transport-wasm` (verified via its own published dependency list) has zero `quinn`, zero `tokio` — just `wasm-bindgen`/`web-sys`/`js-sys`/`url`/`thiserror`.

## Why UDP-class (not TCP)


TCP head-of-line blocking stalls all packets when one is lost.
For position updates at 128 Hz, a dropped packet is just stale — skip it.
UDP (and QUIC, which is UDP-based) delivers the next packet immediately.

QUIC specifically (over raw UDP) buys real, hard-to-replicate-by-hand
wins on top of that base property: proper congestion control (not just
a retransmit timer), TLS 1.3 encryption for free, and connection
migration — a mobile client switching from WiFi to cellular keeps the
same connection instead of dropping it. See "Reliability mechanism" and
"Transport" below for how that changes what this crate hand-rolls vs.
what it leans on the protocol for.

## Two Channels

| Channel | Content | Loss behaviour |
|---|---|---|
| Unreliable | position, rotation, animation | drop freely — QUIC/WebTransport datagram |
| Reliable | join, pickup, damage, events | QUIC/WebTransport reliable stream |

### Validated against Unity and Unreal (checked their actual docs/source, not just recalled)

Both engines converge on exactly this split, which is a good sign this is the right shape:

- **Unreal**: RPCs are Server/Client/NetMulticast, each either Reliable or Unreliable. Their own guidance: unreliable for anything called every tick or non-critical (movement, cosmetic effects); reliable for infrequent-but-critical (spawning, state changes). Explicit warning: overusing Reliable can overflow its queue and force a disconnect — worth having an explicit cap/backpressure policy on the reliable channel rather than an unbounded resend queue.
- **Unity (Netcode for GameObjects)**: same split via `NetworkDelivery`/QoS channels (Reliable vs Unreliable), configurable per message. Same guidance: high-frequency (multiple-times-a-second) → unreliable.
- **Both**: reliable delivery ordering is guaranteed **per-object/per-channel, not globally.** Don't build a single global sequence number across all reliable traffic — that reintroduces TCP-style head-of-line blocking between unrelated entities, which is the exact problem UDP was chosen to avoid. Keep reliable and unreliable streams on independent sequence spaces (confirmed against the general reliable-UDP literature too — mixing them reintroduces the stall you're trying to avoid, since a lost reliable packet's resend-wait shouldn't block delivery of unrelated unreliable packets that were sent after it).

### Reliability mechanism — decided this pass, changed from the original plan

**`PlayerEvent` rides a QUIC/WebTransport reliable stream, not `reliable.rs`'s own ack/retransmit.** The original plan (below, still true as *documentation* of a real, correct, tested technique) assumed an unreliable-only transport, the way raw UDP or a WebRTC unreliable datachannel is — checked naia (a real, mature, cross-platform native+wasm Rust netcode library) for comparison, and confirmed it built exactly this kind of hand-rolled ack-bitfield layer, for exactly that reason: its transports (`webrtc-unreliable` on **both** platforms, not just wasm — checked its actual `Cargo.toml`, not just its README) don't offer reliable streams at all, so it has no choice. We're not in that position once the transport is QUIC: real congestion control (not just a retransmit timer) is a genuinely hard problem to match by hand, so `PlayerEvent` gets it from the protocol instead of reinventing it.

**What that leaves real, not retired:**
- `sequence.rs::Sequence` — still exactly as load-bearing. `PlayerState`'s datagrams can still arrive out of order under QUIC/WebTransport (datagrams are explicitly unordered), so staleness detection is still needed regardless of what carries the bytes.
- `reliable.rs::RetransmitBuffer`/`RttEstimator`/`AckTracker`/`is_acked` — correct, tested (Karn's algorithm and all), and kept, but no longer the primary path for either channel. Available if a future need genuinely wants ack/retransmit over a raw datagram rather than a full stream (e.g. a transport backend without native stream reliability). Two real gaps found comparing against naia's *settled* implementation (the tagged `v0.25.0` release — its untagged `main` branch turned out to have diverged into unrelated private content, not a trustworthy comparison point, see the session history if that matters later) that would apply if/when this *is* used for real: no `should_send_empty_ack`-equivalent (naia guarantees an ack goes out even on a tick with nothing else to send, so the peer's buffer doesn't stall waiting on one) and no rolling loss-percentage telemetry (naia's `loss_monitor`, separate from RTT). Not built — flagging for whenever this path gets picked up again.
- The frame-header concept (kind byte + sequence number) — still how `PlayerState` datagrams identify themselves and get staleness-checked.

Original plan, kept for reference (same shape as Glenn Fiedler's `reliable` library, and naia's own ack manager):

- Each reliable packet carries a monotonically increasing sequence number.
- Receiver tracks the highest sequence number seen plus a 32-bit bitfield of what came before it, piggybacked on every outgoing packet's header.
- Sender keeps unacked packets in a buffer, resends on an RTT-based timeout (`RttEstimator`, same smoothing constants as TCP's Jacobson/Karels RTO estimator).
- Karn's algorithm: a packet's RTT is only sampled the first time it's acked with zero retransmits behind it — an ack for a resent packet is ambiguous about which transmission it confirms.
- Sequence numbers wrap — `Sequence` deliberately has no `PartialOrd`/`Ord` so `<`/`>` can't be reached for by habit.

## Transport

`transport.rs`'s `Transport` trait is the pluggable boundary — same idea
as Unity Netcode's swappable `NetworkTransport` (UTP / WebSocket / a
third-party transport, all underneath one `NetworkManager`). We need at
least two backends no matter what (native, browser), so building the
boundary as a real trait now — rather than an internal `cfg` detail —
means a third backend (Steam Sockets, a custom relay, anything) is just
another impl, not a redesign.

**Checked against the actual `com.unity.netcode` source** (needle-mirror
mirror, Netcode for *Entities* — note this is a different package than
Netcode for GameObjects referenced above; verified which one by reading
`package.json` before trusting anything else in it), not just general
knowledge of the pattern:
- `DefaultDriverConstructor.cs` confirms the exact shape decided above,
  in Unity's own words, not just by inference: its WebSocket driver
  registration comment reads *"Web socket does not require reliable
  pipeline... but they need to be kept around for compatibility
  reasons for cross-platform connections."* That's Unity's own code
  independently landing on "the transport can satisfy 'reliable' natively
  and the pipeline stage becomes a no-op" — the same reasoning that put
  `PlayerEvent` on a QUIC stream instead of `reliable.rs`'s own buffer.
- It registers **three** interchangeable backends under one driver store,
  not two: `UDPNetworkInterface`, `WebSocketNetworkInterface`, and
  `IPCNetworkInterface` (same-process client+server, no real socket at
  all) — `LoopbackTransport` is our version of that third one, not just a
  test convenience Unity skipped.
- `NetworkSnapshotAck.cs` independently confirms the ack-bitfield
  technique itself: "shift the entire mask LEFT by that delta, then apply
  the new mask on top" — the exact operation `sequence.rs::AckTracker`
  implements, arrived at separately.

Deliberately **synchronous and poll-based, not `async fn`**: an async
trait spanning native and `wasm32` hits Rust's `!Send`-on-wasm wall —
checked how the `web-transport` crate itself handles this, and it
doesn't unify native/wasm behind one trait either, it swaps concrete
types per target via `cfg`. Each `Transport` impl is free to use async,
threads, or JS callbacks internally; the trait only asks for a
queue-drain once a tick, matching `reliable.rs`'s existing "no runtime
baked into the protocol logic" principle.

Planned concrete implementations (none built yet — `LoopbackTransport`,
in `transport.rs` now, is the only one that exists, for tests):
- **Native** (desktop + mobile — iOS/Android are ordinary Rust
  cross-compile targets here, nothing quinn-specific to solve): QUIC via
  `quinn`, through the `web-transport-quinn` backend.
- **Browser** (including mobile browsers — Safari 26.4, shipped March
  2026, closed the last real gap; checked current support rather than
  assumed, Safari on iOS was specifically the blocker before that):
  WebTransport via `web-transport-wasm`, zero `quinn`/`tokio` in that
  build.

## Mobile

Two separate claims, both checked rather than assumed:
- **Native mobile apps** (Rust cross-compiled to iOS/Android): no
  blocker. `quinn` is pure Rust + `rustls` — no OpenSSL/platform-TLS
  linking fights — and UDP sockets are standard on both OSes. Same
  native `Transport` backend as desktop, just a different
  cross-compilation target; a build-system concern, not an architecture
  one.
- **Mobile browsers**, if ever relevant: WebTransport is supported on
  Safari iOS (26.4+, March 2026) and Android (Chrome 108+, Firefox
  132+, Samsung Internet 18+) — checked current browser support tables,
  not the pre-2026 state where Safari was the blocker.
- Bonus specific to mobile: QUIC's connection migration means a client
  moving from WiFi to cellular keeps its connection. Raw UDP plus a
  hand-rolled reliability layer would not have gotten this for free.

## Platform & FFI

Two hard constraints, both load-bearing in `reliable.rs`'s design, not just aspirational:

- **No `std::net`, no `std::time::Instant`, anywhere in `packet.rs`/`sequence.rs`/`reliable.rs`.** `std::net::UdpSocket` doesn't exist on `wasm32-unknown-unknown` — checked current browser transport options rather than assuming: WebTransport datagrams (baseline-available across browsers as of March 2026, UDP-like: unreliable, unordered, MTU-sized) are the client-server fit; WebRTC `RTCDataChannel` unreliable mode is the P2P equivalent but needs ICE/STUN/TURN for client-server, more complexity than this needs. Either way, both present the same shape as UDP — a byte buffer in, a byte buffer out, no delivery guarantee — which is exactly the interface `reliable.rs` and `packet.rs` are written against. `socket.rs` is where the actual per-platform transport gets picked (`cfg`-gated); nothing above it needs to change when that happens.
- **Time is a caller-supplied `Timestamp(u64)` (milliseconds), never queried internally.** `std::time::Instant` has no defined layout and isn't meaningful across a C ABI, and it panics on `wasm32-unknown-unknown` without a shim. Pushing "what does now mean" up to the caller sidesteps both problems at once, and as a side effect makes every `reliable.rs` test deterministic — a manually-advanced fake clock, no real sleeping, no timing flakiness.

## DixScript Integration

Packet shapes are documented as `.mdix` reference schema under
`packets/`, but this is **not** a mid-net dependency — decided this pass
and then generalized into a repo-wide policy (see
`docs/architecture.md`, "Technical Mandates"): `dixscript` is not a
dependency of any core crate, not just mid-net, not just "for now."

- **Not a fit for the wire format even setting the dependency question
  aside.** Checked its actual Rust API: it's a dynamic accessor over
  parsed `.mdix` (`data.get::<T>("path")`), not struct codegen — there
  was never a DixScript step `packet.rs`'s hand-written types were
  standing in for. It also ships a real binary Packer/Unpacker (the
  "DLM pipeline"), but that's a general-purpose encode path for
  arbitrary DixScript ASTs — encryption and compression built in,
  self-describing — not tuned for a fixed 28-byte per-tick struct.
- **Not added as a dependency, mandatory or not, by policy now — not a
  case-by-case call.** Even with `default-features = false`, `dixscript`
  pulls in 23 mandatory transitive crates (serde, regex, chrono,
  aes-gcm, chacha20poly1305, argon2, uuid, phf, …). Right budget for a
  general config/data format, wrong one for a core systems crate.
- **Where it actually lives:** `tools/mdix-compiler` — a separate
  binary, not linked into any core crate — now depends on `dixscript`
  for real, since compiling `.mdix` files is exactly its job. That's
  the concrete instance of "DixScript as the engine's convenient data
  format" the policy above points to. mid-net's own `.mdix` files stay
  human-authored reference schema only.

`packet.rs` stays the fast path for in-flight bytes; the `.mdix` files
stay the human-authored source of truth for packet shape, kept in sync
by hand for now.

## Packet Budget

7.8 ms per tick at 128 Hz. Design entity delta budgets early.
This constrains everything else in the networking system.

## Sibling project check

`midn` (the LTE/5G core project) was checked for reusable code — nothing transfers. It's a 3GPP telecom stack (S1AP/NGAP/NAS signaling, GTP-U tunneling, Milenage/TUAK SIM auth), a different protocol domain with different constraints, and its userplane layer is Linux-eBPF/XDP-specific, which conflicts with mid-net's cross-platform requirement. Not worth revisiting for mid-net specifically. (`midn-ecs` is worth a glance later, for mid-ecs — see that crate's notes.)
