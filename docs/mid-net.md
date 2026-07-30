# mid-net

Reliable UDP with DixScript (.mdix) packet definitions.

**Status:** starting now (build order: math → common → geom → **net** → ecs → physics). Currently a skeleton — all files 2-18 lines, no real implementation yet.

## Dependency philosophy

Same mandate as mid-math: **zero external dependencies where at all possible, minimal where not, works on every target with no platform-specific runtime requirement.** No assumption of a particular OS network stack beyond standard UDP sockets — no io_uring-only paths, no eBPF/XDP, nothing that only runs on Linux. This rules out reaching for existing Rust netcode crates (laminar, renet, etc.) as dependencies; they're useful as *reference reading*, not as things to pull in.

**Open question to resolve before writing code:** the line below about "wire encoder uses bincode" needs a decision. If that means the literal `bincode` crate, it contradicts the zero-dependency mandate and should be replaced with a hand-rolled encoder in the same spirit as mid-math's SIMD work — or, if DixScript's own runtime already does binary encoding without depending on the `bincode` crate, the doc just needs its wording fixed so "bincode" isn't confused with the dependency. Pin this down first.

## Why UDP

TCP head-of-line blocking stalls all packets when one is lost.
For position updates at 128 Hz, a dropped packet is just stale — skip it.
UDP delivers the next packet immediately.

## Two Channels

| Channel | Content | Loss behaviour |
|---|---|---|
| Unreliable | position, rotation, animation | drop freely |
| Reliable | join, pickup, damage, events | ACK + retransmit |

### Validated against Unity and Unreal (checked their actual docs/source, not just recalled)

Both engines converge on exactly this split, which is a good sign this is the right shape:

- **Unreal**: RPCs are Server/Client/NetMulticast, each either Reliable or Unreliable. Their own guidance: unreliable for anything called every tick or non-critical (movement, cosmetic effects); reliable for infrequent-but-critical (spawning, state changes). Explicit warning: overusing Reliable can overflow its queue and force a disconnect — worth having an explicit cap/backpressure policy on the reliable channel rather than an unbounded resend queue.
- **Unity (Netcode for GameObjects)**: same split via `NetworkDelivery`/QoS channels (Reliable vs Unreliable), configurable per message. Same guidance: high-frequency (multiple-times-a-second) → unreliable.
- **Both**: reliable delivery ordering is guaranteed **per-object/per-channel, not globally.** Don't build a single global sequence number across all reliable traffic — that reintroduces TCP-style head-of-line blocking between unrelated entities, which is the exact problem UDP was chosen to avoid. Keep reliable and unreliable streams on independent sequence spaces (confirmed against the general reliable-UDP literature too — mixing them reintroduces the stall you're trying to avoid, since a lost reliable packet's resend-wait shouldn't block delivery of unrelated unreliable packets that were sent after it).

### Reliability mechanism (the concrete "ACK + retransmit")

Standard pattern used across the reliable-UDP-for-games space (this is the same approach behind e.g. Glenn Fiedler's `reliable` library, cited widely as the reference design):

- Each reliable packet carries a monotonically increasing sequence number.
- Receiver tracks the highest sequence number seen plus a bitfield of the last N (typically 32) packets received, and sends that back as the ack.
- Sender keeps unacked packets in a small buffer, resends on timeout (RTT-based, not fixed) or when a gap is confirmed via the ack bitfield.
- Sequence numbers wrap — comparisons need wraparound-aware logic (`(a - b) as i16 > 0` style), not raw `>`.
- Keep this as its own small, self-contained module — it's the one part of mid-net worth writing tests against known packet-loss sequences early, since a subtle wraparound or off-by-one bug here silently corrupts delivery guarantees rather than crashing.

## DixScript Integration

Packet shapes are defined in `.mdix` files under `packets/`.

**Important:** benchmark the DixScript deserializer vs a hand-rolled encoder
early in development. At 128 Hz with many entities the per-packet
overhead matters. Use DixScript for definitions; consider a
separate fast path for in-flight bytes if needed. (See the dependency
philosophy note above — pin down what "the wire encoder" actually is
before this benchmark, since the two candidates have different
dependency implications.)

## Packet Budget

7.8 ms per tick at 128 Hz. Design entity delta budgets early.
This constrains everything else in the networking system.

## Sibling project check

`midn` (the LTE/5G core project) was checked for reusable code — nothing transfers. It's a 3GPP telecom stack (S1AP/NGAP/NAS signaling, GTP-U tunneling, Milenage/TUAK SIM auth), a different protocol domain with different constraints, and its userplane layer is Linux-eBPF/XDP-specific, which conflicts with mid-net's cross-platform requirement. Not worth revisiting for mid-net specifically. (`midn-ecs` is worth a glance later, for mid-ecs — see that crate's notes.)
