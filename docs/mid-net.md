# mid-net

Reliable UDP with DixScript (.mdix) packet definitions.

## Why UDP

TCP head-of-line blocking stalls all packets when one is lost.
For position updates at 128 Hz, a dropped packet is just stale — skip it.
UDP delivers the next packet immediately.

## Two Channels

| Channel | Content | Loss behaviour |
|---|---|---|
| Unreliable | position, rotation, animation | drop freely |
| Reliable | join, pickup, damage, events | ACK + retransmit |

## DixScript Integration

Packet shapes are defined in `.mdix` files under `packets/`.
The wire encoder uses bincode for zero-copy byte layout.

**Important:** benchmark the DixScript deserializer vs bincode
early in development. At 128 Hz with many entities the per-packet
overhead matters. Use DixScript for definitions; consider a
separate fast path for in-flight bytes if needed.

## Packet Budget

7.8 ms per tick at 128 Hz. Design entity delta budgets early.
This constrains everything else in the networking system.
