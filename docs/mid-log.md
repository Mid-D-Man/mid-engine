# mid-log

Non-blocking tiered logger using a lock-free SPSC ring buffer (rtrb).

## Design

The game thread pushes log entries into the ring buffer.
A background IO thread drains it and writes to disk.
Zero frame-rate impact — unlike Unity's Debug.Log.

## Log Levels

| Level | Use |
|---|---|
| TRACE | Per-frame firehose — transient mode only |
| INFO | Normal events |
| WARN | Recoverable issues |
| ERROR | Non-fatal failures (stack trace behind 'backtrace' feature) |
| FATAL | Immediate memory dump + shutdown |

## Tier Metadata

Every entry is tagged [LOW] (engine internals) or [HIGH] (gameplay logic).
Prepared for Ubel Stratum integration.
