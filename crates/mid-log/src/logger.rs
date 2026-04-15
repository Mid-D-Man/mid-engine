//! Global logger singleton.
//!
//! Holds the ring buffer producer and owns the IO writer thread.
//! Initialised once per process via `MidLogger::init()`.

use std::sync::{Mutex, OnceLock};
use rtrb::Producer;

use crate::buffer;
use crate::entry::LogEntry;
use crate::level::{LogLevel, Tier};
use crate::writer::LogWriter;

pub struct MidLogger {
    // Mutex here is fine — contention only happens if multiple Rust
    // threads log simultaneously. The Mutex is NOT on the hot path
    // for the single-threaded game loop case. If you need true
    // single-thread zero-contention, store the producer in a
    // thread_local and route everything through one designated caller.
    producer: Mutex<Producer<LogEntry>>,
    _writer:  LogWriter,
}

static INSTANCE: OnceLock<MidLogger> = OnceLock::new();

impl MidLogger {
    /// Initialise the global logger. Returns `true` on success,
    /// `false` if already initialised.
    pub fn init() -> bool {
        let (producer, consumer) = buffer::create();
        let writer = LogWriter::spawn(consumer);
        INSTANCE.set(MidLogger {
            producer: Mutex::new(producer),
            _writer:  writer,
        }).is_ok()
    }

    /// Get the global logger, or `None` if not yet initialised.
    #[inline]
    pub fn get() -> Option<&'static MidLogger> {
        INSTANCE.get()
    }

    /// Push a log entry. Best-effort — if the ring buffer is full
    /// the entry is silently dropped rather than blocking the caller.
    pub fn log(&self, level: LogLevel, tier: Tier, message: String) {
        if let Ok(mut prod) = self.producer.lock() {
            let entry = LogEntry::new(level, tier, message);
            // `push` returns Err(entry) if full — we intentionally drop it.
            let _ = prod.push(entry);
        }
    }

    /// Flush: signal the writer thread to shut down and wait for it to
    /// drain the buffer. Call this at process exit.
    pub fn shutdown() {
        if let Some(logger) = INSTANCE.get() {
            logger._writer.signal_shutdown();
        }
    }
}
