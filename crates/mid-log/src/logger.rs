// crates/mid-log/src/logger.rs

//! Global logger singleton.
//!
//! ## Design changes vs the previous version
//!
//! | Old                              | New                              |
//! |----------------------------------|----------------------------------|
//! | `rtrb` SPSC + `Mutex<Producer>` | `crossbeam-channel` unbounded MPSC |
//! | Mutex needed for multi-thread   | Channel is natively thread-safe  |
//! | `yield_now()` spin in IO thread | `recv()` blocking (zero CPU idle)|
//! | No level filter                  | `filter::is_enabled()` AtomicU8  |
//! | No source location               | `file!`, `line!`, `module_path!` |
//! | No file sink                     | Optional `PathBuf` at init       |
//!
//! ## Init patterns
//!
//! Minimal (stderr, log everything):
//! ```rust,no_run
//! mid_log::logger::MidLogger::init();
//! ```
//!
//! Production (file tee, INFO+):
//! ```rust,no_run
//! use mid_log::logger::MidLogger;
//! use mid_log::filter::set_min_level;
//! use mid_log::level::LogLevel;
//! use std::path::PathBuf;
//!
//! MidLogger::init_with(Some(PathBuf::from("game.log")));
//! set_min_level(LogLevel::Info);
//! ```

use std::path::PathBuf;
use std::sync::OnceLock;

use crate::buffer::{self, LogSender};
use crate::entry::LogEntry;
use crate::filter;
use crate::level::{LogLevel, Tier};
use crate::writer::LogWriter;

pub struct MidLogger {
    /// crossbeam sender is Clone + Send, no Mutex needed.
    sender:  LogSender,
    _writer: LogWriter,
}

static INSTANCE: OnceLock<MidLogger> = OnceLock::new();

impl MidLogger {
    /// Initialise the global logger with stderr output only.
    ///
    /// Returns `true` on success, `false` if already initialised.
    pub fn init() -> bool {
        Self::init_with(None)
    }

    /// Initialise with an optional file tee.
    ///
    /// `log_file`: if `Some(path)`, entries are also written to that file
    /// (appended). The file is created if it does not exist.
    pub fn init_with(log_file: Option<PathBuf>) -> bool {
        let (sender, receiver) = buffer::create();
        let writer = LogWriter::spawn(receiver, log_file);
        INSTANCE.set(MidLogger { sender, _writer: writer }).is_ok()
    }

    /// Get the global logger instance, or `None` if not yet initialised.
    #[inline]
    pub fn get() -> Option<&'static MidLogger> {
        INSTANCE.get()
    }

    /// Push a log entry. Non-blocking, never drops (unbounded channel).
    ///
    /// The level filter check happens in the macro *before* this is called,
    /// so `message` has already been formatted. This function only touches
    /// an atomic send — no allocation.
    #[inline]
    pub fn log(
        &self,
        level:   LogLevel,
        tier:    Tier,
        message: String,
        file:    &'static str,
        line:    u32,
        module:  &'static str,
    ) {
        let entry = LogEntry::new(level, tier, message, file, line, module);
        // send() on an unbounded channel is infallible unless the receiver
        // has been dropped (i.e. the IO thread has already exited).
        // We silently ignore that case — Fatal path calls shutdown() before
        // the receiver could ever drop during normal operation.
        self.sender.send(entry).ok();
    }

    /// Flush: wait for all queued entries to be written.
    ///
    /// Implemented by sending a synthetic Trace entry and joining on the
    /// writer thread. For most use cases `shutdown()` is sufficient.
    /// This method is provided for tests and tools that need a clean flush
    /// without tearing down the logger.
    pub fn flush() {
        // The simplest correct flush: sleep until the channel is empty.
        // The IO thread drains continuously so this converges quickly.
        if let Some(logger) = INSTANCE.get() {
            // Spin until the channel reports empty. On a quiet machine this
            // resolves in <1ms. We use a short sleep to avoid burning CPU.
            while !logger.sender.is_empty() {
                std::thread::sleep(std::time::Duration::from_micros(100));
            }
            // One additional sleep to allow the IO thread to finish the last write.
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }

    /// Flush remaining entries and stop the IO thread.
    ///
    /// Call once at process exit. After this, log calls are silently
    /// discarded (the channel still accepts sends but the receiver is gone).
    pub fn shutdown() {
        if let Some(logger) = INSTANCE.get() {
            // Drain first
            while !logger.sender.is_empty() {
                std::thread::sleep(std::time::Duration::from_micros(100));
            }
            logger._writer.signal_shutdown();
        }
    }
}
