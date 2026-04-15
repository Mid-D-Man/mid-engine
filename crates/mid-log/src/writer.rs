//! Background IO thread — drains the ring buffer and writes to stderr.
//!
//! The thread yields when the buffer is empty rather than spinning,
//! which keeps CPU usage near zero during quiet periods.

use std::thread;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use rtrb::Consumer;
use crate::entry::LogEntry;

pub struct LogWriter {
    shutdown: Arc<AtomicBool>,
    handle:   Option<thread::JoinHandle<()>>,
}

impl LogWriter {
    pub fn spawn(mut consumer: Consumer<LogEntry>) -> Self {
        let shutdown       = Arc::new(AtomicBool::new(false));
        let shutdown_clone = shutdown.clone();

        let handle = thread::Builder::new()
            .name("mid-log-io".into())
            .spawn(move || {
                while !shutdown_clone.load(Ordering::Relaxed) {
                    let mut wrote = false;
                    while let Ok(entry) = consumer.pop() {
                        Self::write(&entry);
                        wrote = true;
                    }
                    if !wrote {
                        thread::yield_now();
                    }
                }
                // Drain anything pushed after the shutdown signal.
                while let Ok(entry) = consumer.pop() {
                    Self::write(&entry);
                }
            })
            .expect("mid-log: failed to spawn IO thread");

        LogWriter { shutdown, handle: Some(handle) }
    }

    fn write(entry: &LogEntry) {
        // Format: [LEVEL][TIER] message
        // Using eprintln so it goes to stderr and doesn't interfere
        // with any stdout-based IPC the game engine might use.
        eprintln!(
            "[{}][{}] {}",
            entry.level,
            entry.tier,
            entry.message,
        );
    }

    pub fn signal_shutdown(&self) {
        self.shutdown.store(true, Ordering::Relaxed);
    }
}

impl Drop for LogWriter {
    fn drop(&mut self) {
        self.signal_shutdown();
        if let Some(handle) = self.handle.take() {
            // Best-effort join — don't panic if the thread already exited.
            let _ = handle.join();
        }
    }
}
