// crates/mid-log/src/writer.rs

//! Background IO thread — drains the channel and writes to the active sink.
//!
//! The thread parks (via `recv()` blocking) when no entries are queued,
//! consuming zero CPU during quiet periods.
//!
//! ## Sink selection
//!
//! | Target                          | Sink                       |
//! |---------------------------------|----------------------------|
//! | Android + `android-logcat` feat | `__android_log_write`      |
//! | Everything else                 | `stderr` (raw `write_all`) |
//!
//! The stderr sink uses `std::io::Write` directly instead of `eprintln!`
//! to bypass Rust's fmt machinery.

use std::thread;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::io::{self, Write};
use std::fs;

use crate::buffer::LogReceiver;
use crate::entry::LogEntry;
use crate::level::LogLevel;

// ── Platform sink ─────────────────────────────────────────────────────────────

#[cfg(all(target_os = "android", feature = "android-logcat"))]
mod android_sink {
    use super::LogEntry;
    use crate::level::LogLevel;
    use std::ffi::CString;

    extern "C" {
        fn __android_log_write(prio: i32, tag: *const i8, text: *const i8) -> i32;
    }

    const ANDROID_LOG_DEBUG:   i32 = 3;
    const ANDROID_LOG_INFO:    i32 = 4;
    const ANDROID_LOG_WARN:    i32 = 5;
    const ANDROID_LOG_ERROR:   i32 = 6;
    const ANDROID_LOG_FATAL:   i32 = 7;
    const TAG: &[u8] = b"mid-engine\0";

    pub fn write(entry: &LogEntry) {
        let prio = match entry.level {
            LogLevel::Trace => ANDROID_LOG_DEBUG,
            LogLevel::Info  => ANDROID_LOG_INFO,
            LogLevel::Warn  => ANDROID_LOG_WARN,
            LogLevel::Error => ANDROID_LOG_ERROR,
            LogLevel::Fatal => ANDROID_LOG_FATAL,
        };
        
        let text = format!(
            "[{}] {} ({}:{})",
            entry.tier.as_str(),
            entry.message,
            entry.file,
            entry.line,
        );
        
        if let Ok(c_text) = CString::new(text) {
            unsafe {
                __android_log_write(prio, TAG.as_ptr() as *const i8, c_text.as_ptr());
            }
        }
    }
}

// ── Formatted output for all non-Android targets ──────────────────────────────

/// Helper to bridge std::fmt::Write (used by write!) to a Vec<u8> (io::Write).
struct VecWriter<'a>(&'a mut Vec<u8>);

impl std::fmt::Write for VecWriter<'_> {
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        self.0.extend_from_slice(s.as_bytes());
        Ok(())
    }
}

/// Format a log entry into the provided buffer.
/// Format: `HH:MM:SS.mmm [LEVEL][TIER] message  (module file:line)\n`
fn format_entry(entry: &LogEntry, buf: &mut Vec<u8>) {
    use std::fmt::Write;
    buf.clear();
    
    let mut w = VecWriter(buf);
    write!(w,
        "{} [{}][{}] {}  ({}  {}:{})\n",
        entry.format_time(),
        entry.level.as_str(),
        entry.tier.as_str(),
        entry.message,
        entry.module,
        entry.file,
        entry.line,
    ).ok();
}

// ── LogWriter ─────────────────────────────────────────────────────────────────

pub struct LogWriter {
    shutdown: Arc<AtomicBool>,
    handle:   Option<thread::JoinHandle<()>>,
}

impl LogWriter {
    /// Spawn the background IO thread.
    pub fn spawn(receiver: LogReceiver, log_file: Option<std::path::PathBuf>) -> Self {
        let shutdown       = Arc::new(AtomicBool::new(false));
        let shutdown_clone = shutdown.clone();

        let handle = thread::Builder::new()
            .name("mid-log-io".into())
            .spawn(move || {
                let stderr = io::stderr();
                let mut buf = Vec::<u8>::with_capacity(256);

                let mut file_sink: Option<io::BufWriter<fs::File>> = log_file.and_then(|p| {
                    match fs::OpenOptions::new().create(true).append(true).open(&p) {
                        Ok(f)  => Some(io::BufWriter::new(f)),
                        Err(e) => {
                            eprintln!("[mid-log] Could not open log file: {}", e);
                            None
                        }
                    }
                });

                loop {
                    match receiver.recv() {
                        Ok(entry) => {
                            Self::write_entry(&entry, &stderr, &mut file_sink, &mut buf);
                            
                            // Drain burst
                            while let Ok(e) = receiver.try_recv() {
                                Self::write_entry(&e, &stderr, &mut file_sink, &mut buf);
                            }
                            
                            if let Some(ref mut f) = file_sink {
                                f.flush().ok();
                            }
                        }
                        Err(_) => {
                            if shutdown_clone.load(Ordering::Relaxed) {
                                break;
                            }
                            break;
                        }
                    }
                }

                if let Some(ref mut f) = file_sink {
                    f.flush().ok();
                }
            })
            .expect("mid-log: failed to spawn IO thread");

        LogWriter { shutdown, handle: Some(handle) }
    }

    fn write_entry(
        entry:     &LogEntry,
        stderr:    &io::Stderr,
        file_sink: &mut Option<io::BufWriter<fs::File>>,
        buf:       &mut Vec<u8>,
    ) {
        #[cfg(all(target_os = "android", feature = "android-logcat"))]
        {
            android_sink::write(entry);
            return;
        }

        #[cfg(not(all(target_os = "android", feature = "android-logcat")))]
        {
            format_entry(entry, buf);

            {
                let mut err = stderr.lock();
                err.write_all(buf).ok();
            }

            if let Some(ref mut f) = file_sink {
                f.write_all(buf).ok();
            }
        }
    }

    pub fn signal_shutdown(&self) {
        self.shutdown.store(true, Ordering::Relaxed);
    }
}

impl Drop for LogWriter {
    fn drop(&mut self) {
        self.signal_shutdown();
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}
