// crates/mid-log/src/writer.rs

//! Background IO thread — drains the channel and writes to the active sink.
//!
//! The thread parks via `recv_timeout()` when the channel is empty — zero CPU idle.
//!
//! ## Per-entry pipeline (in order)
//!
//! 1. Coarse timestamp tick — refreshes `COARSE_TS_MS` so callers pay ~2 ns.
//! 2. Rate limit check (`RateLimiter::check()`).
//! 3. Console buffer push (`console_buffer::push()`).
//! 4. Color scheme snapshot refresh (if `COLOR_SCHEME_GEN` changed).
//! 5. Format snapshot (`FormatSnapshot::take()`).
//! 6. `format_entry()` → `Vec<u8>`.
//! 7. Write to stderr (`write_all`).
//! 8. Write to file sink if configured.

use std::io::{self, Write};
use std::fs;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::thread;
use std::time::Duration;

use crate::buffer::LogReceiver;
use crate::color::{ColorScheme, ResolvedScheme, COLOR_SCHEME_GEN};
use crate::entry::LogEntry;
use crate::format::FormatSnapshot;
use crate::level::{LogLevel, Tier};
use crate::ratelimit::{RateLimitConfig, RateLimiter};
use crate::console_buffer;

// ── Android sink ──────────────────────────────────────────────────────────────

#[cfg(all(target_os = "android", feature = "android-logcat"))]
mod android_sink {
    use crate::entry::LogEntry;
    use crate::level::LogLevel;
    use std::ffi::CString;

    extern "C" {
        fn __android_log_write(prio: i32, tag: *const i8, text: *const i8) -> i32;
    }

    const ANDROID_LOG_DEBUG: i32 = 3;
    const ANDROID_LOG_INFO:  i32 = 4;
    const ANDROID_LOG_WARN:  i32 = 5;
    const ANDROID_LOG_ERROR: i32 = 6;
    const ANDROID_LOG_FATAL: i32 = 7;
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
            "[{}] [F:{}] {}  ({}:{})",
            entry.tier.as_str(), entry.frame,
            entry.message, entry.file, entry.line,
        );
        if let Ok(c) = CString::new(text) {
            unsafe { __android_log_write(prio, TAG.as_ptr() as *const i8, c.as_ptr()); }
        }
    }
}

// ── Formatting ────────────────────────────────────────────────────────────────

struct VecWriter<'a>(&'a mut Vec<u8>);

impl std::fmt::Write for VecWriter<'_> {
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        self.0.extend_from_slice(s.as_bytes());
        Ok(())
    }
}

/// Format one log entry into `buf`.
///
/// Output (all fields enabled, colors on):
/// ```text
/// HH:MM:SS.mmm [LEVEL][TIER] [T:thread] [F:12345] message body  (module  file:line)\n
/// ```
fn format_entry(
    entry:  &LogEntry,
    fmt:    &FormatSnapshot,
    colors: &ResolvedScheme,
    buf:    &mut Vec<u8>,
) {
    use std::fmt::Write;
    buf.clear();
    let mut w = VecWriter(buf);
    let R = colors.reset;

    if fmt.show_timestamp {
        let _ = write!(w, "{}{}{} ", colors.timestamp, entry.format_time(), R);
    }

    let level_color = match entry.level {
        LogLevel::Trace => &colors.trace,
        LogLevel::Info  => &colors.info,
        LogLevel::Warn  => &colors.warn,
        LogLevel::Error => &colors.error,
        LogLevel::Fatal => &colors.fatal,
    };
    let bold_prefix = if entry.level == LogLevel::Fatal { colors.bold } else { "" };
    let _ = write!(w, "{}{}[{}]{} ", bold_prefix, level_color, entry.level.as_str(), R);

    let tier_color = match entry.tier {
        Tier::Low  => &colors.tier_low,
        Tier::Mid  => &colors.tier_mid,
        Tier::High => &colors.tier_high,
    };
    let _ = write!(w, "{}[{}]{} ", tier_color, entry.tier.as_str(), R);

    if fmt.show_thread {
        let _ = write!(w, "{}[T:{}]{} ", colors.thread, entry.thread, R);
    }
    if fmt.show_frame {
        let _ = write!(w, "{}[F:{}]{} ", colors.frame, entry.frame, R);
    }

    if colors.message.is_empty() {
        let _ = write!(w, "{}", entry.message);
    } else {
        let _ = write!(w, "{}{}{}", colors.message, entry.message, R);
    }

    if fmt.show_source_loc {
        let _ = write!(w, "  {}", colors.source);
        if fmt.show_module && !entry.module.is_empty() {
            let _ = write!(w, "{}  ", entry.module);
        }
        let _ = write!(w, "{}:{}{}", entry.file, entry.line, R);
    }

    let _ = write!(w, "\n");
}

// ── LogWriter ─────────────────────────────────────────────────────────────────

pub struct LogWriter {
    shutdown: Arc<AtomicBool>,
    handle:   Option<thread::JoinHandle<()>>,
}

impl LogWriter {
    pub fn spawn(
        receiver:     LogReceiver,
        log_file:     Option<std::path::PathBuf>,
        color_scheme: Arc<Mutex<ColorScheme>>,
    ) -> Self {
        let shutdown       = Arc::new(AtomicBool::new(false));
        let shutdown_clone = shutdown.clone();

        let handle = thread::Builder::new()
            .name("mid-log-io".into())
            .spawn(move || {
                let stderr = io::stderr();
                let mut buf        = Vec::<u8>::with_capacity(256);
                let mut rate_limit = RateLimiter::new();
                let mut rl_config  = RateLimitConfig::default();
                let mut rl_refresh = std::time::Instant::now();

                let mut colors    = ResolvedScheme::no_color();
                let mut color_gen = u64::MAX;

                let mut file_sink: Option<io::BufWriter<fs::File>> =
                    log_file.and_then(|p| {
                        match fs::OpenOptions::new().create(true).append(true).open(&p) {
                            Ok(f)  => Some(io::BufWriter::new(f)),
                            Err(e) => {
                                eprintln!("[mid-log] Could not open log file: {}", e);
                                None
                            }
                        }
                    });

                loop {
                    // ── Coarse timestamp tick ─────────────────────────────────
                    // Refreshes the global AtomicU64 so LogEntry::new() on
                    // calling threads pays ~2 ns instead of ~20–40 ns vDSO.
                    // One real clock call here covers all concurrent threads.
                    crate::entry::tick_coarse_timestamp();

                    // ── Rate-limit config refresh (once per second) ───────────
                    if rl_refresh.elapsed() >= Duration::from_secs(1) {
                        rl_config  = crate::ratelimit::get_rate_limit_config();
                        rl_refresh = std::time::Instant::now();

                        for summary in rate_limit.flush_expired(&rl_config) {
                            Self::emit_entry(
                                &summary, &FormatSnapshot::take(),
                                &colors, &stderr, &mut file_sink, &mut buf,
                            );
                        }
                    }

                    // ── Block until next entry ────────────────────────────────
                    let entry = match receiver.recv_timeout(Duration::from_millis(500)) {
                        Ok(e)  => e,
                        Err(_) => {
                            if shutdown_clone.load(Ordering::Relaxed) { break; }
                            continue;
                        }
                    };

                    // ── Color scheme snapshot refresh ─────────────────────────
                    let gen = COLOR_SCHEME_GEN.load(Ordering::Relaxed);
                    if gen != color_gen {
                        if let Ok(scheme) = color_scheme.lock() {
                            colors = ResolvedScheme::from_scheme(&*scheme);
                        }
                        color_gen = gen;
                    }

                    // ── Rate limit check ──────────────────────────────────────
                    match rate_limit.check(&entry, &rl_config) {
                        crate::ratelimit::RateDecision::Suppress => {
                            console_buffer::push(&entry);
                            continue;
                        }
                        crate::ratelimit::RateDecision::WindowExpired { summary } => {
                            Self::emit_entry(
                                &summary, &FormatSnapshot::take(),
                                &colors, &stderr, &mut file_sink, &mut buf,
                            );
                        }
                        crate::ratelimit::RateDecision::Allow => {}
                    }

                    // ── Console buffer ────────────────────────────────────────
                    console_buffer::push(&entry);

                    // ── Drain burst ───────────────────────────────────────────
                    let fmt = FormatSnapshot::take();
                    Self::emit_entry(&entry, &fmt, &colors, &stderr, &mut file_sink, &mut buf);

                    while let Ok(e) = receiver.try_recv() {
                        match rate_limit.check(&e, &rl_config) {
                            crate::ratelimit::RateDecision::Suppress => {
                                console_buffer::push(&e);
                                continue;
                            }
                            crate::ratelimit::RateDecision::WindowExpired { summary } => {
                                Self::emit_entry(
                                    &summary, &fmt, &colors,
                                    &stderr, &mut file_sink, &mut buf,
                                );
                            }
                            crate::ratelimit::RateDecision::Allow => {}
                        }
                        console_buffer::push(&e);
                        Self::emit_entry(&e, &fmt, &colors, &stderr, &mut file_sink, &mut buf);
                    }

                    if let Some(ref mut f) = file_sink { f.flush().ok(); }
                }

                // Final drain after shutdown signal.
                while let Ok(e) = receiver.try_recv() {
                    console_buffer::push(&e);
                    Self::emit_entry(
                        &e, &FormatSnapshot::take(),
                        &colors, &stderr, &mut file_sink, &mut buf,
                    );
                }
                if let Some(ref mut f) = file_sink { f.flush().ok(); }
            })
            .expect("mid-log: failed to spawn IO thread");

        LogWriter { shutdown, handle: Some(handle) }
    }

    fn emit_entry(
        entry:     &LogEntry,
        fmt:       &FormatSnapshot,
        colors:    &ResolvedScheme,
        stderr:    &io::Stderr,
        file_sink: &mut Option<io::BufWriter<fs::File>>,
        buf:       &mut Vec<u8>,
    ) {
        #[cfg(all(target_os = "android", feature = "android-logcat"))]
        { android_sink::write(entry); return; }

        #[cfg(not(all(target_os = "android", feature = "android-logcat")))]
        {
            format_entry(entry, fmt, colors, buf);
            { let mut err = stderr.lock(); err.write_all(buf).ok(); }
            if let Some(ref mut f) = file_sink { f.write_all(buf).ok(); }
        }
    }

    pub fn signal_shutdown(&self) {
        self.shutdown.store(true, Ordering::Relaxed);
    }
}

impl Drop for LogWriter {
    fn drop(&mut self) {
        self.signal_shutdown();
        if let Some(h) = self.handle.take() { let _ = h.join(); }
    }
}
