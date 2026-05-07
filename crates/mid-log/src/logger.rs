// crates/mid-log/src/logger.rs

//! Global logger singleton.

use std::borrow::Cow;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};

use crate::buffer::{self, LogSender};
use crate::color::{self, ColorScheme};
use crate::entry::LogEntry;
use crate::filter;
use crate::format::{FormatConfig, set_format};
use crate::kv::KvPair;
use crate::level::{LogLevel, Tier};
use crate::writer::LogWriter;

pub struct MidLogger {
    sender:  LogSender,
    _writer: LogWriter,
    pub(crate) color_scheme: Arc<Mutex<ColorScheme>>,
}

static INSTANCE: OnceLock<MidLogger> = OnceLock::new();

impl MidLogger {
    // ── Init ──────────────────────────────────────────────────────────────────

    pub fn init() -> bool {
        Self::init_full(InitConfig::default())
    }

    pub fn init_with(log_file: Option<PathBuf>) -> bool {
        Self::init_full(InitConfig { log_file, ..Default::default() })
    }

    pub fn init_full(config: InitConfig) -> bool {
        set_format(&config.format);
        filter::set_min_level(config.min_level);
        let scheme_arc = color::init_colors(config.color_scheme);

        let old_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            if let Some(logger) = INSTANCE.get() {
                logger.log(
                    LogLevel::Fatal,
                    Tier::Low,
                    Cow::Owned(format!("PANIC: {}", info)),
                    "<panic>", 0, "<panic>",
                );
            }
            Self::flush();
            old_hook(info);
        }));

        let (sender, receiver) = buffer::create();
        let writer = LogWriter::spawn(receiver, config.log_file, scheme_arc.clone());

        INSTANCE.set(MidLogger {
            sender,
            _writer: writer,
            color_scheme: scheme_arc,
        }).is_ok()
    }

    // ── Access ────────────────────────────────────────────────────────────────

    #[inline]
    pub fn get() -> Option<&'static MidLogger> {
        INSTANCE.get()
    }

    // ── Logging — printf path ─────────────────────────────────────────────────

    /// Send a printf-style log entry. `message` is the already-formatted String
    /// produced by `format!()` in the calling macro.
    #[inline]
    pub fn log(
        &self,
        level:   LogLevel,
        tier:    Tier,
        message: Cow<'static, str>,
        file:    &'static str,
        line:    u32,
        module:  &'static str,
    ) {
        let entry = LogEntry::new(level, tier, message, Vec::new(), file, line, module);
        self.sender.send(entry).ok();
    }

    // ── Logging — KV path ─────────────────────────────────────────────────────

    /// Send a structured KV log entry. `message` is a static string literal.
    /// `kvs` contains typed key-value pairs — no `format!()` is called here.
    ///
    /// The IO thread formats `kvs` into `key=value` pairs appended after the
    /// message, before the source location.
    #[inline]
    pub fn log_kv(
        &self,
        level:   LogLevel,
        tier:    Tier,
        message: &'static str,
        kvs:     Vec<KvPair>,
        file:    &'static str,
        line:    u32,
        module:  &'static str,
    ) {
        let entry = LogEntry::new(
            level,
            tier,
            Cow::Borrowed(message),
            kvs,
            file, line, module,
        );
        self.sender.send(entry).ok();
    }

    // ── Flush / shutdown ──────────────────────────────────────────────────────

    pub fn flush() {
        if let Some(logger) = INSTANCE.get() {
            while !logger.sender.is_empty() {
                std::thread::sleep(std::time::Duration::from_micros(100));
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
    }

    pub fn shutdown() {
        if let Some(logger) = INSTANCE.get() {
            while !logger.sender.is_empty() {
                std::thread::sleep(std::time::Duration::from_micros(100));
            }
            logger._writer.signal_shutdown();
        }
    }
}

// ── InitConfig ────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub struct InitConfig {
    pub log_file:     Option<PathBuf>,
    pub min_level:    LogLevel,
    pub format:       FormatConfig,
    pub color_scheme: ColorScheme,
}

impl Default for InitConfig {
    fn default() -> Self {
        Self {
            log_file:     None,
            min_level:    LogLevel::Trace,
            format:       FormatConfig::default(),
            color_scheme: ColorScheme::default(),
        }
    }
}
