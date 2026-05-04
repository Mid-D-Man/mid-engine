// crates/mid-log/src/logger.rs

//! Global logger singleton.

use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};

use crate::buffer::{self, LogSender};
use crate::color::{self, ColorScheme};
use crate::entry::LogEntry;
use crate::filter;
use crate::format::{FormatConfig, set_format};
use crate::level::{LogLevel, Tier};
use crate::writer::LogWriter;

pub struct MidLogger {
    sender:  LogSender,
    _writer: LogWriter,
    /// Shared with the IO thread for live color-scheme updates.
    pub(crate) color_scheme: Arc<Mutex<ColorScheme>>,
}

static INSTANCE: OnceLock<MidLogger> = OnceLock::new();

impl MidLogger {
    // ── Init ──────────────────────────────────────────────────────────────────

    /// Initialise with defaults (stderr, auto-detect colors, default format).
    pub fn init() -> bool {
        Self::init_full(InitConfig::default())
    }

    /// Initialise with an optional file tee (stderr + file).
    pub fn init_with(log_file: Option<PathBuf>) -> bool {
        Self::init_full(InitConfig { log_file, ..Default::default() })
    }

    /// Full initialisation with complete configuration.
    ///
    /// # Example
    /// ```rust,no_run
    /// use mid_log::logger::{MidLogger, InitConfig};
    /// use mid_log::color::ColorScheme;
    /// use mid_log::format::FormatConfig;
    /// use mid_log::level::LogLevel;
    /// use std::path::PathBuf;
    ///
    /// MidLogger::init_full(InitConfig {
    ///     log_file:    Some(PathBuf::from("game.log")),
    ///     min_level:   LogLevel::Info,
    ///     format:      FormatConfig { show_frame: true, ..Default::default() },
    ///     color_scheme: ColorScheme::default(),
    /// });
    /// ```
    pub fn init_full(config: InitConfig) -> bool {
        // Apply format flags.
        set_format(&config.format);

        // Apply level filter.
        filter::set_min_level(config.min_level);

        // Initialize color system and get the shared scheme arc.
        let scheme_arc = color::init_colors(config.color_scheme);

        // Register panic hook so Fatal entries reach the file sink.
        let old_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            if let Some(logger) = INSTANCE.get() {
                logger.log(
                    LogLevel::Fatal,
                    Tier::Low,
                    format!("PANIC: {}", info),
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

    // ── Logging ───────────────────────────────────────────────────────────────

    #[inline]
    pub fn log(
        &self,
        level:  LogLevel,
        tier:   Tier,
        message: String,
        file:   &'static str,
        line:   u32,
        module: &'static str,
    ) {
        let entry = LogEntry::new(level, tier, message, file, line, module);
        self.sender.send(entry).ok();
    }

    // ── Flush / shutdown ──────────────────────────────────────────────────────

    /// Wait for all queued entries to be written without stopping the logger.
    pub fn flush() {
        if let Some(logger) = INSTANCE.get() {
            while !logger.sender.is_empty() {
                std::thread::sleep(std::time::Duration::from_micros(100));
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
    }

    /// Flush then stop the IO thread.
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

/// Full initialization configuration for `MidLogger::init_full()`.
#[derive(Debug)]
pub struct InitConfig {
    /// Optional file path for log tee (created/appended). `None` = stderr only.
    pub log_file:     Option<PathBuf>,
    /// Minimum level to process. Entries below this level are filtered
    /// before `format!()` runs. Default: `Trace` (log everything).
    pub min_level:    LogLevel,
    /// Which fields appear in each formatted line. Default: timestamp + source.
    pub format:       FormatConfig,
    /// Per-field color assignments. Default: standard mid-log palette.
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
