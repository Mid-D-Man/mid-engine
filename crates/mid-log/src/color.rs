// crates/mid-log/src/color.rs

//! ANSI color support — per-field scheme, inline `paint()`, TTY detection.
//!
//! ## Design
//!
//! Color is split into two layers:
//!
//! 1. **`ColorScheme`** — per-field default colors applied by the IO thread
//!    to level badges, tier badges, timestamps, source locations, etc.
//!    Updated via `update_color_scheme(|s| { ... })`.
//!
//! 2. **`paint(value, Color)`** — inline coloring of arbitrary values inside
//!    a log message. Respects the global enable flag automatically.
//!    ```rust,no_run
//!    use mid_log::{mid_warn, level::Tier, color::{Color, paint}};
//!    let hp = 15u32;
//!    mid_warn!(Tier::High, "Low health: {} / {}", paint(hp, Color::Red), 100);
//!    ```
//!
//! ## Color detection
//!
//! On `MidLogger::init*()` the logger checks (in order):
//!   1. `NO_COLOR` env var present → disabled  (<https://no-color.org>)
//!   2. `FORCE_COLOR` env var present → enabled
//!   3. stderr is a TTY → enabled
//!   4. Otherwise → disabled (pipe, CI, file redirect)
//!
//! Override at any time with `set_colors_enabled(bool)`.
//!
//! ## IO thread scheme snapshot
//!
//! The IO thread maintains a local `ResolvedScheme` (pre-rendered ANSI strings)
//! and refreshes it only when `COLOR_SCHEME_GEN` increments. The common path
//! pays one `AtomicU64::load` per entry — zero lock overhead.

use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

// ── Global state ──────────────────────────────────────────────────────────────

/// Whether ANSI color output is currently enabled.
static COLORS_ENABLED: AtomicBool = AtomicBool::new(false);

/// Monotonically incrementing generation counter.
/// Bumped whenever `ColorScheme` or the enable flag changes.
/// The IO thread compares its local copy to decide whether to re-snapshot.
pub(crate) static COLOR_SCHEME_GEN: AtomicU64 = AtomicU64::new(0);

/// Global color scheme storage. Initialized once in `MidLogger::init*()`.
static COLOR_SCHEME: OnceLock<Arc<Mutex<ColorScheme>>> = OnceLock::new();

// ── Public color API ──────────────────────────────────────────────────────────

/// Returns `true` if ANSI color output is currently enabled.
#[inline(always)]
pub fn is_colors_enabled() -> bool {
    COLORS_ENABLED.load(Ordering::Relaxed)
}

/// Manually enable or disable ANSI color output.
///
/// This overrides the TTY auto-detection done at init.
/// The IO thread picks up the change on its next log entry.
pub fn set_colors_enabled(enabled: bool) {
    COLORS_ENABLED.store(enabled, Ordering::Relaxed);
    COLOR_SCHEME_GEN.fetch_add(1, Ordering::Relaxed);
}

/// Detect whether to enable colors based on environment and TTY state.
///
/// Called once at `MidLogger::init*()`. Rules in priority order:
/// 1. `NO_COLOR` is set → disabled.
/// 2. `FORCE_COLOR` is set → enabled.
/// 3. stderr is a TTY → enabled.
/// 4. Otherwise → disabled.
pub(crate) fn detect_colors() -> bool {
    if std::env::var_os("NO_COLOR").is_some() {
        return false;
    }
    if std::env::var_os("FORCE_COLOR").is_some() {
        return true;
    }
    // std::io::IsTerminal is stable since Rust 1.70.
    use std::io::IsTerminal;
    std::io::stderr().is_terminal()
}

/// Initialize the global color system. Called once from `MidLogger::init*()`.
///
/// Returns an `Arc<Mutex<ColorScheme>>` that is shared between `MidLogger`
/// (for public access) and the IO thread (for snapshot refresh).
pub(crate) fn init_colors(scheme: ColorScheme) -> Arc<Mutex<ColorScheme>> {
    let detected = detect_colors();
    COLORS_ENABLED.store(detected, Ordering::Relaxed);
    let arc = Arc::new(Mutex::new(scheme));
    // If init is called twice (second call is a no-op from OnceLock),
    // the second Arc is dropped — the original scheme is preserved.
    let _ = COLOR_SCHEME.set(arc.clone());
    arc
}

/// Update the global `ColorScheme` in place.
///
/// The IO thread will pick up the change on its next log entry.
///
/// # Example
/// ```rust,no_run
/// use mid_log::color::{update_color_scheme, Color};
///
/// update_color_scheme(|s| {
///     s.warn    = Color::BrightYellow;
///     s.error   = Color::Rgb(255, 80, 80);
///     s.message = Color::None; // no tint on message body
/// });
/// ```
pub fn update_color_scheme(f: impl FnOnce(&mut ColorScheme)) {
    if let Some(arc) = COLOR_SCHEME.get() {
        if let Ok(mut guard) = arc.lock() {
            f(&mut *guard);
        }
    }
    COLOR_SCHEME_GEN.fetch_add(1, Ordering::Relaxed);
}

// ── Color enum ────────────────────────────────────────────────────────────────

/// An ANSI foreground color or text style.
///
/// Used in [`ColorScheme`] for per-field defaults and in [`paint()`] for
/// inline coloring of values inside log messages.
///
/// All variants except `Rgb` and `Custom` are `Clone`-friendly and have
/// no runtime allocation until `to_ansi_prefix()` is called (which happens
/// only when colors are enabled and only on the IO thread).
#[derive(Debug, Clone, PartialEq)]
pub enum Color {
    // ── Standard foreground (30–37) ───────────────────────────────────────────
    Black,
    Red,
    Green,
    Yellow,
    Blue,
    Magenta,
    Cyan,
    White,
    // ── Bright foreground (90–97) ─────────────────────────────────────────────
    BrightBlack,    // "dark grey" on most terminals
    BrightRed,
    BrightGreen,
    BrightYellow,
    BrightBlue,
    BrightMagenta,
    BrightCyan,
    BrightWhite,
    // ── Text styles ───────────────────────────────────────────────────────────
    Bold,
    Dim,
    Italic,
    Underline,
    // ── True color ────────────────────────────────────────────────────────────
    /// 24-bit RGB foreground. Supported by most modern terminals (not Windows 7).
    Rgb(u8, u8, u8),
    /// Raw ANSI parameter string placed between `\x1b[` and `m`.
    ///
    /// Examples:
    /// - `"38;5;208"` — xterm-256 orange foreground
    /// - `"1;31"`     — bold red
    /// - `"48;2;0;0;255"` — blue background (RGB)
    Custom(String),
    /// No color. Leaves the terminal in its current state.
    None,
}

impl Color {
    /// Returns the opening ANSI escape sequence, or `None` for `Color::None`.
    pub fn to_ansi_prefix(&self) -> Option<String> {
        Some(match self {
            Color::Black         => "\x1b[30m".to_owned(),
            Color::Red           => "\x1b[31m".to_owned(),
            Color::Green         => "\x1b[32m".to_owned(),
            Color::Yellow        => "\x1b[33m".to_owned(),
            Color::Blue          => "\x1b[34m".to_owned(),
            Color::Magenta       => "\x1b[35m".to_owned(),
            Color::Cyan          => "\x1b[36m".to_owned(),
            Color::White         => "\x1b[37m".to_owned(),
            Color::BrightBlack   => "\x1b[90m".to_owned(),
            Color::BrightRed     => "\x1b[91m".to_owned(),
            Color::BrightGreen   => "\x1b[92m".to_owned(),
            Color::BrightYellow  => "\x1b[93m".to_owned(),
            Color::BrightBlue    => "\x1b[94m".to_owned(),
            Color::BrightMagenta => "\x1b[95m".to_owned(),
            Color::BrightCyan    => "\x1b[96m".to_owned(),
            Color::BrightWhite   => "\x1b[97m".to_owned(),
            Color::Bold          => "\x1b[1m".to_owned(),
            Color::Dim           => "\x1b[2m".to_owned(),
            Color::Italic        => "\x1b[3m".to_owned(),
            Color::Underline     => "\x1b[4m".to_owned(),
            Color::Rgb(r, g, b)  => format!("\x1b[38;2;{};{};{}m", r, g, b),
            Color::Custom(s)     => format!("\x1b[{}m", s),
            Color::None          => return None,
        })
    }

    /// Convenience: `to_ansi_prefix()` with empty string for `Color::None`.
    pub(crate) fn to_ansi_string(&self) -> String {
        self.to_ansi_prefix().unwrap_or_default()
    }

    /// ANSI background equivalent. Used by `paint_bg()`.
    pub(crate) fn to_bg_ansi_string(&self) -> String {
        match self {
            Color::Black         => "\x1b[40m".to_owned(),
            Color::Red           => "\x1b[41m".to_owned(),
            Color::Green         => "\x1b[42m".to_owned(),
            Color::Yellow        => "\x1b[43m".to_owned(),
            Color::Blue          => "\x1b[44m".to_owned(),
            Color::Magenta       => "\x1b[45m".to_owned(),
            Color::Cyan          => "\x1b[46m".to_owned(),
            Color::White         => "\x1b[47m".to_owned(),
            Color::BrightBlack   => "\x1b[100m".to_owned(),
            Color::BrightRed     => "\x1b[101m".to_owned(),
            Color::BrightGreen   => "\x1b[102m".to_owned(),
            Color::BrightYellow  => "\x1b[103m".to_owned(),
            Color::BrightBlue    => "\x1b[104m".to_owned(),
            Color::BrightMagenta => "\x1b[105m".to_owned(),
            Color::BrightCyan    => "\x1b[106m".to_owned(),
            Color::BrightWhite   => "\x1b[107m".to_owned(),
            Color::Rgb(r, g, b)  => format!("\x1b[48;2;{};{};{}m", r, g, b),
            Color::Custom(s)     => format!("\x1b[{}m", s),
            // Styles and None have no background equivalent
            _ => String::new(),
        }
    }
}

// ── ColorScheme ───────────────────────────────────────────────────────────────

/// Per-field color configuration applied by the IO thread when formatting log lines.
///
/// Update at any time via [`update_color_scheme()`]. Changes take effect on
/// the next log entry — no restart needed.
///
/// # Default scheme
///
/// | Field       | Default          | Applies to                          |
/// |-------------|------------------|-------------------------------------|
/// | `trace`     | `Dim`            | `[TRACE]` level badge               |
/// | `info`      | `None`           | `[INFO ]` level badge               |
/// | `warn`      | `Yellow`         | `[WARN ]` level badge               |
/// | `error`     | `Red`            | `[ERROR]` level badge               |
/// | `fatal`     | `BrightRed`      | `[FATAL]` badge (Bold auto-added)   |
/// | `tier_low`  | `Cyan`           | `[LOW ]` tier badge                 |
/// | `tier_mid`  | `Magenta`        | `[MID ]` tier badge                 |
/// | `tier_high` | `Green`          | `[HIGH]` tier badge                 |
/// | `timestamp` | `Dim`            | `HH:MM:SS.mmm`                      |
/// | `source`    | `Dim`            | `file:line`                         |
/// | `module`    | `Dim`            | Rust module path                    |
/// | `thread`    | `Blue`           | `[T:name]` badge                    |
/// | `frame`     | `Dim`            | `[F:n]` badge                       |
/// | `message`   | `None`           | Log message body                    |
#[derive(Debug, Clone)]
pub struct ColorScheme {
    pub trace:     Color,
    pub info:      Color,
    pub warn:      Color,
    pub error:     Color,
    /// Bold is automatically prepended to this color for FATAL entries.
    pub fatal:     Color,
    pub tier_low:  Color,
    pub tier_mid:  Color,
    pub tier_high: Color,
    pub timestamp: Color,
    pub source:    Color,
    pub module:    Color,
    pub thread:    Color,
    pub frame:     Color,
    /// Color applied to the message body text.
    /// `Color::None` leaves the message in the terminal's default color.
    pub message:   Color,
}

impl Default for ColorScheme {
    fn default() -> Self {
        Self {
            trace:     Color::Dim,
            info:      Color::None,
            warn:      Color::Yellow,
            error:     Color::Red,
            fatal:     Color::BrightRed,
            tier_low:  Color::Cyan,
            tier_mid:  Color::Magenta,
            tier_high: Color::Green,
            timestamp: Color::Dim,
            source:    Color::Dim,
            module:    Color::Dim,
            thread:    Color::Blue,
            frame:     Color::Dim,
            message:   Color::None,
        }
    }
}

// ── ResolvedScheme — IO thread local snapshot ─────────────────────────────────

/// Pre-rendered ANSI strings for every color slot.
///
/// Rebuilt from `ColorScheme` whenever `COLOR_SCHEME_GEN` increments.
/// The IO thread pays zero lock overhead on the common path — it checks a
/// single `AtomicU64` and re-locks the scheme only when the value changes.
///
/// Fields are plain `String`s: empty when colors are disabled, ANSI otherwise.
/// `reset` / `bold` are `&'static str` to avoid allocation.
#[derive(Debug, Clone)]
pub(crate) struct ResolvedScheme {
    pub trace:     String,
    pub info:      String,
    pub warn:      String,
    pub error:     String,
    pub fatal:     String,
    pub tier_low:  String,
    pub tier_mid:  String,
    pub tier_high: String,
    pub timestamp: String,
    pub source:    String,
    pub module:    String,
    pub thread:    String,
    pub frame:     String,
    pub message:   String,
    /// `"\x1b[0m"` when colors enabled, `""` otherwise.
    pub reset: &'static str,
    /// `"\x1b[1m"` when colors enabled, `""` otherwise. Prepended to Fatal.
    pub bold:  &'static str,
}

impl ResolvedScheme {
    pub fn from_scheme(scheme: &ColorScheme) -> Self {
        if !is_colors_enabled() {
            return Self::no_color();
        }
        Self {
            trace:     scheme.trace.to_ansi_string(),
            info:      scheme.info.to_ansi_string(),
            warn:      scheme.warn.to_ansi_string(),
            error:     scheme.error.to_ansi_string(),
            fatal:     scheme.fatal.to_ansi_string(),
            tier_low:  scheme.tier_low.to_ansi_string(),
            tier_mid:  scheme.tier_mid.to_ansi_string(),
            tier_high: scheme.tier_high.to_ansi_string(),
            timestamp: scheme.timestamp.to_ansi_string(),
            source:    scheme.source.to_ansi_string(),
            module:    scheme.module.to_ansi_string(),
            thread:    scheme.thread.to_ansi_string(),
            frame:     scheme.frame.to_ansi_string(),
            message:   scheme.message.to_ansi_string(),
            reset:     "\x1b[0m",
            bold:      "\x1b[1m",
        }
    }

    pub fn no_color() -> Self {
        Self {
            trace:     String::new(), info:      String::new(),
            warn:      String::new(), error:     String::new(),
            fatal:     String::new(), tier_low:  String::new(),
            tier_mid:  String::new(), tier_high: String::new(),
            timestamp: String::new(), source:    String::new(),
            module:    String::new(), thread:    String::new(),
            frame:     String::new(), message:   String::new(),
            reset:     "",            bold:      "",
        }
    }
}

// ── paint() — inline coloring ─────────────────────────────────────────────────

/// A colored or plain value for use inside `format!()` strings.
/// Created by [`paint()`] or [`paint_bg()`]. Implements `Display`.
pub struct Painted {
    text: String,
    /// Pre-rendered ANSI prefix. Empty string when colors are disabled.
    ansi: String,
}

impl fmt::Display for Painted {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.ansi.is_empty() {
            f.write_str(&self.text)
        } else {
            // Write: <ansi_prefix><text><reset>
            write!(f, "{}{}\x1b[0m", self.ansi, self.text)
        }
    }
}

/// Apply a foreground color (or style) to a value inside a log message.
///
/// Respects the global `COLORS_ENABLED` flag. When colors are disabled this
/// is a pure pass-through — no ANSI codes, no allocation beyond the
/// `Display` formatting of `text`.
///
/// # Example
/// ```rust,no_run
/// use mid_log::{mid_info, mid_warn, level::Tier, color::{Color, paint}};
///
/// let hp  = 15u32;
/// let max = 100u32;
/// mid_warn!(
///     Tier::High,
///     "HP: {} / {}  — {}",
///     paint(hp,  Color::Red),
///     paint(max, Color::Green),
///     paint("critical", Color::Bold),
/// );
/// ```
pub fn paint(text: impl fmt::Display, fg: Color) -> Painted {
    Painted {
        text: text.to_string(),
        ansi: if is_colors_enabled() { fg.to_ansi_string() } else { String::new() },
    }
}

/// Apply a foreground and background color to a value inside a log message.
///
/// ```rust,no_run
/// use mid_log::{mid_error, level::Tier, color::{Color, paint_bg}};
/// mid_error!(Tier::High, "{}", paint_bg("CRITICAL", Color::White, Color::Red));
/// ```
pub fn paint_bg(text: impl fmt::Display, fg: Color, bg: Color) -> Painted {
    if !is_colors_enabled() {
        return Painted { text: text.to_string(), ansi: String::new() };
    }
    Painted {
        text: text.to_string(),
        ansi: format!("{}{}", fg.to_ansi_string(), bg.to_bg_ansi_string()),
    }
}
