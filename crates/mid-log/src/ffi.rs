// crates/mid-log/src/ffi.rs

//! C-compatible FFI exports — the C face of mid-log.
//!
//! ## New in this version
//!
//! - `mid_log_init_full_c()`      — full init with all options from C
//! - `mid_log_set_colors()`       — enable/disable colors from C
//! - `mid_log_set_format_flags()` — control which fields appear from C
//! - `mid_log_set_frame()`        — set game frame counter from C
//! - `mid_log_console_init()`     — init in-game console buffer from C
//! - `mid_log_console_count()`    — query buffered entry count from C
//! - `mid_log_set_rate_limit()`   — configure rate limiting from C
//! - `mid_log_update_color_c()`   — update one color slot from C

use std::ffi::CStr;
use std::os::raw::c_char;
use std::path::PathBuf;
use std::time::Duration;

use crate::color::{self, Color};
use crate::filter;
use crate::format::{FormatConfig, set_format};
use crate::frame;
use crate::level::{LogLevel, Tier};
use crate::logger::{InitConfig, MidLogger};
use crate::ratelimit::{set_rate_limit_config, RateLimitConfig};

// ── Lifecycle ─────────────────────────────────────────────────────────────────

/// Initialise with stderr only, auto-detect colors, default format.
/// Returns 1 on success, 0 if already initialised.
#[no_mangle]
pub extern "C" fn mid_log_init() -> u8 {
    if MidLogger::init() { 1 } else { 0 }
}

/// Initialise with a file tee.
///
/// `path`: null-terminated UTF-8 path, or NULL for stderr only.
/// Returns 1 on success, 0 if already init or path is invalid.
#[no_mangle]
pub unsafe extern "C" fn mid_log_init_with_file(path: *const c_char) -> u8 {
    let log_file = if path.is_null() {
        None
    } else {
        match CStr::from_ptr(path).to_str() {
            Ok(s)  => Some(PathBuf::from(s)),
            Err(_) => return 0,
        }
    };
    if MidLogger::init_with(log_file) { 1 } else { 0 }
}

/// Initialise with full configuration.
///
/// ```c
/// mid_log_init_full_c(
///     "game.log",      // log_file  — NULL for stderr only
///     MID_LEVEL_INFO,  // min_level
///     1,               // show_timestamp
///     1,               // show_source_loc
///     0,               // show_module
///     1,               // show_thread
///     1,               // show_frame
///     -1,              // colors: -1=auto-detect, 0=disable, 1=force enable
/// );
/// ```
///
/// Returns 1 on success, 0 if already init.
#[no_mangle]
pub unsafe extern "C" fn mid_log_init_full_c(
    log_file:        *const c_char,
    min_level:       u8,
    show_timestamp:  u8,
    show_source_loc: u8,
    show_module:     u8,
    show_thread:     u8,
    show_frame:      u8,
    colors:          i8,   // -1 = auto, 0 = off, 1 = on
) -> u8 {
    let log_file = if log_file.is_null() {
        None
    } else {
        CStr::from_ptr(log_file).to_str().ok().map(PathBuf::from)
    };

    let result = MidLogger::init_full(InitConfig {
        log_file,
        min_level:    LogLevel::from_u8(min_level),
        format:       FormatConfig {
            show_timestamp:  show_timestamp  != 0,
            show_source_loc: show_source_loc != 0,
            show_module:     show_module     != 0,
            show_thread:     show_thread     != 0,
            show_frame:      show_frame      != 0,
        },
        color_scheme: crate::color::ColorScheme::default(),
    });

    // Apply color override after init (init_full handles auto-detect).
    match colors {
        0  => color::set_colors_enabled(false),
        1  => color::set_colors_enabled(true),
        _  => { /* -1 = keep whatever auto-detect set */ }
    }

    if result { 1 } else { 0 }
}

// ── Level filter ──────────────────────────────────────────────────────────────

/// Set minimum log level. 0=TRACE 1=INFO 2=WARN 3=ERROR 4=FATAL.
#[no_mangle]
pub extern "C" fn mid_log_set_min_level(level: u8) {
    filter::set_min_level(LogLevel::from_u8(level));
}

/// Returns current minimum log level.
#[no_mangle]
pub extern "C" fn mid_log_get_min_level() -> u8 {
    filter::get_min_level() as u8
}

// ── Colors ────────────────────────────────────────────────────────────────────

/// Enable or disable ANSI color output.
/// 0 = disable, any other value = enable.
#[no_mangle]
pub extern "C" fn mid_log_set_colors(enabled: u8) {
    color::set_colors_enabled(enabled != 0);
}

/// Returns 1 if colors are currently enabled, 0 otherwise.
#[no_mangle]
pub extern "C" fn mid_log_get_colors() -> u8 {
    if color::is_colors_enabled() { 1 } else { 0 }
}

/// Color slot identifiers for `mid_log_update_color_c()`.
///
/// Match these constants in C with the `MID_COLOR_SLOT_*` defines in `mid_log.h`.
#[repr(u8)]
pub enum ColorSlot {
    Trace    = 0,
    Info     = 1,
    Warn     = 2,
    Error    = 3,
    Fatal    = 4,
    TierLow  = 5,
    TierMid  = 6,
    TierHigh = 7,
    Timestamp = 8,
    Source   = 9,
    Module   = 10,
    Thread   = 11,
    Frame    = 12,
    Message  = 13,
}

/// Update one color slot in the live color scheme.
///
/// `slot`:  one of the `MID_COLOR_SLOT_*` constants.
/// `r,g,b`: RGB components. Pass (0,0,0) with `use_none=1` to remove color.
/// `use_none`: if non-zero, sets the slot to `Color::None` (terminal default).
///
/// Example — make WARN bright orange:
/// ```c
/// mid_log_update_color_c(MID_COLOR_SLOT_WARN, 255, 165, 0, 0);
/// ```
#[no_mangle]
pub extern "C" fn mid_log_update_color_c(
    slot:     u8,
    r:        u8,
    g:        u8,
    b:        u8,
    use_none: u8,
) {
    let new_color = if use_none != 0 {
        Color::None
    } else {
        Color::Rgb(r, g, b)
    };

    color::update_color_scheme(|s| match slot {
        0  => s.trace     = new_color.clone(),
        1  => s.info      = new_color.clone(),
        2  => s.warn      = new_color.clone(),
        3  => s.error     = new_color.clone(),
        4  => s.fatal     = new_color.clone(),
        5  => s.tier_low  = new_color.clone(),
        6  => s.tier_mid  = new_color.clone(),
        7  => s.tier_high = new_color.clone(),
        8  => s.timestamp = new_color.clone(),
        9  => s.source    = new_color.clone(),
        10 => s.module    = new_color.clone(),
        11 => s.thread    = new_color.clone(),
        12 => s.frame     = new_color.clone(),
        13 => s.message   = new_color.clone(),
        _  => {}
    });
}

// ── Format flags ──────────────────────────────────────────────────────────────

/// Set all format flags at once.
///
/// Each parameter: 0 = hide, non-zero = show.
#[no_mangle]
pub extern "C" fn mid_log_set_format_flags(
    show_timestamp:  u8,
    show_source_loc: u8,
    show_module:     u8,
    show_thread:     u8,
    show_frame:      u8,
) {
    set_format(&FormatConfig {
        show_timestamp:  show_timestamp  != 0,
        show_source_loc: show_source_loc != 0,
        show_module:     show_module     != 0,
        show_thread:     show_thread     != 0,
        show_frame:      show_frame      != 0,
    });
}

// ── Frame counter ─────────────────────────────────────────────────────────────

/// Set the current game frame number.
/// Call once at the top of each game tick.
#[no_mangle]
pub extern "C" fn mid_log_set_frame(n: u64) {
    frame::set_frame(n);
}

/// Returns the current game frame number.
#[no_mangle]
pub extern "C" fn mid_log_get_frame() -> u64 {
    frame::current_frame()
}

// ── Rate limiting ─────────────────────────────────────────────────────────────

/// Configure rate limiting.
///
/// `enabled`:        0 = disable, 1 = enable.
/// `window_ms`:      suppression window in milliseconds. Default: 1000.
/// `max_per_window`: max identical entries per window before suppression. Default: 5.
#[no_mangle]
pub extern "C" fn mid_log_set_rate_limit(
    enabled:        u8,
    window_ms:      u32,
    max_per_window: u32,
) {
    set_rate_limit_config(RateLimitConfig {
        enabled:        enabled != 0,
        window:         Duration::from_millis(window_ms as u64),
        max_per_window: max_per_window.max(1),
    });
}

// ── Console buffer ────────────────────────────────────────────────────────────

/// Initialise the in-game console ring buffer.
///
/// Must be called before `mid_log_init*()` to capture all entries.
/// `capacity`: number of entries retained (min 8). Default: 512.
#[no_mangle]
pub extern "C" fn mid_log_console_init(capacity: u32) {
    crate::console_buffer::init_console_buffer(capacity as usize);
}

/// Returns the number of entries currently in the console buffer.
#[no_mangle]
pub extern "C" fn mid_log_console_count() -> u32 {
    crate::console_buffer::snapshot().len() as u32
}

// ── Flush / shutdown ──────────────────────────────────────────────────────────

/// Flush all queued entries without stopping the logger.
#[no_mangle]
pub extern "C" fn mid_log_flush() {
    MidLogger::flush();
}

/// Flush and stop the IO thread. Call once at engine shutdown.
#[no_mangle]
pub extern "C" fn mid_log_shutdown() {
    MidLogger::shutdown();
}

// ── Logging ───────────────────────────────────────────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn mid_log_trace_c(tier: u8, msg: *const c_char) {
    log_c(LogLevel::Trace, tier, msg);
}

#[no_mangle]
pub unsafe extern "C" fn mid_log_info_c(tier: u8, msg: *const c_char) {
    log_c(LogLevel::Info, tier, msg);
}

#[no_mangle]
pub unsafe extern "C" fn mid_log_warn_c(tier: u8, msg: *const c_char) {
    log_c(LogLevel::Warn, tier, msg);
}

#[no_mangle]
pub unsafe extern "C" fn mid_log_error_c(tier: u8, msg: *const c_char) {
    log_c(LogLevel::Error, tier, msg);
}

/// Log at FATAL. Calls `mid_log_shutdown()` automatically.
#[no_mangle]
pub unsafe extern "C" fn mid_log_fatal_c(tier: u8, msg: *const c_char) {
    log_c(LogLevel::Fatal, tier, msg);
    MidLogger::shutdown();
}

unsafe fn log_c(level: LogLevel, tier: u8, msg: *const c_char) {
    if !filter::is_enabled(level) { return; }
    if msg.is_null() { return; }
    let message = CStr::from_ptr(msg)
        .to_str()
        .unwrap_or("<invalid utf-8>")
        .to_owned();
    if let Some(logger) = MidLogger::get() {
        logger.log(level, Tier::from_u8(tier), message, "<ffi>", 0, "<ffi>");
    }
}
