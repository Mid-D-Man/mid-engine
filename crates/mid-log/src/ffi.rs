// crates/mid-log/src/ffi.rs

//! C-compatible FFI exports — the C face of mid-log.

use std::borrow::Cow;
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

#[no_mangle]
pub extern "C" fn mid_log_init() -> u8 {
    if MidLogger::init() { 1 } else { 0 }
}

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

#[no_mangle]
pub unsafe extern "C" fn mid_log_init_full_c(
    log_file:        *const c_char,
    min_level:       u8,
    show_timestamp:  u8,
    show_source_loc: u8,
    show_module:     u8,
    show_thread:     u8,
    show_frame:      u8,
    colors:          i8,
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

    match colors {
        0  => color::set_colors_enabled(false),
        1  => color::set_colors_enabled(true),
        _  => {}
    }

    if result { 1 } else { 0 }
}

// ── Level filter ──────────────────────────────────────────────────────────────

#[no_mangle]
pub extern "C" fn mid_log_set_min_level(level: u8) {
    filter::set_min_level(LogLevel::from_u8(level));
}

#[no_mangle]
pub extern "C" fn mid_log_get_min_level() -> u8 {
    filter::get_min_level() as u8
}

// ── Colors ────────────────────────────────────────────────────────────────────

#[no_mangle]
pub extern "C" fn mid_log_set_colors(enabled: u8) {
    color::set_colors_enabled(enabled != 0);
}

#[no_mangle]
pub extern "C" fn mid_log_get_colors() -> u8 {
    if color::is_colors_enabled() { 1 } else { 0 }
}

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

#[no_mangle]
pub extern "C" fn mid_log_set_frame(n: u64) {
    frame::set_frame(n);
}

#[no_mangle]
pub extern "C" fn mid_log_get_frame() -> u64 {
    frame::current_frame()
}

// ── Rate limiting ─────────────────────────────────────────────────────────────

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

#[no_mangle]
pub extern "C" fn mid_log_console_init(capacity: u32) {
    crate::console_buffer::init_console_buffer(capacity as usize);
}

#[no_mangle]
pub extern "C" fn mid_log_console_count() -> u32 {
    crate::console_buffer::snapshot().len() as u32
}

// ── Flush / shutdown ──────────────────────────────────────────────────────────

#[no_mangle]
pub extern "C" fn mid_log_flush() {
    MidLogger::flush();
}

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

#[no_mangle]
pub unsafe extern "C" fn mid_log_fatal_c(tier: u8, msg: *const c_char) {
    log_c(LogLevel::Fatal, tier, msg);
    MidLogger::shutdown();
}

/// Shared implementation for all C logging entry points.
///
/// The C side always produces a fully-formatted string (via snprintf in
/// the header macros), so FFI entries always take the printf path —
/// `Cow::Owned` wrapping the String decoded from the C pointer.
unsafe fn log_c(level: LogLevel, tier: u8, msg: *const c_char) {
    if !filter::is_enabled(level) { return; }
    if msg.is_null() { return; }
    let message = CStr::from_ptr(msg)
        .to_str()
        .unwrap_or("<invalid utf-8>")
        .to_owned();
    if let Some(logger) = MidLogger::get() {
        // FFI always produces a heap-allocated String — wrap it in Cow::Owned.
        // The KV path is not exposed via C FFI; C callers use snprintf and
        // pass the result as a null-terminated string.
        logger.log(level, Tier::from_u8(tier), Cow::Owned(message), "<ffi>", 0, "<ffi>");
    }
}
