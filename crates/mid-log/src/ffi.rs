// crates/mid-log/src/ffi.rs

//! C-compatible FFI exports — the C face of mid-log.
//!
//! Function names are prefixed `mid_log_` with `_c` suffix on log calls to
//! distinguish them from the Rust macro API at the call site.
//!
//! C consumers include: `headers/mid_log.h`
//!
//! ## Thread safety
//! All exported functions are thread-safe. The channel sender is cloned
//! internally per call — no Mutex, no contention.
//!
//! ## Source location in FFI calls
//! C callers cannot pass Rust `&'static str` easily, so FFI log calls
//! use `"<ffi>"` as the file/module and `0` as the line. If you need
//! per-call C source location, use the `MID_LOG_INFO(msg)` macro in
//! the header (future work — uses `__FILE__` / `__LINE__` and calls
//! `mid_log_info_loc_c`).

use std::ffi::CStr;
use std::os::raw::c_char;
use std::path::PathBuf;
use crate::level::{LogLevel, Tier};
use crate::filter;
use crate::logger::MidLogger;

// ── Lifecycle ─────────────────────────────────────────────────────────────────

/// Initialise the logger (stderr only). Call once at engine startup.
///
/// Returns 1 on success, 0 if already initialised.
#[no_mangle]
pub extern "C" fn mid_log_init() -> u8 {
    if MidLogger::init() { 1 } else { 0 }
}

/// Initialise the logger with a file tee.
///
/// `path`: null-terminated UTF-8 file path. Pass NULL to use stderr only
/// (equivalent to `mid_log_init()`).
///
/// Returns 1 on success, 0 if already initialised or path is invalid UTF-8.
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

/// Set the minimum log level.
///
/// `level`: 0=TRACE, 1=INFO, 2=WARN, 3=ERROR, 4=FATAL.
/// Entries below this level are discarded before string formatting.
#[no_mangle]
pub extern "C" fn mid_log_set_min_level(level: u8) {
    filter::set_min_level(LogLevel::from_u8(level));
}

/// Returns the current minimum log level as a u8.
#[no_mangle]
pub extern "C" fn mid_log_get_min_level() -> u8 {
    filter::get_min_level() as u8
}

/// Flush all queued entries without stopping the logger.
///
/// Blocks until the IO thread has written all pending entries.
/// Safe to call from any thread.
#[no_mangle]
pub extern "C" fn mid_log_flush() {
    MidLogger::flush();
}

/// Flush remaining entries and stop the IO thread.
///
/// Call at engine shutdown. After this, log calls are silently dropped.
#[no_mangle]
pub extern "C" fn mid_log_shutdown() {
    MidLogger::shutdown();
}

// ── Logging ───────────────────────────────────────────────────────────────────
//
// tier: 0 = LOW (engine internals), 1 = MID (engine-adjacent), 2+ = HIGH (gameplay)

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

/// Log at FATAL level. Calls `mid_log_shutdown()` automatically.
#[no_mangle]
pub unsafe extern "C" fn mid_log_fatal_c(tier: u8, msg: *const c_char) {
    log_c(LogLevel::Fatal, tier, msg);
    MidLogger::shutdown();
}

// ── Internal helper ───────────────────────────────────────────────────────────

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
