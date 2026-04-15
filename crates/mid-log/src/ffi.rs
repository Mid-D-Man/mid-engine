//! C-compatible FFI exports — the C face of mid-log.
//!
//! Function names are prefixed `mid_log_` and use `_c` suffix to
//! distinguish them from the Rust macro API at the call site.
//!
//! C consumers include this header: headers/mid_log.h

use std::ffi::CStr;
use std::os::raw::c_char;
use crate::level::{LogLevel, Tier};
use crate::logger::MidLogger;

// ── Lifecycle ─────────────────────────────────────────────────────────────────

/// Initialise the logger. Call once at engine startup.
/// Returns 1 on success, 0 if already initialised or if init failed.
#[no_mangle]
pub extern "C" fn mid_log_init() -> u8 {
    if MidLogger::init() { 1 } else { 0 }
}

/// Flush remaining entries and stop the IO thread.
/// Call at engine shutdown.
#[no_mangle]
pub extern "C" fn mid_log_shutdown() {
    MidLogger::shutdown();
}

// ── Logging ───────────────────────────────────────────────────────────────────

/// `tier`: 0 = LOW (engine internals), 1 = HIGH (gameplay logic)

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
    // Fatal always flushes.
    MidLogger::shutdown();
}

// ── Internal helper ───────────────────────────────────────────────────────────

unsafe fn log_c(level: LogLevel, tier: u8, msg: *const c_char) {
    if msg.is_null() { return; }
    let message = unsafe { CStr::from_ptr(msg) }
        .to_str()
        .unwrap_or("<invalid utf-8>")
        .to_owned();
    if let Some(logger) = MidLogger::get() {
        logger.log(level, Tier::from_u8(tier), message);
    }
      }
