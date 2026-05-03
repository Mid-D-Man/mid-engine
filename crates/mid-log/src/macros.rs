// crates/mid-log/src/macros.rs

//! Rust-side logging macros — the fast path for Mid Engine and Ubel.
//!
//! ## Key design decisions
//!
//! ### Level filter before format!()
//! Every macro checks `filter::is_enabled(level)` before calling `format!()`.
//! If the level is disabled, the macro expands to a single atomic load + branch —
//! no string allocation, no function call. This matches tracing's behaviour on
//! disabled callsites.
//!
//! ### Source location
//! `file!()`, `line!()`, `module_path!()` are captured at the macro expansion
//! site, giving the caller's location (not this file's location). They are
//! `&'static str` / `u32` — zero-cost.
//!
//! ### FFI boundary
//! These macros cannot cross the FFI boundary. For C/C++/Unity use `ffi.rs`.

/// Log at TRACE level.
///
/// # Example
/// ```rust,no_run
/// # use mid_log::{mid_trace, level::Tier};
/// # mid_log::logger::MidLogger::init();
/// mid_trace!(Tier::Low, "entity {} pos ({:.2}, {:.2})", 42, 1.0, 2.5);
/// ```
#[macro_export]
macro_rules! mid_trace {
    ($tier:expr, $($arg:tt)*) => {{
        if $crate::filter::is_enabled($crate::level::LogLevel::Trace) {
            if let Some(logger) = $crate::logger::MidLogger::get() {
                logger.log(
                    $crate::level::LogLevel::Trace,
                    $tier,
                    format!($($arg)*),
                    file!(),
                    line!(),
                    module_path!(),
                );
            }
        }
    }};
}

/// Log at INFO level.
#[macro_export]
macro_rules! mid_info {
    ($tier:expr, $($arg:tt)*) => {{
        if $crate::filter::is_enabled($crate::level::LogLevel::Info) {
            if let Some(logger) = $crate::logger::MidLogger::get() {
                logger.log(
                    $crate::level::LogLevel::Info,
                    $tier,
                    format!($($arg)*),
                    file!(),
                    line!(),
                    module_path!(),
                );
            }
        }
    }};
}

/// Log at WARN level.
#[macro_export]
macro_rules! mid_warn {
    ($tier:expr, $($arg:tt)*) => {{
        if $crate::filter::is_enabled($crate::level::LogLevel::Warn) {
            if let Some(logger) = $crate::logger::MidLogger::get() {
                logger.log(
                    $crate::level::LogLevel::Warn,
                    $tier,
                    format!($($arg)*),
                    file!(),
                    line!(),
                    module_path!(),
                );
            }
        }
    }};
}

/// Log at ERROR level. Non-fatal.
#[macro_export]
macro_rules! mid_error {
    ($tier:expr, $($arg:tt)*) => {{
        if $crate::filter::is_enabled($crate::level::LogLevel::Error) {
            if let Some(logger) = $crate::logger::MidLogger::get() {
                logger.log(
                    $crate::level::LogLevel::Error,
                    $tier,
                    format!($($arg)*),
                    file!(),
                    line!(),
                    module_path!(),
                );
            }
        }
    }};
}

/// Log at FATAL level. Flushes and shuts down the logger.
///
/// After this macro the logger is stopped. Do not log after a fatal.
#[macro_export]
macro_rules! mid_fatal {
    ($tier:expr, $($arg:tt)*) => {{
        if let Some(logger) = $crate::logger::MidLogger::get() {
            logger.log(
                $crate::level::LogLevel::Fatal,
                $tier,
                format!($($arg)*),
                file!(),
                line!(),
                module_path!(),
            );
        }
        $crate::logger::MidLogger::shutdown();
    }};
}
