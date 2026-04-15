//! Rust-side logging macros — the fast path for Mid Engine and Ubel.
//!
//! These cannot cross the FFI boundary. For C/C++/Unity use ffi.rs.

/// Log at TRACE level (per-frame detail — keep disabled in prod).
#[macro_export]
macro_rules! mid_trace {
    ($tier:expr, $($arg:tt)*) => {
        if let Some(logger) = $crate::logger::MidLogger::get() {
            logger.log(
                $crate::level::LogLevel::Trace,
                $tier,
                format!($($arg)*),
            );
        }
    };
}

/// Log at INFO level.
#[macro_export]
macro_rules! mid_info {
    ($tier:expr, $($arg:tt)*) => {
        if let Some(logger) = $crate::logger::MidLogger::get() {
            logger.log(
                $crate::level::LogLevel::Info,
                $tier,
                format!($($arg)*),
            );
        }
    };
}

/// Log at WARN level.
#[macro_export]
macro_rules! mid_warn {
    ($tier:expr, $($arg:tt)*) => {
        if let Some(logger) = $crate::logger::MidLogger::get() {
            logger.log(
                $crate::level::LogLevel::Warn,
                $tier,
                format!($($arg)*),
            );
        }
    };
}

/// Log at ERROR level. Non-fatal.
#[macro_export]
macro_rules! mid_error {
    ($tier:expr, $($arg:tt)*) => {
        if let Some(logger) = $crate::logger::MidLogger::get() {
            logger.log(
                $crate::level::LogLevel::Error,
                $tier,
                format!($($arg)*),
            );
        }
    };
}

/// Log at FATAL level. Flushes and shuts down the logger.
#[macro_export]
macro_rules! mid_fatal {
    ($tier:expr, $($arg:tt)*) => {
        if let Some(logger) = $crate::logger::MidLogger::get() {
            logger.log(
                $crate::level::LogLevel::Fatal,
                $tier,
                format!($($arg)*),
            );
        }
        $crate::logger::MidLogger::shutdown();
    };
              }
