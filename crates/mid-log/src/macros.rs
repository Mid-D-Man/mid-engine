// crates/mid-log/src/macros.rs

//! Rust-side logging macros — printf and structured KV APIs.
//!
//! ## Choosing between printf and KV
//!
//! | Use case                              | Macro family    |
//! |---------------------------------------|-----------------|
//! | Dynamic string, computed values, FFI  | `mid_info!`     |
//! | Scalar fields: entity IDs, positions  | `mid_kvinfo!`   |
//! | Static message only                   | `mid_kvinfo!`   |
//!
//! ## Printf API
//!
//! Calls `format!()` on the calling thread. Straightforward ergonomics,
//! supports all format specifiers. Cost: ~250–500 ns for complex formats.
//!
//! ```rust,no_run
//! # use mid_log::{mid_info, level::Tier};
//! mid_info!(Tier::High, "entity {} pos ({:.2}, {:.2})", id, x, y);
//! ```
//!
//! ## KV API
//!
//! Sends typed scalar values without `format!()`. The IO thread formats
//! them. Cost: ~45–65 ns for 3 KV pairs — ~200–450 ns faster than printf
//! for the same data when floats are involved.
//!
//! ```rust,no_run
//! # use mid_log::{mid_kvinfo, level::Tier};
//! mid_kvinfo!(Tier::High, "entity update"; "id" => id, "x" => x, "y" => y);
//!
//! // Static message only — zero allocation, zero format!():
//! mid_kvinfo!(Tier::High, "physics tick complete");
//! ```

use std::borrow::Cow;

// ═══════════════════════════════════════════════════════════════════════════
//  Printf API — existing macros, unchanged behaviour
// ═══════════════════════════════════════════════════════════════════════════

/// Log at TRACE level (printf style).
#[macro_export]
macro_rules! mid_trace {
    ($tier:expr, $($arg:tt)*) => {{
        if $crate::filter::is_enabled($crate::level::LogLevel::Trace) {
            if let Some(logger) = $crate::logger::MidLogger::get() {
                logger.log(
                    $crate::level::LogLevel::Trace,
                    $tier,
                    ::std::borrow::Cow::Owned(format!($($arg)*)),
                    file!(), line!(), module_path!(),
                );
            }
        }
    }};
}

/// Log at INFO level (printf style).
#[macro_export]
macro_rules! mid_info {
    ($tier:expr, $($arg:tt)*) => {{
        if $crate::filter::is_enabled($crate::level::LogLevel::Info) {
            if let Some(logger) = $crate::logger::MidLogger::get() {
                logger.log(
                    $crate::level::LogLevel::Info,
                    $tier,
                    ::std::borrow::Cow::Owned(format!($($arg)*)),
                    file!(), line!(), module_path!(),
                );
            }
        }
    }};
}

/// Log at WARN level (printf style).
#[macro_export]
macro_rules! mid_warn {
    ($tier:expr, $($arg:tt)*) => {{
        if $crate::filter::is_enabled($crate::level::LogLevel::Warn) {
            if let Some(logger) = $crate::logger::MidLogger::get() {
                logger.log(
                    $crate::level::LogLevel::Warn,
                    $tier,
                    ::std::borrow::Cow::Owned(format!($($arg)*)),
                    file!(), line!(), module_path!(),
                );
            }
        }
    }};
}

/// Log at ERROR level (printf style). Non-fatal.
#[macro_export]
macro_rules! mid_error {
    ($tier:expr, $($arg:tt)*) => {{
        if $crate::filter::is_enabled($crate::level::LogLevel::Error) {
            if let Some(logger) = $crate::logger::MidLogger::get() {
                logger.log(
                    $crate::level::LogLevel::Error,
                    $tier,
                    ::std::borrow::Cow::Owned(format!($($arg)*)),
                    file!(), line!(), module_path!(),
                );
            }
        }
    }};
}

/// Log at FATAL level (printf style). Flushes and shuts down the logger.
#[macro_export]
macro_rules! mid_fatal {
    ($tier:expr, $($arg:tt)*) => {{
        if let Some(logger) = $crate::logger::MidLogger::get() {
            logger.log(
                $crate::level::LogLevel::Fatal,
                $tier,
                ::std::borrow::Cow::Owned(format!($($arg)*)),
                file!(), line!(), module_path!(),
            );
        }
        $crate::logger::MidLogger::shutdown();
    }};
}

// ═══════════════════════════════════════════════════════════════════════════
//  KV API — structured logging, no format!() on the calling thread
// ═══════════════════════════════════════════════════════════════════════════
//
// Syntax:
//   mid_kvinfo!(Tier::High, "static message");
//   mid_kvinfo!(Tier::High, "static message"; "key1" => val1, "key2" => val2);
//
// Keys must be string literals. Values can be any type implementing IntoKvValue
// (bool, i8–i64, u8–u64, f32, f64, &'static str).
//
// The calling thread pays:
//   - 1 filter check (AtomicU8 load)
//   - 1 OnceLock probe (AtomicUsize load)
//   - 1 Vec allocation for the KV pairs (zero allocation if no KVs)
//   - N pair constructions (tag + scalar copy per KV)
//   - 1 channel send
// No format!(), no float-to-string conversion.

/// Log a structured KV entry at TRACE level.
///
/// ```rust,no_run
/// # use mid_log::{mid_kvtrace, level::Tier};
/// mid_kvtrace!(Tier::Low, "physics step"; "dt" => 0.016f32, "bodies" => 1024u32);
/// ```
#[macro_export]
macro_rules! mid_kvtrace {
    ($tier:expr, $msg:literal $(; $($key:literal => $val:expr),+ $(,)?)?) => {{
        if $crate::filter::is_enabled($crate::level::LogLevel::Trace) {
            if let Some(logger) = $crate::logger::MidLogger::get() {
                #[allow(unused_mut)]
                let mut _kvs: Vec<$crate::kv::KvPair> = Vec::new();
                $($( _kvs.push(($key, $crate::kv::IntoKvValue::into_kv_value($val))); )+)?
                logger.log_kv(
                    $crate::level::LogLevel::Trace,
                    $tier,
                    $msg,
                    _kvs,
                    file!(), line!(), module_path!(),
                );
            }
        }
    }};
}

/// Log a structured KV entry at INFO level.
///
/// ```rust,no_run
/// # use mid_log::{mid_kvinfo, level::Tier};
/// // With KVs — no format!(), typed scalar values:
/// mid_kvinfo!(Tier::High, "player spawned"; "id" => 42u32, "x" => 1.0f32, "y" => 2.5f32);
///
/// // Static message only — truly zero allocation:
/// mid_kvinfo!(Tier::High, "frame complete");
/// ```
#[macro_export]
macro_rules! mid_kvinfo {
    ($tier:expr, $msg:literal $(; $($key:literal => $val:expr),+ $(,)?)?) => {{
        if $crate::filter::is_enabled($crate::level::LogLevel::Info) {
            if let Some(logger) = $crate::logger::MidLogger::get() {
                #[allow(unused_mut)]
                let mut _kvs: Vec<$crate::kv::KvPair> = Vec::new();
                $($( _kvs.push(($key, $crate::kv::IntoKvValue::into_kv_value($val))); )+)?
                logger.log_kv(
                    $crate::level::LogLevel::Info,
                    $tier,
                    $msg,
                    _kvs,
                    file!(), line!(), module_path!(),
                );
            }
        }
    }};
}

/// Log a structured KV entry at WARN level.
#[macro_export]
macro_rules! mid_kvwarn {
    ($tier:expr, $msg:literal $(; $($key:literal => $val:expr),+ $(,)?)?) => {{
        if $crate::filter::is_enabled($crate::level::LogLevel::Warn) {
            if let Some(logger) = $crate::logger::MidLogger::get() {
                #[allow(unused_mut)]
                let mut _kvs: Vec<$crate::kv::KvPair> = Vec::new();
                $($( _kvs.push(($key, $crate::kv::IntoKvValue::into_kv_value($val))); )+)?
                logger.log_kv(
                    $crate::level::LogLevel::Warn,
                    $tier,
                    $msg,
                    _kvs,
                    file!(), line!(), module_path!(),
                );
            }
        }
    }};
}

/// Log a structured KV entry at ERROR level. Non-fatal.
#[macro_export]
macro_rules! mid_kverror {
    ($tier:expr, $msg:literal $(; $($key:literal => $val:expr),+ $(,)?)?) => {{
        if $crate::filter::is_enabled($crate::level::LogLevel::Error) {
            if let Some(logger) = $crate::logger::MidLogger::get() {
                #[allow(unused_mut)]
                let mut _kvs: Vec<$crate::kv::KvPair> = Vec::new();
                $($( _kvs.push(($key, $crate::kv::IntoKvValue::into_kv_value($val))); )+)?
                logger.log_kv(
                    $crate::level::LogLevel::Error,
                    $tier,
                    $msg,
                    _kvs,
                    file!(), line!(), module_path!(),
                );
            }
        }
    }};
}
