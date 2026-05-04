// crates/mid-log/src/assert.rs

//! Game engine assertion macros — logger-aware, flush-safe.
//!
//! All hard assertions log their failure at FATAL level and call
//! `MidLogger::flush()` before panicking. This guarantees the entry
//! reaches the file sink even when the process is about to crash.
//!
//! ## Quick reference
//!
//! | Macro                              | Panics? | Build      | Notes                    |
//! |------------------------------------|---------|------------|--------------------------|
//! | `mid_assert!(c)`                   | Yes     | All        | Basic condition check    |
//! | `mid_assert_eq!(a, b)`             | Yes     | All        | Shows left/right values  |
//! | `mid_assert_ne!(a, b)`             | Yes     | All        | Shows both values        |
//! | `mid_assert_approx_eq!(a, b, eps)` | Yes     | All        | Float proximity check    |
//! | `mid_unreachable!()`               | Yes     | All        | Marks impossible paths   |
//! | `mid_soft_assert!(c)`              | No      | All        | Returns `bool`           |
//! | `mid_soft_assert_eq!(a, b)`        | No      | All        | Returns `bool`           |
//! | `mid_soft_assert_ne!(a, b)`        | No      | All        | Returns `bool`           |
//! | `mid_debug_assert!(c)`             | No      | Debug only | Zero cost in release     |
//! | `mid_debug_assert_eq!(a, b)`       | No      | Debug only | Zero cost in release     |
//! | `mid_debug_assert_ne!(a, b)`       | No      | Debug only | Zero cost in release     |
//!
//! ## Example output
//!
//! ```text
//! 12:34:56.789 [FATAL][LOW ] ASSERT_EQ FAILED: `vel.length() == 0.0`
//!   Left:  1247.33
//!   Right: 0.0
//!   Hint:  entity was not stopped before despawn
//!   (physics::step  src/physics.rs:142)
//! ```

// ═══════════════════════════════════════════════════════════════════════════════
//  Hard assertions — log FATAL → flush → panic
// ═══════════════════════════════════════════════════════════════════════════════

/// Hard assertion. Logs at FATAL, flushes the logger, then panics.
///
/// Equivalent to `assert!()` but the failure is guaranteed to reach the
/// log file before the panic unwind.
///
/// # Forms
/// ```rust,no_run
/// # use mid_log::mid_assert;
/// # let health = 0u32;
/// mid_assert!(health > 0);
/// mid_assert!(health > 0, "entity died unexpectedly");
/// mid_assert!(health > 0, "entity {} has {} hp", 42u32, health);
/// ```
#[macro_export]
macro_rules! mid_assert {
    ($cond:expr) => {
        $crate::mid_assert!($cond, "assertion failed");
    };
    ($cond:expr, $($msg:tt)+) => {{
        if !($cond) {
            $crate::mid_fatal!(
                $crate::level::Tier::Low,
                "ASSERT FAILED: `{}`\n  Message: {}",
                stringify!($cond),
                format!($($msg)+),
            );
            $crate::logger::MidLogger::flush();
            panic!(
                "mid_assert failed: `{}`  ({}:{})",
                stringify!($cond), file!(), line!(),
            );
        }
    }};
}

/// Hard equality assertion. Logs both values then panics on mismatch.
///
/// Values are formatted with `{:?}` (Debug). Both must implement `PartialEq + Debug`.
///
/// # Forms
/// ```rust,no_run
/// # use mid_log::mid_assert_eq;
/// # let frame = 1u64; let expected = 1u64;
/// mid_assert_eq!(frame, expected);
/// mid_assert_eq!(frame, expected, "frame counter desync");
/// mid_assert_eq!(frame, expected, "entity {}: frame {} != {}", 99u32, frame, expected);
/// ```
#[macro_export]
macro_rules! mid_assert_eq {
    ($left:expr, $right:expr) => {
        $crate::mid_assert_eq!($left, $right, "values are not equal");
    };
    ($left:expr, $right:expr, $($msg:tt)+) => {{
        let l = &($left);
        let r = &($right);
        if l != r {
            $crate::mid_fatal!(
                $crate::level::Tier::Low,
                "ASSERT_EQ FAILED: `{} == {}`\n  Left:  {:?}\n  Right: {:?}\n  Hint:  {}",
                stringify!($left), stringify!($right),
                l, r,
                format!($($msg)+),
            );
            $crate::logger::MidLogger::flush();
            panic!(
                "mid_assert_eq failed: `{} == {}`  ({}:{})",
                stringify!($left), stringify!($right), file!(), line!(),
            );
        }
    }};
}

/// Hard inequality assertion. Logs both values then panics when `left == right`.
///
/// # Forms
/// ```rust,no_run
/// # use mid_log::mid_assert_ne;
/// # let id_a = 1u32; let id_b = 2u32;
/// mid_assert_ne!(id_a, id_b);
/// mid_assert_ne!(id_a, id_b, "entity IDs must not alias");
/// ```
#[macro_export]
macro_rules! mid_assert_ne {
    ($left:expr, $right:expr) => {
        $crate::mid_assert_ne!($left, $right, "values are unexpectedly equal");
    };
    ($left:expr, $right:expr, $($msg:tt)+) => {{
        let l = &($left);
        let r = &($right);
        if l == r {
            $crate::mid_fatal!(
                $crate::level::Tier::Low,
                "ASSERT_NE FAILED: `{} != {}`\n  Both:  {:?}\n  Hint:  {}",
                stringify!($left), stringify!($right),
                l,
                format!($($msg)+),
            );
            $crate::logger::MidLogger::flush();
            panic!(
                "mid_assert_ne failed: `{} != {}`  ({}:{})",
                stringify!($left), stringify!($right), file!(), line!(),
            );
        }
    }};
}

/// Float approximate-equality assertion. Panics when `|left - right| >= epsilon`.
///
/// Works with any type that supports subtraction and `PartialOrd` — typically
/// `f32` or `f64`. Logs the actual difference alongside both values and epsilon.
///
/// # Example
/// ```rust,no_run
/// # use mid_log::mid_assert_approx_eq;
/// let result = 0.999_999_9_f32;
/// mid_assert_approx_eq!(result, 1.0_f32, 1e-5_f32, "normalized length");
/// ```
#[macro_export]
macro_rules! mid_assert_approx_eq {
    ($left:expr, $right:expr, $eps:expr) => {
        $crate::mid_assert_approx_eq!($left, $right, $eps, "values not approximately equal");
    };
    ($left:expr, $right:expr, $eps:expr, $($msg:tt)+) => {{
        let l   = $left;
        let r   = $right;
        let eps = $eps;
        // Compute |l - r| without requiring Abs — works for f32 and f64.
        let diff = if l > r { l - r } else { r - l };
        if !(diff < eps) {
            $crate::mid_fatal!(
                $crate::level::Tier::Low,
                "ASSERT_APPROX_EQ FAILED: `|{} - {}| < {}`\n  \
                 Left:  {}\n  Right: {}\n  Diff:  {}\n  Eps:   {}\n  Hint:  {}",
                stringify!($left), stringify!($right), stringify!($eps),
                l, r, diff, eps,
                format!($($msg)+),
            );
            $crate::logger::MidLogger::flush();
            panic!(
                "mid_assert_approx_eq failed: diff={} >= eps={}  ({}:{})",
                diff, eps, file!(), line!(),
            );
        }
    }};
}

/// Logger-aware `unreachable!()`. Logs FATAL with full context then panics.
///
/// Use to mark code paths that must never be reached in correct execution.
///
/// # Example
/// ```rust,no_run
/// # use mid_log::mid_unreachable;
/// # let state = 99u32;
/// match state {
///     0 => { /* ... */ }
///     1 => { /* ... */ }
///     _ => mid_unreachable!("unhandled state: {}", state),
/// }
/// ```
#[macro_export]
macro_rules! mid_unreachable {
    () => {
        $crate::mid_unreachable!("entered unreachable code");
    };
    ($($msg:tt)+) => {{
        let msg = format!($($msg)+);
        $crate::mid_fatal!(
            $crate::level::Tier::Low,
            "UNREACHABLE: {}",
            msg,
        );
        $crate::logger::MidLogger::flush();
        panic!("mid_unreachable: {}  ({}:{})", msg, file!(), line!());
    }};
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Soft assertions — log ERROR, return bool, do NOT panic
// ═══════════════════════════════════════════════════════════════════════════════

/// Non-fatal assertion. Logs at ERROR and returns `bool` — does NOT panic.
///
/// Returns `true` if the condition passed, `false` if it failed.
/// Use when the system can continue despite the invariant being broken.
///
/// # Example
/// ```rust,no_run
/// # use mid_log::mid_soft_assert;
/// # let queue_len = 5000usize;
/// if !mid_soft_assert!(queue_len < 4096, "queue unusually large: {}", queue_len) {
///     // trim instead of crashing
/// }
/// ```
#[macro_export]
macro_rules! mid_soft_assert {
    ($cond:expr) => {
        $crate::mid_soft_assert!($cond, "soft assertion failed")
    };
    ($cond:expr, $($msg:tt)+) => {{
        let passed = $cond;
        if !passed {
            $crate::mid_error!(
                $crate::level::Tier::Low,
                "SOFT_ASSERT FAILED: `{}`\n  Message: {}",
                stringify!($cond),
                format!($($msg)+),
            );
        }
        passed
    }};
}

/// Non-fatal equality check. Logs at ERROR when `left != right`. Returns `bool`.
#[macro_export]
macro_rules! mid_soft_assert_eq {
    ($left:expr, $right:expr) => {
        $crate::mid_soft_assert_eq!($left, $right, "values are not equal")
    };
    ($left:expr, $right:expr, $($msg:tt)+) => {{
        let l = &($left);
        let r = &($right);
        let passed = l == r;
        if !passed {
            $crate::mid_error!(
                $crate::level::Tier::Low,
                "SOFT_ASSERT_EQ FAILED: `{} == {}`\n  Left:  {:?}\n  Right: {:?}\n  Hint:  {}",
                stringify!($left), stringify!($right),
                l, r,
                format!($($msg)+),
            );
        }
        passed
    }};
}

/// Non-fatal inequality check. Logs at ERROR when `left == right`. Returns `bool`.
#[macro_export]
macro_rules! mid_soft_assert_ne {
    ($left:expr, $right:expr) => {
        $crate::mid_soft_assert_ne!($left, $right, "values are unexpectedly equal")
    };
    ($left:expr, $right:expr, $($msg:tt)+) => {{
        let l = &($left);
        let r = &($right);
        let passed = l != r;
        if !passed {
            $crate::mid_error!(
                $crate::level::Tier::Low,
                "SOFT_ASSERT_NE FAILED: `{} != {}`\n  Both:  {:?}\n  Hint:  {}",
                stringify!($left), stringify!($right),
                l,
                format!($($msg)+),
            );
        }
        passed
    }};
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Debug-only assertions — #[cfg(debug_assertions)] compiled out in release
// ═══════════════════════════════════════════════════════════════════════════════

/// Debug-only assertion. **Zero cost in release builds.**
///
/// In debug builds: logs at ERROR and continues (does NOT panic).
/// In release builds: the entire macro expands to nothing.
///
/// Use for expensive invariant checks that are too slow for production.
///
/// # Example
/// ```rust,no_run
/// # use mid_log::mid_debug_assert;
/// # let positions: Vec<f32> = vec![];
/// // O(n) check — only runs in debug:
/// mid_debug_assert!(
///     positions.iter().all(|p| p.is_finite()),
///     "NaN detected in position buffer (n={})", positions.len()
/// );
/// ```
#[macro_export]
macro_rules! mid_debug_assert {
    ($cond:expr) => {
        $crate::mid_debug_assert!($cond, "debug assertion failed");
    };
    ($cond:expr, $($msg:tt)+) => {{
        #[cfg(debug_assertions)]
        {
            if !($cond) {
                $crate::mid_error!(
                    $crate::level::Tier::Low,
                    "DEBUG_ASSERT FAILED: `{}`\n  Message: {}",
                    stringify!($cond),
                    format!($($msg)+),
                );
            }
        }
    }};
}

/// Debug-only equality check. **Zero cost in release builds.**
#[macro_export]
macro_rules! mid_debug_assert_eq {
    ($left:expr, $right:expr) => {
        $crate::mid_debug_assert_eq!($left, $right, "values are not equal");
    };
    ($left:expr, $right:expr, $($msg:tt)+) => {{
        #[cfg(debug_assertions)]
        {
            let l = &($left);
            let r = &($right);
            if l != r {
                $crate::mid_error!(
                    $crate::level::Tier::Low,
                    "DEBUG_ASSERT_EQ FAILED: `{} == {}`\n  Left:  {:?}\n  Right: {:?}\n  Hint:  {}",
                    stringify!($left), stringify!($right),
                    l, r,
                    format!($($msg)+),
                );
            }
        }
    }};
}

/// Debug-only inequality check. **Zero cost in release builds.**
#[macro_export]
macro_rules! mid_debug_assert_ne {
    ($left:expr, $right:expr) => {
        $crate::mid_debug_assert_ne!($left, $right, "values are unexpectedly equal");
    };
    ($left:expr, $right:expr, $($msg:tt)+) => {{
        #[cfg(debug_assertions)]
        {
            let l = &($left);
            let r = &($right);
            if l == r {
                $crate::mid_error!(
                    $crate::level::Tier::Low,
                    "DEBUG_ASSERT_NE FAILED: `{} != {}`\n  Both:  {:?}\n  Hint:  {}",
                    stringify!($left), stringify!($right),
                    l,
                    format!($($msg)+),
                );
            }
        }
    }};
}
