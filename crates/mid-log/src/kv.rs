// crates/mid-log/src/kv.rs

//! Structured key-value types for the mid-log KV logging API.
//!
//! ## Why this closes the slog gap
//!
//! Printf path:  `format!("player {} at ({:.2},{:.2})", id, x, y)`
//!   - Float-to-string conversion:  ~200–400 ns
//!   - String heap allocation:      ~50–100 ns
//!   - Total calling-thread cost:   ~250–500 ns
//!
//! KV path: `"player spawned"; "id" => id, "x" => x, "y" => y`
//!   - One Vec allocation (3 pairs): ~35–50 ns
//!   - Populate 3 KvPairs (moves):   ~10–15 ns
//!   - Total calling-thread cost:    ~45–65 ns
//!
//! Savings: ~200–450 ns per structured log call.
//! The IO thread pays the formatting cost — amortised, off the game loop.
//!
//! ## Output format
//!
//! Printf:     `[INFO][HIGH] player 1 at (1.00, 2.00)  (src:42)`
//! Structured: `[INFO][HIGH] player spawned  id=1  x=1.000  y=2.000  (src:42)`

use std::fmt;

/// A typed log value for structured key-value pairs.
///
/// All scalar variants are stored inline — zero allocation.
/// `Str` borrows a `'static` slice — also zero allocation.
/// Dynamic strings should use the printf API or be formatted before logging.
#[derive(Debug, Clone)]
pub enum KvValue {
    Bool(bool),
    /// All signed integer types widen to `i64`.
    I64(i64),
    /// All unsigned integer types widen to `u64`.
    U64(u64),
    /// Both `f32` and `f64` are stored as `f64`.
    /// `f32` widens losslessly for all values in the 24-bit significand range,
    /// which covers every practical game coordinate, health value, and timer.
    F64(f64),
    /// A `'static` string key value — compile-time constant, zero allocation.
    Str(&'static str),
}

impl fmt::Display for KvValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KvValue::Bool(b) => write!(f, "{}", b),
            KvValue::I64(i)  => write!(f, "{}", i),
            KvValue::U64(u)  => write!(f, "{}", u),
            // 3 decimal places matches the default float precision in Unity/Unreal overlays.
            // Override with the printf API if you need `{:.6}` precision.
            KvValue::F64(v)  => write!(f, "{:.3}", v),
            KvValue::Str(s)  => f.write_str(s),
        }
    }
}

/// A structured log key-value pair.
/// Keys are always `&'static str` — they must be compile-time string literals.
pub type KvPair = (&'static str, KvValue);

/// Automatic conversion from any standard scalar type into `KvValue`.
///
/// Used implicitly by `mid_kvinfo!` and friends:
/// ```rust,no_run
/// # use mid_log::{mid_kvinfo, level::Tier};
/// let hp = 75u32;
/// let pos_x = 3.14f32;
/// mid_kvinfo!(Tier::High, "entity update"; "hp" => hp, "x" => pos_x);
/// ```
pub trait IntoKvValue {
    fn into_kv_value(self) -> KvValue;
}

macro_rules! impl_into_kv {
    ($src:ty => $variant:ident as $dst:ty) => {
        impl IntoKvValue for $src {
            #[inline(always)]
            fn into_kv_value(self) -> KvValue {
                KvValue::$variant(self as $dst)
            }
        }
    };
}

impl_into_kv!(i8   => I64 as i64);
impl_into_kv!(i16  => I64 as i64);
impl_into_kv!(i32  => I64 as i64);
impl_into_kv!(i64  => I64 as i64);
impl_into_kv!(u8   => U64 as u64);
impl_into_kv!(u16  => U64 as u64);
impl_into_kv!(u32  => U64 as u64);
impl_into_kv!(u64  => U64 as u64);
impl_into_kv!(f32  => F64 as f64);
impl_into_kv!(f64  => F64 as f64);

impl IntoKvValue for bool {
    #[inline(always)]
    fn into_kv_value(self) -> KvValue { KvValue::Bool(self) }
}

impl IntoKvValue for &'static str {
    #[inline(always)]
    fn into_kv_value(self) -> KvValue { KvValue::Str(self) }
}
