//! mid-log — Non-blocking, tiered logger
//!
//! Lock-free SPSC ring buffer (rtrb) on the hot path.
//! A background IO thread handles formatting and disk writes.
//! Zero frame-rate impact — unlike Unity's Debug.Log.

pub mod buffer;
pub mod level;
pub mod writer;
pub mod ffi;

pub use level::{LogLevel, Tier};
pub use buffer::LogBuffer;
