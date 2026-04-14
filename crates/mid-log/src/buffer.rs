//! Lock-free SPSC ring buffer for the hot logging path.
//!
//! Game thread pushes log entries into rtrb.
//! Background IO thread drains the buffer.
// Auto-generated stub
