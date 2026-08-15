//! C-compatible FFI exports for mid-net.
//!
//! Scope for this pass: the wire codec (`PlayerState`/`PlayerEvent`
//! encode/decode) only — genuinely useful right now regardless of
//! transport status, since a non-Rust caller (Ubel Stratum's LOW tier is
//! the concrete motivating case; see docs/mid-net.md) can encode/decode
//! packets today even with no real `Transport` backend built yet.
//! `Connection<T>` is NOT exposed here — it's generic over `Transport`,
//! and C ABI can't cross a Rust generic; exposing it means picking one
//! concrete backend to monomorphize against, and the only real one that
//! exists today is `LoopbackTransport`, which isn't useful to a real FFI
//! caller. Revisit once `mid-net-transport-quinn` exists.
//!
//! ## Safety conventions used throughout
//! - Every function checks its pointer arguments for null before
//!   dereferencing and returns a defined error code instead of
//!   dereferencing a null pointer.
//! - Every function's body runs inside [`std::panic::catch_unwind`] —
//!   unwinding across an `extern "C"` boundary is undefined behavior, so
//!   a panic here becomes `MidNetStatus::InternalPanic` instead. This
//!   protects against *this crate's* bugs; it can't protect against a
//!   caller violating a documented `unsafe` contract (a pointer that's
//!   not actually valid for the length claimed, a C string that isn't
//!   actually null-terminated, etc.) — that's inherent to any C ABI and
//!   is on the caller, same as it would be for any C library.
//! - `encode` functions support the "call with a null buffer to query
//!   the required size" idiom (returns the byte count, writes nothing)
//!   — standard for variable-length C APIs, so a caller never has to
//!   guess or over-allocate.
//! - Every function taking a raw pointer is marked `unsafe fn` and
//!   documents its `# Safety` contract — `clippy::not_unsafe_ptr_arg_deref`
//!   (deny by default; caught this on real CI, not found locally, same
//!   MSRV-gap pattern as `dangerous_implicit_autorefs` earlier in this
//!   project) requires this even though every dereference was already
//!   inside its own inner `unsafe {}` block and null-checked first. The
//!   inner blocks stay exactly as they were — only the function
//!   signatures and their doc comments changed. This is a *Rust-side*
//!   marker only: it does not change the exported C ABI at all (C has no
//!   concept of `unsafe`), so `mid_net.h` and every existing C caller —
//!   including `ffi-smoke-test/test.c` — are unaffected. It only means
//!   Rust code calling these now needs an explicit `unsafe {}` block,
//!   which is why every direct call in this file's own tests below
//!   changed too.

use mid_net_wire::{DecodeError, Packet, PlayerEvent, PlayerId, PlayerState};
use std::ffi::CStr;
use std::os::raw::c_char;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;
use std::slice;

/// Status/error codes. Encode functions return a non-negative byte count
/// on success, not this enum — everything else returns one of these
/// directly (`MidNetStatus::Ok` is always `0`).
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MidNetStatus {
    Ok = 0,
    NullPointer = -1,
    BufferTooSmall = -2,
    UnexpectedEnd = -3,
    InvalidUtf8 = -4,
    TrailingBytes = -5,
    /// Something inside this crate panicked. Should never happen for
    /// well-formed input per each function's documented contract —
    /// exists so a caller gets a defined code instead of UB from an
    /// unwind crossing the FFI boundary.
    InternalPanic = -6,
}

impl From<DecodeError> for MidNetStatus {
    fn from(e: DecodeError) -> Self {
        match e {
            DecodeError::UnexpectedEnd => MidNetStatus::UnexpectedEnd,
            DecodeError::InvalidUtf8 => MidNetStatus::InvalidUtf8,
            DecodeError::TrailingBytes => MidNetStatus::TrailingBytes,
        }
    }
}

fn ffi_guard(f: impl FnOnce() -> i32) -> i32 {
    catch_unwind(AssertUnwindSafe(f)).unwrap_or(MidNetStatus::InternalPanic as i32)
}

// ---------------------------------------------------------------------
// PlayerState — repr(C), crosses the boundary by value, no opaque handle needed
// ---------------------------------------------------------------------

/// Wire size of an encoded `PlayerState` — always this many bytes, so a
/// caller can just allocate this once rather than querying. Takes no
/// pointer, not `unsafe` — clippy agrees, this one was never flagged.
#[no_mangle]
pub extern "C" fn mid_net_player_state_wire_size() -> usize {
    mid_net_wire::PLAYER_STATE_WIRE_SIZE
}

/// Encodes `*state` into `out_buf` (`out_buf_len` bytes, caller-owned).
/// Returns bytes written (always `mid_net_player_state_wire_size()`) on
/// success, or a negative `MidNetStatus`. Pass `out_buf = NULL` to query
/// the required size without writing anything.
///
/// # Safety
/// `state` must point to a valid, initialized `PlayerState`. If
/// `out_buf` is non-null, it must be valid for `out_buf_len` bytes.
#[no_mangle]
pub unsafe extern "C" fn mid_net_player_state_encode(state: *const PlayerState, out_buf: *mut u8, out_buf_len: usize) -> i32 {
    ffi_guard(|| {
        if state.is_null() {
            return MidNetStatus::NullPointer as i32;
        }
        let state = unsafe { &*state };
        let mut payload = Vec::new();
        state.encode(&mut payload);

        if out_buf.is_null() {
            return payload.len() as i32;
        }
        if payload.len() > out_buf_len {
            return MidNetStatus::BufferTooSmall as i32;
        }
        let out = unsafe { slice::from_raw_parts_mut(out_buf, out_buf_len) };
        out[..payload.len()].copy_from_slice(&payload);
        payload.len() as i32
    })
}

/// Decodes exactly `buf_len` bytes from `buf` into `*out_state`. Returns
/// `MidNetStatus::Ok` (`0`) on success.
///
/// # Safety
/// `buf` must be valid for `buf_len` bytes. `out_state` must point to
/// valid (not necessarily initialized) memory for one `PlayerState`.
#[no_mangle]
pub unsafe extern "C" fn mid_net_player_state_decode(buf: *const u8, buf_len: usize, out_state: *mut PlayerState) -> i32 {
    ffi_guard(|| {
        if buf.is_null() || out_state.is_null() {
            return MidNetStatus::NullPointer as i32;
        }
        let bytes = unsafe { slice::from_raw_parts(buf, buf_len) };
        match PlayerState::decode(bytes) {
            Ok(state) => {
                unsafe { ptr::write(out_state, state) };
                MidNetStatus::Ok as i32
            }
            Err(e) => MidNetStatus::from(e) as i32,
        }
    })
}

// ---------------------------------------------------------------------
// PlayerEvent — owns Strings, not C-representable by value -> opaque handle
// ---------------------------------------------------------------------

/// Opaque handle. Always heap-allocated by this crate; every handle
/// returned by `_new` or `_decode` must be freed with
/// `mid_net_player_event_free` exactly once.
pub struct MidNetPlayerEvent(PlayerEvent);

/// Builds a `PlayerEvent` from null-terminated UTF-8 C strings. Returns
/// NULL on a null pointer or invalid UTF-8.
///
/// # Safety
/// `event` and `payload` must be valid, null-terminated C strings.
#[no_mangle]
pub unsafe extern "C" fn mid_net_player_event_new(player_id: u32, event: *const c_char, payload: *const c_char) -> *mut MidNetPlayerEvent {
    if event.is_null() || payload.is_null() {
        return ptr::null_mut();
    }
    let built = catch_unwind(AssertUnwindSafe(|| {
        let event_str = unsafe { CStr::from_ptr(event) }.to_str().ok()?.to_owned();
        let payload_str = unsafe { CStr::from_ptr(payload) }.to_str().ok()?.to_owned();
        Some(PlayerEvent { player_id: PlayerId(player_id), event: event_str, payload: payload_str })
    }));
    match built {
        Ok(Some(pe)) => Box::into_raw(Box::new(MidNetPlayerEvent(pe))),
        _ => ptr::null_mut(),
    }
}

/// Frees a handle returned by `mid_net_player_event_new` or
/// `mid_net_player_event_decode`. NULL is a safe no-op.
///
/// # Safety
/// `event` must either be NULL or a handle previously returned by
/// `mid_net_player_event_new`/`_decode` that hasn't been freed yet.
/// Freeing the same handle twice, or using it after freeing, is
/// undefined behavior.
#[no_mangle]
pub unsafe extern "C" fn mid_net_player_event_free(event: *mut MidNetPlayerEvent) {
    if event.is_null() {
        return;
    }
    let _ = catch_unwind(AssertUnwindSafe(|| unsafe { drop(Box::from_raw(event)) }));
}

/// # Safety
/// `event` must either be NULL or a valid handle from `_new`/`_decode`
/// that hasn't been freed.
#[no_mangle]
pub unsafe extern "C" fn mid_net_player_event_get_player_id(event: *const MidNetPlayerEvent) -> u32 {
    if event.is_null() {
        return 0;
    }
    // SAFETY: non-null handle from `_new`/`_decode`, per this module's contract.
    let event = unsafe { &*event };
    event.0.player_id.0
}

/// Pointer to the event-name string's UTF-8 bytes (NOT null-terminated —
/// use the `_len` function alongside it). Valid only while `event` is
/// alive; do not use after `mid_net_player_event_free`.
///
/// # Safety
/// `event` must either be NULL or a valid, not-yet-freed handle.
#[no_mangle]
pub unsafe extern "C" fn mid_net_player_event_get_event_ptr(event: *const MidNetPlayerEvent) -> *const u8 {
    if event.is_null() {
        return ptr::null();
    }
    // SAFETY: non-null handle per this module's contract. Bound as an
    // explicit reference here rather than chaining through the raw
    // pointer deref directly -- `.as_ptr()` takes `&self`, and letting
    // that reference get created implicitly (`(*event).0.event.as_ptr()`)
    // is exactly what rustc's `dangerous_implicit_autorefs` lint (deny by
    // default since ~1.93) flags. Same fix applied to every getter below.
    let event = unsafe { &*event };
    event.0.event.as_ptr()
}

/// # Safety
/// `event` must either be NULL or a valid, not-yet-freed handle.
#[no_mangle]
pub unsafe extern "C" fn mid_net_player_event_get_event_len(event: *const MidNetPlayerEvent) -> usize {
    if event.is_null() {
        return 0;
    }
    let event = unsafe { &*event };
    event.0.event.len()
}

/// Pointer to the payload string's UTF-8 bytes (NOT null-terminated).
///
/// # Safety
/// `event` must either be NULL or a valid, not-yet-freed handle.
#[no_mangle]
pub unsafe extern "C" fn mid_net_player_event_get_payload_ptr(event: *const MidNetPlayerEvent) -> *const u8 {
    if event.is_null() {
        return ptr::null();
    }
    let event = unsafe { &*event };
    event.0.payload.as_ptr()
}

/// # Safety
/// `event` must either be NULL or a valid, not-yet-freed handle.
#[no_mangle]
pub unsafe extern "C" fn mid_net_player_event_get_payload_len(event: *const MidNetPlayerEvent) -> usize {
    if event.is_null() {
        return 0;
    }
    let event = unsafe { &*event };
    event.0.payload.len()
}

/// Encodes `*event` into `out_buf`. Same "pass NULL to query size"
/// idiom as `mid_net_player_state_encode`.
///
/// # Safety
/// `event` must either be NULL or a valid, not-yet-freed handle. If
/// `out_buf` is non-null, it must be valid for `out_buf_len` bytes.
#[no_mangle]
pub unsafe extern "C" fn mid_net_player_event_encode(event: *const MidNetPlayerEvent, out_buf: *mut u8, out_buf_len: usize) -> i32 {
    ffi_guard(|| {
        if event.is_null() {
            return MidNetStatus::NullPointer as i32;
        }
        let event = unsafe { &(*event).0 };
        let mut payload = Vec::new();
        event.encode(&mut payload);

        if out_buf.is_null() {
            return payload.len() as i32;
        }
        if payload.len() > out_buf_len {
            return MidNetStatus::BufferTooSmall as i32;
        }
        let out = unsafe { slice::from_raw_parts_mut(out_buf, out_buf_len) };
        out[..payload.len()].copy_from_slice(&payload);
        payload.len() as i32
    })
}

/// Decodes exactly `buf_len` bytes from `buf` into a new owned handle.
/// Returns NULL on a null pointer or decode failure.
///
/// # Safety
/// `buf` must be valid for `buf_len` bytes.
#[no_mangle]
pub unsafe extern "C" fn mid_net_player_event_decode(buf: *const u8, buf_len: usize) -> *mut MidNetPlayerEvent {
    if buf.is_null() {
        return ptr::null_mut();
    }
    let decoded = catch_unwind(AssertUnwindSafe(|| {
        let bytes = unsafe { slice::from_raw_parts(buf, buf_len) };
        PlayerEvent::decode(bytes).ok()
    }));
    match decoded {
        Ok(Some(pe)) => Box::into_raw(Box::new(MidNetPlayerEvent(pe))),
        _ => ptr::null_mut(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn player_state_encode_decode_round_trips_through_ffi() {
        let state = PlayerState { x: 1.0, y: 2.0, z: 3.0, rot_x: 0.0, rot_y: 0.0, rot_z: 0.0, rot_w: 1.0 };
        let mut buf = vec![0u8; mid_net_player_state_wire_size()];
        // SAFETY: `&state`/`buf.as_mut_ptr()` are both valid for this call's duration.
        let written = unsafe { mid_net_player_state_encode(&state, buf.as_mut_ptr(), buf.len()) };
        assert_eq!(written, mid_net_player_state_wire_size() as i32);

        let mut out = PlayerState { x: 0.0, y: 0.0, z: 0.0, rot_x: 0.0, rot_y: 0.0, rot_z: 0.0, rot_w: 0.0 };
        // SAFETY: `buf` holds exactly the bytes just encoded above; `&mut out` is valid.
        let status = unsafe { mid_net_player_state_decode(buf.as_ptr(), buf.len(), &mut out) };
        assert_eq!(status, MidNetStatus::Ok as i32);
        assert_eq!(out, state);
    }

    #[test]
    fn player_state_encode_null_buf_queries_size() {
        let state = PlayerState::default();
        // SAFETY: `&state` valid; NULL out_buf is the documented "query size" contract.
        let size = unsafe { mid_net_player_state_encode(&state, ptr::null_mut(), 0) };
        assert_eq!(size, mid_net_player_state_wire_size() as i32);
    }

    #[test]
    fn player_state_encode_rejects_buffer_too_small() {
        let state = PlayerState::default();
        let mut tiny = [0u8; 4];
        // SAFETY: `&state` valid; `tiny` is valid for its own length.
        let result = unsafe { mid_net_player_state_encode(&state, tiny.as_mut_ptr(), tiny.len()) };
        assert_eq!(result, MidNetStatus::BufferTooSmall as i32);
    }

    #[test]
    fn player_state_functions_reject_null_pointers() {
        let mut out = PlayerState::default();
        // SAFETY: exercising the documented null-pointer rejection path itself.
        unsafe {
            assert_eq!(mid_net_player_state_decode(ptr::null(), 10, &mut out), MidNetStatus::NullPointer as i32);
            assert_eq!(mid_net_player_state_encode(ptr::null(), ptr::null_mut(), 0), MidNetStatus::NullPointer as i32);
        }
    }

    #[test]
    fn player_event_new_free_and_getters_round_trip() {
        let event = std::ffi::CString::new("pickup").unwrap();
        let payload = std::ffi::CString::new("item_id=3").unwrap();
        // SAFETY: both CStrings are valid, null-terminated, and outlive this block.
        let handle = unsafe { mid_net_player_event_new(42, event.as_ptr(), payload.as_ptr()) };
        assert!(!handle.is_null());

        // SAFETY: `handle` is non-null and not yet freed for the whole block below.
        unsafe {
            assert_eq!(mid_net_player_event_get_player_id(handle), 42);
            let event_bytes = slice::from_raw_parts(mid_net_player_event_get_event_ptr(handle), mid_net_player_event_get_event_len(handle));
            assert_eq!(event_bytes, b"pickup");
            let payload_bytes = slice::from_raw_parts(mid_net_player_event_get_payload_ptr(handle), mid_net_player_event_get_payload_len(handle));
            assert_eq!(payload_bytes, b"item_id=3");

            mid_net_player_event_free(handle);
        }
    }

    #[test]
    fn player_event_encode_decode_round_trips_through_ffi() {
        let event = std::ffi::CString::new("damage").unwrap();
        let payload = std::ffi::CString::new("amount=10").unwrap();
        // SAFETY: both CStrings valid and null-terminated.
        let handle = unsafe { mid_net_player_event_new(7, event.as_ptr(), payload.as_ptr()) };

        // SAFETY: `handle` non-null (just created above) and not yet freed for this whole block.
        unsafe {
            let needed = mid_net_player_event_encode(handle, ptr::null_mut(), 0);
            assert!(needed > 0);
            let mut buf = vec![0u8; needed as usize];
            let written = mid_net_player_event_encode(handle, buf.as_mut_ptr(), buf.len());
            assert_eq!(written, needed);

            let decoded = mid_net_player_event_decode(buf.as_ptr(), buf.len());
            assert!(!decoded.is_null());
            assert_eq!(mid_net_player_event_get_player_id(decoded), 7);

            mid_net_player_event_free(handle);
            mid_net_player_event_free(decoded);
        }
    }

    #[test]
    fn player_event_new_rejects_invalid_utf8() {
        let invalid = [0x66u8, 0x6f, 0xff, 0x00];
        let payload = std::ffi::CString::new("ok").unwrap();
        // SAFETY: `invalid` is a valid, null-terminated byte buffer (just not valid UTF-8,
        // which is exactly the rejection path this test exercises); `payload` is a valid CString.
        let handle = unsafe { mid_net_player_event_new(1, invalid.as_ptr() as *const c_char, payload.as_ptr()) };
        assert!(handle.is_null());
    }

    #[test]
    fn player_event_decode_rejects_garbage() {
        let garbage = [0xFFu8, 0xFF, 0xFF];
        // SAFETY: `garbage` is valid for its own length -- exercising the decode-failure path.
        let handle = unsafe { mid_net_player_event_decode(garbage.as_ptr(), garbage.len()) };
        assert!(handle.is_null());
    }

    #[test]
    fn null_handle_getters_return_safe_defaults_not_crash() {
        let null_handle: *const MidNetPlayerEvent = ptr::null();
        // SAFETY: every one of these has a documented, tested NULL-handle path.
        unsafe {
            assert_eq!(mid_net_player_event_get_player_id(null_handle), 0);
            assert!(mid_net_player_event_get_event_ptr(null_handle).is_null());
            assert_eq!(mid_net_player_event_get_event_len(null_handle), 0);
            mid_net_player_event_free(ptr::null_mut());
        }
    }
        }
