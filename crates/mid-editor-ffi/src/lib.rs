// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-editor-ffi.md, section "lib.rs"
// ============================================================================
//! C ABI over [`mid_editor`].
//!
//! Every exported function runs its body inside `catch_unwind`, so a panic
//! never unwinds into C. A panic is counted (`mid_editor_panic_count`) and
//! written to the editor's console log when that is still possible.

use std::ffi::{c_char, c_void, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::OnceLock;

use mid_editor::input::Phase;
use mid_editor::{Editor, PaintItem, TextMeasure};

/// The C type of the host's text-measure callback.
pub type MidTextMeasureFn =
    unsafe extern "C" fn(user: *mut c_void, text: *const c_char, len: usize, size: f32) -> f32;

struct CMeasure {
    f: MidTextMeasureFn,
    user: *mut c_void,
}

impl TextMeasure for CMeasure {
    fn width(&self, text: &str, size: f32) -> f32 {
        // SAFETY: `f` and `user` came from the host, which promised in the
        // header that `f` accepts `user`, a pointer to `len` bytes, and a size.
        let w = unsafe { (self.f)(self.user, text.as_ptr().cast(), text.len(), size) };
        if w.is_finite() && w >= 0.0 {
            w
        } else {
            text.chars().count() as f32 * size * 0.5
        }
    }
}

/// An editor and the count of panics caught while using it.
pub struct MidEditor {
    editor: Editor,
    panics: u32,
}

fn guard<T>(fallback: T, f: impl FnOnce() -> T) -> T {
    catch_unwind(AssertUnwindSafe(f)).unwrap_or(fallback)
}

/// Runs `f` on the editor, counting a panic instead of letting it escape.
fn with_editor(ptr: *mut MidEditor, f: impl FnOnce(&mut Editor)) {
    // SAFETY: the header says `ptr` is NULL or came from `mid_editor_new` and
    // has not been freed, and a handle is used from one thread.
    let Some(handle) = (unsafe { ptr.as_mut() }) else {
        return;
    };
    let ok = catch_unwind(AssertUnwindSafe(|| f(&mut handle.editor))).is_ok();
    if !ok {
        handle.panics = handle.panics.saturating_add(1);
        let _ = catch_unwind(AssertUnwindSafe(|| {
            handle
                .editor
                .log("a panic was caught in the editor core".to_string())
        }));
    }
}

/// Creates an editor. See `mid_editor.h`.
///
/// # Safety
/// `measure` must be a valid function for as long as the editor lives, and
/// `user` whatever it expects.
#[no_mangle]
pub unsafe extern "C" fn mid_editor_new(
    width: f32,
    height: f32,
    measure: Option<MidTextMeasureFn>,
    user: *mut c_void,
) -> *mut MidEditor {
    let Some(f) = measure else {
        return std::ptr::null_mut();
    };
    guard(std::ptr::null_mut(), || {
        let editor = Editor::new(width, height, Box::new(CMeasure { f, user }));
        Box::into_raw(Box::new(MidEditor { editor, panics: 0 }))
    })
}

/// Releases an editor. See `mid_editor.h`.
///
/// # Safety
/// `editor` must be NULL or a pointer from `mid_editor_new` not yet freed.
#[no_mangle]
pub unsafe extern "C" fn mid_editor_free(editor: *mut MidEditor) {
    if !editor.is_null() {
        // SAFETY: by this function's contract the pointer came from Box::into_raw.
        let boxed = unsafe { Box::from_raw(editor) };
        let _ = catch_unwind(AssertUnwindSafe(move || drop(boxed)));
    }
}

/// The screen changed size.
///
/// # Safety
/// `editor` must be NULL or a live pointer from `mid_editor_new`.
#[no_mangle]
pub unsafe extern "C" fn mid_editor_resize(editor: *mut MidEditor, width: f32, height: f32) {
    with_editor(editor, |e| e.resize(width, height));
}

/// One pointer event.
///
/// # Safety
/// `editor` must be NULL or a live pointer from `mid_editor_new`.
#[no_mangle]
pub unsafe extern "C" fn mid_editor_pointer(
    editor: *mut MidEditor,
    phase: i32,
    id: i32,
    x: f32,
    y: f32,
) {
    if let Some(phase) = Phase::from_i32(phase) {
        with_editor(editor, |e| e.pointer(phase, id, x, y));
    }
}

/// Forgets every touch in progress.
///
/// # Safety
/// `editor` must be NULL or a live pointer from `mid_editor_new`.
#[no_mangle]
pub unsafe extern "C" fn mid_editor_cancel_pointers(editor: *mut MidEditor) {
    with_editor(editor, |e| e.cancel_pointers());
}

/// Advances one frame and builds the paint list.
///
/// # Safety
/// `editor` must be NULL or a live pointer from `mid_editor_new`.
#[no_mangle]
pub unsafe extern "C" fn mid_editor_frame(editor: *mut MidEditor, dt: f32) {
    with_editor(editor, |e| e.frame(dt));
}

/// The paint items of the last frame, and how many there are.
///
/// # Safety
/// `editor` must be NULL or a live pointer from `mid_editor_new`; `count` must
/// be NULL or valid for a write.
#[no_mangle]
pub unsafe extern "C" fn mid_editor_paint_items(
    editor: *const MidEditor,
    count: *mut usize,
) -> *const PaintItem {
    // SAFETY: see the contract above.
    let handle = unsafe { editor.as_ref() };
    let (ptr, n) = match handle {
        Some(h) => {
            let items = &h.editor.paint().items;
            (items.as_ptr(), items.len())
        }
        None => (std::ptr::null(), 0),
    };
    // SAFETY: `count` is NULL or valid for a write.
    if let Some(c) = unsafe { count.as_mut() } {
        *c = n;
    }
    ptr
}

/// The text pool of the last frame, and its length in bytes.
///
/// # Safety
/// `editor` must be NULL or a live pointer from `mid_editor_new`; `len` must
/// be NULL or valid for a write.
#[no_mangle]
pub unsafe extern "C" fn mid_editor_paint_text(
    editor: *const MidEditor,
    len: *mut usize,
) -> *const c_char {
    // SAFETY: see the contract above.
    let handle = unsafe { editor.as_ref() };
    let (ptr, n) = match handle {
        Some(h) => {
            let t = &h.editor.paint().text;
            (t.as_ptr().cast(), t.len())
        }
        None => (std::ptr::null(), 0),
    };
    // SAFETY: `len` is NULL or valid for a write.
    if let Some(l) = unsafe { len.as_mut() } {
        *l = n;
    }
    ptr
}

fn host_text<'a>(utf8: *const c_char, len: usize) -> Option<&'a str> {
    if utf8.is_null() {
        return None;
    }
    // SAFETY: the header says `utf8` points to `len` readable bytes.
    let bytes = unsafe { std::slice::from_raw_parts(utf8.cast::<u8>(), len) };
    std::str::from_utf8(bytes).ok()
}

/// Sets the version text.
///
/// # Safety
/// `editor` must be NULL or live; `utf8` must be NULL or point to `len` readable bytes.
#[no_mangle]
pub unsafe extern "C" fn mid_editor_set_version(
    editor: *mut MidEditor,
    utf8: *const c_char,
    len: usize,
) {
    if let Some(s) = host_text(utf8, len) {
        with_editor(editor, |e| e.set_version(s));
    }
}

/// Adds a console log line.
///
/// # Safety
/// `editor` must be NULL or live; `utf8` must be NULL or point to `len` readable bytes.
#[no_mangle]
pub unsafe extern "C" fn mid_editor_log(editor: *mut MidEditor, utf8: *const c_char, len: usize) {
    if let Some(s) = host_text(utf8, len) {
        let line: String = s.chars().take(200).collect();
        with_editor(editor, |e| e.log(line));
    }
}

/// The number of panics caught since the editor was created.
///
/// # Safety
/// `editor` must be NULL or a live pointer from `mid_editor_new`.
#[no_mangle]
pub unsafe extern "C" fn mid_editor_panic_count(editor: *const MidEditor) -> u32 {
    // SAFETY: see the contract above.
    unsafe { editor.as_ref() }.map_or(0, |h| h.panics)
}

/// The library version, NUL-terminated and never NULL.
#[no_mangle]
pub extern "C" fn mid_editor_version() -> *const c_char {
    static VERSION: OnceLock<CString> = OnceLock::new();
    VERSION
        .get_or_init(|| CString::new(env!("CARGO_PKG_VERSION")).unwrap_or_default())
        .as_ptr()
}

#[cfg(test)]
mod tests {
    use super::*;

    unsafe extern "C" fn measure(
        _user: *mut c_void,
        _text: *const c_char,
        len: usize,
        size: f32,
    ) -> f32 {
        len as f32 * size * 0.5
    }

    fn make() -> *mut MidEditor {
        // SAFETY: `measure` is a valid callback and `user` is unused.
        unsafe { mid_editor_new(1080.0, 2408.0, Some(measure), std::ptr::null_mut()) }
    }

    #[test]
    fn a_panic_inside_the_core_is_counted_and_logged_and_does_not_escape() {
        let h = make();
        with_editor(h, |_| panic!("boom"));
        with_editor(h, |_| panic!("boom again"));
        // SAFETY: `h` is live.
        assert_eq!(unsafe { mid_editor_panic_count(h) }, 2);
        // The editor still works afterwards.
        // SAFETY: `h` is live.
        unsafe { mid_editor_frame(h, 0.016) };
        // SAFETY: `h` is live and was not freed.
        unsafe { mid_editor_free(h) };
    }

    #[test]
    fn guard_returns_its_fallback_on_a_panic() {
        assert_eq!(guard(7, || -> i32 { panic!("x") }), 7);
        assert_eq!(guard(7, || 9), 9);
    }

    #[test]
    fn a_measure_that_returns_nonsense_falls_back_to_a_plain_estimate() {
        unsafe extern "C" fn bad(_u: *mut c_void, _t: *const c_char, _l: usize, _s: f32) -> f32 {
            f32::NAN
        }
        let m = CMeasure {
            f: bad,
            user: std::ptr::null_mut(),
        };
        assert_eq!(m.width("abcd", 10.0), 20.0);
    }
}
