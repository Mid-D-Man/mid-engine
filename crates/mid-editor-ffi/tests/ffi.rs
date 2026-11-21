//! The C API driven the way a C host drives it.

use std::ffi::{c_char, c_void, CStr};

use mid_editor::paint::{KIND_LINE, KIND_POP_CLIP, KIND_PUSH_CLIP, KIND_RECT, KIND_TEXT};
use mid_editor::PaintItem;
use mid_editor_ffi::*;

unsafe extern "C" fn measure(
    user: *mut c_void,
    _text: *const c_char,
    len: usize,
    size: f32,
) -> f32 {
    if !user.is_null() {
        // SAFETY: the tests pass a pointer to a live u32 as `user`.
        unsafe { *(user as *mut u32) += 1 };
    }
    len as f32 * size * 0.5
}

fn items(h: *const MidEditor) -> Vec<PaintItem> {
    let mut n = 0usize;
    // SAFETY: `h` is live and `n` is valid for a write.
    let p = unsafe { mid_editor_paint_items(h, &mut n) };
    if p.is_null() {
        return Vec::new();
    }
    // SAFETY: the library promises `n` items at `p` until the next frame.
    unsafe { std::slice::from_raw_parts(p, n) }.to_vec()
}

fn text_of(h: *const MidEditor, it: &PaintItem) -> String {
    let mut n = 0usize;
    // SAFETY: `h` is live and `n` is valid for a write.
    let p = unsafe { mid_editor_paint_text(h, &mut n) };
    // SAFETY: the pool is `n` bytes at `p`, and the item's range lies inside it.
    let pool = unsafe { std::slice::from_raw_parts(p.cast::<u8>(), n) };
    String::from_utf8(pool[it.text_start as usize..(it.text_start + it.text_len) as usize].to_vec())
        .unwrap()
}

fn find(h: *const MidEditor, s: &str) -> Option<PaintItem> {
    items(h)
        .into_iter()
        .find(|i| i.kind == KIND_TEXT && text_of(h, i) == s)
}

#[test]
fn a_full_session_through_the_c_api() {
    let mut calls = 0u32;
    // SAFETY: `measure` is valid and `calls` outlives the editor.
    let h = unsafe {
        mid_editor_new(
            1080.0,
            2408.0,
            Some(measure),
            &mut calls as *mut u32 as *mut c_void,
        )
    };
    assert!(!h.is_null());
    // SAFETY: `h` is live for every call below until the free.
    unsafe {
        mid_editor_set_version(h, b"9.9".as_ptr().cast(), 3);
        mid_editor_log(h, b"hello from the host".as_ptr().cast(), 19);
        mid_editor_frame(h, 0.016);
    }
    assert!(
        calls > 0,
        "the host's measure was used while building the frame"
    );
    let all = items(h);
    assert!(all.len() > 50);
    assert!(all.iter().any(|i| i.kind == KIND_RECT));
    assert!(all.iter().any(|i| i.kind == KIND_LINE));
    assert_eq!(
        all.iter().filter(|i| i.kind == KIND_PUSH_CLIP).count(),
        all.iter().filter(|i| i.kind == KIND_POP_CLIP).count()
    );
    assert!(
        find(h, "v9.9").is_some(),
        "the version text reached the bar"
    );

    // Tap the Console tab, then read the log we wrote.
    let tab = find(h, "Console").expect("a Console tab");
    for phase in [0, 2] {
        // SAFETY: `h` is live.
        unsafe { mid_editor_pointer(h, phase, 0, tab.a + 4.0, tab.b + tab.extra * 0.5) };
        unsafe { mid_editor_frame(h, 0.016) };
    }
    assert!(find(h, "hello from the host").is_some());
    // SAFETY: `h` is live.
    assert_eq!(unsafe { mid_editor_panic_count(h) }, 0);

    // SAFETY: `h` is live and freed once.
    unsafe {
        mid_editor_resize(h, 2408.0, 1080.0);
        mid_editor_cancel_pointers(h);
        mid_editor_frame(h, 0.016);
        mid_editor_free(h);
    }
}

#[test]
fn bad_calls_are_ignored_and_null_is_safe_everywhere() {
    // SAFETY: every pointer is NULL or valid, which is what the header allows.
    unsafe {
        assert!(
            mid_editor_new(100.0, 100.0, None, std::ptr::null_mut()).is_null(),
            "no measure, no editor"
        );
        let h = mid_editor_new(f32::NAN, -5.0, Some(measure), std::ptr::null_mut());
        assert!(!h.is_null(), "a bad size falls back to a default");
        mid_editor_pointer(h, 99, 0, 1.0, 1.0);
        mid_editor_pointer(h, -1, 0, 1.0, 1.0);
        mid_editor_resize(h, 0.0, 0.0);
        mid_editor_set_version(h, std::ptr::null(), 5);
        mid_editor_set_version(h, [0xFF_u8, 0xFE].as_ptr().cast(), 2);
        mid_editor_log(h, [0xC0_u8].as_ptr().cast(), 1);
        mid_editor_frame(h, f32::NAN);
        assert_eq!(mid_editor_panic_count(h), 0);
        mid_editor_free(h);

        let null: *mut MidEditor = std::ptr::null_mut();
        mid_editor_free(null);
        mid_editor_resize(null, 1.0, 1.0);
        mid_editor_pointer(null, 0, 0, 1.0, 1.0);
        mid_editor_cancel_pointers(null);
        mid_editor_frame(null, 0.016);
        mid_editor_set_version(null, b"x".as_ptr().cast(), 1);
        mid_editor_log(null, b"x".as_ptr().cast(), 1);
        assert_eq!(mid_editor_panic_count(null), 0);
        let mut n = 99usize;
        assert!(mid_editor_paint_items(null, &mut n).is_null());
        assert_eq!(n, 0);
        assert!(mid_editor_paint_items(null, std::ptr::null_mut()).is_null());
        assert!(mid_editor_paint_text(null, &mut n).is_null());
        assert!(mid_editor_paint_text(null, std::ptr::null_mut()).is_null());
    }
}

#[test]
fn the_version_is_a_nul_terminated_string_that_never_changes() {
    let a = mid_editor_version();
    assert!(!a.is_null());
    // SAFETY: the library returns a NUL-terminated static string.
    let s = unsafe { CStr::from_ptr(a) }.to_str().unwrap();
    assert_eq!(s, env!("CARGO_PKG_VERSION"));
    assert_eq!(mid_editor_version(), a);
}

/// The names the Rust side exports, read from the source.
fn exported_names() -> Vec<String> {
    let src = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/src/lib.rs")).unwrap();
    let mut out = Vec::new();
    for line in src.lines() {
        let l = line.trim();
        for prefix in ["pub unsafe extern \"C\" fn ", "pub extern \"C\" fn "] {
            if let Some(rest) = l.strip_prefix(prefix) {
                out.push(rest.split('(').next().unwrap().to_string());
            }
        }
    }
    out.sort();
    out
}

fn header() -> String {
    std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/include/mid_editor.h")).unwrap()
}

#[test]
fn the_header_declares_exactly_the_functions_the_library_exports() {
    let h = header();
    let mut declared: Vec<String> = Vec::new();
    for token in h.split(|c: char| !(c.is_alphanumeric() || c == '_')) {
        if token.starts_with("mid_editor_") && !declared.contains(&token.to_string()) {
            declared.push(token.to_string());
        }
    }
    declared.sort();
    assert_eq!(declared, exported_names());
}

#[test]
fn the_header_agrees_with_the_rust_paint_layout_and_constants() {
    let h = header();
    assert_eq!(std::mem::size_of::<PaintItem>(), 36);
    for (name, value) in [
        ("MID_PAINT_RECT", KIND_RECT),
        ("MID_PAINT_PUSH_CLIP", KIND_PUSH_CLIP),
        ("MID_PAINT_POP_CLIP", KIND_POP_CLIP),
        ("MID_PAINT_LINE", KIND_LINE),
        ("MID_PAINT_TEXT", KIND_TEXT),
    ] {
        let line = h
            .lines()
            .find(|l| l.contains(name))
            .unwrap_or_else(|| panic!("{name} missing"));
        let after = line.split(name).nth(1).unwrap();
        let n: u32 = after
            .trim_start()
            .trim_start_matches('=')
            .trim()
            .split(|c: char| !c.is_ascii_digit())
            .next()
            .unwrap()
            .parse()
            .unwrap();
        assert_eq!(n, value, "{name}");
    }
    for (name, value) in [
        ("MID_PHASE_DOWN", 0),
        ("MID_PHASE_MOVE", 1),
        ("MID_PHASE_UP", 2),
        ("MID_PHASE_CANCEL", 3),
    ] {
        let line = h.lines().find(|l| l.contains(name)).unwrap();
        assert!(line.contains(&format!("{name} = {value}")), "{name}");
    }
    let order = [
        "uint32_t kind;",
        "uint32_t color;",
        "float a, b, c, d;",
        "float extra;",
        "uint32_t text_start;",
        "uint32_t text_len;",
    ];
    let mut at = 0;
    for field in order {
        let i = h[at..]
            .find(field)
            .unwrap_or_else(|| panic!("{field} missing or out of order"));
        at += i + field.len();
    }
}
