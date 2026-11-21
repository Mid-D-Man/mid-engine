/* ============================================================================
 * NOTICE: Full documentation, design decisions, and fix history for this file
 * live in docs/mid-editor-ffi.md, section "include/mid_editor.h"
 * ============================================================================
 * C API of the Mid Engine editor core (crate mid-editor-ffi).
 *
 * Written by hand. scripts/build_editor_android.sh checks that every function
 * declared here is exported by the library, and that every exported function
 * is declared here.
 *
 * Threading: a MidEditor is used from one thread. Every function is safe to
 * call with a NULL editor and does nothing.
 */
#ifndef MID_EDITOR_H
#define MID_EDITOR_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MidEditor MidEditor;

/* Pointer phases for mid_editor_pointer. */
enum { MID_PHASE_DOWN = 0, MID_PHASE_MOVE = 1, MID_PHASE_UP = 2, MID_PHASE_CANCEL = 3 };

/* Paint item kinds. */
enum {
    MID_PAINT_RECT = 0,       /* a,b = x,y   c,d = w,h    extra = corner radius        */
    MID_PAINT_PUSH_CLIP = 1,  /* a,b = x,y   c,d = w,h                                 */
    MID_PAINT_POP_CLIP = 2,   /* nothing                                               */
    MID_PAINT_LINE = 3,       /* a,b = start c,d = end    extra = width                */
    MID_PAINT_TEXT = 4        /* a,b = x,y (top-left)     extra = size, text in pool   */
};

/* One shape. 36 bytes. color is 0xRRGGBBAA. */
typedef struct MidPaintItem {
    uint32_t kind;
    uint32_t color;
    float a, b, c, d;
    float extra;
    uint32_t text_start; /* byte offset into the text pool */
    uint32_t text_len;   /* length in bytes */
} MidPaintItem;

/* Returns the width in pixels of UTF-8 text (not NUL-terminated) drawn `size`
 * pixels tall. Called while a frame is being built. */
typedef float (*MidTextMeasureFn)(void *user, const char *text, size_t len, float size);

/* Creates an editor for a screen of width by height pixels. Returns NULL if
 * measure is NULL or creation failed. Release it with mid_editor_free. */
MidEditor *mid_editor_new(float width, float height, MidTextMeasureFn measure, void *user);

/* Releases an editor. NULL is allowed. */
void mid_editor_free(MidEditor *editor);

/* The screen changed size. Sizes that are not positive and finite are ignored. */
void mid_editor_resize(MidEditor *editor, float width, float height);

/* One pointer event. phase is a MID_PHASE_* value; others are ignored. */
void mid_editor_pointer(MidEditor *editor, int32_t phase, int32_t id, float x, float y);

/* Forget every touch in progress (app paused, or touches lost). */
void mid_editor_cancel_pointers(MidEditor *editor);

/* Advance one frame of dt seconds and build the paint list. */
void mid_editor_frame(MidEditor *editor, float dt);

/* The paint list built by the last mid_editor_frame. The pointer and the text
 * pool stay valid until the next call to mid_editor_frame or mid_editor_free. */
const MidPaintItem *mid_editor_paint_items(const MidEditor *editor, size_t *count);
const char *mid_editor_paint_text(const MidEditor *editor, size_t *len);

/* The version text shown in the top bar (kept to 24 characters). */
void mid_editor_set_version(MidEditor *editor, const char *utf8, size_t len);

/* Adds a line to the console log. */
void mid_editor_log(MidEditor *editor, const char *utf8, size_t len);

/* How many panics were caught inside the library since the editor was
 * created. Nonzero means something is wrong and the log has the details. */
uint32_t mid_editor_panic_count(const MidEditor *editor);

/* The library version as a NUL-terminated string. Never NULL. */
const char *mid_editor_version(void);

#ifdef __cplusplus
}
#endif

#endif /* MID_EDITOR_H */
