/* ============================================================================
 * NOTICE: Full documentation, design decisions, and fix history for this file
 * live in docs/mid-editor.md, section "apps/editor-android/main.c"
 * ============================================================================
 * The native host of the Mid Engine editor: a raylib window, the touch points,
 * a font, and a painter for the core's paint list. Everything the editor does
 * lives in the Rust core (crates/mid-editor) behind include/mid_editor.h.
 *
 * Built two ways: for Android by scripts/build_editor_android.sh (-DPLATFORM_ANDROID
 * -DGRAPHICS_API_OPENGL_ES2), and for a desktop self-test by
 * scripts/editor_host_selftest.sh (-DEDITOR_SELFTEST), which runs a scripted
 * session on a software renderer and saves screenshots.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "raylib.h"
#include "mid_editor.h"

#ifndef EDITOR_VERSION
#define EDITOR_VERSION "dev"
#endif

#define FONT_PATH "assets/fonts/DejaVuSans.ttf"
#define FONT_LOAD_SIZE 64
#define MAX_TOUCH 10
#define MAX_CLIPS 8
#define TEXT_BUF 512

#if defined(GRAPHICS_API_OPENGL_ES2)
/* raylib's default ES2 shaders with only the precision changed. Its vertex shader is mediump, which on this phone's GPU has about
 * three decimal digits: screen coordinates past 1000 land a pixel or two off. The game in the Tower of Babylon repository hit the same
 * thing (docs/phone-rendering.md there); this is the same fix. */
static const char *const VS_HIGHP =
    "#version 100                       \n"
    "precision highp float;             \n"
    "attribute vec3 vertexPosition;     \n"
    "attribute vec2 vertexTexCoord;     \n"
    "attribute vec4 vertexColor;        \n"
    "varying vec2 fragTexCoord;         \n"
    "varying vec4 fragColor;            \n"
    "uniform mat4 mvp;                  \n"
    "void main()                        \n"
    "{                                  \n"
    "    fragTexCoord = vertexTexCoord; \n"
    "    fragColor = vertexColor;       \n"
    "    gl_Position = mvp*vec4(vertexPosition, 1.0); \n"
    "}                                  \n";

static const char *const FS_HIGHP =
    "#version 100                       \n"
    "#ifdef GL_FRAGMENT_PRECISION_HIGH  \n"
    "precision highp float;             \n"
    "#else                              \n"
    "precision mediump float;           \n"
    "#endif                             \n"
    "varying vec2 fragTexCoord;         \n"
    "varying vec4 fragColor;            \n"
    "uniform sampler2D texture0;        \n"
    "uniform vec4 colDiffuse;           \n"
    "void main()                        \n"
    "{                                  \n"
    "    vec4 texelColor = texture2D(texture0, fragTexCoord); \n"
    "    gl_FragColor = texelColor*colDiffuse*fragColor;      \n"
    "}                                  \n";
#endif

static Font g_font;
static int g_font_ok = 0;

static Color color_of(uint32_t c) {
    Color out = { (unsigned char)((c >> 24) & 255), (unsigned char)((c >> 16) & 255), (unsigned char)((c >> 8) & 255), (unsigned char)(c & 255) };
    return out;
}

/* Copies UTF-8 bytes into a NUL-terminated buffer, cutting at the buffer's size. */
static void to_c_string(char *dst, size_t cap, const char *src, size_t len) {
    size_t n = len < cap - 1 ? len : cap - 1;
    memcpy(dst, src, n);
    dst[n] = '\0';
}

static float measure_cb(void *user, const char *text, size_t len, float size) {
    char buf[TEXT_BUF];
    (void)user;
    to_c_string(buf, sizeof buf, text, len);
    return MeasureTextEx(g_font, buf, size, 0.0f).x;
}

static void load_font(void) {
    g_font = LoadFontEx(FONT_PATH, FONT_LOAD_SIZE, NULL, 0);
    g_font_ok = g_font.texture.id != 0 && g_font.glyphCount > 0 && g_font.texture.id != GetFontDefault().texture.id;
    if (g_font_ok) {
        SetTextureFilter(g_font.texture, TEXTURE_FILTER_BILINEAR);
    } else {
        g_font = GetFontDefault();
    }
}

static void apply_clip(Rectangle r) {
    BeginScissorMode((int)floorf(r.x), (int)floorf(r.y), (int)ceilf(r.width) + 1, (int)ceilf(r.height) + 1);
}

static void draw_paint(const MidEditor *ed) {
    size_t n = 0, pool_len = 0;
    const MidPaintItem *items = mid_editor_paint_items(ed, &n);
    const char *pool = mid_editor_paint_text(ed, &pool_len);
    Rectangle clips[MAX_CLIPS];
    int depth = 0;
    char buf[TEXT_BUF];
    size_t i;

    for (i = 0; i < n; i++) {
        const MidPaintItem *it = &items[i];
        switch (it->kind) {
        case MID_PAINT_RECT: {
            Rectangle r = { it->a, it->b, it->c, it->d };
            if (it->extra > 0.5f && r.width > 0.0f && r.height > 0.0f) {
                float m = r.width < r.height ? r.width : r.height;
                float roundness = it->extra * 2.0f / m;
                DrawRectangleRounded(r, roundness > 1.0f ? 1.0f : roundness, 6, color_of(it->color));
            } else {
                DrawRectangleRec(r, color_of(it->color));
            }
            break;
        }
        case MID_PAINT_PUSH_CLIP: {
            Rectangle r = { it->a, it->b, it->c, it->d };
            if (depth < MAX_CLIPS) clips[depth] = r;
            depth++;
            apply_clip(r);
            break;
        }
        case MID_PAINT_POP_CLIP:
            if (depth > 0) depth--;
            EndScissorMode();
            if (depth > 0 && depth <= MAX_CLIPS) apply_clip(clips[depth - 1]);
            break;
        case MID_PAINT_LINE: {
            Vector2 p0 = { it->a, it->b }, p1 = { it->c, it->d };
            DrawLineEx(p0, p1, it->extra, color_of(it->color));
            break;
        }
        case MID_PAINT_TEXT: {
            Vector2 at = { it->a, it->b };
            if ((size_t)it->text_start + it->text_len <= pool_len) {
                to_c_string(buf, sizeof buf, pool + it->text_start, it->text_len);
                DrawTextEx(g_font, buf, at, it->extra, 0.0f, color_of(it->color));
            }
            break;
        }
        default:
            break;
        }
    }
    while (depth > 0) { depth--; EndScissorMode(); }
}

/* ---- touch: raylib reports the points that are down; the core wants down, move, up --------------------------------------------------- */
typedef struct { int id; Vector2 pos; } TouchRec;
static TouchRec g_prev[MAX_TOUCH];
static int g_prev_n = 0;

static int find_touch(const TouchRec *list, int n, int id) {
    int i;
    for (i = 0; i < n; i++) if (list[i].id == id) return i;
    return -1;
}

static void pump_touch(MidEditor *ed) {
    TouchRec cur[MAX_TOUCH];
    int n = GetTouchPointCount(), i;
    if (n > MAX_TOUCH) n = MAX_TOUCH;
    for (i = 0; i < n; i++) {
        cur[i].id = GetTouchPointId(i);
        cur[i].pos = GetTouchPosition(i);
    }
    for (i = 0; i < n; i++) {
        int was = find_touch(g_prev, g_prev_n, cur[i].id);
        if (was < 0) {
            mid_editor_pointer(ed, MID_PHASE_DOWN, cur[i].id, cur[i].pos.x, cur[i].pos.y);
        } else if (g_prev[was].pos.x != cur[i].pos.x || g_prev[was].pos.y != cur[i].pos.y) {
            mid_editor_pointer(ed, MID_PHASE_MOVE, cur[i].id, cur[i].pos.x, cur[i].pos.y);
        }
    }
    for (i = 0; i < g_prev_n; i++) {
        if (find_touch(cur, n, g_prev[i].id) < 0) {
            mid_editor_pointer(ed, MID_PHASE_UP, g_prev[i].id, g_prev[i].pos.x, g_prev[i].pos.y);
        }
    }
    memcpy(g_prev, cur, sizeof(TouchRec) * (size_t)n);
    g_prev_n = n;
}

#if defined(EDITOR_SELFTEST)
/* ---- the desktop self-test: a scripted session that taps by label and saves screenshots ----------------------------------------------- */
static int g_selftest_failed = 0;

static int find_label(const MidEditor *ed, const char *label, float *x, float *y) {
    size_t n = 0, pool_len = 0, i, want = strlen(label);
    const MidPaintItem *items = mid_editor_paint_items(ed, &n);
    const char *pool = mid_editor_paint_text(ed, &pool_len);
    for (i = 0; i < n; i++) {
        if (items[i].kind == MID_PAINT_TEXT && items[i].text_len == want && (size_t)items[i].text_start + want <= pool_len &&
            memcmp(pool + items[i].text_start, label, want) == 0) {
            *x = items[i].a + 4.0f;
            *y = items[i].b + items[i].extra * 0.5f;
            return 1;
        }
    }
    return 0;
}

static void tap_label(MidEditor *ed, const char *label) {
    float x, y;
    if (!find_label(ed, label, &x, &y)) {
        fprintf(stderr, "selftest: no text \"%s\" on screen\n", label);
        g_selftest_failed = 1;
        return;
    }
    mid_editor_pointer(ed, MID_PHASE_DOWN, 0, x, y);
    mid_editor_pointer(ed, MID_PHASE_UP, 0, x, y);
}

static void shot(const char *name) {
    const char *dir = getenv("EDITOR_SHOTS");
    char path[512];
    Image img;
    snprintf(path, sizeof path, "%s/%s.png", dir ? dir : ".", name);
    img = LoadImageFromScreen();
    if (!ExportImage(img, path)) { fprintf(stderr, "selftest: could not write %s\n", path); g_selftest_failed = 1; }
    UnloadImage(img);
}

/* Returns 1 when the script is over. Called once per frame, after the frame was built and painted, so a label found now is on screen. */
static int selftest_tick(MidEditor *ed, int frame) {
    float w = (float)GetScreenWidth(), h = (float)GetScreenHeight();
    switch (frame) {
    case 3:  shot("01_initial"); break;
    case 5:  tap_label(ed, "Cube.002"); break;
    case 8:  shot("02_hierarchy_selected"); break;
    case 10: tap_label(ed, "Inspector"); break;
    case 13: shot("03_inspector"); break;
    case 15: tap_label(ed, "Console"); break;
    case 18: shot("04_console"); break;
    case 20: tap_label(ed, "1000"); break;
    case 40: shot("05_console_1000_cubes"); break;
    case 42: tap_label(ed, "2D"); break;
    case 45: shot("06_view_2d"); break;
    case 47: tap_label(ed, "3D"); break;
    case 48: tap_label(ed, "Play"); break;
    case 80: shot("07_play_mode"); break;
    case 82: mid_editor_pointer(ed, MID_PHASE_DOWN, 0, w * 0.3f, h * 0.2f); break;
    case 83: case 84: case 85: case 86: case 87:
        mid_editor_pointer(ed, MID_PHASE_MOVE, 0, w * 0.3f + (float)(frame - 82) * 60.0f, h * 0.2f + (float)(frame - 82) * 10.0f);
        break;
    case 88: mid_editor_pointer(ed, MID_PHASE_UP, 0, w * 0.3f + 360.0f, h * 0.2f + 60.0f); break;
    case 95: shot("08_orbited"); break;
    case 96: return 1;
    default: break;
    }
    return 0;
}
#endif

int main(void) {
    MidEditor *ed;
    int w, h, last_w, last_h;
    char info[160];
#if defined(EDITOR_SELFTEST)
    int frame = 0;
#endif
#if defined(GRAPHICS_API_OPENGL_ES2)
    Shader shader;
    int shader_ok;
#endif

    SetConfigFlags(FLAG_VSYNC_HINT);
#if defined(PLATFORM_ANDROID)
    InitWindow(0, 0, "Mid Editor"); /* 0 by 0: raylib 6.0 uses the display's own resolution, with no letterbox bars */
#else
    InitWindow(540, 1204, "Mid Editor");
#endif
    SetTargetFPS(60);

    load_font();
#if defined(GRAPHICS_API_OPENGL_ES2)
    shader = LoadShaderFromMemory(VS_HIGHP, FS_HIGHP);
    shader_ok = shader.id != 0;
#endif

    w = last_w = GetScreenWidth();
    h = last_h = GetScreenHeight();
    ed = mid_editor_new((float)w, (float)h, measure_cb, NULL);
    if (ed == NULL) {
        TraceLog(LOG_ERROR, "mid_editor_new failed");
        CloseWindow();
        return 1;
    }
    mid_editor_set_version(ed, EDITOR_VERSION, strlen(EDITOR_VERSION));
    snprintf(info, sizeof info, "host: %dx%d, font %s, core %s", w, h, g_font_ok ? "DejaVu Sans" : "raylib default", mid_editor_version());
    mid_editor_log(ed, info, strlen(info));
#if defined(GRAPHICS_API_OPENGL_ES2)
    snprintf(info, sizeof info, "shader: %s", shader_ok ? "highp ES2" : "raylib default (mediump)");
    mid_editor_log(ed, info, strlen(info));
#endif

    while (!WindowShouldClose()) {
        w = GetScreenWidth();
        h = GetScreenHeight();
        if (w != last_w || h != last_h) {
            mid_editor_resize(ed, (float)w, (float)h);
            last_w = w;
            last_h = h;
        }
#if !defined(EDITOR_SELFTEST)
        pump_touch(ed);
#endif
        mid_editor_frame(ed, GetFrameTime());

        BeginDrawing();
        ClearBackground((Color){ 0x1B, 0x1B, 0x1F, 0xFF });
#if defined(GRAPHICS_API_OPENGL_ES2)
        if (shader_ok) BeginShaderMode(shader);
#endif
        draw_paint(ed);
#if defined(GRAPHICS_API_OPENGL_ES2)
        if (shader_ok) EndShaderMode();
#endif
#if defined(EDITOR_SELFTEST)
        if (selftest_tick(ed, frame++)) { EndDrawing(); break; }
#endif
        EndDrawing();
    }

    if (mid_editor_panic_count(ed) != 0) TraceLog(LOG_ERROR, "the editor core caught %u panic(s)", mid_editor_panic_count(ed));
    mid_editor_free(ed);
    if (g_font_ok) UnloadFont(g_font);
#if defined(GRAPHICS_API_OPENGL_ES2)
    if (shader_ok) UnloadShader(shader);
#endif
    CloseWindow();
#if defined(EDITOR_SELFTEST)
    return g_selftest_failed;
#else
    return 0;
#endif
}
