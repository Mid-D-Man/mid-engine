#!/usr/bin/env bash
# ============================================================================
# NOTICE: Full documentation, design decisions, and fix history for this file
# live in docs/mid-editor.md, section "scripts/editor_host_selftest.sh"
# ============================================================================
# Builds the editor's C host for a desktop, runs a scripted session on a software renderer (Xvfb and Mesa), and saves screenshots.
# It exercises the real host: the real C file, the real Rust static library through its C header, and raylib 6.0 (its desktop backend).
# What it cannot show is the phone: OpenGL ES 2 with the highp shader, touch input, the NDK link and the APK.
#
# Needs: a C compiler, make, cargo, git, xvfb-run, and the X11 and GL development packages (libx11-dev libxrandr-dev libxinerama-dev
# libxcursor-dev libxi-dev libgl-dev) plus libgl1-mesa-dri for the software renderer.
# Output: build/editor-selftest/shots/*.png. Exit 0 when the script ran and every expected screenshot was written.
set -euo pipefail
set -E
trap 'printf "error: %s stopped at line %s: %s\n" "$(basename "$0")" "$LINENO" "$BASH_COMMAND" >&2' ERR

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

RAYLIB_TAG="${EDITOR_RAYLIB_TAG:-6.0}"
OUT="${EDITOR_SELFTEST_OUT:-$root/build/editor-selftest}"
RAYLIB_DIR="${EDITOR_RAYLIB_DIR:-$OUT/raylib}"
HOST="$root/apps/editor-android"
FFI="$root/crates/mid-editor-ffi"
SHOTS="$OUT/shots"
EXPECTED=(01_initial 02_hierarchy_selected 03_inspector 04_console 05_console_1000_cubes 06_view_2d 07_play_mode 08_orbited)

log() { printf '\n== %s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

for tool in cc make cargo git xvfb-run; do command -v "$tool" >/dev/null 2>&1 || die "$tool not found"; done
mkdir -p "$OUT" "$SHOTS"
rm -f "$SHOTS"/*.png

log "raylib $RAYLIB_TAG for the desktop"
if [ ! -d "$RAYLIB_DIR/src" ]; then
    git clone --depth 1 --branch "$RAYLIB_TAG" https://github.com/raysan5/raylib.git "$RAYLIB_DIR"
fi
if [ ! -f "$RAYLIB_DIR/src/libraylib.a" ]; then
    make -C "$RAYLIB_DIR/src" PLATFORM=PLATFORM_DESKTOP RAYLIB_LIBTYPE=STATIC -j"$(nproc)"
fi
[ -f "$RAYLIB_DIR/src/libraylib.a" ] || die "raylib did not produce libraylib.a"

log "the Rust core (mid-editor-ffi, release)"
cargo build --release -p mid-editor-ffi
LIB="${CARGO_TARGET_DIR:-$root/target}/release/libmid_editor_ffi.a"
[ -f "$LIB" ] || die "$LIB is missing"

log "the C host with -Wall -Wextra"
cc -std=gnu99 -O1 -Wall -Wextra -Wno-unused-parameter -DPLATFORM_DESKTOP -DEDITOR_SELFTEST -DEDITOR_VERSION='"selftest"' \
    -I"$RAYLIB_DIR/src" -I"$FFI/include" "$HOST/main.c" "$LIB" "$RAYLIB_DIR/src/libraylib.a" \
    -lGL -lm -lpthread -ldl -lrt -lX11 -o "$OUT/editor-host-selftest"

log "the scripted session on a software renderer"
(
    cd "$HOST"
    EDITOR_SHOTS="$SHOTS" LIBGL_ALWAYS_SOFTWARE=1 xvfb-run -a -s "-screen 0 1024x1400x24" "$OUT/editor-host-selftest"
) 2>&1 | tail -25

missing=""
for s in "${EXPECTED[@]}"; do
    if [ ! -s "$SHOTS/$s.png" ]; then missing="$missing $s"; fi
done
[ -z "$missing" ] || die "screenshots missing:$missing"
ls -la "$SHOTS"
echo "self-test passed: ${#EXPECTED[@]} screenshots in $SHOTS"
