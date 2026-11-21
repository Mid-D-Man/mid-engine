#!/usr/bin/env bash
# ============================================================================
# NOTICE: Full documentation, design decisions, and fix history for this file
# live in docs/mid-editor.md, section "scripts/build_editor_android.sh"
# ============================================================================
# Builds the Mid Engine editor as an Android APK for armeabi-v7a (the 32-bit ARM ABI a Galaxy A13 installs), with no Gradle and no Java code:
#   raylib   raylib 6.0 from source for Android (the NDK's clang)
#   editor   crates/mid-editor-ffi as a Rust static library for armv7-linux-androideabi
#   compile  apps/editor-android/main.c to an object file
#   link     libmain.so (the object, the Rust library, raylib) and checks on it
#   package  AndroidManifest.xml, assets, aapt, zipalign, a debug signature
# The shape follows the Tower of Babylon repository's scripts/build_android.sh, which builds the same way for the same phone.
#
# Usage: scripts/build_editor_android.sh [stage ...]    (no argument: every stage, in order)
# Needs: ANDROID_NDK_HOME (r26d is what the other projects use) and ANDROID_HOME with build-tools 34.0.0 and platforms/android-34, a JDK
# (keytool), git, make, rustup with the armv7-linux-androideabi target for the toolchain this repository names, and unzip.
# Output: build/editor-android/mid-editor-armv7-debug.apk
set -euo pipefail
# If any command stops the script (set -e), say which one and where: a silent exit 1 costs an hour.
set -E
trap 'printf "error: %s stopped at line %s: %s\n" "$(basename "$0")" "$LINENO" "$BASH_COMMAND" >&2' ERR

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

API="${EDITOR_ANDROID_API:-29}"
TARGET_SDK="${EDITOR_ANDROID_TARGET_SDK:-34}"
BUILD_TOOLS="${EDITOR_ANDROID_BUILD_TOOLS:-34.0.0}"
PLATFORM="${EDITOR_ANDROID_PLATFORM:-34}"
RAYLIB_TAG="${EDITOR_RAYLIB_TAG:-6.0}"
OPT="${EDITOR_OPT:--O2}"
VERSION_CODE="${EDITOR_VERSION_CODE:-1}"
VERSION_NAME="${EDITOR_VERSION_NAME:-0.1.0}"
ABI="armeabi-v7a"
TRIPLE="${EDITOR_RUST_TRIPLE:-armv7-linux-androideabi}"
CLANG_TRIPLE="${EDITOR_CLANG_TRIPLE:-armv7a-linux-androideabi}"
OUT="${EDITOR_OUT:-$root/build/editor-android}"
HOST="$root/apps/editor-android"
FFI="$root/crates/mid-editor-ffi"
HEADER="${EDITOR_HEADER:-$FFI/include/mid_editor.h}"
RUSTLIB="$OUT/libmid_editor_ffi.a"
APK_NAME="mid-editor-armv7-debug.apk"

NDK="${ANDROID_NDK_HOME:-${NDK_HOME:-}}"
SDK="${ANDROID_HOME:-${ANDROID_SDK_ROOT:-}}"

log() { printf '\n== %s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }
# summary: a line on the run's summary page (Actions only). It ends in "return 0" on purpose: a function whose last command is a failed
# test would trip "set -e".
summary() {
    if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then printf '%s\n' "$*" >> "$GITHUB_STEP_SUMMARY"; fi
    return 0
}

require_tools() {
    if [ -z "$NDK" ] || [ ! -d "$NDK" ]; then
        die "ANDROID_NDK_HOME is not set to an existing NDK directory"
    fi
    TOOLCHAIN="$NDK/toolchains/llvm/prebuilt/linux-x86_64"
    [ -d "$TOOLCHAIN" ] || die "no linux-x86_64 toolchain under $NDK"
    CC="$TOOLCHAIN/bin/${CLANG_TRIPLE}${API}-clang"
    AR="$TOOLCHAIN/bin/llvm-ar"
    NM="$TOOLCHAIN/bin/llvm-nm"
    READELF="$TOOLCHAIN/bin/llvm-readelf"
    [ -x "$CC" ] || die "missing compiler $CC (is API $API supported by this NDK?)"
}

require_sdk() {
    if [ -z "$SDK" ] || [ ! -d "$SDK" ]; then
        die "ANDROID_HOME is not set to an existing SDK directory"
    fi
    BT="$SDK/build-tools/$BUILD_TOOLS"
    ANDROID_JAR="$SDK/platforms/android-$PLATFORM/android.jar"
    local f
    for f in "$BT/aapt" "$BT/zipalign" "$BT/apksigner" "$ANDROID_JAR"; do
        [ -e "$f" ] || die "missing $f (install build-tools;$BUILD_TOOLS and platforms;android-$PLATFORM)"
    done
}

# raylib is built from source for Android, because a libraylib.a from anywhere else is for another platform. The headers the host
# compiles against come from the same checkout, so the library and the headers can never disagree.
stage_raylib() {
    log "raylib $RAYLIB_TAG for Android armv7 (API $API)"
    require_tools
    if [ ! -d "$OUT/raylib" ]; then
        mkdir -p "$OUT"
        git clone --depth 1 --branch "$RAYLIB_TAG" https://github.com/raysan5/raylib.git "$OUT/raylib"
    fi
    make -C "$OUT/raylib/src" clean >/dev/null 2>&1 || true
    make -C "$OUT/raylib/src" \
        PLATFORM=PLATFORM_ANDROID \
        ANDROID_NDK="$NDK" \
        ANDROID_ARCH=arm \
        ANDROID_API_VERSION="$API" \
        RAYLIB_LIBTYPE=STATIC \
        -j"$(nproc)"
    [ -f "$OUT/raylib/src/libraylib.a" ] || die "raylib did not produce libraylib.a"
    ls -la "$OUT/raylib/src/libraylib.a"
}

# The names of the functions the header declares, and the ones the library defines. The NDK's llvm-nm is older than the LLVM inside rustc and
# exits 1 with "Unknown attribute kind" on some members of a Rust static library (the bitcode rustc embeds) while still listing every symbol
# it can read, so its exit status and complaints are ignored (kept in editor-nm.log). The Tower of Babylon build found this out the hard way:
# the exit status stopped the whole script, silently.
check_header_matches_library() {
    local lib="$1" syms fn missing="" extra=""
    syms="$("$NM" --defined-only "$lib" 2>"$OUT/editor-nm.log" || true)"
    if [ -z "$syms" ]; then
        echo "warning: llvm-nm could not read $lib (build/editor-android/editor-nm.log): the check against mid_editor.h is skipped" >&2
        return 0
    fi
    for fn in $(grep -o 'mid_editor_[a-z_]*(' "$HEADER" | tr -d '(' | sort -u); do
        if ! grep -qw "$fn" <<<"$syms"; then missing="$missing $fn"; fi
    done
    [ -z "$missing" ] || die "the library lacks functions that mid_editor.h declares:$missing"
    while IFS= read -r fn; do
        if [ -n "$fn" ] && ! grep -q "${fn}(" "$HEADER"; then extra="$extra $fn"; fi
    done < <(grep -o ' T mid_editor_[a-z_]*' <<<"$syms" | sed 's/ T //' | sort -u)
    [ -z "$extra" ] || die "the library exports functions that mid_editor.h does not declare:$extra"
    echo "mid_editor.h and the library agree"
}

# Builds only the static library (cargo rustc --crate-type staticlib). The NDK's clang is handed to cargo through the variables cargo and the
# cc crate read, which is all cargo-ndk would do for a crate with no C dependencies. The editor crates have no dependencies outside this
# repository, so there is nothing to pin and no network is needed beyond cargo's index for the workspace.
stage_editor() {
    log "mid-editor-ffi static library for $TRIPLE"
    require_tools
    rm -f "$RUSTLIB"
    mkdir -p "$OUT"
    local tool
    for tool in cargo rustc; do command -v "$tool" >/dev/null 2>&1 || die "$tool not found (install Rust, for example with rustup)"; done
    echo "$(rustc --version 2>&1), $(cargo --version 2>&1)"
    local sysroot
    sysroot="$(rustc --print sysroot)"
    if [ ! -d "$sysroot/lib/rustlib/$TRIPLE/lib" ]; then
        echo "rust target $TRIPLE is not installed, adding it"
        command -v rustup >/dev/null 2>&1 || die "the rust target $TRIPLE is not installed and rustup is not available to add it"
        rustup target add "$TRIPLE"
    fi

    local tdir="${EDITOR_TARGET_DIR:-$OUT/target}" upper lower
    upper="$(tr 'a-z-' 'A-Z_' <<<"$TRIPLE")"
    lower="${TRIPLE//-/_}"
    mkdir -p "$tdir"
    echo "cargo rustc, release, $TRIPLE (full log in build/editor-android/editor.log)"
    (
        export "CC_${lower}=$CC"
        export "AR_${lower}=$AR"
        export "CARGO_TARGET_${upper}_LINKER=$CC"
        export ANDROID_NDK_HOME="$NDK"
        export CARGO_TARGET_DIR="$tdir"
        cargo rustc -p mid-editor-ffi --lib --crate-type staticlib --release --target "$TRIPLE"
    ) 2>&1 | tee "$OUT/editor.log"
    local built="$tdir/$TRIPLE/release/libmid_editor_ffi.a"
    [ -f "$built" ] || die "cargo finished but $built is missing (see build/editor-android/editor.log)"
    check_header_matches_library "$built"
    cp "$built" "$RUSTLIB"
    ls -la "$RUSTLIB"
    summary "- Rust core: built for $TRIPLE, libmid_editor_ffi.a $(du -h "$RUSTLIB" | cut -f1) before linking"
}

stage_compile() {
    log "the host as an armv7 Android object"
    require_tools
    [ -f "$OUT/raylib/src/raylib.h" ] || die "raylib.h not found, run the raylib stage first"
    rm -rf "$OUT/obj"
    mkdir -p "$OUT/obj"
    # Notes on the flags:
    #  -I$OUT/raylib/src   the headers of the raylib that was just built.
    #  -DGRAPHICS_API_OPENGL_ES2 -DPLATFORM_ANDROID   what raylib itself was built with; rlgl.h sizes things by them, and the host picks the
    #                      highp shader by the first.
    #  -Werror=implicit-function-declaration   a raylib call this raylib does not have fails here, not at the link.
    local cflags=(-std=gnu99 "$OPT" -fPIC -ffunction-sections -fdata-sections -funwind-tables -fstack-protector-strong
                  -Wall -Wextra -Wno-unused-parameter -Werror=implicit-function-declaration
                  -DPLATFORM_ANDROID -DGRAPHICS_API_OPENGL_ES2 "-DEDITOR_VERSION=\"$VERSION_NAME\""
                  -I"$OUT/raylib/src" -I"$FFI/include")
    "$CC" "${cflags[@]}" -c "$HOST/main.c" -o "$OUT/obj/main.o"
    echo "1 source compiled"
}

# Flags follow raylib's own Android template. The NDK's native app glue is already inside libraylib.a, and -u ANativeActivity_onCreate pulls
# it in. raylib's android_main calls the host's main() directly. --wrap=fopen is required by raylib: its Android backend redirects fopen to
# read from the APK's assets, and the wrapper calls __real_fopen, which only exists when the final link passes this flag. --no-undefined
# makes missing symbols fail here instead of when the app starts. --strip-debug drops the debug info of the parts of Rust's standard library
# that rustc ships precompiled; the host itself is built without -g.
link_so() {
    "$CC" -shared -o "$OUT/stage/lib/$ABI/libmain.so" \
        "$OUT/obj/main.o" \
        "$RUSTLIB" \
        -Wl,--strip-debug \
        "$OUT/raylib/src/libraylib.a" \
        -u ANativeActivity_onCreate \
        -Wl,--wrap=fopen \
        -Wl,-soname,libmain.so -Wl,--exclude-libs,libatomic.a -Wl,--build-id \
        -Wl,--no-undefined -Wl,--gc-sections \
        -Wl,-z,noexecstack -Wl,-z,relro -Wl,-z,now -Wl,--warn-shared-textrel \
        -llog -landroid -lEGL -lGLESv2 -lOpenSLES -lm -ldl -lc
}

# After the link: Rust must be inside libmain.so, and must not have pulled in a library the phone lacks (libc++_shared.so and libunwind.so are
# not in the APK and not on the phone: the app would fail to start with a message nobody sees). Only the system libraries named in the link
# line are allowed.
so_ok() {
    local so="$OUT/stage/lib/$ABI/libmain.so" all needed n bad=""
    all="$("$NM" --defined-only "$so" 2>/dev/null || true)"
    grep -qw mid_editor_new <<<"$all" || die "libmain.so does not contain the editor core (no mid_editor_new)"
    needed="$("$READELF" -d "$so" | grep NEEDED | sed 's/.*\[\(.*\)\]/\1/' | tr '\n' ' ')"
    # EDITOR_EXTRA_NEEDED_OK is empty in a real build. The script's own test sets it, because a stand-in toolchain on a PC links against the PC's libraries.
    for n in $needed; do
        case "$n" in
            libc.so|libm.so|libdl.so|liblog.so|libandroid.so|libEGL.so|libGLESv2.so|libOpenSLES.so|libstdc++.so) ;;
            *)
                local allowed=0 extra
                for extra in ${EDITOR_EXTRA_NEEDED_OK:-}; do
                    if [ "$n" = "$extra" ]; then allowed=1; fi
                done
                if [ "$allowed" -eq 0 ]; then bad="$bad $n"; fi
                ;;
        esac
    done
    [ -z "$bad" ] || die "libmain.so needs libraries the phone does not have:$bad"
    echo "needs: $needed"
}

# The phone installs only 32-bit ARM code. A library built for another architecture would install and then fail to start.
elf_ok() {
    local so="$OUT/stage/lib/$ABI/libmain.so" header
    if [ "${EDITOR_SKIP_ELF_CHECK:-0}" = "1" ]; then echo "ELF check skipped (EDITOR_SKIP_ELF_CHECK=1, for the script's own test)"; return 0; fi
    header="$("$READELF" -h "$so")"
    grep -q 'Class:.*ELF32' <<<"$header" || die "libmain.so is not a 32-bit ELF file"
    grep -q 'Machine:.*ARM' <<<"$header" || die "libmain.so is not built for ARM"
    echo "libmain.so is a 32-bit ARM shared object"
}

stage_link() {
    log "link libmain.so"
    require_tools
    [ -f "$RUSTLIB" ] || die "$RUSTLIB not built yet (run the editor stage)"
    [ -f "$OUT/obj/main.o" ] || die "main.o not built yet (run the compile stage)"
    mkdir -p "$OUT/stage/lib/$ABI"
    link_so 2>&1 | tee "$OUT/link.log"
    [ "${PIPESTATUS[0]}" -eq 0 ] || die "the link failed, see build/editor-android/link.log"
    ls -la "$OUT/stage/lib/$ABI/libmain.so"
    grep -q "ANativeActivity_onCreate" <<<"$("$NM" -D --defined-only "$OUT/stage/lib/$ABI/libmain.so")" || die "ANativeActivity_onCreate is not exported from libmain.so"
    echo "exported entry point: ANativeActivity_onCreate"
    so_ok
    elf_ok
}

stage_package() {
    log "package, align, sign"
    require_sdk
    command -v keytool >/dev/null 2>&1 || die "keytool not found, install a JDK"
    local stage="$OUT/stage" unsigned="$OUT/unsigned.apk" aligned="$OUT/aligned.apk"
    local ks="$OUT/debug.keystore" apk="$OUT/$APK_NAME"
    [ -f "$stage/lib/$ABI/libmain.so" ] || die "libmain.so not built yet (run the link stage)"

    sed -e "s/@MIN_SDK@/$API/" -e "s/@TARGET_SDK@/$TARGET_SDK/" \
        -e "s/@VERSION_CODE@/$VERSION_CODE/" -e "s/@VERSION_NAME@/$VERSION_NAME/" \
        "$HOST/AndroidManifest.xml" > "$stage/AndroidManifest.xml"

    # The host opens its files as "assets/fonts/DejaVuSans.ttf". On Android raylib reads them from the APK's asset folder, whose root is the
    # folder given to aapt, so the host's assets/ goes in as assets/assets/ and every path in the code works as written.
    rm -rf "$stage/assets"
    mkdir -p "$stage/assets"
    cp -r "$HOST/assets" "$stage/assets/assets"

    rm -f "$unsigned" "$aligned" "$apk"
    "$BT/aapt" package -f -M "$stage/AndroidManifest.xml" -I "$ANDROID_JAR" \
        -A "$stage/assets" -F "$unsigned"
    (cd "$stage" && "$BT/aapt" add "$unsigned" "lib/$ABI/libmain.so")
    "$BT/zipalign" -f -p 4 "$unsigned" "$aligned"

    # A throwaway debug key, no secrets involved. A phone only updates an installed app in place when the new APK is signed with the same
    # key, so the workflow keeps this file in the Actions cache between runs.
    if [ ! -f "$ks" ]; then
        echo "no debug keystore yet, making one (the first APK signed with it needs an older install of the app removed)"
        keytool -genkeypair -keystore "$ks" -storepass android -keypass android \
            -alias mideditordebug -keyalg RSA -keysize 2048 -validity 10000 \
            -dname "CN=Mid Editor Debug"
    else
        echo "using the existing debug keystore"
    fi
    "$BT/apksigner" sign --ks "$ks" --ks-pass pass:android --key-pass pass:android \
        --ks-key-alias mideditordebug --out "$apk" "$aligned"
    "$BT/apksigner" verify --verbose "$apk"
    ls -la "$apk"
    "$BT/aapt" dump badging "$apk" | grep -E "^package|^sdkVersion|^targetSdkVersion|^launchable-activity|^native-code" || true
    echo "contents:"
    unzip -l "$apk" | grep -E "AndroidManifest|libmain|DejaVuSans" || true
    summary "- **APK:** $APK_NAME, $(du -h "$apk" | cut -f1), version $VERSION_NAME (code $VERSION_CODE)"
}

stages=("$@")
if [ "${#stages[@]}" -eq 0 ]; then
    stages=(raylib editor compile link package)
fi
if [ "${stages[0]}" != "editor_inner" ]; then
    summary "### Mid Editor, Android armv7 build"
fi
for s in "${stages[@]}"; do
    case "$s" in
        raylib|compile|link|package) "stage_$s" ;;
        editor)
            # The Rust stage runs as a child process, so that a command nobody expected to fail stops only that stage's own shell and the error
            # line above names it.
            bash "${BASH_SOURCE[0]}" editor_inner || die "the Rust stage failed, see build/editor-android/editor.log"
            ;;
        editor_inner) stage_editor ;;
        *) die "unknown stage '$s' (raylib, editor, compile, link, package)" ;;
    esac
done
# The Rust stage's child process has nothing to report: its parent does.
if [ "${stages[0]}" != "editor_inner" ]; then
    echo
    if [ -f "$OUT/$APK_NAME" ] && [[ " ${stages[*]} " == *" package "* ]]; then
        echo "done: $OUT/$APK_NAME"
    else
        echo "done: stages ${stages[*]}"
    fi
fi
