#!/usr/bin/env bash
# ============================================================================
# NOTICE: Full documentation, design decisions, and fix history for this file
# live in docs/mid-editor.md, section "scripts/tests/editor_android_script_check.sh"
# ============================================================================
# Runs scripts/build_editor_android.sh against a stand-in NDK and a stand-in SDK, so its own logic is exercised without Android: the host's
# gcc answers to the NDK's compiler names (and ignores the Android-only link flags), a tiny library stands in for raylib, the Rust stage builds
# for the host's own triple, and the packaging stage uses the real aapt, zipalign and apksigner (the Ubuntu builds, apt packages aapt zipalign
# apksigner and android-sdk-platform-23). It checks the script's success path and the ways it is meant to fail.
# What it cannot check: the real NDK, raylib's Android build, the ARM link, or anything on a phone.
# Needs: gcc, nm, readelf, ar, cargo, keytool, unzip, and the packages above.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
script="$root/scripts/build_editor_android.sh"
raylib_headers="${EDITOR_CHECK_RAYLIB_SRC:-}"
if [ -z "$raylib_headers" ] || [ ! -f "$raylib_headers/raylib.h" ]; then
    echo "set EDITOR_CHECK_RAYLIB_SRC to a raylib src directory (for raylib.h)" >&2
    exit 2
fi

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
pass=0
fail=0
ok() { pass=$((pass + 1)); printf 'ok    %s\n' "$1"; }
bad() { fail=$((fail + 1)); printf 'FAIL  %s\n' "$1"; }
check() { local name="$1"; shift; if "$@" >/dev/null 2>&1; then ok "$name"; else bad "$name"; fi; }

# ---- the stand-ins -------------------------------------------------------------------------------------------------------------------------
ndk="$work/ndk"
bin="$ndk/toolchains/llvm/prebuilt/linux-x86_64/bin"
mkdir -p "$bin"
cat > "$bin/armv7a-linux-androideabi29-clang" <<'WRAP'
#!/usr/bin/env bash
# gcc, minus the flags only Android's linker knows, plus whatever FAKE_LINK_EXTRA adds.
args=()
for a in "$@"; do
    case "$a" in
        -Wl,--no-undefined|-Wl,--wrap=fopen|-Wl,--warn-shared-textrel|-llog|-landroid|-lEGL|-lGLESv2|-lOpenSLES) ;;
        *) args+=("$a") ;;
    esac
done
# shellcheck disable=SC2086
exec gcc "${args[@]}" ${FAKE_LINK_EXTRA:-}
WRAP
chmod +x "$bin/armv7a-linux-androideabi29-clang"
ln -s "$(command -v nm)" "$bin/llvm-nm"
ln -s "$(command -v readelf)" "$bin/llvm-readelf"
ln -s "$(command -v ar)" "$bin/llvm-ar"

sdk="$work/sdk"
mkdir -p "$sdk/build-tools/34.0.0" "$sdk/platforms/android-34"
for t in aapt zipalign apksigner; do ln -s "$(readlink -f "$(command -v "$t")")" "$sdk/build-tools/34.0.0/$t"; done
ln -s /usr/lib/android-sdk/platforms/android-23/android.jar "$sdk/platforms/android-34/android.jar"

# A raylib directory with the real headers and a library that defines only the entry point the link insists on.
fake_raylib() {
    local out="$1"
    mkdir -p "$out/raylib/src"
    cp "$raylib_headers/raylib.h" "$out/raylib/src/"
    printf 'void ANativeActivity_onCreate(void *a, void *b, unsigned long c) { (void)a; (void)b; (void)c; }\n' > "$work/entry.c"
    gcc -c -fPIC "$work/entry.c" -o "$work/entry.o"
    ar rcs "$out/raylib/src/libraylib.a" "$work/entry.o"
}

host_libs="libm.so.6 libc.so.6 ld-linux-x86-64.so.2 libgcc_s.so.1"

run() {  # run <out dir> <stage...>   (extra environment comes from the caller)
    local out="$1"; shift
    ANDROID_NDK_HOME="$ndk" ANDROID_HOME="$sdk" EDITOR_OUT="$out" EDITOR_RUST_TRIPLE=x86_64-unknown-linux-gnu \
        EDITOR_SKIP_ELF_CHECK=1 EDITOR_VERSION_CODE=7 EDITOR_VERSION_NAME=9.8.7 EDITOR_EXTRA_NEEDED_OK="$host_libs" \
        bash "$script" "$@"
}

out="$work/out"
fake_raylib "$out"

# ---- the success path ----------------------------------------------------------------------------------------------------------------------
if run "$out" editor compile link package > "$work/run1.log" 2>&1; then ok "editor, compile, link and package run to the end"; else bad "the stages ran to the end"; tail -20 "$work/run1.log"; fi
apk="$out/mid-editor-armv7-debug.apk"
check "the APK exists" test -s "$apk"
badging="$work/badging.txt"
aapt dump badging "$apk" > "$badging" 2>&1 || true
check "package name is com.midmanstudio.mideditor" grep -q "package: name='com.midmanstudio.mideditor'" "$badging"
check "version code and name are filled in" grep -q "versionCode='7' versionName='9.8.7'" "$badging"
check "min and target SDK are filled in" grep -q "sdkVersion:'29'" "$badging"
check "target SDK 34" grep -q "targetSdkVersion:'34'" "$badging"
check "the launcher activity is NativeActivity" grep -q "launchable-activity: name='android.app.NativeActivity'" "$badging"
check "the native code is armeabi-v7a" grep -q "native-code: 'armeabi-v7a'" "$badging"
check "the manifest asks for OpenGL ES 2" grep -q "0x20000" "$badging"
check "the manifest asks for no permission" bash -c "! grep -q 'uses-permission' '$badging'"
listing="$work/listing.txt"
unzip -l "$apk" > "$listing"
check "the APK holds lib/armeabi-v7a/libmain.so" grep -q "lib/armeabi-v7a/libmain.so" "$listing"
check "the font is at assets/assets/fonts/DejaVuSans.ttf, so the host's path works" grep -q "assets/assets/fonts/DejaVuSans.ttf" "$listing"
check "the font's license ships with it" grep -q "assets/assets/fonts/DejaVu-LICENSE.txt" "$listing"
check "the APK signature verifies" bash -c "apksigner verify '$apk'"
dyn="$(nm -D --defined-only "$out/stage/lib/armeabi-v7a/libmain.so")"
check "libmain.so exports ANativeActivity_onCreate" grep -q "ANativeActivity_onCreate" <<<"$dyn"
check "libmain.so contains the editor core" grep -q "mid_editor_new" <<<"$dyn"
check "the log names the matching header" grep -q "mid_editor.h and the library agree" "$work/run1.log"

# ---- a second run keeps the signing key ----------------------------------------------------------------------------------------------------
cert1="$(apksigner verify --print-certs "$apk" 2>/dev/null | grep -m1 'SHA-256' || true)"
run "$out" package > "$work/run2.log" 2>&1 || true
cert2="$(apksigner verify --print-certs "$apk" 2>/dev/null | grep -m1 'SHA-256' || true)"
check "a second run reuses the debug keystore" grep -q "using the existing debug keystore" "$work/run2.log"
check "and signs with the same certificate" test -n "$cert1" -a "$cert1" = "$cert2"

# ---- defaults ------------------------------------------------------------------------------------------------------------------------------
ANDROID_NDK_HOME="$ndk" ANDROID_HOME="$sdk" EDITOR_OUT="$out" EDITOR_EXTRA_NEEDED_OK="$host_libs" bash "$script" package > "$work/run3.log" 2>&1 || true
if aapt dump badging "$apk" 2>/dev/null | grep -q "versionCode='1' versionName='0.1.0'"; then ok "version defaults are 1 and 0.1.0"; else bad "version defaults are 1 and 0.1.0"; fi

# ---- the ways it is meant to fail ----------------------------------------------------------------------------------------------------------
fails() {  # fails <message fragment> <command...>: the command must exit nonzero and say the fragment
    local frag="$1"; shift
    local log="$work/fail.log"
    if "$@" > "$log" 2>&1; then return 1; fi
    grep -q -- "$frag" "$log"
}
check "no NDK is an error that says so" fails "ANDROID_NDK_HOME is not set" env -u ANDROID_NDK_HOME -u NDK_HOME bash "$script" compile
check "no SDK is an error that says so" fails "ANDROID_HOME is not set" env -u ANDROID_HOME -u ANDROID_SDK_ROOT ANDROID_NDK_HOME="$ndk" bash "$script" package
mkdir -p "$work/nosdk"
check "a missing build tool is named" fails "missing" env ANDROID_NDK_HOME="$ndk" ANDROID_HOME="$work/nosdk" EDITOR_OUT="$out" bash "$script" package
check "an unknown stage is refused" fails "unknown stage 'wibble'" run "$out" wibble
check "link before the editor stage says so" fails "not built yet" run "$work/empty" link
mkdir -p "$work/noheaders/raylib/src"
check "compile without raylib.h says so" fails "raylib.h not found" run "$work/noheaders" compile

# A header that declares a function the library lacks, and a library function the header lacks.
sed 's|mid_editor_version(void);|mid_editor_version(void);\nvoid mid_editor_missing_one(void);|' "$root/crates/mid-editor-ffi/include/mid_editor.h" > "$work/header_extra.h"
check "a header function the library lacks is caught" fails "lacks functions that mid_editor.h declares: mid_editor_missing_one" env EDITOR_HEADER="$work/header_extra.h" EDITOR_TARGET_DIR="$work/tgt1" bash -c "ANDROID_NDK_HOME='$ndk' ANDROID_HOME='$sdk' EDITOR_OUT='$work/o2' EDITOR_RUST_TRIPLE=x86_64-unknown-linux-gnu bash '$script' editor"
grep -v 'mid_editor_log' "$root/crates/mid-editor-ffi/include/mid_editor.h" > "$work/header_short.h"
check "a library function the header lacks is caught" fails "exports functions that mid_editor.h does not declare: mid_editor_log" env EDITOR_HEADER="$work/header_short.h" EDITOR_TARGET_DIR="$work/tgt1" bash -c "ANDROID_NDK_HOME='$ndk' ANDROID_HOME='$sdk' EDITOR_OUT='$work/o3' EDITOR_RUST_TRIPLE=x86_64-unknown-linux-gnu bash '$script' editor"

# A link that pulls in a library the phone does not have.
check "a library the phone lacks is refused" fails "needs libraries the phone does not have" env FAKE_LINK_EXTRA="-Wl,--no-as-needed -lcrypt" bash -c "ANDROID_NDK_HOME='$ndk' ANDROID_HOME='$sdk' EDITOR_OUT='$out' EDITOR_RUST_TRIPLE=x86_64-unknown-linux-gnu EDITOR_SKIP_ELF_CHECK=1 EDITOR_EXTRA_NEEDED_OK='$host_libs' bash '$script' link"

# Code for another architecture must not be packaged (here: the x86-64 library the stand-in toolchain makes).
check "a library that is not 32-bit ARM is refused" fails "is not a 32-bit ELF file" env ANDROID_NDK_HOME="$ndk" ANDROID_HOME="$sdk" EDITOR_OUT="$out" EDITOR_RUST_TRIPLE=x86_64-unknown-linux-gnu EDITOR_EXTRA_NEEDED_OK="$host_libs" bash "$script" link

echo
echo "$pass passed, $fail failed"
[ "$fail" -eq 0 ]
