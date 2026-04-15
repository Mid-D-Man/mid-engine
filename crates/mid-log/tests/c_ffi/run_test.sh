#!/usr/bin/env bash
# run_test.sh — build the Rust cdylib, compile test.c, run it.
#
# Usage (from anywhere):
#   cd crates/mid-log/tests/c_ffi
#   chmod +x run_test.sh
#   ./run_test.sh
#
# Requires: cargo, clang (available on macOS via Xcode CLT)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CRATE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"      # crates/mid-log
WORKSPACE="$(cd "$CRATE_DIR/../.." && pwd)"        # mid-engine root
LIB_DIR="$WORKSPACE/target/debug"

GREEN="\033[0;32m"
YELLOW="\033[0;33m"
RED="\033[0;31m"
RESET="\033[0m"

echo -e "${YELLOW}→ building mid-log cdylib (debug)...${RESET}"
cd "$WORKSPACE"
cargo build -p mid-log
echo -e "${GREEN}✓ build ok${RESET}"

echo -e "${YELLOW}→ compiling test.c...${RESET}"
cd "$SCRIPT_DIR"

# macOS: .dylib, Linux: .so
if [[ "$(uname)" == "Darwin" ]]; then
    LIB_FILE="$LIB_DIR/libmid_log.dylib"
    clang test.c \
        -I "$CRATE_DIR/headers" \
        -L "$LIB_DIR" \
        -l mid_log \
        -rpath "$LIB_DIR" \
        -o test_ffi
else
    LIB_FILE="$LIB_DIR/libmid_log.so"
    clang test.c \
        -I "$CRATE_DIR/headers" \
        -L "$LIB_DIR" \
        -l mid_log \
        -Wl,-rpath,"$LIB_DIR" \
        -o test_ffi
fi

if [[ ! -f "$LIB_FILE" ]]; then
    echo -e "${RED}✗ library not found at $LIB_FILE${RESET}"
    exit 1
fi

echo -e "${GREEN}✓ compile ok${RESET}"

echo -e "${YELLOW}→ running test_ffi...${RESET}"
echo "──────────────────────────────────"
./test_ffi
echo "──────────────────────────────────"
echo -e "${GREEN}✓ FFI boundary verified${RESET}"

# Cleanup
rm -f test_ffi
