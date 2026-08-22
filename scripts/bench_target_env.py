# scripts/bench_target_env.py
# Maps a `target_cpu` workflow-dispatch choice to the RUSTFLAGS / cargo
# env vars a bench workflow needs, replacing the inline bash if/elif
# chain that used to live in each bench yml separately (originally in
# A-mid-math — bench vs all's "Configure target environment" step).
#
# Usage in a workflow step:
#   - name: Configure target environment
#     run: python3 scripts/bench_target_env.py "${{ inputs.target_cpu }}" >> "$GITHUB_ENV"
#
# Prints KEY=VALUE lines, one per line, in the exact format GITHUB_ENV
# expects. Always prints all four vars (empty string for any that don't
# apply to the selected target) so downstream steps can reference
# $BENCH_TARGET / $BENCH_FEATURES / $ACTIVE_CPU_FLAGS unconditionally.
#
#   target_cpu=wasm     -> RUSTFLAGS=-C target-feature=+simd128
#                          BENCH_TARGET=--target wasm32-wasip1
#   target_cpu=neon     -> RUSTFLAGS=-C target-cpu=native
#                          (native aarch64 runner, host==target, safe
#                          as plain RUSTFLAGS — no cross-compile
#                          host/target mismatch risk here)
#   target_cpu=scalar   -> BENCH_FEATURES=--features force-scalar
#   otherwise           -> CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUSTFLAGS
#                          (NOT plain RUSTFLAGS — RUSTFLAGS bleeds into
#                          host build-script compilation too; a runner
#                          without the requested ISA would SIGILL a
#                          build script. CARGO_TARGET_*_RUSTFLAGS only
#                          applies to the target binary.)
#
# Same env var names/semantics as A-mid-math — bench vs all's inline
# version, so summary/diagnostic steps written against that workflow's
# env vars work unmodified against any workflow using this script.

import sys

if len(sys.argv) != 2:
    print("usage: bench_target_env.py <target_cpu>", file=sys.stderr)
    sys.exit(1)

target_cpu = sys.argv[1]

env = {
    "RUSTFLAGS": "",
    "CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUSTFLAGS": "",
    "BENCH_TARGET": "",
    "BENCH_FEATURES": "",
    "ACTIVE_CPU_FLAGS": "",
}

if target_cpu == "wasm":
    env["RUSTFLAGS"] = "-C target-feature=+simd128"
    env["BENCH_TARGET"] = "--target wasm32-wasip1"
    env["ACTIVE_CPU_FLAGS"] = "-C target-feature=+simd128"
elif target_cpu == "neon":
    env["RUSTFLAGS"] = "-C target-cpu=native"
    env["ACTIVE_CPU_FLAGS"] = "-C target-cpu=native"
elif target_cpu == "scalar":
    env["BENCH_FEATURES"] = "--features force-scalar"
else:
    env["CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUSTFLAGS"] = f"-C target-cpu={target_cpu}"
    env["ACTIVE_CPU_FLAGS"] = f"-C target-cpu={target_cpu}"

for key, value in env.items():
    print(f"{key}={value}")
