#!/usr/bin/env python3
# scripts/bench_run_narrow_int.py
#
# Narrow int vecs (IVec4, I8Vec4, etc.) span FOUR separate bench
# targets (vs_int8/vs_int16/vs_int32/vs_int64) — unlike wide-int, which
# is one file (vs_wide_int.rs) filterable by criterion name. There's no
# single `cargo bench` invocation that covers "just i16", so this
# script maps the workflow's `int_type` dropdown to the right set of
# --bench targets, runs each in turn, and concatenates their raw output
# into one file for scripts/bench_vs_all.py to summarize — same
# "running logic lives in a .py file, not the yml" split as
# scripts/bench_target_env.py.
#
# Usage:
#   python3 scripts/bench_run_narrow_int.py <int_type> <output_raw_file>
#
# int_type: all | i8 | i16 | i32 | i64
# Reads BENCH_TARGET / BENCH_FEATURES from the environment (already
# exported to GITHUB_ENV by scripts/bench_target_env.py in an earlier
# step) and forwards them to every `cargo bench` invocation.
#
# Exit code is non-zero if ANY target's bench run failed — a partial
# narrow-int pass (e.g. i8 compiles, i64 doesn't) should fail the job,
# not silently report partial results as success.

import os
import subprocess
import sys

TARGET_MAP = {
    "i8":  ["vs_int8"],
    "i16": ["vs_int16"],
    "i32": ["vs_int32"],
    "i64": ["vs_int64"],
    "all": ["vs_int8", "vs_int16", "vs_int32", "vs_int64"],
}


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: bench_run_narrow_int.py <int_type> <output_raw_file>", file=sys.stderr)
        return 1

    int_type, out_path = sys.argv[1], sys.argv[2]
    targets = TARGET_MAP.get(int_type)
    if targets is None:
        print(f"unknown int_type {int_type!r} — expected one of {sorted(TARGET_MAP)}", file=sys.stderr)
        return 1

    extra_args: list[str] = []
    bench_target = os.environ.get("BENCH_TARGET", "").strip()
    bench_features = os.environ.get("BENCH_FEATURES", "").strip()
    if bench_target:
        extra_args += bench_target.split()
    if bench_features:
        extra_args += bench_features.split()

    any_failed = False
    with open(out_path, "w") as out_file:
        for name in targets:
            cmd = ["cargo", "bench", "--bench", name, "-p", "mid-math", *extra_args]
            print(f"::group::cargo bench --bench {name}")
            print("+", " ".join(cmd))
            out_file.write(f"\n=== {name} ===\n")
            out_file.flush()

            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            out_file.write(result.stdout)
            print(result.stdout)
            print("::endgroup::")

            if result.returncode != 0:
                any_failed = True
                print(f"::error::cargo bench --bench {name} exited {result.returncode}", file=sys.stderr)

    return 1 if any_failed else 0


if __name__ == "__main__":
    sys.exit(main())
