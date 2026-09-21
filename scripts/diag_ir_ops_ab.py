#!/usr/bin/env python3
# ============================================================================
# NOTICE: Full documentation, design decisions, and fix history for this file
# live in docs/mid-ecs.md, section "diag_ir_ops.rs"
# ============================================================================
"""A/B driver for crates/mid-ecs/examples/diag_ir_ops.rs.

Builds the example twice on the SAME machine and toolchain -- once from
the working tree ("current"), once with `crates/mid-ecs/src/archetype.rs`
and `lib.rs` reverted to `--prefix-sha` ("prefix") -- then reports, per
variant: instructions per operation under callgrind (deterministic) and
isolated wall-clock ns per `get_static` lookup. The working tree is
restored afterwards.

Why both: `ecs-vs-bevy-ecs` build #25 showed instruction counts and CI
timing disagreeing for `get_component_random_access`. Instruction counts
say what the code does; timing says what the machine did with it.
Whichever of the two moves (or doesn't) between "prefix" and "current"
here, on CI's own toolchain, is the next question.

Stdlib only. Needs `valgrind` (and optionally `callgrind_annotate`) on
PATH.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys

EXAMPLE = "diag_ir_ops"
REVERT_FILES = [
    "crates/mid-ecs/src/archetype.rs",
    "crates/mid-ecs/src/lib.rs",
]
# (mode, ops counted in the differential run)
OPS = [
    ("get", 20 * 10_000),
    ("insert", 10_000),
    ("remove", 10_000),
    ("spawn", 10_000),
]


def run(cmd, **kw):
    print("+ " + " ".join(cmd), file=sys.stderr, flush=True)
    return subprocess.run(cmd, check=True, **kw)


def build(profile):
    """Builds the example, returns the path of the executable.

    `strip` is forced off for the build: the workspace's `bench` profile
    inherits `release`'s `strip = true`, which would leave callgrind
    reporting raw addresses instead of function names. Stripping changes
    the symbol table only, not the generated code.
    """
    env = dict(os.environ)
    env["CARGO_PROFILE_" + profile.upper().replace("-", "_") + "_STRIP"] = "false"
    proc = subprocess.run(
        [
            "cargo", "build", "--profile", profile, "-p", "mid-ecs",
            "--example", EXAMPLE, "--message-format=json-render-diagnostics",
        ],
        check=True, stdout=subprocess.PIPE, text=True, env=env,
    )
    exe = None
    for line in proc.stdout.splitlines():
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        if (
            msg.get("reason") == "compiler-artifact"
            and msg.get("target", {}).get("name") == EXAMPLE
            and msg.get("executable")
        ):
            exe = msg["executable"]
    if not exe:
        sys.exit("could not find the built example executable in cargo's output")
    return exe


def callgrind_total(path):
    with open(path) as f:
        for line in f:
            if line.startswith("summary:") or line.startswith("totals:"):
                return int(line.split()[1])
    sys.exit(f"no summary line in {path}")


def per_function_self(path):
    """Self-cost Ir per function name from callgrind_annotate."""
    if not shutil.which("callgrind_annotate"):
        return {}
    out = subprocess.run(
        ["callgrind_annotate", path], capture_output=True, text=True
    ).stdout
    table = {}
    for line in out.splitlines():
        m = re.match(r"\s*([\d,]+)\s+\(\s*[\d.]+%\)\s+(\S+?):(.*)$", line)
        if not m:
            continue
        name = m.group(3).split(" [")[0].strip()
        table[name] = table.get(name, 0) + int(m.group(1).replace(",", ""))
    return table


def measure(exe, tag, outdir, reps):
    result = {"ir": {}, "top": {}, "time": None}
    times = []
    for _ in range(reps):
        out = subprocess.run([exe, "time-get"], check=True,
                             capture_output=True, text=True).stdout
        m = re.search(r"best_ns_per_lookup=([\d.]+)", out)
        if m:
            times.append(float(m.group(1)))
    result["time"] = min(times) if times else None
    for mode, ops in OPS:
        files = {}
        for flag in ("0", "1"):
            f = os.path.join(outdir, f"cg.{tag}.{mode}.{flag}.out")
            subprocess.run(
                ["valgrind", "--tool=callgrind", f"--callgrind-out-file={f}",
                 exe, mode, flag],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            files[flag] = f
        result["ir"][mode] = (
            callgrind_total(files["1"]) - callgrind_total(files["0"])
        ) / ops
        if mode in ("get", "insert"):
            a = per_function_self(files["0"])
            b = per_function_self(files["1"])
            diff = {k: (b.get(k, 0) - a.get(k, 0)) / ops for k in set(a) | set(b)}
            result["top"][mode] = sorted(diff.items(), key=lambda kv: -kv[1])[:10]
    return result


def fmt_ratio(a, b):
    if a is None or b is None or b == 0:
        return "n/a"
    return f"{a / b:.2f}x"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="bench")
    ap.add_argument("--prefix-sha", required=True,
                    help="commit whose archetype.rs/lib.rs define the 'prefix' variant")
    ap.add_argument("--out", default="diag-ir-ops-out")
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()

    if not shutil.which("valgrind"):
        sys.exit("valgrind not found on PATH")
    os.makedirs(args.out, exist_ok=True)

    exes = {}
    exes["current"] = shutil.copy(build(args.profile),
                                  os.path.join(args.out, f"{EXAMPLE}.current"))
    try:
        run(["git", "checkout", args.prefix_sha, "--"] + REVERT_FILES)
        exes["prefix"] = shutil.copy(build(args.profile),
                                     os.path.join(args.out, f"{EXAMPLE}.prefix"))
    finally:
        run(["git", "checkout", "HEAD", "--"] + REVERT_FILES)

    res = {tag: measure(exe, tag, args.out, args.reps) for tag, exe in exes.items()}
    rustc = subprocess.run(["rustc", "--version"], capture_output=True,
                           text=True).stdout.strip()

    md = []
    md.append(f"## diag_ir_ops A/B: `{args.profile}` profile")
    md.append("")
    md.append(f"`{rustc}`, prefix = `{args.prefix_sha[:8]}` "
              f"(archetype.rs + lib.rs reverted), current = working tree.")
    md.append("")
    md.append("### Instructions per operation (callgrind, deterministic)")
    md.append("")
    md.append("| op | prefix Ir | current Ir | current / prefix |")
    md.append("|---|---|---|---|")
    for mode, _ in OPS:
        p, c = res["prefix"]["ir"][mode], res["current"]["ir"][mode]
        md.append(f"| {mode} | {p:.1f} | {c:.1f} | {fmt_ratio(c, p)} |")
    md.append("")
    md.append("### Wall-clock `get_static`, isolated (best of "
              f"{args.reps} process runs)")
    md.append("")
    md.append("| variant | ns per lookup | us per 10,000 |")
    md.append("|---|---|---|")
    for tag in ("prefix", "current"):
        t = res[tag]["time"]
        md.append(f"| {tag} | {t:.2f} | {t * 10:.1f} |" if t is not None
                  else f"| {tag} | n/a | n/a |")
    md.append("")
    md.append("Compare against `ecs-vs-bevy-ecs` build #25's "
              "`get_component_random_access`: mid-ecs 362.16 us and "
              "bevy_ecs 75.638 us per 10,000 lookups.")
    for mode in ("get", "insert"):
        for tag in ("current", "prefix"):
            top = res[tag]["top"].get(mode)
            if not top:
                continue
            md.append("")
            md.append(f"<details><summary>{mode}: top self-cost functions, "
                      f"{tag} (Ir per op)</summary>")
            md.append("")
            md.append("```")
            for name, ir in top:
                md.append(f"{ir:8.1f}  {name[:110]}")
            md.append("```")
            md.append("")
            md.append("</details>")
    text = "\n".join(md) + "\n"
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
