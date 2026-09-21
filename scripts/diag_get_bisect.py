#!/usr/bin/env python3
# ============================================================================
# NOTICE: Full documentation, design decisions, and fix history for this file
# live in docs/mid-ecs.md, section "get_bisect.rs"
# ============================================================================
"""Bisect driver for `get_component_random_access`.

Question: the isolated mid-ecs binary measures ~7.5ns per `get_static`
lookup on CI, but `vs_bevy_ecs.rs` measured ~36ns for the same code
(`ecs-vs-bevy-ecs` build #25). What in the real bench binary is
responsible? This runs, on ONE machine and toolchain:

  1. benches/ecs-vs-bevy-ecs/examples/get_bisect.rs (mid-ecs AND bevy_ecs
     linked, both worlds built, NO criterion): wall-clock ns per lookup
     with the bevy world absent/alive, and callgrind instructions per
     lookup for both engines.
  2. The real `vs_bevy_ecs` criterion bench filtered to the one group,
     so no other group has run first in the process.

Reading the result (see docs/mid-ecs.md, "get_bisect.rs"):
  - example fast, filtered bench fast     -> earlier groups' history
    (or build #25 was a transient run).
  - example fast, filtered bench slow     -> criterion / bench-file
    structure.
  - example slow, "alone" fast            -> bevy's world coexisting.
  - example slow, "alone" also slow, Ir up -> bevy linked into the LTO
    unit changed mid-ecs's codegen (callgrind list says where).
  - example slow, Ir ~ 109                -> the machine, not the
    instruction count.

Stdlib only. Needs `valgrind` on PATH.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys

PKG = "ecs-vs-bevy-ecs"
EXAMPLE = "get_bisect"
OPS = 20 * 10_000  # lookups in the callgrind differential


def sh(cmd, **kw):
    print("+ " + " ".join(cmd), file=sys.stderr, flush=True)
    return subprocess.run(cmd, **kw)


def build(profile):
    env = dict(os.environ)
    env["CARGO_PROFILE_" + profile.upper().replace("-", "_") + "_STRIP"] = "false"
    proc = sh(
        ["cargo", "build", "--profile", profile, "-p", PKG,
         "--example", EXAMPLE, "--message-format=json-render-diagnostics"],
        check=True, stdout=subprocess.PIPE, text=True, env=env,
    )
    exe = None
    for line in proc.stdout.splitlines():
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        if (msg.get("reason") == "compiler-artifact"
                and msg.get("target", {}).get("name") == EXAMPLE
                and msg.get("executable")):
            exe = msg["executable"]
    if not exe:
        sys.exit("could not locate the built example executable")
    return exe


def callgrind_total(path):
    with open(path) as f:
        for line in f:
            if line.startswith("summary:") or line.startswith("totals:"):
                return int(line.split()[1])
    sys.exit(f"no summary line in {path}")


def per_function_self(path):
    if not shutil.which("callgrind_annotate"):
        return {}
    out = subprocess.run(["callgrind_annotate", path],
                         capture_output=True, text=True).stdout
    table = {}
    for line in out.splitlines():
        m = re.match(r"\s*([\d,]+)\s+\(\s*[\d.]+%\)\s+(\S+?):(.*)$", line)
        if m:
            name = m.group(3).split(" [")[0].strip()
            table[name] = table.get(name, 0) + int(m.group(1).replace(",", ""))
    return table


def ir_per_lookup(exe, mode, outdir):
    files = {}
    for flag in ("0", "1"):
        f = os.path.join(outdir, f"cg.{mode}.{flag}.out")
        subprocess.run(
            ["valgrind", "--tool=callgrind", f"--callgrind-out-file={f}",
             exe, mode, flag],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        files[flag] = f
    total = (callgrind_total(files["1"]) - callgrind_total(files["0"])) / OPS
    a, b = per_function_self(files["0"]), per_function_self(files["1"])
    diff = {k: (b.get(k, 0) - a.get(k, 0)) / OPS for k in set(a) | set(b)}
    return total, sorted(diff.items(), key=lambda kv: -kv[1])[:10]


def run_time(exe, reps):
    best = {}
    for _ in range(reps):
        out = subprocess.run([exe, "time"], check=True,
                             capture_output=True, text=True).stdout
        for key, val in re.findall(r"(\w+_ns)=([\d.]+)", out):
            best[key] = min(best.get(key, float("inf")), float(val))
    return best


UNIT_US = {"ps": 1e-6, "ns": 1e-3, "us": 1.0, "µs": 1.0, "ms": 1e3, "s": 1e6}


def parse_criterion(text):
    """{bench name: median time in microseconds} from criterion output."""
    res = {}
    name = None
    for line in text.splitlines():
        if "time:" in line and "[" in line:
            m = re.search(r"\[\s*([\d.]+)\s*(\S+)\s+([\d.]+)\s*(\S+)\s+([\d.]+)\s*(\S+)\s*\]",
                          line)
            inline = line.split("time:")[0].strip()
            key = inline or name
            if m and key:
                res[key] = float(m.group(3)) * UNIT_US.get(m.group(4), float("nan"))
        elif line.strip() and not line.startswith(" ") and "/" in line \
                and not line.startswith(("Benchmarking", "Warning", "Found")):
            name = line.strip()
    return res


def run_criterion(outdir):
    proc = sh(
        ["cargo", "bench", "-p", PKG, "--bench", "vs_bevy_ecs", "--",
         "--color", "never", "--warm-up-time", "2", "--measurement-time", "5",
         "get_component_random_access"],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    with open(os.path.join(outdir, "criterion-get-only.txt"), "w") as f:
        f.write(proc.stdout)
    return parse_criterion(proc.stdout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="bench")
    ap.add_argument("--out", default="diag-get-bisect-out")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--skip-criterion", action="store_true")
    args = ap.parse_args()

    if not shutil.which("valgrind"):
        sys.exit("valgrind not found on PATH")
    os.makedirs(args.out, exist_ok=True)

    exe = build(args.profile)
    times = run_time(exe, args.reps)
    mid_ir, mid_top = ir_per_lookup(exe, "mid-ir", args.out)
    bevy_ir, _ = ir_per_lookup(exe, "bevy-ir", args.out)
    crit = {} if args.skip_criterion else run_criterion(args.out)
    rustc = subprocess.run(["rustc", "--version"], capture_output=True,
                           text=True).stdout.strip()
    cpu = ""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass

    def us(ns):
        return f"{ns * 10:.1f}" if ns is not None else "n/a"

    md = [f"## get_bisect: `{args.profile}` profile", "",
          f"`{rustc}`, CPU `{cpu}`", "",
          "All times are microseconds per 10,000 lookups (lower is faster).", "",
          "| experiment | mid-ecs | bevy_ecs |",
          "|---|---|---|",
          f"| `get_bisect`, mid world alone (bevy world not built yet) | {us(times.get('mid_alone_ns'))} | - |",
          f"| `get_bisect`, both worlds alive | {us(times.get('mid_bevy_alive_ns'))} | {us(times.get('bevy_ns'))} |"]
    if crit:
        mid_c = next((v for k, v in crit.items() if "mid" in k), None)
        bevy_c = next((v for k, v in crit.items() if "bevy" in k), None)
        md.append("| real bench, `get_component_random_access` only "
                  f"(criterion, no earlier groups) | {mid_c:.1f} | {bevy_c:.1f} |"
                  if mid_c is not None and bevy_c is not None
                  else "| real bench, filtered | (could not parse, see artifact) | |")
    md += ["| reference: isolated `diag_ir_ops` (A/B builds #1/#2) | 75.4 (bench) / 87.8 (bench-nolto) | - |",
           "| reference: full bench, `ecs-vs-bevy-ecs` build #25 | 362.2 | 75.6 |",
           "",
           f"Instructions per lookup (callgrind, deterministic): mid-ecs **{mid_ir:.1f}**, "
           f"bevy_ecs **{bevy_ir:.1f}**. Isolated `diag_ir_ops` on CI (bench profile): 109.",
           "",
           "<details><summary>mid-ecs lookup: top self-cost functions in the "
           "bisect binary (Ir per lookup)</summary>", "", "```"]
    md += [f"{ir:8.1f}  {name[:110]}" for name, ir in mid_top]
    md += ["```", "", "</details>", ""]
    text = "\n".join(md) + "\n"
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
