# scripts/bench_vs_bevy_ecs.py
# Parses bench-vs-bevy-ecs-raw.txt (criterion output) and prints a markdown
# summary with per-group tables and a mid-ecs-vs-bevy_ecs ratio.
# Called from .github/workflows/bench-vs-bevy-ecs.yml.
#
# Same shape as scripts/bench_vs_mat4_fastest.py / bench_vs_color.py — same
# RE_ANSI/RE_RESULT parsing, same badge thresholds — but does NOT hardcode
# a "root cause" the way bench_vs_mat4_fastest.py does for its own,
# separately-diagnosed storage-layout issue. This is a fair, general-
# purpose reporter: real numbers in, real table out. Any explanation for
# *why* a ratio looks the way it does belongs in a real, separate
# investigation once the numbers are in hand, not guessed at here.

import re
import sys
from collections import OrderedDict

RE_ANSI = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
RE_RESULT = re.compile(
    r'^(\S[^\n]+?)\s+time:\s+\[\s*([\d.]+\s+\S+)\s+([\d.]+\s+\S+)\s+([\d.]+\s+\S+)\s*\]',
    re.MULTILINE,
)

try:
    raw = open('bench-vs-bevy-ecs-raw.txt', encoding='utf-8', errors='replace').read()
except FileNotFoundError:
    print("*(bench-vs-bevy-ecs-raw.txt not found)*")
    sys.exit(0)

text = RE_ANSI.sub('', raw)


def to_ns(s):
    try:
        val, unit = s.strip().split()
        val = float(val)
        if 'µs' in unit or 'us' in unit:
            return val * 1_000
        if 'ms' in unit:
            return val * 1_000_000
        if unit == 's':
            return val * 1_000_000_000
        return val  # assume ns
    except Exception:
        return None


rows = []
for m in RE_RESULT.finditer(text):
    name = m.group(1).strip()
    mean = m.group(3).strip()
    parts = name.split('/')
    group   = '/'.join(parts[:-1]) if len(parts) >= 2 else name
    variant = parts[-1]            if len(parts) >= 2 else ''
    rows.append((group, variant, mean, to_ns(mean)))

groups = OrderedDict()
for g, v, m, ns in rows:
    groups.setdefault(g, []).append((v, m, ns))


def badge(ratio):
    if ratio <= 1.05:
        return "✅ parity"
    if ratio <= 1.5:
        return f"⚠️ {ratio:.2f}×"
    if ratio <= 5.0:
        return f"❌ {ratio:.2f}×"
    return f"🔴 {ratio:.0f}× slower"


overall = []  # (group, mid_ns, bevy_ns) for the closing recap

for group, variants in groups.items():
    bevy_ns = None
    for v, m, ns in variants:
        if 'bevy_ecs' in v.lower():
            bevy_ns = ns
            break

    print(f"#### {group}")
    has_ratio = bevy_ns is not None
    if has_ratio:
        print("| Impl | Mean | vs bevy_ecs |")
        print("|---|---|---|")
    else:
        print("| Impl | Mean |")
        print("|---|---|")

    mid_ns_for_group = None
    for v, m, ns in variants:
        if has_ratio and ns and bevy_ns:
            r = ns / bevy_ns
            print(f"| {v} | {m} | {badge(r)} |")
        else:
            print(f"| {v} | {m} |")
        if 'mid-ecs' in v.lower() and ns:
            mid_ns_for_group = ns

    print()
    if mid_ns_for_group and bevy_ns:
        overall.append((group, mid_ns_for_group, bevy_ns))

if overall:
    print("> **Recap — mid-ecs vs bevy_ecs, this run:**")
    for group, mid_ns, bevy_ns in overall:
        ratio = mid_ns / bevy_ns
        print(f"> - `{group}`: **{ratio:.1f}×** ({mid_ns:,.0f} ns vs {bevy_ns:,.0f} ns)")
    print(">")
    print("> Large ratios are a real signal to go profile, not an accepted")
    print("> cost — see `benches/ecs-vs-bevy-ecs/benches/vs_bevy_ecs.rs`'s")
    print("> own top doc comment for what each workload actually stresses")
    print("> and the one workload (`spawn`) that isn't perfectly apples-to-")
    print("> apples between the two engines' APIs.")
