# scripts/bench_vs_mat4_fastest.py
# Parses bench-mat4-fastest-raw.txt (criterion) and prints a markdown summary
# with per-group tables and a diagnostic for the mat4 storage gap.
# Called from .github/workflows/bench-vs-mat4-fastest.yml.

import re
import sys
from collections import OrderedDict

RE_ANSI = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
RE_RESULT = re.compile(
    r'^(\S[^\n]+?)\s+time:\s+\[\s*([\d.]+\s+\S+)\s+([\d.]+\s+\S+)\s+([\d.]+\s+\S+)\s*\]',
    re.MULTILINE,
)

try:
    raw = open('bench-mat4-fastest-raw.txt', encoding='utf-8', errors='replace').read()
except FileNotFoundError:
    print("*(bench-mat4-fastest-raw.txt not found)*")
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

mid_4x4_ns  = None
glam_4x4_ns = None

for group, variants in groups.items():
    glam_ns = None
    for v, m, ns in variants:
        if 'glam' in v.lower():
            glam_ns = ns
            break

    print(f"#### {group}")
    has_ratio = glam_ns is not None
    if has_ratio:
        print("| Impl | Mean | vs glam |")
        print("|---|---|---|")
    else:
        print("| Impl | Mean |")
        print("|---|---|")

    for v, m, ns in variants:
        if has_ratio and ns and glam_ns:
            r = ns / glam_ns
            if r <= 1.05:
                badge = "✅ parity"
            elif r <= 1.5:
                badge = f"⚠️ {r:.2f}×"
            elif r <= 5.0:
                badge = f"❌ {r:.2f}×"
            else:
                badge = f"🔴 {r:.0f}× (overhead dominated)"
            print(f"| {v} | {m} | {badge} |")
        else:
            print(f"| {v} | {m} |")

        if '4x4_latency' in group:
            if 'mid-math' in v.lower() and 'current' in v.lower() and ns:
                mid_4x4_ns = ns
            if 'glam' in v.lower() and ns:
                glam_4x4_ns = ns

    print()

if mid_4x4_ns and glam_4x4_ns:
    ratio = mid_4x4_ns / glam_4x4_ns
    print(f"> **Key finding:** mid-math 4×4 latency is **{ratio:.1f}×** glam's ({mid_4x4_ns:.1f} ns vs {glam_4x4_ns:.1f} ns).")
    if ratio > 1.1:
        print(">")
        print("> **Root cause:** `[[f32;4];4]` storage → Mat4 passed by pointer → 8× `_mm_load_ps` before any FP math.")
        print("> **Fix (OPT-3):** Change to named `Vec4` fields. ABI then passes columns in XMM0-XMM3. Target: ≤7 ns.")
        print("> **Next (OPT-7):** AVX2 two-column-per-ymm approach → ~3.5 ns.")
    else:
        print("> 🎉 Storage fix working — at or near parity with glam!")
