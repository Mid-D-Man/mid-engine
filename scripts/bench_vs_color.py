# scripts/bench_vs_color.py
# Parses bench-color-raw.txt (criterion output) and prints a markdown summary.
# Called from .github/workflows/bench-vs-color.yml.

import re
import sys

RE_ANSI = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
RE_RESULT = re.compile(
    r'^(\S[^\n]+?)\s+time:\s+\[\s*([\d.]+\s+\S+)\s+([\d.]+\s+\S+)\s+([\d.]+\s+\S+)\s*\]',
    re.MULTILINE,
)

try:
    raw = open('bench-color-raw.txt', encoding='utf-8', errors='replace').read()
except FileNotFoundError:
    print("*(bench-color-raw.txt not found)*")
    sys.exit(0)

text = RE_ANSI.sub('', raw)
current_group = None

for m in RE_RESULT.finditer(text):
    name    = m.group(1).strip()
    lo      = m.group(2).strip()
    mean    = m.group(3).strip()
    hi      = m.group(4).strip()
    parts   = name.split('/')
    group   = '/'.join(parts[:-1]) if len(parts) >= 2 else name
    variant = parts[-1]            if len(parts) >= 2 else ''

    if group != current_group:
        if current_group is not None:
            print()
        print(f"#### {group}")
        print("| Variant | Low | Mean | High |")
        print("|---|---|---|---|")
        current_group = group

    print(f"| {variant} | {lo} | {mean} | {hi} |")
