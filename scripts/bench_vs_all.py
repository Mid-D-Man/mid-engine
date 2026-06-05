# scripts/bench_vs_all.py
# Parses bench-vs-all-raw.txt (criterion output) and prints a markdown summary.
# Called from .github/workflows/bench-vs-all.yml inside a { } >> GITHUB_STEP_SUMMARY block.

import re
import sys

RE_ANSI = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
RE_RESULT = re.compile(
    r'^(\S[^\n]+?)\s+time:\s+\[\s*([\d.]+\s+\S+)\s+([\d.]+\s+\S+)\s+([\d.]+\s+\S+)\s*\]',
    re.MULTILINE,
)

try:
    raw = open('bench-vs-all-raw.txt', encoding='utf-8', errors='replace').read()
except FileNotFoundError:
    print("*(bench-vs-all-raw.txt not found)*")
    sys.exit(0)

text = RE_ANSI.sub('', raw)
current_group = None

for m in RE_RESULT.finditer(text):
    name = m.group(1).strip()
    mean = m.group(3).strip()

    parts = name.split('/')
    group   = '/'.join(parts[:-1]) if len(parts) >= 2 else name
    variant = parts[-1]            if len(parts) >= 2 else ''

    if group != current_group:
        if current_group is not None:
            print()
        print(f"#### {group}")
        print("| Impl | Mean |")
        print("|---|---|")
        current_group = group

    print(f"| {variant} | {mean} |")
