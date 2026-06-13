# scripts/bench_vs_all.py
# Parses bench-vs-all-raw.txt (criterion output) and prints a markdown summary.
# Called from .github/workflows/bemch-vs-all.yml:
#   python3 scripts/bench_vs_all.py >> $GITHUB_STEP_SUMMARY
#
# Env vars:
#   BENCH_RAW_FILE  — path to criterion output (default: bench-vs-all-raw.txt)

import os
import re
import sys

RE_ANSI = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

# Criterion median output line format:
#   vec3/add/mid-math   time:  [960.43 ps  963.28 ps  966.96 ps]
#                               ^^^low^^^  ^^^med^^^  ^^^high^^^
# group(3) is the median (middle value with unit).
RE_RESULT = re.compile(
    r'^(\S[^\n]+?)\s+time:\s+\[\s*([\d.]+\s+\S+)\s+([\d.]+\s+\S+)\s+([\d.]+\s+\S+)\s*\]',
    re.MULTILINE,
)

raw_file = os.environ.get('BENCH_RAW_FILE', 'bench-vs-all-raw.txt')

try:
    raw = open(raw_file, encoding='utf-8', errors='replace').read()
except FileNotFoundError:
    print(f'*(bench raw file not found: `{raw_file}`)*')
    sys.exit(0)

if not raw.strip():
    print('*(bench-vs-all-raw.txt is empty — did the bench step run?)*')
    sys.exit(0)

text = RE_ANSI.sub('', raw)
current_group = None
row_count = 0

for m in RE_RESULT.finditer(text):
    name = m.group(1).strip()
    mean = m.group(3).strip()   # median (middle of the three criterion values)

    # Split "vec3/add/mid-math" → group="vec3/add", variant="mid-math"
    parts = name.split('/')
    group   = '/'.join(parts[:-1]) if len(parts) >= 2 else name
    variant = parts[-1]            if len(parts) >= 2 else ''

    if group != current_group:
        if current_group is not None:
            print()
        print(f'#### {group}')
        print('| Impl | Mean |')
        print('|---|---|')
        current_group = group

    print(f'| {variant} | {mean} |')
    row_count += 1

if row_count == 0:
    print('*(no benchmark results parsed — check bench-vs-all-raw.txt)*')
