# scripts/bench_vs_all.py
# Parses bench-vs-all-raw.txt (criterion output) and prints a markdown summary.
# Called from .github/workflows/Abemch-vs-all.yml:
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

# Patterns indicating criterion itself skipped printing a result for a given
# benchmark — none of these cause a non-zero process exit, so a clean
# "Bench exit code: 0" does not guarantee every benchmark produced a row.
# See criterion 0.5.1 src/analysis/mod.rs: `times.iter().any(|&f| f == 0.0)`
# returns early (logs via the `error!` macro) before the print call that
# RE_RESULT matches against.
RE_DIAGNOSTIC = re.compile(
    r'(took zero time per iteration|Unable to complete \d+ samples|panicked at)',
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

    diag_hits = RE_DIAGNOSTIC.findall(text)
    if diag_hits:
        print()
        print('*Criterion itself reported issues that explain the missing rows '
              '(see Diagnostics section above) — this is not a parser bug.*')
    else:
        print()
        print('*No matching criterion diagnostic strings found either — if this '
              'is a WASM run, check that the runner forwards `CRITERION_DEBUG=1` '
              'and re-run to capture per-benchmark warm-up timing.*')
