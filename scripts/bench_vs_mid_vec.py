# scripts/bench_vs_mid_vec.py
# Parses bench-vs-mid-vec-raw.txt (criterion output) and prints a markdown
# summary — one combined table per (benchmark group, size), every container
# implementation as a row in that table. Mirrors scripts/bench_vs_all.py's
# format and diagnostics; the difference is entirely in the grouping key.
#
# Why this needed its own script instead of reusing bench_vs_all.py's
# "group = everything but the last segment" rule:
#   vs_all ids are 3 segments —      "vec3/add/mid-math"        (group/impl)
#   vs_mid_vec ids are 4 segments —  "mid_vec/construct_push/vec/2"
#                                     (group/impl/size — criterion appends
#                                     the BenchmarkId's parameter as its own
#                                     trailing segment on top of the
#                                     group/name it already had)
# Applying vs_all's rule to a 4-segment id folds the impl name INTO the
# group key ("mid_vec/construct_push/vec" as the "group", "2" as the only
# "variant") — every impl then gets its own one-row table instead of
# sharing one table per size. Nothing was missing or failing to parse; the
# rows just wound up in the wrong tables. Splitting off the last TWO
# segments (impl, then size) instead of one fixes it.
#
# Called from .github/workflows/bench-vs-mid-vec.yml:
#   python3 scripts/bench_vs_mid_vec.py >> $GITHUB_STEP_SUMMARY
#
# Env vars:
#   BENCH_RAW_FILE — path to criterion output (default: bench-vs-mid-vec-raw.txt)

import os
import re
import sys
from collections import OrderedDict

RE_ANSI = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

# Criterion output line format (same as bench_vs_all.py relies on):
#   mid_vec/construct_push/vec/2   time:   [30.251 ns 30.633 ns 31.020 ns]
#                                            ^^^low^^^ ^^^mean^^^ ^^^high^^^
# group(3) is the mean (middle value with unit).
RE_RESULT = re.compile(
    r'^(\S[^\n]+?)\s+time:\s+\[\s*([\d.]+\s+\S+)\s+([\d.]+\s+\S+)\s+([\d.]+\s+\S+)\s*\]',
    re.MULTILINE,
)
RE_THROUGHPUT = re.compile(
    r'^(\S[^\n]+?)\s+thrpt:\s+\[\s*([\d.]+\s+\S+/s)\s+([\d.]+\s+\S+/s)\s+([\d.]+\s+\S+/s)\s*\]',
    re.MULTILINE,
)

# Same diagnostic patterns bench_vs_all.py checks for — a clean process
# exit does not guarantee every benchmark printed a result line (criterion
# silently skips the print for any benchmark whose measured time hit
# exactly 0.0; see criterion 0.5.1 src/analysis/mod.rs).
RE_DIAGNOSTIC = re.compile(
    r'(took zero time per iteration|Unable to complete \d+ samples|panicked at)',
)

raw_file = os.environ.get('BENCH_RAW_FILE', 'bench-vs-mid-vec-raw.txt')

try:
    raw = open(raw_file, encoding='utf-8', errors='replace').read()
except FileNotFoundError:
    print(f'*(bench raw file not found: `{raw_file}`)*')
    sys.exit(0)

if not raw.strip():
    print('*(bench-vs-mid-vec-raw.txt is empty — did the bench step run?)*')
    sys.exit(0)

text = RE_ANSI.sub('', raw)
throughputs = {m.group(1).strip(): m.group(3).strip() for m in RE_THROUGHPUT.finditer(text)}

# key = (group, size) in first-seen order → list of (impl, mean, thrpt) in
# first-seen order. Criterion runs each benchmark_group in declaration
# order and, within a group, in the order bench_with_input was called —
# which in vs_mid_vec.rs is size as the outer loop, impl as the inner call
# — so this dict fills in exactly the display order we want (every impl
# for size=2, then every impl for size=4, ...) with no extra sorting.
tables = OrderedDict()
row_count = 0

for m in RE_RESULT.finditer(text):
    name = m.group(1).strip()
    mean = m.group(3).strip()   # mean (middle of criterion's [low mean high])
    thrpt = throughputs.get(name, '')

    parts = name.split('/')
    if len(parts) >= 4:
        # "mid_vec/construct_push/vec/2" → group="mid_vec/construct_push",
        # impl="vec", size="2"
        group = '/'.join(parts[:-2])
        impl  = parts[-2]
        size  = parts[-1]
    elif len(parts) >= 2:
        # Defensive fallback if a non-parameterized id ever lands in this
        # file — behaves like bench_vs_all.py's grouping.
        group = '/'.join(parts[:-1])
        impl  = parts[-1]
        size  = None
    else:
        group, impl, size = name, '', None

    tables.setdefault((group, size), []).append((impl, mean, thrpt))
    row_count += 1

for (group, size), rows in tables.items():
    header = f"#### {group} — n={size}" if size is not None else f"#### {group}"
    print(header)
    has_thrpt = any(t for _, _, t in rows)
    if has_thrpt:
        print("| Impl | Mean | Throughput |")
        print("|---|---|---|")
        for impl, mean, thrpt in rows:
            print(f"| {impl} | {mean} | {thrpt} |")
    else:
        print("| Impl | Mean |")
        print("|---|---|")
        for impl, mean, _ in rows:
            print(f"| {impl} | {mean} |")
    print()

if row_count == 0:
    print('*(no benchmark results parsed — check bench-vs-mid-vec-raw.txt)*')

    diag_hits = RE_DIAGNOSTIC.findall(text)
    if diag_hits:
        print()
        print('*Criterion itself reported issues that explain the missing rows '
              '(see Diagnostics section above) — this is not a parser bug.*')
    else:
        print()
        print('*No matching criterion diagnostic strings found either — check '
              'that the bench step actually ran and wrote bench-vs-mid-vec-raw.txt.*')
