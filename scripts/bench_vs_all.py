# scripts/bench_vs_all.py
# Canonical criterion-output → markdown-summary parser. Used by every bench
# workflow in this repo via:
#   python3 scripts/bench_vs_all.py
# with an optional override:
#   BENCH_RAW_FILE=some-other-raw.txt python3 scripts/bench_vs_all.py
#
# Handles two shapes of criterion benchmark id, auto-detected per row (not
# per file — a single raw file can mix both):
#
#   3-segment, plain comparison — "vec3/add/mid-math"
#     group="vec3/add", impl="mid-math". One table per group, one row per
#     impl. This is the original vs_all.rs shape.
#
#   4-segment, parameterized — "mid_vec/construct_push/vec/2"
#     (criterion appends a BenchmarkId's parameter as its own trailing
#     segment on top of the group/name it already had). Detected by
#     checking whether the LAST segment parses as a number — group=
#     "mid_vec/construct_push", impl="vec", size="2".
#     - If a group has more than one distinct impl across its sizes (a
#       real head-to-head, e.g. vec/vecdeque/mid_vec/smallvec/tinyvec at
#       each size), one table per size, impl as rows. This is the shape
#       vs_mid_vec.rs and vs_rng.rs's `rng/bool_p` need.
#     - If a group has exactly one impl swept across sizes (a parameter
#       sweep with nothing to compare against, e.g. vs_curves.rs's
#       `cardinal_evaluate` tension sweep, or vs_rng.rs's `rng/advance`
#       delta sweep), that would otherwise produce a silly one-row table
#       per size — instead it collapses to ONE table for the group, with
#       the impl name as the column header and size as the row label.
#
# Why this matters: applying the 3-segment rule to a 4-segment id folds the
# impl name into the group key and gives every impl its own one-row table
# instead of sharing one — real data, fragmented into the wrong tables, not
# missing. That was a real bug found in three separate bench files
# (vs_mid_vec.rs, vs_rng.rs's bool_p group, vs_curves.rs's cardinal_evaluate
# group) before this script was generalized to auto-detect both shapes.
#
# Throughput: criterion prints a second "thrpt: [...]" line for any
# benchmark_group that calls `.throughput(...)`. The previous version of
# this script didn't parse it at all — vs_all.rs's chain_mat4_8,
# 100k/1m_entity_transforms, 100k_quat_slerp, and 5k_inverse_general groups
# all call `.throughput()`, so that data was being computed by criterion
# and then silently dropped before it ever reached the summary. Fixed here:
# a Throughput column is added to a table whenever any row in it has one.
#
# Env vars:
#   BENCH_RAW_FILE — path to criterion output (default: bench-vs-all-raw.txt)

import os
import re
import sys
from collections import OrderedDict

RE_ANSI = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

# Criterion output line format:
#   vec3/add/mid-math   time:  [960.43 ps  963.28 ps  966.96 ps]
#                               ^^^low^^^  ^^^mean^^^  ^^^high^^^
# group(3) is the mean (middle value with unit).
RE_RESULT = re.compile(
    r'^(\S[^\n]+?)\s+time:\s+\[\s*([\d.]+\s+\S+)\s+([\d.]+\s+\S+)\s+([\d.]+\s+\S+)\s*\]',
    re.MULTILINE,
)
RE_THROUGHPUT = re.compile(
    r'^(\S[^\n]+?)\s+thrpt:\s+\[\s*([\d.]+\s+\S+/s)\s+([\d.]+\s+\S+/s)\s+([\d.]+\s+\S+/s)\s*\]',
    re.MULTILINE,
)

# A clean process exit does not guarantee every benchmark printed a result
# row — criterion silently skips the print for any benchmark whose measured
# time hit exactly 0.0 (see criterion 0.5.1 src/analysis/mod.rs).
RE_DIAGNOSTIC = re.compile(
    r'(took zero time per iteration|Unable to complete \d+ samples|panicked at)',
)

RE_NUMERIC = re.compile(r'^-?\d+(\.\d+)?$')


def is_numeric(s):
    return bool(RE_NUMERIC.match(s))


raw_file = os.environ.get('BENCH_RAW_FILE', 'bench-vs-all-raw.txt')

try:
    raw = open(raw_file, encoding='utf-8', errors='replace').read()
except FileNotFoundError:
    print(f'*(bench raw file not found: `{raw_file}`)*')
    sys.exit(0)

if not raw.strip():
    print(f'*({raw_file} is empty — did the bench step run?)*')
    sys.exit(0)

text = RE_ANSI.sub('', raw)
throughputs = {m.group(1).strip(): m.group(3).strip() for m in RE_THROUGHPUT.finditer(text)}

# group -> list of (impl, size_or_None, mean, thrpt), in first-seen order.
groups = OrderedDict()
row_count = 0

for m in RE_RESULT.finditer(text):
    name = m.group(1).strip()
    mean = m.group(3).strip()
    thrpt = throughputs.get(name, '')

    parts = name.split('/')
    if len(parts) >= 3 and is_numeric(parts[-1]):
        group = '/'.join(parts[:-2])
        impl  = parts[-2]
        size  = parts[-1]
    elif len(parts) >= 2:
        group = '/'.join(parts[:-1])
        impl  = parts[-1]
        size  = None
    else:
        group, impl, size = name, '', None

    groups.setdefault(group, []).append((impl, size, mean, thrpt))
    row_count += 1


def print_table(rows, has_thrpt, impl_header='Impl'):
    if has_thrpt:
        print(f"| {impl_header} | Mean | Throughput |")
        print("|---|---|---|")
        for label, mean, thrpt in rows:
            print(f"| {label} | {mean} | {thrpt} |")
    else:
        print(f"| {impl_header} | Mean |")
        print("|---|---|")
        for label, mean, _ in rows:
            print(f"| {label} | {mean} |")


first_group = True
for group, rows in groups.items():
    if not first_group:
        print()
    first_group = False

    has_size  = any(size is not None for _, size, _, _ in rows)
    has_thrpt = any(t for _, _, _, t in rows)
    distinct_impls = list(OrderedDict.fromkeys(impl for impl, _, _, _ in rows))

    if has_size and len(distinct_impls) == 1:
        # Single-impl parameter sweep — one table for the whole group,
        # impl name as the column header, size as the row label.
        print(f"#### {group}")
        table_rows = [(size, mean, thrpt) for _, size, mean, thrpt in rows]
        print_table(table_rows, has_thrpt, impl_header=distinct_impls[0])

    elif has_size:
        # Multiple impls swept across sizes — one table per size.
        by_size = OrderedDict()
        for impl, size, mean, thrpt in rows:
            by_size.setdefault(size, []).append((impl, mean, thrpt))

        first_size = True
        for size, size_rows in by_size.items():
            if not first_size:
                print()
            first_size = False
            print(f"#### {group} — n={size}")
            print_table(size_rows, has_thrpt)

    else:
        print(f"#### {group}")
        table_rows = [(impl, mean, thrpt) for impl, _, mean, thrpt in rows]
        print_table(table_rows, has_thrpt)

if row_count == 0:
    print('*(no benchmark results parsed — check the raw output artifact)*')

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
