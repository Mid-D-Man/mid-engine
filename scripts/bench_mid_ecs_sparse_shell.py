# scripts/bench_mid_ecs_sparse_shell.py
# Parses bench-mid-ecs-sparse-shell-raw.txt (criterion output) and prints
# a markdown summary: one table per group, plus a two-component-overhead
# table (query2_two_components vs query_single_component) -- the Sparse
# Shell's own version of archetype_core's regression guard. Called from
# .github/workflows/bench-mid-ecs-sparse-shell.yml.
#
# Same shape as scripts/bench_mid_ecs_archetype_core.py, adapted for this
# suite's group names (insert_two_components, query_single_component,
# query2_two_components, remove_insert_churn -- see
# crates/mid-ecs/benches/sparse_shell.rs's own header doc comment for
# what each measures and why).

import re
import sys
from collections import OrderedDict

RE_ANSI = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
RE_RESULT = re.compile(
    r'^(\S[^\n]+?)\s+time:\s+\[\s*([\d.]+\s+\S+)\s+([\d.]+\s+\S+)\s+([\d.]+\s+\S+)\s*\]'
    r'(?:\s*\n\s*thrpt:\s+\[\s*([\d.]+\s+\S+)\s+([\d.]+\s+\S+)\s+([\d.]+\s+\S+)\s*\])?',
    re.MULTILINE,
)

try:
    raw = open('bench-mid-ecs-sparse-shell-raw.txt', encoding='utf-8', errors='replace').read()
except FileNotFoundError:
    print("*(bench-mid-ecs-sparse-shell-raw.txt not found)*")
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
        return val
    except Exception:
        return None


# id looks like "insert_two_components/mid-ecs/10000" -- always exactly
# three segments, variant always "mid-ecs" (one implementation), so
# group is just the first segment. See bench_mid_ecs_archetype_core.py's
# own comment for why this isn't "everything before N" -- that was a
# real bug there, avoided here from the start.
rows = []
for m in RE_RESULT.finditer(text):
    full_id = m.group(1).strip()
    mean = m.group(3).strip()
    thrpt_mean = m.group(6).strip() if m.group(6) else None
    parts = full_id.split('/')
    if len(parts) != 3 or not parts[2].isdigit():
        continue
    n = int(parts[2])
    group = parts[0]
    rows.append((group, n, mean, to_ns(mean), thrpt_mean))

by_group = OrderedDict()
for group, n, mean, ns, thrpt in rows:
    by_group.setdefault(group, []).append((n, mean, ns, thrpt))

for group, entries in by_group.items():
    entries.sort(key=lambda e: e[0])
    print(f"#### {group}")
    has_thrpt = any(t for _, _, _, t in entries)
    if has_thrpt:
        print("| N | Mean | Throughput |")
        print("|---|---|---|")
        for n, mean, ns, thrpt in entries:
            print(f"| {n:,} | {mean} | {thrpt or '—'} |")
    else:
        print("| N | Mean |")
        print("|---|---|")
        for n, mean, ns, thrpt in entries:
            print(f"| {n:,} | {mean} |")
    print()

# ── Two-component overhead, Sparse Shell's own version ─────────────────
single = {n: ns for n, _, ns, _ in by_group.get('query_single_component', []) if ns}
two = {n: ns for n, _, ns, _ in by_group.get('query2_two_components', []) if ns}
shared_ns = sorted(set(single) & set(two))

if shared_ns:
    print("#### Two-component overhead")
    print()
    print("`query2_two_components` against `query_single_component` at the same N.")
    print("Not the same regression guard as archetype_core's own table -- Sparse")
    print("Shell's `query2` still does a real per-entity lookup for the second")
    print("component (`SparseShell::get`, one direct hop, no location/archetype")
    print("chain -- see `component.rs`'s own doc comments), so a ratio well above")
    print("1.0x here is expected, not necessarily a bug. No fixed baseline to")
    print("compare against yet -- this is the first real run.")
    print()
    print("| N | query_single (1 col) | query2 (2 col) | Ratio |")
    print("|---|---|---|---|")
    for n in shared_ns:
        ratio = two[n] / single[n]
        print(f"| {n:,} | {single[n]:,.0f} ns | {two[n]:,.0f} ns | {ratio:.2f}× |")
    print()

# ── Sparse Shell vs Archetype Core, if both raw logs are present ───────
# Not a fair head-to-head by construction (different storage, different
# tradeoffs -- see sparse_shell.rs's own header doc comment) but useful
# to see side by side when both have run recently. Purely informational.
try:
    arch_raw = open('bench-mid-ecs-archetype-core-raw.txt', encoding='utf-8', errors='replace').read()
    arch_text = RE_ANSI.sub('', arch_raw)
    arch_rows = {}
    for m in RE_RESULT.finditer(arch_text):
        full_id = m.group(1).strip()
        mean_ns = to_ns(m.group(3).strip())
        parts = full_id.split('/')
        if len(parts) == 3 and parts[2].isdigit() and mean_ns:
            arch_rows[(parts[0], int(parts[2]))] = mean_ns

    pairs = [
        ('insert_two_components', 'spawn_insert_bundle', 'insert / spawn+insert'),
        ('query_single_component', 'query_static_single_component', 'query (1 col)'),
        ('query2_two_components', 'query2_static_two_components', 'query (2 col)'),
        ('remove_insert_churn', 'structural_churn_insert_remove', 'insert+remove churn'),
    ]
    printed_header = False
    for shell_group, arch_group, label in pairs:
        shell_vals = {n: ns for n, _, ns, _ in by_group.get(shell_group, []) if ns}
        common_ns = sorted(n for n in shell_vals if (arch_group, n) in arch_rows)
        if not common_ns:
            continue
        if not printed_header:
            print("#### Sparse Shell vs Archetype Core (informational, not a regression guard)")
            print()
            print("| Operation | N | Sparse Shell | Archetype Core | Faster |")
            print("|---|---|---|---|---|")
            printed_header = True
        for n in common_ns:
            s_ns = shell_vals[n]
            a_ns = arch_rows[(arch_group, n)]
            faster = "Sparse Shell" if s_ns < a_ns else "Archetype Core"
            ratio = max(s_ns, a_ns) / min(s_ns, a_ns)
            print(f"| {label} | {n:,} | {s_ns:,.0f} ns | {a_ns:,.0f} ns | {faster} ({ratio:.2f}×) |")
    if printed_header:
        print()
except FileNotFoundError:
    pass
