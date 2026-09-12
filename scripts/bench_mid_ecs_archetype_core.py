# scripts/bench_mid_ecs_archetype_core.py
# Parses bench-mid-ecs-archetype-core-raw.txt (criterion output) and prints
# a markdown summary: one table per group (N -> time/throughput), plus a
# "two-component overhead" table -- the real regression guard for the
# per-entity-lookup + combinator-chain fixes documented in
# crates/mid-ecs/src/archetype.rs's Iter1/Iter2 doc comments. Called from
# .github/workflows/bench-mid-ecs-archetype-core.yml.
#
# Same RE_ANSI/RE_RESULT parsing shape as scripts/bench_vs_bevy_ecs.py /
# scripts/bench_vs_mat4_fastest.py -- this suite has no natural "vs X"
# baseline (every row is "mid-ecs", there's no second implementation in
# the same run), so instead of a per-row badge this reports the one ratio
# that *is* real and worth tracking here: query2_static_two_components
# against query_static_single_component at the same N. Both touch the
# same archetype's columns, resolved once, not per entity or per item --
# see docs/benching-standards.md and the two Iter1/Iter2 doc comments for
# why that ratio should stay close to 1.0-1.3x, not the ~16-21x it was
# before the fix this script's own workflow exists to guard against.

import re
import sys
from collections import OrderedDict, defaultdict

RE_ANSI = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
RE_RESULT = re.compile(
    r'^(\S[^\n]+?)\s+time:\s+\[\s*([\d.]+\s+\S+)\s+([\d.]+\s+\S+)\s+([\d.]+\s+\S+)\s*\]'
    r'(?:\s*\n\s*thrpt:\s+\[\s*([\d.]+\s+\S+)\s+([\d.]+\s+\S+)\s+([\d.]+\s+\S+)\s*\])?',
    re.MULTILINE,
)

try:
    raw = open('bench-mid-ecs-archetype-core-raw.txt', encoding='utf-8', errors='replace').read()
except FileNotFoundError:
    print("*(bench-mid-ecs-archetype-core-raw.txt not found)*")
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


# id looks like "spawn_insert_bundle/mid-ecs/10000" (variant is always
# the constant "mid-ecs" for those groups -- fold it away, group is
# just parts[0]) or "raw_slice_ceiling/one_field_sum/10000" (variant is
# a real, meaningful distinction between two different loops over the
# same data -- keep it in the group label, or the two variants collapse
# into one ambiguous table with duplicate, unlabeled N rows). Caught by
# actually running this against real captured output before trusting
# it, same as the earlier group-name bug -- inspection alone missed
# this one too.
rows = []
for m in RE_RESULT.finditer(text):
    full_id = m.group(1).strip()
    mean = m.group(3).strip()
    thrpt_mean = m.group(6).strip() if m.group(6) else None
    parts = full_id.split('/')
    if len(parts) != 3 or not parts[2].isdigit():
        continue
    n = int(parts[2])
    group = parts[0] if parts[1] == 'mid-ecs' else f'{parts[0]} — {parts[1]}'
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

# ── The real regression guard ──────────────────────────────────────────
single = {n: ns for n, _, ns, _ in by_group.get('query_static_single_component', []) if ns}
two = {n: ns for n, _, ns, _ in by_group.get('query2_static_two_components', []) if ns}
shared_ns = sorted(set(single) & set(two))

# `single` (query_static_single_component) is the denominator of the
# ratio below, and it has now been directly observed to drift on its
# own -- Archetype Core builds #13, #15, #16 all showed it running
# ~4x its usual cost with ZERO change to its own source (see docs/
# mid-ecs.md's "regression-guard ratio's own instability" section),
# which drags the ratio down to a falsely healthy 1.07-1.13x by making
# the baseline equally bad, not by query2_static getting better. A
# ratio built on a moving denominator isn't trustworthy on its own, so
# cross-check `single` against `raw_slice_ceiling`'s one_field_sum --
# zero ECS abstraction, nothing for a compilation-unit-layout shift to
# grab onto the same way -- and refuse to print a clean ✅ if the
# baseline itself looks drifted this run, regardless of what the raw
# ratio number says.
floor = {n: ns for n, _, ns, _ in by_group.get('raw_slice_ceiling — one_field_sum', []) if ns}
# N=100 excluded here specifically: raw_slice_ceiling's own fixed
# per-call overhead isn't yet amortized away at that size (checked
# directly -- Build #11, a run with a genuinely healthy baseline by
# every other measure, still showed single/floor = 2.6x at N=100 purely
# from this, a false positive this check exists to avoid), while
# N=1,000 and up agree with each other and with the ~4x collapsed
# reading at N=100,000 -- this mirrors why the main ratio table above
# has always read at N>=1,000, not N=100, for the same reason.
baseline_shared = sorted(n for n in (set(single) & set(floor)) if n >= 1000)
baseline_drift = max((single[n] / floor[n] for n in baseline_shared), default=None)
# Healthy runs (Build #11): ~1.0x. Drifted runs (#13, #15, #16): ~4.0x.
# 2.0x sits with real margin in the gap between those two regimes.
BASELINE_DRIFT_THRESHOLD = 2.0

if shared_ns:
    print("#### Two-component overhead (the real regression guard)")
    print()
    print("`query2_static_two_components` against `query_static_single_component`")
    print("at the same N. Both resolve their columns once per matching archetype,")
    print("not per entity or per item (see `Iter1`/`Iter2` in")
    print("`crates/mid-ecs/src/archetype.rs`). `raw_slice_ceiling`'s own")
    print("two_field/one_field ratio (zero ECS abstraction at all) is ~1.0x at")
    print("N>=1,000 -- reading a second field costs, within noise, nothing extra at")
    print("this scale. The ~1.28-1.35x actually observed here is real and stable")
    print("(not sample noise -- checked with --sample-size 50), and not yet fully")
    print("explained; not urgent on its own (this is already within noise of")
    print("bevy_ecs's real numbers, see benches/ecs-vs-bevy-ecs), but if it climbs")
    print("well past that -- especially back toward the ~16-21x this was before the")
    print("Iter1/Iter2 rewrite -- that's a real regression, not this baseline gap.")
    print()

    if baseline_drift is not None and baseline_drift > BASELINE_DRIFT_THRESHOLD:
        print(f"> 🔴 **`query_static_single_component` itself is running {baseline_drift:.1f}× `raw_slice_ceiling`'s floor this run** (usually ~1.0-1.3×). The table below is computed the same as always, but its own denominator looks drifted -- a low ratio here likely means the baseline got worse, not that `query2_static` got better. Cross-check against `raw_slice_ceiling`/`real_query1_inline_never_wrapper` directly rather than trusting the flags below at face value this run.")
        print()

    print("| N | query_static (1 col) | query2_static (2 col) | Ratio |")
    print("|---|---|---|---|")
    worst_ratio = 0.0
    for n in shared_ns:
        ratio = two[n] / single[n]
        worst_ratio = max(worst_ratio, ratio)
        # ~1.28-1.35x is the real, stable, currently-observed state
        # (see this script's printed note above) -- not the 1.0x a
        # zero-abstraction ceiling would suggest, but not a regression
        # either. Thresholds set with margin above that, not at it.
        # Forced to at least ⚠️ when the baseline itself looks drifted
        # (see above) -- a numerically-low ratio isn't "good" if it
        # only looks that way because the denominator broke too.
        if baseline_drift is not None and baseline_drift > BASELINE_DRIFT_THRESHOLD:
            flag = "⚠️" if ratio <= 3.0 else "🔴"
        else:
            flag = "✅" if ratio <= 1.6 else ("⚠️" if ratio <= 3.0 else "🔴")
        print(f"| {n:,} | {single[n]:,.0f} ns | {two[n]:,.0f} ns | {flag} {ratio:.2f}× |")
    print()
    if baseline_drift is not None and baseline_drift > BASELINE_DRIFT_THRESHOLD:
        print(f"> ⚠️ Baseline looks drifted this run (see warning above) -- worst ratio {worst_ratio:.2f}× is not being read as a clean pass regardless of the threshold it'd otherwise clear.")
    elif worst_ratio > 3.0:
        print(f"> 🔴 **Worst observed ratio this run: {worst_ratio:.1f}×** -- investigate before trusting other numbers in this run.")
    elif worst_ratio > 1.6:
        print(f"> ⚠️ Worst observed ratio this run: {worst_ratio:.2f}× -- above the ~1.28-1.35x currently-observed baseline, worth a look.")
    else:
        print(f"> ✅ Worst observed ratio this run: {worst_ratio:.2f}× -- consistent with the ~1.28-1.35x currently-observed baseline.")
