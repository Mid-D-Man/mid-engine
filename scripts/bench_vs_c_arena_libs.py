# scripts/bench_vs_c_arena_libs.py
# Parses C and Rust arena-allocator benchmark output and prints a unified
# markdown comparison. Called from .github/workflows/bench-vs-c-arena-libs.yml.
# Reads: /tmp/tsoding.txt  /tmp/apr.txt  /tmp/talloc.txt  /tmp/rust.txt
#        /tmp/drop_arena.txt
#
# parse_c()/parse_rust()/to_ns()/fmt_ns() below are copied unchanged from
# scripts/bench_vs_c_libs.py (mid-math's own C-vs-Rust comparison script) --
# same output shapes (this project's report()-line convention on the C side,
# criterion's own on the Rust side), so the same regexes apply verbatim. Kept
# as a second copy rather than a shared import: these scripts are each
# invoked as a single free-standing file from their own workflow's `run:`
# step, not as an installed package with a shared module path.

import re
from collections import OrderedDict

RE_ANSI = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

# The actual bug this fix is for: Rust's criterion output reports total
# time for one iteration of the closure, which allocates N items in a
# loop -- so "775.60 us" for SlotArena's insert group means 775.60 us
# for all 100,000 inserts, not one. The C programs' report() function
# already divides by N itself before printing, so a C line reads
# "7.34 ns/op" -- already per-operation. Printing those two numbers in
# the same table column without accounting for that difference is
# exactly what made the C libraries look faster than they are: a
# 4-digit microsecond figure next to a 1-digit nanosecond figure looks
# like a huge gap by eye, when normalized they're close (and the Rust
# side usually wins). Every number this script prints from here on is
# per-operation, Rust included -- the raw batch-total is never shown
# without being divided by the real number of operations it covers.
N = 100_000


def to_ns(val_str, unit):
    v = float(val_str)
    u = unit.lower().strip().rstrip('/op').strip()
    if 'µs' in u or 'us' in u:
        return v * 1_000
    if 'ms' in u:
        return v * 1_000_000
    if u == 's':
        return v * 1_000_000_000
    return v  # ns


def fmt_ns(ns):
    if ns >= 1_000_000:
        return f"{ns / 1_000_000:.2f} ms"
    if ns >= 1_000:
        return f"{ns / 1_000:.2f} µs"
    return f"{ns:.2f} ns"


def parse_c(path):
    """Parse C benchmark output. Format:
       'section_header:'   <- starts a section (unused here -- these three
                               programs are each a single flat namespace of
                               ops, no vec3/mat4-style sections)
       '  op/impl   X.XX ns/op'
    """
    results = {}
    try:
        lines = open(path).readlines()
    except Exception:
        return results

    section = None
    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        if not raw.startswith(' ') and line.endswith(':') and '/op' not in line:
            m = re.match(r'^(\w+)\b', line)
            if m:
                section = m.group(1).lower()
            continue

        if '/op' not in line:
            continue

        parts = line.rsplit(None, 2)
        if len(parts) != 3:
            continue
        label_full, val_str, unit_op = parts
        if '/op' not in unit_op.lower():
            continue

        try:
            ns = to_ns(val_str, unit_op)
        except Exception:
            continue

        op = label_full.split('/')[0].strip()
        key = f"{section}/{op}" if section else op
        results[key] = (f"{val_str} {unit_op.replace('/op', '').strip()}", ns)

    return results


def parse_rust(path):
    """Parse criterion output. Format:
       'group/impl   time:   [lo  MID  hi]'
    Note the two-level grouping here (group/impl), one level shallower than
    bench_vs_c_libs.py's group/op/impl -- this bench's criterion groups
    (insert, get, remove_half_then_reinsert_half, gc) already name the
    operation as the group itself, since (unlike mid-math's vec3/mat4/
    rotation split) every crate in this survey is being compared on the
    exact same operation set, not a per-type API.
    """
    results = {}
    try:
        raw = RE_ANSI.sub('', open(path, errors='replace').read())
    except Exception:
        return results

    RE = re.compile(
        r'^(\S[^\n]+?)\s+time:\s+\[\s*[\d.]+ \S+\s+([\d.]+) (\S+)\s+[\d.]+ \S+\s*\]',
        re.MULTILINE,
    )
    for m in RE.finditer(raw):
        name     = m.group(1).strip()
        mean_val = m.group(2)
        unit     = m.group(3)
        ns = to_ns(mean_val, unit)
        parts = name.split('/')
        if len(parts) >= 2:
            group = parts[0]
            impl  = '/'.join(parts[1:])
            results.setdefault(group, {})[impl] = (f"{mean_val} {unit}", ns)

    return results


tsoding_data = parse_c('/tmp/tsoding.txt')
apr_data     = parse_c('/tmp/apr.txt')
talloc_data  = parse_c('/tmp/talloc.txt')
rust_data    = parse_rust('/tmp/rust.txt')
drop_arena_data = parse_c('/tmp/drop_arena.txt')

# ── insert / get: every source measured the exact same operation ──────────
# Every figure below is per-operation. Rust's raw criterion number
# (total time for a 100,000-op batch) is divided by N here before
# display; the C programs already report per-operation and are shown
# as-is. See the N= comment above for why this matters -- this is the
# fix for a real, repeated confusion, not a style choice.
#
# Grouped by real approach (docs/mid-arena.md's own survey taxonomy),
# not registration order -- comparing SlotArena against slab used to
# mean scanning past six unrelated rows to find slotmap/thunderdome,
# its actual peer group. Fastest-to-slowest within each group; the
# order across groups below is deliberate (safest-and-most-general
# first, most specialized/narrowest-contract last), not alphabetical.

APPROACH_ORDER = [
    "Vec + freelist, ABA-safe (generation-checked)",
    "Vec + freelist, no ABA check",
    "Linked arena chunks (bump, no per-item reuse)",
    "Indexed Vec (no reuse at all)",
    "Sharded / lock-free",
    "Hashset dedup (interning)",
]

APPROACH_OF = {
    "mid-arena/SlotArena": "Vec + freelist, ABA-safe (generation-checked)",
    "mid-arena/CompactSlotArena": "Vec + freelist, ABA-safe (generation-checked)",
    "slotmap": "Vec + freelist, ABA-safe (generation-checked)",
    "generational-arena": "Vec + freelist, ABA-safe (generation-checked)",
    "typed-generational-arena": "Vec + freelist, ABA-safe (generation-checked)",
    "atomic-arena": "Vec + freelist, ABA-safe (generation-checked)",
    "thunderdome": "Vec + freelist, ABA-safe (generation-checked)",
    "slab": "Vec + freelist, no ABA check",
    "mid-arena/BumpArena": "Linked arena chunks (bump, no per-item reuse)",
    "bumpalo": "Linked arena chunks (bump, no per-item reuse)",
    "typed-arena": "Linked arena chunks (bump, no per-item reuse)",
    "id-arena": "Indexed Vec (no reuse at all)",
    "sharded-slab": "Sharded / lock-free",
    "internment/ArenaIntern (unique)": "Hashset dedup (interning)",
}

print("### insert / get — every source, same N=100,000, same 16-byte payload, all figures per-operation")
print("")
print("Grouped by real approach, not benchmark registration order — see "
      "docs/mid-arena.md's own survey taxonomy. `mid-arena`'s own types are "
      "bolded. Fastest to slowest within each group.")
print("")

rust_insert = rust_data.get('insert', {})
rust_get    = rust_data.get('get', {})
rust_impls  = OrderedDict.fromkeys(list(rust_insert.keys()) + list(rust_get.keys()))

rows_by_approach = {a: [] for a in APPROACH_ORDER}
unclassified = []
for impl in rust_impls:
    ins_ns = rust_insert.get(impl, (None, None))[1]
    get_ns = rust_get.get(impl, (None, None))[1]
    approach = APPROACH_OF.get(impl)
    row = (impl, ins_ns, get_ns)
    if approach:
        rows_by_approach[approach].append(row)
    else:
        unclassified.append(row)

for approach in APPROACH_ORDER:
    rows = rows_by_approach[approach]
    if not rows:
        continue
    # Fastest insert first; entries with no insert figure (shouldn't
    # happen for this table, but don't crash if a name mismatches) sort
    # last rather than erroring.
    rows.sort(key=lambda r: (r[1] is None, r[1] if r[1] is not None else 0))
    print(f"**{approach}**")
    print("")
    print("| Implementation | insert (per-op) | get (per-op) |")
    print("|---|---|---|")
    for impl, ins_ns, get_ns in rows:
        name = f"**{impl}**" if impl.startswith("mid-arena/") else impl
        ins = fmt_ns(ins_ns / N) if ins_ns is not None else '—'
        get = fmt_ns(get_ns / N) if get_ns is not None else '—'
        print(f"| {name} | {ins} | {get} |")
    print("")

    # drop_arena isn't a criterion result (see drop_arena_standalone.rs's
    # own doc comment for why -- an invariant-lifetime issue, not a
    # style choice), so it isn't in rust_data/rows_by_approach at all.
    # Printed as its own row right after the bump-allocator group it's
    # actually built on top of (typed-arena + a free list on top), using
    # the same N and payload, already per-operation from its own
    # report() output -- not divided by N again.
    if approach == "Linked arena chunks (bump, no per-item reuse)":
        da_ins = drop_arena_data.get('insert', (None, None))[1]
        da_get = drop_arena_data.get('get', (None, None))[1]
        if da_ins is not None or da_get is not None:
            print("**Bump + free list (typed-arena-based, with reuse)**")
            print("")
            print("`drop_arena` wraps `typed-arena` with a free list so individual "
                  "items can be reclaimed (`docs/mid-arena.md`'s survey) -- a real "
                  "middle ground between the no-reuse bump allocators above and the "
                  "full generation-checked arenas below. Measured with `std::time::Instant`, "
                  "not criterion, for a real, confirmed reason (see "
                  "`drop_arena_standalone.rs`'s own doc comment), so single-run rather "
                  "than statistical -- treat this row as a real but noisier data point "
                  "than the criterion-measured ones above and below it.")
            print("")
            print("| Implementation | insert (per-op) | get (per-op) |")
            print("|---|---|---|")
            ins = fmt_ns(da_ins) if da_ins is not None else '—'
            get = fmt_ns(da_get) if da_get is not None else '—'
            print(f"| drop_arena | {ins} | {get} |")
            print("")

if unclassified:
    print("**Other (not yet categorized in this script)**")
    print("")
    print("| Implementation | insert (per-op) | get (per-op) |")
    print("|---|---|---|")
    for impl, ins_ns, get_ns in unclassified:
        ins = fmt_ns(ins_ns / N) if ins_ns is not None else '—'
        get = fmt_ns(get_ns / N) if get_ns is not None else '—'
        print(f"| {impl} | {ins} | {get} |")
    print("")

print("**C libraries** (own timing methodology — clock_gettime, not "
      "criterion — see the file header of each `.c` source; shown "
      "separately rather than folded into the groups above since they "
      "aren't Rust approaches, but the same per-operation unit applies)")
print("")
print("| Implementation | insert (per-op) | get (per-op) |")
print("|---|---|---|")
for label, data in (
    ("tsoding/arena.h", tsoding_data),
    ("APR pools", apr_data),
    ("talloc_pool", talloc_data),
):
    ins_ns = data.get('insert', (None, None))[1]
    get_ns = data.get('get', (None, None))[1]
    # Already per-op from the C side's own report() -- not divided by N.
    ins = fmt_ns(ins_ns) if ins_ns is not None else '—'
    get = fmt_ns(get_ns) if get_ns is not None else '—'
    print(f"| {label} | {ins} | {get} |")

print("")
print("(Rust: raw criterion batch time / 100,000. C: already per-operation "
      "from each program's own timing. Same unit, same meaning, every row "
      "on this page.)")
print("")

# ── reuse / reset: each source's own real semantics, not forced into one shape ──

print("### Bulk reuse / free — not a shared operation, each API's own real shape")
print("")
print("mid-arena's `SlotArena`/`slab`/`slotmap`/`generational-arena`/`thunderdome` "
      "remove *and reuse a single slot*; `tsoding/arena.h`/APR pools reset the "
      "*whole arena* at once; talloc frees per-item (but a pooled child's bytes "
      "aren't reclaimed until the whole pool is freed — see talloc_bench.c's own "
      "header comment). Three different reuse models, reported separately rather "
      "than collapsed into one row.")
print("")

rust_churn = rust_data.get('remove_half_then_reinsert_half', {})
if rust_churn:
    print("**Rust: remove half, reinsert half (single-slot reuse), per-operation**")
    print("")
    print("Most implementations here run 100,000 inserts + 50,000 removes + "
          "50,000 reinserts (200,000 real operations total) in one measured "
          "closure; `sharded-slab` only runs the insert + remove half (no "
          "reinsert measured for it, 150,000 operations) -- dividing each by "
          "its own real operation count, not one shared N, same reasoning "
          "as the insert/get table above.")
    print("")
    print("| Implementation | per-op |")
    print("|---|---|")
    for impl, (_s, ns) in rust_churn.items():
        ops = 150_000 if 'sharded-slab' in impl else 200_000
        print(f"| {impl} | {fmt_ns(ns / ops)} |")
    da_churn = drop_arena_data.get('remove_half_reinsert_half', (None, None))[1]
    if da_churn is not None:
        print(f"| drop_arena (std::time::Instant, single run — see note above) | {fmt_ns(da_churn)} |")
    print("")

print("**tsoding/arena.h: whole-arena reset**")
print("")
for op in ('reset_whole_arena', 'reinsert_after_reset'):
    if op in tsoding_data:
        print(f"- `{op}`: {tsoding_data[op][0]}")
print("")

print("**APR pools: whole-pool clear**")
print("")
for op in ('clear_whole_pool', 'reinsert_after_clear'):
    if op in apr_data:
        print(f"- `{op}`: {apr_data[op][0]}")
print("")

print("**talloc: per-item free (no reclaim) vs whole-pool free**")
print("")
for op in ('free_half_no_reclaim', 'free_whole_pool'):
    if op in talloc_data:
        print(f"- `{op}`: {talloc_data[op][0]}")
print("")

# ── gc: alloc + collector pause, doesn't share the remove/reuse shape ──────

rust_gc = rust_data.get('gc', {})
if rust_gc:
    print("### gc crate — allocation and collector-pause cost")
    print("")
    print("`alloc` is 100,000 real allocations, shown per-operation like "
          "every other insert figure above. `force_collect` is a single "
          "collection sweep over those same 100,000 live objects, not "
          "100,000 separate operations -- shown as the real total cost of "
          "that one sweep, not divided by N, since dividing it would "
          "understate what a single collection pause actually costs.")
    print("")
    print("| Operation | time |")
    print("|---|---|")
    for impl, (_s, ns) in rust_gc.items():
        if 'alloc' in impl and 'force_collect' not in impl:
            print(f"| {impl} (per-op) | {fmt_ns(ns / N)} |")
        else:
            print(f"| {impl} (total, one sweep over {N:,} live objects) | {fmt_ns(ns)} |")
    print("")
    print("The `force_collect` row is exactly the kind of cost "
          "`docs/mid-arena.md`'s \"Explicitly out of scope: garbage "
          "collection\" section is about: a real, measured pause, not a "
          "hypothetical one.")
