# scripts/bench_vs_c_arena_libs.py
# Parses C and Rust arena-allocator benchmark output and prints a unified
# markdown comparison. Called from .github/workflows/bench-vs-c-arena-libs.yml.
# Reads: /tmp/tsoding.txt  /tmp/apr.txt  /tmp/talloc.txt  /tmp/rust.txt
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

# ── insert / get: every source measured the exact same operation ──────────

print("### insert / get — every source, same N=100,000, same 16-byte payload")
print("")
print("| Implementation | insert | get |")
print("|---|---|---|")

rust_insert = rust_data.get('insert', {})
rust_get    = rust_data.get('get', {})
rust_impls  = OrderedDict.fromkeys(list(rust_insert.keys()) + list(rust_get.keys()))
for impl in rust_impls:
    ins = rust_insert.get(impl, ('—',))[0]
    get = rust_get.get(impl, ('—',))[0]
    print(f"| {impl} | {ins} | {get} |")

for label, data in (
    ("tsoding/arena.h (C)", tsoding_data),
    ("APR pools (C)", apr_data),
    ("talloc_pool (C)", talloc_data),
):
    ins = data.get('insert', ('—',))[0]
    get = data.get('get', ('—',))[0]
    print(f"| {label} | {ins} | {get} |")

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
    print("**Rust: remove half, reinsert half (single-slot reuse)**")
    print("")
    print("| Implementation | time |")
    print("|---|---|")
    for impl, (s, _ns) in rust_churn.items():
        print(f"| {impl} | {s} |")
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
    print("| Operation | time |")
    print("|---|---|")
    for impl, (s, _ns) in rust_gc.items():
        print(f"| {impl} | {s} |")
    print("")
    print("The `force_collect` row is exactly the kind of cost "
          "`docs/mid-arena.md`'s \"Explicitly out of scope: garbage "
          "collection\" section is about: a real, measured pause, not a "
          "hypothetical one.")
