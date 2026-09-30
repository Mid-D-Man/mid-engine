# scripts/bench_mid_ecs_ffi.py
# Parses bench-mid-ecs-ffi-raw.txt (criterion output from
# crates/mid-ecs/benches/ffi_overhead.rs) and prints a markdown summary:
# one table per group, then an "FFI vs Rust" table pairing each `ffi_*`
# variant with its `rust_*` counterpart at the same size. Called from
# .github/workflows/bench-mid-ecs-ffi.yml.
#
# Same RE_ANSI/RE_RESULT parsing shape as scripts/bench_mid_ecs_archetype_core.py.
# Two differences, both because this suite's ids aren't uniform: a group
# either sweeps a size (`group/variant/N`) or doesn't (`group/variant`,
# resource_access), and the variants are named by role, not by a constant
# `mid-ecs` label, so the variant stays in the row key instead of being
# folded away.
#
# No pass/fail thresholds on the ratios. Unlike archetype_core.rs there is
# no real-CI history yet to set one against, and inventing a number now
# would just be a guess dressed as a guard. The one hard check is
# structural: every expected group must have produced results, so a group
# that silently stopped running shows up as a red line instead of a
# shorter table. Once a few real runs exist, thresholds belong here, set
# from those numbers.

import re
import sys
from collections import OrderedDict

RE_ANSI = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
RE_RESULT = re.compile(
    r'^(\S[^\n]+?)\s+time:\s+\[\s*([\d.]+\s+\S+)\s+([\d.]+\s+\S+)\s+([\d.]+\s+\S+)\s*\]'
    r'(?:\s*\n\s*thrpt:\s+\[\s*([\d.]+\s+\S+)\s+([\d.]+\s+\S+)\s+([\d.]+\s+\S+)\s*\])?',
    re.MULTILINE,
)

# group -> the (rust variant, ffi variant) pairs worth a ratio row, plus the
# label the swept parameter has in that group.
EXPECTED = OrderedDict([
    ('lifecycle_spawn_despawn', ('N', [('rust', 'ffi')])),
    ('span_read', ('N', [('rust_span', 'ffi_span'),
                         ('rust_span_call_only', 'ffi_span_call_only')])),
    ('archetypes_matching', ('K', [('rust_collect', 'ffi_count_then_fill')])),
    ('change_rows', ('N', [('rust_query_changed', 'ffi_changed_rows')])),
    ('resource_access', (None, [('rust_read', 'ffi_read'),
                                ('rust_write', 'ffi_write')])),
])

try:
    raw = open('bench-mid-ecs-ffi-raw.txt', encoding='utf-8', errors='replace').read()
except FileNotFoundError:
    print("*(bench-mid-ecs-ffi-raw.txt not found)*")
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
        return val  # ns
    except Exception:
        return None


def fmt_ns(ns):
    if ns is None:
        return '—'
    if ns >= 1_000_000:
        return f'{ns / 1_000_000:,.2f} ms'
    if ns >= 1_000:
        return f'{ns / 1_000:,.2f} µs'
    return f'{ns:,.2f} ns'


# results[group][variant][param] = (mean_str, mean_ns, thrpt_str); param is
# an int for swept groups and None for the unswept one.
results = OrderedDict()
for m in RE_RESULT.finditer(text):
    parts = m.group(1).strip().split('/')
    if len(parts) == 3 and parts[2].isdigit():
        group, variant, param = parts[0], parts[1], int(parts[2])
    elif len(parts) == 2:
        group, variant, param = parts[0], parts[1], None
    else:
        continue
    mean = m.group(3).strip()
    thrpt = m.group(6).strip() if m.group(6) else None
    results.setdefault(group, OrderedDict()).setdefault(variant, {})[param] = (
        mean, to_ns(mean), thrpt)

missing = [g for g in EXPECTED if g not in results]

for group, (label, _) in EXPECTED.items():
    variants = results.get(group)
    if not variants:
        continue
    print(f"#### {group}")
    params = sorted({p for v in variants.values() for p in v}, key=lambda p: (p is None, p))
    if params == [None]:
        print("| Variant | Mean |")
        print("|---|---|")
        for variant, by_param in variants.items():
            print(f"| {variant} | {by_param[None][0]} |")
    else:
        head = "| Variant | " + " | ".join(f"{label}={p:,}" for p in params) + " |"
        print(head)
        print("|---|" + "---|" * len(params))
        for variant, by_param in variants.items():
            cells = [by_param[p][0] if p in by_param else '—' for p in params]
            print(f"| {variant} | " + " | ".join(cells) + " |")
    print()

print("#### FFI vs Rust")
print()
print("`ffi` ÷ `rust` at the same size, both driving the same world in the same")
print("process. The `ffi` side is the `extern \"C\"` function called from Rust, so it")
print("includes its null checks, `catch_unwind` and status plumbing but not the ABI")
print("crossing from a separately compiled C program: read this as a lower bound on")
print("FFI overhead. No thresholds yet, see this script's header for why.")
print()
print("| Group | Pair | Size | Rust | FFI | FFI ÷ Rust |")
print("|---|---|---|---|---|---|")
rows_printed = 0
for group, (label, pairs) in EXPECTED.items():
    variants = results.get(group, {})
    for rust_v, ffi_v in pairs:
        rust_p = variants.get(rust_v, {})
        ffi_p = variants.get(ffi_v, {})
        for p in sorted(set(rust_p) & set(ffi_p), key=lambda p: (p is None, p)):
            r_ns, f_ns = rust_p[p][1], ffi_p[p][1]
            if not r_ns or not f_ns:
                continue
            size = '—' if p is None else f"{label}={p:,}"
            print(f"| {group} | `{rust_v}` → `{ffi_v}` | {size} | "
                  f"{fmt_ns(r_ns)} | {fmt_ns(f_ns)} | {f_ns / r_ns:.2f}× |")
            rows_printed += 1
print()

if missing:
    print(f"> 🔴 **No results parsed for: {', '.join(missing)}.** The bench file defines "
          "these groups, so they either failed before producing output or the raw log "
          "is truncated. Check the raw output below and the diagnostics step.")
elif rows_printed == 0:
    print("> 🔴 No FFI/Rust pairs could be matched. The raw log parsed, but no variant pair "
          "shared a size, which points at a naming change in the bench file, not a slow run.")
else:
    print(f"> ✅ All {len(EXPECTED)} groups produced results ({rows_printed} FFI/Rust pairs compared).")
