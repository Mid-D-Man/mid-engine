# scripts/bench_vs_c_libs.py
# Parses C and Rust benchmark output and prints a unified markdown comparison.
# Called from .github/workflows/bench-vs-c-libs.yml.
# Reads: /tmp/cglm.txt  /tmp/hmm.txt  /tmp/dxm.txt  /tmp/rust.txt

import re
import sys
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
       'section_header:'   <- starts a section
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

        # Section header: starts at column 0, ends with ':', no '/op'
        if not raw.startswith(' ') and line.endswith(':') and '/op' not in line:
            m = re.match(r'^(\w+)\b', line)
            if m:
                section = m.group(1).lower()
            continue

        if '/op' not in line:
            continue

        # Result line: "label   X.XX ns/op"
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
       'group/op/impl   time:   [lo  MID  hi]'
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
        if len(parts) >= 3:
            section = parts[0]
            op      = parts[1]
            impl    = '/'.join(parts[2:])
            key = f"{section}/{op}"
            results.setdefault(key, {})[impl] = (f"{mean_val} {unit}", ns)

    return results


cglm_data = parse_c('/tmp/cglm.txt')
hmm_data  = parse_c('/tmp/hmm.txt')
dxm_data  = parse_c('/tmp/dxm.txt')
rust_data = parse_rust('/tmp/rust.txt')

OPS = [
    ('vec3',     'add',             'Vec3 add'),
    ('vec3',     'dot',             'Vec3 dot'),
    ('vec3',     'cross',           'Vec3 cross'),
    ('vec3',     'normalize',       'Vec3 normalize'),
    ('vec3',     'lerp',            'Vec3 lerp'),
    ('rotation', 'mul',             'Quat mul'),
    ('rotation', 'rotate',          'Quat rotate vec'),
    ('rotation', 'slerp',           'Quat slerp'),
    ('mat4',     'mul',             'Mat4 mul'),
    ('mat4',     'transform_point', 'Mat4 transform_point'),
    ('mat4',     'inverse_general', 'Mat4 inverse_general'),
]

print("### Unified Comparison")
print("")
print("| Operation | mid-math | glam | cglm | HandmadeMath | DirectXMath |")
print("|---|---|---|---|---|---|")

for section, op, label in OPS:
    key = f"{section}/{op}"

    rust_ops = rust_data.get(key, {})
    mid_str  = '—'
    glam_str = '—'
    for k, (s, ns) in rust_ops.items():
        kl = k.lower()
        if 'mid-math' in kl and mid_str  == '—':
            mid_str  = s
        if 'glam'     in kl and glam_str == '—':
            glam_str = s

    c1 = cglm_data.get(key, ('—',))[0]
    c2 = hmm_data.get(key,  ('—',))[0]
    c3 = dxm_data.get(key,  ('—',))[0]

    print(f"| {label} | {mid_str} | {glam_str} | {c1} | {c2} | {c3} |")

print("")

# Mat4 multiply gap analysis
mat4_key = "mat4/mul"
rust_mat4 = rust_data.get(mat4_key, {})
mid_ns  = next((v[1] for k, v in rust_mat4.items() if 'mid-math' in k.lower()), None)
glam_ns = next((v[1] for k, v in rust_mat4.items() if 'glam'     in k.lower()), None)
cglm_ns = cglm_data.get(mat4_key, (None, None))[1]
dxm_ns  = dxm_data.get(mat4_key,  (None, None))[1]

print("#### Mat4 multiply gap analysis")
print("")
if mid_ns and glam_ns:
    ratio = mid_ns / glam_ns
    direction = "slower" if ratio > 1.0 else "faster"
    display_ratio = ratio if ratio > 1.0 else glam_ns / mid_ns
    print(f"- mid-math vs glam:       **{display_ratio:.2f}×** {direction}  ({fmt_ns(mid_ns)} vs {fmt_ns(glam_ns)})")
    if ratio > 1.1:
        print(f"  - Historically caused by `[[f32;4];4]` storage → pointer ABI → 8× `_mm_load_ps` before math")
        print(f"  - Fix: named `Vec4` fields → data in XMM registers → target ≤7 ns")
if cglm_ns and glam_ns:
    print(f"- cglm vs glam:           **{cglm_ns / glam_ns:.2f}×** ({fmt_ns(cglm_ns)} vs {fmt_ns(glam_ns)})")
if dxm_ns and glam_ns:
    print(f"- DirectXMath vs glam:    **{dxm_ns / glam_ns:.2f}×** ({fmt_ns(dxm_ns)} vs {fmt_ns(glam_ns)})")
