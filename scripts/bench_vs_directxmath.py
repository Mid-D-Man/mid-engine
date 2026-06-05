# scripts/bench_vs_directxmath.py
# Parses DirectXMath C++ and criterion Rust output, prints side-by-side markdown.
# Called from .github/workflows/bench-vs-directxmath.yml.
# Reads: dxmath-bench-raw.txt  midmath-bench-raw.txt

import re

RE_ANSI = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

# ── Parse DirectXMath output with section awareness ───────────────────────────
dx = {}
try:
    with open('dxmath-bench-raw.txt', encoding='utf-8', errors='replace') as f:
        raw = f.read()
    section = 'other'
    RE_OP = re.compile(r'^\s+([\w/]+)\s{2,}([\d.]+)\s+ns/op')
    for line in raw.splitlines():
        stripped = line.strip().lower()
        for sec in ('vec3', 'vec4', 'quat', 'mat4'):
            if stripped.startswith(sec):
                section = sec
                break
        m = RE_OP.match(line)
        if m:
            key = m.group(1).strip()
            key = re.sub(r'/directxmath\b', '', key).strip('/ ')
            dx[f'{section}/{key}'] = float(m.group(2))
except FileNotFoundError:
    print("*(dxmath-bench-raw.txt not found)*")

# ── Parse criterion output ────────────────────────────────────────────────────
RE_CRIT = re.compile(
    r'^(\S[^\n]*?)\s*time:\s+\[\s*([\d.]+\s+\S+)\s+([\d.]+\s+\S+)\s+([\d.]+\s+\S+)\s*\]',
    re.MULTILINE,
)
mm = {}
try:
    with open('midmath-bench-raw.txt', encoding='utf-8', errors='replace') as f:
        raw2 = RE_ANSI.sub('', f.read())
    for m in RE_CRIT.finditer(raw2):
        name = m.group(1).strip()
        mean = m.group(3).strip()
        if 'mid-math' in name:
            clean = re.sub(r'/mid-math[^/\s]*', '', name).strip('/ ')
            mm[clean] = mean
except FileNotFoundError:
    print("*(midmath-bench-raw.txt not found)*")

print("### Side-by-side: mid-math vs DirectXMath")
print("")
print("| Operation | mid-math (criterion) | DirectXMath (ns/op) |")
print("|---|---|---|")

ops = [
    ('vec3/add',             'vec3/add',             'vec3 add'),
    ('vec3/dot',             'vec3/dot',             'vec3 dot'),
    ('vec3/cross',           'vec3/cross',           'vec3 cross'),
    ('vec3/normalize',       'vec3/normalize',       'vec3 normalize'),
    ('vec3/lerp',            'vec3/lerp',             'vec3 lerp'),
    ('quat/mul',             'rotation/mul',         'quat mul'),
    ('quat/rotate',          'rotation/rotate',      'quat rotate'),
    ('quat/slerp',           'rotation/slerp',       'quat slerp'),
    ('mat4/mul',             'mat4/mul',             'mat4 mul'),
    ('mat4/transform_point', 'mat4/transform_point', 'mat4 transform_point'),
    ('mat4/inverse_general', 'mat4/inverse_general', 'mat4 inverse'),
]
for dk, mk, label in ops:
    dv = f"{dx[dk]:.2f} ns" if dk in dx else "—"
    mv = mm.get(mk, "—")
    print(f"| {label} | {mv} | {dv} |")

print("")
print("---")
print("### Raw DirectXMath output")
print("```text")
try:
    with open('dxmath-bench-raw.txt') as f:
        print(f.read().strip())
except FileNotFoundError:
    print("(not found)")
print("```")
