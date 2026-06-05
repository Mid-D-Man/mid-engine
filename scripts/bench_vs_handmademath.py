# scripts/bench_vs_handmademath.py
# Parses HandmadeMath C and criterion Rust output, prints side-by-side markdown.
# Called from .github/workflows/bench-vs-handmademath.yml.
# Reads: hmm-bench-raw.txt  midmath-bench-raw.txt
#
# NOTE: The YAML heredoc version had corrupted regex (missing backslashes).
# This file is the corrected canonical version.

import re

RE_ANSI = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

# ── Parse HandmadeMath output with section awareness ──────────────────────────
hmm = {}
try:
    raw = open('hmm-bench-raw.txt', encoding='utf-8', errors='replace').read()
    section = 'other'
    RE_OP = re.compile(r'^\s+([\w/\s()]+?)\s{2,}([\d.]+)\s+ns/op')
    for line in raw.splitlines():
        stripped = line.strip().lower()
        for sec in ('vec3', 'vec4', 'quat', 'mat4'):
            if stripped.startswith(sec):
                section = sec
                break
        m = RE_OP.match(line)
        if m:
            key = m.group(1).strip()
            key = re.sub(r'/handmademath\b', '', key).strip('/ ')
            hmm[f'{section}/{key}'] = float(m.group(2))
except FileNotFoundError:
    print("*(hmm-bench-raw.txt not found)*")

# ── Parse criterion output ────────────────────────────────────────────────────
RE_CRIT = re.compile(
    r'^(\S[^\n]*?)\s*time:\s+\[\s*([\d.]+\s+\S+)\s+([\d.]+\s+\S+)\s+([\d.]+\s+\S+)\s*\]',
    re.MULTILINE,
)
mm = {}
try:
    raw2 = RE_ANSI.sub('', open('midmath-bench-raw.txt', encoding='utf-8', errors='replace').read())
    for m in RE_CRIT.finditer(raw2):
        name = m.group(1).strip()
        mean = m.group(3).strip()
        if 'mid-math' in name:
            clean = re.sub(r'/mid-math[^/\s]*', '', name).strip('/ ')
            mm[clean] = mean
except FileNotFoundError:
    print("*(midmath-bench-raw.txt not found)*")

print("### Side-by-side: mid-math vs HandmadeMath v2")
print("")
print("| Operation | mid-math (criterion) | HandmadeMath (ns/op) |")
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

for hk, mk, label in ops:
    hv = f"{hmm[hk]:.2f} ns" if hk in hmm else "—"
    mv = mm.get(mk, "—")
    print(f"| {label} | {mv} | {hv} |")

print("")
print("---")
print("### Raw HandmadeMath output")
print("```")
try:
    print(open('hmm-bench-raw.txt').read().strip())
except FileNotFoundError:
    print("(not found)")
print("```")
