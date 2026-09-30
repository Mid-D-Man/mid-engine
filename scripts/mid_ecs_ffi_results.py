#!/usr/bin/env python3
# scripts/mid_ecs_ffi_results.py
#
# NOTICE: Full documentation, design decisions, and fix history for this
# file live in docs/mid-ecs.md, section "FFI test and bench"
#
# Turns the raw logs of .github/workflows/mid-ecs-ffi-test.yml into the
# JSON shape the per-crate dashboards read (same shape as
# scripts/parse_test_results.py: build/branch/commit/rust_version/crate,
# summary, suites[{name, passed, failed, ignored, duration_s, tests[]}]),
# and prints a markdown job summary from that JSON.
#
#   mid_ecs_ffi_results.py parse --out results.json --build 3 ... \
#       --cargo-raw ffi-rust-raw.txt \
#       --c-log "C smoke test, libmid_ecs.so=ffi-smoke-so-raw.txt" \
#       --c-log "C smoke test, libmid_ecs.a=ffi-smoke-a-raw.txt" \
#       --valgrind ffi-valgrind-raw.txt --valgrind-status 0
#   mid_ecs_ffi_results.py summary results.json
#
# Suites produced, in order: the Rust-side `ffi::` unit tests (parsed by
# scripts/parse_test_results.py's own parser, not a copy of it), one suite
# per --c-log (each `ok:`/`FAIL:` line of test.c's own output is one test),
# and a "C smoke test under valgrind" suite with one test per condition
# valgrind is asked to enforce.
#
# A C log that is missing, empty, or lacks test.c's final "check(s)
# failed" line becomes one FAILED test named "program ran to completion",
# not an empty passing suite: a crash or a link error prints no `FAIL:`
# lines at all, and would otherwise read as zero failures.

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parse_test_results import parse_file, strip_ansi  # noqa: E402

RE_OK = re.compile(r'^ok:\s+(.*\S)\s*$')
RE_FAIL = re.compile(r'^FAIL:\s+(.*\S)\s+\(line (\d+)\)\s*$')
RE_FINAL = re.compile(r'^=== (\d+) check\(s\) failed ===\s*$')


def suite(name, tests):
    return {
        'name': name,
        'passed': sum(1 for t in tests if t['status'] == 'passed'),
        'failed': sum(1 for t in tests if t['status'] == 'failed'),
        'ignored': 0,
        'duration_s': 0.0,
        'tests': tests,
    }


def test(name, ok, output=''):
    return {
        'name': name,
        'status': 'passed' if ok else 'failed',
        'duration_ms': 0,
        'output': output,
    }


def parse_c_log(label, path):
    try:
        text = strip_ansi(Path(path).read_text(encoding='utf-8', errors='replace'))
    except FileNotFoundError:
        text = ''
    tests = []
    final = None
    for line in text.splitlines():
        m = RE_OK.match(line)
        if m:
            tests.append(test(m.group(1), True))
            continue
        m = RE_FAIL.match(line)
        if m:
            tests.append(test(m.group(1), False, f'test.c line {m.group(2)}'))
            continue
        m = RE_FINAL.match(line)
        if m:
            final = int(m.group(1))
    failed_lines = sum(1 for t in tests if t['status'] == 'failed')
    completed = final is not None and final == failed_lines
    detail = ''
    if final is None:
        detail = ('no "=== N check(s) failed ===" line: the program did not run to '
                  'the end (compile/link error, crash, or missing log)')
    elif final != failed_lines:
        detail = f'program reported {final} failed check(s) but {failed_lines} FAIL lines were logged'
    tests.append(test('program ran to completion', completed, detail))
    return suite(label, tests)


def parse_valgrind(path, status):
    try:
        text = strip_ansi(Path(path).read_text(encoding='utf-8', errors='replace'))
    except FileNotFoundError:
        text = ''
    m = re.search(r'ERROR SUMMARY:\s+(\d+) errors', text)
    errors = int(m.group(1)) if m else None
    leaked = re.search(r'definitely lost:\s+([\d,]+) bytes', text)
    tests = [
        test('valgrind ran and printed an error summary', errors is not None,
             '' if errors is not None else 'no ERROR SUMMARY line in the log'),
        test('no memory errors', errors == 0,
             '' if errors == 0 else
             ('valgrind never reported a summary' if errors is None
              else f'{errors} error(s); see the raw log')),
        test('no definitely-lost heap blocks', leaked is None,
             '' if leaked is None else f'{leaked.group(1)} bytes definitely lost'),
        test('test program exit status is 0', status == 0,
             '' if status == 0 else f'exit status {status} (99 is valgrind\'s own error exit)'),
    ]
    return suite('C smoke test under valgrind', tests)


def cmd_parse(args):
    suites = []
    suites.extend(parse_file(args.cargo_raw, 'ffi unit tests') if args.cargo_raw else [])
    for entry in args.c_log:
        label, _, path = entry.rpartition('=')
        suites.append(parse_c_log(label, path))
    if args.valgrind:
        suites.append(parse_valgrind(args.valgrind, args.valgrind_status))

    total = sum(s['passed'] + s['failed'] + s['ignored'] for s in suites)
    passed = sum(s['passed'] for s in suites)
    failed = sum(s['failed'] for s in suites)
    ignored = sum(s['ignored'] for s in suites)
    result = {
        'build': args.build,
        'branch': args.branch,
        'commit': args.commit[:8],
        'date': args.date,
        'rust_version': args.rust_version,
        'crate': 'mid-ecs-ffi',
        'summary': {
            'total': total,
            'passed': passed,
            'failed': failed,
            'ignored': ignored,
            'duration_s': round(sum(s['duration_s'] for s in suites), 3),
        },
        'suites': suites,
    }
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"mid-ecs-ffi: suites={len(suites)} passed={passed} failed={failed} ignored={ignored}")
    return 0


def cmd_summary(args):
    try:
        result = json.loads(Path(args.json).read_text())
    except FileNotFoundError:
        print("## mid-ecs FFI Test Results\n")
        print("`mid-ecs-ffi-test-results.json` wasn't produced. The parse step above "
              "likely failed before writing it. Check its log.")
        return 0
    s = result['summary']
    out = [f"## mid-ecs FFI Test Results — Build #{result['build']}", '',
           f"Branch: `{result['branch']}` &nbsp;|&nbsp; Commit: `{result['commit']}` "
           f"&nbsp;|&nbsp; Rust: `{result['rust_version']}`", '']
    if s['total'] == 0:
        out.append("### ⚠️ No checks were collected. Check the raw logs")
    elif s['failed'] == 0:
        out.append(f"### ✅ All {s['total']} checks passed")
    else:
        out.append(f"### ❌ {s['failed']} of {s['total']} checks failed")
    out.append('')
    failing = [(x['name'], t) for x in result['suites'] for t in x['tests']
               if t['status'] == 'failed']
    if failing:
        out += ['### Failures', '']
        for sname, t in failing:
            out.append(f"- **{sname} :: {t['name']}**")
            detail = (t.get('output') or '').strip().splitlines()
            if detail:
                out.append(f"  - `{detail[0]}`")
        out.append('')
    out += ['| Suite | Passed | Failed |', '|---|---|---|']
    for x in result['suites']:
        mark = '✅' if x['failed'] == 0 else '❌'
        out.append(f"| {mark} {x['name']} | {x['passed']} | {x['failed']} |")
    out.append('')
    for x in result['suites']:
        out.append(f"<details><summary>{x['name']} — {len(x['tests'])} checks</summary>")
        out += ['', '| Check | Status |', '|---|---|']
        for t in x['tests']:
            out.append(f"| {t['name']} | {'✅' if t['status'] == 'passed' else '❌'} |")
        out += ['', '</details>', '']
    print('\n'.join(out))
    return 0


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)

    pp = sub.add_parser('parse')
    pp.add_argument('--out', required=True)
    pp.add_argument('--build', default='0')
    pp.add_argument('--branch', default='unknown')
    pp.add_argument('--commit', default='unknown')
    pp.add_argument('--date', default='')
    pp.add_argument('--rust-version', default='stable')
    pp.add_argument('--cargo-raw')
    pp.add_argument('--c-log', action='append', default=[],
                    help='"suite label=path" for one raw C smoke test log')
    pp.add_argument('--valgrind')
    pp.add_argument('--valgrind-status', type=int, default=0)
    pp.set_defaults(fn=cmd_parse)

    ps = sub.add_parser('summary')
    ps.add_argument('json')
    ps.set_defaults(fn=cmd_summary)

    args = p.parse_args()
    return args.fn(args)


if __name__ == '__main__':
    sys.exit(main())
