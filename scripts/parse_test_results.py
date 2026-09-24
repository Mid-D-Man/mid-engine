#!/usr/bin/env python3
# scripts/parse_test_results.py
#
# NOTICE: Full documentation, design decisions, and fix history for this
# file live in docs/mid-engine-site.md, section "scripts/parse_test_results.py"
#
# Parses one or more raw `cargo test ... --show-output` logs into a single
# JSON results file, in the shape web/tests/index.html (and any per-crate
# test workflow's own job-summary step) expects:
#
#   { build, branch, commit, rust_version, crate,
#     summary: { total, passed, failed, ignored, duration_s },
#     suites: [ { name, passed, failed, ignored, duration_s, tests: [...] } ] }
#
# Originally written inline, duplicated, inside mid-ptr-test.yml and
# mid-platform-test.yml. Pulled out here so deploy-site.yml (and any future
# per-crate test workflow) can share one implementation instead of a third
# copy-pasted version -- see docs/RUST_AND_CRATE_GUIDELINES.md §7 for why
# copy-pasted-until-corrected CI patterns are exactly the kind of drift this
# project tries to name and stop repeating.
#
# Usage:
#   parse_test_results.py --crate mid-ptr --build 12 --branch main \
#       --commit abc1234 --rust-version "rustc 1.90.0" \
#       --out mid-ptr-test-results.json \
#       raw1.txt[:label1] [raw2.txt:label2 ...]
#
# A raw log path may have an optional ":label" suffix (e.g.
# "nostd-raw.txt:no_std fallback") -- when present, every suite parsed from
# that file gets " [label]" appended to its name, the same way
# mid-platform-test.yml's own inline version distinguished its two feature
# configurations before this script existed.

import argparse
import json
import re
import sys

RE_ANSI = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
RE_RUN = re.compile(r'Running\s+.*?\(\S*?/deps/([A-Za-z0-9_]+)-[0-9a-f]{8,}\)\s*$')
RE_DOC = re.compile(r'^\s*Doc-tests\s+(\S+)\s*$')
RE_TEST = re.compile(
    r'^test\s+([\w:./-]+(?:::\w+)*(?:\s-\s\S+\s\(line\s\d+\))?)\s+\.\.\.\s+'
    r'(ok|FAILED|ignored)(?:\s+\(([\d.]+)s\))?'
)
RE_RESULT = re.compile(r'test result:.*?finished in ([\d.]+)s')


def strip_ansi(s: str) -> str:
    return RE_ANSI.sub('', s)


def parse_file(path: str, label: str | None) -> list[dict]:
    try:
        raw = open(path, encoding='utf-8', errors='replace').read()
    except FileNotFoundError:
        print(f"warning: {path} not found, skipping", file=sys.stderr)
        return []
    text = strip_ansi(raw)

    test_output: dict[str, str] = {}
    for m in re.finditer(
        r'---- ([\w:]+(?:::\w+)*) stdout ----\n(.*?)'
        r'(?=\n---- |\n\n(?:successes|failures):\n|\ntest result:|\Z)',
        text, re.DOTALL,
    ):
        content = m.group(2).strip()
        if content:
            test_output[m.group(1)] = content

    fail_output: dict[str, str] = {}
    fail_section = re.search(r'\nfailures:\n(.*?)(?:\ntest result:|\Z)', text, re.DOTALL)
    if fail_section:
        for m in re.finditer(
            r'---- ([\w:]+(?:::\w+)*) stdout ----\n(.*?)(?=\n---- |\Z)',
            fail_section.group(1), re.DOTALL,
        ):
            content = m.group(2).strip()
            if content:
                fail_output[m.group(1)] = content

    suites: list[dict] = []
    cur_name: str | None = None
    cur_tests: list[dict] = []
    cur_dur = 0.0

    def flush():
        nonlocal cur_name, cur_tests, cur_dur
        if cur_name is not None:
            name = f"{cur_name} [{label}]" if label else cur_name
            suites.append({
                'name': name,
                'passed': sum(1 for t in cur_tests if t['status'] == 'passed'),
                'failed': sum(1 for t in cur_tests if t['status'] == 'failed'),
                'ignored': sum(1 for t in cur_tests if t['status'] == 'ignored'),
                'duration_s': round(cur_dur, 4),
                'tests': cur_tests,
            })
        cur_name, cur_tests, cur_dur = None, [], 0.0

    for line in text.splitlines():
        m = RE_RUN.search(line)
        if m:
            flush()
            cur_name = m.group(1).replace('_', '-')
            continue
        m = RE_DOC.match(line)
        if m:
            flush()
            cur_name = f"{m.group(1)} (doc-tests)"
            continue
        m = RE_TEST.match(line)
        if m:
            if cur_name is None:
                cur_name = 'unknown suite'
            tname, status, dur = m.group(1), m.group(2), m.group(3)
            out = fail_output.get(tname) or test_output.get(tname, '')
            cur_tests.append({
                'name': tname,
                'status': {'ok': 'passed', 'FAILED': 'failed', 'ignored': 'ignored'}.get(status, 'unknown'),
                'duration_ms': int(float(dur) * 1000) if dur else 0,
                'output': out,
            })
            continue
        m = RE_RESULT.search(line)
        if m:
            cur_dur = float(m.group(1))
            flush()

    flush()
    return suites


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--crate', required=True)
    p.add_argument('--build', default='0')
    p.add_argument('--branch', default='unknown')
    p.add_argument('--commit', default='unknown')
    p.add_argument('--rust-version', default='stable')
    p.add_argument('--out', required=True)
    p.add_argument('raw_logs', nargs='+', help='path[:label] for each raw log to parse')
    args = p.parse_args()

    # Simple "path:label" split -- CI runners here are always Linux, so
    # there's no drive-letter-colon case to worry about, and a raw log path
    # from `cargo test | tee ...` never contains a literal colon of its own.
    suites: list[dict] = []
    for entry in args.raw_logs:
        path, _, label = entry.partition(':')
        suites.extend(parse_file(path, label or None))

    total = sum(s['passed'] + s['failed'] + s['ignored'] for s in suites)
    passed = sum(s['passed'] for s in suites)
    failed = sum(s['failed'] for s in suites)
    ignored = sum(s['ignored'] for s in suites)
    duration = sum(s['duration_s'] for s in suites)

    result = {
        'build': args.build,
        'branch': args.branch,
        'commit': args.commit[:8],
        'rust_version': args.rust_version,
        'crate': args.crate,
        'summary': {
            'total': total,
            'passed': passed,
            'failed': failed,
            'ignored': ignored,
            'duration_s': round(duration, 3),
        },
        'suites': suites,
    }

    with open(args.out, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"{args.crate}: suites={len(suites)} passed={passed} failed={failed} ignored={ignored}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
