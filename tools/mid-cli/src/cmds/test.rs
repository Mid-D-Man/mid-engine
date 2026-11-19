// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-cli.md, section "cmds/test.rs"
// ============================================================================
//! `mid test`: run the tests for some crates and summarise them, with the
//! change since the last run of the same thing.

use std::path::{Path, PathBuf};
use std::time::Instant;

use mid_reflect::Value;

use crate::args::{self, Flag};
use crate::cargo_out::parse_test_output;
use crate::proc::{Cmd, Runner};
use crate::report::{count, strings, Report, EXIT_FAILED, EXIT_OK};
use crate::util::{fnv_hex, path_text, write_log};
use crate::{cmds, Env};

const SPEC: &[(&str, Flag, Option<char>)] = &[
    ("package", Flag::Repeat, Some('p')),
    ("all", Flag::Switch, None),
    ("features", Flag::Value, Some('F')),
    ("lib", Flag::Switch, None),
    ("doc", Flag::Switch, None),
];

/// Runs `mid test`.
pub fn run(args: &[String], runner: &dyn Runner, env: &Env, dir: &Path) -> Report {
    let parsed = match args::parse(args, SPEC) {
        Ok(p) => p,
        Err(e) => return cmds::usage("test", &e),
    };
    let packages: Vec<String> = parsed.values("package").to_vec();
    if packages.is_empty() && !parsed.has("all") {
        return cmds::usage("test", "give `-p <crate>` (repeatable) or `--all`");
    }
    let root = match cmds::repo_root("test", runner, dir) {
        Ok(r) => r,
        Err(report) => return report,
    };

    let mut cargo_args: Vec<String> = vec!["test".to_string()];
    if parsed.has("all") {
        cargo_args.push("--workspace".to_string());
    }
    for p in &packages {
        cargo_args.push("-p".to_string());
        cargo_args.push(p.clone());
    }
    if let Some(f) = parsed.value("features") {
        cargo_args.push("--features".to_string());
        cargo_args.push(f.to_string());
    }
    for flag in ["lib", "doc"] {
        if parsed.has(flag) {
            cargo_args.push(format!("--{flag}"));
        }
    }
    cargo_args.extend(["--color=never".to_string(), "--no-fail-fast".to_string()]);
    if !parsed.passthrough.is_empty() {
        cargo_args.push("--".to_string());
        cargo_args.extend(parsed.passthrough.iter().cloned());
    }

    let cmd = Cmd::new("cargo", &cargo_args).in_dir(dir);
    let started = Instant::now();
    let out = runner.run(&cmd);
    let seconds = started.elapsed().as_secs_f64();
    let text = out.combined();
    let log = write_log(&env.temp, "test", &format!("$ {}\n{text}", cmd.display()));
    if !out.started() {
        return Report::error(
            "test",
            crate::report::EXIT_ENV,
            "no-cargo",
            &format!(
                "could not run cargo: {}",
                out.spawn_error.unwrap_or_default()
            ),
        );
    }

    let parse = parse_test_output(&text);
    let (passed, failed, ignored) = (parse.passed(), parse.failed(), parse.ignored());
    let ran = !parse.targets.is_empty();
    let ok = out.success() && failed == 0 && parse.build_errors.is_empty() && ran;

    let scope = if packages.is_empty() {
        "workspace".to_string()
    } else {
        packages.join(", ")
    };
    let key = format!(
        "{}|{}|{}|{}",
        packages.join(","),
        parsed.value("features").unwrap_or(""),
        parsed.has("lib"),
        parsed.has("doc")
    );
    let state = state_path(env, &root, &key);
    let previous = read_state(&state);
    if ran {
        write_state(&state, passed, failed, ignored);
    }
    let delta = previous.map(|(p, _, _)| passed as i64 - p as i64);

    let targets: Vec<Value> = parse
        .targets
        .iter()
        .map(|t| {
            Value::map([
                ("label", Value::from(t.label.as_str())),
                ("passed", Value::Int(t.passed as i64)),
                ("failed", Value::Int(t.failed as i64)),
                ("ignored", Value::Int(t.ignored as i64)),
            ])
        })
        .collect();
    let build_errors: Vec<String> = parse.build_errors.iter().take(10).cloned().collect();

    let mut report = Report::new("test", ok, if ok { EXIT_OK } else { EXIT_FAILED })
        .field("packages", strings(&packages))
        .field("passed", count(passed as usize))
        .field("failed", count(failed as usize))
        .field("ignored", count(ignored as usize))
        .field("targets", Value::List(targets))
        .field("failures", strings(&parse.failures))
        .field("build_errors", strings(&build_errors))
        .field(
            "previous_passed",
            previous.map_or(Value::Null, |(p, _, _)| Value::Int(p as i64)),
        )
        .field("delta_passed", delta.map_or(Value::Null, Value::Int))
        .field("seconds", Value::Float((seconds * 10.0).round() / 10.0))
        .field(
            "log",
            log.as_ref()
                .map_or(Value::Null, |p| Value::from(path_text(p))),
        );

    let change = match delta {
        Some(0) => String::new(),
        Some(d) => format!(" ({d:+} passed since last run)"),
        None => String::new(),
    };
    if !ran && !parse.build_errors.is_empty() {
        report = report.line(format!("test {scope}: build failed ({seconds:.0}s)"));
        for e in &build_errors {
            report = report.line(format!("  {e}"));
        }
    } else if !ran {
        report = report.line(format!("test {scope}: no tests ran ({seconds:.0}s)"));
    } else {
        report = report.line(format!(
            "test {scope}: {passed} passed, {failed} failed, {ignored} ignored{change} ({seconds:.0}s)"
        ));
        for f in parse.failures.iter().take(20) {
            report = report.line(format!("  FAILED {f}"));
        }
    }
    if !ok {
        if let Some(p) = &log {
            report = report.line(format!("  full log: {}", p.display()));
        }
    }
    report
}

fn state_path(env: &Env, root: &Path, key: &str) -> PathBuf {
    env.temp
        .join("mid-cli")
        .join("state")
        .join(fnv_hex(&path_text(root)))
        .join(format!("test-{}.txt", fnv_hex(key)))
}

fn read_state(path: &Path) -> Option<(u64, u64, u64)> {
    let text = std::fs::read_to_string(path).ok()?;
    let mut nums = text.split_whitespace().map(|n| n.parse::<u64>().ok());
    Some((nums.next()??, nums.next()??, nums.next()??))
}

fn write_state(path: &Path, passed: u64, failed: u64, ignored: u64) {
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let _ = std::fs::write(path, format!("{passed} {failed} {ignored}\n"));
}
