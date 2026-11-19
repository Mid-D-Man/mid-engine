//! `doctor`, `test` and `check`, with a scripted stand-in for cargo and git.

mod common;
use common::*;

use mid_cli::{Cmd, Env, Outcome};
use mid_reflect::Value;

fn answer(c: &Cmd, root: &str) -> Option<Outcome> {
    let line = c.args.join(" ");
    match c.program.as_str() {
        "rustc" if line == "--version" => Some(ok("rustc 1.97.1 (abc 2026-07-14)\n")),
        "rustc" => Some(ok("/nonexistent-sysroot\n")),
        "cargo" if line == "--version" => Some(ok("cargo 1.97.1\n")),
        "cargo" if line == "clippy --version" => Some(ok("clippy 0.1.97\n")),
        "rustfmt" | "node" | "tar" => Some(ok("tool 1.0\n")),
        "git" if line == "--version" => Some(ok("git version 2.43.0\n")),
        "git" if line.contains("--show-toplevel") => Some(ok(&format!("{root}\n"))),
        "git" if line.contains("--short") => Some(ok("abc1234\n")),
        "git" if line.contains("--abbrev-ref") => Some(ok("main\n")),
        "git" if line.contains("status") => Some(ok(" M a.rs\n?? b.rs\n")),
        "git" if line.contains("--verify") => Some(ok("deadbeef\n")),
        _ => None,
    }
}

fn setup() -> (TempDir, Env) {
    let t = TempDir::new("cmds");
    let e = env_in(t.path(), t.path());
    (t, e)
}

fn field<'a>(v: &'a Value, path: &str) -> &'a Value {
    v.get_path(path)
        .unwrap_or_else(|e| panic!("{path}: {e}\n{}", v.to_json_pretty()))
}

// ── doctor ──────────────────────────────────────────────────────────────

#[test]
fn doctor_reports_tools_and_the_repository() {
    let (_t, e) = setup();
    let runner = Scripted::closed().on(|c| answer(c, "/repo"));
    let (code, v) = json(&["doctor"], &runner, &e);
    assert_eq!(code, 0);
    assert_eq!(v.get("ok"), Some(&Value::Bool(true)));
    let tools = v.get("tools").and_then(Value::as_list).unwrap();
    let rustc = tools
        .iter()
        .find(|t| t.get("name").and_then(Value::as_str) == Some("rustc"))
        .unwrap();
    assert_eq!(rustc.get("found"), Some(&Value::Bool(true)));
    assert_eq!(rustc.get("required"), Some(&Value::Bool(true)));
    assert_eq!(
        rustc.get("version").and_then(Value::as_str),
        Some("rustc 1.97.1 (abc 2026-07-14)")
    );
    assert_eq!(field(&v, "repo.root"), &Value::from("/repo"));
    assert_eq!(field(&v, "repo.head"), &Value::from("abc1234"));
    assert_eq!(field(&v, "repo.branch"), &Value::from("main"));
    assert_eq!(field(&v, "repo.dirty_paths"), &Value::Int(2));
    assert_eq!(field(&v, "repo.has_origin_main"), &Value::Bool(true));
}

#[test]
fn a_missing_required_tool_is_exit_3_and_is_named() {
    let (_t, e) = setup();
    let runner = Scripted::closed().on(|c| {
        if c.program == "rustc" {
            None
        } else {
            answer(c, "/repo")
        }
    });
    let (code, v) = json(&["doctor"], &runner, &e);
    assert_eq!(code, 3);
    assert_eq!(v.get("ok"), Some(&Value::Bool(false)));
    assert_eq!(strings(&v, "missing"), ["rustc"]);
    let text = mid_cli::run(&args(&["doctor"]), &runner, &e);
    assert_eq!(text.code, 3);
    assert!(
        text.stderr.contains("missing required tools: rustc"),
        "{}",
        text.stderr
    );
}

#[test]
fn a_missing_optional_tool_is_still_exit_0() {
    let (_t, e) = setup();
    let runner = Scripted::closed().on(|c| {
        if c.program == "node" {
            None
        } else {
            answer(c, "/repo")
        }
    });
    let (code, v) = json(&["doctor"], &runner, &e);
    assert_eq!(code, 0);
    let tools = v.get("tools").and_then(Value::as_list).unwrap();
    let node = tools
        .iter()
        .find(|t| t.get("name").and_then(Value::as_str) == Some("node"))
        .unwrap();
    assert_eq!(node.get("found"), Some(&Value::Bool(false)));
    assert_eq!(node.get("version"), Some(&Value::Null));
}

#[test]
fn outside_a_repository_the_repo_is_null_and_doctor_still_passes() {
    let (_t, e) = setup();
    let runner = Scripted::closed().on(|c| {
        if c.program == "git" && c.args.join(" ").contains("--show-toplevel") {
            return Some(failed(128, "", "fatal: not a git repository"));
        }
        answer(c, "/repo")
    });
    let (code, v) = json(&["doctor"], &runner, &e);
    assert_eq!(code, 0);
    assert_eq!(v.get("repo"), Some(&Value::Null));
}

#[test]
fn the_wasm_target_and_the_mdix_tool_are_found_on_disk() {
    let (t, mut e) = setup();
    t.write("sysroot/lib/rustlib/wasm32-unknown-unknown/lib/x", "");
    t.write("mdix/bin/mdix-scaffold.js", "");
    let sysroot = t.path().join("sysroot").to_string_lossy().into_owned();
    e.mdix = Some(t.path().join("mdix").to_string_lossy().into_owned());
    let runner = Scripted::closed().on(move |c| {
        if c.program == "rustc" && c.args.first().map(String::as_str) == Some("--print") {
            return Some(ok(&format!("{sysroot}\n")));
        }
        answer(c, "/repo")
    });
    let (_, v) = json(&["doctor"], &runner, &e);
    let tools = v.get("tools").and_then(Value::as_list).unwrap();
    let found = |name: &str| {
        tools
            .iter()
            .find(|t| t.get("name").and_then(Value::as_str) == Some(name))
            .and_then(|t| t.get("found"))
            .cloned()
    };
    assert_eq!(found("wasm32-unknown-unknown"), Some(Value::Bool(true)));
    assert_eq!(found("mdix"), Some(Value::Bool(true)));
    e.mdix = Some(t.path().join("nowhere").to_string_lossy().into_owned());
    let (_, v) = json(&["doctor"], &runner, &e);
    let tools = v.get("tools").and_then(Value::as_list).unwrap();
    let mdix = tools
        .iter()
        .find(|t| t.get("name").and_then(Value::as_str) == Some("mdix"))
        .unwrap();
    assert_eq!(mdix.get("found"), Some(&Value::Bool(false)));
}

// ── test ────────────────────────────────────────────────────────────────

fn cargo_test_output(lib: u64, doc: u64, failing: bool) -> String {
    let mut s = String::from("   Compiling x v0.1.0\n     Running unittests src/lib.rs (target/debug/deps/x-1)\n\nrunning tests\n");
    if failing {
        s.push_str("test b ... FAILED\ntest result: FAILED. ");
        s.push_str(&format!(
            "{} passed; 1 failed; 0 ignored; 0 measured; 0 filtered out\n",
            lib
        ));
    } else {
        s.push_str(&format!(
            "test result: ok. {lib} passed; 0 failed; 1 ignored; 0 measured; 0 filtered out\n"
        ));
    }
    s.push_str(&format!(
        "\n   Doc-tests x\ntest result: ok. {doc} passed; 0 failed; 0 ignored; 0 measured; 0 filtered out\n"
    ));
    s
}

fn cargo_returning(out: Outcome) -> Scripted {
    Scripted::closed().on(move |c| {
        if c.program == "cargo" && c.args.first().map(String::as_str) == Some("test") {
            return Some(out.clone());
        }
        answer(c, "/repo")
    })
}

#[test]
fn test_summarises_counts_per_target() {
    let (_t, e) = setup();
    let runner = cargo_returning(ok(&cargo_test_output(10, 2, false)));
    let (code, v) = json(&["test", "-p", "x"], &runner, &e);
    assert_eq!(code, 0);
    assert_eq!(v.get("ok"), Some(&Value::Bool(true)));
    assert_eq!(v.get("passed"), Some(&Value::Int(12)));
    assert_eq!(v.get("failed"), Some(&Value::Int(0)));
    assert_eq!(v.get("ignored"), Some(&Value::Int(1)));
    let targets = v.get("targets").and_then(Value::as_list).unwrap();
    assert_eq!(targets.len(), 2);
    assert_eq!(
        targets[0].get("label").and_then(Value::as_str),
        Some("unittests src/lib.rs")
    );
    assert_eq!(
        targets[1].get("label").and_then(Value::as_str),
        Some("doc-tests x")
    );
    assert_eq!(strings(&v, "packages"), ["x"]);
    assert_eq!(v.get("previous_passed"), Some(&Value::Null));
    assert!(
        runner.ran("cargo test -p x --color=never --no-fail-fast"),
        "{:?}",
        runner.calls.borrow()
    );
    let text = mid_cli::run(&args(&["test", "-p", "x"]), &runner, &e);
    assert!(
        text.stdout
            .starts_with("test x: 12 passed, 0 failed, 1 ignored"),
        "{}",
        text.stdout
    );
}

#[test]
fn test_reports_the_change_since_the_last_run_of_the_same_thing() {
    let (_t, e) = setup();
    let first = cargo_returning(ok(&cargo_test_output(10, 2, false)));
    json(&["test", "-p", "x"], &first, &e);
    let second = cargo_returning(ok(&cargo_test_output(15, 2, false)));
    let (_, v) = json(&["test", "-p", "x"], &second, &e);
    assert_eq!(v.get("previous_passed"), Some(&Value::Int(12)));
    assert_eq!(v.get("delta_passed"), Some(&Value::Int(5)));
    let text = mid_cli::run(&args(&["test", "-p", "x"]), &second, &e);
    assert!(
        !text.stdout.contains("(+0 passed") && !text.stdout.contains("since last run"),
        "no change, no note: {}",
        text.stdout
    );
    // A different feature set is a different thing: no previous run.
    let (_, v) = json(&["test", "-p", "x", "--features", "f"], &second, &e);
    assert_eq!(v.get("previous_passed"), Some(&Value::Null));
    // And so is a different crate.
    let (_, v) = json(&["test", "-p", "y"], &second, &e);
    assert_eq!(v.get("previous_passed"), Some(&Value::Null));
    // The note appears when it changed.
    let third = cargo_returning(ok(&cargo_test_output(20, 2, false)));
    let text = mid_cli::run(&args(&["test", "-p", "x"]), &third, &e);
    assert!(
        text.stdout.contains("(+5 passed since last run)"),
        "{}",
        text.stdout
    );
}

#[test]
fn a_failing_test_is_named_and_exit_1() {
    let (_t, e) = setup();
    let runner = cargo_returning(failed(101, &cargo_test_output(1, 0, true), ""));
    let (code, v) = json(&["test", "-p", "x"], &runner, &e);
    assert_eq!(code, 1);
    assert_eq!(v.get("ok"), Some(&Value::Bool(false)));
    assert_eq!(v.get("failed"), Some(&Value::Int(1)));
    assert_eq!(strings(&v, "failures"), ["b"]);
    let text = mid_cli::run(&args(&["test", "-p", "x"]), &runner, &e);
    assert!(text.stdout.contains("FAILED b"), "{}", text.stdout);
    assert!(text.stdout.contains("full log:"));
}

#[test]
fn a_build_error_is_reported_and_is_not_a_pass() {
    let (_t, e) = setup();
    let out = failed(
        101,
        "",
        "error[E0425]: cannot find value `x` in this scope\n --> src/lib.rs:3:5\nerror: could not compile `x` (lib) due to 1 previous error\n",
    );
    let (code, v) = json(&["test", "-p", "x"], &cargo_returning(out.clone()), &e);
    assert_eq!(code, 1);
    assert_eq!(
        strings(&v, "build_errors"),
        ["error[E0425]: cannot find value `x` in this scope"]
    );
    assert_eq!(v.get("passed"), Some(&Value::Int(0)));
    let text = mid_cli::run(&args(&["test", "-p", "x"]), &cargo_returning(out), &e);
    assert!(text.stdout.contains("build failed"), "{}", text.stdout);
}

#[test]
fn running_no_tests_at_all_is_not_a_pass() {
    let (_t, e) = setup();
    let (code, v) = json(&["test", "-p", "x"], &cargo_returning(ok("Finished\n")), &e);
    assert_eq!(code, 1);
    assert_eq!(v.get("ok"), Some(&Value::Bool(false)));
    // But a target with zero tests that ran is fine.
    let zero = ok("     Running unittests src/lib.rs (x)\ntest result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out\n");
    let (code, _) = json(&["test", "-p", "x"], &cargo_returning(zero), &e);
    assert_eq!(code, 0);
}

#[test]
fn test_needs_a_package_or_all_and_a_cargo() {
    let (_t, e) = setup();
    let runner = cargo_returning(ok(""));
    assert_eq!(mid_cli::run(&args(&["test"]), &runner, &e).code, 2);
    let none = Scripted::closed().on(|c| {
        if c.program == "cargo" {
            None
        } else {
            answer(c, "/repo")
        }
    });
    let (code, v) = json(&["test", "-p", "x"], &none, &e);
    assert_eq!(code, 3);
    assert_eq!(v.get_path("error.kind").unwrap(), &Value::from("no-cargo"));
}

#[test]
fn test_passes_its_flags_to_cargo() {
    let (_t, e) = setup();
    let runner = cargo_returning(ok(&cargo_test_output(1, 1, false)));
    mid_cli::run(
        &args(&[
            "--json",
            "test",
            "-p",
            "a",
            "-p",
            "b",
            "--features",
            "f,g",
            "--lib",
            "--doc",
            "--",
            "name",
            "--nocapture",
        ]),
        &runner,
        &e,
    );
    assert!(
        runner.ran("cargo test -p a -p b --features f,g --lib --doc --color=never --no-fail-fast -- name --nocapture"),
        "{:?}",
        runner.calls.borrow()
    );
    let runner = cargo_returning(ok(&cargo_test_output(1, 1, false)));
    json(&["test", "--all"], &runner, &e);
    assert!(
        runner.ran("cargo test --workspace --color=never"),
        "{:?}",
        runner.calls.borrow()
    );
}

// ── check ───────────────────────────────────────────────────────────────

const METADATA: &str = r#"{"packages":[{"name":"x","manifest_path":"/repo/crates/x/Cargo.toml"}]}"#;

fn check_runner(check_out: &'static str, fmt: Outcome, clippy_out: &'static str) -> Scripted {
    Scripted::closed().on(move |c| {
        let line = c.args.join(" ");
        if c.program == "cargo" {
            if line.starts_with("metadata") {
                return Some(ok(METADATA));
            }
            if line.starts_with("check") {
                return Some(ok(check_out));
            }
            if line.starts_with("fmt") {
                return Some(fmt.clone());
            }
            if line.starts_with("clippy") {
                return Some(ok(clippy_out));
            }
        }
        answer(c, "/repo")
    })
}

#[test]
fn check_runs_three_steps_and_counts_only_the_packages_own_diagnostics() {
    let (_t, e) = setup();
    let runner = check_runner(
        "crates/x/src/lib.rs:1:1: warning: unused import\ncrates/other/src/a.rs:2:2: warning: elsewhere\n",
        ok(""),
        "crates/x/src/a.rs:5:1: warning: needless\ncrates/x/src/b.rs:6:1: warning: also\nwarning: `x` (lib) generated 2 warnings\n",
    );
    let (code, v) = json(&["check", "-p", "x"], &runner, &e);
    assert_eq!(code, 0, "warnings do not fail by default");
    let steps = v.get("steps").and_then(Value::as_list).unwrap();
    let names: Vec<&str> = steps
        .iter()
        .map(|s| s.get("step").and_then(Value::as_str).unwrap())
        .collect();
    assert_eq!(names, ["check", "fmt", "clippy"]);
    assert_eq!(steps[0].get("warnings"), Some(&Value::Int(1)));
    assert_eq!(steps[0].get("other_warnings"), Some(&Value::Int(1)));
    assert_eq!(steps[2].get("warnings"), Some(&Value::Int(2)));
    assert_eq!(
        steps[0].get_path("findings[0].path").unwrap(),
        &Value::from("crates/x/src/lib.rs")
    );
    assert_eq!(
        steps[0].get_path("findings[0].line").unwrap(),
        &Value::Int(1)
    );
    assert!(
        runner.ran("cargo check -p x --all-targets --message-format=short --color=never"),
        "{:?}",
        runner.calls.borrow()
    );
    assert!(runner.ran("cargo clippy -p x --all-targets --no-deps --message-format=short"));
}

#[test]
fn check_fails_on_an_error_and_on_a_format_difference() {
    let (_t, e) = setup();
    let runner = check_runner(
        "",
        failed(1, "Diff in /repo/crates/x/src/lib.rs:3:\n", ""),
        "",
    );
    let (code, v) = json(&["check", "-p", "x"], &runner, &e);
    assert_eq!(code, 1);
    let steps = v.get("steps").and_then(Value::as_list).unwrap();
    assert_eq!(steps[1].get("ok"), Some(&Value::Bool(false)));
    assert_eq!(
        steps[1].get("warnings"),
        Some(&Value::Int(1)),
        "one differing file"
    );

    let runner = Scripted::closed().on(|c| {
        let line = c.args.join(" ");
        if c.program == "cargo" && line.starts_with("check") {
            return Some(failed(101, "", "crates/x/src/lib.rs:3:5: error[E0425]: cannot find value\nerror: could not compile `x` (lib)\n"));
        }
        if c.program == "cargo" && line.starts_with("metadata") {
            return Some(ok(METADATA));
        }
        if c.program == "cargo" {
            return Some(ok(""));
        }
        answer(c, "/repo")
    });
    let (code, v) = json(&["check", "-p", "x"], &runner, &e);
    assert_eq!(code, 1);
    let steps = v.get("steps").and_then(Value::as_list).unwrap();
    assert_eq!(steps[0].get("ok"), Some(&Value::Bool(false)));
    assert_eq!(
        steps[0].get("errors"),
        Some(&Value::Int(2)),
        "the located one and the could-not-compile line"
    );
}

#[test]
fn check_can_skip_steps_deny_warnings_and_limit_findings() {
    let (_t, e) = setup();
    let warn = "crates/x/src/a.rs:1:1: warning: w1\ncrates/x/src/a.rs:2:1: warning: w2\ncrates/x/src/a.rs:3:1: warning: w3\n";
    let runner = check_runner(warn, ok(""), "");
    let (code, v) = json(
        &["check", "-p", "x", "--no-fmt", "--no-clippy"],
        &runner,
        &e,
    );
    assert_eq!(code, 0);
    assert_eq!(v.get("steps").and_then(Value::as_list).unwrap().len(), 1);
    assert!(!runner.ran("cargo fmt") && !runner.ran("cargo clippy"));
    let (code, _) = json(
        &[
            "check",
            "-p",
            "x",
            "--no-fmt",
            "--no-clippy",
            "--deny-warnings",
        ],
        &runner,
        &e,
    );
    assert_eq!(code, 1);
    let (_, v) = json(
        &[
            "check",
            "-p",
            "x",
            "--no-fmt",
            "--no-clippy",
            "--limit",
            "2",
        ],
        &runner,
        &e,
    );
    assert_eq!(
        v.get_path("steps[0].findings")
            .unwrap()
            .as_list()
            .unwrap()
            .len(),
        2
    );
    assert_eq!(
        v.get_path("steps[0].warnings").unwrap(),
        &Value::Int(3),
        "the count is not limited"
    );
}

#[test]
fn check_needs_a_package_or_all_and_all_checks_the_workspace() {
    let (_t, e) = setup();
    let runner = check_runner("", ok(""), "");
    assert_eq!(mid_cli::run(&args(&["check"]), &runner, &e).code, 2);
    assert_eq!(
        mid_cli::run(&args(&["check", "-p", "x", "--limit", "many"]), &runner, &e).code,
        2
    );
    let (code, v) = json(&["check", "--all"], &runner, &e);
    assert_eq!(code, 0);
    assert!(
        runner.ran("cargo check --workspace --all-targets"),
        "{:?}",
        runner.calls.borrow()
    );
    assert_eq!(
        v.get_path("steps[0].package").unwrap(),
        &Value::from("workspace")
    );
}

#[test]
fn check_passes_features_and_target_through() {
    let (_t, e) = setup();
    let runner = check_runner("", ok(""), "");
    json(
        &[
            "check",
            "-p",
            "x",
            "--features",
            "math",
            "--target",
            "wasm32-unknown-unknown",
        ],
        &runner,
        &e,
    );
    assert!(
        runner.ran("cargo check -p x --all-targets --features math --target wasm32-unknown-unknown --message-format=short"),
        "{:?}",
        runner.calls.borrow()
    );
}
