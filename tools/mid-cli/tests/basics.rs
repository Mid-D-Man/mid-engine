//! Global behaviour: help, version, usage errors, flags, JSON shape.

mod common;
use common::*;

use mid_cli::run;
use mid_reflect::Value;

fn env() -> (TempDir, mid_cli::Env) {
    let t = TempDir::new("basics");
    let e = env_in(t.path(), t.path());
    (t, e)
}

#[test]
fn help_lists_every_command() {
    let (_t, e) = env();
    let (code, v) = json(&["help"], &Scripted::closed(), &e);
    assert_eq!(code, 0);
    assert_eq!(v.get("schema"), Some(&Value::Int(1)));
    assert_eq!(v.get("command").and_then(Value::as_str), Some("help"));
    let names: Vec<&str> = v
        .get("commands")
        .and_then(Value::as_list)
        .unwrap()
        .iter()
        .map(|c| c.get("name").and_then(Value::as_str).unwrap())
        .collect();
    assert_eq!(
        names,
        [
            "doctor",
            "check",
            "test",
            "docs lint",
            "apply",
            "deliver",
            "help"
        ]
    );
    for c in v.get("commands").and_then(Value::as_list).unwrap() {
        assert!(!c.get("summary").and_then(Value::as_str).unwrap().is_empty());
    }
}

#[test]
fn no_arguments_and_dash_dash_help_print_help() {
    let (_t, e) = env();
    for a in [
        &[][..],
        &["--help"][..],
        &["-h"][..],
        &["doctor", "--help"][..],
    ] {
        let r = run(&args(a), &Scripted::closed(), &e);
        assert_eq!(r.code, 0, "{a:?}");
        assert!(r.stdout.contains("usage: mid"), "{a:?}: {}", r.stdout);
        assert!(r.stderr.is_empty());
    }
}

#[test]
fn version_is_printed() {
    let (_t, e) = env();
    let r = run(&args(&["--version"]), &Scripted::closed(), &e);
    assert_eq!(r.code, 0);
    assert_eq!(r.stdout, format!("mid {}\n", env!("CARGO_PKG_VERSION")));
    let (_, v) = json(&["--version"], &Scripted::closed(), &e);
    assert_eq!(
        v.get("version").and_then(Value::as_str),
        Some(env!("CARGO_PKG_VERSION"))
    );
}

#[test]
fn an_unknown_command_is_a_usage_error_with_the_message_on_stderr() {
    let (_t, e) = env();
    let r = run(&args(&["frobnicate"]), &Scripted::closed(), &e);
    assert_eq!(r.code, 2);
    assert!(r.stdout.is_empty());
    assert!(
        r.stderr.contains("unknown command `frobnicate`"),
        "{}",
        r.stderr
    );
}

#[test]
fn usage_errors_in_json_mode_stay_on_stdout_and_say_what_kind() {
    let (_t, e) = env();
    let (code, v) = json(&["frobnicate"], &Scripted::closed(), &e);
    assert_eq!(code, 2);
    assert_eq!(v.get("ok"), Some(&Value::Bool(false)));
    assert_eq!(v.get_path("error.kind").unwrap(), &Value::from("usage"));
}

#[test]
fn repo_needs_a_value_and_unknown_flags_are_refused() {
    let (_t, e) = env();
    assert_eq!(
        run(&args(&["doctor", "--repo"]), &Scripted::closed(), &e).code,
        2
    );
    let r = run(&args(&["doctor", "--nope"]), &Scripted::closed(), &e);
    assert_eq!(r.code, 2);
    assert!(r.stderr.contains("unknown flag `--nope`"));
    let r = run(&args(&["docs"]), &Scripted::closed(), &e);
    assert_eq!(r.code, 2);
    assert!(r.stderr.contains("`lint`"));
    let r = run(&args(&["docs", "format"]), &Scripted::closed(), &e);
    assert_eq!(r.code, 2);
}

#[test]
fn global_flags_work_before_and_after_the_command() {
    let (_t, e) = env();
    for a in [
        &["--json", "help"][..],
        &["help", "--json"][..],
        &["--repo", "x", "help", "--json"][..],
        &["--repo=x", "--json", "help"][..],
    ] {
        let r = run(&args(a), &Scripted::closed(), &e);
        assert_eq!(r.code, 0, "{a:?}");
        assert!(
            r.stdout.trim_start().starts_with('{'),
            "{a:?}: {}",
            r.stdout
        );
    }
}

#[test]
fn json_output_is_one_line_ending_in_a_newline() {
    let (_t, e) = env();
    let r = run(&args(&["help", "--json"]), &Scripted::closed(), &e);
    assert_eq!(r.stdout.matches('\n').count(), 1);
    assert!(r.stdout.ends_with('\n'));
}

#[test]
fn a_passthrough_argument_is_not_taken_for_a_global_flag() {
    let (t, e) = env();
    // `--json` after `--` belongs to the test binary, not to mid.
    let runner = Scripted::open().on(|c| {
        if c.program == "git" {
            return Some(ok(&format!("{}\n", std::env::temp_dir().display())));
        }
        if c.program == "cargo" {
            assert!(c.args.contains(&"--json".to_string()), "{:?}", c.args);
            return Some(ok(
                "test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out\n",
            ));
        }
        None
    });
    let r = run(&args(&["test", "-p", "x", "--", "--json"]), &runner, &e);
    assert_eq!(r.code, 0, "{}{}", r.stdout, r.stderr);
    assert!(
        !r.stdout.trim_start().starts_with('{'),
        "stayed in text mode"
    );
    drop(t);
}
