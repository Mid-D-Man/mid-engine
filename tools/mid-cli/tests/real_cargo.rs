//! `mid check` against real cargo, in a small workspace with no lockfile:
//! the state of a fresh clone of the engine repository.

mod common;
use common::*;

use mid_reflect::Value;

/// `a` depends on `b`; `b` has a warning, `a` has none.
fn workspace() -> TempDir {
    repo_with(&[
        ("Cargo.toml", "[workspace]\nmembers = [\"a\", \"b\"]\nresolver = \"2\"\n"),
        (
            "a/Cargo.toml",
            "[package]\nname = \"a\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n[dependencies]\nb = { path = \"../b\" }\n",
        ),
        ("a/src/lib.rs", "pub use b::two;\n"),
        (
            "b/Cargo.toml",
            "[package]\nname = \"b\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
        ),
        (
            "b/src/lib.rs",
            "pub fn two() -> u32 {\n    let unused = 1;\n    2\n}\n",
        ),
    ])
}

fn step_number(v: &Value, key: &str) -> i64 {
    v.get("steps")
        .and_then(Value::as_list)
        .and_then(|s| s.first())
        .and_then(|s| s.get(key))
        .and_then(Value::as_i64)
        .unwrap_or(-1)
}

#[test]
fn a_dependencys_warnings_are_not_counted_against_the_package_without_a_lockfile() {
    let r = workspace();
    assert!(!r.exists("Cargo.lock"), "the fixture has no lockfile");
    let t = TempDir::new("env");
    let e = env_in(t.path(), t.path());

    let (code, v) = json(
        &[
            "check",
            "-p",
            "a",
            "--no-fmt",
            "--no-clippy",
            "--repo",
            r.path().to_str().unwrap(),
        ],
        &Scripted::open(),
        &e,
    );
    assert_eq!(code, 0, "{}", v.to_json_pretty());
    assert_eq!(step_number(&v, "warnings"), 0, "{}", v.to_json_pretty());
    assert!(
        step_number(&v, "other_warnings") >= 1,
        "the dependency's warning is reported as another package's: {}",
        v.to_json_pretty()
    );

    let (code, v) = json(
        &[
            "check",
            "-p",
            "b",
            "--no-fmt",
            "--no-clippy",
            "--repo",
            r.path().to_str().unwrap(),
        ],
        &Scripted::open(),
        &e,
    );
    assert_eq!(code, 0, "{}", v.to_json_pretty());
    assert!(step_number(&v, "warnings") >= 1, "{}", v.to_json_pretty());
}

fn target_labels(v: &Value) -> Vec<String> {
    v.get("targets")
        .and_then(Value::as_list)
        .unwrap_or(&[])
        .iter()
        .map(|t| {
            t.get("label")
                .and_then(Value::as_str)
                .unwrap_or("?")
                .to_string()
        })
        .collect()
}

#[test]
fn test_targets_carry_their_labels_in_a_real_run() {
    let r = workspace();
    let t = TempDir::new("env");
    let e = env_in(t.path(), t.path());
    let (code, v) = json(
        &["test", "-p", "a", "--repo", r.path().to_str().unwrap()],
        &Scripted::open(),
        &e,
    );
    assert_eq!(code, 0, "{}", v.to_json_pretty());
    assert_eq!(
        target_labels(&v),
        ["unittests src/lib.rs", "doc-tests a"],
        "{}",
        v.to_json_pretty()
    );
}
