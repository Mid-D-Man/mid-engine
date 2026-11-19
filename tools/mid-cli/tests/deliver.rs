//! `mid deliver`, against real git repositories and real tar.

mod common;
use common::*;

use mid_reflect::Value;

fn base_repo() -> TempDir {
    repo_with(&[
        ("crates/a/src/lib.rs", "one\n"),
        ("crates/a/old.rs", "old\n"),
        ("docs/d.md", "d\n"),
        ("Cargo.toml", "[workspace]\n"),
        (".github/workflows/w.yml", "a\n"),
    ])
}

fn changed_repo() -> TempDir {
    let r = base_repo();
    r.write("crates/a/src/lib.rs", "two\n");
    r.write("crates/a/src/new.rs", "new\n");
    r.write("docs/new.md", "n\n");
    std::fs::remove_file(r.path().join("crates/a/old.rs")).unwrap();
    r.write(".github/workflows/w.yml", "b\n");
    r.write(".github/workflows/n.yml", "n\n");
    r.write("Cargo.toml", "[workspace]\nmembers = []\n");
    r.write("Cargo.lock", "# lock\n");
    r.write("tools/x/Cargo.lock", "# nested lock\n");
    r
}

fn members(tar: &str) -> Vec<String> {
    let out = std::process::Command::new("tar")
        .args(["tzf", tar])
        .output()
        .unwrap();
    let mut m: Vec<String> = String::from_utf8_lossy(&out.stdout)
        .lines()
        .map(str::to_string)
        .collect();
    m.sort();
    m
}

fn tar_file(tar: &str, member: &str) -> String {
    let out = std::process::Command::new("tar")
        .args(["xOzf", tar, member])
        .output()
        .unwrap();
    String::from_utf8_lossy(&out.stdout).into_owned()
}

#[test]
fn files_separate_deleted_and_skipped_are_told_apart() {
    let r = changed_repo();
    let t = TempDir::new("out");
    let e = env_in(t.path(), t.path());
    let out = t.path().join("d.tar.gz");
    let (code, v) = json(
        &[
            "deliver",
            "--base",
            "HEAD",
            "--out",
            out.to_str().unwrap(),
            "--repo",
            r.path().to_str().unwrap(),
        ],
        &Scripted::open(),
        &e,
    );
    assert_eq!(code, 0, "{}", v.to_json_pretty());
    assert_eq!(
        strings(&v, "files"),
        ["crates/a/src/lib.rs", "crates/a/src/new.rs", "docs/new.md"]
    );
    assert_eq!(
        strings(&v, "separate"),
        [
            ".github/workflows/n.yml",
            ".github/workflows/w.yml",
            "Cargo.toml"
        ]
    );
    assert_eq!(strings(&v, "deleted"), ["crates/a/old.rs"]);
    assert_eq!(strings(&v, "skipped"), ["Cargo.lock", "tools/x/Cargo.lock"]);
    assert_eq!(v.get("verified"), Some(&Value::Bool(true)));
    assert_eq!(v.get("members"), Some(&Value::Int(3)));
    assert_eq!(
        members(out.to_str().unwrap()),
        ["crates/a/src/lib.rs", "crates/a/src/new.rs", "docs/new.md"]
    );
    assert_eq!(
        tar_file(out.to_str().unwrap(), "crates/a/src/lib.rs"),
        "two\n"
    );
    assert!(v.get("bytes").and_then(Value::as_i64).unwrap() > 0);
}

#[test]
fn the_text_form_names_what_to_deliver_by_hand_and_what_needs_a_delete() {
    let r = changed_repo();
    let t = TempDir::new("out");
    let e = env_in(t.path(), t.path());
    let out = t.path().join("d.tar.gz");
    let res = mid_cli::run(
        &args(&[
            "deliver",
            "--base",
            "HEAD",
            "--out",
            out.to_str().unwrap(),
            "--repo",
            r.path().to_str().unwrap(),
        ]),
        &Scripted::open(),
        &e,
    );
    assert_eq!(res.code, 0);
    assert!(res.stdout.contains("deliver: 3 files ->"), "{}", res.stdout);
    assert!(
        res.stdout
            .contains("deliver these by hand, outside the tarball:\n  .github/workflows/n.yml"),
        "{}",
        res.stdout
    );
    assert!(
        res.stdout.contains("a tarball cannot delete"),
        "{}",
        res.stdout
    );
    assert!(res.stdout.contains("  crates/a/old.rs"));
}

#[test]
fn a_dry_run_makes_no_tarball() {
    let r = changed_repo();
    let t = TempDir::new("out");
    let e = env_in(t.path(), t.path());
    let (code, v) = json(
        &[
            "deliver",
            "--base",
            "HEAD",
            "--dry-run",
            "--repo",
            r.path().to_str().unwrap(),
        ],
        &Scripted::open(),
        &e,
    );
    assert_eq!(code, 0);
    assert_eq!(v.get("tarball"), Some(&Value::Null));
    assert_eq!(strings(&v, "files").len(), 3);
    assert!(!t.exists("mid-delivery.tar.gz"));
}

#[test]
fn the_default_name_and_a_given_name_land_in_the_working_directory() {
    let r = changed_repo();
    let t = TempDir::new("out");
    let e = env_in(t.path(), t.path());
    let (code, _) = json(
        &[
            "deliver",
            "--base",
            "HEAD",
            "--repo",
            r.path().to_str().unwrap(),
        ],
        &Scripted::open(),
        &e,
    );
    assert_eq!(code, 0);
    assert!(t.exists("mid-delivery.tar.gz"));
    let (code, _) = json(
        &[
            "deliver",
            "--base",
            "HEAD",
            "--name",
            "mid-x",
            "--repo",
            r.path().to_str().unwrap(),
        ],
        &Scripted::open(),
        &e,
    );
    assert_eq!(code, 0);
    assert!(t.exists("mid-x.tar.gz"));
}

#[test]
fn a_tarball_made_inside_the_repository_is_not_delivered_next_time() {
    let r = changed_repo();
    let e = env_in(r.path(), r.path());
    let (code, v) = json(&["deliver", "--base", "HEAD"], &Scripted::open(), &e);
    assert_eq!(code, 0);
    assert!(r.exists("mid-delivery.tar.gz"));
    let (code, v2) = json(&["deliver", "--base", "HEAD"], &Scripted::open(), &e);
    assert_eq!(code, 0);
    assert_eq!(strings(&v2, "files"), strings(&v, "files"));
    assert!(strings(&v2, "skipped").contains(&"mid-delivery.tar.gz".to_string()));
}

#[test]
fn nothing_changed_is_exit_1() {
    let r = base_repo();
    let t = TempDir::new("out");
    let e = env_in(t.path(), t.path());
    let (code, v) = json(
        &[
            "deliver",
            "--base",
            "HEAD",
            "--repo",
            r.path().to_str().unwrap(),
        ],
        &Scripted::open(),
        &e,
    );
    assert_eq!(code, 1);
    assert_eq!(v.get("ok"), Some(&Value::Bool(false)));
    assert!(strings(&v, "files").is_empty());
}

#[test]
fn only_separate_files_changed_is_also_nothing_to_deliver_but_they_are_listed() {
    let r = base_repo();
    r.write(".github/workflows/w.yml", "changed\n");
    let t = TempDir::new("out");
    let e = env_in(t.path(), t.path());
    let res = mid_cli::run(
        &args(&[
            "deliver",
            "--base",
            "HEAD",
            "--repo",
            r.path().to_str().unwrap(),
        ]),
        &Scripted::open(),
        &e,
    );
    assert_eq!(res.code, 1);
    assert!(res.stdout.contains("nothing to put in a tarball"));
    assert!(res.stdout.contains(".github/workflows/w.yml"));
}

#[test]
fn a_base_that_does_not_exist_is_an_environment_error() {
    let r = base_repo();
    let t = TempDir::new("out");
    let e = env_in(t.path(), t.path());
    let (code, v) = json(
        &[
            "deliver",
            "--base",
            "nope",
            "--repo",
            r.path().to_str().unwrap(),
        ],
        &Scripted::open(),
        &e,
    );
    assert_eq!(code, 3);
    assert_eq!(v.get_path("error.kind").unwrap(), &Value::from("no-base"));
    // The default base is origin/main, which a fresh repository does not have.
    let (code, v) = json(
        &["deliver", "--repo", r.path().to_str().unwrap()],
        &Scripted::open(),
        &e,
    );
    assert_eq!(code, 3);
    assert!(v
        .get_path("error.message")
        .unwrap()
        .as_str()
        .unwrap()
        .contains("origin/main"));
}

#[test]
fn outside_a_repository_is_an_environment_error() {
    let t = TempDir::new("plain");
    let e = env_in(t.path(), t.path());
    let (code, v) = json(&["deliver", "--base", "HEAD"], &Scripted::open(), &e);
    assert_eq!(code, 3);
    assert_eq!(
        v.get_path("error.kind").unwrap(),
        &Value::from("not-a-repository")
    );
}

#[test]
fn names_with_spaces_survive() {
    let r = base_repo();
    r.write("docs/a file with spaces.md", "x\n");
    let t = TempDir::new("out");
    let e = env_in(t.path(), t.path());
    let out = t.path().join("d.tar.gz");
    let (code, v) = json(
        &[
            "deliver",
            "--base",
            "HEAD",
            "--out",
            out.to_str().unwrap(),
            "--repo",
            r.path().to_str().unwrap(),
        ],
        &Scripted::open(),
        &e,
    );
    assert_eq!(code, 0, "{}", v.to_json_pretty());
    assert_eq!(
        members(out.to_str().unwrap()),
        ["docs/a file with spaces.md"]
    );
}

#[test]
fn a_staged_change_and_an_unstaged_one_both_count() {
    let r = base_repo();
    r.write("docs/d.md", "staged\n");
    git(r.path(), &["add", "docs/d.md"]);
    r.write("crates/a/src/lib.rs", "unstaged\n");
    let t = TempDir::new("out");
    let e = env_in(t.path(), t.path());
    let (_, v) = json(
        &[
            "deliver",
            "--base",
            "HEAD",
            "--dry-run",
            "--repo",
            r.path().to_str().unwrap(),
        ],
        &Scripted::open(),
        &e,
    );
    assert_eq!(strings(&v, "files"), ["crates/a/src/lib.rs", "docs/d.md"]);
}
