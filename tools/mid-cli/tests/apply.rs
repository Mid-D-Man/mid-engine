//! `mid apply`, against real git repositories and real tar, with a stand-in
//! for the `node` run of the mdix tool.

mod common;
use common::*;

use std::process::Command;

use mid_cli::{Cmd, Outcome};
use mid_reflect::Value;

const TEMPLATE: &str = ".mdix/replacements/replacements.mdix";

fn repo() -> TempDir {
    repo_with(&[
        (TEMPLATE, "// template\n"),
        ("crates/a/src/lib.rs", "old\n"),
        ("docs/d.md", "d\n"),
    ])
}

/// A directory shaped like a clone of the mdix tool. Only the script's
/// presence matters, because `node` is answered by the stand-in.
fn mdix_dir() -> TempDir {
    let d = TempDir::new("mdix");
    d.write("bin/mdix-scaffold.js", "// stand-in\n");
    d
}

fn p(dir: &TempDir) -> &str {
    dir.path().to_str().unwrap()
}

/// A tarball with exactly these members, made by real tar.
fn tarball(dir: &TempDir, name: &str, files: &[(&str, &str)]) -> String {
    let stage = TempDir::new("stage");
    for (rel, text) in files {
        stage.write(rel, text);
    }
    let out = dir.path().join(name);
    let mut cmd = Command::new("tar");
    cmd.arg("czf").arg(&out).arg("-C").arg(stage.path());
    for (rel, _) in files {
        cmd.arg(rel);
    }
    assert!(
        cmd.status().unwrap().success(),
        "tar could not build {name}"
    );
    out.to_string_lossy().into_owned()
}

/// A tarball of a whole staging folder, so its members start with `./` the
/// way `tar czf x.tar.gz -C dir .` writes them.
fn tarball_of_folder(dir: &TempDir, name: &str, files: &[(&str, &str)]) -> String {
    let stage = TempDir::new("stage");
    for (rel, text) in files {
        stage.write(rel, text);
    }
    let out = dir.path().join(name);
    let made = Command::new("tar")
        .arg("czf")
        .arg(&out)
        .arg("-C")
        .arg(stage.path())
        .arg(".")
        .status()
        .unwrap();
    assert!(made.success());
    out.to_string_lossy().into_owned()
}

fn members_of(tar: &str) -> Vec<String> {
    let out = Command::new("tar").args(["tzf", tar]).output().unwrap();
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .map(str::to_string)
        .collect()
}

/// How the stand-in for the mdix tool behaves.
#[derive(Clone)]
enum Mdix {
    /// Unpacks every member and says so.
    Faithful,
    /// Leaves this member out and does not mention it.
    Skips(&'static str),
    /// Places this member too but does not mention it.
    Unreported(&'static str),
    /// Places everything, then leaves different bytes in this member.
    WritesWrong(&'static str),
    /// Exits with this code and places nothing.
    Exits(i32),
    /// Exits 0 but reports one error and places nothing.
    ReportsErrors,
    /// Places everything faithfully but counts one error in its closing line.
    PlacesButCountsAnError,
}

/// Answers `node` the way the tool does: finds the tarball in the
/// replacements folder, unpacks it into the repository, prints one REP or
/// NEW line per member and a closing count.
fn standin(mode: Mdix) -> impl Fn(&Cmd) -> Option<Outcome> + 'static {
    move |cmd| {
        if cmd.program != "node" {
            return None;
        }
        if let Mdix::Exits(code) = &mode {
            return Some(failed(*code, "", "the tool failed"));
        }
        if let Mdix::ReportsErrors = &mode {
            return Some(ok("Done. 0 created, 0 replaced, 0 skipped, 1 error(s).\n"));
        }
        let root = cmd.cwd.clone().expect("node runs in the repository");
        let dry = cmd.args.iter().any(|a| a == "--dry-run");
        let folder = root.join(".mdix/replacements");
        let tar = std::fs::read_dir(&folder)
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .find(|path| path.to_string_lossy().ends_with(".tar.gz"))
            .expect("a tarball in the replacements folder");
        let tar = tar.to_string_lossy().into_owned();

        let skipped = match &mode {
            Mdix::Skips(name) => Some(*name),
            _ => None,
        };
        let unreported = match &mode {
            Mdix::Unreported(name) => Some(*name),
            _ => None,
        };
        let mut out = String::new();
        let (mut created, mut replaced) = (0, 0);
        for member in members_of(&tar) {
            let member = member.strip_prefix("./").unwrap_or(&member).to_string();
            if member.is_empty()
                || member.ends_with('/')
                || Some(member.as_str()) == skipped
                || Some(member.as_str()) == unreported
            {
                continue;
            }
            if root.join(&member).exists() {
                replaced += 1;
                out.push_str(&format!("  REP  ./{member}  (overwritten)\n"));
            } else {
                created += 1;
                out.push_str(&format!("  NEW  ./{member}\n"));
            }
        }
        if !dry {
            let mut extract = Command::new("tar");
            extract.args(["xzf", &tar, "-C"]).arg(&root);
            if let Some(name) = skipped {
                extract.arg("--exclude").arg(name);
            }
            assert!(extract.status().unwrap().success());
            if let Mdix::WritesWrong(name) = &mode {
                std::fs::write(root.join(name), "garbage\n").unwrap();
            }
        }
        let errors = u32::from(matches!(mode, Mdix::PlacesButCountsAnError));
        out.push_str(&format!(
            "Done. {created} created, {replaced} replaced, 0 skipped, {errors} error(s).\n"
        ));
        Some(ok(&out))
    }
}

/// Runs `mid apply <extra...> --repo <repo>` and parses its JSON.
fn run_apply(
    repo: &TempDir,
    extra: &[&str],
    runner: &Scripted,
    mdix: Option<&TempDir>,
) -> (i32, Value) {
    let t = TempDir::new("env");
    let mut e = env_in(t.path(), t.path());
    e.mdix = mdix.map(|d| p(d).to_string());
    let mut list = vec!["apply"];
    list.extend_from_slice(extra);
    list.push("--repo");
    list.push(p(repo));
    json(&list, runner, &e)
}

fn kind(v: &Value) -> String {
    v.get("error")
        .and_then(|e| e.get("kind"))
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string()
}

fn int(n: i64) -> Option<Value> {
    Some(Value::Int(n))
}

fn delivery(out: &TempDir) -> String {
    tarball(
        out,
        "d.tar.gz",
        &[("crates/a/src/lib.rs", "new\n"), ("docs/new.md", "n\n")],
    )
}

#[test]
fn a_delivery_lands_and_every_member_is_verified() {
    let (r, m, out) = (repo(), mdix_dir(), TempDir::new("out"));
    let tar = delivery(&out);
    let runner = Scripted::open().on(standin(Mdix::Faithful));
    let (code, v) = run_apply(&r, &[&tar, "--mdix", p(&m)], &runner, None);
    assert_eq!(code, 0, "{}", v.to_json_pretty());
    assert_eq!(v.get("ok"), Some(&Value::Bool(true)));
    assert_eq!(v.get("verified"), Some(&Value::Bool(true)));
    assert_eq!(v.get("members"), int(2).as_ref());
    assert_eq!(v.get("created"), int(1).as_ref());
    assert_eq!(v.get("replaced"), int(1).as_ref());
    assert_eq!(v.get("errors"), int(0).as_ref());
    assert!(strings(&v, "mismatches").is_empty());
    assert!(strings(&v, "not_placed").is_empty());
    assert_eq!(r.read("crates/a/src/lib.rs"), "new\n");
    assert_eq!(r.read("docs/new.md"), "n\n");
    assert_eq!(
        r.read("docs/d.md"),
        "d\n",
        "a file not in the tarball is untouched"
    );
    assert!(
        !r.exists(".mdix/replacements/d.tar.gz"),
        "the copy of the tarball is removed after the run"
    );
    assert!(runner.ran("bin/mdix-scaffold.js generate --template .mdix/replacements/replacements.mdix --file-strategy overwrite"));
    assert!(!runner.ran("--dry-run"));
}

#[test]
fn members_written_with_a_leading_dot_slash_are_matched_and_the_folder_entry_is_ignored() {
    let (r, m, out) = (repo(), mdix_dir(), TempDir::new("out"));
    let tar = tarball_of_folder(
        &out,
        "d.tar.gz",
        &[("crates/a/src/lib.rs", "new\n"), ("docs/new.md", "n\n")],
    );
    assert!(members_of(&tar).iter().all(|n| n.starts_with("./")));
    let runner = Scripted::open().on(standin(Mdix::Faithful));
    let (code, v) = run_apply(&r, &[&tar, "--mdix", p(&m)], &runner, None);
    assert_eq!(code, 0, "{}", v.to_json_pretty());
    assert_eq!(
        v.get("members"),
        int(2).as_ref(),
        "the `./` entry is not a file"
    );
    assert_eq!(v.get("verified"), Some(&Value::Bool(true)));
    assert!(strings(&v, "not_placed").is_empty());
    assert_eq!(r.read("docs/new.md"), "n\n");
}

#[test]
fn a_dry_run_changes_nothing_and_verifies_nothing() {
    let (r, m, out) = (repo(), mdix_dir(), TempDir::new("out"));
    let tar = delivery(&out);
    let runner = Scripted::open().on(standin(Mdix::Faithful));
    let (code, v) = run_apply(&r, &[&tar, "--dry-run", "--mdix", p(&m)], &runner, None);
    assert_eq!(code, 0, "{}", v.to_json_pretty());
    assert_eq!(v.get("dry_run"), Some(&Value::Bool(true)));
    assert_eq!(v.get("verified"), Some(&Value::Null));
    assert_eq!(r.read("crates/a/src/lib.rs"), "old\n");
    assert!(!r.exists("docs/new.md"));
    assert!(!r.exists(".mdix/replacements/d.tar.gz"));
    assert!(runner.ran("--dry-run"), "the flag reaches the tool");
}

#[test]
fn keep_leaves_the_copy_in_the_replacements_folder_except_on_a_dry_run() {
    let (m, out) = (mdix_dir(), TempDir::new("out"));
    let tar = delivery(&out);

    let r = repo();
    let runner = Scripted::open().on(standin(Mdix::Faithful));
    let (code, v) = run_apply(&r, &[&tar, "--keep", "--mdix", p(&m)], &runner, None);
    assert_eq!(code, 0, "{}", v.to_json_pretty());
    assert!(r.exists(".mdix/replacements/d.tar.gz"));

    let r = repo();
    let runner = Scripted::open().on(standin(Mdix::Faithful));
    let (code, v) = run_apply(
        &r,
        &[&tar, "--keep", "--dry-run", "--mdix", p(&m)],
        &runner,
        None,
    );
    assert_eq!(code, 0, "{}", v.to_json_pretty());
    assert!(
        !r.exists(".mdix/replacements/d.tar.gz"),
        "a dry run never keeps the copy"
    );
}

#[test]
fn a_member_that_leaves_the_repository_is_refused_before_the_tool_runs() {
    let (r, m, out) = (repo(), mdix_dir(), TempDir::new("out"));
    let stage = TempDir::new("stage");
    stage.write("x", "x\n");
    let tar = out.path().join("bad.tar.gz");
    let made = Command::new("tar")
        .args(["czf"])
        .arg(&tar)
        .args(["-C", p(&stage), "--transform", "s,^x$,../x,", "x"])
        .status()
        .unwrap();
    assert!(made.success());
    let tar = tar.to_string_lossy().into_owned();
    assert_eq!(
        members_of(&tar),
        ["../x"],
        "the fixture really has the member"
    );

    let runner = Scripted::open().on(standin(Mdix::Faithful));
    let (code, v) = run_apply(&r, &[&tar, "--mdix", p(&m)], &runner, None);
    assert_eq!(code, 1, "{}", v.to_json_pretty());
    assert_eq!(kind(&v), "unsafe-member");
    assert!(v
        .get("error")
        .and_then(|e| e.get("message"))
        .and_then(Value::as_str)
        .unwrap_or("")
        .contains("../x"));
    assert!(!runner.ran("node"), "nothing runs for an unsafe tarball");
    assert!(!r.exists(".mdix/replacements/bad.tar.gz"));
}

#[test]
fn a_member_the_tool_did_not_place_fails_the_run() {
    let (r, m, out) = (repo(), mdix_dir(), TempDir::new("out"));
    let tar = delivery(&out);
    let runner = Scripted::open().on(standin(Mdix::Skips("docs/new.md")));
    let (code, v) = run_apply(&r, &[&tar, "--mdix", p(&m)], &runner, None);
    assert_eq!(code, 1, "{}", v.to_json_pretty());
    assert_eq!(v.get("ok"), Some(&Value::Bool(false)));
    assert_eq!(strings(&v, "not_placed"), ["docs/new.md"]);
    assert_eq!(strings(&v, "mismatches"), ["docs/new.md"]);
    assert_eq!(v.get("verified"), Some(&Value::Bool(false)));
}

#[test]
fn a_member_with_the_wrong_bytes_is_caught_by_the_comparison() {
    let (r, m, out) = (repo(), mdix_dir(), TempDir::new("out"));
    let tar = delivery(&out);
    let runner = Scripted::open().on(standin(Mdix::WritesWrong("crates/a/src/lib.rs")));
    let (code, v) = run_apply(&r, &[&tar, "--mdix", p(&m)], &runner, None);
    assert_eq!(code, 1, "{}", v.to_json_pretty());
    assert_eq!(strings(&v, "mismatches"), ["crates/a/src/lib.rs"]);
    assert!(strings(&v, "not_placed").is_empty());
    assert_eq!(v.get("verified"), Some(&Value::Bool(false)));
}

#[test]
fn a_member_the_tool_never_reported_fails_the_run_even_if_its_bytes_match() {
    let (r, m, out) = (repo(), mdix_dir(), TempDir::new("out"));
    let tar = delivery(&out);
    let runner = Scripted::open().on(standin(Mdix::Unreported("docs/new.md")));
    let (code, v) = run_apply(&r, &[&tar, "--mdix", p(&m)], &runner, None);
    assert_eq!(code, 1, "{}", v.to_json_pretty());
    assert_eq!(strings(&v, "not_placed"), ["docs/new.md"]);
    assert!(strings(&v, "mismatches").is_empty());
    assert_eq!(v.get("verified"), Some(&Value::Bool(true)));
    assert_eq!(v.get("ok"), Some(&Value::Bool(false)));
}

#[test]
fn the_placeholder_the_tool_rewrites_is_not_compared() {
    let (r, m, out) = (repo(), mdix_dir(), TempDir::new("out"));
    let tar = tarball(
        &out,
        "d.tar.gz",
        &[
            ("crates/a/src/lib.rs", "new\n"),
            (".mdix/replacements/gitkeep", "keep\n"),
        ],
    );
    let runner = Scripted::open().on(standin(Mdix::WritesWrong(".mdix/replacements/gitkeep")));
    let (code, v) = run_apply(&r, &[&tar, "--mdix", p(&m)], &runner, None);
    assert_eq!(code, 0, "{}", v.to_json_pretty());
    assert_eq!(v.get("verified"), Some(&Value::Bool(true)));
    assert!(strings(&v, "mismatches").is_empty());
}

#[test]
fn a_tool_that_exits_nonzero_fails_the_run_without_verifying() {
    let (r, m, out) = (repo(), mdix_dir(), TempDir::new("out"));
    let tar = delivery(&out);
    let runner = Scripted::open().on(standin(Mdix::Exits(3)));
    let (code, v) = run_apply(&r, &[&tar, "--mdix", p(&m)], &runner, None);
    assert_eq!(code, 1, "{}", v.to_json_pretty());
    assert_eq!(v.get("ok"), Some(&Value::Bool(false)));
    assert_eq!(v.get("mdix_exit"), int(3).as_ref());
    assert_eq!(v.get("verified"), Some(&Value::Null));
    assert!(!r.exists(".mdix/replacements/d.tar.gz"));
    assert_eq!(r.read("crates/a/src/lib.rs"), "old\n");
}

#[test]
fn errors_in_the_tools_closing_line_fail_the_run_even_on_exit_zero() {
    let (r, m, out) = (repo(), mdix_dir(), TempDir::new("out"));
    let tar = delivery(&out);
    let runner = Scripted::open().on(standin(Mdix::ReportsErrors));
    let (code, v) = run_apply(&r, &[&tar, "--mdix", p(&m)], &runner, None);
    assert_eq!(code, 1, "{}", v.to_json_pretty());
    assert_eq!(v.get("errors"), int(1).as_ref());
    assert_eq!(v.get("ok"), Some(&Value::Bool(false)));
}

#[test]
fn an_error_in_the_closing_line_fails_the_run_even_when_everything_was_placed() {
    let (r, m, out) = (repo(), mdix_dir(), TempDir::new("out"));
    let tar = delivery(&out);
    let runner = Scripted::open().on(standin(Mdix::PlacesButCountsAnError));
    let (code, v) = run_apply(&r, &[&tar, "--mdix", p(&m)], &runner, None);
    assert_eq!(code, 1, "{}", v.to_json_pretty());
    assert_eq!(v.get("errors"), int(1).as_ref());
    assert_eq!(v.get("verified"), Some(&Value::Bool(true)));
    assert!(strings(&v, "not_placed").is_empty());
    assert_eq!(v.get("ok"), Some(&Value::Bool(false)));
}

#[test]
fn the_comparison_folder_is_removed_from_the_temp_directory() {
    let (r, m, out) = (repo(), mdix_dir(), TempDir::new("out"));
    let tar = delivery(&out);
    let t = TempDir::new("env");
    let e = env_in(t.path(), t.path());
    let runner = Scripted::open().on(standin(Mdix::Faithful));
    let (code, v) = json(
        &["apply", &tar, "--mdix", p(&m), "--repo", p(&r)],
        &runner,
        &e,
    );
    assert_eq!(code, 0, "{}", v.to_json_pretty());
    let left: Vec<_> = std::fs::read_dir(t.path().join("mid-cli"))
        .map(|d| d.filter_map(|e| e.ok()).collect())
        .unwrap_or_default();
    assert!(left.is_empty(), "left behind: {left:?}");
}

#[test]
fn without_the_tool_the_run_is_an_environment_failure() {
    let (r, out) = (repo(), TempDir::new("out"));
    let tar = delivery(&out);

    // No flag, no environment.
    let runner = Scripted::open().on(standin(Mdix::Faithful));
    let (code, v) = run_apply(&r, &[&tar], &runner, None);
    assert_eq!(code, 3, "{}", v.to_json_pretty());
    assert_eq!(kind(&v), "no-mdix");
    assert!(!runner.ran("node"));
    assert!(!r.exists(".mdix/replacements/d.tar.gz"));

    // A directory with no script in it.
    let empty = TempDir::new("empty");
    let (code, v) = run_apply(&r, &[&tar, "--mdix", p(&empty)], &runner, None);
    assert_eq!(code, 3, "{}", v.to_json_pretty());
    assert_eq!(kind(&v), "no-mdix");
}

#[test]
fn the_flag_beats_the_environment_and_the_environment_is_used_alone() {
    let (m, out) = (mdix_dir(), TempDir::new("out"));
    let tar = delivery(&out);

    let r = repo();
    let runner = Scripted::open().on(standin(Mdix::Faithful));
    let (code, v) = run_apply(&r, &[&tar], &runner, Some(&m));
    assert_eq!(code, 0, "{}", v.to_json_pretty());

    let r = repo();
    let empty = TempDir::new("empty");
    let (code, v) = run_apply(&r, &[&tar, "--mdix", p(&m)], &runner, Some(&empty));
    assert_eq!(code, 0, "{}", v.to_json_pretty());
}

#[test]
fn a_repository_without_the_template_is_an_environment_failure() {
    let (m, out) = (mdix_dir(), TempDir::new("out"));
    let r = repo_with(&[("crates/a/src/lib.rs", "old\n")]);
    let tar = delivery(&out);
    let runner = Scripted::open().on(standin(Mdix::Faithful));
    let (code, v) = run_apply(&r, &[&tar, "--mdix", p(&m)], &runner, None);
    assert_eq!(code, 3, "{}", v.to_json_pretty());
    assert_eq!(kind(&v), "no-template");
    assert!(!runner.ran("node"));
}

#[test]
fn a_tarball_already_in_the_replacements_folder_is_never_overwritten_or_removed() {
    let (r, m, out) = (repo(), mdix_dir(), TempDir::new("out"));
    let tar = delivery(&out);
    r.write(".mdix/replacements/d.tar.gz", "stale");
    let runner = Scripted::open().on(standin(Mdix::Faithful));
    let (code, v) = run_apply(&r, &[&tar, "--mdix", p(&m)], &runner, None);
    assert_eq!(code, 1, "{}", v.to_json_pretty());
    assert_eq!(kind(&v), "exists");
    assert_eq!(r.read(".mdix/replacements/d.tar.gz"), "stale");
    assert!(!runner.ran("node"));
}

#[test]
fn a_missing_file_is_a_usage_error() {
    let (r, m) = (repo(), mdix_dir());
    let runner = Scripted::open().on(standin(Mdix::Faithful));
    let (code, v) = run_apply(&r, &["nope.tar.gz", "--mdix", p(&m)], &runner, None);
    assert_eq!(code, 2, "{}", v.to_json_pretty());
    assert_eq!(kind(&v), "no-such-file");
}

#[test]
fn a_file_that_is_not_a_tarball_is_reported_as_one() {
    let (r, m, out) = (repo(), mdix_dir(), TempDir::new("out"));
    out.write("plain.tar.gz", "this is not gzip\n");
    let tar = out
        .path()
        .join("plain.tar.gz")
        .to_string_lossy()
        .into_owned();
    let runner = Scripted::open().on(standin(Mdix::Faithful));
    let (code, v) = run_apply(&r, &[&tar, "--mdix", p(&m)], &runner, None);
    assert_eq!(code, 1, "{}", v.to_json_pretty());
    assert_eq!(kind(&v), "bad-tarball");
    assert!(!runner.ran("node"));
}

#[test]
fn exactly_one_tarball_is_required() {
    let (r, m, out) = (repo(), mdix_dir(), TempDir::new("out"));
    let one = delivery(&out);
    let two = tarball(&out, "e.tar.gz", &[("docs/e.md", "e\n")]);
    let runner = Scripted::open().on(standin(Mdix::Faithful));

    let (code, v) = run_apply(&r, &["--mdix", p(&m)], &runner, None);
    assert_eq!(code, 2, "{}", v.to_json_pretty());

    let (code, v) = run_apply(&r, &[&one, &two, "--mdix", p(&m)], &runner, None);
    assert_eq!(code, 2, "{}", v.to_json_pretty());
    assert!(!runner.ran("node"));
}

#[test]
fn outside_a_git_repository_the_run_is_an_environment_failure() {
    let (m, out, plain) = (mdix_dir(), TempDir::new("out"), TempDir::new("plain"));
    let tar = delivery(&out);
    let runner = Scripted::open().on(standin(Mdix::Faithful));
    let (code, v) = run_apply(&plain, &[&tar, "--mdix", p(&m)], &runner, None);
    assert_eq!(code, 3, "{}", v.to_json_pretty());
    assert_eq!(kind(&v), "not-a-repository");
}

#[test]
fn without_node_the_run_fails_and_the_copy_is_removed() {
    let (r, m, out) = (repo(), mdix_dir(), TempDir::new("out"));
    let tar = delivery(&out);
    let runner = Scripted::open().on(|cmd| {
        (cmd.program == "node").then(|| Outcome {
            spawn_error: Some("not found".to_string()),
            ..Outcome::default()
        })
    });
    let (code, v) = run_apply(&r, &[&tar, "--mdix", p(&m)], &runner, None);
    assert_eq!(code, 3, "{}", v.to_json_pretty());
    assert_eq!(kind(&v), "no-node");
    assert!(!r.exists(".mdix/replacements/d.tar.gz"));
}
