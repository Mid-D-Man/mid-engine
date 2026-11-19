//! `mid docs lint`, against real git repositories.

mod common;
use common::*;

use mid_reflect::Value;

const NOTICE: &str = "\
// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-x.md, section \"FILE\"
// ============================================================================
";

fn notice(file: &str) -> String {
    NOTICE.replace("FILE", file)
}

fn base() -> TempDir {
    repo_with(&[
        (
            "docs/mid-x.md",
            "old \u{2014} dash from before\n# mid-x\n### `lib.rs`\n",
        ),
        (
            "crates/mid-x/src/lib.rs",
            &format!("{}//! x\n", notice("lib.rs")),
        ),
        ("crates/mid-x/src/other.rs", "//! no header\n"),
    ])
}

fn findings(v: &Value) -> Vec<(String, String, i64)> {
    v.get("findings")
        .and_then(Value::as_list)
        .unwrap()
        .iter()
        .map(|f| {
            (
                f.get("rule").and_then(Value::as_str).unwrap().to_string(),
                f.get("file").and_then(Value::as_str).unwrap().to_string(),
                f.get("line").and_then(Value::as_i64).unwrap(),
            )
        })
        .collect()
}

fn lint(r: &TempDir, extra: &[&str]) -> (i32, Value) {
    let t = TempDir::new("lintenv");
    let e = env_in(t.path(), t.path());
    let mut a = vec![
        "docs",
        "lint",
        "--base",
        "HEAD",
        "--repo",
        r.path().to_str().unwrap(),
    ];
    a.extend_from_slice(extra);
    json(&a, &Scripted::open(), &e)
}

#[test]
fn a_clean_tree_is_exit_0_and_old_lines_are_not_blamed() {
    let r = base();
    let (code, v) = lint(&r, &[]);
    assert_eq!(code, 0, "{}", v.to_json_pretty());
    assert!(findings(&v).is_empty());
    assert_eq!(v.get_path("checked.md_files").unwrap(), &Value::Int(0));
}

#[test]
fn added_lines_with_em_dashes_and_filler_are_found_at_their_line_numbers() {
    let r = base();
    r.write(
        "docs/mid-x.md",
        "old \u{2014} dash from before\n# mid-x\n### `lib.rs`\nfine\nit is seamless\nanother \u{2014} dash\n",
    );
    let (code, v) = lint(&r, &[]);
    assert_eq!(code, 1);
    assert_eq!(
        findings(&v),
        vec![
            ("banned-word".to_string(), "docs/mid-x.md".to_string(), 5),
            ("em-dash".to_string(), "docs/mid-x.md".to_string(), 6),
        ]
    );
    assert_eq!(v.get_path("counts.em-dash").unwrap(), &Value::Int(1));
    assert_eq!(v.get_path("checked.md_lines").unwrap(), &Value::Int(3));
}

#[test]
fn an_untracked_markdown_file_is_read_whole() {
    let r = base();
    r.write("docs/new.md", "ok\nbad \u{2014} line\nfine\n");
    let (code, v) = lint(&r, &[]);
    assert_eq!(code, 1);
    assert_eq!(
        findings(&v),
        vec![("em-dash".to_string(), "docs/new.md".to_string(), 2)]
    );
    assert_eq!(v.get_path("checked.md_lines").unwrap(), &Value::Int(3));
}

#[test]
fn the_allow_marker_silences_a_line() {
    let r = base();
    r.write(
        "docs/new.md",
        "the banned word \"leverage\" (mid-lint: allow)\n",
    );
    let (code, _) = lint(&r, &[]);
    assert_eq!(code, 0);
}

#[test]
fn a_changed_source_file_needs_a_notice_header_and_a_doc_heading() {
    let r = base();
    r.write("crates/mid-x/src/other.rs", "//! still no header\n");
    let (code, v) = lint(&r, &[]);
    assert_eq!(code, 1);
    assert_eq!(
        findings(&v),
        vec![
            (
                "notice-header".to_string(),
                "crates/mid-x/src/other.rs".to_string(),
                1
            ),
            (
                "doc-section".to_string(),
                "crates/mid-x/src/other.rs".to_string(),
                0
            ),
        ]
    );
}

#[test]
fn a_header_that_names_a_missing_doc_and_a_header_with_no_doc_are_found() {
    let r = base();
    r.write(
        "crates/mid-x/src/a.rs",
        "// NOTICE: Full documentation\n// live in docs/does-not-exist.md, section \"a.rs\"\n",
    );
    r.write(
        "crates/mid-x/src/b.rs",
        "// NOTICE: Full documentation lives somewhere\n",
    );
    let (_, v) = lint(&r, &[]);
    let f = v.get("findings").and_then(Value::as_list).unwrap();
    let messages: Vec<&str> = f
        .iter()
        .map(|x| x.get("message").and_then(Value::as_str).unwrap())
        .collect();
    assert!(
        messages
            .iter()
            .any(|m| m.contains("`docs/does-not-exist.md`, which does not exist")),
        "{messages:?}"
    );
    assert!(
        messages
            .iter()
            .any(|m| m.contains("does not name a docs/*.md file")),
        "{messages:?}"
    );
}

#[test]
fn a_file_with_a_header_and_a_heading_is_clean() {
    let r = base();
    r.write(
        "crates/mid-x/src/third.rs",
        &format!("{}//! third\n", notice("third.rs")),
    );
    r.write(
        "docs/mid-x.md",
        "old \u{2014} dash from before\n# mid-x\n### `lib.rs`\n### `third.rs`\n",
    );
    let (code, v) = lint(&r, &[]);
    assert_eq!(code, 0, "{}", v.to_json_pretty());
    assert_eq!(v.get_path("checked.rs_files").unwrap(), &Value::Int(1));
}

#[test]
fn split_docs_and_nested_paths_are_understood() {
    let r = base();
    r.write(
        "crates/mid-x/src/cmds/mod.rs",
        &format!("{}//! cmds\n", notice("cmds/mod.rs")),
    );
    r.write(
        "crates/mid-x/src/events.rs",
        &format!("{}//! e\n", notice("events.rs")),
    );
    r.write(
        "docs/mid-x-events.md",
        "# events\n### `events.rs`\n### `cmds/mod.rs`\n",
    );
    let (code, v) = lint(&r, &[]);
    assert_eq!(code, 0, "{}", v.to_json_pretty());
    // A heading for the bare name `mod.rs` is not enough for a nested file.
    r.write(
        "docs/mid-x-events.md",
        "# events\n### `events.rs`\n### `mod.rs`\n",
    );
    let (code, v) = lint(&r, &[]);
    assert_eq!(code, 1);
    assert_eq!(
        findings(&v),
        vec![(
            "doc-section".to_string(),
            "crates/mid-x/src/cmds/mod.rs".to_string(),
            0
        )]
    );
}

#[test]
fn a_crate_with_no_docs_at_all_is_named() {
    let r = base();
    r.write(
        "crates/mid-y/src/lib.rs",
        &format!("{}//! y\n", notice("lib.rs").replace("mid-x", "mid-y")),
    );
    let (_, v) = lint(&r, &[]);
    let f = v.get("findings").and_then(Value::as_list).unwrap();
    assert!(
        f.iter().any(|x| x
            .get("message")
            .and_then(Value::as_str)
            .unwrap()
            .contains("has no docs/mid-y.md")),
        "{}",
        v.to_json_pretty()
    );
}

#[test]
fn only_source_under_src_is_linted() {
    let r = base();
    r.write(
        "crates/mid-x/tests/t.rs",
        "//! no header, and that is fine\n",
    );
    r.write("crates/mid-x/benches/b.rs", "//! none\n");
    r.write("web/ui/src/x.rs", "//! none\n");
    let (code, v) = lint(&r, &[]);
    assert_eq!(code, 0);
    assert_eq!(v.get_path("checked.rs_files").unwrap(), &Value::Int(0));
}

#[test]
fn all_checks_every_line_and_every_source_file() {
    let r = base();
    let (code, v) = lint(&r, &["--all"]);
    assert_eq!(code, 1);
    let f = findings(&v);
    assert!(
        f.contains(&("em-dash".to_string(), "docs/mid-x.md".to_string(), 1)),
        "{f:?}"
    );
    assert!(
        f.iter()
            .any(|(rule, file, _)| rule == "notice-header" && file == "crates/mid-x/src/other.rs"),
        "{f:?}"
    );
    assert_eq!(v.get("base_used"), Some(&Value::from("(all files)")));
}

#[test]
fn a_missing_base_falls_back_to_head_and_says_so() {
    let r = base();
    let t = TempDir::new("lintenv");
    let e = env_in(t.path(), t.path());
    let res = mid_cli::run(
        &args(&["docs", "lint", "--repo", r.path().to_str().unwrap()]),
        &Scripted::open(),
        &e,
    );
    assert_eq!(res.code, 0);
    assert!(
        res.stdout
            .contains("`origin/main` does not exist here, compared with HEAD"),
        "{}",
        res.stdout
    );
    let (_, v) = json(
        &["docs", "lint", "--repo", r.path().to_str().unwrap()],
        &Scripted::open(),
        &e,
    );
    assert_eq!(v.get("base"), Some(&Value::from("origin/main")));
    assert_eq!(v.get("base_used"), Some(&Value::from("HEAD")));
}

#[test]
fn the_text_form_lists_each_problem_with_its_rule_and_place() {
    let r = base();
    r.write("docs/new.md", "one\ntwo \u{2014} three\n");
    let t = TempDir::new("lintenv");
    let e = env_in(t.path(), t.path());
    let res = mid_cli::run(
        &args(&[
            "docs",
            "lint",
            "--base",
            "HEAD",
            "--repo",
            r.path().to_str().unwrap(),
        ]),
        &Scripted::open(),
        &e,
    );
    assert_eq!(res.code, 1);
    assert!(
        res.stdout
            .contains("1 md files (2 lines), 0 source files checked, 1 problems"),
        "{}",
        res.stdout
    );
    assert!(
        res.stdout.contains("[em-dash] docs/new.md:2:"),
        "{}",
        res.stdout
    );
}
