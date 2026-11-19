// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-cli.md, section "git.rs"
// ============================================================================
//! What the commands need from `git`: the repository root, which files
//! differ from a base, and which lines were added.

use std::path::{Path, PathBuf};

use crate::proc::{Cmd, Runner};

/// Files that differ from a base.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Changes {
    /// Added, copied, modified or renamed (the new name), tracked.
    pub changed: Vec<String>,
    /// Deleted, including the old name of a rename.
    pub deleted: Vec<String>,
    /// Not tracked and not ignored.
    pub untracked: Vec<String>,
}

/// One added line of a diff.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AddedLine {
    /// The file, relative to the repository root.
    pub path: String,
    /// The line number in the new file.
    pub line: u64,
    /// The text of the line, without the leading `+`.
    pub text: String,
}

/// The root of the repository containing `dir`.
pub fn toplevel(runner: &dyn Runner, dir: &Path) -> Option<PathBuf> {
    let out = runner.run(&Cmd::new("git", &["rev-parse", "--show-toplevel"]).in_dir(dir));
    if !out.success() {
        return None;
    }
    let line = out.stdout.lines().next()?.trim();
    if line.is_empty() {
        None
    } else {
        Some(PathBuf::from(line))
    }
}

/// Whether `reference` names a commit.
pub fn ref_exists(runner: &dyn Runner, repo: &Path, reference: &str) -> bool {
    let spec = format!("{reference}^{{commit}}");
    runner
        .run(&Cmd::new("git", &["rev-parse", "--verify", "--quiet", &spec]).in_dir(repo))
        .success()
}

fn lines_of(runner: &dyn Runner, repo: &Path, args: &[&str]) -> Result<Vec<String>, String> {
    let out = runner.run(&Cmd::new("git", args).in_dir(repo));
    if !out.started() {
        return Err(format!(
            "could not run git: {}",
            out.spawn_error.unwrap_or_default()
        ));
    }
    if !out.success() {
        return Err(format!(
            "`git {}` failed: {}",
            args.join(" "),
            out.stderr.trim()
        ));
    }
    Ok(out
        .stdout
        .lines()
        .map(str::to_string)
        .filter(|l| !l.is_empty())
        .collect())
}

/// Files that differ from `base`, in the working tree.
pub fn changes_since(runner: &dyn Runner, repo: &Path, base: &str) -> Result<Changes, String> {
    let changed = lines_of(
        runner,
        repo,
        &[
            "diff",
            "--name-only",
            "--no-renames",
            "--diff-filter=ACMR",
            base,
        ],
    )?;
    let deleted = lines_of(
        runner,
        repo,
        &[
            "diff",
            "--name-only",
            "--no-renames",
            "--diff-filter=D",
            base,
        ],
    )?;
    let untracked = lines_of(
        runner,
        repo,
        &["ls-files", "--others", "--exclude-standard"],
    )?;
    Ok(Changes {
        changed,
        deleted,
        untracked,
    })
}

/// The lines added since `base` in files matching `pathspec`.
pub fn added_lines(
    runner: &dyn Runner,
    repo: &Path,
    base: &str,
    pathspec: &str,
) -> Result<Vec<AddedLine>, String> {
    let out = runner.run(
        &Cmd::new(
            "git",
            &[
                "diff",
                "-U0",
                "--no-color",
                "--no-renames",
                base,
                "--",
                pathspec,
            ],
        )
        .in_dir(repo),
    );
    if !out.success() {
        return Err(format!("`git diff` failed: {}", out.stderr.trim()));
    }
    Ok(parse_added_lines(&out.stdout))
}

/// Reads the added lines out of `git diff -U0` output.
pub fn parse_added_lines(diff: &str) -> Vec<AddedLine> {
    let mut lines = Vec::new();
    let mut path: Option<String> = None;
    let mut next: u64 = 0;
    for raw in diff.lines() {
        if let Some(rest) = raw.strip_prefix("+++ ") {
            path = rest.strip_prefix("b/").map(str::to_string);
        } else if raw.starts_with("--- ") {
            continue;
        } else if raw.starts_with("@@") {
            next = raw
                .split_whitespace()
                .find(|t| t.starts_with('+'))
                .and_then(|t| t[1..].split(',').next())
                .and_then(|n| n.parse().ok())
                .unwrap_or(0);
        } else if let Some(text) = raw.strip_prefix('+') {
            if let Some(p) = &path {
                lines.push(AddedLine {
                    path: p.clone(),
                    line: next,
                    text: text.to_string(),
                });
            }
            next += 1;
        }
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn added_lines_carry_their_new_line_numbers() {
        let diff = "\
diff --git a/docs/a.md b/docs/a.md
index 111..222 100644
--- a/docs/a.md
+++ b/docs/a.md
@@ -3,0 +4,2 @@ heading
+first
+second
@@ -10 +12 @@
-old
+changed
diff --git a/docs/new.md b/docs/new.md
--- /dev/null
+++ b/docs/new.md
@@ -0,0 +1 @@
+only
";
        let lines = parse_added_lines(diff);
        let got: Vec<(&str, u64, &str)> = lines
            .iter()
            .map(|l| (l.path.as_str(), l.line, l.text.as_str()))
            .collect();
        assert_eq!(
            got,
            vec![
                ("docs/a.md", 4, "first"),
                ("docs/a.md", 5, "second"),
                ("docs/a.md", 12, "changed"),
                ("docs/new.md", 1, "only"),
            ]
        );
    }

    #[test]
    fn a_deleted_file_has_no_added_lines() {
        let diff = "--- a/x.md\n+++ /dev/null\n@@ -1 +0,0 @@\n-gone\n";
        assert!(parse_added_lines(diff).is_empty());
    }
}
