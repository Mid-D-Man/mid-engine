// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-cli.md, section "cmds/deliver.rs"
// ============================================================================
//! `mid deliver`: a delivery tarball of the files that differ from a base.

use std::path::{Path, PathBuf};

use mid_reflect::Value;

use crate::args::{self, Flag};
use crate::git;
use crate::proc::{Cmd, Runner};
use crate::report::{count, strings, Report, EXIT_ENV, EXIT_FAILED, EXIT_OK};
use crate::util::path_text;
use crate::{cmds, Env};

const SPEC: &[(&str, Flag, Option<char>)] = &[
    ("base", Flag::Value, None),
    ("out", Flag::Value, None),
    ("name", Flag::Value, None),
    ("dry-run", Flag::Switch, None),
];

/// Where a changed file goes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Class {
    /// Into the tarball.
    Deliver,
    /// Delivered by hand, outside the tarball.
    Separate,
    /// Never delivered.
    Skip,
}

/// The delivery convention: workflow files and the root `Cargo.toml` go
/// separately; a lockfile at any depth, the tarballs themselves and build
/// output never.
pub fn classify(path: &str) -> Class {
    if path.starts_with(".github/") || path == "Cargo.toml" {
        Class::Separate
    } else if path == "Cargo.lock"
        || path.ends_with("/Cargo.lock")
        || path.starts_with(".mdix/replacements/")
        || path.ends_with(".tar.gz")
        || path.starts_with("target/")
        || path.contains("/target/")
    {
        Class::Skip
    } else {
        Class::Deliver
    }
}

/// Runs `mid deliver`.
pub fn run(args: &[String], runner: &dyn Runner, env: &Env, dir: &Path) -> Report {
    let parsed = match args::parse(args, SPEC) {
        Ok(p) => p,
        Err(e) => return cmds::usage("deliver", &e),
    };
    let root = match cmds::repo_root("deliver", runner, dir) {
        Ok(r) => r,
        Err(report) => return report,
    };
    let base = parsed.value("base").unwrap_or("origin/main").to_string();
    if !git::ref_exists(runner, &root, &base) {
        return Report::error(
            "deliver",
            EXIT_ENV,
            "no-base",
            &format!("`{base}` does not exist here; fetch it or give `--base <ref>`"),
        );
    }
    let changes = match git::changes_since(runner, &root, &base) {
        Ok(c) => c,
        Err(e) => return Report::error("deliver", EXIT_ENV, "git", &e),
    };

    let mut files: Vec<String> = Vec::new();
    let mut separate: Vec<String> = Vec::new();
    let mut skipped: Vec<String> = Vec::new();
    for path in changes.changed.iter().chain(changes.untracked.iter()) {
        match classify(path) {
            Class::Deliver => files.push(path.clone()),
            Class::Separate => separate.push(path.clone()),
            Class::Skip => skipped.push(path.clone()),
        }
    }
    for list in [&mut files, &mut separate, &mut skipped] {
        list.sort();
        list.dedup();
    }
    let mut deleted: Vec<String> = changes
        .deleted
        .iter()
        .filter(|p| classify(p) != Class::Skip)
        .cloned()
        .collect();
    deleted.sort();

    let out_path = {
        let name = parsed.value("name").unwrap_or("mid-delivery");
        match parsed.value("out") {
            Some(o) => {
                let p = PathBuf::from(o);
                if p.is_absolute() {
                    p
                } else {
                    env.cwd.join(p)
                }
            }
            None => env.cwd.join(format!("{name}.tar.gz")),
        }
    };

    let mut report = Report::new("deliver", true, EXIT_OK)
        .field("base", Value::from(base.as_str()))
        .field("files", strings(&files))
        .field("separate", strings(&separate))
        .field("deleted", strings(&deleted))
        .field("skipped", strings(&skipped));

    if files.is_empty() {
        report.ok = false;
        report.exit = EXIT_FAILED;
        report = report.line(format!(
            "deliver: nothing to put in a tarball against `{base}`"
        ));
        return with_side_lists(report, &separate, &deleted);
    }
    if parsed.has("dry-run") {
        report = report
            .field("tarball", Value::Null)
            .field("bytes", Value::Null)
            .line(format!(
                "deliver (dry run): {} files would go in the tarball",
                files.len()
            ));
        for f in &files {
            report = report.line(format!("  {f}"));
        }
        return with_side_lists(report, &separate, &deleted);
    }

    // The list goes in a file so any number of names fits.
    let list_dir = env.temp.join("mid-cli");
    let _ = std::fs::create_dir_all(&list_dir);
    let list_path = list_dir.join(format!(
        "deliver-{}.txt",
        crate::util::fnv_hex(&path_text(&out_path))
    ));
    if std::fs::write(&list_path, files.join("\n") + "\n").is_err() {
        return Report::error(
            "deliver",
            EXIT_ENV,
            "write",
            "could not write the file list",
        );
    }
    let made = runner.run(&Cmd::new(
        "tar",
        &[
            "czf",
            &path_text(&out_path),
            "-C",
            &path_text(&root),
            "-T",
            &path_text(&list_path),
        ],
    ));
    let _ = std::fs::remove_file(&list_path);
    if !made.started() {
        return Report::error("deliver", EXIT_ENV, "no-tar", "`tar` is not available");
    }
    if !made.success() {
        return Report::error(
            "deliver",
            EXIT_FAILED,
            "tar",
            &format!("`tar` failed: {}", made.stderr.trim()),
        );
    }
    // Read the tarball back and compare its members with the list.
    let listing = runner.run(&Cmd::new("tar", &["tzf", &path_text(&out_path)]));
    let mut members: Vec<String> = listing
        .stdout
        .lines()
        .map(|l| l.trim_end_matches('\r').to_string())
        .filter(|l| !l.is_empty() && !l.ends_with('/'))
        .collect();
    members.sort();
    let verified = members == files;
    let bytes = std::fs::metadata(&out_path).map(|m| m.len()).unwrap_or(0);
    report.ok = verified;
    report.exit = if verified { EXIT_OK } else { EXIT_FAILED };
    report = report
        .field("tarball", Value::from(path_text(&out_path)))
        .field("bytes", Value::Int(bytes as i64))
        .field("members", count(members.len()))
        .field("verified", Value::Bool(verified))
        .line(format!(
            "deliver: {} files -> {} ({} bytes){}",
            files.len(),
            out_path.display(),
            bytes,
            if verified {
                ""
            } else {
                "  MEMBERS DO NOT MATCH THE LIST"
            }
        ));
    with_side_lists(report, &separate, &deleted)
}

fn with_side_lists(mut report: Report, separate: &[String], deleted: &[String]) -> Report {
    if !separate.is_empty() {
        report = report.line("deliver these by hand, outside the tarball:");
        for f in separate {
            report = report.line(format!("  {f}"));
        }
    }
    if !deleted.is_empty() {
        report = report.line("deleted since the base (a tarball cannot delete; needs a patch):");
        for f in deleted {
            report = report.line(format!("  {f}"));
        }
    }
    report
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_delivery_convention() {
        assert_eq!(classify("crates/a/src/lib.rs"), Class::Deliver);
        assert_eq!(classify("docs/x.md"), Class::Deliver);
        assert_eq!(classify("crates/a/Cargo.toml"), Class::Deliver);
        assert_eq!(classify(".github/workflows/x.yml"), Class::Separate);
        assert_eq!(classify(".github/anything.html"), Class::Separate);
        assert_eq!(classify("Cargo.toml"), Class::Separate);
        assert_eq!(classify("Cargo.lock"), Class::Skip);
        assert_eq!(classify("tools/mid-cli/Cargo.lock"), Class::Skip);
        assert_eq!(classify("a/b/Cargo.lock"), Class::Skip);
        assert_eq!(classify("crates/a/NotCargo.lock"), Class::Deliver);
        assert_eq!(classify("Cargo.lock.md"), Class::Deliver);
        assert_eq!(classify(".mdix/replacements/x.tar.gz"), Class::Skip);
        assert_eq!(classify("out/mid-delivery.tar.gz"), Class::Skip);
        assert_eq!(classify("tools/mid-cli/target/debug/x"), Class::Skip);
        assert_eq!(classify("tools/mid-cli/Cargo.toml"), Class::Deliver);
    }
}
