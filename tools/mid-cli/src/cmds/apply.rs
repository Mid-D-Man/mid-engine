// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-cli.md, section "cmds/apply.rs"
// ============================================================================
//! `mid apply`: apply a delivery tarball with the real `mdix` tool, then
//! check that every file in it landed byte for byte.

use std::path::{Path, PathBuf};

use mid_reflect::Value;

use crate::args::{self, Flag};
use crate::proc::{Cmd, Runner};
use crate::report::{count, strings, Report, EXIT_ENV, EXIT_FAILED, EXIT_OK, EXIT_USAGE};
use crate::util::{fnv_hex, path_text};
use crate::{cmds, Env};

const SPEC: &[(&str, Flag, Option<char>)] = &[
    ("mdix", Flag::Value, None),
    ("dry-run", Flag::Switch, None),
    ("keep", Flag::Switch, None),
];

/// Where the tool reads tarballs from, and which template drives it.
const REPLACEMENTS_DIR: &str = ".mdix/replacements";
const TEMPLATE: &str = ".mdix/replacements/replacements.mdix";
/// The placeholder the tool rewrites on every run; not part of a delivery.
const PLACEHOLDER: &str = ".mdix/replacements/gitkeep";

/// What the tool printed.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct MdixResult {
    /// `(REP|NEW, path)` for every file it placed.
    pub placed: Vec<(String, String)>,
    /// The counts of its closing line: created, replaced, skipped, errors.
    pub done: Option<(u64, u64, u64, u64)>,
}

/// Reads the tool's output.
pub fn parse_mdix_output(text: &str) -> MdixResult {
    let mut result = MdixResult::default();
    for line in text.lines() {
        let mut words = line.split_whitespace();
        match words.next() {
            Some(kind @ ("REP" | "NEW")) => {
                if let Some(path) = words.next() {
                    let path = path.strip_prefix("./").unwrap_or(path);
                    result.placed.push((kind.to_string(), path.to_string()));
                }
            }
            Some("Done.") => {
                let nums: Vec<u64> = line
                    .split(|c: char| !c.is_ascii_digit())
                    .filter(|s| !s.is_empty())
                    .filter_map(|s| s.parse().ok())
                    .collect();
                if nums.len() >= 4 {
                    result.done = Some((nums[0], nums[1], nums[2], nums[3]));
                }
            }
            _ => {}
        }
    }
    result
}

/// Whether a tarball member would write outside the repository.
pub fn unsafe_member(name: &str) -> bool {
    name.is_empty()
        || name.starts_with('/')
        || name.contains('\\')
        || name.split('/').any(|part| part == "..")
}

/// Runs `mid apply`.
pub fn run(args: &[String], runner: &dyn Runner, env: &Env, dir: &Path) -> Report {
    let parsed = match args::parse(args, SPEC) {
        Ok(p) => p,
        Err(e) => return cmds::usage("apply", &e),
    };
    let [tarball] = parsed.positional.as_slice() else {
        return cmds::usage(
            "apply",
            "give exactly one tarball: `mid apply <file.tar.gz>`",
        );
    };
    let dry_run = parsed.has("dry-run");
    let tar_path = {
        let p = PathBuf::from(tarball);
        if p.is_absolute() {
            p
        } else {
            env.cwd.join(p)
        }
    };
    if !tar_path.is_file() {
        return Report::error(
            "apply",
            EXIT_USAGE,
            "no-such-file",
            &format!("`{}` is not a file", tar_path.display()),
        );
    }
    let root = match cmds::repo_root("apply", runner, dir) {
        Ok(r) => r,
        Err(report) => return report,
    };

    // The members, and whether any is unsafe.
    let listing = runner.run(&Cmd::new("tar", &["tzf", &path_text(&tar_path)]));
    if !listing.started() {
        return Report::error("apply", EXIT_ENV, "no-tar", "`tar` is not available");
    }
    if !listing.success() {
        return Report::error(
            "apply",
            EXIT_FAILED,
            "bad-tarball",
            &format!("`tar` could not read it: {}", listing.stderr.trim()),
        );
    }
    let members: Vec<String> = listing
        .stdout
        .lines()
        .map(|l| l.trim_end_matches('\r').to_string())
        .filter(|l| !l.is_empty())
        .collect();
    let unsafe_names: Vec<String> = members
        .iter()
        .filter(|m| unsafe_member(m))
        .cloned()
        .collect();
    if !unsafe_names.is_empty() {
        return Report::error(
            "apply",
            EXIT_FAILED,
            "unsafe-member",
            &format!(
                "the tarball has members that leave the repository: {}",
                unsafe_names.join(", ")
            ),
        );
    }
    let files: Vec<String> = members
        .iter()
        .filter(|m| !m.ends_with('/'))
        .map(|m| m.strip_prefix("./").unwrap_or(m).to_string())
        .collect();

    // The tool and its template.
    let mdix_dir = match parsed
        .value("mdix")
        .map(str::to_string)
        .or_else(|| env.mdix.clone())
    {
        Some(d) => PathBuf::from(d),
        None => {
            return Report::error(
                "apply",
                EXIT_ENV,
                "no-mdix",
                "say where the mdix tool is: `--mdix DIR` or the MID_MDIX environment variable",
            )
        }
    };
    let script = mdix_dir.join("bin").join("mdix-scaffold.js");
    if !script.is_file() {
        return Report::error(
            "apply",
            EXIT_ENV,
            "no-mdix",
            &format!("`{}` does not exist", script.display()),
        );
    }
    if !root.join(TEMPLATE).is_file() {
        return Report::error(
            "apply",
            EXIT_ENV,
            "no-template",
            &format!("`{TEMPLATE}` is missing from the repository"),
        );
    }
    let name = tar_path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let dest = root.join(REPLACEMENTS_DIR).join(&name);
    if dest.exists() {
        return Report::error(
            "apply",
            EXIT_FAILED,
            "exists",
            &format!("`{}` is already in the replacements folder", dest.display()),
        );
    }
    if std::fs::copy(&tar_path, &dest).is_err() {
        return Report::error(
            "apply",
            EXIT_ENV,
            "copy",
            "could not copy the tarball into the repository",
        );
    }

    let mut node_args = vec![
        path_text(&script),
        "generate".to_string(),
        "--template".to_string(),
        TEMPLATE.to_string(),
        "--file-strategy".to_string(),
        "overwrite".to_string(),
    ];
    if dry_run {
        node_args.push("--dry-run".to_string());
    }
    let out = runner.run(&Cmd::new("node", &node_args).in_dir(&root));
    if !parsed.has("keep") || dry_run {
        let _ = std::fs::remove_file(&dest);
    }
    if !out.started() {
        return Report::error("apply", EXIT_ENV, "no-node", "`node` is not available");
    }
    let mdix = parse_mdix_output(&out.combined());
    let (created, replaced, skipped, errors) = mdix.done.unwrap_or((0, 0, 0, 0));

    let placed_paths: Vec<&str> = mdix.placed.iter().map(|(_, p)| p.as_str()).collect();
    let unplaced: Vec<String> = files
        .iter()
        .filter(|f| !placed_paths.contains(&f.as_str()))
        .cloned()
        .collect();

    // Verification: extract the tarball and compare it with the repository.
    let mut mismatches: Vec<String> = Vec::new();
    let mut verified: Option<bool> = None;
    if !dry_run && out.success() {
        let tmp = env.temp.join("mid-cli").join(format!(
            "apply-{}",
            fnv_hex(&format!(
                "{}{:?}",
                tar_path.display(),
                std::time::SystemTime::now()
            ))
        ));
        let _ = std::fs::create_dir_all(&tmp);
        let extract = runner.run(&Cmd::new(
            "tar",
            &["xzf", &path_text(&tar_path), "-C", &path_text(&tmp)],
        ));
        if extract.success() {
            for f in &files {
                if f == PLACEHOLDER {
                    continue;
                }
                let want = std::fs::read(tmp.join(f));
                let got = std::fs::read(root.join(f));
                match (want, got) {
                    (Ok(w), Ok(g)) if w == g => {}
                    _ => mismatches.push(f.clone()),
                }
            }
            verified = Some(mismatches.is_empty());
        } else {
            mismatches.push("(could not extract the tarball to compare)".to_string());
            verified = Some(false);
        }
        let _ = std::fs::remove_dir_all(&tmp);
    }

    let ok =
        out.success() && errors == 0 && unplaced.is_empty() && (dry_run || verified == Some(true));
    let mut report = Report::new("apply", ok, if ok { EXIT_OK } else { EXIT_FAILED })
        .field("tarball", Value::from(name.as_str()))
        .field("dry_run", Value::Bool(dry_run))
        .field("members", count(files.len()))
        .field("created", Value::Int(created as i64))
        .field("replaced", Value::Int(replaced as i64))
        .field("skipped", Value::Int(skipped as i64))
        .field("errors", Value::Int(errors as i64))
        .field(
            "mdix_exit",
            out.code.map_or(Value::Null, |c| Value::Int(i64::from(c))),
        )
        .field("not_placed", strings(&unplaced))
        .field("verified", verified.map_or(Value::Null, Value::Bool))
        .field("mismatches", strings(&mismatches));
    report = report.line(format!(
        "apply {name}: {} members, {created} created, {replaced} replaced, {errors} errors{}",
        files.len(),
        if dry_run { " (dry run)" } else { "" }
    ));
    if !unplaced.is_empty() {
        report = report.line(format!("  not placed by mdix: {}", unplaced.join(", ")));
    }
    match verified {
        Some(true) => {
            report = report.line("  verified: every member is byte-identical in the repository")
        }
        Some(false) => report = report.line(format!("  NOT identical: {}", mismatches.join(", "))),
        None => {}
    }
    if !out.success() {
        report = report.line(format!(
            "  mdix exited with {:?}: {}",
            out.code,
            out.stderr.trim()
        ));
    }
    report
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_tools_output_is_read() {
        let text = "\
→ Generating project structure...
  REP  ./crates/a/src/lib.rs  (overwritten)
  NEW  ./docs/new.md
  REP  ./.mdix/replacements/gitkeep  (overwritten)
Done. 1 created, 2 replaced, 0 skipped, 0 error(s).
";
        let r = parse_mdix_output(text);
        assert_eq!(
            r.placed,
            vec![
                ("REP".to_string(), "crates/a/src/lib.rs".to_string()),
                ("NEW".to_string(), "docs/new.md".to_string()),
                ("REP".to_string(), ".mdix/replacements/gitkeep".to_string()),
            ]
        );
        assert_eq!(r.done, Some((1, 2, 0, 0)));
        assert_eq!(parse_mdix_output("nothing useful"), MdixResult::default());
    }

    #[test]
    fn members_that_leave_the_repository_are_refused() {
        assert!(unsafe_member("/etc/passwd"));
        assert!(unsafe_member("../x"));
        assert!(unsafe_member("a/../../x"));
        assert!(unsafe_member("a\\b"));
        assert!(unsafe_member(""));
        assert!(!unsafe_member("a/b/c.rs"));
        assert!(!unsafe_member("./a/b.rs"));
        assert!(!unsafe_member("a/..b/c"));
    }
}
