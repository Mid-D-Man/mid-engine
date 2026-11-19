// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-cli.md, section "cmds/check.rs"
// ============================================================================
//! `mid check`: `cargo check`, `rustfmt --check` and `clippy`, per crate,
//! summarised.

use std::path::{Path, PathBuf};
use std::time::Instant;

use mid_reflect::Value;

use crate::args::{self, Flag};
use crate::cargo_out::{count_unlocated_errors, parse_diags, Diag, Level};
use crate::proc::{Cmd, Runner};
use crate::report::{count, Report, EXIT_FAILED, EXIT_OK};
use crate::util::{path_text, write_log};
use crate::{cmds, Env};

const SPEC: &[(&str, Flag, Option<char>)] = &[
    ("package", Flag::Repeat, Some('p')),
    ("all", Flag::Switch, None),
    ("features", Flag::Value, Some('F')),
    ("target", Flag::Value, None),
    ("no-fmt", Flag::Switch, None),
    ("no-clippy", Flag::Switch, None),
    ("deny-warnings", Flag::Switch, None),
    ("limit", Flag::Value, None),
];

/// Runs `mid check`.
pub fn run(args: &[String], runner: &dyn Runner, env: &Env, dir: &Path) -> Report {
    let parsed = match args::parse(args, SPEC) {
        Ok(p) => p,
        Err(e) => return cmds::usage("check", &e),
    };
    let packages: Vec<String> = parsed.values("package").to_vec();
    if packages.is_empty() && !parsed.has("all") {
        return cmds::usage("check", "give `-p <crate>` (repeatable) or `--all`");
    }
    let limit: usize = match parsed.value("limit").map(str::parse) {
        None => 10,
        Some(Ok(n)) => n,
        Some(Err(_)) => return cmds::usage("check", "`--limit` needs a number"),
    };
    let root = match cmds::repo_root("check", runner, dir) {
        Ok(r) => r,
        Err(report) => return report,
    };
    let mut targets: Vec<Option<String>> = packages.iter().cloned().map(Some).collect();
    if parsed.has("all") {
        targets = vec![None];
    }
    let features = parsed.value("features").map(str::to_string);
    let target = parsed.value("target").map(str::to_string);

    let mut steps = Vec::new();
    let mut lines = Vec::new();
    let mut all_ok = true;
    for package in &targets {
        let label = package.clone().unwrap_or_else(|| "workspace".to_string());
        let pkg_dir = package.as_ref().and_then(|p| package_dir(runner, dir, p));
        if package.is_some() && pkg_dir.is_none() {
            lines.push(format!(
                "note: could not find the directory of `{label}`; every diagnostic is counted against it"
            ));
        }
        let scope: Vec<String> = match package {
            Some(p) => vec!["-p".to_string(), p.clone()],
            None => vec!["--workspace".to_string()],
        };
        let mut extra: Vec<String> = Vec::new();
        if let Some(f) = &features {
            extra.push("--features".to_string());
            extra.push(f.clone());
        }
        if let Some(t) = &target {
            extra.push("--target".to_string());
            extra.push(t.clone());
        }

        let mut build_steps: Vec<(&str, Vec<String>)> = Vec::new();
        let mut check_args = vec!["check".to_string()];
        check_args.extend(scope.clone());
        check_args.extend(["--all-targets".to_string()]);
        check_args.extend(extra.clone());
        check_args.extend([
            "--message-format=short".to_string(),
            "--color=never".to_string(),
        ]);
        build_steps.push(("check", check_args));
        if !parsed.has("no-fmt") {
            let mut fmt_args = vec!["fmt".to_string()];
            fmt_args.extend(scope.clone());
            fmt_args.push("--check".to_string());
            build_steps.push(("fmt", fmt_args));
        }
        if !parsed.has("no-clippy") {
            let mut clippy_args = vec!["clippy".to_string()];
            clippy_args.extend(scope.clone());
            clippy_args.extend(["--all-targets".to_string(), "--no-deps".to_string()]);
            clippy_args.extend(extra.clone());
            clippy_args.extend([
                "--message-format=short".to_string(),
                "--color=never".to_string(),
            ]);
            build_steps.push(("clippy", clippy_args));
        }

        for (name, cargo_args) in build_steps {
            let started = Instant::now();
            let cmd = Cmd::new("cargo", &cargo_args).in_dir(dir);
            let out = runner.run(&cmd);
            let seconds = started.elapsed().as_secs_f64();
            let text = out.combined();
            let log = write_log(
                &env.temp,
                &format!("check-{label}-{name}"),
                &format!("$ {}\n{text}", cmd.display()),
            );
            let (ok, ours_w, ours_e, other_w, findings) = if name == "fmt" {
                let diffs = text.lines().filter(|l| l.starts_with("Diff in ")).count();
                let ok = out.success();
                let findings: Vec<Value> = text
                    .lines()
                    .filter(|l| l.starts_with("Diff in "))
                    .take(limit)
                    .map(|l| Value::map([("message", Value::from(l))]))
                    .collect();
                (ok, diffs, 0, 0, findings)
            } else {
                let diags = parse_diags(&text);
                let (ours, others): (Vec<&Diag>, Vec<&Diag>) = diags
                    .iter()
                    .partition(|d| is_ours(d, &root, pkg_dir.as_deref()));
                let w = ours.iter().filter(|d| d.level == Level::Warning).count();
                let e = ours.iter().filter(|d| d.level == Level::Error).count()
                    + if out.success() {
                        0
                    } else {
                        count_unlocated_errors(&text)
                    };
                let ow = others.iter().filter(|d| d.level == Level::Warning).count();
                let ok = out.success() && e == 0 && out.started();
                let findings: Vec<Value> = ours
                    .iter()
                    .take(limit)
                    .map(|d| {
                        Value::map([
                            ("path", Value::from(d.path.as_str())),
                            ("line", Value::Int(i64::from(d.line))),
                            (
                                "level",
                                Value::from(if d.level == Level::Error {
                                    "error"
                                } else {
                                    "warning"
                                }),
                            ),
                            ("message", Value::from(d.message.as_str())),
                        ])
                    })
                    .collect();
                (ok, w, e, ow, findings)
            };
            let denied = parsed.has("deny-warnings") && name != "fmt" && ours_w > 0;
            let step_ok = ok && !denied;
            all_ok &= step_ok;
            let status = if !out.started() {
                format!(
                    "could not run cargo: {}",
                    out.spawn_error.clone().unwrap_or_default()
                )
            } else if name == "fmt" {
                if ok {
                    "ok".to_string()
                } else {
                    format!("{ours_w} files differ")
                }
            } else {
                format!(
                    "{} {} warnings, {} errors",
                    if step_ok { "ok" } else { "FAILED" },
                    ours_w,
                    ours_e
                )
            };
            lines.push(format!("{name:<6} {label}: {status} ({seconds:.0}s)"));
            for f in &findings {
                let at = f.get("path").and_then(Value::as_str).unwrap_or("");
                let msg = f.get("message").and_then(Value::as_str).unwrap_or("");
                lines.push(format!("         {at} {msg}"));
            }
            steps.push(Value::map([
                ("step", Value::from(name)),
                ("package", Value::from(label.as_str())),
                ("ok", Value::Bool(step_ok)),
                ("warnings", count(ours_w)),
                ("errors", count(ours_e)),
                ("other_warnings", count(other_w)),
                ("seconds", Value::Float((seconds * 10.0).round() / 10.0)),
                ("findings", Value::List(findings)),
                (
                    "log",
                    log.map_or(Value::Null, |p| Value::from(path_text(&p))),
                ),
            ]));
        }
    }
    let mut report = Report::new("check", all_ok, if all_ok { EXIT_OK } else { EXIT_FAILED })
        .field("steps", Value::List(steps));
    for l in lines {
        report = report.line(l);
    }
    report
}

/// Whether a diagnostic belongs to the package being checked. With no
/// directory known (the whole workspace) everything does.
fn is_ours(diag: &Diag, root: &Path, pkg_dir: Option<&Path>) -> bool {
    let Some(pkg_dir) = pkg_dir else {
        return true;
    };
    let path = Path::new(&diag.path);
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        root.join(path)
    };
    absolute.starts_with(pkg_dir)
}

/// The directory of a workspace package, from `cargo metadata --no-deps`,
/// which needs no lockfile (`cargo pkgid` fails without one).
fn package_dir(runner: &dyn Runner, dir: &Path, package: &str) -> Option<PathBuf> {
    let out = runner
        .run(&Cmd::new("cargo", &["metadata", "--no-deps", "--format-version", "1"]).in_dir(dir));
    if !out.success() {
        return None;
    }
    package_dir_from_metadata(&out.stdout, package)
}

/// The directory of the manifest of `package` in `cargo metadata` output.
pub fn package_dir_from_metadata(json: &str, package: &str) -> Option<PathBuf> {
    let metadata = Value::from_json(json.trim()).ok()?;
    metadata
        .get("packages")?
        .as_list()?
        .iter()
        .find(|p| p.get("name").and_then(Value::as_str) == Some(package))
        .and_then(|p| p.get("manifest_path")?.as_str())
        .and_then(|manifest| Path::new(manifest).parent())
        .map(Path::to_path_buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metadata_names_the_directory_of_a_package() {
        let json = r#"{"packages":[
            {"name":"a","manifest_path":"/repo/crates/a/Cargo.toml"},
            {"name":"mid-x","manifest_path":"/repo/crates/mid-x/Cargo.toml","version":"0.1.0"}
        ],"workspace_root":"/repo","metadata":null}"#;
        assert_eq!(
            package_dir_from_metadata(json, "mid-x"),
            Some(PathBuf::from("/repo/crates/mid-x"))
        );
        assert_eq!(package_dir_from_metadata(json, "b"), None);
        assert_eq!(package_dir_from_metadata("not json", "a"), None);
        assert_eq!(package_dir_from_metadata("{}", "a"), None);
        assert_eq!(package_dir_from_metadata("", "a"), None);
    }

    #[test]
    fn a_diagnostic_is_ours_if_it_is_under_the_package() {
        let root = Path::new("/repo");
        let d = |p: &str| Diag {
            path: p.to_string(),
            line: 1,
            level: Level::Warning,
            message: String::new(),
        };
        let pkg = Path::new("/repo/crates/a");
        assert!(is_ours(&d("crates/a/src/lib.rs"), root, Some(pkg)));
        assert!(is_ours(&d("/repo/crates/a/src/lib.rs"), root, Some(pkg)));
        assert!(!is_ours(&d("crates/b/src/lib.rs"), root, Some(pkg)));
        assert!(!is_ours(&d("crates/ab/src/lib.rs"), root, Some(pkg)));
        assert!(is_ours(&d("anything"), root, None));
    }
}
