// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-cli.md, section "cmds/doctor.rs"
// ============================================================================
//! `mid doctor`: what is installed, and what state the repository is in.

use std::path::Path;

use mid_reflect::Value;

use crate::args;
use crate::git;
use crate::proc::{Cmd, Runner};
use crate::report::{count, Report, EXIT_ENV, EXIT_OK};
use crate::{cmds, Env};

/// Runs `mid doctor`.
pub fn run(args: &[String], runner: &dyn Runner, env: &Env, dir: &Path) -> Report {
    if let Err(e) = args::parse(args, &[]) {
        return cmds::usage("doctor", &e);
    }
    // (name, required, program, args)
    let probes: [(&str, bool, &str, &[&str]); 7] = [
        ("rustc", true, "rustc", &["--version"]),
        ("cargo", true, "cargo", &["--version"]),
        ("git", true, "git", &["--version"]),
        ("rustfmt", false, "rustfmt", &["--version"]),
        ("clippy", false, "cargo", &["clippy", "--version"]),
        ("node", false, "node", &["--version"]),
        ("tar", false, "tar", &["--version"]),
    ];
    let mut tools = Vec::new();
    let mut report_lines = Vec::new();
    let mut missing_required = Vec::new();
    for (name, required, program, probe_args) in probes {
        let out = runner.run(&Cmd::new(program, probe_args).in_dir(dir));
        let found = out.started() && out.success();
        let version = if found {
            out.stdout.lines().next().map(|l| l.trim().to_string())
        } else {
            None
        };
        if required && !found {
            missing_required.push(name);
        }
        report_lines.push(format!(
            "{name:<8} {}{}",
            version.clone().unwrap_or_else(|| "not found".to_string()),
            if required { "" } else { " (optional)" }
        ));
        tools.push(Value::map([
            ("name", Value::from(name)),
            ("required", Value::Bool(required)),
            ("found", Value::Bool(found)),
            ("version", version.map_or(Value::Null, Value::from)),
        ]));
    }

    let wasm = wasm32_installed(runner, dir);
    report_lines.push(format!(
        "wasm32   {}",
        if wasm {
            "target installed"
        } else {
            "target not installed (optional)"
        }
    ));
    tools.push(Value::map([
        ("name", Value::from("wasm32-unknown-unknown")),
        ("required", Value::Bool(false)),
        ("found", Value::Bool(wasm)),
        ("version", Value::Null),
    ]));

    let mdix = mdix_state(env);
    report_lines.push(format!("mdix     {}", mdix.1));
    tools.push(Value::map([
        ("name", Value::from("mdix")),
        ("required", Value::Bool(false)),
        ("found", Value::Bool(mdix.0)),
        ("version", Value::Null),
    ]));

    let (repo_value, repo_lines) = repo_state(runner, dir);
    report_lines.extend(repo_lines);

    let ok = missing_required.is_empty();
    let mut report = Report::new("doctor", ok, if ok { EXIT_OK } else { EXIT_ENV })
        .field("tools", Value::List(tools))
        .field("repo", repo_value);
    if !ok {
        report = report.field("missing", crate::report::strings(&missing_required));
        report_lines.push(format!(
            "missing required tools: {}",
            missing_required.join(", ")
        ));
    }
    for line in report_lines {
        report = report.line(line);
    }
    report
}

fn wasm32_installed(runner: &dyn Runner, dir: &Path) -> bool {
    let out = runner.run(&Cmd::new("rustc", &["--print", "sysroot"]).in_dir(dir));
    if !out.success() {
        return false;
    }
    match out.stdout.lines().next() {
        Some(root) => Path::new(root.trim())
            .join("lib")
            .join("rustlib")
            .join("wasm32-unknown-unknown")
            .is_dir(),
        None => false,
    }
}

/// Whether the mdix tool is where `MID_MDIX` says, and a line saying so.
fn mdix_state(env: &Env) -> (bool, String) {
    match &env.mdix {
        None => (false, "MID_MDIX is not set (optional)".to_string()),
        Some(path) => {
            let script = Path::new(path).join("bin").join("mdix-scaffold.js");
            if script.is_file() {
                (true, format!("found at {path}"))
            } else {
                (
                    false,
                    format!("MID_MDIX={path} has no bin/mdix-scaffold.js"),
                )
            }
        }
    }
}

fn repo_state(runner: &dyn Runner, dir: &Path) -> (Value, Vec<String>) {
    let Some(root) = git::toplevel(runner, dir) else {
        return (
            Value::Null,
            vec!["repo     not inside a git repository".to_string()],
        );
    };
    let git_out = |args: &[&str]| -> Option<String> {
        let out = runner.run(&Cmd::new("git", args).in_dir(&root));
        if out.success() {
            out.stdout.lines().next().map(|l| l.trim().to_string())
        } else {
            None
        }
    };
    let head = git_out(&["rev-parse", "--short", "HEAD"]);
    let branch = git_out(&["rev-parse", "--abbrev-ref", "HEAD"]);
    let dirty = {
        let out = runner.run(&Cmd::new("git", &["status", "--porcelain"]).in_dir(&root));
        if out.success() {
            out.stdout.lines().filter(|l| !l.trim().is_empty()).count()
        } else {
            0
        }
    };
    let has_origin_main = git::ref_exists(runner, &root, "origin/main");
    let line = format!(
        "repo     {} at {} on {}, {} changed paths, origin/main {}",
        root.display(),
        head.clone().unwrap_or_else(|| "?".to_string()),
        branch.clone().unwrap_or_else(|| "?".to_string()),
        dirty,
        if has_origin_main { "present" } else { "absent" },
    );
    let value = Value::map([
        ("root", Value::from(crate::util::path_text(&root))),
        ("head", head.map_or(Value::Null, Value::from)),
        ("branch", branch.map_or(Value::Null, Value::from)),
        ("dirty_paths", count(dirty)),
        ("has_origin_main", Value::Bool(has_origin_main)),
    ]);
    (value, vec![line])
}
