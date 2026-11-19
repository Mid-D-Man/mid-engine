// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-cli.md, section "lib.rs"
// ============================================================================
//! The `mid` command line: development-workflow commands with output a
//! program can rely on. See `docs/mid-cli.md`.
//!
//! [`run`] takes the arguments, something that can run other programs, and a
//! description of the environment, and returns what to print and the exit
//! code. Nothing here reads the process environment or prints, so every
//! command can be tested.

pub mod args;
pub mod cargo_out;
pub mod cmds;
pub mod git;
pub mod proc;
pub mod report;
pub mod util;

pub use proc::{Cmd, Outcome, Runner, SystemRunner};
pub use report::Report;

use std::path::PathBuf;

use report::{EXIT_OK, EXIT_USAGE};

/// The parts of the process environment the commands read.
#[derive(Debug, Clone)]
pub struct Env {
    /// The directory commands run in unless `--repo` says otherwise.
    pub cwd: PathBuf,
    /// Where the `mdix-scaffold` tool is (`MID_MDIX`), if known.
    pub mdix: Option<String>,
    /// A directory for logs and saved state.
    pub temp: PathBuf,
}

impl Env {
    /// Reads the real environment.
    pub fn from_process() -> Self {
        Self {
            cwd: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
            mdix: std::env::var("MID_MDIX").ok().filter(|v| !v.is_empty()),
            temp: std::env::temp_dir(),
        }
    }
}

/// What a run produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunResult {
    /// The exit code.
    pub code: i32,
    /// What to write to standard output.
    pub stdout: String,
    /// What to write to standard error.
    pub stderr: String,
}

/// The commands, with one line each. Order is the order `help` prints.
pub const COMMANDS: &[(&str, &str)] = &[
    ("doctor", "Report the toolchain and the repository; exit 3 if something required is missing"),
    ("check", "cargo check, rustfmt --check and clippy for the crates given, summarised"),
    ("test", "Run the tests for the crates given and summarise counts, failures and the change since last run"),
    ("docs lint", "Check the documentation standard on changed files: NOTICE headers, doc sections, writing style"),
    ("apply", "Apply a delivery tarball with the mdix tool and verify every file is byte-identical"),
    ("deliver", "Build a delivery tarball from the files that differ from a base"),
    ("help", "List the commands"),
];

/// The global flags and the rest of the arguments: json, repo, help, version, rest.
type GlobalSplit = (bool, Option<String>, bool, bool, Vec<String>);

/// Splits the global flags (`--json`, `--repo DIR`, `--help`, `--version`)
/// out of `args`.
fn split_global(args: &[String]) -> Result<GlobalSplit, String> {
    let (mut json, mut repo, mut help, mut version) = (false, None, false, false);
    let mut rest = Vec::new();
    let mut i = 0;
    let mut passthrough = false;
    while i < args.len() {
        let a = &args[i];
        i += 1;
        if passthrough {
            rest.push(a.clone());
            continue;
        }
        match a.as_str() {
            "--" => {
                passthrough = true;
                rest.push(a.clone());
            }
            "--json" => json = true,
            "--help" | "-h" => help = true,
            "--version" | "-V" => version = true,
            "--repo" => {
                let Some(v) = args.get(i) else {
                    return Err("`--repo` needs a value".to_string());
                };
                repo = Some(v.clone());
                i += 1;
            }
            _ => match a.strip_prefix("--repo=") {
                Some(v) => repo = Some(v.to_string()),
                None => rest.push(a.clone()),
            },
        }
    }
    Ok((json, repo, help, version, rest))
}

/// Runs one `mid` invocation.
pub fn run(args: &[String], runner: &dyn Runner, env: &Env) -> RunResult {
    let (json, repo, help, version, rest) = match split_global(args) {
        Ok(parts) => parts,
        Err(message) => {
            return render(false, Report::error("mid", EXIT_USAGE, "usage", &message));
        }
    };
    if version {
        let report = Report::new("version", true, EXIT_OK)
            .field(
                "version",
                mid_reflect::Value::from(env!("CARGO_PKG_VERSION")),
            )
            .line(format!("mid {}", env!("CARGO_PKG_VERSION")));
        return render(json, report);
    }
    let repo_dir = match repo {
        Some(r) => {
            let p = PathBuf::from(r);
            if p.is_absolute() {
                p
            } else {
                env.cwd.join(p)
            }
        }
        None => env.cwd.clone(),
    };
    let Some(command) = rest.first().cloned() else {
        return render(json, cmds::help(None));
    };
    if help || command == "help" {
        return render(json, cmds::help(Some(&command)));
    }
    let tail = &rest[1..];
    let report = match command.as_str() {
        "doctor" => cmds::doctor::run(tail, runner, env, &repo_dir),
        "check" => cmds::check::run(tail, runner, env, &repo_dir),
        "test" => cmds::test::run(tail, runner, env, &repo_dir),
        "docs" => match tail.first().map(String::as_str) {
            Some("lint") => cmds::docs_lint::run(&tail[1..], runner, env, &repo_dir),
            _ => Report::error(
                "docs",
                EXIT_USAGE,
                "usage",
                "`mid docs` needs a subcommand: `lint`",
            ),
        },
        "apply" => cmds::apply::run(tail, runner, env, &repo_dir),
        "deliver" => cmds::deliver::run(tail, runner, env, &repo_dir),
        other => Report::error(
            "mid",
            EXIT_USAGE,
            "usage",
            &format!("unknown command `{other}`; try `mid help`"),
        ),
    };
    render(json, report)
}

fn render(json: bool, report: Report) -> RunResult {
    if json {
        return RunResult {
            code: report.exit,
            stdout: format!("{}\n", report.to_json()),
            stderr: String::new(),
        };
    }
    let text = report.to_text();
    if report.exit == EXIT_OK || report.exit == report::EXIT_FAILED {
        RunResult {
            code: report.exit,
            stdout: text,
            stderr: String::new(),
        }
    } else {
        RunResult {
            code: report.exit,
            stdout: String::new(),
            stderr: text,
        }
    }
}
