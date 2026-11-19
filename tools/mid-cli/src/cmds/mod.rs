// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-cli.md, section "cmds/mod.rs"
// ============================================================================
//! The commands, and what they share.

use std::path::{Path, PathBuf};

use mid_reflect::Value;

use crate::git;
use crate::proc::Runner;
use crate::report::{Report, EXIT_ENV, EXIT_OK, EXIT_USAGE};
use crate::COMMANDS;

pub mod apply;
pub mod check;
pub mod deliver;
pub mod docs_lint;
pub mod doctor;
pub mod test;

/// `mid help`: the commands. `asked` is what the user typed, if anything.
pub fn help(asked: Option<&str>) -> Report {
    let mut report = Report::new("help", true, EXIT_OK);
    let mut list = Vec::new();
    report = report.line("mid: the Mid Engine command line");
    report = report.line("usage: mid [--json] [--repo DIR] <command> [flags]");
    report = report.line("");
    for (name, summary) in COMMANDS {
        list.push(Value::map([
            ("name", Value::from(*name)),
            ("summary", Value::from(*summary)),
        ]));
        report = report.line(format!("  {name:<10} {summary}"));
    }
    report = report.line("");
    report = report.line(
        "--json prints one JSON document; exit codes: 0 ok, 1 failed, 2 usage, 3 environment",
    );
    if let Some(a) = asked {
        if a != "help"
            && !a.starts_with('-')
            && !COMMANDS.iter().any(|(n, _)| n.split(' ').next() == Some(a))
        {
            report = report.line(format!("(no command called `{a}`)"));
        }
    }
    report.field("commands", Value::List(list))
}

/// A usage error for `command`.
pub fn usage(command: &str, message: &str) -> Report {
    Report::error(command, EXIT_USAGE, "usage", message)
}

/// The repository root for `dir`, or the report to return if there is none.
pub fn repo_root(command: &str, runner: &dyn Runner, dir: &Path) -> Result<PathBuf, Report> {
    git::toplevel(runner, dir).ok_or_else(|| {
        Report::error(
            command,
            EXIT_ENV,
            "not-a-repository",
            &format!("`{}` is not inside a git repository", dir.display()),
        )
    })
}
