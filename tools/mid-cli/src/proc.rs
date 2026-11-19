// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-cli.md, section "proc.rs"
// ============================================================================
//! Running other programs, behind a trait so tests can stand in for them.

use std::path::{Path, PathBuf};
use std::process::Command;

/// A program to run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cmd {
    /// The program, found on `PATH` unless it has a directory part.
    pub program: String,
    /// Its arguments.
    pub args: Vec<String>,
    /// The directory to run it in.
    pub cwd: Option<PathBuf>,
}

impl Cmd {
    /// A command with arguments.
    pub fn new<S: AsRef<str>>(program: &str, args: &[S]) -> Self {
        Self {
            program: program.to_string(),
            args: args.iter().map(|a| a.as_ref().to_string()).collect(),
            cwd: None,
        }
    }

    /// The same command, run in `dir`.
    pub fn in_dir(mut self, dir: &Path) -> Self {
        self.cwd = Some(dir.to_path_buf());
        self
    }

    /// The command as one line, for logs.
    pub fn display(&self) -> String {
        let mut parts = vec![self.program.clone()];
        parts.extend(self.args.iter().cloned());
        parts.join(" ")
    }
}

/// What a run produced.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Outcome {
    /// The exit code; `None` if the program was not started or was killed.
    pub code: Option<i32>,
    /// Standard output.
    pub stdout: String,
    /// Standard error.
    pub stderr: String,
    /// Why the program could not be started, if it could not.
    pub spawn_error: Option<String>,
}

impl Outcome {
    /// Whether it ran and exited 0.
    pub fn success(&self) -> bool {
        self.code == Some(0)
    }

    /// Whether the program was found and started.
    pub fn started(&self) -> bool {
        self.spawn_error.is_none()
    }

    /// Standard output then standard error.
    pub fn combined(&self) -> String {
        format!("{}{}", self.stdout, self.stderr)
    }
}

/// Something that can run a [`Cmd`].
pub trait Runner {
    /// Runs `cmd` to completion.
    fn run(&self, cmd: &Cmd) -> Outcome;
}

/// Runs real programs.
pub struct SystemRunner;

impl Runner for SystemRunner {
    fn run(&self, cmd: &Cmd) -> Outcome {
        let mut command = Command::new(&cmd.program);
        command.args(&cmd.args);
        if let Some(dir) = &cmd.cwd {
            command.current_dir(dir);
        }
        match command.output() {
            Ok(out) => Outcome {
                code: out.status.code(),
                stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
                stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
                spawn_error: None,
            },
            Err(e) => Outcome {
                spawn_error: Some(e.to_string()),
                ..Outcome::default()
            },
        }
    }
}
