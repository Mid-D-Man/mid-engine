// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-cli.md, section "report.rs"
// ============================================================================
//! What a command returns: one structure that renders as JSON for programs
//! and as plain lines for people.

use mid_reflect::Value;

/// Exit code: the command did what was asked.
pub const EXIT_OK: i32 = 0;
/// Exit code: the thing asked failed (tests failed, the lint found problems).
pub const EXIT_FAILED: i32 = 1;
/// Exit code: the command line was wrong.
pub const EXIT_USAGE: i32 = 2;
/// Exit code: the environment is wrong (a tool or a path is missing).
pub const EXIT_ENV: i32 = 3;

/// The version of the JSON layout. Fields are added, never repurposed.
pub const SCHEMA: i64 = 1;

/// A command's result.
#[derive(Debug, Clone)]
pub struct Report {
    /// The command's name.
    pub command: String,
    /// Whether it succeeded.
    pub ok: bool,
    /// The process exit code.
    pub exit: i32,
    /// The fields of the JSON form, after `schema`, `command` and `ok`.
    pub data: Vec<(String, Value)>,
    /// The plain-text form, one line per entry.
    pub human: Vec<String>,
}

impl Report {
    /// A report for a command that ran.
    pub fn new(command: &str, ok: bool, exit: i32) -> Self {
        Self {
            command: command.to_string(),
            ok,
            exit,
            data: Vec::new(),
            human: Vec::new(),
        }
    }

    /// Adds a field to the JSON form.
    pub fn field(mut self, key: &str, value: Value) -> Self {
        self.data.push((key.to_string(), value));
        self
    }

    /// Adds a line to the plain-text form.
    pub fn line(mut self, text: impl Into<String>) -> Self {
        self.human.push(text.into());
        self
    }

    /// A failure that is not about what the command found but about how it
    /// was run: a bad flag, a missing tool.
    pub fn error(command: &str, exit: i32, kind: &str, message: &str) -> Self {
        Report::new(command, false, exit)
            .field(
                "error",
                Value::map([
                    ("kind", Value::from(kind)),
                    ("message", Value::from(message)),
                ]),
            )
            .line(format!("error: {message}"))
    }

    /// The JSON form, on one line.
    pub fn to_json(&self) -> String {
        let mut entries = vec![
            ("schema".to_string(), Value::Int(SCHEMA)),
            ("command".to_string(), Value::from(self.command.as_str())),
            ("ok".to_string(), Value::Bool(self.ok)),
        ];
        entries.extend(self.data.iter().cloned());
        Value::Map(entries).to_json()
    }

    /// The plain-text form.
    pub fn to_text(&self) -> String {
        let mut out = self.human.join("\n");
        if !out.is_empty() {
            out.push('\n');
        }
        out
    }
}

/// A string list as a [`Value`].
pub fn strings<S: AsRef<str>>(items: &[S]) -> Value {
    Value::List(items.iter().map(|s| Value::from(s.as_ref())).collect())
}

/// A count as a [`Value`].
pub fn count(n: usize) -> Value {
    Value::Int(n as i64)
}
