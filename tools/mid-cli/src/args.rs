// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-cli.md, section "args.rs"
// ============================================================================
//! Argument parsing: global flags, then a command's own flags, by hand.

use std::collections::HashMap;

/// What a flag takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Flag {
    /// A switch: present or not.
    Switch,
    /// A single value; the last one given wins.
    Value,
    /// A value that may be given more than once.
    Repeat,
}

/// One flag in a command's spec: long name, kind, optional short letter.
pub type FlagSpec<'a> = (&'a str, Flag, Option<char>);

/// The parsed flags and positional arguments of one command.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Parsed {
    switches: Vec<String>,
    values: HashMap<String, Vec<String>>,
    /// Arguments that are not flags, in order.
    pub positional: Vec<String>,
    /// Everything after a bare `--`, untouched.
    pub passthrough: Vec<String>,
}

impl Parsed {
    /// Whether a switch was given.
    pub fn has(&self, name: &str) -> bool {
        self.switches.iter().any(|s| s == name)
    }

    /// The last value of a flag.
    pub fn value(&self, name: &str) -> Option<&str> {
        self.values
            .get(name)
            .and_then(|v| v.last())
            .map(String::as_str)
    }

    /// Every value of a repeatable flag.
    pub fn values(&self, name: &str) -> &[String] {
        self.values.get(name).map(Vec::as_slice).unwrap_or(&[])
    }
}

/// Parses `args` against `spec`: `(name, kind, short)` where `name` is the
/// long form without dashes. Unknown flags and missing values are errors.
pub fn parse(args: &[String], spec: &[FlagSpec]) -> Result<Parsed, String> {
    let mut parsed = Parsed::default();
    let mut i = 0;
    while i < args.len() {
        let arg = &args[i];
        i += 1;
        if arg == "--" {
            parsed.passthrough = args[i..].to_vec();
            break;
        }
        let (name, inline): (Option<&FlagSpec>, Option<String>) =
            if let Some(long) = arg.strip_prefix("--") {
                match long.split_once('=') {
                    Some((n, v)) => (spec.iter().find(|s| s.0 == n), Some(v.to_string())),
                    None => (spec.iter().find(|s| s.0 == long), None),
                }
            } else if arg.len() == 2 && arg.starts_with('-') {
                let c = arg.chars().nth(1).unwrap_or(' ');
                (spec.iter().find(|s| s.2 == Some(c)), None)
            } else {
                parsed.positional.push(arg.clone());
                continue;
            };
        let Some(&(name, kind, _)) = name else {
            return Err(format!("unknown flag `{arg}`"));
        };
        match kind {
            Flag::Switch => {
                if inline.is_some() {
                    return Err(format!("`--{name}` does not take a value"));
                }
                if !parsed.has(name) {
                    parsed.switches.push(name.to_string());
                }
            }
            Flag::Value | Flag::Repeat => {
                let value = match inline {
                    Some(v) => v,
                    None => {
                        let Some(v) = args.get(i) else {
                            return Err(format!("`--{name}` needs a value"));
                        };
                        i += 1;
                        v.clone()
                    }
                };
                let slot = parsed.values.entry(name.to_string()).or_default();
                if kind == Flag::Value {
                    slot.clear();
                }
                slot.push(value);
            }
        }
    }
    Ok(parsed)
}
