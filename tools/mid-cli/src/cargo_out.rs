// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-cli.md, section "cargo_out.rs"
// ============================================================================
//! Reading what `cargo` prints: test results, and diagnostics in the short
//! message format.

/// One test target's result line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TargetResult {
    /// Which target: `unittests src/lib.rs`, `tests/math.rs`, `doc-tests x`.
    pub label: String,
    /// Tests that passed.
    pub passed: u64,
    /// Tests that failed.
    pub failed: u64,
    /// Tests that were ignored.
    pub ignored: u64,
}

/// Everything read from a `cargo test` run.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct TestParse {
    /// One entry per `test result:` line.
    pub targets: Vec<TargetResult>,
    /// Names of failed tests.
    pub failures: Vec<String>,
    /// Lines that start a compile error.
    pub build_errors: Vec<String>,
}

impl TestParse {
    /// Total passed.
    pub fn passed(&self) -> u64 {
        self.targets.iter().map(|t| t.passed).sum()
    }

    /// Total failed.
    pub fn failed(&self) -> u64 {
        self.targets.iter().map(|t| t.failed).sum()
    }

    /// Total ignored.
    pub fn ignored(&self) -> u64 {
        self.targets.iter().map(|t| t.ignored).sum()
    }
}

/// Reads `cargo test --color=never` output.
///
/// Cargo prints `Running ...` and `Doc-tests ...` on stderr and the results
/// on stdout, and the two reach here joined, one stream after the other. So
/// the labels are collected on their own and given to the results in order:
/// each target has exactly one of each.
pub fn parse_test_output(text: &str) -> TestParse {
    let mut parse = TestParse::default();
    let mut labels: Vec<String> = Vec::new();
    for raw in text.lines() {
        let line = raw.trim_start();
        if let Some(rest) = line.strip_prefix("Running ") {
            labels.push(rest.split(" (").next().unwrap_or(rest).trim().to_string());
        } else if let Some(rest) = line.strip_prefix("Doc-tests ") {
            labels.push(format!("doc-tests {}", rest.trim()));
        } else if let Some(rest) = line.strip_prefix("test result: ") {
            parse.targets.push(TargetResult {
                label: String::new(),
                passed: count_before(rest, "passed"),
                failed: count_before(rest, "failed"),
                ignored: count_before(rest, "ignored"),
            });
        } else if let Some(rest) = line.strip_prefix("test ") {
            if let Some(name) = rest.strip_suffix(" ... FAILED") {
                parse.failures.push(name.to_string());
            }
        } else if is_build_error(line) {
            parse.build_errors.push(line.to_string());
        }
    }
    for (target, label) in parse.targets.iter_mut().zip(labels) {
        target.label = label;
    }
    parse
}

/// A line that starts a compile error, not one of cargo's own summaries of a
/// failure it already reported.
fn is_build_error(line: &str) -> bool {
    if line.starts_with("error[") {
        return true;
    }
    match line.strip_prefix("error: ") {
        Some(rest) => {
            !(rest.starts_with("could not compile")
                || rest.starts_with("test failed")
                || rest.contains("target failed")
                || rest.contains("targets failed"))
        }
        None => false,
    }
}

/// The number written just before `word` in a `test result:` line.
fn count_before(text: &str, word: &str) -> u64 {
    for part in text.split(';') {
        let part = part.trim();
        if let Some(head) = part.strip_suffix(word) {
            let digits: String = head
                .trim_end()
                .chars()
                .rev()
                .take_while(char::is_ascii_digit)
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect();
            if let Ok(n) = digits.parse() {
                return n;
            }
        }
    }
    0
}

/// Severity of a diagnostic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Level {
    /// A warning.
    Warning,
    /// An error.
    Error,
}

/// One diagnostic from `--message-format=short`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Diag {
    /// The file, as cargo printed it.
    pub path: String,
    /// The line, 0 if none.
    pub line: u32,
    /// Warning or error.
    pub level: Level,
    /// The message.
    pub message: String,
}

/// Reads diagnostics that have a location. Summary lines and diagnostics
/// with no file are not included.
pub fn parse_diags(text: &str) -> Vec<Diag> {
    let mut diags = Vec::new();
    for line in text.lines() {
        for (marker, level) in [(": warning", Level::Warning), (": error", Level::Error)] {
            let Some(at) = line.find(marker) else {
                continue;
            };
            let after = &line[at + marker.len()..];
            if !(after.starts_with(": ") || after.starts_with('[')) {
                continue;
            }
            let location = &line[..at];
            let mut parts = location.rsplitn(3, ':');
            let (Some(col), Some(lineno), Some(path)) = (parts.next(), parts.next(), parts.next())
            else {
                continue;
            };
            if col.parse::<u32>().is_err() {
                continue;
            }
            let Ok(lineno) = lineno.parse::<u32>() else {
                continue;
            };
            let message = after
                .trim_start_matches(|c| c != ' ')
                .trim_start()
                .to_string();
            diags.push(Diag {
                path: path.to_string(),
                line: lineno,
                level,
                message,
            });
            break;
        }
    }
    diags
}

/// Counts lines that report an error with no location (`error: could not
/// compile`), which [`parse_diags`] skips.
pub fn count_unlocated_errors(text: &str) -> usize {
    text.lines()
        .filter(|l| l.starts_with("error:") || l.starts_with("error["))
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_output_is_read_per_target() {
        let text = "\
   Compiling x v0.1.0
     Running unittests src/lib.rs (target/debug/deps/x-1)

running 2 tests
test a ... ok
test b ... FAILED

failures:

---- b stdout ----
test result: FAILED. 1 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

     Running tests/it.rs (target/debug/deps/it-2)
test result: ok. 8 passed; 0 failed; 2 ignored; 0 measured; 0 filtered out; finished in 0.50s

   Doc-tests x
test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.10s
";
        let p = parse_test_output(text);
        assert_eq!(p.targets.len(), 3);
        assert_eq!(p.targets[0].label, "unittests src/lib.rs");
        assert_eq!((p.targets[0].passed, p.targets[0].failed), (1, 1));
        assert_eq!(p.targets[1].label, "tests/it.rs");
        assert_eq!((p.targets[1].passed, p.targets[1].ignored), (8, 2));
        assert_eq!(p.targets[2].label, "doc-tests x");
        assert_eq!((p.passed(), p.failed(), p.ignored()), (13, 1, 2));
        assert_eq!(p.failures, vec!["b".to_string()]);
        assert!(p.build_errors.is_empty());
    }

    #[test]
    fn labels_still_match_when_stdout_comes_before_stderr() {
        // What a captured run looks like: all the results, then all the
        // `Running` lines.
        let text = "\
test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
test result: ok. 8 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
   Compiling x v0.1.0
     Running unittests src/lib.rs (target/debug/deps/x-1)
     Running tests/it.rs (target/debug/deps/it-2)
   Doc-tests x
";
        let p = parse_test_output(text);
        let labels: Vec<&str> = p.targets.iter().map(|t| t.label.as_str()).collect();
        assert_eq!(
            labels,
            ["unittests src/lib.rs", "tests/it.rs", "doc-tests x"]
        );
        assert_eq!(
            p.targets.iter().map(|t| t.passed).collect::<Vec<_>>(),
            [3, 8, 4]
        );
    }

    #[test]
    fn a_result_with_no_label_line_keeps_an_empty_label() {
        let p = parse_test_output("test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s\n");
        assert_eq!(p.targets.len(), 1);
        assert_eq!(p.targets[0].label, "");
    }

    #[test]
    fn a_build_error_is_reported_and_the_cargo_summary_lines_are_not() {
        let text = "\
error[E0425]: cannot find value `x` in this scope
 --> src/lib.rs:3:5
error: could not compile `x` (lib) due to 1 previous error
";
        let p = parse_test_output(text);
        assert_eq!(
            p.build_errors,
            vec!["error[E0425]: cannot find value `x` in this scope"]
        );
        assert!(p.targets.is_empty());
    }

    #[test]
    fn short_diagnostics_are_read_and_summaries_skipped() {
        let text = "\
crates/a/src/lib.rs:12:5: warning: unused variable: `x`
crates/b/src/x.rs:3:1: error[E0425]: cannot find value `y`
warning: `a` (lib) generated 1 warning
error: could not compile `b` (lib)
C:\\dir\\f.rs:9:2: warning: odd
";
        let d = parse_diags(text);
        assert_eq!(d.len(), 3);
        assert_eq!(d[0].path, "crates/a/src/lib.rs");
        assert_eq!((d[0].line, d[0].level), (12, Level::Warning));
        assert_eq!(d[0].message, "unused variable: `x`");
        assert_eq!((d[1].line, d[1].level), (3, Level::Error));
        assert_eq!(d[1].message, "cannot find value `y`");
        assert_eq!(d[2].path, "C:\\dir\\f.rs");
        assert_eq!(count_unlocated_errors(text), 1);
    }

    #[test]
    fn counts_are_read_from_anywhere_in_the_result_line() {
        let l = "ok. 12 passed; 3 failed; 4 ignored; 0 measured; 5 filtered out; finished in 1s";
        assert_eq!(count_before(l, "passed"), 12);
        assert_eq!(count_before(l, "failed"), 3);
        assert_eq!(count_before(l, "ignored"), 4);
        assert_eq!(count_before(l, "missing"), 0);
    }
}
