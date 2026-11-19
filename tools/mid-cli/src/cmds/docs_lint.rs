// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-cli.md, section "cmds/docs_lint.rs"
// ============================================================================
//! `mid docs lint`: the documentation standard
//! (`docs/DOCUMENTATION_AND_COMMENTING_GUIDELINES.md`) checked on the files
//! that changed.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use mid_reflect::Value;

use crate::args::{self, Flag};
use crate::git::{self, AddedLine};
use crate::proc::Runner;
use crate::report::{count, Report, EXIT_FAILED, EXIT_OK};
use crate::{cmds, Env};

const SPEC: &[(&str, Flag, Option<char>)] =
    &[("base", Flag::Value, None), ("all", Flag::Switch, None)];

/// A line carrying this text is not linted.
pub const ALLOW_MARKER: &str = "mid-lint: allow";

/// Filler words the writing style rules out.
const BANNED: &[&str] = &[
    "leverage",
    "leverages",
    "leveraged",
    "leveraging",
    "utilize",
    "utilizes",
    "utilized",
    "utilizing",
    "utilization",
    "seamless",
    "seamlessly",
];

/// One problem found.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Finding {
    /// The rule: `em-dash`, `banned-word`, `notice-header`, `doc-section`.
    pub rule: &'static str,
    /// The file, relative to the repository root.
    pub file: String,
    /// The line, 0 if the problem is about the whole file.
    pub line: u64,
    /// What is wrong.
    pub message: String,
}

/// Runs `mid docs lint`.
pub fn run(args: &[String], runner: &dyn Runner, _env: &Env, dir: &Path) -> Report {
    let parsed = match args::parse(args, SPEC) {
        Ok(p) => p,
        Err(e) => return cmds::usage("docs lint", &e),
    };
    let root = match cmds::repo_root("docs lint", runner, dir) {
        Ok(r) => r,
        Err(report) => return report,
    };
    let all = parsed.has("all");
    let asked_base = parsed.value("base").unwrap_or("origin/main").to_string();
    let base = if git::ref_exists(runner, &root, &asked_base) {
        asked_base.clone()
    } else {
        "HEAD".to_string()
    };

    let mut findings = Vec::new();
    let mut md_files = 0usize;
    let mut md_lines = 0usize;
    let mut rs_files = 0usize;

    let (md_targets, rs_targets, added) = if all {
        let mut md = Vec::new();
        walk(&root.join("docs"), &root, &mut md, |p| p.ends_with(".md"));
        let mut rs = Vec::new();
        for top in ["crates", "tools"] {
            if let Ok(entries) = std::fs::read_dir(root.join(top)) {
                for entry in entries.flatten() {
                    walk(&entry.path().join("src"), &root, &mut rs, |p| {
                        p.ends_with(".rs")
                    });
                }
            }
        }
        (md, rs, BTreeMap::new())
    } else {
        let changes = match git::changes_since(runner, &root, &base) {
            Ok(c) => c,
            Err(e) => {
                return Report::error("docs lint", crate::report::EXIT_ENV, "git", &e);
            }
        };
        let tracked_md: Vec<String> = changes
            .changed
            .iter()
            .filter(|p| p.ends_with(".md"))
            .cloned()
            .collect();
        let untracked_md: Vec<String> = changes
            .untracked
            .iter()
            .filter(|p| p.ends_with(".md"))
            .cloned()
            .collect();
        let added_all = match git::added_lines(runner, &root, &base, "*.md") {
            Ok(a) => a,
            Err(e) => return Report::error("docs lint", crate::report::EXIT_ENV, "git", &e),
        };
        let mut by_file: BTreeMap<String, Vec<AddedLine>> = BTreeMap::new();
        for line in added_all {
            by_file.entry(line.path.clone()).or_default().push(line);
        }
        // Untracked files are read whole, so they have no diff entry.
        for f in &untracked_md {
            let lines = read_lines(&root.join(f))
                .into_iter()
                .enumerate()
                .map(|(i, text)| AddedLine {
                    path: f.clone(),
                    line: i as u64 + 1,
                    text,
                })
                .collect();
            by_file.insert(f.clone(), lines);
        }
        let mut md: Vec<String> = tracked_md;
        md.extend(untracked_md);
        md.sort();
        md.dedup();
        let mut rs: Vec<String> = changes
            .changed
            .iter()
            .chain(changes.untracked.iter())
            .filter(|p| is_lintable_source(p))
            .cloned()
            .collect();
        rs.sort();
        rs.dedup();
        (md, rs, by_file)
    };

    for file in &md_targets {
        md_files += 1;
        let lines: Vec<AddedLine> = if all {
            read_lines(&root.join(file))
                .into_iter()
                .enumerate()
                .map(|(i, text)| AddedLine {
                    path: file.clone(),
                    line: i as u64 + 1,
                    text,
                })
                .collect()
        } else {
            added.get(file).cloned().unwrap_or_default()
        };
        md_lines += lines.len();
        findings.extend(lint_markdown_lines(&lines));
    }
    for file in &rs_targets {
        rs_files += 1;
        findings.extend(lint_source_file(&root, file));
    }

    let mut by_rule: BTreeMap<&str, usize> = BTreeMap::new();
    for f in &findings {
        *by_rule.entry(f.rule).or_default() += 1;
    }
    let ok = findings.is_empty();
    let mut report = Report::new("docs lint", ok, if ok { EXIT_OK } else { EXIT_FAILED })
        .field("base", Value::from(asked_base.as_str()))
        .field(
            "base_used",
            Value::from(if all { "(all files)" } else { base.as_str() }),
        )
        .field(
            "checked",
            Value::map([
                ("md_files", count(md_files)),
                ("md_lines", count(md_lines)),
                ("rs_files", count(rs_files)),
            ]),
        )
        .field(
            "counts",
            Value::map(by_rule.iter().map(|(k, v)| (*k, count(*v)))),
        )
        .field(
            "findings",
            Value::List(
                findings
                    .iter()
                    .map(|f| {
                        Value::map([
                            ("rule", Value::from(f.rule)),
                            ("file", Value::from(f.file.as_str())),
                            ("line", Value::Int(f.line as i64)),
                            ("message", Value::from(f.message.as_str())),
                        ])
                    })
                    .collect(),
            ),
        );
    if !all && base != asked_base {
        report = report.line(format!(
            "note: `{asked_base}` does not exist here, compared with HEAD (uncommitted changes only)"
        ));
    }
    report = report.line(format!(
        "docs lint: {} md files ({} lines), {} source files checked, {} problems",
        md_files,
        md_lines,
        rs_files,
        findings.len()
    ));
    for f in &findings {
        if f.line > 0 {
            report = report.line(format!(
                "  [{}] {}:{}: {}",
                f.rule, f.file, f.line, f.message
            ));
        } else {
            report = report.line(format!("  [{}] {}: {}", f.rule, f.file, f.message));
        }
    }
    report
}

fn read_lines(path: &Path) -> Vec<String> {
    std::fs::read_to_string(path)
        .map(|t| t.lines().map(str::to_string).collect())
        .unwrap_or_default()
}

fn walk(dir: &Path, root: &Path, out: &mut Vec<String>, keep: impl Fn(&str) -> bool + Copy) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            walk(&path, root, out, keep);
        } else if let Ok(rel) = path.strip_prefix(root) {
            let text = rel.to_string_lossy().replace('\\', "/");
            if keep(&text) {
                out.push(text);
            }
        }
    }
    out.sort();
}

/// Source files the header and doc-section rules apply to:
/// `crates/<name>/src/**.rs` and `tools/<name>/src/**.rs`.
pub fn is_lintable_source(path: &str) -> bool {
    let parts: Vec<&str> = path.split('/').collect();
    parts.len() >= 4
        && (parts[0] == "crates" || parts[0] == "tools")
        && parts[2] == "src"
        && path.ends_with(".rs")
}

/// The writing-style rules, on added lines of Markdown.
pub fn lint_markdown_lines(lines: &[AddedLine]) -> Vec<Finding> {
    let mut findings = Vec::new();
    for l in lines {
        if l.text.contains(ALLOW_MARKER) {
            continue;
        }
        if l.text.contains('\u{2014}') {
            findings.push(Finding {
                rule: "em-dash",
                file: l.path.clone(),
                line: l.line,
                message: "em dash; use a comma, a colon, parentheses or a new sentence".to_string(),
            });
        }
        let lower = l.text.to_lowercase();
        for word in BANNED {
            if let Some(at) = find_word(&lower, word) {
                let _ = at;
                findings.push(Finding {
                    rule: "banned-word",
                    file: l.path.clone(),
                    line: l.line,
                    message: format!("`{word}` is filler; say what actually happens"),
                });
                break;
            }
        }
    }
    findings
}

/// Finds `word` where it starts a word and is not part of a longer one.
fn find_word(haystack: &str, word: &str) -> Option<usize> {
    let mut from = 0;
    while let Some(i) = haystack[from..].find(word) {
        let at = from + i;
        let before_ok = !haystack[..at]
            .chars()
            .next_back()
            .is_some_and(char::is_alphanumeric);
        let after_ok = !haystack[at + word.len()..]
            .chars()
            .next()
            .is_some_and(char::is_alphanumeric);
        if before_ok && after_ok {
            return Some(at);
        }
        from = at + word.len();
    }
    None
}

/// The NOTICE header and doc-section rules on one source file.
pub fn lint_source_file(root: &Path, file: &str) -> Vec<Finding> {
    let mut findings = Vec::new();
    let text = match std::fs::read_to_string(root.join(file)) {
        Ok(t) => t,
        Err(_) => return findings,
    };
    let head: Vec<&str> = text.lines().take(12).collect();
    match notice_targets(&head) {
        None => findings.push(Finding {
            rule: "notice-header",
            file: file.to_string(),
            line: 1,
            message: "no NOTICE header in the first lines".to_string(),
        }),
        Some(docs) if docs.is_empty() => findings.push(Finding {
            rule: "notice-header",
            file: file.to_string(),
            line: 1,
            message: "the NOTICE header does not name a docs/*.md file".to_string(),
        }),
        Some(docs) => {
            for d in docs {
                if !root.join(&d).is_file() {
                    findings.push(Finding {
                        rule: "notice-header",
                        file: file.to_string(),
                        line: 1,
                        message: format!("the NOTICE header names `{d}`, which does not exist"),
                    });
                }
            }
        }
    }
    findings.extend(doc_section(root, file));
    findings
}

/// The `docs/...md` paths named in a NOTICE header, or `None` if there is no
/// header.
pub fn notice_targets(head: &[&str]) -> Option<Vec<String>> {
    let start = head.iter().position(|l| l.contains("NOTICE:"))?;
    let block: String = head[start..]
        .iter()
        .take(5)
        .copied()
        .collect::<Vec<_>>()
        .join(" ");
    let mut out = Vec::new();
    let mut rest = block.as_str();
    while let Some(i) = rest.find("docs/") {
        let tail = &rest[i..];
        let end = tail
            .find(|c: char| !(c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '/' | '.')))
            .unwrap_or(tail.len());
        let candidate = tail[..end].trim_end_matches('.');
        if candidate.ends_with(".md") {
            out.push(candidate.to_string());
        }
        rest = &tail[end..];
    }
    Some(out)
}

/// Whether the crate's doc has a heading for this file.
fn doc_section(root: &Path, file: &str) -> Vec<Finding> {
    let parts: Vec<&str> = file.split('/').collect();
    let (crate_name, rel) = (parts[1], parts[3..].join("/"));
    let mut docs: Vec<PathBuf> = Vec::new();
    if let Ok(entries) = std::fs::read_dir(root.join("docs")) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            if name == format!("{crate_name}.md")
                || (name.starts_with(&format!("{crate_name}-")) && name.ends_with(".md"))
            {
                docs.push(entry.path());
            }
        }
    }
    if docs.is_empty() {
        return vec![Finding {
            rule: "doc-section",
            file: file.to_string(),
            line: 0,
            message: format!("crate `{crate_name}` has no docs/{crate_name}.md"),
        }];
    }
    let basename = rel.rsplit('/').next().unwrap_or(&rel).to_string();
    let found = docs.iter().any(|d| {
        std::fs::read_to_string(d)
            .map(|t| {
                t.lines().any(|l| {
                    l.starts_with('#')
                        && (l.contains(&rel) || (!rel.contains('/') && l.contains(&basename)))
                })
            })
            .unwrap_or(false)
    });
    if found {
        Vec::new()
    } else {
        vec![Finding {
            rule: "doc-section",
            file: file.to_string(),
            line: 0,
            message: format!("no heading for `{rel}` in the docs of `{crate_name}`"),
        }]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(text: &str) -> AddedLine {
        AddedLine {
            path: "docs/x.md".to_string(),
            line: 7,
            text: text.to_string(),
        }
    }

    #[test]
    fn em_dashes_and_filler_are_found_with_their_line() {
        let f = lint_markdown_lines(&[
            line("fine text"),
            line("a \u{2014} b"),
            line("we leverage this"),
            line("Seamless handover"),
            line("utilizing it"),
        ]);
        let rules: Vec<&str> = f.iter().map(|f| f.rule).collect();
        assert_eq!(
            rules,
            vec!["em-dash", "banned-word", "banned-word", "banned-word"]
        );
        assert!(f.iter().all(|f| f.line == 7 && f.file == "docs/x.md"));
    }

    #[test]
    fn longer_words_and_the_allow_marker_are_not_flagged() {
        let f = lint_markdown_lines(&[
            line("the leveraged buyout"),
            line("a \u{2014} b <!-- mid-lint: allow -->"),
            line("\"leverage\" is banned (mid-lint: allow)"),
            line("unutilized"),
        ]);
        assert_eq!(f.len(), 1, "{f:?}");
        assert_eq!(
            f[0].rule, "banned-word",
            "`leveraged` is a form of the banned word"
        );
        assert!(f[0].message.contains("leveraged"));
    }

    #[test]
    fn notice_headers_name_their_docs() {
        let head = [
            "// ====",
            "// NOTICE: Full documentation, design decisions, and fix history for this file",
            "// live in docs/mid-x.md, section \"a.rs\"",
            "// ====",
        ];
        assert_eq!(
            notice_targets(&head),
            Some(vec!["docs/mid-x.md".to_string()])
        );
        assert_eq!(notice_targets(&["//! no header"]), None);
        let two = [
            "// NOTICE: Full documentation",
            "// in docs/a.md and docs/sub/b-c.md.",
        ];
        assert_eq!(
            notice_targets(&two),
            Some(vec!["docs/a.md".to_string(), "docs/sub/b-c.md".to_string()])
        );
        assert_eq!(notice_targets(&["// NOTICE: nothing here"]), Some(vec![]));
    }

    #[test]
    fn only_crate_and_tool_sources_are_lintable() {
        assert!(is_lintable_source("crates/mid-x/src/lib.rs"));
        assert!(is_lintable_source("tools/mid-cli/src/cmds/mod.rs"));
        assert!(!is_lintable_source("crates/mid-x/tests/a.rs"));
        assert!(!is_lintable_source("crates/mid-x/benches/a.rs"));
        assert!(!is_lintable_source("web/x/src/lib.rs"));
        assert!(!is_lintable_source("crates/mid-x/src/data.json"));
    }
}
