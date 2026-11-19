//! Shared test support: temporary directories, a scripted process runner,
//! and helpers for building real git repositories.
#![allow(dead_code)]

use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

use mid_cli::{Cmd, Env, Outcome, Runner, SystemRunner};
use mid_reflect::Value;

static COUNTER: AtomicUsize = AtomicUsize::new(0);

/// A directory removed when dropped.
pub struct TempDir(pub PathBuf);

impl TempDir {
    pub fn new(label: &str) -> Self {
        let n = COUNTER.fetch_add(1, Ordering::SeqCst);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let dir = std::env::temp_dir().join(format!(
            "mid-cli-test-{label}-{}-{n}-{nanos}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        TempDir(dir)
    }

    pub fn path(&self) -> &Path {
        &self.0
    }

    pub fn write(&self, rel: &str, text: &str) {
        let p = self.0.join(rel);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, text).unwrap();
    }

    pub fn read(&self, rel: &str) -> String {
        std::fs::read_to_string(self.0.join(rel)).unwrap()
    }

    pub fn exists(&self, rel: &str) -> bool {
        self.0.join(rel).exists()
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

type Handler = Box<dyn Fn(&Cmd) -> Option<Outcome>>;

/// A runner that answers some commands itself and, if allowed, passes the
/// rest to the real system.
pub struct Scripted {
    handlers: Vec<Handler>,
    system: bool,
    pub calls: RefCell<Vec<String>>,
}

impl Scripted {
    /// Answers only what its handlers answer; everything else is "not found".
    pub fn closed() -> Self {
        Self {
            handlers: Vec::new(),
            system: false,
            calls: RefCell::new(Vec::new()),
        }
    }

    /// Answers what its handlers answer and runs everything else for real.
    pub fn open() -> Self {
        Self {
            system: true,
            ..Self::closed()
        }
    }

    pub fn on(mut self, handler: impl Fn(&Cmd) -> Option<Outcome> + 'static) -> Self {
        self.handlers.push(Box::new(handler));
        self
    }

    pub fn ran(&self, needle: &str) -> bool {
        self.calls.borrow().iter().any(|c| c.contains(needle))
    }
}

impl Runner for Scripted {
    fn run(&self, cmd: &Cmd) -> Outcome {
        self.calls.borrow_mut().push(cmd.display());
        for h in &self.handlers {
            if let Some(o) = h(cmd) {
                return o;
            }
        }
        if self.system {
            SystemRunner.run(cmd)
        } else {
            Outcome {
                spawn_error: Some("not scripted".to_string()),
                ..Outcome::default()
            }
        }
    }
}

pub fn ok(stdout: &str) -> Outcome {
    Outcome {
        code: Some(0),
        stdout: stdout.to_string(),
        ..Outcome::default()
    }
}

pub fn failed(code: i32, stdout: &str, stderr: &str) -> Outcome {
    Outcome {
        code: Some(code),
        stdout: stdout.to_string(),
        stderr: stderr.to_string(),
        spawn_error: None,
    }
}

pub fn env_in(cwd: &Path, temp: &Path) -> Env {
    Env {
        cwd: cwd.to_path_buf(),
        mdix: None,
        temp: temp.to_path_buf(),
    }
}

pub fn args(list: &[&str]) -> Vec<String> {
    list.iter().map(|s| s.to_string()).collect()
}

/// Runs `mid` with `--json` and parses what it printed.
pub fn json(list: &[&str], runner: &dyn Runner, env: &Env) -> (i32, Value) {
    let mut a = args(list);
    a.push("--json".to_string());
    let r = mid_cli::run(&a, runner, env);
    let v = Value::from_json(r.stdout.trim())
        .unwrap_or_else(|e| panic!("not JSON ({e}): {:?}", r.stdout));
    (r.code, v)
}

pub fn git(dir: &Path, list: &[&str]) {
    let out = Command::new("git")
        .args([
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@example.com",
            "-c",
            "commit.gpgsign=false",
        ])
        .args(list)
        .current_dir(dir)
        .output()
        .expect("git runs");
    assert!(
        out.status.success(),
        "git {list:?}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

/// A repository with one commit holding `files`.
pub fn repo_with(files: &[(&str, &str)]) -> TempDir {
    let dir = TempDir::new("repo");
    git(dir.path(), &["init", "-q", "-b", "main"]);
    for (rel, text) in files {
        dir.write(rel, text);
    }
    git(dir.path(), &["add", "-A"]);
    git(dir.path(), &["commit", "-q", "-m", "base"]);
    dir
}

pub fn strings(v: &Value, key: &str) -> Vec<String> {
    v.get(key)
        .and_then(Value::as_list)
        .unwrap_or(&[])
        .iter()
        .map(|x| x.as_str().unwrap_or("").to_string())
        .collect()
}
