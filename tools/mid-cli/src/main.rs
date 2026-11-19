// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-cli.md, section "main.rs"
// ============================================================================
//! The `mid` binary: collects the process environment, calls
//! [`mid_cli::run`], prints what it returns, and exits with its code.

use std::io::Write;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let env = mid_cli::Env::from_process();
    let result = mid_cli::run(&args, &mid_cli::SystemRunner, &env);
    let _ = std::io::stdout().write_all(result.stdout.as_bytes());
    let _ = std::io::stderr().write_all(result.stderr.as_bytes());
    std::process::exit(result.code);
}
