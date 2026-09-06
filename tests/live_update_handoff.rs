//! #1045 end-to-end: a live-update handoff must RESUME the MCP session in
//! the replacement process image.
//!
//! The pre-fix #858 handoff spawned the new binary but left it waiting for a
//! second `initialize` the client never sends: every post-handoff
//! `tools/call` answered `-32002 "Not initialized"` and the
//! transport-captured agent identity was lost — "Vault-backed MCP calls
//! return empty results until the session is restarted".
//!
//! These tests drive the REAL binary over stdio (initialize → rebuild →
//! handoff → keep calling) and assert the session survives:
//!
//! 1. `explicit_handoff_resumes_session_on_same_stdio` — the documented
//!    `perseus_vault_handoff_restart {confirm:true}` flow.
//! 2. `auto_handoff_answers_inflight_call_from_new_image` — opt-in
//!    `PERSEUS_VAULT_AUTO_HANDOFF=1`: the replacement image answers the very
//!    call that would otherwise be refused.
//!
//! Unix-only: Windows locks a running executable, so the on-disk binary
//! cannot be replaced mid-session there at all (staleness cannot arise), and
//! `exec` does not exist.
#![cfg(unix)]

use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const BIN: &str = env!("CARGO_BIN_EXE_perseus-vault");
const CALL_DEADLINE: Duration = Duration::from_secs(240);

fn tmp_dir(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "vault-live-update-{tag}-{}-{}",
        std::process::id(),
        uuid::Uuid::new_v4().simple()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// A live stdio MCP client for the real binary, with bounded reads (reader
/// thread + channel) so a silent server can never hang the CI job (#1018).
struct Server {
    child: Child,
    stdin: std::process::ChildStdin,
    rx: mpsc::Receiver<String>,
    stderr_log: Arc<Mutex<String>>,
    dir: std::path::PathBuf,
    bin_path: std::path::PathBuf,
}

impl Server {
    fn spawn(dir: &Path, envs: &[(&str, &str)]) -> Server {
        let bin_path = dir.join("perseus-vault-run");
        let db_path = dir.join("vault.db");
        std::fs::copy(BIN, &bin_path).expect("copy binary into temp dir");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = std::fs::metadata(&bin_path).unwrap().permissions();
            perms.set_mode(0o755);
            std::fs::set_permissions(&bin_path, perms).unwrap();
        }

        let mut cmd = Command::new(&bin_path);
        cmd.args([
            "serve",
            "--db",
            db_path.to_str().expect("temp db path"),
            "--offline",
        ]);
        for (k, v) in envs {
            cmd.env(k, v);
        }
        cmd.stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let mut child = cmd.spawn().expect("spawn perseus-vault serve");
        let stdin = child.stdin.take().unwrap();
        let stdout = child.stdout.take().unwrap();
        let stderr = child.stderr.take().unwrap();

        let (tx, rx) = mpsc::channel::<String>();
        std::thread::spawn(move || {
            let reader = BufReader::new(stdout);
            for line in reader.lines() {
                match line {
                    Ok(l) => {
                        if tx.send(l).is_err() {
                            break;
                        }
                    }
                    Err(_) => break,
                }
            }
        });

        let stderr_log: Arc<Mutex<String>> = Arc::new(Mutex::new(String::new()));
        let log = Arc::clone(&stderr_log);
        std::thread::spawn(move || {
            let reader = BufReader::new(stderr);
            for line in reader.lines() {
                if let Ok(l) = line {
                    if let Ok(mut buf) = log.lock() {
                        buf.push_str(&l);
                        buf.push('\n');
                    }
                }
            }
        });

        Server {
            child,
            stdin,
            rx,
            stderr_log,
            dir: dir.to_path_buf(),
            bin_path,
        }
    }

    fn call(&mut self, method: &str, params: serde_json::Value) -> serde_json::Value {
        let req = serde_json::json!({"jsonrpc":"2.0","id":1,"method":method,"params":params});
        let payload = req.to_string();
        writeln!(self.stdin, "{payload}").expect("write to server stdin");
        self.stdin.flush().unwrap();
        self.read_response()
    }

    fn read_response(&mut self) -> serde_json::Value {
        let deadline = Instant::now() + CALL_DEADLINE;
        loop {
            match self.rx.recv_timeout(Duration::from_secs(5)) {
                Ok(line) => {
                    if let Ok(v) = serde_json::from_str::<serde_json::Value>(&line) {
                        return v;
                    }
                    // Non-JSON stdout line: keep waiting for the response.
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    if let Ok(Some(status)) = self.child.try_wait() {
                        let log = self.stderr_log.lock().unwrap().clone();
                        panic!(
                            "server exited with {status:?} while waiting for a response\nstderr:\n{log}"
                        );
                    }
                    if Instant::now() > deadline {
                        panic!(
                            "timed out ({CALL_DEADLINE:?}) waiting for a server response\nstderr:\n{}",
                            self.stderr_log.lock().unwrap()
                        );
                    }
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    let log = self.stderr_log.lock().unwrap().clone();
                    panic!("server stdout closed unexpectedly\nstderr:\n{log}");
                }
            }
        }
    }

    /// Simulate a rebuild: replace the on-disk binary the server runs from
    /// with a fresh copy (new inode at the same path).
    fn replace_binary(&self) {
        let next = self.dir.join("perseus-vault-next");
        std::fs::copy(BIN, &next).expect("copy replacement binary");
        std::fs::rename(&next, &self.bin_path).expect("rename replacement over running binary");
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

fn initialize(s: &mut Server) {
    let init = s.call(
        "initialize",
        serde_json::json!({
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "live-update-e2e-agent", "version": "1"}
        }),
    );
    assert!(
        init["result"].is_object(),
        "initialize must succeed: {init}"
    );
}

fn call_tool(s: &mut Server, name: &str, args: serde_json::Value) -> serde_json::Value {
    let resp = s.call(
        "tools/call",
        serde_json::json!({"name": name, "arguments": args}),
    );
    assert!(
        resp["error"].is_null() && resp["result"].is_object(),
        "tools/call {name} must return a result, got: {resp}"
    );
    resp
}

fn health_binary_stale(s: &mut Server) -> bool {
    let health = call_tool(s, "perseus_vault_health", serde_json::json!({}));
    health["result"]["structuredContent"]["binary_stale"]
        .as_bool()
        .expect("health must report binary_stale")
}

#[test]
fn explicit_handoff_resumes_session_on_same_stdio() {
    let dir = tmp_dir("explicit");
    let mut s = Server::spawn(&dir, &[]);
    initialize(&mut s);

    // Nominal: not stale, health works.
    assert!(
        !health_binary_stale(&mut s),
        "fresh server must not be stale"
    );

    // Rebuild mid-session → fail loud on every tool except handoff/health.
    s.replace_binary();
    let recall = call_tool(
        &mut s,
        "perseus_vault_recall",
        serde_json::json!({"query": "anything"}),
    );
    let sc = &recall["result"]["structuredContent"];
    assert_eq!(
        sc["isError"],
        serde_json::json!(true),
        "stale call must fail loud: {recall}"
    );
    assert!(
        sc["content"][0]["text"]
            .as_str()
            .unwrap()
            .contains("replaced on disk"),
        "stale refusal must explain the staleness: {recall}"
    );
    assert!(
        health_binary_stale(&mut s),
        "health must report binary_stale after rebuild"
    );

    // Explicit hot-swap, exactly as the #858 workflow documents.
    let handoff = call_tool(
        &mut s,
        "perseus_vault_handoff_restart",
        serde_json::json!({"confirm": true}),
    );
    assert_eq!(
        handoff["result"]["structuredContent"]["status"],
        serde_json::json!("handoff_performed"),
        "handoff must schedule the swap: {handoff}"
    );

    // THE #1045 REGRESSION: the replacement image must already be an
    // initialized session — no `-32002 Not initialized`, identity restored —
    // and must be running the new binary.
    let post_health = call_tool(&mut s, "perseus_vault_health", serde_json::json!({}));
    assert!(
        !post_health["result"]["structuredContent"]["binary_stale"]
            .as_bool()
            .unwrap(),
        "post-handoff health must come from the new image (binary_stale=false)\nhealth: {post_health}\nstderr:\n{}",
        s.stderr_log.lock().unwrap()
    );
    let list = s.call("tools/list", serde_json::json!({}));
    assert!(
        list["result"]["tools"].is_array(),
        "tools/list must work post-handoff (initialized flag restored): {list}"
    );
    // A second rebuild + handoff cycle works too (the child captures its own
    // identity and can hand off again).
    s.replace_binary();
    let handoff2 = call_tool(
        &mut s,
        "perseus_vault_handoff_restart",
        serde_json::json!({"confirm": true}),
    );
    assert_eq!(
        handoff2["result"]["structuredContent"]["status"],
        serde_json::json!("handoff_performed"),
        "repeat handoff must work: {handoff2}"
    );
    assert!(!health_binary_stale(&mut s));
}

#[test]
fn auto_handoff_answers_inflight_call_from_new_image() {
    let dir = tmp_dir("auto");
    let mut s = Server::spawn(&dir, &[("PERSEUS_VAULT_AUTO_HANDOFF", "1")]);
    initialize(&mut s);

    // Prime the staleness identity while the binary is current: the identity
    // is captured at first use, and a real client evaluates it on every call
    // from session start (so this mirrors the production sequence).
    assert!(
        !health_binary_stale(&mut s),
        "fresh server must not be stale"
    );

    // Rebuild mid-session: with auto-handoff the SAME tools/call is answered
    // by the replacement image — one clean response, no isError, no reconnect
    // tool invocation.
    s.replace_binary();
    let recall = call_tool(
        &mut s,
        "perseus_vault_recall",
        serde_json::json!({"query": "anything"}),
    );
    let sc = &recall["result"]["structuredContent"];
    assert!(
        sc.get("isError")
            .map(|v| v != &serde_json::json!(true))
            .unwrap_or(true),
        "auto-handoff must not surface an isError: {recall}"
    );
    let text = recall["result"]["content"][0]["text"].as_str().unwrap();
    assert!(
        !text.contains("replaced on disk"),
        "stale refusal must not appear: {recall}"
    );
    assert!(
        !text.contains("Not initialized"),
        "-32002 must not appear: {recall}"
    );

    // The session continues on the new image.
    assert!(
        !health_binary_stale(&mut s),
        "post-auto-handoff health must come from the new image"
    );
    let list = s.call("tools/list", serde_json::json!({}));
    assert!(
        list["result"]["tools"].is_array(),
        "tools/list must work: {list}"
    );
}
