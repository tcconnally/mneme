//! #858 — live-update / reconnect workflow.
//!
//! A stdio MCP client (e.g. Rovo Dev) spawns the `perseus-vault` child once
//! per session. If the binary is rebuilt/replaced on disk mid-session, the
//! running process image is stale but the client keeps talking to it — and the
//! failure mode was subtle: calls degrading into empty results until a full
//! session restart.
//!
//! This module makes the situation explicit and recoverable:
//!
//! 1. **Detection** — capture the running binary's identity (dev/ino + len +
//!    mtime) once at startup; compare against the current on-disk file on
//!    every tool call (one `stat`).
//! 2. **Fail loud** — when stale, every tool except the handoff tool itself
//!    and `perseus_vault_health` (which reports the staleness in its payload) returns
//!    an explicit `isError` result instead of silently serving stale results.
//!    Override for one-off diagnostics: `PERSEUS_VAULT_IGNORE_STALE_BINARY=1`.
//! 3. **Hot-swap handoff** — `perseus_vault_handoff_restart` replaces the
//!    process image with the on-disk binary on the SAME stdio fds, so the
//!    client's pipes never close and the MCP session continues. On Unix this
//!    is a true `exec` (same PID, no reparenting window); on Windows the
//!    replacement is spawned with inherited stdio and tagged
//!    `PERSEUS_VAULT_HANDOFF_CHILD=1` so the orphan guards do not reap it
//!    when its spawning parent exits.
//!
//!    **Session state is forwarded** (#1045): the client has ALREADY
//!    initialized the old image and will never send `initialize` again, and
//!    the transport-captured agent identity (#684/#855) must survive. The
//!    replacement process therefore carries `PERSEUS_VAULT_HANDOFF_STATE`
//!    (`{initialized, session_agent_id}`), which the new image restores at
//!    startup instead of waiting for a second handshake. Without this the
//!    handoff was a self-destruct button: every post-handoff `tools/call`
//!    answered `-32002 "Not initialized"` and identity-scoped tools degraded
//!    to empty results — the exact #1045 symptom.
//!
//! 4. **Automatic handoff (opt-in)** — with `PERSEUS_VAULT_AUTO_HANDOFF=1`,
//!    a stale-image `tools/call` is not refused: the in-flight request is
//!    serialized and forwarded (`PERSEUS_VAULT_HANDOFF_PENDING_REQUEST`) and
//!    the replacement image answers that very request, so the client sees
//!    one clean response from the new binary — no reconnect tool invocation,
//!    no session restart. Requests over [`MAX_PENDING_REQUEST_BYTES`] fall
//!    back to the loud `isError` path.
//!
//! Safety notes on the handoff:
//! - The swap is window-free (#1045): the stale image never writes a
//!   response for the exchange that triggers the swap — either the in-flight
//!   request is forwarded and answered by the replacement image (auto
//!   mode), or the prepared report is forwarded and written by the
//!   replacement image (explicit tool). No client request can fall into the
//!   gap between a response flush and the exec.
//! - Do not pipeline requests during handoff: the old process's stdin
//!   `BufReader` may hold read-ahead bytes that die with the process. MCP
//!   clients are strictly request/response, so a compliant client is
//!   unaffected.
//! - SQLite is in WAL mode: the child's fresh open recovers any unfinished
//!   state from the old process exactly like the orphan-watcher exit path
//!   (#547) already does.

use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// Identity of a binary file: path + (dev, ino) + (len, mtime_ns).
///
/// dev/ino uniquely identify the file *object* — a rename-replace (the normal
/// `cargo build` / `install` pattern) changes the inode. len/mtime catch
/// in-place rewrites that reuse the same inode (toolchains that patch the file
/// rather than replace it).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BinaryIdentity {
    pub path: PathBuf,
    pub dev: u64,
    pub ino: u64,
    pub len: u64,
    pub mtime_ns: i64,
}

impl BinaryIdentity {
    pub fn capture(path: &Path) -> Option<BinaryIdentity> {
        let md = std::fs::metadata(path).ok()?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            // MetadataExt::mtime() is i64 seconds since epoch; mtime_nsec() is
            // the timespec nanosecond component — combine for ns resolution.
            let mtime_ns = md
                .mtime()
                .saturating_mul(1_000_000_000)
                .saturating_add(md.mtime_nsec());
            Some(BinaryIdentity {
                path: path.to_path_buf(),
                dev: md.dev(),
                ino: md.ino(),
                len: md.len(),
                mtime_ns,
            })
        }
        #[cfg(not(unix))]
        {
            let mtime_ns = md
                .modified()
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_nanos() as i64)
                .unwrap_or(0);
            Some(BinaryIdentity {
                path: path.to_path_buf(),
                dev: 0,
                ino: 0,
                len: md.len(),
                mtime_ns,
            })
        }
    }

    /// True when the file this identity was captured from no longer matches
    /// what is on disk at the same path (replaced, rewritten, or deleted).
    pub fn replaced(&self) -> bool {
        match BinaryIdentity::capture(&self.path) {
            None => true,
            Some(cur) => {
                cur.dev != self.dev
                    || cur.ino != self.ino
                    || cur.len != self.len
                    || cur.mtime_ns != self.mtime_ns
            }
        }
    }
}

/// `current_exe()` normalized for Linux's post-replace `/proc/self/exe`
/// semantics: after the running binary's file is replaced via rename(2), the
/// kernel appends `" (deleted)"` to the link target. Stat-ing or exec-ing
/// that literal path fails with ENOENT, which silently killed the whole
/// live-update mechanism on Linux (staleness detection degraded and the
/// handoff exec could never succeed). Strip the suffix so both staleness
/// detection and the handoff target the real on-disk path.
pub fn executable_path() -> Option<PathBuf> {
    let p = std::env::current_exe().ok()?;
    Some(strip_deleted_suffix(&p))
}

/// Pure suffix-strip (Linux `/proc/self/exe`): `"/x/foo (deleted)"` →
/// `"/x/foo"`; anything else is returned unchanged.
fn strip_deleted_suffix(p: &Path) -> PathBuf {
    let name = match p.file_name() {
        Some(n) => n.to_string_lossy(),
        None => return p.to_path_buf(),
    };
    match name.strip_suffix(" (deleted)") {
        Some(stripped) => p.with_file_name(stripped),
        None => p.to_path_buf(),
    }
}

/// The identity of the binary this process was launched from — captured once
/// (the process image is immutable; only the on-disk file can change).
pub fn running_identity() -> Option<&'static BinaryIdentity> {
    static RUNNING: OnceLock<Option<BinaryIdentity>> = OnceLock::new();
    RUNNING
        .get_or_init(|| executable_path().and_then(|p| BinaryIdentity::capture(&p)))
        .as_ref()
}

/// True when this process was spawned by a `handoff_restart` from an older
/// server instance (env `PERSEUS_VAULT_HANDOFF_CHILD=1`). Only the Windows
/// spawn path sets it — the Unix handoff is an `exec` that keeps the parent
/// (the MCP client) and therefore keeps the orphan guards meaningful.
pub fn handoff_child() -> bool {
    std::env::var_os("PERSEUS_VAULT_HANDOFF_CHILD").is_some()
}

/// Env contract for the handoff (#1045).
pub const HANDOFF_STATE_ENV: &str = "PERSEUS_VAULT_HANDOFF_STATE";
pub const HANDOFF_PENDING_REQUEST_ENV: &str = "PERSEUS_VAULT_HANDOFF_PENDING_REQUEST";
pub const HANDOFF_PENDING_RESPONSE_ENV: &str = "PERSEUS_VAULT_HANDOFF_PENDING_RESPONSE";
/// Forwarded payloads (in-flight request or prepared response) larger than
/// this are not carried across the handoff; the dispatch falls back to the
/// loud `isError` path instead of risking an env-size failure (Windows
/// CreateProcess env blocks cap at 32KB total).
pub const MAX_PENDING_REQUEST_BYTES: usize = 24 * 1024;

/// Session state forwarded across a handoff (#1045). The replacement process
/// must resume the MCP session exactly where the old image left it: the
/// client has ALREADY initialized and will never send `initialize` again,
/// and the transport-captured agent identity (#684/#855) must survive.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct HandoffState {
    pub initialized: bool,
    pub session_agent_id: String,
}

impl HandoffState {
    /// Serialize for the env var. The caller (the server loop) only ever
    /// passes values it produced itself from the live session state.
    pub fn to_env_json(&self) -> Option<String> {
        serde_json::to_string(self).ok()
    }

    /// Parse + validate the env var. Bounds/sanitization are identical to the
    /// `initialize` path so an env-carried value can never bypass them.
    pub fn from_env_json(raw: &str) -> Option<HandoffState> {
        let v: HandoffState = serde_json::from_str(raw).ok()?;
        let sanitized: String = v
            .session_agent_id
            .chars()
            .filter(|c| c.is_alphanumeric() || matches!(c, '-' | '_' | '.' | ':'))
            .take(128)
            .collect();
        Some(HandoffState {
            initialized: v.initialized,
            session_agent_id: sanitized,
        })
    }
}

/// Read + clear the forwarded session state (used by the replacement process
/// at startup; removed so no further self-spawned child inherits a stale one).
pub fn take_handoff_state() -> Option<HandoffState> {
    let raw = std::env::var(HANDOFF_STATE_ENV).ok()?;
    std::env::remove_var(HANDOFF_STATE_ENV);
    HandoffState::from_env_json(&raw)
}

/// Read + clear the forwarded in-flight request (used by the replacement
/// process at startup; see `schedule_auto_handoff` for the sender side).
pub fn take_handoff_pending_request() -> Option<String> {
    let raw = std::env::var(HANDOFF_PENDING_REQUEST_ENV).ok()?;
    std::env::remove_var(HANDOFF_PENDING_REQUEST_ENV);
    if raw.len() > MAX_PENDING_REQUEST_BYTES {
        return None;
    }
    Some(raw)
}

/// Read + clear the forwarded prepared response (used by the replacement
/// process at startup; see `schedule_handoff_with_response` for the sender
/// side).
pub fn take_handoff_pending_response() -> Option<String> {
    let raw = std::env::var(HANDOFF_PENDING_RESPONSE_ENV).ok()?;
    std::env::remove_var(HANDOFF_PENDING_RESPONSE_ENV);
    if raw.len() > MAX_PENDING_REQUEST_BYTES {
        return None;
    }
    Some(raw)
}

/// Opt-in transparent handoff (#1045): `PERSEUS_VAULT_AUTO_HANDOFF=1` makes a
/// stale-image `tools/call` hand off to the replacement binary and have IT
/// answer the very same request, instead of returning the loud `isError`.
pub fn auto_handoff_enabled() -> bool {
    std::env::var_os("PERSEUS_VAULT_AUTO_HANDOFF").is_some()
}

/// Whether the running binary has been replaced on disk since this process
/// started.
pub fn running_stale() -> bool {
    running_identity().map(|i| i.replaced()).unwrap_or(false)
}

/// Pure gate logic: the message to return (or None when the call may proceed).
fn stale_message_for(stale: bool, tool: &str, ignore: bool) -> Option<String> {
    if !stale || ignore {
        return None;
    }
    if tool == "perseus_vault_handoff_restart" || tool == "perseus_vault_health" {
        return None;
    }
    let pid = std::process::id();
    let path = running_identity()
        .map(|i| i.path.display().to_string())
        .unwrap_or_default();
    Some(format!(
        "perseus-vault: the running binary was replaced on disk (pid {pid}, {path}); \
         refusing to serve results from a stale process image. Run \
         perseus_vault_handoff_restart with {{\"confirm\": true}} to hot-swap this \
         session on the same stdio connection, or restart the client session. \
         To override for diagnostics: PERSEUS_VAULT_IGNORE_STALE_BINARY=1"
    ))
}

/// Fail-loud gate for the MCP dispatch: when the binary was replaced
/// mid-session, every tool except the handoff tool itself (and `perseus_vault_health`,
/// which reports the staleness in its payload) refuses with an explicit error
/// — never a silent empty result (#858).
pub fn stale_error_message(tool: &str) -> Option<String> {
    stale_message_for(
        running_stale(),
        tool,
        std::env::var_os("PERSEUS_VAULT_IGNORE_STALE_BINARY").is_some(),
    )
}

/// Set when a handoff is scheduled (explicit tool or auto-handoff); consumed
/// by the MCP server loop after the response is flushed — or instead of a
/// response when the auto-handoff intercepted the in-flight call.
static HANDOFF_PENDING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn handoff_pending() -> bool {
    HANDOFF_PENDING.load(std::sync::atomic::Ordering::SeqCst)
}

fn set_handoff_pending() {
    HANDOFF_PENDING.store(true, std::sync::atomic::Ordering::SeqCst);
}

/// Serialized in-flight request captured when an auto-handoff intercepts a
/// stale-image call; consumed by `perform_handoff` to forward via env.
static PENDING_REQUEST: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);

/// Serialized response captured when the explicit handoff tool schedules a
/// hot-swap (#1045 window-free handoff): the old image does NOT write the
/// report — the replacement image does, from this forwarded value — so no
/// client request can fall into the gap between the response flush and the
/// exec (the pre-fix race that dropped the next request and made the session
/// look dead).
static PENDING_RESPONSE: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);

/// Schedule the explicit window-free handoff: stash the prepared
/// `handoff_performed` response for the replacement image to write, and arm
/// the pending flag for the server loop. Returns false (and does NOT
/// schedule) on an oversized payload — the caller falls back to normal
/// dispatch on the stale image (the fail-loud gate stays active).
pub fn schedule_handoff_with_response(resp: String) -> bool {
    if resp.len() > MAX_PENDING_REQUEST_BYTES {
        return false;
    }
    match PENDING_RESPONSE.lock() {
        Ok(mut slot) => *slot = Some(resp),
        Err(_) => return false,
    }
    set_handoff_pending();
    true
}

/// Attempt to schedule an automatic handoff carrying the in-flight request
/// (#1045). Returns false (and does NOT schedule) when the serialized request
/// is missing or exceeds [`MAX_PENDING_REQUEST_BYTES`] — the caller then falls
/// through to the loud `isError` path.
pub fn schedule_auto_handoff(pending: Option<String>) -> bool {
    let fits = match &pending {
        Some(p) => p.len() <= MAX_PENDING_REQUEST_BYTES,
        None => false,
    };
    if !fits {
        return false;
    }
    match PENDING_REQUEST.lock() {
        Ok(mut slot) => *slot = pending,
        Err(_) => return false,
    }
    set_handoff_pending();
    true
}

/// Clear all handoff-pending state. Used when the handoff fails: the loop
/// keeps serving on the stale image (the fail-loud gate stays active) and
/// the in-flight request is dropped rather than answered by a dead path.
pub fn clear_handoff_pending() {
    HANDOFF_PENDING.store(false, std::sync::atomic::Ordering::SeqCst);
    if let Ok(mut slot) = PENDING_REQUEST.lock() {
        *slot = None;
    }
    if let Ok(mut slot) = PENDING_RESPONSE.lock() {
        *slot = None;
    }
}

/// Clone of the stashed in-flight request, if any. `perform_handoff` uses a
/// clone so a failed exec/spawn leaves the stash intact for the caller's
/// `clear_handoff_pending` and never silently loses the request.
fn pending_request_clone() -> Option<String> {
    PENDING_REQUEST.lock().ok().and_then(|slot| slot.clone())
}

/// Clone of the stashed prepared response, if any (same clone semantics).
fn pending_response_clone() -> Option<String> {
    PENDING_RESPONSE.lock().ok().and_then(|slot| slot.clone())
}

/// Replace the process image with the on-disk binary on the SAME stdio fds,
/// forwarding the live MCP session state so the new image resumes the session
/// instead of waiting for a second `initialize` (#1045).
///
/// - **Unix**: `exec` replaces the image in place — same PID, the client's
///   pipes never close, and the orphan guards keep running against the
///   unchanged parent (the MCP client) exactly as before.
/// - **Windows**: `exec` does not exist; the replacement is spawned with
///   inherited stdio and tagged `PERSEUS_VAULT_HANDOFF_CHILD=1` so the orphan
///   watchers don't reap it when this process exits.
///
/// Returns: Unix — only on failure (a successful `exec` never returns);
/// Windows — `Ok(())` after the spawn, and the caller exits.
pub fn perform_handoff(initialized: bool, session_agent_id: &str) -> Result<(), String> {
    let path = executable_path().ok_or_else(|| "current_exe unavailable".to_string())?;
    let args: Vec<String> = std::env::args().skip(1).collect();
    let state = HandoffState {
        initialized,
        session_agent_id: session_agent_id.to_string(),
    };
    let state_json = state
        .to_env_json()
        .ok_or_else(|| "serialize handoff state".to_string())?;

    let mut cmd = std::process::Command::new(&path);
    cmd.args(&args).env(HANDOFF_STATE_ENV, state_json);
    if let Some(pending) = pending_request_clone() {
        cmd.env(HANDOFF_PENDING_REQUEST_ENV, pending);
    }
    if let Some(resp) = pending_response_clone() {
        cmd.env(HANDOFF_PENDING_RESPONSE_ENV, resp);
    }

    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        // exec only returns on failure; on success this process IS the new
        // image and the caller's subsequent code never runs.
        let err = cmd.exec();
        Err(format!("exec {}: {err}", path.display()))
    }
    #[cfg(not(unix))]
    {
        cmd.env("PERSEUS_VAULT_HANDOFF_CHILD", "1");
        let child = cmd
            .spawn()
            .map_err(|e| format!("spawn {}: {e}", path.display()))?;
        eprintln!(
            "perseus-vault: handoff spawned child pid {} ({}) — old process exiting",
            child.id(),
            path.display()
        );
        Ok(())
    }
}

/// Pure report builder — the four handoff states, testable without touching
/// the running test binary.
fn handoff_report_for(stale: bool, dry_run: bool, confirm: bool) -> Value {
    let ident = running_identity();
    fn identity_json(i: &BinaryIdentity) -> Value {
        json!({"dev": i.dev, "ino": i.ino, "len": i.len, "mtime_ns": i.mtime_ns})
    }
    let mut report = json!({
        "pid": std::process::id(),
        "binary_path": ident.map(|i| i.path.display().to_string()).unwrap_or_default(),
        "running_identity": ident.map(identity_json).unwrap_or(Value::Null),
        "disk_identity": ident
            .and_then(|i| BinaryIdentity::capture(&i.path))
            .map(|b| identity_json(&b))
            .unwrap_or(Value::Null),
        "binary_stale": stale,
    });
    if !stale {
        report["status"] = json!("no_handoff_needed");
        report["note"] = json!(
            "The running binary matches the file on disk; no handoff required. \
             Rebuild the binary and call again to hot-swap."
        );
    } else if dry_run {
        report["status"] = json!("dry_run");
        report["would_handoff"] = json!(true);
        report["note"] = json!(
            "Binary replaced; a confirm:true call would spawn the new binary on \
             this same stdio session and exit this process."
        );
    } else if !confirm {
        report["status"] = json!("confirm_required");
        report["note"] =
            json!("Binary replaced; pass {\"confirm\": true} to perform the hot-swap.");
    } else {
        report["status"] = json!("handoff_performed");
        report["note"] = json!(
            "Hot-swap scheduled: the new binary starts on this session immediately \
             after this response. If the session goes quiet, the spawn failed and \
             the client session should be restarted."
        );
    }
    report
}

/// #858: `perseus_vault_handoff_restart` — explicit session-local reconnect.
///
/// - binary unchanged → `no_handoff_needed` (identity report included)
/// - stale + `dry_run` → `dry_run` report of what would happen
/// - stale, no `confirm` → `confirm_required` (clear feedback, no exec)
/// - stale + `confirm` → schedules the hot-swap; the server loop performs the
///   spawn AFTER flushing this very response, then exits.
pub fn handle_handoff_restart(args: Value) -> Result<String, String> {
    let dry_run = args
        .get("dry_run")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let confirm = args
        .get("confirm")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let stale = running_stale();
    let report = handoff_report_for(stale, dry_run, confirm);
    if stale && confirm && !dry_run {
        set_handoff_pending();
    }
    Ok(report.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::time::Duration;

    fn tmp_file(name: &str, content: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("live-update-test-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let p = dir.join(name);
        fs::write(&p, content).unwrap();
        p
    }

    #[test]
    fn untouched_file_is_not_replaced() {
        let p = tmp_file("same.txt", "hello");
        let id = BinaryIdentity::capture(&p).expect("capture");
        assert!(!id.replaced(), "untouched file must not look replaced");
        fs::remove_file(&p).ok();
    }

    #[test]
    fn in_place_rewrite_is_detected() {
        let p = tmp_file("rewrite.txt", "hello");
        let id = BinaryIdentity::capture(&p).expect("capture");
        std::thread::sleep(Duration::from_millis(20)); // distinct mtime
        fs::write(&p, "hello world — a longer rewrite").unwrap();
        assert!(id.replaced(), "in-place rewrite must be detected");
        fs::remove_file(&p).ok();
    }

    #[test]
    fn rename_replace_is_detected() {
        let p = tmp_file("rename-target.txt", "old image");
        let id = BinaryIdentity::capture(&p).expect("capture");
        // Different length on purpose: on Windows dev/ino are 0, so detection
        // must not hinge on a sub-ms mtime delta (rename preserves the source
        // mtime, which was written only moments after capture — that made this
        // test flaky on Windows runners).
        let replacement = tmp_file("rename-source.txt", "new image with a longer body");
        std::thread::sleep(Duration::from_millis(20));
        fs::rename(&replacement, &p).unwrap();
        assert!(id.replaced(), "rename-replace (new inode) must be detected");
        fs::remove_file(&p).ok();
    }

    #[test]
    fn deletion_is_detected() {
        let p = tmp_file("deleted.txt", "gone soon");
        let id = BinaryIdentity::capture(&p).expect("capture");
        fs::remove_file(&p).unwrap();
        assert!(id.replaced(), "deletion must be detected");
    }

    #[test]
    fn stale_gate_blocks_all_but_handoff_and_health() {
        assert!(stale_message_for(true, "perseus_vault_recall", false).is_some());
        assert!(stale_message_for(true, "perseus_vault_remember", false).is_some());
        // The recovery tools themselves stay callable.
        assert!(stale_message_for(true, "perseus_vault_handoff_restart", false).is_none());
        assert!(stale_message_for(true, "perseus_vault_health", false).is_none());
        // Not stale → no gate; ignore override → no gate.
        assert!(stale_message_for(false, "perseus_vault_recall", false).is_none());
        assert!(stale_message_for(true, "perseus_vault_recall", true).is_none());
    }

    #[test]
    fn handoff_report_covers_all_four_states() {
        let ok = handoff_report_for(false, false, false);
        assert_eq!(ok["status"], "no_handoff_needed");
        assert_eq!(ok["binary_stale"], false);

        let dry = handoff_report_for(true, true, false);
        assert_eq!(dry["status"], "dry_run");
        assert_eq!(dry["would_handoff"], true);

        let need_confirm = handoff_report_for(true, false, false);
        assert_eq!(need_confirm["status"], "confirm_required");

        let go = handoff_report_for(true, false, true);
        assert_eq!(go["status"], "handoff_performed");
        assert!(go["binary_path"]
            .as_str()
            .map(|s| !s.is_empty())
            .unwrap_or(false));
    }

    #[test]
    fn handoff_pending_flag_roundtrip() {
        assert!(!handoff_pending());
        set_handoff_pending();
        assert!(handoff_pending());
        // Reset so parallel tests / later dispatch smoke tests are unaffected.
        HANDOFF_PENDING.store(false, std::sync::atomic::Ordering::SeqCst);
        assert!(!handoff_pending());
    }

    #[test]
    fn handler_no_handoff_needed_when_binary_untouched() {
        // In the test process the running binary (the test harness) is not
        // replaced, so the handler must report no_handoff_needed and must NOT
        // schedule a handoff.
        let out = handle_handoff_restart(json!({})).expect("handler");
        let v: Value = serde_json::from_str(&out).expect("json");
        assert_eq!(v["status"], "no_handoff_needed");
        assert!(!handoff_pending());
    }

    #[test]
    fn strip_deleted_suffix_normalizes_linux_proc_self_exe() {
        assert_eq!(
            strip_deleted_suffix(Path::new("/tmp/foo/perseus-vault-run (deleted)")),
            PathBuf::from("/tmp/foo/perseus-vault-run")
        );
        assert_eq!(
            strip_deleted_suffix(Path::new("/tmp/foo/perseus-vault-run")),
            PathBuf::from("/tmp/foo/perseus-vault-run")
        );
        // Sanity: resolves in the test harness (not deleted).
        assert!(executable_path().is_some());
    }

    #[test]
    fn handoff_state_env_json_roundtrip() {
        let hs = HandoffState {
            initialized: true,
            session_agent_id: "agent-1".to_string(),
        };
        let raw = hs.to_env_json().expect("serialize");
        assert_eq!(HandoffState::from_env_json(&raw), Some(hs));
    }

    #[test]
    fn handoff_state_rejects_garbage_and_sanitizes_identity() {
        assert_eq!(HandoffState::from_env_json("not json"), None);
        assert_eq!(HandoffState::from_env_json("{\"initialized\": 42}"), None);
        // Identity is re-sanitized exactly like the initialize path: bounded
        // to 128 chars and restricted to the same alphanumeric set.
        let raw = r#"{"initialized":true,"session_agent_id":"b<o>b:!.x"}"#;
        let hs = HandoffState::from_env_json(raw).expect("parse");
        assert!(hs.initialized);
        assert_eq!(hs.session_agent_id, "bob:.x");
    }

    #[test]
    fn take_handoff_state_reads_and_clears_env() {
        std::env::set_var(
            HANDOFF_STATE_ENV,
            r#"{"initialized":true,"session_agent_id":"a"}"#,
        );
        let hs = take_handoff_state().expect("take");
        assert!(hs.initialized);
        assert_eq!(hs.session_agent_id, "a");
        assert!(
            std::env::var_os(HANDOFF_STATE_ENV).is_none(),
            "env must be cleared after take"
        );
    }

    #[test]
    fn take_handoff_pending_request_reads_clears_and_caps() {
        std::env::set_var(HANDOFF_PENDING_REQUEST_ENV, "{\"jsonrpc\":\"2.0\"}");
        assert_eq!(
            take_handoff_pending_request().as_deref(),
            Some("{\"jsonrpc\":\"2.0\"}")
        );
        assert!(std::env::var_os(HANDOFF_PENDING_REQUEST_ENV).is_none());

        let huge = "x".repeat(MAX_PENDING_REQUEST_BYTES + 1);
        std::env::set_var(HANDOFF_PENDING_REQUEST_ENV, &huge);
        assert!(take_handoff_pending_request().is_none());
    }

    #[test]
    fn schedule_auto_handoff_accepts_bounded_requests_and_clears() {
        // None → no handoff (nothing to forward).
        assert!(!schedule_auto_handoff(None));
        assert!(!handoff_pending());
        clear_handoff_pending();

        // Oversized → falls back to the loud isError path.
        let huge = "x".repeat(MAX_PENDING_REQUEST_BYTES + 1);
        assert!(!schedule_auto_handoff(Some(huge)));
        assert!(!handoff_pending());
        clear_handoff_pending();

        // Bounded → scheduled, stash visible to perform_handoff's clone.
        let req = r#"{"jsonrpc":"2.0","id":7,"method":"tools/call"}"#.to_string();
        assert!(schedule_auto_handoff(Some(req.clone())));
        assert!(handoff_pending());
        assert_eq!(pending_request_clone().as_deref(), Some(req.as_str()));

        // A failed handoff clears both the flag and the stash.
        clear_handoff_pending();
        assert!(!handoff_pending());
        assert_eq!(pending_request_clone(), None);
    }

    #[test]
    fn schedule_handoff_with_response_stashes_and_clears() {
        let resp = r#"{"jsonrpc":"2.0","id":4,"result":{"content":[]}}"#.to_string();
        assert!(schedule_handoff_with_response(resp.clone()));
        assert!(handoff_pending());
        assert_eq!(pending_response_clone().as_deref(), Some(resp.as_str()));

        clear_handoff_pending();
        let huge = "x".repeat(MAX_PENDING_REQUEST_BYTES + 1);
        assert!(!schedule_handoff_with_response(huge));
        assert!(!handoff_pending());
        assert_eq!(pending_response_clone(), None);
    }

    #[test]
    fn take_handoff_pending_response_reads_clears_and_caps() {
        std::env::set_var(HANDOFF_PENDING_RESPONSE_ENV, "{\"jsonrpc\":\"2.0\"}");
        assert_eq!(
            take_handoff_pending_response().as_deref(),
            Some("{\"jsonrpc\":\"2.0\"}")
        );
        assert!(std::env::var_os(HANDOFF_PENDING_RESPONSE_ENV).is_none());

        let huge = "x".repeat(MAX_PENDING_REQUEST_BYTES + 1);
        std::env::set_var(HANDOFF_PENDING_RESPONSE_ENV, &huge);
        assert!(take_handoff_pending_response().is_none());
    }
}
