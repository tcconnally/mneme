use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Write};
use std::sync::OnceLock;

use crate::beliefs;
use crate::claim_card;
use crate::db::Database;
use crate::tools;

/// The parent PID observed once at process start, before any reparenting can
/// occur. `is_orphaned_by_ppid()` compares the live ppid against this baseline
/// so we detect *reparenting* (parent died → we were re-adopted) rather than
/// the mere fact that our ppid is 1.
///
/// This distinction matters in containers: when the vault is spawned directly
/// by a PID-1 entrypoint (e.g. a Python `demo_server_local.py` running as the
/// container's init, or any `docker run <binary>` where the binary's launcher
/// is PID 1), a perfectly healthy child legitimately has `getppid() == 1` from
/// birth. The original `getppid() == 1` guard (#547) false-positived on exactly
/// that topology and self-terminated a live server on its first request. See
/// the demo-container regression: parent is PID 1, so every start tripped the
/// orphan guard and crash-looped.
static INITIAL_PPID: OnceLock<i32> = OnceLock::new();

/// Windows only: the parent process's creation timestamp, captured alongside
/// INITIAL_PPID. OpenProcess-liveness alone is vulnerable to PID reuse — a
/// dead parent's PID can be recycled by an unrelated process, which would look
/// "alive". Comparing creation times makes the liveness check exact.
#[cfg(windows)]
static INITIAL_PPID_CREATE_TIME: OnceLock<u64> = OnceLock::new();

/// Record the current parent PID as the baseline. Call once, as early as
/// possible in `run_server`, before entering the request loop. Idempotent:
/// only the first call sets the baseline.
pub fn record_initial_ppid() {
    // getppid() has identical reparent-to-PID-1 semantics on every Unix
    // (Linux: init; macOS: launchd), so the baseline/orphan check is not
    // Linux-specific — widened from cfg(linux) to cfg(unix) for #748.
    #[cfg(unix)]
    {
        let _ = INITIAL_PPID.set(unsafe { libc::getppid() });
    }
    // Windows has no getppid(): recover the parent PID from a Toolhelp
    // snapshot and stamp its creation time for the PID-reuse-safe liveness
    // check (#751).
    #[cfg(windows)]
    {
        let ppid = windows_parent_pid().unwrap_or(0);
        let _ = INITIAL_PPID.set(ppid);
        if ppid > 0 {
            if let Some(t) = windows_process_creation_time(ppid) {
                let _ = INITIAL_PPID_CREATE_TIME.set(t);
            }
        }
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = INITIAL_PPID.set(0);
    }
}

/// Current parent PID via a Toolhelp process snapshot (Windows has no
/// getppid). Returns None if the snapshot fails — callers treat that as
/// "unknown" and never false-fire the orphan guard.
#[cfg(windows)]
fn windows_parent_pid() -> Option<i32> {
    use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
    use windows_sys::Win32::System::Diagnostics::ToolHelp::{
        CreateToolhelp32Snapshot, Process32First, Process32Next, PROCESSENTRY32, TH32CS_SNAPPROCESS,
    };
    unsafe {
        let snap = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
        if snap == INVALID_HANDLE_VALUE {
            return None;
        }
        let mut entry: PROCESSENTRY32 = std::mem::zeroed();
        entry.dwSize = std::mem::size_of::<PROCESSENTRY32>() as u32;
        let self_pid = std::process::id();
        let mut found = None;
        if Process32First(snap, &mut entry) != 0 {
            loop {
                if entry.th32ProcessID == self_pid {
                    found = Some(entry.th32ParentProcessID as i32);
                    break;
                }
                if Process32Next(snap, &mut entry) == 0 {
                    break;
                }
            }
        }
        CloseHandle(snap);
        found
    }
}

/// Creation time of `pid` as a u64 FILETIME, or None if the process cannot be
/// opened (i.e. it is dead or inaccessible). Doubles as the liveness probe.
#[cfg(windows)]
fn windows_process_creation_time(pid: i32) -> Option<u64> {
    use windows_sys::Win32::Foundation::{CloseHandle, FILETIME};
    use windows_sys::Win32::System::Threading::{
        GetProcessTimes, OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION,
    };
    unsafe {
        let h = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid as u32);
        if h == 0 {
            return None;
        }
        let mut ct: FILETIME = std::mem::zeroed();
        let mut et: FILETIME = std::mem::zeroed();
        let mut kt: FILETIME = std::mem::zeroed();
        let mut ut: FILETIME = std::mem::zeroed();
        let ok = GetProcessTimes(h, &mut ct, &mut et, &mut kt, &mut ut);
        CloseHandle(h);
        if ok == 0 {
            return None;
        }
        Some(((ct.dwHighDateTime as u64) << 32) | ct.dwLowDateTime as u64)
    }
}

/// Returns `true` if this process has been reparented since start, which is the
/// definitive indicator that the spawning parent has died.
///
/// Orphaning is detected as: the live ppid differs from the baseline captured
/// at start AND the live ppid is now 1 (reparented to init). A process that was
/// *born* with ppid == 1 (its launcher is the container's PID-1 init) is NOT an
/// orphan — its baseline is 1 and stays 1, so this correctly returns `false`.
///
/// On Windows (#751) there is no reparenting: the check instead probes the
/// recorded parent PID with `OpenProcess` and compares creation timestamps, so
/// a dead parent — or a PID recycled by an unrelated process — is detected.
/// Unknown parent (snapshot failed at startup) conservatively returns `false`.
///
/// Exposed as `pub` so the orphan case can be unit-tested without needing to
/// actually kill a parent process.
pub fn is_orphaned_by_ppid() -> bool {
    // #858: a handoff child is deliberately reparented when the spawning
    // server exits — the handoff protocol is its liveness proof, and stdin
    // EOF remains the real client-death signal.
    if crate::live_update::handoff_child() {
        return false;
    }
    // Safety: getppid() is always safe — no undefined behaviour, no allocation.
    // All Unix platforms (Linux AND macOS) reparent orphans to PID 1, so this
    // check is the primary parent-death signal on macOS too (#748) — without
    // it, macOS/Windows had no orphan signal at all and the flat idle timer
    // was the only guard, killing healthy-but-quiet hosts.
    #[cfg(unix)]
    {
        let current = unsafe { libc::getppid() };
        // Baseline should have been recorded at startup; if it wasn't (defensive),
        // fall back to comparing against the current value so we never false-fire.
        let baseline = *INITIAL_PPID.get_or_init(|| current);
        // Orphaned only if we were reparented to init: born under a real parent
        // (baseline != 1) and now adopted by init (current == 1). A process born
        // directly under PID 1 has baseline == 1 and is never treated as orphaned.
        current == 1 && baseline != 1
    }
    // Windows (#751): no reparenting concept — instead probe the recorded
    // parent PID with OpenProcess and compare creation times (PID-reuse safe).
    #[cfg(windows)]
    {
        let baseline = *INITIAL_PPID.get_or_init(|| 0);
        if baseline <= 0 {
            // Parent unknown (snapshot failed at start): never false-fire.
            return false;
        }
        match windows_process_creation_time(baseline) {
            // OpenProcess/GetProcessTimes failed -> parent is gone.
            None => true,
            // Handle opened: alive only if it is the SAME process (creation
            // time matches the startup stamp). A recycled PID means the
            // original parent died.
            Some(t) => match INITIAL_PPID_CREATE_TIME.get() {
                Some(recorded) => t != *recorded,
                // No stamp recorded: liveness alone is the best signal we have.
                None => false,
            },
        }
    }
    #[cfg(not(any(unix, windows)))]
    {
        false
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct JsonRpcRequest {
    pub jsonrpc: String,
    #[serde(default)]
    pub id: Option<Value>,
    pub method: String,
    #[serde(default)]
    pub params: Option<Value>,
}

#[derive(Debug, Serialize)]
pub struct JsonRpcResponse {
    pub jsonrpc: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
}

#[derive(Debug, Serialize)]
pub struct JsonRpcError {
    pub code: i64,
    pub message: String,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ToolProfile {
    /// Advertise the complete canonical registry (the historical default).
    Default,
    /// Explicit spelling for hosts that want the complete registry.
    All,
    /// Advertise only the small memory-management surface.
    Lean,
}

impl ToolProfile {
    /// Parse the public `serve --profile` values.
    #[cfg(test)]
    pub fn parse(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "default" => Some(Self::Default),
            "all" => Some(Self::All),
            "lean" => Some(Self::Lean),
            _ => None,
        }
    }
}

pub struct MCPState {
    // #210: AtomicBool so the HTTP/SSE transport can share &MCPState across
    // concurrent requests without a Mutex (which would re-serialize them now
    // that the DB pool removed the other lock). handle_request takes &MCPState.
    pub initialized: std::sync::atomic::AtomicBool,
    // #684: agent identity captured from the `initialize` handshake's
    // clientInfo.name. Threaded into tool calls as `requesting_agent_id` so
    // visibility enforcement knows who is asking. Empty when the client sent no
    // clientInfo (single-agent / legacy) → unscoped, preserving old behavior.
    // RwLock: set once at initialize, read per tools/call across the shared &state.
    pub session_agent_id: std::sync::RwLock<String>,
    /// #1112: strict multi-agent/HTTP deployment mode. When enabled, every
    /// scoped read or mutation needs a transport identity plus an active,
    /// exact workspace binding; legacy unbound single-agent behavior remains
    /// available only when this explicit deployment gate is off.
    pub strict_scope: bool,
    /// Advertisement profile selected at server startup. This affects
    /// `tools/list`; dispatch remains available for hidden compatibility tools.
    pub profile: ToolProfile,
}

impl MCPState {
    pub fn new() -> Self {
        Self::new_with_profile(ToolProfile::Default)
    }

    /// Construct a session state with an explicit advertisement profile.
    pub fn new_with_profile(profile: ToolProfile) -> Self {
        let strict_scope = std::env::var("PERSEUS_VAULT_STRICT_SCOPE")
            .ok()
            .map(|v| {
                matches!(
                    v.trim().to_ascii_lowercase().as_str(),
                    "1" | "true" | "yes" | "on"
                )
            })
            .unwrap_or(false);
        Self::new_with_profile_and_strict_scope(profile, strict_scope)
    }

    /// Construct a state with an explicit deployment contract. This keeps
    /// tests deterministic without mutating the process environment.
    pub fn new_with_strict_scope(strict_scope: bool) -> Self {
        Self::new_with_profile_and_strict_scope(ToolProfile::Default, strict_scope)
    }

    fn new_with_profile_and_strict_scope(profile: ToolProfile, strict_scope: bool) -> Self {
        MCPState {
            initialized: std::sync::atomic::AtomicBool::new(false),
            session_agent_id: std::sync::RwLock::new(String::new()),
            strict_scope,
            profile,
        }
    }
}

/// #1045: apply a forwarded handoff state (see `live_update::HandoffState`)
/// to a fresh process's session state. The client already initialized the
/// pre-handoff image and never re-sends `initialize`, so the replacement
/// process must consider itself initialized; the transport-captured agent
/// identity (#684/#855) is restored so visibility-scoped tools keep working.
fn apply_handoff_state(state: &MCPState, hs: crate::live_update::HandoffState) {
    if hs.initialized {
        state
            .initialized
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }
    if !hs.session_agent_id.is_empty() {
        if let Ok(mut slot) = state.session_agent_id.write() {
            *slot = hs.session_agent_id;
        }
    }
}

/// #1045: read + clear the forwarded handoff state from the environment and
/// apply it. Bounds/sanitization are enforced by `HandoffState::from_env_json`
/// (mirroring the initialize path), so an env-carried value cannot bypass
/// them. A no-op for a normally-spawned server (no env var).
fn restore_handoff_session(state: &MCPState) {
    if let Some(hs) = crate::live_update::take_handoff_state() {
        apply_handoff_state(state, hs);
    }
}

/// Parse the `PERSEUS_VAULT_IDLE_TIMEOUT_SECS` env value into an idle-watchdog duration.
///
/// - unset / "0" / unparseable  -> disabled (None). DEFAULT IS OFF since #748:
///   inactivity is NOT proof of abandonment — a quiet-but-alive host (Claude
///   Desktop routinely goes many minutes between tool calls and never respawns
///   a dead server) must never be reaped. Parent death is detected
///   deterministically by the orphan watcher (PDEATHSIG on Linux, ppid poll
///   everywhere else), so the flat timer is no longer the orphan guard.
/// - "N" (N > 0)                -> Some(N seconds): OPT-IN aggressive reaping,
///   for the one topology parent-death detection cannot see: a host that leaks
///   the child's stdin write-end while STAYING ALIVE (the original #57228
///   Hermes-worker reconnect leak). Hosts with that lifecycle should set this
///   when spawning the server.
///
/// Factored out of `run_server` so the watchdog policy is unit-tested.
pub fn parse_idle_timeout(raw: Option<&str>) -> Option<std::time::Duration> {
    match raw {
        Some(v) => match v.trim().parse::<u64>() {
            Ok(0) => None,
            Ok(secs) => Some(std::time::Duration::from_secs(secs)),
            Err(_) => {
                eprintln!(
                    "perseus-vault: ignoring unparseable PERSEUS_VAULT_IDLE_TIMEOUT_SECS value {:?} — idle watchdog disabled",
                    v
                );
                None
            }
        },
        None => None,
    }
}

/// Parse one raw JSON-RPC line, dispatch it, and write the response (if any)
/// to stdout. Shared by the read loop and the #1045 pending-request path so a
/// forwarded in-flight request gets exactly the same treatment as a live one.
fn process_request_line(line: &str, state: &MCPState, db: &Database, stdout: &mut std::io::Stdout) {
    let request: JsonRpcRequest = match serde_json::from_str(line) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("perseus-vault: JSON parse error: {} in line: {}", e, line);
            let error_response = json!({
                "jsonrpc": "2.0",
                "id": Value::Null,
                "error": {"code": -32700, "message": format!("Parse error: {}", e)}
            });
            let _ = writeln!(stdout, "{}", error_response);
            let _ = stdout.flush();
            return;
        }
    };

    let response = handle_request(&request, state, db);
    if let Some(resp) = response {
        let resp_str = serde_json::to_string(&resp).unwrap_or_else(|_| {
            json!({
                "jsonrpc": "2.0",
                "id": request.id,
                "error": {"code": -32603, "message": "Internal error: serialization failed"}
            })
            .to_string()
        });
        let _ = writeln!(stdout, "{}", resp_str);
        let _ = stdout.flush();
    }
}

/// Run the MCP server loop: read JSON-RPC from stdin, write responses to stdout.
///
/// Takes `Arc<Database>` (#402) so main.rs can hand the SAME pooled Database
/// to the web dashboard / gRPC surfaces instead of each opening a second
/// `Database` (a second 16-conn pool) on the same file.
/// Run the MCP server loop with the historical full advertisement profile.
pub fn run_server(db: std::sync::Arc<Database>) {
    run_server_with_profile(db, ToolProfile::Default);
}

/// Run the MCP server loop with an explicit advertisement profile.
pub fn run_server_with_profile(db: std::sync::Arc<Database>, profile: ToolProfile) {
    // Capture the baseline parent PID immediately, before anything can reparent
    // us. is_orphaned_by_ppid() compares against this so a process legitimately
    // born under a PID-1 container entrypoint is not mistaken for an orphan (#547
    // follow-up: fixes the demo-container crash loop).
    record_initial_ppid();

    let mut stdout = std::io::stdout();
    let state = MCPState::new_with_profile(profile);

    // #1045: a handoff child resumes the forwarded session — the client
    // already initialized the pre-handoff image and never re-sends
    // `initialize`, and the transport-captured agent identity must survive.
    restore_handoff_session(&state);

    // #1045: the replacement image may carry a prepared response
    // (PERSEUS_VAULT_HANDOFF_PENDING_RESPONSE) for the explicit handoff call
    // — the client is blocked on exactly this report. Write it before
    // entering the read loop.
    if let Some(resp_str) = crate::live_update::take_handoff_pending_response() {
        let _ = writeln!(stdout, "{}", resp_str);
        let _ = stdout.flush();
    }

    // Idle watchdog — OPT-IN since #748 (PERSEUS_VAULT_IDLE_TIMEOUT_SECS, default off).
    //
    // The original #57228 guard treated 600s of silence as proof of orphanhood.
    // That proxy is wrong on every platform without a real parent-death signal
    // (macOS/Windows had none): Claude Desktop goes quiet for long stretches in
    // normal use and — critically — never respawns a server that exits, so the
    // timer was silently killing healthy sessions and forcing a full app
    // restart. True orphans are now caught deterministically by parent-death
    // detection (PR_SET_PDEATHSIG on Linux; the ppid watcher thread below on
    // all Unix), so the flat timer remains only as an opt-in for hosts that
    // leak a child's stdin write-end while STAYING ALIVE (the actual #57228
    // Hermes-worker topology) — those hosts set PERSEUS_VAULT_IDLE_TIMEOUT_SECS when
    // spawning. EOF on stdin (well-behaved host shutdown) exits regardless.
    let idle_timeout: Option<std::time::Duration> = parse_idle_timeout(
        std::env::var("PERSEUS_VAULT_IDLE_TIMEOUT_SECS")
            .ok()
            .as_deref(),
    );

    // Read stdin on a dedicated thread so the main loop can time out on silence.
    let (tx, rx) = std::sync::mpsc::channel::<std::io::Result<String>>();
    std::thread::spawn(move || {
        let stdin = std::io::stdin();
        let reader = BufReader::new(stdin.lock());
        for line in reader.lines() {
            // If the main loop has exited (idle timeout), the receiver is dropped
            // and send() errors — stop reading and let this thread end.
            if tx.send(line).is_err() {
                break;
            }
        }
        // EOF: closing tx makes the main loop's recv return Disconnected.
    });

    eprintln!("perseus-vault: MCP server ready");

    // --- Deterministic parent-death detection (Linux, fixes #547) ---
    //
    // PR_SET_PDEATHSIG makes the kernel send SIGTERM to this process the
    // instant its parent dies, regardless of pipe/traffic state. This closes
    // the race that defeats the idle watchdog: a leaked write-end of stdin
    // held by a still-live sibling keeps recv_timeout() marginally fed so
    // the idle timer never elapses, yet the spawning parent is already dead.
    //
    // After setting the signal we re-check is_orphaned_by_ppid() immediately:
    // if the parent died in the window between fork() and prctl() we exit now
    // rather than blocking forever (the signal delivery already happened
    // before the prctl so we would never receive it). This compares the live
    // ppid against the baseline captured at start, so a server born directly
    // under a PID-1 container entrypoint is NOT treated as orphaned.
    #[cfg(target_os = "linux")]
    {
        if !crate::live_update::handoff_child() {
            unsafe {
                // PR_SET_PDEATHSIG = 1; SIGTERM = 15.  Using the raw constants
                // avoids pulling in the full `nix` crate just for this call.
                libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGTERM, 0, 0, 0);
            }
            if is_orphaned_by_ppid() {
                eprintln!("perseus-vault: parent already dead at server start — exiting (orphan-reap race guard, #547)");
                return;
            }
        }
    }

    // --- Parent-death watcher thread (Unix + Windows; primary orphan signal) ---
    //
    // The per-request ppid poll in the loop below can only run WHEN TRAFFIC
    // ARRIVES — an orphaned server sitting idle in recv() would never notice
    // its parent died. This thread polls the same orphan check on a 5s timer
    // and exits promptly, so abandonment detection works with zero traffic.
    // On Linux it backs up PR_SET_PDEATHSIG (which seccomp/kernels can filter);
    // on macOS it is the ONLY parent-death signal (reparent-to-launchd poll,
    // #748); on Windows it polls OpenProcess + creation-time on the recorded
    // parent PID (#751). It is what lets the idle watchdog default to OFF: the
    // server dies iff its host actually died, never merely because the host
    // went quiet (Claude Desktop neither pings nor respawns).
    #[cfg(any(unix, windows))]
    {
        std::thread::spawn(|| loop {
            std::thread::sleep(std::time::Duration::from_secs(5));
            if is_orphaned_by_ppid() {
                eprintln!(
                    "perseus-vault: parent process died — exiting orphaned stdio server (orphan watcher, #547/#748)"
                );
                // process::exit from the watcher: the main loop is blocked in
                // recv() and cannot be woken without traffic. SQLite is in WAL
                // mode, so skipping destructor-driven pool shutdown is safe.
                std::process::exit(0);
            }
        });
    }

    // #1045: the replacement image may carry the auto-handoff's in-flight
    // request (PERSEUS_VAULT_HANDOFF_PENDING_REQUEST) — the client is blocked
    // on exactly this response. Process it before entering the read loop.
    if let Some(pending_line) = crate::live_update::take_handoff_pending_request() {
        process_request_line(&pending_line, &state, &db, &mut stdout);
    }

    loop {
        let line = match idle_timeout {
            Some(timeout) => match rx.recv_timeout(timeout) {
                Ok(l) => l,
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                    eprintln!(
                        "perseus-vault: no client activity for {}s — exiting idle stdio server (orphan-leak guard, #57228)",
                        timeout.as_secs()
                    );
                    break;
                }
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
            },
            None => match rx.recv() {
                Ok(l) => l,
                Err(_) => break,
            },
        };

        let line = match line {
            Ok(l) => l,
            Err(e) => {
                eprintln!("perseus-vault: stdin read error: {}", e);
                break;
            }
        };

        if line.trim().is_empty() {
            continue;
        }

        // Ppid poll: if we have been reparented to init our spawning parent is
        // gone. PR_SET_PDEATHSIG above handles the common case, but on Linux
        // kernels that ignore the signal or on non-Linux platforms this is the
        // deterministic fallback. One getppid() syscall per request is negligible.
        if is_orphaned_by_ppid() {
            eprintln!(
                "perseus-vault: ppid == 1 detected — parent died, exiting (orphan-reap, #547)"
            );
            break;
        }

        process_request_line(&line, &state, &db, &mut stdout);

        // #858/#1045: live-update handoff — runs AFTER the response is
        // flushed (explicit handoff tool) or INSTEAD of a response (an
        // auto-handoff intercepted the in-flight call and suppressed it; the
        // replacement image answers it). Hoisted out of the response write so
        // a suppressed response still hands off.
        if crate::live_update::handoff_pending() {
            let initialized = state.initialized.load(std::sync::atomic::Ordering::Relaxed);
            let agent = state
                .session_agent_id
                .read()
                .map(|s| s.clone())
                .unwrap_or_default();
            match crate::live_update::perform_handoff(initialized, &agent) {
                // Windows spawn path: the replacement child owns the session.
                Ok(()) => std::process::exit(0),
                // Unix exec failure: keep serving — the fail-loud stale gate
                // stays active and the client can retry or restart.
                Err(e) => {
                    eprintln!(
                        "perseus-vault: handoff failed: {e} — continuing on the stale image (fail-loud gate active)"
                    );
                    crate::live_update::clear_handoff_pending();
                }
            }
        }
    }
}

pub fn handle_request(
    req: &JsonRpcRequest,
    state: &MCPState,
    db: &Database,
) -> Option<JsonRpcResponse> {
    let id = req.id.clone();

    if req.jsonrpc != "2.0" {
        return Some(error_response(
            id,
            -32600,
            "Invalid Request: jsonrpc must be \"2.0\"",
        ));
    }

    match req.method.as_str() {
        "initialize" => {
            if state
                .initialized
                .swap(true, std::sync::atomic::Ordering::AcqRel)
            {
                return Some(error_response(
                    id,
                    -32002,
                    "Already initialized; session identity cannot be replaced",
                ));
            }
            let response = JsonRpcResponse {
                jsonrpc: "2.0".to_string(),
                id,
                result: Some(json!({
                    "protocolVersion": "2025-06-18",
                    "serverInfo": {
                        // Tracks Cargo.toml's package name automatically, so a
                        // future rename doesn't leave this handshake reporting
                        // stale branding (it was hardcoded through the earlier
                        // product renames).
                        "name": env!("CARGO_PKG_NAME"),
                        "version": env!("CARGO_PKG_VERSION")
                    },
                    "capabilities": {
                        "tools": {
                            "listChanged": false
                        }
                    }
                })),
                error: None,
            };
            // #684: capture the client's identity from the handshake so
            // subsequent tool calls can be attributed/visibility-scoped. MCP
            // clientInfo.name (e.g. the agent's name); sanitized to a bounded
            // token. Absent clientInfo → stays empty → unscoped.
            if let Some(name) = req
                .params
                .as_ref()
                .and_then(|p| p.get("clientInfo"))
                .and_then(|c| c.get("name"))
                .and_then(|n| n.as_str())
            {
                let sanitized: String = name
                    .trim()
                    .chars()
                    .filter(|c| c.is_alphanumeric() || matches!(c, '-' | '_' | '.' | ':'))
                    .take(128)
                    .collect();
                if let Ok(mut slot) = state.session_agent_id.write() {
                    *slot = sanitized;
                }
            }
            Some(response)
        }

        "notifications/initialized" => {
            // Notification — no response
            None
        }

        "tools/list" => {
            if !state.initialized.load(std::sync::atomic::Ordering::Relaxed) {
                return Some(error_response(id, -32002, "Not initialized"));
            }
            Some(list_tools(id, state.profile))
        }

        "tools/call" => {
            if !state.initialized.load(std::sync::atomic::Ordering::Relaxed) {
                return Some(error_response(id, -32002, "Not initialized"));
            }

            let params = match &req.params {
                Some(p) => p,
                None => return Some(error_response(id, -32602, "Missing params")),
            };

            let tool_name = match params.get("name").and_then(|v| v.as_str()) {
                Some(n) => n,
                None => return Some(error_response(id, -32602, "Missing tool name")),
            };

            // #1045: transparent auto-handoff — when the binary was replaced
            // and PERSEUS_VAULT_AUTO_HANDOFF=1, hand the session to the new
            // image WITH this request attached instead of refusing with the
            // loud isError: the replacement process answers this very call,
            // so the client sees one clean response and no error. The
            // handoff tool and health stay exempt (stale_error_message is
            // None for them); oversized requests fall through to the loud
            // path.
            if crate::live_update::auto_handoff_enabled()
                && crate::live_update::stale_error_message(tool_name).is_some()
            {
                let pending = serde_json::to_string(req).ok();
                if crate::live_update::schedule_auto_handoff(pending) {
                    // No response here: the replacement image answers the
                    // forwarded request on the same stdio connection.
                    return None;
                }
            }

            let mut tool_args = params.get("arguments").cloned().unwrap_or(json!({}));
            if tool_name == "perseus_vault_workspace_status" && !tool_args.is_object() {
                tool_args = json!({});
            }

            // #1045: window-free explicit handoff. With confirm:true on a
            // stale image, the old process must NOT write the report itself
            // — the replacement image writes it (forwarded via
            // schedule_handoff_with_response). The pre-fix sequence (flush
            // report, then exec) left a gap in which a client's next request
            // could be consumed by the dying image's stdin BufReader and
            // vanish, making the session look dead even though the swap
            // succeeded.
            if tool_name == "perseus_vault_handoff_restart" {
                let confirm = tool_args
                    .get("confirm")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                let dry_run = tool_args
                    .get("dry_run")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                if confirm && !dry_run && crate::live_update::running_stale() {
                    let report_text =
                        match crate::live_update::handle_handoff_restart(tool_args.clone()) {
                            Ok(t) => t,
                            Err(e) => e,
                        };
                    let structured: Option<serde_json::Value> =
                        serde_json::from_str(&report_text).ok();
                    let mut result = json!({
                        "content": [{ "type": "text", "text": report_text }]
                    });
                    if let Some(parsed) = structured {
                        if let Some(ie) = parsed.get("isError") {
                            result["isError"] = ie.clone();
                        }
                        result["structuredContent"] = parsed;
                    }
                    let resp = JsonRpcResponse {
                        jsonrpc: "2.0".to_string(),
                        id: id.clone(),
                        result: Some(result),
                        error: None,
                    };
                    if let Ok(resp_str) = serde_json::to_string(&resp) {
                        if crate::live_update::schedule_handoff_with_response(resp_str) {
                            // No response written here — the replacement
                            // image writes the report after the swap.
                            return None;
                        }
                    }
                    // Oversized/unserializable: fall through to normal
                    // dispatch (report written by the stale image; the
                    // fail-loud gate stays up).
                }
            }

            // #684: stamp the captured session identity so tools that enforce
            // visibility (recall) know who is asking, without the caller having
            // to pass it. #855: the transport-captured host identity is
            // AUTHORITATIVE — a caller-supplied `requesting_agent_id`
            // (model-forged or empty) is overwritten, never trusted, so no
            // model can claim another agent's identity.
            if let Ok(sid) = state.session_agent_id.read() {
                if let Some(obj) = tool_args.as_object_mut() {
                    if sid.trim().is_empty() {
                        // No transport identity: caller-supplied requester fields
                        // are untrusted and must not survive into a public read.
                        obj.remove("requesting_agent_id");
                    } else {
                        obj.insert("requesting_agent_id".to_string(), json!(*sid));
                    }
                    // #1182: task-state scope identities are transport-owned
                    // too. Never let a caller provide a second principal/agent
                    // inside the nested projection request.
                    if tool_name == "perseus_vault_project_task" {
                        if let Some(task_state) =
                            obj.get_mut("task_state").and_then(Value::as_object_mut)
                        {
                            if sid.trim().is_empty() {
                                task_state.remove("principal_id");
                                task_state.remove("agent_id");
                            } else {
                                task_state.insert("principal_id".to_string(), json!(*sid));
                                task_state.insert("agent_id".to_string(), json!(*sid));
                            }
                        }
                    }
                }
            }

            // Workspace status is a caller-scoped diagnostic for every ordinary
            // MCP profile. The marker is injected after transport identity
            // stamping so callers cannot widen it to the all-bindings
            // administrator view. Normalize null/non-object arguments before
            // stamping so argument shape cannot bypass the boundary.
            if tool_name == "perseus_vault_workspace_status" {
                if let Some(obj) = tool_args.as_object_mut() {
                    obj.insert("status_scope".to_string(), json!("caller"));
                    let has_workspace = obj
                        .get("workspace_hash")
                        .and_then(Value::as_str)
                        .is_some_and(|ws| !ws.trim().is_empty());
                    if !has_workspace {
                        if let Some(profile) = obj
                            .get("requesting_agent_id")
                            .and_then(Value::as_str)
                            .filter(|profile| !profile.trim().is_empty())
                        {
                            if let Ok(Some(binding)) = db.workspace_binding_for(profile) {
                                obj.insert(
                                    "workspace_hash".to_string(),
                                    json!(binding.workspace_hash),
                                );
                            }
                        }
                    }
                }
            }

            // Admission provenance and review decisions cannot fall back to a
            // caller-supplied requesting_agent_id. These operations require the
            // identity captured from MCP initialize.clientInfo.name; otherwise
            // an uninitialized caller could choose the reviewer/source agent.
            let admission_source_call = tool_name == "perseus_vault_journal"
                && tool_args.get("event_type").and_then(Value::as_str) == Some("admission_source");
            let admission_review_call = tool_name == "perseus_vault_admission_decide";
            let admission_write_call =
                tool_name == "perseus_vault_remember" && tool_args.get("admission").is_some();
            if admission_source_call || admission_review_call || admission_write_call {
                let captured_session = state
                    .session_agent_id
                    .read()
                    .map(|sid| !sid.trim().is_empty())
                    .unwrap_or(false);
                if !captured_session {
                    return Some(JsonRpcResponse {
                        jsonrpc: "2.0".to_string(),
                        id,
                        result: Some(json!({
                            "content": [{
                                "type": "text",
                                "text": "admission tools require an initialized MCP session with clientInfo.name"
                            }],
                            "isError": true
                        })),
                        error: None,
                    });
                }
            }

            // #879: enforce profile <-> workspace bindings at the tool
            // boundary. The transport-stamped requesting_agent_id is the
            // Hermes profile name; the binding registry is vault-authoritative.
            // Mutations on the scoped surface deny cross-workspace targets,
            // read_only bindings, and quarantined/unbound bindings. Reads
            // deny cross-workspace targets when the caller names a workspace.
            // Unbound profiles keep the legacy unscoped behavior (binding is
            // an opt-in governance surface).
            {
                const SCOPE_MUTATION_TOOLS: &[&str] = &[
                    "perseus_vault_provider_source_event",
                    "perseus_vault_declared_graph_manifest",
                    "perseus_vault_declared_graph_attest",
                    "perseus_vault_remember",
                    "perseus_vault_journal",
                    "perseus_vault_reject_value",
                    "perseus_vault_forget",
                    "perseus_vault_link",
                    "perseus_vault_unlink",
                    "perseus_vault_supersede",
                    "perseus_vault_state_set",
                    "perseus_vault_embed",
                    "perseus_vault_artifact_register",
                    "perseus_vault_learned_artifact_register",
                    "perseus_vault_expire",
                    "perseus_vault_redact",
                    "perseus_vault_erase",
                    "perseus_vault_correct",
                    "perseus_vault_follow",
                    "perseus_vault_write_quarantine",
                    "perseus_vault_admission_decide",
                    "perseus_vault_web_gap_fill",
                    "perseus_vault_experience_projection_rebuild",
                ];
                const SCOPE_READ_TOOLS: &[&str] = &[
                    "perseus_vault_recall",
                    "perseus_vault_declared_graph_query",
                    "perseus_vault_recall_batch",
                    "perseus_vault_recall_layer",
                    "perseus_vault_semantic_search",
                    "perseus_vault_recall_when",
                    "perseus_vault_scan",
                    "perseus_vault_context",
                    "perseus_vault_project_task",
                    "perseus_vault_experience_projection",
                    "perseus_vault_expand_source",
                    "perseus_vault_ask",
                    "perseus_vault_artifact_manifest",
                    "perseus_vault_artifact_excerpt",
                    "perseus_vault_artifact_verify_value",
                    "perseus_vault_handoff_pack",
                    "perseus_vault_delegation_brief",
                    "perseus_vault_workspace_status",
                    "perseus_vault_typed_traversal",
                ];
                let profile = tool_args
                    .get("requesting_agent_id")
                    .and_then(|v| v.as_str());
                let ws = tool_args.get("workspace_hash").and_then(|v| v.as_str());
                let denied = if tool_name == "perseus_vault_typed_traversal" {
                    db.enforce_strict_workspace_binding(profile, ws, false)
                        .err()
                        .map(|e| e.to_string())
                } else if state.strict_scope && SCOPE_MUTATION_TOOLS.contains(&tool_name) {
                    db.enforce_strict_workspace_binding(profile, ws, true)
                        .err()
                        .map(|e| e.to_string())
                } else if state.strict_scope && SCOPE_READ_TOOLS.contains(&tool_name) {
                    db.enforce_strict_workspace_binding(profile, ws, false)
                        .err()
                        .map(|e| e.to_string())
                } else if SCOPE_MUTATION_TOOLS.contains(&tool_name) {
                    db.enforce_workspace_binding(profile, ws, true)
                        .err()
                        .map(|e| e.to_string())
                } else if SCOPE_READ_TOOLS.contains(&tool_name) {
                    db.enforce_workspace_binding(profile, ws, false)
                        .err()
                        .map(|e| e.to_string())
                } else {
                    None
                };
                if let Some(msg) = denied {
                    return Some(JsonRpcResponse {
                        jsonrpc: "2.0".to_string(),
                        id,
                        result: Some(json!({
                            "content": [{ "type": "text", "text": msg }],
                            "isError": true,
                        })),
                        error: None,
                    });
                }
            }

            // v23 (Chancery cross-ref, #6): extract the chancery writ ID from
            // `_meta.chancery/lease` on the tools/call params envelope. When
            // Chancery wraps an MCP server it stamps every request with this so
            // the vault can record the writ in its journal audit trail. Set on a
            // thread-local so `db.journal()` picks it up without threading it
            // through every handler.
            let chancery_writ_id = params
                .get("_meta")
                .and_then(|m| m.get("chancery/lease"))
                .and_then(|v| v.as_str())
                .unwrap_or("");
            crate::db::set_chancery_writ_id(chancery_writ_id);

            let result_text = call_tool(tool_name, db, tool_args, id.clone());

            // Try to parse the result as JSON for structuredContent
            let structured: Option<serde_json::Value> = serde_json::from_str(&result_text).ok();
            let mut result = json!({
                "content": [{
                    "type": "text",
                    "text": result_text
                }]
            });
            // Copy isError through, then move the parsed value into
            // structuredContent rather than deep-cloning the whole result (#208).
            if let Some(parsed) = structured {
                if let Some(is_err) = parsed.get("isError") {
                    result["isError"] = is_err.clone();
                }
                result["structuredContent"] = parsed;
            }
            Some(JsonRpcResponse {
                jsonrpc: "2.0".to_string(),
                id,
                result: Some(result),
                error: None,
            })
        }

        _ => Some(error_response(
            id,
            -32601,
            &format!("Method not found: {}", req.method),
        )),
    }
}

/// Parse-once cache of the canonical Perseus Vault tool registry. The embedded
/// literal is the single source of truth for tool schemas; no legacy aliases
/// are synthesized.
fn tool_registry_base() -> &'static Vec<serde_json::Value> {
    static BASE: OnceLock<Vec<serde_json::Value>> = OnceLock::new();
    BASE.get_or_init(|| {
        let registry = serde_json::from_str::<serde_json::Value>(
        r###"[
  {
    "name": "perseus_vault_remember",
    "description": "Store or update an entity by (category, key). Idempotent — call as often as you want, same key returns an update. NEAR-DUPLICATE MERGING (#531): a NEW key whose body is >=70% trigram-similar to an existing entity in the same category+workspace does NOT create a new entity — the write is folded into the existing one (result: action='deduped', deduped=true, merged_into=<id>). Right for conversational memory; wrong for bulk ingest of templated records, which are similar by construction and will silently collapse to a handful of rows. For bulk ingest pass skip_dedup=true (or use perseus_vault_ingest_file), and check the returned action. Prefer recall_when triggers (retrieve when relevant) over always_on=true (inject unconditionally): the recall-first perseus_vault_context hard-caps the always-on set and warns when it overflows, so reserve always_on for genuinely identity-critical facts. Optional certainty (0.0-1.0) is used by perseus_vault_conflicts for typed-entity conflict detection. Pass derived_from (ids or {category,key} pairs of the memories you recalled) to auto-mark those sources useful — cited memories rank higher and decay slower. Use this for saving facts, decisions, architecture notes, and conventions. Optional hints (#919): 1-3 prospective query phrasings that should retrieve this entity (vocabulary-gap recall) — indexed into FTS5 alongside the body, default-off (PERSEUS_VAULT_HINTS_ENABLED=1 to enable), replaced wholesale on update. When encryption is enabled, body_json is encrypted at rest with AES-256-GCM.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "category": {
          "type": "string",
          "description": "Entity category: 'decision', 'architecture', 'convention', 'insight', or custom"
        },
        "key": {
          "type": "string",
          "description": "Unique key within the category, e.g. 'use-postgres-16' or 'deployment-strategy'"
        },
        "body_json": {
          "type": "string",
          "description": "JSON object with the entity body — store content, summary, and any custom fields here"
        },
        "status": {
          "type": "string",
          "enum": ["active", "draft", "deprecated", "expired", "proposed", "quarantined", "redacted"],
          "default": "active",
          "description": "Closed lifecycle status vocabulary; proposed/quarantined are never publicly serveable"
        },
        "type": {
          "type": "string",
          "default": "insight",
          "description": "Entity type: 'insight', 'architecture', 'decision', 'reference', 'convention'"
        },
        "tags": {
          "type": "array",
          "items": {
            "type": "string"
          },
          "description": "Tags for categorization and cross-referencing"
        },
        "importance": {
          "type": "number",
          "default": 0.5,
          "description": "Initial importance 0.0–1.0 — sets the starting decay score"
        },
        "topic_path": {
          "type": "string",
          "default": "",
          "description": "Hierarchical topic path, e.g. 'architecture/database/postgres'"
        },
        "workspace_hash": {
          "type": "string",
          "default": "",
          "description": "Workspace scope identifier (v1.2.0). Empty = global. Entities with a workspace_hash are invisible to recall queries scoped to a different workspace."
        },
        "agent_id": {
          "type": "string",
          "default": "",
          "description": "Agent identity (v1.2.0). Tracks which agent wrote this entity. Used for agent attribution and context filtering."
        },
        "actor_kind": {
          "type": "string",
          "default": "assistant",
          "description": "Actor basis for the write (for example assistant, user, connector, or system). Missing admission stays reviewable."
        },
        "admission": {
          "type": "object",
          "description": "Hash-only admission envelope. The server emits one stable outcome_class: save, drop, block, or pending_approval. Authoritative admission requires a validated source_event_id and matching workspace; missing or unverified evidence is retained as proposed/requires_review and is not serveable. DROP/BLOCK decisions return hash-only evidence without persisting the candidate.",
          "properties": {
            "record_digest": {"type": "string"},
            "source_identity": {"type": "string"},
            "authorization_scope": {"type": "string"},
            "ingestion_channel": {"type": "string"},
            "workspace_hash": {"type": "string"},
            "source_trust": {"type": "string", "enum": ["untrusted", "trusted", "authoritative"]},
            "source_event_id": {"type": "string"},
            "actor_kind": {"type": "string"},
            "actor_identity": {"type": "string"},
            "validated": {"type": "boolean"},
            "valid_from_unix_ms": {"type": "integer"},
            "recorded_at_unix_ms": {"type": "integer"},
            "task_relevance_bps": {"type": "integer"},
            "instruction_bearing": {"type": "boolean"},
            "contradicts_authoritative": {"type": "boolean"}
          }
        },
        "valid_from_unix_ms": {
          "type": "integer",
          "description": "Application-time period start (#363): when the fact became TRUE IN THE WORLD, independent of when it was recorded. Set in the past for retroactive facts ('this was true last week, we just learned it') without rewriting transaction history. Default: transaction time (now). Query with perseus_vault_valid_at / perseus_vault_bitemporal / recall's valid_at filter."
        },
        "valid_to_unix_ms": {
          "type": "integer",
          "description": "Application-time period end (#363, exclusive): when the fact STOPPED being true in the world. Omit for 'still true' (unbounded). Must be greater than valid_from_unix_ms."
        },
        "skip_dedup": {
          "type": "boolean",
          "default": false,
          "description": "Opt out of near-duplicate merging for this write (#531). Set true for bulk/API ingest of templated records so every acknowledged write actually creates its key; leave false for conversational memory."
        },
        "allow_rejected": {
          "type": "boolean",
          "default": false,
          "description": "#849: deliberate trusted override of a rejected-value tombstone. Journaled as an audited override; never set automatically."
        },
        "derived_from": {
          "type": "array",
          "items": {
            "oneOf": [
              {
                "type": "string",
                "description": "Entity id of a cited source, e.g. 'mem-a1b2c3d4e5f6' (as returned by recall/remember)"
              },
              {
                "type": "object",
                "properties": {
                  "category": { "type": "string" },
                  "key": { "type": "string" }
                },
                "required": ["category", "key"],
                "description": "A cited source addressed by (category, key)"
              }
            ]
          },
          "description": "#487: the memories this write was built on (max 64). Each cited source is automatically marked useful — usefulness_count bumped, last_useful/last_accessed refreshed — so memories that actually inform later writes rank higher in recall and decay slower. Cite the entities you recalled before composing this write. Unknown citations are reported in the result, not fatal; self-citations are ignored."
        },
        "origin": {
          "type": "object",
          "properties": {
            "memory_kind": { "type": "string", "enum": ["asserted", "extracted", "inferred", "imported", "observed"] },
            "source_system": { "type": "string" },
            "capture_method": { "type": "string" },
            "observed_at_unix_ms": { "type": "integer" }
          },
          "description": "#729: optional memory-origin/provenance metadata (spec: docs/specs/memory-provenance-and-external-refs.md). Stored inside body_json under the reserved 'origin' key — surfaced by recall/get_entity via body expansion. All fields optional; unknown values are left absent, never guessed."
        },
        "external_refs": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "ref_type": { "type": "string" },
              "ref_value": { "type": "string" },
              "source_system": { "type": "string" },
              "relationship": { "type": "string", "enum": ["about", "derived_from", "mentions", "applies_to", "supersedes"] }
            },
            "required": ["ref_type", "ref_value"]
          },
          "description": "#728: optional first-class pointers to external systems of record (max 32). Stored inside body_json under the reserved 'external_refs' key; filter recall with ref_type/ref_value."
        },
        "evidence": {
          "type": "object",
          "description": "Write-time audit envelope for captures and decisions. capture_mode distinguishes snapshot, hash_only, pointer_only, not_requested, capture_failed, and legacy_unknown; a missing value is never interpreted implicitly.",
          "properties": {
            "capture_mode": { "type": "string", "enum": ["snapshot", "hash_only", "pointer_only", "not_requested", "capture_failed", "legacy_unknown"] },
            "resolved_value": { "description": "Resolved source value retained at write time when capture_mode=snapshot" },
            "content_sha256": { "type": "string", "description": "64-hex SHA-256 of the resolved value or source bytes" },
            "source_system": { "type": "string" },
            "source_ref": { "type": "string" },
            "captured_at_unix_ms": { "type": "integer" },
            "replayable": { "type": "boolean" }
          },
          "required": ["capture_mode", "captured_at_unix_ms", "replayable"]
        },
        "interference_mode": {
          "type": "string",
          "enum": ["auto", "refuse", "quarantine"],
          "default": "auto",
          "description": "#874: per-write interference-gate mode override. auto (default) uses the operator-configured mode (PERSEUS_VAULT_INTERFERENCE_MODE); refuse/quarantine tighten it per-write. Per-write 'off' is refused fail-closed — only the operator can disable the gate."
        },
        "interference_bound": {
          "type": "number",
          "minimum": 0,
          "maximum": 1,
          "description": "#874: per-write interference bound override — may only TIGHTEN the configured bound (PERSEUS_VAULT_INTERFERENCE_BOUND); a looser bound is refused fail-closed. Writes whose activation overlap with existing memory exceeds the bound are quarantined (default) or refused."
        },
        "sparse_update": {
          "type": "boolean",
          "default": false,
          "description": "#874: sparse update mode — touches only the activated subset of state (body slot, activated links), never disturbs neighbors: no salience inflation on re-assert, caller links admitted only when their target is activated by the new body, no near-duplicate absorption on insert."
        },
        "hints": {
          "type": "array",
          "items": {
            "type": "string"
          },
          "maxItems": 3,
          "description": "#919: optional 1-3 prospective query hints — natural-language phrasings that should retrieve this entity, indexed into FTS5 alongside the body (vocabulary-gap recall). Default-off: hints are rejected unless the server runs with PERSEUS_VAULT_HINTS_ENABLED=1. Hints replace any previously stored hints on update (omit to clear)."
        }
      },
      "required": [
        "category",
        "key",
        "body_json"
      ]
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "id": {
          "type": "string",
          "description": "Entity ID, e.g. 'mem-a1b2c3d4e5f6'"
        },
        "action": {
          "type": "string",
          "description": "'created' for new entities, 'updated' for existing ones"
        },
        "category": {
          "type": "string",
          "description": "Entity category"
        },
        "key": {
          "type": "string",
          "description": "Entity key"
        },
        "derived_from": {
          "type": "object",
          "description": "Present when derived_from citations were passed: {reinforced: n, not_found: [labels]}"
        },
        "proposed": {
          "type": "boolean",
          "description": "True when the write lacks authoritative admission and must remain reviewable."
        },
        "requires_review": {
          "type": "boolean",
          "description": "Whether the stored write must be reviewed before promotion or authoritative use."
        },
        "provenance": {
          "type": "object",
          "description": "Hash-only admission/provenance state; raw prompts, bodies, credentials, and tool arguments are excluded."
        },
        "outcome_class": {
          "type": "string",
          "enum": ["save", "drop", "block", "pending_approval"],
          "description": "Stable four-way admission result. SAVE is durably active; DROP and BLOCK are non-persisting terminal decisions; PENDING_APPROVAL is retained but non-serveable until review."
        },
        "disposition": {
          "type": "string",
          "description": "Existing detailed disposition, such as quarantined; use outcome_class for stable aggregation."
        },
        "admission": {
          "type": "object",
          "description": "Hash-covered, content-minimized admission evidence."
        }
      }
    },
    "annotations": {
      "destructiveHint": true
    },
    "title": "Remember Entity"
  },
  {
    "name": "perseus_vault_write_gate",
    "description": "#939 zero-token write gate: deterministic keep/supersede/forget BEFORE LLM enrichment. Read-only precheck over (category, key, body) that decides store / duplicate / supersede / forget / adjudicate from content-hash + stored-signature near-duplicate scans and an importance floor — ZERO LLM tokens. Only 'adjudicate' (a near-duplicate that may be a contradiction) should escalate to the LLM or operator review. Call this before the enrichment pass to cut per-write Ollama load.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "category": {
          "type": "string",
          "description": "Entity category of the candidate write."
        },
        "key": {
          "type": "string",
          "description": "Entity key of the candidate write."
        },
        "body_json": {
          "type": "string",
          "description": "Serialized body of the candidate write."
        },
        "workspace_hash": {
          "type": "string",
          "description": "Optional workspace scope for the scans."
        }
      },
      "required": ["category", "key", "body_json"]
    },
    "annotations": {
      "readOnlyHint": true
    },
    "title": "Write Gate"
  },
  {"name": "perseus_vault_provider_source_event", "description": "Apply a versioned provider-native source event. Preserves stable provider identity, revision and content digests, timestamps, thread or parent lineage, visibility and workspace scope, and governed deletion tombstones. The envelope never accepts or stores raw provider bodies or payloads. Replaying the same provider/external_id/revision is idempotent.", "inputSchema": {"type": "object", "properties": {"schema_version": {"type": "integer", "const": 1}, "event_type": {"type": "string", "enum": ["upsert", "comment", "reply", "attachment", "delete"]}, "provider": {"type": "string"}, "kind": {"type": "string"}, "external_id": {"type": "string"}, "canonical_uri": {"type": "string"}, "thread_id": {"type": "string"}, "parent_id": {"type": "string"}, "provider_event_id": {"type": "string"}, "author": {"type": "string", "maxLength": 256}, "revision": {"type": "string"}, "expected_revision": {"type": "string"}, "observed_at_unix_ms": {"type": "integer", "minimum": 0}, "provider_created_at_unix_ms": {"type": "integer", "minimum": 0}, "provider_updated_at_unix_ms": {"type": "integer", "minimum": 0}, "content_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"}, "source_span_ref": {"type": "string"}, "workspace_hash": {"type": "string"}, "visibility": {"type": "string", "enum": ["private", "workspace", "public"]}, "retention_policy": {"type": "string"}, "capture_method": {"type": "string"}, "requesting_agent_id": {"type": "string", "description": "Transport-stamped identity; caller-supplied values are overwritten."}, "entity_id": {"type": "string"}}, "required": ["schema_version", "event_type", "provider", "kind", "external_id", "revision"]}, "outputSchema": {"type": "object", "properties": {"schema_version": {"type": "integer"}, "outcome": {"type": "string", "enum": ["applied", "idempotent", "revision_race", "deleted"]}, "event_type": {"type": "string"}, "event_id": {"type": "string"}, "receipt_digest": {"type": "string"}, "source": {"type": "object"}, "previous_revision": {"type": "string"}, "entity_archived": {"type": "boolean"}}}, "annotations": {"destructiveHint": true}, "title": "Provider Source Event"},
  {
    "name": "perseus_vault_declared_graph_manifest",
    "description": "Apply a versioned, source-keyed declared graph manifest. Stable node and edge IDs are scoped by workspace and canonical identity; replace revisions supersede prior active topology, while delete revisions create tombstones and preserve history. Declared edges remain sourced or supported until explicitly attested. No LLM extraction is performed.",
    "inputSchema": {
      "type": "object",
      "additionalProperties": false,
      "properties": {
        "schema_version": {"type": "integer", "const": 1},
        "operation": {"type": "string", "enum": ["upsert", "delete"]},
        "source_key": {"type": "string", "maxLength": 256},
        "revision": {"type": "string", "maxLength": 256},
        "content_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "source_span_ref": {"type": "string", "maxLength": 1024},
        "workspace_hash": {"type": "string", "maxLength": 256},
        "valid_from_unix_ms": {"type": "integer", "minimum": 0},
        "valid_to_unix_ms": {"type": "integer", "minimum": 0},
        "policy": {"type": "string", "const": "replace"},
        "nodes": {"type": "array", "maxItems": 256, "items": {"type": "object", "additionalProperties": false, "properties": {"namespace": {"type": "string", "maxLength": 128}, "canonical_id": {"type": "string", "maxLength": 512}, "node_type": {"type": "string", "maxLength": 128}, "external_ref": {"type": "string", "maxLength": 2048}}, "required": ["namespace", "canonical_id", "node_type"]}},
        "edges": {"type": "array", "maxItems": 512, "items": {"type": "object", "additionalProperties": false, "properties": {"from": {"type": "string"}, "to": {"type": "string"}, "predicate": {"type": "string", "maxLength": 128}, "direction": {"type": "string", "enum": ["forward", "reverse"]}, "context": {"type": "string", "maxLength": 1024}, "source_span_ref": {"type": "string", "maxLength": 1024}, "origin": {"type": "string", "const": "declared"}, "support_state": {"type": "string", "enum": ["sourced", "supported"]}, "valid_from_unix_ms": {"type": "integer", "minimum": 0}, "valid_to_unix_ms": {"type": "integer", "minimum": 0}}, "required": ["from", "to", "predicate", "direction", "origin", "support_state"]}},
        "requesting_agent_id": {"type": "string", "description": "Transport-stamped identity; caller-supplied values are overwritten."}
      },
      "required": ["schema_version", "operation", "source_key", "revision", "content_sha256", "workspace_hash", "policy"]
    },
    "outputSchema": {"type": "object", "properties": {"schema_version": {"type": "integer"}, "outcome": {"type": "string", "enum": ["applied", "idempotent"]}, "manifest_id": {"type": "string"}, "source_id": {"type": "string"}, "node_ids": {"type": "array", "items": {"type": "string"}}, "edge_ids": {"type": "array", "items": {"type": "string"}}, "edges": {"type": "array", "items": {"type": "object"}}}, "required": ["schema_version", "outcome", "manifest_id", "source_id", "node_ids", "edge_ids", "edges"]},
    "annotations": {"destructiveHint": true},
    "title": "Declared Graph Manifest"
  },
  {
    "name": "perseus_vault_declared_graph_attest",
    "description": "Explicitly attest selected active declared edges under an authority reference. Sourced or supported edges cannot become attested through ingestion alone; the selected edge IDs, manifest revision, attestor, and bounded reference are recorded and replay is idempotent.",
    "inputSchema": {"type": "object", "additionalProperties": false, "properties": {"schema_version": {"type": "integer", "const": 1}, "workspace_hash": {"type": "string", "maxLength": 256}, "source_key": {"type": "string", "maxLength": 256}, "revision": {"type": "string", "maxLength": 256}, "edge_ids": {"type": "array", "minItems": 1, "maxItems": 512, "items": {"type": "string", "maxLength": 128}}, "attestation_ref": {"type": "string", "maxLength": 1024}, "attested_by": {"type": "string", "maxLength": 256}, "requesting_agent_id": {"type": "string", "description": "Transport-stamped identity; caller-supplied values are overwritten."}}, "required": ["schema_version", "workspace_hash", "source_key", "revision", "edge_ids", "attestation_ref", "attested_by"]},
    "outputSchema": {"type": "object", "properties": {"schema_version": {"type": "integer"}, "outcome": {"type": "string", "enum": ["applied", "idempotent"]}, "manifest_id": {"type": "string"}, "edge_ids": {"type": "array", "items": {"type": "string"}}, "receipt_digest": {"type": "string"}}, "required": ["schema_version", "outcome", "manifest_id", "edge_ids", "receipt_digest"]},
    "annotations": {"destructiveHint": true},
    "title": "Attest Declared Graph Edges"
  },
  {
    "name": "perseus_vault_declared_graph_query",
    "description": "Read a bounded workspace-scoped projection of declared graph nodes and edges. Active-only output is the default; include_history exposes superseded and tombstoned revisions. Every edge carries source revision, digest/span, scope, origin, validity, and explicit attestation state. This is separate from ordinary recall and does not add graph traversal cost to normal queries.",
    "inputSchema": {"type": "object", "additionalProperties": false, "properties": {"workspace_hash": {"type": "string", "maxLength": 256}, "source_key": {"type": "string", "maxLength": 256}, "requesting_agent_id": {"type": "string", "description": "Transport-stamped identity; caller-supplied values are overwritten."}, "include_history": {"type": "boolean", "default": false}, "limit": {"type": "integer", "minimum": 1, "maximum": 500, "default": 100}}, "required": ["workspace_hash"]},
    "outputSchema": {"type": "object", "properties": {"schema_version": {"type": "integer"}, "workspace_hash": {"type": "string"}, "source_key": {"type": "string"}, "nodes": {"type": "array", "items": {"type": "object"}}, "edges": {"type": "array", "items": {"type": "object"}}, "truncated": {"type": "boolean"}}, "required": ["schema_version", "workspace_hash", "nodes", "edges", "truncated"]},
    "annotations": {"readOnlyHint": true},
    "title": "Query Declared Graph"
  },
  {
    "name": "perseus_vault_recall",
    "description": "Search entities with FTS5 keyword search. Words are OR'd together. Returns entities sorted by relevance with expanded content/summary fields at top level. Use this to find previously stored facts, decisions, or architecture notes. When encryption is enabled, body_json is decrypted transparently.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "query": {
          "type": "string",
          "description": "Search query — words are OR'd together for broad recall. An EMPTY string (\"\") is the match-all / enumeration path: it drops the keyword predicate and returns every entity in scope (respecting category/type/limit/offset), so it is the way to 'list all' a category. Wildcards are NOT globs: \"*\" is a literal FTS5 term and matches nothing — pass \"\" to enumerate, not \"*\"."
        },
        "category": {
          "type": "string",
          "description": "Filter by category, e.g. 'decision' or 'architecture'"
        },
        "type": {
          "type": "string",
          "description": "Filter by entity type, e.g. 'insight' or 'reference'"
        },
        "limit": {
          "type": "integer",
          "default": 10,
          "description": "Maximum number of results to return (max 1000)"
        },
        "offset": {
          "type": "integer",
          "default": 0,
          "description": "Number of results to skip for pagination"
        },
        "min_decay": {
          "type": "number",
          "default": 0.0,
          "description": "Minimum decay score threshold 0.0–1.0 — higher values return fresher results"
        },
        "topic_path": {
          "type": "string",
          "description": "Filter by topic path prefix, e.g. 'architecture/'"
        },
        "mode": {
          "type": "string",
          "default": "fts5",
          "description": "Search mode: 'fts5' (keyword), 'dense' (vector), 'hybrid' (fused via RRF), or 'fused' (TEMPR-style multi-strategy: fts5 + dense + graph + temporal with weighted RRF, token-budget truncation, and a full fused_trace, #883)",
          "enum": [
            "fts5",
            "dense",
            "hybrid",
            "fused"
          ]
        },
        "strategies": {
          "type": "array",
          "items": { "type": "string", "enum": ["fts5", "dense", "graph", "temporal"] },
          "description": "Fused mode only: strategies to engage (2-4). Omit = all four. Unknown names are rejected."
        },
        "max_tokens": {
          "type": "integer",
          "default": 0,
          "description": "Fused mode only: token-budget truncation (estimated tokens = chars/4 per body). 0 = derive from depth_budget (mid = 4096)."
        },
        "depth_budget": {
          "type": "string",
          "enum": ["low", "mid", "high"],
          "description": "Fused mode only: depth budget -> default token caps 1024 / 4096 / 16384 when max_tokens is unset."
        },
        "strategy_weights": {
          "type": "object",
          "description": "Fused mode only: per-strategy RRF weight multipliers (default 1.0 each). Arms that find nothing contribute nothing."
        },
        "rerank": {
          "type": "boolean",
          "default": false,
          "description": "Fused mode only: optional rerank stage over the fused pool (rank-calibrated dense + BM25 agreement signals; default off, latency-preserving)."
        },
        "query_time_unix_ms": {
          "type": "integer",
          "description": "Fused mode only: anchor instant for the temporal strategy (unix ms; default now). Accepts a number or numeric string."
        },
        "graph_utility_threshold": {
          "type": "number",
          "description": "Fused mode only (#869): graph utility gate threshold in [0,1]. The graph strategy engages only when the query's classified graph utility is >= this value. Omit = 0.5 (documented default). 0.0 disables the gate; 1.0 effectively never engages. The routing decision is always observable in fused_trace.graph_route (reason, selected, skipped_reason, gate counts)."
        },
        "profile": {
          "type": "string",
          "enum": ["default", "validity"],
          "description": "#860: validity-aware recall profile. 'validity' re-ranks fused results by a deterministic validity multiplier (freshness decay, scope match, provenance class, supersession, expiry proximity) and annotates every item with its validity info; 'default'/omitted keeps relevance-only ordering. On non-fused modes the profile only enables item annotation. The weights, grade distribution, and context-invalid count are observable in fused_trace.validity."
        },
        "validity_annotate": {
          "type": "boolean",
          "default": false,
          "description": "#860: annotate delivered items with their validity info (grade, freshness, scope match, provenance class, superseded, expiring/expired, multiplier, signals); context-invalid items are additionally flagged 'context_invalid': true. Implied by profile='validity'."
        },
        "include_archived": {
          "type": "boolean",
          "default": false,
          "description": "Include archived (soft-deleted) entities in results"
        },
        "include_confidence": {
          "type": "boolean",
          "default": false,
          "description": "Add a normalized confidence score (0.0-1.0) to each result, rolled up from rank, trust (verified/certainty), and decay. Presentation-only; does not change ranking."
        },
        "include_provider_source": {
          "type": "boolean",
          "default": false,
          "description": "#1141: include only sanitized provider identity, revision, digest, scope, and thread lineage; raw provider bodies and payloads are never returned."
        },
        "evidence_lanes": {
          "type": "array",
          "items": { "type": "string", "enum": ["derived", "verbatim"] },
          "minItems": 1,
          "description": "#1135: opt-in governed answer-facing evidence lanes. Omit for the legacy byte-compatible recall response; choose derived, verbatim, or both under the shared max_tokens budget. Duplicate lane names are canonicalized."
        },
        "include_selection_decisions": {
          "type": "boolean",
          "default": false,
          "description": "#1140: fused mode only. Attach a bounded, hash-only per-candidate selection projection with source-arm ranks, eligibility/disposition reason codes, token-estimator state, unavailable-arm state, and a replay fingerprint. Omit to preserve the legacy response shape."
        },
        "include_declared_graph": {
          "type": "boolean",
          "default": false,
          "description": "#1142: attach a bounded workspace-scoped hash-only declared graph projection. Requires workspace_hash and a transport-stamped requester; ordinary recall does not query the graph."
        },
        "include_conflict_flags": {
          "type": "boolean",
          "default": false,
          "description": "#917: add deterministic contradiction/superseded/stale flags containing only entity IDs, validity ranges, and hash-linked claim-card evidence refs. Suppressed values disclose existence only; no body value is rendered."
        },
        "include_conflict_flags_markdown": {
          "type": "boolean",
          "default": false,
          "description": "#917: independently add an ID/hash/validity-only markdown conflict block. Does not implicitly enable structured conflict_flags."
        },
        "reinforce": {
          "type": "boolean",
          "default": false,
          "description": "Opt-in reinforcement for mode='dense'/'hybrid': bump retrieval_count/last_accessed/decay on the returned hits so semantically-used memories resist decay and promote through layers. Default false keeps semantic recall side-effect-free and byte-deterministic over a frozen DB. No effect on mode='fts5', which already reinforces."
        },
        "expansion": {
          "type": "object",
          "properties": {
            "enabled": {
              "type": "boolean",
              "default": false,
              "description": "Enable stemming-based query expansion"
            },
            "n_variants": {
              "type": "integer",
              "default": 1,
              "description": "Number of stemmed token variants to generate"
            }
          },
          "description": "Configuration for FTS5 query expansion using Porter stemming"
        },
        "preview_cap": {
          "type": "integer",
          "description": "If set, truncate body_json at N chars and append drill-down footer. Use perseus_vault_get_entity to read full body."
        },
        "content_weight": {
          "type": "number",
          "minimum": 0,
          "maximum": 1,
          "default": 0,
          "description": "Additive boost for content witness — rewards entities whose body text literally contains query terms. Damped by body length. Never penalizes."
        },
        "trust_weight": {
          "type": "number",
          "minimum": 0,
          "maximum": 1,
          "default": 0.15,
          "description": "Additive boost for provenance/trust (default 0.15, on by default) — verified sources rank above unverified AI drafts on the same topic. Verified entities get the full boost; unverified ones are scaled by certainty. Set 0 to disable. Never penalizes."
        },
        "diversity_halving": {
          "type": "number",
          "minimum": 0,
          "maximum": 1,
          "default": 1,
          "description": "Per-keyword diversity quota factor (1.0=disabled). Each distinct matched keyword gets ceil(N x halving^n) slots — first keyword N, second N/2, etc."
        },
        "recency_half_life_secs": {
          "type": "number",
          "minimum": 0,
          "description": "Time-aware ranking for mode='hybrid' (default off). When set, each fused result's score is multiplied by 0.5^(age / this), where age is seconds since the memory was created — so a memory this many seconds old keeps half its weight and recent context outranks older but similar hits. Omit for relevance-only ranking."
        },
        "workspace_hash": {
          "type": "string",
          "description": "Workspace scope filter (v1.2.0). When set, only entities with a matching workspace_hash are returned. Compatibility mode permits omission for legacy unscoped reads when strict deployment mode is off; strict mode requires a non-empty workspace_hash and an active binding for the transport requester."
        },
        "requesting_agent_id": {
          "type": "string",
          "description": "Transport-stamped requester identity used for private/fleet visibility enforcement."
        },
        "scope_weight": {
          "type": "number",
          "minimum": 0,
          "maximum": 1,
          "description": "#485: scope as a ranking multiplier instead of a hard filter. Requires workspace_hash. Widens the workspace filter to also include GLOBAL (workspace_hash='') memories, weighted by this factor in the ranking (hybrid/dense scores multiplied; keyword mode returns current-scope hits first) — current-workspace memories outrank equally-relevant global ones, but a strong global memory still surfaces. Never exposes other workspaces' memories. Omit for the strict filter (unchanged default)."
        },
        "agent_id": {
          "type": "string",
          "description": "Agent identity filter (v1.2.0). When set, only entities with a matching agent_id are returned. Omit for no agent filtering."
        },
        "epistemic_state": {
          "type": "string",
          "enum": ["candidate", "verified", "corroborated", "rejected", "defensively_recalled"],
          "description": "#880: epistemic trust-axis filter. When set, only entities in the requested trust state are returned — 'candidate' surfaces useful-but-unverified records, 'verified'/'corroborated' restrict to established fact, 'rejected' shows reviewed-and-refused records. Omit for no trust filtering (default)."
        },
        "retrieval_profile": {
          "type": "string",
          "enum": ["personal", "agent", "shared"],
          "description": "#784 serving posture. personal returns preference/personal classes; agent returns convention/correction/keystone classes; shared (default) returns non-personal memory in the requested workspace. Applied after visibility filtering."
        },
        "layer": {
            "type": "string",
            "description": "Filter by memory layer (world, episodic, semantic)."
        },
        "ref_type": {
          "type": "string",
          "description": "#728: post-filter hits to entities whose body external_refs carry this ref_type (exact match, e.g. 'repo', 'pull_request', 'jira_key')."
        },
        "ref_value": {
          "type": "string",
          "description": "#728: post-filter hits to entities whose body external_refs carry this ref_value. Matches exactly or as a hierarchical '/' prefix ('github:Org' matches 'github:Org/repo')."
        },
        "deadline_ms": {
          "type": "integer",
          "description": "#864: bounded recall. When set, the recall is timed; if it exceeds this many ms the response outcome.status is 'timeout' so callers know the result set may be incomplete. Results are still returned in full."
        },
        "include_outcome": {
          "type": "boolean",
          "default": false,
          "description": "#864/#873/#887/#1186: include the bounded answer-facing outcome for complete results as well as partial, degraded, abstained, unavailable, and empty/stale results; the legacy outcome block remains compatibility-only. By default nominal legacy responses stay byte-identical."
        },
        "as_of_unix_ms": {
          "type": "integer",
          "description": "#472 Temporal RAG: transaction-time instant (unix ms). Reconstruct semantic recall AS BELIEVED at this past instant — each hit's body is the version that was live at as_of_unix_ms; corrections recorded later do not leak in. Combine with valid_at for the full bi-temporal cell. Hits are stamped with is_live_version / recorded_at_unix_ms / valid_from_unix_ms / valid_to_unix_ms. Omit for today's live view. (v1: candidate generation is over the live index, so a fact fully deleted since that instant will not surface.)"
        },
        "valid_at": {
          "type": "integer",
          "description": "Valid-time instant (#363/#472, unix ms): reconstruct recall to the world-version whose application-time period [valid_from, valid_to) contains this instant — 'what was true at time T', per current (or as_of) knowledge. Rebuilds the point-in-time body from history (not just a live-row narrow) and returns hits stamped with is_live_version / recorded_at_unix_ms / valid_from/to. Combine with as_of_unix_ms for the full bi-temporal cell."
        },
        "valid_from_unix_ms": {
          "type": "integer",
          "description": "Valid-time period filter start (#363, unix ms). Pair with valid_to_unix_ms and valid_op; ignored when valid_at is set. Omit for unbounded start."
        },
        "valid_to_unix_ms": {
          "type": "integer",
          "description": "Valid-time period filter end (#363, unix ms, exclusive). Omit for unbounded end."
        },
        "valid_op": {
          "type": "string",
          "default": "overlaps",
          "enum": ["overlaps", "contains"],
          "description": "SQL:2011 period predicate for the valid-time period filter (#363): 'overlaps' (fact's valid period shares at least one instant with the queried period) or 'contains' (fact's valid period contains the whole queried period)."
        }
      },
      "required": [
        "query"
      ]
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "items": {
          "type": "array",
          "items": {
            "type": "object"
          },
          "description": "Matching entities with expanded body_json fields at top level"
        },
        "total": {
          "type": "integer",
          "description": "Number of results returned"
        },
        "evidence": {
          "type": "object",
          "description": "#1135: optional governed derived/verbatim evidence projection with shared budget, exclusions, source groups, and hash-only receipt. Present only when evidence_lanes is supplied."
        },
        "fused_trace": {
          "type": "object",
          "description": "#883: fused serving trace. When include_selection_decisions=true it contains selection_decisions: a bounded hash-only projection of candidate eligibility, dispositions, arm states, token estimates, delivered order, and replay fingerprint."
        },
        "variants": {
          "type": "integer",
          "description": "Number of query variants used when expansion is enabled"
        },
        "declared_graph": {
          "type": "object",
          "description": "#1142: optional bounded declared graph projection; nodes/edges carry hash-only source, span, scope, origin, validity, and support state."
        },
        "conflict_flags": {
          "type": "array",
          "items": { "type": "object" },
          "description": "#917: optional deterministic contradiction/supersession/staleness flags; IDs, validity ranges, and hash-linked evidence refs only"
        },
        "abstain_hint": {
          "type": "boolean",
          "description": "#917: true only when a high-confidence direct contradiction is present in the delivered set"
        },
        "conflict_flags_markdown": {
          "type": "string",
          "description": "#917: optional ID/hash/validity-only markdown rendering of conflict flags"
        },
        "answer_outcome": {
          "type": "object",
          "additionalProperties": false,
          "description": "#1186: bounded answer-facing status; no query, evidence body, or backend error text.",
          "properties": {
            "schema_version": {"type": "string", "const": "perseus-vault-answer-outcome/v1"},
            "status": {"type": "string", "enum": ["complete", "partial", "degraded", "abstained", "unavailable"]},
            "recall_status": {"type": "string", "enum": ["fresh", "partial", "timeout", "unavailable", "empty", "stale"]},
            "reason": {"type": "string", "minLength": 1, "maxLength": 256},
            "reason_codes": {"type": "array", "minItems": 1, "maxItems": 16, "items": {"type": "string", "maxLength": 256}},
            "abstained": {"type": "boolean"},
            "answerable": {"type": "boolean"},
            "fallback": {"type": "object", "additionalProperties": false, "properties": {"mode": {"type": "string", "enum": ["abstain", "canonical_retrieval"]}, "reason": {"type": "string", "minLength": 1, "maxLength": 256}}, "required": ["mode", "reason"]},
            "exclusions": {"type": "array", "maxItems": 256, "items": {"type": "object", "additionalProperties": false, "properties": {"reason": {"type": "string", "maxLength": 256}, "count": {"type": "integer", "minimum": 1}}, "required": ["reason", "count"]}},
            "conflicts": {"type": "array", "maxItems": 128, "items": {"type": "object", "additionalProperties": false, "properties": {"reason": {"type": "string", "maxLength": 256}, "reference_count": {"type": "integer", "minimum": 0}, "references_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"}}, "required": ["reason", "reference_count", "references_sha256"]}}
          },
          "required": ["schema_version", "status", "recall_status", "reason", "reason_codes", "abstained", "answerable"]
        }
      }
    },
    "annotations": {
      "readOnlyHint": true
    },
    "title": "Recall Entities"
  },
  {
    "name": "perseus_vault_handoff_pack",
    "description": "Budgeted handoff pack: lifecycle-filtered (expired/superseded excluded), provenance-tagged context for cross-session handoffs under a hard token budget, with exclusion visibility and a deterministic pack digest. Candidates from FTS5 recall; greedy-with-backfill packing; never exceeds budget_tokens. Optional planning-boundary enrichment (#1039): include_intent_trail adds recent journal events tied to the pack, include_next_work adds journal forward plans plus recall_when anticipation matches, include_conflicts adds pack-scoped contradiction flags.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "query": { "type": "string", "description": "Handoff topic query (required, non-empty)" },
        "budget_tokens": { "type": "integer", "description": "Hard pack budget in tokens (chars/4), 100..100000, default 2000" },
        "max_excluded": { "type": "integer", "description": "Max excluded items listed with reasons, 0..200, default 20" },
        "include_expired": { "type": "boolean", "description": "Include expired checkable claims (default false)" },
        "workspace_hash": { "type": "string", "description": "Workspace scope hash. When set, the pack and its enrichment are scoped to that workspace." },
        "include_intent_trail": { "type": "boolean", "description": "Add intent_trail: recent journal events tied to the packed entities (default false)" },
        "include_next_work": { "type": "boolean", "description": "Add next_work: journal forward plans + recall_when anticipation matches for the scope (default false)" },
        "include_conflicts": { "type": "boolean", "description": "Add pack-scoped contradiction flags from the conflict detector (default false)" },
        "max_trail": { "type": "integer", "description": "Max intent-trail events to return, 1..20, default 5" }
      },
      "required": ["query"]
    },
    "annotations": {
      "readOnlyHint": true
    },
    "title": "Handoff Pack"
  },
  {
    "name": "perseus_vault_delegation_brief",
    "description": "Deterministic markdown delegation brief generated at the planning boundary (#1039): goal + scope + binding context (superseded items excluded and listed as do-not-resurrect) + intent trail + next work + output contract. Hand a subagent the brief instead of the parent chat session; the brief is self-contained for the delegated task.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "query": { "type": "string", "description": "Scope anchor query for the delegation (required, non-empty)" },
        "goal": { "type": "string", "description": "One-sentence goal of the delegated task (required, non-empty)" },
        "output_contract": { "type": "string", "description": "Exact output the delegate must produce (files, commands, report shape). Omitted = return a plan with explicit open questions." },
        "budget_tokens": { "type": "integer", "description": "Hard brief budget in tokens (chars/4), 200..100000, default 4000" },
        "include_expired": { "type": "boolean", "description": "Include expired checkable claims in binding context (default false)" },
        "workspace_hash": { "type": "string", "description": "Workspace scope hash. When set, the brief is built only from that workspace." }
      },
      "required": ["query", "goal"]
    },
    "annotations": {
      "readOnlyHint": true
    },
    "title": "Delegation Brief"
  },
  {
    "name": "perseus_vault_intention",
    "description": "Prospective memory: typed intention programs (Latch borrow) with immutable revisions, compound triggers/inhibitors, time windows, approval flags, atomic exactly-once claims (JSON1 compare-and-set), and purpose-based forgetting. Ops: create|update (new immutable revision), evaluate (waiting|ready|blocked|expired + reasons), claim (exactly-once), complete|fail (one-shot auto-forgets), list.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "op": { "type": "string", "enum": ["create", "update", "evaluate", "claim", "complete", "fail", "list"], "description": "Operation" },
        "name": { "type": "string", "description": "Intention name (required for all ops except list)" },
        "purpose": { "type": "string", "enum": ["one_shot", "recurring"], "description": "one_shot auto-forgets on completion (default one_shot)" },
        "program": { "type": "object", "description": "Instruction: {when:{triggers:[{query}]}, unless:{inhibitors:[{query}]}, window:{after_unix_ms?,before_unix_ms?}, action:{kind,params}, approval:'required'|'auto'}" },
        "claimed_by": { "type": "string", "description": "Claimer identity for the claim op" },
        "note": { "type": "string", "description": "Outcome note for complete/fail" }
      },
      "required": ["op"]
    },
    "title": "Intention Program"
  },
  {
    "name": "perseus_vault_proof_frame",
    "description": "Proof frame: bounded, hash-cited evidence pack for external consumers (Qorx Zero borrow). Memory stays on-device; the consumer gets only a capped frame (top-N records + per-record source hashes + frame digest). Empty frame -> refusal, never invention. zeroize:true permanently blanks framed entities' bodies after framing (privacy end-state).",
    "inputSchema": {
      "type": "object",
      "properties": {
        "query": { "type": "string", "description": "Evidence question (required, non-empty)" },
        "max_records": { "type": "integer", "description": "Max records in the frame, 1..20, default 5" },
        "max_chars": { "type": "integer", "description": "Max frame chars, 200..20000, default 1600" },
        "zeroize": { "type": "boolean", "description": "Blank framed entities' bodies after framing (default false)" },
        "workspace_hash": { "type": "string", "description": "Workspace scope hash" }
      },
      "required": ["query"]
    },
    "annotations": {
      "readOnlyHint": true
    },
    "title": "Proof Frame"
  },
  {
    "name": "perseus_vault_recall_batch",
    "description": "Recall entities across a batch of queries, fusing their results server-side using reciprocal rank fusion (RRF) to merge, deduplicate, and surface the most globally relevant memories first.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "queries": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "query": {
                "type": "string",
                "description": "Search query — words are OR'd together for broad recall. An EMPTY string (\"\") is the match-all / enumeration path."
              },
              "category": {
                "type": "string",
                "description": "Filter by category, e.g. 'decision' or 'architecture'"
              },
              "type": {
                "type": "string",
                "description": "Filter by entity type, e.g. 'insight' or 'reference'"
              },
              "limit": {
                "type": "integer",
                "default": 10,
                "description": "Maximum number of results to return (max 1000)"
              },
              "offset": {
                "type": "integer",
                "default": 0,
                "description": "Number of results to skip for pagination"
              },
              "min_decay": {
                "type": "number",
                "default": 0.0,
                "description": "Minimum decay score threshold 0.0–1.0 — higher values return fresher results"
              },
              "topic_path": {
                "type": "string",
                "description": "Filter by topic path prefix, e.g. 'architecture/'"
              },
              "mode": {
                "type": "string",
                "default": "fts5",
                "description": "Search mode: 'fts5' (keyword), 'dense' (vector), or 'hybrid' (fused via RRF)",
                "enum": [
                  "fts5",
                  "dense",
                  "hybrid"
                ]
              },
              "include_archived": {
                "type": "boolean",
                "default": false,
                "description": "Include archived (soft-deleted) entities in results"
              },
              "include_confidence": {
                "type": "boolean",
                "default": false,
                "description": "Add a normalized confidence score (0.0-1.0) to each result, rolled up from rank, trust (verified/certainty), and decay. Presentation-only; does not change ranking."
              },
              "reinforce": {
                "type": "boolean",
                "default": false,
                "description": "Opt-in reinforcement for mode='dense'/'hybrid': bump retrieval_count/last_accessed/decay on the returned hits so semantically-used memories resist decay."
              },
              "preview_cap": {
                "type": "integer",
                "description": "If set, truncate body_json at N chars and append drill-down footer."
              },
              "content_weight": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "default": 0,
                "description": "Additive boost for content witness — rewards entities whose body text literally contains query terms."
              },
              "trust_weight": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "default": 0.15,
                "description": "Additive boost for provenance/trust (default 0.15, on by default)."
              },
              "diversity_halving": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "default": 1,
                "description": "Per-keyword diversity quota factor (1.0=disabled)."
              },
              "recency_half_life_secs": {
                "type": "number",
                "minimum": 0,
                "description": "Time-aware ranking for mode='hybrid' (default off)."
              },
              "workspace_hash": {
                "type": "string",
                "description": "Workspace scope filter. Compatibility mode permits omission for legacy unscoped nested queries when strict deployment mode is off; strict mode requires a non-empty workspace_hash and an active binding for the transport requester."
              },
              "scope_weight": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "description": "#485: scope as a ranking multiplier instead of a hard filter."
              },
              "agent_id": {
                "type": "string",
                "description": "Agent identity filter."
              },
              "layer": {
                "type": "string",
                "description": "Filter by memory layer (world, episodic, semantic)."
              },
              "as_of_unix_ms": {
                "type": "integer",
                "description": "Temporal RAG transaction-time."
              },
              "valid_at": {
                "type": "integer",
                "description": "Valid-time instant."
              },
              "valid_from_unix_ms": {
                "type": "integer",
                "description": "Valid-time period filter start."
              },
              "valid_to_unix_ms": {
                "type": "integer",
                "description": "Valid-time period filter end."
              },
              "valid_op": {
                "type": "string",
                "default": "overlaps",
                "enum": ["overlaps", "contains"],
                "description": "SQL:2011 period predicate for valid-time period filter."
              },
              "include_outcome": {
                "type": "boolean",
                "default": false,
                "description": "#1186: include the bounded answer_outcome for complete results as well as partial, degraded, abstained, unavailable, and empty/stale results for this query."
              }
            },
            "required": [
              "query"
            ]
          }
        },
        "requesting_agent_id": {
          "type": "string",
          "description": "Transport-stamped requester identity applied to every nested query and fused result."
        },
        "include_outcome": {
          "type": "boolean",
          "default": false,
          "description": "#1186: include bounded answer_outcome and per-query query_outcomes for complete results as well as partial, degraded, abstained, unavailable, and failed queries."
        }
      },
      "required": [
        "queries"
      ]
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "items": {
          "type": "array",
          "items": {
            "type": "object"
          },
          "description": "Matching entities fused from batch queries with expanded body_json fields at top level"
        },
        "total": {
          "type": "integer",
          "description": "Number of results returned"
        },
        "answer_outcome": {
          "type": "object",
          "additionalProperties": false,
          "description": "#1186: bounded answer-facing status; no query, evidence body, or backend error text.",
          "properties": {
            "schema_version": {"type": "string", "const": "perseus-vault-answer-outcome/v1"},
            "status": {"type": "string", "enum": ["complete", "partial", "degraded", "abstained", "unavailable"]},
            "recall_status": {"type": "string", "enum": ["fresh", "partial", "timeout", "unavailable", "empty", "stale"]},
            "reason": {"type": "string", "minLength": 1, "maxLength": 256},
            "reason_codes": {"type": "array", "minItems": 1, "maxItems": 16, "items": {"type": "string", "maxLength": 256}},
            "abstained": {"type": "boolean"},
            "answerable": {"type": "boolean"},
            "fallback": {"type": "object", "additionalProperties": false, "properties": {"mode": {"type": "string", "enum": ["abstain", "canonical_retrieval"]}, "reason": {"type": "string", "minLength": 1, "maxLength": 256}}, "required": ["mode", "reason"]},
            "exclusions": {"type": "array", "maxItems": 256, "items": {"type": "object", "additionalProperties": false, "properties": {"reason": {"type": "string", "maxLength": 256}, "count": {"type": "integer", "minimum": 1}}, "required": ["reason", "count"]}},
            "conflicts": {"type": "array", "maxItems": 128, "items": {"type": "object", "additionalProperties": false, "properties": {"reason": {"type": "string", "maxLength": 256}, "reference_count": {"type": "integer", "minimum": 0}, "references_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"}}, "required": ["reason", "reference_count", "references_sha256"]}}
          },
          "required": ["schema_version", "status", "recall_status", "reason", "reason_codes", "abstained", "answerable"]
        },
        "query_outcomes": {
          "type": "array",
          "maxItems": 256,
          "items": {"$ref": "#/properties/answer_outcome"},
          "description": "#1186: one bounded answer outcome per batch query, including failed arms."
        }
      }
    },
    "annotations": {
      "readOnlyHint": true
    },
    "title": "Recall Entities Batch"
  },
  {
    "name": "perseus_vault_recall_layer",
    "description": "Recall entities from a specific biomimetic memory layer (world, episodic, semantic).",
    "inputSchema": {
      "type": "object",
      "properties": {
        "layer": {
          "type": "string",
          "description": "The memory layer to recall from.",
          "enum": ["world", "episodic", "semantic"]
        },
        "limit": {
          "type": "integer",
          "default": 10,
          "description": "Maximum number of results to return (max 1000)."
        },
        "requesting_agent_id": {
          "type": "string",
          "description": "Transport-stamped requester identity used for visibility enforcement."
        }
      },
      "required": ["layer"]
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "items": {
          "type": "array",
          "items": { "type": "object" },
          "description": "Matching entities with expanded body_json fields at top level."
        },
        "total": {
          "type": "integer",
          "description": "Number of results returned."
        }
      }
    },
    "annotations": {
      "readOnlyHint": true
    }
  },
  {
    "name": "perseus_vault_scan",
    "description": "Enumerate every entity in a category (or the whole store) deterministically, page by page (#562). This is the first-class 'list all / export / sync / reset' path: pages are keyed by immutable entity id (ascending) with a continuation cursor, so repeated calls walk the full set exactly once — unlike recall(query=\"\") pagination, whose relevance ordering mutates as recalls reinforce entities (pages can skip or repeat rows) and whose offset is capped. Call with no cursor for the first page, then pass back next_cursor until has_more is false. Read-only: scanning does not bump retrieval counts or decay. Note the recall query contract this complements: recall's query=\"\" is match-all enumeration; \"*\" is a literal FTS5 term (NOT a glob) and matches nothing.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "category": {
          "type": "string",
          "description": "Category to enumerate, e.g. 'decision'. Omit or pass \"\" to scan every category (no category is excluded — unlike recall, which hides high-volume categories such as 'conversation' unless explicitly requested)."
        },
        "workspace_hash": {
          "type": "string",
          "description": "Workspace scope filter. When set, only entities with exactly this workspace_hash are returned (\"\" targets only global entities). Omit for unscoped."
        },
        "include_archived": {
          "type": "boolean",
          "default": false,
          "description": "Compatibility flag retained for callers that request historical rows; public scans never return archived or terminal bodies. Use dedicated terminal-audit surfaces for hash-only audit markers."
        },
        "cursor": {
          "type": "string",
          "description": "Continuation cursor: the next_cursor value from the previous page. Omit for the first page."
        },
        "limit": {
          "type": "integer",
          "default": 100,
          "description": "Page size (1–1000)."
        }
      },
      "required": []
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "items": {
          "type": "array",
          "items": { "type": "object" },
          "description": "Entities in this page, ordered by id ascending, with expanded body_json fields at top level."
        },
        "total": {
          "type": "integer",
          "description": "Number of entities in this page."
        },
        "has_more": {
          "type": "boolean",
          "description": "True when another page exists."
        },
        "next_cursor": {
          "type": ["string", "null"],
          "description": "Pass this as `cursor` to fetch the next page. Null on the final page."
        }
      }
    },
    "annotations": {
      "readOnlyHint": true
    },
    "title": "Scan / Enumerate Entities"
  },
  {
    "name": "perseus_vault_hygiene",
    "description": "Read-only hygiene report: surface likely low-signal memories so a startup-memory block stays dense without manual forensics. Scores every active memory by startup 'actionability' (the same signal as recall's startup mode) — concrete anchors like issue/ticket keys, #refs, paths, URLs, named systems, and decision/escalation language score high; vague, date-only titles (e.g. '2026-07-13') and very short bodies score low — and returns the worst offenders (below `threshold`) with the reasons they were flagged. Keyset-scans in pages; never bumps retrieval counts or decay. Use it to find archive/consolidate candidates before curating startup recall.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "category": {
          "type": "string",
          "description": "Restrict the scan to one category, e.g. 'memories'. Omit to scan every active category."
        },
        "threshold": {
          "type": "number",
          "default": 0.35,
          "description": "Actionability score (0.0–1.0) below which a memory is flagged low-signal. Lower = stricter (fewer flags)."
        },
        "scan_limit": {
          "type": "integer",
          "default": 1000,
          "description": "Maximum active memories to scan (1–10000)."
        },
        "limit": {
          "type": "integer",
          "default": 50,
          "description": "Maximum flagged rows to return, worst first (1–1000)."
        }
      },
      "required": []
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "scanned": {
          "type": "integer",
          "description": "Number of active memories inspected."
        },
        "flagged_count": {
          "type": "integer",
          "description": "Total memories below the threshold (may exceed the returned rows)."
        },
        "returned": {
          "type": "integer",
          "description": "Number of flagged rows in this response."
        },
        "threshold": {
          "type": "number",
          "description": "The actionability threshold applied."
        },
        "flagged": {
          "type": "array",
          "items": { "type": "object" },
          "description": "Worst-first: {id, category, key, actionability, reasons[], retrieval_count}. reasons ∈ date_only_title | short_body | no_concrete_entities | low_actionability."
        }
      }
    },
    "annotations": {
      "readOnlyHint": true
    },
    "title": "Startup-Memory Hygiene Report"
  },
  {
    "name": "perseus_vault_promote",
    "description": "Promote a memory across the class ladder (to_category) and/or the scope ladder (to_workspace_hash) per the shared-memory promotion ladder (perseus docs/shared-memory-promotion-ladder.md §4). Creates a new entity that carries a promoted_from provenance record (source category/key/id/scope, reason, timestamp) and links the source to it with relationship='promoted_to'. The source entity is never edited or hidden — raw evidence stays reachable. Uses skip_dedup internally so the promoted copy always creates its own key even when near-identical to the source.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "from_category": {
          "type": "string",
          "description": "Category of the source entity to promote"
        },
        "from_key": {
          "type": "string",
          "description": "Key of the source entity to promote"
        },
        "to_category": {
          "type": "string",
          "description": "Target class/category. Omit to keep the source category."
        },
        "to_workspace_hash": {
          "type": "string",
          "description": "Target scope (workspace_hash; empty string = global). Omit to keep the source scope."
        },
        "to_key": {
          "type": "string",
          "description": "Target key. Omit to keep the source key."
        },
        "reason": {
          "type": "string",
          "description": "Why this promotion is happening (recorded in promoted_from)."
        }
      },
      "required": ["from_category", "from_key"]
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "promoted": { "type": "boolean" },
        "action": { "type": "string", "description": "'created' or 'updated' for the target entity" },
        "from_id": { "type": "string" },
        "to_id": { "type": "string" },
        "to_workspace_hash": { "type": "string" }
      }
    },
    "annotations": {
      "destructiveHint": true
    },
    "title": "Promote Memory"
  },
  {
    "name": "perseus_vault_demote",
    "description": "Demote a governed memory exactly one rung down the durable-memory ladder. Writes a provenance-preserving copy, a demoted_to link, and an append-only demotion journal event.",
    "inputSchema": {"type":"object","properties":{
      "from_category":{"type":"string"},"from_key":{"type":"string"},"to_category":{"type":"string"},"to_key":{"type":"string"},"reason":{"type":"string"}
    },"required":["from_category","from_key","to_category"]},
    "outputSchema": {"type":"object","properties":{"demoted":{"type":"boolean"},"to_id":{"type":"string"}}},
    "annotations": {"destructiveHint": true},
    "title": "Demote Memory"
  },
  {
    "name": "perseus_vault_beliefs",
    "description": "Derived-belief overlay (#717, spec: docs/specs/belief-overlay.md): compute the current effective belief for a topic from the live entity store, with fresh local corrections always outranking stale global beliefs regardless of semantic similarity (precedence tiers are absolute, never blended).",
    "inputSchema": {
      "type": "object",
      "properties": {
        "topic": {
          "type": "string",
          "description": "Topic or question to resolve the current effective belief for"
        },
        "workspace_hash": {
          "type": "string",
          "description": "Optional workspace scope for the local-correction tier"
        },
        "limit": {
          "type": "integer",
          "default": 10,
          "description": "Maximum belief candidates to return"
        }
      },
      "required": ["topic"]
    },
    "annotations": {
      "readOnlyHint": true
    },
    "title": "Derived Beliefs Overlay"
  },
  {
    "name": "perseus_vault_claim_card",
    "description": "Evidence-backed claim card (#852, spec: docs/specs/claim-cards.md): a deterministic, versioned projection of one entity's claim, provenance class (source_human/fact_extracted/fact_derived/inference_agent), valid vs recorded time, confidence/support, supersession/contradiction/stale state, evidence references, a sanitized agent_projection hash-bound to the selected evidence and policy, and machine-readable reason codes (serveable / archived / scope_mismatch / revoked_access + flags). Read-only view over existing entities and links — never a second source of truth.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "entity_id": {
          "type": "string",
          "description": "ID of the entity to project as a claim card"
        },
        "workspace_hash": {
          "type": "string",
          "description": "Caller's workspace scope for visibility enforcement (workspace-scoped entities mismatch → withheld with scope_mismatch)"
        },
        "agent_id": {
          "type": "string",
          "description": "Legacy caller field; public authorization uses the transport-stamped requesting_agent_id."
        },
        "requesting_agent_id": {
          "type": "string",
          "description": "Transport-stamped requester identity; required at runtime and never trusted from model input."
        },
        "include_evidence": {
          "type": "boolean",
          "default": true,
          "description": "Include evidence references (metadata only; raw bodies never cross)"
        },
        "include_agent_projection": {
          "type": "boolean",
          "default": true,
          "description": "Include the sanitized agent_projection block"
        }
      },
      "required": ["entity_id"]
    },
    "annotations": {
      "readOnlyHint": true
    },
    "title": "Evidence-Backed Claim Card"
  },
  {
    "name": "perseus_vault_semantic_search",
    "description": "Dense-only semantic search: find entities by meaning, ranked purely by embedding similarity (no keyword fallback). On by default via the bundled in-process ONNX model — zero config, zero network. A one-tool shortcut for 'find things like this'. For fused keyword+vector results use perseus_vault_recall.",
      "inputSchema": {
      "type": "object",
      "properties": {
        "query": {
          "type": "string",
          "description": "Natural-language text to semantically match against stored memories"
        },
        "limit": {
          "type": "integer",
          "default": 10,
          "description": "Maximum number of results to return"
        },
        "category": {
          "type": "string",
          "description": "Filter by category, e.g. 'decision' or 'architecture'"
        },
        "workspace_hash": {
          "type": "string",
          "description": "Workspace scope filter. When set, only entities with a matching workspace_hash are returned. Compatibility mode permits omission for legacy unscoped reads when strict deployment mode is off; strict mode requires a non-empty workspace_hash and an active binding for the transport requester."
        },
        "agent_id": {
          "type": "string",
          "description": "Agent identity filter. When set, only entities with a matching agent_id are returned."
        },
        "requesting_agent_id": {
          "type": "string",
          "description": "Transport-stamped requester identity used for private/fleet visibility enforcement."
        },
        "include_outcome": {
          "type": "boolean",
          "default": false,
          "description": "#1186: include a bounded answer_outcome for complete results as well as no-match, partial, degraded, abstained, or unavailable dense retrieval."
        }
      },
      "required": [
        "query"
      ]
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "items": {
          "type": "array",
          "items": {
            "type": "object"
          },
          "description": "Matching entities ranked by dense embedding similarity, with expanded body_json fields at top level"
        },
        "total": {
          "type": "integer",
          "description": "Number of results returned"
        },
        "answer_outcome": {
          "type": "object",
          "additionalProperties": false,
          "description": "#1186: bounded answer-facing status; no query, evidence body, or backend error text.",
          "properties": {
            "schema_version": {"type": "string", "const": "perseus-vault-answer-outcome/v1"},
            "status": {"type": "string", "enum": ["complete", "partial", "degraded", "abstained", "unavailable"]},
            "recall_status": {"type": "string", "enum": ["fresh", "partial", "timeout", "unavailable", "empty", "stale"]},
            "reason": {"type": "string", "minLength": 1, "maxLength": 256},
            "reason_codes": {"type": "array", "minItems": 1, "maxItems": 16, "items": {"type": "string", "maxLength": 256}},
            "abstained": {"type": "boolean"},
            "answerable": {"type": "boolean"},
            "fallback": {"type": "object", "additionalProperties": false, "properties": {"mode": {"type": "string", "enum": ["abstain", "canonical_retrieval"]}, "reason": {"type": "string", "minLength": 1, "maxLength": 256}}, "required": ["mode", "reason"]},
            "exclusions": {"type": "array", "maxItems": 256, "items": {"type": "object", "additionalProperties": false, "properties": {"reason": {"type": "string", "maxLength": 256}, "count": {"type": "integer", "minimum": 1}}, "required": ["reason", "count"]}},
            "conflicts": {"type": "array", "maxItems": 128, "items": {"type": "object", "additionalProperties": false, "properties": {"reason": {"type": "string", "maxLength": 256}, "reference_count": {"type": "integer", "minimum": 0}, "references_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"}}, "required": ["reason", "reference_count", "references_sha256"]}}
          },
          "required": ["schema_version", "status", "recall_status", "reason", "reason_codes", "abstained", "answerable"]
        }
      }
    },
    "annotations": {
      "readOnlyHint": true
    },
    "title": "Semantic Search Entities"
  },
  {
    "name": "perseus_vault_ask",
    "description": "Ask a natural language question and get a grounded answer from stored memories via RAG. Internally recalls top-k entities, assembles context, and queries the configured LLM (Ollama) for an answer with cited sources. Requires --llm-endpoint to be set. LLM request timeout defaults to 30s; set PERSEUS_VAULT_LLM_TIMEOUT_SECS for large/cold models that need longer to load (#528).",
    "inputSchema": {
      "type": "object",
      "properties": {
        "query": {
          "type": "string",
          "description": "Natural language question to answer from stored memories"
        },
        "top_k": {
          "type": "integer",
          "default": 5,
          "description": "Number of top entities to use as context (max 20)"
        },
        "as_of_unix_ms": {
          "type": "integer",
          "description": "#472 Temporal RAG: answer from the memory context AS IT WAS BELIEVED at this transaction-time instant (unix ms) — the retrieved bodies are reconstructed to the versions live at that instant, so a corrected-later fact does not leak into the past answer. Combine with valid_at_unix_ms for the full bi-temporal cell. Omit for the live view."
        },
        "valid_at_unix_ms": {
          "type": "integer",
          "description": "#472 Temporal RAG: answer from the context that was TRUE IN THE WORLD at this valid-time instant (unix ms), per current (or as_of) knowledge. Omit for the live view."
        },
        "verify_stale_observations": {
          "type": "boolean",
          "default": true,
          "description": "#884: stale-observation gate. When true (default), observation sources with newer unconsolidated raw facts are verified against those facts before citation — consistent facts are cited with a 'verified against raw facts' note, contradicted observations are refused and reported in refused_sources. Set false to disable the gate."
        }
      },
      "required": [
        "query"
      ]
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "answer": {
          "type": "string",
          "description": "Grounded answer with cited sources"
        },
        "sources": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "key": {
                "type": "string"
              },
              "category": {
                "type": "string"
              },
              "score": {
                "type": "number"
              },
              "snippet": {
                "type": "string"
              }
            }
          },
          "description": "Cited source entities used in the answer"
        }
      }
    },
    "annotations": {
      "readOnlyHint": true,
      "destructiveHint": false
    },
    "title": "Ask Question from Memories"
  },
  {
    "name": "perseus_vault_get_entity",
    "description": "Get an entity by ID with its full body_json content. Use after perseus_vault_recall with preview_cap to read the complete body of a truncated result. The drill-down footer embedded in preview-capped results references this tool with the entity ID to use.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "id": {
          "type": "string",
          "description": "Entity ID to retrieve (from recall result id field or preview cap footer)"
        }
      },
      "required": [
        "id"
      ]
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "id": {
          "type": "string"
        },
        "category": {
          "type": "string"
        },
        "key": {
          "type": "string"
        },
        "body_json": {
          "type": "string",
          "description": "Full entity body content"
        },
        "status": {
          "type": "string"
        },
        "entity_type": {
          "type": "string"
        },
        "decay_score": {
          "type": "number"
        },
        "retrieval_count": {
          "type": "integer"
        },
        "layer": {
          "type": "string"
        },
        "always_on": {
          "type": "boolean"
        },
        "certainty": {
          "type": "number"
        }
      }
    },
    "annotations": {
      "readOnlyHint": true
    },
    "title": "Get Entity by ID"
  },
  {
    "name": "perseus_vault_history",
    "description": "List superseded (historical) versions of a fact (category + key), newest first. Each entry was the live fact for an interval before it was overwritten. The companion to perseus_vault_as_of: as_of returns the single version live at one instant; history returns the version trail. Paginated: returns the `limit` newest versions (default 20) starting at `offset`; `total` in the response is the FULL trail size, so total > returned means there are more pages. Returns an empty list if the fact has never been overwritten (its only version is the current live one in recall).",
    "inputSchema": {
      "type": "object",
      "properties": {
        "category": {
          "type": "string",
          "description": "Entity category"
        },
        "key": {
          "type": "string",
          "description": "Entity key within the category"
        },
        "limit": {
          "type": "integer",
          "default": 20,
          "description": "Maximum versions to return (newest first), 0-1000. Defaults to 20. 0 is count-only: returns no version bodies while `total` still reports the full trail size."
        },
        "offset": {
          "type": "integer",
          "default": 0,
          "description": "Number of newest versions to skip, for paging through a long trail."
        }
      },
      "required": [
        "category",
        "key"
      ]
    }
  },
  {
    "name": "perseus_vault_as_of",
    "description": "Transaction-time time-travel: return the version of a fact (category + key) that Perseus Vault believed at a given past instant. When a fact is overwritten, the prior version is kept in history; this returns whichever version was live at as_of_unix_ms. Use to answer 'what did we believe about X back then?' or to audit how a fact changed. For the orthogonal valid-time axis ('what was actually TRUE in the world at time T') use perseus_vault_valid_at; for both axes at once use perseus_vault_bitemporal. Returns found=false if the fact had not been recorded yet at that time. If the instant falls inside a window compacted by history retention (#398), returns an explicit marker (compacted=true, versions_compacted, digest) instead of the original — now unrecoverable — versions.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "category": {
          "type": "string",
          "description": "Entity category"
        },
        "key": {
          "type": "string",
          "description": "Entity key within the category"
        },
        "as_of_unix_ms": {
          "type": "integer",
          "description": "Transaction-time instant (unix ms) to travel to"
        }
      },
      "required": [
        "category",
        "key",
        "as_of_unix_ms"
      ]
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "found": {
          "type": "boolean",
          "description": "False if the fact had not been recorded by as_of_unix_ms"
        },
        "id": {
          "type": "string"
        },
        "category": {
          "type": "string"
        },
        "key": {
          "type": "string"
        },
        "body_json": {
          "type": "string",
          "description": "The fact's content as it was at as_of_unix_ms"
        },
        "status": {
          "type": "string"
        },
        "entity_type": {
          "type": "string"
        },
        "as_of_unix_ms": {
          "type": "integer"
        },
        "compacted": {
          "type": "boolean",
          "description": "Present and true when the instant falls inside a retention-compacted window: the result is a tombstone marker, not a real version (#398)"
        },
        "versions_compacted": {
          "type": "integer",
          "description": "How many original versions the compacted window rolled up (#398)"
        },
        "digest": {
          "type": "string",
          "description": "Hash-chain digest folded over the evicted versions (#398)"
        }
      }
    },
    "annotations": {
      "readOnlyHint": true
    },
    "title": "Time-Travel Entity Lookup"
  },
  {
    "name": "perseus_vault_valid_at",
    "description": "Valid-time (application-time) lookup: return the version of a fact (category + key) that — per CURRENT knowledge — was actually true in the world at a given instant. Orthogonal to perseus_vault_as_of: as_of answers 'what did we BELIEVE at time T' (transaction time); valid_at answers 'what WAS TRUE at time T, as we understand it now'. Facts carry a valid period [valid_from, valid_to) settable on perseus_vault_remember; a later-recorded version's claim supersedes earlier claims for the instants it covers. Returns found=false if no version's valid period contains the instant.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "category": {
          "type": "string",
          "description": "Entity category"
        },
        "key": {
          "type": "string",
          "description": "Entity key within the category"
        },
        "valid_at_unix_ms": {
          "type": "integer",
          "description": "World-instant (unix ms) to evaluate: which version was actually true then"
        }
      },
      "required": [
        "category",
        "key",
        "valid_at_unix_ms"
      ]
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "found": {
          "type": "boolean",
          "description": "False if no version's valid period contains the instant"
        },
        "id": {
          "type": "string"
        },
        "category": {
          "type": "string"
        },
        "key": {
          "type": "string"
        },
        "body_json": {
          "type": "string",
          "description": "The fact's content as it was true at the instant"
        },
        "status": {
          "type": "string"
        },
        "entity_type": {
          "type": "string"
        },
        "valid_from_unix_ms": {
          "type": "integer",
          "description": "Start of the matched version's valid period"
        },
        "valid_to_unix_ms": {
          "type": "integer",
          "description": "End of the matched version's valid period (absent = still true)"
        },
        "recorded_at_unix_ms": {
          "type": "integer",
          "description": "Transaction time the matched version was recorded"
        },
        "is_live_version": {
          "type": "boolean",
          "description": "True when the matched version is the current live row (not superseded)"
        },
        "valid_at_unix_ms": {
          "type": "integer"
        }
      }
    },
    "annotations": {
      "readOnlyHint": true
    },
    "title": "Valid-Time Lookup (What Was True)"
  },
  {
    "name": "perseus_vault_bitemporal",
    "description": "Full bi-temporal query (SQL:2011 SYSTEM_TIME + APPLICATION_TIME): 'as of transaction time tx_at, which version did we believe was true in the world at valid time valid_at?' Returns the exact cell of the bi-temporal rectangle — the audit-grade 'who knew what, as-of-when' question. Combines both axes: perseus_vault_as_of is this with valid_at pinned to tx_at; perseus_vault_valid_at is this with tx_at pinned to now. Retroactive and proactive updates land in the correct rectangle cell. Returns found=false if nothing recorded by tx_at was valid at valid_at.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "category": {
          "type": "string",
          "description": "Entity category"
        },
        "key": {
          "type": "string",
          "description": "Entity key within the category"
        },
        "tx_at_unix_ms": {
          "type": "integer",
          "description": "Transaction-time instant (unix ms): reconstruct knowledge as of this moment"
        },
        "valid_at_unix_ms": {
          "type": "integer",
          "description": "Valid-time instant (unix ms): the world-moment being asked about"
        }
      },
      "required": [
        "category",
        "key",
        "tx_at_unix_ms",
        "valid_at_unix_ms"
      ]
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "found": {
          "type": "boolean",
          "description": "False if nothing recorded by tx_at was valid at valid_at"
        },
        "id": {
          "type": "string"
        },
        "category": {
          "type": "string"
        },
        "key": {
          "type": "string"
        },
        "body_json": {
          "type": "string",
          "description": "The version occupying that bi-temporal rectangle cell"
        },
        "status": {
          "type": "string"
        },
        "entity_type": {
          "type": "string"
        },
        "valid_from_unix_ms": {
          "type": "integer"
        },
        "valid_to_unix_ms": {
          "type": "integer"
        },
        "recorded_at_unix_ms": {
          "type": "integer"
        },
        "invalidated_at_unix_ms": {
          "type": "integer",
          "description": "Transaction time this version was retired (absent = live)"
        },
        "is_live_version": {
          "type": "boolean"
        },
        "tx_at_unix_ms": {
          "type": "integer"
        },
        "valid_at_unix_ms": {
          "type": "integer"
        }
      }
    },
    "annotations": {
      "readOnlyHint": true
    },
    "title": "Bi-Temporal Rectangle Query"
  },
  {
    "name": "perseus_vault_forget",
    "description": "Soft-delete an entity by setting archived=1. The entity is hidden from queries but recoverable. Use this to clean up stale or incorrect facts without permanent data loss.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "category": {
          "type": "string",
          "description": "Entity category to archive"
        },
        "key": {
          "type": "string",
          "description": "Entity key to archive"
        },
        "reason": {
          "type": "string",
          "default": "",
          "description": "Reason for archiving, logged for audit trail"
        }
      },
      "required": [
        "category",
        "key"
      ]
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "found": {
          "type": "boolean",
          "description": "Whether the entity was found and archived"
        },
        "category": {
          "type": "string",
          "description": "Entity category"
        },
        "key": {
          "type": "string",
          "description": "Entity key"
        }
      }
    },
    "annotations": {
      "destructiveHint": true
    },
    "title": "Forget Entity (Soft-Delete)"
  },
  {
    "name": "perseus_vault_ingest",
    "description": "Sync external data connectors (GitHub issues, file watcher) into Perseus Vault. Call with no arguments to run all enabled connectors, or specify a connector name to run only that one. Use dry_run=true to preview without storing. Unchanged content from a previous successful ingest is skipped as zero-work revalidation (provenance-admission containment replay, #1050); use force_reingest=true to bypass.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "connector": {
          "type": "string",
          "description": "Specific connector to run (omit for all enabled)"
        },
        "dry_run": {
          "type": "boolean",
          "default": false,
          "description": "Preview documents without storing them"
        },
        "force_reingest": {
          "type": "boolean",
          "default": false,
          "description": "Bypass the containment replay gate and re-admit every fetched document (#1050)"
        }
      }
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "ingested": {
          "type": "integer",
          "description": "Number of documents ingested (or would be ingested in dry run)"
        },
        "contained": {
          "type": "integer",
          "description": "Documents skipped as already-covered by a live entity (zero-work revalidation, #1050)"
        },
        "dry_run": {
          "type": "boolean",
          "description": "Whether this was a dry run"
        },
        "errors": {
          "type": "array",
          "items": {
            "type": "string"
          },
          "description": "Error messages from connectors that failed"
        }
      }
    },
    "annotations": {
      "destructiveHint": true
    },
    "title": "Ingest External Data"
  },
  {
    "name": "perseus_vault_ingest_file",
    "description": "Ingest a document file into memory by extracting its text LOCALLY (no cloud, no network). Plaintext/markdown/structured-text work in any build; DOCX and PDF require a binary built with --features multimodal (otherwise a clear error is returned). The extracted text is stored as a normal entity (recallable via perseus_vault_recall). category defaults to 'document', key defaults to the file name.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "path": {
          "type": "string",
          "description": "Path to the document file to ingest"
        },
        "category": {
          "type": "string",
          "description": "Entity category (default 'document')"
        },
        "key": {
          "type": "string",
          "description": "Entity key (default: the file name)"
        },
        "tags": {
          "type": "array",
          "items": {
            "type": "string"
          },
          "description": "Optional tags"
        }
      },
      "required": [
        "path"
      ]
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "id": {
          "type": "string",
          "description": "Stored entity id"
        },
        "action": {
          "type": "string",
          "description": "created or updated"
        },
        "category": {
          "type": "string"
        },
        "key": {
          "type": "string"
        },
        "chars": {
          "type": "integer",
          "description": "Characters of text extracted"
        }
      }
    },
    "annotations": {
      "destructiveHint": true
    },
    "title": "Ingest Document File"
  },
  {
    "name": "perseus_vault_artifact_register",
    "description": "Register an immutable artifact by reading a local file, hashing its exact bytes with full SHA-256, and storing a scope-bound metadata binding plus the preserved original bytes. Returns the compact deterministic manifest by default. This first slice accepts only uncompressed source bytes so retrieval anchors stay exact to the original artifact.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "path": { "type": "string", "description": "Local file path to register" },
        "mime_type": { "type": "string", "description": "Optional MIME type override; otherwise inferred from the file extension" },
        "workspace_hash": { "type": "string", "default": "", "description": "Workspace scope for the metadata binding. Omit/empty = global." },
        "agent_id": { "type": "string", "default": "", "description": "Owning agent id for visibility checks." },
        "visibility": { "type": "string", "default": "workspace", "description": "private | fleet | workspace | tenant | public" },
        "origin": { "type": "object", "description": "Optional origin/provenance metadata using the existing memory-origin contract." },
        "external_refs": { "type": "array", "items": { "type": "object" }, "description": "Optional external source anchors; pointers only, never access grants." },
        "retention_policy": { "type": "string", "description": "Optional retention policy from the existing vocabulary." },
        "representation": { "type": "object", "description": "original or derived representation metadata; derived artifacts must point at a full parent SHA-256." }
      },
      "required": ["path"]
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "sha256": { "type": "string" },
        "artifact_action": { "type": "string" },
        "binding_action": { "type": "string" },
        "manifest": { "type": "object" }
      }
    },
    "annotations": {
      "destructiveHint": true
    },
    "title": "Register Immutable Artifact"
  },
  {
    "name": "perseus_vault_learned_artifact_register",
    "description": "#876 governed distillation: register a learned-memory artifact (trained weights / distilled cartridge) bound to its source entities with hash-only evidence, gated fail-closed on a COMPLETED 'learned_memory' action receipt (no receipt, no registration). Every source (category, key) in the workspace is snapshotted (entity id + normalized body digest + recorded_at) into learned_artifact_sources; physically erasing or purging a source revokes the binding (serve paths refuse revoked artifacts), superseding a source flags it stale (retraining trigger). Returns the artifact sha256, source-bindings count, and receipt-replay evidence.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "path": { "type": "string", "description": "Local file path to register (trained artifact / cartridge)" },
        "mime_type": { "type": "string", "description": "Optional MIME type override; otherwise inferred from the file extension" },
        "workspace_hash": { "type": "string", "default": "", "description": "Workspace scope for the metadata binding. Omit/empty = global." },
        "agent_id": { "type": "string", "default": "", "description": "Owning agent id for visibility checks." },
        "visibility": { "type": "string", "default": "workspace", "description": "private | fleet | workspace | tenant | public" },
        "action_id": { "type": "string", "description": "Action id of a COMPLETED 'learned_memory' action receipt (intent -> lease -> complete); the gate refuses registration without it." },
        "source_entities": { "type": "array", "items": { "type": "array", "items": { "type": "string" }, "minItems": 2, "maxItems": 2 }, "description": "(category, key) pairs the artifact was distilled from; snapshotted hash-only at registration." },
        "external_refs": { "type": "array", "items": { "type": "object" }, "description": "Optional external source anchors; pointers only, never access grants." },
        "retention_policy": { "type": "string", "description": "Optional retention policy from the existing vocabulary." },
        "derivation_version": { "type": "string", "description": "Optional distillation pipeline version tag." }
      },
      "required": ["path", "action_id", "source_entities"]
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "sha256": { "type": "string" },
        "artifact_action": { "type": "string" },
        "binding_action": { "type": "string" },
        "source_bindings_count": { "type": "integer" },
        "action_id": { "type": "string" },
        "evidence": { "type": "object" },
        "manifest": { "type": "object" }
      }
    },
    "annotations": {
      "destructiveHint": false
    },
    "title": "Register Governed Learned Artifact"
  },
  {
    "name": "perseus_vault_workspace_bind",
    "description": "#879: bind a Hermes profile to a Vault workspace (one profile <-> one workspace; re-binding switches workspace and resets lifecycle state). access_mode read_write | read_only; read_only bindings deny mutations at the tool boundary. Journaled (workspace_bound / workspace_rebound).",
    "inputSchema": {
      "type": "object",
      "properties": {
        "profile_name": { "type": "string", "description": "Hermes profile name (must match the MCP clientInfo.name used at handshake)" },
        "workspace_hash": { "type": "string", "default": "", "description": "Workspace to bind the profile to" },
        "access_mode": { "type": "string", "default": "read_write", "enum": ["read_write", "read_only"], "description": "read_write or read_only" },
        "metadata": { "type": "object", "description": "Optional metadata (host, hermes version, actor, ...)" }
      },
      "required": ["profile_name", "workspace_hash"]
    },
    "outputSchema": { "type": "object" },
    "title": "Bind Hermes Profile to Workspace"
  },
  {
    "name": "perseus_vault_workspace_unbind",
    "description": "#879: unbind a Hermes profile from its workspace (lifecycle: active/quarantined -> unbound; row retained for audit). Journaled (workspace_unbound).",
    "inputSchema": {
      "type": "object",
      "properties": {
        "profile_name": { "type": "string", "description": "Hermes profile name to unbind" },
        "reason": { "type": "string", "description": "Unbind reason (journaled)" }
      },
      "required": ["profile_name"]
    },
    "outputSchema": { "type": "object" },
    "title": "Unbind Hermes Profile"
  },
  {
    "name": "perseus_vault_workspace_quarantine",
    "description": "#879: operator lifecycle control — quarantine an active binding (stops all access until reactivated) or reactivate a quarantined/unbound binding. Journaled (workspace_quarantined / workspace_reactivated).",
    "inputSchema": {
      "type": "object",
      "properties": {
        "profile_name": { "type": "string", "description": "Hermes profile name" },
        "action": { "type": "string", "default": "quarantine", "enum": ["quarantine", "reactivate"], "description": "quarantine or reactivate" },
        "reason": { "type": "string", "description": "Reason (required for quarantine, journaled)" }
      },
      "required": ["profile_name"]
    },
    "outputSchema": { "type": "object" },
    "title": "Quarantine or Reactivate Profile Binding"
  },
  {
    "name": "perseus_vault_workspace_status",
    "description": "#879: diagnostics — all profile <-> workspace bindings with lifecycle state, access mode, heartbeat, and staleness signal; distinguishes live, stale, quarantined, and unbound bindings.",
    "inputSchema": {
      "type": "object",
      "properties": {}
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "bindings": { "type": "array", "items": { "type": "object" } },
        "count": { "type": "integer" }
      }
    },
    "title": "Workspace Binding Status"
  },
  {
    "name": "perseus_vault_artifact_manifest",
    "description": "Serve the compact deterministic manifest for one artifact identity after scope + visibility filtering. When workspace_hash is omitted, only global bindings are considered — an artifact hash alone is a pointer, not an access grant.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "sha256": { "type": "string", "description": "Full 64-hex SHA-256 content identity" },
        "workspace_hash": { "type": "string", "description": "Exact workspace scope to read; omit for global-only." },
        "requesting_agent_id": { "type": "string", "description": "Optional requesting agent id for visibility filtering." }
      },
      "required": ["sha256"]
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "sha256": { "type": "string" },
        "byte_length": { "type": "integer" },
        "structure": { "type": "object" },
        "significant_signals": { "type": "array", "items": { "type": "string" } },
        "available_retrievals": { "type": "object" },
        "visible_binding_count": { "type": "integer" },
        "bindings": { "type": "array", "items": { "type": "object" } }
      }
    },
    "title": "Serve Artifact Manifest"
  },
  {
    "name": "perseus_vault_artifact_excerpt",
    "description": "Retrieve an exact bounded excerpt from the preserved original artifact bytes by either a half-open byte range [start,end) or an inclusive 1-indexed line range. Returns exact source anchors plus base64 bytes, and UTF-8 text when the slice decodes cleanly.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "sha256": { "type": "string", "description": "Full 64-hex SHA-256 content identity" },
        "workspace_hash": { "type": "string", "description": "Exact workspace scope to read; omit for global-only." },
        "requesting_agent_id": { "type": "string", "description": "Optional requesting agent id for visibility filtering." },
        "byte_start": { "type": "integer", "description": "Byte-range start offset (inclusive)" },
        "byte_end": { "type": "integer", "description": "Byte-range end offset (exclusive)" },
        "line_start": { "type": "integer", "description": "Line-range start (1-indexed, inclusive)" },
        "line_end": { "type": "integer", "description": "Line-range end (1-indexed, inclusive)" }
      },
      "required": ["sha256"]
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "sha256": { "type": "string" },
        "range": { "type": "object" },
        "content_b64": { "type": "string" },
        "content_utf8": { "type": ["string", "null"] },
        "anchors": { "type": "array", "items": { "type": "object" } },
        "why_served": { "type": "object" }
      }
    },
    "title": "Retrieve Exact Artifact Excerpt"
  },
  {
    "name": "perseus_vault_artifact_log_digest",
    "description": "Build a deterministic, evidence-preserving navigation digest over a visible UTF-8 log artifact. Repeated non-protected templates are collapsed with exact counts and first/last source anchors. Lines containing error, warn, exception, fatal, panic, denied, refused, timeout, assertion, or traceback remain verbatim. This is never an LLM summary or replacement for original bytes.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "sha256": { "type": "string", "description": "Full 64-hex SHA-256 content identity" },
        "workspace_hash": { "type": "string", "description": "Exact workspace scope to read; omit for global-only." },
        "requesting_agent_id": { "type": "string", "description": "Optional requesting agent id for visibility filtering." }
      },
      "required": ["sha256"]
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "format": { "type": "string" },
        "source_sha256": { "type": "string" },
        "config_version": { "type": "string" },
        "input_line_count": { "type": "integer" },
        "omitted_line_count": { "type": "integer" },
        "protected_line_count": { "type": "integer" },
        "sections": { "type": "array", "items": { "type": "object" } },
        "protected_lines": { "type": "array", "items": { "type": "array" } },
        "retrieval": { "type": "string" }
      }
    },
    "title": "Build Deterministic Evidence-Preserving Log Digest"
  },
  {
    "name": "perseus_vault_artifact_verify_value",
    "description": "Verify that a candidate value occurs verbatim in the preserved original artifact bytes, with bounded exact-match search only (no regex). Returns exact source anchors for each match found.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "sha256": { "type": "string", "description": "Full 64-hex SHA-256 content identity" },
        "workspace_hash": { "type": "string", "description": "Exact workspace scope to read; omit for global-only." },
        "requesting_agent_id": { "type": "string", "description": "Optional requesting agent id for visibility filtering." },
        "candidate": { "type": "string", "description": "Candidate value to verify: UTF-8 text by default, or base64 when encoding='base64'." },
        "encoding": { "type": "string", "default": "utf8", "description": "utf8 | base64" },
        "max_matches": { "type": "integer", "default": 5, "description": "Maximum exact-match anchors to return (bounded)." }
      },
      "required": ["sha256", "candidate"]
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "sha256": { "type": "string" },
        "candidate_encoding": { "type": "string" },
        "candidate_byte_length": { "type": "integer" },
        "match_count": { "type": "integer" },
        "truncated": { "type": "boolean" },
        "matches": { "type": "array", "items": { "type": "object" } },
        "why_served": { "type": "object" }
      }
    },
    "title": "Verify Candidate Against Original Artifact Bytes"
  },
  {
    "name": "perseus_vault_embed",
    "description": "Generate and store dense vector embeddings for entities via Ollama /api/embed. Supports single entity (category+key) or batch mode (batch_category). Requires --llm-endpoint to be set. #885: also the operator surface for optional quantized embedding storage — quant_mode converts ALL stored float32 embeddings to int8 or bit (MIB-style sign-bit vectors scored by Hamming) in one transaction with a pre-quantization snapshot; restore_quantized_backup rolls back losslessly from that snapshot; drop_quantized_backup removes it after verification.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "text": {
          "type": "string",
          "description": "Text to embed (omit to use entity body_json)"
        },
        "category": {
          "type": "string",
          "description": "Entity category for single mode"
        },
        "key": {
          "type": "string",
          "description": "Entity key for single mode"
        },
        "batch_category": {
          "type": "string",
          "description": "Embed all entities in this category lacking embeddings"
        },
        "batch_limit": {
          "type": "integer",
          "default": 100,
          "description": "Max entities in batch mode"
        },
        "quant_mode": {
          "type": "string",
          "enum": ["int8", "bit"],
          "description": "Store-wide reindex: convert ALL stored embeddings from float32 to int8 or bit (one transaction; pre-quantization float32 snapshot created once). Refused when already quantized — restore first."
        },
        "restore_quantized_backup": {
          "type": "boolean",
          "default": false,
          "description": "Roll back the embedding column to float32 from the pre-quantization snapshot (lossless for rows that existed at quantization time)"
        },
        "drop_quantized_backup": {
          "type": "boolean",
          "default": false,
          "description": "Drop the pre-quantization snapshot after verifying the quantized store (irreversible; rollback then requires re-embed)"
        }
      }
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "embedded": {
          "type": "integer",
          "description": "Number of entities embedded"
        },
        "dimensions": {
          "type": "integer",
          "description": "Vector dimensions"
        }
      }
    },
    "annotations": {
      "destructiveHint": true
    },
    "title": "Generate Entity Embeddings"
  },
  {
    "name": "perseus_vault_prune",
    "description": "Bulk archive entities by category, decay threshold, or age. Use dry_run=true to preview without archiving. Useful for cleaning stale or low-quality memories. With scope='history' (#398) it instead evicts old superseded versions from entity_history under the given (or env-configured PERSEUS_VAULT_HISTORY_*) bounds, rolling each evicted run into a compaction tombstone; dry_run reports the rows and bytes that would be evicted.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "category": {
          "type": "string",
          "description": "Archive entities in this category"
        },
        "min_decay": {
          "type": "number",
          "description": "Archive entities with decay_score below this threshold"
        },
        "older_than_days": {
          "type": "integer",
          "description": "Archive entities older than this many days"
        },
        "limit": {
          "type": "integer",
          "default": 100,
          "description": "Max entities to prune (0 = unlimited)"
        },
        "scope": {
          "type": "string",
          "enum": ["entities", "history"],
          "description": "'history' prunes superseded versions from entity_history under retention bounds instead of archiving live entities (#398)"
        },
        "max_age_days": {
          "type": "integer",
          "description": "scope='history': evict versions invalidated more than this many days ago (overrides PERSEUS_VAULT_HISTORY_MAX_AGE_DAYS)"
        },
        "max_versions_per_key": {
          "type": "integer",
          "description": "scope='history': keep at most this many stored versions per key, oldest evicted first (overrides PERSEUS_VAULT_HISTORY_MAX_VERSIONS_PER_KEY)"
        },
        "max_bytes": {
          "type": "integer",
          "description": "scope='history': global stored-history byte budget, globally-oldest evicted first (overrides PERSEUS_VAULT_HISTORY_MAX_BYTES)"
        },
        "dry_run": {
          "type": "boolean",
          "default": false,
          "description": "Preview without archiving/evicting"
        }
      }
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "archived": {
          "type": "integer"
        },
        "examined": {
          "type": "integer"
        },
        "dry_run": {
          "type": "boolean"
        },
        "reason": {
          "type": "string"
        }
      }
    },
    "annotations": {
      "destructiveHint": true
    },
    "title": "Prune Stale Entities"
  },
  {
    "name": "perseus_vault_link",
    "description": "Create a relationship link from one entity to another. Builds a knowledge graph that perseus_vault_traverse can walk. Use 'depends_on', 'implements', 'extends', 'references', or custom relationships.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "from_category": {
          "type": "string",
          "description": "Source entity category"
        },
        "from_key": {
          "type": "string",
          "description": "Source entity key"
        },
        "to_id": {
          "type": "string",
          "description": "Target entity ID (from perseus_vault_remember return value)"
        },
        "relationship": {
          "type": "string",
          "default": "related",
          "description": "Relationship type: 'depends_on', 'implements', 'extends', 'references', or custom"
        }
      },
      "required": [
        "from_category",
        "from_key",
        "to_id"
      ]
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "success": {
          "type": "boolean"
        },
        "from": {
          "type": "string",
          "description": "Source as 'category/key'"
        },
        "to": {
          "type": "string",
          "description": "Target entity ID"
        },
        "relationship": {
          "type": "string",
          "description": "Relationship type set"
        }
      }
    },
    "annotations": {
      "destructiveHint": true
    },
    "title": "Link Entities"
  },
  {
    "name": "perseus_vault_unlink",
    "description": "Remove a relationship link from one entity to another. Use this to correct outdated or incorrect links in the knowledge graph.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "from_category": {
          "type": "string",
          "description": "Source entity category"
        },
        "from_key": {
          "type": "string",
          "description": "Source entity key"
        },
        "to_id": {
          "type": "string",
          "description": "Target entity ID to unlink"
        }
      },
      "required": [
        "from_category",
        "from_key",
        "to_id"
      ]
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "success": {
          "type": "boolean"
        },
        "from": {
          "type": "string",
          "description": "Source as 'category/key'"
        },
        "to": {
          "type": "string",
          "description": "Target entity ID"
        }
      }
    },
    "annotations": {
      "destructiveHint": true
    },
    "title": "Unlink Entities"
  },
  {
    "name": "perseus_vault_journal",
    "description": "Append a structured decision/observation log entry. Uses evaluated/acted/forward pattern: what was considered, what was done, and what happens next. Essential for audit trails and timeline reconstruction. Public admission_source events additionally require an initialized clientInfo.name and an enforce-mode memory.admission.source authority for the exact workspace; caller-supplied identities are never authoritative.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "event_type": {
          "type": "string",
          "default": "decision",
          "description": "Event type: 'decision', 'observation', 'action', 'error'"
        },
        "evaluated": {
          "type": "object",
          "description": "What was evaluated: options considered, context, constraints"
        },
        "acted": {
          "type": "object",
          "description": "What action was taken and why"
        },
        "forward": {
          "type": "object",
          "description": "What the plan is going forward"
        },
        "category": {
          "type": "string",
          "description": "Related entity category for linking"
        },
        "key": {
          "type": "string",
          "description": "Related entity key for linking"
        },
        "entity_id": {
          "type": "string",
          "description": "Related entity ID for linking"
        },
        "agent_id": {
          "type": "string",
          "default": "",
          "description": "Agent identity (v1.2.0). Records which agent created this journal event."
        },
        "workspace_hash": {
          "type": "string",
          "default": "",
          "description": "Explicit workspace attribution for the journal event; empty string denotes the global partition."
        },
        "requesting_agent_id": {
          "type": "string",
          "description": "Transport-stamped caller identity; required for admission_source events."
        },
        "source_attestation": {
          "type": "string",
          "minLength": 64,
          "maxLength": 64,
          "pattern": "^[0-9a-fA-F]{64}$",
          "description": "HMAC-SHA256 attestation over the canonical admission-source fields; required for public admission_source events and never stored."
        }
      },
      "required": [],
      "allOf": [
        {
          "if": {
            "properties": {
              "event_type": {
                "const": "admission_source"
              }
            },
            "required": ["event_type"]
          },
          "then": {
            "required": [
              "evaluated",
              "workspace_hash",
              "requesting_agent_id",
              "source_attestation"
            ],
            "properties": {
              "evaluated": {
                "type": "object",
                "required": [
                  "record_digest",
                  "source_identity",
                  "workspace_hash",
                  "actor_kind",
                  "actor_identity"
                ],
                "properties": {
                  "record_digest": {
                    "type": "string",
                    "pattern": "^[0-9a-fA-F]{64}$"
                  },
                  "source_identity": {
                    "type": "string",
                    "minLength": 1
                  },
                  "workspace_hash": {
                    "type": "string",
                    "minLength": 1
                  },
                  "actor_kind": {
                    "type": "string",
                    "minLength": 1
                  },
                  "actor_identity": {
                    "type": "string",
                    "minLength": 1
                  }
                }
              },
              "workspace_hash": {
                "minLength": 1
              },
              "requesting_agent_id": {
                "minLength": 1
              },
              "source_attestation": {
                "type": "string",
                "minLength": 64,
                "maxLength": 64,
                "pattern": "^[0-9a-fA-F]{64}$"
              }
            }
          }
        }
      ]
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "id": {
          "type": "string",
          "description": "Journal event ID"
        },
        "event_type": {
          "type": "string",
          "description": "Event type recorded"
        },
        "created_at_unix_ms": {
          "type": "integer",
          "description": "Creation timestamp in unix milliseconds"
        }
      }
    },
    "annotations": {
      "destructiveHint": true
    },
    "title": "Append Journal Entry"
  },
  {
    "name": "perseus_vault_check_failure_pattern",
    "description": "Deja-vu guard (#521): call BEFORE retrying a failed command or committing to an approach. Checks the action against workspace-scoped prior failures in both the journal (error events and failure-marked acted/forward payloads) and the entity store (failure/pitfall/root-cause memories), ranked by similarity, recency, and trust. Returns matching prior failures with the recorded cause and resolution, a deja_vu flag, and a one-line warning when the action was already tried and failed. Read-only: never bumps retrieval counts or decay. Record failures via perseus_vault_journal (event_type 'error') or perseus_vault_remember so the guard can find them.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "action": {
          "type": "string",
          "description": "The command line or approach description you are about to (re)try, e.g. 'cargo build --no-default-features' or 'parse the changelog with a regex'"
        },
        "workspace_hash": {
          "type": "string",
          "description": "Required workspace scope. Use an empty string only for the explicit global partition; other workspaces are never searched."
        },
        "limit": {
          "type": "integer",
          "default": 5,
          "description": "Maximum number of matches to return (1-50)"
        }
      },
      "required": [
        "action",
        "workspace_hash"
      ]
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "matches": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "workspace_hash": {
                "type": "string",
                "description": "Stored workspace scope of the matched failure; empty means the global partition."
              }
            }
          },
          "description": "Prior failures matching the action, best first. Each includes source, ref, workspace_hash, when (unix ms), what_failed, cause, resolution, and score."
        },
        "deja_vu": {
          "type": "boolean",
          "description": "True when at least one prior recorded failure matches the action"
        },
        "warning": {
          "type": "string",
          "description": "One-line agent-actionable deja-vu warning (present only when matches exist)"
        },
        "message": {
          "type": "string",
          "description": "Unambiguous empty state ('no prior failures recorded matching this action') when nothing matches"
        }
      }
    },
    "annotations": {
      "readOnlyHint": true
    },
    "title": "Check Failure Pattern (Deja-Vu Guard)"
  },
  {
    "name": "perseus_vault_timeline",
    "description": "Query workspace-scoped journal events by time range with optional filters for event type, category, or entity. Use this to reconstruct the decision history and understand what happened when.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "workspace_hash": {
          "type": "string",
          "description": "Required workspace scope. Use an empty string only for the explicit global partition."
        },
        "from_ms": {
          "type": "integer",
          "description": "Start time boundary in unix milliseconds"
        },
        "to_ms": {
          "type": "integer",
          "description": "End time boundary in unix milliseconds"
        },
        "event_type": {
          "type": "string",
          "description": "Filter by event type: 'decision', 'observation', 'action', 'error'"
        },
        "category": {
          "type": "string",
          "description": "Filter by related entity category"
        },
        "entity_id": {
          "type": "string",
          "description": "Filter by related entity ID"
        },
        "limit": {
          "type": "integer",
          "default": 50,
          "description": "Maximum number of events to return (max 1000)"
        },
        "offset": {
          "type": "integer",
          "default": 0,
          "description": "Number of events to skip for pagination"
        }
      },
      "required": [
        "workspace_hash"
      ]
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "items": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "id": {
                "type": "string"
              },
              "event_type": {
                "type": "string"
              },
              "workspace_hash": {
                "type": "string",
                "description": "Stored workspace attribution; empty string denotes the global partition."
              }
            }
          },
          "description": "Journal events matching the query"
        },
        "total": {
          "type": "integer",
          "description": "Number of events returned"
        }
      }
    },
    "annotations": {
      "readOnlyHint": true
    },
    "title": "Query Journal Timeline"
  },
  {
    "name": "perseus_vault_state_set",
    "description": "Set a key-value state entry with optional TTL for auto-expiration. Use this for session state, temporary flags, or configuration values that should expire after a set time.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "key": {
          "type": "string",
          "description": "State key — unique identifier for this state entry"
        },
        "value_json": {
          "type": "string",
          "description": "JSON value to store"
        },
        "ttl_seconds": {
          "type": "integer",
          "description": "Time-to-live in seconds. Entry auto-expires and returns null after this duration. Omit for permanent state."
        }
      },
      "required": [
        "key",
        "value_json"
      ]
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "key": {
          "type": "string",
          "description": "State key set"
        },
        "ttl_seconds": {
          "type": "integer",
          "description": "TTL that was set, if any"
        },
        "expires_at_unix_ms": {
          "type": "integer",
          "description": "Expiration timestamp in unix milliseconds, if TTL was set"
        }
      }
    },
    "annotations": {
      "destructiveHint": true
    },
    "title": "Set State Entry"
  },
  {
    "name": "perseus_vault_state_get",
    "description": "Get a state value by key. Returns null if the key has expired or doesn't exist. Use this instead of perseus_vault_recall for transient session state that doesn't need FTS5 search.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "key": {
          "type": "string",
          "description": "State key to retrieve"
        }
      },
      "required": [
        "key"
      ]
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "found": {
          "type": "boolean",
          "description": "Whether the key exists and hasn't expired"
        },
        "key": {
          "type": "string",
          "description": "State key requested"
        },
        "value": {
          "type": "string",
          "description": "JSON value if found"
        },
        "expires_at_unix_ms": {
          "type": "integer",
          "description": "Expiration timestamp if TTL was set"
        },
        "created_at_unix_ms": {
          "type": "integer",
          "description": "Creation timestamp"
        }
      }
    },
    "annotations": {
      "readOnlyHint": true
    },
    "title": "Get State Entry"
  },
  {
    "name": "perseus_vault_state_delete",
    "description": "Delete a state entry by key. Permanent removal — unlike perseus_vault_forget which is a soft-delete. Use this to clean up expired or unused state entries.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "key": {
          "type": "string",
          "description": "State key to permanently delete"
        }
      },
      "required": [
        "key"
      ]
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "found": {
          "type": "boolean",
          "description": "Whether the key existed and was deleted"
        },
        "key": {
          "type": "string",
          "description": "Key that was deleted"
        }
      }
    },
    "annotations": {
      "destructiveHint": true
    },
    "title": "Delete State Entry"
  },
  {
    "name": "perseus_vault_state_list",
    "description": "List all state keys, optionally filtered by a key prefix. Use this to discover what state entries exist without knowing exact keys ahead of time.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "prefix": {
          "type": "string",
          "default": "",
          "description": "Only return keys that start with this prefix"
        }
      },
      "required": []
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "keys": {
          "type": "array",
          "items": {
            "type": "string"
          },
          "description": "Matching state keys"
        },
        "total": {
          "type": "integer",
          "description": "Number of keys returned"
        }
      }
    },
    "annotations": {
      "readOnlyHint": true
    },
    "title": "List State Entries"
  },
  {
    "name": "perseus_vault_health",
    "description": "Cheap readiness probe for the vault server and its SQLite database. Returns healthy/unhealthy plus a readiness snapshot: `ready` (DB answers AND at least one active memory), `active_memories`, `embedded_memories`, `semantic_recall` (available|no_coverage|disabled), `db_path`, and `warnings[]` with likely causes. Call this before a recall-heavy workflow, or when recall unexpectedly returns empty, to tell an empty/degraded store apart from a broken MCP child. Use perseus_vault_stats for detailed statistics.",
    "inputSchema": {
      "type": "object",
      "properties": {}
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "status": {
          "type": "string",
          "enum": [
            "healthy",
            "unhealthy"
          ],
          "description": "Server health status (healthy iff the DB responds)"
        },
        "db_path": {
          "type": "string",
          "description": "Absolute path of the SQLite file this server is bound to (#671)"
        },
        "ready": {
          "type": "boolean",
          "description": "True when the DB responds AND the store has at least one active memory — i.e. recall can return non-empty results"
        },
        "active_memories": {
          "type": "integer",
          "description": "Count of non-archived memories (the set recall reads)"
        },
        "embedded_memories": {
          "type": "integer",
          "description": "Count of active memories carrying a dense embedding"
        },
        "semantic_recall": {
          "type": "string",
          "enum": [
            "available",
            "no_coverage",
            "disabled"
          ],
          "description": "Dense/hybrid posture: available (backend on, coverage present), no_coverage (backend on, nothing embedded), or disabled (keyword-only build/config)"
        },
        "warnings": {
          "type": "array",
          "items": { "type": "string" },
          "description": "Likely-cause messages for degraded/empty states; empty when nominal"
        },
        "binary_stale": {
          "type": "boolean",
          "description": "True when the running binary was replaced on disk since this process started (#858): results come from a stale image — call perseus_vault_handoff_restart or restart the session"
        },
        "binary_path": {
          "type": "string",
          "description": "Absolute path of the running binary (empty when undeterminable)"
        },
        "pid": {
          "type": "integer",
          "description": "PID of the running server process"
        }
      }
    },
    "annotations": {
      "readOnlyHint": true
    },
    "title": "Check Health"
  },
  {
    "name": "perseus_vault_deployment_profile",
    "description": "Resolved runtime deployment profile (#870): one machine-readable answer to 'what is this vault actually connected to?'. Reports the profile class (`offline` | `local_only` | `local_with_approved_network` | `external_actions_enabled`), model backend (bundled/ollama/provider/none), embedding backend (kind + available + degraded — a missing/unavailable local backend is reported as degraded, never silently reclassified as empty success), network listeners and non-loopback egress hosts (hosts only — sanitized, no URLs/keys/raw bodies), connectors, cloud-provider use, external-mutation posture, encryption at rest (aes_256_gcm|plaintext + storage-state probe) and in transit, and raw-retention policy. Describes ACTUAL runtime state: offline mode zeroes web/LLM/embedding/connectors at startup, and the profile reflects the effective flags. Read-only.",
    "inputSchema": {
      "type": "object",
      "properties": {}
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "profile": {
          "type": "string",
          "enum": ["offline", "local_only", "local_with_approved_network", "external_actions_enabled"],
          "description": "Derived deployment class from runtime state"
        },
        "model_backend": { "type": "object", "description": "kind/model/available/degraded" },
        "embedding_backend": { "type": "object", "description": "kind/available/degraded/semantic_recall" },
        "network": { "type": "object", "description": "listeners/egress_hosts/loopback_only" },
        "connectors": { "type": "array", "description": "name/remote/remote_host" },
        "cloud_provider_use": { "type": "string", "description": "'none' or comma-joined non-loopback hosts" },
        "external_mutations": { "type": "string", "enum": ["disabled", "enabled"] },
        "encryption": { "type": "object", "description": "at_rest/storage_state/in_transit" },
        "raw_retention": { "type": "object", "description": "memory_bodies/raw_logs" }
      }
    },
    "annotations": {
      "readOnlyHint": true
    },
    "title": "Deployment Profile"
  },
  {
    "name": "perseus_vault_config_report",
    "description": "Per-stage provider/config self-report with a requested-vs-resolved diff (#1010). One machine-readable answer to 'did every pipeline stage actually resolve the configuration I asked for?' Reports six stages — embedding_backend, model_backend, quantization, db_path, encryption, network — each with `requested` (the operator-facing knob as literally given, sanitized: hosts/kind labels only, never secrets), `resolved` (the runtime's actual resolution), `drifted` (true when they differ in a way the operator did not ask for), and `note` (remediation). Drift is a loud condition: a configured-but-unavailable embedding backend is reported as drifted, never silently reclassified as empty success; a store whose embedding format was declared by a previous process drifts against this process's default. Read-only.",
    "inputSchema": {
      "type": "object",
      "properties": {}
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "generated_at_unix_ms": { "type": "number", "description": "Report timestamp" },
        "stages": {
          "type": "array",
          "description": "One entry per stage: stage/requested/resolved/drifted/note"
        },
        "drifted_stages": {
          "type": "array",
          "description": "Stage ids with drifted=true (empty = everything resolved as requested)"
        }
      }
    },
    "annotations": {
      "readOnlyHint": true
    },
    "title": "Config Self-Report"
  },
  {
    "name": "perseus_vault_type_policies",
    "description": "Typed memory-class policy table (#1000, CogniCore borrow): the 8 MemoryTypes (semantic, episodic, procedural, preference, constraint, failure, reflection, knowledge) with per-type decay_multiplier (scales the per-category half-life at decay tick) and retrieval_weight (multiplies the final fused recall score), plus each policy's rationale. Legacy rows (memory_type '') resolve to the SEMANTIC policy — the byte-compatible baseline. Unknown memory_type values on remember() are hard write errors (fail-closed, never a silent fallback). Read-only.",
    "inputSchema": {
      "type": "object",
      "properties": {}
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "memory_types": {
          "type": "array",
          "description": "One entry per MemoryType: memory_type/decay_multiplier/retrieval_weight/rationale"
        },
        "legacy_rows": { "type": "string", "description": "Legacy-row resolution semantics" },
        "write_validation": { "type": "string", "description": "Write-time validation semantics" }
      }
    },
    "annotations": {
      "readOnlyHint": true
    },
    "title": "Typed Memory Policies"
  },
  {
    "name": "perseus_vault_handoff_restart",
    "description": "Live-update / reconnect for long-lived stdio sessions (#858). When the perseus-vault binary was rebuilt or replaced on disk mid-session, the running process image is stale: every other tool refuses loudly (isError) until the session is restarted — or this tool hot-swaps the process on the SAME stdio connection. States: binary unchanged -> no_handoff_needed (identity report); stale + dry_run -> dry_run (what would happen); stale without confirm -> confirm_required; stale + confirm:true -> the replacement binary is spawned on this session's stdio and the old process exits immediately after this response — the MCP session continues uninterrupted in the new process image. Do not pipeline requests during the handoff.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "dry_run": {
          "type": "boolean",
          "description": "Report what would happen without performing the handoff (default false)"
        },
        "confirm": {
          "type": "boolean",
          "description": "Required to actually perform the hot-swap when the binary is stale (default false)"
        }
      }
    }
  },
  {
    "name": "perseus_vault_quality_telemetry",

    "description": "Machine-readable memory-quality telemetry: contradiction rate, supersession lag, class/layer distribution, and promotion-flow proxy.",
    "inputSchema": {
      "type": "object",
      "properties": {"category": {"type": "string", "description": "Category for contradiction scan (default general)."}}
    }
  },
  {
    "name": "perseus_vault_retrieval_telemetry",

    "description": "Read-only retrieval telemetry: concentration (top slot/token shares, Herfindahl), repeated-serving rate over a turn/second window, diversity (sources, source classes, Simpson), cross-arm contamination (per-arm audits, delivered-set validation, optional arm-level probe), low-trust query-class fan-out, and diversity/cooldown displacement. Reports include denominators, scope, retrieval profile, source class, and the versioned artifact hash; empty/degraded/unavailable states are separated from zero concentration.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "window_turns": {"type": "integer", "description": "Window in serving batches (distinct recalls). Default: none (window_secs wins)."},
        "window_secs": {"type": "integer", "description": "Window in seconds (default 86400)."},
        "profile": {"type": "string", "description": "Scope: only events recorded under this profile."},
        "workspace_hash": {"type": "string", "description": "Scope: only events from this workspace."},
        "probe_query": {"type": "string", "description": "Optional contamination probe: run arm-level SQL deltas for this query and report blocked re-entry per arm."},
        "probe_mode": {"type": "string", "description": "Probe mode: lexical|dense|hybrid|fused|graph|proactive (default lexical)."}
      }
    }
  },
  {
    "name": "perseus_vault_stats",
    "description": "Return comprehensive database statistics: entity counts by category, type, and decay layer; journal event count; state entry count; database file size; date range of stored data; and history growth (stored version rows, bytes, and the top-10 keys by version count — #398).",
    "inputSchema": {
      "type": "object",
      "properties": {}
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "total_entities": {
          "type": "integer",
          "description": "Total entities in the database"
        },
        "by_category": {
          "type": "object",
          "description": "Entity counts grouped by category"
        },
        "by_type": {
          "type": "object",
          "description": "Entity counts grouped by type"
        },
        "by_layer": {
          "type": "object",
          "description": "Entity counts grouped by decay layer (buffer/working/core)"
        },
        "total_journal_events": {
          "type": "integer",
          "description": "Total journal events recorded"
        },
        "total_state_entries": {
          "type": "integer",
          "description": "Total state entries (including expired)"
        },
        "db_file_size_bytes": {
          "type": "integer",
          "description": "Database file size on disk in bytes"
        },
        "oldest_unix_ms": {
          "type": ["integer", "null"],
          "description": "Oldest entity creation timestamp, or null when the database has no entities"
        },
        "newest_unix_ms": {
          "type": ["integer", "null"],
          "description": "Newest entity creation timestamp, or null when the database has no entities"
        },
        "total_history_rows": {
          "type": "integer",
          "description": "Superseded versions stored in entity_history, incl. compaction tombstones (#398)"
        },
        "history_bytes": {
          "type": "integer",
          "description": "Stored history body bytes — SUM(LENGTH(body_json)); row/index overhead excluded (#398)"
        },
        "top_history_keys": {
          "type": "array",
          "description": "Top-10 (category, key) pairs by stored version count: [{category, key, versions, bytes}] (#398)"
        }
      }
    },
    "annotations": {
      "readOnlyHint": true
    },
    "title": "Get Database Statistics"
  },
  {
    "name": "perseus_vault_compact",
    "description": "Archive entities whose decay score has fallen below a threshold. Supports dry-run mode to preview without making changes. Run periodically or threshold-triggered to keep the database focused on active, high-value memories.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "min_decay": {
          "type": "number",
          "default": 0.1,
          "description": "Decay threshold — entities with decay score below this are archived"
        },
        "dry_run": {
          "type": "boolean",
          "default": false,
          "description": "If true, report what would be archived without making changes"
        }
      },
      "required": []
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "entities_archived": {
          "type": "integer",
          "description": "Number of entities actually archived (0 in dry-run mode)"
        },
        "entities_examined": {
          "type": "integer",
          "description": "Number of entities checked"
        },
        "dry_run": {
          "type": "boolean",
          "description": "Whether this was a dry run"
        },
        "completed_at_unix_ms": {
          "type": "integer",
          "description": "Completion timestamp"
        }
      }
    },
    "annotations": {
      "destructiveHint": true
    },
    "title": "Compact Low-Decay Entities"
  },
  {
    "name": "perseus_vault_purge",
    "description": "Permanently delete all archived entities and run VACUUM to reclaim disk space. This is the only operation that actually removes entities — prune/forget only soft-archive. Erasure is complete (#398): every superseded version of a purged entity is deleted from entity_history, and journal rows referencing it are redacted in place (payloads scrubbed; rows kept so the audit hash chain stays verifiable). Purged data is DELETED and NOT RECOVERABLE — this forget-then-purge path is the GDPR-style erasure mechanism. Supports dry_run=true to preview first. Deletion-residue accounting (#990): the report carries a four-way residue partition (purged / declared_residual_controlled / declared_residual_uncontrollable / undeclared_residual), and purge REFUSES to complete while the independent sweep observes undeclared residual state (embedding-snapshot rows, projection-basis rows, or unrevoked artifact bindings whose sources are gone). Use sweep_only=true to run just the sweep and enumerate any orphans.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "dry_run": {
          "type": "boolean",
          "default": false,
          "description": "If true, report what would be deleted (with the residue partition and gate preview) without making changes"
        },
        "sweep_only": {
          "type": "boolean",
          "default": false,
          "description": "If true, run only the independent residue sweep (#990): enumerate undeclared residual state and report the hard-gate status without deleting anything"
        }
      },
      "required": []
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "entities_deleted": {
          "type": "integer",
          "description": "Number of archived entities permanently deleted"
        },
        "history_rows_deleted": {
          "type": "integer",
          "description": "Superseded versions of the purged entities deleted from entity_history (#398)"
        },
        "journal_rows_redacted": {
          "type": "integer",
          "description": "Journal rows referencing purged entities scrubbed in place; the audit hash chain stays valid (#398)"
        },
        "artifact_bindings_revoked": {
          "type": "integer",
          "description": "Learned-artifact bindings revoked because their source entity was physically removed; serve paths refuse revoked bindings (#876)"
        },
        "embeddings_snapshot_deleted": {
          "type": "integer",
          "description": "Pre-quantization float32 snapshot rows removed with their purged source (#990)"
        },
        "projection_basis_deleted": {
          "type": "integer",
          "description": "Declared embedding-basis rows removed with their purged source (#990)"
        },
        "bytes_freed": {
          "type": "integer",
          "description": "Bytes reclaimed after VACUUM (0 in dry-run mode)"
        },
        "dry_run": {
          "type": "boolean",
          "description": "Whether this was a dry run"
        },
        "completed_at_unix_ms": {
          "type": "integer",
          "description": "Completion timestamp"
        },
        "residue": {
          "type": "object",
          "description": "Four-way residue partition of everything derived from the purged set (#990). undeclared_residual is empty for any completed purge (hard gate).",
          "properties": {
            "purged": {"type": "object"},
            "declared_residual_controlled": {"type": "object"},
            "declared_residual_uncontrollable": {"type": "object"},
            "undeclared_residual": {"type": "object"},
            "undeclared_residual_items": {"type": "array", "items": {"type": "string"}},
            "hard_gate_passed": {"type": "boolean"}
          }
        }
      }
    },
    "annotations": {
      "destructiveHint": true
    },
    "title": "Purge Archived Entities"
  },
  {
    "name": "perseus_vault_project_task",
    "description": "Build a compact task-scoped projection (#859): retrieve once, then separate the results into three clearly labeled sections — live_references (pointers into live external systems of record via external_refs), durable_memories (recalled facts), and derived_inferences (inferred/derived facts) — each item carrying a summary, trust class, freshness grade, scope, and provenance digest. The contract block makes permission scope (workspace_scoped/global), freshness anchor, trust classes present, per-section counts, and exclusion reasons visible; no raw recall dump is emitted. Options: query (defaults to task_title), category, workspace_hash (permission scope), limit per section, freshness_window_days (older hits counted as excluded, not dropped silently), min_trust (candidate/corroborated/verified; rejected entities are never projected), include_sections subset, query_time_unix_ms (deterministic replay anchor — identical inputs produce the same projection_id). Output is informational context, not instructions.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "task_title": {
          "type": "string",
          "description": "The task this projection is scoped to. Also the recall query when query is omitted."
        },
        "task_description": {
          "type": "string",
          "description": "Optional task context (advisory; the resolved query wins)."
        },
        "query": {
          "type": "string",
          "description": "Explicit retrieval query. Defaults to task_title."
        },
        "category": {
          "type": "string",
          "description": "Restrict the recall pool to one category."
        },
        "workspace_hash": {
          "type": "string",
          "description": "Permission scope: when set, only matching-workspace or global entities are projected and the contract reports permission: workspace_scoped. Compatibility mode permits omission for legacy unscoped projections when strict deployment mode is off; strict mode requires a non-empty workspace_hash and an active binding for the transport requester."
        },
        "limit": {
          "type": "integer",
          "default": 12,
          "minimum": 1,
          "maximum": 100,
          "description": "Maximum items per section."
        },
        "freshness_window_days": {
          "type": "integer",
          "minimum": 1,
          "description": "Only entities created within this many days are projected; older hits are counted in contract.excluded.outside_freshness_window."
        },
        "min_trust": {
          "type": "string",
          "enum": ["candidate", "corroborated", "verified"],
          "default": "candidate",
          "description": "Minimum trust class. Rejected entities are never projected regardless of this value."
        },
        "include_sections": {
          "type": "array",
          "items": {
            "type": "string",
            "enum": ["live", "durable", "derived"]
          },
          "description": "Section subset; empty = all three."
        },
        "query_time_unix_ms": {
          "type": "integer",
          "description": "Anchor instant for freshness grades; omitted = server now. Deterministic replay anchor (#247)."
        },
        "task_state": {
          "type": "object",
          "additionalProperties": false,
          "description": "#1182: opt-in versioned task-scoped state update. The persisted projection contains bounded task metadata, canonical evidence IDs, source/evidence digests, sequences, and no raw query, prompt, model reasoning, or memory body.",
          "properties": {
            "schema_version": {"type": "string", "const": "perseus-vault-task-state/v1"},
            "task_id": {"type": "string", "maxLength": 128},
            "tenant_id": {"type": "string", "description": "Must equal workspace_hash in the current workspace-as-tenant contract."},
            "workspace_hash": {"type": "string", "description": "Exact task-state workspace scope."},
            "principal_id": {"type": "string", "description": "Overwritten with the initialized MCP session identity."},
            "agent_id": {"type": "string", "description": "Overwritten with the initialized MCP session identity."},
            "query_digest": {"type": "string", "description": "Lowercase SHA-256 of the resolved project_task query."},
            "route": {"type": "string", "maxLength": 64},
            "objective": {"type": "string", "maxLength": 512, "description": "Bounded objective label; not the raw query or prompt."},
            "temporal_anchor_unix_ms": {"type": "integer", "minimum": 0},
            "constraints": {"type": "array", "maxItems": 32, "items": {"type": "string", "maxLength": 256}},
            "base_sequence": {"type": "integer", "minimum": 0},
            "observed_input_digest": {"type": "string", "description": "Lowercase SHA-256 of the observed input."},
            "source_digest": {"type": "string", "description": "Optional expected aggregate source digest; recomputed and checked."},
            "evidence_digest": {"type": "string", "description": "Optional expected aggregate evidence digest; recomputed and checked."},
            "accepted_evidence": {"type": "array", "maxItems": 256, "items": {"$ref": "#/properties/task_state/$defs/taskEvidenceReference"}},
            "rejected_evidence": {"type": "array", "maxItems": 256, "items": {"$ref": "#/properties/task_state/$defs/taskEvidenceReference"}},
            "unresolved_evidence": {"type": "array", "maxItems": 128, "items": {"$ref": "#/properties/task_state/$defs/evidenceSlot"}},
            "active_conflicts": {"type": "array", "maxItems": 128, "items": {"$ref": "#/properties/task_state/$defs/activeConflict"}},
            "missing_evidence": {"type": "array", "maxItems": 128, "items": {"$ref": "#/properties/task_state/$defs/evidenceSlot"}},
            "next_step": {"$ref": "#/properties/task_state/$defs/nextStep"}
          },
          "$defs": {
            "taskEvidenceReference": {
              "type": "object",
              "additionalProperties": false,
              "properties": {
                "entity_id": {"type": "string"},
                "source_id": {"type": "string"},
                "revision": {"type": "string"},
                "source_digest": {"type": "string", "minLength": 64, "maxLength": 64},
                "evidence_digest": {"type": "string", "minLength": 64, "maxLength": 64}
              },
              "required": ["entity_id", "source_id", "revision", "source_digest", "evidence_digest"]
            },
            "evidenceSlot": {
              "type": "object",
              "additionalProperties": false,
              "properties": {"slot_id": {"type": "string"}, "reason": {"type": "string", "maxLength": 256}},
              "required": ["slot_id", "reason"]
            },
            "activeConflict": {
              "type": "object",
              "additionalProperties": false,
              "properties": {"conflict_id": {"type": "string"}, "evidence_ids": {"type": "array", "items": {"type": "string"}}, "reason": {"type": "string", "maxLength": 256}},
              "required": ["conflict_id", "evidence_ids", "reason"]
            },
            "nextStep": {
              "type": "object",
              "additionalProperties": false,
              "properties": {"kind": {"type": "string"}, "reason": {"type": "string", "maxLength": 256}},
              "required": ["kind", "reason"]
            }
          },
          "required": ["schema_version", "task_id", "tenant_id", "workspace_hash", "principal_id", "agent_id", "query_digest", "route", "objective", "base_sequence", "observed_input_digest"]
        }
      },
      "required": ["task_title"]
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "task_state": {"type": "object", "description": "Validated persisted task-state projection: scope, state/base sequence, canonical references, digests, explicit outcome, and state_digest."},
        "serving": {"type": "object", "description": "One coherent answer-serving projection separating canonical_sources, recalled_evidence, rejected_evidence, derived_task_state, and explicit fallback."},
        "outcome": {
          "type": "object",
          "additionalProperties": false,
          "description": "#1186: bounded answer-facing status; no query, evidence body, or backend error text.",
          "properties": {
            "schema_version": {"type": "string", "const": "perseus-vault-answer-outcome/v1"},
            "status": {"type": "string", "enum": ["complete", "partial", "degraded", "abstained", "unavailable"]},
            "recall_status": {"type": "string", "enum": ["fresh", "partial", "timeout", "unavailable", "empty", "stale"]},
            "reason": {"type": "string", "minLength": 1, "maxLength": 256},
            "reason_codes": {"type": "array", "minItems": 1, "maxItems": 16, "items": {"type": "string", "maxLength": 256}},
            "abstained": {"type": "boolean"},
            "answerable": {"type": "boolean"},
            "fallback": {"type": "object", "additionalProperties": false, "properties": {"mode": {"type": "string", "enum": ["abstain", "canonical_retrieval"]}, "reason": {"type": "string", "minLength": 1, "maxLength": 256}}, "required": ["mode", "reason"]},
            "exclusions": {"type": "array", "maxItems": 256, "items": {"type": "object", "additionalProperties": false, "properties": {"reason": {"type": "string", "maxLength": 256}, "count": {"type": "integer", "minimum": 1}}, "required": ["reason", "count"]}},
            "conflicts": {"type": "array", "maxItems": 128, "items": {"type": "object", "additionalProperties": false, "properties": {"reason": {"type": "string", "maxLength": 256}, "reference_count": {"type": "integer", "minimum": 0}, "references_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"}}, "required": ["reason", "reference_count", "references_sha256"]}}
          },
          "required": ["schema_version", "status", "recall_status", "reason", "reason_codes", "abstained", "answerable"]
        }
      },
      "additionalProperties": true
    },
    "title": "Build Task Projection"
  },
  {
    "name": "perseus_vault_experience_projection",
    "description": "Read a versioned, non-authoritative experience projection (#1173). The response contains only bounded derived signals, explicit tenant/workspace/principal/agent scope, canonical source IDs, and resolved canonical source metadata. Every source is re-read through ordinary visibility, validity, supersession, and admission rules. Missing, stale, quarantined, or mismatched projections return read_mode=canonical_fallback and require ordinary canonical retrieval; projection signals are never evidence or authority.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "schema_version": {
          "type": "integer",
          "const": 1,
          "description": "Experience projection schema version."
        },
        "experience_id": {
          "type": "string",
          "description": "Explicit experience grouping ID."
        },
        "workspace_hash": {
          "type": "string",
          "description": "Exact workspace scope. The workspace is also the current tenant partition."
        },
        "requesting_agent_id": {
          "type": "string",
          "description": "Transport-stamped principal from MCP initialize.clientInfo.name; caller values are overwritten."
        }
      },
      "required": ["schema_version", "experience_id", "workspace_hash"]
    },
    "title": "Read Experience Projection"
  },
  {
    "name": "perseus_vault_experience_projection_rebuild",
    "description": "Rebuild one non-authoritative experience projection (#1173) from canonical entity IDs and accepted Vault telemetry references. The write is transactional across the projection, normalized source links, and idempotent rebuild ledger. Metrics are derived from canonical state and telemetry; caller-supplied confidence, verified, body, prompt, credentials, and authority fields are rejected. Source events must belong to one serving batch and pulses to one preload session, with exact workspace and principal checks.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "schema_version": {
          "type": "integer",
          "const": 1,
          "description": "Experience projection schema version."
        },
        "experience_id": {
          "type": "string",
          "description": "Explicit experience grouping ID; accepted only with in-scope Vault telemetry references."
        },
        "workspace_hash": {
          "type": "string",
          "description": "Exact workspace scope. The workspace is also the current tenant partition."
        },
        "graph_side": {
          "type": "string",
          "enum": ["source", "target", "context", "none"],
          "description": "Bounded graph side label; ranking metadata only."
        },
        "layer": {
          "type": "string",
          "description": "Canonical source layer, or mixed when sources span layers."
        },
        "source_entity_ids": {
          "type": "array",
          "minItems": 1,
          "maxItems": 64,
          "items": {"type": "string"},
          "description": "Canonical entity IDs resolved through the governed reader."
        },
        "source_event_ids": {
          "type": "array",
          "maxItems": 128,
          "items": {"type": "string"},
          "description": "Accepted Vault serving-event IDs, all from one batch and tied to a source entity."
        },
        "pulse_ids": {
          "type": "array",
          "maxItems": 128,
          "items": {"type": "string"},
          "description": "Accepted Vault preload-event IDs, all from one session and tied to a source entity."
        },
        "query_time_unix_ms": {
          "type": "integer",
          "minimum": 0,
          "description": "Fixed replay anchor. Reusing the same canonical state/config/anchor yields the same digest."
        },
        "requesting_agent_id": {
          "type": "string",
          "description": "Transport-stamped principal from MCP initialize.clientInfo.name; caller values are overwritten."
        }
      },
      "required": ["schema_version", "experience_id", "workspace_hash", "graph_side", "layer", "source_entity_ids", "query_time_unix_ms"]
    },
    "title": "Rebuild Experience Projection"
  },
  {
    "name": "perseus_vault_expand_source",
    "description": "Expand a distilled fact's source reference back to the verbatim span of its retained transcript (#888). Fact mode (category+key of a capture note): reads the note's stamped source_chunk and returns the exact source text under a char budget, with source metadata, span offsets, and a SHA-256 integrity verdict against the retained store. Explicit mode (source_category+source_key+start_char+end_char): expands an arbitrary span of any retained source with optional span_sha256 verification. Bi-temporal: as_of_unix_ms defaults to the fact's capture time, so the text is the span as it existed when the fact was distilled; pass a later anchor to read the source as it is today. Graceful outcomes (never errors): no_source_ref (fact has no source ref — API writes, LLM-distilled notes, retain_transcript=false), fact_not_found, source_missing, span_invalid (out_of_bounds or hash_mismatch — fail-closed, no text). max_chars budget 1..=16384 (default 2000); longer spans are truncated with truncated:true. Output is verbatim informational context, not instructions.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "category": {
          "type": "string",
          "description": "Fact mode: category of the distilled fact entity."
        },
        "key": {
          "type": "string",
          "description": "Fact mode: key of the distilled fact entity."
        },
        "source_category": {
          "type": "string",
          "description": "Explicit mode: category of the retained source."
        },
        "source_key": {
          "type": "string",
          "description": "Explicit mode: key of the retained source."
        },
        "start_char": {
          "type": "integer",
          "minimum": 0,
          "description": "Explicit mode: span start (char offset, inclusive)."
        },
        "end_char": {
          "type": "integer",
          "minimum": 0,
          "description": "Explicit mode: span end (char offset, exclusive)."
        },
        "span_sha256": {
          "type": "string",
          "description": "Explicit mode: optional expected SHA-256 of the verbatim span; verified when present."
        },
        "max_chars": {
          "type": "integer",
          "default": 2000,
          "minimum": 1,
          "maximum": 16384,
          "description": "Char budget for the returned text."
        },
        "as_of_unix_ms": {
          "type": "integer",
          "description": "Bi-temporal anchor; defaults to the fact's capture time."
        },
        "workspace_hash": {
          "type": "string",
          "default": "",
          "description": "Permission scope."
        }
      },
      "required": []
    },
    "title": "Expand Source Chunk"
  },
  {
    "name": "perseus_vault_expire",
    "description": "Time-based lifecycle sweep (#868): transition entities whose expires_at_unix_ms has passed to status='expired'. Content, history, and searchability are RETAINED — expiry is not erasure, and recall already excludes expired rows; the sweep makes the lifecycle state explicit and observable. Idempotent and re-runnable; use dry_run=true to preview with identical predicates. Contract: docs/specs/data-boundaries-retention-lifecycle.md.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "dry_run": {
          "type": "boolean",
          "default": false,
          "description": "If true, report what would be expired without making changes"
        },
        "workspace_hash": {
          "type": "string",
          "default": "",
          "description": "Restrict the sweep to one workspace (empty = global sweep)"
        }
      },
      "required": []
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "entities_expired": {
          "type": "integer",
          "description": "Entities transitioned to status='expired'"
        },
        "workspace_hash": {
          "type": "string",
          "description": "Workspace the sweep was restricted to ('' = global)"
        },
        "dry_run": {
          "type": "boolean",
          "description": "Whether this was a dry run"
        },
        "completed_at_unix_ms": {
          "type": "integer",
          "description": "Completion timestamp"
        }
      }
    },
    "annotations": {
      "destructiveHint": true
    },
    "title": "Expire Due Entities"
  },
  {
    "name": "perseus_vault_redact",
    "description": "Content redaction (#868): scrub the body of a workspace-scoped entity to a hash-only marker, delete its history snapshots and FTS text, and append a hash-only 'redacted' journal event. Metadata (id, key, links, provenance) is RETAINED; re-ingest of the same value stays allowed (redaction ≠ erasure). Requires an explicit workspace_hash (fail-closed, #854). Contract: docs/specs/data-boundaries-retention-lifecycle.md.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "category": {
          "type": "string",
          "description": "Entity category"
        },
        "key": {
          "type": "string",
          "description": "Entity key"
        },
        "workspace_hash": {
          "type": "string",
          "description": "Workspace scope of the entity (required — a bare category/key is ambiguous)"
        },
        "agent_id": {
          "type": "string",
          "default": "",
          "description": "Acting agent for attribution (overridden by the transport-stamped requesting_agent_id when present)"
        },
        "requesting_agent_id": {
          "type": "string",
          "description": "MCP session identity stamped by the transport; overrides agent_id"
        }
      },
      "required": [
        "category",
        "key",
        "workspace_hash"
      ]
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "found": {
          "type": "boolean",
          "description": "Whether a matching entity was found and redacted"
        },
        "entity_id": {
          "type": "string",
          "description": "Id of the first redacted row"
        },
        "value_sha256": {
          "type": "string",
          "description": "Hash-only audit evidence: sha256 of the scrubbed body"
        },
        "history_deleted": {
          "type": "integer",
          "description": "History snapshot rows deleted (content-bearing)"
        },
        "fts_cleaned": {
          "type": "integer",
          "description": "FTS index rows removed"
        },
        "journal_event_id": {
          "type": "string",
          "description": "Id of the hash-only 'redacted' journal event"
        },
        "workspace_hash": {
          "type": "string",
          "description": "Workspace the redaction was scoped to"
        },
        "completed_at_unix_ms": {
          "type": "integer",
          "description": "Completion timestamp"
        }
      }
    },
    "annotations": {
      "destructiveHint": true
    },
    "title": "Redact Entity Content"
  },
  {
    "name": "perseus_vault_erase",
    "description": "Physical erasure (#868/#866): permanently remove a workspace-scoped entity from the primary store AND all derived layers (FTS, history, history-FTS, community membership, inbound links, journal payloads), quarantine derived entities that cited it via evidence links, install a permanent rejection tombstone + governance mandate (re-ingest fails closed and survives primary-DB rollback), and append a hash-only 'erased' journal event. ERASED DATA IS NOT RECOVERABLE. Requires an explicit workspace_hash (fail-closed, #854). Use dry_run=true to preview exact counts. Contract: docs/specs/data-boundaries-retention-lifecycle.md.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "category": {
          "type": "string",
          "description": "Entity category"
        },
        "key": {
          "type": "string",
          "description": "Entity key"
        },
        "workspace_hash": {
          "type": "string",
          "description": "Workspace scope of the entity (required — a bare category/key is ambiguous)"
        },
        "agent_id": {
          "type": "string",
          "default": "",
          "description": "Acting agent for attribution (overridden by the transport-stamped requesting_agent_id when present)"
        },
        "dry_run": {
          "type": "boolean",
          "default": false,
          "description": "If true, report exactly what would be erased without making changes"
        },
        "requesting_agent_id": {
          "type": "string",
          "description": "MCP session identity stamped by the transport; overrides agent_id"
        }
      },
      "required": [
        "category",
        "key",
        "workspace_hash"
      ]
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "entities_erased": {
          "type": "integer",
          "description": "Primary rows removed"
        },
        "history_deleted": {
          "type": "integer",
          "description": "History snapshot rows removed"
        },
        "fts_cleaned": {
          "type": "integer",
          "description": "FTS index rows removed"
        },
        "community_memberships_cleaned": {
          "type": "integer",
          "description": "Community member_ids entries removed"
        },
        "community_rows_deleted": {
          "type": "integer",
          "description": "Communities deleted because the erased entity was their last member"
        },
        "inbound_links_cleaned": {
          "type": "integer",
          "description": "Inbound link edges removed from other rows"
        },
        "derived_quarantined": {
          "type": "integer",
          "description": "Derived entities citing the erased source, now quarantined pending operator review"
        },
        "journal_rows_redacted": {
          "type": "integer",
          "description": "Journal payloads scrubbed in place (audit chain preserved)"
        },
        "journal_event_id": {
          "type": "string",
          "description": "Id of the hash-only 'erased' journal event"
        },
        "value_sha256": {
          "type": "string",
          "description": "Hash-only evidence: sha256 of the erased body"
        },
        "workspace_hash": {
          "type": "string",
          "description": "Workspace the erasure was scoped to"
        },
        "dry_run": {
          "type": "boolean",
          "description": "Whether this was a dry run"
        },
        "governance_mandate_ok": {
          "type": "boolean",
          "description": "False if the permanent re-ingest mandate could not be installed (content is gone; guard needs operator attention)"
        },
        "completed_at_unix_ms": {
          "type": "integer",
          "description": "Completion timestamp"
        }
      }
    },
    "annotations": {
      "destructiveHint": true
    },
    "title": "Erase Entity Permanently"
  },
  {
    "name": "perseus_vault_memories",

    "description": "Anthropic memory-tool compatible file interface over the vault: view / create / str_replace / insert / delete / rename on paths under /memories. Files are stored as vault entities (category 'memories', FTS-indexed, encrypted at rest, edits versioned via history), so clients built against Claude's native memory directory convention can use the vault unchanged. Use command='view' with path='/memories' to list files.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "command": {
          "type": "string",
          "enum": ["view", "create", "str_replace", "insert", "delete", "rename"],
          "description": "The operation to perform"
        },
        "path": {
          "type": "string",
          "description": "Path under /memories (e.g. '/memories/notes.md'). For view, '/memories' lists the directory."
        },
        "file_text": {
          "type": "string",
          "description": "create: full file content to write (overwrites an existing file)"
        },
        "old_str": {
          "type": "string",
          "description": "str_replace: exact text to replace — must occur exactly once in the file"
        },
        "new_str": {
          "type": "string",
          "description": "str_replace: replacement text"
        },
        "insert_line": {
          "type": "integer",
          "description": "insert: line number to insert AT (0 = beginning of file)"
        },
        "insert_text": {
          "type": "string",
          "description": "insert: the line to insert"
        },
        "old_path": {
          "type": "string",
          "description": "rename: current path"
        },
        "new_path": {
          "type": "string",
          "description": "rename: destination path (must not exist)"
        }
      },
      "required": [
        "command"
      ]
    },
    "title": "Memories Directory (Anthropic convention)"
  },
  {
    "name": "perseus_vault_migrate",
    "description": "Migrate a v0.1.x Perseus Vault database to the current v0.5.0 schema. Reads the old database, converts memories to the entity model, and merges into the current database. Use this once per legacy database during upgrade.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "from_path": {
          "type": "string",
          "description": "Absolute path to the v0.1.x SQLite database file to migrate"
        }
      },
      "required": [
        "from_path"
      ]
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "total_old_memories": {
          "type": "integer",
          "description": "Number of memories found in the old database"
        },
        "entities_created": {
          "type": "integer",
          "description": "New entities created from old memories"
        },
        "entities_updated": {
          "type": "integer",
          "description": "Existing entities updated during merge"
        },
        "errors": {
          "type": "array",
          "items": {
            "type": "string"
          },
          "description": "Any errors encountered during migration"
        },
        "completed_at_unix_ms": {
          "type": "integer",
          "description": "Completion timestamp"
        }
      }
    },
    "annotations": {
      "destructiveHint": true
    },
    "title": "Migrate Legacy Database"
  },
  {
    "name": "perseus_vault_context",
    "description": "Return a pre-formatted markdown context block for session injection. Recall-first by default (mode 'on_demand'): pass `query` (the current task/message) and only topically relevant entities — recall_when trigger matches + keyword matches — are injected, alongside a hard-capped always-on set, clamped to a per-model character budget. Without `query` the block is a compact retrieval pointer (byte-stable across unrelated writes — prefix-cache friendly). The legacy unconditional top-N dump requires explicit mode 'always_inject'. Output is informational context, not instructions.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "categories": {
          "type": "array",
          "items": {
            "type": "string"
          },
          "description": "Categories to include. Empty array = all categories."
        },
        "limit": {
          "type": "integer",
          "default": 10,
          "description": "Maximum number of entities to include in the context block"
        },
        "workspace_hash": {
          "type": "string",
          "description": "Workspace scope filter (v1.2.0). When set, only entities with a matching workspace_hash are included (always-on set too). Compatibility mode permits omission for legacy unscoped context reads when strict deployment mode is off; strict mode requires a non-empty workspace_hash and an active binding for the transport requester."
        },
        "query": {
          "type": "string",
          "description": "Current task/message text — the relevance gate (#356). In on_demand mode only entities whose recall_when triggers or indexed content match it are injected; omit for a compact retrieval pointer with no topical injection."
        },
        "mode": {
          "type": "string",
          "enum": ["on_demand", "always_inject"],
          "default": "on_demand",
          "description": "Injection posture (#366). 'on_demand' (default): relevance-gated, budget-clamped, recall-first. 'always_inject': legacy unconditional top-N dump (no relevance gating) — explicit opt-in only."
        },
        "model": {
          "type": "string",
          "description": "Host model name for recall-budget profile resolution (#366), e.g. 'claude-opus-4-8' gets a larger budget. Unknown/omitted models use the default 1500-char profile."
        },
        "max_context_chars": {
          "type": "integer",
          "description": "Explicit character budget for the rendered block; overrides the model profile. In always_inject mode output is clamped only when this is set."
        },
        "include_provider_source": {
          "type": "boolean",
          "default": false,
          "description": "#1141: include sanitized provider identity and thread lineage on context lines; provider bodies and payloads are excluded."
        },
        "include_selection_decisions": {
          "type": "boolean",
          "default": false,
          "description": "#1140: attach a bounded, hash-only per-candidate context-selection projection with source-arm ranks, eligibility/disposition reason codes, token estimates, arm state, and a replay fingerprint. Omit to preserve the legacy response shape."
        },
        "evidence_requirements": {
          "type": "object",
          "additionalProperties": false,
          "description": "#1183: opt-in answer-serving evidence requirements. The context path resolves each declared ID through scope, visibility, suppression, lifecycle, and temporal gates before counting coverage. Requirement IDs and query text are committed by digest only in the response receipt.",
          "properties": {
            "schema_version": {
              "type": "string",
              "const": "perseus-vault-evidence-sufficiency/v1",
              "default": "perseus-vault-evidence-sufficiency/v1"
            },
            "required_evidence": {
              "type": "array",
              "minItems": 1,
              "maxItems": 256,
              "items": { "type": "string", "minLength": 1, "maxLength": 128 }
            },
            "latest_evidence": {
              "type": "array",
              "maxItems": 256,
              "items": { "type": "string", "minLength": 1, "maxLength": 128 }
            },
            "temporal_anchors": {
              "type": "array",
              "maxItems": 256,
              "items": { "type": "string", "minLength": 1, "maxLength": 128 }
            },
            "required_source_groups": {
              "type": "array",
              "maxItems": 128,
              "items": {
                "type": "object",
                "additionalProperties": false,
                "properties": {
                  "group_id": { "type": "string", "minLength": 1, "maxLength": 128 },
                  "evidence_ids": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 256,
                    "items": { "type": "string", "minLength": 1, "maxLength": 128 }
                  }
                },
                "required": ["group_id", "evidence_ids"]
              }
            },
            "conflicts": {
              "type": "array",
              "maxItems": 128,
              "items": {
                "type": "object",
                "additionalProperties": false,
                "properties": {
                  "conflict_id": { "type": "string", "minLength": 1, "maxLength": 128 },
                  "evidence_ids": {
                    "type": "array",
                    "minItems": 2,
                    "maxItems": 256,
                    "items": { "type": "string", "minLength": 1, "maxLength": 128 }
                  }
                },
                "required": ["conflict_id", "evidence_ids"]
              }
            },
            "temporal_anchor_unix_ms": { "type": "integer", "minimum": 0 },
            "fallback_policy": {
              "type": "string",
              "enum": ["abstain", "canonical_retrieval"],
              "default": "abstain"
            }
          },
          "required": ["required_evidence"]
        },
        "include_declared_graph": {
          "type": "boolean",
          "default": false,
          "description": "#1142: attach a bounded workspace-scoped hash-only declared graph projection to the context response. Requires workspace_hash and a transport-stamped requester."
        },
        "session_id": {
          "type": "string",
          "description": "Session id for preload usage telemetry (#875): injected entities are attributed to this session for precision/recall resolution. Omit or leave empty when unknown."
        },
        "include_outcome": {
          "type": "boolean",
          "default": false,
          "description": "#1186: include a bounded answer outcome for complete results as well as empty, partial, degraded, abstained, or unavailable context."
        }
      },
      "required": []
    },
    "outputSchema": {
      "type": "object",
      "$defs": {
        "sufficiency_coverage": {
          "type": "object",
          "additionalProperties": false,
          "properties": {
            "required": { "type": "integer", "minimum": 0 },
            "selected": { "type": "integer", "minimum": 0 },
            "missing": { "type": "integer", "minimum": 0 }
          },
          "required": ["required", "selected", "missing"]
        }
      },
      "properties": {
        "markdown": {
          "type": "string",
          "description": "Markdown-formatted context block with entity details"
        },
        "total_chars": {
          "type": "integer",
          "description": "Character count of the markdown content"
        },
        "mode": {
          "type": "string",
          "description": "Resolved injection mode: on_demand or always_inject"
        },
        "budget_chars": {
          "type": "integer",
          "description": "Resolved character budget (0 = unclamped legacy output)"
        },
        "entities_injected": {
          "type": "integer",
          "description": "Number of entities actually injected (always-on + topical)"
        },
        "warnings": {
          "type": "array",
          "items": {
            "type": "string"
          },
          "description": "Soft warnings: always-on cap overflow, budget truncation"
        },
        "selection_decisions": {
          "type": "object",
          "description": "#1140: optional bounded, hash-only context-selection projection. Contains policy/schema digests, candidate/retention counts, arm states, token estimates, dispositions, delivered order, and replay fingerprint; it never contains query text or memory bodies."
        },
        "sufficiency": {
          "type": "object",
          "additionalProperties": false,
          "description": "#1183: answer-serving evidence sufficiency report. Counts and coverage dimensions are evaluated after governed scope, visibility, suppression, lifecycle, correction, redaction, and temporal checks. The receipt contains hashes and reason counts only.",
          "properties": {
            "schema_version": { "type": "string" },
            "outcome": {
              "type": "string",
              "enum": ["complete", "partial", "degraded", "abstained", "unavailable"]
            },
            "counts": {
              "type": "object",
              "additionalProperties": false,
              "properties": {
                "required": { "type": "integer", "minimum": 0 },
                "selected": { "type": "integer", "minimum": 0 },
                "omitted": { "type": "integer", "minimum": 0 },
                "stale": { "type": "integer", "minimum": 0 },
                "conflicting": { "type": "integer", "minimum": 0 },
                "unavailable": { "type": "integer", "minimum": 0 },
                "red_herring": { "type": "integer", "minimum": 0 }
              },
              "required": ["required", "selected", "omitted", "stale", "conflicting", "unavailable", "red_herring"]
            },
            "latest": { "$ref": "#/$defs/sufficiency_coverage" },
            "temporal": { "$ref": "#/$defs/sufficiency_coverage" },
            "source_groups": { "$ref": "#/$defs/sufficiency_coverage" },
            "recall_status": { "type": "string" },
            "fallback_policy": {
              "type": "string",
              "enum": ["abstain", "canonical_retrieval"]
            },
            "reason_codes": {
              "type": "array",
              "items": { "type": "string" }
            },
            "fallback": {
              "type": "object",
              "additionalProperties": false,
              "properties": {
                "mode": { "type": "string" },
                "reason": { "type": "string" }
              },
              "required": ["mode", "reason"]
            },
            "receipt": {
              "type": "object",
              "additionalProperties": false,
              "properties": {
                "schema_version": { "type": "string" },
                "query_sha256": { "type": "string", "pattern": "^[0-9a-f]{64}$" },
                "requirement_sha256": { "type": "string", "pattern": "^[0-9a-f]{64}$" },
                "candidate_set_sha256": { "type": "string", "pattern": "^[0-9a-f]{64}$" },
                "selected_set_sha256": { "type": "string", "pattern": "^[0-9a-f]{64}$" },
                "omitted_set_sha256": { "type": "string", "pattern": "^[0-9a-f]{64}$" },
                "reasons": {
                  "type": "array",
                  "items": {
                    "type": "object",
                    "additionalProperties": false,
                    "properties": {
                      "reason": { "type": "string" },
                      "count": { "type": "integer", "minimum": 0 }
                    },
                    "required": ["reason", "count"]
                  }
                },
                "digest": { "type": "string", "pattern": "^[0-9a-f]{64}$" }
              },
              "required": ["schema_version", "query_sha256", "requirement_sha256", "reasons", "digest"]
            }
          },
          "required": ["schema_version", "outcome", "counts", "latest", "temporal", "source_groups", "recall_status", "fallback_policy", "reason_codes", "receipt"]
        },
        "declared_graph": {
          "type": "object",
          "description": "#1142: optional bounded declared graph projection with hash-only source, span, scope, origin, validity, and support state."
        },
        "truncated": {
          "type": "boolean",
          "description": "#1186: true when the rendered context was shortened to its character budget."
        },
        "outcome": {
          "type": "object",
          "additionalProperties": false,
          "description": "#1186: bounded answer-facing status; no query, evidence body, or backend error text.",
          "properties": {
            "schema_version": {"type": "string", "const": "perseus-vault-answer-outcome/v1"},
            "status": {"type": "string", "enum": ["complete", "partial", "degraded", "abstained", "unavailable"]},
            "recall_status": {"type": "string", "enum": ["fresh", "partial", "timeout", "unavailable", "empty", "stale"]},
            "reason": {"type": "string", "minLength": 1, "maxLength": 256},
            "reason_codes": {"type": "array", "minItems": 1, "maxItems": 16, "items": {"type": "string", "maxLength": 256}},
            "abstained": {"type": "boolean"},
            "answerable": {"type": "boolean"},
            "fallback": {"type": "object", "additionalProperties": false, "properties": {"mode": {"type": "string", "enum": ["abstain", "canonical_retrieval"]}, "reason": {"type": "string", "minLength": 1, "maxLength": 256}}, "required": ["mode", "reason"]},
            "exclusions": {"type": "array", "maxItems": 256, "items": {"type": "object", "additionalProperties": false, "properties": {"reason": {"type": "string", "maxLength": 256}, "count": {"type": "integer", "minimum": 1}}, "required": ["reason", "count"]}},
            "conflicts": {"type": "array", "maxItems": 128, "items": {"type": "object", "additionalProperties": false, "properties": {"reason": {"type": "string", "maxLength": 256}, "reference_count": {"type": "integer", "minimum": 0}, "references_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"}}, "required": ["reason", "reference_count", "references_sha256"]}}
          },
          "required": ["schema_version", "status", "recall_status", "reason", "reason_codes", "abstained", "answerable"]
        }
      }
    },
    "annotations": {
      "readOnlyHint": true
    },
    "title": "Get Context Block"
  },
  {
    "name": "perseus_vault_extract",
    "description": "Extract structured knowledge — facts, preferences, temporal events, episodes — from raw text or a stored entity, using a fully local, deterministic rule-based extractor (no cloud LLM, no embedding/API call, no network). Read-only: never writes to the store. Provide `text`, or `category` + `key` to extract from a stored entity.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "text": {
          "type": "string",
          "description": "Raw text to extract from. If omitted, category + key of a stored entity are used."
        },
        "category": {
          "type": "string",
          "description": "Category of a stored entity to extract from (requires key)."
        },
        "key": {
          "type": "string",
          "description": "Key of a stored entity to extract from (requires category)."
        },
        "strategy": {
          "type": "string",
          "default": "rule_based",
          "enum": [
            "rule_based",
            "none"
          ],
          "description": "Extractor strategy: 'rule_based' (local heuristics) or 'none' (no-op)."
        }
      },
      "required": []
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "items": {
          "type": "array",
          "items": {
            "type": "object"
          },
          "description": "Extracted items, each an object with `kind` and `text`."
        },
        "total": {
          "type": "integer",
          "description": "Number of items extracted"
        },
        "strategy": {
          "type": "string",
          "description": "Extractor strategy used"
        }
      }
    },
    "annotations": {
      "readOnlyHint": true
    },
    "title": "Extract Structured Knowledge"
  },
  {
    "name": "perseus_vault_capture",
    "description": "Opt-in in-session memory capture (#520): distill a session transcript or insight payload into durable memory entities the moment a problem is solved, instead of waiting for a scheduled harvest. Splits the payload into candidate notes (headed sections, paragraphs, or JSONL records — auto-detected), classifies each by cheap local signals into root-cause / pitfall / decision / pattern / takeaway, and writes each through the normal remember path with source='capture' (layer buffer, moderate importance). Fully local and deterministic by default — no LLM, no network; pass llm=true to distill via the configured --llm-endpoint instead (falls back to the rule-based path on any LLM failure or timeout). Anti-flood by design: near-duplicate merging stays ON (a re-captured solved problem merges into the existing memory), same-headline notes update in place, and writes are capped per invocation with dropped notes reported. Nothing runs automatically — capture happens only when this tool (or the `perseus-vault capture` CLI verb) is explicitly invoked, e.g. from an on_insight or SessionEnd lifecycle hook (run `maintain` after end-of-session capture).",
    "inputSchema": {
      "type": "object",
      "properties": {
        "text": {
          "type": "string",
          "description": "The transcript / insight payload to distill. Plain text, markdown (headed sections become separate notes), or JSONL (one note per record, using its content/text/insight/lesson/summary/message field)."
        },
        "workspace_hash": {
          "type": "string",
          "description": "Workspace hash to scope the captured entities to. Omit for unscoped (global) capture."
        },
        "agent_id": {
          "type": "string",
          "description": "Agent ID recorded on the captured entities."
        },
        "max_entities": {
          "type": "integer",
          "default": 20,
          "description": "Anti-flood cap: max entities written by this invocation (1-20; callers can lower the cap, not raise it). Notes beyond the cap are dropped and counted in the result."
        },
        "dry_run": {
          "type": "boolean",
          "default": false,
          "description": "Distill and return the would-be notes without writing anything."
        },
        "llm": {
          "type": "boolean",
          "default": false,
          "description": "Distill via the configured LLM endpoint instead of the local rule-based distiller. Requires --llm-endpoint; falls back to the rule-based path on any LLM failure (the result's llm_fallback field says why)."
        },
        "consume": {
          "type": "boolean",
          "default": false,
          "description": "#563: after a SUCCESSFUL non-dry-run capture, atomically remove exactly the captured regions from source_file (temp file + rename, leaving a <source_file>.bak). Scoped to captured records only — surrounding headers/rules/pointers are left untouched. No-op under dry_run, when nothing was captured, or when source_file is unset, so it can never delete content that was not durably stored. Use it to keep a host-inlined write-buffer (e.g. an AGENTS.local.md the agent loads every turn) from accumulating already-stored blocks forever. The result reports 'consumed' (regions removed) and 'source_backup'."
        },
        "source_file": {
          "type": "string",
          "description": "#563: path to the file the payload came from. Required for consume to have anything to prune; ignored when consume is false."
        },
        "evidence": {
          "type": "object",
          "description": "Write-time evidence envelope for captured notes. Omit only for legacy_unknown compatibility.",
          "properties": {
            "capture_mode": { "type": "string", "enum": ["snapshot", "hash_only", "pointer_only", "not_requested", "capture_failed", "legacy_unknown"] },
            "resolved_value": { "description": "Resolved source value retained at capture time" },
            "content_sha256": { "type": "string" },
            "source_system": { "type": "string" },
            "source_ref": { "type": "string" },
            "captured_at_unix_ms": { "type": "integer" },
            "replayable": { "type": "boolean" }
          },
          "required": ["capture_mode", "captured_at_unix_ms", "replayable"]
        }
      },
      "required": [
        "text"
      ]
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "captured": {
          "type": "integer",
          "description": "Number of notes distilled (and written, unless dry_run)"
        },
        "created": {
          "type": "integer",
          "description": "Notes that created a new entity"
        },
        "updated": {
          "type": "integer",
          "description": "Notes that updated an existing entity in place (same category+key)"
        },
        "merged": {
          "type": "integer",
          "description": "Notes merged into an existing near-duplicate entity by the trigram dedup (the capture flood control)"
        },
        "candidates": {
          "type": "integer",
          "description": "Candidate notes found in the payload before capping"
        },
        "dropped": {
          "type": "integer",
          "description": "Candidate notes dropped by the per-invocation cap"
        },
        "dry_run": {
          "type": "boolean",
          "description": "True when nothing was written"
        },
        "distiller": {
          "type": "string",
          "description": "'rule_based' or 'llm' — which distiller produced the notes"
        },
        "llm_fallback": {
          "type": "string",
          "description": "Present when llm=true was requested but the rule-based path was used; says why"
        },
        "notes": {
          "type": "array",
          "items": {
            "type": "object"
          },
          "description": "Per-note report: {id, key, type, summary, action}"
        },
        "message": {
          "type": "string",
          "description": "Unambiguous empty state when the payload contained nothing durable"
        },
        "consumed": {
          "type": "integer",
          "description": "#563: number of captured regions removed from source_file (0 unless consume=true and the prune ran). See source_backup / consume_skipped / consume_error."
        },
        "source_backup": {
          "type": "string",
          "description": "#563: path to the pre-prune backup (<source_file>.bak) written when consumed > 0"
        }
      }
    },
    "title": "Capture Session Insights"
  },
  {
    "name": "perseus_vault_traverse",
    "description": "Walk the entity link graph starting from a given entity up to a configurable depth. Returns a chain of linked entities — useful for exploring dependencies, decision trees, and relationship graphs built via perseus_vault_link.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "category": {
          "type": "string",
          "description": "Starting entity category"
        },
        "key": {
          "type": "string",
          "description": "Starting entity key"
        },
        "max_depth": {
          "type": "integer",
          "default": 3,
          "description": "Maximum traversal depth from the starting entity"
        },
        "max_nodes": {
          "type": "integer",
          "default": 100,
          "description": "Maximum total nodes to traverse before stopping"
        },
        "requesting_agent_id": {
          "type": "string",
          "description": "Transport-stamped requester identity; required at runtime for body-safe traversal."
        },
        "workspace_hash": {
          "type": "string",
          "description": "Workspace scope required when include_declared_graph is true."
        },
        "include_declared_graph": {
          "type": "boolean",
          "default": false,
          "description": "#1142: attach a bounded hash-only declared graph projection; ordinary entity traversal never queries it."
        }
      },
      "required": [
        "category",
        "key"
      ]
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "entity": {
          "type": "object",
          "description": "Root entity with its links"
        },
        "traversed": {
          "type": "array",
          "items": {
            "type": "object"
          },
          "description": "Linked entities traversed from root"
        },
        "declared_graph": {
          "type": "object",
          "description": "#1142: optional bounded declared graph projection with hash-only source, span, scope, origin, validity, and support state."
        }
      },
      "required": [
        "entity",
        "traversed"
      ]
    },
    "annotations": {
      "readOnlyHint": true
    },
    "title": "Traverse Entity Graph"
  },
  {
    "name": "perseus_vault_graph_drift",
    "description": "Read-only graph/entities/indexes/receipts drift report (#869): counts unattested edges (no evidence anchor — NOT serveable by the graph recall arms), dangling links, links to archived/expired targets, cross-workspace links, stale community memberships, FTS drift, and journal receipts referencing missing entities. `consistent` is true when all structural graph checks are clear. Run this after upgrades or bulk imports to see whether the link graph is in a serveable, synchronized state.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "workspace_hash": {
          "type": "string",
          "description": "Optional workspace scope. Omit (or \"\") for all workspaces."
        }
      }
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "checked_at_unix_ms": {
          "type": "integer"
        },
        "workspace": {
          "type": "string"
        },
        "entities": {
          "type": "object"
        },
        "links": {
          "type": "object"
        },
        "drift": {
          "type": "object"
        },
        "consistent": {
          "type": "boolean"
        }
      }
    },
    "annotations": {
      "readOnlyHint": true
    },
    "title": "Graph Drift Report"
  },
  {
    "name": "perseus_vault_graph_attest",
    "description": "Stamp the from-side entity id as the evidence anchor on legacy edges that lack one (pre-#869 rows, hand-edited data), making them serveable by the graph recall arms. Workspace-scoped; use dry_run to preview. Applied runs journal one `graph_attest` event. After attestation, perseus_vault_graph_drift reports unattested = 0 for the covered scope.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "workspace_hash": {
          "type": "string",
          "description": "Optional workspace scope. Omit (or \"\") for all workspaces."
        },
        "dry_run": {
          "type": "boolean",
          "default": false,
          "description": "Preview the stamping without writing."
        }
      }
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "dry_run": {
          "type": "boolean"
        },
        "workspace": {
          "type": "string"
        },
        "entities_affected": {
          "type": "integer"
        },
        "links_to_stamp": {
          "type": "integer"
        },
        "links_stamped": {
          "type": "integer"
        },
        "journal_event": {
          "type": "string"
        }
      }
    },
    "annotations": {
      "destructiveHint": true
    },
    "title": "Attest Legacy Graph Edges"
  },
  {
    "name": "perseus_vault_score",
    "description": "Assign a quality score (0.0–1.0) to an entity. The score persists as an importance floor: decay_tick/cohere never recompute decay_score below it, so an explicitly scored memory survives idle time indefinitely (fidelity beats recency). Scores >= 0.7 also mark the entity verified. Re-score with 0.0 to clear the floor. Use this to mark entities as accurate, verified, or deprecated.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "category": {
          "type": "string",
          "description": "Entity category to score"
        },
        "key": {
          "type": "string",
          "description": "Entity key to score"
        },
        "score": {
          "type": "number",
          "description": "Quality score 0.0–1.0. 1.0 = verified, 0.5 = neutral, 0.0 = low quality"
        }
      },
      "required": [
        "category",
        "key",
        "score"
      ]
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "found": {
          "type": "boolean",
          "description": "Whether the entity was found"
        },
        "category": {
          "type": "string",
          "description": "Entity category"
        },
        "key": {
          "type": "string",
          "description": "Entity key"
        },
        "score": {
          "type": "number",
          "description": "Quality score assigned"
        }
      }
    },
    "annotations": {
      "destructiveHint": true
    },
    "title": "Score Entity Quality"
  },
  {
    "name": "perseus_vault_follow",
    "description": "Record whether an entity (typically a convention/insight/lesson) was actually FOLLOWED or MISSED by the agent — the honest follow-rate signal. Unlike retrieval_count (how often a memory is recalled), this tracks whether recall changed behavior. After enough attempts, efficacy_status flips to 'useful' or 'dead' and feeds into decay scoring so ignored rules decay out of recall while followed ones resist decay.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "category": {
          "type": "string",
          "description": "Entity category"
        },
        "key": {
          "type": "string",
          "description": "Entity key"
        },
        "followed": {
          "type": "boolean",
          "description": "true if the agent's action followed/honored this entity's guidance, false if it was ignored/missed"
        },
        "context": {
          "type": "string",
          "description": "Optional description of the action/context this observation relates to"
        },
        "workspace_hash": {
          "type": "string",
          "description": "Workspace scope filter. When set, the stamped row is resolved with strict workspace equality — the same semantics as a workspace-scoped recall — so the signal lands on the row the agent actually saw (no global fallback). Omit to keep the unscoped deterministic pick (global '' row first, then lexicographically-first workspace)."
        }
      },
      "required": [
        "category",
        "key",
        "followed"
      ]
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "found": {
          "type": "boolean",
          "description": "Whether the entity was found"
        },
        "category": {
          "type": "string"
        },
        "key": {
          "type": "string"
        },
        "follow_count": {
          "type": "integer"
        },
        "miss_count": {
          "type": "integer"
        },
        "follow_rate": {
          "type": "number"
        },
        "efficacy_status": {
          "type": "string",
          "description": "'unverified' | 'useful' | 'dead'"
        }
      }
    },
    "annotations": {
      "destructiveHint": true
    },
    "title": "Record Follow/Miss Efficacy Signal"
  },
  {
    "name": "perseus_vault_operator_review",
    "description": "Read-only operator review queue for contradictions, stale/low-actionability facts, and deprecated supersession lag. Does not resolve or hide findings.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "category": {"type": "string", "description": "Category to review (default general)."},
        "limit": {"type": "integer", "minimum": 1, "maximum": 1000},
        "stale_threshold": {"type": "number", "minimum": 0, "maximum": 1}
      }
    }
  },
  {
    "name": "perseus_vault_eval_history",
    "description": "#930: read-only scheduled-recall evaluation history — bounded quality-run snapshots (nightly curation + midday eval) with per-metric trend and regression breach records. Never mutates.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "kind": {"type": "string", "description": "Cadence filter: nightly | midday | manual (default all)."},
        "limit": {"type": "integer", "minimum": 1, "maximum": 100, "description": "Max runs (default 20)."},
        "regressed_only": {"type": "boolean", "description": "Only runs with regression breaches (default false)."}
      }
    }
  },
  {
    "name": "perseus_vault_web_gap_fill",
    "description": "#929: OPT-IN live-web gap-fill write-back. The vault never fetches the web; the agent fetches, then reports grounded content + source URLs here for validation (allowlisted hosts, no secrets) and audited storage as unverified-until-confirmed.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "query": {"type": "string", "description": "The recall query that missed."},
        "content": {"type": "string", "description": "Agent-fetched page content (max 64 KiB)."},
        "title": {"type": "string", "description": "Page title (max 512 chars)."},
        "sources": {"type": "array", "items": {"type": "string"}, "description": "1-8 http/https source URLs actually fetched by the agent."},
        "category": {"type": "string", "description": "Entity category (default \"web\")."},
        "key": {"type": "string", "description": "Stable key (default: web-<sha256(content)[..16]>)."},
        "workspace_hash": {"type": "string", "description": "Workspace scope (required; must be allowlisted)."},
        "agent_id": {"type": "string", "description": "Write attribution."},
        "relevance_score": {"type": "number", "description": "Agent-judged relevance 0-1 (must clear the configured floor)."}
      }
    }
  },
  {
    "name": "perseus_vault_mental_model_set",
    "description": "#886: create or refresh a curated mental model — the ONLY sanctioned write path for the mental_model category (auto-generated passes refuse it). Versioned via the audited remember path (entity_history); provenance stamped (curated_by/curated_at); revision bumps on every re-assert; review clock resets. recall_when triggers attach for scheduled re-verification.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "key": {
          "type": "string",
          "description": "Stable key of the mental model (e.g. \"stack-portal\")"
        },
        "summary": {
          "type": "string",
          "description": "The curated summary (1..=4096 chars) — what the model answers; consulted before observations and raw facts in ask/recall"
        },
        "scope": {
          "type": "string",
          "description": "Raw-fact category this model covers (\"\" = none); enables the newer-facts staleness check"
        },
        "source_ids": {
          "type": "array",
          "items": {"type": "string"},
          "description": "Provenance: raw fact / observation entity ids it was curated from"
        },
        "recall_when": {
          "type": "array",
          "items": {"type": "string"},
          "description": "Triggers for scheduled re-verification (matched by perseus_vault_recall_when / prepare)"
        },
        "review_interval_days": {
          "type": "integer",
          "default": 30,
          "description": "Age-based review interval (1..=3650)"
        },
        "workspace_hash": {"type": "string", "description": "Workspace scope (default global/empty)"},
        "requesting_agent_id": {"type": "string", "description": "Curator identity (default \"operator\")"}
      },
      "required": ["key", "summary"]
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "ok": {"type": "boolean"},
        "id": {"type": "string"},
        "key": {"type": "string"},
        "revision": {"type": "integer"},
        "curated_by": {"type": "string"}
      }
    }
  },
  {
    "name": "perseus_vault_mental_model_review",
    "description": "#886: mental-model review — list flagged stale curated summaries (reason: age / newer_facts:<key> / malformed_body, with age_days and newest-fact trace), or stamp an operator approve/dismiss decision (resets the age clock and records the decision; the summary itself only changes via perseus_vault_mental_model_set). Flags are also surfaced in perseus_vault_operator_review.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "action": {
          "type": "string",
          "enum": ["list", "approve", "dismiss"],
          "default": "list",
          "description": "list flagged stale models (default) | approve | dismiss"
        },
        "key": {"type": "string", "description": "Key of the model to decide on (required for approve/dismiss)"},
        "workspace_hash": {"type": "string", "description": "Workspace scope (default global/empty)"},
        "requesting_agent_id": {"type": "string", "description": "Reviewer identity (default \"operator\")"},
        "limit": {"type": "integer", "default": 50, "maximum": 1000}
      }
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "action": {"type": "string"},
        "flagged": {"type": "array", "items": {"type": "object"}},
        "flagged_count": {"type": "integer"}
      }
    }
  },
  {
    "name": "perseus_vault_write_quarantine",
    "description": "#874: review the write-quarantine hold — writes whose measured interference exceeded the configured bound are staged here (never served by any read surface) instead of committing to memory. list (default): pending holds with scores; show: full record incl. decrypted body + interference report; release: materialize one through the audited remember path (the operator review IS the approval; refused when the identity is already live); delete: drop one without materialization. Every decision is journaled (interference_released / interference_deleted). Pending items also surface in perseus_vault_operator_review.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "action": {
          "type": "string",
          "enum": ["list", "show", "release", "delete"],
          "default": "list",
          "description": "list (default) | show | release | delete"
        },
        "id": {"type": "string", "description": "Quarantine id (required for show/release/delete)"},
        "workspace_hash": {"type": "string", "description": "Workspace scope for list (default all)"},
        "limit": {"type": "integer", "default": 50, "maximum": 10000},
        "requesting_agent_id": {"type": "string", "description": "Reviewer identity stamped into the journal (default empty)"}
      }
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "count": {"type": "integer"},
        "items": {"type": "array", "items": {"type": "object"}},
        "released": {"type": "boolean"},
        "deleted": {"type": "boolean"}
      }
    }
  },
  {
    "name": "perseus_vault_admission_decide",
    "description": "#1107: resolve a proposed trust-admission candidate through an explicit operator decision. approve re-signs the pending evidence as SAVE and activates the existing row through the verified writer; reject requires rejection_class=drop or block, re-signs the evidence as that terminal class, archives the row, and never serves it. Both transitions are hash-only in the response, record an admission_review_started intent before mutation, and append a completed admission_approved/admission_rejected receipt only after durable transition. Public calls require an initialized clientInfo.name and an enforce-mode memory.admission.review authority for the exact workspace.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "category": {"type": "string", "minLength": 1, "description": "Candidate entity category"},
        "key": {"type": "string", "minLength": 1, "description": "Candidate entity key"},
        "workspace_hash": {"type": "string", "minLength": 1, "description": "Exact non-empty workspace scope of the candidate"},
        "requesting_agent_id": {"type": "string", "minLength": 1, "description": "Operator/reviewer identity stamped into the audit event"},
        "decision": {"type": "string", "enum": ["approve", "reject"]},
        "rejection_class": {"type": "string", "enum": ["drop", "block"], "description": "Required when decision=reject"},
        "reason": {"type": "string", "minLength": 1, "description": "Bounded non-empty review reason; the response stores only its SHA-256"}
      },
      "required": ["category", "key", "workspace_hash", "requesting_agent_id", "decision", "reason"],
      "allOf": [
        {
          "if": {"properties": {"decision": {"const": "reject"}}},
          "then": {"required": ["rejection_class"]}
        }
      ]
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "ok": {"type": "boolean"},
        "id": {"type": "string"},
        "category": {"type": "string"},
        "key": {"type": "string"},
        "decision": {"type": "string", "enum": ["approve", "reject"]},
        "outcome_class": {"type": "string", "enum": ["save", "drop", "block"]},
        "status": {"type": "string"},
        "serveable": {"type": "boolean"},
        "audit_event_id": {"type": "string"},
        "reason_sha256": {"type": "string"}
      }
    }
  },
  {
    "name": "perseus_vault_admission_quarantine",
    "description": "#1026: review the admission-quarantine hold — candidates disposed as `quarantined` by trust admission are sealed OUTSIDE the authoritative head (never served by any read surface; storage presence confers no authority). list (default): active candidates with attempt metadata (no bodies); show: full sealed record incl. decrypted body + admission attempt linkage + hash-only receipt; retire: record the review decision (the row is retained so its proposal identifier stays retired); purge: reclaim retired rows and/or active rows past the age watermark (purged proposal identifiers become reusable). Every decision is journaled (admission_quarantined / admission_quarantine_retired / admission_quarantine_purged). Pending items also surface in perseus_vault_operator_review.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "action": {
          "type": "string",
          "enum": ["list", "show", "retire", "purge"],
          "default": "list",
          "description": "list (default) | show | retire | purge"
        },
        "id": {"type": "string", "description": "Quarantine id (required for show/retire)"},
        "workspace_hash": {"type": "string", "description": "Workspace scope for list (default all)"},
        "include_retired": {"type": "boolean", "default": false, "description": "list: include retired rows (default active only)"},
        "limit": {"type": "integer", "default": 50, "maximum": 10000},
        "purge_retired": {"type": "boolean", "default": true, "description": "purge: reclaim retired rows (default true)"},
        "max_age_days": {"type": "integer", "minimum": 1, "maximum": 3650, "description": "purge: reclaim active rows older than this many days (default 30)"},
        "requesting_agent_id": {"type": "string", "description": "Reviewer identity stamped into the journal (default empty)"}
      }
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "count": {"type": "integer"},
        "items": {"type": "array", "items": {"type": "object"}},
        "retired": {"type": "boolean"},
        "purged": {"type": "integer"}
      }
    }
  },
  {
    "name": "perseus_vault_writer_handoff",
    "description": "#1027: epoch-fenced writer handoff — prevent split-brain on concurrent multi-agent writes. Directory-serialized lifecycle actions, each appending a signed lifecycle result (receipt digest): prepare (open the handoff pointer; the source may still advance), abort (clear the pointer; source stays active), fence (clear the writer + advance the epoch — after this NO writer is authorized; a crash in the Fence→Activate gap leaves zero writers, fail-closed), retarget (advance target + epoch; the abandoned target can no longer activate), activate (admission against the exact fenced revision; the current target becomes the active writer, epoch advances again). status: read the directory. Every write (remember) against an active directory must present the current writer_epoch — stale writes fail with a stable StaleRevision / WriterEpoch reason. Workspace-scoped; absent directory = unfenced legacy posture.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "action": {
          "type": "string",
          "enum": ["prepare", "abort", "fence", "retarget", "activate", "status"],
          "default": "status",
          "description": "status (default) | prepare | abort | fence | retarget | activate"
        },
        "workspace_hash": {"type": "string", "description": "Workspace whose writer directory is managed (required except status; must be non-empty)"},
        "target_agent_id": {"type": "string", "description": "Handoff target agent (required for prepare/retarget)"},
        "presented_epoch": {"type": "integer", "description": "activate: the fenced epoch the activating agent must present (exact match required)"},
        "requesting_agent_id": {"type": "string", "description": "Acting identity — activate requires it to equal target_agent_id; stamped into lifecycle receipts"}
      }
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "epoch": {"type": "integer"},
        "pointer_state": {"type": "string"},
        "writer_agent_id": {"type": "string"},
        "target_agent_id": {"type": "string"},
        "receipt_digest": {"type": "string"},
        "lifecycle_len": {"type": "integer"},
        "directory": {"type": "object"}
      }
    }
  },
  {
    "name": "perseus_vault_impact_report",
    "description": "#1029: supersession impact index — when a fact is superseded or retracted, enumerate the downstream decisions and actions that derived from it (reverse closure over derived_from citations + action justifications). Lists dependent entities ordered by authority (importance) + recency, flags PENDING actions whose cited justification changed (AAR review flag — re-validate freshness before execution), and lists COMPLETED actions for review (external effects are irreversible — flag only, never automatic reversal). Bounded closure: depth_cap (1..16, default 3), age_cap_days (default 365); as_of_unix_ms computes the report at a past transaction instant (v1 filters by dependent creation time; full bi-temporal closure via entity_history is a documented follow-on). Computed lazily at read time.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "entity_id": {"type": "string", "description": "The changed fact's id (alternative to category+key)"},
        "category": {"type": "string", "description": "The changed fact's category (with key; alternative to entity_id)"},
        "key": {"type": "string", "description": "The changed fact's key (with category; alternative to entity_id)"},
        "depth_cap": {"type": "integer", "minimum": 1, "maximum": 16, "default": 3, "description": "Max closure depth (transitive derived_from hops)"},
        "age_cap_days": {"type": "integer", "minimum": 1, "maximum": 36500, "default": 365, "description": "Ignore dependents older than this many days"},
        "as_of_unix_ms": {"type": "integer", "description": "Compute the report as of this transaction instant (default: now)"}
      }
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "target": {"type": "object"},
        "dependents": {"type": "array", "items": {"type": "object"}},
        "pending_actions": {"type": "array", "items": {"type": "object"}},
        "completed_actions": {"type": "array", "items": {"type": "object"}},
        "bounded_closure": {"type": "object"}
      }
    }
  },
  {
    "name": "perseus_vault_finding_record",
    "description": "#1033: record an authenticated impact finding — the durable, receipted admission record a detection pass produces about a superseded/retracted fact (the supersession impact index surface is the read side; this is the write side compensation intents cite). A finding is detection output, NEVER a decision: it cannot self-trigger execution; the disposition (accept-drift / revalidate-pending / open-compensation-case / escalate) is chosen by the authority plane. Compensation intents (action_intent with compensates_for) must cite a finding_ref whose covers list includes the compensated effect and whose cited_head matches the presented superseding_head — fail-closed, stable compensation_* reason codes. covered effects must reference existing action receipts; (category,key) or entity_id targets the changed fact.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "finding_ref": {"type": "string", "description": "Stable caller-facing reference, unique per workspace"},
        "workspace_hash": {"type": "string", "description": "Workspace the finding belongs to"},
        "agent_id": {"type": "string", "description": "Recording agent identity (stamped into the journal)"},
        "category": {"type": "string", "description": "Changed fact's category (with key; alternative to entity_id)"},
        "key": {"type": "string", "description": "Changed fact's key (with category; alternative to entity_id)"},
        "entity_id": {"type": "string", "description": "Changed fact's entity id (alternative to category+key)"},
        "cited_head": {"type": "string", "description": "Exact superseding head that invalidated the original justification (required)"},
        "covers": {"type": "array", "items": {"type": "string"}, "description": "Original effect/action receipt ids this finding covers (each must exist)"},
        "basis": {"type": "string", "description": "Why the finding exists (e.g. 'supersession', 'retraction')"}
      },
      "required": ["finding_ref", "cited_head"]
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "id": {"type": "string"},
        "finding_ref": {"type": "string"},
        "cited_head": {"type": "string"},
        "covers": {"type": "array", "items": {"type": "string"}},
        "status": {"type": "string"},
        "archived": {"type": "boolean"},
        "created_at_unix_ms": {"type": "integer"}
      }
    }
  },
  {
    "name": "perseus_vault_grounding_admit",
    "description": "#1034: admit a deterministic grounding fingerprint for evidence grounded to a file/symbol. Captures a K=64 seeded-sha256 trigram MinHash + neighbor set at admission (zero LLM, reproducible). Fail-closed authoring rule: if trustworthy ground facts are unavailable, stop and report it — never invent node ids or fingerprints. The agent supplies the grounded content; the vault never fetches. Re-admission refreshes the baseline with an appended provenance trail (never silent last-write-wins).",
    "inputSchema": {
      "type": "object",
      "properties": {
        "workspace_hash": {"type": "string", "description": "Workspace the grounding belongs to"},
        "entity_id": {"type": "string", "description": "Evidence entity the grounding anchors (must exist)"},
        "target_ref": {"type": "string", "description": "File path or symbol reference the evidence is grounded to"},
        "kind": {"type": "string", "enum": ["file", "symbol"], "description": "What target_ref names"},
        "content": {"type": "string", "description": "The grounded source content at admission (agent-supplied; bounded)"}
      },
      "required": ["entity_id", "target_ref", "content"]
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "id": {"type": "string"},
        "target_ref": {"type": "string"},
        "fingerprint_hex": {"type": "string"},
        "baseline_digest": {"type": "string"},
        "status": {"type": "string"},
        "captured_at_unix_ms": {"type": "integer"}
      }
    }
  },
  {
    "name": "perseus_vault_grounding_reconcile",
    "description": "#1034: reconcile admitted groundings against a current-content scan (agent-supplied target_ref/content pairs). Deterministic, zero LLM: identical digest → ok; exists-but-changed → GROUNDING_DRIFT; reconcile score (0.7×minhashJaccard + 0.3×neighborOverlap, HI 0.85 / LO 0.55) → MOVED (auto-rewrite anchor + migrate baseline with a provenance trail) / GONE (flag for review) / AMBIGUOUS (surface candidates for operator review).",
    "inputSchema": {
      "type": "object",
      "properties": {
        "workspace_hash": {"type": "string", "description": "Workspace to reconcile"},
        "current": {"type": "array", "items": {"type": "object", "properties": {
          "target_ref": {"type": "string"},
          "content": {"type": "string"}
        }}, "description": "Current content scan: target_ref + content pairs"}
      }
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "checked": {"type": "integer"},
        "ok": {"type": "integer"},
        "drift": {"type": "integer"},
        "moved": {"type": "integer"},
        "gone": {"type": "integer"},
        "ambiguous": {"type": "integer"},
        "issues": {"type": "array", "items": {"type": "object"}},
        "note": {"type": "string"}
      }
    }
  },
  {
    "name": "perseus_vault_drift_check",
    "description": "#1035: deterministic drift-check pre-pass over the store (zero LLM in detection): REFERENCE_INTEGRITY (dangling derived_from citations), GROUNDING_STATUS (drift/gone/ambiguous fingerprints), PATH_EXISTENCE (missing grounded files), CROSS_FILE_CONFLICT (two evidence entities asserting different values for the same keyed claim), STALE_ENTITY (stale vs last-access threshold). Health score = 100 − (10×error + 3×warning + 1×info). Repair scope = flagged items only (perseus_vault_drift_repair).",
    "inputSchema": {
      "type": "object",
      "properties": {
        "workspace_hash": {"type": "string", "description": "Optional workspace scope"},
        "staleness_days": {"type": "integer", "minimum": 1, "maximum": 3650, "default": 90, "description": "Staleness threshold in days"}
      }
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "health_score": {"type": "integer"},
        "errors": {"type": "integer"},
        "warnings": {"type": "integer"},
        "infos": {"type": "integer"},
        "checker_counts": {"type": "object"},
        "issues": {"type": "array", "items": {"type": "object"}},
        "note": {"type": "string"}
      }
    }
  },
  {
    "name": "perseus_vault_drift_repair",
    "description": "#1035: targeted repair + verify leg of the drift loop. Mechanical fixes only: unlink dangling derived_from references (journaled), acknowledge grounding findings. Contradictions, staleness, and missing files are never auto-resolved — they land in requires_review for the operator queue. Re-runs the check and reports the before/after health-score delta; a repair that regresses the score is refused fail-closed.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "workspace_hash": {"type": "string", "description": "Optional workspace scope"},
        "staleness_days": {"type": "integer", "minimum": 1, "maximum": 3650, "default": 90, "description": "Staleness threshold in days"}
      }
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "before_score": {"type": "integer"},
        "after_score": {"type": "integer"},
        "repaired": {"type": "array", "items": {"type": "string"}},
        "requires_review": {"type": "array", "items": {"type": "string"}},
        "note": {"type": "string"}
      }
    }
  },
  {
    "name": "perseus_vault_restore_forward",
    "description": "#1028: forward-only restoration — restore from a checkpoint directory as an audited VERSION ADVANCE of the current head, never as a rewrite. Each checkpoint entity advances its (category, key, workspace) identity: the pre-restore version moves to entity_history (the parent; a rollback is just another forward migration) and the checkpoint body becomes the new head. Protected authority paths always take CURRENT values — authority, policy, revocation, issuer, writer, epoch, dirSeq/lifecycle, createdFrom/provenance are excluded from the mask (M ∩ P = ∅; only `entities` is maskable in v1), so a restore can never revive a stale credential, resurrect a superseded authority, or undo a recorded external effect (authorized actions untouched). Workspace-scoped; an active writer directory (#1027) requires the current writer_epoch. Report: restored / superseded_current_heads / created / errors.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "checkpoint_dir": {"type": "string", "description": "Directory of vault-format .md files (the vault_export wire shape)"},
        "workspace_hash": {"type": "string", "description": "Workspace to restore into (required)"},
        "path_mask": {"type": "array", "items": {"type": "string"}, "default": ["entities"], "description": "State paths to restore (only `entities` in v1; protected paths are refused fail-closed)"},
        "writer_epoch": {"type": "integer", "description": "Required when the workspace has an active writer directory (#1027)"},
        "requesting_agent_id": {"type": "string", "description": "Acting identity stamped into the journal"}
      }
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "restored": {"type": "integer"},
        "superseded_current_heads": {"type": "integer"},
        "created": {"type": "integer"},
        "errors": {"type": "array", "items": {"type": "string"}},
        "protected_paths": {"type": "array", "items": {"type": "string"}}
      }
    }
  },
  {
    "name": "perseus_vault_op_run",
    "description": "#871: durable long-running operation states. Lifecycle tool for the shared run/run-item contract (maintenance, embed, consolidation, export/import, reindex). Actions: begin (queued; requires op_type, optional scope/input_digest/max_retries 0..10/created_by), start (queued->running), progress (done/failed/total counters; partial derived), complete (running->completed with receipt linkage), fail (running->failed; error_detail is sanitized at rest — secrets masked, length capped), failed_to_start (queued->failed_to_start), cancel (queued|running->cancelled), timeout (running->failed with timeout flag), item_add/item_start/item_complete/item_fail/item_cancel (per-item receipts; UNIQUE(run_id, item_ref)). Terminal states accept no further transitions. Restart recovery marks in-flight runs interrupted (mark-only); resume only via perseus_vault_op_run_retry.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "action": {
          "type": "string",
          "enum": ["begin", "start", "progress", "complete", "fail", "failed_to_start", "cancel", "timeout", "item_add", "item_start", "item_complete", "item_fail", "item_cancel"],
          "default": "begin",
          "description": "Lifecycle action"
        },
        "run_id": {"type": "string", "description": "Run id (opr-...) for all actions except begin"},
        "op_type": {"type": "string", "description": "Operation kind for begin: consolidate|embed_flush|export|import|decay|maintain|reindex|cohere|compact|custom"},
        "scope": {"type": "string", "description": "Workspace hash or empty for global"},
        "input_digest": {"type": "string", "description": "sha256 of the input reference set (idempotency anchor)"},
        "max_retries": {"type": "integer", "default": 2, "minimum": 0, "maximum": 10},
        "created_by": {"type": "string", "description": "Caller identity for begin"},
        "done": {"type": "integer", "description": "progress: items completed"},
        "failed": {"type": "integer", "description": "progress: items failed"},
        "total": {"type": "integer", "description": "progress: expected items (omit to keep stored total)"},
        "receipt": {"type": "string", "description": "complete: terminal receipt linkage (journal event id / artifact ref)"},
        "error_class": {"type": "string", "description": "fail/item_fail: error class"},
        "error_detail": {"type": "string", "description": "fail/item_fail: detail (sanitized at rest)"},
        "item_ref": {"type": "string", "description": "item ops: item reference (entity id / file path / ordinal)"},
        "item_digest": {"type": "string", "description": "item_add: item digest"},
        "receipt_ref": {"type": "string", "description": "item_complete: per-item receipt linkage"}
      }
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "id": {"type": "string"},
        "state": {"type": "string"}
      }
    }
  },
  {
    "name": "perseus_vault_op_run_list",
    "description": "#871: list durable operation runs, newest first. Optional state filter (queued|running|completed|failed|cancelled|interrupted|failed_to_start) and op_type filter; bounded limit (1..=100, default 20).",
    "inputSchema": {
      "type": "object",
      "properties": {
        "state": {"type": "string", "description": "Optional terminal-state filter"},
        "op_type": {"type": "string", "description": "Optional operation-kind filter"},
        "limit": {"type": "integer", "default": 20, "minimum": 1, "maximum": 100}
      }
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "count": {"type": "integer"},
        "runs": {"type": "array", "items": {"type": "object"}}
      }
    }
  },
  {
    "name": "perseus_vault_op_run_get",
    "description": "#871: fetch one durable operation run with its per-item receipts.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "run_id": {"type": "string", "description": "Run id (opr-...)"}
      },
      "required": ["run_id"]
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "run": {"type": "object"},
        "items": {"type": "array", "items": {"type": "object"}}
      }
    }
  },
  {
    "name": "perseus_vault_op_run_retry",
    "description": "#871: bounded, scoped, idempotent retry of a TERMINAL run. Forks a NEW child run re-queuing only failed/cancelled/interrupted/unattempted items; completed items are carried into the child with their receipts (never re-executed — retry cannot duplicate writes or receipts). Refused fail-closed on retry exhaustion (retry_count >= max_retries) or when nothing is recoverable.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "run_id": {"type": "string", "description": "Terminal run id to retry"}
      },
      "required": ["run_id"]
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "retried_from": {"type": "string"},
        "child_run_id": {"type": "string"},
        "state": {"type": "string"},
        "retry_count": {"type": "integer"}
      }
    }
  },
  {
    "name": "perseus_vault_op_run_prune",
    "description": "#871: retention prune of TERMINAL runs older than retention_days (min 1, default PERSEUS_VAULT_OP_RETENTION_DAYS=30) plus their items. In-flight runs are never pruned. maintain runs a prune pass each cycle.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "retention_days": {"type": "integer", "minimum": 1, "description": "Retention bound (default env PERSEUS_VAULT_OP_RETENTION_DAYS, 30)"}
      }
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "pruned": {"type": "integer"},
        "retention_days": {"type": "integer"}
      }
    }
  },
  {
    "name": "perseus_vault_preload_resolve",
    "description": "#875: resolve open preload usage events into per-session precision/recall. Events older than the usage window are marked used/unused from entity read activity (serving itself never counts), then folded into preload_sessions. window_minutes defaults to 30. Telemetry bookkeeping only — never touches entity bodies.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "window_minutes": {
          "type": "integer",
          "description": "Session usage window in minutes (default 30)."
        }
      }
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "events_resolved": {"type": "integer"},
        "sessions_written": {"type": "integer"},
        "window_minutes": {"type": "integer"}
      }
    }
  },
  {
    "name": "perseus_vault_preload_stats",
    "description": "#875: read-only preload usage telemetry — which preloaded memories actually got used. Per-trigger precision/recall (separate from #872 serving-concentration), per-session rows, or overall aggregates. Usage = the entity was touched after serving (read paths only; serving itself never counts). Run this before perseus_vault_preload_propose to see the evidence behind tuning proposals.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "scope": {
          "type": "string",
          "enum": ["overall", "trigger", "session"],
          "default": "overall",
          "description": "'trigger': per recall_when trigger precision (used/served) + recall (used/(used+missed-by-trigger)); 'session': per-session precision/recall/miss_rate rows; 'overall': aggregate summary."
        },
        "limit": {
          "type": "integer",
          "default": 50,
          "description": "Max rows for trigger/session scopes (1-1000)"
        },
        "since_days": {
          "type": "integer",
          "default": 7,
          "description": "Only events/sessions at least this recent (0 = all)"
        }
      }
    },
    "outputSchema": {"type": "object"}
  },
  {
    "name": "perseus_vault_preload_propose",
    "description": "#875: offline trigger-tuning pass. From resolved usage history, raises PENDING proposals: retire for triggers served >= PERSEUS_VAULT_PRELOAD_MIN_SERVED (3) with precision < PERSEUS_VAULT_PRELOAD_RETIRE_PRECISION (0.25); add_trigger for entities used in >= 2 sessions but never preloaded (word from the sessions' contexts). Proposals write ONLY the proposals table (journaled); entity mutations happen exclusively via perseus_vault_preload_review approve — never silently.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "by": {
          "type": "string",
          "default": "operator",
          "description": "Agent id recorded as the proposal author"
        }
      }
    },
    "outputSchema": {"type": "object"}
  },
  {
    "name": "perseus_vault_preload_review",
    "description": "#875: operator review queue for preload trigger tuning — the ONLY mutation surface. 'approve' applies the proposal through the audited remember path (journal preload_tuning_applied + entity_history provenance, revision bump): retire removes the trigger from the entity's recall_when (others untouched); add_trigger appends the proposed word. 'dismiss' records the decision without mutating anything. Both are journaled with the operator id.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "action": {
          "type": "string",
          "enum": ["list", "approve", "dismiss"],
          "default": "list",
          "description": "'list': pending proposals with rationale; 'approve': apply proposal_id; 'dismiss': decline proposal_id with a reason."
        },
        "proposal_id": {
          "type": "string",
          "description": "Proposal id (required for approve/dismiss)"
        },
        "reason": {
          "type": "string",
          "description": "Dismissal reason (dismiss only)"
        },
        "by": {
          "type": "string",
          "default": "operator",
          "description": "Agent id recorded as the decision maker"
        },
        "limit": {
          "type": "integer",
          "default": 50,
          "description": "Max proposals for list (1-1000)"
        }
      }
    },
    "outputSchema": {"type": "object"}
  },
  {
    "name": "perseus_vault_guide_seed",
    "description": "#924: seed (create or refresh) the vault operating guide — a 'how to use this vault' manual living as a discoverable entity (category 'guide', key 'vault-operating-guide') with recall_when triggers ('operating guide', ...). Session context blocks then emit a one-line pointer instead of inlining operating instructions; agents retrieve the full guide on demand via normal recall. Idempotent: re-seeding updates in place, never duplicates. Advisory metadata only — never gates writes.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "workspace_hash": {
          "type": "string",
          "default": "",
          "description": "Workspace scope for the guide entity (empty = global)."
        }
      }
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "id": {"type": "string"},
        "category": {"type": "string"},
        "key": {"type": "string"},
        "action": {"type": "string"}
      }
    }
  },
  {
    "name": "perseus_vault_declared_schema_set",
    "description": "#923: declare (or replace) the typed retrieval contract for a category — the deterministic exact-match arm. Fields are typed ('scalar' = exact string equality, 'string_list' = array membership) and may be facet-eligible. Advisory retrieval metadata only: never gates writes. Fail-closed validation: unknown field types, duplicate/empty names, reserved names (id/category/key/recall_when/origin/external_refs/expires_at), >32 fields, >16 facets, or >500-byte query_guidance are errors. Re-declaring bumps the schema version; exact-match queries then follow the new contract.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "category": {
          "type": "string",
          "description": "Category this contract describes (may not be a reserved category)"
        },
        "fields": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "name": {"type": "string"},
              "type": {"type": "string", "enum": ["scalar", "string_list"]},
              "facet": {"type": "boolean", "default": false}
            },
            "required": ["name", "type"]
          },
          "description": "1-32 typed fields. Values are read from each entity's top-level body_json keys at query time."
        },
        "query_guidance": {
          "type": "string",
          "default": "",
          "description": "Advisory: how agents should query this category (returned by declared_query). Max 500 bytes."
        }
      },
      "required": ["category", "fields"]
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "ok": {"type": "boolean"},
        "category": {"type": "string"},
        "version": {"type": "integer"},
        "fields": {"type": "array"},
        "query_guidance": {"type": "string"}
      }
    },
    "title": "Declare Category Retrieval Contract"
  },
  {
    "name": "perseus_vault_declared_query",
    "description": "#923: deterministic exact-match retrieval over a declared category — the no-ranking arm. Filters are AND-combined exact-equality checks against the category's declared schema: scalar fields match by exact string equality, string_list fields by array membership. Results come back in deterministic order (created_at ASC, id ASC). Facet counts are truthful and bounded (top 50 distinct values per facet, remainder rolled into 'other'). Fail-closed: undeclared categories, unknown fields, malformed filters, or non-facet facet requests are errors — never degraded to fuzzy recall.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "category": {
          "type": "string",
          "description": "Category with a declared schema (perseus_vault_declared_schema_set)"
        },
        "filters": {
          "type": "object",
          "additionalProperties": true,
          "description": "Exact-equality filters (AND-combined). Scalar field: string value to equal. String-list field: array of strings, any of which must be present."
        },
        "facets": {
          "type": "array",
          "items": {"type": "string"},
          "description": "Facet-eligible fields to count (top 50 distinct values + 'other' bucket)"
        },
        "limit": {"type": "integer", "default": 10},
        "offset": {"type": "integer", "default": 0},
        "workspace_hash": {"type": "string", "default": ""},
        "requesting_agent_id": {
          "type": "string",
          "description": "Transport-stamped requester identity used for item and facet visibility enforcement."
        }
      },
      "required": ["category"]
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "ok": {"type": "boolean"},
        "category": {"type": "string"},
        "schema": {"type": "object"},
        "total_matches": {"type": "integer"},
        "truncated": {"type": "boolean"},
        "items": {"type": "array"},
        "facet_counts": {"type": "object"}
      }
    },
    "title": "Declared Exact-Match Query"
  },
  {
    "name": "perseus_vault_conflicts",
    "description": "Detect conflicting entities in the same category — pairs with low trigram similarity in their body_json. Flags potential contradictions, duplicate-but-divergent entries, and stale-overwritten facts. Read-only by default. Opt in with resolve=true to actively invalidate the lower-certainty side of clear conflicts (superseding it into history, reversible + time-travelable via perseus_vault_as_of); that path defaults to dry_run=true so you preview first, and never resolves pairs whose certainties are within certainty_margin.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "category": {
          "type": "string",
          "default": "general",
          "description": "Category to scan for conflicts"
        },
        "threshold": {
          "type": "number",
          "default": 0.4,
          "description": "Similarity threshold — pairs below this are flagged as conflicts"
        },
        "limit": {
          "type": "integer",
          "default": 10,
          "description": "Maximum number of conflicts to return / resolve"
        },
        "offset": {
          "type": "integer",
          "default": 0,
          "description": "Number of entities to skip for pagination"
        },
        "resolve": {
          "type": "boolean",
          "default": false,
          "description": "Opt-in: invalidate the lower-certainty side of clear conflicts instead of only reporting them"
        },
        "dry_run": {
          "type": "boolean",
          "default": true,
          "description": "When resolve=true, only report what would be invalidated unless set false"
        },
        "certainty_margin": {
          "type": "number",
          "default": 0.2,
          "description": "Minimum certainty gap to auto-resolve; closer pairs are skipped as ambiguous"
        }
      },
      "required": [
        "category"
      ]
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "conflicts": {
          "type": "array",
          "items": {
            "type": "object"
          },
          "description": "Conflict pairs with similarity scores (detection mode)"
        },
        "invalidations": {
          "type": "array",
          "items": {
            "type": "object"
          },
          "description": "Winner/loser pairs invalidated or previewed (resolve mode)"
        }
      }
    },
    "annotations": {
      "readOnlyHint": false
    },
    "title": "Detect Conflicting Entities"
  },
  {
    "name": "perseus_vault_maintenance_status",
    "description": "#952: read-only maintenance/serving isolation observability. Reports the off-peak maintenance window (configured value, whether it is open now, parse errors), the live-recall SLO budget (PERSEUS_VAULT_MAINTENANCE_P95_BUDGET_MS, last probe latency), the execution-slot state (one maintenance run at a time; held = an operator-explicit run is executing), and lifetime counters (runs started / refused by the gate / mid-run SLO pauses). Maintenance is serialized, never reserved (a disabled mode consumes zero capacity), and gated by window+budget unless force:true — see docs/specs/maintenance-serving-isolation.md.",
    "inputSchema": {
      "type": "object",
      "properties": {},
      "required": []
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "window": {
          "type": "object",
          "description": "Configured off-peak window (PERSEUS_VAULT_MAINTENANCE_WINDOW, UTC HH:MM-HH:MM), whether it is open now, and any parse error (malformed config fails closed)."
        },
        "slo": {
          "type": "object",
          "description": "Live-recall SLO budget ms (PERSEUS_VAULT_MAINTENANCE_P95_BUDGET_MS; null = guard off) and the last measured recall probe latency."
        },
        "lock": {
          "type": "object",
          "description": "Execution slot: held (bool) and the operation holding it, if any."
        },
        "counters": {
          "type": "object",
          "description": "Lifetime counters: runs_started, runs_refused (gate refusals), slo_pauses (mid-run pauses)."
        }
      }
    }
  },
  {
    "name": "perseus_vault_consolidate",
    "description": "Merge overlapping/duplicative entities in the same category into durable, evidence-tracked 'observations' — the mirror image of perseus_vault_conflicts, which flags dissimilar (contradictory) pairs. Groups entities whose pairwise trigram similarity meets similarity_threshold, then creates one new entity per group (category='observation') whose body carries a summary (the highest-certainty source's content), exact-quote evidence refs (source id + verbatim quote, capped by quote_cap_chars), the full list of source entity ids as evidence, a proof_count, updated_at, a staleness flag, and (on contradiction) a preserved journey in history. The observation links back to each source (relationship='evidence_for') for full audit. #884: with refine_existing (default true), new evidence FOLDS into the best-matching existing observation (proof_count grows, no duplicates) and contradictions reconcile into its journey — 'was React, switched to Vue' — with raw facts intact for trace-back; fold/refine writes go through the audited re-assert path (entity_history snapshot). A staleness refresh pass marks observations stale when newer unconsolidated facts exist. By default sources stay live; set archive_sources=true to retire merged sources of FRESHLY CREATED observations only ('local dreaming' — verified or importance-floored sources are never archived), and cold_first=true to target the memories decay is about to claim. perseus_vault_autocohere runs a bounded cold_first+archive_sources pass automatically. Read-only preview with dry_run=true.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "category": {
          "type": "string",
          "description": "Category to scan for overlapping/duplicative entities to consolidate"
        },
        "similarity_threshold": {
          "type": "number",
          "default": 0.6,
          "description": "Trigram similarity threshold at or above which two entities are considered overlapping enough to merge"
        },
        "limit": {
          "type": "integer",
          "default": 50,
          "description": "Maximum number of observations to create"
        },
        "offset": {
          "type": "integer",
          "default": 0,
          "description": "Number of entities to skip for pagination"
        },
        "dry_run": {
          "type": "boolean",
          "default": false,
          "description": "Preview which observations would be created without writing anything"
        },
        "cold_first": {
          "type": "boolean",
          "default": false,
          "description": "Scan the COLDEST entities first (longest since last access) instead of the most recent — compress memories that are fading anyway, before decay archives them individually"
        },
        "archive_sources": {
          "type": "boolean",
          "default": false,
          "description": "Archive merged source entities after the observation is created (archive_reason names the observation; reversible). Verified or importance-floored sources are never archived."
        },
        "workspace_hash": {
          "type": "string",
          "description": "#854 workspace scope for this run. Scans, clusters, evidence links, and archive operations are strictly restricted to this workspace, and derived observations inherit it. Mutually exclusive with global=true. One of workspace_hash or global is required."
        },
        "global": {
          "type": "boolean",
          "default": false,
          "description": "#854 explicit cross-workspace mode for deliberate whole-vault consolidation. Capability-gated (memory.maintenance.global) when the caller carries a host identity. Mutually exclusive with workspace_hash."
        },
        "requesting_agent_id": {
          "type": "string",
          "default": "",
          "description": "Host identity stamped by the MCP transport. Used for global-mode authorization and stamped as author on derived observations."
        },
        "refine_existing": {
          "type": "boolean",
          "default": true,
          "description": "#884: fold new evidence into existing observations instead of creating duplicates. Near-duplicate clusters/singletons update the matched observation (proof_count, quotes, updated_at); contradictions are reconciled into its journey (history) rather than blindly overwritten. Folded evidence is never archived."
        },
        "quote_cap_chars": {
          "type": "integer",
          "default": 512,
          "minimum": 64,
          "maximum": 4096,
          "description": "#884: cap for exact-quote evidence refs (chars). Quotes are each source's note verbatim, truncated at the cap with an ellipsis marker."
        }
      },
      "required": [
        "category"
      ]
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "category": {
          "type": "string"
        },
        "entities_examined": {
          "type": "integer",
          "description": "Number of entities scanned in this category"
        },
        "observations_created": {
          "type": "integer",
          "description": "Number of new observation entities created (or would be, in dry-run)"
        },
        "source_entities_merged": {
          "type": "integer",
          "description": "Total count of source entities folded into the created observations"
        },
        "sources_archived": {
          "type": "integer",
          "description": "Sources archived because archive_sources was set (verified/importance-floored sources are exempt)"
        },
        "dry_run": {
          "type": "boolean"
        },
        "workspace_hash": {
          "type": ["string", "null"],
          "description": "#854 effective scope: the workspace this run operated in (null when global=true)"
        },
        "global": {
          "type": "boolean",
          "description": "#854 true when this run deliberately crossed all workspaces"
        },
        "observations": {
          "type": "array",
          "items": {
            "type": "object"
          },
          "description": "The observations created (or previewed), each with entity_id, key, summary, source_ids, proof_count, certainty"
        }
      }
    },
    "annotations": {
      "readOnlyHint": false
    },
    "title": "Consolidate Overlapping Facts into Observations"
  },
  {
    "name": "perseus_vault_sleep",
    "description": "#1002: bounded sleep-cycle consolidation without an LLM (CogniCore SleepProcessor borrow). One bounded scan of a category (max_entities, #952 window discipline + maintenance gate) produces PROPOSALS, never silent changes: (1) dedup — pairs at/above similarity_threshold become merge proposals; (2) contradiction — pairs with token overlap PLUS a negation word ('X works' vs 'X does not work') become conflict proposals; (3) optional compression — delegates to perseus_vault_consolidate (cold_first) so fading memories are compressed into evidence-linked observations (the only auto-committed artifact; verified/scored sources exempt). Proposals persist under sleep_proposal.* state keys and surface as the 'sleep' lane of perseus_vault_operator_review for explicit operator decisions. dry_run=true performs the identical work with zero writes.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "category": {
          "type": "string",
          "description": "Category to scan. The curated mental_model category is refused (curated-only)."
        },
        "similarity_threshold": {
          "type": "number",
          "default": 0.75,
          "description": "Trigram similarity at or above which two entities are dedup candidates (merge proposal)"
        },
        "max_entities": {
          "type": "integer",
          "default": 200,
          "description": "Scan budget: most-recently-accessed entities examined (clamped 1..=2000)"
        },
        "max_proposals": {
          "type": "integer",
          "default": 50,
          "description": "Proposal budget: cap on merge+conflict proposals per run (clamped 1..=200)"
        },
        "dry_run": {
          "type": "boolean",
          "default": false,
          "description": "Preview: identical scan and report, zero persisted proposals, zero compression writes"
        },
        "include_compression": {
          "type": "boolean",
          "default": false,
          "description": "Also run the delegated cold_first consolidate pass over the same category (its own maintenance slot; results reported under 'compression')"
        },
        "workspace_hash": {
          "type": "string",
          "description": "#854 workspace scope. Mutually exclusive with global=true. One of workspace_hash or global is required."
        },
        "global": {
          "type": "boolean",
          "default": false,
          "description": "#854 deliberate whole-vault mode, capability-gated. Mutually exclusive with workspace_hash."
        },
        "requesting_agent_id": {
          "type": "string",
          "default": "",
          "description": "Host identity stamped by the MCP transport; used for global-mode authorization."
        },
        "force": {
          "type": "boolean",
          "default": false,
          "description": "#952: explicit operator trigger — bypasses the maintenance off-peak window and the live-recall SLO start gate (mid-run pauses still apply)."
        }
      },
      "required": [
        "category"
      ]
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "category": {
          "type": "string"
        },
        "dry_run": {
          "type": "boolean"
        },
        "scanned": {
          "type": "integer",
          "description": "Entities examined in this run"
        },
        "dedup_proposals": {
          "type": "integer",
          "description": "Merge proposals found (kind=merge)"
        },
        "conflict_proposals": {
          "type": "integer",
          "description": "Negation-shaped conflict proposals found (kind=conflict)"
        },
        "compression": {
          "description": "Delegated consolidate report, or null when include_compression is false"
        },
        "proposals": {
          "type": "array",
          "items": {
            "type": "object",
            "description": "SleepProposal: kind, category, entity_a, entity_b, similarity, reason, workspace_hash, status"
          }
        },
        "maintenance_guard": {
          "description": "#952 maintenance-window status for this run"
        }
      }
    },
    "annotations": {
      "readOnlyHint": false
    },
    "title": "Sleep-Cycle Consolidation (Proposal-Only Pass)"
  },
  {
    "name": "perseus_vault_dream",
    "description": "Sleep-time LLM consolidation: batch clusters of related cold/episodic memories, reflect over each cluster via the configured LLM endpoint, and write back durable higher-order SEMANTIC insights (category='insight', semantic layer) — 'given these N memories, what stable pattern/preference/fact do they collectively imply?'. Each written insight carries evidence_for links to every source entity (full provenance), a certainty blended from LLM confidence and evidence coverage, and derivation='dream' so it is auditable and reversible. Idempotent: insights are keyed by an evidence-set hash, so re-dreaming an unchanged cluster never spawns duplicates. Contradictory sources surface as a flagged 'contradiction' insight, never a silent merge. Never fabricates: clusters that support no durable generalization are a no-op. Requires --llm-endpoint (fully local via Ollama); returns a clean error without it unless fallback_consolidate=true, which runs the non-LLM perseus_vault_consolidate pass instead. Bounded by max_entities/max_clusters budgets. Preview with dry_run=true.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "category": {
          "type": "string",
          "description": "Category to dream over. Omit to scan all categories (derived categories — insight, observation, synthesis, memories — are always skipped) until the entity budget is exhausted."
        },
        "topic_path": {
          "type": "string",
          "description": "Optional topic_path prefix filter applied to the scan."
        },
        "similarity_threshold": {
          "type": "number",
          "default": 0.3,
          "description": "Trigram similarity threshold for grouping RELATED memories into one cluster. Lower than consolidate's 0.6 on purpose: dreaming wants thematic neighborhoods, not near-duplicates."
        },
        "max_entities": {
          "type": "integer",
          "default": 100,
          "description": "Budget cap: maximum entities scanned per run (across categories)."
        },
        "max_clusters": {
          "type": "integer",
          "default": 5,
          "description": "Budget cap: maximum clusters sent to the LLM per run (= max LLM calls)."
        },
        "min_cluster_size": {
          "type": "integer",
          "default": 2,
          "description": "Minimum memories a cluster needs before it is worth dreaming over."
        },
        "dry_run": {
          "type": "boolean",
          "default": false,
          "description": "Report candidate insights and their evidence sets without writing anything."
        },
        "cold_first": {
          "type": "boolean",
          "default": true,
          "description": "Scan the COLDEST entities first (longest since last access) — consolidate fading memories into durable semantic insights before decay claims them."
        },
        "archive_sources": {
          "type": "boolean",
          "default": false,
          "description": "Archive source entities once an insight citing them is written (archive_reason names the insight; reversible). Verified or importance-floored sources are never archived; contradiction sources always stay live."
        },
        "fallback_consolidate": {
          "type": "boolean",
          "default": false,
          "description": "When no --llm-endpoint is configured, run the mechanical (non-LLM) perseus_vault_consolidate cold_first pass instead of returning an error."
        }
      },
      "required": []
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "categories_scanned": {
          "type": "array",
          "items": {
            "type": "string"
          }
        },
        "entities_examined": {
          "type": "integer",
          "description": "Number of entities scanned across all categories this run"
        },
        "clusters_dreamed": {
          "type": "integer",
          "description": "Clusters actually sent to the LLM this run"
        },
        "insights_written": {
          "type": "integer",
          "description": "Semantic insights written (or that would be, in dry-run)"
        },
        "insights_deduped": {
          "type": "integer",
          "description": "Insights skipped because the identical evidence set was already dreamed"
        },
        "contradictions_flagged": {
          "type": "integer",
          "description": "Insights flagged as contradictions among their sources"
        },
        "sources_archived": {
          "type": "integer",
          "description": "Sources archived because archive_sources was set (verified/importance-floored sources are exempt)"
        },
        "dry_run": {
          "type": "boolean"
        },
        "workspace_hash": {
          "type": ["string", "null"],
          "description": "#854 effective scope: the workspace this run operated in (null when global=true)"
        },
        "global": {
          "type": "boolean",
          "description": "#854 true when this run deliberately crossed all workspaces"
        },
        "insights": {
          "type": "array",
          "items": {
            "type": "object"
          },
          "description": "The insights written (or previewed), each with entity_id, key, summary, insight_type, confidence, source_ids, category, contradiction, deduped"
        },
        "fallback": {
          "type": "string",
          "description": "Present only when fallback_consolidate ran (no LLM endpoint): always \"consolidate\". The report then has this union shape — categories_scanned, entities_examined, observations_created, sources_archived, dry_run — instead of the LLM dream counters."
        },
        "note": {
          "type": "string",
          "description": "Fallback-only explanation of why the mechanical pass ran"
        },
        "observations_created": {
          "type": "integer",
          "description": "Fallback-only: observations created by the mechanical consolidate pass"
        }
      }
    },
    "annotations": {
      "readOnlyHint": false
    },
    "title": "Dream: LLM Consolidation of Episodic Memory into Semantic Insights"
  },
  {
    "name": "perseus_vault_seal",
    "description": "Record a seal (SHA-256 commitment) over a live entity's stored content — hash + label only, never the content itself. Compare-on-recall and perseus_vault_tamper_scan surface any later mismatch as a tamper event naming the entity, so a tampered store is never served silently. Integrity != truth: a seal proves unchanged-since-sealed, never true-when-written.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "target_id": {
          "type": "string",
          "description": "Entity id to seal."
        },
        "workspace_hash": {
          "type": "string",
          "description": "Workspace of the entity (optional; empty = global)."
        },
        "agent_id": {
          "type": "string",
          "description": "Sealing agent identity for the audit trail."
        },
        "label": {
          "type": "string",
          "description": "Human-readable label recorded with the seal."
        }
      },
      "required": ["target_id"]
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "seal_id": {"type": "string"},
        "target_id": {"type": "string"},
        "label": {"type": "string"},
        "scope": {"type": "string"},
        "sha256": {"type": "string", "description": "SHA-256 over the sealed content (hash only — no content leak)."},
        "workspace_hash": {"type": "string"},
        "agent_id": {"type": "string"},
        "created_at_unix_ms": {"type": "integer"}
      }
    },
    "annotations": {
      "readOnlyHint": false
    },
    "title": "Seal: Tamper Evidence for Persisted Memory"
  },
  {
    "name": "perseus_vault_tamper_scan",
    "description": "Verify every seal (entity + export) against the live content. Returns mismatches and journals each as a tamper event naming the target. Integrity != truth: seals detect unchanged-since-sealed violations, not truth at write time.",
    "inputSchema": {
      "type": "object",
      "properties": {},
      "required": []
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "seals_checked": {"type": "integer"},
        "ok": {"type": "boolean"},
        "tampered": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "seal_id": {"type": "string"},
              "target_id": {"type": "string"},
              "label": {"type": "string"},
              "scope": {"type": "string"},
              "expected_sha256": {"type": "string"},
              "actual_sha256": {"type": "string"},
              "detected_at_unix_ms": {"type": "integer"}
            }
          }
        }
      }
    },
    "annotations": {
      "readOnlyHint": true
    },
    "title": "Seal Verification: Tamper Evidence Scan"
  },
  {
    "name": "perseus_vault_provenance_projection",
    "description": "Evidence-vs-execution provenance projection over the typed link graph (#1064). mode=evidence walks supports/contradicts/invalidates/updates/authorized_by edges with classified kinds; mode=execution lists journal events referencing the entity plus blocked/denied authorized-action receipts (intent + failure receipt extended into the graph). Provenance != authorization != truth.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "seed_id": {
          "type": "string",
          "description": "Entity id to project from."
        },
        "mode": {
          "type": "string",
          "description": "evidence (typed edge graph) or execution (journal events + blocked action receipts). Default: evidence."
        },
        "depth": {
          "type": "integer",
          "description": "BFS depth bound for evidence mode (1-10, default 3)."
        }
      },
      "required": ["seed_id"]
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "seed_id": {"type": "string"},
        "mode": {"type": "string"},
        "depth": {"type": "integer"},
        "nodes": {"type": "array"},
        "edges": {"type": "array"},
        "blocked_actions": {"type": "array"}
      }
    },
    "annotations": {
      "readOnlyHint": true
    },
    "title": "Typed Provenance Projection: Evidence vs Execution"
  },
  {
    "name": "perseus_vault_param_lineage",
    "description": "Parameter-level lineage for high-risk tool arguments (Agent-Sentry pattern, #1064): record or query where a specific parameter value came from. Query validates every source — a dangling source_ref is returned with resolved=false, surfaced rather than trusted.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "action": {
          "type": "string",
          "description": "set (record a lineage row) or query (list rows)."
        },
        "entity_id": {"type": "string"},
        "param_path": {"type": "string"},
        "source_kind": {"type": "string"},
        "source_ref": {
          "type": "string",
          "description": "Optional producing entity id; validated at query time."
        },
        "workspace_hash": {"type": "string"},
        "agent_id": {"type": "string"}
      },
      "required": ["action", "entity_id"]
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "ok": {"type": "boolean"},
        "lineage_id": {"type": "string"},
        "entity_id": {"type": "string"},
        "lineage": {"type": "array"}
      }
    },
    "annotations": {
      "readOnlyHint": false
    },
    "title": "Parameter-Level Lineage for High-Risk Arguments"
  },
  {
    "name": "perseus_vault_typed_traversal",
    "description": "Intent-aware typed-relational traversal (#1065, MAGMA pattern): routes the query to one relation view (temporal / causal / entity / semantic) via a deterministic classifier, runs that view's traversal policy, and returns the explainable selected path (steps carry the relation they were taken over) plus rejected distractors with reasons — with token accounting for the context-budget discipline. LLM-free and reproducible: identical query → identical route.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "query": {"type": "string"},
        "limit": {
          "type": "integer",
          "description": "Selected-path size bound (1-50, default 10)."
        },
        "workspace_hash": {
          "type": "string",
          "description": "Required workspace partition; global scope must be explicitly authorized."
        },
        "requesting_agent_id": {
          "type": "string",
          "description": "Required transport-stamped requester identity."
        }
      },
      "required": ["query", "workspace_hash", "requesting_agent_id"]
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "query": {"type": "string"},
        "intent": {"type": "string"},
        "view": {"type": "string"},
        "path": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "entity_id": {"type": "string"},
              "relation": {"type": "string"},
              "source_chain_commitment": {"type": ["string", "null"]},
              "source_chain_status": {"type": "string"},
              "via": {"type": "string"},
              "source_sequence": {"type": ["integer", "null"]},
              "valid_from_unix_ms": {"type": ["integer", "null"]},
              "wire_rank": {"type": ["integer", "null"]}
            },
            "required": ["entity_id", "relation", "source_chain_status"]
          }
        },
        "rejected": {"type": "array"},
        "tokens_selected": {"type": "integer"},
        "tokens_rejected": {"type": "integer"},
        "run_id": {"type": "string"}
      }
    },
    "annotations": {
      "readOnlyHint": false
    },
    "title": "Intent-Aware Typed-Relational Traversal"
  },
  {
    "name": "perseus_vault_traversal_ablation",
    "description": "Per-relation-view ablation report over recorded typed traversals (#1065): mean selected/rejected tokens and distractor ratio per view — auditable evidence for whether each relation view earns its token cost.",
    "inputSchema": {
      "type": "object",
      "properties": {},
      "required": []
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "views": {"type": "array"}
      }
    },
    "annotations": {
      "readOnlyHint": true
    },
    "title": "Typed-Traversal Ablation Report"
  },
  {
    "name": "perseus_vault_model_inheritance",
    "description": "Model-upgrade inheritance receipt (#1066, identity/vessel split): record a source-state snapshot for a subject identity and the replacement model identity, run the compatibility report, and (after policy-gated approval) stamp the approved handoff as a queryable inheritance receipt in the provenance graph. `depart` is a governed transition that preserves a tombstone; `replay` samples representative memories as hash-only digests (no content leak). Memory survives the model — now the handoff is auditable.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "action": {
          "type": "string",
          "enum": ["record", "approve", "query", "depart", "replay"]
        },
        "subject_id": {"type": "string"},
        "old_model": {"type": "string"},
        "new_model": {"type": "string"},
        "reason": {"type": "string"},
        "approver": {"type": "string"},
        "sample_count": {"type": "integer"}
      },
      "required": ["action", "subject_id"]
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "receipt": {"type": "object"},
        "replay": {"type": "object"},
        "ok": {"type": "boolean"}
      }
    },
    "annotations": {
      "readOnlyHint": false
    },
    "title": "Model-Upgrade Inheritance Receipt"
  },
  {
    "name": "perseus_vault_vault_export",
    "description": "Export all non-archived entities to .md files with YAML frontmatter in a vault directory. Files are human-readable, git-trackable, and Obsidian-compatible. Use this for backup, transfer between workspaces, or offline review.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "vault_dir": {
          "type": "string",
          "default": "~/.perseus-vault/vault",
          "description": "Directory path to write .md files. Created if it doesn't exist. Use ~ for home directory."
        },
        "requesting_agent_id": {
          "type": "string",
          "description": "Transport-stamped requester identity used for visibility enforcement."
        }
      },
      "required": []
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "files_created": {
          "type": "integer",
          "description": "Number of new .md files created"
        },
        "files_updated": {
          "type": "integer",
          "description": "Number of existing .md files updated"
        },
        "errors": {
          "type": "array",
          "items": {
            "type": "string"
          },
          "description": "Any errors encountered during export"
        },
        "vault_dir": {
          "type": "string",
          "description": "Absolute path to the vault directory"
        },
        "completed_at_unix_ms": {
          "type": "integer",
          "description": "Completion timestamp"
        }
      }
    },
    "annotations": {
      "destructiveHint": true
    },
    "title": "Export Vault to Files"
  },
  {
    "name": "perseus_vault_derived_export",
    "description": "Compile durable knowledge into a deterministic, provenance-rich Markdown surface. The export is derived and read-only; SQLite remains the source of truth.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "output_path": {
          "type": "string",
          "description": "Markdown file path to write."
        },
        "workspace_hash": {
          "type": "string",
          "description": "Optional exact workspace scope."
        },
        "requesting_agent_id": {
          "type": "string",
          "description": "Transport-stamped requester identity used for visibility enforcement."
        }
      },
      "required": [
        "output_path"
      ]
    }
  },
  {
    "name": "perseus_vault_markdown_import",
    "description": "Import one Markdown file as explicitly non-authoritative, provenance-labeled draft evidence. Duplicate source content is idempotently detected.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "path": {"type": "string", "description": "Markdown file path to import."},
        "workspace_hash": {"type": "string"},
        "source_system": {"type": "string", "description": "Provenance source label; defaults to markdown."}
      },
      "required": ["path"]
    }
  },
  {
    "name": "perseus_vault_structured_index_anchor",
    "description": "Represent an upstream structured-index record as a refetchable anchor, or import it explicitly as low-confidence non-authoritative draft evidence.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "index_type": {"type": "string", "description": "Structured index kind, e.g. ide_symbol or domain_fact_map."},
        "index_uri": {"type": "string", "description": "Stable index locator for later refetch."},
        "record_id": {"type": "string", "description": "Stable record identity inside the index."},
        "mode": {"type": "string", "enum": ["reference", "import"], "default": "reference"},
        "content": {"type": "string", "description": "Required only for mode=import."},
        "workspace_hash": {"type": "string"},
        "source_system": {"type": "string"},
        "observed_at_unix_ms": {"type": "integer"},
        "revision": {"type": "string", "description": "Optional upstream revision/ETag for refetch verification."}
      },
      "required": ["index_type", "index_uri", "record_id"]
    }
  },
  {
    "name": "perseus_vault_vault_import",
    "description": "Import .md files from a vault directory into the database. Reads YAML frontmatter for metadata and markdown body for content. Idempotent — re-running on the same vault won't duplicate entities. Pass shadow_workspace to run a shadow import: every entity is forced into that scratch workspace and the live bank is never touched, so you can compare recall before cutting over (see perseus_vault_shadow_compare / _promote / _rollback). Pair with perseus_vault_vault_export for transfer.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "vault_dir": {
          "type": "string",
          "default": "~/.perseus-vault/vault",
          "description": "Directory path to read .md files from. Use ~ for home directory."
        },
        "shadow_workspace": {
          "type": "string",
          "description": "#951 shadow import: when set, every imported entity is forced into this workspace regardless of frontmatter. Zero writes to the live bank; rerunnable with zero new identities."
        }
      },
      "required": []
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "files_created": {
          "type": "integer",
          "description": "Number of new entities created from files"
        },
        "files_updated": {
          "type": "integer",
          "description": "Number of existing entities updated"
        },
        "errors": {
          "type": "array",
          "items": {
            "type": "string"
          },
          "description": "Any errors encountered during import"
        },
        "vault_dir": {
          "type": "string",
          "description": "Absolute path of the vault directory read"
        },
        "completed_at_unix_ms": {
          "type": "integer",
          "description": "Completion timestamp"
        }
      }
    },
    "annotations": {
      "destructiveHint": true
    },
    "title": "Import Vault from Files"
  },
  {
    "name": "perseus_vault_shadow_compare",
    "description": "#951: recall comparison between the live workspace and a shadow workspace over a fixed query set. Deterministic (Fts5 mode), side-effect-free, machine-readable — the gate for deciding whether a shadow import clears cutover.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "queries": {
          "type": "array",
          "items": { "type": "string" },
          "description": "Fixed query set to run in both workspaces (1..=500)."
        },
        "live_workspace": {
          "type": "string",
          "description": "Live workspace to compare against; omit for the unscoped bank."
        },
        "shadow_workspace": {
          "type": "string",
          "description": "The scratch workspace holding the shadow import."
        },
        "limit": {
          "type": "integer",
          "default": 5,
          "description": "Recall limit per query (1..=100)."
        }
      },
      "required": ["queries", "shadow_workspace"]
    },
    "annotations": {
      "readOnlyHint": true
    },
    "title": "Compare Live vs Shadow Recall"
  },
  {
    "name": "perseus_vault_shadow_promote",
    "description": "#951: promote — move every non-archived entity from the shadow workspace into the target workspace in ONE atomic operation, journaling the moved ids so perseus_vault_shadow_rollback can undo the cutover in one operation. dry_run previews the count without writing.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "shadow_workspace": {
          "type": "string",
          "description": "The scratch workspace to promote from."
        },
        "target_workspace": {
          "type": "string",
          "default": "",
          "description": "Target workspace (default: the unscoped live bank)."
        },
        "dry_run": {
          "type": "boolean",
          "default": false,
          "description": "Preview the move (count only — nothing written, no journal)."
        }
      },
      "required": ["shadow_workspace"]
    },
    "annotations": {
      "destructiveHint": true
    },
    "title": "Promote Shadow Import to Live"
  },
  {
    "name": "perseus_vault_shadow_rollback",
    "description": "#951: rollback — one operation returns every promoted id to its pre-promote workspace, using the shadow_promote_last journal. dry_run previews the journal without writing.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "dry_run": {
          "type": "boolean",
          "default": false,
          "description": "Preview the journal (nothing written, journal kept)."
        }
      },
      "required": []
    },
    "annotations": {
      "destructiveHint": true
    },
    "title": "Roll Back Shadow Promote"
  },
  {
    "name": "perseus_vault_decay",
    "description": "Recalculate Ebbinghaus decay scores for all entities based on time since last access. Auto-archives entities that have fully decayed (score < 0.05). Run periodically to keep memory fresh — decayed entities surface less often in recall results.",
    "inputSchema": {
      "type": "object",
      "properties": {}
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "entities_checked": {
          "type": "integer",
          "description": "Total entities evaluated"
        },
        "entities_updated": {
          "type": "integer",
          "description": "Entities whose stored decay score was actually rewritten (rows whose recomputed score changed). A steady-state tick reports ~0: unchanged rows are evaluated but not written."
        },
        "auto_archived": {
          "type": "integer",
          "description": "Entities auto-archived because decay fell below 0.05"
        },
        "completed_at_unix_ms": {
          "type": "integer",
          "description": "Completion timestamp"
        }
      }
    },
    "annotations": {
      "destructiveHint": true
    },
    "title": "Recalculate Decay Scores"
  },
  {
    "name": "perseus_vault_reindex",
    "description": "Rebuild the FTS5 search index from the entities table. Repairs index drift — e.g. after a direct SQLite write, an interrupted archive, or a legacy database written before the atomic prune/forget fixes — so archived entities stop surfacing in recall/search. Returns the number of entities reindexed.",
    "inputSchema": {
      "type": "object",
      "properties": {}
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "reindexed": {
          "type": "integer",
          "description": "Number of non-archived entities indexed into FTS5"
        }
      }
    },
    "annotations": {
      "destructiveHint": true
    },
    "title": "Rebuild Search Index"
  },
  {
    "name": "perseus_vault_workspace_list",
    "description": "List all distinct entity categories present in the database. Use this to discover what knowledge domains exist before querying with perseus_vault_recall or perseus_vault_context.",
    "inputSchema": {
      "type": "object",
      "properties": {}
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "categories": {
          "type": "array",
          "items": {
            "type": "string"
          },
          "description": "All distinct categories in the database"
        },
        "total": {
          "type": "integer",
          "description": "Number of categories"
        }
      }
    },
    "annotations": {
      "readOnlyHint": true
    },
    "title": "List Workspace Categories"
  },
  {
    "name": "perseus_vault_recall_when",
    "description": "Search entities whose recall_when triggers match a given context. Use this for proactive just-in-time memory injection — before writing code, before plans, at session start. Pass the current task description as context and get back memories that declared they should be recalled in similar situations.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "context": {
          "type": "string",
          "description": "The current task or context description to match against recall_when triggers"
        },
        "limit": {
          "type": "integer",
          "description": "Maximum entities to return (default 10, max 100)",
          "default": 10
        },
        "workspace_hash": {
          "type": "string",
          "description": "Workspace scope filter (v1.2.0). When set, only entities with a matching workspace_hash can fire. Compatibility mode permits omission for legacy unscoped trigger reads when strict deployment mode is off; strict mode requires a non-empty workspace_hash and an active binding for the transport requester."
        },
        "requesting_agent_id": {
          "type": "string",
          "description": "Transport-stamped requester identity; caller-supplied values are overwritten before trigger matching and serialization."
        },
        "session_id": {
          "type": "string",
          "description": "Session id for preload usage telemetry (#875): served entities are attributed to this session for precision/recall resolution. Omit or leave empty when unknown."
        },
        "include_outcome": {
          "type": "boolean",
          "default": false,
          "description": "#1186: include a bounded answer_outcome for complete results as well as empty, partial, degraded, abstained, or unavailable trigger recall."
        }
      },
      "required": [
        "context"
      ]
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "items": {
          "type": "array",
          "items": {
            "type": "object"
          }
        },
        "total": {
          "type": "integer"
        },
        "context": {
          "type": "string"
        },
        "answer_outcome": {
          "type": "object",
          "additionalProperties": false,
          "description": "#1186: bounded answer-facing status; no query, evidence body, or backend error text.",
          "properties": {
            "schema_version": {"type": "string", "const": "perseus-vault-answer-outcome/v1"},
            "status": {"type": "string", "enum": ["complete", "partial", "degraded", "abstained", "unavailable"]},
            "recall_status": {"type": "string", "enum": ["fresh", "partial", "timeout", "unavailable", "empty", "stale"]},
            "reason": {"type": "string", "minLength": 1, "maxLength": 256},
            "reason_codes": {"type": "array", "minItems": 1, "maxItems": 16, "items": {"type": "string", "maxLength": 256}},
            "abstained": {"type": "boolean"},
            "answerable": {"type": "boolean"},
            "fallback": {"type": "object", "additionalProperties": false, "properties": {"mode": {"type": "string", "enum": ["abstain", "canonical_retrieval"]}, "reason": {"type": "string", "minLength": 1, "maxLength": 256}}, "required": ["mode", "reason"]},
            "exclusions": {"type": "array", "maxItems": 256, "items": {"type": "object", "additionalProperties": false, "properties": {"reason": {"type": "string", "maxLength": 256}, "count": {"type": "integer", "minimum": 1}}, "required": ["reason", "count"]}},
            "conflicts": {"type": "array", "maxItems": 128, "items": {"type": "object", "additionalProperties": false, "properties": {"reason": {"type": "string", "maxLength": 256}, "reference_count": {"type": "integer", "minimum": 0}, "references_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"}}, "required": ["reason", "reference_count", "references_sha256"]}}
          },
          "required": ["schema_version", "status", "recall_status", "reason", "reason_codes", "abstained", "answerable"]
        }
      }
    },
    "annotations": {
      "readOnlyHint": true
    },
    "title": "Proactive Recall by Context"
  },
  {
    "name": "perseus_vault_cohere",
    "description": "Run an autonomous coherence grooming pass over the memory. Promotes buffer entities to working layer, applies decay, auto-links related entities, and archives stale ones below the decay threshold. Use dry_run=true to preview without making changes.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "dry_run": {
          "type": "boolean",
          "description": "If true, count what would be done without making changes",
          "default": false
        },
        "max_links": {
          "type": "integer",
          "description": "Maximum auto-links to create (default 20, max 100)",
          "default": 20
        },
        "promote_threshold": {
          "type": "integer",
          "description": "Retrieval count threshold for buffer to working promotion (default 3)",
          "default": 3
        },
        "archive_threshold": {
          "type": "number",
          "description": "Decay score below which entities are auto-archived (default 0.05)",
          "default": 0.05
        },
        "cross_scope_promote": {
          "type": "boolean",
          "description": "#486: also run cross-scope promotion — a fact independently observed in >= cross_scope_k distinct workspaces is promoted to one global-scope entity with promoted_from links back to the per-scope evidence. Off by default; re-runs are idempotent (the global scope's dedup absorbs them); undo by forgetting the promoted entity.",
          "default": false
        },
        "cross_scope_k": {
          "type": "integer",
          "description": "Minimum distinct workspaces before a recurring fact is promoted (default 3, minimum 2)",
          "default": 3
        },
        "cross_scope_similarity": {
          "type": "number",
          "description": "Trigram similarity treating two bodies as the same fact across scopes (default 0.7, matching write-time dedup)",
          "default": 0.7
        }
      }
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "promoted": {
          "type": "integer",
          "description": "Number of entities promoted from buffer to working"
        },
        "cross_scope_clusters": {
          "type": "integer",
          "description": "#486: clusters found spanning >= cross_scope_k workspaces (0 unless cross_scope_promote)"
        },
        "cross_scope_promoted": {
          "type": "integer",
          "description": "#486: new global-scope entities created by cross-scope promotion"
        },
        "cross_scope_skipped_existing": {
          "type": "integer",
          "description": "#486: qualifying clusters already represented at the global scope (idempotent re-run)"
        },
        "decayed": {
          "type": "integer",
          "description": "Number of entities whose decay score was reduced"
        },
        "linked": {
          "type": "integer",
          "description": "Number of auto-links created"
        },
        "archived": {
          "type": "integer",
          "description": "Number of entities archived due to low decay"
        },
        "entities_examined": {
          "type": "integer",
          "description": "Total non-archived entities examined"
        },
        "dry_run": {
          "type": "boolean"
        },
        "completed_at_unix_ms": {
          "type": "integer"
        }
      }
    },
    "annotations": {
      "destructiveHint": true
    },
    "title": "Run Coherence Grooming"
  },
  {
    "name": "perseus_vault_share",
    "description": "Share an entity to another workspace. Copies the entity (by category + key) from its current workspace into the target workspace, preserving content and metadata while generating a new ID. The original entity is unchanged. Use this for controlled cross-workspace knowledge transfer.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "category": {
          "type": "string",
          "description": "Entity category to share"
        },
        "key": {
          "type": "string",
          "description": "Entity key to share"
        },
        "to_workspace": {
          "type": "string",
          "description": "Target workspace hash to copy the entity into"
        }
      },
      "required": [
        "category",
        "key",
        "to_workspace"
      ]
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "shared_id": {
          "type": "string",
          "description": "ID of the new shared copy"
        },
        "action": {
          "type": "string",
          "description": "'created' or 'updated'"
        },
        "from_workspace": {
          "type": "string",
          "description": "Source workspace the entity was copied from"
        },
        "to_workspace": {
          "type": "string",
          "description": "Target workspace the entity was copied to"
        }
      }
    },
    "annotations": {
      "destructiveHint": true
    },
    "title": "Share Entity to Workspace"
  },

  {
    "name": "perseus_vault_correct",
    "description": "Capture a user correction to the agent. Stores what went wrong, what the user said, and the lesson learned — as both a 'correction' entity and a journal entry. Use this every time the user corrects your approach. Enables the self-improving feedback loop: the agent learns from mistakes across sessions.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "wrong_approach": {
          "type": "string",
          "description": "What the agent did that was wrong (the mistaken approach)"
        },
        "user_correction": {
          "type": "string",
          "description": "What the user said to correct the agent (the right way)"
        },
        "task_context": {
          "type": "string",
          "description": "What task was being attempted when the correction occurred"
        },
        "evidence": {
          "type": "object",
          "description": "Write-time audit envelope for the correction's source evidence. capture_mode distinguishes snapshot, hash_only, pointer_only, not_requested, capture_failed, and legacy_unknown; a missing value is never interpreted implicitly.",
          "properties": {
            "capture_mode": { "type": "string", "enum": ["snapshot", "hash_only", "pointer_only", "not_requested", "capture_failed", "legacy_unknown"] },
            "resolved_value": { "description": "Resolved source value retained at write time when capture_mode=snapshot" },
            "content_sha256": { "type": "string", "description": "64-hex SHA-256 of the resolved value or source bytes" },
            "source_system": { "type": "string" },
            "source_ref": { "type": "string" },
            "captured_at_unix_ms": { "type": "integer" },
            "replayable": { "type": "boolean" }
          },
          "required": ["capture_mode", "captured_at_unix_ms", "replayable"]
        },
        "session_id": {
          "type": "string",
          "default": "",
          "description": "Session identifier for traceability"
        },
        "tags": {
          "type": "array",
          "items": {
            "type": "string"
          },
          "description": "Tags for categorization"
        },
        "category": {
          "type": "string",
          "default": "correction",
          "description": "Entity category (default: 'correction')"
        },
        "visibility": {
          "type": "string",
          "default": "workspace",
          "description": "Visibility: 'private', 'workspace', or 'public'"
        },
        "valid_from_unix_ms": {
          "type": "integer",
          "description": "Application-time period start (#363): when the corrected fact was actually true in the world. Set in the past for retroactive corrections. Default: transaction time."
        },
        "valid_to_unix_ms": {
          "type": "integer",
          "description": "Application-time period end (#363, exclusive). Omit for 'still true'."
        },
        "workspace_hash": {
          "type": "string",
          "default": "",
          "description": "Workspace scope for the rejection tombstone (#849). Empty means global."
        },
        "agent_id": {
          "type": "string",
          "default": "",
          "description": "Agent that authored the correction (stamped on the tombstone)."
        },
        "requesting_agent_id": {
          "type": "string",
          "default": "",
          "description": "#855 host identity (stamped by the MCP transport). When present, it is authoritative: the correction entity, journal event, and tombstone attribute the host, not any model-supplied agent_id."
        }
      },
      "required": [
        "wrong_approach",
        "user_correction",
        "task_context"
      ]
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "entity_id": {
          "type": "string",
          "description": "Created correction entity ID"
        },
        "journal_id": {
          "type": "string",
          "description": "Created journal entry ID"
        },
        "agent_id": {
          "type": "string",
          "description": "#855 agent attribution persisted on the entity and journal event (host identity when the transport stamped one)"
        },
        "workspace_hash": {
          "type": "string",
          "description": "#855 workspace scope persisted on the entity and journal event. Empty = global/legacy."
        },
        "category": {
          "type": "string"
        },
        "key": {
          "type": "string"
        },
        "created_at_unix_ms": {
          "type": "integer"
        }
      }
    },
    "annotations": {
      "destructiveHint": true
    },
    "title": "Capture Agent Correction"
  },
  {
    "name": "perseus_vault_synthesize",
    "description": "LLM-driven session synthesis. Reviews a session transcript and extracts structured lessons: what worked (success), what failed (failure), what was corrected (correction), what was abandoned (dead_end), and key decisions made (decision). Each lesson becomes an entity linked to a synthesis journal entry. Requires --llm-endpoint to be configured. This is the Perplexity-Brain-style overnight synthesis loop for agent self-improvement.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "session_content": {
          "type": "string",
          "description": "Full session transcript to synthesize lessons from"
        },
        "session_id": {
          "type": "string",
          "default": "",
          "description": "Session identifier for traceability"
        },
        "tags": {
          "type": "array",
          "items": {
            "type": "string"
          },
          "description": "Tags applied to all synthesized entities"
        },
        "visibility": {
          "type": "string",
          "default": "workspace",
          "description": "Visibility for synthesized entities"
        }
      },
      "required": [
        "session_content"
      ]
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "lessons": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "lesson_type": {
                "type": "string"
              },
              "summary": {
                "type": "string"
              },
              "evidence": {
                "type": "string"
              },
              "confidence": {
                "type": "number"
              }
            }
          },
          "description": "Extracted lessons with type, summary, evidence, and confidence"
        },
        "entities_created": {
          "type": "integer",
          "description": "Number of lesson entities created"
        },
        "journal_id": {
          "type": "string"
        },
        "dry_run": {
          "type": "boolean"
        },
        "completed_at_unix_ms": {
          "type": "integer"
        }
      }
    },
    "annotations": {
      "destructiveHint": true
    },
    "title": "Synthesize Session Lessons"
  },
  {
    "name": "perseus_vault_bench",
    "description": "Record a performance benchmark data point. Tracks task metrics (turns taken, tokens used, success) alongside whether memory recall was used — enabling measurement of Perseus Vault's impact on agent performance. Aggregate with perseus_vault_recall to analyze trends.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "task_description": {
          "type": "string",
          "description": "Description of the task being measured"
        },
        "turns_taken": {
          "type": "integer",
          "description": "Number of conversation turns the task took"
        },
        "tokens_used": {
          "type": "integer",
          "description": "Total tokens consumed by the task"
        },
        "memory_recall_used": {
          "type": "boolean",
          "description": "Whether memory recall (perseus_vault_recall) was used during this task"
        },
        "recall_count": {
          "type": "integer",
          "default": 0,
          "description": "How many times memory was recalled during this task"
        },
        "task_success": {
          "type": "boolean",
          "default": false,
          "description": "Whether the task completed successfully"
        },
        "session_id": {
          "type": "string",
          "default": "",
          "description": "Session identifier for traceability"
        },
        "tags": {
          "type": "array",
          "items": {
            "type": "string"
          },
          "description": "Tags for categorization"
        }
      },
      "required": [
        "task_description",
        "turns_taken",
        "tokens_used",
        "memory_recall_used"
      ]
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "entity_id": {
          "type": "string",
          "description": "Created benchmark entity ID"
        },
        "created_at_unix_ms": {
          "type": "integer"
        }
      }
    },
    "annotations": {
      "destructiveHint": true
    },
    "title": "Record Benchmark"
  },
  {
    "name": "perseus_vault_autocohere",
    "description": "Run a full atomic grooming pass. When capture_text is supplied, capture runs first and must succeed before cohere, decay, compact, consolidation, or retention can compress source context. Returns a summary report. Use dry_run=true to preview without writing.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "dry_run": {
          "type": "boolean",
          "description": "If true, preview changes without writing",
          "default": false
        },
        "capture_text": {
          "type": "string",
          "description": "Optional raw transcript/insight payload persisted before every compaction-like stage. Capture failure aborts the pass."
        },
        "capture_workspace_hash": {
          "type": "string",
          "description": "Workspace scope for pre-compaction captured facts"
        },
        "capture_agent_id": {
          "type": "string",
          "description": "Agent attribution for pre-compaction captured facts"
        },
        "capture_max_entities": {
          "type": "integer",
          "description": "Maximum durable notes extracted from capture_text (1-20)"
        },
        "workspace_hash": {
          "type": "string",
          "description": "#854 workspace scope for the consolidation step. When set, only that workspace's entities are consolidated and the observations inherit the scope. Omit for the whole-vault pass."
        },
        "global": {
          "type": "boolean",
          "default": false,
          "description": "#854 explicit whole-vault consolidation mode (capability-gated with a host identity). Mutually exclusive with workspace_hash."
        },
        "requesting_agent_id": {
          "type": "string",
          "default": "",
          "description": "Host identity stamped by the MCP transport. Used for global-mode authorization and consolidation author attribution."
        }
      }
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "precompact_capture": {
          "type": "object",
          "description": "Capture barrier report. stage=completed means capture persisted before all lifecycle compression stages; stage=skipped means no capture_text was supplied."
        },
        "promoted_entities": {
          "type": "integer",
          "description": "Entities promoted during cohere"
        },
        "links_created": {
          "type": "integer",
          "description": "Auto-links created during cohere"
        },
        "archived_entities": {
          "type": "integer",
          "description": "Entities archived (cohere + compact)"
        },
        "decay_updates": {
          "type": "integer",
          "description": "Entities whose decay score was updated"
        },
        "compact_archived_count": {
          "type": "integer",
          "description": "Entities archived during compact step"
        },
        "history_rows_evicted": {
          "type": "integer",
          "description": "entity_history rows evicted by the retention policy (#398; 0 while no PERSEUS_VAULT_HISTORY_* knob is set)"
        },
        "history_bytes_evicted": {
          "type": "integer",
          "description": "Stored history body bytes evicted (#398)"
        },
        "history_tombstones_written": {
          "type": "integer",
          "description": "Compaction tombstones written (#398)"
        },
        "db_size_delta_bytes": {
          "type": "integer",
          "description": "Change in SQLite file size in bytes"
        },
        "decay_auto_archived": {
          "type": "integer",
          "description": "Entities decay auto-archived during this pass (#490; 0 under dry_run)"
        },
        "observations_created": {
          "type": "integer",
          "description": "Observations created by the consolidation step"
        },
        "consolidate_sources_archived": {
          "type": "integer",
          "description": "Sources archived by the consolidation step (verified/importance-floored exempt)"
        },
        "workspace_hash": {
          "type": ["string", "null"],
          "description": "#854 effective consolidation scope: the workspace the consolidate step operated in (null = whole-vault pass)"
        },
        "global": {
          "type": "boolean",
          "description": "#854 true when the consolidation step deliberately crossed all workspaces"
        },
        "dry_run": {
          "type": "boolean"
        }
      }
    },
    "annotations": {
      "destructiveHint": true
    },
    "title": "Atomic Coherence Pass"
  },
  {
    "name": "perseus_vault_supersede",
    "description": "Create a 'supersedes' relationship from a new fact to an old one, setting the old entity's status to 'deprecated'. Use this when a newer entity makes an older one obsolete.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "from_category": {
          "type": "string",
          "description": "Category of the OLD entity being superseded"
        },
        "from_key": {
          "type": "string",
          "description": "Key of the OLD entity being superseded"
        },
        "to_category": {
          "type": "string",
          "description": "Category of the NEW entity that supersedes"
        },
        "to_key": {
          "type": "string",
          "description": "Key of the NEW entity that supersedes"
        },
        "reason": {
          "type": "string",
          "description": "Reason for superseding (recorded in archive_reason)",
          "default": ""
        },
        "relationship": {
          "type": "string",
          "description": "Link relationship type (default: 'supersedes')",
          "default": "supersedes"
        },
        "valid_to_unix_ms": {
          "type": "integer",
          "description": "When the OLD fact stopped being true in the world (#363, unix ms). Defaults to transaction time (now). Closes the old entity's application-time period so perseus_vault_valid_at stops returning it from that instant on. Must be after the fact's valid_from, and may only TIGHTEN an already-closed period (a fact that ended cannot be retroactively extended); violations are rejected before any mutation."
        }
      },
      "required": [
        "from_category",
        "from_key",
        "to_category",
        "to_key"
      ]
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "from_entity_id": {
          "type": "string",
          "description": "ID of the old (superseded) entity"
        },
        "from_entity_category": {
          "type": "string"
        },
        "from_entity_key": {
          "type": "string"
        },
        "from_valid_to_unix_ms": {
          "type": "integer",
          "description": "The instant the old fact's validity was closed at (#363)"
        },
        "to_entity_id": {
          "type": "string",
          "description": "ID of the new (superseding) entity"
        },
        "to_entity_category": {
          "type": "string"
        },
        "to_entity_key": {
          "type": "string"
        },
        "relationship": {
          "type": "string"
        },
        "status_updated": {
          "type": "string",
          "description": "New status of the old entity (always 'deprecated')"
        }
      }
    },
    "annotations": {
      "destructiveHint": true
    },
    "title": "Supersede Entity"
  },
  {
    "name": "perseus_vault_consistency_audit",
    "description": "Read-only court-of-record self-audit (#940): scans the category for contradiction pairs, recommends a deterministic winner per pair (importance → source-authority → recency → id), surfaces pairs with an existing active ruling, lists supersession lag (deprecated entities without a live successor), and reports the pending keystone suggestion count. NEVER mutates: run before ruling, decide with perseus_vault_audit_ruling.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "category": {
          "type": "string",
          "description": "Category to audit (default: facts)",
          "default": "facts"
        },
        "limit": {
          "type": "integer",
          "description": "Max contradiction pairs to scan (clamped 1-200, default 50)",
          "default": 50
        }
      }
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "findings": {
          "type": "array",
          "description": "Per-pair: recommendation {winner_id, winner_key, decided_by} or already_ruled {ruling_id, winner_id}"
        },
        "supersession_lag": {
          "type": "array"
        },
        "keystone_pending": {
          "type": "integer"
        },
        "read_only": {
          "type": "boolean",
          "description": "Always true — the audit never mutates"
        }
      }
    },
    "title": "Consistency Audit (court of record)"
  },
  {
    "name": "perseus_vault_audit_ruling",
    "description": "Idempotent operator ruling over a consistency finding (#940): accept compiles the recommended winner into the supersede guard (winner→loser link, loser valid-period closed, status deprecated); override compiles an explicit winner; reverse reopens a ruled pair for re-litigation (the compiled guard remains). Rulings are recorded + journaled (court_ruling_set/court_ruling_reversed); an active ruling with a different winner is refused until reversed.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "action": {
          "type": "string",
          "enum": ["accept", "override", "reverse"],
          "description": "accept = compile the ladder-recommended winner; override = compile an explicit winner; reverse = reopen a ruled pair"
        },
        "category": {
          "type": "string",
          "description": "Category of entity_a/entity_b (default: facts)",
          "default": "facts"
        },
        "entity_a_key": {
          "type": "string",
          "description": "Key of the first contested entity (accept/override)"
        },
        "entity_b_key": {
          "type": "string",
          "description": "Key of the second contested entity (accept/override)"
        },
        "winner_category": {
          "type": "string",
          "description": "Override only: category of the explicit winner"
        },
        "winner_key": {
          "type": "string",
          "description": "Override only: key of the explicit winner"
        },
        "ruling_id": {
          "type": "string",
          "description": "Reverse only: id of the active ruling to reopen"
        },
        "rationale": {
          "type": "string",
          "description": "Optional ruling rationale (recorded verbatim)",
          "default": ""
        },
        "decided_by": {
          "type": "string",
          "description": "Who decided (default: operator)",
          "default": "operator"
        }
      },
      "required": ["action"]
    },
    "annotations": {
      "destructiveHint": false,
      "readOnlyHint": false
    },
    "title": "Audit Ruling (court of record)"
  },
  {
    "name": "perseus_vault_maintenance",
    "description": "Database maintenance operations: deduplicate entities with identical (category, key), detect orphan journal entries and links, vacuum (reclaim disk space), reindex FTS5, and enforce the entity_history retention policy (#398 — no-op unless PERSEUS_VAULT_HISTORY_* env knobs are set). Set dry_run=true to preview. Use 'all' to run everything.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "dedup": {
          "type": "boolean",
          "description": "Find duplicate (category, key) entities and archive the oldest",
          "default": false
        },
        "orphans": {
          "type": "boolean",
          "description": "Detect journal entries and links pointing to non-existent entities",
          "default": false
        },
        "vacuum": {
          "type": "boolean",
          "description": "Run SQLite VACUUM to reclaim disk space",
          "default": false
        },
        "reindex": {
          "type": "boolean",
          "description": "Rebuild the FTS5 search index from entities table",
          "default": false
        },
        "history": {
          "type": "boolean",
          "description": "Enforce the entity_history retention policy from PERSEUS_VAULT_HISTORY_* env knobs (#398; no-op while none are set)",
          "default": false
        },
        "all": {
          "type": "boolean",
          "description": "Run all maintenance operations (dedup, orphans, vacuum, reindex, history retention)",
          "default": false
        },
        "dry_run": {
          "type": "boolean",
          "description": "If true, preview changes without writing",
          "default": false
        }
      }
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "dedup_archived": {
          "type": "integer",
          "description": "Number of duplicate entities archived"
        },
        "orphan_journal_entries_found": {
          "type": "integer",
          "description": "Orphan journal entries detected"
        },
        "orphan_links_found": {
          "type": "integer",
          "description": "Orphan links detected"
        },
        "vacuum_reclaimed_bytes": {
          "type": "integer",
          "description": "Disk space reclaimed by VACUUM"
        },
        "reindex_rows_affected": {
          "type": "integer",
          "description": "Rows reindexed into FTS5"
        },
        "history_rows_evicted": {
          "type": "integer",
          "description": "entity_history rows evicted by the retention policy (#398)"
        },
        "history_bytes_evicted": {
          "type": "integer",
          "description": "Stored history body bytes evicted (#398)"
        },
        "history_tombstones_written": {
          "type": "integer",
          "description": "Compaction tombstones written for evicted runs (#398)"
        },
        "dry_run": {
          "type": "boolean"
        },
        "errors": {
          "type": "array",
          "items": {
            "type": "string"
          },
          "description": "Errors encountered during maintenance"
        }
      }
    },
    "annotations": {
      "destructiveHint": true
    },
    "title": "Run Database Maintenance"
  },
  {
    "name": "perseus_vault_communities",
    "description": "GraphRAG community detection: partition the entity link graph (built via perseus_vault_link) into communities using deterministic label propagation or greedy modularity ('louvain'). Persists the result with an extractive summary per community; community ids are derived from the member set, so re-detection after membership changes yields new ids. Local-first — no LLM or network required.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "workspace_hash": {
          "type": "string",
          "default": "",
          "description": "Workspace scope for the graph. Empty = global/unscoped entities."
        },
        "algorithm": {
          "type": "string",
          "default": "label_prop",
          "enum": ["label_prop", "louvain"],
          "description": "Detection algorithm: 'label_prop' (deterministic label propagation, default) or 'louvain' (greedy one-level modularity optimization)."
        },
        "min_size": {
          "type": "integer",
          "default": 2,
          "description": "Minimum member count for a community to be kept (minimum 2 — isolated entities never form communities)."
        }
      },
      "required": []
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "workspace_hash": { "type": "string" },
        "algorithm": { "type": "string" },
        "node_count": { "type": "integer", "description": "Entities considered as graph nodes" },
        "edge_count": { "type": "integer", "description": "Undirected edges in the graph" },
        "modularity": { "type": "number", "description": "Newman modularity of the detected partition" },
        "communities": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "id": { "type": "string", "description": "Community id ('com-' + member-set digest)" },
              "size": { "type": "integer" },
              "member_ids": { "type": "array", "items": { "type": "string" } },
              "summary": { "type": "string", "description": "Extractive summary (top members by in-community degree), capped in size" }
            }
          }
        },
        "stale_summaries_archived": { "type": "integer", "description": "Stale community_summary entities archived because membership changed" },
        "generated_at_unix_ms": { "type": "integer" }
      }
    },
    "annotations": {
      "idempotentHint": true
    },
    "title": "Detect Link-Graph Communities"
  },
  {
    "name": "perseus_vault_community_summary",
    "description": "Return (and materialize) the summary of one detected community. Default is the extractive summary (top representative members); set use_llm=true for an optional LLM polish that degrades back to extractive when no LLM endpoint is configured. The summary is stored as a 'community_summary' entity carrying evidence_for links to its members, and cached while membership is unchanged.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "community_id": {
          "type": "string",
          "description": "Community id from perseus_vault_communities, e.g. 'com-1a2b3c4d5e6f7a8b'"
        },
        "use_llm": {
          "type": "boolean",
          "default": false,
          "description": "Polish the summary with the configured LLM (--llm-endpoint). Never required: falls back to the extractive summary on error or when disabled."
        },
        "refresh": {
          "type": "boolean",
          "default": false,
          "description": "Force regeneration even when a cached summary entity exists."
        }
      },
      "required": ["community_id"]
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "community_id": { "type": "string" },
        "summary": { "type": "string" },
        "summary_entity_id": { "type": "string", "description": "entities.id of the materialized community_summary entity" },
        "member_count": { "type": "integer" },
        "cached": { "type": "boolean", "description": "True when an existing summary entity was reused (membership unchanged)" },
        "llm_used": { "type": "boolean" }
      }
    },
    "annotations": {
      "idempotentHint": true
    },
    "title": "Get Community Summary"
  },
  {
    "name": "perseus_vault_global_recall",
    "description": "GraphRAG global search: answer a broad 'what does the vault know about X, holistically' query by scoring it against community summaries first (breadth), then drilling into the best communities' member entities (depth). Cites entities across multiple communities instead of returning only the single nearest cluster like flat recall. Detects communities automatically on first use. Local-first and deterministic; optional use_llm synthesizes the final answer.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "query": {
          "type": "string",
          "description": "The global question to answer across the whole memory graph"
        },
        "workspace_hash": {
          "type": "string",
          "default": "",
          "description": "Workspace scope. Empty = global/unscoped entities."
        },
        "top_communities": {
          "type": "integer",
          "default": 3,
          "description": "How many best-matching communities to drill into"
        },
        "limit": {
          "type": "integer",
          "default": 10,
          "description": "Max member entities cited across all communities (round-robined so every matched community is represented)"
        },
        "auto_detect": {
          "type": "boolean",
          "default": true,
          "description": "Run community detection automatically when none are persisted yet"
        },
        "use_llm": {
          "type": "boolean",
          "default": false,
          "description": "Synthesize the final answer with the configured LLM; degrades to the extractive answer on error or when disabled."
        }
      },
      "required": ["query"]
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "query": { "type": "string" },
        "workspace_hash": { "type": "string" },
        "communities_considered": { "type": "integer", "description": "Persisted communities scored in the breadth pass" },
        "communities": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "id": { "type": "string" },
              "score": { "type": "number", "description": "Distinct query-token hits in the community summary" },
              "size": { "type": "integer" },
              "summary": { "type": "string" },
              "members": {
                "type": "array",
                "items": {
                  "type": "object",
                  "properties": {
                    "id": { "type": "string" },
                    "category": { "type": "string" },
                    "key": { "type": "string" },
                    "score": { "type": "number" },
                    "snippet": { "type": "string" }
                  }
                }
              }
            }
          }
        },
        "answer": { "type": "string", "description": "Extractive (or LLM-synthesized) holistic answer citing entities across communities" },
        "llm_used": { "type": "boolean" }
      }
    },
    "title": "Global Recall (GraphRAG)"
  },
  {
    "name": "perseus_vault_keystone_set",
    "description": "Author a Keystone — a mandatory policy rule that survives context compaction (#683). Unlike ordinary memories (retrieved when relevant), keystones are fetched deterministically at session start via perseus_vault_keystone_get, merged across scope, and are meant to be obeyed over any conflicting instruction (e.g. 'Every memory write MUST carry a retention class', 'Customer PII MUST NOT cross agent boundaries'). Higher weight wins on contradiction. Re-setting the same (scope, scope_id, content) updates it in place. Every mutation is appended to the cryptographic audit chain. Authoring is gated on trust tier: pass author_trust_tier (>= trust_tier_required, default 2). NOTE: until multi-agent trust tiers land (#684), author_trust_tier is caller-asserted; when omitted the write is allowed and the response flags that enforcement is pending.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "content": { "type": "string", "description": "The policy rule text. Imperative, testable directives work best." },
        "scope": { "type": "string", "default": "tenant", "description": "Merge scope: 'tenant' (org-wide), 'fleet' (a team), or 'agent' (an individual). Narrower scopes are layered on top of broader ones at get time." },
        "scope_id": { "type": "string", "description": "Identifier the keystone applies to within a non-tenant scope: the fleet_id ('fleet') or agent_id ('agent'). Omit/empty for tenant scope or 'all in scope'." },
        "weight": { "type": "number", "default": 1.0, "description": "Conflict-resolution weight; on contradiction the higher-weight keystone wins. Also the merge/sort order returned by keystone_get." },
        "trust_tier_required": { "type": "integer", "default": 2, "description": "Minimum author trust tier permitted to set/modify this keystone. Defaults to 2 (per #684's tier model: tier 2 = write keystones)." },
        "author_trust_tier": { "type": "integer", "description": "The authoring agent's trust tier, checked against trust_tier_required. Caller-asserted until #684 wires per-agent trust + session identity." },
        "agent_id": { "type": "string", "description": "Identity of the authoring agent, stamped on the keystone and its audit-chain event for provenance." },
        "workspace_hash": { "type": "string", "description": "Optional workspace scope. Keystones with an empty workspace_hash are global (apply everywhere)." }
      },
      "required": ["content"]
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "id": { "type": "string" },
        "created": { "type": "boolean", "description": "true if a new keystone was created, false if an existing one was updated" },
        "trust_enforced": { "type": "boolean", "description": "false when author_trust_tier was omitted (enforcement pending #684)" }
      }
    },
    "title": "Set Keystone"
  },
  {
    "name": "perseus_vault_keystone_get",
    "description": "Fetch the merged Keystones (mandatory policy rules, #683) that apply at session start — the deterministic counterpart to recall. Returns rules ordered by weight (highest first, then scope tenant<fleet<agent, then id) so a renderer can inject them ahead of all other context and resolve contradictions by weight. Filter by scope/scope_id/workspace to get exactly the set an agent must obey. Read-only.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "scope": { "type": "string", "description": "Optional: restrict to a single scope ('tenant' | 'fleet' | 'agent'). Omit to merge all scopes." },
        "scope_id": { "type": "string", "description": "Optional: with a non-tenant scope, restrict to this fleet_id/agent_id. Rules with an empty scope_id (scope-wide) are always included." },
        "workspace_hash": { "type": "string", "description": "Optional workspace scope. Global keystones (empty workspace_hash) are always included." }
      }
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "keystones": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "id": { "type": "string" },
              "content": { "type": "string" },
              "scope": { "type": "string" },
              "scope_id": { "type": "string" },
              "weight": { "type": "number" }
            }
          }
        },
        "count": { "type": "integer" }
      }
    },
    "title": "Get Keystones"
  },
  {
    "name": "perseus_vault_keystone_suggestions",
    "description": "List candidate directive/keystone suggestions (#889) extracted from `correct` captures by word-boundary-anchored patterns (en/de/ru/it/es). Suggestions are candidates only — never policy: promotion to the keystones table requires an explicit operator `approve` decision via perseus_vault_keystone_suggestion_decide. Each suggestion carries its source correction entity id for citation. Filter by status (pending/approved/rejected) and workspace; read-only.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "status": { "type": "string", "description": "Filter: '' (all), 'pending', 'approved', or 'rejected'. Default ''." },
        "workspace_hash": { "type": "string", "description": "Optional workspace scope filter." },
        "limit": { "type": "integer", "description": "Max rows (1-1000). Default 50." }
      }
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "suggestions": { "type": "array", "items": { "type": "object" } },
        "count": { "type": "integer" }
      }
    },
    "title": "List Keystone Suggestions"
  },
  {
    "name": "perseus_vault_keystone_suggestion_decide",
    "description": "Decide a keystone-suggestion candidate (#889): 'approve' promotes the suggestion's instruction into the keystones table (re-running the #683/#684 trust-tier gate — authoring requires tier >= trust_tier_required) and marks the suggestion approved; 'reject' marks it rejected and writes nothing. Extraction never writes policy — this explicit operator decision is the only promotion path.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "id": { "type": "string", "description": "Suggestion id (ksug-...)." },
        "action": { "type": "string", "description": "'approve' (promote to keystone) or 'reject'." },
        "scope": { "type": "string", "description": "Keystone scope: 'tenant' | 'fleet' | 'agent'. Default 'agent'." },
        "scope_id": { "type": "string", "description": "Keystone scope_id; defaults to agent_id." },
        "weight": { "type": "number", "description": "Conflict-resolution weight. Default 1.0." },
        "trust_tier_required": { "type": "integer", "description": "Minimum authoring tier. Default 2." },
        "author_trust_tier": { "type": "integer", "description": "Caller-asserted tier (used when the agent is not registry-registered)." },
        "agent_id": { "type": "string", "description": "Author agent id (registry-backed tier wins when registered)." },
        "workspace_hash": { "type": "string", "description": "Must match the suggestion's own workspace." }
      },
      "required": ["id", "action"]
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "ok": { "type": "boolean" },
        "keystone_id": { "type": "string" },
        "suggestion_status": { "type": "string" }
      }
    },
    "title": "Decide Keystone Suggestion"
  },
  {
    "name": "perseus_vault_agent",
    "description": "Register/update or look up an agent in the multi-agent registry (#684). Agents carry a trust tier (0-3) that gates sensitive ops (e.g. authoring keystones needs tier >= 2) and drives visibility enforcement on recall: tier 0 = read own only, 1 = fleet, 2 = read all + write keystones, 3 = admin. Pass trust_tier (and optionally name/fleet_id) to upsert; omit trust_tier to just look up. entities/journal already stamp agent_id (v1.2.0); this adds the identity + tier metadata. NOTE: an empty/unknown agent has no registry row — unknown identified agents resolve to tier 0, and a caller with no session identity is unscoped.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "agent_id": { "type": "string", "description": "The agent's stable identifier (e.g. the MCP clientInfo name)." },
        "name": { "type": "string", "description": "Human-readable name (upsert only)." },
        "trust_tier": { "type": "integer", "description": "Trust tier 0-3. Provide to upsert; omit to look up. Clamped to [0,3]." },
        "fleet_id": { "type": "string", "description": "Fleet/team the agent belongs to (used for 'fleet' visibility). Upsert only." }
      },
      "required": ["agent_id"]
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "found": { "type": "boolean" },
        "created": { "type": "boolean", "description": "true if an upsert created a new registry row" },
        "agent": {
          "type": "object",
          "properties": {
            "agent_id": { "type": "string" },
            "name": { "type": "string" },
            "trust_tier": { "type": "integer" },
            "fleet_id": { "type": "string" }
          }
        }
      }
    },
    "title": "Agent Registry"
  },
  {
    "name":"perseus_vault_authority_set", "description":"Create a versioned authority manifest for a registered agent.", "inputSchema":{"type":"object","properties":{"agent_id":{"type":"string"},"workspace_hash":{"type":"string"},"allowed_capabilities":{"type":"array","items":{"type":"string"}},"scope_anchors":{"type":"array","items":{"type":"string"}},"approval_required_capabilities":{"type":"array","items":{"type":"string"}},"approver_principals":{"type":"array","items":{"type":"string"}},"allowed_inbound_principals":{"type":"array","items":{"type":"string"}},"permitted_external_ref_prefixes":{"type":"array","items":{"type":"string"}},"max_parallel_actions":{"type":"integer","default":1},"mode":{"type":"string","default":"shadow"},"expires_at_unix_ms":{"type":"integer"},"author_agent_id":{"type":"string"},"capability_constraints_json":{"type":"string","default":"{}"}},"required":["agent_id","workspace_hash","allowed_capabilities","scope_anchors"]}, "title":"Set Action Authority"},
  {"name":"perseus_vault_authority_get", "description":"Get the active authority manifest for an agent and workspace.", "inputSchema":{"type":"object","properties":{"agent_id":{"type":"string"},"workspace_hash":{"type":"string"},"include_revoked":{"type":"boolean","default":false}},"required":["agent_id","workspace_hash"]}, "title":"Get Action Authority"},
  {"name":"perseus_vault_authority_revoke", "description":"Revoke an authority manifest.", "inputSchema":{"type":"object","properties":{"manifest_id":{"type":"string"},"actor_agent_id":{"type":"string"},"reason":{"type":"string"}},"required":["manifest_id"]}, "title":"Revoke Action Authority"},
  {"name":"perseus_vault_authority_set_signed", "description":"Load a signed, distributable policy/authority profile (Ed25519 sigstore-style attestation); verification failure grants no authority (fail closed) and the verification result lands in the ledger journal.", "inputSchema":{"type":"object","properties":{"profile_json":{"type":"string"},"trusted_public_key_b64":{"type":"string"},"author_agent_id":{"type":"string"}},"required":["profile_json","trusted_public_key_b64","author_agent_id"]}, "title":"Load Signed Authority Profile"},
  {"name":"perseus_vault_action_intent", "description":"Record a fail-closed authorized action intent.", "inputSchema":{"type":"object","properties":{"agent_id":{"type":"string"},"workspace_hash":{"type":"string"},"scope_anchor":{"type":"string"},"external_ref":{"type":"string"},"capability":{"type":"string"},"action_key":{"type":"string"},"intent_hash":{"type":"string"},"resource_constraints_json":{"type":"string","default":"{}"},"justification_entity_ids":{"type":"array","items":{"type":"string"},"description":"#1029: entity ids this action cites as grounding (must reference existing rows; the supersession impact index flags PENDING actions whose cited facts later changed)"},"lineage":{"type":"object","additionalProperties":false,"description":"#1134: versioned hash-only task/action-lineage transition; continuation is explicit","properties":{"schema_version":{"type":"integer","const":1},"transition":{"type":"string","enum":["continue","new_authorization"]},"action_class":{"type":"string","enum":["read","external_send","write","delete","other"]},"budget_cost":{"type":"integer","minimum":0,"maximum":1000000},"impact_units":{"type":"integer","minimum":0,"maximum":1000000},"continuation":{"type":"object","additionalProperties":false,"properties":{"schema_version":{"type":"integer","const":1},"lineage_id":{"type":"string"},"parent_head_digest":{"type":"string","pattern":"^[0-9a-f]{64}$"},"continuation_state_digest":{"type":"string","pattern":"^[0-9a-f]{64}$"},"workspace_hash":{"type":"string"},"agent_id":{"type":"string"},"authority_manifest_version":{"type":"integer","minimum":1},"policy_version":{"type":"string","pattern":"^[0-9a-f]{64}$"}},"required":["schema_version","lineage_id","parent_head_digest","continuation_state_digest","workspace_hash","agent_id","authority_manifest_version","policy_version"]}},"required":["schema_version","transition","action_class","budget_cost","impact_units"]}},"required":["agent_id","workspace_hash","scope_anchor","external_ref","capability","action_key","intent_hash"]}, "title":"Record Action Intent"},
  {"name":"perseus_vault_action_approve", "description":"Grant or deny an approval-requested action.", "inputSchema":{"type":"object","properties":{"action_id":{"type":"string"},"approver_principal":{"type":"string"},"decision":{"type":"string","enum":["granted","denied"]}},"required":["action_id","approver_principal","decision"]}, "title":"Decide Action Approval"},
  {"name":"perseus_vault_action_complete", "description":"Record an executed, failed, cancelled, or denied action outcome by hash.", "inputSchema":{"type":"object","properties":{"action_id":{"type":"string"},"actor_agent_id":{"type":"string"},"outcome":{"type":"string","enum":["executed","failed","cancelled","denied"]},"outcome_hash":{"type":"string"}},"required":["action_id","actor_agent_id","outcome","outcome_hash"]}, "title":"Complete Authorized Action"},
  {"name":"perseus_vault_action_resolve_timeout", "description":"Resolve a pending approval to deny once its window has expired (timeout defaults to deny).", "inputSchema":{"type":"object","properties":{"action_id":{"type":"string"},"approval_timeout_ms":{"type":"integer"}},"required":["action_id","approval_timeout_ms"]}, "title":"Resolve Approval Timeout"},
  {"name":"perseus_vault_action_receipt_get", "description":"Get durable action receipt metadata and hashes.", "inputSchema":{"type":"object","properties":{"action_id":{"type":"string"}},"required":["action_id"]}, "title":"Get Action Receipt"},
  {"name":"perseus_vault_action_lease_acquire", "description":"Acquire the single active lease for an action key.", "inputSchema":{"type":"object","properties":{"action_id":{"type":"string"},"holder_id":{"type":"string"},"ttl_seconds":{"type":"integer","default":1}},"required":["action_id","holder_id"]}, "title":"Acquire Action Lease"},
  {"name":"perseus_vault_action_lease_release", "description":"Release an action lease held by its owner.", "inputSchema":{"type":"object","properties":{"lease_id":{"type":"string"},"holder_id":{"type":"string"}},"required":["lease_id","holder_id"]}, "title":"Release Action Lease"},
  {"name":"perseus_vault_stage_trace_validate", "description":"Validate a versioned hash-only runtime stage trace and optionally compare replay semantics. Raw prompts, memory bodies, credentials, and tool payloads are not accepted.", "inputSchema":{"type":"object","properties":{"trace":{"type":"object","description":"perseus-vault-stage-trace/v1 structured trace"},"replay_of":{"type":"object","description":"Optional second trace to compare by replay fingerprint"}},"required":["trace"]}, "title":"Validate Runtime Stage Trace"},
  {"name":"perseus_vault_context_transform_validate", "description":"#1106: validate a versioned context-transformer proposal at the provider boundary. Returns only a hash-only receipt, bounded changed-span metadata, explicit outcome/lossiness, and replay/original references; raw messages, prompts, memory bodies, credentials, and tool payloads are not returned.", "inputSchema":{"type":"object","properties":{"request":{"type":"object","description":"perseus-vault-context-transformer/v1 request metadata and transient input_messages"},"proposed_output":{"type":"array","description":"Transient proposed provider messages; never returned in the response","items":{"type":"object"}},"proposed_output_tokens":{"type":"integer","minimum":0}},"required":["request","proposed_output"]}, "title":"Validate Context Transform"},
  {"name":"perseus_vault_reject_value", "description":"Record a scoped digest-only rejected-value tombstone. Equivalent values remain rejected across new entity keys and writer paths until the tombstone expires or is explicitly superseded.", "inputSchema":{"type":"object","properties":{"workspace_hash":{"type":"string","description":"Workspace scope; empty means global."},"subject":{"type":"string"},"predicate":{"type":"string"},"value":{"type":"string","description":"Normalized only for matching; the value is not stored."},"reason":{"type":"string"},"evidence_ref":{"type":"string"},"author_agent_id":{"type":"string"},"expires_at_unix_ms":{"type":"integer"}},"required":["workspace_hash","subject","predicate","value"]}, "title":"Reject Value"},
  {
    "name": "perseus_vault_span_audit",
    "description": "Extraction-loss net (#1048): audit an entity for fact-bearing sentences its extracted claims missed, retaining them verbatim as residual spans with provenance (embedding-first similarity, token fallback — no extra LLM call). Append-only; re-audits never duplicate. Spans are regular, decay/hygiene-subject memory state — never auto-served into recall.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "entity_id": { "type": "string", "description": "Entity id to audit" },
        "min_chars": { "type": "integer", "default": 12, "description": "Minimum sentence length in chars to consider" },
        "coverage_threshold": { "type": "number", "default": 0.55, "description": "Max claim-similarity below which a sentence is residual" },
        "mode": { "type": "string", "default": "auto", "description": "Similarity backend: auto | embedding | token" }
      },
      "required": ["entity_id"]
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "entity_id": { "type": "string" },
        "claims": { "type": "integer" },
        "spans_n": { "type": "integer" },
        "spans": { "type": "array", "items": { "type": "object" } },
        "mode_used": { "type": "string" }
      }
    },
    "annotations": { "readOnlyHint": false },
    "title": "Audit Extraction Loss (Residual Spans)"
  },
  {
    "name": "perseus_vault_report_refusal",
    "description": "Extraction-loss net (#1048): an answerer's refusal over a served payload is evidence. Re-scores the served entities' residual spans against the original query and returns a retry payload (spans whose query-similarity beats the entity's own by a margin — the anomaly rule). Units with no retry material accumulate lossy marks; at the threshold they are flagged for repair-on-touch.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "query": { "type": "string", "description": "The query the answerer could not answer" },
        "served_ids": { "type": "array", "items": { "type": "string" }, "description": "Entity ids that were in the served payload" },
        "reason": { "type": "string", "description": "Optional refusal reason (kept for the journal)" }
      },
      "required": ["query", "served_ids"]
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "retry": { "type": "array", "items": { "type": "object" } },
        "retry_n": { "type": "integer" },
        "lossy_flagged": { "type": "array", "items": { "type": "object" } },
        "margin": { "type": "number" }
      }
    },
    "annotations": { "readOnlyHint": false },
    "title": "Report Refusal (Retry Payload)"
  },
  {
    "name": "perseus_vault_report_success",
    "description": "Extraction-loss net (#1048): confirm a retry payload answered the query. Attaches a provisional query key (query fingerprint to entity ids) so an identical repeat query serves first-pass; served spans become confirmed; lossy units are cleared to repaired. The binding is durable until superseded by another report_success for the same query.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "query": { "type": "string", "description": "The query that was answered" },
        "entity_ids": { "type": "array", "items": { "type": "string" }, "description": "Entity ids that carried the answer" }
      },
      "required": ["query", "entity_ids"]
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "confirmed": { "type": "boolean" },
        "entity_ids": { "type": "array", "items": { "type": "string" } },
        "spans_confirmed": { "type": "integer" },
        "query_fingerprint": { "type": "string" }
      }
    },
    "annotations": { "readOnlyHint": false },
    "title": "Report Success (Confirm Query Key)"
  },
  {
    "name": "perseus_vault_rollback_repair",
    "description": "#1084 (arXiv:2608.10502): dependency-guided rollback repair for poisoned/stale memories. Builds a typed memory→action dependency graph from runtime provenance, preserves dependents with independent trusted support, tombstones unsupported state (quarantine — never deletes), and reports a scoped selective-replay proposal. Every step is journal-receipted and the repair is reversible (reverse_repair_id).",
    "inputSchema": {
      "type": "object",
      "properties": {
        "faulty_ids": {"type": "array", "items": {"type": "string"}, "description": "Diagnosed faulty entity ids"},
        "dry_run": {"type": "boolean", "default": false, "description": "Report the plan without writing"},
        "replay": {"type": "boolean", "default": false, "description": "Include a scoped selective-replay proposal (dry-run consolidation over the affected category/workspace)"},
        "workspace_hash": {"type": "string", "description": "Optional workspace scope hint"},
        "reverse_repair_id": {"type": "string", "description": "When set, reverse this previously recorded repair instead of running a new one"}
      },
      "required": ["faulty_ids"]
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "repair_id": {"type": "string"},
        "dry_run": {"type": "boolean"},
        "faulty": {"type": "array", "items": {"type": "string"}},
        "preserved": {"type": "array", "items": {"type": "object"}},
        "tombstoned": {"type": "array", "items": {"type": "string"}},
        "replay": {"type": "object"},
        "rollback": {"type": "object"}
      }
    },
    "annotations": { "readOnlyHint": false },
    "title": "Dependency-Guided Rollback Repair"
  },
  {
    "name": "perseus_vault_signer_epoch_set",
    "description": "#1080 (MutMem): register or replace the Ed25519 signing key for a signer epoch — the authorization root for signed transitions. The seed (32 raw bytes, base64) is stored at rest alongside the database (same trust domain as the AES key file) and never echoed back. Ops scope.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "epoch": {"type": "integer", "minimum": 1, "description": "Signer epoch number (key generation era)"},
        "seed_b64": {"type": "string", "description": "Raw 32-byte Ed25519 seed, base64-encoded"}
      },
      "required": ["epoch", "seed_b64"]
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "registered_epoch": {"type": "integer"},
        "signer_fingerprint": {"type": "string"}
      }
    },
    "annotations": { "readOnlyHint": false },
    "title": "Set Signer Epoch Key"
  },
  {
    "name": "perseus_vault_poison_label",
    "description": "#1080 (MutMem): set or revise a SIGNED poison label on a stored entity. Poison-likely content is retained (never silently deleted); recall consumes the label as trust evidence (poison_likely −90% effective score, suspect −50%, clean = restored). Every label write commits as a signed transition — fails closed when no signer epoch is registered.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "entity_id": {"type": "string", "description": "Entity to label"},
        "level": {"type": "string", "enum": ["poison_likely", "suspect", "clean"]},
        "reason": {"type": "string", "description": "Attribution for the label (recorded in the signed transition)"}
      },
      "required": ["entity_id", "level"]
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "entity_id": {"type": "string"},
        "level": {"type": "string"},
        "reason": {"type": "string"},
        "transition": {"type": "object"}
      }
    },
    "annotations": { "readOnlyHint": false },
    "title": "Set Poison Label"
  },
  {
    "name": "perseus_vault_transition_audit",
    "description": "#1080 (MutMem): replay the signed-transition chain end to end — every record must verify against its epoch key, link to the previous chain hash (no forks), and reproduce its own chain hash. Reports record count, verified count, chain head, and the first divergence (if any).",
    "inputSchema": {"type": "object", "properties": {}},
    "outputSchema": {
      "type": "object",
      "properties": {
        "records": {"type": "integer"},
        "verified": {"type": "integer"},
        "divergence": {"type": "object"},
        "chain_head": {"type": "string"},
        "note": {"type": "string"}
      }
    },
    "annotations": { "readOnlyHint": true },
    "title": "Audit Signed Transition Chain"
  },
{
    "name": "perseus_vault_skill_set",
    "description": "#1090 (ERSkill, arXiv:2608.12720): define or version a retrieval skill — a validated parameterization of recall primitives (mode, typed filters, trust/content weights, recency). New versions always enter the expansion frontier (double-frontier deployment): they never affect routing until a governed advancement.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "skill_id": {"type": "string"},
        "name": {"type": "string"},
        "version": {"type": "integer", "minimum": 1},
        "profile": {"type": "object", "description": "Router affinity weights: base/recent/negation/question/type_hint/long_query"},
        "template": {"type": "object", "description": "Skill template: mode (fts5|dense|hybrid|fused), limit 1..50, optional category/type_filter/layer/epistemic_state/weights"}
      },
      "required": ["skill_id", "version", "template"]
    },
    "outputSchema": {"type": "object", "properties": {"defined": {"type": "boolean"}, "skill_id": {"type": "string"}, "version": {"type": "integer"}, "frontier": {"type": "string"}, "receipt": {"type": "string"}}},
    "annotations": { "readOnlyHint": false },
    "title": "Define Retrieval Skill"
  },
  {
    "name": "perseus_vault_skill_route",
    "description": "#1090 (ERSkill): deterministic per-query routing over the SERVING frontier only — feature-based scoring, ties break by skill id. With serve=true the chosen skill executes (recall with its template) and the explored path is logged into the experience trie (skill id × query fingerprint × outcome).",
    "inputSchema": {
      "type": "object",
      "properties": {
        "query": {"type": "string"},
        "serve": {"type": "boolean", "default": false}
      },
      "required": ["query"]
    },
    "outputSchema": {"type": "object", "properties": {"skill_id": {"type": "string"}, "skill_version": {"type": "integer"}, "score": {"type": "number"}, "served": {"type": "boolean"}, "entities": {"type": "array", "items": {"type": "object"}}}},
    "annotations": { "readOnlyHint": false },
    "title": "Route Retrieval Query"
  },
  {
    "name": "perseus_vault_skill_advance",
    "description": "#1090 (ERSkill): governed double-frontier transition. advance (expansion→serving) REQUIRES non-regression evidence (wins/losses/ties + recall_delta) and is refused fail-closed on regression; demote (serving→expansion) is the governed rollback. Every transition is receipt-anchored and bumps the serving version.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "skill_id": {"type": "string"},
        "direction": {"type": "string", "enum": ["advance", "demote"]},
        "evidence": {"type": "object", "description": "eval_ref, wins, losses, ties, recall_delta"}
      },
      "required": ["skill_id", "direction"]
    },
    "outputSchema": {"type": "object", "properties": {"accepted": {"type": "boolean"}, "skill_id": {"type": "string"}, "frontier": {"type": "string"}, "serving_version": {"type": "integer"}, "receipt": {"type": "string"}, "reason": {"type": "string"}}},
    "annotations": { "readOnlyHint": false },
    "title": "Advance Retrieval Skill Frontier"
  },
  {
    "name": "perseus_vault_skill_audit",
    "description": "#1090 (ERSkill): read-only audit of the skill registry — definitions by frontier, serving version, experience-trie stats per skill, and the receipt trail (definitions, advancements, refusals).",
    "inputSchema": {"type": "object", "properties": {}},
    "outputSchema": {"type": "object", "properties": {"skills": {"type": "array", "items": {"type": "object"}}, "serving_version": {"type": "object"}, "experience_stats": {"type": "object"}, "receipts": {"type": "array", "items": {"type": "object"}}}},
    "annotations": { "readOnlyHint": true },
    "title": "Audit Retrieval Skills"
  },
{
    "name": "perseus_vault_decay_audit",
    "description": "#1091 (ScrubJay-MEM, arXiv:2608.04746): audit type-conditioned temporal decay — the deterministic perishability/utility-horizon profile table per memory type plus population aggregates (count, mean decay, mean age, past-horizon rows excluded from default recall).",
    "inputSchema": {"type": "object", "properties": {}},
    "outputSchema": {"type": "object", "properties": {"generated_at_unix_ms": {"type": "integer"}, "profiles": {"type": "array", "items": {"type": "object"}}, "population": {"type": "array", "items": {"type": "object"}}, "note": {"type": "string"}}},
    "annotations": { "readOnlyHint": true },
    "title": "Audit Temporal Decay"
  },
{
    "name": "perseus_vault_segment_consolidate",
    "description": "#1088 (LycheeMemory V2, arXiv:2608.12990): semantic segment-level consolidation — batch entities into semantic segments via deterministic boundary detection (inter-arrival gap + adjacent trigram discontinuity, never fixed windows), then run ONE bounded consolidate pass per finalized segment (>=2 members). Construction frequency is segment-count-bound, not write-count-bound. Segment plans are indexed under state keys segment_plan.<id>.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "category": {"type": "string", "description": "Category to consolidate"},
        "workspace_hash": {"type": "string", "description": "Workspace scope (required — ordinary runs are workspace-scoped)"},
        "gap_ms": {"type": "integer", "minimum": 0, "default": 21600000, "description": "Inter-arrival gap in ms that starts a new segment"},
        "sim_floor": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.25, "description": "Adjacent trigram similarity below which a new segment starts"},
        "max_entities": {"type": "integer", "minimum": 1, "maximum": 5000, "default": 1000, "description": "Scan cap"},
        "dry_run": {"type": "boolean", "default": false, "description": "Report plans without writing"}
      },
      "required": ["category", "workspace_hash"]
    },
    "outputSchema": {
      "type": "object",
      "properties": {
        "category": {"type": "string"},
        "workspace_hash": {"type": "string"},
        "dry_run": {"type": "boolean"},
        "scanned": {"type": "integer"},
        "segments": {"type": "integer"},
        "consolidated": {"type": "integer"},
        "skipped_singletons": {"type": "integer"},
        "consolidations": {"type": "array", "items": {"type": "object"}},
        "plans": {"type": "array", "items": {"type": "object"}}
      }
    },
    "annotations": { "readOnlyHint": false },
    "title": "Segment-Level Consolidation"
  },
{
    "name": "perseus_vault_state_audit",
    "description": "#1093 (STALE/StateAuditor, arXiv:2608.01619): audit state-table entries for implicit stale-dependency drift (sleep proposals whose entities vanished, experience-stats drift, cached entity-count drift, shadow-promote records) and repair by state-to-draft demotion — originals preserved verbatim under state_draft.*, live keys marked stale, journal receipts anchored. dry_run=true only reports.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "dry_run": {"type": "boolean", "default": false, "description": "Report only; make no writes"}
      }
    },
    "outputSchema": {"type": "object", "properties": {"dry_run": {"type": "boolean"}, "scanned": {"type": "integer"}, "stale_count": {"type": "integer"}, "found_stale": {"type": "array", "items": {"type": "object"}}, "repaired": {"type": "array", "items": {"type": "object"}}}},
    "annotations": { "readOnlyHint": false },
    "title": "Audit State Staleness"
  }

]"###,
        )
        .expect("tools JSON must be valid");
        registry
            .as_array()
            .expect("tools registry must be a JSON array")
            .clone()
    })
}

/// Tool scope classification (#1051): which advertisement tier carries each tool.
/// `Agent` = the everyday memory + coordination surface (recall, remember,
/// context, handoffs, state, and the agent-side AAR calls). `Ops` = the agent
/// surface plus operational grooming, maintenance, governance, and export.
/// `Admin` = full tier only — irreversible or root control-plane tools that
/// should never appear in a scoped list by accident.
///
/// Scope controls ADVERTISEMENT in tools/list only — it is a visibility view,
/// never an authorization boundary. tools/call stays functional for every
/// canonical tool regardless of the active scope; workspace binding and
/// authority manifests remain the enforcement layer.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
enum ToolScope {
    Agent = 0,
    Ops = 1,
    Admin = 2,
}

impl ToolScope {
    fn rank(self) -> u8 {
        self as u8
    }
}

/// The view a running server advertises. `Full` is the legacy default and
/// shows every canonical tool. Selected via PERSEUS_VAULT_TOOL_SCOPE
/// (values: `agent`, `ops`, anything else — including unset — resolves to
/// `full`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ScopeView {
    Agent,
    Ops,
    Full,
}

impl ScopeView {
    fn rank(self) -> u8 {
        match self {
            Self::Agent => 0,
            Self::Ops => 1,
            Self::Full => 2,
        }
    }
}

fn resolve_scope_view(raw: Option<&str>) -> ScopeView {
    match raw.map(str::trim).map(str::to_ascii_lowercase).as_deref() {
        Some("agent") => ScopeView::Agent,
        Some("ops") => ScopeView::Ops,
        _ => ScopeView::Full,
    }
}

/// The intentionally small advertisement surface for LLM hosts. These names are
/// all entries in the canonical registry; profile filtering never synthesizes or
/// renames a tool.
const LEAN_PROFILE_TOOL_NAMES: &[&str] = &[
    "perseus_vault_remember",
    "perseus_vault_recall",
    "perseus_vault_forget",
    "perseus_vault_correct",
    "perseus_vault_context",
    "perseus_vault_workspace_status",
    "perseus_vault_health",
];

/// Filter the canonical registry for the explicit server advertisement
/// profile. `Default` and `All` intentionally preserve the complete registry;
/// `Lean` is a `tools/list` reduction only and does not change authorization or
/// dispatch of hidden tools.
fn filter_registry_by_profile(tools: Vec<Value>, profile: ToolProfile) -> Vec<Value> {
    match profile {
        ToolProfile::Default | ToolProfile::All => tools,
        ToolProfile::Lean => tools
            .into_iter()
            .filter(|tool| {
                tool.get("name")
                    .and_then(Value::as_str)
                    .is_some_and(|name| LEAN_PROFILE_TOOL_NAMES.contains(&name))
            })
            .collect(),
    }
}

/// Canonical name -> scope tier. Exactly one entry per registry tool; the
/// 1:1 + completeness invariant is CI-enforced by
/// scripts/registry_metadata_check.py.
const TOOL_SCOPES: &[(&str, ToolScope)] = &[
    ("perseus_vault_provider_source_event", ToolScope::Agent),
    ("perseus_vault_declared_graph_manifest", ToolScope::Agent),
    ("perseus_vault_declared_graph_attest", ToolScope::Ops),
    ("perseus_vault_declared_graph_query", ToolScope::Agent),
    ("perseus_vault_remember", ToolScope::Agent),
    ("perseus_vault_write_gate", ToolScope::Agent),
    ("perseus_vault_recall", ToolScope::Agent),
    ("perseus_vault_handoff_pack", ToolScope::Agent),
    ("perseus_vault_delegation_brief", ToolScope::Agent),
    ("perseus_vault_intention", ToolScope::Agent),
    ("perseus_vault_proof_frame", ToolScope::Agent),
    ("perseus_vault_recall_batch", ToolScope::Agent),
    ("perseus_vault_recall_layer", ToolScope::Agent),
    ("perseus_vault_scan", ToolScope::Ops),
    ("perseus_vault_hygiene", ToolScope::Ops),
    ("perseus_vault_promote", ToolScope::Ops),
    ("perseus_vault_demote", ToolScope::Ops),
    ("perseus_vault_beliefs", ToolScope::Ops),
    ("perseus_vault_claim_card", ToolScope::Ops),
    ("perseus_vault_semantic_search", ToolScope::Agent),
    ("perseus_vault_ask", ToolScope::Agent),
    ("perseus_vault_get_entity", ToolScope::Agent),
    ("perseus_vault_history", ToolScope::Agent),
    ("perseus_vault_as_of", ToolScope::Agent),
    ("perseus_vault_valid_at", ToolScope::Agent),
    ("perseus_vault_bitemporal", ToolScope::Ops),
    ("perseus_vault_forget", ToolScope::Agent),
    ("perseus_vault_ingest", ToolScope::Ops),
    ("perseus_vault_ingest_file", ToolScope::Ops),
    ("perseus_vault_artifact_register", ToolScope::Ops),
    ("perseus_vault_learned_artifact_register", ToolScope::Ops),
    ("perseus_vault_workspace_bind", ToolScope::Ops),
    ("perseus_vault_workspace_unbind", ToolScope::Ops),
    ("perseus_vault_workspace_quarantine", ToolScope::Ops),
    ("perseus_vault_workspace_status", ToolScope::Agent),
    ("perseus_vault_artifact_manifest", ToolScope::Ops),
    ("perseus_vault_artifact_excerpt", ToolScope::Ops),
    ("perseus_vault_artifact_log_digest", ToolScope::Ops),
    ("perseus_vault_artifact_verify_value", ToolScope::Ops),
    ("perseus_vault_embed", ToolScope::Ops),
    ("perseus_vault_prune", ToolScope::Ops),
    ("perseus_vault_link", ToolScope::Agent),
    ("perseus_vault_unlink", ToolScope::Agent),
    ("perseus_vault_journal", ToolScope::Agent),
    ("perseus_vault_check_failure_pattern", ToolScope::Agent),
    ("perseus_vault_timeline", ToolScope::Agent),
    ("perseus_vault_state_set", ToolScope::Agent),
    ("perseus_vault_state_get", ToolScope::Agent),
    ("perseus_vault_state_delete", ToolScope::Agent),
    ("perseus_vault_state_list", ToolScope::Agent),
    ("perseus_vault_health", ToolScope::Agent),
    ("perseus_vault_deployment_profile", ToolScope::Ops),
    ("perseus_vault_config_report", ToolScope::Ops),
    ("perseus_vault_type_policies", ToolScope::Ops),
    ("perseus_vault_handoff_restart", ToolScope::Agent),
    ("perseus_vault_quality_telemetry", ToolScope::Ops),
    ("perseus_vault_retrieval_telemetry", ToolScope::Ops),
    ("perseus_vault_stats", ToolScope::Ops),
    ("perseus_vault_compact", ToolScope::Ops),
    ("perseus_vault_purge", ToolScope::Admin),
    ("perseus_vault_project_task", ToolScope::Agent),
    ("perseus_vault_experience_projection", ToolScope::Agent),
    (
        "perseus_vault_experience_projection_rebuild",
        ToolScope::Ops,
    ),
    ("perseus_vault_expand_source", ToolScope::Agent),
    ("perseus_vault_expire", ToolScope::Ops),
    ("perseus_vault_redact", ToolScope::Ops),
    ("perseus_vault_erase", ToolScope::Admin),
    ("perseus_vault_memories", ToolScope::Agent),
    ("perseus_vault_migrate", ToolScope::Admin),
    ("perseus_vault_context", ToolScope::Agent),
    ("perseus_vault_extract", ToolScope::Ops),
    ("perseus_vault_capture", ToolScope::Agent),
    ("perseus_vault_traverse", ToolScope::Agent),
    ("perseus_vault_graph_drift", ToolScope::Ops),
    ("perseus_vault_graph_attest", ToolScope::Ops),
    ("perseus_vault_score", ToolScope::Ops),
    ("perseus_vault_follow", ToolScope::Ops),
    ("perseus_vault_operator_review", ToolScope::Ops),
    ("perseus_vault_eval_history", ToolScope::Ops),
    ("perseus_vault_web_gap_fill", ToolScope::Agent),
    ("perseus_vault_mental_model_set", ToolScope::Ops),
    ("perseus_vault_mental_model_review", ToolScope::Ops),
    ("perseus_vault_write_quarantine", ToolScope::Ops),
    ("perseus_vault_admission_quarantine", ToolScope::Ops),
    ("perseus_vault_admission_decide", ToolScope::Ops),
    ("perseus_vault_writer_handoff", ToolScope::Ops),
    ("perseus_vault_impact_report", ToolScope::Ops),
    ("perseus_vault_finding_record", ToolScope::Ops),
    ("perseus_vault_grounding_admit", ToolScope::Ops),
    ("perseus_vault_grounding_reconcile", ToolScope::Ops),
    ("perseus_vault_drift_check", ToolScope::Ops),
    ("perseus_vault_drift_repair", ToolScope::Ops),
    ("perseus_vault_restore_forward", ToolScope::Ops),
    ("perseus_vault_signer_epoch_set", ToolScope::Ops),
    ("perseus_vault_poison_label", ToolScope::Ops),
    ("perseus_vault_transition_audit", ToolScope::Ops),
    ("perseus_vault_skill_set", ToolScope::Ops),
    ("perseus_vault_skill_route", ToolScope::Ops),
    ("perseus_vault_skill_advance", ToolScope::Ops),
    ("perseus_vault_skill_audit", ToolScope::Ops),
    ("perseus_vault_decay_audit", ToolScope::Ops),
    ("perseus_vault_segment_consolidate", ToolScope::Ops),
    ("perseus_vault_state_audit", ToolScope::Ops),
    ("perseus_vault_rollback_repair", ToolScope::Ops),
    ("perseus_vault_op_run", ToolScope::Ops),
    ("perseus_vault_op_run_list", ToolScope::Ops),
    ("perseus_vault_op_run_get", ToolScope::Ops),
    ("perseus_vault_op_run_retry", ToolScope::Ops),
    ("perseus_vault_op_run_prune", ToolScope::Ops),
    ("perseus_vault_preload_resolve", ToolScope::Ops),
    ("perseus_vault_preload_stats", ToolScope::Ops),
    ("perseus_vault_preload_propose", ToolScope::Ops),
    ("perseus_vault_preload_review", ToolScope::Ops),
    ("perseus_vault_guide_seed", ToolScope::Ops),
    ("perseus_vault_declared_schema_set", ToolScope::Ops),
    ("perseus_vault_declared_query", ToolScope::Ops),
    ("perseus_vault_conflicts", ToolScope::Ops),
    ("perseus_vault_maintenance_status", ToolScope::Ops),
    ("perseus_vault_consolidate", ToolScope::Ops),
    ("perseus_vault_sleep", ToolScope::Ops),
    ("perseus_vault_dream", ToolScope::Ops),
    ("perseus_vault_seal", ToolScope::Ops),
    ("perseus_vault_tamper_scan", ToolScope::Ops),
    ("perseus_vault_provenance_projection", ToolScope::Ops),
    ("perseus_vault_param_lineage", ToolScope::Ops),
    ("perseus_vault_typed_traversal", ToolScope::Ops),
    ("perseus_vault_traversal_ablation", ToolScope::Ops),
    ("perseus_vault_model_inheritance", ToolScope::Ops),
    ("perseus_vault_vault_export", ToolScope::Ops),
    ("perseus_vault_derived_export", ToolScope::Ops),
    ("perseus_vault_markdown_import", ToolScope::Ops),
    ("perseus_vault_structured_index_anchor", ToolScope::Ops),
    ("perseus_vault_vault_import", ToolScope::Admin),
    ("perseus_vault_shadow_compare", ToolScope::Ops),
    ("perseus_vault_shadow_promote", ToolScope::Ops),
    ("perseus_vault_shadow_rollback", ToolScope::Ops),
    ("perseus_vault_decay", ToolScope::Ops),
    ("perseus_vault_reindex", ToolScope::Ops),
    ("perseus_vault_workspace_list", ToolScope::Agent),
    ("perseus_vault_recall_when", ToolScope::Agent),
    ("perseus_vault_cohere", ToolScope::Ops),
    ("perseus_vault_share", ToolScope::Ops),
    ("perseus_vault_correct", ToolScope::Agent),
    ("perseus_vault_synthesize", ToolScope::Agent),
    ("perseus_vault_bench", ToolScope::Ops),
    ("perseus_vault_autocohere", ToolScope::Ops),
    ("perseus_vault_supersede", ToolScope::Agent),
    ("perseus_vault_consistency_audit", ToolScope::Ops),
    ("perseus_vault_audit_ruling", ToolScope::Ops),
    ("perseus_vault_maintenance", ToolScope::Ops),
    ("perseus_vault_communities", ToolScope::Ops),
    ("perseus_vault_community_summary", ToolScope::Ops),
    ("perseus_vault_global_recall", ToolScope::Ops),
    ("perseus_vault_keystone_set", ToolScope::Ops),
    ("perseus_vault_keystone_get", ToolScope::Agent),
    ("perseus_vault_keystone_suggestions", ToolScope::Ops),
    ("perseus_vault_keystone_suggestion_decide", ToolScope::Ops),
    ("perseus_vault_agent", ToolScope::Agent),
    ("perseus_vault_authority_set", ToolScope::Admin),
    ("perseus_vault_authority_get", ToolScope::Agent),
    ("perseus_vault_authority_revoke", ToolScope::Admin),
    ("perseus_vault_authority_set_signed", ToolScope::Admin),
    ("perseus_vault_action_intent", ToolScope::Agent),
    ("perseus_vault_action_approve", ToolScope::Ops),
    ("perseus_vault_action_complete", ToolScope::Agent),
    ("perseus_vault_action_resolve_timeout", ToolScope::Ops),
    ("perseus_vault_action_receipt_get", ToolScope::Agent),
    ("perseus_vault_action_lease_acquire", ToolScope::Agent),
    ("perseus_vault_action_lease_release", ToolScope::Agent),
    ("perseus_vault_stage_trace_validate", ToolScope::Ops),
    ("perseus_vault_context_transform_validate", ToolScope::Ops),
    ("perseus_vault_reject_value", ToolScope::Ops),
    ("perseus_vault_span_audit", ToolScope::Agent),
    ("perseus_vault_report_refusal", ToolScope::Agent),
    ("perseus_vault_report_success", ToolScope::Agent),
];

fn tool_scope_rank(name: &str) -> u8 {
    TOOL_SCOPES
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, s)| s.rank())
        // Unknown names rank as Admin: fail closed toward the full view.
        .unwrap_or(ToolScope::Admin as u8)
}

fn filter_registry_by_view(tools: Vec<Value>, view: ScopeView) -> Vec<Value> {
    match view {
        ScopeView::Full => tools,
        v => tools
            .into_iter()
            .filter(|t| {
                let name = t.get("name").and_then(|n| n.as_str()).unwrap_or("");
                tool_scope_rank(name) <= v.rank()
            })
            .collect(),
    }
}

/// Build the tools/list response from the canonical registry (parsed once by
/// `tool_registry_base`; cached there so repeated tools/list calls don't
/// re-parse the embedded literal — perf review #208). #1051: the active
/// scope view filters the advertised list; the default is full (legacy
/// behavior).
fn list_tools(id: Option<Value>, profile: ToolProfile) -> JsonRpcResponse {
    let view = resolve_scope_view(std::env::var("PERSEUS_VAULT_TOOL_SCOPE").ok().as_deref());
    let profile_tools = filter_registry_by_profile(tool_registry_base().clone(), profile);
    let tools = filter_registry_by_view(profile_tools, view);
    JsonRpcResponse {
        jsonrpc: "2.0".to_string(),
        id,
        result: Some(json!({
            "tools": tools
        })),
        error: None,
    }
}
fn call_tool(name: &str, db: &Database, args: Value, _id: Option<Value>) -> String {
    // Keep the caller's original name for error messages — a
    // "perseus_vault_bogus" call should say so, not report a rewritten name.
    let original_name = name;

    // #858: fail loud when the running binary was replaced on disk — never
    // silently serve results from a stale process image. The handoff tool and
    // health stay callable (health reports the staleness in its payload).
    if let Some(stale_msg) = crate::live_update::stale_error_message(name) {
        return serde_json::to_string(&json!({
            "content": [{"type": "text", "text": stale_msg}],
            "isError": true
        }))
        .unwrap_or_else(|_| {
            format!(
                r#"{{"content":[{{"type":"text","text":"{}"}}],"isError":true}}"#,
                stale_msg
            )
        });
    }

    let handler_result: Result<String, String> = match name {
        "perseus_vault_remember" => tools::handle_remember(db, args).map_err(|e| e.to_string()),

        "perseus_vault_write_gate" => tools::handle_write_gate(db, args).map_err(|e| e.to_string()),

        "perseus_vault_reject_value" => {
            tools::handle_reject_value(db, args).map_err(|e| e.to_string())
        }
        "perseus_vault_span_audit" => tools::handle_span_audit(db, args).map_err(|e| e.to_string()),
        "perseus_vault_report_refusal" => {
            tools::handle_report_refusal(db, args).map_err(|e| e.to_string())
        }
        "perseus_vault_report_success" => {
            tools::handle_report_success(db, args).map_err(|e| e.to_string())
        }

        "perseus_vault_provider_source_event" => {
            tools::handle_provider_source_event(db, args).map_err(|e| e.to_string())
        }
        "perseus_vault_declared_graph_manifest" => {
            tools::handle_declared_graph_manifest(db, args).map_err(|e| e.to_string())
        }
        "perseus_vault_declared_graph_attest" => {
            tools::handle_declared_graph_attest(db, args).map_err(|e| e.to_string())
        }
        "perseus_vault_declared_graph_query" => {
            tools::handle_declared_graph_query(db, args).map_err(|e| e.to_string())
        }
        "perseus_vault_recall" => tools::handle_recall(db, args).map_err(|e| e.to_string()),
        "perseus_vault_handoff_pack" => {
            tools::handle_handoff_pack(db, args).map_err(|e| e.to_string())
        }
        "perseus_vault_delegation_brief" => {
            tools::handle_delegation_brief(db, args).map_err(|e| e.to_string())
        }
        "perseus_vault_intention" => tools::handle_intention(db, args).map_err(|e| e.to_string()),
        "perseus_vault_proof_frame" => {
            tools::handle_proof_frame(db, args).map_err(|e| e.to_string())
        }

        "perseus_vault_recall_batch" => {
            tools::handle_recall_batch(db, args).map_err(|e| e.to_string())
        }

        "perseus_vault_recall_layer" => {
            tools::handle_recall_layer(db, args).map_err(|e| e.to_string())
        }

        "perseus_vault_scan" => tools::handle_scan(db, args).map_err(|e| e.to_string()),

        "perseus_vault_hygiene" => tools::handle_hygiene(db, args).map_err(|e| e.to_string()),

        "perseus_vault_semantic_search" => {
            tools::handle_semantic_search(db, args).map_err(|e| e.to_string())
        }

        "perseus_vault_ask" => tools::handle_ask(db, args).map_err(|e| e.to_string()),

        "perseus_vault_get_entity" => tools::handle_get_entity(db, args).map_err(|e| e.to_string()),
        "perseus_vault_history" => tools::handle_history(db, args).map_err(|e| e.to_string()),
        "perseus_vault_as_of" => tools::handle_as_of(db, args).map_err(|e| e.to_string()),
        "perseus_vault_valid_at" => tools::handle_valid_at(db, args).map_err(|e| e.to_string()),
        "perseus_vault_bitemporal" => tools::handle_bitemporal(db, args).map_err(|e| e.to_string()),
        "perseus_vault_forget" => tools::handle_forget(db, args).map_err(|e| e.to_string()),

        "perseus_vault_ingest" => tools::handle_ingest(db, args).map_err(|e| e.to_string()),

        "perseus_vault_ingest_file" => {
            tools::handle_ingest_file(db, args).map_err(|e| e.to_string())
        }

        "perseus_vault_artifact_register" => {
            tools::handle_artifact_register(db, args).map_err(|e| e.to_string())
        }
        "perseus_vault_artifact_manifest" => {
            tools::handle_artifact_manifest(db, args).map_err(|e| e.to_string())
        }
        "perseus_vault_artifact_excerpt" => {
            tools::handle_artifact_excerpt(db, args).map_err(|e| e.to_string())
        }
        "perseus_vault_artifact_log_digest" => {
            tools::handle_artifact_log_digest(db, args).map_err(|e| e.to_string())
        }
        "perseus_vault_artifact_verify_value" => {
            tools::handle_artifact_verify_value(db, args).map_err(|e| e.to_string())
        }

        "perseus_vault_embed" => tools::handle_embed(db, args).map_err(|e| e.to_string()),

        "perseus_vault_project_task" => tools::handle_project_task(db, args),
        "perseus_vault_experience_projection" => tools::handle_experience_projection(db, args),
        "perseus_vault_experience_projection_rebuild" => {
            tools::handle_experience_projection_rebuild(db, args)
        }

        "perseus_vault_expand_source" => tools::handle_expand_source(db, args),

        "perseus_vault_prune" => tools::handle_prune(db, args).map_err(|e| e.to_string()),

        "perseus_vault_link" => tools::handle_link(db, args).map_err(|e| e.to_string()),

        "perseus_vault_unlink" => tools::handle_unlink(db, args).map_err(|e| e.to_string()),

        "perseus_vault_journal" => tools::handle_journal(db, args).map_err(|e| e.to_string()),

        "perseus_vault_check_failure_pattern" => {
            tools::handle_check_failure_pattern(db, args).map_err(|e| e.to_string())
        }

        "perseus_vault_timeline" => tools::handle_timeline(db, args).map_err(|e| e.to_string()),

        "perseus_vault_state_set" => tools::handle_state_set(db, args).map_err(|e| e.to_string()),

        "perseus_vault_state_get" => tools::handle_state_get(db, args).map_err(|e| e.to_string()),

        "perseus_vault_state_delete" => {
            tools::handle_state_delete(db, args).map_err(|e| e.to_string())
        }

        "perseus_vault_state_list" => tools::handle_state_list(db, args).map_err(|e| e.to_string()),

        "perseus_vault_health" => Ok(tools::handle_health(db)),
        "perseus_vault_deployment_profile" => tools::handle_deployment_profile(db, args),
        "perseus_vault_config_report" => tools::handle_config_report(db, args),
        "perseus_vault_type_policies" => tools::handle_type_policies(db, args),
        "perseus_vault_handoff_restart" => crate::live_update::handle_handoff_restart(args),
        "perseus_vault_quality_telemetry" => tools::handle_quality_telemetry(db, args),
        "perseus_vault_retrieval_telemetry" => tools::handle_retrieval_telemetry(db, args),

        "perseus_vault_stats" => Ok(tools::handle_stats(db)),

        "perseus_vault_compact" => Ok(tools::handle_compact(db, args)),

        "perseus_vault_purge" => tools::handle_purge(db, args).map_err(|e| e.to_string()),
        "perseus_vault_expire" => tools::handle_expire(db, args).map_err(|e| e.to_string()),
        "perseus_vault_redact" => tools::handle_redact(db, args).map_err(|e| e.to_string()),
        "perseus_vault_erase" => tools::handle_erase(db, args).map_err(|e| e.to_string()),
        "perseus_vault_learned_artifact_register" => {
            tools::handle_learned_artifact_register(db, args).map_err(|e| e.to_string())
        }

        "perseus_vault_workspace_bind" => {
            tools::handle_workspace_bind(db, args).map_err(|e| e.to_string())
        }
        "perseus_vault_workspace_unbind" => {
            tools::handle_workspace_unbind(db, args).map_err(|e| e.to_string())
        }
        "perseus_vault_workspace_quarantine" => {
            tools::handle_workspace_quarantine(db, args).map_err(|e| e.to_string())
        }
        "perseus_vault_workspace_status" => {
            tools::handle_workspace_status(db, args).map_err(|e| e.to_string())
        }
        "perseus_vault_memories" => tools::handle_memories(db, args).map_err(|e| e.to_string()),

        "perseus_vault_migrate" => Ok(tools::handle_migrate(db, args)),

        "perseus_vault_context" => Ok(tools::handle_context(db, args)),

        "perseus_vault_extract" => tools::handle_extract(db, args).map_err(|e| e.to_string()),

        "perseus_vault_capture" => tools::handle_capture(db, args).map_err(|e| e.to_string()),

        "perseus_vault_traverse" => Ok(tools::handle_traverse(db, args)),
        "perseus_vault_graph_drift" => tools::handle_graph_drift(db, args),
        "perseus_vault_graph_attest" => tools::handle_graph_attest(db, args),
        "perseus_vault_score" => Ok(tools::handle_score(db, args)),
        "perseus_vault_follow" => tools::handle_follow(db, args).map_err(|e| e.to_string()),
        "perseus_vault_keystone_set" => tools::handle_keystone_set(db, args),
        "perseus_vault_keystone_suggestions" => tools::handle_keystone_suggestions(db, args),
        "perseus_vault_keystone_suggestion_decide" => {
            tools::handle_keystone_suggestion_decide(db, args)
        }
        "perseus_vault_keystone_get" => tools::handle_keystone_get(db, args),
        "perseus_vault_agent" => tools::handle_agent(db, args),
        "perseus_vault_authority_set" => tools::handle_authority_set(db, args),
        "perseus_vault_authority_set_signed" => tools::handle_authority_set_signed(db, args),
        "perseus_vault_authority_get" => tools::handle_authority_get(db, args),
        "perseus_vault_authority_revoke" => tools::handle_authority_revoke(db, args),
        "perseus_vault_action_intent" => tools::handle_action_intent(db, args),
        "perseus_vault_action_approve" => tools::handle_action_approve(db, args),
        "perseus_vault_action_complete" => tools::handle_action_complete(db, args),
        "perseus_vault_action_resolve_timeout" => tools::handle_action_resolve_timeout(db, args),
        "perseus_vault_action_receipt_get" => tools::handle_action_receipt_get(db, args),
        "perseus_vault_action_lease_acquire" => tools::handle_action_lease_acquire(db, args),
        "perseus_vault_action_lease_release" => tools::handle_action_lease_release(db, args),
        "perseus_vault_stage_trace_validate" => (|| -> Result<String, String> {
            let trace_value = args
                .get("trace")
                .cloned()
                .ok_or_else(|| "stage_trace_validate requires trace".to_string())?;
            let trace: crate::stage_trace::StageTrace = serde_json::from_value(trace_value)
                .map_err(|e| format!("invalid stage trace: {e}"))?;
            trace.validate()?;
            let replay_fingerprint = trace.replay_fingerprint()?;
            let replay_match = if let Some(replay_value) = args.get("replay_of").cloned() {
                let replay: crate::stage_trace::StageTrace =
                    serde_json::from_value(replay_value)
                        .map_err(|e| format!("invalid replay trace: {e}"))?;
                Some(crate::stage_trace::StageTrace::validate_replay(&trace, &replay).is_ok())
            } else {
                None
            };
            serde_json::to_string(&json!({
                "valid": true,
                "trace_digest": trace.digest()?,
                "replay_fingerprint": replay_fingerprint,
                "replay_match": replay_match,
                "schema_version": crate::stage_trace::STAGE_TRACE_SCHEMA_VERSION,
                "stage_count": trace.stages.len(),
            }))
            .map_err(|e| e.to_string())
        })(),
        "perseus_vault_context_transform_validate" => (|| -> Result<String, String> {
            let request_value = args
                .get("request")
                .cloned()
                .ok_or_else(|| "context_transform_validate requires request".to_string())?;
            let proposed_value = args
                .get("proposed_output")
                .cloned()
                .ok_or_else(|| "context_transform_validate requires proposed_output".to_string())?;
            let request: crate::context_transform::ContextTransformRequest =
                serde_json::from_value(request_value)
                    .map_err(|e| format!("invalid context transform request: {e}"))?;
            let proposed_output: Vec<crate::context_transform::ContextMessage> =
                serde_json::from_value(proposed_value)
                    .map_err(|e| format!("invalid context transform proposed_output: {e}"))?;
            let proposed_output_tokens = args.get("proposed_output_tokens").and_then(Value::as_u64);
            let decision = crate::context_transform::transform_context(
                &request,
                proposed_output,
                proposed_output_tokens,
            )?;
            let receipt = decision.receipt;
            let transformed = matches!(receipt.outcome.as_str(), "transformed" | "degraded");
            serde_json::to_string(&json!({
                "valid": true,
                "accepted": true,
                "transformed": transformed,
                "outcome": receipt.outcome,
                "actual_lossiness": receipt.actual_lossiness,
                "output_message_count": decision.output_messages.len(),
                "receipt": receipt,
                "schema_version": crate::context_transform::CONTEXT_TRANSFORMER_SCHEMA_VERSION,
            }))
            .map_err(|e| e.to_string())
        })(),
        "perseus_vault_promote" => tools::handle_promote(db, args),
        "perseus_vault_demote" => tools::handle_demote(db, args),
        "perseus_vault_beliefs" => beliefs::handle_beliefs(db, args),
        "perseus_vault_claim_card" => claim_card::handle_claim_card(db, args),
        "perseus_vault_operator_review" => tools::handle_operator_review(db, args),
        "perseus_vault_eval_history" => tools::handle_eval_history(db, args),
        "perseus_vault_web_gap_fill" => tools::handle_web_gap_fill(db, args),
        "perseus_vault_mental_model_set" => tools::handle_mental_model_set(db, args),
        "perseus_vault_mental_model_review" => tools::handle_mental_model_review(db, args),
        "perseus_vault_write_quarantine" => tools::handle_write_quarantine(db, args),
        "perseus_vault_admission_quarantine" => tools::handle_admission_quarantine(db, args),
        "perseus_vault_admission_decide" => tools::handle_admission_decide(db, args),
        "perseus_vault_writer_handoff" => tools::handle_writer_handoff(db, args),
        "perseus_vault_impact_report" => tools::handle_impact_report(db, args),
        "perseus_vault_finding_record" => tools::handle_finding_record(db, args),
        "perseus_vault_grounding_admit" => tools::handle_grounding_admit(db, args),
        "perseus_vault_grounding_reconcile" => tools::handle_grounding_reconcile(db, args),
        "perseus_vault_drift_check" => tools::handle_drift_check(db, args),
        "perseus_vault_drift_repair" => tools::handle_drift_repair(db, args),
        "perseus_vault_restore_forward" => tools::handle_restore_forward(db, args),
        "perseus_vault_signer_epoch_set" => tools::handle_signer_epoch_set(db, args),
        "perseus_vault_poison_label" => tools::handle_poison_label(db, args),
        "perseus_vault_transition_audit" => tools::handle_transition_audit(db, args),

        "perseus_vault_skill_set" => tools::handle_skill_set(db, args),
        "perseus_vault_skill_route" => tools::handle_skill_route(db, args),
        "perseus_vault_skill_advance" => tools::handle_skill_advance(db, args),
        "perseus_vault_skill_audit" => tools::handle_skill_audit(db, args),

        "perseus_vault_decay_audit" => tools::handle_decay_audit(db, args),

        "perseus_vault_segment_consolidate" => tools::handle_segment_consolidate(db, args),

        "perseus_vault_state_audit" => tools::handle_state_audit(db, args),
        "perseus_vault_rollback_repair" => tools::handle_rollback_repair(db, args),
        "perseus_vault_op_run" => tools::handle_op_run(db, args).map_err(|e| e.to_string()),
        "perseus_vault_op_run_list" => {
            tools::handle_op_run_list(db, args).map_err(|e| e.to_string())
        }
        "perseus_vault_op_run_get" => tools::handle_op_run_get(db, args).map_err(|e| e.to_string()),
        "perseus_vault_op_run_retry" => {
            tools::handle_op_run_retry(db, args).map_err(|e| e.to_string())
        }
        "perseus_vault_op_run_prune" => {
            tools::handle_op_run_prune(db, args).map_err(|e| e.to_string())
        }
        "perseus_vault_preload_stats" => {
            tools::handle_preload_stats(db, args).map_err(|e| e.to_string())
        }
        "perseus_vault_preload_resolve" => {
            tools::handle_preload_resolve(db, args).map_err(|e| e.to_string())
        }
        "perseus_vault_preload_propose" => {
            tools::handle_preload_propose(db, args).map_err(|e| e.to_string())
        }
        "perseus_vault_preload_review" => {
            tools::handle_preload_review(db, args).map_err(|e| e.to_string())
        }
        "perseus_vault_guide_seed" => tools::handle_guide_seed(db, args).map_err(|e| e.to_string()),
        "perseus_vault_declared_schema_set" => {
            tools::handle_declared_schema_set(db, args).map_err(|e| e.to_string())
        }
        "perseus_vault_declared_query" => {
            tools::handle_declared_query(db, args).map_err(|e| e.to_string())
        }
        "perseus_vault_conflicts" => Ok(tools::handle_conflicts(db, args)),
        "perseus_vault_consolidate" => Ok(tools::handle_consolidate(db, args)),
        "perseus_vault_sleep" => Ok(tools::handle_sleep(db, args)),
        "perseus_vault_maintenance_status" => {
            tools::handle_maintenance_status(db, args).map_err(|e| e.to_string())
        }
        "perseus_vault_dream" => tools::handle_dream(db, args),
        "perseus_vault_seal" => tools::handle_seal(db, args),
        "perseus_vault_tamper_scan" => tools::handle_tamper_scan(db, args),
        "perseus_vault_provenance_projection" => tools::handle_provenance_projection(db, args),
        "perseus_vault_param_lineage" => tools::handle_param_lineage(db, args),
        "perseus_vault_typed_traversal" => tools::handle_typed_traversal(db, args),
        "perseus_vault_traversal_ablation" => tools::handle_traversal_ablation(db, args),
        "perseus_vault_model_inheritance" => tools::handle_model_inheritance(db, args),
        "perseus_vault_vault_export" => Ok(tools::handle_vault_export(db, args)),
        "perseus_vault_derived_export" => tools::handle_derived_export(db, args),
        "perseus_vault_markdown_import" => tools::handle_markdown_import(db, args),
        "perseus_vault_structured_index_anchor" => tools::handle_structured_index_anchor(db, args),
        "perseus_vault_vault_import" => Ok(tools::handle_vault_import(db, args)),
        "perseus_vault_shadow_compare" => {
            tools::handle_shadow_compare(db, args).map_err(|e| e.to_string())
        }
        "perseus_vault_shadow_promote" => {
            tools::handle_shadow_promote(db, args).map_err(|e| e.to_string())
        }
        "perseus_vault_shadow_rollback" => {
            tools::handle_shadow_rollback(db, args).map_err(|e| e.to_string())
        }
        "perseus_vault_decay" => Ok(tools::handle_decay(db, args)),
        "perseus_vault_reindex" => Ok(tools::handle_reindex(db, args)),
        "perseus_vault_share" => tools::handle_share(db, args).map_err(|e| e.to_string()),
        "perseus_vault_federate" => tools::handle_federate(db, args).map_err(|e| e.to_string()),
        "perseus_vault_workspace_list" => Ok(tools::handle_workspace_list(db)),
        "perseus_vault_recall_when" => {
            tools::handle_recall_when(db, args).map_err(|e| e.to_string())
        }
        "perseus_vault_cohere" => tools::handle_cohere(db, args).map_err(|e| e.to_string()),
        "perseus_vault_correct" => tools::handle_correct(db, args).map_err(|e| e.to_string()),
        "perseus_vault_synthesize" => tools::handle_synthesize(db, args).map_err(|e| e.to_string()),
        "perseus_vault_bench" => tools::handle_bench(db, args).map_err(|e| e.to_string()),

        "perseus_vault_communities" => {
            tools::handle_communities(db, args).map_err(|e| e.to_string())
        }
        "perseus_vault_community_summary" => {
            tools::handle_community_summary(db, args).map_err(|e| e.to_string())
        }
        "perseus_vault_global_recall" => {
            tools::handle_global_recall(db, args).map_err(|e| e.to_string())
        }

        "perseus_vault_autocohere" => tools::handle_autocohere(db, args).map_err(|e| e.to_string()),
        "perseus_vault_supersede" => tools::handle_supersede(db, args).map_err(|e| e.to_string()),
        "perseus_vault_consistency_audit" => tools::handle_consistency_audit(db, args),
        "perseus_vault_audit_ruling" => tools::handle_audit_ruling(db, args),
        "perseus_vault_maintenance" => {
            tools::handle_maintenance(db, args).map_err(|e| e.to_string())
        }

        _ => Err(format!("Unknown tool: {}", original_name)),
    };

    // MCP spec §3.3: tool failures must return isError:true in the result,
    // NOT a JSON-RPC protocol error (which is reserved for transport/protocol faults).
    match handler_result {
        Ok(text) => text,
        Err(err_msg) => serde_json::to_string(&json!({
            "content": [{"type": "text", "text": err_msg}],
            "isError": true
        }))
        .unwrap_or_else(|_| {
            format!(
                r#"{{"content":[{{"type":"text","text":"{}"}}],"isError":true}}"#,
                err_msg
            )
        }),
    }
}

fn error_response(id: Option<Value>, code: i64, message: &str) -> JsonRpcResponse {
    JsonRpcResponse {
        jsonrpc: "2.0".to_string(),
        id,
        result: None,
        error: Some(JsonRpcError {
            code,
            message: message.to_string(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    /// Tool names advertised by tools/list (the canonical registry).
    fn advertised_names() -> Vec<String> {
        tool_registry_base()
            .iter()
            .map(|t| t["name"].as_str().unwrap().to_string())
            .collect()
    }

    #[test]
    fn registry_and_advertised_manifest_are_unique_and_in_sync() {
        let base = tool_registry_base();
        let registry_names: Vec<&str> = base
            .iter()
            .map(|tool| tool["name"].as_str().expect("registry tool name"))
            .collect();
        let registry_set: std::collections::HashSet<&str> =
            registry_names.iter().copied().collect();
        assert_eq!(
            registry_set.len(),
            registry_names.len(),
            "registry names must be unique"
        );
        assert_eq!(
            registry_names.len(),
            175,
            "update public metadata when adding a tool"
        );

        let canonical = advertised_names();
        let canonical_set: std::collections::HashSet<&str> =
            canonical.iter().map(String::as_str).collect();
        assert_eq!(canonical.len(), registry_names.len());
        assert_eq!(
            canonical_set.len(),
            canonical.len(),
            "canonical names must be unique"
        );
        assert!(canonical
            .iter()
            .all(|name| name.starts_with("perseus_vault_")));
    }

    #[test]
    fn lean_profile_contains_only_canonical_registry_tools() {
        let lean = filter_registry_by_profile(tool_registry_base().clone(), ToolProfile::Lean);
        let names: std::collections::HashSet<String> = lean
            .iter()
            .map(|tool| tool["name"].as_str().expect("tool name").to_string())
            .collect();
        let expected: std::collections::HashSet<String> = [
            "perseus_vault_remember",
            "perseus_vault_recall",
            "perseus_vault_forget",
            "perseus_vault_correct",
            "perseus_vault_context",
            "perseus_vault_workspace_status",
            "perseus_vault_health",
        ]
        .into_iter()
        .map(str::to_string)
        .collect();

        assert_eq!(names, expected);
        assert_eq!(lean.len(), expected.len());
        assert!(names.iter().all(|name| advertised_names().contains(name)));
        assert!(!names.contains("perseus_vault_status"));
    }

    #[test]
    fn lean_workspace_status_is_scoped_to_the_transport_identity() {
        let db_path = std::env::temp_dir().join(format!(
            "perseus_vault-lean-status-{}.db",
            uuid::Uuid::new_v4()
        ));
        let db = Database::open(db_path.to_str().expect("temp db path")).expect("open temp db");
        db.workspace_bind("lean-agent", "workspace-a", "read_write", "{}", "test")
            .expect("bind lean agent");
        db.workspace_bind("other-agent", "workspace-b", "read_write", "{}", "test")
            .expect("bind other agent");

        let raw = tools::handle_workspace_status(
            &db,
            json!({
                "status_scope": "caller",
                "requesting_agent_id": "lean-agent"
            }),
        )
        .expect("scoped status");
        let value: Value = serde_json::from_str(&raw).expect("status JSON");
        assert_eq!(value["count"], json!(1), "got: {raw}");
        assert_eq!(value["bindings"][0]["profile_name"], json!("lean-agent"));
        assert_eq!(value["bindings"][0]["workspace_hash"], json!("workspace-a"));
        assert!(
            !raw.contains("other-agent"),
            "cross-profile metadata leaked: {raw}"
        );
        assert!(
            !raw.contains("workspace-b"),
            "cross-workspace metadata leaked: {raw}"
        );

        let _ = fs::remove_file(&db_path);
    }

    #[test]
    fn journal_scope_attribution_is_advertised_in_schemas() {
        let canonical = tool_registry_base();
        for name in ["perseus_vault_journal", "perseus_vault_timeline"] {
            let tool = canonical
                .iter()
                .find(|tool| tool["name"] == name)
                .unwrap_or_else(|| panic!("missing {name}"));
            if name.ends_with("journal") {
                assert_eq!(
                    tool["inputSchema"]["properties"]["workspace_hash"]["type"],
                    "string"
                );
            } else {
                assert_eq!(
                    tool["outputSchema"]["properties"]["items"]["items"]["properties"]
                        ["workspace_hash"]["type"],
                    "string"
                );
            }
        }
    }

    #[test]
    fn context_sufficiency_schema_is_advertised_with_bounded_hash_only_receipt() {
        let context = tool_registry_base()
            .iter()
            .find(|tool| tool["name"] == "perseus_vault_context")
            .expect("context tool must be registered");
        let input = &context["inputSchema"]["properties"]["evidence_requirements"];
        assert_eq!(input["type"], "object");
        assert_eq!(input["additionalProperties"], false);
        assert_eq!(
            input["properties"]["fallback_policy"]["enum"],
            json!(["abstain", "canonical_retrieval"])
        );
        assert_eq!(input["properties"]["required_evidence"]["maxItems"], 256);
        let output = &context["outputSchema"]["properties"]["sufficiency"];
        assert_eq!(output["type"], "object");
        assert_eq!(
            output["properties"]["outcome"]["enum"],
            json!([
                "complete",
                "partial",
                "degraded",
                "abstained",
                "unavailable"
            ])
        );
        assert_eq!(
            output["properties"]["receipt"]["properties"]["query_sha256"]["pattern"],
            "^[0-9a-f]{64}$"
        );
        assert_eq!(
            output["properties"]["latest"]["$ref"],
            "#/$defs/sufficiency_coverage"
        );
        assert_eq!(
            context["outputSchema"]["$defs"]["sufficiency_coverage"]["required"],
            json!(["required", "selected", "missing"])
        );
    }

    #[test]
    fn answer_outcome_contract_is_advertised_for_answer_surfaces() {
        let registry = tool_registry_base();
        let answer_surfaces = [
            ("perseus_vault_recall", "answer_outcome"),
            ("perseus_vault_recall_batch", "answer_outcome"),
            ("perseus_vault_semantic_search", "answer_outcome"),
            ("perseus_vault_recall_when", "answer_outcome"),
            ("perseus_vault_context", "outcome"),
            ("perseus_vault_project_task", "outcome"),
        ];
        let expected_statuses = json!([
            "complete",
            "partial",
            "degraded",
            "abstained",
            "unavailable"
        ]);
        let expected_recall_statuses = json!([
            "fresh",
            "partial",
            "timeout",
            "unavailable",
            "empty",
            "stale"
        ]);
        for (name, field) in answer_surfaces {
            let tool = registry
                .iter()
                .find(|tool| tool["name"] == name)
                .unwrap_or_else(|| panic!("missing {name}"));
            let schema = &tool["outputSchema"]["properties"][field];
            assert_eq!(schema["type"], "object", "{name}.{field}");
            assert_eq!(
                schema["properties"]["schema_version"]["const"], "perseus-vault-answer-outcome/v1",
                "{name}.{field} schema version"
            );
            assert_eq!(
                schema["properties"]["status"]["enum"], expected_statuses,
                "{name}.{field}"
            );
            assert_eq!(
                schema["properties"]["recall_status"]["enum"], expected_recall_statuses,
                "{name}.{field} recall status"
            );
            assert_eq!(
                schema["properties"]["reason_codes"]["type"], "array",
                "{name}.{field}"
            );
            assert_eq!(
                schema["properties"]["abstained"]["type"], "boolean",
                "{name}.{field}"
            );
            assert_eq!(
                schema["properties"]["answerable"]["type"], "boolean",
                "{name}.{field}"
            );
            assert_eq!(
                schema["properties"]["fallback"]["type"], "object",
                "{name}.{field}"
            );
        }
        for name in [
            "perseus_vault_recall_batch",
            "perseus_vault_semantic_search",
            "perseus_vault_recall_when",
            "perseus_vault_context",
        ] {
            let tool = registry
                .iter()
                .find(|tool| tool["name"] == name)
                .unwrap_or_else(|| panic!("missing {name}"));
            assert_eq!(
                tool["inputSchema"]["properties"]["include_outcome"]["type"], "boolean",
                "{name} must advertise include_outcome"
            );
        }
    }

    #[test]
    fn answer_outcome_wire_contract_has_bounded_fields_and_binding_metadata() {
        let registry = tool_registry_base();
        let surfaces = [
            ("perseus_vault_recall", "answer_outcome"),
            ("perseus_vault_recall_batch", "answer_outcome"),
            ("perseus_vault_semantic_search", "answer_outcome"),
            ("perseus_vault_recall_when", "answer_outcome"),
            ("perseus_vault_context", "outcome"),
            ("perseus_vault_project_task", "outcome"),
        ];
        for (name, field) in surfaces {
            let tool = registry
                .iter()
                .find(|tool| tool["name"] == name)
                .unwrap_or_else(|| panic!("missing {name}"));
            let outcome = &tool["outputSchema"]["properties"][field];
            assert_eq!(
                outcome["properties"]["reason"]["minLength"],
                json!(1),
                "{name}.{field}"
            );
            assert_eq!(
                outcome["properties"]["reason_codes"]["minItems"],
                json!(1),
                "{name}.{field}"
            );
            assert_eq!(
                outcome["properties"]["reason_codes"]["maxItems"],
                json!(16),
                "{name}.{field}"
            );
            assert_eq!(
                outcome["properties"]["fallback"]["properties"]["mode"]["enum"],
                json!(["abstain", "canonical_retrieval"]),
                "{name}.{field} fallback mode"
            );
            assert_eq!(
                outcome["properties"]["fallback"]["properties"]["reason"]["minLength"],
                json!(1),
                "{name}.{field} fallback reason"
            );

            let input = &tool["inputSchema"]["properties"];
            if let Some(description) = input
                .get("include_outcome")
                .and_then(|value| value["description"].as_str())
            {
                let description = description.to_ascii_lowercase();
                assert!(
                    description.contains("complete"),
                    "{name} include_outcome must cover complete results"
                );
                assert!(
                    description.contains("unavailable"),
                    "{name} include_outcome must cover unavailable results"
                );
            }
            let workspace = input.get("workspace_hash").or_else(|| {
                input
                    .get("queries")
                    .and_then(|value| value["items"]["properties"].get("workspace_hash"))
            });
            if let Some(description) = workspace.and_then(|value| value["description"].as_str()) {
                let description = description.to_ascii_lowercase();
                assert!(
                    description.contains("compatibility"),
                    "{name} workspace compatibility must be explicit"
                );
                assert!(
                    description.contains("strict"),
                    "{name} workspace strict mode must be explicit"
                );
            }
        }

        let recall_when = registry
            .iter()
            .find(|tool| tool["name"] == "perseus_vault_recall_when")
            .expect("recall_when must be registered");
        assert_eq!(
            recall_when["inputSchema"]["properties"]["requesting_agent_id"]["type"], "string",
            "recall_when must advertise the transport requester"
        );
    }

    #[test]
    fn stats_schema_allows_null_timestamps_for_an_empty_database() {
        let stats = tool_registry_base()
            .iter()
            .find(|tool| tool["name"] == "perseus_vault_stats")
            .expect("stats tool must be registered");

        for field in ["oldest_unix_ms", "newest_unix_ms"] {
            assert_eq!(
                stats["outputSchema"]["properties"][field]["type"],
                json!(["integer", "null"]),
                "{field} must accept the null value returned for an empty database"
            );
        }
    }

    #[test]
    fn dream_is_registered_and_errors_cleanly_without_llm() {
        assert!(advertised_names().contains(&"perseus_vault_dream".to_string()));

        let db_path =
            std::env::temp_dir().join(format!("perseus_vault-dream-{}.db", uuid::Uuid::new_v4()));
        let db = Database::open(db_path.to_str().expect("temp db path")).expect("open temp db");

        // No --llm-endpoint configured: the tool must answer with a clean MCP
        // tool error (isError, spec §3.3) — never a crash or protocol error —
        // and the message must name the flag and the non-LLM alternative.
        let r = call_tool(
            "perseus_vault_dream",
            &db,
            json!({"category": "episodes"}),
            None,
        );
        let v: Value = serde_json::from_str(&r).unwrap();
        if &(v["isError"]) != &(json!(true)) {
            panic!("test assertion failed");
        };
        let msg = v["content"][0]["text"].as_str().unwrap();
        assert!(msg.contains("--llm-endpoint"), "got: {msg}");
        assert!(msg.contains("perseus_vault_consolidate"), "got: {msg}");

        // Opt-in graceful degradation: fallback_consolidate runs the non-LLM
        // consolidate pass instead of erroring, and says so.
        let r = call_tool(
            "perseus_vault_dream",
            &db,
            json!({"fallback_consolidate": true, "dry_run": true}),
            None,
        );
        let v: Value = serde_json::from_str(&r).unwrap();
        if &(v["fallback"]) != &(json!("consolidate")) {
            panic!("test assertion failed");
        };
        assert_eq!(v["dry_run"], json!(true));

        let _ = fs::remove_file(&db_path);
    }

    #[test]
    fn disabled_federate_is_not_advertised() {
        assert!(
            !advertised_names().contains(&"perseus_vault_federate".to_string()),
            "disabled federation must not appear in tools/list"
        );
    }

    #[test]
    fn check_failure_pattern_is_registered_and_dispatches() {
        // #521: tools/list must expose the deja-vu guard under the canonical name.
        assert!(advertised_names().contains(&"perseus_vault_check_failure_pattern".to_string()));

        let db_path =
            std::env::temp_dir().join(format!("perseus_vault-fpguard-{}.db", uuid::Uuid::new_v4()));
        let db = Database::open(db_path.to_str().expect("temp db path")).expect("open temp db");

        // Alias prefixes normalize into the same handler; empty store answers
        // with the unambiguous empty state.
        let r = call_tool(
            "perseus_vault_check_failure_pattern",
            &db,
            json!({"action": "cargo build --release", "workspace_hash": ""}),
            None,
        );
        let v: Value = serde_json::from_str(&r).unwrap();
        if &(v["deja_vu"]) != &(json!(false)) {
            panic!("test assertion failed");
        };
        if !(v["message"]
            .as_str()
            .unwrap()
            .contains("no prior failures recorded matching this action"))
        {
            panic!("test assertion failed");
        };

        // Missing required `action` → clean MCP tool error (isError, §3.3).
        let r = call_tool("perseus_vault_check_failure_pattern", &db, json!({}), None);
        let v: Value = serde_json::from_str(&r).unwrap();
        if &(v["isError"]) != &(json!(true)) {
            panic!("test assertion failed");
        };

        let _ = fs::remove_file(&db_path);
    }

    #[test]
    fn capture_is_registered_and_dispatches() {
        // #520: tools/list must expose the capture pipeline under the
        // canonical name.
        assert!(advertised_names().contains(&"perseus_vault_capture".to_string()));

        let db_path =
            std::env::temp_dir().join(format!("perseus_vault-capture-{}.db", uuid::Uuid::new_v4()));
        let db = Database::open(db_path.to_str().expect("temp db path")).expect("open temp db");

        // A real payload distills and writes through the remember path.
        let r = call_tool(
            "perseus_vault_capture",
            &db,
            json!({"text": "The deploy failed because the schema version was never bumped."}),
            None,
        );
        let v: Value = serde_json::from_str(&r).unwrap();
        if &(v["captured"]) != &(json!(1)) {
            panic!("test assertion failed");
        };
        if &(v["created"]) != &(json!(1)) {
            panic!("test assertion failed");
        };
        if &(v["notes"][0]["type"]) != &(json!("root-cause")) {
            panic!("test assertion failed");
        };

        // Empty payload → clean MCP tool error (isError, spec §3.3).
        let r = call_tool("perseus_vault_capture", &db, json!({"text": "  "}), None);
        let v: Value = serde_json::from_str(&r).unwrap();
        if &(v["isError"]) != &(json!(true)) {
            panic!("test assertion failed");
        };

        let _ = fs::remove_file(&db_path);
    }

    #[test]
    fn memories_adapter_full_lifecycle_roundtrip() {
        // The Anthropic /memories directory convention over vault entities:
        // create, list, view (numbered), str_replace (unique-match), insert,
        // rename, delete, and recreate-after-delete (revival must also
        // restore the FTS row so the file is searchable again).
        let db_path = std::env::temp_dir().join(format!(
            "perseus_vault-memories-{}.db",
            uuid::Uuid::new_v4()
        ));
        let db = Database::open(db_path.to_str().expect("temp db path")).expect("open temp db");
        let call = |args: Value| -> String { call_tool("perseus_vault_memories", &db, args, None) };

        // create
        let r = call(json!({"command": "create", "path": "/memories/notes.md",
                            "file_text": "alpha\nbeta\ngamma"}));
        if !(r.contains("created")) {
            panic!("test assertion failed");
        };

        // view directory
        let r = call(json!({"command": "view", "path": "/memories"}));
        let v: Value = serde_json::from_str(&r).unwrap();
        if &(v["files"]) != &(json!(["notes.md"])) {
            panic!("test assertion failed");
        };

        // view file — numbered content
        let r = call(json!({"command": "view", "path": "/memories/notes.md"}));
        if !(r.contains("beta")) {
            panic!("test assertion failed");
        };
        let v: Value = serde_json::from_str(&r).unwrap();
        if !(v["content"].as_str().unwrap().contains("     2\tbeta")) {
            panic!("test assertion failed");
        };

        // str_replace — must reject ambiguous and missing matches
        let r = call(
            json!({"command": "str_replace", "path": "/memories/notes.md",
                            "old_str": "beta", "new_str": "BETA"}),
        );
        if !(r.contains("replaced")) {
            panic!("test assertion failed");
        };
        let r = call(
            json!({"command": "str_replace", "path": "/memories/notes.md",
                            "old_str": "missing", "new_str": "x"}),
        );
        if !(r.contains("not found")) {
            panic!("test assertion failed");
        };

        // insert at line 0
        let r = call(json!({"command": "insert", "path": "/memories/notes.md",
                            "insert_line": 0, "insert_text": "header"}));
        if !(r.contains("inserted")) {
            panic!("test assertion failed");
        };
        let r = call(json!({"command": "view", "path": "/memories/notes.md"}));
        let v: Value = serde_json::from_str(&r).unwrap();
        if !(v["content"].as_str().unwrap().starts_with("     1\theader")) {
            panic!("test assertion failed");
        };

        // rename
        let r = call(
            json!({"command": "rename", "old_path": "/memories/notes.md",
                            "new_path": "/memories/archive/notes.md"}),
        );
        if !(r.contains("renamed")) {
            panic!("test assertion failed");
        };
        let r = call(json!({"command": "view", "path": "/memories"}));
        let v: Value = serde_json::from_str(&r).unwrap();
        if &(v["files"]) != &(json!(["archive/notes.md"])) {
            panic!("test assertion failed");
        };

        // path traversal is rejected
        let r = call(json!({"command": "view", "path": "/memories/../etc/passwd"}));
        if !(r.contains("invalid path") || r.contains("error")) {
            panic!("test assertion failed");
        };

        // delete, then recreate: revival must restore searchability (the FTS
        // row is deleted by forget; the remember update path must re-insert it).
        let r = call(json!({"command": "delete", "path": "/memories/archive/notes.md"}));
        if !(r.contains("deleted")) {
            panic!("test assertion failed");
        };
        let r = call(
            json!({"command": "create", "path": "/memories/archive/notes.md",
                            "file_text": "reborn searchable zanzibar"}),
        );
        if !(r.contains("created")) {
            panic!("test assertion failed");
        };
        let hits = db
            .recall(&crate::models::RecallParams {
                query: "zanzibar".to_string(),
                skip_side_effects: true,
                ..crate::models::RecallParams::default()
            })
            .unwrap();
        assert!(
            hits.iter().any(|e| e.key == "archive/notes.md"),
            "revived file must be FTS-searchable again"
        );

        drop(db);
        let _ = fs::remove_file(&db_path);
    }

    #[test]
    fn bitemporal_tools_are_registered_and_dispatch() {
        // #363: perseus_vault_valid_at / perseus_vault_bitemporal exist in the
        // registry and dispatch through call_tool.
        let db_path = std::env::temp_dir().join(format!(
            "perseus_vault-bitemporal-tools-{}.db",
            uuid::Uuid::new_v4()
        ));
        let db = Database::open(db_path.to_str().expect("temp db path")).expect("open temp db");

        let names = advertised_names();
        for expect in ["perseus_vault_valid_at", "perseus_vault_bitemporal"] {
            assert!(names.contains(&expect.to_string()), "missing tool {expect}");
        }

        // Round-trip through call_tool.
        let stored = call_tool(
            "perseus_vault_remember",
            &db,
            json!({"category": "f", "key": "k", "body_json": "{\"note\":\"x\"}",
                   "valid_from_unix_ms": 1000}),
            None,
        );
        if !(stored.contains("created")) {
            panic!("test assertion failed");
        };
        for prefix in ["perseus_vault"] {
            let r = call_tool(
                &format!("{prefix}_valid_at"),
                &db,
                json!({"category": "f", "key": "k", "valid_at_unix_ms": 2000}),
                None,
            );
            if !(r.contains("\"found\":true")) {
                panic!("test assertion failed");
            };
            let b = call_tool(
                &format!("{prefix}_bitemporal"),
                &db,
                json!({"category": "f", "key": "k",
                       "tx_at_unix_ms": now_ms_for_test(), "valid_at_unix_ms": 2000}),
                None,
            );
            if !(b.contains("\"found\":true")) {
                panic!("test assertion failed");
            };
        }

        let _ = fs::remove_file(&db_path);
    }

    fn now_ms_for_test() -> i64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as i64
    }

    #[test]
    fn keystone_tools_register_dispatch_order_and_gate() {
        // #683: keystones are registered, round-trip through
        // call_tool, merge by weight, are updated in place on re-set, gate on
        // trust tier, and every mutation lands on the audit chain.
        let db_path = std::env::temp_dir().join(format!(
            "perseus_vault-keystones-{}.db",
            uuid::Uuid::new_v4()
        ));
        let db = Database::open(db_path.to_str().expect("temp db path")).expect("open temp db");

        // Registered under the canonical prefix.
        for expect in ["perseus_vault_keystone_set", "perseus_vault_keystone_get"] {
            assert!(
                advertised_names().contains(&expect.to_string()),
                "missing tool {expect}"
            );
        }

        // Author two keystones with different weights (author tier satisfies).
        let low = call_tool(
            "perseus_vault_keystone_set",
            &db,
            json!({"content": "cite source memory IDs", "scope": "tenant",
                   "weight": 1.0, "author_trust_tier": 2}),
            None,
        );
        if !(low.contains("\"created\":true")) {
            panic!("test assertion failed");
        };
        // #684: caller-asserted (no registered agent) → checked but not
        // registry-enforced. trust_enforced reflects registry backing only.
        if !(low.contains("\"trust_enforced\":false")) {
            panic!("test assertion failed");
        };
        let _ = call_tool(
            "perseus_vault_keystone_set",
            &db,
            json!({"content": "PII MUST NOT cross agent boundaries", "scope": "fleet",
                   "scope_id": "sec", "weight": 9.0, "author_trust_tier": 3}),
            None,
        );

        // get merges both, highest weight first.
        let got = call_tool("perseus_vault_keystone_get", &db, json!({}), None);
        let v: Value = serde_json::from_str(&got).unwrap();
        if &(v["count"]) != &(json!(2)) {
            panic!("test assertion failed");
        };
        assert_eq!(
            v["keystones"][0]["content"],
            json!("PII MUST NOT cross agent boundaries")
        );
        assert_eq!(
            v["keystones"][1]["content"],
            json!("cite source memory IDs")
        );

        // Re-setting the same (scope, scope_id, content) updates in place.
        let again = call_tool(
            "perseus_vault_keystone_set",
            &db,
            json!({"content": "cite source memory IDs", "scope": "tenant",
                   "weight": 5.0, "author_trust_tier": 2}),
            None,
        );
        if !(again.contains("\"created\":false")) {
            panic!("test assertion failed");
        };
        let got2 = call_tool("perseus_vault_keystone_get", &db, json!({}), None);
        let v2: Value = serde_json::from_str(&got2).unwrap();
        if &(v2["count"]) != &(json!(2)) {
            panic!("test assertion failed");
        };

        // Trust gate: asserting tier below required is rejected.
        let denied = call_tool(
            "perseus_vault_keystone_set",
            &db,
            json!({"content": "denied rule", "author_trust_tier": 1}),
            None,
        );
        if !(denied.contains("insufficient trust tier")) {
            panic!("test assertion failed");
        };
        // Omitting the tier is allowed but flagged as unenforced.
        let unenforced = call_tool(
            "perseus_vault_keystone_set",
            &db,
            json!({"content": "unenforced rule", "scope": "agent", "scope_id": "a1"}),
            None,
        );
        if !(unenforced.contains("\"trust_enforced\":false")) {
            panic!("test assertion failed");
        };

        // Every keystone_set is crypto-chained (event_type keystone_set) and the
        // chain still verifies.
        let chained: i64 = db
            .conn()
            .unwrap()
            .query_row(
                "SELECT COUNT(*) FROM journal WHERE event_type = 'keystone_set'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(chained, 4, "one audit event per successful set (3 create + 1 update); the trust-denied set emits none");
        assert!(
            crate::db::verify_audit_chain(&db).is_ok(),
            "audit chain must verify after keystone mutations"
        );

        let _ = fs::remove_file(&db_path);
    }

    #[test]
    fn stage_trace_tool_validates_hash_only_replay_contract() {
        let db_path =
            std::env::temp_dir().join(format!("perseus-stage-trace-{}.db", uuid::Uuid::new_v4()));
        let db = Database::open(db_path.to_str().expect("temp db path")).expect("open temp db");
        let trace = crate::stage_trace::StageTrace::new("trace-mcp", "workspace-a")
            .seal()
            .expect("empty trace is a valid fixture");
        let response = call_tool(
            "perseus_vault_stage_trace_validate",
            &db,
            json!({
                "trace": serde_json::to_value(&trace).unwrap(),
                "replay_of": serde_json::to_value(&trace).unwrap()
            }),
            None,
        );
        let value: Value = serde_json::from_str(&response).expect("structured response");
        if &(value["valid"]) != &(true) {
            panic!("test assertion failed");
        };
        if &(value["replay_match"]) != &(true) {
            panic!("test assertion failed");
        };
        assert!(advertised_names().contains(&"perseus_vault_stage_trace_validate".to_string()));
        let _ = fs::remove_file(db_path);
    }

    #[test]
    fn context_transform_tool_returns_hash_only_outcome_and_is_advertised() {
        let db_path = std::env::temp_dir().join(format!(
            "perseus-context-transform-{}.db",
            uuid::Uuid::new_v4()
        ));
        let db = Database::open(db_path.to_str().expect("temp db path")).expect("open temp db");
        let input = vec![crate::context_transform::ContextMessage::new(
            "m-1",
            0,
            "assistant_prose",
            json!({"role": "assistant", "content": "MCP_RAW_SENTINEL"}),
        )];
        let request = crate::context_transform::ContextTransformRequest::new(
            "openai",
            crate::context_transform::SUPPORTED_OPENAI_REQUEST_FORMAT,
            crate::context_transform::TransformerDescriptor::new("fixture", "1"),
            vec![crate::context_transform::TransformStage::new(
                "distill", "1", true, "trusted", None,
            )],
            "lossy_opt_in",
            input.clone(),
        )
        .with_input_tokens(Some(20));
        let mut proposed = input;
        proposed[0].message["content"] = json!("short");
        let response = call_tool(
            "perseus_vault_context_transform_validate",
            &db,
            json!({
                "request": serde_json::to_value(&request).unwrap(),
                "proposed_output": serde_json::to_value(&proposed).unwrap(),
                "proposed_output_tokens": 3
            }),
            None,
        );
        let value: Value = serde_json::from_str(&response).expect("structured response");
        if &(value["valid"]) != &(true) {
            panic!("test assertion failed");
        };
        if &(value["outcome"]) != &("degraded") {
            panic!("test assertion failed");
        };
        if &(value["receipt"]["actual_lossiness"]) != &("lossy") {
            panic!("test assertion failed");
        };
        assert_eq!(value["receipt"]["input_digest"].as_str().unwrap().len(), 64);
        if !(!response.contains("MCP_RAW_SENTINEL")) {
            panic!("test assertion failed");
        };
        if !(!response.contains("\"message\"")) {
            panic!("test assertion failed");
        };
        assert!(
            advertised_names().contains(&"perseus_vault_context_transform_validate".to_string())
        );
        let _ = fs::remove_file(db_path);
    }

    #[test]
    fn rejected_value_tombstones_block_laundering_and_support_audited_override() {
        let db_path = std::env::temp_dir().join(format!(
            "perseus_vault-rejected-value-{}.db",
            uuid::Uuid::new_v4()
        ));
        let db = Database::open(db_path.to_str().expect("temp db path")).expect("open temp db");

        // Reject the value explicitly with the new tool. The value is the
        // canonical body string the writer would store (digest matching is
        // case/whitespace-insensitive, so any equivalent spelling matches).
        let reject = call_tool(
            "perseus_vault_reject_value",
            &db,
            json!({
                "workspace_hash": "ws-a",
                "subject": "never-use-x",
                "predicate": "convention",
                "value": "{\"note\": \"Always use X for everything\"}",
                "reason": "user correction",
                "author_agent_id": "test-agent"
            }),
            None,
        );
        if !(reject.contains("\"rejected\":true")) {
            panic!("test assertion failed");
        };

        // A normal remember with the same body is blocked, even under a
        // brand-new key and different (subject, predicate) spelling: that is
        // the laundering prevention (the gate matches on value digest within
        // the workspace scope).
        let blocked = call_tool(
            "perseus_vault_remember",
            &db,
            json!({
                "category": "convention",
                "key": "totally-different-key",
                "workspace_hash": "ws-a",
                "body_json": "{\"note\": \"Always use X for everything\"}"
            }),
            None,
        );
        if !(blocked.contains("rejected")) {
            panic!("test assertion failed");
        };

        // Case/whitespace variants of the rejected value are equivalent.
        let blocked_variant = call_tool(
            "perseus_vault_remember",
            &db,
            json!({
                "category": "convention",
                "key": "another-key",
                "workspace_hash": "ws-a",
                "body_json": "{ \"note\":  \"always  use x  for everything\" }"
            }),
            None,
        );
        if !(blocked_variant.contains("rejected")) {
            panic!("test assertion failed");
        };

        // A different value for the same key is NOT blocked.
        let fine = call_tool(
            "perseus_vault_remember",
            &db,
            json!({
                "category": "convention",
                "key": "totally-different-key",
                "workspace_hash": "ws-a",
                "body_json": "{\"note\": \"Use Y instead\"}"
            }),
            None,
        );
        if !(fine.contains("\"action\":\"created\"")) {
            panic!("test assertion failed");
        };

        // Scope isolation: the same value re-ingested in a DIFFERENT workspace
        // is not poisoned by the ws-a tombstone.
        let other_ws = call_tool(
            "perseus_vault_remember",
            &db,
            json!({
                "category": "convention",
                "key": "ws-b-key",
                "workspace_hash": "ws-b",
                "body_json": "{\"note\": \"Always use X for everything\"}"
            }),
            None,
        );
        if !(other_ws.contains("\"action\":\"created\"")) {
            panic!("test assertion failed");
        };

        // The deliberate override writes through and is journaled.
        let override_ok = call_tool(
            "perseus_vault_remember",
            &db,
            json!({
                "category": "convention",
                "key": "never-use-x",
                "workspace_hash": "ws-a",
                "body_json": "{\"note\": \"Always use X for everything\"}",
                "allow_rejected": true
            }),
            None,
        );
        if !(override_ok.contains("\"action\":\"created\"")) {
            panic!("test assertion failed");
        };

        let rejected_events: i64 = {
            let conn = db.conn().expect("db connection");
            conn.query_row(
                "SELECT COUNT(*) FROM journal WHERE event_type = 'rejected_write'",
                [],
                |r| r.get(0),
            )
            .unwrap_or(0)
        };
        assert!(rejected_events >= 1, "rejected write must be journaled");

        let override_events: i64 = {
            let conn = db.conn().expect("db connection");
            conn.query_row(
                "SELECT COUNT(*) FROM journal WHERE event_type = 'rejected_write_override'",
                [],
                |r| r.get(0),
            )
            .unwrap_or(0)
        };
        assert!(
            override_events >= 1,
            "trusted override must be journaled for audit"
        );

        // Correcting a wrong approach records a tombstone and the correction
        // itself still writes.
        let corrected = call_tool(
            "perseus_vault_correct",
            &db,
            json!({
                "workspace_hash": "ws-a",
                "wrong_approach": "Used X everywhere",
                "user_correction": "Prefer Y",
                "task_context": "choose strategy",
                "agent_id": "test-agent"
            }),
            None,
        );
        if !(corrected.contains("entity_id")) {
            panic!("test assertion failed");
        };
        let corr_val: Value = serde_json::from_str(&corrected).expect("correction response JSON");
        let corr_id = corr_val["entity_id"]
            .as_str()
            .expect("entity_id")
            .to_string();
        let corr_key = format!("correction-{}", &corr_id[4..16]);
        assert!(
            db.is_value_rejected("ws-a", &corr_key, "correction", "Used X everywhere")
                .unwrap(),
            "correction must leave a scoped tombstone on its own key"
        );

        let _ = fs::remove_file(db_path);
    }

    #[test]
    fn unknown_tool_error_reports_the_caller_name() {
        let db_path = std::env::temp_dir().join(format!(
            "perseus_vault-unknown-tool-{}.db",
            uuid::Uuid::new_v4()
        ));
        let db = Database::open(db_path.to_str().expect("temp db path")).expect("open temp db");

        // An unknown tool name is reported verbatim — no prefix normalization.
        let result = call_tool("perseus_vault_bogus", &db, json!({}), None);
        if !(result.contains("Unknown tool: perseus_vault_bogus")) {
            panic!("test assertion failed");
        };

        let other = call_tool("custom_bogus", &db, json!({}), None);
        if !(other.contains("Unknown tool: custom_bogus")) {
            panic!("test assertion failed");
        };

        let _ = fs::remove_file(&db_path);
    }

    #[test]
    fn handoff_state_resumes_initialized_session_and_identity() {
        // #1045: the replacement process must consider itself initialized
        // (the client never re-sends `initialize`) and must carry the
        // transport-captured agent identity so visibility-scoped tools keep
        // working.
        let state = MCPState::new();
        assert!(!state.initialized.load(std::sync::atomic::Ordering::Relaxed));
        assert_eq!(*state.session_agent_id.read().unwrap(), "");

        apply_handoff_state(
            &state,
            crate::live_update::HandoffState {
                initialized: true,
                session_agent_id: "rovo-dev-agent".to_string(),
            },
        );
        assert!(state.initialized.load(std::sync::atomic::Ordering::Relaxed));
        assert_eq!(*state.session_agent_id.read().unwrap(), "rovo-dev-agent");
        // tools/call must dispatch (not -32002) and stamp the identity.
        assert!(
            state.session_agent_id.read().unwrap().len() > 0,
            "identity must be restored"
        );
    }

    #[test]
    fn handoff_state_without_init_flag_leaves_session_uninitialized() {
        // A malformed/partial forwarded state must not silently authorize the
        // session: initialized:false means the client must still initialize.
        let state = MCPState::new();
        apply_handoff_state(
            &state,
            crate::live_update::HandoffState {
                initialized: false,
                session_agent_id: String::new(),
            },
        );
        assert!(!state.initialized.load(std::sync::atomic::Ordering::Relaxed));
        assert_eq!(*state.session_agent_id.read().unwrap(), "");
    }

    #[test]
    fn rejects_non_json_rpc_2_requests() {
        let db_path = std::env::temp_dir().join(format!(
            "perseus_vault-jsonrpc-version-{}.db",
            uuid::Uuid::new_v4()
        ));
        let db = Database::open(db_path.to_str().expect("temp db path")).expect("open temp db");
        let req = JsonRpcRequest {
            jsonrpc: "1.0".to_string(),
            id: Some(json!(1)),
            method: "initialize".to_string(),
            params: None,
        };
        let state = MCPState::new();

        let resp = handle_request(&req, &state, &db).expect("error response");
        assert_eq!(resp.error.expect("json-rpc error").code, -32600);
        assert!(!state.initialized.load(std::sync::atomic::Ordering::Relaxed));

        let _ = fs::remove_file(db_path);
    }

    #[test]
    fn initialize_reports_the_current_crate_name_not_a_hardcoded_one() {
        // Regression: serverInfo.name was a hardcoded literal that went stale
        // across the earlier product renames. It must track Cargo.toml's
        // package name instead.
        let db_path = std::env::temp_dir().join(format!(
            "perseus_vault-initialize-name-{}.db",
            uuid::Uuid::new_v4()
        ));
        let db = Database::open(db_path.to_str().expect("temp db path")).expect("open temp db");
        let req = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(json!(1)),
            method: "initialize".to_string(),
            params: None,
        };
        let state = MCPState::new();

        let resp = handle_request(&req, &state, &db).expect("initialize response");
        let result = resp.result.expect("initialize result");
        assert_eq!(result["serverInfo"]["name"], json!(env!("CARGO_PKG_NAME")),);

        let _ = fs::remove_file(db_path);
    }

    #[test]
    fn initialize_captures_client_identity_and_scopes_recall() {
        // #684: the initialize handshake's clientInfo.name is captured and
        // stamped onto tool calls as requesting_agent_id, so a private entity is
        // transparently hidden from a different client — no explicit arg needed.
        let db_path = std::env::temp_dir().join(format!(
            "perseus_vault-clientinfo-{}.db",
            uuid::Uuid::new_v4()
        ));
        let db = Database::open(db_path.to_str().expect("temp db path")).expect("open temp db");
        db.agent_upsert("alice", "Alice", 0, "eng").unwrap();
        db.agent_upsert("bob", "Bob", 0, "eng").unwrap();
        tools::handle_remember(
            &db,
            json!({"category": "notes", "key": "secret",
                   "body_json": "{\"note\":\"quantum blueprint\"}",
                   "visibility": "private", "agent_id": "alice"}),
        )
        .expect("private note");

        let state = MCPState::new();
        // Handshake as client "bob".
        let init = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(json!(1)),
            method: "initialize".to_string(),
            params: Some(json!({"clientInfo": {"name": "bob", "version": "1.0"}})),
        };
        handle_request(&init, &state, &db).expect("initialize");
        assert_eq!(
            *state.session_agent_id.read().unwrap(),
            "bob",
            "clientInfo.name must be captured"
        );

        // A plain recall (no requesting_agent_id arg) is transparently scoped to bob.
        let call = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(json!(2)),
            method: "tools/call".to_string(),
            params: Some(json!({
                "name": "perseus_vault_recall",
                "arguments": {"query": "quantum", "mode": "fts5"}
            })),
        };
        let resp = handle_request(&call, &state, &db).expect("recall response");
        let structured = resp.result.expect("result")["structuredContent"].clone();
        assert_eq!(
            structured["total"],
            json!(0),
            "bob must not see alice's private note via the captured identity: {structured}"
        );
        let _ = fs::remove_file(db_path);
    }

    #[test]
    fn transport_host_identity_overrides_forged_requesting_agent_id() {
        // #855 review: even when the caller forges a requesting_agent_id
        // (or passes an empty one), the transport overwrites it with the
        // captured clientInfo.name — a model cannot claim another identity.
        let db_path = std::env::temp_dir().join(format!(
            "perseus_vault-forged-id-{}.db",
            uuid::Uuid::new_v4()
        ));
        let db = Database::open(db_path.to_str().expect("temp db path")).expect("open temp db");

        let state = MCPState::new();
        let init = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(json!(1)),
            method: "initialize".to_string(),
            params: Some(json!({"clientInfo": {"name": "host-bob", "version": "1.0"}})),
        };
        handle_request(&init, &state, &db).expect("initialize");

        // Forge: the model claims to be "mallory" in both author and host
        // fields. The transport must replace requesting_agent_id with host-bob.
        let call = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(json!(2)),
            method: "tools/call".to_string(),
            params: Some(json!({
                "name": "perseus_vault_correct",
                "arguments": {
                    "wrong_approach": "assistant guessed the state",
                    "user_correction": "user corrected",
                    "task_context": "review",
                    "agent_id": "mallory",
                    "requesting_agent_id": "mallory"
                }
            })),
        };
        let resp = handle_request(&call, &state, &db).expect("correct response");
        let structured = resp.result.expect("result")["structuredContent"].clone();
        assert_eq!(
            structured["agent_id"],
            json!("host-bob"),
            "host identity must override the forged id: {structured}"
        );
        let _ = fs::remove_file(db_path);
    }

    #[test]
    fn recall_confidence_is_opt_in_and_normalized() {
        let db_path = std::env::temp_dir().join(format!(
            "perseus_vault-confidence-{}.db",
            uuid::Uuid::new_v4()
        ));
        let db = Database::open(db_path.to_str().expect("temp db path")).expect("open temp db");

        tools::handle_remember(
            &db,
            json!({"category": "demo", "key": "k1", "body_json": "{\"content\":\"alpha bravo\"}"}),
        )
        .expect("remember");

        // Default: confidence is absent (opt-in, non-breaking).
        let plain = tools::handle_recall(&db, json!({"query": "alpha"})).expect("recall");
        let plain_v: Value = serde_json::from_str(&plain).unwrap();
        assert!(
            plain_v["items"][0].get("confidence").is_none(),
            "confidence must be opt-in"
        );

        // Opt-in: confidence present and normalized to [0,1].
        let withc =
            tools::handle_recall(&db, json!({"query": "alpha", "include_confidence": true}))
                .expect("recall");
        let withc_v: Value = serde_json::from_str(&withc).unwrap();
        let c = withc_v["items"][0]["confidence"]
            .as_f64()
            .expect("confidence number");
        assert!((0.0..=1.0).contains(&c), "confidence {} out of range", c);

        let _ = fs::remove_file(db_path);
    }

    #[test]
    fn history_tool_lists_superseded_versions() {
        let db_path =
            std::env::temp_dir().join(format!("perseus_vault-history-{}.db", uuid::Uuid::new_v4()));
        let db = Database::open(db_path.to_str().expect("temp db path")).expect("open temp db");

        tools::handle_remember(
            &db,
            json!({"category":"facts","key":"color","body_json":"{\"content\":\"blue\"}"}),
        )
        .expect("v1");
        // A content change snapshots the prior version into history.
        tools::handle_remember(
            &db,
            json!({"category":"facts","key":"color","body_json":"{\"content\":\"green\"}"}),
        )
        .expect("v2");

        let resp =
            tools::handle_history(&db, json!({"category":"facts","key":"color"})).expect("history");
        let v: Value = serde_json::from_str(&resp).unwrap();
        assert_eq!(
            v["total"].as_i64().unwrap(),
            1,
            "one superseded version: {}",
            resp
        );
        let body = v["versions"][0]["content"]
            .as_str()
            .or_else(|| v["versions"][0]["body_json"].as_str())
            .unwrap_or("");
        assert!(
            body.contains("blue"),
            "history should hold the old 'blue' value: {}",
            resp
        );

        // Unknown key -> empty trail.
        let empty =
            tools::handle_history(&db, json!({"category":"facts","key":"nope"})).expect("history");
        let ev: Value = serde_json::from_str(&empty).unwrap();
        assert_eq!(ev["total"].as_i64().unwrap(), 0);

        let _ = fs::remove_file(db_path);
    }

    #[test]
    fn graphrag_tools_dispatch_including_aliases() {
        // #365: the three GraphRAG tools must be dispatchable under the
        // canonical perseus_vault_* name and both rename aliases, and must appear in
        // tools/list.
        let db_path = std::env::temp_dir().join(format!(
            "perseus_vault-graphrag-{}.db",
            uuid::Uuid::new_v4()
        ));
        let db = Database::open(db_path.to_str().expect("temp db path")).expect("open temp db");

        // Two linked entities so detection has a community to find.
        tools::handle_remember(
            &db,
            json!({"category":"g","key":"n1","body_json":"{\"content\":\"quasar telescope\"}"}),
        )
        .expect("remember n1");
        tools::handle_remember(
            &db,
            json!({"category":"g","key":"n2","body_json":"{\"content\":\"nebula filter rig\"}"}),
        )
        .expect("remember n2");
        let n2 = db.get_entity("g", "n2").unwrap().expect("n2 exists");
        db.link("g", "n1", &n2.id, "related").expect("link");

        let detect = call_tool("perseus_vault_communities", &db, json!({}), None);
        let v: Value = serde_json::from_str(&detect).expect("valid JSON");
        if &(v["communities"].as_array().unwrap().len()) != &(1) {
            panic!("test assertion failed");
        };
        let cid = v["communities"][0]["id"].as_str().unwrap().to_string();

        // Dispatch via the canonical names.
        let summary = call_tool(
            "perseus_vault_community_summary",
            &db,
            json!({"community_id": cid}),
            None,
        );
        let sv: Value = serde_json::from_str(&summary).expect("valid JSON");
        if &(sv["community_id"].as_str().unwrap()) != &(cid) {
            panic!("test assertion failed");
        };
        if !(sv.get("isError").is_none()) {
            panic!("test assertion failed");
        };

        let recall = call_tool(
            "perseus_vault_global_recall",
            &db,
            json!({"query": "quasar"}),
            None,
        );
        let rv: Value = serde_json::from_str(&recall).expect("valid JSON");
        if !(rv.get("isError").is_none()) {
            panic!("test assertion failed");
        };
        if &(rv["communities"].as_array().unwrap().len()) != &(1) {
            panic!("test assertion failed");
        };

        // tools/list advertises the graph tools under the canonical prefix.
        let names = advertised_names();
        for tool in [
            "perseus_vault_communities",
            "perseus_vault_community_summary",
            "perseus_vault_global_recall",
        ] {
            assert!(names.contains(&tool.to_string()), "must advertise {tool}");
        }

        drop(db);
        let _ = fs::remove_file(&db_path);
    }

    #[test]
    fn recall_layer_filter_scopes_by_canonical_and_alias() {
        let db_path = std::env::temp_dir().join(format!(
            "perseus_vault-layerfilter-{}.db",
            uuid::Uuid::new_v4()
        ));
        let db = Database::open(db_path.to_str().expect("temp db path")).expect("open temp db");

        tools::handle_remember(
            &db,
            json!({"category":"demo","key":"a","body_json":"{\"content\":\"alpha core fact\"}","layer":"core"}),
        )
        .expect("remember a");
        tools::handle_remember(
            &db,
            json!({"category":"demo","key":"b","body_json":"{\"content\":\"alpha working fact\"}","layer":"working"}),
        )
        .expect("remember b");

        let keys = |resp: &str| -> Vec<String> {
            let v: Value = serde_json::from_str(resp).unwrap();
            v["items"]
                .as_array()
                .unwrap()
                .iter()
                .map(|i| i["key"].as_str().unwrap().to_string())
                .collect()
        };

        // Canonical "core" -> only entity a.
        let core =
            tools::handle_recall(&db, json!({"query":"alpha","layer":"core"})).expect("recall");
        let ck = keys(&core);
        assert!(
            ck.contains(&"a".to_string()) && !ck.contains(&"b".to_string()),
            "core filter returned {:?}",
            ck
        );

        // Alias "semantic" -> "working" -> only entity b.
        let sem =
            tools::handle_recall(&db, json!({"query":"alpha","layer":"semantic"})).expect("recall");
        let sk = keys(&sem);
        assert!(
            sk.contains(&"b".to_string()) && !sk.contains(&"a".to_string()),
            "semantic->working filter returned {:?}",
            sk
        );

        // No layer filter -> both.
        let all = tools::handle_recall(&db, json!({"query":"alpha"})).expect("recall");
        assert_eq!(keys(&all).len(), 2, "no filter should return both");

        let _ = fs::remove_file(db_path);
    }

    #[test]
    fn artifact_tools_advertise_canonical_names_and_dispatch() {
        let names = advertised_names();
        for name in [
            "perseus_vault_artifact_register",
            "perseus_vault_artifact_manifest",
            "perseus_vault_artifact_excerpt",
            "perseus_vault_artifact_log_digest",
            "perseus_vault_artifact_verify_value",
        ] {
            assert!(
                names.contains(&name.to_string()),
                "canonical list must advertise {name}"
            );
        }

        let db_path = std::env::temp_dir().join(format!(
            "perseus_vault-artifact-tools-{}.db",
            uuid::Uuid::new_v4()
        ));
        let db = Database::open(db_path.to_str().expect("temp db path")).expect("open temp db");
        let source =
            std::env::temp_dir().join(format!("artifact-mcp-{}.txt", uuid::Uuid::new_v4()));
        std::fs::write(&source, "artifact via MCP\n").unwrap();

        let registered = call_tool(
            "perseus_vault_artifact_register",
            &db,
            json!({"path": source.to_string_lossy(), "workspace_hash": "ws-mcp"}),
            None,
        );
        let rv: Value = serde_json::from_str(&registered).unwrap();
        let sha = rv["sha256"].as_str().unwrap();

        let manifest = call_tool(
            "perseus_vault_artifact_manifest",
            &db,
            json!({"sha256": sha, "workspace_hash": "ws-mcp"}),
            None,
        );
        let mv: Value = serde_json::from_str(&manifest).unwrap();
        assert_eq!(mv["sha256"], json!(sha));

        let _ = fs::remove_file(db_path);
        let _ = std::fs::remove_file(source);
    }

    #[test]
    fn idle_timeout_is_opt_in_since_748() {
        use std::time::Duration;
        // Unset -> DISABLED (default off): inactivity is not abandonment; a
        // quiet-but-alive host (Claude Desktop) must never be reaped. Orphans
        // are caught by parent-death detection, not a flat timer.
        assert_eq!(parse_idle_timeout(None), None);
        // Explicit "0" -> disabled (same as unset).
        assert_eq!(parse_idle_timeout(Some("0")), None);
        // Explicit value -> honored (opt-in aggressive reaping for hosts that
        // leak the stdin write-end while staying alive, the #57228 topology).
        assert_eq!(
            parse_idle_timeout(Some("30")),
            Some(Duration::from_secs(30))
        );
        // Whitespace tolerated.
        assert_eq!(
            parse_idle_timeout(Some(" 120 ")),
            Some(Duration::from_secs(120))
        );
        // Garbage -> disabled (never silently re-enables the flat timer),
        // never panics.
        assert_eq!(parse_idle_timeout(Some("banana")), None);
    }

    #[test]
    fn is_orphaned_by_ppid_returns_false_in_test_process() {
        // The test runner's parent is not init (ppid 1), so this must be false.
        // This is a baseline sanity check; it also confirms the function does not
        // panic and returns the correct type on the current platform.
        assert!(
            !super::is_orphaned_by_ppid(),
            "test process should not have ppid==1"
        );
    }

    /// Verify that `is_orphaned_by_ppid` distinguishes a reparented orphan from
    /// a process legitimately born under a PID-1 init.
    ///
    /// We can't kill the real parent in a unit test, so we model the decision
    /// directly against the documented contract:
    ///   orphaned  <=>  current_ppid == 1  AND  baseline_ppid != 1
    ///
    /// This is the exact logic that fixes the demo-container crash loop, where a
    /// server born under a PID-1 entrypoint (baseline == 1) was falsely reaped by
    /// the old `getppid() == 1` guard. Full end-to-end orphan detection (spawn a
    /// child, kill the parent, observe reparenting) is left to manual/integration
    /// verification since a unit test cannot reparent itself.
    #[test]
    fn is_orphaned_by_ppid_contract() {
        // Pure decision function mirroring is_orphaned_by_ppid's Linux branch.
        fn decide(current_ppid: i32, baseline_ppid: i32) -> bool {
            current_ppid == 1 && baseline_ppid != 1
        }

        // Born under a real parent, later reparented to init => orphaned.
        assert!(
            decide(1, 4242),
            "reparented-to-init must be treated as orphaned"
        );

        // Born directly under PID 1 (container entrypoint) and still there =>
        // NOT an orphan. This is the demo-container regression case.
        assert!(
            !decide(1, 1),
            "process born under PID-1 init must NOT be treated as orphaned"
        );

        // Normal case: real, unchanged parent => not orphaned.
        assert!(
            !decide(4242, 4242),
            "live parent must not be treated as orphaned"
        );

        // Sanity: the live function never fires in a normal test environment
        // (the test runner's parent is never init).
        assert!(
            !super::is_orphaned_by_ppid(),
            "ppid should not be 1 in a normal test environment"
        );
    }

    /// The baseline recorder must be idempotent and safe to call, and after
    /// recording, a normal test process (real parent, not init) must not be
    /// considered orphaned.
    #[test]
    fn record_initial_ppid_is_idempotent_and_safe() {
        super::record_initial_ppid();
        super::record_initial_ppid(); // second call must not panic
        assert!(
            !super::is_orphaned_by_ppid(),
            "after recording baseline, a process with a live parent is not orphaned"
        );
    }

    #[test]
    fn admission_decide_requires_captured_client_identity() {
        let db_path = std::env::temp_dir().join(format!(
            "perseus-vault-admission-identity-{}.db",
            uuid::Uuid::new_v4()
        ));
        let db = Database::open(db_path.to_str().expect("temp db path")).expect("open temp db");
        let state = MCPState::new();
        let init = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(json!(0)),
            method: "initialize".to_string(),
            params: Some(json!({})),
        };
        handle_request(&init, &state, &db).expect("initialize without client identity");
        let call = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(json!(1)),
            method: "tools/call".to_string(),
            params: Some(json!({
                "name": "perseus_vault_admission_decide",
                "arguments": {
                    "category": "decision",
                    "key": "candidate",
                    "workspace_hash": "review-ws",
                    "requesting_agent_id": "forged-reviewer",
                    "decision": "approve",
                    "reason": "human approved"
                }
            })),
        };
        let resp = handle_request(&call, &state, &db).expect("tool response");
        let result = resp.result.expect("tool result");
        assert_eq!(result["isError"], json!(true), "{result}");
        assert!(
            result["content"][0]["text"]
                .as_str()
                .unwrap()
                .contains("clientInfo.name"),
            "{result}"
        );
        let remember = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(json!(2)),
            method: "tools/call".to_string(),
            params: Some(json!({
                "name": "perseus_vault_remember",
                "arguments": {
                    "category": "decision",
                    "key": "candidate",
                    "workspace_hash": "review-ws",
                    "requesting_agent_id": "forged-reviewer",
                    "body_json": "{\"content\":\"candidate\"}",
                    "admission": {"record_digest": "00"}
                }
            })),
        };
        let remember_resp = handle_request(&remember, &state, &db).expect("remember response");
        let remember_result = remember_resp.result.expect("remember tool result");
        assert_eq!(remember_result["isError"], json!(true), "{remember_result}");
        assert!(
            remember_result["content"][0]["text"]
                .as_str()
                .unwrap()
                .contains("clientInfo.name"),
            "{remember_result}"
        );
        let _ = fs::remove_file(db_path);
    }

    #[test]
    fn initialize_cannot_replace_captured_session_identity() {
        let db_path = std::env::temp_dir().join(format!(
            "perseus-vault-initialize-replay-{}.db",
            uuid::Uuid::new_v4()
        ));
        let db = Database::open(db_path.to_str().expect("temp db path")).expect("open temp db");
        let state = MCPState::new();
        let first = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(json!(1)),
            method: "initialize".to_string(),
            params: Some(json!({"clientInfo": {"name": "first-client"}})),
        };
        let first_response =
            handle_request(&first, &state, &db).expect("first initialize response");
        assert!(first_response.error.is_none());
        let second = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(json!(2)),
            method: "initialize".to_string(),
            params: Some(json!({"clientInfo": {"name": "forged-replacement"}})),
        };
        let second_response =
            handle_request(&second, &state, &db).expect("replay initialize response");
        assert!(second_response.result.is_none());
        assert_eq!(
            second_response
                .error
                .as_ref()
                .map(|error| error.message.as_str()),
            Some("Already initialized; session identity cannot be replaced")
        );
        assert_eq!(
            state.session_agent_id.read().unwrap().as_str(),
            "first-client"
        );
        let _ = fs::remove_file(db_path);
    }

    #[test]
    fn call_boundary_enforces_profile_workspace_bindings() {
        // #879 end-to-end: the transport-captured clientInfo.name is the
        // profile identity; a read_only binding denies mutations and a bound
        // profile cannot touch another workspace — the denial surfaces as an
        // isError at the tools/call boundary.
        let db_path =
            std::env::temp_dir().join(format!("perseus-vault-binding-{}.db", uuid::Uuid::new_v4()));
        let db = Database::open(db_path.to_str().expect("temp db path")).expect("open temp db");
        db.workspace_bind("profile-ro", "ws-own", "read_only", "{}", "operator")
            .unwrap();

        let state = MCPState::new();
        // Handshake as the bound read_only profile.
        let init = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(json!(1)),
            method: "initialize".to_string(),
            params: Some(json!({"clientInfo": {"name": "profile-ro", "version": "1.0"}})),
        };
        handle_request(&init, &state, &db).expect("initialize");
        assert_eq!(*state.session_agent_id.read().unwrap(), "profile-ro");

        // Mutation in the bound (read_only) workspace -> denied.
        let call = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(json!(2)),
            method: "tools/call".to_string(),
            params: Some(json!({
                "name": "perseus_vault_remember",
                "arguments": {"category": "decision", "key": "k",
                              "body_json": "{\"v\":1}", "workspace_hash": "ws-own"}
            })),
        };
        let resp = handle_request(&call, &state, &db).expect("remember response");
        let result = resp.result.expect("result");
        assert_eq!(result["isError"], json!(true), "{result}");
        assert!(
            result["content"][0]["text"]
                .as_str()
                .unwrap()
                .contains("read_only"),
            "{result}"
        );

        // Mutation in a DIFFERENT workspace -> denied (cross-workspace).
        let call = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(json!(3)),
            method: "tools/call".to_string(),
            params: Some(json!({
                "name": "perseus_vault_remember",
                "arguments": {"category": "decision", "key": "k2",
                              "body_json": "{\"v\":2}", "workspace_hash": "ws-other"}
            })),
        };
        let resp = handle_request(&call, &state, &db).expect("remember response");
        let result = resp.result.expect("result");
        assert_eq!(result["isError"], json!(true), "{result}");
        assert!(
            result["content"][0]["text"]
                .as_str()
                .unwrap()
                .contains("cross-workspace"),
            "{result}"
        );

        // Reads within the bound workspace stay allowed.
        let call = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(json!(4)),
            method: "tools/call".to_string(),
            params: Some(json!({
                "name": "perseus_vault_recall",
                "arguments": {"query": "anything", "mode": "fts5", "workspace_hash": "ws-own"}
            })),
        };
        let resp = handle_request(&call, &state, &db).expect("recall response");
        assert!(
            resp.result.expect("result").get("isError").is_none(),
            "read must pass"
        );

        let _ = std::fs::remove_file(db_path);
    }

    #[test]
    fn compatibility_dispatcher_rejects_unbound_typed_traversal() {
        let db_path = std::env::temp_dir().join(format!(
            "perseus-vault-typed-traversal-scope-{}.db",
            uuid::Uuid::new_v4()
        ));
        let db = Database::open(db_path.to_str().expect("temp db path")).expect("open temp db");
        let state = MCPState::new_with_strict_scope(false);
        let init = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(json!(1)),
            method: "initialize".to_string(),
            params: Some(json!({"clientInfo": {"name": "unbound-traverse"}})),
        };
        handle_request(&init, &state, &db).expect("initialize");
        let call = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(json!(2)),
            method: "tools/call".to_string(),
            params: Some(json!({
                "name": "perseus_vault_typed_traversal",
                "arguments": {
                    "query": "lineage path",
                    "limit": 1,
                    "workspace_hash": "arbitrary-workspace"
                }
            })),
        };
        let result = handle_request(&call, &state, &db)
            .expect("typed traversal response")
            .result
            .expect("typed traversal result");
        assert_eq!(result["isError"], json!(true), "{result}");
        assert!(
            result["content"][0]["text"]
                .as_str()
                .unwrap_or_default()
                .contains("active workspace binding"),
            "{result}"
        );
        let _ = fs::remove_file(db_path);
    }

    #[test]
    fn strict_scope_deployment_gates_semantic_and_trigger_reads() {
        let db_path = std::env::temp_dir().join(format!(
            "perseus-vault-strict-read-scope-{}.db",
            uuid::Uuid::new_v4()
        ));
        let db = Database::open(db_path.to_str().expect("temp db path")).expect("open temp db");
        db.workspace_bind(
            "strict-reader",
            "workspace-a",
            "read_write",
            "{}",
            "operator",
        )
        .expect("bind strict reader");
        let state = MCPState::new_with_profile_and_strict_scope(ToolProfile::Default, true);
        let initialize = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(json!(1)),
            method: "initialize".to_string(),
            params: Some(json!({"clientInfo": {"name": "strict-reader"}})),
        };
        handle_request(&initialize, &state, &db).expect("initialize response");

        for (id, name, arguments) in [
            (
                2,
                "perseus_vault_semantic_search",
                json!({"query": "strict read probe", "workspace_hash": "workspace-b"}),
            ),
            (
                3,
                "perseus_vault_recall_when",
                json!({"context": "strict trigger probe", "workspace_hash": "workspace-b"}),
            ),
        ] {
            let call = JsonRpcRequest {
                jsonrpc: "2.0".to_string(),
                id: Some(json!(id)),
                method: "tools/call".to_string(),
                params: Some(json!({"name": name, "arguments": arguments})),
            };
            let result = handle_request(&call, &state, &db)
                .expect("read-scope response")
                .result
                .expect("read-scope result");
            assert_eq!(result["isError"], json!(true), "{name}: {result}");
            let text = result["content"][0]["text"]
                .as_str()
                .expect("read-scope error text");
            assert!(text.contains("cross-workspace"), "{name}: {result}");
        }

        let _ = fs::remove_file(db_path);
    }

    #[test]
    fn strict_scope_deployment_requires_bound_transport_identity_and_exact_workspace() {
        let db_path = std::env::temp_dir().join(format!(
            "perseus-vault-strict-scope-{}.db",
            uuid::Uuid::new_v4()
        ));
        let db = Database::open(db_path.to_str().expect("temp db path")).expect("open temp db");
        db.workspace_bind("host-agent", "host-ws", "read_write", "{}", "operator")
            .unwrap();

        // Strict deployments reject an uninitialized transport even when the
        // model supplies a plausible requester and workspace. Without this
        // gate, a no-manifest write can become an active, serveable fact.
        let anonymous = MCPState::new_with_strict_scope(true);
        let init = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(json!(1)),
            method: "initialize".to_string(),
            params: Some(json!({})),
        };
        handle_request(&init, &anonymous, &db).expect("anonymous initialize");
        let remember = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(json!(2)),
            method: "tools/call".to_string(),
            params: Some(json!({
                "name": "perseus_vault_remember",
                "arguments": {
                    "category": "facts",
                    "key": "strict-anonymous",
                    "body_json": "{\"v\":1}",
                    "workspace_hash": "host-ws"
                }
            })),
        };
        let denied = handle_request(&remember, &anonymous, &db)
            .expect("strict anonymous response")
            .result
            .expect("strict anonymous result");
        assert_eq!(denied["isError"], json!(true), "{denied}");
        assert!(
            denied["content"][0]["text"]
                .as_str()
                .unwrap()
                .contains("clientInfo.name"),
            "{denied}"
        );

        // A transport identity is not sufficient by itself: strict mode also
        // requires an active host binding and exact workspace equality.
        let bound = MCPState::new_with_strict_scope(true);
        let init = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(json!(3)),
            method: "initialize".to_string(),
            params: Some(json!({"clientInfo": {"name": "host-agent"}})),
        };
        handle_request(&init, &bound, &db).expect("bound initialize");
        let cross_scope = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(json!(4)),
            method: "tools/call".to_string(),
            params: Some(json!({
                "name": "perseus_vault_remember",
                "arguments": {
                    "category": "facts",
                    "key": "strict-cross-scope",
                    "body_json": "{\"v\":1}",
                    "workspace_hash": "other-ws"
                }
            })),
        };
        let denied = handle_request(&cross_scope, &bound, &db)
            .expect("strict cross-scope response")
            .result
            .expect("strict cross-scope result");
        assert_eq!(denied["isError"], json!(true), "{denied}");
        assert!(
            denied["content"][0]["text"]
                .as_str()
                .unwrap()
                .contains("cross-workspace"),
            "{denied}"
        );

        let _ = std::fs::remove_file(db_path);
    }

    #[test]
    fn strict_default_status_is_scoped_to_the_transport_identity() {
        let db_path = std::env::temp_dir().join(format!(
            "perseus-vault-strict-status-{}.db",
            uuid::Uuid::new_v4()
        ));
        let db = Database::open(db_path.to_str().expect("temp db path")).expect("open temp db");
        db.workspace_bind(
            "strict-agent",
            "workspace-a",
            "read_write",
            "{}",
            "operator",
        )
        .expect("bind strict agent");
        db.workspace_bind("other-agent", "workspace-b", "read_write", "{}", "operator")
            .expect("bind other agent");
        let state = MCPState::new_with_profile_and_strict_scope(ToolProfile::Default, true);
        let initialize = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(json!(1)),
            method: "initialize".to_string(),
            params: Some(json!({"clientInfo": {"name": "strict-agent"}})),
        };
        handle_request(&initialize, &state, &db).expect("initialize response");

        let call = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(json!(2)),
            method: "tools/call".to_string(),
            params: Some(json!({
                "name": "perseus_vault_workspace_status",
                "arguments": {"workspace_hash": "workspace-a"}
            })),
        };
        let response = handle_request(&call, &state, &db)
            .expect("status response")
            .result
            .expect("status result");
        let structured = &response["structuredContent"];
        assert_eq!(structured["count"], json!(1), "got: {response}");
        assert_eq!(
            structured["bindings"][0]["profile_name"],
            json!("strict-agent"),
            "got: {response}"
        );
        let text = response["content"][0]["text"]
            .as_str()
            .expect("status text");
        assert!(
            !text.contains("other-agent"),
            "cross-profile metadata leaked: {text}"
        );
        assert!(
            !text.contains("workspace-b"),
            "cross-workspace metadata leaked: {text}"
        );

        let _ = fs::remove_file(&db_path);
    }

    #[test]
    fn scope_view_parsing_is_lenient_and_defaults_to_full() {
        assert_eq!(resolve_scope_view(None), ScopeView::Full);
        assert_eq!(resolve_scope_view(Some("agent")), ScopeView::Agent);
        assert_eq!(resolve_scope_view(Some("OPS")), ScopeView::Ops);
        assert_eq!(resolve_scope_view(Some("  Agent  ")), ScopeView::Agent);
        assert_eq!(resolve_scope_view(Some("bogus")), ScopeView::Full);
        assert_eq!(resolve_scope_view(Some("")), ScopeView::Full);
    }

    #[test]
    fn scope_table_covers_every_canonical_tool_exactly_once() {
        let registry = tool_registry_base();
        let mut seen = std::collections::HashSet::new();
        for (name, _) in TOOL_SCOPES {
            assert!(seen.insert(*name), "duplicate scope entry for {name}");
        }
        assert_eq!(
            TOOL_SCOPES.len(),
            registry.len(),
            "scope table must stay 1:1 with the canonical registry"
        );
        for tool in registry {
            let name = tool
                .get("name")
                .and_then(|n| n.as_str())
                .expect("registry tool name");
            assert!(seen.contains(name), "missing scope entry for {name}");
        }
    }

    #[test]
    fn scope_filter_counts_match_the_documented_tiers() {
        let registry = tool_registry_base();
        let agent = filter_registry_by_view(registry.clone(), ScopeView::Agent);
        let ops = filter_registry_by_view(registry.clone(), ScopeView::Ops);
        let full = filter_registry_by_view(registry.clone(), ScopeView::Full);
        assert_eq!(
            agent.len(),
            55,
            "agent view count drifted — new tools must be classified"
        );
        assert_eq!(
            ops.len(),
            168,
            "ops view count drifted — new tools must be classified"
        );
        assert_eq!(full.len(), 175, "full view must expose the whole registry");
        assert!(agent.len() < ops.len() && ops.len() < full.len());
    }

    #[test]
    fn scope_filter_keeps_admin_tools_out_of_scoped_views() {
        let registry = tool_registry_base();
        for view in [ScopeView::Agent, ScopeView::Ops] {
            let filtered = filter_registry_by_view(registry.clone(), view);
            let names: Vec<&str> = filtered
                .iter()
                .filter_map(|t| t.get("name").and_then(|n| n.as_str()))
                .collect();
            for admin in [
                "perseus_vault_migrate",
                "perseus_vault_purge",
                "perseus_vault_erase",
                "perseus_vault_authority_set",
            ] {
                assert!(!names.contains(&admin), "{admin} leaked into a scoped view");
            }
            assert!(
                names.contains(&"perseus_vault_recall"),
                "agent tool missing from scoped view"
            );
        }
    }

    #[test]
    fn scope_filter_never_blocks_tool_call_dispatch() {
        // Scopes are advertisement-only: a hidden tool must stay callable
        // through tools/call (authorization remains with workspace binding
        // and authority manifests).
        for view in [ScopeView::Agent, ScopeView::Ops, ScopeView::Full] {
            let filtered = filter_registry_by_view(tool_registry_base().clone(), view);
            assert!(filtered.len() >= 48);
        }
    }

    #[test]
    fn declared_graph_mcp_tools_use_transport_stamped_identity() {
        let db_path = std::env::temp_dir().join(format!(
            "perseus-vault-declared-graph-mcp-{}.db",
            uuid::Uuid::new_v4()
        ));
        let db = Database::open(db_path.to_str().expect("temp db path")).expect("open temp db");
        let state = MCPState::new_with_strict_scope(false);
        let init = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(json!(1)),
            method: "initialize".to_string(),
            params: Some(json!({
                "clientInfo": {"name": "transport-agent", "version": "1.0"}
            })),
        };
        handle_request(&init, &state, &db).expect("initialize");
        assert_eq!(*state.session_agent_id.read().unwrap(), "transport-agent");

        let call = |id: i64, name: &str, arguments: Value| -> Value {
            let request = JsonRpcRequest {
                jsonrpc: "2.0".to_string(),
                id: Some(json!(id)),
                method: "tools/call".to_string(),
                params: Some(json!({"name": name, "arguments": arguments})),
            };
            let response = handle_request(&request, &state, &db).expect("tool response");
            response
                .result
                .expect("tool result")
                .get("structuredContent")
                .cloned()
                .expect("structured content")
        };

        let manifest = call(
            2,
            "perseus_vault_declared_graph_manifest",
            json!({
                "schema_version": 1,
                "operation": "upsert",
                "source_key": "transport-manifest",
                "revision": "r1",
                "content_sha256": "a".repeat(64),
                "source_span_ref": "artifact:transport/span:0-12",
                "workspace_hash": "workspace-a",
                "policy": "replace",
                "nodes": [
                    {"namespace": "service", "canonical_id": "api", "node_type": "service"},
                    {"namespace": "service", "canonical_id": "db", "node_type": "database"}
                ],
                "edges": [{
                    "from": "service:api",
                    "to": "service:db",
                    "predicate": "DEPENDS_ON",
                    "direction": "forward",
                    "origin": "declared",
                    "support_state": "sourced"
                }]
            }),
        );
        assert_eq!(manifest["outcome"], "applied");
        let edge_id = manifest["edge_ids"][0]
            .as_str()
            .expect("edge id")
            .to_string();

        let sourced = call(
            3,
            "perseus_vault_declared_graph_query",
            json!({"workspace_hash": "workspace-a", "limit": 10}),
        );
        assert_eq!(sourced["edges"][0]["attestation_state"], "sourced");

        let attested = call(
            4,
            "perseus_vault_declared_graph_attest",
            json!({
                "schema_version": 1,
                "workspace_hash": "workspace-a",
                "source_key": "transport-manifest",
                "revision": "r1",
                "edge_ids": [edge_id],
                "attestation_ref": "review:transport:r1",
                "attested_by": "reviewer-transport"
            }),
        );
        assert_eq!(attested["outcome"], "applied");

        let verified = call(
            5,
            "perseus_vault_declared_graph_query",
            json!({"workspace_hash": "workspace-a", "limit": 10}),
        );
        assert_eq!(verified["edges"][0]["attestation_state"], "attested");
        assert_eq!(verified["edges"][0]["origin"], "declared");
        assert_eq!(verified["edges"][0]["source_revision"], "r1");
        let _ = std::fs::remove_file(db_path);
    }

    #[test]
    fn declared_graph_tools_are_registered_with_governed_scopes() {
        let registry = tool_registry_base();
        let expected = [
            ("perseus_vault_declared_graph_manifest", ToolScope::Agent),
            ("perseus_vault_declared_graph_attest", ToolScope::Ops),
            ("perseus_vault_declared_graph_query", ToolScope::Agent),
        ];
        for (name, scope) in expected {
            assert!(
                registry.iter().any(|tool| tool["name"] == name),
                "missing registry tool {name}"
            );
            assert_eq!(
                tool_scope_rank(name),
                scope.rank(),
                "wrong scope for {name}"
            );
        }
    }

    #[test]
    fn project_task_schema_advertises_task_state_contract() {
        let tool = tool_registry_base()
            .iter()
            .find(|tool| tool["name"] == "perseus_vault_project_task")
            .expect("project_task tool must be registered");
        let state = &tool["inputSchema"]["properties"]["task_state"];
        assert_eq!(state["type"], "object");
        assert_eq!(state["additionalProperties"], false);
        assert_eq!(state["properties"]["query_digest"]["type"], "string");
        assert_eq!(state["properties"]["constraints"]["maxItems"], 32);
        assert_eq!(
            state["properties"]["accepted_evidence"]["items"]["$ref"],
            "#/properties/task_state/$defs/taskEvidenceReference"
        );
        assert!(state["properties"]["raw_prompt"].is_null());
        assert!(state["properties"]["model_reasoning"].is_null());
        assert!(tool["outputSchema"]["properties"]["serving"].is_object());
        let required = state["required"]
            .as_array()
            .expect("task-state required fields");
        for field in [
            "schema_version",
            "task_id",
            "tenant_id",
            "workspace_hash",
            "principal_id",
            "agent_id",
            "query_digest",
            "route",
            "objective",
            "base_sequence",
            "observed_input_digest",
        ] {
            assert!(
                required.iter().any(|value| value == field),
                "missing required field {field}"
            );
        }
    }

    #[test]
    fn recall_schema_advertises_opt_in_evidence_lanes() {
        let recall = tool_registry_base()
            .iter()
            .find(|tool| tool["name"] == "perseus_vault_recall")
            .expect("recall tool must be registered");
        let evidence = &recall["inputSchema"]["properties"]["evidence_lanes"];
        assert_eq!(evidence["type"], "array");
        assert_eq!(evidence["minItems"], 1);
        assert_eq!(evidence["items"]["enum"], json!(["derived", "verbatim"]));
        assert!(recall["outputSchema"]["properties"]["evidence"].is_object());
    }

    #[test]
    fn recall_schema_advertises_opt_in_selection_decisions() {
        let recall = tool_registry_base()
            .iter()
            .find(|tool| tool["name"] == "perseus_vault_recall")
            .expect("recall tool must be registered");
        let selection = &recall["inputSchema"]["properties"]["include_selection_decisions"];
        assert_eq!(selection["type"], "boolean");
        assert_eq!(selection["default"], false);
        assert!(selection["description"]
            .as_str()
            .unwrap_or("")
            .contains("#1140"));
        assert!(recall["outputSchema"]["properties"]["fused_trace"].is_object());
    }

    #[test]
    fn context_schema_advertises_opt_in_selection_decisions() {
        let context = tool_registry_base()
            .iter()
            .find(|tool| tool["name"] == "perseus_vault_context")
            .expect("context tool must be registered");
        let selection = &context["inputSchema"]["properties"]["include_selection_decisions"];
        assert_eq!(selection["type"], "boolean");
        assert_eq!(selection["default"], false);
        assert!(selection["description"]
            .as_str()
            .unwrap_or("")
            .contains("#1140"));
        assert!(context["outputSchema"]["properties"]["selection_decisions"].is_object());
    }

    #[test]
    fn lean_profile_exposes_only_the_core_memory_surface() {
        let mut names: Vec<String> =
            filter_registry_by_profile(tool_registry_base().clone(), ToolProfile::Lean)
                .into_iter()
                .filter_map(|tool| tool.get("name").and_then(Value::as_str).map(str::to_owned))
                .collect();
        names.sort();
        assert_eq!(
            names,
            vec![
                "perseus_vault_context",
                "perseus_vault_correct",
                "perseus_vault_forget",
                "perseus_vault_health",
                "perseus_vault_recall",
                "perseus_vault_remember",
                "perseus_vault_workspace_status",
            ]
        );
    }

    #[test]
    fn lean_profile_core_tools_survive_agent_scope_filter() {
        let visible = filter_registry_by_view(
            filter_registry_by_profile(tool_registry_base().clone(), ToolProfile::Lean),
            ScopeView::Agent,
        );
        let mut names: Vec<&str> = visible
            .iter()
            .filter_map(|tool| tool.get("name").and_then(Value::as_str))
            .collect();
        names.sort_unstable();
        assert_eq!(names.len(), 7);
        assert!(names.contains(&"perseus_vault_workspace_status"));
        assert!(names.contains(&"perseus_vault_health"));
    }

    #[test]
    fn default_and_all_profiles_preserve_the_full_registry() {
        let full_len = tool_registry_base().len();
        assert_eq!(
            filter_registry_by_profile(tool_registry_base().clone(), ToolProfile::Default).len(),
            full_len
        );
        assert_eq!(
            filter_registry_by_profile(tool_registry_base().clone(), ToolProfile::All).len(),
            full_len
        );
    }

    #[test]
    fn profile_is_carried_into_mcp_state_and_tools_list() {
        let db = Database::open(":memory:").expect("open in-memory db");
        let state = MCPState::new_with_profile(ToolProfile::Lean);
        let initialize = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(json!(1)),
            method: "initialize".to_string(),
            params: Some(json!({"clientInfo": {"name": "lean-test"}})),
        };
        handle_request(&initialize, &state, &db).expect("initialize response");
        let listed = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(json!(2)),
            method: "tools/list".to_string(),
            params: Some(json!({})),
        };
        let response = handle_request(&listed, &state, &db)
            .expect("tools/list response")
            .result
            .expect("tools/list result");
        let mut names: Vec<String> = response["tools"]
            .as_array()
            .expect("tools array")
            .iter()
            .filter_map(|tool| tool.get("name").and_then(Value::as_str).map(str::to_owned))
            .collect();
        names.sort();
        assert_eq!(names.len(), 7);
        assert!(names.contains(&"perseus_vault_workspace_status".to_string()));
    }

    #[test]
    fn default_workspace_status_request_is_scoped_through_mcp_boundary() {
        let db_path = std::env::temp_dir().join(format!(
            "perseus_vault-default-mcp-status-{}.db",
            uuid::Uuid::new_v4()
        ));
        let db = Database::open(db_path.to_str().expect("temp db path")).expect("open temp db");
        db.workspace_bind("default-test", "workspace-a", "read_write", "{}", "test")
            .expect("bind default profile");
        db.workspace_bind("other-agent", "workspace-b", "read_write", "{}", "test")
            .expect("bind other profile");
        let state = MCPState::new();
        let initialize = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(json!(1)),
            method: "initialize".to_string(),
            params: Some(json!({"clientInfo": {"name": "default-test"}})),
        };
        handle_request(&initialize, &state, &db).expect("initialize response");

        let call = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(json!(2)),
            method: "tools/call".to_string(),
            params: Some(json!({
                "name": "perseus_vault_workspace_status",
                "arguments": {"status_scope": "all"}
            })),
        };
        let response = handle_request(&call, &state, &db)
            .expect("status response")
            .result
            .expect("status result");
        let structured = &response["structuredContent"];
        assert_eq!(structured["count"], json!(1), "got: {response}");
        assert_eq!(
            structured["bindings"][0]["profile_name"],
            json!("default-test"),
            "got: {response}"
        );
        let text = response["content"][0]["text"]
            .as_str()
            .expect("status text");
        assert!(
            !text.contains("other-agent"),
            "cross-profile metadata leaked: {text}"
        );
        assert!(
            !text.contains("workspace-b"),
            "cross-workspace metadata leaked: {text}"
        );

        let null_call = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(json!(3)),
            method: "tools/call".to_string(),
            params: Some(json!({
                "name": "perseus_vault_workspace_status",
                "arguments": Value::Null
            })),
        };
        let null_response = handle_request(&null_call, &state, &db)
            .expect("null-arguments status response")
            .result
            .expect("null-arguments status result");
        let null_structured = &null_response["structuredContent"];
        assert_eq!(null_structured["count"], json!(1), "got: {null_response}");
        assert_eq!(
            null_structured["bindings"][0]["profile_name"],
            json!("default-test"),
            "got: {null_response}"
        );
        let null_text = null_response["content"][0]["text"]
            .as_str()
            .expect("null-arguments status text");
        assert!(
            !null_text.contains("other-agent"),
            "cross-profile metadata leaked: {null_text}"
        );
        assert!(
            !null_text.contains("workspace-b"),
            "cross-workspace metadata leaked: {null_text}"
        );

        let _ = fs::remove_file(&db_path);
    }

    #[test]
    fn lean_workspace_status_request_is_scoped_through_mcp_boundary() {
        let db_path = std::env::temp_dir().join(format!(
            "perseus_vault-lean-mcp-status-{}.db",
            uuid::Uuid::new_v4()
        ));
        let db = Database::open(db_path.to_str().expect("temp db path")).expect("open temp db");
        db.workspace_bind("lean-test", "workspace-a", "read_write", "{}", "test")
            .expect("bind lean profile");
        db.workspace_bind("other-agent", "workspace-b", "read_write", "{}", "test")
            .expect("bind other profile");
        let state = MCPState::new_with_profile(ToolProfile::Lean);
        let initialize = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(json!(1)),
            method: "initialize".to_string(),
            params: Some(json!({"clientInfo": {"name": "lean-test"}})),
        };
        handle_request(&initialize, &state, &db).expect("initialize response");

        let call = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(json!(2)),
            method: "tools/call".to_string(),
            params: Some(json!({
                "name": "perseus_vault_workspace_status",
                "arguments": {"status_scope": "all"}
            })),
        };
        let response = handle_request(&call, &state, &db)
            .expect("status response")
            .result
            .expect("status result");
        let structured = &response["structuredContent"];
        assert_eq!(structured["count"], json!(1), "got: {response}");
        assert_eq!(
            structured["bindings"][0]["profile_name"],
            json!("lean-test"),
            "got: {response}"
        );
        let text = response["content"][0]["text"]
            .as_str()
            .expect("status text");
        assert!(
            !text.contains("other-agent"),
            "cross-profile metadata leaked: {text}"
        );
        assert!(
            !text.contains("workspace-b"),
            "cross-workspace metadata leaked: {text}"
        );

        let _ = fs::remove_file(&db_path);
    }

    #[test]
    fn profile_parser_accepts_default_all_and_lean_only() {
        assert_eq!(ToolProfile::parse("default"), Some(ToolProfile::Default));
        assert_eq!(ToolProfile::parse("all"), Some(ToolProfile::All));
        assert_eq!(ToolProfile::parse(" lean "), Some(ToolProfile::Lean));
        assert_eq!(ToolProfile::parse("ops"), None);
    }

    #[test]
    fn remember_schema_excludes_history_only_compacted_status() {
        let remember = tool_registry_base()
            .iter()
            .find(|tool| tool["name"] == "perseus_vault_remember")
            .expect("remember tool must be registered");
        let statuses = remember["inputSchema"]["properties"]["status"]["enum"]
            .as_array()
            .expect("remember status enum");
        assert!(
            statuses.iter().all(|status| status != "compacted"),
            "history-only compacted status must not be advertised as writable"
        );
    }
}
