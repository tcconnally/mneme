//! #952: maintenance / serving isolation.
//!
//! consolidate/cohere/dream (and the autocohere composite) are LLM+DB heavy
//! and must never degrade live recall. Three mechanisms, all operator-
//! configured and fail-closed:
//!
//! 1. **Off-peak window** (`PERSEUS_VAULT_MAINTENANCE_WINDOW`, UTC
//!    `HH:MM-HH:MM`): maintenance refuses to start outside the window unless
//!    the caller passes `force: true` (explicit trigger). Unset = no window
//!    gate (explicit trigger only — nothing auto-starts inside the serving
//!    path; the #908 session-end/cron mis-trigger class is closed by this
//!    same gate).
//! 2. **Serialized, non-reserved concurrency**: one maintenance run at a
//!    time (fail-early on overlap, observable via op_run_*). The lock is
//!    held ONLY while a run is actually executing — a disabled maintenance
//!    mode never reserves capacity, and an idle server holds zero
//!    maintenance capacity.
//! 3. **Live-recall SLO guard** (`PERSEUS_VAULT_MAINTENANCE_P95_BUDGET_MS`):
//!    a timed recall probe measures serving latency; the run refuses to
//!    start when the probe exceeds the budget (unless `force`), and pauses
//!    mid-run (partial report + `slo_paused: true`) when a periodic probe
//!    trips. Unset = guard off. `0` = pause immediately (test seam and the
//!    strictest setting).
//!
//! The guard block (`window`, `slo`, `lock`) is stamped into every
//! maintenance report and exposed read-only via
//! `perseus_vault_maintenance_status`.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard};

use crate::db::Database;
use serde_json::json;
use std::collections::HashMap;

pub const WINDOW_ENV: &str = "PERSEUS_VAULT_MAINTENANCE_WINDOW";
pub const BUDGET_ENV: &str = "PERSEUS_VAULT_MAINTENANCE_P95_BUDGET_MS";
/// Probe check frequency for mid-run pauses: once per maintenance phase
/// boundary (cohere's link window, dream's cluster loop, consolidate's write
/// phase). The probe itself is a bounded limit-1 recall.
const PROBE_QUERY: &str = "_maintenance_probe_952";

static RUNS_STARTED: AtomicU64 = AtomicU64::new(0);
static RUNS_REFUSED: AtomicU64 = AtomicU64::new(0);
static SLO_PAUSES: AtomicU64 = AtomicU64::new(0);
static LAST_PROBE_MS: AtomicU64 = AtomicU64::new(0);

// #952: pause stamp for the CURRENT call thread. Thread-local on purpose:
// concurrent maintenance runs on different stores (or the parallel test
// suite) must never cross-stamp each other's reports; a run's mid-run
// pause flag is only meaningful to the call that produced it.
thread_local! {
    static SLO_PAUSED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

// #952 (tests only): hermetic, THREAD-LOCAL overrides for the gate config.
// Tests must never mutate the process env — the suite runs in parallel and
// other (pre-existing) handler tests read the real env, so any process-wide
// write (env var or static) would transiently break them. A thread-local
// override is visible only to the test thread and its own handler calls
// (all maintenance work is synchronous on the calling thread), so gate
// tests are fully isolated. None falls through to the real env var.
#[cfg(test)]
thread_local! {
    static TEST_BUDGET_MS: std::cell::RefCell<Option<u64>> =
        const { std::cell::RefCell::new(None) };
    static TEST_WINDOW_SPEC: std::cell::RefCell<Option<String>> =
        const { std::cell::RefCell::new(None) };
}

/// Tests only: set (or clear, with None) the SLO budget override for THIS
/// thread.
#[cfg(test)]
pub(crate) fn set_test_budget(v: Option<u64>) {
    TEST_BUDGET_MS.with(|b| *b.borrow_mut() = v);
}

/// Tests only: set (or clear, with None) the window override for THIS
/// thread.
#[cfg(test)]
pub(crate) fn set_test_window(spec: Option<&str>) {
    TEST_WINDOW_SPEC.with(|b| *b.borrow_mut() = spec.map(str::to_string));
}

fn lock_unpoisoned<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|p| p.into_inner())
}

// ─── window ──────────────────────────────────────────────────────────────

/// Parse `HH:MM-HH:MM` (UTC) into (start_min, end_min). `None` on malformed
/// input — a malformed window is treated as "never open" (fail-closed), and
/// the parse error is surfaced in status().
pub fn parse_window(spec: &str) -> Result<(u32, u32), String> {
    let (a, b) = spec
        .trim()
        .split_once('-')
        .ok_or_else(|| format!("maintenance window {spec:?} must be HH:MM-HH:MM (UTC)"))?;
    let parse = |t: &str| -> Result<u32, String> {
        let (h, m) = t
            .trim()
            .split_once(':')
            .ok_or_else(|| format!("maintenance window time {t:?} must be HH:MM"))?;
        let h: u32 = h.trim().parse().map_err(|_| format!("bad hour in {t:?}"))?;
        let m: u32 = m
            .trim()
            .parse()
            .map_err(|_| format!("bad minute in {t:?}"))?;
        if h > 23 || m > 59 {
            return Err(format!("time {t:?} out of range"));
        }
        Ok(h * 60 + m)
    };
    let start = parse(a)?;
    let end = parse(b)?;
    if start == end {
        return Err("maintenance window start == end".to_string());
    }
    Ok((start, end))
}

/// Is `now_min` (minutes since UTC midnight) inside the window? A window
/// with start > end wraps midnight (e.g. 22:00-04:00).
pub fn window_contains(win: (u32, u32), now_min: u32) -> bool {
    let (start, end) = win;
    if start < end {
        now_min >= start && now_min < end
    } else {
        now_min >= start || now_min < end
    }
}

fn configured_window() -> Option<Result<(u32, u32), String>> {
    #[cfg(test)]
    {
        let v = TEST_WINDOW_SPEC.with(|b| b.borrow().clone());
        if let Some(v) = v {
            return Some(parse_window(&v));
        }
    }
    std::env::var(WINDOW_ENV).ok().map(|s| parse_window(&s))
}

pub fn window_open() -> bool {
    match configured_window() {
        None => true,
        Some(Ok(win)) => {
            let now = chrono_now_minutes();
            window_contains(win, now)
        }
        // Malformed config: fail closed (maintenance must be explicitly
        // forced rather than run on garbage).
        Some(Err(_)) => false,
    }
}

fn chrono_now_minutes() -> u32 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    // UTC wall clock minutes (no chrono dependency in the lean tree).
    let days = secs / 86400;
    let rem = secs % 86400;
    let _ = days;
    ((rem / 60) % 1440) as u32
}

// ─── SLO budget ──────────────────────────────────────────────────────────

/// ms budget; None = guard off; 0 = pause/refuse immediately.
pub fn slo_budget_ms() -> Option<u64> {
    #[cfg(test)]
    {
        let v = TEST_BUDGET_MS.with(|b| *b.borrow());
        if v.is_some() {
            return v;
        }
    }
    std::env::var(BUDGET_ENV)
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok())
}

/// Timed live-recall probe (limit-1 FTS5 lookup — the serving path's first
/// hop). Records LAST_PROBE_MS. Returns measured milliseconds.
pub fn recall_probe_ms(db: &Database) -> f64 {
    let start = std::time::Instant::now();
    let params = crate::models::RecallParams {
        query: PROBE_QUERY.to_string(),
        limit: 1,
        ..Default::default()
    };
    let _ = db.recall(&params);
    let ms = start.elapsed().as_secs_f64() * 1000.0;
    LAST_PROBE_MS.store(ms as u64, Ordering::Relaxed);
    ms
}

/// Mid-run pause check: probe exceeds the budget → set the pause flag and
/// return true. Never panics on a failed probe (a probe error is not a
/// budget breach; it is recorded as 0).
pub fn probe_over_budget(db: &Database) -> bool {
    let Some(budget) = slo_budget_ms() else {
        return false;
    };
    let ms = recall_probe_ms(db);
    if ms > budget as f64 {
        SLO_PAUSED.with(|f| f.set(true));
        SLO_PAUSES.fetch_add(1, Ordering::Relaxed);
        true
    } else {
        false
    }
}

// ─── entry gate ──────────────────────────────────────────────────────────

/// Fail-closed entry gate: window (unless `force`) then live-recall probe vs
/// budget (unless `force`). `force` = explicit operator trigger: it bypasses
/// both gates (the run may still pause mid-run on a budget trip).
pub fn check_start(db: &Database, force: bool) -> Result<(), String> {
    if !force {
        match configured_window() {
            None => {}
            Some(Err(e)) => {
                RUNS_REFUSED.fetch_add(1, Ordering::Relaxed);
                return Err(format!(
                    "{e} — maintenance fail-closed; fix {WINDOW_ENV} or pass force:true"
                ));
            }
            Some(Ok(win)) if !window_contains(win, chrono_now_minutes()) => {
                RUNS_REFUSED.fetch_add(1, Ordering::Relaxed);
                return Err(format!(
                    "maintenance window closed (configured {WINDOW_ENV}={}, UTC now HH:MM={:02}:{:02}); \
                     pass force:true for an explicit trigger",
                    std::env::var(WINDOW_ENV).unwrap_or_default(),
                    chrono_now_minutes() / 60,
                    chrono_now_minutes() % 60
                ));
            }
            Some(Ok(_)) => {}
        }
        if let Some(budget) = slo_budget_ms() {
            let ms = recall_probe_ms(db);
            if ms > budget as f64 {
                RUNS_REFUSED.fetch_add(1, Ordering::Relaxed);
                return Err(format!(
                    "maintenance refused: live recall probe {ms:.1}ms exceeds budget {budget}ms \
                     ({BUDGET_ENV}); pass force:true to run anyway (the run may still pause mid-run)"
                ));
            }
        }
    }
    Ok(())
}

// ─── serialized execution lock (per store) ───────────────────────────────

/// One maintenance run at a time PER STORE, keyed by database path. A
/// process-global slot would wrongly serialize unrelated stores and make the
/// parallel test suite collide; maintenance is a property of the store being
/// maintained, so its slot is too. The mutexes leak deliberately (bounded by
/// the number of distinct store paths a process ever opens — a handful).
static STORE_LOCKS: std::sync::OnceLock<Mutex<HashMap<String, &'static Mutex<()>>>> =
    std::sync::OnceLock::new();
static LOCK_HOLDERS: std::sync::OnceLock<Mutex<HashMap<String, String>>> =
    std::sync::OnceLock::new();

fn store_locks() -> &'static Mutex<HashMap<String, &'static Mutex<()>>> {
    STORE_LOCKS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn lock_holders() -> &'static Mutex<HashMap<String, String>> {
    LOCK_HOLDERS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn store_lock(db: &Database) -> &'static Mutex<()> {
    let path = db.db_path();
    lock_unpoisoned(store_locks())
        .entry(path)
        .or_insert_with(|| Box::leak(Box::new(Mutex::new(()))))
        .clone()
}

/// RAII holder of a store's maintenance execution slot. The slot is never
/// reserved: it exists only while a run is actually executing.
#[derive(Debug)]
pub struct MaintenanceLock {
    _guard: MutexGuard<'static, ()>,
    path: String,
}

impl Drop for MaintenanceLock {
    fn drop(&mut self) {
        lock_unpoisoned(lock_holders()).remove(&self.path);
    }
}

pub fn acquire_maintenance(db: &Database, op: &str) -> Result<MaintenanceLock, String> {
    // Bounded retry (~100ms): maintenance is serialized per store, but a
    // brief wait absorbs hand-over overlap (e.g. parallel test threads or a
    // scheduler racing the previous run's tail). Sustained contention still
    // fails fast so a stuck run is visible via op_run_* instead of silently
    // queueing forever.
    let path = db.db_path();
    let m = store_lock(db);
    for attempt in 0..4u32 {
        match m.try_lock() {
            Ok(g) => {
                lock_unpoisoned(lock_holders()).insert(path.clone(), op.to_string());
                RUNS_STARTED.fetch_add(1, Ordering::Relaxed);
                return Ok(MaintenanceLock { _guard: g, path });
            }
            Err(_) if attempt < 3 => {
                std::thread::sleep(std::time::Duration::from_millis(25));
            }
            Err(_) => {
                let holder = lock_unpoisoned(lock_holders())
                    .get(&path)
                    .cloned()
                    .unwrap_or_else(|| "unknown".to_string());
                return Err(format!(
                    "another maintenance run is in progress on this store ({}) — maintenance \
                     is serialized; retry after it finishes (op_run_* exposes start/stop/queue)",
                    holder
                ));
            }
        }
    }
    unreachable!("retry loop always returns")
}

pub fn maintenance_lock_held(db: &Database) -> Option<String> {
    lock_unpoisoned(lock_holders()).get(&db.db_path()).cloned()
}

// ─── report stamping + status ────────────────────────────────────────────

/// The guard block stamped into every maintenance report:
/// {window_open, force, slo_budget_ms, last_probe_ms, slo_paused}.
pub fn guard_block(force: bool) -> serde_json::Value {
    let paused = SLO_PAUSED.with(|f| f.replace(false));
    json!({
        "window": {
            "configured": std::env::var(WINDOW_ENV).ok(),
            "open": window_open(),
        },
        "slo": {
            "budget_ms": slo_budget_ms(),
            "last_probe_ms": LAST_PROBE_MS.load(Ordering::Relaxed),
        },
        "force": force,
        "slo_paused": paused,
    })
}

/// Read-only observability for `perseus_vault_maintenance_status`.
pub fn status(db: &Database) -> serde_json::Value {
    let window = configured_window();
    let holder = maintenance_lock_held(db);
    json!({
        "window": {
            "configured": std::env::var(WINDOW_ENV).ok(),
            "open": window_open(),
            "parse_error": match window {
                Some(Err(e)) => Some(e),
                _ => None,
            },
        },
        "slo": {
            "budget_ms": slo_budget_ms(),
            "last_probe_ms": LAST_PROBE_MS.load(Ordering::Relaxed),
        },
        "lock": {
            "held": holder.is_some(),
            "op": holder,
        },
        "counters": {
            "runs_started": RUNS_STARTED.load(Ordering::Relaxed),
            "runs_refused": RUNS_REFUSED.load(Ordering::Relaxed),
            "slo_pauses": SLO_PAUSES.load(Ordering::Relaxed),
        },
        "note": "maintenance is serialized (one run at a time), never reserved, and gated by \
                 window/budget unless force:true; see docs/specs/maintenance-serving-isolation.md",
    })
}

// ─── tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    #[test]
    fn window_parse_and_contains() {
        assert_eq!(parse_window("02:00-05:00").unwrap(), (120, 300));
        assert_eq!(parse_window("22:00-04:00").unwrap(), (1320, 240));
        assert!(window_contains((120, 300), 121));
        assert!(window_contains((120, 300), 299));
        assert!(!window_contains((120, 300), 300));
        assert!(!window_contains((120, 300), 60));
        // Wrapping window: 23:30 is inside 22:00-04:00; 12:00 is not.
        assert!(window_contains((1320, 240), 1410));
        assert!(window_contains((1320, 240), 30));
        assert!(!window_contains((1320, 240), 720));
        assert!(parse_window("25:00-26:00").is_err());
        assert!(parse_window("02:00").is_err());
        assert!(parse_window("02:00-02:00").is_err());
        assert!(parse_window("junk").is_err());
    }

    #[test]
    fn malformed_window_fails_closed() {
        set_test_window(Some("not-a-window"));
        assert!(!window_open(), "malformed window must read as closed");
        set_test_window(None);
    }

    #[test]
    fn window_env_gate_and_force_bypass() {
        // A window guaranteed closed at the assertion moment: the previous
        // and next minute (wrap-around safe, never "now").
        let now = chrono_now_minutes();
        let closed = format!(
            "{:02}:{:02}-{:02}:{:02}",
            (now + 1) / 60 % 24,
            (now + 1) % 60,
            (now + 2) / 60 % 24,
            (now + 2) % 60
        );
        set_test_window(Some(&closed));
        assert!(!window_open(), "window {closed} must be closed now");
        set_test_window(None);
        assert!(window_open(), "unset window = always open");
    }

    #[test]
    fn budget_zero_refuses_at_gate_and_force_bypasses() {
        set_test_budget(Some(0));
        assert_eq!(slo_budget_ms(), Some(0));
        let fixture = crate::db::TestDatabase::new("perseus_vault-test-db-guard");
        let path = fixture.path().to_string();
        let db: &Database = &fixture;
        let err = check_start(db, false).unwrap_err();
        assert!(
            err.contains("exceeds budget"),
            "gate must refuse on a 0 budget, got: {err}"
        );
        // force bypasses the START gate (mid-run pause still applies).
        check_start(db, true).expect("force must bypass the start gate");
        set_test_budget(None);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn maintenance_lock_is_exclusive_and_non_reserved() {
        let fixture = crate::db::TestDatabase::new("perseus_vault-test-db-lockunit");
        let db: &Database = &fixture;
        let l1 = acquire_maintenance(db, "test-op").expect("first acquire");
        assert_eq!(maintenance_lock_held(db).as_deref(), Some("test-op"));
        let err = acquire_maintenance(db, "second").unwrap_err();
        assert!(
            err.contains("another maintenance run is in progress"),
            "got: {err}"
        );
        drop(l1);
        assert_eq!(maintenance_lock_held(db), None, "lock must release on drop");
        // Non-reserved: immediately acquirable again (no lingering capacity).
        let l2 = acquire_maintenance(db, "test-op-2").expect("re-acquire after drop");
        drop(l2);
    }

    #[test]
    fn status_shape_is_stable() {
        let fixture = crate::db::TestDatabase::new("perseus_vault-test-db-status");
        let db: &Database = &fixture;
        let s = status(db);
        assert!(s["window"]["open"].is_boolean());
        assert!(s["slo"]["budget_ms"].is_null() || s["slo"]["budget_ms"].is_u64());
        assert!(s["lock"]["held"].is_boolean());
        assert!(s["counters"]["runs_started"].is_u64());
    }
}
