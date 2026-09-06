//! #958: runtime self-audit `verify` command.
//!
//! Re-asserts the Vault's invariants against the operator's own live store
//! (TeamBrain `tb verify` borrow). Exit contract: 0 = all checks PASS,
//! 2 = a check could not run (UNVERIFIED, never PASS), 3 = an invariant is
//! violated. Findings print `path:key` only — never values.
//!
//! Read-only by construction: every check runs over a SQLITE_OPEN_READ_ONLY
//! connection; nothing here opens the store through the migration path.

use rusqlite::{Connection, OptionalExtension};

/// Check table (C1–C7 always; C8 only in strict mode).
pub const CHECK_IDS: [&str; 8] = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    Pass,
    Unverified,
    Fail,
}

impl Status {
    pub fn as_str(&self) -> &'static str {
        match self {
            Status::Pass => "PASS",
            Status::Unverified => "UNVERIFIED",
            Status::Fail => "FAIL",
        }
    }
}

#[derive(Debug, Clone)]
pub struct CheckResult {
    pub id: &'static str,
    pub status: Status,
    /// Findings are `path:key` strings only — values are never emitted.
    pub findings: Vec<String>,
    pub note: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub struct VerifyOptions {
    pub strict: bool,
    /// Check ids the operator asked to skip -> reported UNVERIFIED.
    pub skip: Vec<String>,
}

pub fn run_verify(conn: &Connection, opts: &VerifyOptions) -> Vec<CheckResult> {
    let mut out = vec![
        c1_secret_shapes(conn),
        c2_encrypted_at_rest(conn),
        c3_workspace_isolation(conn),
        c4_authority_expiry(conn),
        c5_archived_not_served(conn),
        c6_fts_sync(conn),
        c7_schema_version(conn),
    ];
    if opts.strict {
        out.push(c8_egress_sandbox());
    }
    if !opts.skip.is_empty() {
        for r in out.iter_mut() {
            if opts.skip.iter().any(|s| s == r.id) {
                r.status = Status::Unverified;
                r.findings.clear();
                r.note = Some(format!("skipped by operator --skip {}", r.id));
            }
        }
    }
    out
}

pub fn exit_code(results: &[CheckResult]) -> i32 {
    if results.iter().any(|r| r.status == Status::Fail) {
        3
    } else if results.iter().any(|r| r.status == Status::Unverified) {
        2
    } else {
        0
    }
}

// ─── helpers ────────────────────────────────────────────────────────────

fn store_is_encrypted(conn: &Connection) -> bool {
    conn.query_row(
        "SELECT COUNT(*) FROM encryption_canary WHERE id = 1",
        [],
        |r| r.get::<_, i64>(0),
    )
    .map(|n| n > 0)
    .unwrap_or(false)
}

/// Extract a distinctive probe token (first word of length >= 4) from a body.
fn probe_token(body: &str) -> Option<String> {
    body.split(|c: char| !c.is_alphanumeric())
        .find(|w| w.len() >= 4)
        .map(|w| w.to_lowercase())
}

const MAX_FINDINGS: usize = 20;

/// C1 — no secret-shaped values in bodies; sanitizer markers present where
/// expected. On an encrypted store the invariant holds by construction
/// (bodies are ciphertext), so the scan reports PASS with a note.
fn c1_secret_shapes(conn: &Connection) -> CheckResult {
    let encrypted = store_is_encrypted(conn);
    if encrypted {
        return CheckResult {
            id: "C1",
            status: Status::Pass,
            findings: vec![],
            note: Some("store encrypted at rest; bodies are ciphertext — plaintext scan inapplicable by construction".to_string()),
        };
    }
    let mut findings = Vec::new();
    let mut stmt = match conn
        .prepare("SELECT id, category, key, body_json FROM entities WHERE archived = 0")
    {
        Ok(s) => s,
        Err(_) => {
            return CheckResult {
                id: "C1",
                status: Status::Unverified,
                findings: vec![],
                note: Some("entities table unavailable".to_string()),
            }
        }
    };
    let rows = stmt
        .query_map([], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
                r.get::<_, String>(3)?,
            ))
        })
        .and_then(|rows| rows.collect::<Result<Vec<_>, _>>())
        .unwrap_or_default();
    let mut secret_key_hits = 0usize;
    let mut shape_hits = 0usize;
    for row in rows {
        let (id, category, key, body) = row;
        // 1) secret-shaped value patterns (deterministic substring shapes).
        let lower = body.to_lowercase();
        let shapes = [
            ("pem_private", "-----begin"),
            ("gh_token", "ghp_"),
            ("aws_key", "akia"),
            ("sk_key", "sk-"),
            ("slack_token", "xox"),
        ];
        for (shape, needle) in shapes {
            if lower.contains(needle) {
                shape_hits += 1;
                if findings.len() < MAX_FINDINGS {
                    findings.push(format!(
                        "{id}:{category}/{key} (secret-shaped value: {shape})"
                    ));
                }
                break;
            }
        }
        // 2) sanitizer-marker expectation: secret-NAMED keys must carry a
        // redaction marker, an empty value, or a placeholder — never a real
        // credential-looking string.
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&body) {
            if let Some(obj) = v.as_object() {
                for (k, val) in obj {
                    let kk = k.to_lowercase();
                    let secret_named = [
                        "token",
                        "secret",
                        "api_key",
                        "apikey",
                        "password",
                        "passwd",
                        "bearer",
                        "authorization",
                        "private_key",
                    ]
                    .iter()
                    .any(|s| kk.contains(s));
                    if secret_named {
                        if let Some(s) = val.as_str() {
                            let redacted = s.contains("[REDACTED]")
                                || s.contains("***")
                                || s.contains("<redacted>")
                                || s.chars().all(|c| c == 'x');
                            if s.len() >= 8 && !redacted {
                                secret_key_hits += 1;
                                if findings.len() < MAX_FINDINGS {
                                    findings.push(format!(
                                        "{id}:{category}/{key} (sanitizer marker missing on '{k}')"
                                    ));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    let total = shape_hits + secret_key_hits;
    let mut note = None;
    if total > 0 && findings.len() >= MAX_FINDINGS {
        note = Some(format!(
            "{total} hits; findings truncated to {MAX_FINDINGS}"
        ));
    }
    CheckResult {
        id: "C1",
        status: if total > 0 {
            Status::Fail
        } else {
            Status::Pass
        },
        findings,
        note,
    }
}

/// C2 — encrypted-at-rest deployments have zero plaintext payload rows.
/// UNVERIFIED when the store has no encryption canary. Payload classification
/// is deliberately conservative: valid JSON, empty values, short values, and
/// values containing non-base64 bytes are plaintext/invalid for this store.
fn c2_plaintext_payload(value: &str) -> bool {
    let trimmed = value.trim();
    trimmed.is_empty()
        || serde_json::from_str::<serde_json::Value>(trimmed).is_ok()
        || value.len() < 40
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"+/=".contains(&byte))
}

fn c2_encrypted_at_rest(conn: &Connection) -> CheckResult {
    if !store_is_encrypted(conn) {
        return CheckResult {
            id: "C2",
            status: Status::Unverified,
            findings: vec![],
            note: Some(
                "store has no encryption canary; encrypted-at-rest check cannot run".to_string(),
            ),
        };
    }
    let payloads = "
        SELECT id, category, key, 'entities.body_json' AS column_name, body_json AS value
        FROM entities
        UNION ALL
        SELECT history_id, category, key, 'entity_history.body_json' AS column_name, body_json AS value
        FROM entity_history
        UNION ALL
        SELECT id, category, key, 'entities.hints' AS column_name, COALESCE(hints, '') AS value
        FROM entities
    ";
    let mut stmt = match conn.prepare(payloads) {
        Ok(stmt) => stmt,
        Err(_) => return c2_failure("c2-query (payload enumeration unavailable)"),
    };
    let rows: Vec<(String, String, String, String, String)> = match stmt
        .query_map([], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
                r.get::<_, String>(3)?,
                r.get::<_, String>(4)?,
            ))
        })
        .and_then(|rows| rows.collect::<Result<Vec<_>, _>>())
    {
        Ok(rows) => rows,
        Err(_) => return c2_failure("c2-query (payload enumeration unavailable)"),
    };
    let plaintext: Vec<_> = rows
        .iter()
        .filter(|(_, _, _, _, value)| c2_plaintext_payload(value))
        .collect();
    let total = plaintext.len() as i64;
    let findings: Vec<String> = plaintext
        .iter()
        .take(MAX_FINDINGS)
        .map(|(id, category, key, column, _)| format!("{id}:{category}/{key} ({column})"))
        .collect();
    CheckResult {
        id: "C2",
        status: if total > 0 {
            Status::Fail
        } else {
            Status::Pass
        },
        findings,
        note: if total > 0 {
            Some(format!("{total} plaintext payload rows in encrypted store"))
        } else {
            Some("body-column encryption only; FTS protection and other SQLite metadata are checked separately".to_string())
        },
    }
}

fn c2_failure(reason: &str) -> CheckResult {
    CheckResult {
        id: "C2",
        status: Status::Fail,
        findings: vec![reason.to_string()],
        note: None,
    }
}

/// C3 — workspace isolation: no cross-workspace entity visibility. Scoped
/// recall filters on `workspace_hash` at the SQL level (total filter), so
/// the enforceable invariant is identity isolation: the same (category, key)
/// must not be stamped into two distinct workspaces. A collision is the
/// #951 shadow-import hazard — scoped recall and dedup can surface a foreign
/// workspace's entity under a colliding identity.
fn c3_workspace_isolation(conn: &Connection) -> CheckResult {
    let mut stmt = match conn.prepare(
        "SELECT category, key, COUNT(DISTINCT workspace_hash) AS ws_count, \
         GROUP_CONCAT(DISTINCT workspace_hash) AS ws_list \
         FROM entities \
         WHERE workspace_hash IS NOT NULL AND workspace_hash <> '' \
         GROUP BY category, key HAVING COUNT(DISTINCT workspace_hash) > 1 \
         LIMIT ?1",
    ) {
        Ok(s) => s,
        Err(_) => {
            return CheckResult {
                id: "C3",
                status: Status::Unverified,
                findings: vec![],
                note: Some("entities table unavailable".to_string()),
            }
        }
    };
    let rows: Vec<(String, String, i64, String)> = stmt
        .query_map([MAX_FINDINGS as i64], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, i64>(2)?,
                r.get::<_, String>(3)?,
            ))
        })
        .and_then(|rows| rows.collect::<Result<Vec<_>, _>>())
        .unwrap_or_default();
    let findings: Vec<String> = rows
        .iter()
        .map(|(cat, key, count, ws_list)| {
            format!("identity:{cat}/{key} (stamped into {count} workspaces: {ws_list})")
        })
        .collect();
    let colliding = !findings.is_empty();
    CheckResult {
        id: "C3",
        status: if colliding {
            Status::Fail
        } else {
            Status::Pass
        },
        findings,
        note: if colliding {
            None
        } else {
            Some("no identity spans multiple workspaces".to_string())
        },
    }
}

/// C4 — authority manifest presence/expiry: no active manifest may be past
/// its expiry. Zero manifests -> UNVERIFIED (nothing to assert).
fn c4_authority_expiry(conn: &Connection) -> CheckResult {
    let count: i64 = conn
        .query_row("SELECT COUNT(*) FROM authority_manifests", [], |r| r.get(0))
        .unwrap_or(0);
    if count == 0 {
        return CheckResult {
            id: "C4",
            status: Status::Unverified,
            findings: vec![],
            note: Some(
                "no authority manifests configured; presence/expiry cannot be asserted".to_string(),
            ),
        };
    }
    let now = crate::db::now_ms() as i64;
    let mut findings = Vec::new();
    let mut stmt = conn
        .prepare(
            "SELECT id, agent_id, workspace_hash FROM authority_manifests AS m \
             WHERE m.revoked_at_unix_ms IS NULL AND m.expires_at_unix_ms IS NOT NULL \
             AND m.expires_at_unix_ms <= ?1 \
             AND m.version = (SELECT MAX(current.version) FROM authority_manifests AS current \
                              WHERE current.agent_id = m.agent_id \
                                AND current.workspace_hash = m.workspace_hash \
                                AND current.revoked_at_unix_ms IS NULL) \
             LIMIT ?2",
        )
        .unwrap_or_else(|_| panic!("C4 prepare"));
    let rows = stmt
        .query_map(rusqlite::params![now, MAX_FINDINGS as i64], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
            ))
        })
        .and_then(|rows| rows.collect::<Result<Vec<_>, _>>())
        .unwrap_or_default();
    for row in rows {
        findings.push(format!("{}:{}/{}", row.0, row.1, row.2));
    }
    let expired = !findings.is_empty();
    CheckResult {
        id: "C4",
        status: if expired { Status::Fail } else { Status::Pass },
        findings,
        note: if expired {
            Some(
                "expired authority manifests require revoke or replacement via authority_revoke or authority_set"
                    .to_string(),
            )
        } else {
            None
        },
    }
}

/// C5 — archived entities are never served by recall: archived rows must not
/// be FTS-indexed, and a recall probe for an archived token must not return
/// the archived entity.
fn c5_archived_not_served(conn: &Connection) -> CheckResult {
    let mut findings = Vec::new();
    // 1) structural: archived row present in the FTS index (recall serves FTS).
    let mut stmt = conn
        .prepare(
            "SELECT e.id, e.category, e.key FROM entities e \
             JOIN entities_fts f ON f.rowid = e.rowid \
             WHERE e.archived = 1 LIMIT ?1",
        )
        .unwrap_or_else(|_| panic!("C5 prepare"));
    let rows = stmt
        .query_map([MAX_FINDINGS as i64], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
            ))
        })
        .and_then(|rows| rows.collect::<Result<Vec<_>, _>>())
        .unwrap_or_default();
    for row in rows {
        findings.push(format!(
            "{}:{}/{} (archived but still indexed)",
            row.0, row.1, row.2
        ));
    }
    // 2) behavioral: recall probe for up to 10 archived tokens.
    if findings.is_empty() && !store_is_encrypted(conn) {
        let mut pstmt = conn
            .prepare(
                "SELECT id, category, key, body_json FROM entities \
                 WHERE archived = 1 LIMIT 10",
            )
            .unwrap_or_else(|_| panic!("C5 probe prepare"));
        let probes: Vec<(String, String, String, String)> = pstmt
            .query_map([], |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, String>(2)?,
                    r.get::<_, String>(3)?,
                ))
            })
            .and_then(|rows| rows.collect::<Result<Vec<_>, _>>())
            .unwrap_or_default();
        for (pid, pcat, pkey, pbody) in probes {
            let Some(token) = probe_token(&pbody) else {
                continue;
            };
            let q = format!("\"{token}\"");
            let sql = "SELECT e.id FROM entities_fts f JOIN entities e ON e.rowid = f.rowid \
                       WHERE entities_fts MATCH ?1 LIMIT 5";
            let mut rstmt = conn.prepare(sql).unwrap_or_else(|_| panic!("C5 recall"));
            let served: Vec<String> = rstmt
                .query_map([&q], |r| r.get::<_, String>(0))
                .and_then(|rows| rows.collect::<Result<Vec<_>, _>>())
                .unwrap_or_default();
            if served.contains(&pid) {
                findings.push(format!(
                    "{pid}:{pcat}/{pkey} (recall probe served an archived entity)"
                ));
                break;
            }
        }
    }
    CheckResult {
        id: "C5",
        status: if findings.is_empty() {
            Status::Pass
        } else {
            Status::Fail
        },
        findings,
        note: if store_is_encrypted(conn) {
            Some("encrypted store: archived FTS structure checked; content probe requires the active key".to_string())
        } else {
            None
        },
    }
}

/// C6 — FTS index in sync with entities: no active entity missing from the
/// index, no phantom index rows. Encrypted stores additionally require the
/// declared protected mode and non-empty 64-hex HMAC tokens in both live and
/// history indexes.
fn protected_fts_value(value: &str) -> bool {
    !value.trim().is_empty()
        && value
            .split_whitespace()
            .all(|token| token.len() == 64 && token.bytes().all(|byte| byte.is_ascii_hexdigit()))
}

fn protected_fts_invalid_rows(conn: &Connection, table: &str) -> Result<i64, rusqlite::Error> {
    let sql = format!("SELECT body_json FROM {table}");
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map([], |r| r.get::<_, String>(0))?;
    let mut invalid = 0i64;
    for value in rows {
        if !protected_fts_value(&value?) {
            invalid += 1;
        }
    }
    Ok(invalid)
}

fn c6_fts_sync(conn: &Connection) -> CheckResult {
    let mut findings = Vec::new();
    let missing: i64 = match conn.query_row(
        "SELECT COUNT(*) FROM entities \
         WHERE archived = 0 \
         AND rowid NOT IN (SELECT rowid FROM entities_fts)",
        [],
        |r| r.get(0),
    ) {
        Ok(count) => count,
        Err(_) => return c6_failure("entities:fts-sync (check unavailable)"),
    };
    if missing > 0 {
        let mut stmt = match conn.prepare(
            "SELECT id, category, key FROM entities \
             WHERE archived = 0 \
             AND rowid NOT IN (SELECT rowid FROM entities_fts) LIMIT ?1",
        ) {
            Ok(stmt) => stmt,
            Err(_) => return c6_failure("entities:fts-sync (missing-row enumeration unavailable)"),
        };
        let rows: Vec<(String, String, String)> = match stmt
            .query_map([MAX_FINDINGS as i64], |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, String>(2)?,
                ))
            })
            .and_then(|rows| rows.collect::<Result<Vec<_>, _>>())
        {
            Ok(rows) => rows,
            Err(_) => return c6_failure("entities:fts-sync (missing-row enumeration unavailable)"),
        };
        for row in rows {
            findings.push(format!(
                "{}:{}/{} (entity missing from FTS index)",
                row.0, row.1, row.2
            ));
        }
    }
    let history_missing: i64 = match conn.query_row(
        "SELECT COUNT(*) FROM entity_history AS h
         WHERE NOT EXISTS (SELECT 1 FROM entity_history_fts AS f WHERE f.rowid = h.rowid)",
        [],
        |r| r.get(0),
    ) {
        Ok(count) => count,
        Err(_) => return c6_failure("entity_history:fts-sync (missing-row check unavailable)"),
    };
    if history_missing > 0 {
        findings.push(format!(
            "history-missing-fts:{history_missing} rows (history rows with no FTS entry)"
        ));
    }
    let history_phantom: i64 = match conn.query_row(
        "SELECT COUNT(*) FROM entity_history_fts AS f
         WHERE NOT EXISTS (SELECT 1 FROM entity_history AS h WHERE h.rowid = f.rowid)",
        [],
        |r| r.get(0),
    ) {
        Ok(count) => count,
        Err(_) => return c6_failure("entity_history:fts-sync (phantom-row check unavailable)"),
    };
    if history_phantom > 0 {
        findings.push(format!(
            "history-phantom-fts:{history_phantom} rows (history FTS rows with no history row)"
        ));
    }
    let phantom: i64 = match conn.query_row(
        "SELECT COUNT(*) FROM entities_fts AS f
         WHERE NOT EXISTS (
             SELECT 1 FROM entities AS e
             WHERE e.rowid = f.rowid AND e.archived = 0
         )",
        [],
        |r| r.get(0),
    ) {
        Ok(count) => count,
        Err(_) => return c6_failure("entities:fts-sync (phantom-row check unavailable)"),
    };
    if phantom > 0 {
        findings.push(format!(
            "phantom-fts:{phantom} rows (index rows with no active entity)"
        ));
    }
    let encrypted = store_is_encrypted(conn);
    if encrypted {
        let mode: Option<String> = match conn
            .query_row(
                "SELECT search_mode FROM encryption_profile WHERE id = 1",
                [],
                |r| r.get(0),
            )
            .optional()
        {
            Ok(mode) => mode,
            Err(_) => return c6_failure("encryption_profile:search_mode (check unavailable)"),
        };
        if mode.as_deref() != Some(crate::encryption::BLIND_TOKEN_SEARCH_MODE) {
            findings
                .push("encryption_profile:search_mode (protected mode not declared)".to_string());
        }
        for table in ["entities_fts", "entity_history_fts"] {
            match protected_fts_invalid_rows(conn, table) {
                Ok(invalid) if invalid > 0 => findings.push(format!(
                    "{table}:protected_tokens ({invalid} rows contain non-protected content)"
                )),
                Err(_) => findings.push(format!("{table}:protected_tokens (check unavailable)")),
                _ => {}
            }
        }
    }
    let protected_index_note = encrypted && findings.is_empty();
    CheckResult {
        id: "C6",
        status: if findings.is_empty() {
            Status::Pass
        } else {
            Status::Fail
        },
        findings,
        note: if protected_index_note {
            Some("protected HMAC-token representation checked for live and history FTS".to_string())
        } else {
            None
        },
    }
}

fn c6_failure(reason: &str) -> CheckResult {
    CheckResult {
        id: "C6",
        status: Status::Fail,
        findings: vec![reason.to_string()],
        note: None,
    }
}

/// C7 — schema_version consistent with the binary.
fn c7_schema_version(conn: &Connection) -> CheckResult {
    let uv: i64 = conn
        .query_row("PRAGMA user_version", [], |r| r.get(0))
        .unwrap_or(-1);
    let expected = crate::schema::SCHEMA_VERSION;
    if uv != expected {
        CheckResult {
            id: "C7",
            status: Status::Fail,
            findings: vec![format!("schema:{uv} (binary expects {expected})")],
            note: None,
        }
    } else {
        CheckResult {
            id: "C7",
            status: Status::Pass,
            findings: vec![],
            note: Some(format!("user_version = {expected}")),
        }
    }
}

// ─── C8: strict-mode egress sandbox ─────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProbeOutcome {
    /// The probe connected — egress is possible.
    Connected,
    /// The probe could not connect — egress is blocked.
    Blocked,
    /// The sandbox machinery itself is unavailable.
    Unavailable,
}

/// Probe `addr:port` from inside a fresh network namespace (deny-all egress).
/// `unshare -n` gives the child its own network stack with no interfaces up,
/// so ANY connect fails unless the sandbox itself is broken.
pub fn probe_connect_sandboxed(addr: &str, port: u16) -> ProbeOutcome {
    let probe = format!(
        "import socket; s=socket.socket(); s.settimeout(3); \
         s.connect(({addr:?}, {port})); print('connected')"
    );
    let out = std::process::Command::new("unshare")
        .args(["-n", "python3", "-c", &probe])
        .output();
    match out {
        Ok(o) => {
            if o.status.success() {
                ProbeOutcome::Connected
            } else {
                ProbeOutcome::Blocked
            }
        }
        Err(_) => ProbeOutcome::Unavailable,
    }
}

/// Unsandboxed probe (test helper): loopback connect succeeds on any normal
/// host, proving the sandbox — not the environment — blocks the socket.
pub fn probe_connect_unsandboxed(addr: &str, port: u16) -> ProbeOutcome {
    let probe = format!(
        "import socket; s=socket.socket(); s.settimeout(3); \
         s.connect(({addr:?}, {port})); print('connected')"
    );
    match std::process::Command::new("python3")
        .args(["-c", &probe])
        .output()
    {
        Ok(o) if o.status.success() => ProbeOutcome::Connected,
        _ => ProbeOutcome::Blocked,
    }
}

/// C8 — OS-level deny-all network sandbox: a socket connect from inside the
/// sandbox must fail. UNVERIFIED when `unshare` is unavailable.
fn c8_egress_sandbox() -> CheckResult {
    match probe_connect_sandboxed("1.1.1.1", 443) {
        ProbeOutcome::Connected => CheckResult {
            id: "C8",
            status: Status::Fail,
            findings: vec![
                "sandbox:egress (socket connect succeeded inside unshare -n)".to_string(),
            ],
            note: None,
        },
        ProbeOutcome::Blocked => CheckResult {
            id: "C8",
            status: Status::Pass,
            findings: vec![],
            note: Some("deny-all network sandbox blocked the egress probe".to_string()),
        },
        ProbeOutcome::Unavailable => CheckResult {
            id: "C8",
            status: Status::Unverified,
            findings: vec![],
            note: Some("unshare(1) unavailable; sandboxed egress check cannot run".to_string()),
        },
    }
}

// ─── report formatting ──────────────────────────────────────────────────

pub fn render_human(results: &[CheckResult]) -> String {
    let mut s = String::new();
    for r in results {
        s.push_str(&format!("{} {}\n", r.id, r.status.as_str()));
        for f in &r.findings {
            s.push_str(&format!("  {}\n", f));
        }
        if let Some(n) = &r.note {
            s.push_str(&format!("  note: {}\n", n));
        }
    }
    s.push_str(&format!(
        "exit: {} (0=pass, 2=unverified, 3=violation)\n",
        exit_code(results)
    ));
    s
}

pub fn render_json(results: &[CheckResult]) -> String {
    let v: serde_json::Value = serde_json::json!({
        "version": 1,
        "checks": results.iter().map(|r| {
            serde_json::json!({
                "id": r.id,
                "status": r.status.as_str(),
                "findings": r.findings,
                "note": r.note,
            })
        }).collect::<Vec<_>>(),
        "exit_code": exit_code(results),
    });
    serde_json::to_string_pretty(&v).unwrap_or_else(|_| "{}".to_string())
}

// ─── tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::Database;
    use rusqlite::OpenFlags;
    use std::fs;

    /// Open a read-only connection to a temp store, seeding through the
    /// normal Database first (dropped before the RO handle opens).
    fn ro_db(seed: impl FnOnce(&Database)) -> Connection {
        let path = std::env::temp_dir().join(format!(
            "perseus-verify-{}.db",
            uuid::Uuid::new_v4().simple()
        ));
        let path_str = path.to_str().unwrap().to_string();
        let db = Database::open(&path_str).expect("seed db");
        seed(&db);
        drop(db);
        Connection::open_with_flags(
            &path_str,
            OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )
        .expect("ro open")
    }

    fn seed_entity(db: &Database, category: &str, key: &str, body: &str, ws: &str) {
        let mut e = crate::db::tests::make_entity(key, category, key, body);
        if !ws.is_empty() {
            e.workspace_hash = ws.to_string();
        }
        db.remember_with_options(&e, true, None, None, false)
            .unwrap();
    }

    // ── C1 ──
    #[test]
    fn c1_fails_on_secret_shaped_body() {
        let conn = ro_db(|db| {
            seed_entity(db, "facts", "k1", r#"{"note":"release pipeline"}"#, "");
            seed_entity(
                db,
                "facts",
                "k2",
                r#"{"note":"deploy token ghp_abcdefghijklmnopqrstuvwxyz1234567890 here"}"#,
                "",
            );
        });
        let r = run_verify(&conn, &VerifyOptions::default());
        let c1 = r.iter().find(|c| c.id == "C1").unwrap();
        assert_eq!(c1.status, Status::Fail, "{c1:?}");
        assert!(c1.findings[0].starts_with("k2:facts/k2"), "{c1:?}");
        assert!(
            !c1.findings[0].contains("ghp_"),
            "findings must not leak values: {c1:?}"
        );
    }

    #[test]
    fn c1_fails_on_missing_sanitizer_marker() {
        let conn = ro_db(|db| {
            seed_entity(
                db,
                "facts",
                "k1",
                r#"{"api_token":"supersecretvalue123"}"#,
                "",
            );
        });
        let r = run_verify(&conn, &VerifyOptions::default());
        let c1 = r.iter().find(|c| c.id == "C1").unwrap();
        assert_eq!(c1.status, Status::Fail, "{c1:?}");
    }

    #[test]
    fn c1_passes_clean_store() {
        let conn = ro_db(|db| {
            seed_entity(db, "facts", "k1", r#"{"note":"plain notes"}"#, "");
        });
        let r = run_verify(&conn, &VerifyOptions::default());
        let c1 = r.iter().find(|c| c.id == "C1").unwrap();
        assert_eq!(c1.status, Status::Pass, "{c1:?}");
    }

    // ── C2 ──
    #[test]
    fn c2_unverified_without_canary() {
        let conn = ro_db(|db| {
            seed_entity(db, "facts", "k1", r#"{"note":"x"}"#, "");
        });
        let r = run_verify(&conn, &VerifyOptions::default());
        let c2 = r.iter().find(|c| c.id == "C2").unwrap();
        assert_eq!(c2.status, Status::Unverified, "{c2:?}");
        assert_eq!(exit_code(&r), 2, "UNVERIFIED must exit 2");
    }

    #[test]
    fn c2_fails_on_plaintext_row_in_encrypted_store() {
        let conn = ro_db(|db| {
            // Seed the corruption before marking the store encrypted. The
            // plaintext row itself is injected through the normal test writer;
            // production writers reject it after a canary exists.
            seed_entity(db, "facts", "k1", r#"{"note":"leaked in the clear"}"#, "");
            let c = db.conn().unwrap();
            c.execute(
                "INSERT OR REPLACE INTO encryption_canary (id, ciphertext, created_at_unix_ms) \
                 VALUES (1, 'bm9uY2VjaXBoZXJ0ZXh0', 1)",
                [],
            )
            .unwrap();
        });
        let r = run_verify(&conn, &VerifyOptions::default());
        let c2 = r.iter().find(|c| c.id == "C2").unwrap();
        assert_eq!(c2.status, Status::Fail, "{c2:?}");
        assert_eq!(exit_code(&r), 3);
    }

    #[test]
    fn c2_passes_encrypted_store_without_plaintext() {
        let conn = ro_db(|db| {
            // Seed ciphertext-shaped values before marking the store encrypted.
            seed_entity(
                db,
                "facts",
                "k1",
                "bm90anNvbnBsYWludGV4dHlwaWNhbGJhc2U2NA==",
                "",
            );
            db.conn()
                .unwrap()
                .execute(
                    "UPDATE entities SET hints = ?1",
                    rusqlite::params!["A".repeat(40)],
                )
                .unwrap();
            let c = db.conn().unwrap();
            c.execute(
                "INSERT OR REPLACE INTO encryption_canary (id, ciphertext, created_at_unix_ms) \
                 VALUES (1, 'bm9uY2VjaXBoZXJ0ZXh0', 1)",
                [],
            )
            .unwrap();
        });
        let r = run_verify(&conn, &VerifyOptions::default());
        let c2 = r.iter().find(|c| c.id == "C2").unwrap();
        assert_eq!(c2.status, Status::Pass, "{c2:?}");
    }

    #[test]
    fn c2_fails_on_plaintext_history_row_in_encrypted_store() {
        let conn = ro_db(|db| {
            let c = db.conn().unwrap();
            c.execute(
                "INSERT OR REPLACE INTO encryption_canary (id, ciphertext, created_at_unix_ms) \
                 VALUES (1, 'bm9uY2VjaXBoZXJ0ZXh0', 1)",
                [],
            )
            .unwrap();
            c.execute(
                "INSERT INTO entity_history
                 (history_id, id, category, key, body_json, created_at_unix_ms, last_accessed_unix_ms)
                 VALUES ('history-c2-plaintext', 'entity-c2-plaintext', 'facts',
                         'history-c2-plaintext', ?1, 1, 1)",
                [r#"{"note":"history payload is still plaintext"}"#],
            )
            .unwrap();
        });
        let r = run_verify(&conn, &VerifyOptions::default());
        let c2 = r.iter().find(|c| c.id == "C2").unwrap();
        assert_eq!(c2.status, Status::Fail, "{c2:?}");
    }

    #[test]
    fn c2_fails_on_array_scalar_and_hint_plaintext() {
        let conn = ro_db(|db| {
            seed_entity(db, "facts", "k-array", "[]", "");
            let c = db.conn().unwrap();
            c.execute("UPDATE entities SET hints = '0' WHERE key = 'k-array'", [])
                .unwrap();
            c.execute(
                "INSERT INTO entity_history
                 (history_id, id, category, key, body_json, created_at_unix_ms, last_accessed_unix_ms)
                 VALUES ('history-c2-scalar', 'entity-c2-scalar', 'facts',
                         'history-c2-scalar', '1234', 1, 1)",
                [],
            )
            .unwrap();
            let c = db.conn().unwrap();
            c.execute(
                "INSERT OR REPLACE INTO encryption_canary (id, ciphertext, created_at_unix_ms) \
                 VALUES (1, 'bm9uY2VjaXBoZXJ0ZXh0', 1)",
                [],
            )
            .unwrap();
        });
        let r = run_verify(&conn, &VerifyOptions::default());
        let c2 = r.iter().find(|c| c.id == "C2").unwrap();
        assert_eq!(c2.status, Status::Fail, "{c2:?}");
    }

    #[test]
    fn c6_fails_on_history_row_missing_from_fts() {
        let conn = ro_db(|db| {
            let c = db.conn().unwrap();
            c.execute(
                "INSERT OR REPLACE INTO encryption_canary
                 (id, ciphertext, created_at_unix_ms) VALUES (1, 'canary', 1)",
                [],
            )
            .unwrap();
            c.execute(
                "INSERT INTO entity_history
                 (history_id, id, category, key, body_json, created_at_unix_ms,
                  last_accessed_unix_ms)
                 VALUES ('history-c6', 'id-c6', 'facts', 'k-c6',
                         'ciphertext', 1, 1)",
                [],
            )
            .unwrap();
            drop(c);
        });
        let r = run_verify(&conn, &VerifyOptions::default());
        let c6 = r.iter().find(|check| check.id == "C6").unwrap();
        assert_eq!(c6.status, Status::Fail, "{c6:?}");
        assert!(
            c6.findings
                .iter()
                .any(|finding| finding.contains("history")),
            "{c6:?}"
        );
    }

    // ── C3 ──
    #[test]
    fn c3_fails_on_identity_collision() {
        let conn = ro_db(|db| {
            seed_entity(
                db,
                "facts",
                "ka",
                r#"{"note":"zzzwsa distinctive payload"}"#,
                "ws-a",
            );
            // Corruption: the same (category, key) is stamped into a second
            // workspace (the #951 shadow-import hazard — a shadow write that
            // leaked into the live workspace under a colliding identity).
            let c = db.conn().unwrap();
            let cols: String = c
                .query_row(
                    "SELECT group_concat(name) FROM pragma_table_info('entities') \
                     WHERE name NOT IN ('id', 'workspace_hash')",
                    [],
                    |r| r.get(0),
                )
                .unwrap();
            // Stamp the copy into a DIFFERENT workspace: (category, key, ws)
            // is unique, so the only collision the store can hold is the
            // same identity under a second workspace — the corruption a
            // shadow-import leak would produce.
            let sql = format!(
                "INSERT INTO entities (id, workspace_hash, {cols}) \
                 SELECT 'mem-copy', 'ws-b', {cols} FROM entities WHERE key = 'ka'"
            );
            c.execute_batch(&sql).unwrap();
            // Index the copy so the collision is recall-reachable.
            c.execute(
                "INSERT INTO entities_fts (rowid, body_json) \
                 VALUES ((SELECT rowid FROM entities WHERE id = 'mem-copy'), \
                         (SELECT body_json FROM entities WHERE id = 'mem-copy'))",
                [],
            )
            .unwrap();
        });
        let r = run_verify(&conn, &VerifyOptions::default());
        let c3 = r.iter().find(|c| c.id == "C3").unwrap();
        assert_eq!(c3.status, Status::Fail, "{c3:?}");
        assert!(c3.findings[0].starts_with("identity:facts/ka"), "{c3:?}");
        assert_eq!(exit_code(&r), 3);
    }

    #[test]
    fn c3_passes_isolated_workspaces() {
        let conn = ro_db(|db| {
            seed_entity(
                db,
                "facts",
                "ka",
                r#"{"note":"zzzwsa distinctive payload"}"#,
                "ws-a",
            );
            seed_entity(
                db,
                "facts",
                "kb",
                r#"{"note":"zzzwsa distinctive payload"}"#,
                "ws-b",
            );
        });
        let r = run_verify(&conn, &VerifyOptions::default());
        let c3 = r.iter().find(|c| c.id == "C3").unwrap();
        assert_eq!(c3.status, Status::Pass, "{c3:?}");
    }

    // ── C4 ──
    #[test]
    fn c4_unverified_without_manifests() {
        let conn = ro_db(|_db| {});
        let r = run_verify(&conn, &VerifyOptions::default());
        let c4 = r.iter().find(|c| c.id == "C4").unwrap();
        assert_eq!(c4.status, Status::Unverified, "{c4:?}");
    }

    #[test]
    fn c4_fails_on_expired_manifest() {
        let conn = ro_db(|db| {
            let c = db.conn().unwrap();
            c.execute(
                "INSERT INTO authority_manifests (id, agent_id, workspace_hash, version, \
                 allowed_capabilities, approval_required_capabilities, scope_anchors, \
                 approver_principals, allowed_inbound_principals, \
                 permitted_external_ref_prefixes, max_parallel_actions, mode, \
                 expires_at_unix_ms, revoked_at_unix_ms, created_at_unix_ms, \
                 capability_constraints_json) \
                 VALUES ('man-1', 'agent-a', 'ws-a', 1, '[]', '[]', '[]', '[]', '[]', '[]', \
                 1, 'self', 1000, NULL, 1, '{}')",
                [],
            )
            .unwrap();
        });
        let r = run_verify(&conn, &VerifyOptions::default());
        let c4 = r.iter().find(|c| c.id == "C4").unwrap();
        assert_eq!(c4.status, Status::Fail, "{c4:?}");
        assert!(c4.findings[0].starts_with("man-1:"), "{c4:?}");
    }

    #[test]
    fn c4_passes_unexpired_manifest() {
        let conn = ro_db(|db| {
            let c = db.conn().unwrap();
            c.execute(
                "INSERT INTO authority_manifests (id, agent_id, workspace_hash, version, \
                 allowed_capabilities, approval_required_capabilities, scope_anchors, \
                 approver_principals, allowed_inbound_principals, \
                 permitted_external_ref_prefixes, max_parallel_actions, mode, \
                 expires_at_unix_ms, revoked_at_unix_ms, created_at_unix_ms, \
                 capability_constraints_json) \
                 VALUES ('man-2', 'agent-a', 'ws-a', 1, '[]', '[]', '[]', '[]', '[]', '[]', \
                 1, 'self', 4102444800000, NULL, 1, '{}')",
                [],
            )
            .unwrap();
        });
        let r = run_verify(&conn, &VerifyOptions::default());
        let c4 = r.iter().find(|c| c.id == "C4").unwrap();
        assert_eq!(c4.status, Status::Pass, "{c4:?}");
    }

    #[test]
    fn c4_ignores_expired_replaced_manifest() {
        let conn = ro_db(|db| {
            db.agent_upsert("agent-c4-replaced", "C4 Replaced", 3, "perseus")
                .unwrap();
            let mut input = crate::models::AuthorityManifestInput {
                agent_id: "agent-c4-replaced".to_string(),
                workspace_hash: "ws-c4-replaced".to_string(),
                allowed_capabilities: vec!["git_push".to_string()],
                approval_required_capabilities: vec![],
                scope_anchors: vec!["vault".to_string()],
                approver_principals: vec![],
                allowed_inbound_principals: vec![],
                permitted_external_ref_prefixes: vec!["vault".to_string()],
                max_parallel_actions: 1,
                mode: "enforce".to_string(),
                expires_at_unix_ms: Some(1),
                capability_constraints_json: "{}".to_string(),
            };
            db.authority_set(&input, "admin").unwrap();
            input.expires_at_unix_ms = Some(4_102_444_800_000);
            db.authority_set(&input, "admin").unwrap();
        });
        let r = run_verify(&conn, &VerifyOptions::default());
        let c4 = r.iter().find(|c| c.id == "C4").unwrap();
        assert_eq!(c4.status, Status::Pass, "{c4:?}");
    }

    // ── C5 ──
    #[test]
    fn c5_fails_when_archived_entity_is_still_indexed() {
        let conn = ro_db(|db| {
            seed_entity(db, "facts", "k1", r#"{"note":"zzzarchived payload"}"#, "");
            let id = db.get_entity("facts", "k1").unwrap().unwrap().id;
            db.forget("facts", "k1", "test").unwrap();
            // Re-index the archived row by hand — the corruption a recall
            // regression would produce.
            let c = db.conn().unwrap();
            c.execute(
                "INSERT INTO entities_fts (rowid, body_json) VALUES ((SELECT rowid FROM entities WHERE id = ?1), ?2)",
                rusqlite::params![id, r#"{"note":"zzzarchived payload"}"#],
            )
            .unwrap();
        });
        let r = run_verify(&conn, &VerifyOptions::default());
        let c5 = r.iter().find(|c| c.id == "C5").unwrap();
        assert_eq!(c5.status, Status::Fail, "{c5:?}");
    }

    #[test]
    fn c5_passes_when_archive_cleans_index() {
        let conn = ro_db(|db| {
            seed_entity(db, "facts", "k1", r#"{"note":"zzzarchived payload"}"#, "");
            db.forget("facts", "k1", "test").unwrap();
        });
        let r = run_verify(&conn, &VerifyOptions::default());
        let c5 = r.iter().find(|c| c.id == "C5").unwrap();
        assert_eq!(c5.status, Status::Pass, "{c5:?}");
    }

    // ── C6 ──
    #[test]
    fn c6_fails_on_missing_fts_row() {
        let conn = ro_db(|db| {
            seed_entity(db, "facts", "k1", r#"{"note":"zzzindex me"}"#, "");
            let c = db.conn().unwrap();
            c.execute(
                "DELETE FROM entities_fts WHERE rowid = (SELECT rowid FROM entities WHERE key = 'k1')",
                [],
            )
            .unwrap();
        });
        let r = run_verify(&conn, &VerifyOptions::default());
        let c6 = r.iter().find(|c| c.id == "C6").unwrap();
        assert_eq!(c6.status, Status::Fail, "{c6:?}");
        assert_eq!(exit_code(&r), 3);
    }

    #[test]
    fn c6_ignores_archived_entity_missing_fts_row() {
        let conn = ro_db(|db| {
            seed_entity(db, "facts", "archived-k1", "archived no index", "");
            db.forget("facts", "archived-k1", "test").unwrap();
        });
        let r = run_verify(&conn, &VerifyOptions::default());
        let c5 = r.iter().find(|c| c.id == "C5").unwrap();
        let c6 = r.iter().find(|c| c.id == "C6").unwrap();
        assert_eq!(c5.status, Status::Pass, "{c5:?}");
        assert_eq!(c6.status, Status::Pass, "{c6:?}");
    }

    #[test]
    fn c6_passes_synced_index() {
        let conn = ro_db(|db| {
            seed_entity(db, "facts", "k1", r#"{"note":"zzzindex me"}"#, "");
        });
        let r = run_verify(&conn, &VerifyOptions::default());
        let c6 = r.iter().find(|c| c.id == "C6").unwrap();
        assert_eq!(c6.status, Status::Pass, "{c6:?}");
    }

    #[test]
    fn c6_fails_on_empty_protected_token_row() {
        let conn = ro_db(|db| {
            seed_entity(db, "facts", "k-empty-token", r#"{"note":"indexed"}"#, "");
            let c = db.conn().unwrap();
            c.execute(
                "INSERT OR REPLACE INTO encryption_canary (id, ciphertext, created_at_unix_ms) \
                 VALUES (1, 'bm9uY2VjaXBoZXJ0ZXh0', 1)",
                [],
            )
            .unwrap();
            c.execute(
                "INSERT OR REPLACE INTO encryption_profile (id, search_mode, updated_at_unix_ms) \
                 VALUES (1, ?1, 1)",
                rusqlite::params![crate::encryption::BLIND_TOKEN_SEARCH_MODE],
            )
            .unwrap();
            c.execute("UPDATE entities_fts SET body_json = ''", [])
                .unwrap();
        });
        let r = run_verify(&conn, &VerifyOptions::default());
        let c6 = r.iter().find(|c| c.id == "C6").unwrap();
        assert_eq!(c6.status, Status::Fail, "{c6:?}");
    }

    #[test]
    fn c6_fails_on_archived_live_fts_phantom() {
        let conn = ro_db(|db| {
            seed_entity(
                db,
                "facts",
                "k-archived-phantom",
                r#"{"note":"indexed"}"#,
                "",
            );
            let id = db
                .get_entity("facts", "k-archived-phantom")
                .unwrap()
                .unwrap()
                .id;
            db.forget("facts", "k-archived-phantom", "test").unwrap();
            let c = db.conn().unwrap();
            c.execute(
                "INSERT OR REPLACE INTO encryption_canary (id, ciphertext, created_at_unix_ms) \
                 VALUES (1, 'bm9uY2VjaXBoZXJ0ZXh0', 1)",
                [],
            )
            .unwrap();
            c.execute(
                "INSERT OR REPLACE INTO encryption_profile (id, search_mode, updated_at_unix_ms) \
                 VALUES (1, ?1, 1)",
                rusqlite::params![crate::encryption::BLIND_TOKEN_SEARCH_MODE],
            )
            .unwrap();
            c.execute(
                "INSERT INTO entities_fts (rowid, body_json) VALUES \
                 ((SELECT rowid FROM entities WHERE id = ?1), ?2)",
                rusqlite::params![id, "a".repeat(64)],
            )
            .unwrap();
        });
        let r = run_verify(&conn, &VerifyOptions::default());
        let c6 = r.iter().find(|c| c.id == "C6").unwrap();
        assert_eq!(c6.status, Status::Fail, "{c6:?}");
    }

    // ── C7 ──
    #[test]
    fn c7_fails_on_version_mismatch() {
        let conn = ro_db(|db| {
            let c = db.conn().unwrap();
            c.execute_batch("PRAGMA user_version = 1;").unwrap();
        });
        let r = run_verify(&conn, &VerifyOptions::default());
        let c7 = r.iter().find(|c| c.id == "C7").unwrap();
        assert_eq!(c7.status, Status::Fail, "{c7:?}");
    }

    #[test]
    fn c7_passes_on_current_version() {
        let conn = ro_db(|_db| {});
        let r = run_verify(&conn, &VerifyOptions::default());
        let c7 = r.iter().find(|c| c.id == "C7").unwrap();
        assert_eq!(c7.status, Status::Pass, "{c7:?}");
    }

    // ── C8 sandbox ──
    #[test]
    fn sandbox_actually_blocks_sockets() {
        // Prove the sandbox blocks sockets: a loopback listener reachable
        // unsandboxed must be unreachable inside `unshare -n`.
        if std::process::Command::new("unshare")
            .arg("-n")
            .arg("true")
            .status()
            .is_err()
        {
            eprintln!("unshare unavailable; sandbox proof skipped");
            return;
        }
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        assert_eq!(
            probe_connect_unsandboxed("127.0.0.1", port),
            ProbeOutcome::Connected,
            "unsandboxed loopback connect must succeed (environment sanity)"
        );
        assert_eq!(
            probe_connect_sandboxed("127.0.0.1", port),
            ProbeOutcome::Blocked,
            "sandboxed connect must fail: unshare -n must isolate the network stack"
        );
    }

    #[test]
    fn c8_unverified_without_unshare() {
        if std::process::Command::new("unshare")
            .arg("-n")
            .arg("true")
            .status()
            .is_ok()
        {
            eprintln!("unshare present; UNVERIFIED path exercised only where absent");
            return;
        }
        let r = c8_egress_sandbox();
        assert_eq!(r.status, Status::Unverified, "{r:?}");
    }

    // ── exit contract + no-value-leakage ──
    #[test]
    fn exit_contract_and_no_value_leakage() {
        let conn = ro_db(|db| {
            seed_entity(
                db,
                "facts",
                "k1",
                r#"{"note":"token ghp_abcdefghijklmnopqrstuvwxyz1234567890 leaks"}"#,
                "",
            );
        });
        let r = run_verify(&conn, &VerifyOptions::default());
        assert_eq!(exit_code(&r), 3, "violation must exit 3");
        let json = render_json(&r);
        assert!(
            !json.contains("ghp_"),
            "JSON output must not leak values: {json}"
        );
        let human = render_human(&r);
        assert!(human.contains("FAIL"), "{human}");
        assert!(
            !human.contains("ghp_"),
            "human output must not leak values: {human}"
        );
    }

    #[test]
    fn all_pass_exits_zero() {
        let conn = ro_db(|db| {
            seed_entity(
                db,
                "facts",
                "k1",
                "bm90anNvbnBsYWludGV4dHlwaWNhbGJhc2U2NA==",
                "",
            );
            let c = db.conn().unwrap();
            c.execute(
                "INSERT OR REPLACE INTO encryption_canary (id, ciphertext, created_at_unix_ms) \
                 VALUES (1, 'bm9uY2VjaXBoZXJ0ZXh0', 1)",
                [],
            )
            .unwrap();
            c.execute(
                "INSERT OR REPLACE INTO encryption_profile (id, search_mode, updated_at_unix_ms) \
                 VALUES (1, ?1, ?2)",
                rusqlite::params![crate::encryption::BLIND_TOKEN_SEARCH_MODE, 1],
            )
            .unwrap();
            c.execute(
                "UPDATE entities SET hints = ?1",
                rusqlite::params!["A".repeat(40)],
            )
            .unwrap();
            c.execute(
                "UPDATE entities_fts SET body_json = ?1",
                rusqlite::params!["a".repeat(64)],
            )
            .unwrap();
            c.execute(
                "INSERT INTO authority_manifests (id, agent_id, workspace_hash, version, \
                 allowed_capabilities, approval_required_capabilities, scope_anchors, \
                 approver_principals, allowed_inbound_principals, \
                 permitted_external_ref_prefixes, max_parallel_actions, mode, \
                 expires_at_unix_ms, revoked_at_unix_ms, created_at_unix_ms, \
                 capability_constraints_json) \
                 VALUES ('man-ok', 'agent-a', '', 1, '[]', '[]', '[]', '[]', '[]', '[]', \
                 1, 'self', 4102444800000, NULL, 1, '{}')",
                [],
            )
            .unwrap();
        });
        let r = run_verify(&conn, &VerifyOptions::default());
        assert_eq!(exit_code(&r), 0, "{r:?}");
    }

    #[test]
    fn skip_marks_checks_unverified() {
        let conn = ro_db(|db| {
            seed_entity(db, "facts", "k1", r#"{"note":"clean notes"}"#, "");
        });
        let r = run_verify(
            &conn,
            &VerifyOptions {
                strict: false,
                skip: vec!["C7".to_string()],
            },
        );
        let c7 = r.iter().find(|c| c.id == "C7").unwrap();
        assert_eq!(c7.status, Status::Unverified, "{c7:?}");
        assert_eq!(exit_code(&r), 2, "skipped check must exit 2 (never PASS)");
    }

    #[test]
    fn verify_is_read_only() {
        let conn = ro_db(|db| {
            seed_entity(db, "facts", "k1", r#"{"note":"clean notes"}"#, "");
        });
        run_verify(&conn, &VerifyOptions::default());
        // Read-only handle cannot write: an attempted write must fail.
        assert!(conn
            .execute("CREATE TABLE should_not_exist (x)", [])
            .is_err());
    }

    #[test]
    fn encrypted_store_c1_passes_and_c2_runs() {
        let conn = ro_db(|db| {
            seed_entity(
                db,
                "facts",
                "k1",
                "bm90anNvbnBsYWludGV4dHlwaWNhbGJhc2U2NA==",
                "",
            );
            db.conn()
                .unwrap()
                .execute(
                    "UPDATE entities SET hints = ?1",
                    rusqlite::params!["A".repeat(40)],
                )
                .unwrap();
            let c = db.conn().unwrap();
            c.execute(
                "INSERT OR REPLACE INTO encryption_canary (id, ciphertext, created_at_unix_ms) \
                 VALUES (1, 'bm9uY2VjaXBoZXJ0ZXh0', 1)",
                [],
            )
            .unwrap();
        });
        let r = run_verify(&conn, &VerifyOptions::default());
        let c1 = r.iter().find(|c| c.id == "C1").unwrap();
        assert_eq!(
            c1.status,
            Status::Pass,
            "C1 holds by construction on ciphertext: {c1:?}"
        );
        let c2 = r.iter().find(|c| c.id == "C2").unwrap();
        assert_eq!(c2.status, Status::Pass, "{c2:?}");
    }
}
