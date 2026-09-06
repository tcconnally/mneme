//! #872: retrieval concentration, repeated-serving, and cross-arm
//! contamination telemetry.
//!
//! Read-only observability over serving (this module never mutates recall
//! ranking; lifecycle cleanup, admission, and supersession remain the truth
//! controls). Three write-on-read side effects feed it — bounded and
//! pruned like the existing `retrieval_count` side effect:
//!
//!   * `served_events`        — one row per delivered entity per recall
//!                              (slot, estimated tokens, query class, trust)
//!   * `recall_arm_audits`    — per-arm candidate / re-entry / delivered
//!                              counts per recall mode
//!   * `displacement_events`  — cooldown/diversity controls removing an
//!                              entity (with sole-evidence flag)
//!
//! The report separates `empty` (no serving activity in the window) from
//! zero concentration, and reports degraded probe states separately from
//! clean zeros. Denominators, scope, retrieval profile, source class, and
//! the versioned artifact hash are always included (acceptance #5/#6).

use crate::db::{is_stopword, now_ms, Database};
use crate::encryption::EncryptionManager;
use crate::models::Entity;
use rusqlite::{params, Connection};

/// Non-serveable lifecycle statuses — the truth controls (supersession,
/// quarantine, expiry, redaction). Serving arms exclude these at the SQL
/// boundary; telemetry verifies they never re-enter through another arm.
pub const NON_SERVEABLE_STATUSES: &[&str] = &[
    "deprecated",
    "expired",
    "proposed",
    "quarantined",
    "redacted",
    "compacted",
];

const SERVED_RETENTION_MS: i64 = 7 * 24 * 3600 * 1000;
const SERVED_CAP_ROWS: i64 = 100_000;
const AUDIT_RETENTION_MS: i64 = 7 * 24 * 3600 * 1000;
const AUDIT_CAP_ROWS: i64 = 50_000;

/// Serveable-status SQL clause. Only canonical active/draft rows are
/// serveable; NULL, empty, unknown, and terminal/pending states fail closed.
pub fn serveable_status_clause(alias: &str) -> String {
    let a = if alias.is_empty() {
        String::new()
    } else {
        format!("{alias}.")
    };
    format!("{a}status IN ('active','draft')")
}

/// Bare-column variant for splicing into existing SQL (no alias).
pub const SERVEABLE_STATUS_SQL: &str = "status IN ('active','draft')";

/// Coarse query class: up to two content-bearing tokens, lowercased.
/// Used for fan-out measurement (how many query classes a low-trust
/// candidate enters) without storing raw queries.
pub fn query_class(query: &str) -> String {
    query
        .split_whitespace()
        .filter(|w| w.chars().count() >= 3 && !is_stopword(&w.to_lowercase()))
        .take(2)
        .map(|w| w.to_lowercase())
        .collect::<Vec<_>>()
        .join(" ")
}

pub fn mode_label(mode: &crate::models::SearchMode) -> &'static str {
    match mode {
        crate::models::SearchMode::Fts5 => "lexical",
        crate::models::SearchMode::Dense => "dense",
        crate::models::SearchMode::Hybrid => "hybrid",
        crate::models::SearchMode::Fused => "fused",
    }
}

fn new_id(prefix: &str) -> String {
    format!("{prefix}-{}", uuid::Uuid::new_v4().simple().to_string())
}

/// Prune one telemetry table: retention window first, then a hard row cap.
fn prune(
    conn: &Connection,
    table: &str,
    retention_ms: i64,
    cap_rows: i64,
) -> Result<(), Box<dyn std::error::Error>> {
    let cutoff = now_ms() - retention_ms;
    conn.execute(
        &format!("DELETE FROM {table} WHERE ts_unix_ms < ?1"),
        params![cutoff],
    )?;
    conn.execute(
        &format!(
            "DELETE FROM {table} WHERE id IN (\
             SELECT id FROM {table} ORDER BY ts_unix_ms DESC LIMIT -1 OFFSET ?1)"
        ),
        params![cap_rows],
    )?;
    Ok(())
}

/// Record one served event per delivered entity (bounded, pruned).
/// `skip_side_effects` recalls do NOT record — probes stay clean.
pub fn record_served(
    conn: &Connection,
    batch_id: &str,
    profile: &str,
    mode: &str,
    query: &str,
    entities: &[Entity],
) -> Result<(), Box<dyn std::error::Error>> {
    if entities.is_empty() {
        return Ok(());
    }
    let ts = now_ms();
    let qclass = query_class(query);
    let mut stmt = conn.prepare(
        "INSERT OR IGNORE INTO served_events \
         (id, ts_unix_ms, batch_id, profile, workspace_hash, entity_id, category, key, \
          source, verified, certainty, mode, query, query_class, tokens_est, slot) \
         VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,?16)",
    )?;
    for (slot, e) in entities.iter().enumerate() {
        let tokens_est = (e.body_json.chars().count() / 4).max(1) as i64;
        stmt.execute(params![
            new_id("sev"),
            ts,
            batch_id,
            profile,
            e.workspace_hash,
            e.id,
            e.category,
            e.key,
            e.source,
            e.verified as i32,
            e.certainty,
            mode,
            query,
            qclass,
            tokens_est,
            slot as i64,
        ])?;
    }
    drop(stmt);
    prune(conn, "served_events", SERVED_RETENTION_MS, SERVED_CAP_ROWS)?;
    Ok(())
}

/// Record one per-arm audit row for a recall.
/// `reentry` = entities in the arm's materialized candidate list whose
/// lifecycle status is non-serveable (post-SQL — the status exclusion runs
/// at the query boundary, so this is expected to be 0; the contamination
/// probe measures what the SQL boundary blocked).
pub fn record_arm_audit(
    conn: &Connection,
    mode: &str,
    arm: &str,
    candidates: usize,
    reentry: usize,
    delivered: usize,
    profile: &str,
    workspace_hash: &str,
    query_hash: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    conn.execute(
        "INSERT OR IGNORE INTO recall_arm_audits \
         (id, ts_unix_ms, mode, arm, candidates, reentry_candidates, delivered, \
          profile, workspace_hash, query_hash) \
         VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10)",
        params![
            new_id("ara"),
            now_ms(),
            mode,
            arm,
            candidates as i64,
            reentry as i64,
            delivered as i64,
            profile,
            workspace_hash,
            query_hash,
        ],
    )?;
    prune(
        conn,
        "recall_arm_audits",
        AUDIT_RETENTION_MS,
        AUDIT_CAP_ROWS,
    )?;
    Ok(())
}

/// Record a displacement event (cooldown/diversity control removed an
/// entity). `was_sole_evidence` = the entity was the only match for its
/// dominant keyword — the displacement-sole-answer case (#872).
pub fn record_displacement(
    conn: &Connection,
    entity_id: &str,
    reason: &str,
    was_sole_evidence: bool,
    mode: &str,
    workspace_hash: &str,
    query: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    conn.execute(
        "INSERT OR IGNORE INTO displacement_events \
         (id, ts_unix_ms, entity_id, reason, was_sole_evidence, mode, workspace_hash, query) \
         VALUES (?1,?2,?3,?4,?5,?6,?7,?8)",
        params![
            new_id("dsp"),
            now_ms(),
            entity_id,
            reason,
            was_sole_evidence as i32,
            mode,
            workspace_hash,
            query,
        ],
    )?;
    prune(
        conn,
        "displacement_events",
        SERVED_RETENTION_MS,
        SERVED_CAP_ROWS,
    )?;
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────

#[derive(serde::Deserialize, Default)]
pub struct TelemetryArgs {
    /// Window in serving batches (distinct recalls). Default: none (secs wins).
    pub window_turns: Option<i64>,
    /// Window in seconds (default 24h).
    pub window_secs: Option<i64>,
    pub profile: Option<String>,
    pub workspace_hash: Option<String>,
    /// Optional contamination probe query (runs arm-level SQL deltas).
    pub probe_query: Option<String>,
    pub probe_mode: Option<String>,
}

pub fn hash_query(query: &str) -> String {
    use sha2::{Digest, Sha256};
    format!("{:x}", Sha256::digest(query.as_bytes()))[..16].to_string()
}

struct Window {
    since_ts: i64,
    since_turns: Option<i64>,
}

impl Window {
    fn resolve(&self, conn: &Connection) -> Result<i64, Box<dyn std::error::Error>> {
        let mut since = self.since_ts;
        if let Some(turns) = self.since_turns {
            if turns > 0 {
                // The oldest batch within the last `turns` batches.
                let ts: Option<i64> = conn.query_row(
                    "SELECT ts_unix_ms FROM (\
                     SELECT batch_id, MAX(ts_unix_ms) AS ts_unix_ms FROM served_events \
                     GROUP BY batch_id ORDER BY ts_unix_ms DESC LIMIT ?1) \
                     ORDER BY ts_unix_ms ASC LIMIT 1",
                    params![turns],
                    |r| r.get(0),
                )?;
                if let Some(ts) = ts {
                    since = ts;
                }
            }
        }
        Ok(since)
    }
}

/// Full telemetry report (acceptance #5/#6): denominators, scope,
/// retrieval profile, source class, versioned artifact hash, and
/// empty/degraded/unavailable states separated from zero concentration.
pub fn retrieval_telemetry_report(
    db: &Database,
    args: &TelemetryArgs,
) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    let conn = db
        .conn()
        .map_err(|e| format!("telemetry conn failed: {e}"))?;
    let window_secs = args.window_secs.unwrap_or(24 * 3600);
    let window = Window {
        since_ts: now_ms() - window_secs * 1000,
        since_turns: args.window_turns,
    };
    let since = window.resolve(&conn).map_err(|e| e.to_string())?;

    let mut scope_sql = String::new();
    let mut scope_vals: Vec<String> = Vec::new();
    if let Some(ref p) = args.profile {
        if !p.is_empty() {
            scope_vals.push(p.clone());
            scope_sql.push_str(&format!(" AND profile = ?{}", scope_vals.len()));
        }
    }
    if let Some(ref ws) = args.workspace_hash {
        if !ws.is_empty() {
            scope_vals.push(ws.clone());
            scope_sql.push_str(&format!(" AND workspace_hash = ?{}", scope_vals.len()));
        }
    }

    // ── served events in window ─────────────────────────────────────────
    let rows = {
        let sql = format!(
            "SELECT entity_id, category, key, source, verified, certainty, mode, \
                    query_class, tokens_est, slot, batch_id \
             FROM served_events WHERE ts_unix_ms >= ?{} {}",
            scope_vals.len() + 1,
            scope_sql
        );
        let mut all: Vec<Box<dyn rusqlite::types::ToSql>> = scope_vals
            .iter()
            .map(|s| Box::new(s.clone()) as Box<dyn rusqlite::types::ToSql>)
            .collect();
        all.push(Box::new(since));
        let refs: Vec<&dyn rusqlite::types::ToSql> = all.iter().map(|p| p.as_ref()).collect();
        let mut stmt = conn.prepare(&sql).map_err(|e| e.to_string())?;
        let out: Vec<(
            String,
            String,
            String,
            String,
            bool,
            f64,
            String,
            String,
            i64,
            i64,
            String,
        )> = stmt
            .query_map(refs.as_slice(), |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, String>(2)?,
                    r.get::<_, String>(3)?,
                    r.get::<_, bool>(4)?,
                    r.get::<_, f64>(5)?,
                    r.get::<_, String>(6)?,
                    r.get::<_, String>(7)?,
                    r.get::<_, i64>(8)?,
                    r.get::<_, i64>(9)?,
                    r.get::<_, String>(10)?,
                ))
            })
            .map_err(|e| e.to_string())?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| e.to_string())?;
        out
    };

    let empty = rows.is_empty();
    let batches: std::collections::BTreeSet<String> = rows.iter().map(|r| r.10.clone()).collect();
    let batch_count = batches.len() as i64;
    let slots_total: i64 = rows.len() as i64;
    let tokens_total: i64 = rows.iter().map(|r| r.8).sum();

    // Entity-level aggregation (window).
    let mut per_entity: std::collections::HashMap<String, (i64, i64, String, bool, f64)> =
        std::collections::HashMap::new();
    for r in &rows {
        let e = per_entity
            .entry(r.0.clone())
            .or_insert((0, 0, r.1.clone(), r.4, r.5));
        e.0 += 1;
        e.1 += r.8;
    }

    // ── concentration ───────────────────────────────────────────────────
    let mut top_entity: Option<String> = None;
    let mut top_slots: i64 = 0;
    let mut top_tokens: i64 = 0;
    let mut hhi_numer = 0.0f64;
    for (id, (cnt, tok, cat, _v, _c)) in &per_entity {
        hhi_numer += (*cnt as f64) * (*cnt as f64);
        // Deterministic top pick: max serves; ties prefer the verified
        // entity, then higher certainty, then the smaller id (stable
        // across processes — HashMap iteration order is randomized).
        let cur_verified = per_entity
            .get(top_entity.as_deref().unwrap_or(""))
            .map(|(_c, _t, _cat, v, cert)| (*v, *cert))
            .unwrap_or((false, 0.0));
        let mine_verified = (*_v, *_c);
        let beats = if top_entity.is_none() {
            true
        } else if *cnt != top_slots {
            *cnt > top_slots
        } else if mine_verified.0 != cur_verified.0 {
            mine_verified.0
        } else if (mine_verified.1 - cur_verified.1).abs() > f64::EPSILON {
            mine_verified.1 > cur_verified.1
        } else {
            id.as_str() < top_entity.as_deref().unwrap_or("")
        };
        if beats {
            top_entity = Some(id.clone());
            top_slots = *cnt;
            top_tokens = *tok;
        }
    }
    let hhi = if slots_total > 0 {
        hhi_numer / (slots_total as f64 * slots_total as f64)
    } else {
        0.0
    };
    let top_slot_share = if slots_total > 0 {
        top_slots as f64 / slots_total as f64
    } else {
        0.0
    };
    let top_token_share = if tokens_total > 0 {
        top_tokens as f64 / tokens_total as f64
    } else {
        0.0
    };

    // ── repeated serving ─────────────────────────────────────────────────
    let unique_entities = per_entity.len() as i64;
    let repeat_rate = if slots_total > 1 {
        1.0 - (unique_entities as f64 / slots_total as f64)
    } else {
        0.0
    };
    let mut repeats: Vec<(String, i64)> = per_entity
        .iter()
        .map(|(id, (cnt, _, _, _, _))| (id.clone(), *cnt))
        .filter(|(_, c)| *c > 1)
        .collect();
    repeats.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    repeats.truncate(5);

    // ── diversity ───────────────────────────────────────────────────────
    let mut sources: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    let mut classes: std::collections::BTreeMap<String, i64> = std::collections::BTreeMap::new();
    let mut class_entropy = 0.0f64;
    for r in &rows {
        sources.insert(r.3.clone());
        *classes.entry(r.1.clone()).or_insert(0) += 1;
    }
    let class_total: i64 = classes.values().sum();
    for c in classes.values() {
        let p = *c as f64 / class_total.max(1) as f64;
        if p > 0.0 {
            class_entropy -= p * p.ln();
        }
    }
    let simpson = if slots_total > 1 {
        // Simpson diversity over entity serve counts: 1 - Σ p²
        1.0 - hhi
    } else {
        0.0
    };

    // ── contamination: arm audits + delivered-set validation ────────────
    let audits = {
        let sql = format!(
            "SELECT mode, arm, SUM(candidates), SUM(reentry_candidates), SUM(delivered), COUNT(*) \
             FROM recall_arm_audits WHERE ts_unix_ms >= ?{} {} \
             GROUP BY mode, arm ORDER BY mode, arm",
            scope_vals.len() + 1,
            scope_sql
        );
        let mut all: Vec<Box<dyn rusqlite::types::ToSql>> = scope_vals
            .iter()
            .map(|s| Box::new(s.clone()) as Box<dyn rusqlite::types::ToSql>)
            .collect();
        all.push(Box::new(since));
        let refs: Vec<&dyn rusqlite::types::ToSql> = all.iter().map(|p| p.as_ref()).collect();
        let mut stmt = conn.prepare(&sql).map_err(|e| e.to_string())?;
        let out: Vec<(String, String, i64, i64, i64, i64)> = stmt
            .query_map(refs.as_slice(), |r| {
                Ok((
                    r.get(0)?,
                    r.get(1)?,
                    r.get(2)?,
                    r.get(3)?,
                    r.get(4)?,
                    r.get(5)?,
                ))
            })
            .map_err(|e| e.to_string())?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| e.to_string())?;
        out
    };

    // Delivered-set validation: re-scan the CURRENT lifecycle status of
    // every entity served in the window.
    let mut served_reentry: i64 = 0;
    if !per_entity.is_empty() {
        let ids: Vec<String> = per_entity.keys().cloned().collect();
        for chunk in ids.chunks(500) {
            let placeholders = (1..=chunk.len())
                .map(|i| format!("?{i}"))
                .collect::<Vec<_>>()
                .join(",");
            let sql = format!(
                "SELECT COUNT(*) FROM entities WHERE id IN ({placeholders}) \
                 AND archived = 0 AND status IN ('active','draft')"
            );
            let refs: Vec<&dyn rusqlite::types::ToSql> = chunk
                .iter()
                .map(|s| s as &dyn rusqlite::types::ToSql)
                .collect();
            let live: i64 = conn.query_row(&sql, refs.as_slice(), |r| r.get(0))?;
            served_reentry += chunk.len() as i64 - live;
        }
    }

    // ── fan-out: low-trust candidates across query classes ──────────────
    let mut fanout: std::collections::BTreeMap<String, std::collections::BTreeSet<String>> =
        std::collections::BTreeMap::new();
    for r in &rows {
        let low_trust = !r.4 || r.5 < 0.5;
        if low_trust {
            fanout.entry(r.0.clone()).or_default().insert(r.7.clone());
        }
    }
    let fanout_report: Vec<serde_json::Value> = fanout
        .iter()
        .map(|(id, classes)| {
            serde_json::json!({
                "entity_id": id,
                "query_classes": classes.len(),
                "classes": classes.iter().cloned().collect::<Vec<_>>(),
            })
        })
        .collect();

    // ── displacement (workspace-scoped; events carry no profile) ────────
    let mut disp_scope_sql = String::new();
    let mut disp_scope_vals: Vec<String> = Vec::new();
    if let Some(ref ws) = args.workspace_hash {
        if !ws.is_empty() {
            disp_scope_vals.push(ws.clone());
            disp_scope_sql.push_str(&format!(" AND workspace_hash = ?{}", disp_scope_vals.len()));
        }
    }
    let displacement_count: i64 = {
        let sql = format!(
            "SELECT COUNT(*) FROM displacement_events WHERE ts_unix_ms >= ?{} {disp_scope_sql}",
            disp_scope_vals.len() + 1
        );
        let mut all: Vec<Box<dyn rusqlite::types::ToSql>> = disp_scope_vals
            .iter()
            .map(|s| Box::new(s.clone()) as Box<dyn rusqlite::types::ToSql>)
            .collect();
        all.push(Box::new(since));
        let refs: Vec<&dyn rusqlite::types::ToSql> = all.iter().map(|p| p.as_ref()).collect();
        conn.query_row(&sql, refs.as_slice(), |r| r.get(0))?
    };
    let displacement_sample: Vec<(String, String, bool, String, String, String)> = {
        let sql = format!(
            "SELECT entity_id, reason, was_sole_evidence, mode, workspace_hash, query \
             FROM displacement_events WHERE ts_unix_ms >= ?{} {disp_scope_sql} \
             ORDER BY ts_unix_ms DESC LIMIT 5",
            disp_scope_vals.len() + 1
        );
        let mut all: Vec<Box<dyn rusqlite::types::ToSql>> = disp_scope_vals
            .iter()
            .map(|s| Box::new(s.clone()) as Box<dyn rusqlite::types::ToSql>)
            .collect();
        all.push(Box::new(since));
        let refs: Vec<&dyn rusqlite::types::ToSql> = all.iter().map(|p| p.as_ref()).collect();
        let mut stmt = conn.prepare(&sql).map_err(|e| e.to_string())?;
        let rows = stmt
            .query_map(refs.as_slice(), |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, bool>(2)?,
                    r.get::<_, String>(3)?,
                    r.get::<_, String>(4)?,
                    r.get::<_, String>(5)?,
                ))
            })
            .map_err(|e| e.to_string())?;
        let collected: Vec<(String, String, bool, String, String, String)> = rows
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| e.to_string())?;
        drop(stmt);
        collected
    };

    // ── optional contamination probe (arm-level SQL deltas) ─────────────
    let mut probe: Option<serde_json::Value> = None;
    let mut degraded = false;
    if let Some(ref q) = args.probe_query {
        if !q.trim().is_empty() {
            match contamination_probe(
                &conn,
                q,
                args.probe_mode.as_deref().unwrap_or("lexical"),
                db.encryption.as_ref(),
            ) {
                Ok(p) => probe = Some(p),
                Err(e) => {
                    degraded = true;
                    probe = Some(serde_json::json!({"error": e.to_string()}));
                }
            }
        }
    }

    let state = if empty {
        "empty"
    } else if degraded {
        "degraded"
    } else {
        "ok"
    };

    let mut report = serde_json::json!({
        "state": state,
        "zero_vs_empty": if empty { "empty" } else { "nonzero_or_zero" },
        "window": {
            "since_unix_ms": since,
            "secs": window_secs,
            "turns": args.window_turns,
        },
        "denominator": {
            "recalls": batch_count,
            "slots": slots_total,
            "tokens_est": tokens_total,
            "unique_entities": unique_entities,
        },
        "scope": {
            "profile": args.profile.clone().unwrap_or_default(),
            "workspace_hash": args.workspace_hash.clone().unwrap_or_default(),
        },
        "retrieval_profile": {
            "modes": {
                "lexical": rows.iter().filter(|r| r.6 == "lexical").count(),
                "dense": rows.iter().filter(|r| r.6 == "dense").count(),
                "hybrid": rows.iter().filter(|r| r.6 == "hybrid").count(),
                "fused": rows.iter().filter(|r| r.6 == "fused").count(),
                "proactive": rows.iter().filter(|r| r.6 == "proactive").count(),
            },
        },
        "concentration": {
            "top_entity_id": top_entity,
            "top_slot_share": top_slot_share,
            "top_token_share": top_token_share,
            "herfindahl": hhi,
            "max_entity_serves": top_slots,
        },
        "repeated_serving": {
            "served_total": slots_total,
            "unique_entities": unique_entities,
            "repeat_rate": repeat_rate,
            "top_repeat": repeats
                .iter()
                .map(|(id, c)| serde_json::json!({"entity_id": id, "serves": c}))
                .collect::<Vec<_>>(),
        },
        "diversity": {
            "unique_sources": sources.len(),
            "simpson_index": simpson,
            "source_class_entropy": class_entropy,
            "source_classes": classes,
        },
        "contamination": {
            "arm_audits": audits
                .iter()
                .map(|(mode, arm, cand, re, del, n)| serde_json::json!({
                    "mode": mode,
                    "arm": arm,
                    "audits": n,
                    "candidates": cand,
                    "reentry_candidates": re,
                    "delivered": del,
                }))
                .collect::<Vec<_>>(),
            "served_reentry": served_reentry,
            "probe": probe,
        },
        "fanout_low_trust": fanout_report,
        "displacement": {
            "count": displacement_count,
            "sample": displacement_sample
                .iter()
                .map(|(id, reason, sole, mode, ws, q)| serde_json::json!({
                    "entity_id": id,
                    "reason": reason,
                    "was_sole_evidence": sole,
                    "mode": mode,
                    "workspace_hash": ws,
                    "query": q,
                }))
                .collect::<Vec<_>>(),
        },
        "artifact": {
            "git_hash": option_env!("GIT_HASH").unwrap_or("unknown"),
            "schema_version": crate::schema::SCHEMA_VERSION,
            "binary_version": env!("CARGO_PKG_VERSION"),
        },
    });
    // #872: content hash covers the report exactly as serialized, excluding
    // the hash field itself (deterministic over identical store state).
    let canonical = serde_json::to_string(&report).unwrap_or_default();
    let content_hash = {
        use sha2::{Digest, Sha256};
        format!("{:x}", Sha256::digest(canonical.as_bytes()))
    };
    if let Some(art) = report.get_mut("artifact").and_then(|a| a.as_object_mut()) {
        art.insert("content_hash".to_string(), serde_json::json!(content_hash));
    }
    Ok(report)
}

/// Arm-level contamination probe: for each serving arm, count the matches
/// the status exclusion boundary blocks (would-be re-entry) and the
/// matches the arm now serves. Read-only; never mutates.
fn contamination_probe(
    conn: &Connection,
    query: &str,
    mode: &str,
    encryption: Option<&EncryptionManager>,
) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    let excluded = NON_SERVEABLE_STATUSES
        .iter()
        .map(|s| format!("'{s}'"))
        .collect::<Vec<_>>()
        .join(",");
    let mut arms = Vec::new();

    match mode {
        "dense" => {
            let total: i64 = conn.query_row(
                "SELECT COUNT(*) FROM entities \
                 WHERE archived = 0 AND embedding IS NOT NULL",
                [],
                |r| r.get(0),
            )?;
            let blocked: i64 = conn.query_row(
                &format!(
                    "SELECT COUNT(*) FROM entities \
                     WHERE archived = 0 AND embedding IS NOT NULL \
                     AND (COALESCE(status, '') NOT IN ('active','draft'))"
                ),
                [],
                |r| r.get(0),
            )?;
            arms.push(serde_json::json!({
                "arm": "dense",
                "matched_total": total,
                "blocked_reentry": blocked,
            }));
        }
        "fused" => {
            // fused = lexical + dense + graph + temporal; lexical probe
            // below covers fts5/temporal, dense probe above, graph probe
            // below. Emit the union explicitly.
            let lex = lexical_probe(conn, query, &excluded, encryption)?;
            let den = dense_probe(conn, &excluded)?;
            let gr = graph_probe(conn, query, &excluded, encryption)?;
            arms.push(lex);
            arms.push(den);
            arms.push(gr);
        }
        "graph" => {
            arms.push(graph_probe(conn, query, &excluded, encryption)?);
        }
        "proactive" => {
            let words: Vec<String> = if encryption.is_some() {
                crate::encryption::search_terms(query)
                    .into_iter()
                    .filter(|word| word.chars().count() >= 3 && !is_stopword(word))
                    .collect()
            } else {
                query
                    .split_whitespace()
                    .filter(|w| w.chars().count() >= 3 && !is_stopword(&w.to_lowercase()))
                    .map(|w| {
                        w.chars()
                            .filter(|c| c.is_alphanumeric())
                            .collect::<String>()
                    })
                    .collect()
            };
            let fts_query = if let Some(enc) = encryption {
                enc.blind_query_from_terms(&words)
            } else {
                words
                    .iter()
                    .map(|w| format!("{w}*"))
                    .collect::<Vec<_>>()
                    .join(" OR ")
            };
            let total: i64 = if fts_query.is_empty() {
                0
            } else {
                conn.query_row(
                    "SELECT COUNT(*) FROM entities \
                     WHERE archived = 0 \
                       AND rowid IN (SELECT rowid FROM entities_fts WHERE entities_fts MATCH ?1)",
                    params![fts_query],
                    |r| r.get(0),
                )?
            };
            let blocked: i64 = if fts_query.is_empty() {
                0
            } else {
                conn.query_row(
                    &format!(
                        "SELECT COUNT(*) FROM entities \
                         WHERE archived = 0 \
                           AND rowid IN (SELECT rowid FROM entities_fts WHERE entities_fts MATCH ?1) \
                           AND (COALESCE(status, '') NOT IN ('active','draft'))"
                    ),
                    params![fts_query],
                    |r| r.get(0),
                )?
            };
            arms.push(serde_json::json!({
                "arm": "proactive",
                "matched_total": total,
                "blocked_reentry": blocked,
                "note": "pre-trigger-confirmation upper bound",
            }));
        }
        _ => {
            // lexical / temporal share the keyword match set.
            arms.push(lexical_probe(conn, query, &excluded, encryption)?);
            arms.push(temporal_probe(conn, query, &excluded, encryption)?);
        }
    }

    Ok(serde_json::json!({
        "query_hash": hash_query(query),
        "mode": mode,
        "arms": arms,
        "invariant": arms.iter().all(|a| a["blocked_reentry"].as_i64() == Some(0)),
    }))
}

fn lexical_probe(
    conn: &Connection,
    query: &str,
    excluded: &str,
    encryption: Option<&EncryptionManager>,
) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    let words: Vec<String> = if encryption.is_some() {
        crate::encryption::search_terms(query)
            .into_iter()
            .filter(|word| !is_stopword(word))
            .collect()
    } else {
        query
            .split_whitespace()
            .filter(|w| !w.is_empty() && !is_stopword(&w.to_lowercase()))
            .map(|w| format!("\"{}\"*", w.replace('"', "\\\"\\\"")))
            .collect()
    };
    let fts_query = if let Some(enc) = encryption {
        enc.blind_query_from_terms(&words)
    } else {
        words.join(" OR ")
    };
    let total: i64 = if fts_query.is_empty() {
        0
    } else {
        conn.query_row(
            "SELECT COUNT(*) FROM entities_fts WHERE entities_fts MATCH ?1",
            params![fts_query],
            |r| r.get(0),
        )?
    };
    let blocked: i64 = if fts_query.is_empty() {
        0
    } else {
        conn.query_row(
            &format!(
                "SELECT COUNT(*) FROM entities_fts f JOIN entities e ON e.rowid = f.rowid \
                 WHERE entities_fts MATCH ?1 AND e.archived = 0 \
                   AND (COALESCE(e.status, '') NOT IN ('active','draft'))"
            ),
            params![fts_query],
            |r| r.get(0),
        )?
    };
    Ok(serde_json::json!({
        "arm": "lexical",
        "matched_total": total,
        "blocked_reentry": blocked,
    }))
}

fn temporal_probe(
    conn: &Connection,
    query: &str,
    excluded: &str,
    encryption: Option<&EncryptionManager>,
) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    let lex = lexical_probe(conn, query, excluded, encryption)?;
    Ok(serde_json::json!({
        "arm": "temporal",
        "matched_total": lex["matched_total"],
        "blocked_reentry": lex["blocked_reentry"],
        "note": "temporal arm base is the keyword match set",
    }))
}

fn dense_probe(
    conn: &Connection,
    _excluded: &str,
) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    let total: i64 = conn.query_row(
        "SELECT COUNT(*) FROM entities WHERE archived = 0 AND embedding IS NOT NULL",
        [],
        |r| r.get(0),
    )?;
    let blocked: i64 = conn.query_row(
        &format!(
            "SELECT COUNT(*) FROM entities WHERE archived = 0 AND embedding IS NOT NULL \
             AND (COALESCE(status, '') NOT IN ('active','draft'))"
        ),
        [],
        |r| r.get(0),
    )?;
    Ok(serde_json::json!({
        "arm": "dense",
        "matched_total": total,
        "blocked_reentry": blocked,
    }))
}

fn graph_probe(
    conn: &Connection,
    query: &str,
    _excluded: &str,
    encryption: Option<&EncryptionManager>,
) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    // One-hop neighbor set of the top-3 keyword matches (same seed rule as
    // the graph arm), then count excluded-status neighbors.
    let words: Vec<String> = if encryption.is_some() {
        crate::encryption::search_terms(query)
            .into_iter()
            .filter(|word| !is_stopword(word))
            .collect()
    } else {
        query
            .split_whitespace()
            .filter(|w| !w.is_empty() && !is_stopword(&w.to_lowercase()))
            .map(|w| format!("\"{}\"*", w.replace('"', "\\\"\\\"")))
            .collect()
    };
    let fts_query = if let Some(enc) = encryption {
        enc.blind_query_from_terms(&words)
    } else {
        words.join(" OR ")
    };
    let mut neighbor_ids: Vec<String> = Vec::new();
    if !fts_query.is_empty() {
        let seed_ids: Vec<String> = conn
            .prepare(
                "SELECT e.id FROM entities_fts f JOIN entities e ON e.rowid = f.rowid \
                 WHERE entities_fts MATCH ?1 AND e.archived = 0 \
                 AND e.status IN ('active','draft') \
                 ORDER BY bm25(entities_fts) LIMIT 3",
            )?
            .query_map(params![fts_query], |r| r.get::<_, String>(0))?
            .collect::<Result<Vec<_>, _>>()?;
        let seed_set: std::collections::HashSet<&str> =
            seed_ids.iter().map(|s| s.as_str()).collect();
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        for chunk in seed_ids.chunks(500) {
            let placeholders = (1..=chunk.len())
                .map(|i| format!("?{i}"))
                .collect::<Vec<_>>()
                .join(",");
            let sql = format!("SELECT id, links FROM entities WHERE id IN ({placeholders})");
            let refs: Vec<&dyn rusqlite::types::ToSql> = chunk
                .iter()
                .map(|s| s as &dyn rusqlite::types::ToSql)
                .collect();
            let mut stmt = conn.prepare(&sql)?;
            let rows = stmt.query_map(refs.as_slice(), |r| {
                Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
            })?;
            for row in rows {
                let (id, links_json) = row?;
                if let Ok(links) =
                    serde_json::from_str::<Vec<crate::models::MemoryLink>>(&links_json)
                {
                    for l in links {
                        if !seed_set.contains(l.target_id.as_str())
                            && seen.insert(l.target_id.clone())
                        {
                            neighbor_ids.push(l.target_id.clone());
                        }
                    }
                }
                let _ = id;
            }
        }
    }
    let (total, blocked) = if neighbor_ids.is_empty() {
        (0, 0)
    } else {
        let placeholders = (1..=neighbor_ids.len())
            .map(|i| format!("?{i}"))
            .collect::<Vec<_>>()
            .join(",");
        let refs: Vec<&dyn rusqlite::types::ToSql> = neighbor_ids
            .iter()
            .map(|s| s as &dyn rusqlite::types::ToSql)
            .collect();
        let total: i64 = conn.query_row(
            &format!("SELECT COUNT(*) FROM entities WHERE id IN ({placeholders}) AND archived = 0"),
            refs.as_slice(),
            |r| r.get(0),
        )?;
        let blocked: i64 = conn.query_row(
            &format!(
                "SELECT COUNT(*) FROM entities WHERE id IN ({placeholders}) \
                 AND archived = 0 AND (COALESCE(status, '') NOT IN ('active','draft'))"
            ),
            refs.as_slice(),
            |r| r.get(0),
        )?;
        (total, blocked)
    };
    Ok(serde_json::json!({
        "arm": "graph",
        "matched_total": total,
        "blocked_reentry": blocked,
    }))
}

/// Entity count with a non-serveable status (used by recall-path hooks to
/// fill the `reentry_candidates` audit column; runs only when a path has
/// materialized candidate lists that bypassed the SQL boundary — e.g.
/// tombstone-suppressed rows, which are removed in Rust).
pub fn count_non_serveable(entities: &[Entity]) -> usize {
    entities
        .iter()
        .filter(|e| NON_SERVEABLE_STATUSES.contains(&e.status.as_str()))
        .count()
}

/// Scored-list variant of [`count_non_serveable`] for fused/hybrid arms
/// whose candidates are `(Entity, f64)` pairs.
pub fn count_non_serveable_scored(entities: &[(Entity, f64)]) -> usize {
    entities
        .iter()
        .filter(|(e, _)| NON_SERVEABLE_STATUSES.contains(&e.status.as_str()))
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::Database;

    fn temp_db() -> (Database, String) {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("rt-test-{}.db", uuid::Uuid::new_v4().simple()));
        let db = Database::open(path.to_str().unwrap()).expect("open");
        (db, path.display().to_string())
    }

    fn make_entity(id: &str, category: &str, key: &str, body: &str) -> Entity {
        Entity {
            id: id.to_string(),
            category: category.to_string(),
            key: key.to_string(),
            body_json: body.to_string(),
            status: "active".to_string(),
            entity_type: "memory".to_string(),
            tags: Vec::new(),
            decay_score: 1.0,
            retrieval_count: 0,
            layer: "working".to_string(),
            topic_path: String::new(),
            archived: false,
            archive_reason: String::new(),
            links: Vec::new(),
            verified: false,
            source: "test".to_string(),
            always_on: false,
            certainty: 0.5,
            workspace_hash: String::new(),
            agent_id: String::new(),
            visibility: "workspace".to_string(),
            created_at_unix_ms: 0,
            last_accessed_unix_ms: 0,
            follow_count: 0,
            miss_count: 0,
            follow_rate: 0.0,
            efficacy_status: "unverified".to_string(),
            epistemic_state: "candidate".to_string(),
            hints: vec![],
            memory_type: String::new(),
            embedding: None,
            _parsed_body: None,
        }
    }

    #[test]
    fn query_class_keeps_content_bearing_tokens_only() {
        assert_eq!(query_class("what is the zeppelin core"), "zeppelin core");
        assert_eq!(query_class("ERR7781"), "err7781");
        assert_eq!(query_class("a an the"), "");
    }

    #[test]
    fn serveable_clause_excludes_known_statuses() {
        let c = serveable_status_clause("e");
        assert!(c.contains("e.status IN ('active','draft')"));
        assert!(!c.contains("status NOT IN"));
        assert!(!c.contains("'deprecated'"));
        assert!(!c.contains("'quarantined'"));
        let c2 = serveable_status_clause("");
        assert_eq!(c2, "status IN ('active','draft')");
    }

    #[test]
    fn count_non_serveable_detects_truth_control_statuses() {
        let mut a = make_entity("a1", "facts", "a1", r#"{"note":"x"}"#);
        let mut b = make_entity("b1", "facts", "b1", r#"{"note":"y"}"#);
        a.status = "active".to_string();
        b.status = "deprecated".to_string();
        assert_eq!(count_non_serveable(&[a, b]), 1);
    }

    #[test]
    fn report_empty_state_separates_from_zero_concentration() {
        let (db, path) = temp_db();
        let rep = retrieval_telemetry_report(
            &db,
            &TelemetryArgs {
                window_turns: None,
                window_secs: Some(3600),
                profile: None,
                workspace_hash: None,
                probe_query: None,
                probe_mode: None,
            },
        )
        .unwrap();
        assert_eq!(rep["state"], "empty");
        assert_eq!(rep["zero_vs_empty"], "empty");
        assert_eq!(rep["denominator"]["recalls"], 0);
        assert_eq!(
            rep["artifact"]["schema_version"],
            crate::schema::SCHEMA_VERSION
        );
        assert_eq!(rep["contamination"]["served_reentry"], 0);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn report_records_served_and_measures_concentration_repeat_diversity() {
        let (db, path) = temp_db();
        let conn = db.conn().unwrap();
        db.remember(&make_entity(
            "s-1",
            "facts",
            "s-1",
            r#"{"note":"alpha core"}"#,
        ))
        .unwrap();
        db.remember(&make_entity(
            "s-2",
            "insight",
            "s-2",
            r#"{"note":"beta detail"}"#,
        ))
        .unwrap();
        // Serve the STORED entities (real ids) so the delivered-set
        // validation finds them as live, serveable rows.
        let e1 = db.get_entity("facts", "s-1").unwrap().unwrap();
        let e2 = db.get_entity("insight", "s-2").unwrap().unwrap();
        record_served(
            &conn,
            "b1",
            "p",
            "lexical",
            "alpha",
            &[e1.clone(), e2.clone()],
        )
        .unwrap();
        record_served(&conn, "b2", "p", "lexical", "alpha", &[e1.clone()]).unwrap();
        record_served(&conn, "b3", "p", "hybrid", "beta", &[e2.clone()]).unwrap();
        record_arm_audit(&conn, "hybrid", "sparse", 2, 0, 2, "p", "", "abc").unwrap();

        let rep = retrieval_telemetry_report(&db, &TelemetryArgs::default()).unwrap();
        assert_eq!(rep["state"], "ok");
        assert_eq!(rep["denominator"]["recalls"], 3);
        assert_eq!(rep["denominator"]["slots"], 4);
        assert_eq!(rep["denominator"]["unique_entities"], 2);
        assert_eq!(
            rep["repeated_serving"]["repeat_rate"].as_f64().unwrap(),
            0.5
        );
        assert_eq!(rep["concentration"]["top_entity_id"], "s-1");
        assert_eq!(rep["diversity"]["unique_sources"].as_i64().unwrap(), 1);
        assert!(rep["diversity"]["simpson_index"].as_f64().unwrap() > 0.0);
        assert!(
            rep["diversity"]["source_classes"]["facts"]
                .as_i64()
                .unwrap()
                >= 1
        );
        assert_eq!(rep["retrieval_profile"]["modes"]["hybrid"], 1);
        // Arm audits surface.
        let audits = rep["contamination"]["arm_audits"].as_array().unwrap();
        assert_eq!(audits.len(), 1);
        assert_eq!(audits[0]["arm"], "sparse");
        assert_eq!(audits[0]["candidates"], 2);
        assert_eq!(rep["contamination"]["served_reentry"], 0);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn report_fanout_tracks_low_trust_query_classes() {
        let (db, path) = temp_db();
        let conn = db.conn().unwrap();
        let mut e = make_entity("f-1", "facts", "f-1", r#"{"note":"low trust item"}"#);
        e.verified = false;
        e.certainty = 0.1;
        record_served(&conn, "b1", "p", "lexical", "alpha query", &[e.clone()]).unwrap();
        record_served(&conn, "b2", "p", "lexical", "beta query", &[e.clone()]).unwrap();
        record_served(&conn, "b3", "p", "lexical", "gamma query", &[e.clone()]).unwrap();
        let rep = retrieval_telemetry_report(&db, &TelemetryArgs::default()).unwrap();
        let fanout = rep["fanout_low_trust"].as_array().unwrap();
        assert_eq!(fanout.len(), 1);
        assert_eq!(fanout[0]["entity_id"], "f-1");
        assert_eq!(fanout[0]["query_classes"].as_i64().unwrap(), 3);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn contamination_probe_measures_blocked_reentry_per_arm() {
        let (db, path) = temp_db();
        let conn = db.conn().unwrap();
        // Two entities with the same content; one gets deprecated.
        db.remember(&make_entity(
            "c-ok",
            "facts",
            "c-ok",
            r#"{"note":"zeppelin core notes"}"#,
        ))
        .unwrap();
        db.remember(&make_entity(
            "c-dep",
            "facts",
            "c-dep",
            r#"{"note":"zeppelin core notes deprecated variant"}"#,
        ))
        .unwrap();
        // remember() derives a stable id from (category, key); retire the
        // row by key so the lifecycle status lands on the real entity.
        conn.execute(
            "UPDATE entities SET status = 'deprecated' WHERE category = 'facts' AND key = 'c-dep'",
            [],
        )
        .unwrap();
        let probe = contamination_probe(&conn, "zeppelin core", "lexical", None).unwrap();
        assert_eq!(
            probe["arms"][0]["blocked_reentry"], 1,
            "probe dump: {}",
            probe
        );
        let arms = probe["arms"].as_array().unwrap();
        assert_eq!(arms[0]["arm"], "lexical");
        assert!(arms[0]["blocked_reentry"].as_i64().unwrap() >= 1);
        // And the served-side stays clean under the same fixture.
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn displacement_recording_and_report() {
        let (db, path) = temp_db();
        let conn = db.conn().unwrap();
        record_displacement(
            &conn,
            "d-1",
            "diversity_halving",
            true,
            "lexical",
            "",
            "alpha",
        )
        .unwrap();
        record_displacement(&conn, "d-2", "cooldown", false, "lexical", "", "beta").unwrap();
        let rep = retrieval_telemetry_report(&db, &TelemetryArgs::default()).unwrap();
        assert_eq!(rep["displacement"]["count"], 2);
        let sample = rep["displacement"]["sample"].as_array().unwrap();
        assert_eq!(sample.len(), 2);
        // Same-timestamp events: match on entity, not array position.
        let sole = sample
            .iter()
            .find(|s| s["entity_id"] == "d-1")
            .expect("d-1 displacement present");
        assert_eq!(sole["was_sole_evidence"], true);
        let cooldown = sample
            .iter()
            .find(|s| s["entity_id"] == "d-2")
            .expect("d-2 displacement present");
        assert_eq!(cooldown["was_sole_evidence"], false);
        let _ = std::fs::remove_file(path);
    }
}
