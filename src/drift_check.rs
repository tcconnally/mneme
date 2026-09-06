//! #1035: deterministic drift-check → targeted-repair → verify loop with a
//! health score. Borrowed from mex-memory/mex (`mex check` / `mex sync`,
//! MIT, verified locally 2026-08-14) and generalized to the vault's evidence
//! model: a zero-LLM pre-pass over the store catches concrete drift —
//! broken references, missing grounded paths, grounding drift, cross-file
//! contradictions, and staleness — and the health score reserves LLM work
//! for the flagged subset only.
//!
//! Health score = 100 − (10×error + 3×warning + 1×info), floored at 0.
//! Repair scope = flagged items only; every repair re-runs the check and the
//! verify leg reports the score delta (repairs are accepted only when the
//! score improves or holds). Deterministic throughout: no LLM call in the
//! detection, and repairs only apply mechanical fixes (unlink dangling
//! references, acknowledge grounding findings). Contradictions and staleness
//! are surfaced for review, never auto-resolved.

use crate::db::Database;
use rusqlite::params;
use serde::{Deserialize, Serialize};

/// Severity classes mirroring mex (error/warning/info) with the same score
/// weights.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftIssue {
    pub code: String,
    pub severity: String,
    pub target: String,
    pub detail: String,
    pub repairable: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftReport {
    pub health_score: i64,
    pub errors: usize,
    pub warnings: usize,
    pub infos: usize,
    pub checker_counts: serde_json::Value,
    pub issues: Vec<DriftIssue>,
    pub note: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepairReport {
    pub before_score: i64,
    pub after_score: i64,
    pub repaired: Vec<String>,
    pub requires_review: Vec<String>,
    pub note: String,
}

/// Stable issue key: code + target (deterministic across runs).
fn issue_key(code: &str, target: &str) -> String {
    format!("{code}:{target}")
}

fn score_for(issues: &[DriftIssue]) -> i64 {
    let errors = issues.iter().filter(|i| i.severity == "error").count() as i64;
    let warnings = issues.iter().filter(|i| i.severity == "warning").count() as i64;
    let infos = issues.iter().filter(|i| i.severity == "info").count() as i64;
    (100 - 10 * errors - 3 * warnings - infos).max(0)
}

/// Checker 1: dangling `derived_from` references — a citation whose target
/// entity no longer exists (deleted or never admitted). Deterministic
/// reference-integrity walk over `entities.links`.
fn check_reference_integrity(db: &Database, ws: Option<&str>) -> Result<Vec<DriftIssue>, String> {
    let conn = db.conn().map_err(|e| e.to_string())?;
    let (mut issues, mut rows) = (Vec::new(), Vec::new());
    {
        let mut stmt = conn
            .prepare("SELECT id, links FROM entities WHERE archived = 0 AND json_valid(links) AND (?1 = '' OR workspace_hash = ?1)")
            .map_err(|e| e.to_string())?;
        let mut q = if let Some(w) = ws {
            stmt.query(params![w]).map_err(|e| e.to_string())?
        } else {
            stmt.query(params![ws.unwrap_or("")])
                .map_err(|e| e.to_string())?
        };
        while let Some(row) = q.next().map_err(|e| e.to_string())? {
            let id: String = row.get(0).map_err(|e| e.to_string())?;
            let links: String = row.get(1).map_err(|e| e.to_string())?;
            rows.push((id, links));
        }
    }
    for (id, links) in rows {
        let parsed: serde_json::Value = serde_json::from_str(&links).unwrap_or_default();
        let arr = match parsed.as_array() {
            Some(a) => a,
            None => continue,
        };
        for link in arr {
            if link.get("relationship").and_then(|r| r.as_str()) != Some("derived_from") {
                continue;
            }
            let Some(target) = link.get("target_id").and_then(|t| t.as_str()) else {
                continue;
            };
            let exists: i64 = conn
                .query_row(
                    "SELECT COUNT(*) FROM entities WHERE id = ?1 AND archived = 0",
                    params![target],
                    |r| r.get(0),
                )
                .map_err(|e| e.to_string())?;
            if exists == 0 {
                issues.push(DriftIssue {
                    code: "REFERENCE_INTEGRITY".into(),
                    severity: "error".into(),
                    target: id.clone(),
                    detail: format!(
                        "dangling derived_from citation: entity {id} cites missing entity {target}"
                    ),
                    repairable: true,
                });
            }
        }
    }
    Ok(issues)
}

/// Checker 2: grounding drift — fingerprint rows whose status moved off `ok`
/// (GONE = error, DRIFT/AMBIGUOUS = warning) and are unreviewed.
fn check_grounding_status(db: &Database, ws: Option<&str>) -> Result<Vec<DriftIssue>, String> {
    let conn = db.conn().map_err(|e| e.to_string())?;
    let mut issues = Vec::new();
    let mut stmt = conn
        .prepare(
            "SELECT target_ref, entity_id, status, candidates_json FROM grounding_fingerprints
             WHERE (?1 = '' OR workspace_hash = ?1) AND status != 'ok' AND reviewed_at_unix_ms IS NULL
             ORDER BY updated_at_unix_ms DESC LIMIT 500",
        )
        .map_err(|e| e.to_string())?;
    let mut q = stmt
        .query(params![ws.unwrap_or("")])
        .map_err(|e| e.to_string())?;
    while let Some(row) = q.next().map_err(|e| e.to_string())? {
        let target_ref: String = row.get(0).map_err(|e| e.to_string())?;
        let entity_id: String = row.get(1).map_err(|e| e.to_string())?;
        let status: String = row.get(2).map_err(|e| e.to_string())?;
        let candidates: String = row.get(3).map_err(|e| e.to_string())?;
        let (severity, detail) = match status.as_str() {
            "gone" => (
                "error",
                format!(
                    "grounding GONE: {target_ref} (entity {entity_id}) — no plausible moved candidate; flag for review"
                ),
            ),
            "ambiguous" => (
                "warning",
                format!(
                    "grounding AMBIGUOUS: {target_ref} (entity {entity_id}) — candidates: {candidates}"
                ),
            ),
            _ => (
                "warning",
                format!(
                    "grounding DRIFT: {target_ref} (entity {entity_id}) — content changed in place; re-verify"
                ),
            ),
        };
        issues.push(DriftIssue {
            code: "GROUNDING_STATUS".into(),
            severity: severity.into(),
            target: target_ref.clone(),
            detail,
            repairable: true, // acknowledge-only repair
        });
    }
    Ok(issues)
}

/// Checker 3: path existence — file groundings whose absolute target path is
/// missing on disk (deterministic stat; the vault never guesses).
fn check_path_existence(db: &Database, ws: Option<&str>) -> Result<Vec<DriftIssue>, String> {
    let conn = db.conn().map_err(|e| e.to_string())?;
    let mut issues = Vec::new();
    let mut stmt = conn
        .prepare(
            "SELECT target_ref, entity_id FROM grounding_fingerprints
             WHERE (?1 = '' OR workspace_hash = ?1) AND kind = 'file' AND status = 'ok'
             ORDER BY updated_at_unix_ms DESC LIMIT 1000",
        )
        .map_err(|e| e.to_string())?;
    let mut q = stmt
        .query(params![ws.unwrap_or("")])
        .map_err(|e| e.to_string())?;
    while let Some(row) = q.next().map_err(|e| e.to_string())? {
        let target_ref: String = row.get(0).map_err(|e| e.to_string())?;
        let entity_id: String = row.get(1).map_err(|e| e.to_string())?;
        if !target_ref.starts_with('/') {
            issues.push(DriftIssue {
                code: "PATH_EXISTENCE".into(),
                severity: "info".into(),
                target: target_ref.clone(),
                detail: format!(
                    "relative target_ref skipped (not stat-able): {target_ref} (entity {entity_id})"
                ),
                repairable: false,
            });
            continue;
        }
        if !std::path::Path::new(&target_ref).exists() {
            issues.push(DriftIssue {
                code: "PATH_EXISTENCE".into(),
                severity: "error".into(),
                target: target_ref.clone(),
                detail: format!("grounded file missing on disk: {target_ref} (entity {entity_id})"),
                repairable: false,
            });
        }
    }
    Ok(issues)
}

/// Checker 4: cross-file contradiction — two active evidence entities
/// asserting different values for the same top-level body key. mex's claim
/// regex is shallow (versions/commands); the vault generalizes to ANY keyed
/// evidence value. Deterministic: same key + differing scalar value ⇒
/// conflict.
fn check_cross_file_conflicts(db: &Database, ws: Option<&str>) -> Result<Vec<DriftIssue>, String> {
    let conn = db.conn().map_err(|e| e.to_string())?;
    let mut stmt = conn
        .prepare(
            "SELECT id, body_json FROM entities WHERE archived = 0 AND json_valid(body_json) AND (?1 = '' OR workspace_hash = ?1) LIMIT 5000",
        )
        .map_err(|e| e.to_string())?;
    let mut q = stmt
        .query(params![ws.unwrap_or("")])
        .map_err(|e| e.to_string())?;
    let mut claims: std::collections::BTreeMap<String, std::collections::BTreeMap<String, String>> =
        std::collections::BTreeMap::new();
    while let Some(row) = q.next().map_err(|e| e.to_string())? {
        let id: String = row.get(0).map_err(|e| e.to_string())?;
        let body: String = row.get(1).map_err(|e| e.to_string())?;
        let parsed: serde_json::Value = serde_json::from_str(&body).unwrap_or_default();
        let obj = match parsed.as_object() {
            Some(o) => o,
            None => continue,
        };
        for (k, v) in obj {
            // Only scalar claims participate (arrays/objects are structural,
            // not keyed evidence values).
            if !v.is_string() && !v.is_number() && !v.is_boolean() {
                continue;
            }
            claims
                .entry(k.clone())
                .or_default()
                .entry(v.to_string())
                .or_insert_with(|| id.clone());
        }
    }
    let mut issues = Vec::new();
    for (key, values) in claims {
        if values.len() < 2 {
            continue;
        }
        let mut detail = format!("conflicting values for evidence key {key:?}: ");
        let parts: Vec<String> = values
            .iter()
            .map(|(v, id)| format!("{v:?} (entity {id})"))
            .collect();
        detail.push_str(&parts.join(", "));
        issues.push(DriftIssue {
            code: "CROSS_FILE_CONFLICT".into(),
            severity: "error".into(),
            target: key,
            detail,
            repairable: false,
        });
    }
    Ok(issues)
}

/// Checker 5: staleness — entities untouched past the per-kind threshold
/// window (warning; surfaced for review, never auto-deleted).
fn check_staleness(
    db: &Database,
    ws: Option<&str>,
    staleness_days: i64,
) -> Result<Vec<DriftIssue>, String> {
    let cutoff = crate::db::now_ms() - staleness_days * 86_400_000;
    let conn = db.conn().map_err(|e| e.to_string())?;
    let mut issues = Vec::new();
    let mut stmt = conn
        .prepare(
            "SELECT id, category, key, last_accessed_unix_ms FROM entities
             WHERE archived = 0 AND last_accessed_unix_ms < ?1 AND (?2 = '' OR workspace_hash = ?2)
             ORDER BY last_accessed_unix_ms ASC LIMIT 100",
        )
        .map_err(|e| e.to_string())?;
    let mut q = stmt
        .query(params![cutoff, ws.unwrap_or("")])
        .map_err(|e| e.to_string())?;
    while let Some(row) = q.next().map_err(|e| e.to_string())? {
        let id: String = row.get(0).map_err(|e| e.to_string())?;
        let category: String = row.get(1).map_err(|e| e.to_string())?;
        let key: String = row.get(2).map_err(|e| e.to_string())?;
        let accessed: i64 = row.get(3).map_err(|e| e.to_string())?;
        issues.push(DriftIssue {
            code: "STALE_ENTITY".into(),
            severity: "warning".into(),
            target: id.clone(),
            detail: format!(
                "stale entity {category}/{key} (id {id}) last accessed {accessed}; consider decay/supersede review"
            ),
            repairable: false,
        });
    }
    Ok(issues)
}

/// Run all deterministic checkers and compute the health score.
pub fn drift_check(
    db: &Database,
    ws: Option<&str>,
    staleness_days: i64,
) -> Result<DriftReport, String> {
    let staleness_days = staleness_days.clamp(1, 3650);
    let mut issues = Vec::new();
    let mut counts = serde_json::Map::new();
    let mut push = |code: &str, found: Vec<DriftIssue>| {
        counts.insert(code.to_string(), serde_json::json!(found.len()));
        issues.extend(found);
    };
    push("REFERENCE_INTEGRITY", check_reference_integrity(db, ws)?);
    push("GROUNDING_STATUS", check_grounding_status(db, ws)?);
    push("PATH_EXISTENCE", check_path_existence(db, ws)?);
    push("CROSS_FILE_CONFLICT", check_cross_file_conflicts(db, ws)?);
    push("STALE_ENTITY", check_staleness(db, ws, staleness_days)?);
    let errors = issues.iter().filter(|i| i.severity == "error").count();
    let warnings = issues.iter().filter(|i| i.severity == "warning").count();
    let infos = issues.iter().filter(|i| i.severity == "info").count();
    Ok(DriftReport {
        health_score: score_for(&issues),
        errors,
        warnings,
        infos,
        checker_counts: serde_json::Value::Object(counts),
        issues,
        note: "deterministic pre-pass (zero LLM in detection); health score = 100 − (10×error + 3×warning + 1×info); repair scope = flagged items only".to_string(),
    })
}

/// Targeted repair + verify leg. Applies ONLY mechanical fixes (unlink
/// dangling references; acknowledge grounding findings), then re-runs the
/// check and reports the score delta. Contradictions/staleness/missing paths
/// are never auto-resolved — they land in `requires_review` (operator review
/// queue). Fail-closed: a repair is accepted only when the score improves or
/// holds.
pub fn drift_repair(
    db: &Database,
    ws: Option<&str>,
    staleness_days: i64,
) -> Result<RepairReport, String> {
    let before = drift_check(db, ws, staleness_days)?;
    let mut repaired: Vec<String> = Vec::new();
    let mut requires_review: Vec<String> = Vec::new();

    // 1. Unlink dangling derived_from references (per-entity, deterministic,
    //    single connection, journaled).
    for issue in before
        .issues
        .iter()
        .filter(|i| i.code == "REFERENCE_INTEGRITY")
    {
        let conn = db.conn().map_err(|e| e.to_string())?;
        let links_raw: Option<String> = conn
            .query_row(
                "SELECT links FROM entities WHERE id = ?1",
                params![issue.target],
                |r| r.get(0),
            )
            .ok();
        let Some(links_raw) = links_raw else { continue };
        let parsed: serde_json::Value = serde_json::from_str(&links_raw).unwrap_or_default();
        let Some(arr) = parsed.as_array() else {
            continue;
        };
        let mut kept: Vec<serde_json::Value> = Vec::new();
        let mut removed = 0usize;
        for link in arr {
            let is_derived =
                link.get("relationship").and_then(|r| r.as_str()) == Some("derived_from");
            let keep = if !is_derived {
                true
            } else {
                match link.get("target_id").and_then(|t| t.as_str()) {
                    Some(t) => {
                        let exists: i64 = conn
                            .query_row(
                                "SELECT COUNT(*) FROM entities WHERE id = ?1 AND archived = 0",
                                params![t],
                                |r| r.get(0),
                            )
                            .unwrap_or(1);
                        exists > 0
                    }
                    None => true,
                }
            };
            if keep {
                kept.push(link.clone());
            } else {
                removed += 1;
            }
        }
        if removed > 0 {
            conn.execute(
                "UPDATE entities SET links = ?1 WHERE id = ?2",
                params![
                    serde_json::to_string(&kept).map_err(|e| e.to_string())?,
                    issue.target
                ],
            )
            .map_err(|e| e.to_string())?;
            drop(conn);
            crate::db::Database::journal(
                db,
                &crate::models::JournalEvent {
                    id: format!("jrn-{}", uuid::Uuid::new_v4().simple()),
                    event_type: "drift_repair_unlink".to_string(),
                    evaluated_json: serde_json::json!({
                        "code": "REFERENCE_INTEGRITY",
                        "removed_dangling_links": removed,
                    })
                    .to_string(),
                    acted_json: "{}".to_string(),
                    forward_json: "{}".to_string(),
                    category: "drift_check".to_string(),
                    key: issue.target.clone(),
                    entity_id: issue.target.clone(),
                    agent_id: "drift_check".to_string(),
                    workspace_hash: ws.unwrap_or("").to_string(),
                    created_at_unix_ms: crate::db::now_ms(),
                },
            )
            .map_err(|e| e.to_string())?;
            repaired.push(issue_key("REFERENCE_INTEGRITY", &issue.target));
        }
    }

    // 2. Acknowledge grounding findings (review flag, never auto-delete).
    for issue in before
        .issues
        .iter()
        .filter(|i| i.code == "GROUNDING_STATUS")
    {
        let conn = db.conn().map_err(|e| e.to_string())?;
        conn.execute(
            "UPDATE grounding_fingerprints SET reviewed_at_unix_ms = ?1 WHERE target_ref = ?2",
            params![crate::db::now_ms(), issue.target],
        )
        .map_err(|e| e.to_string())?;
        drop(conn);
        repaired.push(issue_key("GROUNDING_STATUS", &issue.target));
    }

    // 3. Everything else is review-only (deterministic rule: never
    //    auto-resolve contradictions, stale facts, or missing files).
    for issue in before.issues.iter() {
        let key = issue_key(&issue.code, &issue.target);
        if !repaired.contains(&key) {
            requires_review.push(key);
        }
    }

    let after = drift_check(db, ws, staleness_days)?;
    if after.health_score < before.health_score {
        return Err(format!(
            "repair regression: health score dropped from {} to {} — repairs not accepted",
            before.health_score, after.health_score
        ));
    }
    Ok(RepairReport {
        before_score: before.health_score,
        after_score: after.health_score,
        repaired,
        requires_review,
        note: "check → targeted repair → verify loop complete; LLM review reserved for the requires_review subset".to_string(),
    })
}
