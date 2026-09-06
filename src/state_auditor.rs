//! #1093: state-to-draft audit — implicit stale-dependency repair
//! (STALE / StateAuditor, arXiv:2608.01619).
//!
//! State-table entries are written against assumptions about the entities
//! they reference. When those dependencies change silently (an entity is
//! consolidated away, a cohort is pruned, a cached count drifts), the state
//! entry becomes STALE: it still parses, still serves, but its premise is
//! gone. Serving it quietly is a lie; deleting it destroys evidence.
//!
//! This module implements the implicit repair:
//!
//! 1. **Dependency rules** — deterministic per-family checks over the
//!    state table. A finding names the key and the exact broken
//!    dependency (never a heuristic score).
//! 2. **State-to-draft demotion** — the stale value is preserved verbatim
//!    as a non-authoritative draft under `state_draft.<receipt_id>`, the
//!    live key is rewritten as `status: "stale"` (shape-compatible so the
//!    review lanes still render it, clearly marked), and a
//!    `state_stale_repaired` journal receipt anchors the repair.
//!
//! Rules:
//! - `sleep_proposal.*` → stale when entity_a or entity_b no longer exists.
//! - `shadow_promote_last` → stale when any promoted id no longer exists.
//! - `skill.exp.stats.<id>` → stale when n != count of experience paths.
//! - any state value carrying `snapshot_entity_count` → stale when the
//!   embedded count disagrees with the live entities table.

use serde::Serialize;

/// Draft staging prefix: original stale values are preserved here,
/// verbatim, keyed by the repair receipt.
pub const STATE_DRAFT_PREFIX: &str = "state_draft.";

#[derive(Debug, Clone, Serialize)]
pub struct StaleFinding {
    pub key: String,
    pub reason: String,
}

fn entity_exists(db: &crate::db::Database, id: &str) -> bool {
    let conn = match db.conn() {
        Ok(c) => c,
        Err(_) => return true, // audit must not fail on transient conn issues
    };
    conn.query_row("SELECT COUNT(*) FROM entities WHERE id = ?1", [id], |r| {
        r.get::<_, i64>(0)
    })
    .map(|n| n > 0)
    .unwrap_or(true)
}

/// Evaluate one state entry against its family's dependency rules.
/// Returns the stale reason when the entry's implicit dependencies have
/// drifted.
pub fn evaluate(db: &crate::db::Database, key: &str, value_json: &str) -> Option<StaleFinding> {
    // Rule 1: sleep proposals depend on both referenced entities.
    if key.starts_with(crate::sleep::STATE_PREFIX) {
        let v: serde_json::Value = serde_json::from_str(value_json).ok()?;
        // Already demoted entries are not re-flagged.
        if v.get("status").and_then(|s| s.as_str()) == Some("stale") {
            return None;
        }
        let a = v.get("entity_a")?.as_str()?;
        let b = v.get("entity_b")?.as_str()?;
        let missing: Vec<&str> = [a, b]
            .into_iter()
            .filter(|id| !entity_exists(db, id))
            .collect();
        if !missing.is_empty() {
            return Some(StaleFinding {
                key: key.to_string(),
                reason: format!(
                    "referenced entity(ies) no longer exist: {}",
                    missing.join(", ")
                ),
            });
        }
        return None;
    }
    // Rule 2: shadow promotion record depends on every promoted id.
    if key == "shadow_promote_last" {
        let v: serde_json::Value = serde_json::from_str(value_json).ok()?;
        let ids = v.get("ids")?.as_array()?;
        let missing: Vec<String> = ids
            .iter()
            .filter_map(|i| i.as_str())
            .filter(|id| !entity_exists(db, id))
            .map(|s| s.to_string())
            .collect();
        if !missing.is_empty() {
            return Some(StaleFinding {
                key: key.to_string(),
                reason: format!("promoted id(s) no longer exist: {}", missing.join(", ")),
            });
        }
        return None;
    }
    // Rule 3: experience-trie stats must match the logged paths.
    if let Some(skill_id) = key.strip_prefix("skill.exp.stats.") {
        let v: serde_json::Value = serde_json::from_str(value_json).ok()?;
        let claimed = v.get("n").and_then(|n| n.as_i64()).unwrap_or(0);
        let conn = db.conn().ok()?;
        let actual: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM state WHERE key LIKE ?1",
                [format!("skill.exp.{skill_id}.%")],
                |r| r.get(0),
            )
            .unwrap_or(0);
        if claimed != actual {
            return Some(StaleFinding {
                key: key.to_string(),
                reason: format!("stats.n={claimed} but {actual} experience paths are logged"),
            });
        }
        return None;
    }
    // Rule 4: generic embedded snapshot counts must match live counts.
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(value_json) {
        if let Some(snap) = v.get("snapshot_entity_count").and_then(|n| n.as_i64()) {
            let conn = db.conn().ok()?;
            let live: i64 = conn
                .query_row(
                    "SELECT COUNT(*) FROM entities WHERE archived = 0",
                    [],
                    |r| r.get(0),
                )
                .unwrap_or(snap); // on transient failure, do not flag
            if snap != live {
                return Some(StaleFinding {
                    key: key.to_string(),
                    reason: format!("snapshot_entity_count={snap} but live active count is {live}"),
                });
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entity_fixture(id: &str) -> crate::models::Entity {
        crate::models::Entity {
            id: id.to_string(),
            category: "facts".to_string(),
            key: id.to_string(),
            body_json: format!("{{\"content\": \"{id} body\"}}"),
            status: "active".to_string(),
            entity_type: "fact".to_string(),
            tags: vec![],
            decay_score: 1.0,
            retrieval_count: 0,
            layer: "working".to_string(),
            topic_path: String::new(),
            archived: false,
            archive_reason: String::new(),
            links: vec![],
            verified: false,
            source: "agent".to_string(),
            always_on: false,
            certainty: 0.5,
            workspace_hash: String::new(),
            agent_id: String::new(),
            visibility: "workspace".to_string(),
            created_at_unix_ms: crate::db::now_ms(),
            last_accessed_unix_ms: crate::db::now_ms(),
            follow_count: 0,
            miss_count: 0,
            follow_rate: 0.0,
            efficacy_status: "unverified".to_string(),
            epistemic_state: "candidate".to_string(),
            hints: vec![],
            memory_type: "semantic".to_string(),
            embedding: None,
            _parsed_body: None,
        }
    }

    fn seed(db: &crate::db::Database, e: &crate::models::Entity) {
        db.remember_with_write_options(
            e,
            true,
            None,
            None,
            false,
            crate::interference::WriteGateOptions {
                mode_override: Some(crate::interference::InterferenceMode::Off),
                ..Default::default()
            },
        )
        .unwrap();
    }

    fn set_state(db: &crate::db::Database, key: &str, value: serde_json::Value) {
        db.state_set(&crate::models::StateEntry {
            key: key.to_string(),
            value_json: value.to_string(),
            expires_at_unix_ms: None,
            created_at_unix_ms: crate::db::now_ms(),
        })
        .unwrap();
    }

    #[test]
    fn sleep_proposal_with_missing_entity_is_stale() {
        let db = crate::db::TestDatabase::new("stateauditor-sleep");
        seed(&db, &entity_fixture("sa-alive"));
        let alive = serde_json::json!({
            "kind": "merge", "category": "facts", "entity_a": "sa-alive",
            "entity_b": "sa-alive", "similarity": 0.9, "reason": "dup",
            "workspace_hash": "", "status": "pending"
        });
        let dead = serde_json::json!({
            "kind": "merge", "category": "facts", "entity_a": "sa-alive",
            "entity_b": "sa-gone", "similarity": 0.9, "reason": "dup",
            "workspace_hash": "", "status": "pending"
        });
        assert!(evaluate(&db, "sleep_proposal.ok", &alive.to_string()).is_none());
        let f = evaluate(&db, "sleep_proposal.dead", &dead.to_string()).expect("must flag");
        assert!(f.reason.contains("sa-gone"));
        set_state(&db, "sleep_proposal.ok", alive);
        set_state(&db, "sleep_proposal.dead", dead);
    }

    #[test]
    fn experience_stats_rule_counts_paths() {
        let db = crate::db::TestDatabase::new("stateauditor-stats");
        set_state(&db, "skill.exp.s-a.fp1", serde_json::json!({"served": 1}));
        set_state(&db, "skill.exp.s-a.fp2", serde_json::json!({"served": 0}));
        set_state(
            &db,
            "skill.exp.stats.s-a",
            serde_json::json!({"n": 5, "ok": 1}),
        );
        let f = evaluate(&db, "skill.exp.stats.s-a", "{\"n\": 5, \"ok\": 1}").expect("must flag");
        assert!(f.reason.contains("5"));
        let ok = evaluate(&db, "skill.exp.stats.s-a", "{\"n\": 2, \"ok\": 1}");
        assert!(ok.is_none());
    }

    #[test]
    fn audit_demotes_stale_to_draft_with_receipts() {
        let db = crate::db::TestDatabase::new("stateauditor-e2e");
        seed(&db, &entity_fixture("sa-alive"));
        set_state(
            &db,
            "sleep_proposal.ok",
            serde_json::json!({
                "kind": "merge", "category": "facts", "entity_a": "sa-alive",
                "entity_b": "sa-alive", "similarity": 0.9, "reason": "dup",
                "workspace_hash": "", "status": "pending"
            }),
        );
        set_state(
            &db,
            "sleep_proposal.dead",
            serde_json::json!({
                "kind": "merge", "category": "facts", "entity_a": "sa-alive",
                "entity_b": "sa-gone", "similarity": 0.9, "reason": "dup",
                "workspace_hash": "", "status": "pending"
            }),
        );
        set_state(
            &db,
            "shadow_promote_last",
            serde_json::json!({"ids": ["sa-alive", "sa-gone"], "from": "s", "to": "m"}),
        );
        set_state(
            &db,
            "cache.report",
            serde_json::json!({"snapshot_entity_count": 99}),
        );

        // Dry-run: 3 findings, zero writes.
        let dry = db.state_audit(true).unwrap();
        assert_eq!(dry["dry_run"], true);
        assert_eq!(dry["stale_count"].as_i64().unwrap(), 3);
        let live = db.state_get("sleep_proposal.dead").unwrap().unwrap();
        assert!(live.value_json.contains("\"pending\""));
        assert_eq!(db.state_list("state_draft.").unwrap().len(), 0);

        // Real run: demotion + drafts + receipts.
        let rep = db.state_audit(false).unwrap();
        assert_eq!(rep["stale_count"].as_i64().unwrap(), 3);
        assert_eq!(rep["repaired"].as_array().unwrap().len(), 3);
        // Drafts preserve the originals.
        let drafts = db.state_list("state_draft.").unwrap();
        assert_eq!(drafts.len(), 3);
        // Sleep proposal demoted shape-compatibly: still a SleepProposal.
        let demoted = db.state_get("sleep_proposal.dead").unwrap().unwrap();
        assert!(demoted.value_json.contains("\"status\":\"stale\""));
        assert!(
            demoted.value_json.contains("sa-gone"),
            "original refs preserved"
        );
        // Journal receipts anchored.
        let conn = db.conn().unwrap();
        let receipts: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM journal WHERE event_type = 'state_stale_repaired'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(receipts, 3);
        // Second pass: already-demoted entries are not re-flagged.
        let rep2 = db.state_audit(true).unwrap();
        assert_eq!(
            rep2["stale_count"].as_i64().unwrap(),
            0,
            "repair is idempotent — demoted entries are not re-flagged"
        );
    }

    #[test]
    fn snapshot_count_rule_flags_drift() {
        let db = crate::db::TestDatabase::new("stateauditor-count");
        seed(&db, &entity_fixture("sa-count-a"));
        set_state(
            &db,
            "cache.report",
            serde_json::json!({"snapshot_entity_count": 7}),
        );
        let f = evaluate(&db, "cache.report", "{\"snapshot_entity_count\": 7}").expect("must flag");
        assert!(f.reason.contains("7"));
        assert!(evaluate(&db, "cache.report", "{\"snapshot_entity_count\": 1}").is_none());
    }
}
