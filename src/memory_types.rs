//! #1000: typed memory classes with per-type policies (CogniCore borrow).
//!
//! CogniCore (cognicore-dev/cognicore-my-openenv, memory/base.py) defines 8
//! `MemoryType`s with a `TypePolicy` per type (storage strategy, retrieval
//! weight, scoring, decay). The vault has categories (arbitrary strings) but
//! no type-level behavior — every category decays and ranks identically.
//! This module borrows the TAXONOMY (not CogniCore's thin enforcement):
//!
//! - `MemoryType`: SEMANTIC | EPISODIC | PROCEDURAL | PREFERENCE |
//!   CONSTRAINT | FAILURE | REFLECTION | KNOWLEDGE.
//! - `TypePolicy`: per-type decay_multiplier (scales the #941 category
//!   half-life at decay tick — composes, never overrides) and
//!   retrieval_weight (multiplies the final fused recall score).
//!
//! Divergence from the borrow, deliberate: CogniCore's `from_string` silently
//! falls back to SEMANTIC on unknown input. The vault fails closed — an
//! unknown type on a write is a hard error (see #998 fail-loud gating), and
//! legacy rows (memory_type = '') get the SEMANTIC POLICY but keep their
//! stored value unchanged (a legacy row is never silently rewritten).

use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum MemoryType {
    Semantic,
    Episodic,
    Procedural,
    Preference,
    Constraint,
    Failure,
    Reflection,
    Knowledge,
}

impl MemoryType {
    pub const ALL: [MemoryType; 8] = [
        MemoryType::Semantic,
        MemoryType::Episodic,
        MemoryType::Procedural,
        MemoryType::Preference,
        MemoryType::Constraint,
        MemoryType::Failure,
        MemoryType::Reflection,
        MemoryType::Knowledge,
    ];

    pub fn as_str(&self) -> &'static str {
        match self {
            MemoryType::Semantic => "semantic",
            MemoryType::Episodic => "episodic",
            MemoryType::Procedural => "procedural",
            MemoryType::Preference => "preference",
            MemoryType::Constraint => "constraint",
            MemoryType::Failure => "failure",
            MemoryType::Reflection => "reflection",
            MemoryType::Knowledge => "knowledge",
        }
    }

    /// Fail-closed parse: unknown/empty input is a hard error for WRITES.
    /// Callers that need the legacy-row policy use `policy_for("")` instead.
    pub fn parse(s: &str) -> Result<MemoryType, String> {
        match s.trim().to_lowercase().as_str() {
            "semantic" => Ok(MemoryType::Semantic),
            "episodic" => Ok(MemoryType::Episodic),
            "procedural" => Ok(MemoryType::Procedural),
            "preference" => Ok(MemoryType::Preference),
            "constraint" => Ok(MemoryType::Constraint),
            "failure" => Ok(MemoryType::Failure),
            "reflection" => Ok(MemoryType::Reflection),
            "knowledge" => Ok(MemoryType::Knowledge),
            other => Err(format!(
                "unknown memory type '{other}' — expected one of: semantic, episodic, \
                 procedural, preference, constraint, failure, reflection, knowledge"
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct TypePolicy {
    pub memory_type: MemoryType,
    /// Scales the #941 per-category half-life at decay tick. >1 = lives
    /// longer than the category default; <1 = rots faster. "never" (the
    /// huge-value encoding) survives any finite multiplier by design.
    pub decay_multiplier: f64,
    /// Multiplies the final fused recall score (the #8504 scoring loop).
    /// >1 = the type outranks pure similarity; <1 = damped.
    pub retrieval_weight: f64,
    pub rationale: &'static str,
}

/// The policy table. Rationales follow CogniCore's type docstrings mapped
/// onto governed memory: PROCEDURAL (how-to), CONSTRAINT (rules/boundaries),
/// FAILURE (anti-patterns), REFLECTION (meta-observations) are the four the
/// vault's governance regime cares most about — they outlive events and
/// outrank raw similarity.
pub fn policies() -> [TypePolicy; 8] {
    [
        TypePolicy {
            memory_type: MemoryType::Semantic,
            decay_multiplier: 1.0,
            retrieval_weight: 1.0,
            rationale: "facts and definitions — the neutral baseline; legacy rows resolve here",
        },
        TypePolicy {
            memory_type: MemoryType::Episodic,
            decay_multiplier: 0.5,
            retrieval_weight: 0.9,
            rationale: "events and conversations — rot faster than the category default; a transient event must not crowd out durable knowledge",
        },
        TypePolicy {
            memory_type: MemoryType::Procedural,
            decay_multiplier: 2.0,
            retrieval_weight: 1.2,
            rationale: "how-to and working procedures — proven recipes outlive and outrank",
        },
        TypePolicy {
            memory_type: MemoryType::Preference,
            decay_multiplier: 4.0,
            retrieval_weight: 1.3,
            rationale: "user preferences and style — near-durable, and a preference should beat an episode on similarity alone",
        },
        TypePolicy {
            memory_type: MemoryType::Constraint,
            decay_multiplier: 4.0,
            retrieval_weight: 1.4,
            rationale: "rules, restrictions, boundaries — the most durable and most rank-worthy class in governed memory",
        },
        TypePolicy {
            memory_type: MemoryType::Failure,
            decay_multiplier: 1.5,
            retrieval_weight: 1.3,
            rationale: "mistakes and anti-patterns — anti-patterns must keep resurfacing until the lesson is learned",
        },
        TypePolicy {
            memory_type: MemoryType::Reflection,
            decay_multiplier: 1.2,
            retrieval_weight: 1.1,
            rationale: "meta-observations and insights — modest boost; insights earn rank by usefulness",
        },
        TypePolicy {
            memory_type: MemoryType::Knowledge,
            decay_multiplier: 1.5,
            retrieval_weight: 1.0,
            rationale: "structured domain knowledge — durable, neutral rank",
        },
    ]
}

/// Policy for a stored `memory_type` value. Empty string = legacy row → the
/// SEMANTIC policy (baseline behavior, byte-compatible with pre-#1000).
pub fn policy_for(stored: &str) -> TypePolicy {
    if stored.is_empty() {
        return policies()[0];
    }
    MemoryType::parse(stored)
        .ok()
        .and_then(|t| policies().iter().copied().find(|p| p.memory_type == t))
        .unwrap_or(policies()[0])
}

pub fn retrieval_weight(stored: &str) -> f64 {
    policy_for(stored).retrieval_weight
}

pub fn decay_multiplier(stored: &str) -> f64 {
    policy_for(stored).decay_multiplier
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::TestDatabase;
    use crate::models::Entity;

    fn test_entity(category: &str, key: &str, body: &str, memory_type: &str) -> Entity {
        let now = crate::db::now_ms();
        Entity {
            id: uuid::Uuid::new_v4().to_string(),
            category: category.to_string(),
            key: key.to_string(),
            body_json: body.to_string(),
            status: "active".to_string(),
            entity_type: "fact".to_string(),
            tags: vec![],
            decay_score: 0.5,
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
            created_at_unix_ms: now,
            last_accessed_unix_ms: now,
            follow_count: 0,
            miss_count: 0,
            follow_rate: 0.0,
            efficacy_status: "unverified".to_string(),
            epistemic_state: crate::models::default_epistemic_state(),
            hints: vec![],
            memory_type: memory_type.to_string(),
            embedding: None,
            _parsed_body: None,
        }
    }

    #[test]
    fn parse_accepts_all_eight_and_rejects_unknown() {
        for t in MemoryType::ALL {
            assert_eq!(MemoryType::parse(t.as_str()).unwrap(), t);
        }
        assert!(MemoryType::parse("").is_err());
        assert!(MemoryType::parse("garbage").is_err());
        assert!(MemoryType::parse("PROCEDURAL").is_ok());
    }

    #[test]
    fn legacy_rows_resolve_to_semantic_policy() {
        let p = policy_for("");
        assert_eq!(p.memory_type, MemoryType::Semantic);
        assert_eq!(p.decay_multiplier, 1.0);
        assert_eq!(p.retrieval_weight, 1.0);
    }

    #[test]
    fn constraint_is_most_durable_and_most_ranked() {
        let c = policy_for("constraint");
        assert_eq!(c.decay_multiplier, 4.0);
        assert_eq!(c.retrieval_weight, 1.4);
        assert!(c.retrieval_weight > policy_for("episodic").retrieval_weight);
        assert!(c.decay_multiplier > policy_for("episodic").decay_multiplier);
    }

    #[test]
    fn episodic_rots_faster_than_baseline() {
        assert_eq!(policy_for("episodic").decay_multiplier, 0.5);
    }

    #[test]
    fn unknown_stored_value_falls_back_to_baseline_policy() {
        // Stored values are validated at write time, but a hand-edited store
        // must never panic the policy layer.
        let p = policy_for("not-a-type");
        assert_eq!(p.memory_type, MemoryType::Semantic);
    }

    #[test]
    fn policy_table_is_complete_and_serializable() {
        let ps = policies();
        assert_eq!(ps.len(), 8);
        let json = serde_json::to_string(&ps).unwrap();
        assert!(json.contains("constraint"));
        assert!(json.contains("retrieval_weight"));
    }

    // ── integration: write / read / decay / filter ──────────────────────

    #[test]
    fn typed_write_persists_and_recalls_memory_type() {
        let db = TestDatabase::new("memtype-write");
        db.remember(&test_entity(
            "rules",
            "k1",
            "{\"text\":\"never commit secrets to git\"}",
            "constraint",
        ))
        .unwrap();
        let mut p = crate::models::RecallParams::default();
        p.query = "commit secrets git".to_string();
        let hits = db.recall(&p).unwrap();
        assert!(!hits.is_empty());
        assert_eq!(hits[0].memory_type, "constraint");
    }

    #[test]
    fn legacy_write_keeps_empty_memory_type() {
        let db = TestDatabase::new("memtype-legacy");
        db.remember(&test_entity(
            "facts",
            "k1",
            "{\"text\":\"plain legacy fact\"}",
            "",
        ))
        .unwrap();
        let mut p = crate::models::RecallParams::default();
        p.query = "plain legacy fact".to_string();
        let hits = db.recall(&p).unwrap();
        assert_eq!(hits[0].memory_type, "");
        assert_eq!(
            policy_for(&hits[0].memory_type).memory_type,
            MemoryType::Semantic
        );
    }

    #[test]
    fn unknown_type_write_fails_closed() {
        let db = TestDatabase::new("memtype-reject");
        let args = serde_json::json!({
            "category": "rules",
            "key": "k1",
            "body_json": "{\"text\":\"x\"}",
            "memory_type": "bogus"
        });
        let err = crate::tools::handle_remember(&db, args).unwrap_err();
        assert!(err.contains("unknown memory type"), "err: {err}");
    }

    #[test]
    fn constraint_decays_slower_than_episodic() {
        let db = TestDatabase::new("memtype-decay");
        let constraint = test_entity(
            "lessons",
            "c1",
            "{\"text\":\"always lock the db\"}",
            "constraint",
        );
        let episodic = test_entity(
            "lessons",
            "e1",
            "{\"text\":\"tuesday standup was noisy\"}",
            "episodic",
        );
        let cid = constraint.id.clone();
        let eid = episodic.id.clone();
        db.remember(&constraint).unwrap();
        db.remember(&episodic).unwrap();
        // Same category, same age: backdate both to 10 days ago.
        let old = crate::db::now_ms() - 10 * 86_400_000;
        let conn = db.conn().unwrap();
        conn.execute(
            "UPDATE entities SET last_accessed_unix_ms = ?1 WHERE id IN (?2, ?3)",
            rusqlite::params![old, cid, eid],
        )
        .unwrap();
        db.decay_tick().unwrap();
        let cscore: f64 = conn
            .query_row(
                "SELECT decay_score FROM entities WHERE id=?1",
                rusqlite::params![cid],
                |r| r.get(0),
            )
            .unwrap();
        let escore: f64 = conn
            .query_row(
                "SELECT decay_score FROM entities WHERE id=?1",
                rusqlite::params![eid],
                |r| r.get(0),
            )
            .unwrap();
        // decay_score DECREASES toward forgetting (0 = dead). The constraint
        // (4x half-life) must retain more score than the episode (0.5x).
        assert!(
            cscore > escore,
            "constraint {cscore} should outlive episodic {escore}"
        );
    }

    #[test]
    fn type_filter_narrows_fused_recall_and_validates() {
        let db = TestDatabase::new("memtype-filter");
        db.remember(&test_entity(
            "rules",
            "c1",
            "{\"text\":\"never store tokens\"}",
            "constraint",
        ))
        .unwrap();
        db.remember(&test_entity(
            "sessions",
            "e1",
            "{\"text\":\"monday sync about tokens\"}",
            "episodic",
        ))
        .unwrap();
        db.remember(&test_entity(
            "facts",
            "l1",
            "{\"text\":\"legacy tokens note\"}",
            "",
        ))
        .unwrap();

        let mut p = crate::models::RecallParams::default();
        p.query = "tokens".to_string();
        p.type_filter = Some("constraint".to_string());
        let hits = db.recall(&p).unwrap();
        assert!(!hits.is_empty());
        assert!(hits.iter().all(|h| h.memory_type == "constraint"));

        p.type_filter = Some("semantic".to_string());
        let hits = db.recall(&p).unwrap();
        // Legacy rows ('' ) satisfy the semantic filter; the episodic row
        // must not.
        assert!(hits.iter().any(|h| h.memory_type.is_empty()));
        assert!(hits.iter().all(|h| h.memory_type != "episodic"));

        p.type_filter = Some("bogus".to_string());
        assert!(db.recall(&p).is_err());
    }
}
