//! #1001: utility-driven lifecycle promotion (CogniCore borrow).
//!
//! CogniCore models `MemoryState` (CANDIDATE/ACTIVE/VERIFIED/ARCHIVED/
//! DELETED) plus a `utility_score` updated on store/search. The borrowable
//! part is the SIGNAL DESIGN — retrieval-count → promotion, outcome
//! feedback → verification — which pairs with the vault's existing decay
//! machinery. The ladder maps onto the vault's #880 epistemic states:
//!
//! - CANDIDATE  = epistemic_state 'candidate' (write default)
//! - ACTIVE     = status 'active' (write default; not a stored rung)
//! - VERIFIED   = epistemic_state 'verified'
//! - ARCHIVED   = archived = 1 (existing decay machinery — unchanged)
//!
//! Divergences from the borrow, deliberate:
//! 1. Auto-promotion moves ONE rung (candidate → verified) and NEVER
//!    demotes; corroboration still requires independent-source evidence
//!    (#880) and is never auto-granted.
//! 2. Utility accrues ONLY on side-effect-bearing interactions — reinforced
//!    recalls, citations (derived_from), outcome feedback. The #247
//!    frozen-recall determinism contract is preserved: a default recall is
//!    a pure read and never mutates utility.
//! 3. Every transition is journaled (event_type 'auto_promotion') and the
//!    journal is the evidence surface; there is no silent mutation.

/// Utility delta for one reinforced recall hit.
pub const RETRIEVAL_HIT_DELTA: f64 = 1.0;
/// Utility delta for one citation (derived_from / mark_useful).
pub const CITATION_DELTA: f64 = 5.0;
/// Utility cap — saturating, so a heavily-reused memory converges instead
/// of growing without bound.
pub const UTILITY_CAP: f64 = 100.0;
/// Utility a candidate must reach before auto-promotion is eligible.
pub const PROMOTE_UTILITY_THRESHOLD: f64 = 10.0;
/// Independent outcome signals (citations) required alongside utility.
pub const PROMOTE_OUTCOMES_THRESHOLD: i64 = 1;

/// Pure, deterministic transition function: (state, utility, outcomes) →
/// the next epistemic state, or None when nothing moves.
///
/// Rules:
/// - candidate → verified  when utility >= threshold AND outcomes >= threshold
/// - verified/corroborated/rejected/defensively_recalled → None (never
///   auto-demote; corroboration requires independent sources)
/// - unknown state → None (never invent a state)
pub fn next_epistemic_state(state: &str, utility: f64, outcomes: i64) -> Option<&'static str> {
    match state {
        "candidate"
            if utility >= PROMOTE_UTILITY_THRESHOLD && outcomes >= PROMOTE_OUTCOMES_THRESHOLD =>
        {
            Some("verified")
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn candidate_promotes_to_verified_at_both_thresholds() {
        assert_eq!(next_epistemic_state("candidate", 10.0, 1), Some("verified"));
        assert_eq!(next_epistemic_state("candidate", 25.0, 3), Some("verified"));
    }

    #[test]
    fn candidate_does_not_promote_below_either_threshold() {
        assert_eq!(next_epistemic_state("candidate", 9.9, 5), None);
        assert_eq!(next_epistemic_state("candidate", 50.0, 0), None);
        assert_eq!(next_epistemic_state("candidate", 1.0, 0), None);
    }

    #[test]
    fn never_auto_demotes_or_auto_corroborates() {
        for state in [
            "verified",
            "corroborated",
            "rejected",
            "defensively_recalled",
        ] {
            assert_eq!(next_epistemic_state(state, 0.0, 0), None, "{state}");
            assert_eq!(next_epistemic_state(state, 100.0, 10), None, "{state}");
        }
    }

    #[test]
    fn unknown_state_never_moves() {
        assert_eq!(next_epistemic_state("garbage", 100.0, 10), None);
        assert_eq!(next_epistemic_state("", 100.0, 10), None);
    }

    #[test]
    fn function_is_pure_and_total() {
        // Same inputs, same output; every legal input produces Option<&str>.
        let a = next_epistemic_state("candidate", 10.0, 1);
        let b = next_epistemic_state("candidate", 10.0, 1);
        assert_eq!(a, b);
    }

    // ── integration: accrual hooks + journaled transition ───────────────

    #[test]
    fn reinforced_recall_accrues_utility_without_promoting_early() {
        let db = crate::db::TestDatabase::new("util-recall");
        let now = crate::db::now_ms();
        let entity = crate::models::Entity {
            id: "u-recall-1".to_string(),
            category: "lessons".to_string(),
            key: "k1".to_string(),
            body_json: "{\"text\":\"x\"}".to_string(),
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
            epistemic_state: "candidate".to_string(),
            hints: vec![],
            embedding: None,
            memory_type: String::new(),
            _parsed_body: None,
        };
        db.remember(&entity).unwrap();
        db.apply_recall_side_effects(&["u-recall-1".to_string()])
            .unwrap();
        let conn = db.conn().unwrap();
        let utility: f64 = conn
            .query_row(
                "SELECT utility_score FROM entities WHERE id='u-recall-1'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(utility, RETRIEVAL_HIT_DELTA);
        let state: String = conn
            .query_row(
                "SELECT epistemic_state FROM entities WHERE id='u-recall-1'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(state, "candidate", "below thresholds, no promotion");
    }

    #[test]
    fn citations_promote_candidate_to_verified_with_journal_evidence() {
        let db = crate::db::TestDatabase::new("util-promote");
        let now = crate::db::now_ms();
        let mut entity = crate::models::Entity {
            id: "u-promote-1".to_string(),
            category: "lessons".to_string(),
            key: "k2".to_string(),
            body_json: "{\"text\":\"y\"}".to_string(),
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
            epistemic_state: "candidate".to_string(),
            hints: vec![],
            embedding: None,
            memory_type: String::new(),
            _parsed_body: None,
        };
        entity = {
            let mut e = entity;
            e.category = "lessons".to_string();
            e
        };
        db.remember(&entity).unwrap();
        // Two citations: utility 10.0 + outcomes 2 → threshold crossed.
        assert!(db.mark_useful_by_id("u-promote-1").unwrap());
        assert!(db.mark_useful_by_id("u-promote-1").unwrap());
        let conn = db.conn().unwrap();
        let (state, utility, outcomes): (String, f64, i64) = conn
            .query_row(
                "SELECT epistemic_state, utility_score, usefulness_count \
                 FROM entities WHERE id='u-promote-1'",
                [],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
            .unwrap();
        assert_eq!(state, "verified", "auto-promotion at thresholds");
        assert_eq!(utility, CITATION_DELTA * 2.0);
        assert_eq!(outcomes, 2);
        let journaled: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM journal WHERE event_type='auto_promotion' AND entity_id='u-promote-1'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(journaled, 1, "transition must be journaled exactly once");
    }
}
