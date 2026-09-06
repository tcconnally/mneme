//! #1088: semantic segment-level consolidation (LycheeMemory V2,
//! arXiv:2608.12990).
//!
//! Eager consolidation invokes the encoding pass after every interaction, so
//! memory-construction cost grows with conversation length. Segment-level
//! consolidation batches exchanges into SEMANTIC segments (boundary
//! detection, not fixed windows) and runs one encoding pass per finalized
//! segment — construction frequency is segment-count-bound, not
//! turn-count-bound. Segments feed the existing #1002/#1026 governance
//! pipeline unchanged: granularity changes the schedule, not the authority
//! rules.
//!
//! Deterministic boundary detection: an incoming entity starts a new segment
//! when (a) the inter-arrival gap since the previous entity exceeds the
//! configured gap threshold, or (b) the trigram similarity between adjacent
//! entities drops below the semantic floor (a topic discontinuity). Fixed
//! windows are deliberately NOT used: event-level and temporal evidence
//! stays coherent inside one segment.

use std::collections::HashSet;

/// One candidate entity in arrival order for segmentation.
#[derive(Debug, Clone)]
pub struct SegmentInput {
    pub entity_id: String,
    pub created_at_unix_ms: i64,
    pub body_text: String,
}

/// A finalized segment: the member indices into the input slice.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SegmentGroup {
    pub members: Vec<String>,
    pub start_ms: i64,
    pub end_ms: i64,
    /// Why the segment AFTER this one started ('' for the last segment).
    pub next_boundary_reason: String,
}

/// Deterministic character trigrams (lowercased, alphanumeric-only) —
/// the same family of measure the dedup/consolidate machinery uses.
fn trigrams(text: &str) -> HashSet<[char; 3]> {
    let chars: Vec<char> = text
        .to_lowercase()
        .chars()
        .filter(|c| c.is_alphanumeric())
        .collect();
    chars.windows(3).map(|w| [w[0], w[1], w[2]]).collect()
}

/// Trigram Jaccard similarity in [0.0, 1.0]. Empty sets score 0.0 unless
/// both texts are equal (which scores 1.0).
pub fn trigram_similarity(a: &str, b: &str) -> f64 {
    if a == b {
        return 1.0;
    }
    let ta = trigrams(a);
    let tb = trigrams(b);
    if ta.is_empty() || tb.is_empty() {
        return 0.0;
    }
    let inter = ta.intersection(&tb).count();
    let union = ta.len() + tb.len() - inter;
    if union == 0 {
        0.0
    } else {
        inter as f64 / union as f64
    }
}

/// Segment `items` (expected in created_at ascending order) into semantic
/// segments. A boundary fires between two adjacent entities when their
/// inter-arrival gap exceeds `gap_ms` or their trigram similarity falls
/// below `sim_floor`.
pub fn detect_segments(items: &[SegmentInput], gap_ms: i64, sim_floor: f64) -> Vec<SegmentGroup> {
    if items.is_empty() {
        return Vec::new();
    }
    let mut groups: Vec<Vec<usize>> = vec![vec![0]];
    let mut reasons: Vec<&str> = Vec::new();
    for i in 1..items.len() {
        let gap = items[i].created_at_unix_ms - items[i - 1].created_at_unix_ms;
        let sim = trigram_similarity(&items[i - 1].body_text, &items[i].body_text);
        if gap > gap_ms {
            groups.push(vec![i]);
            reasons.push("time-gap");
        } else if sim < sim_floor {
            groups.push(vec![i]);
            reasons.push("semantic-discontinuity");
        } else {
            groups.last_mut().expect("at least one group").push(i);
        }
    }
    groups
        .into_iter()
        .enumerate()
        .map(|(g, idxs)| SegmentGroup {
            members: idxs.iter().map(|&i| items[i].entity_id.clone()).collect(),
            start_ms: items[*idxs.first().unwrap()].created_at_unix_ms,
            end_ms: items[*idxs.last().unwrap()].created_at_unix_ms,
            next_boundary_reason: reasons.get(g).copied().unwrap_or("").to_string(),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item(id: &str, t: i64, body: &str) -> SegmentInput {
        SegmentInput {
            entity_id: id.to_string(),
            created_at_unix_ms: t,
            body_text: body.to_string(),
        }
    }

    #[test]
    fn time_gap_splits_segments() {
        let items = vec![
            item("a", 1000, "alpha theme one"),
            item("b", 2000, "alpha theme two"),
            item("c", 900_000, "alpha theme three"), // 898s gap
        ];
        let groups = detect_segments(&items, 60_000, 0.2);
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].members, vec!["a", "b"]);
        assert_eq!(groups[0].next_boundary_reason, "time-gap");
        assert_eq!(groups[1].members, vec!["c"]);
    }

    #[test]
    fn semantic_discontinuity_splits_segments() {
        let items = vec![
            item("a", 1000, "alpha theme one"),
            item("b", 2000, "alpha theme two"),
            item("c", 3000, "completely unrelated zebra quantum"),
            item("d", 4000, "completely unrelated zebra quantum two"),
        ];
        let groups = detect_segments(&items, 60_000, 0.2);
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].members, vec!["a", "b"]);
        assert_eq!(groups[0].next_boundary_reason, "semantic-discontinuity");
        assert_eq!(groups[1].members, vec!["c", "d"]);
    }

    #[test]
    fn uniform_stream_stays_one_segment() {
        let items = vec![
            item("a", 1000, "alpha theme one"),
            item("b", 2000, "alpha theme two"),
            item("c", 3000, "alpha theme three"),
        ];
        let groups = detect_segments(&items, 60_000, 0.2);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].members.len(), 3);
    }

    #[test]
    fn trigram_similarity_bounds() {
        assert!((trigram_similarity("alpha", "alpha") - 1.0).abs() < 1e-9);
        assert_eq!(trigram_similarity("alpha", ""), 0.0);
        assert!(trigram_similarity("alpha beta gamma", "alpha beta delta") > 0.3);
        assert!(trigram_similarity("alpha beta gamma", "xylophone quantum") < 0.2);
    }

    #[test]
    fn empty_input_no_segments() {
        assert!(detect_segments(&[], 1000, 0.2).is_empty());
    }

    // ── DB integration ──

    fn seg_entity(id: &str, body: &str, created: i64, ws: &str) -> crate::models::Entity {
        crate::models::Entity {
            id: id.to_string(),
            category: "turns".to_string(),
            key: id.to_string(),
            body_json: body.to_string(),
            status: "active".to_string(),
            entity_type: "insight".to_string(),
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
            workspace_hash: ws.to_string(),
            agent_id: String::new(),
            visibility: "workspace".to_string(),
            created_at_unix_ms: created,
            last_accessed_unix_ms: created,
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

    /// Fixture write with the interference gate OFF (the established
    /// db.rs test pattern): synthetic fixtures are trigram-similar by
    /// construction and would otherwise be quarantined instead of written.
    fn write_fixture(
        db: &crate::db::Database,
        e: &crate::models::Entity,
    ) -> Result<(String, String), Box<dyn std::error::Error>> {
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
    }

    fn state_count(db: &crate::db::Database) -> i64 {
        db.conn()
            .unwrap()
            .query_row("SELECT COUNT(*) FROM state", [], |r| r.get(0))
            .unwrap()
    }

    #[test]
    fn segment_consolidate_end_to_end() {
        let db = crate::db::TestDatabase::new("segment-e2e");
        let ws = "ws-seg-e2e";
        // Three similar alpha-theme entities, then a 6h+ gap, then two
        // similar beta-theme entities → two semantic segments.
        write_fixture(&db, &seg_entity("a1", "alpha theme one", 1_000, ws)).unwrap();
        write_fixture(&db, &seg_entity("a2", "alpha theme two", 2_000, ws)).unwrap();
        write_fixture(&db, &seg_entity("a3", "alpha theme three", 3_000, ws)).unwrap();
        write_fixture(&db, &seg_entity("b1", "beta theme one", 1_000_000, ws)).unwrap();
        write_fixture(&db, &seg_entity("b2", "beta theme two", 1_001_000, ws)).unwrap();

        // Dry run: plans reported, nothing written.
        let report = db
            .segment_consolidate("turns", ws, 60_000, 0.25, 100, true)
            .unwrap();
        assert_eq!(report["dry_run"], true);
        assert_eq!(report["scanned"], 5);
        assert_eq!(report["segments"], 2);
        assert_eq!(report["consolidated"], 2);
        assert_eq!(report["skipped_singletons"], 0);
        assert_eq!(state_count(&db), 0, "dry run must not write plans");

        // Execute: one bounded consolidate pass per segment.
        let report = db
            .segment_consolidate("turns", ws, 60_000, 0.25, 100, false)
            .unwrap();
        assert_eq!(report["dry_run"], false);
        assert_eq!(report["consolidated"], 2);
        let consolidations = report["consolidations"].as_array().unwrap();
        assert_eq!(consolidations.len(), 2);
        for c in consolidations {
            assert!(c["error"].is_null(), "unexpected segment error: {c}");
            let rep = &c["report"];
            assert!(rep["observations_created"].as_i64().unwrap() >= 1);
            assert!(rep["source_entities_merged"].as_i64().unwrap() >= 2);
        }
        // Lightweight structured index: one segment_plan state record per
        // segment.
        assert_eq!(state_count(&db), 2);
        let conn = db.conn().unwrap();
        let plans: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM state WHERE key LIKE 'segment_plan.%'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(plans, 2);
    }

    #[test]
    fn candidate_scoped_consolidate_touches_only_listed_ids() {
        let db = crate::db::TestDatabase::new("segment-candidate");
        let ws = "ws-seg-cand";
        // Five similar entities (no gap) — a single segment.
        write_fixture(&db, &seg_entity("c1", "gamma topic one", 1_000, ws)).unwrap();
        write_fixture(&db, &seg_entity("c2", "gamma topic two", 2_000, ws)).unwrap();
        write_fixture(&db, &seg_entity("c3", "gamma topic three", 3_000, ws)).unwrap();
        write_fixture(&db, &seg_entity("c4", "gamma topic four", 4_000, ws)).unwrap();
        write_fixture(&db, &seg_entity("c5", "gamma topic five", 5_000, ws)).unwrap();
        // Direct candidate-scoped consolidate over the first three only.
        let params: crate::models::ConsolidateParams = serde_json::from_value(serde_json::json!({
            "category": "turns",
            "workspace_hash": ws,
        }))
        .unwrap();
        let report = db
            .consolidate_with_candidates(&params, Some(&["c1".into(), "c2".into(), "c3".into()]))
            .unwrap();
        assert_eq!(report.observations_created, 1);
        assert_eq!(report.source_entities_merged, 3);
        // c4/c5 untouched by this scoped pass: still live, no observation
        // covers them yet — a follow-up scoped run over them merges them.
        let params2: crate::models::ConsolidateParams = serde_json::from_value(serde_json::json!({
            "category": "turns",
            "workspace_hash": ws,
        }))
        .unwrap();
        let report2 = db
            .consolidate_with_candidates(&params2, Some(&["c4".into(), "c5".into()]))
            .unwrap();
        assert_eq!(report2.observations_created, 1);
        assert_eq!(report2.source_entities_merged, 2);
    }

    #[test]
    fn singletons_are_skipped_not_consolidated() {
        let db = crate::db::TestDatabase::new("segment-single");
        let ws = "ws-seg-single";
        write_fixture(&db, &seg_entity("s1", "solo entity", 1_000, ws)).unwrap();
        let report = db
            .segment_consolidate("turns", ws, 60_000, 0.25, 100, false)
            .unwrap();
        assert_eq!(report["segments"], 1);
        assert_eq!(report["consolidated"], 0);
        assert_eq!(report["skipped_singletons"], 1);
        assert_eq!(state_count(&db), 0);
    }
}
