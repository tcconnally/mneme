//! #1002: sleep-cycle consolidation with dedup + contradiction resolution
//! (CogniCore SleepProcessor borrow).
//!
//! CogniCore's SleepProcessor runs off-peak: dedup, contradiction detection
//! (jaccard + negation heuristics), episode compression. The vault's version
//! reuses the EXISTING bounded primitives instead of new scan machinery:
//!
//! - **dedup phase** — pairwise trigram similarity >= threshold within the
//!   #952 bounded window → MERGE PROPOSAL (never auto-merged).
//! - **contradiction phase** — the CogniCore cheap prefilter: token overlap
//!   PLUS a negation word in one body ("X works" vs "X does NOT work") →
//!   CONFLICT PROPOSAL. Escalates to the operator review queue; never
//!   auto-resolved.
//! - **compression phase** — delegates to the existing #952 `consolidate`
//!   (bounded, exempts verified/scored, emits summary observations with
//!   supersedes links) — the only auto-committed artifact, and it is already
//!   derived + hash-linked by that machinery.
//!
//! Proposals are stored under state keys `sleep_proposal.<uuid>` (no schema
//! change) and surfaced as the `sleep` lane of `perseus_vault_operator_review`.
//! Dry-run reports without persisting. Every phase is bounded by the #952
//! window + a per-run budget so the pass cannot contend with live recall.

use serde::Serialize;

pub const NEGATION_WORDS: [&str; 12] = [
    "not",
    "no",
    "never",
    "cannot",
    "can't",
    "won't",
    "must not",
    "mustn't",
    "no longer",
    "don't",
    "do not",
    "stop",
];

#[derive(Debug, Clone, Serialize)]
pub struct SleepProposal {
    pub kind: String, // "merge" | "conflict"
    pub category: String,
    pub entity_a: String,
    pub entity_b: String,
    pub similarity: f64,
    pub reason: String,
    pub workspace_hash: String,
    pub status: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct SleepReport {
    pub category: String,
    pub dry_run: bool,
    pub scanned: usize,
    pub dedup_proposals: usize,
    pub conflict_proposals: usize,
    pub compression: serde_json::Value,
    pub proposals: Vec<SleepProposal>,
}

pub const STATE_PREFIX: &str = "sleep_proposal.";

/// The CogniCore cheap contradiction prefilter: the two bodies share at
/// least two non-trivial tokens AND one of them carries a negation word —
/// the "X works" vs "X does not work" shape. Never a verdict — only the
/// escalation signal.
pub fn negation_prefilter(a: &str, b: &str) -> bool {
    let tokens = |s: &str| -> Vec<String> {
        s.to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|t| t.len() >= 3)
            .map(|t| t.to_string())
            .collect()
    };
    let ta = tokens(a);
    let tb = tokens(b);
    let overlap = ta.iter().filter(|t| tb.contains(t)).count();
    if overlap < 2 {
        return false;
    }
    let has_negation = |s: &str| {
        let lower = s.to_lowercase();
        let words: Vec<&str> = lower.split_whitespace().collect();
        NEGATION_WORDS.iter().any(|w| {
            if w.contains(' ') {
                // Phrase negation ("must not", "no longer"): bigram match.
                words
                    .windows(2)
                    .any(|pair| format!("{} {}", pair[0], pair[1]) == *w)
            } else {
                words.iter().any(|t| t == w)
            }
        })
    };
    has_negation(a) || has_negation(b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn negation_prefilter_requires_overlap_and_negation() {
        assert!(negation_prefilter(
            "the import works fine",
            "the import does not work"
        ));
        assert!(!negation_prefilter(
            "the import works fine",
            "the import works today"
        ));
        assert!(!negation_prefilter("alpha", "beta does not work"));
        assert!(!negation_prefilter("import works", "export broken"));
    }

    #[test]
    fn negation_words_cover_contractions_and_phrases() {
        assert!(negation_prefilter(
            "alpha beta gamma must not run",
            "alpha beta gamma runs"
        ));
        assert!(negation_prefilter(
            "alpha beta gamma never runs",
            "alpha beta gamma runs"
        ));
        assert!(negation_prefilter(
            "alpha beta gamma runs",
            "alpha beta gamma no longer runs"
        ));
        // "nothing" contains "not" as a substring — token matching must not
        // false-positive on it.
        assert!(!negation_prefilter(
            "alpha beta nothing here",
            "alpha beta gamma"
        ));
    }

    #[test]
    fn run_sleep_dry_run_proposes_without_persisting() {
        let db = crate::db::TestDatabase::new("sleep-run");
        let now = crate::db::now_ms();
        let mut seed = |id: &str, body: &str| crate::models::Entity {
            id: id.to_string(),
            category: "facts".to_string(),
            key: id.to_string(),
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
            workspace_hash: "ws-a".to_string(),
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
            embedding: None,
            memory_type: String::new(),
            _parsed_body: None,
        };
        // remember() write-dedups near-duplicates (signature-based), so seed
        // via remember_skip_dedup — the sleep pass exists precisely to catch
        // rows that never went through the write-time dedup gate (legacy
        // imports, skip_dedup writers).
        db.remember_skip_dedup(&seed(
            "s-1",
            "the import pipeline handles csv files correctly",
        ))
        .unwrap();
        db.remember_skip_dedup(&seed(
            "s-2",
            "the import pipeline handles csv files perfectly",
        ))
        .unwrap();
        db.remember_skip_dedup(&seed("s-3", "the import pipeline does not work on linux"))
            .unwrap();
        let params = crate::models::SleepParams {
            category: "facts".to_string(),
            similarity_threshold: 0.6,
            max_entities: 50,
            max_proposals: 50,
            dry_run: true,
            include_compression: false,
            workspace_hash: Some("ws-a".to_string()),
            global: false,
            requesting_agent_id: String::new(),
            force: true,
        };
        let report = db.run_sleep(&params).unwrap();
        assert_eq!(report.scanned, 3, "scan must see all three seeded entities");
        assert!(
            report.dedup_proposals >= 1,
            "near-duplicate pair must be proposed as merge"
        );
        assert!(
            report.proposals.iter().any(|p| p.kind == "conflict" && {
                (p.entity_a == "s-1" || p.entity_a == "s-3")
                    && (p.entity_b == "s-1" || p.entity_b == "s-3")
            }),
            "negation pair must be proposed as conflict even when textually similar"
        );
        // Dry-run: zero persisted proposals.
        let keys = db.state_list(crate::sleep::STATE_PREFIX).unwrap();
        assert!(keys.is_empty());
    }
}
