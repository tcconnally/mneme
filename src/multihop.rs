//! #1003: multi-hop retrieval arm — graph traversal + entity-coverage
//! selection (CogniCore MultiHopMemoryBackend borrow).
//!
//! CogniCore's MultiHopMemoryBackend: hop-1 dense+BM25 anchors → graph
//! traversal (graph_next/graph_prev) → final selection by ENTITY COVERAGE —
//! the set that jointly covers the most entities named in the query, not the
//! individually highest-scoring chunks (LongMemEval STRICT R@5 +6.4%).
//!
//! The vault slots this in as a SELECTION STRATEGY over the fused pool
//! (#883), not a new index:
//!
//! 1. **Hop expansion** — the top fused anchors' links are followed via the
//!    existing `graph_expand` (#869 attested-edge gate applies), neighbors
//!    discounted per hop (0.8^hop) and appended to the pool.
//! 2. **Coverage selection** — query entities are the deterministic
//!    stopword-filtered significant tokens (the LongMemEval entity-lexicon
//!    proxy, no LLM); greedy set-cover within the caller limit + #942 token
//!    budget replaces the ranked walk.
//!
//! Guards: opt-in (`multihop`, default OFF — a default recall stays
//! byte-identical, #247); hop budget bounded; workspace scoping inherited
//! from graph_expand; the raw-query arm keeps its place — expansion only
//! ADDS neighbors, it never reorders the anchors themselves.

use serde::Serialize;

pub const DEFAULT_ANCHORS: usize = 3;
pub const DEFAULT_HOP_BUDGET: usize = 1;
pub const HOP_DISCOUNT: f64 = 0.8;
pub const MAX_QUERY_ENTITIES: usize = 12;
pub const MAX_NEIGHBORS_PER_HOP: usize = 20;

/// English stopwords filtered from query-entity extraction (plus the vault's
/// own tool words that carry no coverage signal).
pub const STOPWORDS: [&str; 42] = [
    "the", "and", "for", "with", "that", "this", "these", "those", "what", "which", "when",
    "where", "who", "whom", "whose", "how", "why", "did", "does", "was", "were", "are", "is",
    "been", "being", "have", "has", "had", "from", "into", "about", "your", "their", "they",
    "there", "here", "then", "than", "them", "will", "would", "should",
];

#[derive(Debug, Clone, Serialize)]
pub struct MultiHopTrace {
    pub hop_expanded: usize,
    pub expanded_ids: Vec<String>,
    pub selection_order: Vec<String>,
    pub covered_entities: Vec<String>,
    pub uncovered_entities: Vec<String>,
}

/// Extract the query's significant tokens: lowercase, alphanumeric-only,
/// stopword-filtered, length >= 3, deduped in first-seen order, capped.
pub fn query_entities(query: &str) -> Vec<String> {
    let mut seen: Vec<String> = Vec::new();
    for tok in query
        .split(|c: char| !c.is_alphanumeric())
        .map(|t| t.to_lowercase())
        .filter(|t| t.len() >= 3 && !STOPWORDS.contains(&t.as_str()))
    {
        if !seen.contains(&tok) {
            seen.push(tok);
            if seen.len() >= MAX_QUERY_ENTITIES {
                break;
            }
        }
    }
    seen
}

/// How many query entities the body covers (token containment — the
/// LongMemEval lexicon-coverage primitive). The body is compared lowercased.
pub fn coverage(body: &str, entities: &[String]) -> usize {
    let lower = body.to_lowercase();
    entities
        .iter()
        .filter(|t| lower.contains(t.as_str()))
        .count()
}

/// Greedy entity-coverage selection within the caller limit and the token
/// budget (estimated tokens = chars/4, the #942 convention).
///
/// Repeatedly pick the remaining item covering the most still-uncovered
/// query entities; ties break by score desc then id asc. Once every query
/// entity is covered (or nothing left covers any uncovered entity), the
/// walk falls back to plain score order. Deterministic.
pub fn coverage_select(
    pool: &[(crate::models::Entity, f64)],
    entities: &[String],
    limit: usize,
    token_budget: i64,
) -> (Vec<crate::models::Entity>, MultiHopTrace) {
    let mut remaining: Vec<usize> = (0..pool.len()).collect();
    let mut selected: Vec<usize> = Vec::new();
    let mut covered: Vec<String> = Vec::new();
    let mut tokens_used: i64 = 0;

    while selected.len() < limit && !remaining.is_empty() {
        // Greedy: best coverage of UNCOVERED entities, tie-break score/id.
        let mut best: Option<usize> = None;
        let mut best_cov: usize = 0;
        for &idx in remaining.iter() {
            let (e, _) = &pool[idx];
            let cov_eff = entities
                .iter()
                .filter(|t| !covered.contains(t) && e.body_json.to_lowercase().contains(t.as_str()))
                .count();
            let better = match best {
                None => cov_eff > 0,
                Some(bidx) => {
                    let (be, bs) = &pool[bidx];
                    cov_eff > best_cov
                        || (cov_eff == best_cov && pool[idx].1 > *bs)
                        || (cov_eff == best_cov && pool[idx].1 == *bs && e.id < be.id)
                }
            };
            if better {
                best = Some(idx);
                best_cov = cov_eff;
            }
        }
        // Fall back to score order when nothing covers anything uncovered.
        let pick = match best {
            Some(idx) => idx,
            None => {
                let mut best_idx = remaining[0];
                for &idx in remaining.iter().skip(1) {
                    if pool[idx].1 > pool[best_idx].1
                        || (pool[idx].1 == pool[best_idx].1 && pool[idx].0.id < pool[best_idx].0.id)
                    {
                        best_idx = idx;
                    }
                }
                best_idx
            }
        };
        let (e, _) = &pool[pick];
        let est = (e.body_json.chars().count() / 4).max(1) as i64;
        if !selected.is_empty() && tokens_used + est > token_budget {
            break;
        }
        tokens_used += est;
        selected.push(pick);
        remaining.retain(|&i| i != pick);
        for t in entities {
            if !covered.contains(t) && e.body_json.to_lowercase().contains(t.as_str()) {
                covered.push(t.clone());
            }
        }
    }

    let uncovered: Vec<String> = entities
        .iter()
        .filter(|t| !covered.contains(t))
        .cloned()
        .collect();
    let expanded_ids = Vec::new(); // filled by the caller (db integration)
    let entities_out = selected.iter().map(|&i| pool[i].0.clone()).collect();
    let selection_order = selected.iter().map(|&i| pool[i].0.id.clone()).collect();
    (
        entities_out,
        MultiHopTrace {
            hop_expanded: 0,
            expanded_ids,
            selection_order,
            covered_entities: covered,
            uncovered_entities: uncovered,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn query_entities_filters_stopwords_and_dedupes() {
        let q = "What was the decision about the database migration for the import pipeline?";
        let ents = query_entities(q);
        assert!(ents.contains(&"decision".to_string()));
        assert!(ents.contains(&"database".to_string()));
        assert!(ents.contains(&"migration".to_string()));
        assert!(!ents.contains(&"the".to_string()));
        assert!(!ents.contains(&"was".to_string()));
        assert!(!ents.contains(&"for".to_string()));
    }

    #[test]
    fn coverage_counts_contained_entities() {
        let body = "the database migration completed on schedule";
        let ents = vec![
            "database".to_string(),
            "migration".to_string(),
            "pipeline".to_string(),
        ];
        assert_eq!(coverage(body, &ents), 2);
    }

    #[test]
    fn coverage_select_prefers_covering_items_within_budget() {
        // e1 covers migration+database, e2 covers nothing, e3 covers pipeline.
        let mk = |id: &str, body: &str, score: f64| {
            (
                crate::models::Entity {
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
                    workspace_hash: String::new(),
                    agent_id: String::new(),
                    visibility: "workspace".to_string(),
                    created_at_unix_ms: 0,
                    last_accessed_unix_ms: 0,
                    follow_count: 0,
                    miss_count: 0,
                    follow_rate: 0.0,
                    efficacy_status: "unverified".to_string(),
                    epistemic_state: crate::models::default_epistemic_state(),
                    hints: vec![],
                    memory_type: String::new(),
                    embedding: None,
                    _parsed_body: None,
                },
                score,
            )
        };
        let pool = vec![
            mk("e2", "something entirely unrelated", 0.99),
            mk("e1", "the database migration finished", 0.9),
            mk("e3", "the import pipeline is failing", 0.8),
        ];
        let ents = vec![
            "database".to_string(),
            "migration".to_string(),
            "pipeline".to_string(),
        ];
        let (selected, trace) = coverage_select(&pool, &ents, 2, 100);
        // e1 (covers 2) beats e2 (covers 0) despite e2's higher score.
        assert_eq!(selected[0].id, "e1");
        // Budget-bounded selection of 2: next best coverage = e3.
        assert_eq!(selected.len(), 2);
        assert!(trace.covered_entities.contains(&"database".to_string()));
        assert_eq!(trace.uncovered_entities.len(), 0);
    }

    #[test]
    fn coverage_select_respects_token_budget() {
        let mk = |id: &str, body: &str| {
            (
                crate::models::Entity {
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
                    workspace_hash: String::new(),
                    agent_id: String::new(),
                    visibility: "workspace".to_string(),
                    created_at_unix_ms: 0,
                    last_accessed_unix_ms: 0,
                    follow_count: 0,
                    miss_count: 0,
                    follow_rate: 0.0,
                    efficacy_status: "unverified".to_string(),
                    epistemic_state: crate::models::default_epistemic_state(),
                    hints: vec![],
                    memory_type: String::new(),
                    embedding: None,
                    _parsed_body: None,
                },
                1.0,
            )
        };
        let pool = vec![
            mk("a", "database migration done"),
            mk("b", "pipeline import fixed"),
        ];
        let ents = vec!["database".to_string(), "pipeline".to_string()];
        // Budget fits ~1 item (each body is ~20 chars => est 5 tokens;
        // 5+5=10 > 9 breaks before the second).
        let (selected, _trace) = coverage_select(&pool, &ents, 10, 9);
        assert_eq!(selected.len(), 1);
    }
}
