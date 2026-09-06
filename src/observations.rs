//! #884: evidence-grounded observations.
//!
//! Consolidation artifacts ("observations", category `observation`) carry
//! evidence refs (source memory id + exact quote), a proof count, an
//! updated_at, a staleness flag, and — when evidence contradicts the
//! observation — a preserved journey (e.g. "was React, switched to Vue")
//! instead of a blind overwrite. Raw facts stay intact for trace-back.
//!
//! This module is the pure, deterministic core: parsing the observation
//! body schema, quote extraction, match classification (fold / contradiction
//! / unrelated), and refinement folding. All DB access lives in `db.rs`
//! (consolidate, ask gate); everything here is unit-testable without a DB.
//!
//! Body schema v2 (v1 bodies — summary/source_ids/proof_count/
//! merged_from_category only — parse tolerantly):
//! ```json
//! {
//!   "summary": "stack uses React",
//!   "source_ids": ["mem-..."],
//!   "quotes": [{"source_id": "mem-...", "quote": "stack uses React"}],
//!   "proof_count": 2,
//!   "merged_from_category": "tech",
//!   "updated_at_unix_ms": 1750000000000,
//!   "stale": false,
//!   "history": [{"from": "stack uses React", "to": "stack switched to Vue",
//!                "changed_at_unix_ms": ..., "triggered_by": "mem-...",
//!                "reason": "contradiction"}]
//! }
//! ```

use serde::{Deserialize, Serialize};

/// Category of observation entities.
pub const OBSERVATION_CATEGORY: &str = "observation";

/// Staleness/verification threshold for the ask gate. Matches
/// `default_consolidate_threshold` so a fact the consolidator would fold is
/// also a fact the gate treats as consistent.
pub const OBSERVATION_VERIFY_THRESHOLD: f64 = 0.6;

/// A single exact-quote evidence ref (source memory id + verbatim quote).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct QuoteRef {
    pub source_id: String,
    pub quote: String,
}

/// One preserved step of the observation's journey (correction/extension).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct JourneyEntry {
    /// The observation's summary before the change.
    pub from: String,
    /// The observation's summary after the change.
    pub to: String,
    pub changed_at_unix_ms: i64,
    /// The raw fact that triggered the change (trace-back anchor).
    pub triggered_by: String,
    /// "contradiction" (0 < sim < threshold). Folded evidence never creates
    /// a journey entry — it only strengthens the observation.
    pub reason: String,
}

/// Parsed observation body (v1 and v2).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ObservationMeta {
    pub summary: String,
    pub source_ids: Vec<String>,
    pub quotes: Vec<QuoteRef>,
    pub proof_count: i64,
    pub merged_from_category: String,
    pub updated_at_unix_ms: i64,
    pub stale: bool,
    pub history: Vec<JourneyEntry>,
}

/// How a candidate raw fact relates to an existing observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatchClass {
    /// Trigram similarity >= threshold (or near-subset/superset
    /// containment): same claim, strengthens it.
    Fold,
    /// 0 < similarity < threshold: same topic, contradicts/revises it.
    Contradiction,
    /// No shared trigrams (or empty bodies): unrelated, do not touch.
    Unrelated,
}

/// #884: a candidate whose trigram set is a near-subset/superset of the
/// summary's (containment >= this floor) is an extension of the same claim
/// and folds, even when raw Jaccard sits below the similarity threshold.
/// 0.8 keeps genuine revisions ("server runs postgres" vs "server runs
/// mysql" — ~0.64 containment) classified as contradictions.
pub const FOLD_CONTAINMENT_FLOOR: f64 = 0.8;

/// Extract the exact quote for a source body: the `note` field verbatim
/// when present, else the body itself. Truncated at `cap` chars with an
/// ellipsis marker (bounded deterministic quote; `cap` is validated by the
/// caller).
pub fn quote_for(body: &str, cap: usize) -> String {
    let text = body_text(body);
    if text.chars().count() > cap {
        let head: String = text.chars().take(cap).collect();
        format!("{head}…")
    } else {
        text
    }
}

/// The human-readable text of a body: `note` field when the body parses as
/// an object with a string note, else the body itself (or compact JSON for
/// objects without a note).
pub fn body_text(body: &str) -> String {
    match serde_json::from_str::<serde_json::Value>(body) {
        Ok(serde_json::Value::Object(map)) => {
            if let Some(s) = map.get("note").and_then(|v| v.as_str()) {
                if !s.trim().is_empty() {
                    return s.to_string();
                }
            }
            // Objects with a summary field are already human-readable
            // (legacy observation bodies).
            if let Some(s) = map.get("summary").and_then(|v| v.as_str()) {
                if !s.trim().is_empty() {
                    return s.to_string();
                }
            }
            serde_json::Value::Object(map).to_string()
        }
        Ok(serde_json::Value::String(s)) => s,
        _ => body.to_string(),
    }
}

/// Tolerant parse of an observation body. v1 bodies (no quotes/updated_at/
/// stale/history) parse with defaults; `fallback_updated_at` anchors v1
/// bodies at their entity creation time so staleness math stays correct.
pub fn parse_observation(body_json: &str, fallback_updated_at: i64) -> Option<ObservationMeta> {
    let v: serde_json::Value = serde_json::from_str(body_json).ok()?;
    let obj = v.as_object()?;
    let summary = obj
        .get("summary")
        .map(|s| {
            s.as_str()
                .map(|x| x.to_string())
                .unwrap_or_else(|| s.to_string())
        })
        .unwrap_or_default();
    let source_ids: Vec<String> = obj
        .get("source_ids")
        .and_then(|s| s.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|x| x.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();
    let quotes: Vec<QuoteRef> = obj
        .get("quotes")
        .and_then(|q| q.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|x| {
                    let o = x.as_object()?;
                    Some(QuoteRef {
                        source_id: o.get("source_id")?.as_str()?.to_string(),
                        quote: o.get("quote")?.as_str()?.to_string(),
                    })
                })
                .collect()
        })
        .unwrap_or_default();
    let history: Vec<JourneyEntry> = obj
        .get("history")
        .and_then(|h| h.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|x| {
                    let o = x.as_object()?;
                    Some(JourneyEntry {
                        from: o.get("from")?.as_str()?.to_string(),
                        to: o.get("to")?.as_str()?.to_string(),
                        changed_at_unix_ms: o.get("changed_at_unix_ms")?.as_i64()?,
                        triggered_by: o.get("triggered_by")?.as_str()?.to_string(),
                        reason: o
                            .get("reason")
                            .and_then(|r| r.as_str())
                            .unwrap_or("contradiction")
                            .to_string(),
                    })
                })
                .collect()
        })
        .unwrap_or_default();
    let proof_count = obj
        .get("proof_count")
        .and_then(|p| p.as_i64())
        .unwrap_or(source_ids.len() as i64);
    Some(ObservationMeta {
        summary,
        source_ids,
        quotes,
        proof_count,
        merged_from_category: obj
            .get("merged_from_category")
            .and_then(|m| m.as_str())
            .unwrap_or_default()
            .to_string(),
        updated_at_unix_ms: obj
            .get("updated_at_unix_ms")
            .and_then(|u| u.as_i64())
            .unwrap_or(fallback_updated_at),
        stale: obj.get("stale").and_then(|s| s.as_bool()).unwrap_or(false),
        history,
    })
}

/// Classify a candidate raw fact against an observation summary.
pub fn classify(obs_summary: &str, candidate_body: &str, threshold: f64) -> MatchClass {
    let candidate = body_text(candidate_body);
    // Exact text match (bodies shorter than one trigram) is always a fold —
    // the same rule consolidate's cluster scan uses for short bodies.
    if !obs_summary.is_empty() && obs_summary == candidate {
        return MatchClass::Fold;
    }
    let sim = crate::db::Database::trigram_overlap_public(obs_summary, &candidate);
    // #884: a candidate whose trigrams are a near-subset/superset of the
    // summary is an EXTENSION of the same claim (e.g. "stack uses react" vs
    // "stack uses react with hooks" — Jaccard 0.56 < threshold but the
    // shorter text is fully contained). Such candidates fold instead of
    // being recorded as contradictions. Must agree with the cluster scan.
    let containment = crate::db::Database::trigram_containment_public(obs_summary, &candidate);
    if sim >= threshold || containment >= FOLD_CONTAINMENT_FLOOR {
        MatchClass::Fold
    } else if sim > 0.0 {
        MatchClass::Contradiction
    } else {
        MatchClass::Unrelated
    }
}

/// Fold new evidence into an observation. Deterministic:
/// - `new_sources` are deduplicated against existing source_ids and sorted
///   by id before folding;
/// - each new source is classified against the CURRENT summary; the first
///   contradiction (in id order) revises the summary and appends a journey
///   entry (`reason: "contradiction"`), later sources classify against the
///   revised summary;
/// - quotes are appended (capped), proof_count and updated_at bumped;
/// - sources are never removed — the full evidence trail survives.
pub fn refine(
    existing: &ObservationMeta,
    new_sources: &[(String, String)],
    now: i64,
    threshold: f64,
    quote_cap: usize,
) -> ObservationMeta {
    let mut out = existing.clone();
    let mut added: Vec<&(String, String)> = new_sources
        .iter()
        .filter(|(id, _)| !out.source_ids.iter().any(|s| s == id))
        .collect();
    if added.is_empty() {
        return out;
    }
    added.sort_by(|a, b| a.0.as_str().cmp(b.0.as_str()));

    let mut folded_any = false;
    for (id, body) in added {
        let text = body_text(body);
        // Unrelated evidence is NOT folded in: it neither supports nor
        // contradicts the observation — it only keeps it stale.
        if classify(&out.summary, &text, threshold) == MatchClass::Unrelated {
            continue;
        }
        out.source_ids.push(id.clone());
        out.quotes.push(QuoteRef {
            source_id: id.clone(),
            quote: quote_for(body, quote_cap),
        });
        out.proof_count += 1;
        folded_any = true;
        match classify(&out.summary, &text, threshold) {
            MatchClass::Contradiction => {
                let to = text.clone();
                out.history.push(JourneyEntry {
                    from: out.summary.clone(),
                    to: to.clone(),
                    changed_at_unix_ms: now,
                    triggered_by: id.clone(),
                    reason: "contradiction".to_string(),
                });
                out.summary = to;
            }
            MatchClass::Fold | MatchClass::Unrelated => {}
        }
    }
    if folded_any {
        out.updated_at_unix_ms = now;
        out.stale = false;
    }
    out
}

/// Serialize an observation body (v2 schema).
pub fn observation_body(meta: &ObservationMeta) -> String {
    serde_json::json!({
        "summary": meta.summary,
        "source_ids": meta.source_ids,
        "quotes": meta.quotes,
        "proof_count": meta.proof_count,
        "merged_from_category": meta.merged_from_category,
        "updated_at_unix_ms": meta.updated_at_unix_ms,
        "stale": meta.stale,
        "history": meta.history,
    })
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    const NOW: i64 = 1_750_000_000_000;

    #[test]
    fn quote_uses_note_verbatim_and_caps() {
        assert_eq!(
            quote_for(r#"{"note": "stack uses react"}"#, 512),
            "stack uses react"
        );
        let long = format!("{}", "x".repeat(600));
        let q = quote_for(&format!(r#"{{"note": "{long}"}}"#), 512);
        assert_eq!(q.chars().count(), 513); // 512 + ellipsis
        assert!(q.ends_with('…'));
        // Plain-text bodies quote themselves.
        assert_eq!(quote_for("plain body text", 512), "plain body text");
    }

    #[test]
    fn body_text_prefers_note_then_summary_then_compact() {
        assert_eq!(body_text(r#"{"note": "n"}"#), "n");
        assert_eq!(body_text(r#"{"summary": "s", "other": 1}"#), "s");
        assert_eq!(body_text("plain"), "plain");
        assert!(body_text(r#"{"a": 1}"#).contains('"'));
    }

    #[test]
    fn parse_legacy_v1_body_tolerantly() {
        let meta = parse_observation(
            r#"{"summary": "legacy", "source_ids": ["m1"], "proof_count": 1,
                "merged_from_category": "tech"}"#,
            NOW,
        )
        .unwrap();
        assert_eq!(meta.summary, "legacy");
        assert_eq!(meta.updated_at_unix_ms, NOW);
        assert!(meta.quotes.is_empty());
        assert!(!meta.stale);
        assert!(meta.history.is_empty());
    }

    #[test]
    fn parse_v2_body_roundtrip() {
        let meta = ObservationMeta {
            summary: "s".to_string(),
            source_ids: vec!["m1".to_string()],
            quotes: vec![QuoteRef {
                source_id: "m1".to_string(),
                quote: "q".to_string(),
            }],
            proof_count: 1,
            merged_from_category: "tech".to_string(),
            updated_at_unix_ms: NOW,
            stale: true,
            history: vec![JourneyEntry {
                from: "a".to_string(),
                to: "b".to_string(),
                changed_at_unix_ms: NOW,
                triggered_by: "m1".to_string(),
                reason: "contradiction".to_string(),
            }],
        };
        let parsed = parse_observation(&observation_body(&meta), 0).unwrap();
        assert_eq!(parsed, meta);
    }

    #[test]
    fn classify_fold_contradiction_unrelated() {
        // Near-duplicate → fold.
        assert_eq!(
            classify(
                "stack uses react",
                r#"{"note": "stack uses react with hooks"}"#,
                0.5
            ),
            MatchClass::Fold
        );
        // Same topic, revised claim → contradiction.
        assert_eq!(
            classify(
                "stack uses react",
                r#"{"note": "stack switched to vue"}"#,
                0.5
            ),
            MatchClass::Contradiction
        );
        // Unrelated topic → no shared trigrams.
        assert_eq!(
            classify(
                "stack uses react",
                r#"{"note": "the weather is sunny today in berlin"}"#,
                0.5
            ),
            MatchClass::Unrelated
        );
        // Exact duplicate body → fold (identical trigram sets).
        assert_eq!(
            classify("stack uses react", "stack uses react", 0.5),
            MatchClass::Fold
        );
    }

    #[test]
    fn refine_folds_evidence_without_journey() {
        let existing = ObservationMeta {
            summary: "stack uses react".to_string(),
            source_ids: vec!["m1".to_string()],
            quotes: vec![],
            proof_count: 1,
            merged_from_category: "tech".to_string(),
            updated_at_unix_ms: NOW - 1000,
            stale: true,
            history: vec![],
        };
        let out = refine(
            &existing,
            &[(
                "m2".to_string(),
                r#"{"note": "stack uses react with hooks"}"#.to_string(),
            )],
            NOW,
            0.5,
            512,
        );
        assert_eq!(out.proof_count, 2);
        assert_eq!(out.source_ids, vec!["m1".to_string(), "m2".to_string()]);
        assert_eq!(out.updated_at_unix_ms, NOW);
        assert!(!out.stale, "folded evidence clears staleness");
        assert!(
            out.history.is_empty(),
            "fold must not create a journey entry"
        );
        assert_eq!(out.summary, "stack uses react");
        assert_eq!(out.quotes[0].source_id, "m2");
        assert_eq!(out.quotes[0].quote, "stack uses react with hooks");
    }

    #[test]
    fn refine_preserves_journey_on_contradiction() {
        let existing = ObservationMeta {
            summary: "stack uses react".to_string(),
            source_ids: vec!["m1".to_string()],
            quotes: vec![],
            proof_count: 1,
            merged_from_category: "tech".to_string(),
            updated_at_unix_ms: NOW - 1000,
            stale: true,
            history: vec![],
        };
        let out = refine(
            &existing,
            &[(
                "m2".to_string(),
                r#"{"note": "stack switched to vue"}"#.to_string(),
            )],
            NOW,
            0.5,
            512,
        );
        assert_eq!(out.summary, "stack switched to vue");
        assert_eq!(out.history.len(), 1);
        let entry = &out.history[0];
        assert_eq!(entry.from, "stack uses react");
        assert_eq!(entry.to, "stack switched to vue");
        assert_eq!(entry.triggered_by, "m2");
        assert_eq!(entry.reason, "contradiction");
        assert_eq!(out.changed_at_unix_ms(), NOW);
        assert_eq!(out.proof_count, 2);
        assert!(!out.stale);
        // Raw facts intact: every source id still listed.
        assert_eq!(out.source_ids, vec!["m1".to_string(), "m2".to_string()]);
    }

    #[test]
    fn refine_dedupes_already_folded_sources() {
        let existing = ObservationMeta {
            summary: "s".to_string(),
            source_ids: vec!["m1".to_string()],
            quotes: vec![QuoteRef {
                source_id: "m1".to_string(),
                quote: "s".to_string(),
            }],
            proof_count: 1,
            merged_from_category: "tech".to_string(),
            updated_at_unix_ms: NOW,
            stale: false,
            history: vec![],
        };
        let out = refine(
            &existing,
            &[("m1".to_string(), "s".to_string())],
            NOW + 100,
            0.5,
            512,
        );
        assert_eq!(out.proof_count, 1, "re-folding must not double-count");
        assert_eq!(out.updated_at_unix_ms, NOW, "no new sources -> no touch");
    }

    #[test]
    fn refine_unrelated_source_is_ignored() {
        let existing = ObservationMeta {
            summary: "stack uses react".to_string(),
            source_ids: vec!["m1".to_string()],
            quotes: vec![],
            proof_count: 1,
            merged_from_category: "tech".to_string(),
            updated_at_unix_ms: NOW - 1000,
            stale: true,
            history: vec![],
        };
        let out = refine(
            &existing,
            &[(
                "m9".to_string(),
                r#"{"note": "the weather is sunny in berlin"}"#.to_string(),
            )],
            NOW,
            0.5,
            512,
        );
        // Unrelated evidence is NOT folded in (it does not support or
        // contradict the observation; it only keeps it stale).
        assert_eq!(out.source_ids, vec!["m1".to_string()]);
        assert_eq!(out.proof_count, 1);
        assert_eq!(out.updated_at_unix_ms, NOW - 1000);
        assert!(
            out.stale,
            "unrelated newer facts keep the observation stale"
        );
    }

    impl ObservationMeta {
        fn changed_at_unix_ms(&self) -> i64 {
            self.history
                .iter()
                .map(|h| h.changed_at_unix_ms)
                .max()
                .unwrap_or(0)
        }
    }
}
