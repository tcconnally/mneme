//! #886: mental-models tier — user/agent-curated summaries consulted first.
//!
//! Hindsight borrow: curated summaries for frequent queries sit ABOVE
//! consolidated observations and raw facts in hierarchical retrieval, because
//! they are explicitly curated (not auto-generated), versioned, and carry
//! provenance. This module is the pure, deterministic core: body schema
//! parsing/serialization, tier classification, staleness reasons, and the
//! curation validation rules. All DB access lives in `db.rs` (set/review
//! tools, recall tier ordering, ask marking, operator-review integration);
//! everything here is unit-testable without a DB.
//!
//! Body schema v1 (all fields optional except `summary`):
//! ```json
//! {
//!   "summary": "stack uses vue for the portal",
//!   "scope": "tech",                       // raw-fact category this model covers ("" = none)
//!   "source_ids": ["mem-1"],               // provenance: raw facts/observations it was curated from
//!   "recall_when": ["stack", "portal"],    // triggers for scheduled re-verification via recall_when
//!   "curated_by": "operator",              // provenance: who curated/re-asserted it
//!   "curated_at_unix_ms": 1750000000000,
//!   "reviewed_at_unix_ms": 1750000000000,  // last operator review stamp
//!   "review_interval_days": 30,            // age-based staleness policy
//!   "revision": 2,                         // bump on every curated re-assert
//!   "stale": false,                        // derived snapshot; read-time computation wins
//!   "stale_reason": "",                    // "age" | "newer_facts" | "" (derived)
//!   "last_review_decision": "approved"     // operator decision history ("approved"|"dismissed"|"")
//! }
//! ```
//!
//! Staleness is computed at read time (like observations, #884):
//! - `age`: now - reviewed_at (or curated_at) > review_interval_days;
//! - `newer_facts`: a raw fact exists in `scope` category created after
//!   curated_at that is not already a source (checked in db.rs — the DB
//!   owns that query; the pure rule set is here).

use serde::{Deserialize, Serialize};

/// Category of mental-model entities.
pub const MENTAL_MODEL_CATEGORY: &str = "mental_model";

/// Entity type stamped on mental-model entities (parallels the category, so
/// `type` filters and the MCP surface are consistent).
pub const MENTAL_MODEL_TYPE: &str = "mental_model";

/// Default review interval (days) when the curator does not specify one.
pub const DEFAULT_REVIEW_INTERVAL_DAYS: i64 = 30;

/// Hard bounds for curated content, validated fail-closed at the set tool.
pub const SUMMARY_MIN_CHARS: usize = 1;
pub const SUMMARY_MAX_CHARS: usize = 4096;
pub const REVIEW_INTERVAL_MIN_DAYS: i64 = 1;
pub const REVIEW_INTERVAL_MAX_DAYS: i64 = 3650;
pub const RECALL_WHEN_MAX_ENTRIES: usize = 32;
pub const RECALL_WHEN_MAX_CHARS: usize = 128;
pub const SOURCE_IDS_MAX: usize = 256;

/// Retrieval tiers for the hierarchy mental models → observations → raw facts.
pub const TIER_MENTAL_MODEL: u8 = 0;
pub const TIER_OBSERVATION: u8 = 1;
pub const TIER_RAW: u8 = 2;

/// The tier of an entity by category. Unknown categories are raw facts.
pub fn tier_of(category: &str) -> u8 {
    match category {
        MENTAL_MODEL_CATEGORY => TIER_MENTAL_MODEL,
        crate::observations::OBSERVATION_CATEGORY => TIER_OBSERVATION,
        _ => TIER_RAW,
    }
}

/// Parsed mental-model body (v1).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MentalModelMeta {
    pub summary: String,
    pub scope: String,
    pub source_ids: Vec<String>,
    pub recall_when: Vec<String>,
    pub curated_by: String,
    pub curated_at_unix_ms: i64,
    pub reviewed_at_unix_ms: i64,
    pub review_interval_days: i64,
    pub revision: i64,
    pub stale: bool,
    pub stale_reason: String,
    pub last_review_decision: String,
}

impl MentalModelMeta {
    /// A fresh meta for a first curation.
    pub fn new(
        summary: &str,
        scope: &str,
        source_ids: Vec<String>,
        recall_when: Vec<String>,
        curated_by: &str,
        now_ms: i64,
        review_interval_days: i64,
    ) -> Self {
        MentalModelMeta {
            summary: summary.to_string(),
            scope: scope.to_string(),
            source_ids,
            recall_when,
            curated_by: curated_by.to_string(),
            curated_at_unix_ms: now_ms,
            reviewed_at_unix_ms: now_ms,
            review_interval_days,
            revision: 1,
            stale: false,
            stale_reason: String::new(),
            last_review_decision: String::new(),
        }
    }

    /// Effective anchor of the age clock: the last review stamp, else the
    /// curation timestamp.
    pub fn review_anchor_ms(&self) -> i64 {
        if self.reviewed_at_unix_ms > 0 {
            self.reviewed_at_unix_ms
        } else {
            self.curated_at_unix_ms
        }
    }

    /// Pure staleness reasons — the subset computable without the DB:
    /// age-based flags only. `newer_facts` is appended by the DB check
    /// (`db.rs`), which knows the scope category's raw facts.
    pub fn stale_reasons_pure(&self, now_ms: i64) -> Vec<String> {
        let mut reasons = Vec::new();
        let anchor = self.review_anchor_ms();
        if self.review_interval_days > 0 && anchor > 0 {
            let age_days = (now_ms - anchor).max(0) / 86_400_000;
            if age_days > self.review_interval_days {
                reasons.push("age".to_string());
            }
        }
        reasons
    }

    /// Full staleness decision given the optional newest-fact evidence from
    /// the DB (`newer_fact_key` = the newest raw fact in scope created after
    /// curation and not already a source; None = no such fact).
    pub fn staleness(&self, now_ms: i64, newer_fact_key: Option<&str>) -> (bool, String) {
        let mut reasons = self.stale_reasons_pure(now_ms);
        if let Some(k) = newer_fact_key {
            reasons.push(format!("newer_facts:{k}"));
        }
        if reasons.is_empty() {
            (false, String::new())
        } else {
            (true, reasons.join(","))
        }
    }
}

/// Parse a mental-model body tolerantly: unknown/missing fields get safe
/// defaults; a non-object or missing summary yields None (the caller treats
/// it as a malformed mental model — surfaced, never silently ranked).
pub fn parse_mental_model(body: &str) -> Option<MentalModelMeta> {
    let v: serde_json::Value = serde_json::from_str(body).ok()?;
    let obj = v.as_object()?;
    let summary = obj.get("summary").and_then(|s| s.as_str()).unwrap_or("");
    if summary.trim().is_empty() {
        return None;
    }
    let arr = |k: &str| -> Vec<String> {
        obj.get(k)
            .and_then(|x| x.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|s| s.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default()
    };
    let num = |k: &str| -> i64 { obj.get(k).and_then(|x| x.as_i64()).unwrap_or_default() };
    let interval = num("review_interval_days");
    Some(MentalModelMeta {
        summary: summary.to_string(),
        scope: obj
            .get("scope")
            .and_then(|s| s.as_str())
            .unwrap_or("")
            .to_string(),
        source_ids: arr("source_ids"),
        recall_when: arr("recall_when"),
        curated_by: obj
            .get("curated_by")
            .and_then(|s| s.as_str())
            .unwrap_or("operator")
            .to_string(),
        curated_at_unix_ms: num("curated_at_unix_ms"),
        reviewed_at_unix_ms: num("reviewed_at_unix_ms"),
        // Preserve the stored interval when present (1..=3650 validates
        // at the set tool); default 30 only when missing or zero.
        review_interval_days: if interval > 0 {
            interval
        } else {
            DEFAULT_REVIEW_INTERVAL_DAYS
        },
        revision: num("revision").max(1),
        stale: obj.get("stale").and_then(|b| b.as_bool()).unwrap_or(false),
        stale_reason: obj
            .get("stale_reason")
            .and_then(|s| s.as_str())
            .unwrap_or("")
            .to_string(),
        last_review_decision: obj
            .get("last_review_decision")
            .and_then(|s| s.as_str())
            .unwrap_or("")
            .to_string(),
    })
}

/// Canonical serialization of a mental-model body (deterministic key order —
/// serde_json preserves struct field order for a struct, and this struct is
/// the single writer of the canonical form).
pub fn serialize_meta(meta: &MentalModelMeta) -> String {
    serde_json::to_string(meta).unwrap_or_else(|_| "{}".to_string())
}

/// Curation validation — fail-closed at the set tool. Returns the first
/// problem as an error string, None when valid.
pub fn validate_curation(
    key: &str,
    summary: &str,
    review_interval_days: i64,
    recall_when: &[String],
    source_ids: &[String],
) -> Option<String> {
    if key.trim().is_empty() {
        return Some("key is required for a mental model".to_string());
    }
    if key.chars().count() > 256 {
        return Some("key must be at most 256 chars".to_string());
    }
    let slen = summary.chars().count();
    if slen < SUMMARY_MIN_CHARS || slen > SUMMARY_MAX_CHARS {
        return Some(format!(
            "summary must be {}..={} chars (got {slen})",
            SUMMARY_MIN_CHARS, SUMMARY_MAX_CHARS
        ));
    }
    if review_interval_days < REVIEW_INTERVAL_MIN_DAYS
        || review_interval_days > REVIEW_INTERVAL_MAX_DAYS
    {
        return Some(format!(
            "review_interval_days must be {}..={} (got {review_interval_days})",
            REVIEW_INTERVAL_MIN_DAYS, REVIEW_INTERVAL_MAX_DAYS
        ));
    }
    if recall_when.len() > RECALL_WHEN_MAX_ENTRIES {
        return Some(format!(
            "recall_when must have at most {RECALL_WHEN_MAX_ENTRIES} triggers"
        ));
    }
    for t in recall_when {
        let tc = t.chars().count();
        if tc == 0 || tc > RECALL_WHEN_MAX_CHARS {
            return Some(format!(
                "each recall_when trigger must be 1..={RECALL_WHEN_MAX_CHARS} chars"
            ));
        }
    }
    if source_ids.len() > SOURCE_IDS_MAX {
        return Some(format!(
            "source_ids must have at most {SOURCE_IDS_MAX} entries"
        ));
    }
    None
}

/// Stable in-place tier reorder: mental models first, then observations,
/// then raw facts — preserving relative order within each tier. This is the
/// #886 hierarchy for single-list retrieval: curated summaries are consulted
/// before consolidated beliefs, which are consulted before raw facts. It
/// reorders the list only (membership and scores untouched).
pub fn apply_tier_order(entities: &mut Vec<crate::models::Entity>) {
    if entities.len() < 2 {
        return;
    }
    // Stable partition by tier via sort_by_key on the tier value — Rust's
    // sort is stable, so equal tiers keep their original (rank) order.
    entities.sort_by_key(|e| tier_of(&e.category));
}

/// The ask-context prefix for a mental-model entity: `None` for non-mental
/// models, else `[mental model: <key>]` with the pending-review marker when
/// `stale` (the summary is curated, so it is flagged, never silently
/// dropped — refresh is the operator's review decision).
pub fn context_prefix(entity_key: &str, category: &str, stale: bool) -> Option<String> {
    if category != MENTAL_MODEL_CATEGORY {
        return None;
    }
    let mut prefix = format!("[mental model: {entity_key}]");
    if stale {
        prefix.push_str(" (stale — pending operator review)");
    }
    Some(prefix)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tier_of_classifies_hierarchy() {
        assert_eq!(tier_of(MENTAL_MODEL_CATEGORY), TIER_MENTAL_MODEL);
        assert_eq!(
            tier_of(crate::observations::OBSERVATION_CATEGORY),
            TIER_OBSERVATION
        );
        assert_eq!(tier_of("facts"), TIER_RAW);
        assert_eq!(tier_of(""), TIER_RAW);
    }

    #[test]
    fn parse_round_trips_canonical_body() {
        let meta = MentalModelMeta::new(
            "stack uses vue",
            "tech",
            vec!["mem-1".to_string()],
            vec!["stack".to_string(), "portal".to_string()],
            "operator",
            1_750_000_000_000,
            30,
        );
        let body = serialize_meta(&meta);
        let parsed = parse_mental_model(&body).expect("round trip");
        assert_eq!(parsed.summary, "stack uses vue");
        assert_eq!(parsed.scope, "tech");
        assert_eq!(parsed.source_ids, vec!["mem-1".to_string()]);
        assert_eq!(
            parsed.recall_when,
            vec!["stack".to_string(), "portal".to_string()]
        );
        assert_eq!(parsed.curated_by, "operator");
        assert_eq!(parsed.revision, 1);
        assert_eq!(parsed.review_interval_days, 30);
    }

    #[test]
    fn parse_tolerates_garbage_and_missing_summary() {
        assert!(parse_mental_model("not json").is_none());
        assert!(parse_mental_model("{}").is_none());
        assert!(parse_mental_model(r#"{"summary":""}"#).is_none());
        assert!(parse_mental_model(r#"{"summary":"  "}"#).is_none());
        // Tolerant defaults for missing fields.
        let m = parse_mental_model(r#"{"summary":"ok"}"#).unwrap();
        assert_eq!(m.scope, "");
        assert_eq!(m.curated_by, "operator");
        assert_eq!(m.review_interval_days, DEFAULT_REVIEW_INTERVAL_DAYS);
        assert_eq!(m.revision, 1);
        assert!(m.source_ids.is_empty());
        assert!(m.recall_when.is_empty());
    }

    #[test]
    fn staleness_age_rule_and_review_anchor() {
        let now = 1_750_000_000_000i64;
        let day = 86_400_000i64;
        let m = MentalModelMeta::new("s", "", vec![], vec![], "operator", now - 31 * day, 30);
        // 31 days since curation, no review stamp → stale by age.
        let (stale, reason) = m.staleness(now, None);
        assert!(stale);
        assert!(reason.contains("age"));
        // A review stamp resets the clock.
        let mut m2 = m.clone();
        m2.reviewed_at_unix_ms = now - 5 * day;
        let (stale, _) = m2.staleness(now, None);
        assert!(!stale);
        // Interval not yet elapsed → fresh.
        let m3 = MentalModelMeta::new("s", "", vec![], vec![], "operator", now - 10 * day, 30);
        let (stale, _) = m3.staleness(now, None);
        assert!(!stale);
    }

    #[test]
    fn staleness_newer_fact_rule() {
        let now = 1_750_000_000_000i64;
        let m = MentalModelMeta::new("s", "tech", vec![], vec![], "operator", now, 30);
        let (stale, reason) = m.staleness(now, Some("mem-99"));
        assert!(stale);
        assert!(reason.contains("newer_facts:mem-99"));
        // Zero interval disables the age rule (facts-only policy).
        let mut m2 = m.clone();
        m2.review_interval_days = 0;
        let (stale, _) = m2.staleness(now, None);
        assert!(!stale);
    }

    #[test]
    fn validation_is_fail_closed() {
        assert!(validate_curation("", "s", 30, &[], &[]).is_some());
        assert!(validate_curation("k", "", 30, &[], &[]).is_some());
        assert!(validate_curation("k", &"x".repeat(5000), 30, &[], &[]).is_some());
        assert!(validate_curation("k", "s", 0, &[], &[]).is_some());
        assert!(validate_curation("k", "s", 99999, &[], &[]).is_some());
        assert!(validate_curation("k", "s", 30, &["".to_string()], &[]).is_some());
        let many_sources: Vec<String> = (0..300).map(|i| format!("src-{i}")).collect();
        assert!(validate_curation("k", "s", 30, &[], &many_sources).is_some());
        assert!(validate_curation("k", "s", 30, &[], &[]).is_none());
        assert!(
            validate_curation("k", "s", 1, &["stack".to_string()], &["mem-1".to_string()])
                .is_none()
        );
    }

    #[test]
    fn context_prefix_marks_only_mental_models() {
        assert_eq!(context_prefix("k", "facts", false), None);
        assert_eq!(
            context_prefix("k", MENTAL_MODEL_CATEGORY, false),
            Some("[mental model: k]".to_string())
        );
        assert_eq!(
            context_prefix("k", MENTAL_MODEL_CATEGORY, true),
            Some("[mental model: k] (stale — pending operator review)".to_string())
        );
    }
}
