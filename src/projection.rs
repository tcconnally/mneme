//! #859: task-scoped projection surfaces — separate live context references,
//! durable recalled memory, and derived inference artifacts for ONE task.
//!
//! A projection is a compact, client-consumable artifact: instead of
//! stitching raw recall dumps, a client asks for a task projection and gets
//! three clearly labeled sections (live_references / durable_memories /
//! derived_inferences) plus a contract block where permission scope,
//! freshness, provenance, and trust class are all visible.
//!
//! Everything here is deterministic (given the same DB + anchor time, the
//! same projection is produced — #247). Freshness reuses the validity
//! scorer from #860; trust class comes from the entity's epistemic state
//! (#880); live/derived classification reads the reserved `origin` /
//! `external_refs` provenance keys stored in `body_json`
//! (memory-provenance-and-external-refs.md).

use crate::db::Database;
use crate::models::{Entity, RecallParams, SearchMode};
use crate::validity::{self, ValidityWeights};
use serde::Serialize;

// ─── Section classification ─────────────────────────────────────────────

/// The three projection sections (#859).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Section {
    /// Pointers to live external systems of record (first-class
    /// `external_refs`).
    Live,
    /// Recalled facts/observations that are durable memory.
    Durable,
    /// Derived inferences (inferred origin, `fact_derived` type, or
    /// `derived_from` citations).
    Derived,
}

impl Section {
    pub fn name(self) -> &'static str {
        match self {
            Section::Live => "live",
            Section::Derived => "derived",
            Section::Durable => "durable",
        }
    }
}

/// Parse the reserved provenance keys out of `body_json`. Absent or
/// unparseable bodies yield `(None, vec![])` — never guessed (#247, and the
/// provenance spec's "absent means unlabeled" rule).
pub fn parse_body_provenance(
    body_json: &str,
) -> (
    Option<crate::models::OriginRecord>,
    Vec<crate::models::ExternalRef>,
) {
    let Ok(value) = serde_json::from_str::<serde_json::Value>(body_json) else {
        return (None, Vec::new());
    };
    let origin = value
        .get("origin")
        .and_then(|v| serde_json::from_value(v.clone()).ok());
    let external_refs = value
        .get("external_refs")
        .and_then(|v| serde_json::from_value(v.clone()).ok())
        .unwrap_or_default();
    (origin, external_refs)
}

/// Deterministic three-way classification. Derived takes precedence over
/// live (an inference that cites live sources is still an inference);
/// anything without a derivation marker and without external refs is
/// durable memory.
pub fn classify(entity: &Entity) -> Section {
    let (origin, external_refs) = parse_body_provenance(&entity.body_json);
    let inferred_origin =
        origin.as_ref().and_then(|o| o.memory_kind.as_deref()) == Some("inferred");
    let derived_type = entity.entity_type == "fact_derived";
    let derived_ref = external_refs
        .iter()
        .any(|r| r.relationship.as_deref() == Some("derived_from"));
    if inferred_origin || derived_type || derived_ref {
        return Section::Derived;
    }
    if !external_refs.is_empty() {
        return Section::Live;
    }
    Section::Durable
}

/// Trust ranking over the epistemic axis (#880). Unclassified/unknown is
/// usable (rank 2, same as candidate); `defensively_recalled` is
/// explicitly untrusted; `rejected` is never projected.
pub fn trust_rank(state: &str) -> u8 {
    match state {
        "verified" => 4,
        "corroborated" => 3,
        "candidate" | "" => 2,
        "defensively_recalled" => 1,
        "rejected" => 0,
        _ => 2,
    }
}

/// Map a min_trust option to its rank floor. Fail-closed: unknown options
/// are rejected by validation before this is called.
pub fn min_trust_rank(option: &str) -> u8 {
    match option {
        "verified" => 4,
        "corroborated" => 3,
        _ => 2, // "candidate"
    }
}

// ─── Output contract ────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct FreshnessInfo {
    pub grade: String,
    pub value: f64,
    pub created_at_unix_ms: i64,
    pub age_days: f64,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ExternalRefBrief {
    pub ref_type: String,
    pub ref_value: String,
    pub relationship: Option<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ProvenanceInfo {
    pub memory_kind: Option<String>,
    pub source_system: Option<String>,
    pub capture_method: Option<String>,
    pub external_refs: Vec<ExternalRefBrief>,
    pub evidence_hash: Option<String>,
}

/// One compact projection item. Deliberately NOT a raw recall dump: a
/// summary, the section label, trust class, freshness, scope, and a
/// provenance digest — everything a consuming agent needs to decide
/// whether to trust and how current the item is.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ProjectionItem {
    pub id: String,
    pub key: String,
    pub section: String,
    pub summary: String,
    pub trust_class: String,
    pub freshness: FreshnessInfo,
    pub scope: String,
    pub provenance: ProvenanceInfo,
    /// "live_external" for live references (pointers into a live system of
    /// record), "memory_internal" for everything recalled from memory.
    pub source_of_truth_hint: String,
}

const SUMMARY_CAP: usize = 280;

/// Compact summary: the body's `note` when present, else the body itself,
/// truncated deterministically.
pub fn summarize(body_json: &str) -> String {
    let parsed = serde_json::from_str::<serde_json::Value>(body_json).ok();
    let note = parsed
        .as_ref()
        .and_then(|v| v.get("note"))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| body_json.to_string());
    truncate(&note, SUMMARY_CAP)
}

fn truncate(text: &str, cap: usize) -> String {
    if text.chars().count() <= cap {
        return text.to_string();
    }
    let mut out: String = text.chars().take(cap.saturating_sub(1)).collect();
    out.push('…');
    out
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct SectionCounts {
    pub live: usize,
    pub durable: usize,
    pub derived: usize,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ExcludedCounts {
    /// `rejected` epistemic state — never projected, regardless of min_trust.
    pub rejected: usize,
    /// Below the requested min_trust floor (incl. defensively_recalled).
    pub below_min_trust: usize,
    /// Older than the requested freshness window.
    pub outside_freshness_window: usize,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ProjectionContract {
    /// The three separated surfaces this projection always distinguishes.
    pub separates: Vec<String>,
    /// "workspace_scoped" when a workspace was requested, else "global".
    pub permission: String,
    /// The anchor instant for all freshness grades (query_time_unix_ms or
    /// generation time).
    pub freshness_anchor_unix_ms: i64,
    /// Distinct trust classes present in the projection.
    pub trust_classes: Vec<String>,
    pub counts: SectionCounts,
    pub excluded: ExcludedCounts,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ProjectionTrace {
    /// "task-projection-v1"
    pub method: String,
    /// Recall pool size before sectioning/filters.
    pub pool_size: usize,
    /// The recall modes the pool was built from.
    pub recall_modes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ProjectionReport {
    pub task: TaskBrief,
    pub generated_at_unix_ms: i64,
    pub scope: ProjectionScope,
    pub sections: ProjectionSections,
    pub contract: ProjectionContract,
    pub trace: ProjectionTrace,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct TaskBrief {
    pub title: String,
    /// Deterministic content hash of the projection inputs — stable across
    /// identical requests (replay, #247).
    pub projection_id: String,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ProjectionScope {
    pub workspace_hash: Option<String>,
    pub category: Option<String>,
    pub permission: String,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ProjectionSections {
    pub live_references: Vec<ProjectionItem>,
    pub durable_memories: Vec<ProjectionItem>,
    pub derived_inferences: Vec<ProjectionItem>,
}

// ─── Request + build ────────────────────────────────────────────────────

/// Validated projection request. Built from the MCP args by
/// `ProjectionRequest::parse` (fail-closed) — see tools.rs ProjectTaskArgs.
#[derive(Debug, Clone)]
pub struct ProjectionRequest {
    pub task_title: String,
    pub query: String,
    pub category: Option<String>,
    pub workspace_hash: Option<String>,
    pub limit: i64,
    pub freshness_window_days: Option<i64>,
    pub min_trust: String,
    pub include_sections: Vec<Section>,
    pub query_time_unix_ms: Option<i64>,
}

impl ProjectionRequest {
    pub fn parse(
        task_title: String,
        query: Option<String>,
        category: Option<String>,
        workspace_hash: Option<String>,
        limit: i64,
        freshness_window_days: Option<i64>,
        min_trust: String,
        include_sections: Vec<String>,
        query_time_unix_ms: Option<i64>,
    ) -> Result<Self, String> {
        let title = task_title.trim().to_string();
        if title.is_empty() {
            return Err("task_title must be a non-empty string".to_string());
        }
        if !(1..=100).contains(&limit) {
            return Err("limit must be between 1 and 100".to_string());
        }
        if let Some(days) = freshness_window_days {
            if days < 1 {
                return Err("freshness_window_days must be >= 1".to_string());
            }
        }
        if !matches!(
            min_trust.as_str(),
            "candidate" | "corroborated" | "verified"
        ) {
            return Err(format!(
                "invalid min_trust '{min_trust}': expected 'candidate', 'corroborated', or 'verified'"
            ));
        }
        let mut sections = Vec::new();
        for name in include_sections {
            let section = match name.as_str() {
                "live" => Section::Live,
                "durable" => Section::Durable,
                "derived" => Section::Derived,
                other => {
                    return Err(format!(
                        "invalid include_sections entry '{other}': expected 'live', 'durable', or 'derived'"
                    ))
                }
            };
            if !sections.contains(&section) {
                sections.push(section);
            }
        }
        if sections.is_empty() {
            sections = vec![Section::Live, Section::Durable, Section::Derived];
        }
        let resolved_query = query
            .map(|q| q.trim().to_string())
            .filter(|q| !q.is_empty())
            .unwrap_or_else(|| title.clone());
        Ok(ProjectionRequest {
            task_title: title,
            query: resolved_query,
            category,
            workspace_hash,
            limit,
            freshness_window_days,
            min_trust,
            include_sections: sections,
            query_time_unix_ms,
        })
    }
}

/// Build a task projection: fused recall (fts5 + temporal arms) into a
/// pool, deterministic sectioning + trust/freshness filtering, then the
/// compact report contract.
pub fn build_projection(
    db: &Database,
    req: &ProjectionRequest,
) -> Result<ProjectionReport, String> {
    let now_ms = req.query_time_unix_ms.unwrap_or_else(crate::db::now_ms);
    let weights = ValidityWeights::default();
    let pool_limit = req.limit.saturating_mul(3).min(300);

    let params = RecallParams {
        query: req.query.clone(),
        category: req.category.clone(),
        entity_type: None,
        type_filter: None,
        budget_profile: None,
        multihop: false,
        limit: pool_limit,
        offset: 0,
        min_decay: 0.0,
        topic_path: None,
        include_archived: false,
        skip_side_effects: true,
        mode: SearchMode::Fused,
        embedding: None,
        preview_cap: None,
        always_on: None,
        content_weight: 0.0,
        trust_weight: 0.0,
        max_prior_overturn: crate::models::default_max_prior_overturn(),
        diversity_halving: 1.0,
        diversity_per_query_share: 0.0,
        recency_half_life_secs: None,
        enforce_utility_horizon: true,
        workspace_hash: req.workspace_hash.clone(),
        scope_weight: None,
        agent_id: None,
        epistemic_state: None,
        visibility: None,
        layer: None,
        reinforce: false,
        strategies: vec!["fts5".to_string(), "temporal".to_string()],
        max_tokens: 0,
        depth_budget: None,
        strategy_weights: None,
        rerank: false,
        profile: None,
        validity_annotate: false,
        query_time_unix_ms: req.query_time_unix_ms,
        graph_utility_threshold: None,
        tier_order: false,
        declared_category: None,
        declared_filters: None,
        anchor_expansion: false,
    };
    let (entities, _count, _trace) = db
        .fused_recall(&params)
        .map_err(|e| format!("projection recall failed: {e}"))?;

    let min_rank = min_trust_rank(&req.min_trust);
    let window_ms = req
        .freshness_window_days
        .map(|d| d.saturating_mul(86_400_000));

    let mut live: Vec<ProjectionItem> = Vec::new();
    let mut durable: Vec<ProjectionItem> = Vec::new();
    let mut derived: Vec<ProjectionItem> = Vec::new();
    let mut rejected = 0usize;
    let mut below_min_trust = 0usize;
    let mut outside_freshness_window = 0usize;
    let mut trust_classes: Vec<String> = Vec::new();

    for entity in &entities {
        let section = classify(entity);
        if !req.include_sections.contains(&section) {
            continue;
        }
        let state = entity.epistemic_state.trim();
        let rank = trust_rank(state);
        if rank == 0 {
            rejected += 1;
            continue;
        }
        if rank < min_rank {
            below_min_trust += 1;
            continue;
        }
        let age_days = ((now_ms - entity.created_at_unix_ms).max(0) as f64) / 86_400_000.0;
        if let Some(win) = window_ms {
            if entity.created_at_unix_ms < now_ms.saturating_sub(win) {
                outside_freshness_window += 1;
                continue;
            }
        }
        let validity_info = validity::score(
            now_ms,
            entity.created_at_unix_ms,
            crate::db::entity_expiry_ms(&entity.body_json),
            &entity.workspace_hash,
            req.workspace_hash.as_deref(),
            &entity.epistemic_state,
            &entity.status,
            &weights,
        );
        let item = build_item(entity, section, now_ms, age_days, &validity_info);
        let trust_class = if state.is_empty() {
            "unclassified".to_string()
        } else {
            state.to_string()
        };
        if !trust_classes.contains(&trust_class) {
            trust_classes.push(trust_class);
        }
        match section {
            Section::Live => live.push(item),
            Section::Durable => durable.push(item),
            Section::Derived => derived.push(item),
        }
    }

    let sort_section = |items: &mut Vec<ProjectionItem>| {
        items.sort_by(|a, b| {
            b.freshness
                .value
                .partial_cmp(&a.freshness.value)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.id.cmp(&b.id))
        });
        items.truncate(req.limit as usize);
    };
    sort_section(&mut live);
    sort_section(&mut durable);
    sort_section(&mut derived);

    trust_classes.sort();
    let permission = if req.workspace_hash.as_deref().is_some_and(|w| !w.is_empty()) {
        "workspace_scoped".to_string()
    } else {
        "global".to_string()
    };

    let counts = SectionCounts {
        live: live.len(),
        durable: durable.len(),
        derived: derived.len(),
    };
    let projection_id = projection_id(req, now_ms);
    let report = ProjectionReport {
        task: TaskBrief {
            title: req.task_title.clone(),
            projection_id,
        },
        generated_at_unix_ms: now_ms,
        scope: ProjectionScope {
            workspace_hash: req.workspace_hash.clone(),
            category: req.category.clone(),
            permission: permission.clone(),
        },
        sections: ProjectionSections {
            live_references: live,
            durable_memories: durable,
            derived_inferences: derived,
        },
        contract: ProjectionContract {
            separates: vec![
                "live".to_string(),
                "durable".to_string(),
                "derived".to_string(),
            ],
            permission,
            freshness_anchor_unix_ms: now_ms,
            trust_classes,
            counts,
            excluded: ExcludedCounts {
                rejected,
                below_min_trust,
                outside_freshness_window,
            },
        },
        trace: ProjectionTrace {
            method: "task-projection-v1".to_string(),
            pool_size: entities.len(),
            recall_modes: vec![
                "fused".to_string(),
                "fts5".to_string(),
                "temporal".to_string(),
            ],
        },
    };
    Ok(report)
}

fn build_item(
    entity: &Entity,
    section: Section,
    now_ms: i64,
    age_days: f64,
    validity_info: &validity::ValidityInfo,
) -> ProjectionItem {
    let (origin, external_refs) = parse_body_provenance(&entity.body_json);
    let parsed_body = serde_json::from_str::<serde_json::Value>(&entity.body_json).ok();
    let evidence_hash = parsed_body
        .as_ref()
        .and_then(|v| v.get("evidence"))
        .and_then(|v| v.get("content_sha256"))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    ProjectionItem {
        id: entity.id.clone(),
        key: entity.key.clone(),
        section: section.name().to_string(),
        summary: summarize(&entity.body_json),
        trust_class: if entity.epistemic_state.trim().is_empty() {
            "unclassified".to_string()
        } else {
            entity.epistemic_state.clone()
        },
        freshness: FreshnessInfo {
            grade: validity_info.grade.clone(),
            value: validity_info.freshness,
            created_at_unix_ms: entity.created_at_unix_ms,
            age_days,
        },
        scope: validity_info.scope_match.clone(),
        provenance: ProvenanceInfo {
            memory_kind: origin.as_ref().and_then(|o| o.memory_kind.clone()),
            source_system: origin.as_ref().and_then(|o| o.source_system.clone()),
            capture_method: origin.as_ref().and_then(|o| o.capture_method.clone()),
            external_refs: external_refs
                .iter()
                .map(|r| ExternalRefBrief {
                    ref_type: r.ref_type.clone(),
                    ref_value: r.ref_value.clone(),
                    relationship: r.relationship.clone(),
                })
                .collect(),
            evidence_hash,
        },
        source_of_truth_hint: match section {
            Section::Live => "live_external".to_string(),
            _ => "memory_internal".to_string(),
        },
    }
}

/// Deterministic projection id over the request inputs (not the data — the
/// id identifies the projection request, so identical requests replay to
/// the same id per #247).
fn projection_id(req: &ProjectionRequest, now_ms: i64) -> String {
    let input = format!(
        "perseus-vault-task-projection-v1|{}|{}|{}|{}|{}|{}|{}|{}",
        req.workspace_hash.as_deref().unwrap_or(""),
        req.category.as_deref().unwrap_or(""),
        req.task_title,
        req.query,
        req.limit,
        req.freshness_window_days
            .map(|d| d.to_string())
            .unwrap_or_else(|| "all".to_string()),
        req.min_trust,
        now_ms,
    );
    let digest: String = {
        use sha2::{Digest, Sha256};
        format!("{:x}", Sha256::digest(input.as_bytes()))
    };
    digest.chars().take(16).collect()
}

// ─── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::Entity;

    fn entity_with_body(body: &str) -> Entity {
        Entity {
            id: "mem-test".to_string(),
            category: "c".to_string(),
            key: "k".to_string(),
            body_json: body.to_string(),
            status: "active".to_string(),
            entity_type: "insight".to_string(),
            tags: vec![],
            decay_score: 1.0,
            retrieval_count: 0,
            layer: "memory".to_string(),
            topic_path: String::new(),
            archived: false,
            archive_reason: String::new(),
            links: vec![],
            verified: false,
            source: String::new(),
            always_on: false,
            certainty: 0.5,
            workspace_hash: "ws".to_string(),
            agent_id: String::new(),
            visibility: "workspace".to_string(),
            created_at_unix_ms: 1_700_000_000_000,
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
    fn classify_live_via_external_refs() {
        let e = entity_with_body(
            r#"{"note": "incident 901", "external_refs": [{"ref_type": "jira_key", "ref_value": "PLT-901", "source_system": "jira", "relationship": "about"}]}"#,
        );
        assert_eq!(classify(&e), Section::Live);
    }

    #[test]
    fn classify_derived_via_inferred_origin() {
        let e = entity_with_body(
            r#"{"note": "deploy window", "origin": {"memory_kind": "inferred", "source_system": "quality-fixture", "capture_method": "rule_based_extractor"}}"#,
        );
        assert_eq!(classify(&e), Section::Derived);
    }

    #[test]
    fn classify_derived_via_fact_derived_type() {
        let mut e = entity_with_body(r#"{"note": "synthesized"}"#);
        e.entity_type = "fact_derived".to_string();
        assert_eq!(classify(&e), Section::Derived);
    }

    #[test]
    fn classify_derived_precedence_over_live() {
        // An inference that also cites live sources is still derived.
        let e = entity_with_body(
            r#"{"note": "x", "origin": {"memory_kind": "inferred"}, "external_refs": [{"ref_type": "url", "ref_value": "https://example.com"}]}"#,
        );
        assert_eq!(classify(&e), Section::Derived);
    }

    #[test]
    fn classify_durable_otherwise() {
        let e = entity_with_body(r#"{"note": "plain observed fact"}"#);
        assert_eq!(classify(&e), Section::Durable);
        // Malformed body → durable, never guessed.
        let e2 = entity_with_body("not json at all");
        assert_eq!(classify(&e2), Section::Durable);
    }

    #[test]
    fn trust_rank_ordering() {
        assert_eq!(trust_rank("verified"), 4);
        assert_eq!(trust_rank("corroborated"), 3);
        assert_eq!(trust_rank("candidate"), 2);
        assert_eq!(trust_rank(""), 2); // unclassified is usable
        assert_eq!(trust_rank("defensively_recalled"), 1);
        assert_eq!(trust_rank("rejected"), 0);
    }

    #[test]
    fn summarize_prefers_note_and_truncates() {
        assert_eq!(summarize(r#"{"note": "hello"}"#), "hello");
        let long = "x".repeat(400);
        let out = summarize(&format!(r#"{{"note": "{long}"}}"#));
        assert_eq!(out.chars().count(), SUMMARY_CAP);
        assert!(out.ends_with('…'));
        assert_eq!(summarize("raw body"), "raw body");
    }

    #[test]
    fn parse_rejects_invalid_requests_fail_closed() {
        assert!(ProjectionRequest::parse(
            "  ".to_string(),
            None,
            None,
            None,
            12,
            None,
            "candidate".to_string(),
            vec![],
            None,
        )
        .is_err());
        assert!(ProjectionRequest::parse(
            "t".to_string(),
            None,
            None,
            None,
            0,
            None,
            "candidate".to_string(),
            vec![],
            None,
        )
        .is_err());
        assert!(ProjectionRequest::parse(
            "t".to_string(),
            None,
            None,
            None,
            12,
            None,
            "bogus".to_string(),
            vec![],
            None,
        )
        .is_err());
        assert!(ProjectionRequest::parse(
            "t".to_string(),
            None,
            None,
            None,
            12,
            Some(0),
            "candidate".to_string(),
            vec![],
            None,
        )
        .is_err());
        assert!(ProjectionRequest::parse(
            "t".to_string(),
            None,
            None,
            None,
            12,
            None,
            "candidate".to_string(),
            vec!["nope".to_string()],
            None,
        )
        .is_err());
    }

    #[test]
    fn parse_defaults_and_query_fallback() {
        let req = ProjectionRequest::parse(
            "Deploy windows".to_string(),
            None,
            None,
            None,
            12,
            None,
            "candidate".to_string(),
            vec![],
            None,
        )
        .unwrap();
        assert_eq!(req.query, "Deploy windows"); // title fallback
        assert_eq!(req.include_sections.len(), 3);
        let req2 = ProjectionRequest::parse(
            "t".to_string(),
            Some("  ".to_string()),
            None,
            None,
            12,
            None,
            "candidate".to_string(),
            vec!["live".to_string(), "live".to_string()],
            None,
        )
        .unwrap();
        assert_eq!(req2.include_sections, vec![Section::Live]);
    }

    #[test]
    fn projection_id_is_deterministic() {
        let a = ProjectionRequest::parse(
            "task".to_string(),
            None,
            None,
            Some("ws".to_string()),
            5,
            None,
            "candidate".to_string(),
            vec![],
            Some(1_700_000_000_000),
        )
        .unwrap();
        let id1 = projection_id(&a, 1_700_000_000_000);
        let id2 = projection_id(&a, 1_700_000_000_000);
        assert_eq!(id1, id2);
        assert_eq!(id1.len(), 16);
    }
}
