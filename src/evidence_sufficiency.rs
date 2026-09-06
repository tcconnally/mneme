//! Answer-serving evidence sufficiency contract (#1183).
//!
//! This module separates requirement coverage from ordinary recall presence. A
//! task may require several evidence items, a latest replacement, a temporal
//! anchor, or one representative from each declared source group. The serving
//! path evaluates those requirements only after governed scope, visibility,
//! lifecycle, and temporal checks. Reports contain counts and digests, never
//! raw queries, prompts, memory bodies, or evidence identifiers.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

pub const EVIDENCE_SUFFICIENCY_SCHEMA_VERSION: &str = "perseus-vault-evidence-sufficiency/v1";
const MAX_REQUIRED_EVIDENCE: usize = 256;
const MAX_SOURCE_GROUPS: usize = 128;
const MAX_CONFLICTS: usize = 128;
const MAX_IDENTIFIER_CHARS: usize = 128;

fn default_schema_version() -> String {
    EVIDENCE_SUFFICIENCY_SCHEMA_VERSION.to_string()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SufficiencyOutcome {
    Complete,
    Partial,
    Degraded,
    Abstained,
    Unavailable,
}

impl SufficiencyOutcome {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Complete => "complete",
            Self::Partial => "partial",
            Self::Degraded => "degraded",
            Self::Abstained => "abstained",
            Self::Unavailable => "unavailable",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SufficiencyFallbackPolicy {
    Abstain,
    CanonicalRetrieval,
}

impl Default for SufficiencyFallbackPolicy {
    fn default() -> Self {
        Self::Abstain
    }
}

impl SufficiencyFallbackPolicy {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Abstain => "abstain",
            Self::CanonicalRetrieval => "canonical_retrieval",
        }
    }
}

/// A source-group coverage requirement declared by the task.
///
/// `evidence_ids` are opaque canonical IDs supplied by the caller. The group
/// label is a bounded task-local grouping key, not a source body or prompt.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceGroupRequirement {
    pub group_id: String,
    pub evidence_ids: Vec<String>,
}

/// An unresolved conflict declared by the task's evidence planner.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConflictRequirement {
    pub conflict_id: String,
    pub evidence_ids: Vec<String>,
}

/// Opt-in answer-serving requirements for one context request.
///
/// Omission of this object preserves the pre-existing context response. The
/// requirement lists are metadata only; the server still resolves every ID
/// through governed readers before counting it as available or selected.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceRequirementSet {
    #[serde(default = "default_schema_version")]
    pub schema_version: String,
    pub required_evidence: Vec<String>,
    #[serde(default)]
    pub latest_evidence: Vec<String>,
    #[serde(default)]
    pub temporal_anchors: Vec<String>,
    #[serde(default)]
    pub required_source_groups: Vec<SourceGroupRequirement>,
    #[serde(default)]
    pub conflicts: Vec<ConflictRequirement>,
    #[serde(default)]
    pub temporal_anchor_unix_ms: Option<i64>,
    #[serde(default)]
    pub fallback_policy: SufficiencyFallbackPolicy,
}

impl EvidenceRequirementSet {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != EVIDENCE_SUFFICIENCY_SCHEMA_VERSION {
            return Err(format!(
                "unsupported evidence sufficiency schema_version '{}'; expected {}",
                self.schema_version, EVIDENCE_SUFFICIENCY_SCHEMA_VERSION
            ));
        }
        if self.required_evidence.is_empty() {
            return Err("required_evidence must contain at least one evidence ID".to_string());
        }
        if self.required_evidence.len() > MAX_REQUIRED_EVIDENCE {
            return Err(format!(
                "required_evidence may contain at most {MAX_REQUIRED_EVIDENCE} entries"
            ));
        }
        let required = validate_unique_ids("required_evidence", &self.required_evidence)?;
        validate_subset("latest_evidence", &self.latest_evidence, &required)?;
        validate_subset("temporal_anchors", &self.temporal_anchors, &required)?;
        if let Some(anchor) = self.temporal_anchor_unix_ms {
            if anchor < 0 {
                return Err("temporal_anchor_unix_ms must be non-negative".to_string());
            }
        }
        if self.required_source_groups.len() > MAX_SOURCE_GROUPS {
            return Err(format!(
                "required_source_groups may contain at most {MAX_SOURCE_GROUPS} entries"
            ));
        }
        let mut group_ids = BTreeSet::new();
        for (index, group) in self.required_source_groups.iter().enumerate() {
            validate_identifier(
                &format!("required_source_groups[{index}].group_id"),
                &group.group_id,
            )?;
            if !group_ids.insert(&group.group_id) {
                return Err(format!(
                    "required_source_groups contains duplicate group_id '{}'",
                    group.group_id
                ));
            }
            if group.evidence_ids.is_empty() {
                return Err(format!(
                    "required_source_groups[{index}].evidence_ids must not be empty"
                ));
            }
            validate_subset(
                &format!("required_source_groups[{index}].evidence_ids"),
                &group.evidence_ids,
                &required,
            )?;
        }
        if self.conflicts.len() > MAX_CONFLICTS {
            return Err(format!(
                "conflicts may contain at most {MAX_CONFLICTS} entries"
            ));
        }
        let mut conflict_ids = BTreeSet::new();
        for (index, conflict) in self.conflicts.iter().enumerate() {
            validate_identifier(
                &format!("conflicts[{index}].conflict_id"),
                &conflict.conflict_id,
            )?;
            if !conflict_ids.insert(&conflict.conflict_id) {
                return Err(format!(
                    "conflicts contains duplicate conflict_id '{}'",
                    conflict.conflict_id
                ));
            }
            if conflict.evidence_ids.len() < 2 {
                return Err(format!(
                    "conflicts[{index}].evidence_ids must contain at least two IDs"
                ));
            }
            validate_subset(
                &format!("conflicts[{index}].evidence_ids"),
                &conflict.evidence_ids,
                &required,
            )?;
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<String, String> {
        self.validate()?;
        let value = serde_json::to_value(self)
            .map_err(|error| format!("evidence requirement serialization failed: {error}"))?;
        Ok(sha256_hex(canonical_json(&value).as_bytes()))
    }
}

/// Metadata about one required evidence ID after governed lookup.
#[derive(Debug, Clone)]
pub struct EvidenceCandidate {
    pub evidence_id: String,
    pub selected: bool,
    pub available: bool,
    pub stale: bool,
    pub temporal_valid: bool,
    pub budget_omitted: bool,
}

impl EvidenceCandidate {
    pub fn new(
        evidence_id: impl Into<String>,
        selected: bool,
        available: bool,
        stale: bool,
        temporal_valid: bool,
        budget_omitted: bool,
    ) -> Self {
        Self {
            evidence_id: evidence_id.into(),
            selected,
            available,
            stale,
            temporal_valid,
            budget_omitted,
        }
    }
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct CoverageCounts {
    pub required: usize,
    pub selected: usize,
    pub missing: usize,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct SufficiencyCounts {
    pub required: usize,
    pub selected: usize,
    pub omitted: usize,
    pub stale: usize,
    pub conflicting: usize,
    pub unavailable: usize,
    pub red_herring: usize,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct SufficiencyReasonCount {
    pub reason: String,
    pub count: usize,
}

/// Answer-visible conflict metadata. The conflict and reference identities are
/// commitments only; raw IDs never cross the answer-facing boundary.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct SufficiencyConflict {
    pub conflict_id_sha256: String,
    pub reference_count: usize,
    pub reference_digests: Vec<String>,
    pub references_sha256: String,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct SufficiencyReceipt {
    pub schema_version: String,
    pub query_sha256: String,
    pub requirement_sha256: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub candidate_set_sha256: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub selected_set_sha256: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub omitted_set_sha256: Option<String>,
    pub reasons: Vec<SufficiencyReasonCount>,
    pub digest: String,
}

impl SufficiencyReceipt {
    pub fn verify(&self) -> bool {
        is_sha256(&self.digest) && self.digest == self.compute_digest()
    }

    fn compute_digest(&self) -> String {
        let value = serde_json::json!({
            "schema_version": self.schema_version,
            "query_sha256": self.query_sha256,
            "requirement_sha256": self.requirement_sha256,
            "candidate_set_sha256": self.candidate_set_sha256,
            "selected_set_sha256": self.selected_set_sha256,
            "omitted_set_sha256": self.omitted_set_sha256,
            "reasons": self.reasons,
        });
        sha256_hex(canonical_json(&value).as_bytes())
    }
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct SufficiencyFallback {
    pub mode: String,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct EvidenceSufficiencyReport {
    pub schema_version: String,
    pub outcome: SufficiencyOutcome,
    pub counts: SufficiencyCounts,
    pub latest: CoverageCounts,
    pub temporal: CoverageCounts,
    pub source_groups: CoverageCounts,
    pub recall_status: String,
    pub fallback_policy: SufficiencyFallbackPolicy,
    pub reason_codes: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fallback: Option<SufficiencyFallback>,
    /// Reason-only metadata for evidence that could not be admitted.
    pub excluded: Vec<SufficiencyReasonCount>,
    /// Hash-only commitments to unresolved competing references.
    pub conflicts: Vec<SufficiencyConflict>,
    pub receipt: SufficiencyReceipt,
}

impl EvidenceSufficiencyReport {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != EVIDENCE_SUFFICIENCY_SCHEMA_VERSION {
            return Err("unsupported evidence sufficiency report schema_version".to_string());
        }
        if self.counts.selected > self.counts.required {
            return Err("sufficiency selected count exceeds required count".to_string());
        }
        if self.counts.omitted != self.counts.required.saturating_sub(self.counts.selected) {
            return Err("sufficiency omitted count does not match required coverage".to_string());
        }
        for coverage in [&self.latest, &self.temporal, &self.source_groups] {
            if coverage.selected > coverage.required
                || coverage.missing != coverage.required.saturating_sub(coverage.selected)
            {
                return Err("sufficiency coverage counts are inconsistent".to_string());
            }
        }
        if !self.receipt.verify() {
            return Err("sufficiency receipt digest is invalid".to_string());
        }
        for exclusion in &self.excluded {
            if exclusion.reason.is_empty() || exclusion.reason.len() > MAX_IDENTIFIER_CHARS {
                return Err("sufficiency exclusion reason is invalid".to_string());
            }
            if exclusion.count == 0 {
                return Err("sufficiency exclusion count must be positive".to_string());
            }
        }
        for conflict in &self.conflicts {
            if !is_sha256(&conflict.conflict_id_sha256)
                || !is_sha256(&conflict.references_sha256)
                || conflict.reference_count < 2
                || conflict.reference_digests.len() != conflict.reference_count
                || conflict
                    .reference_digests
                    .iter()
                    .any(|digest| !is_sha256(digest))
                || conflict.reason.is_empty()
            {
                return Err("sufficiency conflict metadata is invalid".to_string());
            }
        }
        Ok(())
    }
}

/// Evaluate an already-governed candidate set against a task requirement set.
///
/// The query is hashed into the receipt and is never retained in the report.
/// Candidate IDs are likewise represented only through set commitments in the
/// receipt. `EvidenceCandidate` is intentionally metadata-only.
pub fn evaluate(
    requirements: &EvidenceRequirementSet,
    candidates: &[EvidenceCandidate],
    recall_outcome: &crate::models::RecallOutcome,
    query: &str,
) -> Result<EvidenceSufficiencyReport, String> {
    requirements.validate()?;
    let mut by_id = BTreeMap::new();
    for candidate in candidates {
        validate_identifier("candidate evidence_id", &candidate.evidence_id)?;
        if by_id
            .insert(candidate.evidence_id.clone(), candidate)
            .is_some()
        {
            return Err(format!(
                "candidate evidence_id '{}' appears more than once",
                candidate.evidence_id
            ));
        }
    }

    let required: BTreeSet<&str> = requirements
        .required_evidence
        .iter()
        .map(String::as_str)
        .collect();
    let selected_for = |id: &str| {
        by_id
            .get(id)
            .is_some_and(|candidate| candidate.selected && candidate.available && !candidate.stale)
    };

    let selected_ids: Vec<String> = requirements
        .required_evidence
        .iter()
        .filter(|id| selected_for(id))
        .cloned()
        .collect();
    let omitted_ids: Vec<String> = requirements
        .required_evidence
        .iter()
        .filter(|id| !selected_for(id))
        .cloned()
        .collect();
    let stale = requirements
        .required_evidence
        .iter()
        .filter(|id| {
            by_id
                .get(id.as_str())
                .is_some_and(|candidate| candidate.stale)
        })
        .count();
    let unavailable = requirements
        .required_evidence
        .iter()
        .filter(|id| {
            by_id
                .get(id.as_str())
                .is_none_or(|candidate| !candidate.available)
        })
        .count();
    let budget_drops = requirements
        .required_evidence
        .iter()
        .filter(|id| {
            by_id.get(id.as_str()).is_some_and(|candidate| {
                candidate.budget_omitted
                    && candidate.available
                    && !candidate.stale
                    && !selected_for(id)
            })
        })
        .count();
    let coverage_drops = requirements
        .required_evidence
        .iter()
        .filter(|id| {
            by_id.get(id.as_str()).is_some_and(|candidate| {
                candidate.available
                    && !candidate.stale
                    && !candidate.budget_omitted
                    && !selected_for(id)
            })
        })
        .count();
    let red_herring = by_id
        .values()
        .filter(|candidate| {
            candidate.selected && !required.contains(candidate.evidence_id.as_str())
        })
        .count();

    let latest = coverage(&requirements.latest_evidence, |id| selected_for(id));
    let temporal = coverage(&requirements.temporal_anchors, |id| {
        by_id
            .get(id)
            .is_some_and(|candidate| selected_for(id) && candidate.temporal_valid)
    });
    let source_groups = coverage(
        &requirements
            .required_source_groups
            .iter()
            .map(|group| group.group_id.clone())
            .collect::<Vec<_>>(),
        |group_id| {
            requirements
                .required_source_groups
                .iter()
                .find(|group| group.group_id == group_id)
                .is_some_and(|group| group.evidence_ids.iter().any(|id| selected_for(id)))
        },
    );
    let conflicting = requirements
        .conflicts
        .iter()
        .filter(|conflict| conflict.evidence_ids.iter().any(|id| selected_for(id)))
        .count();

    let mut reasons = BTreeMap::new();
    if !omitted_ids.is_empty() {
        increment_reason(&mut reasons, "missing_required", omitted_ids.len());
    }
    if latest.missing > 0 {
        increment_reason(&mut reasons, "missing_latest_version", latest.missing);
    }
    if temporal.missing > 0 {
        if requirements.temporal_anchor_unix_ms.is_none() {
            increment_reason(&mut reasons, "missing_temporal_anchor", temporal.missing);
        } else {
            increment_reason(&mut reasons, "missing_temporal_evidence", temporal.missing);
        }
    }
    if source_groups.missing > 0 {
        increment_reason(&mut reasons, "missing_source_group", source_groups.missing);
    }
    if stale > 0 {
        increment_reason(&mut reasons, "stale_evidence", stale);
    }
    if unavailable > 0 {
        increment_reason(&mut reasons, "unavailable_evidence", unavailable);
    }
    if conflicting > 0 {
        increment_reason(&mut reasons, "conflicting_evidence", conflicting);
    }
    if budget_drops > 0 {
        increment_reason(&mut reasons, "dropped_budget", budget_drops);
    }
    if coverage_drops > 0 {
        increment_reason(&mut reasons, "dropped_coverage", coverage_drops);
    }
    if red_herring > 0 {
        increment_reason(&mut reasons, "red_herring_ignored", red_herring);
    }
    let recall_status = recall_outcome.status.as_str().to_string();
    match recall_outcome.status {
        crate::models::RecallStatus::Partial => increment_reason(&mut reasons, "recall_partial", 1),
        crate::models::RecallStatus::Timeout => increment_reason(&mut reasons, "recall_timeout", 1),
        crate::models::RecallStatus::Stale => increment_reason(&mut reasons, "recall_stale", 1),
        crate::models::RecallStatus::Unavailable => {
            increment_reason(&mut reasons, "recall_unavailable", 1)
        }
        crate::models::RecallStatus::Fresh | crate::models::RecallStatus::Empty => {}
    }

    let has_coverage_gap = omitted_ids.len() != 0
        || latest.missing != 0
        || temporal.missing != 0
        || source_groups.missing != 0;
    let outcome = if conflicting > 0 {
        SufficiencyOutcome::Abstained
    } else if unavailable > 0 || recall_outcome.status == crate::models::RecallStatus::Unavailable {
        SufficiencyOutcome::Unavailable
    } else if has_coverage_gap {
        match requirements.fallback_policy {
            SufficiencyFallbackPolicy::Abstain => SufficiencyOutcome::Abstained,
            SufficiencyFallbackPolicy::CanonicalRetrieval => SufficiencyOutcome::Partial,
        }
    } else if matches!(
        recall_outcome.status,
        crate::models::RecallStatus::Partial
            | crate::models::RecallStatus::Timeout
            | crate::models::RecallStatus::Stale
    ) {
        SufficiencyOutcome::Degraded
    } else {
        SufficiencyOutcome::Complete
    };

    let reason_codes: Vec<String> = reasons.keys().cloned().collect();
    let reason_records: Vec<SufficiencyReasonCount> = reasons
        .into_iter()
        .map(|(reason, count)| SufficiencyReasonCount { reason, count })
        .collect();
    let excluded = reason_records.clone();
    let conflicts: Vec<SufficiencyConflict> = requirements
        .conflicts
        .iter()
        .filter(|conflict| conflict.evidence_ids.iter().any(|id| selected_for(id)))
        .map(conflict_summary)
        .collect();
    let fallback = (outcome != SufficiencyOutcome::Complete).then(|| SufficiencyFallback {
        mode: requirements.fallback_policy.as_str().to_string(),
        reason: if reason_codes.is_empty() {
            outcome.as_str().to_string()
        } else {
            reason_codes.join(",")
        },
    });
    let all_candidate_ids: Vec<String> = by_id.keys().cloned().collect();
    let receipt = make_receipt(
        query,
        requirements,
        &all_candidate_ids,
        &selected_ids,
        &omitted_ids,
        reason_records,
    )?;
    let report = EvidenceSufficiencyReport {
        schema_version: EVIDENCE_SUFFICIENCY_SCHEMA_VERSION.to_string(),
        outcome,
        counts: SufficiencyCounts {
            required: requirements.required_evidence.len(),
            selected: selected_ids.len(),
            omitted: omitted_ids.len(),
            stale,
            conflicting,
            unavailable,
            red_herring,
        },
        latest,
        temporal,
        source_groups,
        recall_status,
        fallback_policy: requirements.fallback_policy,
        reason_codes,
        fallback,
        excluded,
        conflicts,
        receipt,
    };
    report.validate()?;
    Ok(report)
}

/// Evaluate task requirements against the live governed entity store.
///
/// This boundary is deliberately separate from ordinary context selection. It
/// resolves every declared requirement through the public requester gate, then
/// asks the pure evaluator to classify coverage. Inaccessible IDs become an
/// unavailable count; their values are never returned.
pub fn evaluate_context_requirements(
    db: &crate::db::Database,
    requirements: &EvidenceRequirementSet,
    candidate_ids: &BTreeSet<String>,
    delivered_ids: &BTreeSet<String>,
    query: &str,
    workspace_hash: Option<&str>,
    requesting_agent_id: Option<&str>,
    recall_outcome: &crate::models::RecallOutcome,
) -> Result<EvidenceSufficiencyReport, String> {
    requirements.validate()?;
    let now = crate::db::now_ms();
    let mut candidates = Vec::with_capacity(requirements.required_evidence.len());
    for evidence_id in &requirements.required_evidence {
        let raw = db
            .get_entity_by_id_unfiltered(evidence_id)
            .map_err(|error| format!("sufficiency evidence lookup failed: {error}"))?;
        let mut available = false;
        let mut stale = false;
        let mut temporal_valid = false;
        if let Some(entity) = raw {
            let in_scope = scope_allows(&entity.workspace_hash, workspace_hash);
            let requester_allowed =
                db.requester_can_read(requesting_agent_id, &entity.visibility, &entity.agent_id);
            let terminal_stale = matches!(entity.status.as_str(), "deprecated" | "expired");
            let publicly_readable = db
                .get_entity_by_id_for_requester(evidence_id, requesting_agent_id)
                .map_err(|error| format!("sufficiency visibility lookup failed: {error}"))?
                .is_some();
            if in_scope && requester_allowed && (publicly_readable || terminal_stale) {
                match entity.status.as_str() {
                    "redacted" | "quarantined" | "compacted" => {}
                    _ if entity.archived => {}
                    "deprecated" | "expired" => {
                        available = true;
                        stale = true;
                    }
                    _ => {
                        available = true;
                        stale = crate::db::entity_expiry_ms(&entity.body_json)
                            .is_some_and(|expires| expires <= now);
                    }
                }
                if let Some(anchor) = requirements.temporal_anchor_unix_ms {
                    let mut historical = vec![entity.clone()];
                    db.resolve_temporal_versions(&mut historical, Some(anchor), None)
                        .map_err(|error| format!("sufficiency temporal lookup failed: {error}"))?;
                    if let Some(at_anchor) = historical.into_iter().next() {
                        temporal_valid = scope_allows(&at_anchor.workspace_hash, workspace_hash)
                            && db.requester_can_read(
                                requesting_agent_id,
                                &at_anchor.visibility,
                                &at_anchor.agent_id,
                            )
                            && !at_anchor.archived
                            && !matches!(
                                at_anchor.status.as_str(),
                                "deprecated" | "expired" | "redacted" | "quarantined" | "compacted"
                            )
                            && !crate::db::entity_expiry_ms(&at_anchor.body_json)
                                .is_some_and(|expires| expires <= anchor);
                    }
                }
            }
        }
        let selected = available && !stale && delivered_ids.contains(evidence_id);
        let budget_omitted = available
            && !stale
            && candidate_ids.contains(evidence_id)
            && !delivered_ids.contains(evidence_id);
        candidates.push(EvidenceCandidate::new(
            evidence_id,
            selected,
            available,
            stale,
            temporal_valid,
            budget_omitted,
        ));
    }
    let required_ids: BTreeSet<&str> = requirements
        .required_evidence
        .iter()
        .map(String::as_str)
        .collect();
    for evidence_id in candidate_ids {
        if !required_ids.contains(evidence_id.as_str()) {
            candidates.push(EvidenceCandidate::new(
                evidence_id.clone(),
                delivered_ids.contains(evidence_id.as_str()),
                true,
                false,
                false,
                false,
            ));
        }
    }
    evaluate(requirements, &candidates, recall_outcome, query)
}

fn coverage<F>(ids: &[String], mut selected: F) -> CoverageCounts
where
    F: FnMut(&str) -> bool,
{
    let selected_count = ids.iter().filter(|id| selected(id)).count();
    CoverageCounts {
        required: ids.len(),
        selected: selected_count,
        missing: ids.len().saturating_sub(selected_count),
    }
}

fn make_receipt(
    query: &str,
    requirements: &EvidenceRequirementSet,
    candidate_ids: &[String],
    selected_ids: &[String],
    omitted_ids: &[String],
    reasons: Vec<SufficiencyReasonCount>,
) -> Result<SufficiencyReceipt, String> {
    let mut candidate_ids = candidate_ids.to_vec();
    let mut selected_ids = selected_ids.to_vec();
    let mut omitted_ids = omitted_ids.to_vec();
    candidate_ids.sort();
    selected_ids.sort();
    omitted_ids.sort();
    let mut receipt = SufficiencyReceipt {
        schema_version: EVIDENCE_SUFFICIENCY_SCHEMA_VERSION.to_string(),
        query_sha256: sha256_hex(query.as_bytes()),
        requirement_sha256: requirements.digest()?,
        candidate_set_sha256: hash_optional_set(&candidate_ids),
        selected_set_sha256: hash_optional_set(&selected_ids),
        omitted_set_sha256: hash_optional_set(&omitted_ids),
        reasons,
        digest: String::new(),
    };
    receipt.digest = receipt.compute_digest();
    Ok(receipt)
}

fn conflict_summary(conflict: &ConflictRequirement) -> SufficiencyConflict {
    let mut references = conflict.evidence_ids.clone();
    references.sort();
    references.dedup();
    let reference_digests: Vec<String> = references
        .iter()
        .map(|reference| sha256_hex(reference.as_bytes()))
        .collect();
    SufficiencyConflict {
        conflict_id_sha256: sha256_hex(conflict.conflict_id.as_bytes()),
        reference_count: references.len(),
        reference_digests,
        references_sha256: sha256_hex(canonical_json(&serde_json::json!(references)).as_bytes()),
        reason: "unresolved_conflict".to_string(),
    }
}

fn hash_optional_set(values: &[String]) -> Option<String> {
    if values.is_empty() {
        None
    } else {
        Some(sha256_hex(
            canonical_json(&serde_json::json!(values)).as_bytes(),
        ))
    }
}

fn increment_reason(reasons: &mut BTreeMap<String, usize>, reason: &str, count: usize) {
    let entry = reasons.entry(reason.to_string()).or_default();
    *entry = entry.saturating_add(count);
}

fn validate_unique_ids(label: &str, values: &[String]) -> Result<BTreeSet<String>, String> {
    let mut ids = BTreeSet::new();
    for (index, value) in values.iter().enumerate() {
        validate_identifier(&format!("{label}[{index}]"), value)?;
        if !ids.insert(value.clone()) {
            return Err(format!("{label} contains duplicate ID '{value}'"));
        }
    }
    Ok(ids)
}

fn validate_subset(
    label: &str,
    values: &[String],
    parent: &BTreeSet<String>,
) -> Result<(), String> {
    let ids = validate_unique_ids(label, values)?;
    if let Some(value) = ids.iter().find(|value| !parent.contains(*value)) {
        return Err(format!(
            "{label} contains ID outside required_evidence: '{value}'"
        ));
    }
    Ok(())
}

fn validate_identifier(label: &str, value: &str) -> Result<(), String> {
    let valid = !value.is_empty()
        && value.len() <= MAX_IDENTIFIER_CHARS
        && value.bytes().enumerate().all(|(index, byte)| {
            (index == 0 && byte.is_ascii_alphanumeric())
                || (index > 0
                    && (byte.is_ascii_alphanumeric()
                        || matches!(byte, b'.' | b'_' | b':' | b'/' | b'-')))
        });
    if valid {
        Ok(())
    } else {
        Err(format!("{label} must be a bounded opaque identifier"))
    }
}

fn scope_allows(source_workspace: &str, requested_workspace: Option<&str>) -> bool {
    match requested_workspace {
        None => true,
        Some("") => source_workspace.is_empty(),
        Some(workspace) => source_workspace.is_empty() || source_workspace == workspace,
    }
}

fn canonical_json(value: &serde_json::Value) -> String {
    fn sort(value: &serde_json::Value) -> serde_json::Value {
        match value {
            serde_json::Value::Object(map) => {
                let mut entries: Vec<(String, serde_json::Value)> = map
                    .iter()
                    .map(|(key, value)| (key.clone(), sort(value)))
                    .collect();
                entries.sort_by(|left, right| left.0.cmp(&right.0));
                serde_json::Value::Object(entries.into_iter().collect())
            }
            serde_json::Value::Array(values) => {
                serde_json::Value::Array(values.iter().map(sort).collect())
            }
            other => other.clone(),
        }
    }
    serde_json::to_string(&sort(value)).unwrap_or_default()
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value.bytes().all(|byte| byte.is_ascii_hexdigit())
        && value == value.to_ascii_lowercase()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn requirement_set() -> EvidenceRequirementSet {
        EvidenceRequirementSet {
            schema_version: EVIDENCE_SUFFICIENCY_SCHEMA_VERSION.to_string(),
            required_evidence: vec!["evidence-a".to_string(), "evidence-b".to_string()],
            latest_evidence: vec!["evidence-b".to_string()],
            temporal_anchors: vec!["evidence-a".to_string(), "evidence-b".to_string()],
            required_source_groups: vec![SourceGroupRequirement {
                group_id: "group-bridge".to_string(),
                evidence_ids: vec!["evidence-a".to_string(), "evidence-b".to_string()],
            }],
            conflicts: Vec::new(),
            temporal_anchor_unix_ms: Some(100),
            fallback_policy: SufficiencyFallbackPolicy::Abstain,
        }
    }

    fn candidate(
        id: &str,
        selected: bool,
        available: bool,
        stale: bool,
        temporal_valid: bool,
        budget_omitted: bool,
    ) -> EvidenceCandidate {
        EvidenceCandidate::new(
            id,
            selected,
            available,
            stale,
            temporal_valid,
            budget_omitted,
        )
    }

    fn fresh_recall() -> crate::models::RecallOutcome {
        crate::models::RecallOutcome {
            status: crate::models::RecallStatus::Fresh,
            ..Default::default()
        }
    }

    #[test]
    fn missing_bridge_and_budget_are_explicit_and_receipt_is_hash_only() {
        let requirements = requirement_set();
        let report = evaluate(
            &requirements,
            &[
                candidate("evidence-a", true, true, false, true, false),
                candidate("evidence-b", false, true, false, false, true),
            ],
            &fresh_recall(),
            "raw query must not enter the receipt",
        )
        .unwrap();
        assert_eq!(report.outcome, SufficiencyOutcome::Abstained);
        assert_eq!(report.counts.required, 2);
        assert_eq!(report.counts.selected, 1);
        assert_eq!(report.counts.omitted, 1);
        assert_eq!(report.latest.missing, 1);
        assert_eq!(report.temporal.missing, 1);
        assert_eq!(report.source_groups.missing, 0);
        assert!(report.reason_codes.contains(&"dropped_budget".to_string()));
        let encoded = serde_json::to_string(&report).unwrap();
        assert!(!encoded.contains("raw query must not enter the receipt"));
        assert!(!encoded.contains("evidence-a"));
        assert!(report.receipt.verify());
        assert!(report.validate().is_ok());
    }

    #[test]
    fn canonical_retrieval_fallback_is_partial_not_silent_narrowing() {
        let mut requirements = requirement_set();
        requirements.fallback_policy = SufficiencyFallbackPolicy::CanonicalRetrieval;
        let report = evaluate(
            &requirements,
            &[
                candidate("evidence-a", true, true, false, true, false),
                candidate("evidence-b", false, true, false, false, true),
            ],
            &fresh_recall(),
            "query",
        )
        .unwrap();
        assert_eq!(report.outcome, SufficiencyOutcome::Partial);
        assert_eq!(
            report.fallback.as_ref().unwrap().mode,
            "canonical_retrieval"
        );
    }

    #[test]
    fn stale_conflict_and_unavailable_states_win_over_presence() {
        let mut conflict = requirement_set();
        conflict.conflicts = vec![ConflictRequirement {
            conflict_id: "conflict-a".to_string(),
            evidence_ids: vec!["evidence-a".to_string(), "evidence-b".to_string()],
        }];
        let report = evaluate(
            &conflict,
            &[
                candidate("evidence-a", true, true, false, true, false),
                candidate("evidence-b", true, true, true, false, false),
            ],
            &fresh_recall(),
            "query",
        )
        .unwrap();
        assert_eq!(report.outcome, SufficiencyOutcome::Abstained);
        assert_eq!(report.counts.conflicting, 1);
        assert_eq!(report.counts.stale, 1);

        let unavailable = evaluate(
            &requirement_set(),
            &[
                candidate("evidence-a", false, false, false, false, false),
                candidate("evidence-b", false, true, false, false, false),
            ],
            &fresh_recall(),
            "query",
        )
        .unwrap();
        assert_eq!(unavailable.outcome, SufficiencyOutcome::Unavailable);
        assert_eq!(unavailable.counts.unavailable, 1);
    }

    #[test]
    fn stale_required_evidence_never_counts_as_complete() {
        let mut requirements = requirement_set();
        requirements.required_evidence = vec!["evidence-a".to_string()];
        requirements.latest_evidence.clear();
        requirements.temporal_anchors.clear();
        requirements.required_source_groups.clear();
        let report = evaluate(
            &requirements,
            &[candidate("evidence-a", true, true, true, false, false)],
            &fresh_recall(),
            "query",
        )
        .unwrap();
        assert_eq!(report.outcome, SufficiencyOutcome::Abstained);
        assert_eq!(report.counts.stale, 1);
    }

    #[test]
    fn recall_degradation_is_separate_from_complete_requirement_coverage() {
        let mut requirements = requirement_set();
        requirements.latest_evidence.clear();
        requirements.temporal_anchors.clear();
        requirements.required_source_groups.clear();
        let recall = crate::models::RecallOutcome {
            status: crate::models::RecallStatus::Partial,
            ..Default::default()
        };
        let report = evaluate(
            &requirements,
            &[
                candidate("evidence-a", true, true, false, false, false),
                candidate("evidence-b", true, true, false, false, false),
                candidate("red-herring", true, true, false, false, false),
            ],
            &recall,
            "query",
        )
        .unwrap();
        assert_eq!(report.outcome, SufficiencyOutcome::Degraded);
        assert_eq!(report.counts.selected, 2);
        assert_eq!(report.counts.red_herring, 1);
    }

    #[test]
    fn malformed_requirement_sets_and_raw_fields_fail_closed() {
        let mut duplicate = requirement_set();
        duplicate.required_evidence.push("evidence-a".to_string());
        assert!(duplicate.validate().is_err());
        let mut unknown = serde_json::to_value(requirement_set()).unwrap();
        unknown["raw_prompt"] = json!("forbidden");
        assert!(serde_json::from_value::<EvidenceRequirementSet>(unknown).is_err());
        let mut outside = requirement_set();
        outside.latest_evidence = vec!["not-required".to_string()];
        assert!(outside.validate().is_err());
    }

    #[test]
    fn excluded_evidence_and_conflicting_references_are_answer_visible() {
        let mut requirements = requirement_set();
        requirements
            .required_evidence
            .push("evidence-missing".to_string());
        requirements.conflicts = vec![ConflictRequirement {
            conflict_id: "conflict-deployment".to_string(),
            evidence_ids: vec!["evidence-a".to_string(), "evidence-b".to_string()],
        }];
        let report = evaluate(
            &requirements,
            &[
                candidate("evidence-a", true, true, false, true, false),
                candidate("evidence-b", true, true, false, true, false),
            ],
            &fresh_recall(),
            "query",
        )
        .unwrap();
        let value = serde_json::to_value(&report).unwrap();
        assert_eq!(value["outcome"], "abstained");
        assert_eq!(value["counts"]["conflicting"], 1);
        assert!(value["excluded"].is_array(), "{value}");
        assert!(value["excluded"]
            .as_array()
            .unwrap()
            .iter()
            .any(|entry| entry["reason"] == "unavailable_evidence" && entry["count"] == 1));
        assert_eq!(value["conflicts"][0]["reference_count"], 2, "{value}");
        assert_eq!(
            value["conflicts"][0]["references_sha256"]
                .as_str()
                .unwrap()
                .len(),
            64,
            "{value}"
        );
        let serialized = value.to_string();
        assert!(
            !serialized.contains("evidence-a"),
            "raw conflict IDs leaked: {value}"
        );
        assert!(
            !serialized.contains("evidence-b"),
            "raw conflict IDs leaked: {value}"
        );
    }
}
