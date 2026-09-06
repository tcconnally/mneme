//! Answer-facing safe outcome contracts for retrieval and context assembly (#1186).
//!
//! The retrieval layer has a richer health vocabulary (`fresh`, `empty`,
//! `stale`, ...), while an answerer needs a small closed set of safe decisions.
//! This module performs that translation without treating an empty or invalid
//! projection as a successful answer.  Metadata is deliberately reason-only:
//! evidence identifiers and bodies are represented by bounded counts or
//! commitments, never copied into the answer outcome.

use crate::evidence_lanes::{EvidenceProjection, ExclusionRecord};
use crate::evidence_sufficiency::EvidenceSufficiencyReport;
use crate::models::{RecallOutcome, RecallStatus};
use crate::task_state::{ActiveConflict, TaskFallback, TaskStateOutcome};
use serde::Serialize;
use sha2::{Digest, Sha256};

pub const SAFE_OUTCOME_SCHEMA_VERSION: &str = "perseus-vault-answer-outcome/v1";
pub const SAFE_OUTCOME_STATUSES: [&str; 5] = [
    "complete",
    "partial",
    "degraded",
    "abstained",
    "unavailable",
];

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct SafeFallback {
    pub mode: String,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ReasonOnly {
    pub reason: String,
    pub count: usize,
}

/// A conflict summary carries a commitment to the competing references rather
/// than their IDs.  This lets an answer-facing consumer detect that values
/// were not blended while avoiding an existence/identifier leak.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ConflictSummary {
    pub reason: String,
    pub reference_count: usize,
    pub references_sha256: String,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct AnswerOutcome {
    pub schema_version: String,
    pub status: String,
    /// The lower-level recall status is retained so `empty` remains distinct
    /// from `unavailable`, even though an empty answer is abstained.
    pub recall_status: String,
    pub reason: String,
    pub reason_codes: Vec<String>,
    pub abstained: bool,
    pub answerable: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fallback: Option<SafeFallback>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub exclusions: Vec<ReasonOnly>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub conflicts: Vec<ConflictSummary>,
}

pub fn object(
    status: &str,
    recall_status: &str,
    reason: &str,
    answerable: bool,
) -> serde_json::Value {
    let status = if SAFE_OUTCOME_STATUSES.contains(&status) {
        status
    } else {
        "unavailable"
    };
    let abstained = matches!(status, "abstained" | "unavailable") || !answerable;
    let answerable = answerable && !abstained;
    let reason = safe_reason(reason);
    let mut outcome = serde_json::json!({
        "schema_version": SAFE_OUTCOME_SCHEMA_VERSION,
        "status": status,
        "recall_status": safe_recall_status(recall_status),
        "reason": reason.clone(),
        "reason_codes": [reason.clone()],
        "abstained": abstained,
        "answerable": answerable,
    });
    if status != "complete" {
        outcome["fallback"] = serde_json::json!({
            "mode": "canonical_retrieval",
            "reason": reason,
        });
    }
    outcome
}

impl AnswerOutcome {
    fn new(
        status: &str,
        recall_status: &str,
        reason: impl Into<String>,
        abstained: bool,
        answerable: bool,
        fallback: Option<SafeFallback>,
    ) -> Self {
        debug_assert!(SAFE_OUTCOME_STATUSES.contains(&status));
        let reason = safe_reason(&reason.into());
        let fallback = fallback.map(|value| SafeFallback {
            mode: safe_fallback_mode(&value.mode),
            reason: safe_reason(&value.reason),
        });
        Self {
            schema_version: SAFE_OUTCOME_SCHEMA_VERSION.to_string(),
            status: status.to_string(),
            recall_status: safe_recall_status(recall_status),
            reason_codes: vec![reason.clone()],
            reason,
            abstained,
            answerable,
            fallback,
            exclusions: Vec::new(),
            conflicts: Vec::new(),
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != SAFE_OUTCOME_SCHEMA_VERSION {
            return Err("unsupported answer outcome schema_version".to_string());
        }
        if !SAFE_OUTCOME_STATUSES.contains(&self.status.as_str()) {
            return Err(format!(
                "unsupported answer outcome status '{}':",
                self.status
            ));
        }
        if self.reason.is_empty() || self.reason.len() > 256 {
            return Err("answer outcome reason must be bounded and non-empty".to_string());
        }
        if self.reason_codes.is_empty()
            || self.reason_codes.len() > 16
            || self.reason_codes.iter().any(|reason| {
                reason.is_empty() || reason.len() > 256 || reason != &safe_reason(reason)
            })
            || !self.reason_codes.contains(&self.reason)
        {
            return Err("answer outcome reason_codes are invalid".to_string());
        }
        if self.status == "complete" && (!self.answerable || self.abstained) {
            return Err("complete answer outcome must be answerable and non-abstained".to_string());
        }
        if matches!(self.status.as_str(), "abstained" | "unavailable")
            && (!self.abstained || self.answerable)
        {
            return Err("abstained/unavailable outcome has inconsistent answerability".to_string());
        }
        if self.status != "complete" && self.fallback.is_none() {
            return Err("incomplete answer outcome requires a safe fallback".to_string());
        }
        Ok(())
    }
}

/// A context answer is complete only when matching evidence was actually
/// delivered and the rendered block was not truncated. `recall_status`
/// deliberately remains `empty` for no-match/unspecified-query cases so
/// callers can distinguish those from a backend fault.
pub fn for_context(query: Option<&str>, injected: i64, truncated: bool) -> AnswerOutcome {
    let has_query = query.is_some_and(|value| !value.trim().is_empty());
    if has_query && injected > 0 && !truncated {
        return AnswerOutcome::new("complete", "fresh", "context assembled", false, true, None);
    }
    if has_query && injected > 0 && truncated {
        return AnswerOutcome::new(
            "partial",
            "partial",
            "context_truncated",
            false,
            true,
            Some(SafeFallback {
                mode: "canonical_retrieval".to_string(),
                reason: "context_truncated".to_string(),
            }),
        );
    }
    let reason = if has_query { "no_match" } else { "no_query" };
    AnswerOutcome::new(
        "abstained",
        "empty",
        reason,
        true,
        false,
        Some(SafeFallback {
            mode: "canonical_retrieval".to_string(),
            reason: reason.to_string(),
        }),
    )
}

/// Preserve an explicit evidence-sufficiency decision at the answer boundary.
/// A canonical-retrieval policy is therefore reported as `partial`, while an
/// abstention policy remains an abstention; neither is inferred from the
/// rendered entity count.
pub fn for_sufficiency(report: &EvidenceSufficiencyReport) -> AnswerOutcome {
    let status = report.outcome.as_str();
    let reason = report
        .fallback
        .as_ref()
        .map(|fallback| fallback.reason.as_str())
        .or_else(|| report.reason_codes.first().map(String::as_str))
        .unwrap_or(status);
    let fallback = report.fallback.as_ref().map(|fallback| SafeFallback {
        mode: fallback.mode.clone(),
        reason: safe_reason(&fallback.reason),
    });
    let mut answer = AnswerOutcome::new(
        status,
        &report.recall_status,
        reason,
        matches!(status, "abstained" | "unavailable"),
        status == "complete",
        fallback,
    );
    answer.exclusions = report
        .excluded
        .iter()
        .map(|entry| ReasonOnly {
            reason: safe_reason(&entry.reason),
            count: entry.count,
        })
        .collect();
    answer.conflicts = report
        .conflicts
        .iter()
        .map(|conflict| ConflictSummary {
            reason: safe_reason(&conflict.reason),
            reference_count: conflict.reference_count,
            references_sha256: conflict.references_sha256.clone(),
        })
        .collect();
    answer.reason_codes = merge_reason_codes(
        &answer.reason,
        report
            .reason_codes
            .iter()
            .map(String::as_str)
            .chain(answer.exclusions.iter().map(|entry| entry.reason.as_str())),
    );
    answer
}

/// Apply the serving-layer delivery boundary to an answer outcome. A complete
/// evidence decision is only answerable when the complete rendered context was
/// delivered; explicit truncation therefore downgrades it to a partial answer.
/// Existing partial, abstained, and unavailable outcomes are left unchanged.
pub fn with_context_delivery(mut base: AnswerOutcome, truncated: bool) -> AnswerOutcome {
    if truncated && base.status == "complete" {
        base.status = "partial".to_string();
        base.reason = "context_truncated".to_string();
        base.reason_codes = vec![base.reason.clone()];
        base.abstained = false;
        base.answerable = true;
        base.fallback = Some(SafeFallback {
            mode: "canonical_retrieval".to_string(),
            reason: base.reason.clone(),
        });
    }
    base
}

/// Convert the existing recall health result into the closed answer-facing
/// status set. `hits` is the post-governance count that can actually support an
/// answer; it is not inferred from a backend estimate.
pub fn for_recall(recall: &RecallOutcome, hits: usize) -> AnswerOutcome {
    let recall_status = recall.status.as_str();
    let reason = if recall.reason.is_empty() {
        default_recall_reason(&recall.status, hits)
    } else {
        safe_reason(&recall.reason)
    };
    let (status, abstained, answerable) = if recall.abstained {
        // The lower-level contract's abstention bit is authoritative.  Do not
        // let an inconsistent status/hit count turn a withheld result into a
        // complete answer.
        if matches!(recall.status, RecallStatus::Unavailable) {
            ("unavailable", true, false)
        } else {
            ("abstained", true, false)
        }
    } else {
        match recall.status {
            RecallStatus::Fresh if hits > 0 => ("complete", false, true),
            RecallStatus::Fresh | RecallStatus::Empty => ("abstained", true, false),
            RecallStatus::Partial | RecallStatus::Timeout if hits > 0 => ("partial", false, true),
            RecallStatus::Partial | RecallStatus::Timeout => ("degraded", true, false),
            RecallStatus::Stale if hits > 0 => ("degraded", false, true),
            RecallStatus::Stale => ("degraded", true, false),
            RecallStatus::Unavailable => ("unavailable", true, false),
        }
    };
    let fallback = (status != "complete").then(|| SafeFallback {
        mode: "canonical_retrieval".to_string(),
        reason: reason.clone(),
    });
    AnswerOutcome::new(
        status,
        recall_status,
        reason,
        abstained,
        answerable,
        fallback,
    )
}

/// Map a successful task-state serving response into the shared answer outcome
/// vocabulary. Task-state evidence lists stay in their existing response
/// surface; this summary only carries reason/count and hash-only conflict data.
pub fn for_task(
    outcome: &TaskStateOutcome,
    fallback: Option<&TaskFallback>,
    accepted: usize,
    rejected: usize,
    unresolved: usize,
    missing: usize,
    conflicts: &[ActiveConflict],
) -> AnswerOutcome {
    let status = outcome.as_str();
    let recall_status = match outcome {
        TaskStateOutcome::Complete => "fresh",
        TaskStateOutcome::Partial => "partial",
        TaskStateOutcome::Degraded => "partial",
        TaskStateOutcome::Abstained => "empty",
        TaskStateOutcome::Unavailable => "unavailable",
    };
    let reason = match outcome {
        TaskStateOutcome::Complete => "task_state_complete",
        TaskStateOutcome::Partial => "task_state_partial",
        TaskStateOutcome::Degraded => "task_state_degraded",
        TaskStateOutcome::Abstained if !conflicts.is_empty() => "unresolved_conflict",
        TaskStateOutcome::Abstained if unresolved > 0 || missing > 0 => "missing_evidence",
        TaskStateOutcome::Abstained => "no_evidence",
        TaskStateOutcome::Unavailable => "task_state_unavailable",
    };
    let mut answer = AnswerOutcome::new(
        status,
        recall_status,
        reason,
        matches!(
            outcome,
            TaskStateOutcome::Abstained | TaskStateOutcome::Unavailable
        ),
        matches!(outcome, TaskStateOutcome::Complete)
            || matches!(
                outcome,
                TaskStateOutcome::Partial | TaskStateOutcome::Degraded
            ) && accepted > 0,
        fallback
            .map(|fallback| SafeFallback {
                mode: fallback.mode.clone(),
                reason: safe_reason(&fallback.reason),
            })
            .or_else(|| {
                (status != "complete").then(|| SafeFallback {
                    mode: "canonical_retrieval".to_string(),
                    reason: reason.to_string(),
                })
            }),
    );
    let mut exclusions = Vec::new();
    if rejected > 0 {
        exclusions.push(ReasonOnly {
            reason: "rejected_evidence".to_string(),
            count: rejected,
        });
    }
    if unresolved > 0 {
        exclusions.push(ReasonOnly {
            reason: "unresolved_evidence".to_string(),
            count: unresolved,
        });
    }
    if missing > 0 {
        exclusions.push(ReasonOnly {
            reason: "missing_evidence".to_string(),
            count: missing,
        });
    }
    answer.exclusions = exclusions;
    answer.conflicts = conflicts
        .iter()
        .map(|conflict| conflict_summary(&conflict.reason, &conflict.evidence_ids))
        .collect();
    answer.reason_codes = merge_reason_codes(
        &answer.reason,
        answer.exclusions.iter().map(|entry| entry.reason.as_str()),
    );
    answer
}

/// Convert a task-state resolver failure to a non-disclosing safe response.
/// Lifecycle and integrity failures abstain; infrastructure/unknown failures
/// remain unavailable. The raw resolver text is never serialized.
pub fn for_task_failure(error: &str) -> AnswerOutcome {
    let reason = failure_reason(error);
    let unavailable = matches!(
        reason.as_str(),
        "unavailable_evidence" | "backend_unavailable"
    );
    let status = if unavailable {
        "unavailable"
    } else {
        "abstained"
    };
    let mut answer = AnswerOutcome::new(
        status,
        if unavailable { "unavailable" } else { "stale" },
        &reason,
        true,
        false,
        Some(SafeFallback {
            mode: "canonical_retrieval".to_string(),
            reason: reason.clone(),
        }),
    );
    answer.exclusions = vec![ReasonOnly { reason, count: 1 }];
    answer
}

fn failure_reason(error: &str) -> String {
    let lower = error.to_ascii_lowercase();
    if lower.contains("superseded") {
        "superseded".to_string()
    } else if lower.contains("expired") {
        "expired".to_string()
    } else if lower.contains("revoked") || lower.contains("tombstoned") {
        "revoked".to_string()
    } else if lower.contains("stale source") || lower.contains("stale evidence") {
        "stale".to_string()
    } else if lower.contains("workspace scope") || lower.contains("scope mismatch") {
        "out_of_scope".to_string()
    } else if lower.contains("invisible")
        || lower.contains("requester")
        || lower.contains("suppressed")
    {
        "invisible".to_string()
    } else if lower.contains("digest") || lower.contains("malformed") {
        "invalid_evidence".to_string()
    } else if lower.contains("unavailable")
        || lower.contains("database")
        || lower.contains("lookup failed")
    {
        "unavailable_evidence".to_string()
    } else {
        "invalid_evidence".to_string()
    }
}

/// An unavailable handler/backend is not a successful empty response.
pub fn unavailable(reason: impl AsRef<str>) -> AnswerOutcome {
    let reason = safe_reason(reason.as_ref());
    AnswerOutcome::new(
        "unavailable",
        "unavailable",
        reason.clone(),
        true,
        false,
        Some(SafeFallback {
            mode: "canonical_retrieval".to_string(),
            reason,
        }),
    )
}

/// Translate an evidence projection's exclusions into answer-facing metadata.
/// If a requested lane has no valid items, the base result is forced to an
/// abstention. A malformed or unavailable evidence projection therefore cannot
/// become an ordinary empty success.
pub fn with_evidence(mut base: AnswerOutcome, evidence: &EvidenceProjection) -> AnswerOutcome {
    base.exclusions = reason_only_counts(&evidence.excluded);
    if evidence.items.is_empty() {
        let reason = evidence
            .excluded
            .first()
            .map(|entry| safe_reason(&entry.reason))
            .unwrap_or_else(|| {
                if base.status == "unavailable" {
                    base.reason.clone()
                } else {
                    "no_evidence".to_string()
                }
            });
        let unavailable = base.status == "unavailable"
            || base.recall_status == "unavailable"
            || reason == "unavailable"
            || reason == "unavailable_evidence";
        base.status = if unavailable {
            "unavailable".to_string()
        } else {
            "abstained".to_string()
        };
        base.recall_status = if unavailable {
            "unavailable".to_string()
        } else {
            base.recall_status.clone()
        };
        base.reason = reason.clone();
        base.abstained = true;
        base.answerable = false;
        base.fallback = Some(SafeFallback {
            mode: "canonical_retrieval".to_string(),
            reason,
        });
    } else if !evidence.excluded.is_empty() && base.status == "complete" {
        base.status = "partial".to_string();
        base.reason = "excluded_evidence".to_string();
        base.fallback = Some(SafeFallback {
            mode: "canonical_retrieval".to_string(),
            reason: "excluded_evidence".to_string(),
        });
    }
    base.reason = safe_reason(&base.reason);
    base.reason_codes = merge_reason_codes(
        &base.reason,
        base.exclusions.iter().map(|entry| entry.reason.as_str()),
    );
    base
}

pub fn reason_only_counts(exclusions: &[ExclusionRecord]) -> Vec<ReasonOnly> {
    let mut sorted = exclusions.to_vec();
    sorted.sort_by(|left, right| left.reason.cmp(&right.reason));
    let mut result: Vec<ReasonOnly> = Vec::new();
    for exclusion in sorted {
        if let Some(previous) = result.last_mut() {
            if previous.reason == exclusion.reason {
                previous.count = previous.count.saturating_add(exclusion.count);
                continue;
            }
        }
        result.push(ReasonOnly {
            reason: safe_reason(&exclusion.reason),
            count: exclusion.count,
        });
    }
    result
}

pub fn conflict_summary(reason: &str, references: &[String]) -> ConflictSummary {
    let mut normalized = references.to_vec();
    normalized.sort();
    normalized.dedup();
    let mut hasher = Sha256::new();
    for reference in &normalized {
        hasher.update(reference.len().to_string().as_bytes());
        hasher.update([0]);
        hasher.update(reference.as_bytes());
        hasher.update([0]);
    }
    ConflictSummary {
        reason: safe_reason(reason),
        reference_count: normalized.len(),
        references_sha256: format!("{:x}", hasher.finalize()),
    }
}

fn default_recall_reason(status: &RecallStatus, hits: usize) -> String {
    match status {
        RecallStatus::Empty if hits == 0 => "no_match".to_string(),
        RecallStatus::Unavailable => "unavailable".to_string(),
        RecallStatus::Stale => "stale_evidence".to_string(),
        RecallStatus::Timeout => "deadline_elapsed".to_string(),
        RecallStatus::Partial => "partial_recall".to_string(),
        RecallStatus::Fresh => "no_match".to_string(),
        RecallStatus::Empty => "no_match".to_string(),
    }
}

fn safe_reason(value: &str) -> String {
    let lower = value.trim().to_ascii_lowercase();
    let first = lower
        .split(|character: char| matches!(character, ':' | '\n' | '\r' | ',' | ';' | '|'))
        .next()
        .unwrap_or("")
        .trim();
    let known = [
        "abstained",
        "backend_unavailable",
        "budget_exhausted",
        "context assembled",
        "context_truncated",
        "context_unavailable",
        "conflicting_evidence",
        "db_unhealthy",
        "deadline_elapsed",
        "degraded",
        "dropped_budget",
        "dropped_coverage",
        "empty_store",
        "excluded_evidence",
        "expired",
        "incomplete_evidence",
        "index_behind",
        "invisible",
        "invalid_evidence",
        "malformed_reference",
        "missing_evidence",
        "missing_latest_version",
        "missing_required",
        "missing_source_group",
        "missing_temporal_anchor",
        "missing_temporal_evidence",
        "no_evidence",
        "no_match",
        "no_query",
        "no_eligible_candidates",
        "out_of_scope",
        "partial",
        "partial_arms",
        "partial_recall",
        "pending_embeds",
        "recall_partial",
        "recall_stale",
        "recall_timeout",
        "recall_unavailable",
        "red_herring_ignored",
        "rejected_evidence",
        "revoked",
        "selection_projection_unavailable",
        "source_missing",
        "stale",
        "stale_evidence",
        "superseded",
        "task_state_complete",
        "task_state_degraded",
        "task_state_partial",
        "task_state_unavailable",
        "unavailable",
        "unavailable_evidence",
        "unknown_chain_identity",
        "unresolved_conflict",
        "unresolved_evidence",
        "unsupported_lane",
        "wrong_chain_identity",
    ];
    if let Some(reason) = known.iter().find(|candidate| **candidate == first) {
        return (*reason).to_string();
    }
    if lower.contains("unavailable")
        || lower.contains("database")
        || lower.contains("lookup failed")
        || lower.contains("serialization failed")
    {
        "unavailable".to_string()
    } else if lower.contains("conflict") {
        "unresolved_conflict".to_string()
    } else if lower.contains("supersed") {
        "superseded".to_string()
    } else if lower.contains("expire") {
        "expired".to_string()
    } else if lower.contains("stale") {
        "stale_evidence".to_string()
    } else if lower.contains("scope") || lower.contains("invisible") {
        "out_of_scope".to_string()
    } else {
        "unspecified".to_string()
    }
}

fn bounded_reason(value: &str) -> String {
    value.chars().take(256).collect()
}

fn safe_fallback_mode(value: &str) -> String {
    match value {
        "abstain" | "canonical_retrieval" => value.to_string(),
        _ => "canonical_retrieval".to_string(),
    }
}

fn safe_recall_status(value: &str) -> String {
    match value {
        "fresh" | "partial" | "timeout" | "unavailable" | "empty" | "stale" => value.to_string(),
        _ => "unavailable".to_string(),
    }
}

fn merge_reason_codes<'a>(
    primary: &'a str,
    additional: impl IntoIterator<Item = &'a str>,
) -> Vec<String> {
    let mut result = Vec::new();
    for value in std::iter::once(primary).chain(additional) {
        let value = safe_reason(value);
        if !result.contains(&value) {
            result.push(value);
        }
        if result.len() >= 16 {
            break;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn revoked_task_evidence_has_a_reason_only_outcome() {
        let outcome = for_task_failure("evidence source opaque-ref is revoked");
        assert_eq!(outcome.status, "abstained");
        assert_eq!(outcome.reason, "revoked");
        assert!(outcome.answerable == false);
        assert_eq!(outcome.exclusions[0].reason, "revoked");
    }

    #[test]
    fn an_abstained_recall_never_maps_to_a_complete_answer() {
        let recall = RecallOutcome {
            status: RecallStatus::Fresh,
            abstained: true,
            ..RecallOutcome::default()
        };
        let outcome = for_recall(&recall, 1);
        assert_eq!(outcome.status, "abstained");
        assert!(!outcome.answerable);
        assert!(outcome.abstained);
    }

    #[test]
    fn serialized_answer_outcome_has_bounded_reason_codes_and_fallback() {
        let recall = RecallOutcome {
            status: RecallStatus::Unavailable,
            abstained: true,
            reason: "backend unavailable: internal database detail".to_string(),
            ..RecallOutcome::default()
        };
        let outcome = for_recall(&recall, 0);
        let value = serde_json::to_value(&outcome).unwrap();
        assert_eq!(value["status"], "unavailable");
        assert_eq!(value["abstained"], true);
        assert_eq!(value["answerable"], false);
        assert!(value["reason_codes"].is_array(), "{value}");
        assert_eq!(value["reason_codes"][0], "unavailable");
        assert_eq!(value["fallback"]["mode"], "canonical_retrieval");
        let serialized = value.to_string();
        assert!(!serialized.contains("internal database detail"), "{value}");
        assert!(outcome.validate().is_ok(), "{value}");
    }

    #[test]
    fn arbitrary_recall_status_is_not_serialized_into_answer_outcome() {
        let outcome = AnswerOutcome::new(
            "partial",
            "internal error: request-id-123",
            "partial_recall",
            false,
            true,
            Some(SafeFallback {
                mode: "canonical_retrieval".to_string(),
                reason: "partial_recall".to_string(),
            }),
        );
        assert_eq!(outcome.recall_status, "unavailable");
        assert!(!serde_json::to_string(&outcome)
            .unwrap()
            .contains("request-id-123"));
    }
}
