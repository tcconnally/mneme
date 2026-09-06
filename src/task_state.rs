//! Versioned task-scoped state and governed serving projection (#1182).
//!
//! Task state is a rebuildable projection over canonical Vault entities.  It is
//! deliberately not a second memory model: the durable row contains bounded
//! task metadata, canonical IDs, source revisions, and digests only.  Entity
//! bodies, prompts, and model reasoning remain outside the receipt and are
//! resolved through the normal governed readers when a serving response is
//! assembled.

use crate::db::Database;
use crate::models::{Entity, RecallOutcome, RecallParams, RecallStatus, SearchMode};
use rusqlite::OptionalExtension;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

pub const TASK_STATE_SCHEMA_VERSION: &str = "perseus-vault-task-state/v1";
pub const TASK_STATE_OUTCOMES: [&str; 5] = [
    "complete",
    "partial",
    "degraded",
    "abstained",
    "unavailable",
];

const MAX_ID_CHARS: usize = 256;
const MAX_TASK_ID_CHARS: usize = 128;
const MAX_ROUTE_CHARS: usize = 64;
const MAX_OBJECTIVE_CHARS: usize = 512;
const MAX_CONSTRAINTS: usize = 32;
const MAX_CONSTRAINT_CHARS: usize = 256;
const MAX_EVIDENCE_REFS: usize = 256;
const MAX_SLOTS: usize = 128;
const MAX_SLOT_REASON_CHARS: usize = 256;
const MAX_CONFLICTS: usize = 128;
const MAX_NEXT_STEP_REASON_CHARS: usize = 256;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TaskStateOutcome {
    Complete,
    Partial,
    Degraded,
    Abstained,
    Unavailable,
}

impl TaskStateOutcome {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Complete => "complete",
            Self::Partial => "partial",
            Self::Degraded => "degraded",
            Self::Abstained => "abstained",
            Self::Unavailable => "unavailable",
        }
    }

    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "complete" => Ok(Self::Complete),
            "partial" => Ok(Self::Partial),
            "degraded" => Ok(Self::Degraded),
            "abstained" => Ok(Self::Abstained),
            "unavailable" => Ok(Self::Unavailable),
            other => Err(format!("unsupported task-state outcome '{other}'")),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct TaskStateScope {
    pub tenant_id: String,
    pub workspace_hash: String,
    pub principal_id: String,
    pub agent_id: String,
    pub task_id: String,
}

impl TaskStateScope {
    pub fn validate(&self) -> Result<(), String> {
        validate_text("tenant_id", &self.tenant_id, MAX_ID_CHARS)?;
        validate_text("workspace_hash", &self.workspace_hash, MAX_ID_CHARS)?;
        validate_text("principal_id", &self.principal_id, MAX_ID_CHARS)?;
        validate_text("agent_id", &self.agent_id, MAX_ID_CHARS)?;
        validate_text("task_id", &self.task_id, MAX_TASK_ID_CHARS)?;
        // The repository's existing scoped projection contract uses the
        // workspace as its tenant partition.  Keeping that relation explicit
        // prevents an unbound tenant/global fallback.
        if self.tenant_id != self.workspace_hash {
            return Err("tenant_id must equal workspace_hash for task state".to_string());
        }
        // The current transport has one authenticated session principal.  Do
        // not permit a caller to update a different agent's task projection.
        if self.principal_id != self.agent_id {
            return Err("principal_id and agent_id must match for task state".to_string());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct TaskEvidenceReference {
    /// Canonical Vault entity ID.  The ID is resolved again on every update and
    /// rebuild; it is never treated as evidence by itself.
    pub entity_id: String,
    /// `entity:<entity_id>` in this v1 canonical-entity lane, or a provider
    /// source ID retained by `provider_sources`.
    pub source_id: String,
    /// Canonical source revision at the time the evidence was observed.
    pub revision: String,
    /// Digest of the canonical source representation.
    pub source_digest: String,
    /// Digest of the exact evidence representation consumed by the task.
    pub evidence_digest: String,
}

impl TaskEvidenceReference {
    fn validate(&self, label: &str) -> Result<(), String> {
        validate_text(&format!("{label}.entity_id"), &self.entity_id, MAX_ID_CHARS)?;
        validate_text(&format!("{label}.source_id"), &self.source_id, MAX_ID_CHARS)?;
        validate_text(&format!("{label}.revision"), &self.revision, MAX_ID_CHARS)?;
        validate_sha256(&format!("{label}.source_digest"), &self.source_digest)?;
        validate_sha256(&format!("{label}.evidence_digest"), &self.evidence_digest)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct EvidenceSlot {
    pub slot_id: String,
    pub reason: String,
}

impl EvidenceSlot {
    fn validate(&self, label: &str) -> Result<(), String> {
        validate_text(&format!("{label}.slot_id"), &self.slot_id, MAX_ID_CHARS)?;
        validate_text(
            &format!("{label}.reason"),
            &self.reason,
            MAX_SLOT_REASON_CHARS,
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ActiveConflict {
    pub conflict_id: String,
    pub evidence_ids: Vec<String>,
    pub reason: String,
}

impl ActiveConflict {
    fn validate(&self, label: &str) -> Result<(), String> {
        validate_text(
            &format!("{label}.conflict_id"),
            &self.conflict_id,
            MAX_ID_CHARS,
        )?;
        validate_text(
            &format!("{label}.reason"),
            &self.reason,
            MAX_SLOT_REASON_CHARS,
        )?;
        if self.evidence_ids.is_empty() || self.evidence_ids.len() > MAX_EVIDENCE_REFS {
            return Err(format!(
                "{label}.evidence_ids must contain between 1 and {MAX_EVIDENCE_REFS} IDs"
            ));
        }
        let mut ids = BTreeSet::new();
        for (index, id) in self.evidence_ids.iter().enumerate() {
            validate_text(&format!("{label}.evidence_ids[{index}]"), id, MAX_ID_CHARS)?;
            if !ids.insert(id) {
                return Err(format!("{label}.evidence_ids contains duplicate ID '{id}'"));
            }
        }
        Ok(())
    }
}

impl Default for ActiveConflict {
    fn default() -> Self {
        Self {
            conflict_id: String::new(),
            evidence_ids: Vec::new(),
            reason: String::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct NextStepMetadata {
    pub kind: String,
    pub reason: String,
}

impl Default for NextStepMetadata {
    fn default() -> Self {
        Self {
            kind: "none".to_string(),
            reason: "no next step declared".to_string(),
        }
    }
}

impl NextStepMetadata {
    fn validate(&self) -> Result<(), String> {
        validate_text("next_step.kind", &self.kind, MAX_ROUTE_CHARS)?;
        validate_text("next_step.reason", &self.reason, MAX_NEXT_STEP_REASON_CHARS)
    }
}

/// Input to one task-state projection/update.  `source_digest` and
/// `evidence_digest` are optional expectations: when supplied, the server
/// recomputes them from governed canonical references and rejects a mismatch.
/// They are always populated on the persisted `TaskState`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskStateRequest {
    pub schema_version: String,
    pub task_id: String,
    pub tenant_id: String,
    pub workspace_hash: String,
    /// Overwritten from the initialized transport session at the MCP boundary.
    pub principal_id: String,
    /// Overwritten from the initialized transport session at the MCP boundary.
    pub agent_id: String,
    #[serde(rename = "query_digest", alias = "task_digest")]
    pub task_digest: String,
    pub route: String,
    /// A bounded task objective/label, not the raw query or prompt.
    pub objective: String,
    #[serde(default)]
    pub temporal_anchor_unix_ms: Option<i64>,
    #[serde(default)]
    #[serde(rename = "constraints", alias = "load_bearing_constraints")]
    pub load_bearing_constraints: Vec<String>,
    pub base_sequence: u64,
    pub observed_input_digest: String,
    #[serde(default)]
    pub source_digest: Option<String>,
    #[serde(default)]
    pub evidence_digest: Option<String>,
    #[serde(default)]
    pub accepted_evidence: Vec<TaskEvidenceReference>,
    #[serde(default)]
    pub rejected_evidence: Vec<TaskEvidenceReference>,
    #[serde(default)]
    pub unresolved_evidence: Vec<EvidenceSlot>,
    #[serde(default)]
    pub active_conflicts: Vec<ActiveConflict>,
    #[serde(default)]
    pub missing_evidence: Vec<EvidenceSlot>,
    #[serde(default)]
    pub next_step: NextStepMetadata,
}

impl TaskStateRequest {
    pub(crate) fn validate_for_project_task(
        &self,
        query: &str,
        requested_workspace: Option<&str>,
        requester: &str,
    ) -> Result<(), String> {
        self.validate(query, requested_workspace, requester)
    }

    fn validate(
        &self,
        query: &str,
        requested_workspace: Option<&str>,
        requester: &str,
    ) -> Result<(), String> {
        if self.schema_version != TASK_STATE_SCHEMA_VERSION {
            return Err(format!(
                "unsupported task-state schema_version '{}'; expected {}",
                self.schema_version, TASK_STATE_SCHEMA_VERSION
            ));
        }
        let scope = self.scope();
        scope.validate()?;
        if requested_workspace != Some(self.workspace_hash.as_str()) {
            return Err("task-state workspace scope does not match project_task scope".to_string());
        }
        if self.principal_id != requester || self.agent_id != requester {
            return Err(
                "task-state principal/agent scope does not match transport identity".to_string(),
            );
        }
        let expected_task_digest = sha256_hex(query);
        if self.task_digest != expected_task_digest {
            return Err("task_digest does not match the resolved task query".to_string());
        }
        validate_sha256("observed_input_digest", &self.observed_input_digest)?;
        validate_text("route", &self.route, MAX_ROUTE_CHARS)?;
        validate_text("objective", &self.objective, MAX_OBJECTIVE_CHARS)?;
        if let Some(anchor) = self.temporal_anchor_unix_ms {
            if anchor < 0 {
                return Err("temporal_anchor_unix_ms must be non-negative".to_string());
            }
        }
        if self.load_bearing_constraints.len() > MAX_CONSTRAINTS {
            return Err(format!(
                "load_bearing_constraints may contain at most {MAX_CONSTRAINTS} entries"
            ));
        }
        let mut constraints = BTreeSet::new();
        for (index, constraint) in self.load_bearing_constraints.iter().enumerate() {
            validate_text(
                &format!("load_bearing_constraints[{index}]"),
                constraint,
                MAX_CONSTRAINT_CHARS,
            )?;
            if !constraints.insert(constraint) {
                return Err(format!(
                    "load_bearing_constraints contains duplicate value '{constraint}'"
                ));
            }
        }
        validate_optional_digest("source_digest", self.source_digest.as_deref())?;
        validate_optional_digest("evidence_digest", self.evidence_digest.as_deref())?;
        validate_reference_lists(&self.accepted_evidence, &self.rejected_evidence)?;
        validate_slots("unresolved_evidence", &self.unresolved_evidence)?;
        validate_slots("missing_evidence", &self.missing_evidence)?;
        if self.active_conflicts.len() > MAX_CONFLICTS {
            return Err(format!(
                "active_conflicts may contain at most {MAX_CONFLICTS} entries"
            ));
        }
        let mut conflict_ids = BTreeSet::new();
        for (index, conflict) in self.active_conflicts.iter().enumerate() {
            conflict.validate(&format!("active_conflicts[{index}]"))?;
            if !conflict_ids.insert(&conflict.conflict_id) {
                return Err(format!(
                    "active_conflicts contains duplicate ID '{}'",
                    conflict.conflict_id
                ));
            }
        }
        self.next_step.validate()?;
        task_search_mode(&self.route).map(|_| ())
    }

    fn scope(&self) -> TaskStateScope {
        TaskStateScope {
            tenant_id: self.tenant_id.clone(),
            workspace_hash: self.workspace_hash.clone(),
            principal_id: self.principal_id.clone(),
            agent_id: self.agent_id.clone(),
            task_id: self.task_id.clone(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct TaskState {
    pub schema_version: String,
    pub scope: TaskStateScope,
    #[serde(rename = "query_digest", alias = "task_digest")]
    pub task_digest: String,
    pub route: String,
    pub objective: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temporal_anchor_unix_ms: Option<i64>,
    #[serde(rename = "constraints", alias = "load_bearing_constraints")]
    pub load_bearing_constraints: Vec<String>,
    pub accepted_evidence: Vec<TaskEvidenceReference>,
    pub rejected_evidence: Vec<TaskEvidenceReference>,
    pub unresolved_evidence: Vec<EvidenceSlot>,
    pub active_conflicts: Vec<ActiveConflict>,
    pub missing_evidence: Vec<EvidenceSlot>,
    pub next_step: NextStepMetadata,
    pub state_sequence: u64,
    pub base_sequence: u64,
    pub observed_input_digest: String,
    pub source_digest: String,
    pub evidence_digest: String,
    pub outcome: TaskStateOutcome,
    pub state_digest: String,
}

impl TaskState {
    fn from_request(
        request: &TaskStateRequest,
        accepted_evidence: Vec<TaskEvidenceReference>,
        rejected_evidence: Vec<TaskEvidenceReference>,
        missing_evidence: Vec<EvidenceSlot>,
        source_digest: String,
        evidence_digest: String,
        outcome: TaskStateOutcome,
    ) -> Result<Self, String> {
        let state_sequence = request
            .base_sequence
            .checked_add(1)
            .ok_or_else(|| "task-state base_sequence overflow".to_string())?;
        let mut state = Self {
            schema_version: TASK_STATE_SCHEMA_VERSION.to_string(),
            scope: request.scope(),
            task_digest: request.task_digest.clone(),
            route: request.route.clone(),
            objective: request.objective.clone(),
            temporal_anchor_unix_ms: request.temporal_anchor_unix_ms,
            load_bearing_constraints: request.load_bearing_constraints.clone(),
            accepted_evidence,
            rejected_evidence,
            unresolved_evidence: request.unresolved_evidence.clone(),
            active_conflicts: request.active_conflicts.clone(),
            missing_evidence,
            next_step: request.next_step.clone(),
            state_sequence,
            base_sequence: request.base_sequence,
            observed_input_digest: request.observed_input_digest.clone(),
            source_digest,
            evidence_digest,
            outcome,
            state_digest: String::new(),
        };
        state.seal()?;
        Ok(state)
    }

    pub fn validate(&self) -> Result<(), String> {
        self.scope.validate()?;
        if self.schema_version != TASK_STATE_SCHEMA_VERSION {
            return Err(format!(
                "unsupported persisted task-state schema_version '{}'; expected {}",
                self.schema_version, TASK_STATE_SCHEMA_VERSION
            ));
        }
        let expected_sequence = self
            .base_sequence
            .checked_add(1)
            .ok_or_else(|| "task-state base_sequence overflow".to_string())?;
        if self.state_sequence == 0 || self.state_sequence != expected_sequence {
            return Err("task-state sequence must be base_sequence + 1".to_string());
        }
        validate_sha256("task_digest", &self.task_digest)?;
        validate_sha256("observed_input_digest", &self.observed_input_digest)?;
        validate_sha256("source_digest", &self.source_digest)?;
        validate_sha256("evidence_digest", &self.evidence_digest)?;
        validate_text("route", &self.route, MAX_ROUTE_CHARS)?;
        validate_text("objective", &self.objective, MAX_OBJECTIVE_CHARS)?;
        if self.temporal_anchor_unix_ms.is_some_and(|value| value < 0) {
            return Err("temporal_anchor_unix_ms must be non-negative".to_string());
        }
        if self.load_bearing_constraints.len() > MAX_CONSTRAINTS {
            return Err(format!(
                "load_bearing_constraints may contain at most {MAX_CONSTRAINTS} entries"
            ));
        }
        let mut constraints = BTreeSet::new();
        for constraint in &self.load_bearing_constraints {
            validate_text("load_bearing_constraints", constraint, MAX_CONSTRAINT_CHARS)?;
            if !constraints.insert(constraint) {
                return Err("load_bearing_constraints contains duplicate values".to_string());
            }
        }
        validate_reference_lists(&self.accepted_evidence, &self.rejected_evidence)?;
        validate_slots("unresolved_evidence", &self.unresolved_evidence)?;
        validate_slots("missing_evidence", &self.missing_evidence)?;
        if self.active_conflicts.len() > MAX_CONFLICTS {
            return Err(format!(
                "active_conflicts may contain at most {MAX_CONFLICTS} entries"
            ));
        }
        let mut conflict_ids = BTreeSet::new();
        for conflict in &self.active_conflicts {
            conflict.validate("active_conflicts")?;
            if !conflict_ids.insert(&conflict.conflict_id) {
                return Err("active_conflicts contains duplicate IDs".to_string());
            }
        }
        self.next_step.validate()?;
        TaskStateOutcome::parse(self.outcome.as_str())?;
        let expected = self.compute_state_digest()?;
        if expected != self.state_digest {
            return Err("state_digest does not match task-state contents".to_string());
        }
        Ok(())
    }

    pub fn compute_state_digest(&self) -> Result<String, String> {
        let mut value = serde_json::to_value(self)
            .map_err(|error| format!("task-state serialization failed: {error}"))?;
        value
            .as_object_mut()
            .ok_or_else(|| "task-state serialization was not an object".to_string())?
            .remove("state_digest");
        Ok(domain_digest("perseus-vault/task-state/v1", &value))
    }

    fn seal(&mut self) -> Result<(), String> {
        self.state_digest = self.compute_state_digest()?;
        self.validate()
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct CanonicalSource {
    pub id: String,
    pub source_id: String,
    pub category: String,
    pub key: String,
    pub workspace_hash: String,
    pub agent_id: String,
    pub status: String,
    pub revision: String,
    pub source_digest: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct TaskFallback {
    pub mode: String,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct TaskServingProjection {
    pub canonical_sources: Vec<CanonicalSource>,
    pub recalled_evidence: Vec<TaskEvidenceReference>,
    pub rejected_evidence: Vec<TaskEvidenceReference>,
    pub unresolved_evidence: Vec<EvidenceSlot>,
    pub missing_evidence: Vec<EvidenceSlot>,
    pub derived_task_state: TaskState,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fallback: Option<TaskFallback>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TaskServingResponse {
    pub schema_version: String,
    pub outcome: TaskStateOutcome,
    pub task_state: TaskState,
    pub serving: TaskServingProjection,
}

#[derive(Debug, Clone)]
struct ResolvedReference {
    reference: TaskEvidenceReference,
}

/// One coherent, provider-free serving path for the opt-in `project_task`
/// contract. The query is used transiently for recall and is never copied into
/// `TaskState` or the persisted receipt.
pub fn serve_project_task(
    db: &Database,
    request: &TaskStateRequest,
    query: &str,
    category: Option<&str>,
    limit: i64,
    requesting_agent_id: &str,
    requested_workspace: Option<&str>,
) -> Result<TaskServingResponse, String> {
    request.validate(query, requested_workspace, requesting_agent_id)?;
    let scope = request.scope();
    let previous = if request.base_sequence > 0 {
        db.task_state_get(&scope)?
    } else {
        None
    };
    if let Some(previous) = previous.as_ref() {
        ensure_immutable_spec(previous, request)?;
    }

    let mut params = RecallParams::default();
    params.query = query.to_string();
    params.category = category.map(str::to_owned);
    params.limit = limit.clamp(1, MAX_EVIDENCE_REFS as i64);
    params.offset = 0;
    params.mode = task_search_mode(&request.route)?;
    params.skip_side_effects = true;
    params.workspace_hash = Some(scope.workspace_hash.clone());
    params.enforce_utility_horizon = true;
    let query_embedding_available = matches!(params.mode, SearchMode::Fts5);
    let (candidates, recall_outcome) =
        match db.recall_for_requester(&params, Some(&scope.principal_id)) {
            Ok(candidates) => {
                let outcome = db.recall_outcome(
                    &params.mode,
                    query_embedding_available,
                    candidates.len(),
                    None,
                );
                (candidates, outcome)
            }
            Err(error) => {
                let reason = error.to_string();
                let outcome =
                    db.recall_outcome(&params.mode, query_embedding_available, 0, Some(&reason));
                (Vec::new(), outcome)
            }
        };

    let auto_recall = request.accepted_evidence.is_empty() && request.rejected_evidence.is_empty();
    let accepted_input = if auto_recall {
        candidates.iter().map(canonical_reference).collect()
    } else {
        request.accepted_evidence.clone()
    };
    let rejected_input = request.rejected_evidence.clone();

    let mut sources = Vec::new();
    let accepted = resolve_references(
        db,
        &scope,
        request.temporal_anchor_unix_ms,
        &accepted_input,
        &mut sources,
    )?;
    let rejected = resolve_references(
        db,
        &scope,
        request.temporal_anchor_unix_ms,
        &rejected_input,
        &mut sources,
    )?;
    let accepted_refs: Vec<TaskEvidenceReference> = accepted
        .iter()
        .map(|resolved| resolved.reference.clone())
        .collect();
    let rejected_refs: Vec<TaskEvidenceReference> = rejected
        .iter()
        .map(|resolved| resolved.reference.clone())
        .collect();

    let mut missing = request.missing_evidence.clone();
    if accepted_refs.is_empty() && request.unresolved_evidence.is_empty() && missing.is_empty() {
        missing.push(EvidenceSlot {
            slot_id: "task-evidence".to_string(),
            reason: match recall_outcome.status {
                RecallStatus::Unavailable => "governed evidence backend unavailable".to_string(),
                RecallStatus::Partial | RecallStatus::Timeout | RecallStatus::Stale => {
                    "governed evidence lane is incomplete".to_string()
                }
                RecallStatus::Fresh | RecallStatus::Empty => {
                    if candidates.is_empty() {
                        "no governed evidence matched the task query".to_string()
                    } else {
                        "no governed evidence survived validation".to_string()
                    }
                }
            },
        });
    }
    validate_slots("missing_evidence", &missing)?;

    let source_digest = digest_sources(&sources);
    let evidence_digest = digest_references(&accepted_refs, &rejected_refs);
    if let Some(expected) = request.source_digest.as_deref() {
        if expected != source_digest {
            return Err("source digest mismatch".to_string());
        }
    }
    if let Some(expected) = request.evidence_digest.as_deref() {
        if expected != evidence_digest {
            return Err("evidence digest mismatch".to_string());
        }
    }

    let outcome = derive_outcome(
        accepted_refs.len(),
        rejected_refs.len(),
        request.unresolved_evidence.len(),
        request.active_conflicts.len(),
        missing.len(),
        auto_recall.then_some(&recall_outcome),
    );
    let state = TaskState::from_request(
        request,
        accepted_refs.clone(),
        rejected_refs.clone(),
        missing.clone(),
        source_digest,
        evidence_digest,
        outcome.clone(),
    )?;
    db.task_state_compare_and_swap(&state)?;

    sources.sort_by(|left, right| {
        (&left.id, &left.source_id, &left.revision).cmp(&(
            &right.id,
            &right.source_id,
            &right.revision,
        ))
    });
    sources.dedup_by(|left, right| left.id == right.id && left.source_id == right.source_id);
    let fallback = match outcome {
        TaskStateOutcome::Complete => None,
        TaskStateOutcome::Partial => Some(TaskFallback {
            mode: "canonical_retrieval".to_string(),
            reason: "task state has incomplete evidence".to_string(),
        }),
        TaskStateOutcome::Degraded => Some(TaskFallback {
            mode: "canonical_retrieval".to_string(),
            reason: "task state was served from a degraded evidence lane".to_string(),
        }),
        TaskStateOutcome::Abstained => Some(TaskFallback {
            mode: "canonical_retrieval".to_string(),
            reason: "task state has no sufficient unconflicted evidence".to_string(),
        }),
        TaskStateOutcome::Unavailable => Some(TaskFallback {
            mode: "canonical_retrieval".to_string(),
            reason: "task-state projection is unavailable".to_string(),
        }),
    };
    Ok(TaskServingResponse {
        schema_version: TASK_STATE_SCHEMA_VERSION.to_string(),
        outcome: outcome.clone(),
        task_state: state.clone(),
        serving: TaskServingProjection {
            canonical_sources: sources,
            recalled_evidence: accepted_refs,
            rejected_evidence: rejected_refs,
            unresolved_evidence: state.unresolved_evidence.clone(),
            missing_evidence: state.missing_evidence.clone(),
            derived_task_state: state,
            fallback,
        },
    })
}

/// Revalidate a persisted projection entirely from its retained canonical
/// references. No prompt, body, or prior model reasoning is needed.
pub fn rebuild_task_state(db: &Database, scope: &TaskStateScope) -> Result<TaskState, String> {
    scope.validate()?;
    let state = db
        .task_state_get(scope)?
        .ok_or_else(|| "task state is unavailable".to_string())?;
    let mut sources = Vec::new();
    let accepted = resolve_references(
        db,
        scope,
        state.temporal_anchor_unix_ms,
        &state.accepted_evidence,
        &mut sources,
    )?;
    let rejected = resolve_references(
        db,
        scope,
        state.temporal_anchor_unix_ms,
        &state.rejected_evidence,
        &mut sources,
    )?;
    let accepted_refs: Vec<_> = accepted.into_iter().map(|item| item.reference).collect();
    let rejected_refs: Vec<_> = rejected.into_iter().map(|item| item.reference).collect();
    let source_digest = digest_sources(&sources);
    let evidence_digest = digest_references(&accepted_refs, &rejected_refs);
    if source_digest != state.source_digest {
        return Err("source digest mismatch while rebuilding task state".to_string());
    }
    if evidence_digest != state.evidence_digest {
        return Err("evidence digest mismatch while rebuilding task state".to_string());
    }
    state.validate()?;
    Ok(state)
}

fn ensure_immutable_spec(previous: &TaskState, request: &TaskStateRequest) -> Result<(), String> {
    if previous.scope != request.scope()
        || previous.task_digest != request.task_digest
        || previous.route != request.route
        || previous.objective != request.objective
        || previous.temporal_anchor_unix_ms != request.temporal_anchor_unix_ms
        || previous.load_bearing_constraints != request.load_bearing_constraints
    {
        return Err("task-state immutable specification changed".to_string());
    }
    Ok(())
}

fn task_search_mode(route: &str) -> Result<SearchMode, String> {
    match route {
        "dense" => Ok(SearchMode::Dense),
        "hybrid" => Ok(SearchMode::Hybrid),
        "fused" => Ok(SearchMode::Fused),
        "fts5" | "fts" | "project_task" => Ok(SearchMode::Fts5),
        _ => Err("unsupported task route".to_string()),
    }
}

fn derive_outcome(
    accepted: usize,
    rejected: usize,
    unresolved: usize,
    conflicts: usize,
    missing: usize,
    recall: Option<&RecallOutcome>,
) -> TaskStateOutcome {
    if conflicts > 0 {
        return TaskStateOutcome::Abstained;
    }
    if let Some(recall) = recall {
        match recall.status {
            RecallStatus::Unavailable => {
                if accepted > 0 && recall.reason == "semantic_backend_not_serving" {
                    return TaskStateOutcome::Degraded;
                }
                return TaskStateOutcome::Unavailable;
            }
            RecallStatus::Partial | RecallStatus::Timeout | RecallStatus::Stale => {
                return TaskStateOutcome::Degraded;
            }
            RecallStatus::Fresh | RecallStatus::Empty => {}
        }
    }
    if accepted == 0 && (unresolved > 0 || missing > 0) {
        return TaskStateOutcome::Abstained;
    }
    if accepted == 0 {
        TaskStateOutcome::Abstained
    } else if rejected > 0 || unresolved > 0 || missing > 0 {
        TaskStateOutcome::Partial
    } else {
        TaskStateOutcome::Complete
    }
}

fn validate_reference_lists(
    accepted: &[TaskEvidenceReference],
    rejected: &[TaskEvidenceReference],
) -> Result<(), String> {
    if accepted.len() > MAX_EVIDENCE_REFS || rejected.len() > MAX_EVIDENCE_REFS {
        return Err(format!(
            "evidence reference lists may contain at most {MAX_EVIDENCE_REFS} entries"
        ));
    }
    let mut ids = BTreeSet::new();
    for (index, reference) in accepted.iter().enumerate() {
        reference.validate(&format!("accepted_evidence[{index}]"))?;
        if !ids.insert(&reference.entity_id) {
            return Err(format!(
                "duplicate evidence reference ID '{}'",
                reference.entity_id
            ));
        }
    }
    for (index, reference) in rejected.iter().enumerate() {
        reference.validate(&format!("rejected_evidence[{index}]"))?;
        if !ids.insert(&reference.entity_id) {
            return Err(format!(
                "duplicate evidence reference ID '{}'",
                reference.entity_id
            ));
        }
    }
    Ok(())
}

fn validate_slots(label: &str, slots: &[EvidenceSlot]) -> Result<(), String> {
    if slots.len() > MAX_SLOTS {
        return Err(format!("{label} may contain at most {MAX_SLOTS} entries"));
    }
    let mut ids = BTreeSet::new();
    for (index, slot) in slots.iter().enumerate() {
        slot.validate(&format!("{label}[{index}]"))?;
        if !ids.insert(&slot.slot_id) {
            return Err(format!(
                "{label} contains duplicate slot ID '{}'",
                slot.slot_id
            ));
        }
    }
    Ok(())
}

fn canonical_reference(entity: &Entity) -> TaskEvidenceReference {
    let source_digest = sha256_hex(&entity.body_json);
    TaskEvidenceReference {
        entity_id: entity.id.clone(),
        source_id: format!("entity:{}", entity.id),
        revision: format!("entity-v1:{source_digest}"),
        source_digest: source_digest.clone(),
        evidence_digest: source_digest,
    }
}

fn resolve_references(
    db: &Database,
    scope: &TaskStateScope,
    temporal_anchor_unix_ms: Option<i64>,
    references: &[TaskEvidenceReference],
    sources: &mut Vec<CanonicalSource>,
) -> Result<Vec<ResolvedReference>, String> {
    let mut resolved = Vec::with_capacity(references.len());
    for reference in references {
        reference.validate("evidence")?;
        let Some(raw) = db
            .get_entity_by_id_unfiltered(&reference.entity_id)
            .map_err(|error| format!("canonical evidence lookup failed: {error}"))?
        else {
            return Err(format!(
                "unknown evidence reference ID '{}'",
                reference.entity_id
            ));
        };
        if raw.workspace_hash != scope.workspace_hash && !raw.workspace_hash.is_empty() {
            return Err(format!(
                "evidence workspace scope mismatch for '{}'",
                reference.entity_id
            ));
        }
        if !db.requester_can_read(Some(&scope.principal_id), &raw.visibility, &raw.agent_id) {
            return Err(format!(
                "evidence reference '{}' is invisible to the task principal",
                reference.entity_id
            ));
        }
        if raw.archived {
            return Err(format!(
                "evidence reference '{}' is archived",
                reference.entity_id
            ));
        }
        match raw.status.as_str() {
            "deprecated" => {
                return Err(format!(
                    "evidence reference '{}' is superseded",
                    reference.entity_id
                ));
            }
            "expired" => {
                return Err(format!(
                    "evidence reference '{}' is expired",
                    reference.entity_id
                ));
            }
            "active" | "draft" => {}
            other => {
                return Err(format!(
                    "evidence reference '{}' has non-serveable lifecycle status '{other}'",
                    reference.entity_id
                ));
            }
        }
        let temporal_bounds: (Option<i64>, Option<i64>, Option<i64>) = {
            let conn = db
                .conn()
                .map_err(|error| format!("canonical temporal lookup failed: {error}"))?;
            conn.query_row(
                "SELECT valid_from_unix_ms, valid_to_unix_ms, invalidated_at_unix_ms
                 FROM entities WHERE id = ?1",
                rusqlite::params![&raw.id],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .map_err(|error| format!("canonical temporal lookup failed: {error}"))?
        };
        let anchor = temporal_anchor_unix_ms.unwrap_or_else(crate::db::now_ms);
        if anchor < raw.created_at_unix_ms
            || temporal_bounds
                .0
                .is_some_and(|valid_from| anchor < valid_from)
            || temporal_bounds.1.is_some_and(|valid_to| anchor >= valid_to)
            || temporal_bounds
                .2
                .is_some_and(|invalidated_at| anchor >= invalidated_at)
        {
            return Err(format!(
                "evidence reference '{}' is not valid at temporal anchor",
                reference.entity_id
            ));
        }
        if crate::db::entity_expiry_ms(&raw.body_json).is_some_and(|expiry| expiry <= anchor) {
            return Err(format!(
                "evidence reference '{}' is expired",
                reference.entity_id
            ));
        }
        let Some(entity) = db
            .get_entity_by_id_for_requester(&reference.entity_id, Some(&scope.principal_id))
            .map_err(|error| format!("governed evidence lookup failed: {error}"))?
        else {
            return Err(format!(
                "evidence reference '{}' is suppressed or unavailable",
                reference.entity_id
            ));
        };

        let source = if reference.source_id == format!("entity:{}", entity.id) {
            let digest = sha256_hex(&entity.body_json);
            let revision = format!("entity-v1:{digest}");
            if reference.source_digest != digest || reference.evidence_digest != digest {
                return Err(format!(
                    "source digest mismatch for evidence reference '{}'",
                    reference.entity_id
                ));
            }
            if reference.revision != revision {
                return Err(format!(
                    "stale source version for evidence reference '{}'",
                    reference.entity_id
                ));
            }
            CanonicalSource {
                id: entity.id.clone(),
                source_id: reference.source_id.clone(),
                category: entity.category.clone(),
                key: entity.key.clone(),
                workspace_hash: entity.workspace_hash.clone(),
                agent_id: entity.agent_id.clone(),
                status: entity.status.clone(),
                revision,
                source_digest: digest,
            }
        } else {
            resolve_provider_source(db, scope, &entity, reference)?
        };
        sources.push(source);
        resolved.push(ResolvedReference {
            reference: reference.clone(),
        });
    }
    Ok(resolved)
}

fn resolve_provider_source(
    db: &Database,
    scope: &TaskStateScope,
    entity: &Entity,
    reference: &TaskEvidenceReference,
) -> Result<CanonicalSource, String> {
    let conn = db
        .conn()
        .map_err(|error| format!("provider source lookup failed: {error}"))?;
    let row: Option<(
        Option<String>,
        String,
        Option<String>,
        String,
        String,
        String,
        Option<i64>,
    )> = conn
        .query_row(
            "SELECT entity_id, revision, content_sha256, workspace_hash, visibility,
                    state, deleted_at_unix_ms
             FROM provider_sources WHERE source_id = ?1",
            rusqlite::params![reference.source_id],
            |row| {
                Ok((
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    row.get(4)?,
                    row.get(5)?,
                    row.get(6)?,
                ))
            },
        )
        .optional()
        .map_err(|error| format!("provider source lookup failed: {error}"))?;
    let Some((entity_id, revision, content_sha256, workspace_hash, visibility, state, deleted_at)) =
        row
    else {
        return Err(format!("unknown source identity '{}'", reference.source_id));
    };
    if state != "active" || deleted_at.is_some() {
        return Err(format!(
            "evidence source '{}' is revoked",
            reference.source_id
        ));
    }
    if entity_id.as_deref() != Some(entity.id.as_str()) {
        return Err(format!(
            "source identity mismatch for evidence reference '{}'",
            reference.entity_id
        ));
    }
    if workspace_hash != scope.workspace_hash && !workspace_hash.is_empty() {
        return Err(format!(
            "source workspace scope mismatch for '{}'",
            reference.source_id
        ));
    }
    if !db.requester_can_read(Some(&scope.principal_id), &visibility, &entity.agent_id) {
        return Err(format!(
            "evidence source '{}' is invisible to the task principal",
            reference.source_id
        ));
    }
    let Some(content_digest) = content_sha256 else {
        return Err(format!(
            "source digest unavailable for '{}'",
            reference.source_id
        ));
    };
    validate_sha256("provider source content_sha256", &content_digest)?;
    if reference.revision != revision {
        return Err(format!(
            "stale source version for evidence source '{}'",
            reference.source_id
        ));
    }
    if reference.source_digest != content_digest {
        return Err(format!(
            "source digest mismatch for evidence source '{}'",
            reference.source_id
        ));
    }
    Ok(CanonicalSource {
        id: entity.id.clone(),
        source_id: reference.source_id.clone(),
        category: entity.category.clone(),
        key: entity.key.clone(),
        workspace_hash: entity.workspace_hash.clone(),
        agent_id: entity.agent_id.clone(),
        status: entity.status.clone(),
        revision,
        source_digest: content_digest,
    })
}

fn digest_sources(sources: &[CanonicalSource]) -> String {
    let mut values = sources.to_vec();
    values.sort_by(|left, right| {
        (
            &left.id,
            &left.source_id,
            &left.revision,
            &left.source_digest,
        )
            .cmp(&(
                &right.id,
                &right.source_id,
                &right.revision,
                &right.source_digest,
            ))
    });
    domain_digest("perseus-vault/task-state-sources/v1", &values)
}

fn digest_references(
    accepted: &[TaskEvidenceReference],
    rejected: &[TaskEvidenceReference],
) -> String {
    let mut value = serde_json::json!({
        "accepted": accepted,
        "rejected": rejected,
    });
    if let Some(object) = value.as_object_mut() {
        object.insert(
            "accepted_count".to_string(),
            Value::from(accepted.len() as u64),
        );
        object.insert(
            "rejected_count".to_string(),
            Value::from(rejected.len() as u64),
        );
    }
    domain_digest("perseus-vault/task-state-evidence/v1", &value)
}

fn ensure_text_without_control(value: &str) -> bool {
    !value.chars().any(char::is_control)
}

fn validate_text(label: &str, value: &str, max_chars: usize) -> Result<(), String> {
    if value.trim().is_empty() {
        return Err(format!("{label} must be non-empty"));
    }
    if value.chars().count() > max_chars || !ensure_text_without_control(value) {
        return Err(format!(
            "{label} is invalid or exceeds {max_chars} characters"
        ));
    }
    Ok(())
}

fn validate_sha256(label: &str, value: &str) -> Result<(), String> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err(format!("{label} must be a lowercase SHA-256 digest"));
    }
    Ok(())
}

fn validate_optional_digest(label: &str, value: Option<&str>) -> Result<(), String> {
    if let Some(value) = value {
        validate_sha256(label, value)?;
    }
    Ok(())
}

fn sha256_hex(value: &str) -> String {
    format!("{:x}", Sha256::digest(value.as_bytes()))
}

fn canonical_value(value: &Value) -> Value {
    match value {
        Value::Object(object) => {
            let mut keys: Vec<&String> = object.keys().collect();
            keys.sort();
            let mut sorted = Map::new();
            for key in keys {
                sorted.insert(key.clone(), canonical_value(&object[key]));
            }
            Value::Object(sorted)
        }
        Value::Array(values) => Value::Array(values.iter().map(canonical_value).collect()),
        other => other.clone(),
    }
}

fn domain_digest(domain: &str, value: &impl Serialize) -> String {
    let canonical = serde_json::to_value(value)
        .map(|value| canonical_value(&value).to_string())
        .unwrap_or_else(|_| "null".to_string());
    sha256_hex(&format!("{domain}|{canonical}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(value: &str) -> String {
        sha256_hex(value)
    }

    fn scope() -> TaskStateScope {
        TaskStateScope {
            tenant_id: "workspace-a".to_string(),
            workspace_hash: "workspace-a".to_string(),
            principal_id: "agent-a".to_string(),
            agent_id: "agent-a".to_string(),
            task_id: "task-a".to_string(),
        }
    }

    fn request_for(query: &str, task_id: &str) -> TaskStateRequest {
        TaskStateRequest {
            schema_version: TASK_STATE_SCHEMA_VERSION.to_string(),
            task_id: task_id.to_string(),
            tenant_id: "workspace-a".to_string(),
            workspace_hash: "workspace-a".to_string(),
            principal_id: "agent-a".to_string(),
            agent_id: "agent-a".to_string(),
            task_digest: digest(query),
            route: "project_task".to_string(),
            objective: "review evidence".to_string(),
            temporal_anchor_unix_ms: None,
            load_bearing_constraints: Vec::new(),
            base_sequence: 0,
            observed_input_digest: digest("observation"),
            source_digest: None,
            evidence_digest: None,
            accepted_evidence: Vec::new(),
            rejected_evidence: Vec::new(),
            unresolved_evidence: Vec::new(),
            active_conflicts: Vec::new(),
            missing_evidence: Vec::new(),
            next_step: NextStepMetadata::default(),
        }
    }

    fn reference_for(entity_id: &str) -> TaskEvidenceReference {
        let source_digest = digest(&format!("source:{entity_id}"));
        TaskEvidenceReference {
            entity_id: entity_id.to_string(),
            source_id: format!("entity:{entity_id}"),
            revision: format!("entity-v1:{source_digest}"),
            source_digest: source_digest.clone(),
            evidence_digest: source_digest,
        }
    }

    #[test]
    fn task_state_outcomes_are_closed_and_serializable() {
        for name in TASK_STATE_OUTCOMES {
            let outcome = TaskStateOutcome::parse(name).unwrap();
            assert_eq!(outcome.as_str(), name);
            assert_eq!(serde_json::to_value(outcome).unwrap(), Value::from(name));
        }
        assert!(TaskStateOutcome::parse("unknown").is_err());
    }

    #[test]
    fn task_state_rejects_duplicate_evidence_references_before_resolution() {
        let db = crate::db::TestDatabase::new("task-state-duplicate");
        let mut request = request_for("needle", "task-duplicate");
        let reference = reference_for("duplicate-id");
        request.accepted_evidence = vec![reference.clone(), reference];
        let error = serve_project_task(
            &db,
            &request,
            "needle",
            None,
            10,
            "agent-a",
            Some("workspace-a"),
        )
        .expect_err("duplicate evidence IDs must be rejected");
        assert!(
            error.contains("duplicate evidence reference"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn task_state_rejects_unknown_evidence_references() {
        let db = crate::db::TestDatabase::new("task-state-unknown");
        let mut request = request_for("needle", "task-unknown");
        request.accepted_evidence = vec![reference_for("does-not-exist")];
        let error = serve_project_task(
            &db,
            &request,
            "needle",
            None,
            10,
            "agent-a",
            Some("workspace-a"),
        )
        .expect_err("unknown evidence IDs must be rejected");
        assert!(
            error.contains("unknown evidence reference"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn task_state_rejects_evidence_from_another_workspace() {
        let db = crate::db::TestDatabase::new("task-state-scope");
        let mut entity = crate::db::tests::make_entity(
            "cross-workspace-evidence",
            "facts",
            "cross-workspace",
            r#"{\"content\":\"cross workspace evidence\"}"#,
        );
        entity.workspace_hash = "workspace-b".to_string();
        entity.agent_id = "agent-a".to_string();
        db.remember_skip_dedup(&entity).unwrap();

        let mut request = request_for("needle", "task-scope");
        request.accepted_evidence = vec![canonical_reference(&entity)];
        let error = serve_project_task(
            &db,
            &request,
            "needle",
            None,
            10,
            "agent-a",
            Some("workspace-a"),
        )
        .expect_err("cross-workspace evidence must be rejected");
        assert!(
            error.contains("workspace scope mismatch"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn task_state_rejects_a_source_digest_mismatch() {
        let db = crate::db::TestDatabase::new("task-state-digest");
        let mut entity = crate::db::tests::make_entity(
            "digest-evidence",
            "facts",
            "digest",
            r#"{\"content\":\"digest-bound evidence\"}"#,
        );
        entity.workspace_hash = "workspace-a".to_string();
        entity.agent_id = "agent-a".to_string();
        db.remember_skip_dedup(&entity).unwrap();

        let mut reference = canonical_reference(&entity);
        reference.source_digest = "0".repeat(64);
        let mut request = request_for("needle", "task-digest");
        request.accepted_evidence = vec![reference];
        let error = serve_project_task(
            &db,
            &request,
            "needle",
            None,
            10,
            "agent-a",
            Some("workspace-a"),
        )
        .expect_err("a forged source digest must be rejected");
        assert!(
            error.contains("source digest mismatch"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn task_state_rejects_superseded_evidence() {
        let db = crate::db::TestDatabase::new("task-state-superseded");
        let mut entity = crate::db::tests::make_entity(
            "superseded-evidence",
            "facts",
            "superseded",
            r#"{\"content\":\"old deployment evidence\"}"#,
        );
        entity.workspace_hash = "workspace-a".to_string();
        entity.agent_id = "agent-a".to_string();
        db.remember_skip_dedup(&entity).unwrap();
        {
            let conn = db.conn().unwrap();
            conn.execute(
                "UPDATE entities SET status = 'deprecated', archived = 0 WHERE id = ?1",
                rusqlite::params![entity.id],
            )
            .unwrap();
        }

        let mut request = request_for("needle", "task-superseded");
        request.accepted_evidence = vec![canonical_reference(&entity)];
        let error = serve_project_task(
            &db,
            &request,
            "needle",
            None,
            10,
            "agent-a",
            Some("workspace-a"),
        )
        .expect_err("superseded evidence must be rejected");
        assert!(error.contains("superseded"), "unexpected error: {error}");
    }

    #[test]
    fn task_state_rejects_expired_evidence() {
        let db = crate::db::TestDatabase::new("task-state-expired");
        let mut entity = crate::db::tests::make_entity(
            "expired-evidence",
            "facts",
            "expired",
            r#"{"content":"expired deployment evidence","expires_at":1}"#,
        );
        entity.workspace_hash = "workspace-a".to_string();
        entity.agent_id = "agent-a".to_string();
        db.remember_skip_dedup(&entity).unwrap();

        let mut request = request_for("needle", "task-expired");
        request.accepted_evidence = vec![canonical_reference(&entity)];
        let error = serve_project_task(
            &db,
            &request,
            "needle",
            None,
            10,
            "agent-a",
            Some("workspace-a"),
        )
        .expect_err("expired evidence must be rejected");
        assert!(error.contains("expired"), "unexpected error: {error}");
    }

    #[test]
    fn task_state_rejects_evidence_not_valid_at_temporal_anchor() {
        let db = crate::db::TestDatabase::new("task-state-valid-time");
        let anchor = crate::db::now_ms();
        let mut entity = crate::db::tests::make_entity(
            "future-evidence",
            "facts",
            "future",
            r#"{"content":"future deployment evidence"}"#,
        );
        entity.workspace_hash = "workspace-a".to_string();
        entity.agent_id = "agent-a".to_string();
        db.remember_skip_dedup(&entity).unwrap();
        {
            let conn = db.conn().unwrap();
            conn.execute(
                "UPDATE entities SET valid_from_unix_ms = ?1 WHERE id = ?2",
                rusqlite::params![anchor + 60_000, entity.id],
            )
            .unwrap();
        }

        let mut request = request_for("needle", "task-valid-time");
        request.temporal_anchor_unix_ms = Some(anchor);
        request.accepted_evidence = vec![canonical_reference(&entity)];
        let error = serve_project_task(
            &db,
            &request,
            "needle",
            None,
            10,
            "agent-a",
            Some("workspace-a"),
        )
        .expect_err("future evidence must not serve before valid_from");
        assert!(
            error.contains("not valid at temporal anchor"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn task_state_maps_recall_statuses_to_explicit_outcomes() {
        let cases = [
            (RecallStatus::Fresh, 1, 0, 0, 0, TaskStateOutcome::Complete),
            (
                RecallStatus::Partial,
                1,
                0,
                0,
                0,
                TaskStateOutcome::Degraded,
            ),
            (
                RecallStatus::Timeout,
                1,
                0,
                0,
                0,
                TaskStateOutcome::Degraded,
            ),
            (RecallStatus::Stale, 1, 0, 0, 0, TaskStateOutcome::Degraded),
            (
                RecallStatus::Unavailable,
                1,
                0,
                0,
                0,
                TaskStateOutcome::Unavailable,
            ),
            (RecallStatus::Empty, 0, 0, 0, 1, TaskStateOutcome::Abstained),
        ];
        for (status, accepted, rejected, unresolved, missing, expected) in cases {
            let recall = RecallOutcome {
                status,
                ..RecallOutcome::default()
            };
            assert_eq!(
                derive_outcome(accepted, rejected, unresolved, 0, missing, Some(&recall),),
                expected
            );
        }
        assert_eq!(
            derive_outcome(1, 1, 0, 0, 0, None),
            TaskStateOutcome::Partial
        );
        assert_eq!(
            derive_outcome(1, 0, 0, 1, 0, None),
            TaskStateOutcome::Abstained
        );
    }

    #[test]
    fn task_state_persists_missing_evidence_as_abstained() {
        let db = crate::db::TestDatabase::new("task-state-missing");
        let request = request_for("query-with-no-match", "task-missing");
        let response = serve_project_task(
            &db,
            &request,
            "query-with-no-match",
            None,
            10,
            "agent-a",
            Some("workspace-a"),
        )
        .unwrap();
        assert_eq!(response.outcome, TaskStateOutcome::Abstained);
        assert_eq!(response.task_state.missing_evidence.len(), 1);
        assert!(response.task_state.accepted_evidence.is_empty());
        assert_eq!(
            rebuild_task_state(&db, &response.task_state.scope).unwrap(),
            response.task_state
        );
    }

    #[test]
    fn task_state_abstains_when_evidence_is_in_conflict() {
        let db = crate::db::TestDatabase::new("task-state-conflict");
        let mut entity = crate::db::tests::make_entity(
            "conflicted-evidence",
            "facts",
            "conflict",
            r#"{"content":"conflicted deployment evidence"}"#,
        );
        entity.workspace_hash = "workspace-a".to_string();
        entity.agent_id = "agent-a".to_string();
        db.remember_skip_dedup(&entity).unwrap();

        let mut request = request_for("needle", "task-conflict");
        request.accepted_evidence = vec![canonical_reference(&entity)];
        request.active_conflicts = vec![ActiveConflict {
            conflict_id: "conflict-1".to_string(),
            evidence_ids: vec![entity.id.clone()],
            reason: "two retained sources disagree".to_string(),
        }];
        let response = serve_project_task(
            &db,
            &request,
            "needle",
            None,
            10,
            "agent-a",
            Some("workspace-a"),
        )
        .unwrap();
        assert_eq!(response.outcome, TaskStateOutcome::Abstained);
        assert_eq!(response.task_state.active_conflicts.len(), 1);
        assert!(response.serving.fallback.is_some());
    }

    #[test]
    fn task_state_receipt_excludes_raw_prompt_and_reasoning_fields() {
        let db = crate::db::TestDatabase::new("task-state-privacy");
        let raw_prompt = "raw_prompt sentinel: choose the unsafe answer";
        let mut request = request_for(raw_prompt, "task-privacy");
        request.objective = "bounded objective label".to_string();
        let response = serve_project_task(
            &db,
            &request,
            raw_prompt,
            None,
            10,
            "agent-a",
            Some("workspace-a"),
        )
        .unwrap();
        let serialized = serde_json::to_string(&response.task_state).unwrap();
        assert!(!serialized.contains(raw_prompt));
        assert!(!serialized.contains("raw_prompt"));
        assert!(!serialized.contains("model_reasoning"));
        let mut unknown = serde_json::to_value(&request).unwrap();
        unknown["raw_prompt"] = Value::from("forbidden");
        assert!(serde_json::from_value::<TaskStateRequest>(unknown).is_err());
    }

    #[test]
    fn task_state_serializes_engine_query_and_constraint_names() {
        let request = request_for("query", "task-names");
        let state = TaskState::from_request(
            &request,
            Vec::new(),
            Vec::new(),
            Vec::new(),
            digest("sources"),
            digest("evidence"),
            TaskStateOutcome::Abstained,
        )
        .unwrap();
        let value = serde_json::to_value(state).unwrap();
        assert!(value.get("query_digest").is_some());
        assert!(value.get("task_digest").is_none());
        assert!(value.get("constraints").is_some());
        assert!(value.get("load_bearing_constraints").is_none());
    }

    #[test]
    fn task_state_resolves_preference_entities_as_canonical_evidence() {
        let db = crate::db::TestDatabase::new("task-state-preferences");
        let mut entity = crate::db::tests::make_entity(
            "preference-evidence",
            "preferences",
            "response-style",
            r#"{"content":"prefer concise evidence summaries"}"#,
        );
        entity.workspace_hash = "workspace-a".to_string();
        entity.agent_id = "agent-a".to_string();
        db.remember_skip_dedup(&entity).unwrap();

        let mut request = request_for("preference", "task-preferences");
        request.accepted_evidence = vec![canonical_reference(&entity)];
        let response = serve_project_task(
            &db,
            &request,
            "preference",
            None,
            10,
            "agent-a",
            Some("workspace-a"),
        )
        .unwrap();
        assert_eq!(response.outcome, TaskStateOutcome::Complete);
        assert_eq!(response.serving.canonical_sources.len(), 1);
        assert_eq!(
            response.serving.canonical_sources[0].category,
            "preferences"
        );
    }

    #[test]
    fn task_state_reports_unavailable_for_an_unserved_dense_route() {
        let db = crate::db::TestDatabase::new("task-state-unavailable");
        let mut request = request_for("semantic query", "task-unavailable");
        request.route = "dense".to_string();
        let response = serve_project_task(
            &db,
            &request,
            "semantic query",
            None,
            10,
            "agent-a",
            Some("workspace-a"),
        )
        .expect("unavailable retrieval is an explicit response, not a handler error");
        assert!(
            if cfg!(feature = "bundled-embeddings") {
                response.outcome == TaskStateOutcome::Degraded
            } else {
                response.outcome == TaskStateOutcome::Unavailable
            },
            "dense route must never report complete without served semantic evidence: {:?}",
            response.outcome
        );
        assert!(response.task_state.accepted_evidence.is_empty());
        assert!(response.serving.fallback.is_some());
    }

    #[test]
    fn task_state_reports_degraded_for_hybrid_without_semantic_backend() {
        let db = crate::db::TestDatabase::new("task-state-degraded");
        let mut entity = crate::db::tests::make_entity(
            "hybrid-evidence",
            "facts",
            "hybrid",
            r#"{"content":"hybrid deployment evidence"}"#,
        );
        entity.workspace_hash = "workspace-a".to_string();
        entity.agent_id = "agent-a".to_string();
        db.remember_skip_dedup(&entity).unwrap();

        let mut request = request_for("hybrid deployment", "task-degraded");
        request.route = "hybrid".to_string();
        let response = serve_project_task(
            &db,
            &request,
            "hybrid deployment",
            Some("facts"),
            10,
            "agent-a",
            Some("workspace-a"),
        )
        .expect("sparse fallback remains a valid serving response");
        assert_eq!(response.outcome, TaskStateOutcome::Degraded);
        assert!(!response.task_state.accepted_evidence.is_empty());
        assert!(response.serving.fallback.is_some());
    }

    #[test]
    fn task_state_isolated_by_principal_for_the_same_task_id() {
        let db = crate::db::TestDatabase::new("task-state-sessions");
        let mut entity = crate::db::tests::make_entity(
            "shared-session-evidence",
            "facts",
            "shared",
            r#"{"content":"shared session deployment evidence"}"#,
        );
        entity.workspace_hash = "workspace-a".to_string();
        entity.agent_id = "agent-a".to_string();
        entity.visibility = "workspace".to_string();
        db.remember_skip_dedup(&entity).unwrap();

        let first_request = request_for("shared session deployment", "same-task");
        let first = serve_project_task(
            &db,
            &first_request,
            "shared session deployment",
            Some("facts"),
            10,
            "agent-a",
            Some("workspace-a"),
        )
        .unwrap();
        let mut second_request = first_request.clone();
        second_request.principal_id = "agent-b".to_string();
        second_request.agent_id = "agent-b".to_string();
        second_request.base_sequence = 0;
        let second = serve_project_task(
            &db,
            &second_request,
            "shared session deployment",
            Some("facts"),
            10,
            "agent-b",
            Some("workspace-a"),
        )
        .unwrap();

        assert_eq!(
            first.task_state.scope.task_id,
            second.task_state.scope.task_id
        );
        assert_ne!(
            first.task_state.scope.principal_id,
            second.task_state.scope.principal_id
        );
        assert_ne!(
            first.task_state.state_digest,
            second.task_state.state_digest
        );
        assert_eq!(
            db.task_state_get(&first.task_state.scope).unwrap(),
            Some(first.task_state)
        );
        assert_eq!(
            db.task_state_get(&second.task_state.scope).unwrap(),
            Some(second.task_state)
        );
    }

    #[test]
    fn task_state_rebuild_rejects_external_canonical_state_drift() {
        let db = crate::db::TestDatabase::new("task-state-drift");
        let mut entity = crate::db::tests::make_entity(
            "drifting-evidence",
            "facts",
            "drift",
            r#"{"content":"original canonical evidence"}"#,
        );
        entity.workspace_hash = "workspace-a".to_string();
        entity.agent_id = "agent-a".to_string();
        db.remember_skip_dedup(&entity).unwrap();

        let mut request = request_for("drift", "task-drift");
        request.accepted_evidence = vec![canonical_reference(&entity)];
        let response = serve_project_task(
            &db,
            &request,
            "drift",
            Some("facts"),
            10,
            "agent-a",
            Some("workspace-a"),
        )
        .unwrap();
        {
            let conn = db.conn().unwrap();
            conn.execute(
                "UPDATE entities SET body_json = ?1 WHERE id = ?2",
                rusqlite::params![r#"{"content":"drifted canonical evidence"}"#, entity.id],
            )
            .unwrap();
        }
        let error = rebuild_task_state(&db, &response.task_state.scope)
            .expect_err("rebuild must detect canonical source drift");
        assert!(
            error.contains("source digest mismatch"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn task_state_digest_is_stable_and_excludes_digest_field() {
        let request = TaskStateRequest {
            schema_version: TASK_STATE_SCHEMA_VERSION.to_string(),
            task_id: "task-a".to_string(),
            tenant_id: "workspace-a".to_string(),
            workspace_hash: "workspace-a".to_string(),
            principal_id: "agent-a".to_string(),
            agent_id: "agent-a".to_string(),
            task_digest: digest("query"),
            route: "project_task".to_string(),
            objective: "triage".to_string(),
            temporal_anchor_unix_ms: Some(1),
            load_bearing_constraints: vec!["workspace only".to_string()],
            base_sequence: 0,
            observed_input_digest: digest("observation"),
            source_digest: None,
            evidence_digest: None,
            accepted_evidence: Vec::new(),
            rejected_evidence: Vec::new(),
            unresolved_evidence: Vec::new(),
            active_conflicts: Vec::new(),
            missing_evidence: vec![EvidenceSlot {
                slot_id: "slot-a".to_string(),
                reason: "awaiting evidence".to_string(),
            }],
            next_step: NextStepMetadata {
                kind: "review".to_string(),
                reason: "inspect sources".to_string(),
            },
        };
        let state = TaskState::from_request(
            &request,
            Vec::new(),
            Vec::new(),
            request.missing_evidence.clone(),
            digest("sources"),
            digest("evidence"),
            TaskStateOutcome::Abstained,
        )
        .unwrap();
        assert!(state.validate().is_ok());
        assert_eq!(state.state_digest, state.compute_state_digest().unwrap());
        let mut tampered = state.clone();
        tampered.state_digest = "0".repeat(64);
        assert!(tampered.validate().is_err());
    }

    #[test]
    fn task_state_compare_and_swap_rejects_a_stale_base_sequence() {
        let db = crate::db::TestDatabase::new("task-state-cas");
        let request = TaskStateRequest {
            schema_version: TASK_STATE_SCHEMA_VERSION.to_string(),
            task_id: "task-cas".to_string(),
            tenant_id: "workspace-a".to_string(),
            workspace_hash: "workspace-a".to_string(),
            principal_id: "agent-a".to_string(),
            agent_id: "agent-a".to_string(),
            task_digest: digest("query"),
            route: "project_task".to_string(),
            objective: "cas".to_string(),
            temporal_anchor_unix_ms: Some(1),
            load_bearing_constraints: Vec::new(),
            base_sequence: 0,
            observed_input_digest: digest("observation"),
            source_digest: None,
            evidence_digest: None,
            accepted_evidence: Vec::new(),
            rejected_evidence: Vec::new(),
            unresolved_evidence: Vec::new(),
            active_conflicts: Vec::new(),
            missing_evidence: Vec::new(),
            next_step: NextStepMetadata::default(),
        };
        let first = TaskState::from_request(
            &request,
            Vec::new(),
            Vec::new(),
            Vec::new(),
            digest("sources"),
            digest("evidence"),
            TaskStateOutcome::Complete,
        )
        .unwrap();
        db.task_state_compare_and_swap(&first).unwrap();
        let error = db
            .task_state_compare_and_swap(&first)
            .expect_err("the same base sequence cannot be committed twice");
        assert!(
            error.contains("stale base sequence"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn task_state_reader_rejects_tampered_projection_columns() {
        let db = crate::db::TestDatabase::new("task-state-column-tamper");
        let request = request_for("query", "task-column-tamper");
        let state = TaskState::from_request(
            &request,
            Vec::new(),
            Vec::new(),
            Vec::new(),
            digest("sources"),
            digest("evidence"),
            TaskStateOutcome::Abstained,
        )
        .unwrap();
        db.task_state_compare_and_swap(&state).unwrap();
        {
            let conn = db.conn().unwrap();
            conn.execute(
                "UPDATE task_state_projections SET state_digest = ?1
                 WHERE tenant_id = ?2 AND workspace_hash = ?3 AND principal_id = ?4
                   AND agent_id = ?5 AND task_id = ?6",
                rusqlite::params![
                    "0".repeat(64),
                    state.scope.tenant_id,
                    state.scope.workspace_hash,
                    state.scope.principal_id,
                    state.scope.agent_id,
                    state.scope.task_id,
                ],
            )
            .unwrap();
        }
        let error = db
            .task_state_get(&state.scope)
            .expect_err("projection column tampering must fail closed");
        assert!(
            error.contains("stored task-state") || error.contains("state_digest"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn task_state_serving_updates_with_the_current_base_sequence() {
        let db = crate::db::TestDatabase::new("task-state-update");
        let mut entity = crate::db::tests::make_entity(
            "task-state-update-evidence",
            "facts",
            "alpha",
            r#"{\"content\":\"alpha deployment evidence\"}"#,
        );
        entity.workspace_hash = "workspace-a".to_string();
        entity.agent_id = "agent-a".to_string();
        db.remember_skip_dedup(&entity).unwrap();

        let mut request = TaskStateRequest {
            schema_version: TASK_STATE_SCHEMA_VERSION.to_string(),
            task_id: "task-update".to_string(),
            tenant_id: "workspace-a".to_string(),
            workspace_hash: "workspace-a".to_string(),
            principal_id: "agent-a".to_string(),
            agent_id: "agent-a".to_string(),
            task_digest: digest("alpha deployment"),
            route: "project_task".to_string(),
            objective: "deployment review".to_string(),
            temporal_anchor_unix_ms: Some(crate::db::now_ms()),
            load_bearing_constraints: vec!["workspace evidence only".to_string()],
            base_sequence: 0,
            observed_input_digest: digest("observation-1"),
            source_digest: None,
            evidence_digest: None,
            accepted_evidence: Vec::new(),
            rejected_evidence: Vec::new(),
            unresolved_evidence: Vec::new(),
            active_conflicts: Vec::new(),
            missing_evidence: Vec::new(),
            next_step: NextStepMetadata {
                kind: "review".to_string(),
                reason: "check current deployment".to_string(),
            },
        };
        let first = serve_project_task(
            &db,
            &request,
            "alpha deployment",
            Some("facts"),
            10,
            "agent-a",
            Some("workspace-a"),
        )
        .unwrap();
        assert_eq!(first.task_state.state_sequence, 1);
        assert_eq!(first.task_state.accepted_evidence.len(), 1);

        request.base_sequence = 1;
        request.observed_input_digest = digest("observation-2");
        request.next_step.reason = "recheck after observation".to_string();
        let second = serve_project_task(
            &db,
            &request,
            "alpha deployment",
            Some("facts"),
            10,
            "agent-a",
            Some("workspace-a"),
        )
        .unwrap();
        assert_eq!(second.task_state.state_sequence, 2);
        assert_eq!(second.task_state.base_sequence, 1);
        assert_eq!(
            second.task_state.accepted_evidence,
            first.task_state.accepted_evidence
        );
        assert_eq!(
            rebuild_task_state(&db, &second.task_state.scope).unwrap(),
            second.task_state
        );
    }
}
