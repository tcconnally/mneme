//! Versioned, hash-only runtime-stage trace contract (#822).
//!
//! This module intentionally does not know about transport or policy execution.
//! It provides the stable record/validation boundary that those layers can emit
//! and that offline benchmark consumers can replay without receiving raw
//! prompts, memory bodies, credentials, or tool payloads.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

pub const STAGE_TRACE_SCHEMA_VERSION: &str = "perseus-vault-stage-trace/v1";
pub const STAGE_VOCABULARY: [&str; 8] = [
    "context_candidate_generation",
    "context_selection",
    "validation_provenance",
    "policy_evaluation",
    "mediation_escalation",
    "tool_execution",
    "recovery",
    "receipt_persistence",
];
pub const STAGE_OUTCOMES: [&str; 7] = [
    "in_progress",
    "completed",
    "degraded",
    "abstained",
    "failed",
    "timeout",
    "skipped",
];
const CAUSAL_KEYS: [&str; 6] = [
    "context_digest",
    "decision_digest",
    "action_id",
    "lease_id",
    "receipt_id",
    "parent_trace_digest",
];

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct StageRecord {
    pub stage: String,
    pub sequence: u32,
    pub started_at_unix_ms: i64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ended_at_unix_ms: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deadline_class: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub priority_class: Option<String>,
    pub outcome: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_provider: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_digest: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_digest: Option<String>,
    pub workspace_hash: String,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub causal_links: BTreeMap<String, String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason_code: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct StageTrace {
    pub schema_version: String,
    pub trace_id: String,
    pub workspace_hash: String,
    pub stages: Vec<StageRecord>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trace_digest: Option<String>,
}

impl StageTrace {
    pub fn new(trace_id: impl Into<String>, workspace_hash: impl Into<String>) -> Self {
        Self {
            schema_version: STAGE_TRACE_SCHEMA_VERSION.to_string(),
            trace_id: trace_id.into(),
            workspace_hash: workspace_hash.into(),
            stages: Vec::new(),
            trace_digest: None,
        }
    }

    pub fn push(&mut self, stage: StageRecord) -> Result<(), String> {
        let expected = self.stages.len() as u32;
        if stage.sequence != expected {
            return Err(format!(
                "stage sequence must be {expected}, got {}",
                stage.sequence
            ));
        }
        self.stages.push(stage);
        self.trace_digest = None;
        Ok(())
    }

    pub fn seal(mut self) -> Result<Self, String> {
        self.validate_without_digest()?;
        self.trace_digest = Some(self.compute_digest()?);
        Ok(self)
    }

    pub fn validate(&self) -> Result<(), String> {
        self.validate_without_digest()?;
        if let Some(expected) = &self.trace_digest {
            let actual = self.compute_digest()?;
            if expected != &actual {
                return Err("trace_digest does not match canonical trace".to_string());
            }
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<String, String> {
        self.validate()?;
        self.compute_digest()
    }

    /// Compare replay semantics while deliberately ignoring wall-clock times.
    /// This lets an offline replay prove the same ordered decisions without
    /// pretending that it ran at the same time or latency.
    pub fn replay_fingerprint(&self) -> Result<String, String> {
        self.validate()?;
        let material: Vec<ReplayStage<'_>> = self.stages.iter().map(ReplayStage::from).collect();
        canonical_digest(&(
            self.schema_version.as_str(),
            self.workspace_hash.as_str(),
            material,
        ))
    }

    pub fn validate_replay(expected: &Self, actual: &Self) -> Result<(), String> {
        let left = expected.replay_fingerprint()?;
        let right = actual.replay_fingerprint()?;
        if left == right {
            Ok(())
        } else {
            Err("replay fingerprint mismatch".to_string())
        }
    }

    fn validate_without_digest(&self) -> Result<(), String> {
        if self.schema_version != STAGE_TRACE_SCHEMA_VERSION {
            return Err(format!(
                "unsupported schema_version: {}",
                self.schema_version
            ));
        }
        validate_identifier("trace_id", &self.trace_id)?;
        validate_identifier("workspace_hash", &self.workspace_hash)?;
        let mut seen = BTreeSet::new();
        for (index, stage) in self.stages.iter().enumerate() {
            if stage.sequence != index as u32 {
                return Err(format!("stages must be ordered and contiguous at {index}"));
            }
            if !STAGE_VOCABULARY.contains(&stage.stage.as_str()) {
                return Err(format!("unsupported stage: {}", stage.stage));
            }
            if !seen.insert(stage.stage.as_str()) {
                return Err(format!("duplicate stage: {}", stage.stage));
            }
            if stage.started_at_unix_ms < 0 {
                return Err("started_at_unix_ms must be non-negative".to_string());
            }
            if let Some(end) = stage.ended_at_unix_ms {
                if end < stage.started_at_unix_ms {
                    return Err(format!("stage {} ends before it starts", stage.stage));
                }
            } else if stage.outcome != "in_progress" {
                return Err(format!(
                    "incomplete stage {} must be in_progress",
                    stage.stage
                ));
            }
            if !STAGE_OUTCOMES.contains(&stage.outcome.as_str()) {
                return Err(format!("unsupported outcome: {}", stage.outcome));
            }
            validate_identifier("stage.workspace_hash", &stage.workspace_hash)?;
            if stage.workspace_hash != self.workspace_hash {
                return Err(format!("stage {} crosses workspace scope", stage.stage));
            }
            for digest in [&stage.input_digest, &stage.output_digest] {
                if let Some(value) = digest {
                    validate_sha256(value)?;
                }
            }
            for (key, value) in &stage.causal_links {
                if !CAUSAL_KEYS.contains(&key.as_str()) {
                    return Err(format!("unsupported causal link key: {key}"));
                }
                validate_identifier("causal link", value)?;
                if key.ends_with("digest") {
                    validate_sha256(value)?;
                }
            }
            for value in [
                &stage.deadline_class,
                &stage.priority_class,
                &stage.model_provider,
                &stage.reason_code,
            ] {
                if let Some(value) = value {
                    validate_identifier("stage metadata", value)?;
                }
            }
        }
        if let Some(digest) = &self.trace_digest {
            validate_sha256(digest)?;
        }
        Ok(())
    }

    fn compute_digest(&self) -> Result<String, String> {
        let material = (
            &self.schema_version,
            &self.trace_id,
            &self.workspace_hash,
            &self.stages,
        );
        canonical_digest(&material)
    }
}

#[derive(Serialize)]
struct ReplayStage<'a> {
    stage: &'a str,
    sequence: u32,
    outcome: &'a str,
    model_provider: &'a Option<String>,
    input_tokens: &'a Option<u64>,
    output_tokens: &'a Option<u64>,
    input_digest: &'a Option<String>,
    output_digest: &'a Option<String>,
    causal_links: &'a BTreeMap<String, String>,
    reason_code: &'a Option<String>,
}

impl<'a> From<&'a StageRecord> for ReplayStage<'a> {
    fn from(stage: &'a StageRecord) -> Self {
        Self {
            stage: &stage.stage,
            sequence: stage.sequence,
            outcome: &stage.outcome,
            model_provider: &stage.model_provider,
            input_tokens: &stage.input_tokens,
            output_tokens: &stage.output_tokens,
            input_digest: &stage.input_digest,
            output_digest: &stage.output_digest,
            causal_links: &stage.causal_links,
            reason_code: &stage.reason_code,
        }
    }
}

fn canonical_digest<T: Serialize>(value: &T) -> Result<String, String> {
    let bytes =
        serde_json::to_vec(value).map_err(|e| format!("trace serialization failed: {e}"))?;
    let mut digest = Sha256::new();
    digest.update(bytes);
    Ok(format!("{:x}", digest.finalize()))
}

fn validate_sha256(value: &str) -> Result<(), String> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err("digest must be a lowercase SHA-256 value".to_string());
    }
    Ok(())
}

fn validate_identifier(label: &str, value: &str) -> Result<(), String> {
    if value.is_empty()
        || value.len() > 256
        || value.chars().any(|c| c.is_control() || c.is_whitespace())
    {
        return Err(format!(
            "{label} must be a bounded non-whitespace identifier"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(seed: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(seed.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    fn stage(name: &str, sequence: u32, outcome: &str) -> StageRecord {
        StageRecord {
            stage: name.to_string(),
            sequence,
            started_at_unix_ms: 10 + sequence as i64,
            ended_at_unix_ms: Some(20 + sequence as i64),
            deadline_class: Some("interactive".to_string()),
            priority_class: Some("normal".to_string()),
            outcome: outcome.to_string(),
            model_provider: Some("fixture-provider".to_string()),
            input_tokens: Some(10),
            output_tokens: Some(5),
            input_digest: Some(digest("input")),
            output_digest: Some(digest("output")),
            workspace_hash: "workspace-a".to_string(),
            causal_links: BTreeMap::new(),
            reason_code: None,
        }
    }

    fn complete_trace() -> StageTrace {
        let mut trace = StageTrace::new("trace-1", "workspace-a");
        trace
            .push(stage("context_candidate_generation", 0, "completed"))
            .unwrap();
        trace
            .push(stage("context_selection", 1, "completed"))
            .unwrap();
        trace
            .push(stage("validation_provenance", 2, "completed"))
            .unwrap();
        trace.seal().unwrap()
    }

    #[test]
    fn seals_and_validates_hash_only_trace() {
        let trace = complete_trace();
        assert_eq!(trace.trace_digest.as_ref().unwrap().len(), 64);
        assert_eq!(trace.digest().unwrap(), trace.trace_digest.clone().unwrap());
    }

    #[test]
    fn timeout_and_partial_stage_are_explicit() {
        let mut trace = StageTrace::new("trace-timeout", "workspace-a");
        let mut item = stage("policy_evaluation", 0, "timeout");
        item.ended_at_unix_ms = Some(item.started_at_unix_ms);
        trace.push(item).unwrap();
        assert!(trace.seal().is_ok());

        let mut partial = StageTrace::new("trace-partial", "workspace-a");
        let mut item = stage("tool_execution", 0, "in_progress");
        item.ended_at_unix_ms = None;
        partial.push(item).unwrap();
        assert!(partial.seal().is_ok());
    }

    #[test]
    fn rejects_reordered_duplicate_and_cross_scope_stages() {
        let mut trace = StageTrace::new("trace-invalid", "workspace-a");
        assert!(trace
            .push(stage("context_selection", 1, "completed"))
            .is_err());
        trace
            .push(stage("context_candidate_generation", 0, "completed"))
            .unwrap();
        assert!(trace
            .push(stage("context_candidate_generation", 1, "completed"))
            .is_ok());
        assert!(trace.validate().is_err());

        let mut scope = complete_trace();
        scope.stages[0].workspace_hash = "workspace-b".to_string();
        scope.trace_digest = None;
        assert!(scope.validate().is_err());
    }

    #[test]
    fn detects_tampering_and_replay_mismatch() {
        let expected = complete_trace();
        let mut tampered = expected.clone();
        tampered.stages[1].output_tokens = Some(999);
        assert!(tampered.validate().is_err());
        assert!(StageTrace::validate_replay(&expected, &tampered).is_err());
    }

    #[test]
    fn rejects_raw_like_causal_keys_and_uppercase_digest() {
        let mut trace = complete_trace();
        trace.stages[0]
            .causal_links
            .insert("raw_prompt".to_string(), "x".to_string());
        trace.trace_digest = None;
        assert!(trace.validate().is_err());
        assert!(validate_sha256(&"A".repeat(64)).is_err());
    }
}
