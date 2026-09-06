//! Versioned, fail-closed context-transformer contract (#1106).
//!
//! This module is deliberately a serving boundary, not a compression backend.
//! A caller may propose a transformed provider request, but the contract decides
//! whether that proposal can be admitted. Durable receipts contain only bounded
//! metadata, identifiers, and digests; the provider-shaped messages stay in the
//! caller's process or in the separately governed original/replay store.

use crate::source_chain::SourceChainIdentity;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

pub const CONTEXT_TRANSFORMER_SCHEMA_VERSION: &str = "perseus-vault-context-transformer/v1";
pub const SUPPORTED_OPENAI_REQUEST_FORMAT: &str = "openai-chat/v1";
pub const TRANSFORM_OUTCOMES: [&str; 5] = [
    "passthrough",
    "transformed",
    "degraded",
    "rejected",
    "unavailable",
];
pub const LOSSINESS_POLICIES: [&str; 3] = ["passthrough", "reversible", "lossy_opt_in"];
pub const ACTUAL_LOSSINESS: [&str; 3] = ["none", "reversible", "lossy"];
pub const TOKEN_COUNT_STATES: [&str; 3] = ["known", "partial", "missing"];
pub const PROTECTED_CONTENT_CLASSES: [&str; 5] = [
    "system",
    "policy",
    "keystone_policy",
    "authority",
    "user_constraint",
];
pub const KNOWN_CONTENT_CLASSES: [&str; 11] = [
    "system",
    "policy",
    "keystone_policy",
    "authority",
    "user_constraint",
    "assistant_prose",
    "tool_use",
    "tool_result",
    "source_code",
    "fenced",
    "multimodal",
];
const MAX_MESSAGES: usize = 512;
const MAX_STAGES: usize = 32;
const MAX_CHANGED_SPANS: usize = 128;
const MAX_IDENTIFIER_LEN: usize = 256;
const MAX_MESSAGE_BYTES: usize = 2 * 1024 * 1024;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TransformerDescriptor {
    pub name: String,
    pub version: String,
}

impl TransformerDescriptor {
    pub fn new(name: impl Into<String>, version: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            version: version.into(),
        }
    }

    fn validate(&self) -> Result<(), String> {
        validate_identifier("transformer.name", &self.name)?;
        validate_identifier("transformer.version", &self.version)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TransformStage {
    pub name: String,
    pub version: String,
    pub enabled: bool,
    /// `trusted` is an explicitly governed stage; `untrusted` is a heuristic
    /// stage and may never rewrite protected content.
    pub trust: String,
    /// Configuration is represented by a commitment, never by arbitrary
    /// provider/config payloads in the receipt.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub config_digest: Option<String>,
}

impl TransformStage {
    pub fn new(
        name: impl Into<String>,
        version: impl Into<String>,
        enabled: bool,
        trust: impl Into<String>,
        config_digest: Option<String>,
    ) -> Self {
        Self {
            name: name.into(),
            version: version.into(),
            enabled,
            trust: trust.into(),
            config_digest,
        }
    }

    fn validate(&self) -> Result<(), String> {
        validate_identifier("stage.name", &self.name)?;
        validate_identifier("stage.version", &self.version)?;
        if self.trust != "trusted" && self.trust != "untrusted" {
            return Err("stage trust must be trusted or untrusted".to_string());
        }
        if let Some(digest) = &self.config_digest {
            validate_sha256(digest)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BoundedReference {
    pub kind: String,
    pub id: String,
    pub digest: String,
}

impl BoundedReference {
    pub fn new(kind: impl Into<String>, id: impl Into<String>, digest: impl Into<String>) -> Self {
        Self {
            kind: kind.into(),
            id: id.into(),
            digest: digest.into(),
        }
    }

    fn validate(&self, label: &str) -> Result<(), String> {
        validate_identifier(&format!("{label}.kind"), &self.kind)?;
        validate_identifier(&format!("{label}.id"), &self.id)?;
        validate_sha256(&self.digest)
    }
}

/// One transient provider-shaped message plus an explicit assembly-side class.
/// The message is accepted for OpenAI-shaped multimodal content, but is never
/// copied into a `ContextTransformReceipt`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ContextMessage {
    pub id: String,
    pub order: u32,
    pub content_class: String,
    pub message: Value,
    /// Optional chain identity from the governed source/assembly layer. It is
    /// metadata only; the provider-shaped message remains separate.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_chain: Option<SourceChainIdentity>,
}

impl ContextMessage {
    pub fn new(
        id: impl Into<String>,
        order: u32,
        content_class: impl Into<String>,
        message: Value,
    ) -> Self {
        Self {
            id: id.into(),
            order,
            content_class: content_class.into(),
            message,
            source_chain: None,
        }
    }

    pub fn with_source_chain(mut self, source_chain: SourceChainIdentity) -> Self {
        self.source_chain = Some(source_chain);
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TokenCounts {
    pub input: Option<u64>,
    pub output: Option<u64>,
    /// `known`, `partial`, or `missing`; absence is never silently represented
    /// as zero.
    pub status: String,
}

impl TokenCounts {
    pub fn new(input: Option<u64>, output: Option<u64>) -> Self {
        let status = match (input.is_some(), output.is_some()) {
            (true, true) => "known",
            (false, false) => "missing",
            _ => "partial",
        };
        Self {
            input,
            output,
            status: status.to_string(),
        }
    }

    fn validate(&self) -> Result<(), String> {
        if !TOKEN_COUNT_STATES.contains(&self.status.as_str()) {
            return Err(format!("unsupported token count status: {}", self.status));
        }
        let expected = match (self.input.is_some(), self.output.is_some()) {
            (true, true) => "known",
            (false, false) => "missing",
            _ => "partial",
        };
        if self.status != expected {
            return Err("token count status does not match supplied values".to_string());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ContextTransformRequest {
    pub provider: String,
    pub request_format: String,
    pub transformer: TransformerDescriptor,
    pub stages: Vec<TransformStage>,
    pub lossiness_policy: String,
    pub input_messages: Vec<ContextMessage>,
    pub input_tokens: Option<u64>,
    /// A retained original must have a bounded locator. A boolean without a
    /// locator is rejected because it cannot be audited or recovered.
    #[serde(default)]
    pub original_retained: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub original_ref: Option<BoundedReference>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub replay_envelope_ref: Option<BoundedReference>,
}

impl ContextTransformRequest {
    pub fn new(
        provider: impl Into<String>,
        request_format: impl Into<String>,
        transformer: TransformerDescriptor,
        stages: Vec<TransformStage>,
        lossiness_policy: impl Into<String>,
        input_messages: Vec<ContextMessage>,
    ) -> Self {
        Self {
            provider: provider.into(),
            request_format: request_format.into(),
            transformer,
            stages,
            lossiness_policy: lossiness_policy.into(),
            input_messages,
            input_tokens: None,
            original_retained: false,
            original_ref: None,
            replay_envelope_ref: None,
        }
    }

    pub fn with_input_tokens(mut self, input_tokens: Option<u64>) -> Self {
        self.input_tokens = input_tokens;
        self
    }

    pub fn with_original_ref(mut self, original_ref: BoundedReference) -> Self {
        self.original_retained = true;
        self.original_ref = Some(original_ref);
        self
    }

    pub fn with_replay_envelope_ref(mut self, replay_envelope_ref: BoundedReference) -> Self {
        self.replay_envelope_ref = Some(replay_envelope_ref);
        self
    }

    pub fn validate(&self) -> Result<(), String> {
        validate_identifier("provider", &self.provider)?;
        validate_identifier("request_format", &self.request_format)?;
        self.transformer.validate()?;
        if self.stages.len() > MAX_STAGES {
            return Err("too many transformer stages".to_string());
        }
        let mut stage_names = BTreeSet::new();
        for stage in &self.stages {
            stage.validate()?;
            if !stage_names.insert(stage.name.as_str()) {
                return Err(format!("duplicate transformer stage: {}", stage.name));
            }
        }
        if !LOSSINESS_POLICIES.contains(&self.lossiness_policy.as_str()) {
            return Err(format!(
                "unsupported lossiness policy: {}",
                self.lossiness_policy
            ));
        }
        if self.original_retained != self.original_ref.is_some() {
            return Err("original_retained must agree with original_ref".to_string());
        }
        if let Some(reference) = &self.original_ref {
            reference.validate("original_ref")?;
        }
        if let Some(reference) = &self.replay_envelope_ref {
            reference.validate("replay_envelope_ref")?;
        }
        validate_message_list(&self.input_messages)?;
        Ok(())
    }

    fn has_enabled_stage(&self) -> bool {
        self.stages.iter().any(|stage| stage.enabled)
    }

    fn has_untrusted_enabled_stage(&self) -> bool {
        self.stages
            .iter()
            .any(|stage| stage.enabled && stage.trust == "untrusted")
    }

    fn recovery_available(&self) -> bool {
        self.original_retained && self.original_ref.is_some() && self.replay_envelope_ref.is_some()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChangedSpan {
    pub message_id: String,
    pub content_class: String,
    pub input_chars: u32,
    pub output_chars: u32,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReplayMembership {
    pub source_id: String,
    pub input_order: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_order: Option<u32>,
    pub content_class: String,
    pub input_digest: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_digest: Option<String>,
    pub disposition: String,
    #[serde(
        default,
        serialize_with = "crate::source_chain::serialize_source_chain_receipt"
    )]
    pub source_chain: SourceChainIdentity,
}

#[derive(Serialize)]
struct LegacyReplayMembership<'a> {
    source_id: &'a str,
    input_order: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    output_order: Option<u32>,
    content_class: &'a str,
    input_digest: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    output_digest: Option<&'a str>,
    disposition: &'a str,
}

fn legacy_replay_fingerprint(membership: &[ReplayMembership]) -> Result<String, String> {
    let legacy: Vec<_> = membership
        .iter()
        .map(|item| LegacyReplayMembership {
            source_id: &item.source_id,
            input_order: item.input_order,
            output_order: item.output_order,
            content_class: &item.content_class,
            input_digest: &item.input_digest,
            output_digest: item.output_digest.as_deref(),
            disposition: &item.disposition,
        })
        .collect();
    canonical_digest(&legacy)
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReplayPlan {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub envelope_ref: Option<BoundedReference>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub original_ref: Option<BoundedReference>,
    pub membership: Vec<ReplayMembership>,
    pub fingerprint: String,
}

impl ReplayPlan {
    fn new(
        envelope_ref: Option<BoundedReference>,
        original_ref: Option<BoundedReference>,
        membership: Vec<ReplayMembership>,
    ) -> Result<Self, String> {
        let fingerprint = canonical_digest(&membership)?;
        Ok(Self {
            envelope_ref,
            original_ref,
            membership,
            fingerprint,
        })
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.membership.is_empty() {
            return Err("replay membership must not be empty".to_string());
        }
        if let Some(reference) = &self.envelope_ref {
            reference.validate("replay.envelope_ref")?;
        }
        if let Some(reference) = &self.original_ref {
            reference.validate("replay.original_ref")?;
        }
        let mut source_ids = BTreeSet::new();
        let mut output_orders = BTreeSet::new();
        for (index, item) in self.membership.iter().enumerate() {
            if item.input_order != index as u32 {
                return Err("replay membership input order must be contiguous".to_string());
            }
            validate_identifier("replay source_id", &item.source_id)?;
            item.source_chain.validate()?;
            validate_content_class(&item.content_class)?;
            validate_sha256(&item.input_digest)?;
            if !source_ids.insert(item.source_id.as_str()) {
                return Err("replay membership source ids must be unique".to_string());
            }
            if let Some(output_order) = item.output_order {
                if !output_orders.insert(output_order) {
                    return Err("replay output order must be unique".to_string());
                }
                if output_order as usize >= self.membership.len() {
                    return Err("replay output order is out of bounds".to_string());
                }
                let output_digest = item
                    .output_digest
                    .as_ref()
                    .ok_or_else(|| "output membership requires output_digest".to_string())?;
                validate_sha256(output_digest)?;
            } else if item.output_digest.is_some() {
                return Err("omitted membership cannot carry output_digest".to_string());
            }
            if !["retained", "transformed", "reordered", "omitted"]
                .contains(&item.disposition.as_str())
            {
                return Err(format!(
                    "unsupported replay disposition: {}",
                    item.disposition
                ));
            }
            if item.output_order.is_none() && item.disposition != "omitted" {
                return Err("only omitted membership may have no output order".to_string());
            }
        }
        let expected_outputs: BTreeSet<u32> = (0..output_orders.len() as u32).collect();
        if output_orders != expected_outputs {
            return Err("replay output order must be contiguous".to_string());
        }
        let current_fingerprint = canonical_digest(&self.membership)?;
        let legacy_fingerprint = if self
            .membership
            .iter()
            .all(|item| item.source_chain.is_unknown())
        {
            Some(legacy_replay_fingerprint(&self.membership)?)
        } else {
            None
        };
        if self.fingerprint != current_fingerprint
            && Some(self.fingerprint.as_str()) != legacy_fingerprint.as_deref()
        {
            return Err("replay fingerprint does not match membership".to_string());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReplayMembershipResult {
    /// Source identities in the exact order admitted to the provider request.
    pub ordered_source_ids: Vec<String>,
    /// Chain commitments aligned with `ordered_source_ids`; unknown identities
    /// are explicit `null` entries rather than hash-looking commitments.
    pub source_chain_commitments: Vec<Option<String>>,
    /// Source identities omitted by the transformed request, in original order.
    pub omitted_source_ids: Vec<String>,
    /// Bounded locator for the retained original/retrieval envelope, if one was
    /// recorded. The body is never returned by this function.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub original_ref: Option<BoundedReference>,
    pub replay_fingerprint: String,
}

/// Replay only the hash-bound membership/order decision against a separately
/// retrieved original. The caller resolves `original_ref` through its governed
/// retrieval/replay store; this function never fetches or returns a raw body.
pub fn replay_membership(
    plan: &ReplayPlan,
    original_messages: &[ContextMessage],
) -> Result<ReplayMembershipResult, String> {
    plan.validate()?;
    validate_message_list(original_messages)?;
    let by_id: BTreeMap<&str, &ContextMessage> = original_messages
        .iter()
        .map(|message| (message.id.as_str(), message))
        .collect();
    let mut ordered: Vec<(u32, String, Option<String>)> = Vec::new();
    let mut omitted = Vec::new();
    for member in &plan.membership {
        let source = by_id
            .get(member.source_id.as_str())
            .ok_or_else(|| "replay source is unavailable".to_string())?;
        if !digest_message_matches_legacy(source, &member.input_digest)? {
            return Err("replay source digest mismatch".to_string());
        }
        let source_chain = source.source_chain.clone().unwrap_or_default();
        if source_chain != member.source_chain {
            return Err("replay source-chain identity mismatch".to_string());
        }
        if let Some(output_order) = member.output_order {
            ordered.push((
                output_order,
                member.source_id.clone(),
                member
                    .source_chain
                    .is_known()
                    .then(|| member.source_chain.commitment().to_string()),
            ));
        } else {
            omitted.push(member.source_id.clone());
        }
    }
    ordered.sort_by_key(|(order, _, _)| *order);
    let ordered_source_ids = ordered.iter().map(|(_, id, _)| id.clone()).collect();
    let source_chain_commitments = ordered
        .into_iter()
        .map(|(_, _, commitment)| commitment)
        .collect();
    Ok(ReplayMembershipResult {
        ordered_source_ids,
        source_chain_commitments,
        omitted_source_ids: omitted,
        original_ref: plan.original_ref.clone(),
        replay_fingerprint: plan.fingerprint.clone(),
    })
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProviderShapeReport {
    pub format: String,
    pub input_valid: bool,
    pub proposed_output_valid: bool,
    pub accepted_output_valid: bool,
    pub pairing_checked: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_code: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ContextTransformReceipt {
    pub schema_version: String,
    pub provider: String,
    pub request_format: String,
    pub transformer: TransformerDescriptor,
    pub stages: Vec<TransformStage>,
    pub lossiness_policy: String,
    pub outcome: String,
    pub actual_lossiness: String,
    pub token_counts: TokenCounts,
    pub input_digest: String,
    pub output_digest: String,
    /// Commitment to the proposed output when it was not admitted. This keeps
    /// a rejected attempt auditable without retaining its body.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub candidate_output_digest: Option<String>,
    pub changed_content_classes: Vec<String>,
    pub changed_spans: Vec<ChangedSpan>,
    pub original_retained: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub original_ref: Option<BoundedReference>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub replay: Option<ReplayPlan>,
    pub provider_shape: ProviderShapeReport,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason_code: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub receipt_digest: Option<String>,
}

impl ContextTransformReceipt {
    pub fn validate(&self) -> Result<(), String> {
        self.validate_without_digest()?;
        if let Some(expected) = &self.receipt_digest {
            let actual = self.compute_digest()?;
            if expected != &actual {
                return Err("receipt_digest does not match canonical receipt".to_string());
            }
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<String, String> {
        self.validate()?;
        self.compute_digest()
    }

    fn seal(mut self) -> Result<Self, String> {
        self.receipt_digest = None;
        self.validate_without_digest()?;
        self.receipt_digest = Some(self.compute_digest()?);
        Ok(self)
    }

    fn validate_without_digest(&self) -> Result<(), String> {
        if self.schema_version != CONTEXT_TRANSFORMER_SCHEMA_VERSION {
            return Err(format!(
                "unsupported schema_version: {}",
                self.schema_version
            ));
        }
        validate_identifier("provider", &self.provider)?;
        validate_identifier("request_format", &self.request_format)?;
        self.transformer.validate()?;
        if self.stages.len() > MAX_STAGES {
            return Err("too many transformer stages".to_string());
        }
        let mut stage_names = BTreeSet::new();
        for stage in &self.stages {
            stage.validate()?;
            if !stage_names.insert(stage.name.as_str()) {
                return Err(format!("duplicate transformer stage: {}", stage.name));
            }
        }
        if !LOSSINESS_POLICIES.contains(&self.lossiness_policy.as_str()) {
            return Err(format!(
                "unsupported lossiness policy: {}",
                self.lossiness_policy
            ));
        }
        if !TRANSFORM_OUTCOMES.contains(&self.outcome.as_str()) {
            return Err(format!("unsupported transform outcome: {}", self.outcome));
        }
        if !ACTUAL_LOSSINESS.contains(&self.actual_lossiness.as_str()) {
            return Err(format!(
                "unsupported actual lossiness: {}",
                self.actual_lossiness
            ));
        }
        self.token_counts.validate()?;
        validate_sha256(&self.input_digest)?;
        validate_sha256(&self.output_digest)?;
        if let Some(digest) = &self.candidate_output_digest {
            validate_sha256(digest)?;
        }
        if let Some(reason) = &self.reason_code {
            validate_identifier("reason_code", reason)?;
        }
        if self.changed_content_classes.len() > MAX_CHANGED_SPANS {
            return Err("too many changed content classes".to_string());
        }
        for class in &self.changed_content_classes {
            validate_content_class(class)?;
        }
        if self.changed_spans.len() > MAX_CHANGED_SPANS {
            return Err("too many changed spans".to_string());
        }
        for span in &self.changed_spans {
            validate_identifier("changed_span.message_id", &span.message_id)?;
            validate_content_class(&span.content_class)?;
            validate_identifier("changed_span.reason", &span.reason)?;
            if span.input_chars > 10_000_000 || span.output_chars > 10_000_000 {
                return Err("changed span is too large".to_string());
            }
        }
        if self.original_retained != self.original_ref.is_some() {
            return Err("original_retained must agree with original_ref".to_string());
        }
        if let Some(reference) = &self.original_ref {
            reference.validate("original_ref")?;
        }
        self.provider_shape.validate()?;
        if let Some(replay) = &self.replay {
            replay.validate()?;
            if replay.original_ref != self.original_ref {
                return Err("replay original_ref does not match receipt original_ref".to_string());
            }
        }
        match self.outcome.as_str() {
            "passthrough" | "rejected" | "unavailable" => {
                if self.actual_lossiness != "none" || self.input_digest != self.output_digest {
                    return Err("non-admitted outcome must be an identity output".to_string());
                }
            }
            "transformed" => {
                if self.actual_lossiness != "reversible" || self.input_digest == self.output_digest
                {
                    return Err(
                        "transformed outcome must be a changed reversible output".to_string()
                    );
                }
            }
            "degraded" => {
                if self.actual_lossiness != "lossy" || self.input_digest == self.output_digest {
                    return Err("degraded outcome must be a changed lossy output".to_string());
                }
            }
            _ => unreachable!(),
        }
        if !self.provider_shape.accepted_output_valid {
            return Err("accepted output must satisfy provider shape".to_string());
        }
        if self.receipt_digest.is_some() {
            validate_sha256(self.receipt_digest.as_ref().unwrap())?;
        }
        Ok(())
    }

    fn compute_digest(&self) -> Result<String, String> {
        let mut unsigned = self.clone();
        unsigned.receipt_digest = None;
        canonical_digest(&unsigned)
    }
}

impl ProviderShapeReport {
    fn validate(&self) -> Result<(), String> {
        validate_identifier("provider_shape.format", &self.format)?;
        if let Some(code) = &self.error_code {
            validate_identifier("provider_shape.error_code", code)?;
        }
        if self.accepted_output_valid == false && self.input_valid {
            return Err("accepted output must be valid for a receipt".to_string());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TransformDecision {
    pub output_messages: Vec<ContextMessage>,
    pub receipt: ContextTransformReceipt,
}

/// Compute the exact digest of the ordered provider-shaped input, including
/// message IDs, assembly classes, and order. JSON object keys are serialized by
/// serde_json in canonical sorted-map order.
pub fn digest_messages(messages: &[ContextMessage]) -> Result<String, String> {
    validate_message_list(messages)?;
    canonical_digest(&messages)
}

/// Compute a content commitment for one message without exposing its body.
pub fn digest_message(message: &ContextMessage) -> Result<String, String> {
    validate_message(message)?;
    canonical_digest(&(
        &message.content_class,
        &message.message,
        &message.source_chain,
    ))
}

fn digest_message_matches_legacy(message: &ContextMessage, expected: &str) -> Result<bool, String> {
    if digest_message(message)? == expected {
        return Ok(true);
    }
    if message
        .source_chain
        .as_ref()
        .is_some_and(SourceChainIdentity::is_known)
    {
        return Ok(false);
    }
    Ok(canonical_digest(&(&message.content_class, &message.message))? == expected)
}

/// Enforce the transformer contract over a proposed provider request.
///
/// The returned `output_messages` are the bytes the provider adapter may send.
/// Rejected, unsupported, or unavailable paths return the original messages;
/// they do not silently send the lossy proposal.
pub fn transform_context(
    request: &ContextTransformRequest,
    proposed_output: Vec<ContextMessage>,
    proposed_output_tokens: Option<u64>,
) -> Result<TransformDecision, String> {
    request.validate()?;
    let input_shape = inspect_provider_shape(&request.request_format, &request.input_messages);
    if !input_shape.valid {
        return Err("input provider shape is invalid".to_string());
    }

    let proposed_basic_valid = validate_message_list(&proposed_output).is_ok();
    let proposed_shape = if proposed_basic_valid {
        inspect_provider_shape(&request.request_format, &proposed_output)
    } else {
        ShapeInspection {
            valid: false,
            pairing_checked: request.request_format == SUPPORTED_OPENAI_REQUEST_FORMAT,
            error_code: Some("provider_shape_invalid".to_string()),
        }
    };
    let diff = if proposed_basic_valid && proposed_shape.valid {
        compare_messages(&request.input_messages, &proposed_output).ok()
    } else {
        None
    };
    let changed = diff.as_ref().map(|item| item.changed).unwrap_or(true);
    let shape_report = |accepted: &[ContextMessage]| ProviderShapeReport {
        format: request.request_format.clone(),
        input_valid: input_shape.valid,
        proposed_output_valid: proposed_shape.valid,
        accepted_output_valid: inspect_provider_shape(&request.request_format, accepted).valid,
        pairing_checked: input_shape.pairing_checked || proposed_shape.pairing_checked,
        error_code: proposed_shape.error_code.clone(),
    };

    if !changed {
        return make_decision(
            request,
            request.input_messages.clone(),
            request.input_tokens,
            None,
            "passthrough",
            "none",
            None,
            shape_report(&request.input_messages),
        );
    }

    if !proposed_shape.valid {
        return make_decision(
            request,
            request.input_messages.clone(),
            request.input_tokens,
            diff,
            "rejected",
            "none",
            Some("provider_shape_invalid"),
            shape_report(&request.input_messages),
        );
    }

    // A new message ID or malformed membership is not something a generic
    // transformer can safely interpret. Unknown/unsupported input defaults to
    // the original context rather than guessing what the stage meant.
    let Some(diff) = diff else {
        return make_decision(
            request,
            request.input_messages.clone(),
            request.input_tokens,
            None,
            "passthrough",
            "none",
            Some("unsupported_membership_passthrough"),
            shape_report(&request.input_messages),
        );
    };

    if request.request_format != SUPPORTED_OPENAI_REQUEST_FORMAT {
        return make_decision(
            request,
            request.input_messages.clone(),
            request.input_tokens,
            Some(diff),
            "passthrough",
            "none",
            Some("unsupported_provider_format_passthrough"),
            shape_report(&request.input_messages),
        );
    }
    if !request.has_enabled_stage() {
        return make_decision(
            request,
            request.input_messages.clone(),
            request.input_tokens,
            Some(diff),
            "passthrough",
            "none",
            Some("no_enabled_transform_stage"),
            shape_report(&request.input_messages),
        );
    }
    if diff
        .changed_classes
        .iter()
        .any(|class| !KNOWN_CONTENT_CLASSES.contains(&class.as_str()))
    {
        return make_decision(
            request,
            request.input_messages.clone(),
            request.input_tokens,
            Some(diff),
            "passthrough",
            "none",
            Some("unknown_content_passthrough"),
            shape_report(&request.input_messages),
        );
    }
    if request.lossiness_policy == "passthrough" {
        return make_decision(
            request,
            request.input_messages.clone(),
            request.input_tokens,
            Some(diff),
            "passthrough",
            "none",
            Some("lossiness_policy_passthrough"),
            shape_report(&request.input_messages),
        );
    }

    let protected_changed = diff
        .changed_classes
        .iter()
        .any(|class| PROTECTED_CONTENT_CLASSES.contains(&class.as_str()));
    if protected_changed && request.has_untrusted_enabled_stage() {
        return make_decision(
            request,
            request.input_messages.clone(),
            request.input_tokens,
            Some(diff),
            "rejected",
            "none",
            Some("protected_content_untrusted_stage"),
            shape_report(&request.input_messages),
        );
    }

    let recovery_available = request.recovery_available();
    if request.lossiness_policy == "reversible" && !recovery_available {
        return make_decision(
            request,
            request.input_messages.clone(),
            request.input_tokens,
            Some(diff),
            "rejected",
            "none",
            Some("reversible_transform_missing_recovery_reference"),
            shape_report(&request.input_messages),
        );
    }
    // Protected content may be transformed only by a trusted stage and only
    // when the exact original can be located again, even if the caller asks
    // for lossy opt-in. This prevents policy/user constraints from becoming a
    // permanently lossy prompt projection.
    if protected_changed && !recovery_available {
        return make_decision(
            request,
            request.input_messages.clone(),
            request.input_tokens,
            Some(diff),
            "rejected",
            "none",
            Some("protected_content_missing_recovery_reference"),
            shape_report(&request.input_messages),
        );
    }
    let actual_lossiness = if recovery_available {
        "reversible"
    } else {
        "lossy"
    };
    let outcome = if actual_lossiness == "reversible" {
        "transformed"
    } else {
        "degraded"
    };
    make_decision(
        request,
        proposed_output,
        proposed_output_tokens,
        Some(diff),
        outcome,
        actual_lossiness,
        Some(if actual_lossiness == "lossy" {
            "lossy_opt_in"
        } else {
            "reversible_transform_admitted"
        }),
        shape_report(&request.input_messages),
    )
}

/// Explicitly record an unavailable transformer/backend. This is distinct from
/// a valid empty/passthrough context and is therefore scoreable by adapters.
pub fn unavailable_context(
    request: &ContextTransformRequest,
    reason_code: impl Into<String>,
) -> Result<TransformDecision, String> {
    request.validate()?;
    let shape = inspect_provider_shape(&request.request_format, &request.input_messages);
    if !shape.valid {
        return Err("input provider shape is invalid".to_string());
    }
    let report = ProviderShapeReport {
        format: request.request_format.clone(),
        input_valid: true,
        proposed_output_valid: true,
        accepted_output_valid: true,
        pairing_checked: shape.pairing_checked,
        error_code: None,
    };
    let reason_code = reason_code.into();
    make_decision(
        request,
        request.input_messages.clone(),
        request.input_tokens,
        None,
        "unavailable",
        "none",
        Some(&reason_code),
        report,
    )
}

/// Alias used by adapters that call the operation an evaluation rather than a
/// transform. Keeping this pure avoids a second integration contract.
pub fn evaluate_transform(
    request: &ContextTransformRequest,
    proposed_output: Vec<ContextMessage>,
    proposed_output_tokens: Option<u64>,
) -> Result<TransformDecision, String> {
    transform_context(request, proposed_output, proposed_output_tokens)
}

fn make_decision(
    request: &ContextTransformRequest,
    accepted_output: Vec<ContextMessage>,
    accepted_output_tokens: Option<u64>,
    diff: Option<Diff>,
    outcome: &str,
    actual_lossiness: &str,
    reason_code: Option<&str>,
    provider_shape: ProviderShapeReport,
) -> Result<TransformDecision, String> {
    let input_digest = digest_messages(&request.input_messages)?;
    let output_digest = digest_messages(&accepted_output)?;
    let candidate_output_digest = diff
        .as_ref()
        .and_then(|item| item.candidate_digest.clone())
        .filter(|digest| digest != &output_digest);
    let changed_content_classes = diff
        .as_ref()
        .map(|item| item.changed_classes.clone())
        .unwrap_or_default();
    let changed_spans = diff
        .as_ref()
        .map(|item| item.changed_spans.clone())
        .unwrap_or_default();
    let replay = if let Some(item) = diff.as_ref().filter(|item| item.changed) {
        Some(ReplayPlan::new(
            request.replay_envelope_ref.clone(),
            request.original_ref.clone(),
            item.membership.clone(),
        )?)
    } else {
        None
    };
    let token_counts = if input_digest == output_digest {
        TokenCounts::new(request.input_tokens, request.input_tokens)
    } else {
        TokenCounts::new(request.input_tokens, accepted_output_tokens)
    };
    let receipt = ContextTransformReceipt {
        schema_version: CONTEXT_TRANSFORMER_SCHEMA_VERSION.to_string(),
        provider: request.provider.clone(),
        request_format: request.request_format.clone(),
        transformer: request.transformer.clone(),
        stages: request.stages.clone(),
        lossiness_policy: request.lossiness_policy.clone(),
        outcome: outcome.to_string(),
        actual_lossiness: actual_lossiness.to_string(),
        token_counts,
        input_digest,
        output_digest,
        candidate_output_digest,
        changed_content_classes,
        changed_spans,
        original_retained: request.original_retained,
        original_ref: request.original_ref.clone(),
        replay,
        provider_shape,
        reason_code: reason_code.map(str::to_string),
        receipt_digest: None,
    }
    .seal()?;
    Ok(TransformDecision {
        output_messages: accepted_output,
        receipt,
    })
}

#[derive(Debug, Clone)]
struct ShapeInspection {
    valid: bool,
    pairing_checked: bool,
    error_code: Option<String>,
}

fn inspect_provider_shape(format: &str, messages: &[ContextMessage]) -> ShapeInspection {
    if format != SUPPORTED_OPENAI_REQUEST_FORMAT {
        return ShapeInspection {
            valid: validate_message_list(messages).is_ok(),
            pairing_checked: false,
            error_code: None,
        };
    }
    match validate_openai_messages(messages) {
        Ok(()) => ShapeInspection {
            valid: true,
            pairing_checked: true,
            error_code: None,
        },
        Err(code) => ShapeInspection {
            valid: false,
            pairing_checked: true,
            error_code: Some(code),
        },
    }
}

fn validate_message_list(messages: &[ContextMessage]) -> Result<(), String> {
    if messages.len() > MAX_MESSAGES {
        return Err("too many context messages".to_string());
    }
    let mut ids = BTreeSet::new();
    for (index, message) in messages.iter().enumerate() {
        if message.order != index as u32 {
            return Err("context message order must be contiguous".to_string());
        }
        validate_message(message)?;
        if !ids.insert(message.id.as_str()) {
            return Err("context message ids must be unique".to_string());
        }
    }
    Ok(())
}

fn validate_message(message: &ContextMessage) -> Result<(), String> {
    validate_identifier("message.id", &message.id)?;
    validate_content_class(&message.content_class)?;
    if let Some(source_chain) = &message.source_chain {
        source_chain.validate()?;
    }
    if !message.message.is_object() {
        return Err("provider message must be an object".to_string());
    }
    let bytes = serde_json::to_vec(&message.message)
        .map_err(|e| format!("message serialization failed: {e}"))?;
    if bytes.len() > MAX_MESSAGE_BYTES {
        return Err("provider message exceeds bounded size".to_string());
    }
    Ok(())
}

fn validate_content_class(class: &str) -> Result<(), String> {
    validate_identifier("content_class", class)
}

fn validate_openai_messages(messages: &[ContextMessage]) -> Result<(), String> {
    let mut tool_calls = BTreeSet::new();
    let mut tool_results = BTreeSet::new();
    for message in messages {
        let object = message
            .message
            .as_object()
            .ok_or_else(|| "provider_shape_invalid".to_string())?;
        let role = object
            .get("role")
            .and_then(Value::as_str)
            .ok_or_else(|| "provider_shape_missing_role".to_string())?;
        match role {
            "system" | "developer" | "user" | "assistant" | "tool" => {}
            _ => return Err("provider_shape_unknown_role".to_string()),
        }
        if let Some(content) = object.get("content") {
            validate_openai_content(content)?;
        }
        if role == "tool" {
            let call_id = object
                .get("tool_call_id")
                .and_then(Value::as_str)
                .filter(|value| !value.is_empty())
                .ok_or_else(|| "provider_shape_missing_tool_call_id".to_string())?;
            if !tool_results.insert(call_id) || !tool_calls.contains(call_id) {
                return Err("provider_shape_tool_pairing".to_string());
            }
        }
        if let Some(calls) = object.get("tool_calls") {
            if role != "assistant" {
                return Err("provider_shape_tool_calls_role".to_string());
            }
            let calls = calls
                .as_array()
                .ok_or_else(|| "provider_shape_tool_calls_not_array".to_string())?;
            for call in calls {
                let call_object = call
                    .as_object()
                    .ok_or_else(|| "provider_shape_tool_call_invalid".to_string())?;
                let id = call_object
                    .get("id")
                    .and_then(Value::as_str)
                    .filter(|value| !value.is_empty())
                    .ok_or_else(|| "provider_shape_tool_call_missing_id".to_string())?;
                if !tool_calls.insert(id) {
                    return Err("provider_shape_tool_pairing".to_string());
                }
                let function = call_object
                    .get("function")
                    .and_then(Value::as_object)
                    .ok_or_else(|| "provider_shape_tool_function_invalid".to_string())?;
                if function.get("name").and_then(Value::as_str).is_none()
                    || function.get("arguments").and_then(Value::as_str).is_none()
                {
                    return Err("provider_shape_tool_function_fields".to_string());
                }
            }
        }
    }
    if tool_calls != tool_results {
        return Err("provider_shape_tool_pairing".to_string());
    }
    Ok(())
}

fn validate_openai_content(content: &Value) -> Result<(), String> {
    match content {
        Value::Null | Value::String(_) => Ok(()),
        Value::Array(parts) => {
            for part in parts {
                let object = part
                    .as_object()
                    .ok_or_else(|| "provider_shape_multimodal_part".to_string())?;
                let kind = object
                    .get("type")
                    .and_then(Value::as_str)
                    .filter(|value| !value.is_empty())
                    .ok_or_else(|| "provider_shape_multimodal_type".to_string())?;
                match kind {
                    "text" | "input_text" => {
                        if object.get("text").and_then(Value::as_str).is_none() {
                            return Err("provider_shape_multimodal_text".to_string());
                        }
                    }
                    "image_url" | "input_image" => {
                        if !object.contains_key("image_url")
                            && !object.contains_key("image_url_ref")
                        {
                            return Err("provider_shape_multimodal_image".to_string());
                        }
                    }
                    _ => {}
                }
            }
            Ok(())
        }
        _ => Err("provider_shape_content_type".to_string()),
    }
}

#[derive(Debug, Clone)]
struct Diff {
    changed: bool,
    changed_classes: Vec<String>,
    changed_spans: Vec<ChangedSpan>,
    membership: Vec<ReplayMembership>,
    candidate_digest: Option<String>,
}

fn compare_messages(input: &[ContextMessage], output: &[ContextMessage]) -> Result<Diff, String> {
    let input_by_id: BTreeMap<&str, &ContextMessage> =
        input.iter().map(|item| (item.id.as_str(), item)).collect();
    let output_by_id: BTreeMap<&str, &ContextMessage> =
        output.iter().map(|item| (item.id.as_str(), item)).collect();
    if output_by_id.len() != output.len()
        || output_by_id.keys().any(|id| !input_by_id.contains_key(id))
    {
        return Err("unsupported_membership".to_string());
    }
    let mut classes = BTreeSet::new();
    let mut spans = Vec::new();
    let mut membership = Vec::with_capacity(input.len());
    let mut changed = false;
    for source in input {
        let source_digest = digest_message(source)?;
        let Some(candidate) = output_by_id.get(source.id.as_str()) else {
            changed = true;
            classes.insert(source.content_class.clone());
            spans.push(ChangedSpan {
                message_id: source.id.clone(),
                content_class: source.content_class.clone(),
                input_chars: message_chars(source),
                output_chars: 0,
                reason: "omitted".to_string(),
            });
            membership.push(ReplayMembership {
                source_id: source.id.clone(),
                input_order: source.order,
                output_order: None,
                content_class: source.content_class.clone(),
                input_digest: source_digest,
                output_digest: None,
                disposition: "omitted".to_string(),
                source_chain: source.source_chain.clone().unwrap_or_default(),
            });
            continue;
        };
        let candidate_digest = digest_message(candidate)?;
        let same_content =
            source.content_class == candidate.content_class && source.message == candidate.message;
        let same_chain = source.source_chain == candidate.source_chain;
        let same_order = source.order == candidate.order;
        let disposition = if same_content && same_order && same_chain {
            "retained"
        } else if same_content && same_order {
            "transformed"
        } else if same_content {
            "reordered"
        } else {
            "transformed"
        };
        if disposition != "retained" {
            changed = true;
            classes.insert(source.content_class.clone());
            if source.content_class != candidate.content_class {
                classes.insert(candidate.content_class.clone());
            }
            spans.push(ChangedSpan {
                message_id: source.id.clone(),
                content_class: source.content_class.clone(),
                input_chars: message_chars(source),
                output_chars: message_chars(candidate),
                reason: disposition.to_string(),
            });
        }
        membership.push(ReplayMembership {
            source_id: source.id.clone(),
            input_order: source.order,
            output_order: Some(candidate.order),
            content_class: source.content_class.clone(),
            input_digest: source_digest,
            output_digest: Some(candidate_digest),
            disposition: disposition.to_string(),
            source_chain: source.source_chain.clone().unwrap_or_default(),
        });
    }
    if spans.len() > MAX_CHANGED_SPANS {
        return Err("too_many_changed_spans".to_string());
    }
    Ok(Diff {
        changed,
        changed_classes: classes.into_iter().collect(),
        changed_spans: spans,
        membership,
        candidate_digest: Some(digest_messages(output)?),
    })
}

fn message_chars(message: &ContextMessage) -> u32 {
    serde_json::to_string(&message.message)
        .map(|text| text.chars().count().min(u32::MAX as usize) as u32)
        .unwrap_or(0)
}

fn canonical_digest<T: Serialize>(value: &T) -> Result<String, String> {
    let bytes =
        serde_json::to_vec(value).map_err(|e| format!("canonical serialization failed: {e}"))?;
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    Ok(format!("{:x}", hasher.finalize()))
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
        || value.len() > MAX_IDENTIFIER_LEN
        || value
            .chars()
            .any(|character| character.is_control() || character.is_whitespace())
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

    fn sample_request() -> ContextTransformRequest {
        ContextTransformRequest::new(
            "openai",
            SUPPORTED_OPENAI_REQUEST_FORMAT,
            TransformerDescriptor::new("test", "1"),
            vec![TransformStage::new("stage", "1", true, "trusted", None)],
            "passthrough",
            vec![ContextMessage::new(
                "m-1",
                0,
                "assistant_prose",
                serde_json::json!({"role": "assistant", "content": "hello"}),
            )],
        )
    }

    #[test]
    fn receipt_digest_is_hash_only_and_tamper_evident() {
        let request = sample_request();
        let decision =
            transform_context(&request, request.input_messages.clone(), Some(1)).unwrap();
        let mut receipt = decision.receipt;
        assert_eq!(receipt.digest().unwrap().len(), 64);
        receipt.output_digest = "a".repeat(64);
        assert!(receipt.validate().is_err());
    }

    #[test]
    fn openai_shape_checks_tool_pairing_and_multimodal_parts() {
        let mut request = sample_request();
        request.input_messages = vec![ContextMessage::new(
            "m-1",
            0,
            "multimodal",
            serde_json::json!({
                "role": "user",
                "content": [{"type": "text", "text": "hello"}, {"type": "image_url", "image_url": {"url": "fixture"}}]
            }),
        )];
        let result = transform_context(&request, request.input_messages.clone(), None).unwrap();
        assert!(result.receipt.provider_shape.pairing_checked);
        assert!(result.receipt.provider_shape.accepted_output_valid);
    }

    #[test]
    fn unavailable_is_not_an_empty_success() {
        let request = sample_request();
        let decision = unavailable_context(&request, "provider_unavailable").unwrap();
        assert_eq!(decision.receipt.outcome, "unavailable");
        assert_eq!(
            decision.receipt.reason_code.as_deref(),
            Some("provider_unavailable")
        );
        assert!(decision.receipt.validate().is_ok());
    }
}
