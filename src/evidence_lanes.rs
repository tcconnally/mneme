use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap, HashSet};

use crate::models::Entity;
use crate::source_chain::SourceChainIdentity;
use rusqlite::OptionalExtension;

pub const EVIDENCE_RECEIPT_SCHEMA_VERSION: u32 = 1;
const MAX_EVIDENCE_TOKENS: i64 = 65_536;
const MAX_SOURCE_BODY_BYTES: usize = 4 * 1024 * 1024;
const MAX_SOURCE_SPAN_CHARS: usize = MAX_EVIDENCE_TOKENS as usize * 4;
const MAX_RESIDUAL_ITEMS: i64 = 256;

/// The two governed evidence representations exposed by #1135.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceLane {
    Derived,
    Verbatim,
}

impl EvidenceLane {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "derived" => Ok(Self::Derived),
            "verbatim" => Ok(Self::Verbatim),
            other => Err(format!(
                "unknown evidence lane {other:?}; expected derived or verbatim"
            )),
        }
    }

    fn order(self) -> u8 {
        match self {
            Self::Derived => 0,
            Self::Verbatim => 1,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Derived => "derived",
            Self::Verbatim => "verbatim",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LaneSelection {
    lanes: Vec<EvidenceLane>,
}

impl LaneSelection {
    pub fn lanes(&self) -> &[EvidenceLane] {
        &self.lanes
    }

    pub fn contains(&self, lane: EvidenceLane) -> bool {
        self.lanes.contains(&lane)
    }
}

/// Parse and canonicalize an explicit lane selection.
///
/// The caller distinguishes an omitted field (`None`) from an explicit JSON
/// array. An explicit array must contain at least one known lane; duplicate
/// lane names are harmless and collapse into the canonical order.
pub fn parse_lane_selection(value: &serde_json::Value) -> Result<LaneSelection, String> {
    let values = value
        .as_array()
        .ok_or_else(|| "evidence_lanes must be an array".to_string())?;
    if values.is_empty() {
        return Err("evidence_lanes must contain at least one lane".to_string());
    }
    let mut lanes = Vec::with_capacity(values.len());
    for value in values {
        let name = value
            .as_str()
            .ok_or_else(|| "evidence_lanes entries must be strings".to_string())?;
        let lane = EvidenceLane::parse(name)?;
        if !lanes.contains(&lane) {
            lanes.push(lane);
        }
    }
    lanes.sort_by_key(|lane| lane.order());
    Ok(LaneSelection { lanes })
}

/// Stable identity for one retained source revision and character span.
///
/// `start_char` and `end_char` are Unicode scalar-value offsets, matching the
/// existing capture/expand-source contract. `content_sha256` is empty only
/// when the source has no expected digest; a non-empty value must be a
/// lowercase SHA-256 digest. The identity never contains raw source text.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceGroup {
    pub source_id: String,
    pub revision: String,
    pub start_char: usize,
    pub end_char: usize,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub content_sha256: String,
}

impl SourceGroup {
    pub fn new(
        source_id: impl Into<String>,
        revision: impl Into<String>,
        start_char: usize,
        end_char: usize,
        content_sha256: impl Into<String>,
    ) -> Result<Self, String> {
        let source_id = source_id.into();
        let revision = revision.into();
        let content_sha256 = content_sha256.into();
        if source_id.trim().is_empty() {
            return Err("source group source_id must not be empty".to_string());
        }
        if revision.trim().is_empty() {
            return Err("source group revision must not be empty".to_string());
        }
        if start_char > end_char {
            return Err("source group span must satisfy start_char <= end_char".to_string());
        }
        if !content_sha256.is_empty() && !is_sha256(&content_sha256) {
            return Err(
                "source group content_sha256 must be a lowercase SHA-256 digest".to_string(),
            );
        }
        Ok(Self {
            source_id,
            revision,
            start_char,
            end_char,
            content_sha256,
        })
    }

    fn canonical_key(&self) -> Vec<u8> {
        fn append_text(output: &mut Vec<u8>, value: &str) {
            output.extend_from_slice(&(value.len() as u64).to_be_bytes());
            output.extend_from_slice(value.as_bytes());
        }

        let mut output = b"perseus-source-group-v1".to_vec();
        append_text(&mut output, &self.source_id);
        append_text(&mut output, &self.revision);
        output.extend_from_slice(&(self.start_char as u64).to_be_bytes());
        output.extend_from_slice(&(self.end_char as u64).to_be_bytes());
        append_text(&mut output, &self.content_sha256);
        output
    }

    pub fn id(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(self.canonical_key());
        format!("sg-{:x}", hasher.finalize())
    }
}

/// A checked character range in a retained source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceSpan {
    pub start_char: usize,
    pub end_char: usize,
}

impl SourceSpan {
    pub fn new(start_char: usize, end_char: usize) -> Result<Self, String> {
        if start_char >= end_char {
            return Err("source span must satisfy start_char < end_char".to_string());
        }
        Ok(Self {
            start_char,
            end_char,
        })
    }
}

#[derive(Debug, Deserialize)]
struct RetainedSourceBody {
    content: Option<String>,
    #[serde(default)]
    chunk_hashes: HashMap<String, String>,
}

fn parse_retained_source_body(source_body: &str) -> Result<RetainedSourceBody, String> {
    if source_body.len() > MAX_SOURCE_BODY_BYTES {
        return Err("source_too_large".to_string());
    }
    serde_json::from_str(source_body).map_err(|_| "source_missing".to_string())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceChunkRef {
    pub source_category: String,
    pub source_key: String,
    pub span: SourceSpan,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EntityClassification {
    pub derived: bool,
    pub source_chunk: Option<SourceChunkRef>,
    pub support_ids: Vec<String>,
    pub reason: Option<String>,
}

impl EntityClassification {
    pub fn is_derived(&self) -> bool {
        self.derived
    }
}

/// Classify an entity from stored provenance metadata without inspecting its
/// prose. Malformed optional metadata is represented as an exclusion reason,
/// not upgraded into evidence.
pub fn classify_entity(entity: &crate::models::Entity) -> Result<EntityClassification, String> {
    let body: serde_json::Value = serde_json::from_str(&entity.body_json)
        .map_err(|error| format!("entity {} has invalid body JSON: {error}", entity.id))?;

    let source_chunk = match body.get("source_chunk") {
        None => None,
        Some(value) => match parse_source_chunk(value) {
            Ok(reference) => Some(reference),
            Err(_) => {
                return Ok(EntityClassification {
                    derived: false,
                    source_chunk: None,
                    support_ids: Vec::new(),
                    reason: Some("malformed_reference".to_string()),
                });
            }
        },
    };

    let mut support_ids: Vec<String> = entity
        .links
        .iter()
        .filter(|link| {
            matches!(
                link.relationship.as_str(),
                "supports" | "derived_from" | "evidence_for" | "promoted_to"
            ) || crate::models::classify_relation(&link.relationship)
                == crate::models::RelationKind::Supports
                || link.kind == Some(crate::models::RelationKind::Supports)
        })
        .filter(|link| !link.target_id.trim().is_empty())
        .map(|link| link.target_id.clone())
        .collect();
    support_ids.sort();
    support_ids.dedup();

    let derived = source_chunk.is_some() || !support_ids.is_empty();
    let reason = if derived {
        None
    } else if body
        .get("origin")
        .and_then(|origin| origin.get("memory_kind"))
        .and_then(serde_json::Value::as_str)
        == Some("inferred")
    {
        Some("missing_provenance".to_string())
    } else {
        None
    };

    Ok(EntityClassification {
        derived,
        source_chunk,
        support_ids,
        reason,
    })
}

fn parse_source_chunk(value: &serde_json::Value) -> Result<SourceChunkRef, String> {
    let object = value
        .as_object()
        .ok_or_else(|| "source_chunk must be an object".to_string())?;
    let source_category = object
        .get("source_category")
        .and_then(serde_json::Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| "source_chunk source_category is missing".to_string())?;
    let source_key = object
        .get("source_key")
        .and_then(serde_json::Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| "source_chunk source_key is missing".to_string())?;
    let span = object
        .get("span")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| "source_chunk span is missing".to_string())?;
    let start_char =
        span.get("start_char")
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| "source_chunk span start_char is missing".to_string())? as usize;
    let end_char =
        span.get("end_char")
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| "source_chunk span end_char is missing".to_string())? as usize;
    Ok(SourceChunkRef {
        source_category: source_category.to_string(),
        source_key: source_key.to_string(),
        span: SourceSpan::new(start_char, end_char)?,
    })
}

#[derive(Debug, Clone)]
pub struct RecoveredSource {
    pub source_group: SourceGroup,
    pub span: SourceSpan,
    pub text: Option<String>,
    pub span_sha256: String,
    pub verification: VerificationState,
    pub trust: TrustState,
    pub reason: Option<String>,
}

/// Recover one source span from a retained entity body. A bad hash returns a
/// structured result with no text; malformed storage or bounds are errors that
/// the caller maps to a non-disclosing exclusion reason.
pub fn recover_source_span(
    source_id: &str,
    source_category: &str,
    source_key: &str,
    revision: &str,
    source_body: &str,
    span: SourceSpan,
    expected_hash: Option<&str>,
) -> Result<RecoveredSource, String> {
    let body = parse_retained_source_body(source_body)?;
    let content = body
        .content
        .as_deref()
        .ok_or_else(|| "source_missing".to_string())?;
    let mut start_byte = None;
    let mut end_byte = None;
    for (index, (byte, character)) in content.char_indices().enumerate() {
        if index == span.start_char {
            start_byte = Some(byte);
        }
        if index.saturating_add(1) == span.end_char {
            end_byte = Some(byte + character.len_utf8());
            break;
        }
    }
    let (Some(start_byte), Some(end_byte)) = (start_byte, end_byte) else {
        return Err("span_out_of_bounds".to_string());
    };
    let span_chars = span.end_char - span.start_char;
    if span_chars > MAX_SOURCE_SPAN_CHARS {
        return Err("source_too_large".to_string());
    }

    let span_text = &content[start_byte..end_byte];
    let span_sha256 = sha256_hex(span_text.as_bytes());
    let full_source_sha256 = sha256_hex(content.as_bytes());
    let manifest_key = format!("{}:{}", span.start_char, span.end_char);
    let manifest_hash = body
        .chunk_hashes
        .get(&manifest_key)
        .map(String::as_str)
        .filter(|value| !value.is_empty());
    let expected_hash = expected_hash
        .filter(|value| !value.is_empty())
        .or(manifest_hash);
    if let Some(expected_hash) = expected_hash {
        if !is_sha256(expected_hash) {
            return Err("malformed_reference".to_string());
        }
    }

    let source_group = SourceGroup::new(
        source_id,
        revision,
        span.start_char,
        span.end_char,
        full_source_sha256,
    )?;
    if let Some(expected_hash) = expected_hash {
        if expected_hash != span_sha256 {
            return Ok(RecoveredSource {
                source_group,
                span,
                text: None,
                span_sha256,
                verification: VerificationState::HashMismatch,
                trust: TrustState::Untrusted,
                reason: Some("hash_mismatch".to_string()),
            });
        }
        return Ok(RecoveredSource {
            source_group,
            span,
            text: Some(span_text.to_string()),
            span_sha256,
            verification: VerificationState::Verified,
            trust: TrustState::Untrusted,
            reason: None,
        });
    }

    Ok(RecoveredSource {
        source_group,
        span,
        text: Some(span_text.to_string()),
        span_sha256,
        verification: VerificationState::Unchecked,
        trust: TrustState::Untrusted,
        reason: None,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VerificationState {
    EvidenceLinked,
    Verified,
    Unchecked,
    HashMismatch,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrustState {
    Trusted,
    Untrusted,
}

/// A single selected answer-facing item. `text` is deliberately confined to
/// this response projection; receipts use `ReceiptSelection` instead.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceItem {
    pub lane: EvidenceLane,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub entity_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<SourceLocator>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub span: Option<SourceSpan>,
    pub source_groups: Vec<String>,
    /// Stable chain metadata for this item. Unknown is explicit and is never
    /// treated as compatible with another unknown item.
    #[serde(serialize_with = "crate::source_chain::serialize_source_chain_receipt")]
    pub chain_identity: SourceChainIdentity,
    pub verification: VerificationState,
    pub trust: TrustState,
    pub tokens: usize,
    pub revision: String,
    pub span_sha256: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceLocator {
    pub id: String,
    pub category: String,
    pub key: String,
    pub revision: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LaneBudget {
    pub lane: EvidenceLane,
    pub selected_items: usize,
    pub omitted_items: usize,
    pub selected_tokens: usize,
    pub omitted_tokens: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceBudget {
    pub max_tokens: usize,
    pub selected_tokens: usize,
    pub omitted_tokens: usize,
    pub per_lane: Vec<LaneBudget>,
}

impl EvidenceBudget {
    pub fn new(max_tokens: i64) -> Result<Self, String> {
        if !(1..=MAX_EVIDENCE_TOKENS).contains(&max_tokens) {
            return Err(format!(
                "evidence max_tokens must be between 1 and {MAX_EVIDENCE_TOKENS}, got {max_tokens}"
            ));
        }
        Ok(Self {
            max_tokens: max_tokens as usize,
            selected_tokens: 0,
            omitted_tokens: 0,
            per_lane: Vec::new(),
        })
    }

    fn lane_mut(&mut self, lane: EvidenceLane) -> &mut LaneBudget {
        if let Some(index) = self.per_lane.iter().position(|entry| entry.lane == lane) {
            return &mut self.per_lane[index];
        }
        self.per_lane.push(LaneBudget {
            lane,
            selected_items: 0,
            omitted_items: 0,
            selected_tokens: 0,
            omitted_tokens: 0,
        });
        self.per_lane.sort_by_key(|entry| entry.lane.order());
        self.per_lane
            .iter_mut()
            .find(|entry| entry.lane == lane)
            .expect("lane inserted")
    }

    pub fn account_selected(&mut self, lane: EvidenceLane, tokens: usize) {
        self.selected_tokens = self.selected_tokens.saturating_add(tokens);
        let entry = self.lane_mut(lane);
        entry.selected_items += 1;
        entry.selected_tokens = entry.selected_tokens.saturating_add(tokens);
    }

    pub fn account_omitted(&mut self, lane: EvidenceLane, tokens: usize) {
        self.omitted_tokens = self.omitted_tokens.saturating_add(tokens);
        let entry = self.lane_mut(lane);
        entry.omitted_items += 1;
        entry.omitted_tokens = entry.omitted_tokens.saturating_add(tokens);
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ExclusionRecord {
    pub reason: String,
    pub count: usize,
}

impl ExclusionRecord {
    pub fn new(reason: impl Into<String>, count: usize) -> Self {
        Self {
            reason: reason.into(),
            count,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReceiptSelection {
    pub lane: EvidenceLane,
    pub entity_id: String,
    pub source_groups: Vec<String>,
    #[serde(serialize_with = "crate::source_chain::serialize_source_chain_receipt")]
    pub chain_identity: SourceChainIdentity,
    pub revision: String,
    pub span_sha256: String,
    pub verification: VerificationState,
    pub trust: TrustState,
    pub tokens: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceReceipt {
    pub schema_version: u32,
    pub query_sha256: String,
    pub lanes: Vec<EvidenceLane>,
    pub max_tokens: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub workspace_hash: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub requesting_agent_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub as_of_unix_ms: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub valid_at: Option<i64>,
    pub selected: Vec<ReceiptSelection>,
    pub excluded: Vec<ExclusionRecord>,
    pub budget: EvidenceBudget,
    pub digest: String,
}

impl EvidenceReceipt {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        query: &str,
        selection: &LaneSelection,
        budget: &EvidenceBudget,
        workspace_hash: Option<&str>,
        requesting_agent_id: Option<&str>,
        as_of_unix_ms: Option<i64>,
        valid_at: Option<i64>,
        selected: Vec<ReceiptSelection>,
        excluded: Vec<ExclusionRecord>,
    ) -> Self {
        let mut receipt = Self {
            schema_version: EVIDENCE_RECEIPT_SCHEMA_VERSION,
            query_sha256: sha256_hex(query.as_bytes()),
            lanes: selection.lanes.clone(),
            max_tokens: budget.max_tokens,
            workspace_hash: nonempty(workspace_hash),
            requesting_agent_id: nonempty(requesting_agent_id),
            as_of_unix_ms,
            valid_at,
            selected: normalize_selected(selected),
            excluded: normalize_excluded(excluded),
            budget: budget.clone(),
            digest: String::new(),
        };
        receipt.digest = receipt.compute_digest();
        receipt
    }

    fn digest_input(&self) -> serde_json::Value {
        serde_json::json!({
            "schema_version": self.schema_version,
            "query_sha256": self.query_sha256,
            "lanes": self.lanes,
            "max_tokens": self.max_tokens,
            "workspace_hash": self.workspace_hash,
            "requesting_agent_id": self.requesting_agent_id,
            "as_of_unix_ms": self.as_of_unix_ms,
            "valid_at": self.valid_at,
            "selected": normalize_selected(self.selected.clone()),
            "excluded": normalize_excluded(self.excluded.clone()),
            "budget": self.budget,
        })
    }

    fn compute_digest(&self) -> String {
        sha256_hex(canonical_json(&self.digest_input()).as_bytes())
    }

    pub fn digest(&self) -> String {
        self.compute_digest()
    }

    pub fn verify(&self) -> bool {
        self.digest == self.compute_digest() && is_sha256(&self.digest)
    }
}

fn nonempty(value: Option<&str>) -> Option<String> {
    value.filter(|value| !value.is_empty()).map(str::to_string)
}

fn normalize_selected(mut selected: Vec<ReceiptSelection>) -> Vec<ReceiptSelection> {
    for entry in &mut selected {
        entry.source_groups.sort();
        entry.source_groups.dedup();
    }
    selected.sort_by(|a, b| {
        (
            a.lane.order(),
            &a.entity_id,
            &a.revision,
            &a.span_sha256,
            a.chain_identity.commitment(),
        )
            .cmp(&(
                b.lane.order(),
                &b.entity_id,
                &b.revision,
                &b.span_sha256,
                b.chain_identity.commitment(),
            ))
    });
    selected
}

fn normalize_excluded(mut excluded: Vec<ExclusionRecord>) -> Vec<ExclusionRecord> {
    excluded.sort_by(|a, b| a.reason.cmp(&b.reason));
    let mut normalized: Vec<ExclusionRecord> = Vec::with_capacity(excluded.len());
    for entry in excluded {
        if let Some(previous) = normalized.last_mut() {
            if previous.reason == entry.reason {
                previous.count = previous.count.saturating_add(entry.count);
                continue;
            }
        }
        normalized.push(entry);
    }
    normalized
}

fn canonical_json(value: &serde_json::Value) -> String {
    fn sort(value: &serde_json::Value) -> serde_json::Value {
        match value {
            serde_json::Value::Object(map) => {
                let mut entries: Vec<(String, serde_json::Value)> = map
                    .iter()
                    .map(|(key, value)| (key.clone(), sort(value)))
                    .collect();
                entries.sort_by(|a, b| a.0.cmp(&b.0));
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

pub fn estimate_tokens(text: &str) -> usize {
    (text.chars().count().saturating_add(3) / 4).max(1)
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value.bytes().all(|b| b.is_ascii_hexdigit())
        && value == value.to_ascii_lowercase()
}

#[derive(Debug, Clone, Serialize)]
pub struct EvidenceProjection {
    pub lanes: Vec<EvidenceLane>,
    pub items: Vec<EvidenceItem>,
    pub budget: EvidenceBudget,
    pub excluded: Vec<ExclusionRecord>,
    pub receipt: EvidenceReceipt,
}

#[derive(Debug, Clone)]
struct ProviderMetadata {
    source_id: String,
    entity_id: Option<String>,
    revision: String,
    content_sha256: Option<String>,
    source_span_ref: Option<String>,
    workspace_hash: String,
    visibility: String,
    authority_agent_id: String,
    state: String,
    deleted_at_unix_ms: Option<i64>,
}

#[derive(Debug, Clone)]
struct GovernedEntity {
    entity: crate::models::Entity,
    revision: String,
    provider: Option<ProviderMetadata>,
}

/// Keep a chain-sensitive candidate set on one known lineage before any lane
/// or token-budget selection. Missing identity is an explicit exclusion, not a
/// wildcard that can mix with a known chain.
pub(crate) fn select_chain_coherent_entities(
    entities: Vec<Entity>,
) -> (Vec<Entity>, Vec<ExclusionRecord>) {
    let mut groups: BTreeMap<String, Vec<Entity>> = BTreeMap::new();
    let mut first_indices: BTreeMap<String, usize> = BTreeMap::new();
    let mut excluded = Vec::new();
    for (index, entity) in entities.into_iter().enumerate() {
        let Some(key) = (match entity_chain_key(&entity) {
            Ok(key) => key,
            Err(reason) => {
                excluded.push(ExclusionRecord::new(reason, 1));
                continue;
            }
        }) else {
            excluded.push(ExclusionRecord::new("unknown_chain_identity", 1));
            continue;
        };
        first_indices.entry(key.clone()).or_insert(index);
        groups.entry(key).or_default().push(entity);
    }
    let Some(winner_key) = groups
        .iter()
        .max_by(|(left_key, left_members), (right_key, right_members)| {
            left_members.len().cmp(&right_members.len()).then_with(|| {
                first_indices
                    .get(*right_key)
                    .cmp(&first_indices.get(*left_key))
            })
        })
        .map(|(key, _)| key.clone())
    else {
        return (Vec::new(), excluded);
    };
    let selected = groups.get(&winner_key).cloned().unwrap_or_default();
    for (key, members) in &groups {
        if key != &winner_key {
            excluded.push(ExclusionRecord::new("incompatible_chain", members.len()));
        }
    }
    (selected, excluded)
}

pub(crate) fn select_chain_coherent_scored(
    entities: Vec<(Entity, f64)>,
) -> (Vec<(Entity, f64)>, Vec<ExclusionRecord>) {
    let mut groups: BTreeMap<String, HashSet<String>> = BTreeMap::new();
    let mut first_indices: BTreeMap<String, usize> = BTreeMap::new();
    let mut keys = Vec::with_capacity(entities.len());
    let mut excluded = Vec::new();
    for (index, (entity, _score)) in entities.iter().enumerate() {
        match entity_chain_key(entity) {
            Ok(Some(key)) => {
                first_indices.entry(key.clone()).or_insert(index);
                groups
                    .entry(key.clone())
                    .or_default()
                    .insert(entity.id.clone());
                keys.push(Some(key));
            }
            Ok(None) => {
                excluded.push(ExclusionRecord::new("unknown_chain_identity", 1));
                keys.push(None);
            }
            Err(reason) => {
                excluded.push(ExclusionRecord::new(reason, 1));
                keys.push(None);
            }
        }
    }
    let Some(winner_key) = groups
        .iter()
        .max_by(|(left_key, left_ids), (right_key, right_ids)| {
            left_ids.len().cmp(&right_ids.len()).then_with(|| {
                first_indices
                    .get(*right_key)
                    .cmp(&first_indices.get(*left_key))
            })
        })
        .map(|(key, _)| key.clone())
    else {
        return (Vec::new(), excluded);
    };
    let mut selected = Vec::new();
    for ((entity, score), key) in entities.into_iter().zip(keys) {
        if key.as_deref() == Some(winner_key.as_str()) {
            selected.push((entity, score));
        } else if key.is_some() {
            excluded.push(ExclusionRecord::new("incompatible_chain", 1));
        }
    }
    selected.sort_by(|(_, left_score), (_, right_score)| {
        right_score
            .partial_cmp(left_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    (selected, excluded)
}

fn is_chain_sensitive_query(query: &str) -> bool {
    crate::source_chain::is_chain_sensitive_query(query)
}

/// Build the opt-in answer-facing evidence block over already-ranked recall
/// candidates. The ordinary recall path never calls this function.
pub fn project_recall_evidence(
    db: &crate::db::Database,
    candidates: &[crate::models::Entity],
    query: &str,
    selection: &LaneSelection,
    max_tokens: i64,
    workspace_hash: Option<&str>,
    requesting_agent_id: Option<&str>,
    as_of: Option<i64>,
    valid_at: Option<i64>,
) -> Result<EvidenceProjection, String> {
    let effective_max_tokens = if max_tokens == 0 { 256 } else { max_tokens };
    let mut budget = EvidenceBudget::new(effective_max_tokens)?;
    let mut excluded = Vec::new();
    let mut items = Vec::new();
    let mut seen_groups = HashSet::new();

    let mut ordered: Vec<crate::models::Entity> = Vec::new();
    for candidate in candidates {
        match govern_entity(
            db,
            candidate.clone(),
            workspace_hash,
            requesting_agent_id,
            as_of,
            valid_at,
        ) {
            Ok(governed) => ordered.push(governed.entity),
            Err(reason) => excluded.push(ExclusionRecord::new(reason, 1)),
        }
    }
    if is_chain_sensitive_query(query) {
        let (coherent, chain_excluded) = select_chain_coherent_entities(ordered);
        ordered = coherent;
        excluded.extend(chain_excluded);
    }

    if selection.contains(EvidenceLane::Derived) {
        for candidate in &ordered {
            let classification = match classify_entity(candidate) {
                Ok(classification) => classification,
                Err(_) => {
                    excluded.push(ExclusionRecord::new("malformed_reference", 1));
                    continue;
                }
            };
            if !classification.is_derived() {
                excluded.push(ExclusionRecord::new(
                    classification
                        .reason
                        .as_deref()
                        .unwrap_or("unsupported_lane"),
                    1,
                ));
                continue;
            }
            match derived_item(
                db,
                candidate,
                &classification,
                workspace_hash,
                requesting_agent_id,
                as_of,
                valid_at,
            ) {
                Ok(item) => {
                    if item
                        .source_groups
                        .iter()
                        .any(|group| seen_groups.contains(group))
                    {
                        continue;
                    }
                    let source_groups = item.source_groups.clone();
                    if account_item(item, &mut items, &mut budget, &mut excluded) {
                        seen_groups.extend(source_groups);
                    }
                }
                Err(reason) => excluded.push(ExclusionRecord::new(reason, 1)),
            }
        }
    }

    if selection.contains(EvidenceLane::Verbatim) {
        for candidate in &ordered {
            let classification = match classify_entity(candidate) {
                Ok(classification) => classification,
                Err(_) => {
                    excluded.push(ExclusionRecord::new("malformed_reference", 1));
                    continue;
                }
            };
            match verbatim_items(
                db,
                candidate,
                &classification,
                workspace_hash,
                requesting_agent_id,
                as_of,
                valid_at,
            ) {
                Ok(candidate_items) => {
                    for item in candidate_items {
                        if item
                            .source_groups
                            .iter()
                            .any(|group| seen_groups.contains(group))
                        {
                            continue;
                        }
                        let source_groups = item.source_groups.clone();
                        if account_item(item, &mut items, &mut budget, &mut excluded) {
                            seen_groups.extend(source_groups);
                        }
                    }
                }
                Err(reason) => excluded.push(ExclusionRecord::new(reason, 1)),
            }
        }
    }

    let excluded = normalize_excluded(excluded);
    let selected = items
        .iter()
        .map(|item| ReceiptSelection {
            lane: item.lane,
            entity_id: item.entity_id.clone().unwrap_or_default(),
            source_groups: item.source_groups.clone(),
            chain_identity: item.chain_identity.clone(),
            revision: item.revision.clone(),
            span_sha256: item.span_sha256.clone(),
            verification: item.verification,
            trust: item.trust,
            tokens: item.tokens,
        })
        .collect();
    let receipt = EvidenceReceipt::new(
        query,
        selection,
        &budget,
        workspace_hash,
        requesting_agent_id,
        as_of,
        valid_at,
        selected,
        excluded.clone(),
    );
    Ok(EvidenceProjection {
        lanes: selection.lanes.clone(),
        items,
        budget,
        excluded,
        receipt,
    })
}

fn account_item(
    item: EvidenceItem,
    items: &mut Vec<EvidenceItem>,
    budget: &mut EvidenceBudget,
    excluded: &mut Vec<ExclusionRecord>,
) -> bool {
    if budget.selected_tokens.saturating_add(item.tokens) <= budget.max_tokens {
        budget.account_selected(item.lane, item.tokens);
        items.push(item);
        true
    } else {
        budget.account_omitted(item.lane, item.tokens);
        excluded.push(ExclusionRecord::new("insufficient_budget", 1));
        false
    }
}

fn derived_item(
    db: &crate::db::Database,
    candidate: &crate::models::Entity,
    classification: &EntityClassification,
    workspace_hash: Option<&str>,
    requesting_agent_id: Option<&str>,
    as_of: Option<i64>,
    valid_at: Option<i64>,
) -> Result<EvidenceItem, String> {
    let mut groups = Vec::new();
    let mut revisions = Vec::new();
    let expected_chain_key = entity_chain_key(candidate)?;
    if let Some(reference) = &classification.source_chunk {
        let governed = lookup_source(
            db,
            &reference.source_category,
            &reference.source_key,
            workspace_hash,
            requesting_agent_id,
            as_of,
            valid_at,
        )?;
        if let Some(expected) = expected_chain_key.as_deref() {
            match entity_chain_key(&governed.entity)? {
                Some(actual) if actual == expected => {}
                Some(_) => return Err("wrong_chain_identity".to_string()),
                None => return Err("unknown_chain_identity".to_string()),
            }
        }
        let recovered = recover_governed_span(&governed, reference.span, None)?;
        revisions.push(recovered.source_group.revision.clone());
        groups.push(recovered.source_group);
    }
    for support_id in &classification.support_ids {
        let group = support_group(
            db,
            support_id,
            workspace_hash,
            requesting_agent_id,
            as_of,
            valid_at,
            expected_chain_key.as_deref(),
        )?;
        revisions.push(group.revision.clone());
        groups.push(group);
    }
    if groups.is_empty() {
        return Err("missing_provenance".to_string());
    }
    groups.sort_by_key(SourceGroup::id);
    groups.dedup();
    let source_groups: Vec<String> = groups.iter().map(SourceGroup::id).collect();
    let span_sha256 = groups
        .first()
        .map(|group| group.content_sha256.clone())
        .unwrap_or_default();
    let revision = revisions.join("|");
    let text = entity_answer_text(candidate);
    let chain_identity = entity_chain_identity(candidate)?;
    Ok(EvidenceItem {
        lane: EvidenceLane::Derived,
        entity_id: Some(candidate.id.clone()),
        source: None,
        span: None,
        source_groups,
        chain_identity,
        verification: VerificationState::EvidenceLinked,
        trust: TrustState::Trusted,
        tokens: estimate_tokens(&text),
        revision,
        span_sha256,
        text: None,
    })
}

fn verbatim_items(
    db: &crate::db::Database,
    candidate: &crate::models::Entity,
    classification: &EntityClassification,
    workspace_hash: Option<&str>,
    requesting_agent_id: Option<&str>,
    as_of: Option<i64>,
    valid_at: Option<i64>,
) -> Result<Vec<EvidenceItem>, String> {
    let mut items = Vec::new();
    let expected_chain_key = entity_chain_key(candidate)?;
    if let Some(reference) = &classification.source_chunk {
        let governed = lookup_source(
            db,
            &reference.source_category,
            &reference.source_key,
            workspace_hash,
            requesting_agent_id,
            as_of,
            valid_at,
        )?;
        if let Some(expected) = expected_chain_key.as_deref() {
            match entity_chain_key(&governed.entity)? {
                Some(actual) if actual == expected => {}
                Some(_) => return Err("wrong_chain_identity".to_string()),
                None => return Err("unknown_chain_identity".to_string()),
            }
        }
        let recovered = recover_governed_span(&governed, reference.span, None)?;
        items.push(verbatim_item(&governed, recovered)?);
    } else if is_retained_source(candidate) {
        let governed = govern_entity(
            db,
            candidate.clone(),
            workspace_hash,
            requesting_agent_id,
            as_of,
            valid_at,
        )?;
        let content = source_content(&governed.entity)?;
        let span = SourceSpan::new(0, content.chars().count())?;
        let recovered = recover_governed_span(&governed, span, None)?;
        items.push(verbatim_item(&governed, recovered)?);
    } else if classification.reason.is_some() || !classification.is_derived() {
        return Err(classification
            .reason
            .clone()
            .unwrap_or_else(|| "missing_provenance".to_string()));
    }

    items.extend(residual_items(db, candidate, as_of, valid_at)?);
    if items.is_empty() {
        return Err("missing_provenance".to_string());
    }
    Ok(items)
}

fn verbatim_item(
    governed: &GovernedEntity,
    recovered: RecoveredSource,
) -> Result<EvidenceItem, String> {
    let text = recovered.text.clone().ok_or_else(|| {
        recovered
            .reason
            .clone()
            .unwrap_or_else(|| "hash_mismatch".to_string())
    })?;
    let source_groups = vec![recovered.source_group.id()];
    let chain_identity =
        entity_chain_identity(&governed.entity)?.with_source_group(source_groups[0].clone())?;
    Ok(EvidenceItem {
        lane: EvidenceLane::Verbatim,
        entity_id: Some(governed.entity.id.clone()),
        source: Some(SourceLocator {
            id: governed.entity.id.clone(),
            category: governed.entity.category.clone(),
            key: governed.entity.key.clone(),
            revision: governed.revision.clone(),
        }),
        span: Some(recovered.span),
        source_groups,
        chain_identity,
        verification: recovered.verification,
        trust: recovered.trust,
        tokens: estimate_tokens(&text),
        revision: governed.revision.clone(),
        span_sha256: recovered.span_sha256,
        text: Some(text),
    })
}

fn entity_chain_identity(entity: &Entity) -> Result<SourceChainIdentity, String> {
    let body: serde_json::Value =
        serde_json::from_str(&entity.body_json).map_err(|_| "malformed_reference".to_string())?;
    SourceChainIdentity::from_entity_body(&body).map_err(|_| "malformed_reference".to_string())
}

pub(crate) fn entity_chain_key(entity: &Entity) -> Result<Option<String>, String> {
    let identity = entity_chain_identity(entity)?;
    Ok(identity
        .compatibility_key()
        .and_then(|key| serde_json::to_string(&(entity.workspace_hash.as_str(), key)).ok()))
}

fn support_group(
    db: &crate::db::Database,
    support_id: &str,
    workspace_hash: Option<&str>,
    requesting_agent_id: Option<&str>,
    as_of: Option<i64>,
    valid_at: Option<i64>,
    expected_chain_key: Option<&str>,
) -> Result<SourceGroup, String> {
    let support = db
        .get_entity_by_id_unfiltered(support_id)
        .map_err(|_| "source_missing".to_string())?
        .ok_or_else(|| "source_missing".to_string())?;
    let governed = govern_entity(
        db,
        support.clone(),
        workspace_hash,
        requesting_agent_id,
        as_of,
        valid_at,
    )?;
    if let Some(expected) = expected_chain_key {
        match entity_chain_key(&governed.entity)? {
            Some(actual) if actual == expected => {}
            Some(_) => return Err("wrong_chain_identity".to_string()),
            None => return Err("unknown_chain_identity".to_string()),
        }
    }
    let classification =
        classify_entity(&governed.entity).map_err(|_| "malformed_reference".to_string())?;
    if let Some(reference) = classification.source_chunk {
        let governed_source = lookup_source(
            db,
            &reference.source_category,
            &reference.source_key,
            workspace_hash,
            requesting_agent_id,
            as_of,
            valid_at,
        )?;
        if let Some(expected) = expected_chain_key {
            match entity_chain_key(&governed_source.entity)? {
                Some(actual) if actual == expected => {}
                Some(_) => return Err("wrong_chain_identity".to_string()),
                None => return Err("unknown_chain_identity".to_string()),
            }
        }
        return Ok(recover_governed_span(&governed_source, reference.span, None)?.source_group);
    }
    let body: serde_json::Value = serde_json::from_str(&governed.entity.body_json)
        .map_err(|_| "malformed_reference".to_string())?;
    let body_digest = sha256_hex(canonical_json(&body).as_bytes());
    SourceGroup::new(
        format!("entity:{}", governed.entity.id),
        governed.revision,
        0,
        0,
        body_digest,
    )
}

fn recover_reference(
    db: &crate::db::Database,
    reference: &SourceChunkRef,
    workspace_hash: Option<&str>,
    requesting_agent_id: Option<&str>,
    as_of: Option<i64>,
    valid_at: Option<i64>,
) -> Result<RecoveredSource, String> {
    let governed = lookup_source(
        db,
        &reference.source_category,
        &reference.source_key,
        workspace_hash,
        requesting_agent_id,
        as_of,
        valid_at,
    )?;
    recover_governed_span(&governed, reference.span, None)
}

fn recover_governed_span(
    governed: &GovernedEntity,
    span: SourceSpan,
    explicit_hash: Option<&str>,
) -> Result<RecoveredSource, String> {
    let body = governed.entity.body_json.as_str();
    if let Some(provider) = &governed.provider {
        if let Some(expected) = provider.content_sha256.as_deref() {
            let content = source_content(&governed.entity)?;
            if sha256_hex(content.as_bytes()) != expected {
                return Err("hash_mismatch".to_string());
            }
        }
        if let Some(reference) = provider.source_span_ref.as_deref() {
            let wanted = format!("{}:{}", span.start_char, span.end_char);
            if !reference.is_empty() && reference != wanted {
                return Err("malformed_reference".to_string());
            }
        }
    }
    let source_id = governed
        .provider
        .as_ref()
        .map(|provider| provider.source_id.as_str())
        .unwrap_or(governed.entity.id.as_str());
    recover_source_span(
        source_id,
        &governed.entity.category,
        &governed.entity.key,
        &governed.revision,
        body,
        span,
        explicit_hash,
    )
}

fn lookup_source(
    db: &crate::db::Database,
    category: &str,
    key: &str,
    workspace_hash: Option<&str>,
    requesting_agent_id: Option<&str>,
    as_of: Option<i64>,
    valid_at: Option<i64>,
) -> Result<GovernedEntity, String> {
    let source = db
        .get_entity(category, key)
        .map_err(|_| "source_missing".to_string())?
        .ok_or_else(|| "source_missing".to_string())?;
    govern_entity(
        db,
        source,
        workspace_hash,
        requesting_agent_id,
        as_of,
        valid_at,
    )
}

fn govern_entity(
    db: &crate::db::Database,
    entity: crate::models::Entity,
    workspace_hash: Option<&str>,
    requesting_agent_id: Option<&str>,
    as_of: Option<i64>,
    valid_at: Option<i64>,
) -> Result<GovernedEntity, String> {
    let mut resolved = vec![entity];
    if as_of.is_some() || valid_at.is_some() {
        db.resolve_temporal_versions(&mut resolved, as_of, valid_at)
            .map_err(|_| "source_missing".to_string())?;
    }
    let entity = resolved
        .into_iter()
        .next()
        .ok_or_else(|| "source_missing".to_string())?;
    if !scope_allows(&entity, workspace_hash) {
        return Err("scope_mismatch".to_string());
    }
    if !db.requester_can_read(requesting_agent_id, &entity.visibility, &entity.agent_id) {
        return Err("requester_mismatch".to_string());
    }
    if let Some(reason) = lifecycle_reason(&entity) {
        return Err(reason.to_string());
    }
    let provider = provider_metadata(db, &entity.id, as_of)?;
    if let Some(provider) = &provider {
        if provider.entity_id.as_deref() != Some(entity.id.as_str()) {
            return Err("source_missing".to_string());
        }
        if !scope_allows_workspace(&provider.workspace_hash, workspace_hash) {
            return Err("scope_mismatch".to_string());
        }
        if !db.requester_can_read(
            requesting_agent_id,
            &provider.visibility,
            &provider.authority_agent_id,
        ) {
            return Err("requester_mismatch".to_string());
        }
        if provider.state != "active" || provider.deleted_at_unix_ms.is_some() {
            return Err("tombstoned".to_string());
        }
    }
    let revision = provider
        .as_ref()
        .map(|provider| provider.revision.clone())
        .filter(|revision| !revision.is_empty())
        .unwrap_or_else(|| revision_from_entity(&entity));
    Ok(GovernedEntity {
        entity,
        revision,
        provider,
    })
}

fn provider_metadata(
    db: &crate::db::Database,
    entity_id: &str,
    as_of: Option<i64>,
) -> Result<Option<ProviderMetadata>, String> {
    let conn = db.conn().map_err(|_| "source_missing".to_string())?;
    let current = conn
        .query_row(
            "SELECT source_id, entity_id, revision, content_sha256, source_span_ref,
                    workspace_hash, visibility, authority_agent_id, state,
                    deleted_at_unix_ms
             FROM provider_sources
             WHERE entity_id=?1
             ORDER BY updated_at_unix_ms DESC LIMIT 1",
            rusqlite::params![entity_id],
            provider_metadata_from_row,
        )
        .optional()
        .map_err(|_| "source_missing".to_string())?;
    let Some(current) = current else {
        return Ok(None);
    };

    let Some(as_of) = as_of else {
        return Ok(Some(current));
    };
    let historical = conn
        .query_row(
            "SELECT source_id, entity_id, revision, content_sha256, source_span_ref,
                    workspace_hash, visibility, authority_agent_id, state_after,
                    deleted_at_unix_ms
             FROM provider_source_events
             WHERE source_id=?1 AND recorded_at_unix_ms <= ?2
             ORDER BY recorded_at_unix_ms DESC, event_id DESC LIMIT 1",
            rusqlite::params![current.source_id, as_of],
            provider_metadata_from_row,
        )
        .optional()
        .map_err(|_| "source_missing".to_string())?;
    let Some(historical) = historical else {
        return Err("source_missing".to_string());
    };
    if historical.entity_id.as_deref() != Some(entity_id) {
        return Err("source_missing".to_string());
    }
    Ok(Some(historical))
}

fn provider_metadata_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<ProviderMetadata> {
    Ok(ProviderMetadata {
        source_id: row.get(0)?,
        entity_id: row.get(1)?,
        revision: row.get(2)?,
        content_sha256: row.get(3)?,
        source_span_ref: row.get(4)?,
        workspace_hash: row.get(5)?,
        visibility: row.get(6)?,
        authority_agent_id: row.get(7)?,
        state: row.get(8)?,
        deleted_at_unix_ms: row.get(9)?,
    })
}

fn residual_items(
    db: &crate::db::Database,
    candidate: &crate::models::Entity,
    as_of: Option<i64>,
    valid_at: Option<i64>,
) -> Result<Vec<EvidenceItem>, String> {
    if valid_at.is_some() {
        // Residual spans are transaction-time rows without a valid-time
        // coordinate. Do not attach current residuals to a valid-time view.
        return Ok(Vec::new());
    }
    let conn = db.conn().map_err(|_| "source_missing".to_string())?;
    let mut statement = conn
        .prepare(
            "SELECT id, span_text, status
             FROM residual_spans
             WHERE entity_id=?1
               AND status IN ('active', 'confirmed')
               AND length(span_text) <= ?2
               AND (?3 IS NULL OR created_ms <= ?3)
             ORDER BY created_ms ASC, id ASC
             LIMIT ?4",
        )
        .map_err(|_| "source_missing".to_string())?;
    let rows = statement
        .query_map(
            rusqlite::params![
                &candidate.id,
                MAX_SOURCE_SPAN_CHARS as i64,
                as_of,
                MAX_RESIDUAL_ITEMS,
            ],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                ))
            },
        )
        .map_err(|_| "source_missing".to_string())?;
    let mut items = Vec::new();
    for row in rows {
        let (id, text, status) = row.map_err(|_| "source_missing".to_string())?;
        if text.is_empty() {
            continue;
        }
        let span = SourceSpan::new(0, text.chars().count())?;
        let span_sha256 = sha256_hex(text.as_bytes());
        let group = SourceGroup::new(
            format!("residual:{id}"),
            format!("residual:{status}"),
            span.start_char,
            span.end_char,
            span_sha256.clone(),
        )?;
        items.push(EvidenceItem {
            lane: EvidenceLane::Verbatim,
            entity_id: Some(candidate.id.clone()),
            source: Some(SourceLocator {
                id: id.clone(),
                category: "residual".to_string(),
                key: id,
                revision: format!("residual:{status}"),
            }),
            span: Some(span),
            source_groups: vec![group.id()],
            chain_identity: entity_chain_identity(candidate)?,
            verification: VerificationState::Unchecked,
            trust: TrustState::Untrusted,
            tokens: estimate_tokens(&text),
            revision: format!("residual:{status}"),
            span_sha256,
            text: Some(text),
        });
    }
    Ok(items)
}

fn scope_allows(entity: &crate::models::Entity, workspace_hash: Option<&str>) -> bool {
    scope_allows_workspace(&entity.workspace_hash, workspace_hash)
}

fn scope_allows_workspace(source_workspace: &str, workspace_hash: Option<&str>) -> bool {
    workspace_hash
        .filter(|workspace| !workspace.is_empty())
        .map_or(true, |workspace| {
            source_workspace.is_empty() || source_workspace == workspace
        })
}

fn lifecycle_reason(entity: &crate::models::Entity) -> Option<&'static str> {
    if entity.archived {
        return Some("archived");
    }
    match entity.status.as_str() {
        "deprecated" => Some("superseded"),
        "expired" => Some("stale"),
        "redacted" | "quarantined" | "compacted" => Some("archived"),
        _ => None,
    }
}

fn is_retained_source(entity: &crate::models::Entity) -> bool {
    if entity.category == "transcript" {
        return true;
    }
    serde_json::from_str::<serde_json::Value>(&entity.body_json)
        .ok()
        .and_then(|body| {
            Some(
                body.get("chunk_hashes").is_some()
                    || body
                        .get("retained_source")
                        .and_then(serde_json::Value::as_bool)
                        == Some(true),
            )
        })
        .unwrap_or(false)
}

fn source_content(entity: &crate::models::Entity) -> Result<String, String> {
    let body: serde_json::Value =
        serde_json::from_str(&entity.body_json).map_err(|_| "source_missing".to_string())?;
    body.get("content")
        .and_then(serde_json::Value::as_str)
        .map(str::to_string)
        .filter(|content| !content.is_empty())
        .ok_or_else(|| "source_missing".to_string())
}

fn entity_answer_text(entity: &crate::models::Entity) -> String {
    serde_json::from_str::<serde_json::Value>(&entity.body_json)
        .ok()
        .and_then(|body| {
            body.get("content")
                .or_else(|| body.get("text"))
                .or_else(|| body.get("claim"))
                .or_else(|| body.get("note"))
                .and_then(serde_json::Value::as_str)
                .map(str::to_string)
        })
        .filter(|text| !text.is_empty())
        .unwrap_or_else(|| format!("{}:{}", entity.category, entity.key))
}

fn revision_from_entity(entity: &crate::models::Entity) -> String {
    serde_json::from_str::<serde_json::Value>(&entity.body_json)
        .ok()
        .and_then(|body| {
            body.get("revision")
                .and_then(serde_json::Value::as_str)
                .filter(|revision| !revision.is_empty())
                .map(str::to_string)
        })
        .unwrap_or_else(|| format!("entity-created:{}", entity.created_at_unix_ms))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn lane_selection_canonicalizes_union_order() {
        let selection = parse_lane_selection(&json!(["verbatim", "derived"])).unwrap();
        assert_eq!(
            selection.lanes(),
            &[EvidenceLane::Derived, EvidenceLane::Verbatim]
        );
    }

    #[test]
    fn source_group_identity_is_order_independent_and_uses_character_spans() {
        let first =
            SourceGroup::new("transcript:meeting-1", "revision-7", 12, 28, "a".repeat(64)).unwrap();
        let second =
            SourceGroup::new("transcript:meeting-1", "revision-7", 12, 28, "a".repeat(64)).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.id(), second.id());
        assert!(first.id().starts_with("sg-"));
        assert_eq!(first.start_char, 12);
        assert_eq!(first.end_char, 28);
    }

    #[test]
    fn source_group_identity_is_unambiguous_for_delimiter_values() {
        let first = SourceGroup::new("a\\nrevision=b", "r", 1, 2, "a".repeat(64)).unwrap();
        let second = SourceGroup::new("a", "b\\nrevision=r", 1, 2, "a".repeat(64)).unwrap();
        assert_ne!(
            first.id(),
            second.id(),
            "distinct source identity fields must never share a canonical hash input"
        );
    }

    #[test]
    fn explicit_budget_is_strict_and_receipt_is_order_independent_and_content_free() {
        assert!(EvidenceBudget::new(-1).is_err());
        assert!(EvidenceBudget::new(0).is_err());

        let selection = parse_lane_selection(&json!(["verbatim", "derived"])).unwrap();
        let budget = EvidenceBudget::new(64).unwrap();
        let selected = vec![ReceiptSelection {
            lane: EvidenceLane::Verbatim,
            entity_id: "derived-1".to_string(),
            source_groups: vec!["sg-z".to_string(), "sg-a".to_string()],
            chain_identity: SourceChainIdentity::unknown(),
            revision: "r1".to_string(),
            span_sha256: "b".repeat(64),
            verification: VerificationState::Unchecked,
            trust: TrustState::Untrusted,
            tokens: 4,
        }];
        let first = EvidenceReceipt::new(
            "same query",
            &selection,
            &budget,
            Some("workspace-a"),
            Some("agent-a"),
            Some(12),
            Some(34),
            selected.clone(),
            vec![ExclusionRecord::new("insufficient_budget", 1)],
        );
        let mut reversed = selected;
        reversed[0].source_groups.reverse();
        let second = EvidenceReceipt::new(
            "same query",
            &selection,
            &budget,
            Some("workspace-a"),
            Some("agent-a"),
            Some(12),
            Some(34),
            reversed,
            vec![ExclusionRecord::new("insufficient_budget", 1)],
        );
        assert_eq!(first.digest(), second.digest());
        let encoded = serde_json::to_string(&first).unwrap();
        assert!(!encoded.contains("raw source text"));
        assert!(!encoded.contains("same query"));
        assert!(!encoded.contains("commitment_sha256"));
        assert!(encoded.contains("\"status\":\"unknown\""));
        assert!(first.verify());
    }

    #[test]
    fn inferred_content_without_support_is_not_derived() {
        let mut entity = crate::db::tests::make_entity(
            "inference-unlinked",
            "insight",
            "unlinked",
            r#"{"content":"an agent guess","origin":{"memory_kind":"inferred"}}"#,
        );
        entity.links.clear();
        let classified = classify_entity(&entity).unwrap();
        assert!(!classified.is_derived());
        assert_eq!(classified.reason.as_deref(), Some("missing_provenance"));
    }

    #[test]
    fn support_link_places_inferred_content_in_derived_lane() {
        let mut entity = crate::db::tests::make_entity(
            "inference-linked",
            "insight",
            "linked",
            r#"{"content":"an evidence-backed inference","origin":{"memory_kind":"inferred"}}"#,
        );
        entity.links = vec![crate::models::MemoryLink {
            target_id: "source-1".to_string(),
            relationship: "derived_from".to_string(),
            weight: 1.0,
            source: Some(entity.id.clone()),
            kind: Some(crate::models::RelationKind::Supports),
            asserted_at_unix_ms: Some(1),
        }];
        let classified = classify_entity(&entity).unwrap();
        assert!(classified.is_derived());
        assert_eq!(classified.support_ids, vec!["source-1"]);
    }

    #[test]
    fn promoted_to_link_places_inferred_content_in_derived_lane() {
        let mut entity = crate::db::tests::make_entity(
            "inference-promoted",
            "insight",
            "promoted",
            r#"{"content":"a promoted inference","origin":{"memory_kind":"inferred"}}"#,
        );
        entity.links = vec![crate::models::MemoryLink {
            target_id: "source-1".to_string(),
            relationship: "promoted_to".to_string(),
            weight: 1.0,
            source: Some(entity.id.clone()),
            kind: None,
            asserted_at_unix_ms: Some(1),
        }];
        let classified = classify_entity(&entity).unwrap();
        assert!(classified.is_derived());
        assert_eq!(classified.support_ids, vec!["source-1"]);
    }

    #[test]
    fn source_recovery_is_utf8_safe_and_fail_closed_on_hash_mismatch() {
        let content = "préfixe — café — suffixe";
        let span = SourceSpan::new(11, 17).unwrap();
        let expected_text: String = content.chars().skip(11).take(6).collect();
        let expected_hash = sha256_hex(expected_text.as_bytes());
        let source_body = serde_json::json!({
            "content": content,
            "chunk_hashes": {"11:17": expected_hash},
        })
        .to_string();

        let recovered = recover_source_span(
            "transcript-1",
            "transcript",
            "meeting-1",
            "revision-1",
            &source_body,
            span,
            None,
        )
        .unwrap();
        assert_eq!(recovered.text.as_deref(), Some(expected_text.as_str()));
        assert_eq!(recovered.verification, VerificationState::Verified);
        assert_eq!(recovered.trust, TrustState::Untrusted);

        let mismatch = recover_source_span(
            "transcript-1",
            "transcript",
            "meeting-1",
            "revision-1",
            &source_body,
            span,
            Some(&"f".repeat(64)),
        )
        .unwrap();
        assert!(mismatch.text.is_none());
        assert_eq!(mismatch.reason.as_deref(), Some("hash_mismatch"));
        assert_eq!(
            recover_source_span(
                "source-1",
                "document",
                "doc-1",
                "revision-1",
                r#"{"content":"retained bytes"}"#,
                SourceSpan {
                    start_char: 8,
                    end_char: 2,
                },
                None,
            )
            .unwrap_err(),
            "span_out_of_bounds"
        );
    }

    #[test]
    fn source_without_expected_hash_is_explicitly_unchecked() {
        let recovered = recover_source_span(
            "source-1",
            "document",
            "doc-1",
            "revision-1",
            r#"{"content":"retained bytes"}"#,
            SourceSpan::new(0, 8).unwrap(),
            None,
        )
        .unwrap();
        assert_eq!(recovered.verification, VerificationState::Unchecked);
        assert_eq!(recovered.trust, TrustState::Untrusted);
        assert!(recovered.reason.is_none());
    }

    #[test]
    fn projection_union_recovers_verified_untrusted_source_and_derived_fact() {
        let (db, _path) = crate::db::tests::temp_db();
        let content = "alpha café omega";
        let span = SourceSpan::new(0, 10).unwrap();
        let span_text: String = content.chars().take(10).collect();
        let source_body = serde_json::json!({
            "content": content,
            "chunk_hashes": {"0:10": sha256_hex(span_text.as_bytes())},
            "source_chain": {"schema_version": 1, "chain_id": "chain-a", "episode_id": "episode-1"}
        })
        .to_string();
        let mut source =
            crate::db::tests::make_entity("source-1", "transcript", "meeting-1", &source_body);
        source.workspace_hash = "workspace-a".to_string();
        source.agent_id = "agent-a".to_string();
        db.remember(&source).unwrap();

        let fact_body = serde_json::json!({
            "content": "the meeting covered the café",
            "origin": {"memory_kind": "extracted"},
            "source_chunk": {
                "source_category": "transcript",
                "source_key": "meeting-1",
                "span": {"start_char": 0, "end_char": 10}
            },
            "source_chain": {"schema_version": 1, "chain_id": "chain-a", "episode_id": "episode-1"}
        })
        .to_string();
        let mut fact = crate::db::tests::make_entity("fact-1", "fact", "meeting-cafe", &fact_body);
        fact.workspace_hash = "workspace-a".to_string();
        fact.agent_id = "agent-a".to_string();
        db.remember(&fact).unwrap();

        let selection = parse_lane_selection(&json!(["verbatim", "derived"])).unwrap();
        let projection = project_recall_evidence(
            &db,
            &[fact],
            "café",
            &selection,
            64,
            Some("workspace-a"),
            Some("agent-a"),
            None,
            None,
        )
        .unwrap();
        assert_eq!(
            projection.items.len(),
            1,
            "one source group must not be emitted twice across the lane union"
        );
        assert_eq!(projection.items[0].lane, EvidenceLane::Derived);
        assert_eq!(
            projection.items[0].chain_identity.chain_id.as_deref(),
            Some("chain-a")
        );
        assert_eq!(projection.items[0].chain_identity.status, "known");
        assert!(
            projection
                .items
                .iter()
                .flat_map(|item| item.source_groups.iter())
                .collect::<std::collections::HashSet<_>>()
                .len()
                == projection
                    .items
                    .iter()
                    .map(|item| item.source_groups.len())
                    .sum::<usize>()
        );
        assert!(projection.receipt.verify());
    }

    #[test]
    fn over_budget_group_does_not_suppress_later_fitting_representative() {
        let (db, _path) = crate::db::tests::temp_db();
        let mut support = crate::db::tests::make_entity(
            "support-1",
            "source",
            "support-1",
            r#"{"content":"support"}"#,
        );
        support.workspace_hash = "workspace-a".to_string();
        support.agent_id = "agent-a".to_string();
        db.remember(&support).unwrap();

        let link = |source: &str| crate::models::MemoryLink {
            target_id: "support-1".to_string(),
            relationship: "supports".to_string(),
            weight: 1.0,
            source: Some(source.to_string()),
            kind: Some(crate::models::RelationKind::Supports),
            asserted_at_unix_ms: Some(1),
        };
        let mut large = crate::db::tests::make_entity(
            "fact-large",
            "fact",
            "large",
            &json!({"content": "xxxxxxxxxxxxxxxxxxxxxxxx", "origin": {"memory_kind": "extracted"}})
                .to_string(),
        );
        large.workspace_hash = "workspace-a".to_string();
        large.agent_id = "agent-a".to_string();
        large.links = vec![link(&large.id)];
        let mut small = crate::db::tests::make_entity(
            "fact-small",
            "fact",
            "small",
            &json!({"content": "x", "origin": {"memory_kind": "extracted"}}).to_string(),
        );
        small.workspace_hash = "workspace-a".to_string();
        small.agent_id = "agent-a".to_string();
        small.links = vec![link(&small.id)];

        let selection = parse_lane_selection(&json!(["derived"])).unwrap();
        let projection = project_recall_evidence(
            &db,
            &[large, small],
            "same support",
            &selection,
            1,
            Some("workspace-a"),
            Some("agent-a"),
            None,
            None,
        )
        .unwrap();
        assert_eq!(projection.items.len(), 1);
        assert_eq!(projection.items[0].entity_id.as_deref(), Some("fact-small"));
        let derived_budget = projection
            .budget
            .per_lane
            .iter()
            .find(|entry| entry.lane == EvidenceLane::Derived)
            .expect("derived lane budget");
        assert_eq!(derived_budget.omitted_items, 1);
        assert_eq!(projection.budget.selected_tokens, 1);
    }

    #[test]
    fn recall_attaches_evidence_only_when_lanes_are_explicit() {
        let (db, _path) = crate::db::tests::temp_db();
        let mut source = crate::db::tests::make_entity(
            "source-handler",
            "transcript",
            "handler-meeting",
            &json!({"content": "alpha café omega", "retained_source": true}).to_string(),
        );
        source.visibility = "public".to_string();
        db.remember(&source).unwrap();

        let legacy =
            crate::tools::handle_recall(&db, json!({"query": "café", "mode": "fts5", "limit": 10}))
                .unwrap();
        let legacy_value: serde_json::Value = serde_json::from_str(&legacy).unwrap();
        assert!(legacy_value.get("evidence").is_none());

        let explicit = crate::tools::handle_recall(
            &db,
            json!({
                "query": "café",
                "mode": "fts5",
                "limit": 10,
                "max_tokens": 64,
                "evidence_lanes": ["verbatim"]
            }),
        )
        .unwrap();
        let explicit_value: serde_json::Value = serde_json::from_str(&explicit).unwrap();
        assert!(explicit_value.get("evidence").is_some());
        assert_eq!(explicit_value["evidence"]["lanes"], json!(["verbatim"]));
        assert!(explicit_value["evidence"]["items"].is_array());
    }

    #[test]
    fn recall_expansion_preserves_explicit_evidence_projection() {
        let (db, _path) = crate::db::tests::temp_db();
        let mut source = crate::db::tests::make_entity(
            "source-expansion",
            "transcript",
            "expansion-meeting",
            &json!({"content": "running systems", "retained_source": true}).to_string(),
        );
        source.visibility = "public".to_string();
        db.remember(&source).unwrap();

        let response = crate::tools::handle_recall(
            &db,
            json!({
                "query": "running",
                "mode": "fts5",
                "limit": 10,
                "expansion": {"enabled": true, "n_variants": 1},
                "evidence_lanes": ["verbatim"]
            }),
        )
        .unwrap();
        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert!(value.get("evidence").is_some());
    }

    #[test]
    fn recall_expansion_filters_incompatible_source_chains_before_delivery() {
        let (db, _path) = crate::db::tests::temp_db();
        for (id, key, chain_id, content) in [
            (
                "expand-chain-a1",
                "expand-chain-a1",
                "chain-a",
                "studies expansion record a1",
            ),
            (
                "expand-chain-a2",
                "expand-chain-a2",
                "chain-a",
                "studies expansion record a2",
            ),
            (
                "expand-chain-b1",
                "expand-chain-b1",
                "chain-b",
                "studio expansion record b1",
            ),
            (
                "expand-chain-b2",
                "expand-chain-b2",
                "chain-b",
                "studio expansion record b2",
            ),
            (
                "expand-chain-b3",
                "expand-chain-b3",
                "chain-b",
                "studio expansion record b3",
            ),
        ] {
            let mut entity = crate::db::tests::make_entity(
                id,
                "transcript",
                key,
                &json!({
                    "content": content,
                    "source_chain": {"schema_version": 1, "chain_id": chain_id}
                })
                .to_string(),
            );
            entity.visibility = "public".to_string();
            db.remember_skip_dedup(&entity).unwrap();
        }
        let response = crate::tools::handle_recall(
            &db,
            json!({
                "query": "studies lineage",
                "mode": "fts5",
                "limit": 10,
                "expansion": {"enabled": true, "n_variants": 1}
            }),
        )
        .unwrap();
        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        let ids: std::collections::HashSet<&str> = value["items"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|item| item["id"].as_str())
            .collect();
        let expected: std::collections::HashSet<&str> =
            ["expand-chain-b1", "expand-chain-b2", "expand-chain-b3"]
                .into_iter()
                .collect();
        assert_eq!(ids, expected, "mixed or incorrect chain delivery: {value}");
    }

    #[test]
    fn malformed_lane_requests_fail_closed_at_handler_boundary() {
        let (db, _path) = crate::db::tests::temp_db();
        for lanes in [json!([]), json!(["unknown"]), json!([1]), json!("derived")] {
            let error = crate::tools::handle_recall(
                &db,
                json!({"query": "anything", "evidence_lanes": lanes}),
            )
            .expect_err("malformed evidence lanes must fail closed");
            assert!(error.contains("Invalid evidence_lanes"), "{error}");
        }
    }

    #[test]
    fn residual_projection_respects_temporal_anchor_and_bounds() {
        let (db, _path) = crate::db::tests::temp_db();
        let mut fact = crate::db::tests::make_entity(
            "fact-residual",
            "fact",
            "residual",
            r#"{"content":"inferred"}"#,
        );
        fact.workspace_hash = "workspace-a".to_string();
        fact.agent_id = "agent-a".to_string();
        fact.links = vec![crate::models::MemoryLink {
            target_id: "support-residual".to_string(),
            relationship: "supports".to_string(),
            weight: 1.0,
            source: Some("fact-residual".to_string()),
            kind: None,
            asserted_at_unix_ms: Some(1),
        }];
        db.remember(&fact).unwrap();

        let conn = db.conn().unwrap();
        conn.execute(
            "INSERT INTO residual_spans (id, entity_id, span_text, status, created_ms)
             VALUES ('residual-old', 'fact-residual', 'old', 'active', 10)",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO residual_spans (id, entity_id, span_text, status, created_ms)
             VALUES ('residual-future', 'fact-residual', 'future', 'active', 20)",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO residual_spans (id, entity_id, span_text, status, created_ms)
             VALUES ('residual-long', 'fact-residual', ?1, 'active', 1)",
            rusqlite::params!["x".repeat(MAX_SOURCE_SPAN_CHARS + 1)],
        )
        .unwrap();
        drop(conn);

        let anchored = residual_items(&db, &fact, Some(15), None).unwrap();
        assert_eq!(anchored.len(), 1);
        assert_eq!(anchored[0].source.as_ref().unwrap().id, "residual-old");
        assert!(residual_items(&db, &fact, None, Some(15))
            .unwrap()
            .is_empty());
    }

    #[test]
    fn provider_metadata_replays_event_history_at_as_of_anchor() {
        let (db, _path) = crate::db::tests::temp_db();
        let conn = db.conn().unwrap();
        conn.execute(
            "INSERT INTO provider_sources (
                source_id, workspace_hash, provider, kind, external_id, revision,
                visibility, capture_method, authority_agent_id, entity_id, state,
                current_event_id, receipt_digest, recorded_at_unix_ms, updated_at_unix_ms
             ) VALUES (?1, 'workspace-a', 'test', 'message', 'external-1', 'r2',
                       'workspace', 'event_feed', 'agent-a', 'source-1', 'deleted',
                       'evt-2', 'receipt-2', 200, 200)",
            rusqlite::params!["src-test-1"],
        )
        .unwrap();
        for (event_id, revision, event_type, state, recorded) in [
            ("evt-1", "r1", "upsert", "active", 100_i64),
            ("evt-2", "r2", "delete", "deleted", 200_i64),
        ] {
            conn.execute(
                "INSERT INTO provider_source_events (
                    event_id, source_id, entity_id, workspace_hash, provider, kind, external_id,
                    revision, event_type, state_after, request_digest, receipt_digest,
                    recorded_at_unix_ms
                 ) VALUES (?1, 'src-test-1', 'source-1', 'workspace-a', 'test', 'message',
                           'external-1', ?2, ?3, ?4, 'request', 'receipt', ?5)",
                rusqlite::params![event_id, revision, event_type, state, recorded],
            )
            .unwrap();
        }
        drop(conn);

        let historical = provider_metadata(&db, "source-1", Some(150))
            .unwrap()
            .expect("active event must be visible at the anchor");
        assert_eq!(historical.revision, "r1");
        assert_eq!(historical.state, "active");
        assert!(provider_metadata(&db, "source-1", Some(50)).is_err());
        let deleted = provider_metadata(&db, "source-1", Some(250))
            .unwrap()
            .expect("delete event must be visible after its anchor");
        assert_eq!(deleted.revision, "r2");
        assert_eq!(deleted.state, "deleted");
        assert!(deleted.deleted_at_unix_ms.is_none());

        let conn = db.conn().unwrap();
        conn.execute(
            "INSERT INTO provider_source_events (
                event_id, source_id, workspace_hash, provider, kind, external_id,
                revision, event_type, state_after, request_digest, receipt_digest,
                recorded_at_unix_ms
             ) VALUES ('evt-null', 'src-test-1', 'workspace-a', 'test', 'message',
                       'external-1', 'r3', 'upsert', 'active', 'request', 'receipt', 300)",
            [],
        )
        .unwrap();
        drop(conn);
        assert!(provider_metadata(&db, "source-1", Some(350)).is_err());
    }

    #[test]
    fn omitted_evidence_field_is_byte_compatible_with_explicit_null() {
        let (db, _path) = crate::db::tests::temp_db();
        let without_field = crate::tools::handle_recall(
            &db,
            json!({"query": "no-such-evidence-lane-fixture", "mode": "fts5", "limit": 10}),
        )
        .unwrap();
        let explicit_null = crate::tools::handle_recall(
            &db,
            json!({
                "query": "no-such-evidence-lane-fixture",
                "mode": "fts5",
                "limit": 10,
                "evidence_lanes": null
            }),
        )
        .unwrap();
        assert_eq!(without_field, explicit_null);
    }

    #[test]
    fn nested_source_chain_mismatch_is_rejected_before_evidence_recovery() {
        let (db, _path) = crate::db::tests::temp_db();
        let mut source = crate::db::tests::make_entity(
            "nested-source",
            "transcript",
            "nested-meeting",
            &serde_json::json!({
                "content": "source bytes",
                "source_chain": {"schema_version": 1, "chain_id": "chain-b"}
            })
            .to_string(),
        );
        source.workspace_hash = "workspace-a".to_string();
        db.remember(&source).unwrap();

        let mut support = crate::db::tests::make_entity(
            "nested-support",
            "insight",
            "nested-support",
            &serde_json::json!({
                "content": "derived support",
                "origin": {"memory_kind": "extracted"},
                "source_chunk": {
                    "source_category": "transcript",
                    "source_key": "nested-meeting",
                    "span": {"start_char": 0, "end_char": 6}
                },
                "source_chain": {"schema_version": 1, "chain_id": "chain-a"}
            })
            .to_string(),
        );
        support.workspace_hash = "workspace-a".to_string();
        db.remember(&support).unwrap();

        let mut candidate = crate::db::tests::make_entity(
            "nested-candidate",
            "insight",
            "nested-candidate",
            &serde_json::json!({
                "content": "candidate",
                "origin": {"memory_kind": "inferred"},
                "source_chain": {"schema_version": 1, "chain_id": "chain-a"}
            })
            .to_string(),
        );
        candidate.workspace_hash = "workspace-a".to_string();
        candidate.links = vec![crate::models::MemoryLink {
            target_id: support.id.clone(),
            relationship: "derived_from".to_string(),
            weight: 1.0,
            source: Some(candidate.id.clone()),
            kind: Some(crate::models::RelationKind::Supports),
            asserted_at_unix_ms: Some(1),
        }];
        db.remember(&candidate).unwrap();

        let selection = parse_lane_selection(&serde_json::json!(["derived"])).unwrap();
        let projection = project_recall_evidence(
            &db,
            &[candidate],
            "candidate",
            &selection,
            64,
            Some("workspace-a"),
            None,
            None,
            None,
        )
        .expect("nested mismatch becomes an exclusion");
        assert!(projection.items.is_empty());
        assert!(projection
            .excluded
            .iter()
            .any(|entry| entry.reason == "wrong_chain_identity"));
    }

    #[test]
    fn scored_chain_selection_is_invariant_to_equal_score_input_order() {
        let make = |id: &str, chain: &str| {
            let body = json!({
                "content": id,
                "source_chain": {"schema_version": 1, "chain_id": chain}
            })
            .to_string();
            crate::db::tests::make_entity(id, "fact", id, &body)
        };
        let a = make("a", "chain-a");
        let b = make("b", "chain-b");
        let (selected_ab, _) =
            select_chain_coherent_scored(vec![(a.clone(), 1.0), (b.clone(), 1.0)]);
        let (selected_ba, _) = select_chain_coherent_scored(vec![(b, 1.0), (a, 1.0)]);
        let ids_ab = selected_ab
            .into_iter()
            .map(|(entity, _)| entity.id)
            .collect::<Vec<_>>();
        let ids_ba = selected_ba
            .into_iter()
            .map(|(entity, _)| entity.id)
            .collect::<Vec<_>>();
        assert_eq!(ids_ab, vec!["a"]);
        assert_eq!(ids_ba, vec!["b"]);
    }

    #[test]
    fn scored_chain_selection_preserves_ranked_order_for_equal_scores() {
        let make = |id: &str| {
            crate::db::tests::make_entity(
                id,
                "fact",
                id,
                &json!({
                    "content": id,
                    "source_chain": {"schema_version": 1, "chain_id": "same-chain"}
                })
                .to_string(),
            )
        };
        let (selected, _) =
            select_chain_coherent_scored(vec![(make("z-ranked"), 1.0), (make("a-ranked"), 1.0)]);
        assert_eq!(
            selected
                .into_iter()
                .map(|(entity, _)| entity.id)
                .collect::<Vec<_>>(),
            vec!["z-ranked", "a-ranked"]
        );
    }

    #[test]
    fn unscored_chain_selection_preserves_ranked_input_order() {
        let make = |id: &str| {
            crate::db::tests::make_entity(
                id,
                "fact",
                id,
                &json!({
                    "content": id,
                    "source_chain": {"schema_version": 1, "chain_id": "chain-ranked"}
                })
                .to_string(),
            )
        };
        let selected = select_chain_coherent_entities(vec![make("z-ranked"), make("a-ranked")]).0;
        assert_eq!(
            selected
                .into_iter()
                .map(|entity| entity.id)
                .collect::<Vec<_>>(),
            vec!["z-ranked", "a-ranked"]
        );
    }

    #[test]
    fn evidence_projection_preserves_ranked_candidate_order() {
        let (db, path) = crate::db::tests::temp_db();
        let make = |id: &str| {
            crate::db::tests::make_entity(
                id,
                "transcript",
                id,
                &json!({
                    "content": format!("{id} retained source"),
                    "retained_source": true,
                    "source_chain": {"schema_version": 1, "chain_id": "ordered-chain"}
                })
                .to_string(),
            )
        };
        let z = make("z-evidence");
        let a = make("a-evidence");
        db.remember(&z).unwrap();
        db.remember(&a).unwrap();
        let selection = parse_lane_selection(&json!(["verbatim"])).unwrap();
        let projection = project_recall_evidence(
            &db,
            &[z, a],
            "ordered evidence",
            &selection,
            64,
            None,
            None,
            None,
            None,
        )
        .unwrap();
        assert_eq!(
            projection
                .items
                .into_iter()
                .map(|item| item.entity_id.unwrap())
                .collect::<Vec<_>>(),
            vec!["z-evidence", "a-evidence"]
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn chain_sensitive_selection_keeps_one_known_chain_and_excludes_unknowns() {
        let make = |id: &str, chain: Option<&str>| {
            let body = match chain {
                Some(chain_id) => json!({
                    "content": id,
                    "source_chain": {
                        "schema_version": 1,
                        "chain_id": chain_id
                    }
                })
                .to_string(),
                None => json!({"content": id}).to_string(),
            };
            crate::db::tests::make_entity(id, "fact", id, &body)
        };
        let (selected, excluded) = select_chain_coherent_entities(vec![
            make("chain-a-1", Some("chain-a")),
            make("unknown", None),
            make("chain-b-1", Some("chain-b")),
            make("chain-a-2", Some("chain-a")),
        ]);
        assert_eq!(
            selected
                .into_iter()
                .map(|entity| entity.id)
                .collect::<Vec<_>>(),
            vec!["chain-a-1", "chain-a-2"]
        );
        assert!(excluded
            .iter()
            .any(|entry| entry.reason == "unknown_chain_identity"));
        assert!(excluded
            .iter()
            .any(|entry| entry.reason == "incompatible_chain"));
    }
}
