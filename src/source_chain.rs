//! Stable, bounded source-chain identity for coherent evidence assembly.
//!
//! This is provenance metadata, not evidence content. It is optional on legacy
//! records, explicit when missing, and hash-bound when present. Bodies remain
//! resolved by governed readers rather than copied into receipts.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

pub const SOURCE_CHAIN_SCHEMA_VERSION: u32 = 1;
const _SC_MAX_ID_CHARS: usize = 256;
const _SC_MAX_SUBJECT_IDS: usize = 16;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SourceChainIdentity {
    pub schema_version: u32,
    /// `known` when at least one stable lineage anchor is present; `unknown`
    /// is an explicit legacy/missing-identity state.
    pub status: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_group_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub episode_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub experience_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chain_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thread_id: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub subject_ids: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sequence: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub valid_from_unix_ms: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub valid_to_unix_ms: Option<i64>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub commitment_sha256: String,
}

impl Default for SourceChainIdentity {
    fn default() -> Self {
        Self::unknown()
    }
}

impl SourceChainIdentity {
    pub fn unknown() -> Self {
        let identity = Self {
            schema_version: SOURCE_CHAIN_SCHEMA_VERSION,
            status: "unknown".to_string(),
            source_group_id: None,
            episode_id: None,
            experience_id: None,
            chain_id: None,
            thread_id: None,
            subject_ids: Vec::new(),
            parent_id: None,
            sequence: None,
            valid_from_unix_ms: None,
            valid_to_unix_ms: None,
            commitment_sha256: String::new(),
        };
        identity
    }

    pub fn from_body(body: &Value) -> Result<Self, String> {
        let Some(value) = body.get("source_chain") else {
            return Ok(Self::unknown());
        };
        let input: _ScInput = serde_json::from_value(value.clone())
            .map_err(|_| "source_chain metadata is malformed".to_string())?;
        if input.schema_version != SOURCE_CHAIN_SCHEMA_VERSION {
            return Err("unsupported source-chain schema_version".to_string());
        }
        _sc_build_known(input)
    }

    pub fn from_entity_body(body: &Value) -> Result<Self, String> {
        let valid_from = body.get("valid_from_unix_ms").and_then(Value::as_i64);
        let valid_to = body.get("valid_to_unix_ms").and_then(Value::as_i64);
        Self::from_body(body)?.with_valid_time(valid_from, valid_to)
    }

    /// Attach a stable source-group anchor to an otherwise unknown identity.
    /// This is used for retained source spans whose producer did not provide a
    /// richer episode/experience/thread identity.
    pub fn for_source_group(source_group_id: impl Into<String>) -> Result<Self, String> {
        let mut input = _ScInput::default();
        input.schema_version = SOURCE_CHAIN_SCHEMA_VERSION;
        input.source_group_id = Some(source_group_id.into());
        _sc_build_known(input)
    }

    /// Add a source-group anchor without discarding richer metadata.
    pub fn with_source_group(mut self, source_group_id: impl Into<String>) -> Result<Self, String> {
        if self.source_group_id.is_none() {
            self.source_group_id = Some(source_group_id.into());
            self.status = "known".to_string();
            self.commitment_sha256 = _sc_identity_commitment(&self);
            self.validate()?;
        }
        Ok(self)
    }

    /// Fill missing valid-time coordinates from the canonical entity columns.
    /// Producer-supplied source-chain values win if both representations exist.
    pub fn with_valid_time(
        mut self,
        valid_from_unix_ms: Option<i64>,
        valid_to_unix_ms: Option<i64>,
    ) -> Result<Self, String> {
        if self.valid_from_unix_ms.is_none() {
            self.valid_from_unix_ms = valid_from_unix_ms;
        }
        if self.valid_to_unix_ms.is_none() {
            self.valid_to_unix_ms = valid_to_unix_ms;
        }
        if self.is_known() {
            self.commitment_sha256 = _sc_identity_commitment(&self);
        } else {
            self.commitment_sha256.clear();
        }
        self.validate()?;
        Ok(self)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != SOURCE_CHAIN_SCHEMA_VERSION {
            return Err("unsupported source-chain schema_version".to_string());
        }
        if self.status != "known" && self.status != "unknown" {
            return Err("source-chain status must be known or unknown".to_string());
        }
        for (label, value) in [
            ("source_group_id", self.source_group_id.as_deref()),
            ("episode_id", self.episode_id.as_deref()),
            ("experience_id", self.experience_id.as_deref()),
            ("chain_id", self.chain_id.as_deref()),
            ("thread_id", self.thread_id.as_deref()),
            ("parent_id", self.parent_id.as_deref()),
        ] {
            if let Some(value) = value {
                _sc_validate_id(label, value)?;
            }
        }
        if self.subject_ids.len() > _SC_MAX_SUBJECT_IDS {
            return Err("source-chain subject_ids exceeds bound".to_string());
        }
        for value in &self.subject_ids {
            _sc_validate_id("subject_id", value)?;
        }
        if let (Some(from), Some(to)) = (self.valid_from_unix_ms, self.valid_to_unix_ms) {
            if to < from {
                return Err("source-chain valid-time range is inverted".to_string());
            }
        }
        if self.status == "unknown" && self._has_anchor() {
            return Err("unknown source-chain identity cannot carry anchors".to_string());
        }
        if self.status == "known" && !self._has_anchor() {
            return Err("known source-chain identity requires an anchor".to_string());
        }
        if self.status == "unknown" {
            if !self.commitment_sha256.is_empty() {
                return Err("unknown source-chain identity cannot carry a commitment".to_string());
            }
        } else {
            if !is_sha256(&self.commitment_sha256) {
                return Err(
                    "source-chain commitment must be a lowercase SHA-256 digest".to_string()
                );
            }
            if self.commitment_sha256 != _sc_identity_commitment(self) {
                return Err("source-chain commitment does not match identity".to_string());
            }
        }
        Ok(())
    }

    pub fn commitment(&self) -> &str {
        &self.commitment_sha256
    }

    pub fn is_known(&self) -> bool {
        self.status == "known"
    }

    pub fn is_unknown(&self) -> bool {
        !self.is_known()
    }

    pub fn compatibility_key(&self) -> Option<String> {
        if !self.is_known() {
            return None;
        }
        serde_json::to_string(&(
            self.source_group_id.as_deref(),
            self.episode_id.as_deref(),
            self.experience_id.as_deref(),
            self.chain_id.as_deref(),
            self.thread_id.as_deref(),
            self.subject_ids.as_slice(),
            self.parent_id.as_deref(),
        ))
        .ok()
    }

    /// Unknown identities never become compatible merely because they are both
    /// missing. A chain-sensitive route must either select one known key or
    /// report that identity is unavailable.
    pub fn compatible_with(&self, other: &Self) -> bool {
        self.compatibility_key().is_some() && self.compatibility_key() == other.compatibility_key()
    }

    fn _has_anchor(&self) -> bool {
        self.source_group_id.is_some()
            || self.episode_id.is_some()
            || self.experience_id.is_some()
            || self.chain_id.is_some()
            || self.thread_id.is_some()
            || !self.subject_ids.is_empty()
            || self.parent_id.is_some()
    }
}

#[derive(Serialize)]
struct _ScReceiptProjection<'a> {
    status: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    commitment_sha256: Option<&'a str>,
}

/// Serialize source-chain identity for public receipts without exposing
/// anchors, subjects, chronology, or other internal identity fields.
pub fn serialize_source_chain_receipt<S>(
    identity: &SourceChainIdentity,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    _ScReceiptProjection {
        status: &identity.status,
        commitment_sha256: identity.is_known().then_some(identity.commitment()),
    }
    .serialize(serializer)
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct _ScInput {
    schema_version: u32,
    #[serde(default)]
    source_group_id: Option<String>,
    #[serde(default)]
    episode_id: Option<String>,
    #[serde(default)]
    experience_id: Option<String>,
    #[serde(default)]
    chain_id: Option<String>,
    #[serde(default)]
    thread_id: Option<String>,
    #[serde(default)]
    subject_ids: Vec<String>,
    #[serde(default)]
    parent_id: Option<String>,
    #[serde(default)]
    sequence: Option<u64>,
    #[serde(default)]
    valid_from_unix_ms: Option<i64>,
    #[serde(default)]
    valid_to_unix_ms: Option<i64>,
}

fn _sc_build_known(input: _ScInput) -> Result<SourceChainIdentity, String> {
    let mut subject_ids = input.subject_ids;
    subject_ids.sort();
    subject_ids.dedup();
    let mut identity = SourceChainIdentity {
        schema_version: input.schema_version,
        status: if input.source_group_id.is_some()
            || input.episode_id.is_some()
            || input.experience_id.is_some()
            || input.chain_id.is_some()
            || input.thread_id.is_some()
            || !subject_ids.is_empty()
            || input.parent_id.is_some()
        {
            "known".to_string()
        } else {
            "unknown".to_string()
        },
        source_group_id: input.source_group_id,
        episode_id: input.episode_id,
        experience_id: input.experience_id,
        chain_id: input.chain_id,
        thread_id: input.thread_id,
        subject_ids,
        parent_id: input.parent_id,
        sequence: input.sequence,
        valid_from_unix_ms: input.valid_from_unix_ms,
        valid_to_unix_ms: input.valid_to_unix_ms,
        commitment_sha256: String::new(),
    };
    if !identity._has_anchor() {
        identity.status = "unknown".to_string();
    }
    identity.commitment_sha256 = if identity.is_known() {
        _sc_identity_commitment(&identity)
    } else {
        String::new()
    };
    identity.validate()?;
    Ok(identity)
}

fn _sc_identity_material(identity: &SourceChainIdentity) -> Value {
    serde_json::json!({
        "schema_version": identity.schema_version,
        "status": identity.status,
        "source_group_id": identity.source_group_id,
        "episode_id": identity.episode_id,
        "experience_id": identity.experience_id,
        "chain_id": identity.chain_id,
        "thread_id": identity.thread_id,
        "subject_ids": identity.subject_ids,
        "parent_id": identity.parent_id,
        "sequence": identity.sequence,
        "valid_from_unix_ms": identity.valid_from_unix_ms,
        "valid_to_unix_ms": identity.valid_to_unix_ms,
    })
}

fn _sc_identity_commitment(identity: &SourceChainIdentity) -> String {
    _sc_commitment(&_sc_identity_material(identity))
}

pub fn is_chain_sensitive_query(query: &str) -> bool {
    let lower = query.to_ascii_lowercase();
    [
        "chain",
        "lineage",
        "episode",
        "experience",
        "thread",
        "sequence",
        "chronolog",
        "before",
        "after",
        "then",
        "path",
        "multi-hop",
        "multihop",
    ]
    .iter()
    .any(|marker| lower.contains(marker))
}

fn _sc_commitment(value: &Value) -> String {
    let bytes = serde_json::to_vec(value).unwrap_or_default();
    let mut hasher = Sha256::new();
    hasher.update(b"perseus-source-chain/v1|");
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn _sc_validate_id(label: &str, value: &str) -> Result<(), String> {
    let count = value.chars().count();
    if value.trim().is_empty() || count > _SC_MAX_ID_CHARS || value.chars().any(char::is_control) {
        return Err(format!(
            "source-chain {label} is empty, oversized, or contains control text"
        ));
    }
    Ok(())
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

    #[test]
    fn parses_known_identity_and_binds_all_lineage_coordinates() {
        let body = json!({
            "source_chain": {
                "schema_version": 1,
                "source_group_id": "group-a",
                "episode_id": "episode-7",
                "experience_id": "experience-2",
                "chain_id": "chain-a",
                "thread_id": "thread-9",
                "subject_ids": ["subject-a"],
                "parent_id": "parent-1",
                "sequence": 3,
                "valid_from_unix_ms": 100,
                "valid_to_unix_ms": 200
            }
        });
        let identity = SourceChainIdentity::from_body(&body).expect("identity");
        assert_eq!(identity.status, "known");
        assert_eq!(identity.chain_id.as_deref(), Some("chain-a"));
        assert_eq!(identity.sequence, Some(3));
        assert!(identity.validate().is_ok());
        assert_eq!(identity.commitment().len(), 64);
    }

    #[test]
    fn full_subject_scope_distinguishes_related_but_different_chains() {
        let left = SourceChainIdentity::from_body(&json!({
            "source_chain": {
                "schema_version": 1,
                "chain_id": "chain-a",
                "subject_ids": ["subject-a", "subject-b"],
                "sequence": 1
            }
        }))
        .unwrap();
        let right = SourceChainIdentity::from_body(&json!({
            "source_chain": {
                "schema_version": 1,
                "chain_id": "chain-a",
                "subject_ids": ["subject-a", "subject-c"],
                "sequence": 2
            }
        }))
        .unwrap();
        assert!(!left.compatible_with(&right));
        assert_eq!(left.compatibility_key().is_some(), true);
    }

    #[test]
    fn missing_identity_is_explicitly_unknown() {
        let identity = SourceChainIdentity::from_body(&json!({"content": "legacy"})).unwrap();
        assert!(!identity.is_known());
        assert_eq!(identity.status, "unknown");
        assert!(identity.validate().is_ok());
    }

    #[test]
    fn sequence_only_identity_is_unknown_and_not_chain_compatible() {
        let identity = SourceChainIdentity::from_body(&json!({
            "source_chain": {
                "schema_version": 1,
                "sequence": 7
            }
        }))
        .unwrap();
        assert!(identity.is_unknown());
        assert_eq!(identity.compatibility_key(), None);
        assert!(identity.validate().is_ok());
    }
}
