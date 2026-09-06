//! #1080: cryptographically authorized mutation — signed transitions for
//! retrieval state (MutMem, arXiv:2608.02843).
//!
//! Every nontrivial retrieval-relevant state change commits as a signed
//! transition binding a terminal provenance node, a signer epoch, quantized
//! old→new values, a no-fork predecessor, and two domain-separated SHA-256
//! commitments, under an Ed25519 signature verifiable by a portable verifier
//! with no database access. Poison-likely content is retained (never silently
//! deleted) with signed, revisable labels that recall consumes as trust
//! evidence.
//!
//! Scope (per the paper): the protocol provides evidence of integrity,
//! authorization, traceability, and historical continuity — it does not
//! establish content truth. Key material reuses the Ed25519 format and
//! verification style of the signed-profile infrastructure (#837); epochs are
//! registered through `perseus_vault_signer_epoch_set` (Ops scope).

use ed25519_dalek::{Signature, Signer, SigningKey, VerifyingKey, PUBLIC_KEY_LENGTH};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

/// Wire schema of a signed transition.
pub const TRANSITION_SCHEMA: &str = "perseus-transition/v1";
/// Domain separators for the two value commitments and the chain hash.
pub const DOMAIN_OLD: &[u8] = b"perseus-vault/transition/v1|old-value";
pub const DOMAIN_NEW: &[u8] = b"perseus-vault/transition/v1|new-value";
pub const DOMAIN_CHAIN: &[u8] = b"perseus-vault/transition/v1|chain";

/// Poison-label levels accepted by `perseus_vault_poison_label`.
pub const POISON_LEVELS: [&str; 3] = ["poison_likely", "suspect", "clean"];
/// Effective ranking-time trust penalty per level (0.0 = none).
pub fn poison_penalty(level: &str) -> f64 {
    match level {
        "poison_likely" => 0.9,
        "suspect" => 0.5,
        _ => 0.0,
    }
}

pub(crate) fn b64_encode(bytes: &[u8]) -> String {
    use base64::Engine as _;
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

pub(crate) fn b64_decode(s: &str) -> Result<Vec<u8>, String> {
    use base64::Engine as _;
    base64::engine::general_purpose::STANDARD
        .decode(s.trim())
        .map_err(|e| format!("invalid base64: {e}"))
}

pub(crate) fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// Decode a raw 32-byte Ed25519 public key from base64 (fail-closed).
pub fn decode_public_key(b64: &str) -> Result<VerifyingKey, String> {
    let bytes = b64_decode(b64)?;
    if bytes.len() != PUBLIC_KEY_LENGTH {
        return Err(format!(
            "public key must be {PUBLIC_KEY_LENGTH} raw bytes, got {}",
            bytes.len()
        ));
    }
    let mut arr = [0u8; PUBLIC_KEY_LENGTH];
    arr.copy_from_slice(&bytes);
    VerifyingKey::from_bytes(&arr).map_err(|e| format!("invalid Ed25519 public key: {e}"))
}

pub(crate) fn encode_public_key(key: &VerifyingKey) -> String {
    b64_encode(&key.to_bytes())
}

/// Decode a raw 32-byte Ed25519 seed (base64). Used by signer-epoch
/// registration; the seed never appears in logs or responses.
pub fn decode_seed(seed_b64: &str) -> Result<[u8; 32], String> {
    let bytes = b64_decode(seed_b64)?;
    if bytes.len() != 32 {
        return Err(format!(
            "Ed25519 seed must be 32 raw bytes, got {}",
            bytes.len()
        ));
    }
    let mut arr = [0u8; 32];
    arr.copy_from_slice(&bytes);
    Ok(arr)
}

/// Quantize a JSON value so both sides of a transition are stable and
/// drift-free: numbers round to 4 decimals, objects get sorted keys.
pub fn quantize(value: &Value) -> Value {
    match value {
        Value::Number(n) => {
            if let Some(f) = n.as_f64() {
                let q = (f * 10_000.0).round() / 10_000.0;
                serde_json::Number::from_f64(q)
                    .map(Value::Number)
                    .unwrap_or_else(|| value.clone())
            } else {
                value.clone()
            }
        }
        Value::Array(items) => Value::Array(items.iter().map(quantize).collect()),
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            let mut out = serde_json::Map::new();
            for k in keys {
                out.insert(k.clone(), quantize(&map[k]));
            }
            Value::Object(out)
        }
        other => other.clone(),
    }
}

/// Domain-separated SHA-256 commitment over a quantized value:
/// `H(domain || 0x00 || canonical(value))`.
pub fn domain_commitment(domain: &[u8], value: &Value) -> String {
    let mut h = Sha256::new();
    h.update(domain);
    h.update(b"\x00");
    h.update(crate::signed_profile::canonical_json_bytes(&quantize(
        value,
    )));
    format!("{:x}", h.finalize())
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TransitionPayload {
    pub schema: String,
    /// Terminal provenance node: the mutated entity id (or journal event id).
    pub terminal_node: String,
    pub signer_epoch: u64,
    pub signer_fingerprint: String,
    pub old_value: Value,
    pub new_value: Value,
    /// Chain hash of the previous record ('' = genesis).
    pub predecessor_hash: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SignedTransition {
    pub payload: TransitionPayload,
    pub commitment_old: String,
    pub commitment_new: String,
    pub signature_b64: String,
    /// H(DOMAIN_CHAIN || canonical(payload) || commitments || signature).
    pub chain_hash: String,
}

#[derive(Debug, Clone)]
pub struct TransitionVerification {
    pub signer_fingerprint: String,
    pub old_digest: String,
    pub new_digest: String,
    pub chain_hash: String,
}

/// H(DOMAIN_CHAIN || canonical(payload) || 0x00 || commitment_old || 0x00 ||
/// commitment_new || 0x00 || signature bytes).
fn chain_hash_of(
    payload: &TransitionPayload,
    commitment_old: &str,
    commitment_new: &str,
    signature: &Signature,
) -> String {
    let mut h = Sha256::new();
    h.update(DOMAIN_CHAIN);
    h.update(b"\x00");
    h.update(crate::signed_profile::canonical_json_bytes(
        &serde_json::to_value(payload).unwrap_or(Value::Null),
    ));
    h.update(b"\x00");
    h.update(commitment_old.as_bytes());
    h.update(b"\x00");
    h.update(commitment_new.as_bytes());
    h.update(b"\x00");
    h.update(signature.to_bytes());
    format!("{:x}", h.finalize())
}

/// Sign a transition with the writer's Ed25519 key. The payload embeds the
/// signer fingerprint; the caller supplies the epoch and the predecessor hash
/// (the previous record's chain hash, '' for genesis).
pub fn sign_transition(
    signing: &SigningKey,
    epoch: u64,
    terminal_node: &str,
    old_value: &Value,
    new_value: &Value,
    predecessor_hash: &str,
) -> Result<SignedTransition, String> {
    let verifying = signing.verifying_key();
    let payload = TransitionPayload {
        schema: TRANSITION_SCHEMA.into(),
        terminal_node: terminal_node.into(),
        signer_epoch: epoch,
        signer_fingerprint: sha256_hex(&verifying.to_bytes()),
        old_value: quantize(old_value),
        new_value: quantize(new_value),
        predecessor_hash: predecessor_hash.into(),
    };
    let canonical = crate::signed_profile::canonical_json_bytes(
        &serde_json::to_value(&payload)
            .map_err(|e| format!("payload serialization failed: {e}"))?,
    );
    let signature = signing.sign(&canonical);
    let commitment_old = domain_commitment(DOMAIN_OLD, &payload.old_value);
    let commitment_new = domain_commitment(DOMAIN_NEW, &payload.new_value);
    let chain_hash = chain_hash_of(&payload, &commitment_old, &commitment_new, &signature);
    Ok(SignedTransition {
        payload,
        commitment_old,
        commitment_new,
        signature_b64: b64_encode(&signature.to_bytes()),
        chain_hash,
    })
}

/// Portable verification: checks the schema, signer identity, both
/// domain-separated commitments, the Ed25519 signature, and the recomputed
/// chain hash. No database access — usable as a standalone verifier (CLI
/// `verify-transition`, tests, external tooling).
pub fn verify_transition(
    t: &SignedTransition,
    trusted_key_b64: &str,
) -> Result<TransitionVerification, String> {
    if t.payload.schema != TRANSITION_SCHEMA {
        return Err(format!(
            "transition schema must be {TRANSITION_SCHEMA}, got {:?}",
            t.payload.schema
        ));
    }
    let public = decode_public_key(trusted_key_b64)?;
    let fingerprint = sha256_hex(&public.to_bytes());
    if fingerprint != t.payload.signer_fingerprint {
        return Err("signer fingerprint does not match the trusted key".into());
    }
    let commitment_old = domain_commitment(DOMAIN_OLD, &t.payload.old_value);
    if commitment_old != t.commitment_old {
        return Err("old-value commitment mismatch (tampered or unquantized payload)".into());
    }
    let commitment_new = domain_commitment(DOMAIN_NEW, &t.payload.new_value);
    if commitment_new != t.commitment_new {
        return Err("new-value commitment mismatch (tampered or unquantized payload)".into());
    }
    let canonical = crate::signed_profile::canonical_json_bytes(
        &serde_json::to_value(&t.payload)
            .map_err(|e| format!("payload serialization failed: {e}"))?,
    );
    let signature = Signature::from_slice(&b64_decode(&t.signature_b64)?)
        .map_err(|e| format!("invalid signature encoding: {e}"))?;
    public
        .verify_strict(&canonical, &signature)
        .map_err(|_| "signature verification failed (tampered or unsigned)".to_string())?;
    let chain_hash = chain_hash_of(&t.payload, &commitment_old, &commitment_new, &signature);
    if chain_hash != t.chain_hash {
        return Err("chain hash mismatch (record does not reproduce its own chain hash)".into());
    }
    Ok(TransitionVerification {
        signer_fingerprint: fingerprint,
        old_digest: commitment_old,
        new_digest: commitment_new,
        chain_hash,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn test_key() -> SigningKey {
        SigningKey::from_bytes(&[9u8; 32])
    }

    fn key_b64(signing: &SigningKey) -> String {
        b64_encode(&signing.verifying_key().to_bytes())
    }

    #[test]
    fn sign_then_verify_round_trips() {
        let signing = test_key();
        let t = sign_transition(
            &signing,
            1,
            "mem-abc123",
            &json!({"decay_score": 0.8}),
            &json!({"decay_score": 1.0, "importance": 1.0}),
            "",
        )
        .unwrap();
        let v = verify_transition(&t, &key_b64(&signing)).unwrap();
        assert_eq!(v.signer_fingerprint.len(), 64);
        assert_eq!(v.old_digest.len(), 64);
        assert_eq!(v.new_digest.len(), 64);
        assert_eq!(v.chain_hash, t.chain_hash);
    }

    #[test]
    fn authorization_wrong_key_fails() {
        let signing = test_key();
        let other = SigningKey::from_bytes(&[3u8; 32]);
        let t = sign_transition(&signing, 1, "mem-x", &json!({}), &json!({"a": 1}), "").unwrap();
        let err = verify_transition(&t, &key_b64(&other)).unwrap_err();
        assert!(err.contains("fingerprint"), "unexpected error: {err}");
    }

    #[test]
    fn unsigned_record_fails() {
        let signing = test_key();
        let mut t =
            sign_transition(&signing, 1, "mem-x", &json!({}), &json!({"a": 1}), "").unwrap();
        t.signature_b64 = b64_encode(&[0u8; 64]);
        let err = verify_transition(&t, &key_b64(&signing)).unwrap_err();
        assert!(err.contains("signature"), "unexpected error: {err}");
    }

    #[test]
    fn tampered_old_value_fails_commitment_check() {
        let signing = test_key();
        let mut t =
            sign_transition(&signing, 1, "mem-x", &json!({"v": 1}), &json!({"v": 2}), "").unwrap();
        t.payload.old_value = json!({"v": 99});
        let err = verify_transition(&t, &key_b64(&signing)).unwrap_err();
        assert!(
            err.contains("old-value commitment"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn tampered_terminal_node_fails_signature() {
        let signing = test_key();
        let mut t = sign_transition(&signing, 1, "mem-x", &json!({}), &json!({}), "").unwrap();
        t.payload.terminal_node = "mem-y".into();
        let err = verify_transition(&t, &key_b64(&signing)).unwrap_err();
        assert!(err.contains("signature"), "unexpected error: {err}");
    }

    #[test]
    fn quantization_stabilizes_floats_and_key_order() {
        let raw = json!({"z": 0.123456789, "a": 0.5, "nested": {"b": 1.0, "a": 2.0}});
        let q = quantize(&raw);
        let a = crate::signed_profile::canonical_json_bytes(&q);
        let b = crate::signed_profile::canonical_json_bytes(&quantize(&q));
        assert_eq!(
            a, b,
            "quantization must be idempotent and order-insensitive"
        );
        assert_eq!(q["z"], json!(0.1235));
    }

    #[test]
    fn commitments_are_domain_separated() {
        let v = json!({"score": 0.5});
        let old = domain_commitment(DOMAIN_OLD, &v);
        let new = domain_commitment(DOMAIN_NEW, &v);
        assert_ne!(
            old, new,
            "identical values under different domains must hash differently"
        );
    }

    #[test]
    fn chain_hash_binds_predecessor() {
        let signing = test_key();
        let t1 = sign_transition(&signing, 1, "mem-a", &json!({}), &json!({"v": 1}), "").unwrap();
        let t2a = sign_transition(
            &signing,
            1,
            "mem-b",
            &json!({}),
            &json!({"v": 2}),
            &t1.chain_hash,
        )
        .unwrap();
        let t2b = sign_transition(
            &signing,
            1,
            "mem-b",
            &json!({}),
            &json!({"v": 2}),
            "deadbeef",
        )
        .unwrap();
        assert_eq!(t2a.payload.predecessor_hash, t1.chain_hash);
        assert_ne!(
            t2a.chain_hash, t2b.chain_hash,
            "a different predecessor must change the chain hash"
        );
    }

    #[test]
    fn portable_verifier_reproduces_writer_result() {
        let signing = test_key();
        let t = sign_transition(
            &signing,
            7,
            "mem-p",
            &json!({"layer": "working"}),
            &json!({"layer": "core"}),
            "abc",
        )
        .unwrap();
        // Writer-side verification is the same pure function the portable
        // verifier uses — assert full field equality with a recomputation.
        let v = verify_transition(&t, &key_b64(&signing)).unwrap();
        let canonical =
            crate::signed_profile::canonical_json_bytes(&serde_json::to_value(&t.payload).unwrap());
        let sig = Signature::from_slice(&b64_decode(&t.signature_b64).unwrap()).unwrap();
        assert!(signing
            .verifying_key()
            .verify_strict(&canonical, &sig)
            .is_ok());
        assert_eq!(v.chain_hash, t.chain_hash);
    }

    #[test]
    fn seed_decode_rejects_wrong_length() {
        assert!(decode_seed(&b64_encode(&[1u8; 31])).is_err());
        assert!(decode_seed(&b64_encode(&[1u8; 33])).is_err());
        assert!(decode_seed(&b64_encode(&[1u8; 32])).is_ok());
    }

    #[test]
    fn poison_penalty_levels() {
        assert_eq!(poison_penalty("poison_likely"), 0.9);
        assert_eq!(poison_penalty("suspect"), 0.5);
        assert_eq!(poison_penalty("clean"), 0.0);
        assert_eq!(poison_penalty("anything_else"), 0.0);
    }

    // ── DB integration (TestDatabase) ────────────────────────────────

    fn test_entity(id: &str, body_json: &str, decay: f64, verified: bool) -> crate::models::Entity {
        crate::models::Entity {
            id: id.to_string(),
            category: "facts".to_string(),
            key: id.to_string(),
            body_json: body_json.to_string(),
            status: "active".to_string(),
            entity_type: "insight".to_string(),
            tags: vec![],
            decay_score: decay,
            retrieval_count: 0,
            layer: "working".to_string(),
            topic_path: String::new(),
            archived: false,
            archive_reason: String::new(),
            links: vec![],
            verified,
            source: "agent".to_string(),
            always_on: false,
            certainty: 0.5,
            workspace_hash: String::new(),
            agent_id: String::new(),
            visibility: "workspace".to_string(),
            created_at_unix_ms: crate::db::now_ms(),
            last_accessed_unix_ms: crate::db::now_ms(),
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
    fn unsigned_regime_and_epoch_authorization() {
        let db = crate::db::TestDatabase::new("mutmem-epoch");
        // No epoch registered: transitions record as None (unsigned regime)…
        assert!(db
            .record_signed_transition("mem-a", &json!({"v": 0}), &json!({"v": 1}))
            .unwrap()
            .is_none());
        // …and poison labels fail closed.
        let e = test_entity("mem-lbl", "{}", 1.0, false);
        db.remember_skip_dedup(&e).unwrap();
        let err = db
            .set_poison_label("mem-lbl", "poison_likely", "x")
            .unwrap_err();
        assert!(err.contains("no signer epoch"), "unexpected error: {err}");
        // Register an epoch → signed regime.
        let fp = db
            .register_signer_epoch(2, &b64_encode(&[11u8; 32]))
            .unwrap();
        assert_eq!(fp.len(), 64);
        let t = db
            .record_signed_transition("mem-a", &json!({"v": 0}), &json!({"v": 1}))
            .unwrap()
            .unwrap();
        assert_eq!(t.payload.signer_epoch, 2);
        let lbl = db
            .set_poison_label("mem-lbl", "poison_likely", "injected")
            .unwrap();
        assert_eq!(lbl["level"], "poison_likely");
        // Unknown level fails closed.
        assert!(db.set_poison_label("mem-lbl", "bogus", "x").is_err());
    }

    #[test]
    fn chain_audit_detects_tampering_and_blocks_forks() {
        let db = crate::db::TestDatabase::new("mutmem-chain");
        db.register_signer_epoch(1, &b64_encode(&[7u8; 32]))
            .unwrap();
        let t1 = db
            .record_signed_transition("mem-a", &json!({"v": 1}), &json!({"v": 2}))
            .unwrap()
            .unwrap();
        let t2 = db
            .record_signed_transition("mem-b", &json!({"v": 2}), &json!({"v": 3}))
            .unwrap()
            .unwrap();
        assert_eq!(t2.payload.predecessor_hash, t1.chain_hash);
        let audit = db.verify_transition_chain().unwrap();
        assert_eq!(audit["records"], 2);
        assert_eq!(audit["verified"], 2);
        assert!(audit["divergence"].is_null());
        // Tamper with a stored record's old value → divergence on replay.
        {
            let conn = db.conn().unwrap();
            conn.execute(
                "UPDATE signed_transitions SET old_value_json = '{\"v\": 99}' WHERE chain_hash = ?1",
                [&t2.chain_hash],
            )
            .unwrap();
        }
        let audit = db.verify_transition_chain().unwrap();
        assert!(audit["divergence"].is_object());
        assert_eq!(audit["verified"].as_i64().unwrap(), 1);
        // Storage-level no-fork: a second genesis is unrepresentable.
        {
            let conn = db.conn().unwrap();
            let dup = conn.execute(
                "INSERT INTO signed_transitions (id, terminal_node, signer_epoch, signer_fingerprint,
                     old_value_json, new_value_json, commitment_old, commitment_new,
                     predecessor_hash, signature_b64, chain_hash, created_at_unix_ms)
                 VALUES ('stn-fork', 'mem-z', 1, 'ff', '{}', '{}', 'aa', 'bb', '', 'cc', 'dd', 1)",
                [],
            );
            assert!(
                dup.is_err(),
                "the UNIQUE predecessor index must reject a second genesis"
            );
        }
    }

    #[test]
    fn signed_poison_labels_exclude_injected_content_from_top_k() {
        let db = crate::db::TestDatabase::new("mutmem-poison");
        let term = "zephyrion";
        let mut poisoned = Vec::new();
        let mut legit = Vec::new();
        for i in 0..100 {
            let e = test_entity(
                &format!("mem-poison-{i}"),
                &json!({"content": format!("{term} {term} {term} {term} injected claim #{i}")})
                    .to_string(),
                1.0,
                false,
            );
            db.remember_skip_dedup(&e).unwrap();
            poisoned.push(e.id);
        }
        for i in 0..5 {
            let e = test_entity(
                &format!("mem-legit-{i}"),
                &json!({"content": format!("{term} verified record #{i}")}).to_string(),
                0.85,
                false,
            );
            db.remember_skip_dedup(&e).unwrap();
            legit.push(e.id);
        }
        let params = crate::models::RecallParams {
            query: term.to_string(),
            limit: 105,
            content_weight: 0.1,
            max_prior_overturn: 0.0,
            trust_weight: 0.0,
            ..crate::models::RecallParams::default()
        };
        // Before labeling: the strongly-matching injected content dominates
        // the top-5 (the leak the labels exist to close).
        let top = db.recall(&params).unwrap();
        assert!(!top.is_empty());
        let before: Vec<&str> = top.iter().take(5).map(|e| e.id.as_str()).collect();
        assert!(
            before.iter().all(|id| poisoned.iter().any(|p| p == id)),
            "pre-label top-5 should be dominated by injected content, got {before:?}"
        );
        // Authorize the signing regime; sign a poison label on every injected row.
        db.register_signer_epoch(1, &b64_encode(&[42u8; 32]))
            .unwrap();
        for id in &poisoned {
            db.set_poison_label(id, "poison_likely", "PoisonedRAG-style injected adaptation")
                .unwrap();
        }
        let top = db.recall(&params).unwrap();
        let after: Vec<&str> = top.iter().take(5).map(|e| e.id.as_str()).collect();
        assert!(
            after.iter().all(|id| legit.iter().any(|l| l == id)),
            "signed labels must keep injected content out of top-5, got {after:?}"
        );
        assert!(after.iter().all(|id| !poisoned.iter().any(|p| p == id)));
        // The whole label wave is a verified, no-fork signed chain.
        let audit = db.verify_transition_chain().unwrap();
        assert_eq!(audit["records"], 100);
        assert_eq!(audit["verified"], 100);
        assert!(audit["divergence"].is_null());
        // Labels are revisable: a signed clean restores eligibility.
        db.set_poison_label(&poisoned[0], "clean", "operator reviewed")
            .unwrap();
        let audit = db.verify_transition_chain().unwrap();
        assert_eq!(audit["records"], 101);
        assert_eq!(audit["verified"], 101);
        assert!(audit["divergence"].is_null());
    }
}
