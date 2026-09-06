//! Signed, distributable policy/authority profiles (perseus-vault #837).
//!
//! Sigstore-style attestation for manifest/policy bundles: a profile is
//! canonical JSON carrying an authority-manifest payload, the signer's
//! Ed25519 public key, and a signature over the canonicalized payload.
//! On load the profile is verified before the manifest takes effect;
//! verification failure = no authority (fail closed). The verification
//! result (identity fingerprint, payload digest, outcome) is recorded in the
//! ledger by the caller (`Database::authority_set_signed`).
//!
//! Bounded scope: this is content signing for manifest/policy loading, not a
//! universal security claim from one attestation scheme, and not a key
//! management system.

use ed25519_dalek::{Signature, Signer, SigningKey, PUBLIC_KEY_LENGTH};
use serde_json::Value;

/// Profile schema version this verifier accepts.
pub const PROFILE_SCHEMA: &str = "perseus-policy-profile/v1";

/// Canonical bytes of a JSON value: sorted keys, no insignificant
/// whitespace. The signature binds these exact bytes, so a tampered field —
/// or a re-ordered one — breaks verification.
pub fn canonical_json_bytes(value: &Value) -> Vec<u8> {
    canonical_value(value).as_bytes().to_vec()
}

fn canonical_value(value: &Value) -> String {
    match value {
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            let body: Vec<String> = keys
                .iter()
                .map(|k| format!("{k}:{}", canonical_value(&map[*k])))
                .collect();
            format!("{{{}}}", body.join(","))
        }
        Value::Array(items) => {
            let body: Vec<String> = items.iter().map(canonical_value).collect();
            format!("[{}]", body.join(","))
        }
        other => other.to_string(),
    }
}

/// The result of a successful profile verification: the signer's key
/// fingerprint and the SHA-256 of the canonical payload.
#[derive(Debug, Clone)]
pub struct ProfileVerification {
    pub signer_fingerprint: String,
    pub payload_digest: String,
}

/// Decode a raw 32-byte Ed25519 public key from base64.
fn decode_public_key(b64: &str) -> Result<ed25519_dalek::VerifyingKey, String> {
    let bytes = base64_decode(b64)?;
    if bytes.len() != PUBLIC_KEY_LENGTH {
        return Err(format!(
            "public key must be {PUBLIC_KEY_LENGTH} raw bytes, got {}",
            bytes.len()
        ));
    }
    let mut arr = [0u8; PUBLIC_KEY_LENGTH];
    arr.copy_from_slice(&bytes);
    ed25519_dalek::VerifyingKey::from_bytes(&arr)
        .map_err(|e| format!("invalid Ed25519 public key: {e}"))
}

fn base64_decode(b64: &str) -> Result<Vec<u8>, String> {
    use base64::Engine as _;
    base64::engine::general_purpose::STANDARD
        .decode(b64.trim())
        .map_err(|e| format!("invalid base64: {e}"))
}

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

/// Verify a signed profile against a trusted public key.
///
/// Fails closed: a malformed, unsigned, tampered, or wrong-key profile is
/// rejected with an error and grants no authority. On success returns the
/// signer fingerprint and the canonical payload digest.
pub fn verify_profile(
    profile_json: &str,
    trusted_public_key_b64: &str,
) -> Result<ProfileVerification, String> {
    let profile: Value = serde_json::from_str(profile_json)
        .map_err(|e| format!("profile is not valid JSON: {e}"))?;
    let obj = profile.as_object().ok_or("profile must be a JSON object")?;
    if obj.get("schema").and_then(Value::as_str) != Some(PROFILE_SCHEMA) {
        return Err(format!("profile schema must be {PROFILE_SCHEMA}"));
    }
    let signer_b64 = obj
        .get("signer_key_b64")
        .and_then(Value::as_str)
        .ok_or("profile missing signer_key_b64")?;
    let payload = obj.get("payload").ok_or("profile missing payload")?;
    let signature_b64 = obj
        .get("signature_b64")
        .and_then(Value::as_str)
        .ok_or("profile missing signature_b64")?;

    let public = decode_public_key(signer_b64)?;
    // Identity check: the embedded signer must be the trusted key.
    if signer_b64.trim() != trusted_public_key_b64.trim() {
        return Err("profile signer is not the trusted key".into());
    }

    let canonical = canonical_json_bytes(payload);
    let signature = Signature::from_slice(&base64_decode(signature_b64)?)
        .map_err(|e| format!("invalid signature encoding: {e}"))?;
    public
        .verify_strict(&canonical, &signature)
        .map_err(|_| "profile signature verification failed (tampered or unsigned)".to_string())?;

    let fingerprint = sha256_hex(&public.to_bytes());
    let payload_digest = sha256_hex(&canonical);
    Ok(ProfileVerification {
        signer_fingerprint: fingerprint,
        payload_digest,
    })
}

/// Sign a profile payload (test/authoring helper). Returns the full profile
/// document with the raw (uncompressed) public key embedded.
pub fn sign_profile(signing_key_bytes: &[u8; 32], payload: &Value) -> Result<Value, String> {
    use base64::Engine as _;
    let signing = SigningKey::from_bytes(signing_key_bytes);
    let canonical = canonical_json_bytes(payload);
    let signature = signing.sign(&canonical);
    let verifying = signing.verifying_key();
    let mut obj = serde_json::Map::new();
    obj.insert("schema".into(), Value::String(PROFILE_SCHEMA.to_string()));
    obj.insert(
        "signer_key_b64".into(),
        Value::String(base64::engine::general_purpose::STANDARD.encode(verifying.to_bytes())),
    );
    obj.insert("payload".into(), payload.clone());
    obj.insert(
        "signature_b64".into(),
        Value::String(base64::engine::general_purpose::STANDARD.encode(signature.to_bytes())),
    );
    Ok(Value::Object(obj))
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::Engine as _;
    use serde_json::json;

    fn test_key() -> SigningKey {
        SigningKey::from_bytes(&[7u8; 32])
    }

    #[test]
    fn verify_accepts_a_profile_signed_by_the_trusted_key() {
        let signing = test_key();
        let payload = json!({"agent_id": "agent-837", "workspace_hash": "ws-837",
                             "allowed_capabilities": ["tool.run"], "mode": "enforce"});
        let profile = sign_profile(signing.as_bytes(), &payload).unwrap();
        let trusted =
            base64::engine::general_purpose::STANDARD.encode(signing.verifying_key().to_bytes());
        let v = verify_profile(&profile.to_string(), &trusted).unwrap();
        assert_eq!(v.payload_digest.len(), 64);
        assert_eq!(v.signer_fingerprint.len(), 64);
    }

    #[test]
    fn verify_rejects_a_tampered_payload() {
        let signing = test_key();
        let payload = json!({"agent_id": "agent-837", "workspace_hash": "ws-837",
                             "allowed_capabilities": ["tool.run"], "mode": "enforce"});
        let mut profile = sign_profile(signing.as_bytes(), &payload).unwrap();
        profile["payload"]["allowed_capabilities"] = json!(["rm_rf"]);
        let trusted =
            base64::engine::general_purpose::STANDARD.encode(signing.verifying_key().to_bytes());
        let err = verify_profile(&profile.to_string(), &trusted).unwrap_err();
        assert!(err.contains("verification failed"), "got: {err}");
    }

    #[test]
    fn verify_rejects_a_profile_signed_by_an_unknown_key() {
        let signing = test_key();
        let payload = json!({"agent_id": "agent-837", "workspace_hash": "ws-837",
                             "allowed_capabilities": ["tool.run"]});
        let profile = sign_profile(signing.as_bytes(), &payload).unwrap();
        let trusted = base64::engine::general_purpose::STANDARD.encode([9u8; 32]); // different trusted key
        let err = verify_profile(&profile.to_string(), &trusted).unwrap_err();
        assert!(err.contains("not the trusted key"), "got: {err}");
    }

    #[test]
    fn verify_rejects_an_unsigned_profile() {
        let payload = json!({"agent_id": "agent-837", "workspace_hash": "ws-837",
                             "allowed_capabilities": ["tool.run"]});
        let profile = json!({"schema": PROFILE_SCHEMA, "payload": payload,
                             "signer_key_b64": "AAAA", "signature_b64": "AAAA"});
        let err = verify_profile(&profile.to_string(), "AAAA").unwrap_err();
        assert!(!err.is_empty());
    }
}
