//! Optional quantized embedding storage (#885) — MIB-style vector compression.
//!
//! The `entities.embedding` BLOB column historically stored raw little-endian
//! f32 vectors (4·dim bytes). This module adds two opt-in quantized encodings,
//! each self-describing via a one-byte tag prefix so reads never depend on a
//! store-wide flag (a mixed corpus decodes correctly row by row):
//!
//! | format | tag byte | payload | 384-dim size | vs float32 |
//! |--------|----------|---------|--------------|------------|
//! | float32 (legacy) | none | 4·dim LE f32 bytes | 1536 | 1.00× |
//! | int8 | `0x01` | f32 LE scale + dim i8 codes | 389 | 0.25× |
//! | bit | `0x02` | dim/8 sign bits | 49 | 0.032× |
//!
//! - **int8**: per-vector scale `max|v|/127` (0 for an all-zero vector),
//!   `code_i = clamp(round(v_i/scale), -127, 127)`. Decodes to
//!   `v_i ≈ scale·code_i`; cosine ranking on the approximation tracks the
//!   exact ranking (standard scalar-quantization recall behavior).
//! - **bit**: sign bits — bit i set iff `v[i] > 0.0`, byte-packed with the
//!   SAME rule as `db::embedding_signature`, so a bit-stored vector's payload
//!   is byte-identical to its `emb_sig`. Distance is Hamming over the stored
//!   bits (in-store distance scoring), normalized to a similarity in [0,1]
//!   comparable with cosine.
//!
//! Decoding is length-validated and fail-closed: a blob that matches no known
//! layout for the query dim decodes to `None` and the caller's dim filter
//! drops the row — the same end state as the pre-existing dim-mismatch path
//! (mixed embedding backends). Layouts are unambiguous for real dims
//! (384/768/1536): `4d` vs `5+d` vs `1+d/8` never collide. (Pathological
//! dim-1 float32 vs dim-24 bit rows would collide at length 4; float32 is
//! checked first and wins deterministically.)
//!
//! float32 remains the default storage format; quantization is opt-in via
//! `PERSEUS_VAULT_EMBEDDING_QUANT` / `--embedding-quant` and the
//! `perseus_vault_embed` `quant_mode` reindex path (see
//! `docs/specs/vector-compression.md`).

use crate::db::signature_hamming;

/// Storage format of the `entities.embedding` BLOB column.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EmbeddingQuant {
    /// Raw little-endian f32 (legacy, default).
    F32,
    /// Tag `0x01` + f32 LE scale + dim i8 codes.
    Int8,
    /// Tag `0x02` + dim/8 sign bits (MIB-style).
    Bit,
}

/// Tag byte for int8-encoded blobs.
pub(crate) const TAG_INT8: u8 = 0x01;
/// Tag byte for bit-encoded blobs.
pub(crate) const TAG_BIT: u8 = 0x02;

impl EmbeddingQuant {
    /// Parse a config value: `float32`/`f32`/`none` → F32, `int8` → Int8,
    /// `bit`/`binary` → Bit. Anything else → None (fail-closed at config).
    pub fn parse(s: &str) -> Option<EmbeddingQuant> {
        match s.trim().to_ascii_lowercase().as_str() {
            "float32" | "f32" | "none" => Some(EmbeddingQuant::F32),
            "int8" => Some(EmbeddingQuant::Int8),
            "bit" | "binary" => Some(EmbeddingQuant::Bit),
            _ => None,
        }
    }

    /// Canonical config spelling (also the value stored in `embedding_format`).
    pub fn as_str(self) -> &'static str {
        match self {
            EmbeddingQuant::F32 => "float32",
            EmbeddingQuant::Int8 => "int8",
            EmbeddingQuant::Bit => "bit",
        }
    }

    /// Byte tag for the in-memory atomic (see `Database::embedding_quant`).
    pub(crate) fn to_byte(self) -> u8 {
        match self {
            EmbeddingQuant::F32 => 0,
            EmbeddingQuant::Int8 => 1,
            EmbeddingQuant::Bit => 2,
        }
    }

    /// Inverse of `to_byte`; unknown bytes resolve to F32 (the default).
    pub(crate) fn from_byte(b: u8) -> EmbeddingQuant {
        match b {
            1 => EmbeddingQuant::Int8,
            2 => EmbeddingQuant::Bit,
            _ => EmbeddingQuant::F32,
        }
    }
}

/// A decoded stored vector: either a full-precision (float32 or int8-approx)
/// vector that scores by cosine, or raw sign bits that score by Hamming.
#[derive(Clone, Debug, PartialEq)]
pub enum StoredVec {
    /// Full vector (float32 original, or the int8 scale·code approximation).
    Full(Vec<f32>),
    /// Raw sign bits (bit mode payload — byte-identical to `emb_sig`).
    Bits(Vec<u8>),
}

/// Encode `v` as int8: tag + f32 LE scale + dim i8 codes (little-endian
/// two's-complement bytes). `scale = max|v|/127`; an all-zero vector stores
/// scale 0 and all-zero codes (its dot products are 0 — matches the true
/// vector).
pub fn quantize_int8(v: &[f32]) -> Vec<u8> {
    let max_abs = v.iter().fold(0f32, |m, &x| m.max(x.abs()));
    let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 0.0 };
    let mut out = Vec::with_capacity(1 + 4 + v.len());
    out.push(TAG_INT8);
    out.extend_from_slice(&scale.to_le_bytes());
    for &x in v {
        let code = if scale == 0.0 {
            0i8
        } else {
            (x / scale).round().clamp(-127.0, 127.0) as i8
        };
        out.push(code as u8);
    }
    out
}

/// Encode `v` as bit: tag + sign bits (bit i set iff `v[i] > 0.0`, byte
/// order matches `db::embedding_signature` — the payload IS the signature).
pub fn quantize_bit(v: &[f32]) -> Vec<u8> {
    let sig = crate::db::embedding_signature(v);
    let mut out = Vec::with_capacity(1 + sig.len());
    out.push(TAG_BIT);
    out.extend_from_slice(&sig);
    out
}

/// Decode a stored `embedding` blob for a query of `dim` dimensions.
///
/// Returns `None` when the blob matches no known layout for `dim` — the
/// caller's dim filter drops the row (fail-closed; same end state as the
/// legacy dim-mismatch path). Layout priority: float32 first (legacy rows
/// carry no tag), then tagged int8, then tagged bit.
pub fn decode_stored(blob: &[u8], dim: usize) -> Option<StoredVec> {
    if blob.len() == dim * 4 {
        let v: Vec<f32> = blob
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        return Some(StoredVec::Full(v));
    }
    if blob.len() == 1 + 4 + dim && blob.first() == Some(&TAG_INT8) {
        let scale = f32::from_le_bytes([blob[1], blob[2], blob[3], blob[4]]);
        let v: Vec<f32> = blob[5..]
            .iter()
            .map(|&b| scale * (b as i8 as f32))
            .collect();
        return Some(StoredVec::Full(v));
    }
    if blob.len() == 1 + dim.div_ceil(8) && blob.first() == Some(&TAG_BIT) {
        return Some(StoredVec::Bits(blob[1..].to_vec()));
    }
    None
}

/// Classify a stored blob without a query dim — used by the reindex/restore
/// reporting (the store's own dim is derived from each row's layout).
/// Priority and collision behavior match `decode_stored`.
pub fn classify_stored(blob: &[u8]) -> Option<(EmbeddingQuant, usize)> {
    if blob.len() >= 4 && blob.len() % 4 == 0 {
        // Legacy float32 (checked first, matching decode_stored's priority).
        // A tagged row whose total length is also a multiple of 4 is
        // impossible for real dims (int8: 5+d ≡ 1 mod 4; bit: 1+d/8 has
        // d/8 ≡ 0 mod 4 only when d ≡ 0 mod 32, then total ≡ 1 mod 4).
        return Some((EmbeddingQuant::F32, blob.len() / 4));
    }
    if blob.len() >= 6 && blob.first() == Some(&TAG_INT8) {
        return Some((EmbeddingQuant::Int8, blob.len() - 5));
    }
    if blob.len() >= 2 && blob.first() == Some(&TAG_BIT) {
        return Some((EmbeddingQuant::Bit, (blob.len() - 1) * 8));
    }
    None
}

/// Hamming similarity of a query's sign bits against stored bit vectors:
/// `1 − hamming/dim` ∈ [0,1], comparable with cosine (higher = closer).
/// A length mismatch (foreign dim) scores 0 — it can never win a slot.
pub fn bit_similarity(query_bits: &[u8], stored_bits: &[u8]) -> f64 {
    if query_bits.is_empty() || query_bits.len() != stored_bits.len() {
        return 0.0;
    }
    let dim = (query_bits.len() * 8) as f64;
    let h = signature_hamming(query_bits, stored_bits) as f64;
    1.0 - h / dim
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantize_int8_round_trips_within_scale() {
        let v: Vec<f32> = (0..16).map(|i| (i as f32 - 8.0) * 0.5).collect();
        let blob = quantize_int8(&v);
        assert_eq!(blob.len(), 1 + 4 + 16);
        assert_eq!(blob[0], TAG_INT8);
        let scale = f32::from_le_bytes([blob[1], blob[2], blob[3], blob[4]]);
        // max|v| over [-4.0, 3.5] is 4.0 → scale = 4/127.
        assert!((scale - 4.0 / 127.0).abs() < 1e-6);
        let decoded = match decode_stored(&blob, 16) {
            Some(StoredVec::Full(v)) => v,
            other => panic!("expected Full, got {other:?}"),
        };
        // Max-|v| element survives exactly; quantization error is bounded by
        // half a code step for the rest.
        assert!((decoded[0] - v[0]).abs() < scale / 2.0);
        // Cosine between original and approx is near 1 for this smooth vector.
        let dot: f32 = v.iter().zip(&decoded).map(|(a, b)| a * b).sum();
        let na: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = decoded.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(dot / (na * nb) > 0.99);
    }

    #[test]
    fn quantize_int8_zero_vector_stores_zero_scale_and_codes() {
        let blob = quantize_int8(&[0.0; 8]);
        assert_eq!(blob[0], TAG_INT8);
        assert_eq!(
            f32::from_le_bytes([blob[1], blob[2], blob[3], blob[4]]),
            0.0
        );
        assert!(blob[5..].iter().all(|&b| b == 0));
    }

    #[test]
    fn quantize_bit_matches_embedding_signature_payload() {
        let v = [0.5, -0.5, 0.0, 1.0, -2.0, 0.1, -0.1, 3.0, -3.0];
        let blob = quantize_bit(&v);
        assert_eq!(blob[0], TAG_BIT);
        assert_eq!(blob.len(), 1 + 2); // 9 dims → 2 bytes
        let sig = crate::db::embedding_signature(&v);
        assert_eq!(&blob[1..], &sig[..]);
        // bit i set iff v[i] > 0: bits 0,3,5,7 → 1 + 8 + 32 + 128 = 169;
        // v[8] = -3.0 → byte 1 all zero.
        assert_eq!(blob[1], 0b1010_1001);
        assert_eq!(blob[2], 0b0000_0000);
    }

    #[test]
    fn decode_stored_f32_legacy_and_fail_closed_lengths() {
        let v: Vec<f32> = (0..12).map(|i| i as f32 * 0.25).collect();
        let mut blob = Vec::new();
        for f in &v {
            blob.extend_from_slice(&f.to_le_bytes());
        }
        assert_eq!(decode_stored(&blob, 12), Some(StoredVec::Full(v.clone())));
        // Wrong dim → None (fail-closed), never garbage.
        assert_eq!(decode_stored(&blob, 3), None);
        // Truncated blob → None.
        assert_eq!(decode_stored(&blob[..blob.len() - 1], 12), None);
        // Empty → None.
        assert_eq!(decode_stored(&[], 384), None);
    }

    #[test]
    fn decode_stored_int8_requires_tag_and_exact_length() {
        let v = [1.0, -2.0, 3.0, -4.0];
        let blob = quantize_int8(&v);
        assert!(matches!(decode_stored(&blob, 4), Some(StoredVec::Full(_))));
        // Tag stripped → misread as nothing (len 4 ≠ 16, no tag) → None.
        assert_eq!(decode_stored(&blob[1..], 4), None);
        // Wrong dim → None.
        assert_eq!(decode_stored(&blob, 5), None);
    }

    #[test]
    fn decode_stored_bit_requires_tag_and_exact_length() {
        let v = [1.0, -1.0, 1.0, -1.0];
        let blob = quantize_bit(&v);
        assert!(matches!(decode_stored(&blob, 4), Some(StoredVec::Bits(_))));
        // Wrong dim → None (dim 16 expects 1+16/8 = 3 bytes, blob is 2).
        assert_eq!(decode_stored(&blob, 16), None);
        // Tag stripped → misread as nothing (1 byte ≠ 4·dim, no tag) → None.
        assert_eq!(decode_stored(&blob[1..], 4), None);
    }

    #[test]
    fn bit_similarity_ranks_closer_bits_higher() {
        let a = [0b1111_0000u8, 0b1010_1010];
        let b = [0b1111_0000u8, 0b1010_1010]; // identical
        let c = [0b0000_1111u8, 0b0101_0101]; // 16 bits differ
        assert_eq!(bit_similarity(&a, &b), 1.0);
        assert_eq!(bit_similarity(&a, &c), 0.0);
        assert!(bit_similarity(&a, &b) > bit_similarity(&a, &c));
        // Length mismatch → 0 (never wins).
        assert_eq!(bit_similarity(&a, &a[..1]), 0.0);
    }

    #[test]
    fn parse_accepts_documented_spellings_and_rejects_garbage() {
        assert_eq!(EmbeddingQuant::parse("float32"), Some(EmbeddingQuant::F32));
        assert_eq!(EmbeddingQuant::parse("F32"), Some(EmbeddingQuant::F32));
        assert_eq!(EmbeddingQuant::parse("none"), Some(EmbeddingQuant::F32));
        assert_eq!(EmbeddingQuant::parse("int8"), Some(EmbeddingQuant::Int8));
        assert_eq!(EmbeddingQuant::parse("bit"), Some(EmbeddingQuant::Bit));
        assert_eq!(EmbeddingQuant::parse("binary"), Some(EmbeddingQuant::Bit));
        assert_eq!(EmbeddingQuant::parse("int4"), None);
        assert_eq!(EmbeddingQuant::parse(""), None);
        assert_eq!(EmbeddingQuant::as_str(EmbeddingQuant::F32), "float32");
        assert_eq!(EmbeddingQuant::as_str(EmbeddingQuant::Int8), "int8");
        assert_eq!(EmbeddingQuant::as_str(EmbeddingQuant::Bit), "bit");
    }

    #[test]
    fn classify_stored_detects_each_layout() {
        assert_eq!(classify_stored(&[0u8; 8]), Some((EmbeddingQuant::F32, 2)));
        let i8b = quantize_int8(&[0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8]);
        assert_eq!(classify_stored(&i8b), Some((EmbeddingQuant::Int8, 8)));
        let bb = quantize_bit(&[1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]);
        assert_eq!(classify_stored(&bb), Some((EmbeddingQuant::Bit, 8)));
        assert_eq!(classify_stored(&[]), None);
        // A 5-byte f32 blob is dim 1.25 — impossible; tags must not fire on
        // random bytes when the length is also a multiple of 4 (priority).
        assert_eq!(
            classify_stored(&[0x01, 0, 0, 0, 0, 0, 0, 0]),
            Some((EmbeddingQuant::F32, 2))
        );
    }
}
