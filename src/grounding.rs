//! #1034: deterministic grounding verification — symbol fingerprints +
//! MOVED/GONE reconcile for rendered evidence.
//!
//! Borrowed from mex-memory/mex (MIT, verified locally 2026-08-14) and
//! generalized to the vault's evidence model. A content fingerprint (K=64
//! seeded-sha256 trigram MinHash + neighbor set) is captured at admission
//! time — deterministic, zero LLM. On maintenance/hygiene passes the current
//! content is diffed against the baseline:
//!
//! - identical digest → `ok`;
//! - exists-but-changed → GROUNDING_DRIFT;
//! - reconcile score = 0.7×minhashJaccard + 0.3×neighborOverlap, with HI/LO
//!   thresholds: ≥HI against one candidate → MOVED (auto-rewrite the anchor +
//!   migrate the baseline, with a supersede-style provenance trail — never
//!   silent last-write-wins); <LO → GONE (flag for review); in-band or
//!   multi-candidate → AMBIGUOUS (surface candidates for operator review).
//!
//! Fail-closed authoring rule adopted verbatim from mex: "If trustworthy
//! ground facts are unavailable, stop and report it. Never invent node ids
//! or fingerprints."
use std::collections::HashSet;

/// MinHash signature width.
pub const K: usize = 64;
/// Minimum trigram count for a fingerprintable body. Below this the content
/// is too short to ground deterministically → admission refused (fail-closed).
pub const MIN_TOKENS: usize = 30;
/// Reconcile-score HI threshold: ≥HI → MOVED (single candidate).
pub const HI: f64 = 0.85;
/// Reconcile-score LO threshold: <LO → GONE (no plausible moved candidate).
pub const LO: f64 = 0.55;
/// MinHash weight in the reconcile score (mex's constant).
pub const MINHASH_WEIGHT: f64 = 0.7;
/// Neighbor-overlap weight in the reconcile score (mex's constant).
pub const NEIGHBOR_WEIGHT: f64 = 0.3;
/// Max neighbor-set size (bounded for pathological bodies).
pub const MAX_NEIGHBORS: usize = 4096;
/// Max body length admitted for fingerprinting (bounded for pathological
/// bodies).
pub const MAX_BODY_LEN: usize = 256 * 1024;
/// Fixed derivation seed for all vault fingerprints (deterministic corpus).
pub const SEED: u64 = 0x5045_5253_4555_53;
/// Max neighbor-set size persisted per grounding row (bounded storage;
/// sufficient for the overlap estimate).
pub const MAX_STORED_NEIGHBORS: usize = 256;

/// Extract normalized trigram shingles from a body. Trigrams are
/// byte-oriented (language-agnostic, deterministic). Non-alphanumeric runs
/// collapse to a single space; case is preserved (content fidelity).
pub fn trigrams(body: &str) -> Vec<String> {
    let mut cleaned: Vec<u8> = Vec::with_capacity(body.len());
    let mut last_space = false;
    for b in body.bytes() {
        if b.is_ascii_alphanumeric() || b == b'_' || b == b'-' || b == b'/' || b == b'.' {
            cleaned.push(b);
            last_space = false;
        } else if !last_space && !cleaned.is_empty() {
            cleaned.push(b' ');
            last_space = true;
        }
    }
    let s = String::from_utf8_lossy(&cleaned);
    let chars: Vec<&str> = s.split_whitespace().collect();
    if chars.is_empty() {
        return Vec::new();
    }
    // Space-joined tokens preserve word boundaries; trigrams slide over the
    // joined string by 3-byte window (mex-style shingles).
    let joined: Vec<u8> = chars.join(" ").into_bytes();
    if joined.len() < 3 {
        // Degenerate: single very short token; use the token itself as the
        // only shingle (still deterministic).
        return vec![String::from_utf8_lossy(&joined).to_string()];
    }
    (0..=joined.len() - 3)
        .map(|i| String::from_utf8_lossy(&joined[i..i + 3]).to_string())
        .collect()
}

fn sha256_bytes(data: &[u8]) -> [u8; 32] {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(data);
    let out = h.finalize();
    let mut arr = [0u8; 32];
    arr.copy_from_slice(&out);
    arr
}

/// K=64 seeded-sha256 trigram MinHash. Deterministic: identical bodies
/// produce identical signatures. Returns `None` when the body has fewer than
/// `MIN_TOKENS` trigrams (fail-closed: not fingerprintable).
pub fn minhash(body: &str, seed: u64) -> Option<Vec<u64>> {
    let tokens = trigrams(body);
    if tokens.len() < MIN_TOKENS {
        return None;
    }
    let mut out = Vec::with_capacity(K);
    for i in 0..K {
        let mut best: Option<u64> = None;
        for tok in &tokens {
            let mut buf = Vec::with_capacity(8 + 8 + tok.len());
            buf.extend_from_slice(&seed.to_le_bytes());
            buf.extend_from_slice(&(i as u64).to_le_bytes());
            buf.extend_from_slice(tok.as_bytes());
            let h = u64::from_le_bytes(sha256_bytes(&buf)[..8].try_into().unwrap());
            best = Some(match best {
                Some(b) => b.min(h),
                None => h,
            });
        }
        out.push(best.unwrap());
    }
    Some(out)
}

/// The neighbor set: unique trigram shingles (bounded at `MAX_NEIGHBORS`).
pub fn neighbor_set(body: &str) -> HashSet<String> {
    let tokens = trigrams(body);
    tokens.into_iter().take(MAX_NEIGHBORS).collect()
}

/// Estimated Jaccard similarity from two MinHash signatures (matching-hash
/// fraction, the classic K-width estimator).
pub fn minhash_jaccard(a: &[u64], b: &[u64]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let matches = a.iter().zip(b.iter()).filter(|(x, y)| x == y).count();
    matches as f64 / a.len() as f64
}

/// Jaccard overlap of two neighbor sets.
pub fn neighbor_overlap(a: &HashSet<String>, b: &HashSet<String>) -> f64 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let inter = a.intersection(b).count();
    let union = a.union(b).count();
    if union == 0 {
        return 0.0;
    }
    inter as f64 / union as f64
}

/// mex's reconcile score: 0.7×minhashJaccard + 0.3×neighborOverlap.
pub fn reconcile_score(
    a_sig: &[u64],
    b_sig: &[u64],
    a_neighbors: &HashSet<String>,
    b_neighbors: &HashSet<String>,
) -> f64 {
    MINHASH_WEIGHT * minhash_jaccard(a_sig, b_sig)
        + NEIGHBOR_WEIGHT * neighbor_overlap(a_neighbors, b_neighbors)
}

/// Fingerprint hex string: seed + K×u64 hex, colon-separated. The seed is
/// embedded so a signature is self-describing (which derivation produced it).
pub fn fingerprint_hex(body: &str, seed: u64) -> Option<String> {
    let sig = minhash(body, seed)?;
    let mut out = format!("seed={seed:016x};k={K};");
    for (i, h) in sig.iter().enumerate() {
        if i > 0 {
            out.push(':');
        }
        out.push_str(&format!("{h:016x}"));
    }
    Some(out)
}

/// Parse a fingerprint hex string back into (seed, signature). Returns `None`
/// on malformed input (fail-closed: never guess).
pub fn parse_fingerprint(fp: &str) -> Option<(u64, Vec<u64>)> {
    // Format: "seed=<hex>;k=<K>;<h1>:<h2>:..." (the k terminator is a
    // semicolon; hashes are colon-separated).
    let (head, body) = fp.split_once(';')?;
    let seed = head.strip_prefix("seed=")?;
    let seed = u64::from_str_radix(seed, 16).ok()?;
    let (k_part, hashes_part) = body.split_once(';')?;
    let k_val: usize = k_part.strip_prefix("k=")?.parse().ok()?;
    if k_val != K {
        return None;
    }
    let hashes: Option<Vec<u64>> = hashes_part
        .split(':')
        .map(|h| u64::from_str_radix(h, 16).ok())
        .collect();
    Some((seed, hashes?))
}

fn bytes_to_hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push_str(&format!("{b:02x}"));
    }
    out
}

/// sha256 hex digest of a body — the cheap identical-content fast path and
/// the baseline digest stored at admission.
pub fn content_digest(body: &str) -> String {
    format!("sha256:{}", bytes_to_hex(&sha256_bytes(body.as_bytes())))
}

/// Reconcile classification for one grounding whose current content changed
/// or disappeared.
#[derive(Debug, Clone, PartialEq)]
pub enum ReconcileClass {
    Ok,
    Drift,
    Moved { to_target: String, score: f64 },
    Gone,
    Ambiguous { candidates: Vec<(String, f64)> },
}

#[cfg(test)]
mod tests {
    use super::*;

    const SEED: u64 = 0xC0FFEE;
    const BODY: &str = "fn compute_total(items: &[u32]) -> u64 { \
        // Sum every item, saturating at u64::MAX, with a per-item audit note. \
        let mut total: u64 = 0; \
        for (index, item) in items.iter().enumerate() { \
            let contribution = *item as u64; \
            total = total.saturating_add(contribution); \
            debug_assert!(total >= contribution, \"overflow impossible under saturating add\"); \
            if index % 100 == 0 { log::trace!(\"progress\", index = index); } \
        } \
        total \
    }";

    #[test]
    fn fingerprints_are_deterministic_and_content_sensitive() {
        let a = fingerprint_hex(BODY, SEED).unwrap();
        let b = fingerprint_hex(BODY, SEED).unwrap();
        assert_eq!(
            a, b,
            "identical content must produce identical fingerprints"
        );
        let mutated = BODY.replace("u64", "u128");
        let c = fingerprint_hex(&mutated, SEED).unwrap();
        assert_ne!(a, c, "mutated content must drift");
        // Parse round-trip.
        let (seed, sig) = parse_fingerprint(&a).unwrap();
        assert_eq!(seed, SEED);
        assert_eq!(sig.len(), K);
    }

    #[test]
    fn short_content_is_not_fingerprintable() {
        assert!(fingerprint_hex("tiny", SEED).is_none());
        assert!(fingerprint_hex("", SEED).is_none());
    }

    #[test]
    fn minhash_jaccard_estimates_similarity() {
        let sig_a = minhash(BODY, SEED).unwrap();
        let sig_same = minhash(BODY, SEED).unwrap();
        assert!((minhash_jaccard(&sig_a, &sig_same) - 1.0).abs() < 1e-9);
        let other = "completely different content about nothing in common at all, long enough to fingerprint deterministically";
        let sig_other = minhash(other, SEED).unwrap();
        assert!(minhash_jaccard(&sig_a, &sig_other) < 0.3);
    }

    #[test]
    fn reconcile_score_prefers_moved_content_over_foreign_content() {
        let moved = format!("{BODY} // relocated comment");
        let sig_base = minhash(BODY, SEED).unwrap();
        let sig_moved = minhash(&moved, SEED).unwrap();
        let nb_base = neighbor_set(BODY);
        let nb_moved = neighbor_set(&moved);
        let score = reconcile_score(&sig_base, &sig_moved, &nb_base, &nb_moved);
        assert!(
            score >= HI,
            "moved content should clear the HI threshold: {score}"
        );
        let foreign = "fn unrelated() { let x = vec![1,2,3,4,5,6,7,8,9]; println!(\"{}\", x.iter().sum::<i32>()); }";
        let sig_foreign = minhash(foreign, SEED).unwrap();
        let nb_foreign = neighbor_set(foreign);
        let foreign_score = reconcile_score(&sig_base, &sig_foreign, &nb_base, &nb_foreign);
        assert!(
            foreign_score < LO,
            "foreign content should fall below the LO threshold: {foreign_score}"
        );
    }
}
