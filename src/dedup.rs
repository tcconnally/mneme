//! Stored per-row near-duplicate signatures (#392).
//!
//! `find_near_duplicate` used to rebuild a candidate's character-trigram set
//! from `body_json` on every insert — 30µs per candidate at 1KB bodies, so a
//! 50k-row category cost ~1.6s per `remember()`. This module stores each
//! row's trigram set ONCE at write time and lets the dedup scan compute the
//! exact same Jaccard verdict from the stored signature.
//!
//! LOSSLESSNESS is the design contract: every function here either computes a
//! value IDENTICAL to what the exhaustive trigram scan computes, or a bound
//! that can only ever skip candidates whose verdict is provably "not a
//! near-duplicate". The pieces:
//!
//! - `pack_trigram`: a `[char; 3]` trigram packed into a `u64`. Each `char`
//!   is a Unicode scalar value <= U+10FFFF (21 bits), so three fit in 63 bits
//!   with no truncation — the packing is INJECTIVE. Set cardinalities and
//!   intersections over packed values are therefore exactly those of the
//!   original trigram sets, and Jaccard over packed sets equals Jaccard over
//!   `HashSet<[char; 3]>` bit-for-bit (same integer counts, same f64 division).
//! - `encode_sig` / streaming decode: the sorted packed set, delta-encoded
//!   with LEB128 varints (sorted trigrams of real text sit close together, so
//!   this is ~2x smaller than raw u64s). Decoding validates strict
//!   monotonicity and element count; anything malformed is reported so the
//!   caller can fall back to rebuilding from the body.
//! - `histogram`: 256-bucket occupancy counts of the packed set. For two sets
//!   A and B, `|A ∩ B| <= Σ_j min(hA[j], hB[j])` — intersection members in
//!   bucket j number at most min of the two bucket sizes. Buckets are u8; a
//!   set that would overflow any bucket gets NO histogram (`None`) rather
//!   than a saturated one, because a clamped count could understate the
//!   ceiling and turn the prune lossy.
//! - `jaccard_verdict_from_sig`: merge-intersects the target set against the
//!   decoded signature with early abandon. It returns the verdict of
//!   `exact_jaccard(A, B) >= threshold` — abandoning only when even the
//!   maximum possible remaining intersection cannot reach the threshold.

/// Pack one character trigram into a `u64`. Injective: each `char` is a
/// Unicode scalar value (`c as u32 <= 0x10FFFF < 2^21`), so the three
/// 21-bit fields never overlap or truncate.
#[inline]
fn pack_trigram(c0: char, c1: char, c2: char) -> u64 {
    ((c0 as u64) << 42) | ((c1 as u64) << 21) | (c2 as u64)
}

/// The packed, sorted, deduplicated character-trigram set of `s`.
/// Cardinality equals `|trigrams(s)|` exactly (injectivity of `pack_trigram`).
pub fn packed_trigrams(s: &str) -> Vec<u64> {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() < 3 {
        return Vec::new();
    }
    let mut packed: Vec<u64> = chars
        .windows(3)
        .map(|w| pack_trigram(w[0], w[1], w[2]))
        .collect();
    packed.sort_unstable();
    packed.dedup();
    packed
}

/// Exact Jaccard over two packed trigram sets (sorted, deduplicated).
/// Bit-for-bit identical to `trigram_overlap` on the corresponding
/// `HashSet<[char; 3]>` sets: same empty-set guards, same integer
/// intersection/union counts, same single f64 division.
pub fn exact_jaccard(a: &[u64], b: &[u64]) -> f64 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let mut i = 0usize;
    let (mut ai, mut bi) = (0usize, 0usize);
    while ai < a.len() && bi < b.len() {
        match a[ai].cmp(&b[bi]) {
            std::cmp::Ordering::Less => ai += 1,
            std::cmp::Ordering::Greater => bi += 1,
            std::cmp::Ordering::Equal => {
                i += 1;
                ai += 1;
                bi += 1;
            }
        }
    }
    let union = a.len() + b.len() - i;
    if union == 0 {
        return 0.0;
    }
    i as f64 / union as f64
}

/// Delta + LEB128-varint encoding of a sorted, deduplicated packed set.
/// First element absolute, the rest strictly-positive deltas.
pub fn encode_sig(sorted_set: &[u64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(sorted_set.len() * 4);
    let mut prev = 0u64;
    for (idx, &v) in sorted_set.iter().enumerate() {
        let delta = if idx == 0 { v } else { v - prev };
        let mut d = delta;
        loop {
            let byte = (d & 0x7F) as u8;
            d >>= 7;
            if d == 0 {
                out.push(byte);
                break;
            }
            out.push(byte | 0x80);
        }
        prev = v;
    }
    out
}

/// Streaming decoder for `encode_sig` blobs. Yields values in strictly
/// increasing order; any malformed input (varint overrun, zero delta after
/// the first element, truncation) surfaces as `Err(())` from `next_value`.
struct SigDecoder<'a> {
    buf: &'a [u8],
    pos: usize,
    prev: u64,
    first: bool,
}

impl<'a> SigDecoder<'a> {
    fn new(buf: &'a [u8]) -> Self {
        SigDecoder {
            buf,
            pos: 0,
            prev: 0,
            first: true,
        }
    }

    fn exhausted(&self) -> bool {
        self.pos >= self.buf.len()
    }

    /// Next decoded value, or Err(()) on malformed input.
    fn next_value(&mut self) -> Result<u64, ()> {
        let mut delta = 0u64;
        let mut shift = 0u32;
        loop {
            let &byte = self.buf.get(self.pos).ok_or(())?;
            self.pos += 1;
            if shift >= 64 {
                return Err(());
            }
            delta |= ((byte & 0x7F) as u64) << shift;
            if byte & 0x80 == 0 {
                break;
            }
            shift += 7;
        }
        let v = if self.first {
            self.first = false;
            delta
        } else {
            // Strictly increasing: a zero delta (duplicate) or overflow is
            // malformed.
            if delta == 0 {
                return Err(());
            }
            self.prev.checked_add(delta).ok_or(())?
        };
        self.prev = v;
        Ok(v)
    }
}

/// Number of histogram buckets. 256 keeps the per-row overhead at 256 bytes
/// while making the min-sum intersection ceiling sharp enough to prune
/// unrelated kilobyte-scale bodies (expected ~4 trigrams/bucket at 1KB).
const HISTO_BUCKETS: usize = 256;

/// Bucket index for a packed trigram. The multiplier is the 64-bit golden
/// ratio (Fibonacci hashing) so trigrams differing only in one character
/// still spread across buckets — a plain `% 256` would collapse everything
/// onto the low bits of the LAST character.
#[inline]
fn histo_bucket(v: u64) -> usize {
    ((v.wrapping_mul(0x9E37_79B9_7F4A_7C15)) >> 56) as usize
}

/// 256-bucket occupancy histogram of a packed set, or `None` if any bucket
/// would exceed u8::MAX. `None` (rather than clamping) keeps the min-sum
/// prune provably lossless: a saturated count could understate the true
/// bucket size and with it the intersection ceiling.
pub fn histogram(sorted_set: &[u64]) -> Option<Vec<u8>> {
    let mut h = vec![0u8; HISTO_BUCKETS];
    for &v in sorted_set {
        let b = histo_bucket(v);
        if h[b] == u8::MAX {
            return None;
        }
        h[b] += 1;
    }
    Some(h)
}

/// Lossless intersection ceiling from two histograms:
/// `|A ∩ B| <= Σ_j min(hA[j], hB[j])`, because the intersection members that
/// fall in bucket j are members of both A's and B's bucket-j populations.
pub fn histo_intersection_ceiling(ha: &[u8], hb: &[u8]) -> usize {
    ha.iter()
        .zip(hb.iter())
        .map(|(&x, &y)| x.min(y) as usize)
        .sum()
}

/// Stable 64-bit content hash of a stored body — the signature freshness
/// guard. Defined inline (NOT std's `DefaultHasher`) because the value is
/// PERSISTED: the algorithm must never drift across Rust or crate versions,
/// or every stored signature would read as stale at once. 8-byte
/// multiply-rotate chunks keep it roughly an order of magnitude cheaper than
/// a byte-wise FNV on kilobyte bodies — the scan verifies it for every
/// candidate row. Not a security boundary: it defends against
/// signature-unaware writers (a rolled-back pre-v10 binary, direct SQL), not
/// adversarial collisions — a length-only guard provably cannot catch a
/// same-length rewrite, which both AES-GCM re-encryption and ordinary
/// same-size edits produce.
pub fn body_hash64(s: &str) -> i64 {
    let bytes = s.as_bytes();
    // Seed mixes the length so zero-padded tail chunks are unambiguous.
    let mut h: u64 =
        0xcbf2_9ce4_8422_2325 ^ (bytes.len() as u64).wrapping_mul(0x0000_0100_0000_01b3);
    for chunk in bytes.chunks(8) {
        let mut buf = [0u8; 8];
        buf[..chunk.len()].copy_from_slice(chunk);
        h = (h ^ u64::from_le_bytes(buf)).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        h = h.rotate_left(23);
    }
    // Final avalanche so short bodies don't leave the high bits undermixed.
    h ^= h >> 31;
    h = h.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    (h ^ (h >> 29)) as i64
}

/// Everything the dedup scan needs about one stored row, computed from the
/// STORED `body_json` column value (ciphertext when encryption is on — see
/// the call sites in `db.rs` for why that is both the exactness-preserving
/// and the non-leaking choice).
pub struct RowSignature {
    /// Byte length of the stored body — the cheap half of the freshness
    /// guard: a signature whose recorded length disagrees with the fetched
    /// body is stale and must not be trusted.
    pub body_len: i64,
    /// `body_hash64` of the stored body — the exact half of the freshness
    /// guard, catching same-length rewrites by signature-unaware writers.
    pub body_hash: i64,
    /// Cardinality of the trigram set.
    pub tg_count: i64,
    /// `encode_sig` blob of the sorted packed set.
    pub sig: Vec<u8>,
    /// Bucket histogram, absent when any bucket saturates.
    pub histo: Option<Vec<u8>>,
}

/// Build the signature row for a stored body value.
pub fn build_row_signature(stored_body: &str) -> RowSignature {
    let set = packed_trigrams(stored_body);
    RowSignature {
        body_len: stored_body.len() as i64,
        body_hash: body_hash64(stored_body),
        tg_count: set.len() as i64,
        sig: encode_sig(&set),
        histo: histogram(&set),
    }
}

/// The near-duplicate verdict `exact_jaccard(target, candidate) >= threshold`
/// computed from the candidate's stored signature, with lossless early
/// abandon. Returns `None` when the blob is malformed or its element count
/// disagrees with `cand_count` — the caller must then fall back to rebuilding
/// the set from the body.
///
/// Early-abandon proof: at any point, the final intersection satisfies
/// `i_final <= i + min(remaining_target, remaining_candidate) = m`, and
/// `x / (a + b - x)` is monotonically increasing in `x`, so
/// `sim_final <= m / (a + b - m)`. Integer-to-f64 conversion and f64 division
/// are monotone, so if the f64 bound is `< threshold`, the f64 sim the exact
/// scan would compute is also `< threshold` — the verdict is provably false.
/// When the merge completes, the verdict is the SAME expression the exhaustive
/// scan evaluates: `(i as f64 / union as f64) >= threshold`.
pub fn jaccard_verdict_from_sig(
    target: &[u64],
    sig_blob: &[u8],
    cand_count: usize,
    threshold: f64,
) -> Option<bool> {
    let a = target.len();
    let b = cand_count;
    if a == 0 || b == 0 {
        // Matches exact_jaccard's empty-set guard: sim = 0.0.
        return Some(0.0 >= threshold);
    }
    let mut dec = SigDecoder::new(sig_blob);
    let mut i = 0usize; // intersection so far
    let mut ai = 0usize; // target elements consumed
    let mut seen_b = 0usize; // candidate elements consumed
    while seen_b < b {
        let v = dec.next_value().ok()?;
        seen_b += 1;
        while ai < a && target[ai] < v {
            ai += 1;
        }
        if ai < a && target[ai] == v {
            i += 1;
            ai += 1;
        }
        // Lossless early abandon (see doc comment).
        let m = i + (a - ai).min(b - seen_b);
        if ((m as f64) / ((a + b - m) as f64)) < threshold {
            return Some(false);
        }
    }
    if !dec.exhausted() {
        // Trailing bytes: blob disagrees with tg_count — treat as malformed.
        return None;
    }
    let union = a + b - i;
    Some((i as f64 / union as f64) >= threshold)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    /// Deterministic xorshift64* so tests need no rand dependency.
    struct XorShift(u64);
    impl XorShift {
        fn new(seed: u64) -> Self {
            XorShift(seed.max(1))
        }
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x.wrapping_mul(0x2545F4914F6CDD1D)
        }
        fn below(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }
    }

    /// Reference trigram set — verbatim the shape `db::Database::trigrams`
    /// builds, kept local so this module's math is testable in isolation.
    fn ref_trigrams(s: &str) -> HashSet<[char; 3]> {
        let chars: Vec<char> = s.chars().collect();
        if chars.len() < 3 {
            return HashSet::new();
        }
        chars.windows(3).map(|w| [w[0], w[1], w[2]]).collect()
    }

    /// Reference Jaccard — verbatim `db::Database::trigram_overlap`.
    fn ref_overlap(ta: &HashSet<[char; 3]>, tb: &HashSet<[char; 3]>) -> f64 {
        if ta.is_empty() || tb.is_empty() {
            return 0.0;
        }
        let intersection = ta.intersection(tb).count();
        let union = ta.len() + tb.len() - intersection;
        if union == 0 {
            return 0.0;
        }
        intersection as f64 / union as f64
    }

    fn random_body(rng: &mut XorShift, words: usize) -> String {
        const POOL: &[&str] = &[
            "alpha",
            "bravo",
            "charlie",
            "delta",
            "echo",
            "foxtrot",
            "golf",
            "hotel",
            "india",
            "juliet",
            "kilo",
            "lima",
            "mike",
            "november",
            "oscar",
            "papa",
            "café",
            "naïve",
            "日本語",
            "🌍",
            "Ω",
            "données",
            "vault",
            "perseus",
        ];
        let mut s = String::from("{\"note\":\"");
        for _ in 0..words {
            s.push_str(POOL[rng.below(POOL.len())]);
            s.push(' ');
        }
        s.push_str("\"}");
        s
    }

    #[test]
    fn packing_is_injective_and_cardinality_matches_reference() {
        let mut rng = XorShift::new(42);
        for _ in 0..200 {
            let s = {
                let w = 1 + rng.below(60);
                random_body(&mut rng, w)
            };
            let packed = packed_trigrams(&s);
            let reference = ref_trigrams(&s);
            assert_eq!(
                packed.len(),
                reference.len(),
                "packed cardinality must equal trigram-set cardinality for {s:?}"
            );
            // Strictly sorted (deduped).
            assert!(packed.windows(2).all(|w| w[0] < w[1]));
        }
        // Tiny / empty strings.
        for s in ["", "a", "ab", "é", "日本", "abc"] {
            assert_eq!(packed_trigrams(s).len(), ref_trigrams(s).len());
        }
    }

    #[test]
    fn exact_jaccard_is_bitwise_equal_to_hashset_reference() {
        let mut rng = XorShift::new(7);
        for _ in 0..300 {
            let a = {
                let w = 1 + rng.below(50);
                random_body(&mut rng, w)
            };
            let b = if rng.below(3) == 0 {
                a.clone() // identical
            } else {
                {
                    let w = 1 + rng.below(50);
                    random_body(&mut rng, w)
                }
            };
            let got = exact_jaccard(&packed_trigrams(&a), &packed_trigrams(&b));
            let want = ref_overlap(&ref_trigrams(&a), &ref_trigrams(&b));
            // Same integer counts, same single f64 division: exact equality.
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "jaccard mismatch for {a:?} vs {b:?}: {got} != {want}"
            );
        }
    }

    #[test]
    fn sig_roundtrip_and_verdict_match_exact_jaccard() {
        let mut rng = XorShift::new(1234);
        for round in 0..300 {
            let a = {
                let w = 1 + rng.below(50);
                random_body(&mut rng, w)
            };
            let b = match rng.below(4) {
                0 => a.clone(),
                1 => format!("{a} extra tail"),
                _ => {
                    let w = 1 + rng.below(50);
                    random_body(&mut rng, w)
                }
            };
            let ta = packed_trigrams(&a);
            let tb = packed_trigrams(&b);
            let blob = encode_sig(&tb);
            let sim = exact_jaccard(&ta, &tb);
            // Sweep thresholds including ones AT the computed sim, so the
            // >= boundary itself is exercised.
            for threshold in [0.0, 0.3, 0.7, 0.9, 1.0, sim] {
                let want = sim >= threshold;
                let got = jaccard_verdict_from_sig(&ta, &blob, tb.len(), threshold)
                    .expect("well-formed sig must decode");
                assert_eq!(
                    got, want,
                    "verdict mismatch round {round} threshold {threshold}: sim={sim}"
                );
            }
        }
    }

    #[test]
    fn malformed_sig_is_rejected_not_misjudged() {
        let set = packed_trigrams("{\"note\":\"hello world hello vault\"}");
        let blob = encode_sig(&set);
        // Wrong count (claims one more element than the blob holds).
        assert_eq!(
            jaccard_verdict_from_sig(&set, &blob, set.len() + 1, 0.0),
            None
        );
        // Trailing garbage.
        let mut long = blob.clone();
        long.push(0x01);
        assert_eq!(jaccard_verdict_from_sig(&set, &long, set.len(), 0.0), None);
        // Truncated varint (continuation bit set at end of buffer). threshold
        // 1.0 keeps the early-abandon from short-circuiting before the
        // malformed tail is reached.
        let mut trunc = blob.clone();
        trunc.pop();
        trunc.push(0x80);
        assert_eq!(jaccard_verdict_from_sig(&set, &trunc, set.len(), 1.0), None);
    }

    #[test]
    fn histogram_ceiling_never_undercounts_intersection() {
        let mut rng = XorShift::new(99);
        for _ in 0..200 {
            let a = {
                let w = 1 + rng.below(60);
                random_body(&mut rng, w)
            };
            let b = if rng.below(2) == 0 {
                format!("{a} shared suffix material")
            } else {
                {
                    let w = 1 + rng.below(60);
                    random_body(&mut rng, w)
                }
            };
            let ta = packed_trigrams(&a);
            let tb = packed_trigrams(&b);
            let (Some(ha), Some(hb)) = (histogram(&ta), histogram(&tb)) else {
                continue;
            };
            let ceiling = histo_intersection_ceiling(&ha, &hb);
            let true_i = {
                let sa: HashSet<u64> = ta.iter().copied().collect();
                tb.iter().filter(|v| sa.contains(v)).count()
            };
            assert!(
                ceiling >= true_i,
                "histogram ceiling {ceiling} undercounts true intersection {true_i}"
            );
        }
    }

    #[test]
    fn histogram_refuses_to_saturate() {
        // >255 distinct trigrams in one bucket requires a set large enough to
        // overflow a u8 bucket; synthesize one directly.
        let big: Vec<u64> = (0..200_000u64).map(|i| i * 7919).collect();
        let mut sorted = big.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(
            histogram(&sorted),
            None,
            "a bucket-overflowing set must yield no histogram, not a clamped one"
        );
    }

    #[test]
    fn body_hash64_is_stable_and_length_independent_of_content() {
        // The hash is PERSISTED (freshness guard): pin concrete values so an
        // accidental algorithm change fails loudly instead of silently
        // invalidating (or worse, silently trusting) every stored signature.
        assert_eq!(body_hash64(""), body_hash64(""));
        assert_eq!(body_hash64("perseus-vault"), body_hash64("perseus-vault"));
        assert_ne!(body_hash64(""), body_hash64("a"));
        // Same length, different content — the case a length-only guard
        // cannot distinguish (review defect on #392).
        let a = r#"{"note":"alpha bravo charlie delta echo foxtrot golf"}"#;
        let b = r#"{"memo":"zulu yankee xray whiskey victor uniform tan"}"#;
        assert_eq!(a.len(), b.len());
        assert_ne!(body_hash64(a), body_hash64(b));
        // Tail-chunk sensitivity: bodies differing only in the last byte.
        assert_ne!(body_hash64("12345678x"), body_hash64("12345678y"));
        // Zero-padding must not alias a shorter body onto a longer one.
        assert_ne!(body_hash64("abc"), body_hash64("abc\0\0"));

        // LITERAL value pins. Self-consistency alone (both sides calling the
        // same fn) would let a real algorithm change — e.g. rotate_left(23)
        // -> (24) — pass while silently invalidating every persisted
        // signature across binaries. These hardcoded constants are the ONLY
        // check that actually fails on such a drift: if you change the hash
        // algorithm you MUST recompute and update them (and accept that all
        // existing v10 signatures self-heal on next touch via the freshness
        // guard). Values captured from the current implementation.
        assert_eq!(
            body_hash64("perseus-vault"),
            -4349344705766122978,
            "body_hash64 algorithm changed — persisted signatures across binaries would silently mismatch"
        );
        assert_eq!(
            body_hash64(""),
            1530470515733238723,
            "body_hash64 empty-input value changed — see the pin comment above"
        );

        // The pin values must also be exactly what build_row_signature (and
        // therefore the scan's freshness guard) stores.
        assert_eq!(
            build_row_signature("perseus-vault").body_hash,
            -4349344705766122978
        );
    }

    #[test]
    fn build_row_signature_is_consistent() {
        let body = "{\"note\":\"the quick brown fox — 日本語テスト 🌍\"}";
        let rs = build_row_signature(body);
        let set = packed_trigrams(body);
        assert_eq!(rs.body_len, body.len() as i64);
        assert_eq!(rs.tg_count, set.len() as i64);
        assert_eq!(rs.sig, encode_sig(&set));
        // Verdict from the stored form equals the direct computation.
        let probe = packed_trigrams("{\"note\":\"the quick brown fox — 日本語テスト 🌍!\"}");
        let sim = exact_jaccard(&probe, &set);
        assert_eq!(
            jaccard_verdict_from_sig(&probe, &rs.sig, set.len(), 0.7),
            Some(sim >= 0.7)
        );
    }
}
