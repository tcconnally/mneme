// #1020: deterministic subword-HDC fingerprint tier (Hillock borrow).
//
// Hillock's `SubwordHDCEncoder` (v0.4.1, AGPL-3.0; algorithm mirrored, not
// ported) encodes any text into a 10,000-dim bipolar vector by superposing
// one random ±1 vector per character 3/4/5-gram of the padded, lowercased
// text and taking the sign. The GloVe-50d sign-random-projection half of
// Hillock's combined encoder is intentionally omitted: that path needs a
// 10MB external vocabulary file, while this tier exists for zero-API,
// deterministic, GPU-free environments — subword-only is the OOV-robust
// deterministic core.
//
// Differences from Hillock, by design:
//   - Per-n-gram bipolar vectors come from an FNV-1a-seeded splitmix64
//     stream instead of numpy MT19937. Byte-identical output with Hillock
//     is NOT a goal (and not possible with its non-seeded empty-text
//     random fallback); determinism within this binary IS: the same input
//     always produces the same bytes, on every platform, because the
//     bit source is a fixed 64-bit integer recurrence — no RNG state,
//     no HashMap iteration, no floating point.
//   - Empty/short text yields the all-+1 vector (sign(0) => +1, the same
//     tie rule as Hillock) instead of Hillock's unseeded random draw.
//   - Similarity is `1 - hamming/dim` in [0,1], the same metric the #885
//     bit-quantized dense arm already uses (`vector_quant::bit_similarity`),
//     so fingerprint scores are rank-comparable with dense scores. An
//     unrelated text pair lands at ~0.5 (the noise floor), not 0.
//
// Storage: 10,000 bits = 1,250 bytes per entity (entities.fingerprint),
// written on content change when the fingerprint tier is enabled
// (PERSEUS_VAULT_EMBEDDING_FINGERPRINT / --embedding-fingerprint). The
// tier is a lexical-adjacent FALLBACK ranker: never primary while dense
// embeddings exist, engaged when the embedding backend is unavailable.
//
// Unicode note: n-grams walk `char` boundaries (like Python's str
// indexing) but the hash input is the UTF-8 encoding, and case folding is
// `str::to_lowercase`. Identical input is byte-identical within one
// binary build; Unicode table versions may differ across Rust releases.

/// Dimension of the bipolar HDC space (Hillock's HDC_DIMENSION).
pub const FINGERPRINT_DIM: usize = 10_000;
/// Packed size of one fingerprint: one sign bit per dimension.
pub const FINGERPRINT_BYTES: usize = FINGERPRINT_DIM / 8; // 1,250

/// FNV-1a 64-bit over the n-gram's UTF-8 bytes — the per-vector seed.
/// Deterministic across platforms and std versions by construction.
fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// splitmix64 — a fixed 64-bit integer recurrence used as the ±1 bit
/// source for one n-gram's vector. Every dimension's bit is a pure
/// function of (ngram, dim), so the whole fingerprint is deterministic.
struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        SplitMix64(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// Encode `text` into its 10,000-bit subword-HDC fingerprint.
///
/// Algorithm (mirrors Hillock `SubwordHDCEncoder.encode`):
///   pad -> `#<lowercased trimmed text>#`, then for n in {3,4,5}, slide a
///   window of n chars and add that n-gram's random ±1 vector into an
///   i32 accumulator. The final bit is set iff the accumulated count is
///   >= 0 (Hillock's sign rule with zeros resolving to +1).
///
/// Cost: ~3·len(char) n-grams × 157 splitmix draws + 10k adds. A 1KB body
/// is single-digit milliseconds in release builds. Deterministic: no
/// environment, thread, or platform dependence.
pub fn fingerprint_bytes(text: &str) -> Vec<u8> {
    let mut padded: Vec<char> = Vec::with_capacity(text.len() + 2);
    padded.push('#');
    padded.extend(text.trim().to_lowercase().chars());
    padded.push('#');

    let mut counts = vec![0i32; FINGERPRINT_DIM];
    let mut ngram = String::new();
    for n in 3..=5usize {
        if padded.len() < n {
            continue;
        }
        for i in 0..=(padded.len() - n) {
            ngram.clear();
            ngram.extend(padded[i..i + n].iter().copied());
            let mut rng = SplitMix64::new(fnv1a64(ngram.as_bytes()));
            for chunk in counts.chunks_mut(64) {
                let mut word = rng.next_u64();
                for slot in chunk.iter_mut() {
                    if word & 1 == 1 {
                        *slot += 1;
                    } else {
                        *slot -= 1;
                    }
                    word >>= 1;
                }
            }
        }
    }

    let mut out = vec![0u8; FINGERPRINT_BYTES];
    for (i, &count) in counts.iter().enumerate() {
        if count >= 0 {
            out[i / 8] |= 1u8 << (i % 8);
        }
    }
    out
}

/// Popcount (Hamming) distance between two packed fingerprints.
/// Lengths must match; a mismatch counts as 0 similarity upstream.
pub fn hamming_distance(a: &[u8], b: &[u8]) -> u32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x ^ y).count_ones())
        .sum()
}

/// Hamming similarity in [0,1]: `1 - hamming/dim`. Identical vectors score
/// 1.0, unrelated vectors score ~0.5 (the bipolar noise floor), opposites
/// 0.0. A length mismatch scores 0.0 — it can never win a rank slot.
pub fn fingerprint_similarity(a: &[u8], b: &[u8]) -> f64 {
    if a.is_empty() || a.len() != b.len() {
        return 0.0;
    }
    let dim = (a.len() * 8) as f64;
    1.0 - f64::from(hamming_distance(a, b)) / dim
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Acceptance (#1020): same input -> identical vector, byte for byte.
    #[test]
    fn deterministic_same_input_identical_bytes() {
        let a = fingerprint_bytes("The build fails on Windows CI after the rustfmt update");
        let b = fingerprint_bytes("The build fails on Windows CI after the rustfmt update");
        assert_eq!(a, b);
        // And stable across many repeated encodes (no hidden state).
        for _ in 0..20 {
            assert_eq!(
                a,
                fingerprint_bytes("The build fails on Windows CI after the rustfmt update")
            );
        }
    }

    /// Acceptance (#1020): storage-cost assertion — exactly 10,000 bits.
    #[test]
    fn storage_cost_is_exactly_10k_packed_bits() {
        assert_eq!(FINGERPRINT_DIM, 10_000);
        assert_eq!(FINGERPRINT_BYTES, 1_250);
        for text in [
            "a",
            "hello world",
            "much longer body text with many tokens and symbols *&^%$#@",
        ] {
            assert_eq!(fingerprint_bytes(text).len(), FINGERPRINT_BYTES);
        }
    }

    /// A fingerprint is not a degenerate constant: distinct texts produce
    /// distinct bit patterns, and self-similarity is exactly 1.0.
    #[test]
    fn distinct_texts_distinct_patterns_self_similarity_one() {
        let a = fingerprint_bytes("alpha memory tier");
        let b = fingerprint_bytes("beta evidence ledger");
        assert_ne!(a, b);
        assert_eq!(fingerprint_similarity(&a, &a), 1.0);
        assert_eq!(fingerprint_similarity(&b, &b), 1.0);
    }

    /// Acceptance (#1020): binding-orthogonality analog — unrelated texts
    /// land at the bipolar noise floor (~0.5 similarity, i.e. dot ~ 0).
    /// With 10k independent bits the sampling spread is ~±0.005.
    #[test]
    fn unrelated_texts_sit_at_the_noise_floor() {
        let pairs = [
            (
                "quantum chromodynamics renormalization",
                "the ssh agent socket path on unraid",
            ),
            (
                "perl one-liner for csv quoting",
                "biosketch other support attachment",
            ),
            (
                "zebra migrations in the serengeti",
                "rust borrow checker lifetime elision",
            ),
        ];
        for (x, y) in pairs {
            let sim = fingerprint_similarity(&fingerprint_bytes(x), &fingerprint_bytes(y));
            assert!(
                (sim - 0.5).abs() < 0.02,
                "{x:?} vs {y:?} scored {sim}, expected ~0.5"
            );
        }
    }

    /// Acceptance (#1020): near-miss spelling robustness — a separator
    /// variant stays far above the noise floor, and stays well above an
    /// unrelated comparison.
    #[test]
    fn near_miss_spelling_stays_above_noise_and_above_unrelated() {
        let a = fingerprint_bytes("Alan_Turing");
        let b = fingerprint_bytes("Alan Turing");
        let c = fingerprint_bytes("Grace Hopper");
        let near_miss = fingerprint_similarity(&a, &b);
        let unrelated = fingerprint_similarity(&a, &c);
        assert!(
            near_miss > 0.55,
            "Alan_Turing vs Alan Turing scored {near_miss}, expected well above 0.5"
        );
        assert!(
            near_miss > unrelated + 0.03,
            "near-miss {near_miss} not clearly above unrelated {unrelated}"
        );
    }

    /// Case folding is part of the encoder: identical modulo case and
    /// surrounding whitespace produces identical bytes.
    #[test]
    fn case_and_whitespace_invariant() {
        assert_eq!(
            fingerprint_bytes("Hello World"),
            fingerprint_bytes("  hello world  ")
        );
        // Pure case difference only — separators are content, not noise
        // (underscore vs space is a near-miss, covered separately).
        assert_eq!(
            fingerprint_bytes("ALAN TURING"),
            fingerprint_bytes("alan turing")
        );
    }

    /// Empty and single-char inputs are defined and deterministic (the
    /// all-+1 vector for empty — Hillock's sign rule with zero counts).
    #[test]
    fn empty_and_short_inputs_are_defined_and_deterministic() {
        let empty = fingerprint_bytes("");
        assert_eq!(empty.len(), FINGERPRINT_BYTES);
        assert_eq!(empty, vec![0xFFu8; FINGERPRINT_BYTES]); // all counts 0 -> all +1
        let short = fingerprint_bytes("x");
        assert_eq!(short, fingerprint_bytes("x"));
        assert_ne!(short, empty);
    }

    /// Non-ASCII input is deterministic and walks char boundaries.
    #[test]
    fn unicode_input_is_deterministic() {
        let a = fingerprint_bytes("Überarbeitung der Spezifikation");
        let b = fingerprint_bytes("Überarbeitung der Spezifikation");
        assert_eq!(a, b);
        let cjk = fingerprint_bytes("记忆层设计规范");
        assert_eq!(cjk, fingerprint_bytes("记忆层设计规范"));
        assert_ne!(a, cjk);
    }

    /// Length mismatch scores 0 (fail-closed: a foreign-format blob can
    /// never win a rank slot).
    #[test]
    fn similarity_length_mismatch_is_zero() {
        let a = fingerprint_bytes("anything");
        assert_eq!(fingerprint_similarity(&a, &a[..a.len() - 1]), 0.0);
        assert_eq!(fingerprint_similarity(&[], &[]), 0.0);
    }

    /// Popcount comparison probe for the spec numbers (run explicitly:
    /// `cargo test --no-default-features popcount_comparison_probe -- --ignored`).
    #[test]
    #[ignore = "comparison throughput probe — run explicitly for docs/specs/fingerprint-tier.md"]
    fn popcount_comparison_probe() {
        let texts: Vec<String> = (0..100)
            .map(|i| {
                format!(
                    "entity body number {i} about memory tooling, corrections, and retrieval quality"
                )
            })
            .collect();
        let fps: Vec<Vec<u8>> = texts.iter().map(|t| fingerprint_bytes(t)).collect();
        // Spec numbers: the acceptance-framing similarities.
        let near_miss = fingerprint_similarity(
            &fingerprint_bytes("Alan_Turing"),
            &fingerprint_bytes("Alan Turing"),
        );
        let unrelated = fingerprint_similarity(
            &fingerprint_bytes("Alan Turing"),
            &fingerprint_bytes("Grace Hopper"),
        );
        let unrelated2 = fingerprint_similarity(
            &fingerprint_bytes("quantum chromodynamics renormalization"),
            &fingerprint_bytes("the ssh agent socket path on unraid"),
        );
        eprintln!("probe sim: near_miss(Alan_Turing|Alan Turing) = {near_miss:.4}");
        eprintln!("probe sim: unrelated(Alan Turing|Grace Hopper) = {unrelated:.4}");
        eprintln!("probe sim: unrelated(long strings) = {unrelated2:.4}");
        // Encode throughput (write-path cost when the tier is enabled).
        let t_enc = std::time::Instant::now();
        let encoded = texts.len();
        for t in &texts {
            let _ = fingerprint_bytes(t);
        }
        let enc_elapsed = t_enc.elapsed();
        eprintln!(
            "probe: {encoded} encodes in {enc_elapsed:?} ({:.1} µs/encode)",
            enc_elapsed.as_micros() as f64 / encoded as f64,
        );
        let t0 = std::time::Instant::now();
        let mut comparisons: u64 = 0;
        let mut acc: u64 = 0;
        for (i, a) in fps.iter().enumerate() {
            for b in fps.iter().skip(i + 1) {
                acc += u64::from(hamming_distance(a, b));
                comparisons += 1;
            }
        }
        let elapsed = t0.elapsed();
        eprintln!(
            "probe: {comparisons} hamming comparisons in {elapsed:?} ({:.1} ns/comparison, {:.2} M byte-pairs/s)",
            elapsed.as_nanos() as f64 / comparisons as f64,
            (comparisons as f64 * FINGERPRINT_BYTES as f64) / elapsed.as_secs_f64() / 1e6,
        );
        assert!(acc > 0);
    }
}
