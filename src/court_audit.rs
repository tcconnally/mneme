//! #940: court-of-record — deterministic audit recommendations.
//!
//! Pure helpers for the consistency self-audit: pair fingerprints and the
//! winner-recommendation ladder (importance → source authority → recency →
//! id). No IO; the handlers in tools.rs own the read-only audit and the
//! idempotent ruling write.

use sha2::{Digest, Sha256};

/// Candidate entity attributes used by the ladder.
#[derive(Debug, Clone)]
pub struct Candidate {
    pub id: String,
    pub category: String,
    pub key: String,
    pub importance: f64,
    pub source: String,
    pub created_at_unix_ms: i64,
    /// #960: source-grounded encoding strength (1..=5, S1 weakest). A
    /// higher tier dominates deterministically BEFORE any other rung —
    /// ground-truth code (S5) can never be overruled by inference (S1-S2).
    pub encoding_strength: u8,
}

/// Authority rank for a source label (higher = more authoritative).
/// `curated` 4 > `capture` 3 > `agent` 2 > `web_gap_fill` 1 > default 1.
pub fn source_authority_rank(source: &str) -> i64 {
    match source {
        "curated" => 4,
        "capture" => 3,
        "agent" => 2,
        "web_gap_fill" => 1,
        _ => 1,
    }
}

/// Pair fingerprint: sha256 over the two entity ids joined by '|', sorted.
/// Order-independent so the same pair always maps to one ruling slot.
pub fn pair_fingerprint(id_a: &str, id_b: &str) -> String {
    let (first, second) = if id_a <= id_b {
        (id_a, id_b)
    } else {
        (id_b, id_a)
    };
    let digest = Sha256::digest(format!("{first}|{second}").as_bytes());
    format!("{digest:x}")
}

/// A winner recommendation for one contradiction pair.
#[derive(Debug, Clone)]
pub struct Recommendation {
    pub winner: Candidate,
    pub loser: Candidate,
    /// Which rung decided: "encoding_strength" | "importance" | "authority"
    /// | "recency" | "id".
    pub decided_by: &'static str,
}

/// Recommend a winner: encoding strength desc (S5 > S1, hard rule), then
/// importance desc, then source-authority desc, then recency desc, then id
/// asc (deterministic tiebreak). Cross-tier pairs NEVER reach consolidation
/// or operator review — only same-tier conflicts fall through to the other
/// rungs.
pub fn recommend(a: &Candidate, b: &Candidate) -> Recommendation {
    let (winner, loser, decided_by) = if a.encoding_strength != b.encoding_strength {
        if a.encoding_strength > b.encoding_strength {
            (a, b, "encoding_strength")
        } else {
            (b, a, "encoding_strength")
        }
    } else if a.importance > b.importance {
        (a, b, "importance")
    } else if b.importance > a.importance {
        (b, a, "importance")
    } else {
        let ra = source_authority_rank(&a.source);
        let rb = source_authority_rank(&b.source);
        if ra > rb {
            (a, b, "authority")
        } else if rb > ra {
            (b, a, "authority")
        } else if a.created_at_unix_ms > b.created_at_unix_ms {
            (a, b, "recency")
        } else if b.created_at_unix_ms > a.created_at_unix_ms {
            (b, a, "recency")
        } else if a.id < b.id {
            (a, b, "id")
        } else {
            (b, a, "id")
        }
    };
    Recommendation {
        winner: winner.clone(),
        loser: loser.clone(),
        decided_by,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cand(id: &str, importance: f64, source: &str, created: i64) -> Candidate {
        Candidate {
            id: id.to_string(),
            category: "facts".to_string(),
            key: format!("key-{id}"),
            importance,
            source: source.to_string(),
            created_at_unix_ms: created,
            encoding_strength: 1,
        }
    }

    #[test]
    fn encoding_strength_dominates_every_other_rung() {
        // Code-grounded (S5) beats single inference (S1) even when the
        // inference has higher importance, newer recency, and higher source
        // authority — the hard rule, not a tiebreak.
        let mut code = cand("code", 0.3, "agent", 100);
        code.encoding_strength = 5;
        let mut chat = cand("chat", 0.9, "curated", 999);
        chat.encoding_strength = 1;
        let r = recommend(&chat, &code);
        assert_eq!(r.winner.id, "code");
        assert_eq!(r.decided_by, "encoding_strength");
        // Order of arguments does not matter.
        let r2 = recommend(&code, &chat);
        assert_eq!(r2.winner.id, "code");
        assert_eq!(r2.decided_by, "encoding_strength");
        // Corroborated (S2) also loses to code (S5).
        let mut corr = cand("corr", 0.9, "curated", 999);
        corr.encoding_strength = 2;
        let r3 = recommend(&corr, &code);
        assert_eq!(r3.winner.id, "code");
    }

    #[test]
    fn same_tier_falls_through_to_importance() {
        // Equal encoding strength: the old ladder applies unchanged.
        let mut a = cand("a", 0.8, "agent", 100);
        a.encoding_strength = 3;
        let mut b = cand("b", 0.5, "agent", 200);
        b.encoding_strength = 3;
        let r = recommend(&a, &b);
        assert_eq!(r.winner.id, "a");
        assert_eq!(r.decided_by, "importance");
    }

    #[test]
    fn fingerprint_is_order_independent_and_pair_specific() {
        let f1 = pair_fingerprint("mem-a", "mem-b");
        assert_eq!(f1, pair_fingerprint("mem-b", "mem-a"));
        assert_ne!(f1, pair_fingerprint("mem-a", "mem-c"));
        assert_eq!(f1.len(), 64);
    }

    #[test]
    fn importance_decides_first() {
        let r = recommend(&cand("a", 0.8, "agent", 100), &cand("b", 0.5, "agent", 200));
        assert_eq!(r.winner.id, "a");
        assert_eq!(r.loser.id, "b");
        assert_eq!(r.decided_by, "importance");
    }

    #[test]
    fn authority_decides_importance_ties() {
        // equal importance: curated beats agent regardless of recency
        let r = recommend(
            &cand("old-curated", 0.5, "curated", 100),
            &cand("new-agent", 0.5, "agent", 999),
        );
        assert_eq!(r.winner.id, "old-curated");
        assert_eq!(r.decided_by, "authority");
    }

    #[test]
    fn recency_decides_importance_and_authority_ties() {
        let r = recommend(
            &cand("old", 0.5, "agent", 100),
            &cand("new", 0.5, "agent", 200),
        );
        assert_eq!(r.winner.id, "new");
        assert_eq!(r.decided_by, "recency");
    }

    #[test]
    fn id_breaks_full_ties_deterministically() {
        let r = recommend(
            &cand("mem-b", 0.5, "agent", 100),
            &cand("mem-a", 0.5, "agent", 100),
        );
        assert_eq!(r.winner.id, "mem-a");
        assert_eq!(r.decided_by, "id");
        // and the reverse ordering still picks the same winner
        let r2 = recommend(
            &cand("mem-a", 0.5, "agent", 100),
            &cand("mem-b", 0.5, "agent", 100),
        );
        assert_eq!(r2.winner.id, "mem-a");
    }

    #[test]
    fn authority_rank_orders_sources() {
        assert!(source_authority_rank("curated") > source_authority_rank("capture"));
        assert!(source_authority_rank("capture") > source_authority_rank("agent"));
        assert!(source_authority_rank("agent") > source_authority_rank("web_gap_fill"));
        assert_eq!(
            source_authority_rank("unknown"),
            source_authority_rank("web_gap_fill")
        );
    }
}
