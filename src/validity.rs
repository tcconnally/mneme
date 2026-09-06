//! #860: validity-aware recall — deterministic scoring of whether a memory
//! is VALID for the current task context, not merely semantically similar.
//!
//! Recent memory-system research (contextual reinstatement, validity-aware
//! retrieval) argues recall should optimize for "is this memory appropriate
//! and still valid here", not just "is it related". This module scores five
//! validity signals per candidate — temporal fit (freshness decay), entity
//! scope match, supersession state, provenance class, and expiry proximity —
//! into a single multiplier and an explicit grade:
//!
//! - `valid`           — all signals nominal; safe to rely on.
//! - `stale`           — aged or expiring soon; usable with caution.
//! - `context_invalid` — superseded, expired, or so old that the memory is
//!                       likely a distractor for the current context.
//!
//! Everything here is a pure function of (entity fields, now): no LLM, no
//! network, no hidden state — the same DB yields the same grades, so recall
//! stays byte-deterministic (#247).

use serde::Serialize;

/// Default validity weights. Half-life = 30 days: a memory 30 days old keeps
/// half its freshness, 90 days (3 half-lives) drops below the
/// context-invalid floor.
pub const DEFAULT_FRESHNESS_HALF_LIFE_SECS: f64 = 2_592_000.0;

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ValidityWeights {
    /// Freshness half-life in seconds (0.5^(age/half_life)).
    pub freshness_half_life_secs: f64,
    /// Multiplicative bonus for exact workspace match.
    pub scope_bonus: f64,
    /// Multiplicative boost for established-fact provenance
    /// (verified/corroborated).
    pub provenance_boost: f64,
    /// Multiplicative penalty for superseded (deprecated) memories.
    pub superseded_penalty: f64,
    /// Multiplicative penalty for memories expiring within one half-life.
    pub expiring_penalty: f64,
    /// Freshness below this grades the memory `stale`.
    pub stale_freshness: f64,
    /// Freshness below this grades the memory `context_invalid`.
    pub context_invalid_freshness: f64,
}

impl Default for ValidityWeights {
    fn default() -> Self {
        Self {
            freshness_half_life_secs: DEFAULT_FRESHNESS_HALF_LIFE_SECS,
            scope_bonus: 0.10,
            provenance_boost: 0.15,
            superseded_penalty: 0.35,
            expiring_penalty: 0.70,
            stale_freshness: 0.50,
            context_invalid_freshness: 0.125,
        }
    }
}

/// Per-candidate validity annotation, attached to recall items and used to
/// re-rank the fused pool under the `validity` profile.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ValidityInfo {
    /// "valid" | "stale" | "context_invalid".
    pub grade: String,
    /// 0.5^(age/half_life), 1.0 = just written, -> 0.0 as it ages.
    pub freshness: f64,
    /// "exact" (entity ws == query ws) | "global" (entity ws empty) | "none".
    pub scope_match: String,
    /// The entity's epistemic trust class (verified/corroborated/candidate/
    /// rejected/defensively_recalled/unknown).
    pub provenance_class: String,
    /// True when the entity status is `deprecated` (supersession contract).
    pub superseded: bool,
    /// True when expiry is within one freshness half-life of now.
    pub expiring_soon: bool,
    /// True when the entity has passed its declared expiry.
    pub expired: bool,
    pub expires_at_unix_ms: Option<i64>,
    /// Combined validity multiplier applied to the base relevance score.
    /// > 1.0 boosts, < 1.0 penalizes; always > 0 (never hard-excludes).
    pub multiplier: f64,
    /// Human-readable signal list for trace-back ("freshness:0.031",
    /// "scope:exact", "provenance:verified", "superseded:deprecated").
    pub signals: Vec<String>,
}

/// Score one candidate's validity. All inputs are plain entity fields.
pub fn score(
    now_unix_ms: i64,
    created_at_unix_ms: i64,
    expires_at_unix_ms: Option<i64>,
    workspace_hash: &str,
    query_workspace: Option<&str>,
    epistemic_state: &str,
    status: &str,
    weights: &ValidityWeights,
) -> ValidityInfo {
    let age_secs = ((now_unix_ms - created_at_unix_ms).max(0) as f64) / 1000.0;
    let freshness = if weights.freshness_half_life_secs > 0.0 {
        0.5f64.powf(age_secs / weights.freshness_half_life_secs)
    } else {
        1.0
    };

    let scope_match = match query_workspace {
        Some(ws) if !ws.is_empty() => {
            if workspace_hash == ws {
                "exact"
            } else if workspace_hash.is_empty() {
                "global"
            } else {
                "none"
            }
        }
        _ => "none",
    };

    let provenance_class = if epistemic_state.is_empty() {
        "unknown"
    } else {
        epistemic_state
    };

    let superseded = status == "deprecated";
    let expired = expires_at_unix_ms.map(|e| now_unix_ms > e).unwrap_or(false);
    let expiring_soon = !expired
        && expires_at_unix_ms
            .map(|e| {
                let remaining_ms = e - now_unix_ms;
                remaining_ms > 0 && remaining_ms as f64 <= weights.freshness_half_life_secs * 1000.0
            })
            .unwrap_or(false);

    let mut multiplier = freshness;
    let mut signals = vec![format!("freshness:{freshness:.3}")];

    if scope_match == "exact" {
        multiplier *= 1.0 + weights.scope_bonus;
        signals.push("scope:exact".to_string());
    } else if scope_match == "global" {
        signals.push("scope:global".to_string());
    } else {
        signals.push("scope:none".to_string());
    }

    if matches!(provenance_class, "verified" | "corroborated") {
        multiplier *= 1.0 + weights.provenance_boost;
        signals.push(format!("provenance:{provenance_class}"));
    }

    if superseded {
        multiplier *= weights.superseded_penalty;
        signals.push("superseded:deprecated".to_string());
    }
    if expired {
        multiplier *= weights.superseded_penalty;
        signals.push("expired".to_string());
    } else if expiring_soon {
        multiplier *= weights.expiring_penalty;
        signals.push("expiring_soon".to_string());
    }

    let grade = if expired || superseded || freshness < weights.context_invalid_freshness {
        "context_invalid"
    } else if expiring_soon || freshness < weights.stale_freshness {
        "stale"
    } else {
        "valid"
    };

    ValidityInfo {
        grade: grade.to_string(),
        freshness,
        scope_match: scope_match.to_string(),
        provenance_class: provenance_class.to_string(),
        superseded,
        expiring_soon,
        expired,
        expires_at_unix_ms,
        multiplier,
        signals,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const NOW: i64 = 1_750_000_000_000; // fixed instant; all ages derived from it
    const DAY_MS: i64 = 86_400_000;

    fn base() -> ValidityWeights {
        ValidityWeights::default()
    }

    #[test]
    fn fresh_exact_verified_is_valid_and_boosted() {
        let info = score(
            NOW,
            NOW - 60_000, // one minute old
            None,
            "ws-a",
            Some("ws-a"),
            "verified",
            "active",
            &base(),
        );
        assert_eq!(info.grade, "valid");
        assert!((info.freshness - 1.0).abs() < 0.01);
        assert_eq!(info.scope_match, "exact");
        assert!(
            info.multiplier > 1.0,
            "exact-scope + verified boost multiplier"
        );
        assert!(info.signals.iter().any(|s| s == "scope:exact"));
        assert!(info.signals.iter().any(|s| s == "provenance:verified"));
    }

    #[test]
    fn one_half_life_halves_freshness_and_marks_stale_under_floor() {
        // 30 days old: freshness ≈ 0.5 -> stale (below 0.5? boundary: equal).
        let info = score(
            NOW,
            NOW - 30 * DAY_MS,
            None,
            "",
            Some("ws-a"),
            "candidate",
            "active",
            &base(),
        );
        assert!((info.freshness - 0.5).abs() < 1e-9);
        assert_eq!(info.scope_match, "global");
        // freshness 0.5 is NOT < 0.5, so valid; 0.5^1 = 0.5 exact boundary.
        assert_eq!(info.grade, "valid");

        // 61 days old: freshness ≈ 0.5^(61/30) ≈ 0.244 < 0.5 -> stale.
        let info = score(
            NOW,
            NOW - 61 * DAY_MS,
            None,
            "ws-a",
            Some("ws-b"), // mismatched scope
            "candidate",
            "active",
            &base(),
        );
        assert_eq!(info.grade, "stale");
        assert_eq!(info.scope_match, "none");
        assert!(info.multiplier < 0.5);
    }

    #[test]
    fn superseded_and_expired_are_context_invalid() {
        let info = score(
            NOW,
            NOW - 60_000,
            None,
            "ws-a",
            Some("ws-a"),
            "verified",
            "deprecated",
            &base(),
        );
        assert_eq!(info.grade, "context_invalid");
        assert!(info.superseded);
        assert!(info.signals.iter().any(|s| s == "superseded:deprecated"));

        let info = score(
            NOW,
            NOW - 60_000,
            Some(NOW - 1000), // expired one second ago
            "ws-a",
            Some("ws-a"),
            "verified",
            "active",
            &base(),
        );
        assert_eq!(info.grade, "context_invalid");
        assert!(info.expired);
    }

    #[test]
    fn expiring_soon_grades_stale_with_penalty() {
        let info = score(
            NOW,
            NOW - 60_000,
            Some(NOW + 60_000), // expires in one minute
            "ws-a",
            Some("ws-a"),
            "corroborated",
            "active",
            &base(),
        );
        assert_eq!(info.grade, "stale");
        assert!(info.expiring_soon);
        // freshness for a 1-minute-old memory is ~0.99998 (not exactly 1),
        // so tolerance-based: penalty * scope * provenance, near-exact.
        let expected =
            base().expiring_penalty * (1.0 + base().scope_bonus) * (1.0 + base().provenance_boost);
        assert!(
            (info.multiplier - expected).abs() < 0.01,
            "multiplier {} vs expected {expected}",
            info.multiplier
        );
    }

    #[test]
    fn very_old_memory_is_context_invalid_even_when_active() {
        // 200 days = 6.7 half-lives -> freshness ≈ 0.5^6.7 ≈ 0.0096 < 0.125.
        let info = score(
            NOW,
            NOW - 200 * DAY_MS,
            None,
            "ws-a",
            Some("ws-a"),
            "candidate",
            "active",
            &base(),
        );
        assert_eq!(info.grade, "context_invalid");
        assert!(info.freshness < base().context_invalid_freshness);
    }

    #[test]
    fn multiplier_is_never_zero_and_composes_deterministically() {
        // Worst case: expired + superseded + ancient.
        let info = score(
            NOW,
            NOW - 400 * DAY_MS,
            Some(NOW - 1),
            "ws-a",
            Some("ws-b"),
            "candidate",
            "deprecated",
            &base(),
        );
        assert!(info.multiplier > 0.0);
        assert_eq!(info.grade, "context_invalid");
        // Best case multiplier: fresh + exact + verified.
        let best = score(
            NOW,
            NOW - 1,
            None,
            "ws-a",
            Some("ws-a"),
            "verified",
            "active",
            &base(),
        );
        assert!(best.multiplier > 1.0);
        assert!(best.multiplier > info.multiplier);
    }
}
