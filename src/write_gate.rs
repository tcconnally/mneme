//! #939: zero-token write gate — deterministic keep/supersede/forget BEFORE
//! LLM enrichment.
//!
//! Every Vault write currently costs an LLM pass in the agent pipeline (local
//! Ollama qwen3.5:9b on Greg); most turns say nothing new. This gate is the
//! cheap deterministic precheck the provider/agent flow can call BEFORE
//! enrichment: content-hash dedup, key supersession, and a stored-signature
//! near-duplicate scan (the "dumb signal") decide Store / Duplicate /
//! Supersede / Forget with ZERO LLM tokens. Only `Adjudicate` — a genuine
//! near-duplicate that may be a contradiction — escalates to the LLM (or
//! operator review). Read-only: the gate never mutates.
//!
//! Design contract: every verdict is deterministic over the store state;
//! `Forget` is deliberately conservative (only vague/empty notes) so the gate
//! can never drop a substantive fact.

use crate::db::Database;

/// The zero-token verdict for one candidate write.
#[derive(Debug, Clone, PartialEq)]
pub enum GateVerdict {
    /// Nothing triggered: store without an LLM pass.
    Store { note: &'static str },
    /// Exact/near-exact content already present in (category, workspace).
    Duplicate { matched_id: String },
    /// Candidate is too vague to justify a write (importance floor).
    Forget { reason: String },
    /// Same (category, key) already exists — a deterministic supersede.
    Supersede { target_id: String },
    /// Near-duplicate with divergent content — the ONLY LLM-eligible path.
    Adjudicate { matched_id: String, reason: String },
}

/// Near-exact match threshold: same content, maybe whitespace/format drift.
const DUP_THRESHOLD: f64 = 0.97;
/// Near-duplicate threshold: same subject area but divergent wording —
/// a contradiction candidate, not a silent dedup. Jaccard over character
/// trigrams is strict on longer bodies; 0.55 separates "same topic, maybe
/// same claim" from "unrelated".
const ADJUDICATE_THRESHOLD: f64 = 0.55;
/// Importance floor: a body shorter than this with no concrete markers is a
/// vague note ("ok", "noted") — forget rather than store.
const MIN_SUBSTANTIVE_CHARS: usize = 24;

impl GateVerdict {
    /// Stable machine name for the verdict.
    pub fn name(&self) -> &'static str {
        match self {
            GateVerdict::Store { .. } => "store",
            GateVerdict::Duplicate { .. } => "duplicate",
            GateVerdict::Forget { .. } => "forget",
            GateVerdict::Supersede { .. } => "supersede",
            GateVerdict::Adjudicate { .. } => "adjudicate",
        }
    }

    /// True when the caller may skip the LLM enrichment pass entirely.
    pub fn needs_llm(&self) -> bool {
        matches!(self, GateVerdict::Adjudicate { .. })
    }
}

/// Run the deterministic write gate for one candidate (category, key, body).
/// Never mutates the store.
pub fn run_gate(
    db: &Database,
    category: &str,
    key: &str,
    body: &str,
    workspace_hash: Option<&str>,
) -> Result<GateVerdict, String> {
    let ws = workspace_hash.unwrap_or("");
    let trimmed = body.trim();

    // 1. Exact / near-exact content dedup (stored-signature scan, no full
    //    store pass): same content already admitted -> Duplicate.
    if let Some(id) = db
        .find_near_duplicate(category, ws, body, DUP_THRESHOLD)
        .map_err(|e| format!("duplicate scan failed: {e}"))?
    {
        return Ok(GateVerdict::Duplicate { matched_id: id });
    }

    // 2. Same (category, key) exists -> deterministic supersede target.
    if let Some(existing) = db
        .get_entity(category, key)
        .map_err(|e| format!("entity lookup failed: {e}"))?
    {
        if existing.workspace_hash == ws {
            return Ok(GateVerdict::Supersede {
                target_id: existing.id,
            });
        }
    }

    // 3. Near-duplicate with divergent content: contradiction candidate —
    //    the only verdict that may consume LLM tokens (or operator review).
    if let Some(id) = db
        .find_near_duplicate(category, ws, body, ADJUDICATE_THRESHOLD)
        .map_err(|e| format!("contradiction scan failed: {e}"))?
    {
        return Ok(GateVerdict::Adjudicate {
            matched_id: id,
            reason: "near_duplicate_divergent_content".to_string(),
        });
    }

    // 4. Importance floor: vague/empty notes are forgotten, never stored.
    if trimmed.chars().count() < MIN_SUBSTANTIVE_CHARS && !has_concrete_markers(trimmed) {
        return Ok(GateVerdict::Forget {
            reason: format!(
                "below_importance_floor: {} chars, no concrete markers",
                trimmed.chars().count()
            ),
        });
    }

    // 5. Nothing triggered.
    Ok(GateVerdict::Store {
        note: "zero-token gate cleared; no duplicate, supersede, or conflict",
    })
}

/// A body with digits, URLs, paths, or identifiers is concrete even when
/// short ("v1.2.3", "src/main.rs", "port 8080").
fn has_concrete_markers(body: &str) -> bool {
    body.contains('/')
        || body.chars().any(|c| c.is_ascii_digit())
        || body.contains('.')
        || body.split_whitespace().any(|w| w.len() >= 12)
}
