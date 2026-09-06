//! #874: Activation-gated sparse writes — interference-aware memory updates.
//!
//! Grounding: "Continual Learning via Sparse Memory Finetuning" (Lin et al.,
//! Meta FAIR + UC Berkeley, arXiv:2510.15103, Oct 2025). Updating only the
//! memory slots highly activated by new knowledge — sparse memory
//! finetuning — cuts catastrophic forgetting: NaturalQuestions F1 drops 89%
//! after full fine-tuning and 71% with LoRA on the same new facts, versus
//! 11% with sparse updates. The vault analog is a governed write discipline:
//!
//!   * every landing write (fresh insert or content-changing update)
//!     measures its activation overlap with the existing corpus — the
//!     interference score;
//!   * the commit is gated on that score against a configurable bound,
//!     fail-closed to a reviewable write quarantine (or refusal) instead of
//!     silently merging into unrelated memory;
//!   * sparse update mode touches only the activated subset of state
//!     (the body slot, activated links) and never disturbs neighbors.
//!
//! Scoring is a weighted mean over the components that can be measured:
//!
//!   * token containment — how much of the incoming fact's vocabulary is
//!     already present in an existing entity (activation via shared text);
//!   * link containment — how much of the incoming link set an existing
//!     entity already covers (activation via shared edges);
//!   * embedding similarity — max cosine to existing vectors when the
//!     write path has a vector for the incoming fact (opt-in synchronous
//!     compute — see PERSEUS_VAULT_INTERFERENCE_EMBED; #271 kept ONNX
//!     inference off the default write path).
//!
//! Missing components do not dilute the score (weights renormalize over the
//! components actually measured). Telemetry: every landing write journals
//! `interference_scored` with the full report; gate firings journal
//! `interference_quarantined` / `interference_refused`. Drift over time is
//! observable via `perseus_vault_timeline`.

use crate::models::Entity;
use serde::{Deserialize, Serialize};

/// Default interference bound: a write whose weighted activation overlap
/// with any single existing entity exceeds this is fail-closed.
pub const DEFAULT_INTERFERENCE_BOUND: f64 = 0.90;
/// Default candidate-set size for the activation overlap scan (FTS top-k).
pub const DEFAULT_INTERFERENCE_TOP_K: usize = 16;
/// Activation threshold for sparse-update link admission: a caller-supplied
/// link is stored in sparse mode only when the target's body shares at least
/// this fraction of the incoming body's tokens.
pub const DEFAULT_SPARSE_ACTIVATION: f64 = 0.30;
/// Component weights (embedding / link / token). Weights renormalize over
/// the components actually measured — a missing modality never dilutes.
pub const EMB_WEIGHT: f64 = 0.5;
pub const LINK_WEIGHT: f64 = 0.25;
pub const TOKEN_WEIGHT: f64 = 0.25;

/// How the gate treats a write that exceeds the bound.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterferenceMode {
    /// Store the write in the reviewable `write_quarantine` table (never
    /// served) instead of committing it to memory. Default.
    Quarantine,
    /// Reject the write with an error (nothing stored anywhere).
    Refuse,
    /// Bypass enforcement entirely (operator-only; telemetry still runs).
    Off,
}

impl InterferenceMode {
    /// Parse a mode string case-insensitively; `None` for unknown values
    /// (callers decide whether that is fail-closed or a no-op).
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "quarantine" | "quarantined" => Some(Self::Quarantine),
            "refuse" | "refused" => Some(Self::Refuse),
            "off" | "disabled" | "none" => Some(Self::Off),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Quarantine => "quarantine",
            Self::Refuse => "refuse",
            Self::Off => "off",
        }
    }

    pub fn from_env() -> Self {
        std::env::var("PERSEUS_VAULT_INTERFERENCE_MODE")
            .ok()
            .and_then(|v| Self::parse(&v))
            .unwrap_or(Self::Quarantine)
    }
}

/// Resolved gate configuration. Read from the environment at gate time (the
/// same pattern as the #619 dense-scan dials), so a running server reflects
/// operator changes without a restart.
#[derive(Debug, Clone)]
pub struct InterferenceConfig {
    pub mode: InterferenceMode,
    pub bound: f64,
    pub top_k: usize,
    /// Synchronously compute the incoming fact's embedding for the gate.
    /// Default off: #271 kept ONNX inference off the default write path.
    pub embed: bool,
    pub sparse_activation: f64,
    /// Scan ceiling for the embedding component (mirrors the dense arm's
    /// `PERSEUS_VAULT_DENSE_MAX_SCAN` semantics; 0 = unbounded).
    pub max_scan: usize,
}

impl Default for InterferenceConfig {
    fn default() -> Self {
        Self {
            mode: InterferenceMode::Quarantine,
            bound: DEFAULT_INTERFERENCE_BOUND,
            top_k: DEFAULT_INTERFERENCE_TOP_K,
            embed: false,
            sparse_activation: DEFAULT_SPARSE_ACTIVATION,
            max_scan: 50_000,
        }
    }
}

impl InterferenceConfig {
    pub fn from_env() -> Self {
        let base = Self::default();
        Self {
            mode: InterferenceMode::from_env(),
            bound: std::env::var("PERSEUS_VAULT_INTERFERENCE_BOUND")
                .ok()
                .and_then(|v| v.trim().parse::<f64>().ok())
                .filter(|b| (0.0..=1.0).contains(b))
                .unwrap_or(base.bound),
            top_k: std::env::var("PERSEUS_VAULT_INTERFERENCE_TOP_K")
                .ok()
                .and_then(|v| v.trim().parse::<usize>().ok())
                .filter(|k| *k >= 1 && *k <= 10_000)
                .unwrap_or(base.top_k),
            embed: match std::env::var("PERSEUS_VAULT_INTERFERENCE_EMBED")
                .unwrap_or_default()
                .trim()
                .to_ascii_lowercase()
                .as_str()
            {
                "1" | "true" | "on" | "yes" => true,
                _ => false,
            },
            sparse_activation: std::env::var("PERSEUS_VAULT_SPARSE_ACTIVATION")
                .ok()
                .and_then(|v| v.trim().parse::<f64>().ok())
                .filter(|a| (0.0..=1.0).contains(a))
                .unwrap_or(base.sparse_activation),
            max_scan: crate::db::Database::dense_max_scan_from_env(),
        }
    }
}

/// Per-write gate options carried from the tool surface (or internal
/// callers) into the remember path.
#[derive(Debug, Clone, Default)]
pub struct WriteGateOptions {
    /// Sparse update mode: touch only the activated subset of state and
    /// never disturb neighbors (no salience inflation, activation-filtered
    /// caller links, no near-duplicate absorption on insert).
    pub sparse_update: bool,
    /// Per-write mode override. On the MCP surface only `refuse` /
    /// `quarantine` are accepted (per-write `off` would let a caller bypass
    /// the gate); internal trusted callers may pass any mode.
    pub mode_override: Option<InterferenceMode>,
    /// Per-write bound override — may only TIGHTEN the configured bound
    /// (a lower bound fires more often; a looser one is refused fail-closed).
    pub bound_override: Option<f64>,
    /// Memory slots this write is INTENTIONALLY updating (its own identity,
    /// cited sources being consolidated, curated source ids). Activation
    /// overlap is measured against the rest of the corpus — the paper's
    /// "slots activated by the new knowledge" are the update targets, and
    /// interference is disturbance of everything else.
    pub exclude_ids: Vec<String>,
}

impl WriteGateOptions {
    pub fn none() -> Self {
        Self::default()
    }

    /// The sparse link-activation threshold, resolved from the environment
    /// at gate time (the options carry no per-write threshold override).
    pub fn sparse_activation_param(&self) -> f64 {
        InterferenceConfig::from_env().sparse_activation
    }
}

/// The measured activation overlap of one incoming write against the
/// existing corpus, plus the gate verdict. Serialized into the journal so
/// drift is observable over time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterferenceReport {
    /// Weighted mean over measured components, in [0, 1].
    pub score: f64,
    /// Max token containment of the incoming vocabulary in one existing
    /// entity (undefined components are omitted and reported as -1).
    pub max_token_containment: f64,
    /// Max link-target containment of the incoming link set in one existing
    /// entity.
    pub max_link_containment: f64,
    /// Max cosine similarity to one existing embedded entity.
    pub max_emb_similarity: f64,
    /// The existing entity that contributed the top component (empty when
    /// no component was measurable).
    pub top_entity_id: String,
    /// How many existing entities were scored as candidates.
    pub candidates: usize,
    /// Which components were actually measured: subset of
    /// ["token", "link", "embedding"].
    pub components: Vec<String>,
    /// Gate verdict: "allowed" | "quarantined" | "refused" | "off".
    pub decision: String,
    /// The effective bound the verdict was checked against.
    pub bound: f64,
    /// The effective mode.
    pub mode: String,
}

impl InterferenceReport {
    fn empty(mode: InterferenceMode, bound: f64) -> Self {
        Self {
            score: 0.0,
            max_token_containment: -1.0,
            max_link_containment: -1.0,
            max_emb_similarity: -1.0,
            top_entity_id: String::new(),
            candidates: 0,
            components: Vec::new(),
            decision: "allowed".to_string(),
            bound,
            mode: mode.as_str().to_string(),
        }
    }
}

/// Outcome of the interference gate for one write.
#[derive(Debug)]
pub(crate) enum GateVerdict {
    /// Gate passed (report journaled as `interference_scored`); the
    /// Option is None when no gate was needed (identical re-assert).
    Proceed(Option<InterferenceReport>),
    /// Write staged in `write_quarantine`; carries (quarantine id, action).
    Quarantined(String, String),
    /// Write refused; the caller returns the error.
    Refused,
}

/// Gate decision for a write.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterferenceDecision {
    Allow,
    Quarantine,
    Refuse,
}

/// Deterministic lowercase word tokenizer: ASCII-lowercases, splits on
/// non-alphanumeric runs, keeps tokens of length >= 2 — plus pure-digit
/// tokens of ANY length (record indices like "0"/"1" are the tokens that
/// tell templated records apart; dropping them makes "probe 0" and
/// "probe 1" token-identical). Dedupes preserving first-seen order.
/// Shared by the token-containment component and the sparse link-activation
/// test.
pub fn tokenize(text: &str) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    for chunk in text
        .to_lowercase()
        .split(|c: char| !c.is_ascii_alphanumeric())
    {
        let is_digits = !chunk.is_empty() && chunk.bytes().all(|b| b.is_ascii_digit());
        if (chunk.len() >= 2 || is_digits) && seen.insert(chunk.to_string()) {
            out.push(chunk.to_string());
        }
    }
    out
}

/// Containment of `incoming` in `existing`: |incoming ∩ existing| / |incoming|.
/// 0.0 when `incoming` is empty (nothing to contain); 1.0 when every
/// incoming token is already present.
pub fn containment(incoming: &[String], existing: &[String]) -> f64 {
    if incoming.is_empty() {
        return 0.0;
    }
    let set: std::collections::HashSet<&str> = existing.iter().map(String::as_str).collect();
    let hits = incoming.iter().filter(|t| set.contains(t.as_str())).count();
    hits as f64 / incoming.len() as f64
}

/// Content tokens of a body: the string VALUES of the JSON document
/// (recursively), falling back to the raw text when the body is not valid
/// JSON. Structural JSON keys and identity fields are deliberately
/// excluded — boilerplate every entity shares (`content`, `note`, the
/// category/key) would dilute the activation overlap of the fact itself.
pub fn body_tokens(body_json: &str) -> Vec<String> {
    let mut collected: Vec<String> = Vec::new();
    match serde_json::from_str::<serde_json::Value>(body_json) {
        Ok(v) => collect_strings(&v, &mut collected),
        Err(_) => collected.extend(tokenize(body_json)),
    }
    let mut seen = std::collections::HashSet::new();
    collected.retain(|t| seen.insert(t.clone()));
    collected
}

fn collect_strings(v: &serde_json::Value, out: &mut Vec<String>) {
    match v {
        serde_json::Value::String(s) => out.extend(tokenize(s)),
        serde_json::Value::Array(items) => {
            for item in items {
                collect_strings(item, out);
            }
        }
        serde_json::Value::Object(map) => {
            for (_k, val) in map {
                collect_strings(val, out);
            }
        }
        _ => {}
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0f64;
    let mut na = 0f64;
    let mut nb = 0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += (*x as f64) * (*y as f64);
        na += (*x as f64) * (*x as f64);
        nb += (*y as f64) * (*y as f64);
    }
    if na <= 0.0 || nb <= 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

/// The FTS match expression for a token set: `"t1" OR "t2" ...`. Terms are
/// alphanumeric by construction (tokenize), so quoting is safe.
fn fts_match_expr(tokens: &[String]) -> String {
    tokens
        .iter()
        .map(|t| format!("\"{}\"", t.replace('"', "\"\"")))
        .collect::<Vec<_>>()
        .join(" OR ")
}

/// Compute the activation overlap of `incoming` against the corpus (same
/// workspace, unarchived, excluding `exclude_ids`), measured over the
/// activated candidate set: FTS top-k by shared vocabulary, plus stored
/// embeddings when `incoming_vec` is provided. Pure read — never mutates.
/// `decrypt(raw, category, key)` resolves AES-GCM at-rest candidate bodies
/// to plaintext (the FTS index stores plaintext only in legacy mode and keyed
/// blind tokens in protected mode; the entities row stores ciphertext on
/// encrypted deployments — measuring ciphertext scores 0.0, the same class as
/// the #884 consolidate scan fix). Auth failures return
/// empty (fail closed: tampered content is never treated as evidence).
pub fn compute_activation_overlap(
    conn: &rusqlite::Connection,
    incoming: &Entity,
    incoming_vec: Option<&[f32]>,
    exclude_ids: &[String],
    cfg: &InterferenceConfig,
    decrypt: &dyn Fn(&str, &str, &str) -> String,
) -> Result<InterferenceReport, String> {
    compute_activation_overlap_with_search(
        conn,
        incoming,
        incoming_vec,
        exclude_ids,
        cfg,
        decrypt,
        None,
    )
}

/// Strict encrypted variant. Unlike the compatibility wrapper, the decrypt
/// callback can reject an unauthenticated candidate instead of silently
/// converting it into an empty body.
pub fn compute_activation_overlap_strict(
    conn: &rusqlite::Connection,
    incoming: &Entity,
    incoming_vec: Option<&[f32]>,
    exclude_ids: &[String],
    cfg: &InterferenceConfig,
    decrypt: &dyn Fn(&str, &str, &str) -> Result<String, String>,
) -> Result<InterferenceReport, String> {
    compute_activation_overlap_with_search_strict(
        conn,
        incoming,
        incoming_vec,
        exclude_ids,
        cfg,
        decrypt,
        None,
    )
}

/// Variant used by encrypted databases. The callback is the active
/// domain-separated blind-token encoder; plaintext callers keep the legacy
/// expression through [`compute_activation_overlap`].
pub fn compute_activation_overlap_with_search(
    conn: &rusqlite::Connection,
    incoming: &Entity,
    incoming_vec: Option<&[f32]>,
    exclude_ids: &[String],
    cfg: &InterferenceConfig,
    decrypt: &dyn Fn(&str, &str, &str) -> String,
    encryption: Option<&crate::encryption::EncryptionManager>,
) -> Result<InterferenceReport, String> {
    let strict_decrypt =
        |raw: &str, category: &str, key: &str| Ok::<String, String>(decrypt(raw, category, key));
    compute_activation_overlap_with_search_strict(
        conn,
        incoming,
        incoming_vec,
        exclude_ids,
        cfg,
        &strict_decrypt,
        encryption,
    )
}

/// Encrypted-search implementation with fail-closed body authentication.
pub fn compute_activation_overlap_with_search_strict(
    conn: &rusqlite::Connection,
    incoming: &Entity,
    incoming_vec: Option<&[f32]>,
    exclude_ids: &[String],
    cfg: &InterferenceConfig,
    decrypt: &dyn Fn(&str, &str, &str) -> Result<String, String>,
    encryption: Option<&crate::encryption::EncryptionManager>,
) -> Result<InterferenceReport, String> {
    let mut report = InterferenceReport::empty(cfg.mode, cfg.bound);
    let incoming_tokens = body_tokens(&incoming.body_json);

    // Exclusion placeholders start at ?3 — ?1 (MATCH) and ?2 (workspace)
    // are fixed. Starting at ?1 would collide with the MATCH parameter and
    // silently disable the exclusion.
    let excl_clause = if exclude_ids.is_empty() {
        String::new()
    } else {
        let ph = (3..3 + exclude_ids.len())
            .map(|i| format!("?{i}"))
            .collect::<Vec<_>>()
            .join(",");
        format!(" AND e.id NOT IN ({ph})")
    };
    let excl_params: Vec<&str> = exclude_ids.iter().map(String::as_str).collect();

    // ── token + link components over the FTS candidate set ──────────────
    let mut candidates: Vec<(String, String, String, String, String)> = Vec::new(); // id, body, links, category, key
    if !incoming_tokens.is_empty() {
        // Placeholder indexes shift with the exclusion clause: LIMIT is
        // always the last parameter (2 fixed + exclude_ids + 1).
        let limit_ph = 3 + excl_params.len();
        let sql = format!(
            "SELECT e.id, e.body_json, e.links, e.category, e.key \
             FROM entities_fts f JOIN entities e ON e.rowid = f.rowid \
             WHERE entities_fts MATCH ?1 AND e.workspace_hash = ?2 AND e.archived = 0 \
             {excl_clause} ORDER BY rank LIMIT ?{limit_ph}"
        );
        let expr = match encryption {
            Some(enc) => enc.blind_query_from_terms(&incoming_tokens),
            None => fts_match_expr(&incoming_tokens),
        };
        let top_k = cfg.top_k as i64;
        let mut params: Vec<&dyn rusqlite::ToSql> = vec![
            &expr as &dyn rusqlite::ToSql,
            &incoming.workspace_hash as &dyn rusqlite::ToSql,
        ];
        for p in &excl_params {
            params.push(p as &dyn rusqlite::ToSql);
        }
        params.push(&top_k as &dyn rusqlite::ToSql);
        let mut stmt = conn
            .prepare(&sql)
            .map_err(|e| format!("interference candidate prepare: {e}"))?;
        let rows = stmt
            .query_map(rusqlite::params_from_iter(params.iter().copied()), |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, String>(2)?,
                    r.get::<_, String>(3)?,
                    r.get::<_, String>(4)?,
                ))
            })
            .map_err(|e| format!("interference candidate query: {e}"))?;
        for row in rows {
            let row = row.map_err(|e| format!("interference candidate row: {e}"))?;
            // Decrypt AES-GCM at-rest bodies ONCE here — the FTS index holds
            // plaintext but the entities row stores ciphertext on encrypted
            // deployments; measuring ciphertext scores 0.0 (#884-class).
            // Auth failures resolve to empty (fail closed: tampered content
            // is never treated as evidence). Downstream consumers (report
            // loop, best_candidate_id) see plaintext.
            let (id, body, links, cand_cat, cand_key) = row;
            candidates.push((
                id,
                decrypt(&body, &cand_cat, &cand_key)
                    .map_err(|e| format!("interference candidate authentication failed: {e}"))?,
                links,
                cand_cat,
                cand_key,
            ));
        }
    }

    if !candidates.is_empty() {
        report.components.push("token".to_string());
        report.candidates += candidates.len();
        let incoming_targets = incoming_targets(incoming);
        for (_id, body, links_json, _cand_cat, _cand_key) in &candidates {
            let cand_tokens = body_tokens(body);
            let t = containment(&incoming_tokens, &cand_tokens);
            if t > report.max_token_containment {
                report.max_token_containment = t;
            }
            if !incoming_targets.is_empty() {
                if !report.components.contains(&"link".to_string()) {
                    report.components.push("link".to_string());
                }
                let cand_targets: Vec<String> =
                    serde_json::from_str::<Vec<crate::models::MemoryLink>>(links_json)
                        .unwrap_or_default()
                        .iter()
                        .map(|l| l.target_id.clone())
                        .collect();
                let l = containment(&incoming_targets, &cand_targets);
                if l > report.max_link_containment {
                    report.max_link_containment = l;
                }
            }
        }
        report.top_entity_id = best_candidate_id(&candidates, &incoming_tokens, &incoming_targets);
    }

    // ── embedding component (bounded scan over stored vectors) ──────────
    if let Some(vec) = incoming_vec {
        if !vec.is_empty() {
            let limit_ph = 2 + excl_params.len();
            let sql = format!(
                "SELECT id, embedding FROM entities \
                 WHERE workspace_hash = ?1 AND archived = 0 AND embedding IS NOT NULL \
                 {excl_clause} LIMIT ?{limit_ph}"
            );
            let mut params: Vec<&dyn rusqlite::ToSql> =
                vec![&incoming.workspace_hash as &dyn rusqlite::ToSql];
            for p in &excl_params {
                params.push(p as &dyn rusqlite::ToSql);
            }
            let limit = if cfg.max_scan == 0 {
                i64::MAX
            } else {
                cfg.max_scan as i64
            };
            params.push(&limit as &dyn rusqlite::ToSql);
            let mut stmt = conn
                .prepare(&sql)
                .map_err(|e| format!("interference embed prepare: {e}"))?;
            let rows = stmt
                .query_map(rusqlite::params_from_iter(params.iter().copied()), |r| {
                    Ok((r.get::<_, String>(0)?, r.get::<_, Vec<u8>>(1)?))
                })
                .map_err(|e| format!("interference embed query: {e}"))?;
            let mut scanned = 0usize;
            for row in rows {
                let (id, blob) = row.map_err(|e| format!("interference embed row: {e}"))?;
                scanned += 1;
                if let Some(stored) = crate::vector_quant::decode_stored(&blob, vec.len()) {
                    if let crate::vector_quant::StoredVec::Full(v) = stored {
                        let s = cosine(vec, &v);
                        if s > report.max_emb_similarity {
                            report.max_emb_similarity = s;
                        }
                        if !report.components.contains(&"embedding".to_string()) {
                            report.components.push("embedding".to_string());
                        }
                    }
                    // Bit-mode blobs have no reconstructible vector — skipped.
                }
            }
            report.candidates += scanned;
        }
    }

    // ── weighted mean over measured components ──────────────────────────
    let mut weight_sum = 0f64;
    let mut acc = 0f64;
    if report.components.contains(&"embedding".to_string()) {
        weight_sum += EMB_WEIGHT;
        acc += EMB_WEIGHT * report.max_emb_similarity.max(0.0);
    }
    if report.components.contains(&"link".to_string()) {
        weight_sum += LINK_WEIGHT;
        acc += LINK_WEIGHT * report.max_link_containment.max(0.0);
    }
    if report.components.contains(&"token".to_string()) {
        weight_sum += TOKEN_WEIGHT;
        acc += TOKEN_WEIGHT * report.max_token_containment.max(0.0);
    }
    report.score = if weight_sum > 0.0 {
        acc / weight_sum
    } else {
        0.0
    };
    report.score = report.score.clamp(0.0, 1.0);
    Ok(report)
}

fn incoming_targets(incoming: &Entity) -> Vec<String> {
    incoming.links.iter().map(|l| l.target_id.clone()).collect()
}

fn best_candidate_id(
    candidates: &[(String, String, String, String, String)],
    incoming_tokens: &[String],
    incoming_targets: &[String],
) -> String {
    let mut best: Option<(f64, String)> = None;
    for (id, body, links_json, _cat, _key) in candidates {
        // Candidates are already decrypted by the caller-provided callback.
        let cand_tokens = body_tokens(body);
        let cand_targets: Vec<String> =
            serde_json::from_str::<Vec<crate::models::MemoryLink>>(links_json)
                .unwrap_or_default()
                .iter()
                .map(|l| l.target_id.clone())
                .collect();
        let t = containment(incoming_tokens, &cand_tokens);
        let l = if incoming_targets.is_empty() {
            -1.0
        } else {
            containment(incoming_targets, &cand_targets)
        };
        let v = t.max(l).max(0.0);
        if v > best.as_ref().map(|(b, _)| *b).unwrap_or(-1.0) {
            best = Some((v, id.clone()));
        }
    }
    best.map(|(_, id)| id).unwrap_or_default()
}

/// Apply the gate: decide allow / quarantine / refuse for a measured score.
/// Overrides are validated fail-closed — a per-write `off` mode and a
/// looser-than-configured bound are refused by the caller (see
/// `validate_overrides`).
pub fn evaluate(
    cfg: &InterferenceConfig,
    mode_override: Option<InterferenceMode>,
    bound_override: Option<f64>,
    report: &mut InterferenceReport,
) -> InterferenceDecision {
    let mode = mode_override.unwrap_or(cfg.mode);
    let bound = bound_override.unwrap_or(cfg.bound);
    report.bound = bound;
    report.mode = mode.as_str().to_string();
    match mode {
        InterferenceMode::Off => {
            report.decision = "off".to_string();
            InterferenceDecision::Allow
        }
        InterferenceMode::Quarantine | InterferenceMode::Refuse => {
            if report.score > bound {
                report.decision = mode.as_str().to_string();
                if mode == InterferenceMode::Quarantine {
                    InterferenceDecision::Quarantine
                } else {
                    InterferenceDecision::Refuse
                }
            } else {
                report.decision = "allowed".to_string();
                InterferenceDecision::Allow
            }
        }
    }
}

/// Fail-closed override validation for the MCP surface: a per-write mode may
/// only be `refuse` or `quarantine` (never `off`), and a per-write bound may
/// only tighten the configured bound.
pub fn validate_overrides(
    cfg: &InterferenceConfig,
    mode_override: Option<InterferenceMode>,
    bound_override: Option<f64>,
) -> Result<(), String> {
    if let Some(m) = mode_override {
        if m == InterferenceMode::Off {
            return Err(
                "interference_mode override 'off' refused: per-write bypass of the \
                 interference gate is not allowed; set PERSEUS_VAULT_INTERFERENCE_MODE=off \
                 at the operator level"
                    .to_string(),
            );
        }
    }
    if let Some(b) = bound_override {
        if !(0.0..=1.0).contains(&b) {
            return Err(format!("interference_bound must be within [0,1], got {b}"));
        }
        if b > cfg.bound {
            return Err(format!(
                "interference_bound override {b} would LOOSEN the configured bound {} — \
                 per-write overrides may only tighten the gate",
                cfg.bound
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenize_is_deterministic_and_filters_noise() {
        let t1 = tokenize("The quick brown FOX jumps! fox over the lazy dog");
        let t2 = tokenize("the quick brown fox jumps over the lazy dog");
        assert_eq!(t1, t2, "case/punctuation-insensitive, deduped");
        assert!(t1.contains(&"fox".to_string()));
        assert!(
            !t1.iter()
                .any(|t| t.len() < 2 && !t.bytes().all(|b| b.is_ascii_digit())),
            "single non-digit chars dropped"
        );
        // Pure-digit tokens survive at ANY length: they are the record
        // indices that tell templated entries apart ("probe 0" vs "probe 1").
        assert_eq!(
            tokenize("probe 0 and probe 9"),
            vec!["probe", "0", "and", "9"]
        );
    }

    #[test]
    fn containment_is_incoming_normalized() {
        let inc: Vec<String> = ["a", "b", "c"].iter().map(|s| s.to_string()).collect();
        let full: Vec<String> = ["a", "b", "c", "d"].iter().map(|s| s.to_string()).collect();
        let half: Vec<String> = ["a", "b"].iter().map(|s| s.to_string()).collect();
        assert_eq!(containment(&inc, &full), 1.0);
        assert_eq!(containment(&inc, &half), 2.0 / 3.0);
        assert_eq!(
            containment(&[], &full),
            0.0,
            "empty incoming contains nothing"
        );
    }

    #[test]
    fn mode_parse_is_case_insensitive_and_fail_closed_on_unknown() {
        assert_eq!(
            InterferenceMode::parse("QUARANTINE"),
            Some(InterferenceMode::Quarantine)
        );
        assert_eq!(
            InterferenceMode::parse("refuse"),
            Some(InterferenceMode::Refuse)
        );
        assert_eq!(InterferenceMode::parse("Off"), Some(InterferenceMode::Off));
        assert_eq!(InterferenceMode::parse("banana"), None);
        // Unknown env value resolves to the fail-closed default.
        assert_eq!(InterferenceMode::from_env(), InterferenceMode::Quarantine);
    }

    #[test]
    fn evaluate_fires_only_strictly_above_bound_and_honors_mode() {
        let cfg = InterferenceConfig::default();
        let mut r = InterferenceReport::empty(InterferenceMode::Quarantine, cfg.bound);
        r.score = cfg.bound; // exactly at the bound: allowed (strict >)
        assert_eq!(
            evaluate(&cfg, None, None, &mut r),
            InterferenceDecision::Allow
        );
        r.score = cfg.bound + 0.001;
        assert_eq!(
            evaluate(&cfg, None, None, &mut r),
            InterferenceDecision::Quarantine
        );
        assert_eq!(
            evaluate(&cfg, Some(InterferenceMode::Refuse), None, &mut r),
            InterferenceDecision::Refuse
        );
        assert_eq!(
            evaluate(&cfg, Some(InterferenceMode::Off), None, &mut r),
            InterferenceDecision::Allow
        );
        assert_eq!(r.decision, "off");
    }

    #[test]
    fn validate_overrides_tighten_only_and_never_off() {
        let cfg = InterferenceConfig::default();
        assert!(validate_overrides(&cfg, Some(InterferenceMode::Refuse), Some(0.8)).is_ok());
        assert!(validate_overrides(&cfg, Some(InterferenceMode::Quarantine), None).is_ok());
        assert!(validate_overrides(&cfg, None, None).is_ok());
        let err = validate_overrides(&cfg, Some(InterferenceMode::Off), None).unwrap_err();
        assert!(err.contains("bypass"), "per-write off must be refused");
        let err = validate_overrides(&cfg, None, Some(0.95)).unwrap_err();
        assert!(err.contains("LOOSEN"), "loosening override must be refused");
        assert!(validate_overrides(&cfg, None, Some(1.5)).is_err());
    }

    #[test]
    fn config_from_env_defaults_fail_closed() {
        // No env set in the test process: defaults apply.
        let cfg = InterferenceConfig::from_env();
        assert_eq!(cfg.mode, InterferenceMode::Quarantine);
        assert_eq!(cfg.bound, DEFAULT_INTERFERENCE_BOUND);
        assert!(!cfg.embed);
    }
}
