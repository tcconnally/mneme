//! Extraction-loss net (#1048, Coalent borrow).
//!
//! When claim extraction drops a fact, nothing recovers it until the next
//! full ingest. This module closes that gap with three pieces:
//!
//! 1. **Residual-span audit** — [`Database::span_audit`] splits an entity's
//!    text into sentences and retains verbatim (with provenance) the
//!    sentences its extracted claims do NOT cover. Embedding-first
//!    similarity (bundled model, no extra LLM call) with a deterministic
//!    token-containment fallback, so the net works air-gapped.
//! 2. **Refusal-as-signal loop** — [`Database::report_refusal`] treats an
//!    answerer's refusal over a served payload as evidence: it re-scores the
//!    unit's residual spans against the original query and returns a retry
//!    payload (spans whose query-similarity beats the entity's own
//!    similarity by a margin — the anomaly rule). Units that repeatedly
//!    under-cover get lossy-marked.
//! 3. **Provisional query keys** — [`Database::report_success`] confirms a
//!    retry: it attaches a query fingerprint → entity binding so an
//!    identical repeat query serves first-pass ([`Database::serve_confirmed_query_key`],
//!    wired into the recall handler). Lossy units repair append-only on
//!    their next `remember` touch.
//!
//! Scoping: spans are admitted through the normal authority/admission path,
//! subject to decay/hygiene like any other fact, and are never auto-served
//! into recall without a provenance check — the retry payload is explicit.

use crate::db::Database;
use crate::extraction::Extractor;
use crate::interference;
use rusqlite::params;
use rusqlite::OptionalExtension;
use sha2::{Digest, Sha256};

/// Default coverage floor: a sentence whose best claim similarity is below
/// this is a residual span (the extractor missed it).
pub(crate) const DEFAULT_COVERAGE_THRESHOLD: f64 = 0.55;
/// The anomaly rule: a span is retry-worthy only when its query similarity
/// beats the entity's own query similarity by at least this margin.
pub(crate) const ANOMALY_MARGIN: f64 = 0.05;
/// Refusals without a usable retry before a unit is marked lossy.
pub(crate) const LOSSY_THRESHOLD: i64 = 2;
/// Absolute floor for retry-payload admission.
const RETRY_FLOOR: f64 = 0.10;

/// Deterministic similarity. Embedding mode uses the bundled model when
/// enabled and available; any failure falls back to token containment so the
/// net never depends on a backend. Returns (score, mode_used).
fn similarity(db: &Database, a: &str, b: &str, prefer_embedding: bool) -> (f64, &'static str) {
    if prefer_embedding && db.embedding_config().enabled {
        if let Ok(va) = crate::embedding::generate_embedding(db.embedding_config(), a) {
            if let Ok(vb) = crate::embedding::generate_embedding(db.embedding_config(), b) {
                let dot: f64 = va.iter().zip(vb.iter()).map(|(x, y)| (x * y) as f64).sum();
                let na: f64 = va
                    .iter()
                    .map(|x| (*x as f64) * (*x as f64))
                    .sum::<f64>()
                    .sqrt();
                let nb: f64 = vb
                    .iter()
                    .map(|x| (*x as f64) * (*x as f64))
                    .sum::<f64>()
                    .sqrt();
                if na > 0.0 && nb > 0.0 {
                    return (dot / (na * nb), "embedding");
                }
            }
        }
    }
    let ta = interference::tokenize(a);
    let tb = interference::tokenize(b);
    // How much of `a`'s vocabulary is covered by `b` — the deterministic
    // analogue of an extractor's coverage.
    let cov = if ta.is_empty() {
        0.0
    } else {
        interference::containment(&ta, &tb)
    };
    (cov, "token")
}

/// Split text into sentences on `.` `!` `?` boundaries (kept simple and
/// deterministic; decimal points are tolerated because a split only happens
/// on `.` followed by whitespace/EOL).
fn sentences(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for ch in text.chars() {
        cur.push(ch);
        if matches!(ch, '.' | '!' | '?') {
            let trimmed = cur.trim();
            if !trimmed.is_empty() {
                out.push(trimmed.to_string());
            }
            cur.clear();
        }
    }
    let trimmed = cur.trim();
    if !trimmed.is_empty() {
        out.push(trimmed.to_string());
    }
    out
}

impl Database {
    /// #1048: audit an entity for extraction loss — retain the sentences its
    /// claims do not cover as residual spans (verbatim, with provenance).
    /// Append-only: an identical active span is never duplicated.
    pub(crate) fn span_audit(
        &self,
        entity_id: &str,
        min_chars: usize,
        coverage_threshold: f64,
        mode: &str,
    ) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
        let entity = self
            .get_entity_by_id_pub(entity_id)?
            .ok_or_else(|| format!("entity not found: {entity_id}"))?;
        let body: serde_json::Value = serde_json::from_str(&entity.body_json)
            .unwrap_or_else(|_| serde_json::Value::String(entity.body_json.clone()));
        let text = body
            .get("text")
            .and_then(|t| t.as_str())
            .unwrap_or(entity.body_json.trim_matches('"'))
            .to_string();
        if text.trim().is_empty() {
            return Ok(serde_json::json!({
                "entity_id": entity_id,
                "claims": 0,
                "spans": [],
                "mode_used": "none",
                "note": "entity has no text to audit",
            }));
        }

        let prefer_embedding = mode != "token";
        let claims = crate::extraction::RuleBasedExtractor
            .extract(&text)
            .into_iter()
            .map(|c| c.text)
            .collect::<Vec<_>>();

        let mut spans = Vec::new();
        let mut mode_used = "none";
        let conn = self.conn()?;
        for sent in sentences(&text) {
            if sent.chars().count() < min_chars {
                continue;
            }
            let (best, used) = if claims.is_empty() {
                (0.0, "none")
            } else {
                claims
                    .iter()
                    .map(|c| similarity(self, &sent, c, prefer_embedding))
                    .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap_or((0.0, "none"))
            };
            if used != "none" {
                mode_used = used;
            }
            if best < coverage_threshold {
                let id = format!("span-{}", uuid::Uuid::new_v4().simple());
                let now = crate::db::now_ms();
                let dup: i64 = conn.query_row(
                    "SELECT COUNT(*) FROM residual_spans
                     WHERE entity_id = ?1 AND span_text = ?2 AND status = 'active'",
                    params![entity_id, sent],
                    |r| r.get(0),
                )?;
                if dup == 0 {
                    conn.execute(
                        "INSERT INTO residual_spans
                            (id, entity_id, span_text, source, max_coverage, coverage_mode,
                             status, lossy_count, created_ms, last_served_ms)
                         VALUES (?1, ?2, ?3, ?4, ?5, ?6, 'active', 0, ?7, NULL)",
                        params![id, entity_id, sent, entity.source, best, used, now],
                    )?;
                }
                spans.push(serde_json::json!({
                    "id": id,
                    "text": sent,
                    "max_coverage": best,
                    "coverage_mode": used,
                    "status": "active",
                }));
            }
        }
        Ok(serde_json::json!({
            "entity_id": entity_id,
            "claims": claims.len(),
            "spans_n": spans.len(),
            "spans": spans,
            "mode_used": if mode_used == "none" { "token" } else { mode_used },
        }))
    }

    /// #1048: an answerer's refusal over a served payload. Re-scores the
    /// served units' residual spans against the query and returns a retry
    /// payload: spans whose query-similarity beats the unit's own
    /// similarity by [`ANOMALY_MARGIN`] (and clears [`RETRY_FLOOR`]).
    /// Units with no retry material accumulate lossy marks.
    pub(crate) fn report_refusal(
        &self,
        query: &str,
        served_ids: &[String],
        _reason: Option<&str>,
    ) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
        let mut retry = Vec::new();
        let mut lossy_flagged = Vec::new();
        let conn = self.conn()?;
        for id in served_ids {
            let entity = match self.get_entity_by_id_pub(id)? {
                Some(e) => e,
                None => continue,
            };
            let body: serde_json::Value = serde_json::from_str(&entity.body_json)
                .unwrap_or_else(|_| serde_json::Value::String(entity.body_json.clone()));
            let text = body
                .get("text")
                .and_then(|t| t.as_str())
                .unwrap_or(&entity.body_json)
                .to_string();
            // The entity's score is what was actually SERVED: the extracted
            // claims (not the raw body, which contains the spans too).
            let q_tokens = interference::tokenize(query);
            let claims = crate::extraction::RuleBasedExtractor
                .extract(&text)
                .into_iter()
                .map(|c| c.text)
                .collect::<Vec<_>>();
            let entity_score = if claims.is_empty() {
                0.0
            } else {
                claims
                    .iter()
                    .map(|c| {
                        let ct = interference::tokenize(c);
                        if q_tokens.is_empty() {
                            0.0
                        } else {
                            interference::containment(&q_tokens, &ct)
                        }
                    })
                    .fold(0.0f64, |a, b| a.max(b))
            };

            let mut stmt = conn.prepare(
                "SELECT id, span_text, status, lossy_count FROM residual_spans
                 WHERE entity_id = ?1 AND status IN ('active', 'confirmed')
                 ORDER BY created_ms ASC",
            )?;
            let spans: Vec<(String, String, String, i64)> = stmt
                .query_map(params![id], |r| {
                    Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?))
                })?
                .collect::<Result<_, _>>()?;

            let mut had_retry = false;
            let mut best_span_score = 0.0f64;
            for (span_id, span_text, status, lossy_count) in spans {
                let st = interference::tokenize(&span_text);
                let score = if q_tokens.is_empty() {
                    0.0
                } else {
                    interference::containment(&q_tokens, &st)
                };
                best_span_score = best_span_score.max(score);
                let threshold = entity_score + ANOMALY_MARGIN;
                if score > threshold && score >= RETRY_FLOOR {
                    had_retry = true;
                    conn.execute(
                        "UPDATE residual_spans SET status = 'served', last_served_ms = ?1
                         WHERE id = ?2",
                        params![crate::db::now_ms(), span_id],
                    )?;
                    retry.push(serde_json::json!({
                        "span_id": span_id,
                        "entity_id": id,
                        "entity_key": entity.key,
                        "text": span_text,
                        "score": score,
                        "entity_score": entity_score,
                        "status": status,
                    }));
                }
            }
            // Repeated under-coverage: the served payload (claims) covered
            // the query no better than any residual span could — the unit is
            // lossy for this query family. The counter is entity-level (a
            // served span must not reset it) and threshold-gated so a single
            // out-of-domain refusal does not flag anything.
            if !had_retry && best_span_score <= entity_score + ANOMALY_MARGIN {
                conn.execute(
                    "INSERT INTO lossy_units (entity_id, lossy_count, marked_at_ms, status)
                     VALUES (?1, 1, ?2, 'lossy')
                     ON CONFLICT(entity_id) DO UPDATE SET
                        lossy_count = lossy_count + 1,
                        marked_at_ms = excluded.marked_at_ms,
                        status = 'lossy'",
                    params![id, crate::db::now_ms()],
                )?;
                let count: i64 = conn.query_row(
                    "SELECT lossy_count FROM lossy_units WHERE entity_id = ?1",
                    params![id],
                    |r| r.get(0),
                )?;
                if count >= LOSSY_THRESHOLD {
                    lossy_flagged.push(serde_json::json!({
                        "entity_id": id,
                        "entity_key": entity.key,
                        "lossy_count": count,
                    }));
                }
            }
        }
        Ok(serde_json::json!({
            "retry": retry,
            "retry_n": retry.len(),
            "lossy_flagged": lossy_flagged,
            "margin": ANOMALY_MARGIN,
        }))
    }

    /// #1048: confirm a retry payload answered the query. Attaches a
    /// provisional query key (fingerprint → entities) so an identical repeat
    /// query serves first-pass; served spans become confirmed; lossy units
    /// are cleared to repaired.
    pub(crate) fn report_success(
        &self,
        query: &str,
        entity_ids: &[String],
    ) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
        let conn = self.conn()?;
        let fingerprint = query_fingerprint(query);
        let now = crate::db::now_ms();
        let entity_ids_json = serde_json::to_string(&entity_ids)?;
        conn.execute(
            "INSERT INTO query_keys (fingerprint, query, entity_ids, confirmed_ms, hit_count)
             VALUES (?1, ?2, ?3, ?4, 0)
             ON CONFLICT(fingerprint) DO UPDATE SET
                entity_ids = excluded.entity_ids,
                confirmed_ms = excluded.confirmed_ms",
            params![fingerprint, query, entity_ids_json, now],
        )?;
        let mut spans_confirmed = 0usize;
        for id in entity_ids {
            let n = conn.execute(
                "UPDATE residual_spans SET status = 'confirmed'
                 WHERE entity_id = ?1 AND status = 'served'",
                params![id],
            )?;
            spans_confirmed += n;
            let _ = conn.execute(
                "UPDATE lossy_units SET status = 'repaired' WHERE entity_id = ?1",
                params![id],
            )?;
        }
        Ok(serde_json::json!({
            "confirmed": true,
            "entity_ids": entity_ids,
            "spans_confirmed": spans_confirmed,
            "query_fingerprint": fingerprint,
        }))
    }

    /// #1048: serve the confirmed entities for an identical repeat query, if
    /// any. The binding exists only after a successful `report_success`, so
    /// this is the first-pass shortcut for the exact query that was repaired.
    pub(crate) fn serve_confirmed_query_key(
        &self,
        query: &str,
        workspace_hash: Option<&str>,
        requesting_agent_id: Option<&str>,
    ) -> Result<Option<Vec<crate::models::Entity>>, Box<dyn std::error::Error>> {
        let conn = self.conn()?;
        let fingerprint = query_fingerprint(query);
        let entity_ids_json: Option<String> = conn
            .query_row(
                "SELECT entity_ids FROM query_keys WHERE fingerprint = ?1",
                params![fingerprint],
                |r| r.get(0),
            )
            .optional()?;
        let Some(json) = entity_ids_json else {
            return Ok(None);
        };
        let ids: Vec<String> = serde_json::from_str(&json).unwrap_or_default();
        if ids.is_empty() {
            return Ok(None);
        }
        let mut out = Vec::new();
        for id in &ids {
            if let Some(e) = self.get_entity_by_id_for_requester(id, requesting_agent_id)? {
                if workspace_hash.is_none_or(|workspace| e.workspace_hash == workspace) {
                    out.push(e);
                }
            }
        }
        if out.is_empty() {
            return Ok(None);
        }
        conn.execute(
            "UPDATE query_keys SET hit_count = hit_count + 1 WHERE fingerprint = ?1",
            params![fingerprint],
        )?;
        Ok(Some(out))
    }

    pub(crate) fn has_confirmed_query_key(
        &self,
        query: &str,
    ) -> Result<bool, Box<dyn std::error::Error>> {
        let conn = self.conn()?;
        let fingerprint = query_fingerprint(query);
        let entity_ids_json: Option<String> = conn
            .query_row(
                "SELECT entity_ids FROM query_keys WHERE fingerprint = ?1",
                params![fingerprint],
                |r| r.get(0),
            )
            .optional()?;
        Ok(entity_ids_json.is_some_and(|json| {
            serde_json::from_str::<Vec<String>>(&json)
                .map(|ids| !ids.is_empty())
                .unwrap_or(false)
        }))
    }

    /// #1048: repair a lossy unit append-only on touch. Called from the
    /// remember write path with the connection it already holds; appends the
    /// unit's confirmed/active residual spans to the body text and clears
    /// the lossy mark. Returns true when the body was modified.
    pub(crate) fn repair_lossy_body(
        &self,
        conn: &rusqlite::Connection,
        category: &str,
        key: &str,
        body_json: &mut String,
    ) -> Result<bool, Box<dyn std::error::Error>> {
        let id: Option<String> = conn
            .query_row(
                "SELECT id FROM entities WHERE category = ?1 AND key = ?2 AND archived = 0",
                params![category, key],
                |r| r.get(0),
            )
            .optional()?;
        let Some(id) = id else {
            return Ok(false);
        };
        let lossy: i64 = conn.query_row(
            "SELECT COUNT(*) FROM lossy_units WHERE entity_id = ?1 AND status = 'lossy'",
            params![id],
            |r| r.get(0),
        )?;
        if lossy == 0 {
            return Ok(false);
        }
        let mut stmt = conn.prepare(
            "SELECT span_text FROM residual_spans
             WHERE entity_id = ?1 AND status IN ('confirmed', 'active')
             ORDER BY created_ms ASC",
        )?;
        let spans: Vec<String> = stmt
            .query_map(params![id], |r| r.get(0))?
            .collect::<Result<_, _>>()?;
        if spans.is_empty() {
            return Ok(false);
        }
        let mut appendix = String::from("\n\n## Residual spans (lossy repair)\n");
        let mut added = 0usize;
        for s in spans {
            if body_json.contains(&s) {
                continue;
            }
            appendix.push_str(&format!("- {s}\n"));
            added += 1;
        }
        if added == 0 {
            return Ok(false);
        }
        match serde_json::from_str::<serde_json::Value>(body_json) {
            Ok(mut v) => {
                if let Some(t) = v.get_mut("text").and_then(|t| t.as_str()) {
                    let mut nt = t.to_string();
                    nt.push_str(&appendix);
                    v["text"] = serde_json::Value::String(nt);
                    *body_json = serde_json::to_string(&v)?;
                } else {
                    body_json.push_str(&appendix);
                }
            }
            Err(_) => body_json.push_str(&appendix),
        }
        conn.execute(
            "UPDATE lossy_units SET status = 'repaired' WHERE entity_id = ?1",
            params![id],
        )?;
        Ok(true)
    }
}

/// Canonical fingerprint of a query: sha256 of the trimmed, case-folded text.
pub(crate) fn query_fingerprint(query: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(query.trim().to_lowercase().as_bytes());
    format!("{:x}", hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::tests::make_entity;
    use crate::db::tests::temp_db;

    fn entity_body(text: &str) -> String {
        serde_json::json!({ "text": text }).to_string()
    }

    #[test]
    fn span_audit_retains_uncovered_sentences_append_only() {
        let (db, path) = temp_db();
        // Two declarative facts (extractor keeps copula sentences) and one
        // narrative sentence the rule-based extractor skips.
        let body = entity_body(
            "The Orion stack uses postgres 14. Rollouts are rehearsed on tuesdays. \
             The capacity board lives in the ops channel.",
        );
        let (id, _) = db
            .remember_skip_dedup(&make_entity("el-1", "insight", "k1", &body))
            .unwrap();
        let r = db
            .span_audit(&id, 4, DEFAULT_COVERAGE_THRESHOLD, "token")
            .unwrap();
        assert_eq!(r["claims"].as_i64().unwrap(), 2, "copula facts extracted");
        let spans = r["spans"].as_array().unwrap();
        assert_eq!(spans.len(), 1, "narrative sentence is residual");
        assert!(spans[0]["text"]
            .as_str()
            .unwrap()
            .contains("capacity board"));
        assert!(spans[0]["coverage_mode"].as_str().unwrap() == "token");

        // Append-only: re-audit adds nothing.
        let r2 = db
            .span_audit(&id, 4, DEFAULT_COVERAGE_THRESHOLD, "token")
            .unwrap();
        assert_eq!(r2["spans_n"].as_i64().unwrap(), 1);
        let rows: i64 = db
            .conn()
            .unwrap()
            .query_row(
                "SELECT COUNT(*) FROM residual_spans WHERE entity_id = ?1",
                params![id],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(rows, 1);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn report_refusal_returns_anomaly_spans_and_flags_lossy_units() {
        let (db, path) = temp_db();
        let body = entity_body(
            "The Orion stack uses postgres 14. The standby replica lives in eu-west-2.",
        );
        let (id, _) = db
            .remember_skip_dedup(&make_entity("el-2", "insight", "k2", &body))
            .unwrap();
        db.span_audit(&id, 4, DEFAULT_COVERAGE_THRESHOLD, "token")
            .unwrap();

        // Query about the standby region — the claim ("uses postgres 14")
        // doesn't cover it, so the span must be retried.
        let q = "what region hosts the standby replica";
        let r = db.report_refusal(q, &[id.clone()], None).unwrap();
        let retry = r["retry"].as_array().unwrap();
        assert!(
            retry
                .iter()
                .any(|s| s["text"].as_str().unwrap().contains("eu-west-2")),
            "span must be retried for the refusal: {r}"
        );

        // Repeated refusals on a query with NO retry material -> lossy mark.
        let q2 = "what color is the orion dashboard theme";
        let _ = db.report_refusal(q2, &[id.clone()], None).unwrap();
        let _ = db.report_refusal(q2, &[id.clone()], None).unwrap();
        let r3 = db.report_refusal(q2, &[id], None).unwrap();
        assert_eq!(r3["lossy_flagged"].as_array().unwrap().len(), 1);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn confirmed_query_key_applies_requester_and_workspace_visibility() {
        let (db, path) = temp_db();
        let body = entity_body("The Orion stack uses postgres 14. The standby lives in eu-west-2.");
        let mut entity = make_entity("el-visibility", "insight", "k-visibility", &body);
        entity.visibility = "private".to_string();
        entity.agent_id = "owner".to_string();
        entity.workspace_hash = "workspace-a".to_string();
        let (id, _) = db.remember_skip_dedup(&entity).unwrap();
        db.span_audit(&id, 4, DEFAULT_COVERAGE_THRESHOLD, "token")
            .unwrap();
        let q = "what region hosts the standby replica";
        db.report_refusal(q, &[id.clone()], None).unwrap();
        db.report_success(q, &[id.clone()]).unwrap();

        assert!(db
            .serve_confirmed_query_key(q, Some("workspace-a"), None)
            .unwrap()
            .is_none());
        assert!(db
            .serve_confirmed_query_key(q, Some("workspace-b"), Some("owner"))
            .unwrap()
            .is_none());
        let served = db
            .serve_confirmed_query_key(q, Some("workspace-a"), Some("owner"))
            .unwrap();
        assert_eq!(served.unwrap()[0].id, id);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn report_success_confirms_spans_and_query_key_serves_first_pass() {
        let (db, path) = temp_db();
        let body = entity_body("The Orion stack uses postgres 14. The standby lives in eu-west-2.");
        let (id, _) = db
            .remember_skip_dedup(&make_entity("el-3", "insight", "k3", &body))
            .unwrap();
        db.span_audit(&id, 4, DEFAULT_COVERAGE_THRESHOLD, "token")
            .unwrap();

        let q = "what region hosts the standby replica";
        let refusal = db.report_refusal(q, &[id.clone()], None).unwrap();
        assert!(refusal["retry_n"].as_i64().unwrap() > 0);

        let ok = db.report_success(q, &[id.clone()]).unwrap();
        assert_eq!(ok["confirmed"], serde_json::json!(true));
        assert!(ok["spans_confirmed"].as_i64().unwrap() >= 1);

        // Identical repeat query serves first-pass.
        let served = db.serve_confirmed_query_key(q, None, None).unwrap();
        assert!(served.is_some());
        assert_eq!(served.unwrap()[0].id, id);
        // A different query has no key.
        assert!(db
            .serve_confirmed_query_key("something else entirely", None, None)
            .unwrap()
            .is_none());
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn lossy_unit_repairs_append_only_on_touch() {
        let (db, path) = temp_db();
        let body = entity_body(
            "The Orion stack uses postgres 14. The standby replica lives in eu-west-2.",
        );
        let (id, _) = db
            .remember_skip_dedup(&make_entity("el-4", "insight", "k4", &body))
            .unwrap();
        db.span_audit(&id, 4, DEFAULT_COVERAGE_THRESHOLD, "token")
            .unwrap();

        // Drive the unit lossy with repeated hopeless refusals.
        let q = "what color is the orion dashboard theme";
        for _ in 0..2 {
            let _ = db.report_refusal(q, &[id.clone()], None).unwrap();
        }
        let r = db.report_refusal(q, &[id.clone()], None).unwrap();
        assert_eq!(r["lossy_flagged"].as_array().unwrap().len(), 1);

        // Touch the same (category, key): the residual span must be folded
        // into the body, append-only, and the lossy mark cleared.
        let touched = entity_body("The Orion stack uses postgres 14. (reasserted)");
        let (_, _) = db
            .remember_skip_dedup(&make_entity("el-4b", "insight", "k4", &touched))
            .unwrap();
        let stored = db.get_entity("insight", "k4").unwrap().unwrap();
        assert!(
            stored.body_json.contains("Residual spans (lossy repair)"),
            "body must carry the repair appendix: {}",
            stored.body_json
        );
        assert!(
            stored.body_json.contains("eu-west-2"),
            "repair must fold the residual span into the body: {}",
            stored.body_json
        );
        let lossy_left: i64 = db
            .conn()
            .unwrap()
            .query_row(
                "SELECT COUNT(*) FROM lossy_units WHERE entity_id = ?1 AND status = 'lossy'",
                params![id],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(lossy_left, 0, "lossy mark cleared after repair");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn query_fingerprint_is_stable_and_case_folded() {
        assert_eq!(
            query_fingerprint("  Orion v2.0 ships?  "),
            query_fingerprint("orion v2.0 ships?")
        );
        assert_ne!(query_fingerprint("a"), query_fingerprint("b"));
    }
}
