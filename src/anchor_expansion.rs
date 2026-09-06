//! #1009: active-decision anchor query expansion (MindCache borrow).
//!
//! MindCache takes the top-k currently-ACTIVE decisions and uses each one's
//! text as an additional lexical (BM25) query at retrieval time — expansion,
//! not fusion — so memories sharing terms with a standing decision rank up
//! even with near-zero similarity to the raw query ("whenever I buy
//! something expensive, warranty matters" surfacing on a mattress query).
//!
//! The vault's anchor set is strictly richer: keystones (#683, mandatory
//! policy rules with weights) plus ACTIVE decision-category entities (active
//! = no successor claims it via the #363/#472 supersede chain — a structural
//! fact, not an LLM status label). This module:
//!
//! 1. `load_anchors` — keystones (weight-ranked, workspace-scoped) + active
//!    decisions (recency-ranked, workspace-scoped, category list configurable
//!    via PERSEUS_VAULT_ANCHOR_CATEGORIES, default "decision").
//! 2. `anchor_queries` — sanitized lexical queries from anchor text.
//! 3. `boost_anchor_matches` — FTS5 re-check of fused candidates against the
//!    anchor queries; matches get a CAPPED boost and are recorded as
//!    anchor-matched (visible in the fused trace — no silent reranking).
//!
//! Guards, per the issue:
//! - Anchor domination cap: per-match boost is capped (ANCHOR_BOOST_CAP) and
//!   the cumulative boost is capped (MAX_ANCHOR_BOOST) — anchors steer, the
//!   raw-query arm keeps the floor.
//! - Workspace scoping: anchors are per-workspace (+ global '') — never
//!   cross-tenant.
//! - Opt-in: `anchor_expansion` on RecallParams, default OFF. A default
//!   recall stays byte-identical (#247).

use crate::db::Database;
use rusqlite::params;
use serde::Serialize;

/// Default number of anchors taken per source (keystones, decisions).
pub const DEFAULT_ANCHOR_K: usize = 3;
/// Boost per anchor match (×1.15), applied cumulatively.
pub const ANCHOR_BOOST: f64 = 0.15;
/// Cumulative boost cap (×1.5) — anchors steer, never dominate.
pub const MAX_ANCHOR_BOOST: f64 = 1.5;
/// Anchor text truncated to this many chars per query.
pub const ANCHOR_QUERY_MAX_CHARS: usize = 200;

#[derive(Debug, Clone, Serialize)]
pub struct Anchor {
    pub text: String,
    pub weight: f64,
    pub source: &'static str,
}

fn anchor_categories() -> Vec<String> {
    std::env::var("PERSEUS_VAULT_ANCHOR_CATEGORIES")
        .unwrap_or_else(|_| "decision".to_string())
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

/// Load anchors from a plaintext store. Encrypted callers must use
/// [`load_anchors_with_encryption`] so decision bodies are authenticated and
/// decrypted before their terms influence retrieval.
pub fn load_anchors(
    conn: &rusqlite::Connection,
    workspace_hash: Option<&str>,
    k: usize,
) -> Result<Vec<Anchor>, Box<dyn std::error::Error>> {
    load_anchors_with_encryption(conn, workspace_hash, k, None)
}

/// Load anchors with an optional body-encryption manager. When encryption is
/// present, every decision body is authenticated with the canonical AAD (plus
/// the deliberate legacy-AAD read fallback) before it is projected into an
/// anchor query. Authentication failure aborts the expansion rather than
/// allowing ciphertext or a forged body to become retrieval evidence.
pub fn load_anchors_with_encryption(
    conn: &rusqlite::Connection,
    workspace_hash: Option<&str>,
    k: usize,
    encryption: Option<&crate::encryption::EncryptionManager>,
) -> Result<Vec<Anchor>, Box<dyn std::error::Error>> {
    let mut anchors: Vec<Anchor> = Vec::new();
    let ws = workspace_hash.unwrap_or("");
    let k = k.max(1);

    // Keystones: mandatory policy (#683). Global ('') or this workspace.
    let mut stmt = conn.prepare(
        "SELECT content, weight FROM keystones \
         WHERE (workspace_hash = ?1 OR workspace_hash = '') \
         ORDER BY weight DESC, id ASC LIMIT ?2",
    )?;
    let rows = stmt.query_map(params![ws, k as i64], |r| {
        Ok((r.get::<_, String>(0)?, r.get::<_, f64>(1).unwrap_or(1.0)))
    })?;
    for row in rows {
        let (content, weight) = row?;
        if !content.trim().is_empty() {
            anchors.push(Anchor {
                text: content,
                weight,
                source: "keystone",
            });
        }
    }
    drop(stmt);

    // Active decisions: category in the configured list, live, not replaced
    // by any successor (supersede chain), not invalidated, workspace-scoped.
    // Placeholders are EXPLICITLY numbered (anonymous `?` mixed with `?N`
    // reuses slot 1 in SQLite and silently binds the wrong value).
    let cats = anchor_categories();
    let cat_placeholders: Vec<String> = (0..cats.len()).map(|i| format!("?{}", i + 2)).collect();
    let limit_slot = cats.len() + 2;
    let sql = format!(
        "SELECT e.category, e.key, e.body_json FROM entities e \
         WHERE e.category IN ({}) \
           AND e.archived = 0 \
           AND e.status IN ('active','draft') \
           AND e.invalidated_at_unix_ms IS NULL \
           AND (e.workspace_hash = ?1 OR e.workspace_hash = '') \
           AND NOT EXISTS (SELECT 1 FROM entities e2 WHERE e2.supersedes = e.id) \
         ORDER BY e.last_accessed_unix_ms DESC LIMIT ?{}",
        cat_placeholders.join(","),
        limit_slot
    );
    let mut binds: Vec<&dyn rusqlite::types::ToSql> = vec![&ws];
    for c in &cats {
        binds.push(c);
    }
    let k_bind = k.to_string();
    binds.push(&k_bind);
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map(binds.as_slice(), |r| {
        Ok((
            r.get::<_, String>(0)?,
            r.get::<_, String>(1)?,
            r.get::<_, String>(2)?,
        ))
    })?;
    for row in rows {
        let (category, key, raw_body) = row?;
        let body = match encryption {
            Some(enc) => {
                match Database::decrypt_body_with_aad_fallback(enc, &raw_body, &category, &key) {
                    crate::encryption::BodyDecrypt::Plaintext(body)
                    | crate::encryption::BodyDecrypt::LegacyPlaintext(body) => body,
                    crate::encryption::BodyDecrypt::AuthFailed(_) => {
                        return Err(format!(
                            "anchor body authentication failed for {category}:{key}"
                        )
                        .into());
                    }
                }
            }
            None => raw_body,
        };
        let text = extract_text(&body);
        if !text.trim().is_empty() {
            anchors.push(Anchor {
                text,
                weight: 1.0,
                source: "decision",
            });
        }
    }
    Ok(anchors)
}

/// Extract the searchable text from an entity body: prefer the `text` /
/// `content` / `rule` JSON field; fall back to the raw string when the body
/// is not an object. Never fails — returns "" for opaque bodies.
pub fn extract_text(body_json: &str) -> String {
    let value: serde_json::Value = match serde_json::from_str(body_json) {
        Ok(v) => v,
        Err(_) => return body_json.to_string(),
    };
    if let Some(obj) = value.as_object() {
        for field in ["text", "content", "rule", "decision"] {
            if let Some(s) = obj.get(field).and_then(|v| v.as_str()) {
                if !s.trim().is_empty() {
                    return s.to_string();
                }
            }
        }
        // No known field: fall back to the compact serialization.
        return value.to_string();
    }
    value
        .as_str()
        .map(|s| s.to_string())
        .unwrap_or_else(|| value.to_string())
}

/// Build sanitized lexical queries from anchor texts: lowercase, strip
/// non-word chars, keep up to 8 longest tokens per anchor (OR-ed). Empty
/// anchors drop out.
fn anchor_query_terms(anchors: &[Anchor], max_chars: usize) -> Vec<Vec<String>> {
    let mut groups = Vec::new();
    for a in anchors {
        let text: String = a
            .text
            .chars()
            .take(max_chars)
            .map(|c| {
                if c.is_alphanumeric() || c.is_whitespace() {
                    c
                } else {
                    ' '
                }
            })
            .collect();
        let mut tokens: Vec<String> = text
            .split_whitespace()
            .map(|t| t.to_lowercase())
            .filter(|t| t.len() >= 3)
            .collect();
        tokens.sort_by(|a, b| b.len().cmp(&a.len()));
        tokens.truncate(8);
        if !tokens.is_empty() {
            groups.push(tokens);
        }
    }
    groups
}

pub fn anchor_queries(anchors: &[Anchor], max_chars: usize) -> Vec<String> {
    anchor_query_terms(anchors, max_chars)
        .into_iter()
        .map(|tokens| {
            tokens
                .into_iter()
                .map(|token| format!("\"{token}\""))
                .collect::<Vec<_>>()
                .join(" OR ")
        })
        .collect()
}

/// Re-check fused candidates against the anchor queries via FTS5 and apply
/// the capped boost. Returns the number of boosted entities. Read-only —
/// ranking only, no access-state writes (#247).
pub fn boost_anchor_matches(
    conn: &rusqlite::Connection,
    scored: &mut Vec<(crate::models::Entity, f64)>,
    anchors: &[Anchor],
    matched_ids: &mut Vec<String>,
) -> Result<usize, Box<dyn std::error::Error>> {
    boost_anchor_matches_with_encryption(conn, scored, anchors, matched_ids, None)
}

/// Protected variant of [`boost_anchor_matches`]. Encrypted stores hash each
/// anchor term before querying the blind-token FTS index; plaintext stores keep
/// the original query shape through the compatibility wrapper above.
pub fn boost_anchor_matches_with_encryption(
    conn: &rusqlite::Connection,
    scored: &mut Vec<(crate::models::Entity, f64)>,
    anchors: &[Anchor],
    matched_ids: &mut Vec<String>,
    encryption: Option<&crate::encryption::EncryptionManager>,
) -> Result<usize, Box<dyn std::error::Error>> {
    let queries = if let Some(enc) = encryption {
        anchor_query_terms(anchors, ANCHOR_QUERY_MAX_CHARS)
            .iter()
            .map(|terms| enc.blind_query_from_terms(terms))
            .collect::<Vec<_>>()
    } else {
        anchor_queries(anchors, ANCHOR_QUERY_MAX_CHARS)
    };
    if queries.is_empty() || scored.is_empty() {
        return Ok(0);
    }
    let mut boosted = 0usize;
    for (entity, score) in scored.iter_mut() {
        let mut matches = 0usize;
        for q in &queries {
            let hit: bool = conn.query_row(
                "SELECT EXISTS(SELECT 1 FROM entities_fts \
                 WHERE entities_fts MATCH ?1 AND rowid = (SELECT rowid FROM entities \
                 WHERE id = ?2))",
                params![q, entity.id],
                |r| r.get(0),
            )?;
            if hit {
                matches += 1;
            }
        }
        if matches > 0 {
            let factor = (1.0 + ANCHOR_BOOST * matches as f64).min(MAX_ANCHOR_BOOST);
            *score *= factor;
            matched_ids.push(entity.id.clone());
            boosted += 1;
        }
    }
    Ok(boosted)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_text_prefers_named_fields_and_falls_back() {
        assert_eq!(
            extract_text("{\"text\":\"warranty matters\"}"),
            "warranty matters"
        );
        assert_eq!(extract_text("{\"rule\":\"no secrets\"}"), "no secrets");
        assert_eq!(extract_text("plain body"), "plain body");
        assert!(extract_text("{\"opaque\": 1}").contains("opaque"));
    }

    #[test]
    fn anchor_queries_sanitize_and_drop_garbage() {
        let anchors = vec![
            Anchor {
                text: "Warranty MATTERS!!! for expensive buys".into(),
                weight: 1.0,
                source: "decision",
            },
            Anchor {
                text: "   ".into(),
                weight: 1.0,
                source: "decision",
            },
        ];
        let qs = anchor_queries(&anchors, 200);
        assert_eq!(qs.len(), 1);
        assert!(qs[0].contains("warranty"));
        assert!(qs[0].contains("expensive"));
        assert!(!qs[0].contains("!!!"));
    }

    #[test]
    fn boost_applies_capped_factor_and_records_matches() {
        let db = crate::db::TestDatabase::new("anchor-boost");
        let now = crate::db::now_ms();
        let mut make = |id: &str, body: &str| crate::models::Entity {
            id: id.to_string(),
            category: "facts".to_string(),
            key: id.to_string(),
            body_json: body.to_string(),
            status: "active".to_string(),
            entity_type: "fact".to_string(),
            tags: vec![],
            decay_score: 0.5,
            retrieval_count: 0,
            layer: "working".to_string(),
            topic_path: String::new(),
            archived: false,
            archive_reason: String::new(),
            links: vec![],
            verified: false,
            source: "agent".to_string(),
            always_on: false,
            certainty: 0.5,
            workspace_hash: String::new(),
            agent_id: String::new(),
            visibility: "workspace".to_string(),
            created_at_unix_ms: now,
            last_accessed_unix_ms: now,
            follow_count: 0,
            miss_count: 0,
            follow_rate: 0.0,
            efficacy_status: "unverified".to_string(),
            epistemic_state: crate::models::default_epistemic_state(),
            hints: vec![],
            embedding: None,
            memory_type: String::new(),
            _parsed_body: None,
        };
        let e1 = make("a-1", "{\"text\":\"mattress warranty coverage details\"}");
        let e2 = make("a-2", "{\"text\":\"unrelated gardening tips\"}");
        db.remember(&e1).unwrap();
        db.remember(&e2).unwrap();
        let conn = db.conn().unwrap();
        let anchors = vec![Anchor {
            text: "warranty matters for expensive purchases".into(),
            weight: 1.0,
            source: "keystone",
        }];
        let mut scored = vec![(e1.clone(), 1.0), (e2.clone(), 1.0)];
        let mut matched = Vec::new();
        let n = boost_anchor_matches(&conn, &mut scored, &anchors, &mut matched).unwrap();
        assert_eq!(n, 1);
        assert_eq!(matched, vec!["a-1".to_string()]);
        assert!(scored[0].1 > scored[1].1);
        assert!(scored[0].1 <= 1.0 + ANCHOR_BOOST + 1e-9);
    }

    #[test]
    fn load_anchors_respects_workspace_and_active_only() {
        let db = crate::db::TestDatabase::new("anchor-load");
        let conn = db.conn().unwrap();
        // Keystone global + another workspace's keystone.
        conn.execute(
            "INSERT INTO keystones (id, content, scope, scope_id, weight, \
             trust_tier_required, workspace_hash, author_agent_id, \
             created_at_unix_ms, updated_at_unix_ms) \
             VALUES ('k1','never log secrets','','',2.0,0,'','',1,1), \
                    ('k2','other-tenant rule','','',2.0,0,'ws-other','',1,1)",
            [],
        )
        .unwrap();
        let now = crate::db::now_ms();
        // Active decision (no successor) + superseded decision (has successor).
        conn.execute(
            "INSERT INTO entities (id, category, key, body_json, status, type, tags, \
             decay_score, retrieval_count, layer, topic_path, archived, archive_reason, \
             links, verified, source, always_on, certainty, created_at_unix_ms, \
             last_accessed_unix_ms, workspace_hash, agent_id, visibility, \
             recorded_at_unix_ms, valid_from_unix_ms, valid_to_unix_ms, epistemic_state, \
             expires_at_unix_ms, hints, supersedes, invalidated_at_unix_ms) \
             VALUES ('d1','decision','d1','{\"text\":\"active rule\"}','active','fact','[]', \
             0.5,0,'working','',0,'','[]',0,'agent',0,0.5,?1,?1,'ws-a','','workspace',?1,NULL,NULL,'candidate',NULL,'[]','',NULL)",
            [now],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO entities (id, category, key, body_json, status, type, tags, \
             decay_score, retrieval_count, layer, topic_path, archived, archive_reason, \
             links, verified, source, always_on, certainty, created_at_unix_ms, \
             last_accessed_unix_ms, workspace_hash, agent_id, visibility, \
             recorded_at_unix_ms, valid_from_unix_ms, valid_to_unix_ms, epistemic_state, \
             expires_at_unix_ms, hints, supersedes, invalidated_at_unix_ms) \
             VALUES ('d2','decision','d2','{\"text\":\"old rule\"}','active','fact','[]', \
             0.5,0,'working','',0,'','[]',0,'agent',0,0.5,?1,?1,'ws-a','','workspace',?1,NULL,NULL,'candidate',NULL,'[]','',NULL), \
                    ('d3','decision','d3','{\"text\":\"newer rule\"}','active','fact','[]', \
             0.5,0,'working','',0,'','[]',0,'agent',0,0.5,?1,?1,'ws-a','','workspace',?1,NULL,NULL,'candidate',NULL,'[]','d2',NULL)",
            [now],
        )
        .unwrap();
        let anchors = load_anchors(&conn, Some("ws-a"), 5).unwrap();
        // k1 (global keystone) in; k2 (other tenant) out; d1 active in;
        // d2 superseded by d3 → out; d3 in.
        let texts: Vec<&str> = anchors.iter().map(|a| a.text.as_str()).collect();
        assert!(texts.contains(&"never log secrets"));
        assert!(!texts.contains(&"other-tenant rule"));
        assert!(texts.contains(&"active rule"));
        assert!(texts.contains(&"newer rule"));
        assert!(!texts.contains(&"old rule"));
    }
}
