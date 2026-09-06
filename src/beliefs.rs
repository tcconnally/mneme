//! beliefs.rs — first-class belief entities as a derived overlay (#717).
//!
//! Spec: docs/specs/memory-taxonomy-and-precedence.md §5. A belief is NOT a
//! new storage entity: it is a claim-level view derived at request time from
//! the existing store — claim, scope, confidence, support count, evidence
//! links, and supersession state. Derivation rules:
//!
//! - one belief per live entity (archived rows never derive)
//! - supporters are OTHER live entities linked to it with an evidence
//!   relationship (evidence_for / derived_from / promoted_to); merged
//!   near-duplicates count once because dedup folds them into one row
//! - confidence = certainty scaled by verification state
//! - superseded = status 'deprecated' (perseus_vault_supersede's marker)
//!
//! Retrieval prefers beliefs by scope, confidence, freshness, and support
//! count, and every returned belief explains that preference.

use crate::db::Database;
use serde::Serialize;
use serde_json::{json, Value};

/// Link relationships counted as independent support for a claim.
const EVIDENCE_RELS: [&str; 3] = ["evidence_for", "derived_from", "promoted_to"];

/// Confidence multiplier applied to unverified entities (verified = 1.0).
const UNVERIFIED_CONFIDENCE_FACTOR: f64 = 0.8;

#[derive(Debug, Clone, Serialize)]
pub struct Belief {
    /// The backing entity this belief is derived from.
    pub entity_id: String,
    pub claim: String,
    pub category: String,
    pub key: String,
    /// Workspace scope ("" = global).
    pub scope: String,
    pub confidence: f64,
    pub support_count: i64,
    pub supporting_entity_ids: Vec<String>,
    pub last_revalidated_at_unix_ms: i64,
    pub superseded: bool,
    /// Why this belief ranked where it did (compact, prompt-safe).
    pub explanation: String,
}

#[derive(Debug, serde::Deserialize)]
pub struct BeliefsArgs {
    #[serde(default)]
    pub query: Option<String>,
    #[serde(default)]
    pub category: Option<String>,
    #[serde(default)]
    pub workspace_hash: Option<String>,
    #[serde(default = "default_beliefs_limit")]
    pub limit: i64,
    #[serde(default)]
    pub min_confidence: f64,
    #[serde(default)]
    pub include_superseded: bool,
}

fn default_beliefs_limit() -> i64 {
    10
}

/// Extract a human-readable claim from an entity body: summary, then
/// content, then claim, falling back to the key. Truncated for prompt use.
fn extract_claim(body: &Value, key: &str) -> String {
    for field in ["summary", "content", "claim"] {
        if let Some(s) = body.get(field).and_then(|v| v.as_str()) {
            if !s.trim().is_empty() {
                let trimmed = s.trim();
                return if trimmed.chars().count() > 240 {
                    trimmed.chars().take(240).collect::<String>() + "…"
                } else {
                    trimmed.to_string()
                };
            }
        }
    }
    key.to_string()
}

/// Derive beliefs for all live entities (optionally one category).
/// Returns (beliefs, entities_scanned).
pub fn derive_beliefs(db: &Database, category: Option<&str>) -> Result<Vec<Belief>, String> {
    derive_beliefs_scoped(db, category, None)
}

fn derive_beliefs_scoped(
    db: &Database,
    category: Option<&str>,
    workspace_hash: Option<&str>,
) -> Result<Vec<Belief>, String> {
    let conn = db
        .conn()
        .map_err(|e| format!("db connection failed: {}", e))?;
    let scope_clause = if workspace_hash.is_some() {
        " AND (workspace_hash = ?2 OR workspace_hash = '')"
    } else {
        ""
    };
    let sql = match category {
        Some(_) => format!(
            "SELECT id, category, key, body_json, certainty, verified, status, \
                    workspace_hash, last_accessed_unix_ms, links \
             FROM entities WHERE archived = 0 AND status IN ('active','draft') AND category = ?1{}",
            scope_clause
        ),
        None => {
            let scope_clause = if workspace_hash.is_some() {
                " AND (workspace_hash = ?1 OR workspace_hash = '')"
            } else {
                ""
            };
            format!(
                "SELECT id, category, key, body_json, certainty, verified, status, \
                        workspace_hash, last_accessed_unix_ms, links \
                 FROM entities WHERE archived = 0 AND status IN ('active','draft'){}",
                scope_clause
            )
        }
    };
    let mut stmt = conn
        .prepare(&sql)
        .map_err(|e| format!("belief scan prepare failed: {}", e))?;
    let map_row = |r: &rusqlite::Row| {
        Ok((
            r.get::<_, String>(0)?,
            r.get::<_, String>(1)?,
            r.get::<_, String>(2)?,
            r.get::<_, String>(3)?,
            r.get::<_, f64>(4).unwrap_or(0.5),
            r.get::<_, bool>(5).unwrap_or(false),
            r.get::<_, String>(6)
                .unwrap_or_else(|_| "active".to_string()),
            r.get::<_, String>(7).unwrap_or_default(),
            r.get::<_, i64>(8).unwrap_or(0),
            r.get::<_, String>(9).unwrap_or_else(|_| "[]".to_string()),
        ))
    };
    let raw_rows: Vec<_> = match (category, workspace_hash) {
        (Some(c), Some(ws)) => {
            let rows = stmt
                .query_map(rusqlite::params![c, ws], map_row)
                .map_err(|e| format!("belief scan failed: {}", e))?;
            rows.collect::<Result<Vec<_>, _>>()
                .map_err(|e| format!("belief scan row failed: {}", e))?
        }
        (Some(c), None) => {
            let rows = stmt
                .query_map(rusqlite::params![c], map_row)
                .map_err(|e| format!("belief scan failed: {}", e))?;
            rows.collect::<Result<Vec<_>, _>>()
                .map_err(|e| format!("belief scan row failed: {}", e))?
        }
        (None, Some(ws)) => {
            let rows = stmt
                .query_map(rusqlite::params![ws], map_row)
                .map_err(|e| format!("belief scan failed: {}", e))?;
            rows.collect::<Result<Vec<_>, _>>()
                .map_err(|e| format!("belief scan row failed: {}", e))?
        }
        (None, None) => {
            let rows = stmt
                .query_map(rusqlite::params![], map_row)
                .map_err(|e| format!("belief scan failed: {}", e))?;
            rows.collect::<Result<Vec<_>, _>>()
                .map_err(|e| format!("belief scan row failed: {}", e))?
        }
    };
    drop(stmt);
    let rows: Vec<_> = raw_rows
        .into_iter()
        .map(
            |(id, cat, key, raw_body, certainty, verified, status, ws, touched, links)| {
                let body_json = db
                    .decrypt_body_for_read(&raw_body, &cat, &key)
                    .map_err(|e| format!("belief body hydration failed: {e}"))?;
                Ok((
                    id, cat, key, body_json, certainty, verified, status, ws, touched, links,
                ))
            },
        )
        .collect::<Result<Vec<_>, String>>()?;

    // Reverse evidence map: target entity id -> ids of supporter entities.
    let mut supporters: std::collections::HashMap<String, Vec<String>> =
        std::collections::HashMap::new();
    for (id, _, _, _, _, _, _, _, _, links_json) in &rows {
        let links: Vec<crate::models::MemoryLink> =
            serde_json::from_str(links_json).unwrap_or_default();
        for link in links {
            if EVIDENCE_RELS.contains(&link.relationship.as_str())
                && !link.target_id.is_empty()
                && link.target_id != *id
            {
                supporters
                    .entry(link.target_id)
                    .or_default()
                    .push(id.clone());
            }
        }
    }

    let mut beliefs = Vec::with_capacity(rows.len());
    for (id, cat, key, body_json, certainty, verified, status, ws, touched, _) in rows {
        let body: Value = serde_json::from_str(&body_json).unwrap_or_else(|_| json!({}));
        let claim = extract_claim(&body, &key);
        let mut supporter_ids = supporters.get(&id).cloned().unwrap_or_default();
        supporter_ids.sort();
        supporter_ids.dedup();
        let support_count = 1 + supporter_ids.len() as i64; // self + supporters
        let confidence = certainty
            * if verified {
                1.0
            } else {
                UNVERIFIED_CONFIDENCE_FACTOR
            };
        let superseded = status == "deprecated";
        beliefs.push(Belief {
            entity_id: id,
            claim,
            category: cat,
            key,
            scope: ws,
            confidence: (confidence * 100.0).round() / 100.0,
            support_count,
            supporting_entity_ids: supporter_ids,
            last_revalidated_at_unix_ms: touched,
            superseded,
            explanation: String::new(), // filled in by the ranking layer
        });
    }
    Ok(beliefs)
}

pub fn handle_beliefs(db: &Database, args: Value) -> Result<String, String> {
    let a: BeliefsArgs =
        serde_json::from_value(args).map_err(|e| format!("Invalid beliefs arguments: {}", e))?;
    let limit = a.limit.clamp(1, 100) as usize;
    if !(0.0..=1.0).contains(&a.min_confidence) {
        return Err(format!(
            "min_confidence must be between 0.0 and 1.0, got {}",
            a.min_confidence
        ));
    }
    let ws_filter = a.workspace_hash.as_deref().filter(|s| !s.is_empty());

    let mut beliefs = derive_beliefs_scoped(db, a.category.as_deref(), ws_filter)?;

    // Filters: query substring, confidence floor, supersession, scope
    // visibility (a scoped query sees its own workspace plus global).
    let q = a.query.as_deref().map(|s| s.to_lowercase());
    beliefs.retain(|b| {
        if let Some(ref q) = q {
            let hay = format!("{} {} {}", b.claim, b.category, b.key).to_lowercase();
            if !q.split_whitespace().any(|t| hay.contains(t)) {
                return false;
            }
        }
        if b.confidence < a.min_confidence {
            return false;
        }
        if b.superseded && !a.include_superseded {
            return false;
        }
        if let Some(ws) = ws_filter {
            if b.scope != ws && !b.scope.is_empty() {
                return false;
            }
        }
        true
    });

    // Rank: exact scope match first, then confidence, then independent
    // support, then freshness — the belief-level precedence of spec §5.
    beliefs.sort_by(|x, y| {
        let scope_rank = |b: &Belief| match ws_filter {
            Some(ws) if b.scope == ws => 2,
            _ if b.scope.is_empty() => 1,
            _ => 0,
        };
        scope_rank(y)
            .cmp(&scope_rank(x))
            .then_with(|| {
                y.confidence
                    .partial_cmp(&x.confidence)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| y.support_count.cmp(&x.support_count))
            .then_with(|| {
                y.last_revalidated_at_unix_ms
                    .cmp(&x.last_revalidated_at_unix_ms)
            })
    });

    let total = beliefs.len();
    beliefs.truncate(limit);
    for b in beliefs.iter_mut() {
        let scope_label = match ws_filter {
            Some(ws) if b.scope == ws => "workspace-exact",
            _ if b.scope.is_empty() => "global",
            _ => "workspace-other",
        };
        b.explanation = format!(
            "scope:{}; confidence:{:.2} (certainty × verification); support:{} ({} linked + self); superseded:{}; last_revalidated_ms:{}",
            scope_label,
            b.confidence,
            b.support_count,
            b.supporting_entity_ids.len(),
            b.superseded,
            b.last_revalidated_at_unix_ms
        );
    }

    serde_json::to_string(&json!({
        "beliefs": beliefs,
        "total": total,
        "derivation": "belief-overlay-v1",
    }))
    .map_err(|e| format!("Serialization failed: {}", e))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_db() -> Database {
        let path =
            std::env::temp_dir().join(format!("perseus_vault-beliefs-{}.db", uuid::Uuid::new_v4()));
        Database::open(path.to_str().expect("temp db path")).expect("open temp db")
    }

    fn remember(db: &Database, category: &str, key: &str, content: &str) -> String {
        let out = crate::tools::handle_remember(
            db,
            json!({
                "category": category,
                "key": key,
                "body_json": json!({"content": content}).to_string(),
                "skip_dedup": true,
            }),
        )
        .expect("remember");
        let v: Value = serde_json::from_str(&out).expect("remember json");
        v["id"].as_str().expect("entity id").to_string()
    }

    fn remember_in_workspace(
        db: &Database,
        category: &str,
        key: &str,
        content: &str,
        workspace_hash: &str,
    ) -> String {
        let out = crate::tools::handle_remember(
            db,
            json!({
                "category": category,
                "key": key,
                "body_json": json!({"content": content}).to_string(),
                "workspace_hash": workspace_hash,
                "skip_dedup": true,
            }),
        )
        .expect("remember");
        let v: Value = serde_json::from_str(&out).expect("remember json");
        v["id"].as_str().expect("entity id").to_string()
    }

    #[test]
    fn beliefs_do_not_count_supporters_from_other_workspace() {
        let db = temp_db();
        let target = remember_in_workspace(&db, "fact", "target", "shared claim", "workspace-a");
        let _same = remember_in_workspace(
            &db,
            "fact",
            "support-a",
            "workspace-a support",
            "workspace-a",
        );
        let _other = remember_in_workspace(
            &db,
            "fact",
            "support-b",
            "workspace-b support",
            "workspace-b",
        );
        db.link("fact", "support-a", &target, "evidence_for")
            .expect("same-workspace link");
        db.link("fact", "support-b", &target, "evidence_for")
            .expect("cross-workspace link");

        let out = handle_beliefs(
            &db,
            json!({"category": "fact", "workspace_hash": "workspace-a"}),
        )
        .expect("scoped beliefs");
        let value: Value = serde_json::from_str(&out).unwrap();
        let target_belief = value["beliefs"]
            .as_array()
            .unwrap()
            .iter()
            .find(|belief| belief["key"] == "target")
            .expect("target belief");
        assert_eq!(target_belief["support_count"], 2);
        assert_eq!(
            target_belief["supporting_entity_ids"]
                .as_array()
                .unwrap()
                .len(),
            1
        );
    }

    #[test]
    fn belief_derives_with_support_from_evidence_link() {
        let db = temp_db();
        let a = remember(&db, "fact", "alpha", "claim alpha");
        let _b = remember(&db, "fact", "beta", "supporting observation");
        db.link("fact", "beta", &a, "evidence_for").expect("link");

        let out = handle_beliefs(&db, json!({})).expect("beliefs");
        let v: Value = serde_json::from_str(&out).unwrap();
        let beliefs = v["beliefs"].as_array().unwrap();
        let alpha = beliefs
            .iter()
            .find(|b| b["key"] == "alpha")
            .expect("alpha belief");
        assert_eq!(alpha["support_count"], 2, "self + one linked supporter");
        assert_eq!(alpha["supporting_entity_ids"].as_array().unwrap().len(), 1);
        assert_eq!(alpha["superseded"], false);
        assert!(!alpha["explanation"].as_str().unwrap().is_empty());
    }

    #[test]
    fn superseded_excluded_from_public_reader_even_when_opted_in() {
        let db = temp_db();
        let a = remember(&db, "fact", "old-claim", "stale claim");
        let _b = remember(&db, "fact", "new-claim", "fresh claim");
        let out = crate::tools::handle_supersede(
            &db,
            json!({"from_category": "fact", "from_key": "old-claim",
                   "to_category": "fact", "to_key": "new-claim"}),
        );
        assert!(out.is_ok(), "supersede: {out:?}");

        let out = handle_beliefs(&db, json!({})).expect("beliefs");
        let v: Value = serde_json::from_str(&out).unwrap();
        assert!(!v["beliefs"]
            .as_array()
            .unwrap()
            .iter()
            .any(|b| b["entity_id"] == a));

        let out = handle_beliefs(&db, json!({"include_superseded": true})).expect("beliefs");
        let v: Value = serde_json::from_str(&out).unwrap();
        assert!(!v["beliefs"]
            .as_array()
            .unwrap()
            .iter()
            .any(|b| b["entity_id"] == a));
    }

    #[test]
    fn workspace_exact_outranks_global() {
        let db = temp_db();
        let _g = remember(&db, "fact", "global-claim", "org-wide belief");
        let out = crate::tools::handle_remember(
            &db,
            json!({
                "category": "fact",
                "key": "local-claim",
                "body_json": json!({"content": "local fact"}).to_string(),
                "workspace_hash": "ws-1",
                "skip_dedup": true,
            }),
        );
        assert!(out.is_ok(), "scoped remember: {out:?}");

        let out = handle_beliefs(&db, json!({"workspace_hash": "ws-1"})).expect("beliefs");
        let v: Value = serde_json::from_str(&out).unwrap();
        let beliefs = v["beliefs"].as_array().unwrap();
        assert_eq!(
            beliefs.len(),
            2,
            "ws-1 sees local + global, not other scopes"
        );
        assert_eq!(
            beliefs[0]["key"], "local-claim",
            "workspace-exact ranks first"
        );
        assert!(beliefs[0]["explanation"]
            .as_str()
            .unwrap()
            .contains("workspace-exact"));
    }

    #[test]
    fn query_and_confidence_filters() {
        let db = temp_db();
        let _a = remember(&db, "fact", "deploy-procedure", "how deploys work");
        let _b = remember(&db, "fact", "lunch-menu", "tacos on friday");

        let out = handle_beliefs(&db, json!({"query": "deploy"})).expect("beliefs");
        let v: Value = serde_json::from_str(&out).unwrap();
        let beliefs = v["beliefs"].as_array().unwrap();
        assert_eq!(beliefs.len(), 1);
        assert_eq!(beliefs[0]["key"], "deploy-procedure");

        let out = handle_beliefs(&db, json!({"min_confidence": 0.9})).expect("beliefs");
        let v: Value = serde_json::from_str(&out).unwrap();
        // default certainty 0.5 × unverified 0.8 = 0.4 < 0.9
        assert_eq!(v["beliefs"].as_array().unwrap().len(), 0);
    }

    #[test]
    fn deprecated_bodies_are_excluded_from_public_beliefs() {
        let db = temp_db();
        let id = remember(
            &db,
            "fact",
            "deprecated-public-claim",
            "deprecated body must not be projected",
        );
        db.update_entity_status(&id, "deprecated", "retired")
            .expect("deprecate test entity");

        let out = handle_beliefs(&db, json!({"include_superseded": true})).expect("beliefs");
        let v: Value = serde_json::from_str(&out).unwrap();
        assert!(!v["beliefs"]
            .as_array()
            .unwrap()
            .iter()
            .any(|belief| belief["key"] == "deprecated-public-claim"));
    }
}
