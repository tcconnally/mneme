//! #918: read-only inspector data layer.
//!
//! Opens a vault database STRICTLY read-only (SQLite `SQLITE_OPEN_READ_ONLY`
//! + `PRAGMA query_only` — no migrations, no writes) and exposes the
//! surfaces the TUI renders:
//!
//! * entity state (archived / quarantined / superseded / active), decay
//!   scores, categories;
//! * claim-card state (#852: entities whose body carries a claim-card
//!   revision, or tagged `contradiction`);
//! * recall-arm telemetry (`served_events`, `recall_arm_audits`,
//!   `displacement_events`, schema v31);
//! * bi-temporal history (`entity_history` — valid-time + transaction-time
//!   columns, supersession edges);
//! * quarantined writes (`write_quarantine`, #874).
//!
//! Decryption: when a key file is supplied (arg, or `PERSEUS_VAULT_KEY_FILE`
//! env — the same env the server honors) stored bodies are decrypted with
//! the current AAD scheme and the legacy fallback, exactly like server
//! reads. Without a key, plaintext rows are shown as-is and ciphertext rows
//! are flagged `(encrypted at rest)` — the raw bytes are never surfaced.

use rusqlite::Connection;

/// A single entity as the inspector renders it: stored metadata plus
/// derived state flags. `body_plaintext` carries the DECRYPTED body when a
/// key is available, the plaintext body when the row was stored unencrypted,
/// or an explicit marker for ciphertext-at-rest rows.
#[derive(Clone, Debug)]
pub struct InspectEntity {
    pub id: String,
    pub category: String,
    pub key: String,
    pub status: String,
    pub entity_type: String,
    pub tags: Vec<String>,
    pub decay_score: f64,
    pub retrieval_count: i64,
    pub layer: String,
    pub topic_path: String,
    pub archived: bool,
    pub archive_reason: String,
    pub verified: bool,
    pub source: String,
    pub created_at_unix_ms: i64,
    pub last_accessed_unix_ms: i64,
    pub always_on: bool,
    pub certainty: f64,
    pub workspace_hash: String,
    pub agent_id: String,
    pub visibility: String,
    pub follow_rate: f64,
    pub efficacy_status: String,
    pub epistemic_state: String,
    pub links_json: String,
    pub body_plaintext: String,
    pub quarantined: bool,
    pub superseded: bool,
    pub claim_card: bool,
}

#[derive(Clone, Debug, Default)]
pub struct EntityFilter {
    pub category: Option<String>,
    /// Case-insensitive substring over key OR decrypted body.
    pub text: Option<String>,
    /// "all" (default) | "active" | "archived" | "quarantined" | "superseded" | "claims"
    pub state: Option<String>,
}

#[derive(Clone, Debug)]
pub struct HistoryRow {
    pub history_id: String,
    pub recorded_at_unix_ms: Option<i64>,
    pub valid_from_unix_ms: Option<i64>,
    pub valid_to_unix_ms: Option<i64>,
    pub invalidated_at_unix_ms: Option<i64>,
    pub supersedes: String,
    pub superseded_by: String,
    pub archived: bool,
    pub body_plaintext: String,
}

#[derive(Clone, Debug)]
pub struct LinkRow {
    pub target_id: String,
    pub rel: String,
}

#[derive(Clone, Debug)]
pub struct Overview {
    pub total_entities: i64,
    pub active: i64,
    pub archived: i64,
    pub quarantined: i64,
    pub superseded: i64,
    pub claim_cards: i64,
    pub categories: Vec<(String, i64)>,
    pub served_events: i64,
    pub arm_audits: i64,
    pub displacement_events: i64,
    /// Decay buckets over ACTIVE entities: 0.0–0.2 … 0.8–1.0.
    pub decay_buckets: [(String, i64); 5],
}

#[derive(Clone, Debug)]
pub struct ServedEvent {
    pub ts_unix_ms: i64,
    pub batch_id: String,
    pub profile: String,
    pub entity_id: String,
    pub category: String,
    pub key: String,
    pub mode: String,
    pub query: String,
    pub tokens_est: i64,
    pub slot: i64,
}

#[derive(Clone, Debug)]
pub struct ArmAudit {
    pub ts_unix_ms: i64,
    pub mode: String,
    pub arm: String,
    pub candidates: i64,
    pub reentry_candidates: i64,
    pub delivered: i64,
    pub profile: String,
}

#[derive(Clone, Debug)]
pub struct DisplacementEvent {
    pub ts_unix_ms: i64,
    pub entity_id: String,
    pub reason: String,
    pub was_sole_evidence: bool,
    pub mode: String,
}

const ENCRYPTED_BODY_SENTINEL: &str = "(encrypted at rest — pass --key-file to decrypt)";

/// Render a body for the read-only inspector. Keyed reads use the database's
/// strict canonical-AAD helper; only the explicitly keyless plaintext-store
/// path may return raw JSON.
fn decrypt_body(
    enc: Option<&crate::encryption::EncryptionManager>,
    raw: &str,
    category: &str,
    key: &str,
) -> Result<String, String> {
    let Some(enc) = enc else {
        // No key: plaintext rows start with '{' (JSON); anything else is
        // ciphertext-at-rest and must not be surfaced raw.
        if raw.trim_start().starts_with('{') {
            return Ok(raw.to_string());
        }
        return Ok(ENCRYPTED_BODY_SENTINEL.to_string());
    };
    match crate::db::Database::decrypt_body_with_aad_fallback(enc, raw, category, key) {
        crate::encryption::BodyDecrypt::Plaintext(p) => Ok(p),
        crate::encryption::BodyDecrypt::LegacyPlaintext(_) => Err(format!(
            "plaintext body found in encrypted store for {category}:{key}"
        )),
        crate::encryption::BodyDecrypt::AuthFailed(_) => Err(format!(
            "decryption failed for encrypted body {category}:{key}"
        )),
    }
}

pub struct Inspector {
    conn: Connection,
    enc: Option<crate::encryption::EncryptionManager>,
    protected_store: bool,
}

/// Column list matching `db.rs` entity queries, so rows map cleanly.
const ENTITY_COLUMNS: &str = "id, category, key, body_json, status, type, tags, decay_score, \
     retrieval_count, layer, topic_path, archived, archive_reason, links, verified, source, \
     created_at_unix_ms, last_accessed_unix_ms, NULL as embedding, always_on, certainty, \
     workspace_hash, agent_id, visibility, follow_count, miss_count, follow_rate, \
     efficacy_status, epistemic_state, hints, memory_type";

impl Inspector {
    /// Open a vault database read-only. `key_file` overrides
    /// `PERSEUS_VAULT_KEY_FILE`; both are optional (ciphertext rows are
    /// flagged rather than surfaced).
    pub fn open_ro(path: &str, key_file: Option<&str>) -> Result<Self, String> {
        Self::open_ro_inner(path, key_file, true)
    }

    fn open_ro_inner(
        path: &str,
        key_file: Option<&str>,
        use_env_key: bool,
    ) -> Result<Self, String> {
        let conn = Connection::open_with_flags(
            path,
            rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )
        .map_err(|e| format!("cannot open {path} read-only: {e}"))?;
        // Belt and braces: any accidental write attempt fails loudly.
        conn.pragma_update(None, "query_only", true)
            .map_err(|e| format!("read-only pragma failed: {e}"))?;
        let canary_table_exists = conn
            .query_row(
                "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='encryption_canary')",
                [],
                |row| row.get::<_, i64>(0),
            )
            .unwrap_or(1)
            != 0;
        let canary_present = if canary_table_exists {
            conn.query_row(
                "SELECT COUNT(*) FROM encryption_canary WHERE id = 1 AND length(ciphertext) > 0",
                [],
                |row| row.get::<_, i64>(0),
            )
            .map(|count| count > 0)
            .unwrap_or(true)
        } else {
            false
        };
        let profile_table_exists = conn
            .query_row(
                "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='encryption_profile')",
                [],
                |row| row.get::<_, i64>(0),
            )
            .unwrap_or(1)
            != 0;
        let profile_protected = if profile_table_exists {
            match conn.query_row(
                "SELECT search_mode FROM encryption_profile WHERE id = 1",
                [],
                |row| row.get::<_, String>(0),
            ) {
                Ok(mode) => {
                    mode == crate::encryption::BLIND_TOKEN_SEARCH_MODE
                        || mode == "migration-pending"
                }
                Err(rusqlite::Error::QueryReturnedNoRows) => false,
                Err(_) => true,
            }
        } else {
            false
        };
        let protected_store = canary_present || profile_protected;
        let key_file = key_file.map(|s| s.to_string()).or_else(|| {
            use_env_key
                .then(|| std::env::var("PERSEUS_VAULT_KEY_FILE").ok())
                .flatten()
        });
        let enc = match key_file {
            Some(kf) => Some(
                crate::encryption::EncryptionManager::from_key_file(&kf)
                    .map_err(|e| format!("key file: {e}"))?,
            ),
            None => None,
        };
        Ok(Self {
            conn,
            enc,
            protected_store,
        })
    }

    #[cfg(test)]
    fn open_ro_without_env_key(path: &str) -> Result<Self, String> {
        Self::open_ro_inner(path, None, false)
    }

    fn table_exists(&self, name: &str) -> bool {
        self.conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?1",
                [name],
                |r| r.get::<_, i64>(0),
            )
            .unwrap_or(0)
            > 0
    }

    fn count(&self, sql: &str) -> i64 {
        self.conn.query_row(sql, [], |r| r.get(0)).unwrap_or(0)
    }

    /// Derived state flags for every entity row (quarantine + supersession
    /// are table lookups; claim-card state needs the decrypted body).
    fn load_entities(&self) -> Result<Vec<InspectEntity>, String> {
        let sql = format!(
            "SELECT {ENTITY_COLUMNS} FROM entities ORDER BY last_accessed_unix_ms DESC LIMIT 2000"
        );
        let mut stmt = self
            .conn
            .prepare(&sql)
            .map_err(|e| format!("entities query failed: {e}"))?;
        let rows = stmt
            .query_map([], |row| crate::db::entity_from_row(row, self.enc.as_ref()))
            .map_err(|e| format!("entities scan failed: {e}"))?;
        let mut entities: Vec<InspectEntity> = Vec::new();
        for row in rows {
            let e = row.map_err(|e| format!("entity row: {e}"))?;
            let quarantined = self
                .conn
                .query_row(
                    "SELECT COUNT(*) FROM write_quarantine WHERE id=?1",
                    [&e.id],
                    |r| r.get::<_, i64>(0),
                )
                .unwrap_or(0)
                > 0;
            let superseded = self
                .conn
                .query_row(
                    "SELECT COUNT(*) FROM entity_history \
                     WHERE id=?1 AND (supersedes != '' OR superseded_by != '')",
                    [&e.id],
                    |r| r.get::<_, i64>(0),
                )
                .unwrap_or(0)
                > 0;
            let body = if self.enc.is_some() {
                e.body_json.clone()
            } else if self.protected_store {
                ENCRYPTED_BODY_SENTINEL.to_string()
            } else {
                decrypt_body(None, &e.body_json, &e.category, &e.key)?
            };
            let claim_card = e.tags.iter().any(|t| t == "contradiction")
                || body.contains("\"claim_card_version\"");
            entities.push(InspectEntity {
                id: e.id.clone(),
                category: e.category.clone(),
                key: e.key.clone(),
                status: e.status.clone(),
                entity_type: e.entity_type.clone(),
                tags: e.tags.clone(),
                decay_score: e.decay_score,
                retrieval_count: e.retrieval_count,
                layer: e.layer.clone(),
                topic_path: e.topic_path.clone(),
                archived: e.archived,
                archive_reason: e.archive_reason.clone(),
                verified: e.verified,
                source: e.source.clone(),
                created_at_unix_ms: e.created_at_unix_ms,
                last_accessed_unix_ms: e.last_accessed_unix_ms,
                always_on: e.always_on,
                certainty: e.certainty,
                workspace_hash: e.workspace_hash.clone(),
                agent_id: e.agent_id.clone(),
                visibility: e.visibility.clone(),
                follow_rate: e.follow_rate,
                efficacy_status: e.efficacy_status.clone(),
                epistemic_state: e.epistemic_state.clone(),
                links_json: serde_json::to_string(&e.links).unwrap_or_default(),
                body_plaintext: body,
                quarantined,
                superseded,
                claim_card,
            });
        }
        Ok(entities)
    }

    /// Overview numbers for the landing tab.
    pub fn overview(&self) -> Result<Overview, String> {
        let total = self.count("SELECT COUNT(*) FROM entities");
        let archived = self.count("SELECT COUNT(*) FROM entities WHERE archived = 1");
        let quarantined = if self.table_exists("write_quarantine") {
            self.count("SELECT COUNT(*) FROM write_quarantine")
        } else {
            0
        };
        let superseded = if self.table_exists("entity_history") {
            self.count(
                "SELECT COUNT(DISTINCT id) FROM entity_history \
                 WHERE supersedes != '' OR superseded_by != ''",
            )
        } else {
            0
        };
        let categories: Vec<(String, i64)> = {
            let mut stmt = self
                .conn
                .prepare(
                    "SELECT category, COUNT(*) AS n FROM entities \
                     GROUP BY category ORDER BY n DESC LIMIT 20",
                )
                .map_err(|e| e.to_string())?;
            let rows = stmt
                .query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?)))
                .map_err(|e| e.to_string())?;
            rows.collect::<Result<Vec<_>, _>>()
                .map_err(|e| e.to_string())?
        };
        let served_events = if self.table_exists("served_events") {
            self.count("SELECT COUNT(*) FROM served_events")
        } else {
            0
        };
        let arm_audits = if self.table_exists("recall_arm_audits") {
            self.count("SELECT COUNT(*) FROM recall_arm_audits")
        } else {
            0
        };
        let displacement_events = if self.table_exists("displacement_events") {
            self.count("SELECT COUNT(*) FROM displacement_events")
        } else {
            0
        };
        // Claim-card + decay buckets come from the decrypted scan.
        let entities = self.load_entities()?;
        let claim_cards = entities.iter().filter(|e| e.claim_card).count() as i64;
        let mut decay_buckets = [
            ("0.0–0.2".to_string(), 0_i64),
            ("0.2–0.4".to_string(), 0_i64),
            ("0.4–0.6".to_string(), 0_i64),
            ("0.6–0.8".to_string(), 0_i64),
            ("0.8–1.0".to_string(), 0_i64),
        ];
        for e in entities.iter().filter(|e| !e.archived) {
            let idx = ((e.decay_score * 5.0).floor() as usize).min(4);
            decay_buckets[idx].1 += 1;
        }
        Ok(Overview {
            total_entities: total,
            active: total - archived,
            archived,
            quarantined,
            superseded,
            claim_cards,
            categories,
            served_events,
            arm_audits,
            displacement_events,
            decay_buckets,
        })
    }

    /// Filtered entity list (state + category + text over key/body).
    pub fn entities(
        &self,
        filter: &EntityFilter,
        limit: usize,
    ) -> Result<Vec<InspectEntity>, String> {
        let all = self.load_entities()?;
        let text = filter.text.as_deref().map(|s| s.to_lowercase());
        let mut out: Vec<InspectEntity> = all
            .into_iter()
            .filter(|e| {
                if let Some(cat) = &filter.category {
                    if !cat.is_empty() && &e.category != cat {
                        return false;
                    }
                }
                match filter.state.as_deref().unwrap_or("all") {
                    "active" => {
                        if e.archived || e.quarantined || e.superseded {
                            return false;
                        }
                    }
                    "archived" => {
                        if !e.archived {
                            return false;
                        }
                    }
                    "quarantined" => {
                        if !e.quarantined {
                            return false;
                        }
                    }
                    "superseded" => {
                        if !e.superseded {
                            return false;
                        }
                    }
                    "claims" => {
                        if !e.claim_card {
                            return false;
                        }
                    }
                    _ => {}
                }
                if let Some(t) = &text {
                    if !e.key.to_lowercase().contains(t)
                        && !e.category.to_lowercase().contains(t)
                        && !e.body_plaintext.to_lowercase().contains(t)
                    {
                        return false;
                    }
                }
                true
            })
            .take(limit)
            .collect();
        out.sort_by(|a, b| {
            b.last_accessed_unix_ms
                .cmp(&a.last_accessed_unix_ms)
                .then_with(|| a.id.cmp(&b.id))
        });
        Ok(out)
    }

    /// One entity's full record: current state + bi-temporal history +
    /// link rows.
    pub fn entity_detail(
        &self,
        id: &str,
    ) -> Result<Option<(InspectEntity, Vec<HistoryRow>, Vec<LinkRow>)>, String> {
        let mut entities = self.entities(
            &EntityFilter {
                text: None,
                ..Default::default()
            },
            0,
        )?;
        // entities() caps at `limit`; fetch by exact id directly instead.
        let sql = format!("SELECT {ENTITY_COLUMNS} FROM entities WHERE id = ?1 LIMIT 1");
        let mut stmt = self.conn.prepare(&sql).map_err(|e| e.to_string())?;
        let mut rows = stmt
            .query_map([id], |row| {
                crate::db::entity_from_row(row, self.enc.as_ref())
            })
            .map_err(|e| e.to_string())?;
        let Some(e) = rows.next().transpose().map_err(|e| e.to_string())? else {
            return Ok(None);
        };
        let quarantined = self
            .conn
            .query_row(
                "SELECT COUNT(*) FROM write_quarantine WHERE id=?1",
                [id],
                |r| r.get::<_, i64>(0),
            )
            .unwrap_or(0)
            > 0;
        let superseded = self
            .conn
            .query_row(
                "SELECT COUNT(*) FROM entity_history \
                 WHERE id=?1 AND (supersedes != '' OR superseded_by != '')",
                [id],
                |r| r.get::<_, i64>(0),
            )
            .unwrap_or(0)
            > 0;
        let body = if self.enc.is_some() {
            e.body_json.clone()
        } else if self.protected_store {
            ENCRYPTED_BODY_SENTINEL.to_string()
        } else {
            decrypt_body(None, &e.body_json, &e.category, &e.key)?
        };
        let claim_card =
            e.tags.iter().any(|t| t == "contradiction") || body.contains("\"claim_card_version\"");
        let entity = InspectEntity {
            id: e.id.clone(),
            category: e.category.clone(),
            key: e.key.clone(),
            status: e.status.clone(),
            entity_type: e.entity_type.clone(),
            tags: e.tags.clone(),
            decay_score: e.decay_score,
            retrieval_count: e.retrieval_count,
            layer: e.layer.clone(),
            topic_path: e.topic_path.clone(),
            archived: e.archived,
            archive_reason: e.archive_reason.clone(),
            verified: e.verified,
            source: e.source.clone(),
            created_at_unix_ms: e.created_at_unix_ms,
            last_accessed_unix_ms: e.last_accessed_unix_ms,
            always_on: e.always_on,
            certainty: e.certainty,
            workspace_hash: e.workspace_hash.clone(),
            agent_id: e.agent_id.clone(),
            visibility: e.visibility.clone(),
            follow_rate: e.follow_rate,
            efficacy_status: e.efficacy_status.clone(),
            epistemic_state: e.epistemic_state.clone(),
            links_json: serde_json::to_string(&e.links).unwrap_or_default(),
            body_plaintext: body,
            quarantined,
            superseded,
            claim_card,
        };

        let history = if self.table_exists("entity_history") {
            let mut stmt = self
                .conn
                .prepare(
                    "SELECT history_id, recorded_at_unix_ms, valid_from_unix_ms, \
                            valid_to_unix_ms, invalidated_at_unix_ms, supersedes, \
                            superseded_by, archived, body_json \
                     FROM entity_history WHERE id = ?1 \
                     ORDER BY created_at_unix_ms ASC, history_id ASC",
                )
                .map_err(|e| e.to_string())?;
            let rows = stmt
                .query_map([id], |r| {
                    let raw_body: String = r.get(8)?;
                    let body_plaintext = if self.enc.is_none() && self.protected_store {
                        ENCRYPTED_BODY_SENTINEL.to_string()
                    } else {
                        decrypt_body(self.enc.as_ref(), &raw_body, &entity.category, &entity.key)
                            .map_err(|error| {
                                rusqlite::Error::FromSqlConversionFailure(
                                    8,
                                    rusqlite::types::Type::Text,
                                    Box::new(std::io::Error::other(error)),
                                )
                            })?
                    };
                    Ok(HistoryRow {
                        history_id: r.get(0)?,
                        recorded_at_unix_ms: r.get(1)?,
                        valid_from_unix_ms: r.get(2)?,
                        valid_to_unix_ms: r.get(3)?,
                        invalidated_at_unix_ms: r.get(4)?,
                        supersedes: r.get(5)?,
                        superseded_by: r.get(6)?,
                        archived: r.get::<_, i64>(7)? != 0,
                        body_plaintext,
                    })
                })
                .map_err(|e| e.to_string())?;
            rows.collect::<Result<Vec<_>, _>>()
                .map_err(|e| e.to_string())?
        } else {
            Vec::new()
        };

        let links: Vec<LinkRow> = e
            .links
            .iter()
            .map(|l| LinkRow {
                target_id: l.target_id.clone(),
                rel: l.relationship.clone(),
            })
            .collect();

        Ok(Some((entity, history, links)))
    }

    pub fn recent_served(&self, limit: usize) -> Result<Vec<ServedEvent>, String> {
        if !self.table_exists("served_events") {
            return Ok(Vec::new());
        }
        let mut stmt = self
            .conn
            .prepare(
                "SELECT ts_unix_ms, batch_id, profile, entity_id, category, key, \
                        mode, query, tokens_est, slot \
                 FROM served_events ORDER BY ts_unix_ms DESC LIMIT ?1",
            )
            .map_err(|e| e.to_string())?;
        let rows = stmt
            .query_map([limit as i64], |r| {
                Ok(ServedEvent {
                    ts_unix_ms: r.get(0)?,
                    batch_id: r.get(1)?,
                    profile: r.get(2)?,
                    entity_id: r.get(3)?,
                    category: r.get(4)?,
                    key: r.get(5)?,
                    mode: r.get(6)?,
                    query: r.get(7)?,
                    tokens_est: r.get(8)?,
                    slot: r.get(9)?,
                })
            })
            .map_err(|e| e.to_string())?;
        rows.collect::<Result<Vec<_>, _>>()
            .map_err(|e| e.to_string())
    }

    pub fn recent_arm_audits(&self, limit: usize) -> Result<Vec<ArmAudit>, String> {
        if !self.table_exists("recall_arm_audits") {
            return Ok(Vec::new());
        }
        let mut stmt = self
            .conn
            .prepare(
                "SELECT ts_unix_ms, mode, arm, candidates, reentry_candidates, \
                        delivered, profile \
                 FROM recall_arm_audits ORDER BY ts_unix_ms DESC LIMIT ?1",
            )
            .map_err(|e| e.to_string())?;
        let rows = stmt
            .query_map([limit as i64], |r| {
                Ok(ArmAudit {
                    ts_unix_ms: r.get(0)?,
                    mode: r.get(1)?,
                    arm: r.get(2)?,
                    candidates: r.get(3)?,
                    reentry_candidates: r.get(4)?,
                    delivered: r.get(5)?,
                    profile: r.get(6)?,
                })
            })
            .map_err(|e| e.to_string())?;
        rows.collect::<Result<Vec<_>, _>>()
            .map_err(|e| e.to_string())
    }

    pub fn recent_displacements(&self, limit: usize) -> Result<Vec<DisplacementEvent>, String> {
        if !self.table_exists("displacement_events") {
            return Ok(Vec::new());
        }
        let mut stmt = self
            .conn
            .prepare(
                "SELECT ts_unix_ms, entity_id, reason, was_sole_evidence, mode \
                 FROM displacement_events ORDER BY ts_unix_ms DESC LIMIT ?1",
            )
            .map_err(|e| e.to_string())?;
        let rows = stmt
            .query_map([limit as i64], |r| {
                Ok(DisplacementEvent {
                    ts_unix_ms: r.get(0)?,
                    entity_id: r.get(1)?,
                    reason: r.get(2)?,
                    was_sole_evidence: r.get::<_, i64>(3)? != 0,
                    mode: r.get(4)?,
                })
            })
            .map_err(|e| e.to_string())?;
        rows.collect::<Result<Vec<_>, _>>()
            .map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::TestDatabase;
    use std::io::Write;

    /// Mirrors db.rs tests' make_entity (that helper is module-private).
    fn mk(id: &str, category: &str, key: &str, body: &str) -> crate::models::Entity {
        crate::models::Entity {
            id: id.to_string(),
            category: category.to_string(),
            key: key.to_string(),
            body_json: body.to_string(),
            status: "active".to_string(),
            entity_type: "insight".to_string(),
            tags: vec![],
            decay_score: 1.0,
            retrieval_count: 0,
            layer: "working".to_string(),
            topic_path: String::new(),
            archived: false,
            archive_reason: String::new(),
            links: vec![],
            verified: false,
            source: "test".to_string(),
            always_on: false,
            certainty: 0.5,
            workspace_hash: String::new(),
            agent_id: String::new(),
            visibility: "workspace".to_string(),
            created_at_unix_ms: crate::db::now_ms(),
            last_accessed_unix_ms: crate::db::now_ms(),
            follow_count: 0,
            miss_count: 0,
            follow_rate: 0.0,
            efficacy_status: "unverified".to_string(),
            epistemic_state: crate::models::default_epistemic_state(),
            hints: vec![],
            memory_type: String::new(),
            embedding: None,
            _parsed_body: None,
        }
    }

    fn write_key_file(path: &str) {
        let key = crate::encryption::EncryptionManager::generate_key();
        let mut f = std::fs::File::create(path).unwrap();
        f.write_all(key.as_bytes()).unwrap();
    }

    #[test]
    fn open_ro_rejects_writes() {
        let db = TestDatabase::new("inspect-ro-test");
        let path = db.path().to_string();
        let insp = Inspector::open_ro_without_env_key(&path).unwrap();
        let err = insp
            .conn
            .execute("INSERT INTO entities (id) VALUES ('x')", [])
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("readonly") || err.contains("query_only"),
            "write must fail loudly on the read-only inspector: {err}"
        );
        drop(insp);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn overview_and_entities_reflect_state() {
        let db = TestDatabase::new("inspect-overview-test");
        let path = db.path().to_string();
        let now = crate::db::now_ms();

        // Active entity.
        let mut e1 = mk("i-1", "alpha", "k1", r#"{"note":"first"}"#);
        e1.decay_score = 0.9;
        db.remember(&e1).unwrap();
        // Archived entity.
        let mut e2 = mk("i-2", "beta", "k2", r#"{"note":"second"}"#);
        e2.decay_score = 0.1;
        db.remember(&e2).unwrap();
        {
            let conn = db.conn().unwrap();
            conn.execute(
                "UPDATE entities SET archived = 1, archive_reason = 'stale' WHERE id = 'i-2'",
                [],
            )
            .unwrap();
        }
        // Superseded entity (history edge).
        let mut e3 = mk("i-3", "beta", "k3", r#"{"note":"third"}"#);
        e3.decay_score = 0.5;
        db.remember(&e3).unwrap();
        {
            let conn = db.conn().unwrap();
            conn.execute(
                "INSERT INTO entity_history \
                 (history_id, id, category, key, body_json, created_at_unix_ms, \
                  last_accessed_unix_ms, superseded_by) \
                 VALUES ('h-1', 'i-3', 'beta', 'k3', '{}', ?1, ?1, 'i-9')",
                [now],
            )
            .unwrap();
        }
        // Quarantined write.
        {
            let conn = db.conn().unwrap();
            conn.execute(
                "INSERT INTO write_quarantine \
                 (id, category, key, body_json, interference_score, \
                  interference_json, created_at_unix_ms) \
                 VALUES ('q-1', 'gamma', 'kq', '{}', 0.9, '{\"source\":\"test\"}', ?1)",
                [now],
            )
            .unwrap();
        }
        // Served events + arm audit + displacement.
        {
            let conn = db.conn().unwrap();
            conn.execute(
                "INSERT INTO served_events \
                 (id, ts_unix_ms, batch_id, entity_id, query) \
                 VALUES ('s-1', ?1, 'b-1', 'i-1', 'first')",
                [now],
            )
            .unwrap();
            conn.execute(
                "INSERT INTO recall_arm_audits \
                 (id, ts_unix_ms, mode, arm, candidates, reentry_candidates, delivered) \
                 VALUES ('a-1', ?1, 'fused', 'fts5', 12, 0, 3)",
                [now],
            )
            .unwrap();
            conn.execute(
                "INSERT INTO displacement_events \
                 (id, ts_unix_ms, entity_id, reason) \
                 VALUES ('d-1', ?1, 'i-2', 'diversity')",
                [now],
            )
            .unwrap();
        }
        drop(db);

        let insp = Inspector::open_ro_without_env_key(&path).unwrap();
        let ov = insp.overview().unwrap();
        assert_eq!(ov.total_entities, 3);
        assert_eq!(ov.archived, 1);
        assert_eq!(ov.active, 2);
        assert_eq!(ov.quarantined, 1);
        assert_eq!(ov.superseded, 1);
        assert_eq!(ov.served_events, 1);
        assert_eq!(ov.arm_audits, 1);
        assert_eq!(ov.displacement_events, 1);
        assert_eq!(
            ov.categories
                .iter()
                .find(|(c, _)| c == "beta")
                .map(|(_, n)| *n),
            Some(2)
        );
        // Decay buckets: e1 0.9 → 0.8–1.0, e3 0.5 → 0.4–0.6, e2 archived (excluded).
        assert_eq!(ov.decay_buckets[4].1, 1);
        assert_eq!(ov.decay_buckets[2].1, 1);

        let all = insp.entities(&EntityFilter::default(), 100).unwrap();
        assert_eq!(all.len(), 3);
        let e1 = all.iter().find(|e| e.id == "i-1").unwrap();
        assert!(!e1.archived && !e1.quarantined && !e1.superseded);
        let e2 = all.iter().find(|e| e.id == "i-2").unwrap();
        assert!(e2.archived);
        let e3 = all.iter().find(|e| e.id == "i-3").unwrap();
        assert!(e3.superseded);
        assert_eq!(e3.body_plaintext, r#"{"note":"third"}"#);

        // State filters.
        let archived = insp
            .entities(
                &EntityFilter {
                    state: Some("archived".into()),
                    ..Default::default()
                },
                100,
            )
            .unwrap();
        assert_eq!(archived.len(), 1);
        assert_eq!(archived[0].id, "i-2");
        let quarantined = insp
            .entities(
                &EntityFilter {
                    state: Some("quarantined".into()),
                    ..Default::default()
                },
                100,
            )
            .unwrap();
        assert!(
            quarantined.is_empty(),
            "quarantine table rows are not live entities"
        );
        let superseded = insp
            .entities(
                &EntityFilter {
                    state: Some("superseded".into()),
                    ..Default::default()
                },
                100,
            )
            .unwrap();
        assert_eq!(superseded.len(), 1);

        // Category + text filters.
        let beta = insp
            .entities(
                &EntityFilter {
                    category: Some("beta".into()),
                    ..Default::default()
                },
                100,
            )
            .unwrap();
        assert_eq!(beta.len(), 2);
        let text = insp
            .entities(
                &EntityFilter {
                    text: Some("first".into()),
                    ..Default::default()
                },
                100,
            )
            .unwrap();
        assert_eq!(text.len(), 1);
        assert_eq!(text[0].id, "i-1");

        // Detail: history + links.
        let (detail, history, links) = insp.entity_detail("i-3").unwrap().unwrap();
        assert!(detail.superseded);
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].superseded_by, "i-9");
        assert!(links.is_empty());

        // Telemetry rows.
        let served = insp.recent_served(10).unwrap();
        assert_eq!(served.len(), 1);
        assert_eq!(served[0].query, "first");
        let audits = insp.recent_arm_audits(10).unwrap();
        assert_eq!(audits.len(), 1);
        assert_eq!(audits[0].arm, "fts5");
        let displ = insp.recent_displacements(10).unwrap();
        assert_eq!(displ.len(), 1);
        assert_eq!(displ[0].reason, "diversity");

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn encrypted_bodies_require_key_and_decrypt_with_aad_scheme() {
        let db = TestDatabase::new("inspect-enc-test");
        let path = db.path().to_string();
        let key_path = format!("{path}.key");
        write_key_file(&key_path);
        let key = std::fs::read_to_string(&key_path).unwrap();
        let enc = crate::encryption::EncryptionManager::from_key_file(&key_path).unwrap();
        let _ = key;
        // Simulate an encrypted-at-rest row exactly like db.rs writes it
        // (AAD = len(category):category:key).
        let plain = r#"{"note":"classified memory"}"#;
        let aad = format!("{}:{}:{}", "enc".len(), "enc", "k1");
        let cipher = enc.encrypt(plain, aad.as_bytes()).unwrap();
        {
            let conn = db.conn().unwrap();
            conn.execute(
                "INSERT INTO entities (id, category, key, body_json, created_at_unix_ms, \
                 last_accessed_unix_ms, type, status, tags, layer, topic_path, links, \
                 source, always_on, certainty, workspace_hash, agent_id, visibility, \
                 decay_score, retrieval_count, archived, archive_reason, verified, \
                 follow_count, miss_count, follow_rate, efficacy_status, epistemic_state, hints, memory_type) \
                 VALUES ('i-9', 'enc', 'k1', ?1, 1, 1, 'insight', 'active', '[]', \
                 'working', '', '[]', 'agent', 0, 0.5, '', '', 'workspace', 1.0, 0, 0, '', \
                 0, 0, 0, 0.0, 'unverified', 'candidate', '[]', '')",
                [&cipher],
            )
            .unwrap();
        }
        drop(db);

        // Without a key the body is flagged, never surfaced.
        let insp = Inspector::open_ro_without_env_key(&path).unwrap();
        let all = insp.entities(&EntityFilter::default(), 10).unwrap();
        assert_eq!(all.len(), 1);
        assert!(all[0].body_plaintext.contains("encrypted at rest"));
        assert!(!all[0].body_plaintext.contains("classified"));

        // With the key the body decrypts through the current AAD scheme.
        let insp = Inspector::open_ro(&path, Some(&key_path)).unwrap();
        let all = insp.entities(&EntityFilter::default(), 10).unwrap();
        assert_eq!(all[0].body_plaintext, plain);

        let _ = std::fs::remove_file(path);
        let _ = std::fs::remove_file(&key_path);
    }

    #[test]
    fn encrypted_history_detail_decrypts_current_aad() {
        let db = TestDatabase::new("inspect-history-enc-test");
        let path = db.path().to_string();
        let key_path = format!("{path}.key");
        write_key_file(&key_path);
        let enc = crate::encryption::EncryptionManager::from_key_file(&key_path).unwrap();
        let body = r#"{"note":"current encrypted history"}"#;
        let ciphertext = enc
            .encrypt(body, crate::db::Database::build_aad("enc", "k1").as_bytes())
            .unwrap();
        {
            let conn = db.conn().unwrap();
            conn.execute(
                "INSERT INTO entities (id, category, key, body_json, created_at_unix_ms, \
                 last_accessed_unix_ms, type, status, tags, layer, topic_path, links, \
                 source, always_on, certainty, workspace_hash, agent_id, visibility, \
                 decay_score, retrieval_count, archived, archive_reason, verified, \
                 follow_count, miss_count, follow_rate, efficacy_status, epistemic_state, hints, memory_type) \
                 VALUES ('i-history', 'enc', 'k1', ?1, 1, 1, 'insight', 'active', '[]', \
                 'working', '', '[]', 'agent', 0, 0.5, '', '', 'workspace', 1.0, 0, 0, '', \
                 0, 0, 0, 0.0, 'unverified', 'candidate', '[]', '')",
                [&ciphertext],
            )
            .unwrap();
            conn.execute(
                "INSERT INTO entity_history \
                 (history_id, id, category, key, body_json, created_at_unix_ms, \
                  last_accessed_unix_ms, superseded_by) \
                 VALUES ('h-history', 'i-history', 'enc', 'k1', ?1, 1, 1, '')",
                [&ciphertext],
            )
            .unwrap();
        }
        drop(db);

        let insp = Inspector::open_ro(&path, Some(&key_path)).unwrap();
        let (_, history, _) = insp.entity_detail("i-history").unwrap().unwrap();
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].body_plaintext, body);

        let _ = std::fs::remove_file(path);
        let _ = std::fs::remove_file(&key_path);
    }

    #[test]
    fn keyless_inspector_hides_plaintext_rows_in_mixed_encrypted_store() {
        let mut db = TestDatabase::new("inspect-mixed-enc-test");
        let path = db.path().to_string();
        let key_path = format!("{path}.key");
        write_key_file(&key_path);
        db.set_encryption(&key_path).unwrap();
        let sentinel = "INSPECT_MIXED_PLAINTEXT_SENTINEL";
        {
            let conn = db.conn().unwrap();
            conn.execute(
                "INSERT INTO entities (id, category, key, body_json, created_at_unix_ms, \
                 last_accessed_unix_ms, type, status, tags, layer, topic_path, links, \
                 source, always_on, certainty, workspace_hash, agent_id, visibility, \
                 decay_score, retrieval_count, archived, archive_reason, verified, \
                 follow_count, miss_count, follow_rate, efficacy_status, epistemic_state, hints, memory_type) \
                 VALUES ('i-mixed', 'enc', 'mixed', ?1, 1, 1, 'insight', 'active', '[]', \
                 'working', '', '[]', 'agent', 0, 0.5, '', '', 'workspace', 1.0, 0, 0, '', \
                 0, 0, 0, 0.0, 'unverified', 'candidate', '[]', '')",
                [format!("{{\"note\":\"{sentinel}\"}}")],
            )
            .unwrap();
        }
        drop(db);

        let insp = Inspector::open_ro_without_env_key(&path).unwrap();
        let all = insp.entities(&EntityFilter::default(), 10).unwrap();
        let mixed = all.iter().find(|entity| entity.id == "i-mixed").unwrap();
        assert!(!mixed.body_plaintext.contains(sentinel));
        assert!(mixed.body_plaintext.contains("encrypted at rest"));

        let _ = std::fs::remove_file(path);
        let _ = std::fs::remove_file(&key_path);
    }

    #[test]
    fn claim_card_state_is_flagged() {
        let db = TestDatabase::new("inspect-claims-test");
        let path = db.path().to_string();
        let mut c1 = mk(
            "c-1",
            "claims",
            "card-1",
            r#"{"note":"a card","claim_card_version":1}"#,
        );
        c1.tags = vec!["claim_card".to_string()];
        db.remember(&c1).unwrap();
        let mut c2 = mk("c-2", "claims", "card-2", r#"{"note":"contradicted card"}"#);
        c2.tags = vec!["contradiction".to_string()];
        db.remember(&c2).unwrap();
        drop(db);

        let insp = Inspector::open_ro_without_env_key(&path).unwrap();
        let claims = insp
            .entities(
                &EntityFilter {
                    state: Some("claims".into()),
                    ..Default::default()
                },
                10,
            )
            .unwrap();
        assert_eq!(
            claims.len(),
            2,
            "claim_card_version body + contradiction tag both count"
        );
        let ov = insp.overview().unwrap();
        assert_eq!(ov.claim_cards, 2);
        let _ = std::fs::remove_file(path);
    }
}
