//! #1091: type-conditioned temporal decay — perishability + utility
//! horizon (ScrubJay-MEM, arXiv:2608.04746).
//!
//! #1000 gave each memory type a half-life multiplier (decay_multiplier)
//! and a retrieval weight. What it deliberately did NOT provide is a
//! surfacing-age CEILING: a residual-score entity that has not been
//! touched in years still competes in recall on whatever decay remains.
//!
//! ScrubJay completes the type-conditioned picture with two per-type
//! values, resolved deterministically from the stored `memory_type`:
//!
//! - **perishability**: the type's intrinsic useful lifetime in days
//!   (documentation + audit baseline; the half-life multiplier remains the
//!   decay engine — this does not double-condition the decay tick).
//! - **utility horizon**: the maximum surfacing age in days. Past the
//!   horizon an entity is excluded from recall REGARDLESS of residual
//!   decay score (still reachable via explicit as-of/history surfaces).
//!
//! Perishable types (episodic, failure, reflection) expire from the
//! serving surface quickly; durable types (preference, constraint,
//! procedural) carry an infinite horizon. The gate is query-adaptive via
//! `RecallParams.enforce_utility_horizon` (default ON) and is auditable
//! through `perseus_vault_decay_audit`.

use serde::Serialize;

/// Days encoding for "never expires" — mirrors the decay tick's
/// never-encoding so horizons and half-lives share one convention.
pub const HORIZON_NEVER_DAYS: i64 = i64::MAX / 86_400_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ScrubJayProfile {
    pub perishability_days: i64,
    pub utility_horizon_days: i64,
}

impl ScrubJayProfile {
    pub const fn new(perishability_days: i64, utility_horizon_days: i64) -> Self {
        Self {
            perishability_days,
            utility_horizon_days,
        }
    }
}

/// Deterministic type → (perishability, utility horizon) table.
/// Legacy rows (empty memory_type) resolve to the semantic profile.
pub fn profile_for(memory_type: &str) -> ScrubJayProfile {
    match memory_type.trim().to_lowercase().as_str() {
        "semantic" => ScrubJayProfile::new(90, 365),
        "episodic" => ScrubJayProfile::new(14, 30),
        "procedural" => ScrubJayProfile::new(180, HORIZON_NEVER_DAYS),
        "preference" => ScrubJayProfile::new(180, HORIZON_NEVER_DAYS),
        "constraint" => ScrubJayProfile::new(180, HORIZON_NEVER_DAYS),
        "failure" => ScrubJayProfile::new(7, 21),
        "reflection" => ScrubJayProfile::new(14, 60),
        "knowledge" => ScrubJayProfile::new(90, 365),
        // Legacy rows and unknown stored values: baseline semantic profile.
        _ => ScrubJayProfile::new(90, 365),
    }
}

/// True when the entity's age exceeds its type's utility horizon.
/// `HORIZON_NEVER_DAYS` never returns true. Epoch/negative timestamps are
/// "no age signal" (legacy rows migrated without timestamps) and are never
/// gated — no signal, no expiry.
pub fn past_utility_horizon(created_at_ms: i64, memory_type: &str, now_ms: i64) -> bool {
    if created_at_ms <= 0 {
        return false;
    }
    let horizon = profile_for(memory_type).utility_horizon_days;
    if horizon >= HORIZON_NEVER_DAYS {
        return false;
    }
    let age_days = now_ms.saturating_sub(created_at_ms) / 86_400_000;
    age_days > horizon
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_table_is_deterministic() {
        assert_eq!(profile_for("episodic"), ScrubJayProfile::new(14, 30));
        assert_eq!(
            profile_for("constraint"),
            ScrubJayProfile::new(180, HORIZON_NEVER_DAYS)
        );
        assert_eq!(
            profile_for("preference"),
            ScrubJayProfile::new(180, HORIZON_NEVER_DAYS)
        );
        assert_eq!(profile_for("failure"), ScrubJayProfile::new(7, 21));
        // Legacy + unknown fall back to the semantic profile.
        assert_eq!(profile_for(""), profile_for("semantic"));
        assert_eq!(profile_for("not-a-type"), profile_for("semantic"));
    }

    #[test]
    fn horizon_gate_boundaries() {
        let now = 1_000_000_000_000i64;
        let day = 86_400_000i64;
        // Episodic horizon = 30 days: 30 days old is within, 31 is past.
        let fresh = now - 30 * day;
        let stale = now - 31 * day;
        assert!(!past_utility_horizon(fresh, "episodic", now));
        assert!(past_utility_horizon(stale, "episodic", now));
        // Durable types never pass their horizon.
        assert!(!past_utility_horizon(now - 40_000 * day, "constraint", now));
        // Fresh entity of any type is always within.
        assert!(!past_utility_horizon(now - day, "failure", now));
        // No age signal (epoch 0 / negative): never gated.
        assert!(!past_utility_horizon(0, "episodic", now));
        assert!(!past_utility_horizon(-5, "episodic", now));
    }

    fn e2e_entity(id: &str, body: &str, memory_type: &str, age_days: i64) -> crate::models::Entity {
        let now = crate::db::now_ms();
        crate::models::Entity {
            id: id.to_string(),
            category: "facts".to_string(),
            key: id.to_string(),
            body_json: body.to_string(),
            status: "active".to_string(),
            entity_type: "fact".to_string(),
            tags: vec![],
            decay_score: 0.9,
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
            created_at_unix_ms: now - age_days * 86_400_000,
            last_accessed_unix_ms: now - age_days * 86_400_000,
            follow_count: 0,
            miss_count: 0,
            follow_rate: 0.0,
            efficacy_status: "unverified".to_string(),
            epistemic_state: "candidate".to_string(),
            hints: vec![],
            memory_type: memory_type.to_string(),
            embedding: None,
            _parsed_body: None,
        }
    }

    fn write_fixture(
        db: &crate::db::Database,
        e: &crate::models::Entity,
    ) -> Result<(String, String), Box<dyn std::error::Error>> {
        db.remember_with_write_options(
            e,
            true,
            None,
            None,
            false,
            crate::interference::WriteGateOptions {
                mode_override: Some(crate::interference::InterferenceMode::Off),
                ..Default::default()
            },
        )
    }

    fn recall_ids(db: &crate::db::Database, query: &str, enforce: bool) -> Vec<String> {
        let mut params = crate::models::RecallParams {
            query: query.to_string(),
            limit: 20,
            ..Default::default()
        };
        params.enforce_utility_horizon = enforce;
        let mut ids: Vec<String> = db
            .recall(&params)
            .expect("recall failed")
            .into_iter()
            .map(|e| e.id)
            .collect();
        ids.sort();
        ids
    }

    #[test]
    fn horizon_gate_excludes_only_past_horizon_perishable_types() {
        let db = crate::db::TestDatabase::new("scrubjay-horizon");
        write_fixture(
            &db,
            &e2e_entity(
                "v-stale-episodic",
                "{\"content\": \"scrubjaytoken alpha\"}",
                "episodic",
                40,
            ),
        )
        .unwrap();
        write_fixture(
            &db,
            &e2e_entity(
                "v-fresh-episodic",
                "{\"content\": \"scrubjaytoken beta\"}",
                "episodic",
                1,
            ),
        )
        .unwrap();
        write_fixture(
            &db,
            &e2e_entity(
                "v-old-preference",
                "{\"content\": \"scrubjaytoken gamma\"}",
                "preference",
                400,
            ),
        )
        .unwrap();
        // enforce backdated created_at for the stale episodic (remember()
        // normalizes to now)
        let conn = db.conn().unwrap();
        conn.execute(
            "UPDATE entities SET created_at_unix_ms = ?1 WHERE id = ?2",
            rusqlite::params![crate::db::now_ms() - 40 * 86_400_000, "v-stale-episodic"],
        )
        .unwrap();
        drop(conn);

        // Horizon ON: stale episodic (40d > 30d horizon) is gone; old
        // preference is durable (never expires) and fresh episodic stays.
        assert_eq!(
            recall_ids(&db, "scrubjaytoken", true),
            vec!["v-fresh-episodic", "v-old-preference"]
        );
        // Horizon OFF: everything surfaces.
        assert_eq!(
            recall_ids(&db, "scrubjaytoken", false),
            vec!["v-fresh-episodic", "v-old-preference", "v-stale-episodic"]
        );

        // Audit: the past-horizon row is counted and attributable.
        let audit = db.decay_audit().unwrap();
        let profiles = audit["profiles"].as_array().unwrap();
        assert_eq!(profiles.len(), 9, "8 types + legacy row");
        let pop = audit["population"].as_array().unwrap();
        let episodic_row = pop
            .iter()
            .find(|r| r["memory_type"] == "episodic")
            .expect("episodic population row missing");
        assert_eq!(episodic_row["count"].as_i64().unwrap(), 2);
        assert_eq!(episodic_row["past_horizon_count"].as_i64().unwrap(), 1);
        let pref_row = pop
            .iter()
            .find(|r| r["memory_type"] == "preference")
            .expect("preference population row missing");
        assert_eq!(pref_row["past_horizon_count"].as_i64().unwrap(), 0);
    }
}
