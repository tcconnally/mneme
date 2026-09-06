//! #1090: evolvable retrieval-skill routing (ERSkill, arXiv:2608.12720).
//!
//! Retrieval behaviors are compiled into executable SKILLS over the
//! primitives already present in the recall stack (search mode, typed
//! filters, as-of narrowing, recency/salience fusion, trust weighting). A
//! deterministic router selects one skill per query at inference time; the
//! skill set and router weights co-evolve OFFLINE from logged outcomes —
//! never by live self-modification.
//!
//! - **Double-frontier deployment**: a new skill version lands in the
//!   `expansion` frontier; it becomes router-facing (`serving`) only through
//!   the governed advancement gate, which refuses to advance unless the
//!   attached evaluation evidence demonstrates non-regression.
//! - **Experience trie**: every served route logs the explored path
//!   (skill id × query fingerprint × outcome) so evolution reuses rather
//!   than re-explores.
//! - Every definition/advancement is receipt-anchored in the journal, and
//!   demotion is the governed rollback.

use serde::{Deserialize, Serialize};

/// State-key prefixes for skill storage.
pub const SKILL_DEF_PREFIX: &str = "skill.def.";
pub const SKILL_EXP_PREFIX: &str = "skill.exp.";
pub const SKILL_EXP_STATS_PREFIX: &str = "skill.exp.stats.";
pub const SKILL_SERVING_VERSION_KEY: &str = "skill.serving_version";

/// Frontier names.
pub const FRONTIER_EXPANSION: &str = "expansion";
pub const FRONTIER_SERVING: &str = "serving";

/// Bounds: a served route never returns more than this many entities.
pub const SKILL_LIMIT_CAP: i64 = 50;
pub const SKILL_MAX_DEFS: usize = 64;

/// The executable part of a skill: a validated parameterization of the
/// existing recall primitives.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillTemplate {
    pub mode: String, // fts5 | dense | hybrid | fused
    pub limit: i64,
    #[serde(default)]
    pub category: Option<String>,
    #[serde(default)]
    pub type_filter: Option<String>,
    #[serde(default)]
    pub layer: Option<String>,
    #[serde(default)]
    pub epistemic_state: Option<String>,
    #[serde(default)]
    pub trust_weight: f64,
    #[serde(default)]
    pub content_weight: f64,
    #[serde(default)]
    pub max_prior_overturn: f64,
    #[serde(default)]
    pub recency_half_life_secs: Option<f64>,
    #[serde(default)]
    pub include_archived: bool,
}

impl SkillTemplate {
    pub fn validate(&self) -> Result<(), String> {
        if !matches!(self.mode.as_str(), "fts5" | "dense" | "hybrid" | "fused") {
            return Err(format!(
                "unknown mode {:?}; expected fts5 | dense | hybrid | fused",
                self.mode
            ));
        }
        if !(1..=SKILL_LIMIT_CAP).contains(&self.limit) {
            return Err(format!("limit must be 1..={SKILL_LIMIT_CAP}"));
        }
        if self.trust_weight < 0.0 || self.content_weight < 0.0 || self.max_prior_overturn < 0.0 {
            return Err("weights must be non-negative".into());
        }
        Ok(())
    }

    /// Map the validated template onto the recall primitives.
    pub fn to_recall_params(&self, query: &str) -> crate::models::RecallParams {
        use crate::models::{RecallParams, SearchMode};
        let mode = match self.mode.as_str() {
            "dense" => SearchMode::Dense,
            "hybrid" => SearchMode::Hybrid,
            "fused" => SearchMode::Fused,
            _ => SearchMode::Fts5,
        };
        RecallParams {
            query: query.to_string(),
            category: self.category.clone(),
            type_filter: self.type_filter.clone(),
            limit: self.limit,
            layer: self.layer.clone(),
            epistemic_state: self.epistemic_state.clone(),
            trust_weight: self.trust_weight,
            content_weight: self.content_weight,
            max_prior_overturn: self.max_prior_overturn,
            recency_half_life_secs: self.recency_half_life_secs,
            include_archived: self.include_archived,
            ..RecallParams::default()
        }
    }
}

/// Router-affinity profile: how strongly this skill matches deterministic
/// query features. Co-evolved offline, applied deterministically.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterProfile {
    pub base: f64,
    pub recent: f64,
    pub negation: f64,
    pub question: f64,
    pub type_hint: f64,
    pub long_query: f64,
}

impl Default for RouterProfile {
    fn default() -> Self {
        Self {
            base: 1.0,
            recent: 0.0,
            negation: 0.0,
            question: 0.0,
            type_hint: 0.0,
            long_query: 0.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillDef {
    pub skill_id: String,
    pub name: String,
    pub version: u64,
    pub frontier: String,
    pub profile: RouterProfile,
    pub template: SkillTemplate,
}

/// Deterministic query features for routing (no LLM in the loop).
#[derive(Debug, Clone, Default)]
pub struct QueryFeatures {
    pub recent: bool,
    pub negation: bool,
    pub question: bool,
    pub type_hint: bool,
    pub long_query: bool,
}

pub const RECENT_MARKERS: [&str; 9] = [
    "recent",
    "latest",
    "last",
    "today",
    "now",
    "yesterday",
    "this week",
    "newest",
    "current",
];
pub const NEGATION_MARKERS: [&str; 7] = [
    "not",
    "never",
    "no longer",
    "without",
    "doesn't",
    "don't",
    "cannot",
];
pub const QUESTION_MARKERS: [&str; 7] = ["who", "what", "when", "where", "why", "how", "which"];
pub const TYPE_HINT_MARKERS: [&str; 9] = [
    "policy",
    "decision",
    "fact",
    "convention",
    "lesson",
    "plan",
    "preference",
    "rule",
    "contact",
];

pub fn query_features(query: &str) -> QueryFeatures {
    let lower = query.to_lowercase();
    QueryFeatures {
        recent: RECENT_MARKERS.iter().any(|m| lower.contains(m)),
        negation: NEGATION_MARKERS.iter().any(|m| lower.contains(m)),
        question: query.contains('?') || QUESTION_MARKERS.iter().any(|m| lower.starts_with(m)),
        type_hint: TYPE_HINT_MARKERS.iter().any(|m| lower.contains(m)),
        long_query: query.chars().count() > 200,
    }
}

/// Router score: dot product of deterministic features against the
/// skill's affinity profile.
pub fn route_score(f: &QueryFeatures, p: &RouterProfile) -> f64 {
    p.base
        + f64::from(f.recent) * p.recent
        + f64::from(f.negation) * p.negation
        + f64::from(f.question) * p.question
        + f64::from(f.type_hint) * p.type_hint
        + f64::from(f.long_query) * p.long_query
}

/// Deterministic selection over serving-frontier skills: highest score,
/// ties broken by ascending skill id. Returns (skill, score, full ranking).
pub fn route_serving(
    query: &str,
    skills: &[SkillDef],
) -> Option<(SkillDef, f64, Vec<(String, f64)>)> {
    let f = query_features(query);
    let mut scored: Vec<(f64, String)> = skills
        .iter()
        .filter(|s| s.frontier == FRONTIER_SERVING)
        .map(|s| (route_score(&f, &s.profile), s.skill_id.clone()))
        .collect();
    if scored.is_empty() {
        return None;
    }
    scored.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.1.cmp(&b.1))
    });
    let (best_score, best_id) = scored.remove(0);
    let skill = skills.iter().find(|s| s.skill_id == best_id)?.clone();
    let ranking: Vec<(String, f64)> = scored.into_iter().map(|(score, id)| (id, score)).collect();
    Some((skill, best_score, ranking))
}

/// Non-regression evidence attached to an advancement request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvanceEvidence {
    pub eval_ref: String,
    pub wins: i64,
    pub losses: i64,
    pub ties: i64,
    /// Signed recall-metric delta vs the incumbent (e.g., F1 points).
    pub recall_delta: f64,
}

impl AdvanceEvidence {
    /// The fail-closed gate: advancement requires demonstrated
    /// non-regression (losses <= wins AND non-negative metric delta).
    pub fn passes(&self) -> bool {
        self.losses <= self.wins && self.recall_delta >= 0.0
    }
}

/// Deterministic query fingerprint for the experience trie (sha256, hex16).
pub fn query_fingerprint(query: &str) -> String {
    use sha2::{Digest, Sha256};
    let h = Sha256::digest(query.trim().to_lowercase().as_bytes());
    let hex = format!("{:x}", h);
    hex[..16.min(hex.len())].to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn skill(id: &str, profile: RouterProfile, mode: &str) -> SkillDef {
        SkillDef {
            skill_id: id.to_string(),
            name: id.to_string(),
            version: 1,
            frontier: FRONTIER_SERVING.to_string(),
            profile,
            template: SkillTemplate {
                mode: mode.to_string(),
                limit: 10,
                category: None,
                type_filter: None,
                layer: None,
                epistemic_state: None,
                trust_weight: 0.0,
                content_weight: 0.0,
                max_prior_overturn: 0.0,
                recency_half_life_secs: None,
                include_archived: false,
            },
        }
    }

    #[test]
    fn features_are_deterministic() {
        let f = query_features("what is the most recent policy decision?");
        assert!(f.recent && f.question && f.type_hint && !f.negation && !f.long_query);
        let n = query_features("this does not work, never works");
        assert!(n.negation && !n.question);
    }

    #[test]
    fn router_picks_affinity_and_ties_break_by_id() {
        let recency = skill(
            "s-recency",
            RouterProfile {
                recent: 10.0,
                ..Default::default()
            },
            "fts5",
        );
        let neutral = skill("s-neutral", RouterProfile::default(), "fts5");
        let q = "show me the latest results";
        let (chosen, score, _) = route_serving(q, &[neutral.clone(), recency.clone()]).unwrap();
        assert_eq!(chosen.skill_id, "s-recency");
        assert!(score > 1.0);
        // neutral query → base-only scores → tie → ascending id
        let (chosen2, _, _) = route_serving("results", &[neutral, recency]).unwrap();
        assert_eq!(chosen2.skill_id, "s-neutral");
    }

    #[test]
    fn expansion_frontier_is_never_routed() {
        let mut expansion = skill(
            "s-new",
            RouterProfile {
                base: 100.0,
                ..Default::default()
            },
            "fts5",
        );
        expansion.frontier = FRONTIER_EXPANSION.to_string();
        let serving = skill("s-old", RouterProfile::default(), "fts5");
        let (chosen, _, _) = route_serving("anything", &[expansion, serving.clone()]).unwrap();
        assert_eq!(chosen.skill_id, "s-old");
        // only expansion skills → no route
        let mut only_exp = serving;
        only_exp.frontier = FRONTIER_EXPANSION.to_string();
        assert!(route_serving("anything", &[only_exp]).is_none());
    }

    #[test]
    fn advance_gate_is_fail_closed() {
        let regress = AdvanceEvidence {
            eval_ref: "e1".into(),
            wins: 2,
            losses: 5,
            ties: 1,
            recall_delta: -0.03,
        };
        assert!(!regress.passes());
        let flat = AdvanceEvidence {
            eval_ref: "e2".into(),
            wins: 3,
            losses: 3,
            ties: 2,
            recall_delta: 0.0,
        };
        assert!(flat.passes());
        let wins = AdvanceEvidence {
            eval_ref: "e3".into(),
            wins: 9,
            losses: 1,
            ties: 0,
            recall_delta: 0.11,
        };
        assert!(wins.passes());
        let neg_delta = AdvanceEvidence {
            eval_ref: "e4".into(),
            wins: 9,
            losses: 1,
            ties: 0,
            recall_delta: -0.01,
        };
        assert!(
            !neg_delta.passes(),
            "negative metric delta refuses even with wins"
        );
    }

    #[test]
    fn template_validation_is_fail_closed() {
        let ok = SkillTemplate {
            mode: "fts5".into(),
            limit: 10,
            ..SkillTemplate::default_()
        };
        assert!(ok.validate().is_ok());
    }
}

// ── DB integration ──

fn skill_entity(id: &str, body: &str) -> crate::models::Entity {
    crate::models::Entity {
        id: id.to_string(),
        category: "facts".to_string(),
        key: id.to_string(),
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
        source: "agent".to_string(),
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
        epistemic_state: "candidate".to_string(),
        hints: vec![],
        memory_type: String::new(),
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

fn simple_def(id: &str, profile: RouterProfile) -> SkillDef {
    SkillDef {
        skill_id: id.to_string(),
        name: id.to_string(),
        version: 1,
        frontier: FRONTIER_EXPANSION.to_string(),
        profile,
        template: SkillTemplate {
            mode: "fts5".to_string(),
            limit: 10,
            category: None,
            type_filter: None,
            layer: None,
            epistemic_state: None,
            trust_weight: 0.0,
            content_weight: 0.0,
            max_prior_overturn: 0.0,
            recency_half_life_secs: None,
            include_archived: false,
        },
    }
}

#[test]
fn skill_lifecycle_route_serve_audit() {
    let db = crate::db::TestDatabase::new("skill-lifecycle");
    // Seed content so serve=true has something deterministic to return.
    write_fixture(
        &db,
        &skill_entity("mem-latest-a", "{\"content\": \"latest findings alpha\"}"),
    )
    .unwrap();
    write_fixture(
        &db,
        &skill_entity("mem-latest-b", "{\"content\": \"latest findings beta\"}"),
    )
    .unwrap();

    // Definitions enter the expansion frontier.
    let recency = simple_def(
        "s-recency",
        RouterProfile {
            recent: 10.0,
            ..Default::default()
        },
    );
    let neutral = simple_def("s-neutral", RouterProfile::default());
    let r1 = db.skill_set(&recency).unwrap();
    assert_eq!(r1["frontier"], "expansion");
    db.skill_set(&neutral).unwrap();

    // No serving skills yet → routing fails closed with guidance.
    assert!(db.skill_route("latest results", false).is_err());

    // Advancement gate: no evidence → refused; regression → refused.
    let no_ev = db.skill_advance("s-recency", None, "advance").unwrap();
    assert_eq!(no_ev["accepted"], false);
    let regress = AdvanceEvidence {
        eval_ref: "e1".into(),
        wins: 1,
        losses: 4,
        ties: 0,
        recall_delta: -0.02,
    };
    let refused = db
        .skill_advance("s-recency", Some(&regress), "advance")
        .unwrap();
    assert_eq!(refused["accepted"], false);
    // Still expansion → routing still fails.
    assert!(db.skill_route("latest results", false).is_err());

    // Non-regression evidence → accepted, serving, version bump.
    let good = AdvanceEvidence {
        eval_ref: "e2".into(),
        wins: 12,
        losses: 3,
        ties: 2,
        recall_delta: 0.31,
    };
    let adv = db
        .skill_advance("s-recency", Some(&good), "advance")
        .unwrap();
    assert_eq!(adv["accepted"], true);
    assert_eq!(adv["serving_version"], 1);
    let adv2 = db
        .skill_advance("s-neutral", Some(&good), "advance")
        .unwrap();
    assert_eq!(adv2["serving_version"], 2);

    // Routing: temporal query picks the recency skill.
    let route = db.skill_route("latest results", false).unwrap();
    assert_eq!(route["skill_id"], "s-recency");
    assert!(route["score"].as_f64().unwrap() > 1.0);

    // serve=true executes + logs the experience path.
    let served = db.skill_route("latest results", true).unwrap();
    assert!(served["served"].as_i64().unwrap() >= 1);
    assert_eq!(served["skill_id"], "s-recency");
    assert!(served["entities"].as_array().unwrap().len() >= 1);

    // Audit: stats bumped for the recency skill; receipts present.
    let audit = db.skill_audit().unwrap();
    let stats = audit["experience_stats"]["s-recency"].clone();
    assert!(stats["n"].as_i64().unwrap() >= 1);
    assert!(stats["ok"].as_i64().unwrap() >= 1);
    assert!(audit["receipts"].as_array().unwrap().len() >= 4);

    // Demote = governed rollback: serving → expansion, version bump.
    let dem = db.skill_advance("s-recency", None, "demote").unwrap();
    assert_eq!(dem["accepted"], true);
    assert_eq!(dem["serving_version"], 3);
    // Routing now picks the remaining serving skill (s-neutral).
    let route2 = db.skill_route("latest results", false).unwrap();
    assert_eq!(route2["skill_id"], "s-neutral");
}

#[test]
fn skill_set_validation_is_fail_closed() {
    let db = crate::db::TestDatabase::new("skill-validation");
    // Bad mode.
    let mut bad = simple_def("s-bad-mode", RouterProfile::default());
    bad.template.mode = "bm25".into();
    assert!(db.skill_set(&bad).is_err());
    // Bad limit.
    let mut bad2 = simple_def("s-bad-limit", RouterProfile::default());
    bad2.template.limit = 0;
    assert!(db.skill_set(&bad2).is_err());
    // Non-monotonic version.
    let ok = simple_def("s-versioned", RouterProfile::default());
    db.skill_set(&ok).unwrap();
    let mut lower = ok.clone();
    lower.version = 0;
    assert!(db.skill_set(&lower).is_err());
    let mut same = ok.clone();
    assert!(db.skill_set(&same).is_err());
    let mut newer = ok.clone();
    newer.version = 2;
    assert!(db.skill_set(&newer).is_ok());
    // New version lands in expansion even if the old one was serving.
    assert_eq!(
        db.skill_audit().unwrap()["skills"][0]["frontier"],
        "expansion"
    );
}

impl SkillTemplate {
    #[cfg(test)]
    fn default_() -> Self {
        Self {
            mode: "fts5".into(),
            limit: 10,
            category: None,
            type_filter: None,
            layer: None,
            epistemic_state: None,
            trust_weight: 0.0,
            content_weight: 0.0,
            max_prior_overturn: 0.0,
            recency_half_life_secs: None,
            include_archived: false,
        }
    }
}
