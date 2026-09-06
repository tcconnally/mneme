//! #1008: per-type retrieval context budgets at context assembly
//! (MindCache borrow, extends #1000).
//!
//! MindCache partitions its RRF candidate pools by memory type and enforces
//! per-type floors/caps when assembling the final context — diverse evidence
//! beats a wall of similar high-scoring ephemeral entries, and standing
//! policy (decisions) keeps a guaranteed share instead of being outvoted by
//! vector similarity. With typed classes landed (#1000), this is the
//! retrieval-side counterpart.
//!
//! Purely a SELECTION layer over the fused pool: floors pull floor-worthy
//! items up from below the caller limit, caps stop any one class from
//! crowding out the rest, and everything is reported in the truncation
//! trace (never silent). Off by default — a default recall stays
//! byte-identical (#247).

use serde::Serialize;

/// Pseudo-class: the entity's category is "decision" — the vault's standing
/// policy records. Allocation-wise it overrides the row's memory_type so
/// governed decisions get their own floor/cap lane. (Keystone-backed
/// detection is deferred: keystones are policy, and decision-category rows
/// are their usual carrier.)
pub const DECISION_CLASS: &str = "decision";

#[derive(Debug, Clone, Serialize)]
pub struct TypeAllocation {
    pub class: String,
    pub floor: usize,
    pub cap: usize,
    pub retained: usize,
    pub floor_shortfall: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct AllocationReport {
    pub profile: String,
    pub allocations: Vec<TypeAllocation>,
}

#[derive(Debug, Clone)]
pub struct TypeBudgetProfile {
    pub name: &'static str,
    /// (class, min slots) — floors are satisfied FIRST, in rank order.
    pub floors: Vec<(String, usize)>,
    /// (class, max slots) — caps applied during the ranked walk.
    pub caps: Vec<(String, usize)>,
}

impl TypeBudgetProfile {
    pub fn floor_for(&self, class: &str) -> usize {
        self.floors
            .iter()
            .find(|(c, _)| c == class)
            .map(|(_, n)| *n)
            .unwrap_or(0)
    }
    pub fn cap_for(&self, class: &str) -> Option<usize> {
        self.caps.iter().find(|(c, _)| c == class).map(|(_, n)| *n)
    }
}

/// The MindCache "diverse evidence" default: every class keeps a bounded
/// share; decisions and constraints get guaranteed floors.
pub fn profile_diverse() -> TypeBudgetProfile {
    TypeBudgetProfile {
        name: "diverse",
        floors: vec![("decision".into(), 3), ("constraint".into(), 2)],
        caps: vec![
            ("semantic".into(), 15),
            ("episodic".into(), 10),
            ("procedural".into(), 10),
            ("knowledge".into(), 10),
            ("preference".into(), 5),
            ("failure".into(), 5),
            ("reflection".into(), 5),
            ("constraint".into(), 8),
            ("decision".into(), 8),
        ],
    }
}

/// Fact lookups: tight episodic cap, wide semantic/knowledge lanes.
pub fn profile_fact_lookup() -> TypeBudgetProfile {
    TypeBudgetProfile {
        name: "fact_lookup",
        floors: vec![("decision".into(), 2)],
        caps: vec![
            ("semantic".into(), 20),
            ("knowledge".into(), 15),
            ("episodic".into(), 5),
            ("procedural".into(), 8),
            ("preference".into(), 8),
            ("failure".into(), 8),
            ("reflection".into(), 8),
            ("constraint".into(), 8),
            ("decision".into(), 6),
        ],
    }
}

/// Broad/synthesis queries: summary-level coverage — decisions and
/// constraints dominate the floors; low-signal classes are tightly capped.
pub fn profile_broad() -> TypeBudgetProfile {
    TypeBudgetProfile {
        name: "broad",
        floors: vec![
            ("decision".into(), 4),
            ("constraint".into(), 3),
            ("semantic".into(), 3),
        ],
        caps: vec![
            ("semantic".into(), 12),
            ("knowledge".into(), 12),
            ("episodic".into(), 8),
            ("procedural".into(), 6),
            ("preference".into(), 4),
            ("failure".into(), 4),
            ("reflection".into(), 4),
            ("constraint".into(), 10),
            ("decision".into(), 10),
        ],
    }
}

/// Resolve a profile by name. Unknown names return None (caller fails
/// closed with a validation error — never a silent fallback to unshaped).
pub fn profile(name: &str) -> Option<TypeBudgetProfile> {
    match name {
        "diverse" => Some(profile_diverse()),
        "fact_lookup" => Some(profile_fact_lookup()),
        "broad" => Some(profile_broad()),
        _ => None,
    }
}

/// The allocation class of an entity: decision-category rows map to the
/// pseudo-class `decision`; everything else maps by memory_type with the
/// legacy '' normalized to `semantic` (#1000).
pub fn class_of(category: &str, memory_type: &str) -> String {
    if category == "decision" {
        DECISION_CLASS.to_string()
    } else if memory_type.is_empty() {
        "semantic".to_string()
    } else {
        memory_type.to_string()
    }
}

/// Apply floors + caps to a rank-ordered pool.
///
/// - Floors first: for each (class, n), pull the top-scored n items of that
///   class (in pool order) that are not already retained — even from below
///   the caller `limit`. Shortfalls are recorded, never fatal.
/// - Then the ranked walk: retain in pool order, skipping any item whose
///   class count has reached its cap, until `limit` items are retained.
///
/// Returns (retained, report). Deterministic: stable on equal scores.
pub fn apply(
    pool: &[(crate::models::Entity, f64)],
    prof: &TypeBudgetProfile,
    limit: usize,
) -> (Vec<crate::models::Entity>, AllocationReport) {
    let mut retained: Vec<usize> = Vec::new();
    let mut class_counts: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();
    let mut floors_met: std::collections::HashMap<String, usize> = std::collections::HashMap::new();

    // Phase 1: floors, in profile order, each satisfied from the best-scored
    // not-yet-retained items of that class anywhere in the pool.
    for (class, floor) in &prof.floors {
        let mut taken = 0usize;
        for (idx, (e, _)) in pool.iter().enumerate() {
            if taken >= *floor {
                break;
            }
            if retained.contains(&idx) {
                continue;
            }
            if class_of(&e.category, &e.memory_type) == *class {
                retained.push(idx);
                *class_counts.entry(class.clone()).or_insert(0) += 1;
                taken += 1;
            }
        }
        floors_met.insert(class.clone(), taken);
    }

    // Phase 2: caps + caller limit over the ranked walk.
    for (idx, (e, _)) in pool.iter().enumerate() {
        if retained.len() >= limit {
            break;
        }
        if retained.contains(&idx) {
            continue;
        }
        let class = class_of(&e.category, &e.memory_type);
        let count = *class_counts.entry(class.clone()).or_insert(0);
        if let Some(cap) = prof.cap_for(&class) {
            if count >= cap {
                continue;
            }
        }
        retained.push(idx);
        *class_counts.entry(class).or_insert(0) += 1;
    }

    // Final ordering: the pool's original rank order (floors first means
    // floor pulls may appear before higher-scored later items — that is the
    // point; they get the guaranteed slot).
    let mut ordered = retained.clone();
    ordered.sort_unstable();

    let allocations: Vec<TypeAllocation> = {
        let mut classes: Vec<String> = prof
            .floors
            .iter()
            .map(|(c, _)| c.clone())
            .chain(prof.caps.iter().map(|(c, _)| c.clone()))
            .collect();
        classes.sort();
        classes.dedup();
        classes
            .into_iter()
            .map(|class| {
                let floor = prof.floor_for(&class);
                let met = floors_met.get(&class).copied().unwrap_or(0);
                TypeAllocation {
                    retained: *class_counts.get(&class).unwrap_or(&0),
                    floor,
                    cap: prof.cap_for(&class).unwrap_or(usize::MAX.min(limit)),
                    floor_shortfall: floor.saturating_sub(met),
                    class,
                }
            })
            .collect()
    };

    let entities = ordered.iter().map(|&i| pool[i].0.clone()).collect();
    (
        entities,
        AllocationReport {
            profile: prof.name.to_string(),
            allocations,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ent(
        id: &str,
        category: &str,
        memory_type: &str,
        score: f64,
    ) -> (crate::models::Entity, f64) {
        let e = crate::models::Entity {
            id: id.to_string(),
            category: category.to_string(),
            key: id.to_string(),
            body_json: format!("body of {id}"),
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
            created_at_unix_ms: 0,
            last_accessed_unix_ms: 0,
            follow_count: 0,
            miss_count: 0,
            follow_rate: 0.0,
            efficacy_status: "unverified".to_string(),
            epistemic_state: crate::models::default_epistemic_state(),
            hints: vec![],
            memory_type: memory_type.to_string(),
            embedding: None,
            _parsed_body: None,
        };
        (e, score)
    }

    #[test]
    fn profiles_are_resolvable_and_unknown_fails_closed() {
        assert!(profile("diverse").is_some());
        assert!(profile("fact_lookup").is_some());
        assert!(profile("broad").is_some());
        assert!(profile("nope").is_none());
    }

    #[test]
    fn class_of_maps_decision_and_legacy() {
        assert_eq!(class_of("decision", "episodic"), "decision");
        assert_eq!(class_of("facts", ""), "semantic");
        assert_eq!(class_of("facts", "episodic"), "episodic");
    }

    #[test]
    fn floors_pull_from_below_limit_and_caps_stop_crowding() {
        // 12 episodic entries outscore the lone decision; limit = 4.
        let mut pool = Vec::new();
        for i in 0..12 {
            pool.push(ent(
                &format!("e{i}"),
                "events",
                "episodic",
                1.0 - i as f64 * 0.01,
            ));
        }
        pool.push(ent("d1", "decision", "procedural", 0.1));
        let prof = profile_diverse();
        let (retained, report) = apply(&pool, &prof, 4);
        // Floor: decision gets a slot even at score 0.1.
        assert!(retained.iter().any(|e| e.id == "d1"));
        // Cap: episodic capped at 10 — with limit 4, 3 episodic + decision.
        assert_eq!(retained.len(), 4);
        let ep = report
            .allocations
            .iter()
            .find(|a| a.class == "episodic")
            .unwrap();
        assert_eq!(ep.retained, 3);
        let dec = report
            .allocations
            .iter()
            .find(|a| a.class == "decision")
            .unwrap();
        assert_eq!(dec.retained, 1);
        assert_eq!(
            dec.floor_shortfall, 2,
            "diverse floor is 3, only 1 decision exists"
        );
    }

    #[test]
    fn floor_shortfall_is_recorded_when_class_absent() {
        let pool = vec![ent("e1", "events", "episodic", 1.0)];
        let prof = profile_broad();
        let (retained, report) = apply(&pool, &prof, 10);
        assert_eq!(retained.len(), 1);
        for a in report.allocations.iter().filter(|a| a.floor > 0) {
            assert!(a.floor_shortfall > 0 || a.retained > 0);
        }
        assert_eq!(report.profile, "broad");
    }

    #[test]
    fn caps_bound_even_without_limit_pressure() {
        let mut pool = Vec::new();
        for i in 0..30 {
            pool.push(ent(
                &format!("e{i}"),
                "events",
                "episodic",
                1.0 - i as f64 * 0.001,
            ));
        }
        let prof = profile_diverse();
        let (retained, report) = apply(&pool, &prof, 30);
        let ep = report
            .allocations
            .iter()
            .find(|a| a.class == "episodic")
            .unwrap();
        assert_eq!(ep.retained, 10, "episodic cap 10 must bound a 30-item wall");
        assert_eq!(retained.len(), 10);
    }
}
