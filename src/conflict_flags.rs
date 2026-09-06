//! Deterministic, pre-injection contradiction flags for recall (#917).
//!
//! This module is deliberately a projection over existing conflict detection,
//! governance suppression, and claim cards. It never persists a flag and never
//! returns a body/claim string: the public surface contains only entity IDs,
//! validity metadata, and hash-linked card references.

use crate::claim_card::{build_claim_card, build_claim_card_for_conflict, ClaimCard};
use crate::db::Database;
use crate::models::Entity;
use serde_json::{json, Value};
use std::collections::{BTreeSet, HashMap, HashSet};

#[derive(Debug)]
pub(crate) struct ConflictFlagReport {
    pub flags: Vec<Value>,
    pub abstain_hint: bool,
    pub markdown: String,
}

fn raw_entity(
    db: &Database,
    id: &str,
    cache: &mut HashMap<String, Entity>,
) -> Result<Entity, String> {
    if let Some(entity) = cache.get(id) {
        return Ok(entity.clone());
    }
    let entity = db
        .get_entity_by_id_unfiltered(id)
        .map_err(|e| format!("conflict flags: entity lookup failed: {e}"))?
        .ok_or_else(|| format!("conflict flags: detector returned missing entity '{id}'"))?;
    cache.insert(id.to_string(), entity.clone());
    Ok(entity)
}

fn card_for(
    db: &Database,
    id: &str,
    suppressed: bool,
    workspace_hash: Option<&str>,
    agent_id: Option<&str>,
    now: i64,
    cache: &mut HashMap<String, ClaimCard>,
) -> Result<ClaimCard, String> {
    if let Some(card) = cache.get(id) {
        return Ok(card.clone());
    }
    // The projection never serializes this card. Suppressed rows use the
    // internal loader only so their identity/validity/card digest can be
    // linked; claim text remains inside this process and is not returned.
    let card = if suppressed {
        build_claim_card_for_conflict(db, id, workspace_hash, agent_id, true, false, now)?
    } else {
        build_claim_card(db, id, workspace_hash, agent_id, true, false, now)?
    };
    cache.insert(id.to_string(), card.clone());
    Ok(card)
}

fn pair_ids(pair: &Value) -> Option<(String, String)> {
    let a = pair
        .get("entity_a")
        .and_then(|v| v.get("id"))
        .and_then(Value::as_str)?;
    let b = pair
        .get("entity_b")
        .and_then(|v| v.get("id"))
        .and_then(Value::as_str)?;
    Some((a.to_string(), b.to_string()))
}

fn workspace_compatible(
    a: &Entity,
    b: &Entity,
    requested_workspace: Option<&str>,
    scope_weight: Option<f64>,
) -> bool {
    if a.workspace_hash == b.workspace_hash {
        return true;
    }
    // The only permitted cross-partition pairing is an explicitly widened
    // current-workspace + global recall. Never compare two arbitrary tenants.
    let Some(ws) = requested_workspace.filter(|ws| !ws.is_empty()) else {
        return false;
    };
    scope_weight.is_some_and(|_| {
        (a.workspace_hash == ws && b.workspace_hash.is_empty())
            || (b.workspace_hash == ws && a.workspace_hash.is_empty())
    })
}

fn high_confidence(card: &ClaimCard) -> bool {
    !card.state.archived
        && !card.state.quarantined
        && (card.verified || matches!(card.epistemic_state.as_str(), "verified" | "corroborated"))
}

fn flag_kind(candidate: &ClaimCard, claim: &ClaimCard) -> &'static str {
    if candidate.state.superseded || claim.state.superseded {
        "superseded"
    } else if candidate.state.stale || claim.state.stale {
        "stale"
    } else {
        "contradiction"
    }
}

fn validity(card: &ClaimCard) -> Value {
    // `valid_from` is backfilled for current rows, but use recorded time as the
    // effective opening for legacy rows so the flag always carries a usable
    // bi-temporal range rather than inventing a world-time instant.
    json!({
        "valid_from_unix_ms": card.times.valid_from_unix_ms.or(card.times.recorded_at_unix_ms),
        "valid_to_unix_ms": card.times.valid_to_unix_ms,
        "recorded_at_unix_ms": card.times.recorded_at_unix_ms,
        "invalidated_at_unix_ms": card.times.invalidated_at_unix_ms,
    })
}

fn evidence_refs(candidate: &ClaimCard, claim: &ClaimCard) -> Value {
    json!([
        {"entity_id": candidate.entity_id, "card_digest": candidate.digest},
        {"entity_id": claim.entity_id, "card_digest": claim.digest},
    ])
}

fn push_direction(flags: &mut Vec<Value>, candidate: &ClaimCard, claim: &ClaimCard) -> bool {
    // A flag is useful only when its established side is claim-card high-grade.
    // A lower-grade retrieved candidate can still be marked against that fact;
    // a pair of unestablished drafts is not an abstention signal.
    if !high_confidence(claim) {
        return false;
    }
    let kind = flag_kind(candidate, claim);
    flags.push(json!({
        "candidate_id": candidate.entity_id,
        "claim_id": claim.entity_id,
        "kind": kind,
        "validity": {
            "candidate": validity(candidate),
            "claim": validity(claim),
        },
        "evidence_refs": evidence_refs(candidate, claim),
        "confidence": "high",
        "disposition": "flag",
        "disclose_existence": true,
        "disclose_value": false,
    }));
    true
}

fn render_markdown(flags: &[Value]) -> String {
    let mut out = String::from("### Conflict flags\n\n");
    if flags.is_empty() {
        out.push_str("No conflicts detected.");
        return out;
    }
    out.push_str("The retrieved context contains deterministic conflict flags; do not blend conflicting values without re-verification.\n\n");
    for flag in flags {
        let candidate = flag
            .get("candidate_id")
            .and_then(Value::as_str)
            .unwrap_or("");
        let claim = flag.get("claim_id").and_then(Value::as_str).unwrap_or("");
        let kind = flag
            .get("kind")
            .and_then(Value::as_str)
            .unwrap_or("conflict");
        let confidence = flag
            .get("confidence")
            .and_then(Value::as_str)
            .unwrap_or("unknown");
        out.push_str(&format!(
            "- conflict kind={kind} candidate_id={candidate} claim_id={claim} confidence={confidence} disclose_value=false\n"
        ));
    }
    out
}

/// Assemble flags for the already-filtered recall candidates. The detector is
/// invoked only when a caller opted into one of the two projections, and every
/// operation here is observational: conflict detection, suppression filtering,
/// raw identity lookup, and claim-card construction are all read paths.
pub(crate) fn assemble(
    db: &Database,
    candidates: &[Entity],
    requested_workspace: Option<&str>,
    scope_weight: Option<f64>,
    requesting_agent_id: Option<&str>,
) -> Result<ConflictFlagReport, String> {
    if candidates.is_empty() {
        return Ok(ConflictFlagReport {
            flags: Vec::new(),
            abstain_hint: false,
            markdown: render_markdown(&[]),
        });
    }

    let retrieved_ids: HashSet<String> = candidates.iter().map(|e| e.id.clone()).collect();
    let candidate_by_id: HashMap<String, &Entity> = candidates
        .iter()
        .map(|entity| (entity.id.clone(), entity))
        .collect();
    let categories: BTreeSet<String> = candidates
        .iter()
        .map(|entity| entity.category.clone())
        .collect();
    let requester = requesting_agent_id.unwrap_or("");
    let now = crate::db::now_ms();
    let mut raw_cache = HashMap::new();
    let mut suppressed_cache: HashMap<String, bool> = HashMap::new();
    let mut card_cache = HashMap::new();
    let mut flags = Vec::new();
    let mut abstain_hint = false;

    for category in categories {
        let report = db
            .detect_conflicts(&category, 0.6, 1000, 0)
            .map_err(|e| format!("conflict flags: detector failed: {e}"))?;
        let Some(pairs) = report.get("conflicts").and_then(Value::as_array) else {
            continue;
        };
        for pair in pairs {
            let Some((a_id, b_id)) = pair_ids(pair) else {
                continue;
            };
            let conflict_likely = pair
                .get("conflict_likely")
                .and_then(Value::as_bool)
                .or_else(|| {
                    pair.get("similarity")
                        .and_then(Value::as_f64)
                        .map(|similarity| similarity < 0.3)
                })
                .unwrap_or(false);
            if !conflict_likely {
                continue;
            }

            let a = raw_entity(db, &a_id, &mut raw_cache)?;
            let b = raw_entity(db, &b_id, &mut raw_cache)?;
            if !workspace_compatible(&a, &b, requested_workspace, scope_weight) {
                continue;
            }
            if !db.can_read(requester, &a.visibility, &a.agent_id)
                || !db.can_read(requester, &b.visibility, &b.agent_id)
            {
                continue;
            }

            // Reuse the same governance interceptor as recall. A missing id
            // from this kept set is an existence-only, suppressed claim side.
            let kept = db
                .filter_suppressed(vec![a.clone(), b.clone()])
                .map_err(|e| format!("conflict flags: suppression check failed: {e}"))?;
            let kept_ids: HashSet<String> = kept.into_iter().map(|entity| entity.id).collect();
            let a_suppressed = !kept_ids.contains(&a_id);
            let b_suppressed = !kept_ids.contains(&b_id);
            suppressed_cache.insert(a_id.clone(), a_suppressed);
            suppressed_cache.insert(b_id.clone(), b_suppressed);

            let a_card = card_for(
                db,
                &a_id,
                a_suppressed,
                requested_workspace,
                requesting_agent_id,
                now,
                &mut card_cache,
            )?;
            let b_card = card_for(
                db,
                &b_id,
                b_suppressed,
                requested_workspace,
                requesting_agent_id,
                now,
                &mut card_cache,
            )?;

            let a_retrieved = retrieved_ids.contains(&a_id) && !a_suppressed;
            let b_retrieved = retrieved_ids.contains(&b_id) && !b_suppressed;
            if !a_retrieved && !b_retrieved {
                continue;
            }

            if a_retrieved {
                let _ = push_direction(&mut flags, &a_card, &b_card);
            }
            if b_retrieved {
                let _ = push_direction(&mut flags, &b_card, &a_card);
            }

            // The abstention hint is intentionally narrower than flag
            // emission: both sides must be in the delivered, unsuppressed set
            // and at least one side must be a high-grade claim-card fact.
            if a_retrieved
                && b_retrieved
                && flag_kind(&a_card, &b_card) == "contradiction"
                && (high_confidence(&a_card) || high_confidence(&b_card))
            {
                abstain_hint = true;
            }
        }
    }

    flags.sort_by(|a, b| {
        a.get("candidate_id")
            .and_then(Value::as_str)
            .cmp(&b.get("candidate_id").and_then(Value::as_str))
            .then_with(|| {
                a.get("claim_id")
                    .and_then(Value::as_str)
                    .cmp(&b.get("claim_id").and_then(Value::as_str))
            })
            .then_with(|| {
                a.get("kind")
                    .and_then(Value::as_str)
                    .cmp(&b.get("kind").and_then(Value::as_str))
            })
    });

    // Keep the map alive until all cards are built; this also makes it explicit
    // that suppression state is computed from IDs only in the output path.
    let _ = suppressed_cache;
    let markdown = render_markdown(&flags);
    Ok(ConflictFlagReport {
        flags,
        abstain_hint,
        markdown,
    })
}
