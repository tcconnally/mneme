//! Graph utility gate (#869): deterministic query-shape classification.
//!
//! Graph traversal is only worth its cost (edge hydration, extra latency,
//! a second ranking arm) for questions that are actually *graph-shaped*:
//! multi-hop lineage/impact/dependency questions, named-entity hubs, global
//! structure/overview questions. Plain factual lookups are better served by
//! the keyword/semantic arms alone — running the graph arm on them adds
//! latency without candidates.
//!
//! This module is the pure, deterministic classifier behind the gate. It
//! emits a `utility` score in [0, 1] plus a `reason` code so the routing
//! decision is observable (it lands on every fused recall's `fused_trace`
//! as `graph_route`). No LLM, no network, no state: identical query →
//! identical route, so benchmarks and tests are reproducible.
//!
//! Signal design follows `docs/specs/graph-first-retrieval.md` §1: relational
//! verbs (depends/supports/caused/blocks/supersedes...) and named hubs are the
//! primary graph shapes; global overview words and date markers are secondary.
//! Pure-temporal questions ("what shipped on 2026-06-20") are NOT routed to
//! the graph arm: fused mode already serves them with the dedicated temporal
//! strategy, so the gate keeps the graph arm off them by construction.
//!
//! Default threshold: [`DEFAULT_GRAPH_UTILITY_THRESHOLD`]. Callers may lower
//! it (0.0 = always engage the graph arm when requested) or raise it
//! (1.0 = effectively never).

use crate::db::is_stopword;
use crate::extraction::RuleBasedExtractor;

/// Default utility threshold: a query must score at least this high for the
/// graph arm to engage (>= comparison, so exactly 0.5 engages).
pub const DEFAULT_GRAPH_UTILITY_THRESHOLD: f64 = 0.5;

/// The dominant question shape, as a machine-readable reason code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphRouteReason {
    /// Two or more relational connectors, or a strong connector with enough
    /// substance (entities/content) — lineage, impact, dependency questions.
    MultiHop,
    /// Overview / whole-system / structure words ("overview", "everything").
    Global,
    /// A date/time marker — served by the temporal strategy, NOT the graph.
    Temporal,
    /// One or more named-entity references (capitalized tokens, acronyms,
    /// quoted spans, `#refs`) without a multi-hop shape.
    EntityCentric,
    /// A single relational connector without entity anchors.
    Relational,
    /// A plain factual lookup.
    Ordinary,
    /// Empty or near-empty query — nothing to classify.
    NoSignal,
}

impl GraphRouteReason {
    pub fn as_str(&self) -> &'static str {
        match self {
            GraphRouteReason::MultiHop => "multi_hop",
            GraphRouteReason::Global => "global",
            GraphRouteReason::Temporal => "temporal",
            GraphRouteReason::EntityCentric => "entity_centric",
            GraphRouteReason::Relational => "relational",
            GraphRouteReason::Ordinary => "ordinary",
            GraphRouteReason::NoSignal => "no_signal",
        }
    }
}

/// The routing decision for one query.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GraphRoute {
    /// Utility in [0, 1]; higher = more graph-shaped.
    pub utility: f64,
    /// The dominant question shape.
    pub reason: GraphRouteReason,
}

impl GraphRoute {
    /// Whether the graph arm should engage at the given threshold.
    pub fn selected(&self, threshold: f64) -> bool {
        let threshold = threshold.clamp(0.0, 1.0);
        self.utility >= threshold
    }
}

/// Strong connectors: dependency/lineage/impact verbs that almost always mark
/// a graph question ("what depends on X", "what supports Y", "lineage of Z").
const STRONG_CONNECTORS: &[&str] = &[
    "depends",
    "dependency",
    "dependencies",
    "dependent",
    "dependents",
    "depended",
    "support",
    "supports",
    "supported",
    "supporting",
    "blocked",
    "blocks",
    "blocking",
    "caused",
    "causes",
    "cause of",
    "derived",
    "derives",
    "derivation",
    "lineage",
    "impact",
    "impacts",
    "because of",
    "follows",
    "follow from",
    "follows from",
    "implements",
    "implemented",
    "references",
    "referenced",
    "supersedes",
    "superseded by",
];

/// Weak connectors: relational words that mark a graph question only when
/// combined with entity anchors or other signals.
const WEAK_CONNECTORS: &[&str] = &[
    "related",
    "relates",
    "relationship",
    "relationships",
    "connected",
    "connects",
    "connection",
    "connections",
    "linking",
    "links",
    "linked",
    "link between",
    "between",
    "through",
    "path",
    "paths",
    "chain",
    "chains",
    "network",
    "networks",
    "ties",
    "tied",
    "association",
    "associations",
    "changed",
    "change",
    "influences",
    "influenced",
    "affects",
    "affected",
    "drives",
    "driven",
];

/// Global / overview question words.
const GLOBAL_WORDS: &[&str] = &[
    "overview",
    "everything",
    "whole",
    "landscape",
    "globally",
    "entire",
    "structure of",
    "map of",
    "big picture",
    "summary of the",
    "all of",
];

/// Contribution per signal (capped so no single signal saturates the score).
const STRONG_CONNECTOR_POINTS: f64 = 0.35;
const WEAK_CONNECTOR_POINTS: f64 = 0.2;
const ENTITY_POINTS: f64 = 0.3;
const CONTENT_WORD_POINTS: f64 = 0.1;
const TEMPORAL_POINTS: f64 = 0.15;
const GLOBAL_POINTS: f64 = 0.3;

const CONTENT_WORD_CAP: usize = 2;
const ENTITY_TOKEN_CAP: usize = 2;

/// Classify a query's graph utility. Pure and deterministic.
pub fn classify_graph_utility(query: &str) -> GraphRoute {
    let trimmed = query.trim();
    if trimmed.is_empty() {
        return GraphRoute {
            utility: 0.0,
            reason: GraphRouteReason::NoSignal,
        };
    }

    let lower = trimmed.to_lowercase();
    let strong = STRONG_CONNECTORS
        .iter()
        .filter(|w| lower.contains(**w))
        .count();
    let weak = WEAK_CONNECTORS
        .iter()
        .filter(|w| lower.contains(**w))
        .count();
    let global = GLOBAL_WORDS.iter().filter(|w| lower.contains(**w)).count() > 0;
    let temporal = RuleBasedExtractor::has_temporal_marker(&lower);
    let entity_tokens = count_entity_tokens(trimmed);
    let content_words = count_content_words(&lower);

    let mut utility = 0.0;
    if strong > 0 {
        utility += STRONG_CONNECTOR_POINTS * strong.min(2) as f64;
    }
    if weak > 0 {
        utility += WEAK_CONNECTOR_POINTS * weak.min(2) as f64;
    }
    utility += ENTITY_POINTS * entity_tokens.min(ENTITY_TOKEN_CAP) as f64;
    utility += CONTENT_WORD_POINTS * content_words.min(CONTENT_WORD_CAP) as f64;
    if temporal {
        utility += TEMPORAL_POINTS;
    }
    if global {
        utility += GLOBAL_POINTS;
    }
    utility = utility.clamp(0.0, 1.0);

    // Reason priority: multi-hop > global > temporal > entity-centric >
    // relational > ordinary > no-signal.
    let reason = if content_words == 0 && strong == 0 && weak == 0 && entity_tokens == 0 {
        GraphRouteReason::NoSignal
    } else if strong >= 2
        || strong + weak >= 3
        || (strong >= 1 && entity_tokens >= 1)
        || (strong >= 1 && content_words >= 3)
        || (weak >= 1 && entity_tokens >= 2)
    {
        GraphRouteReason::MultiHop
    } else if global {
        GraphRouteReason::Global
    } else if temporal {
        GraphRouteReason::Temporal
    } else if entity_tokens >= 1 {
        GraphRouteReason::EntityCentric
    } else if strong + weak >= 1 {
        GraphRouteReason::Relational
    } else {
        GraphRouteReason::Ordinary
    };

    GraphRoute { utility, reason }
}

/// Count named-entity references: capitalized tokens not at sentence start,
/// ALL-CAPS acronyms, quoted/backticked spans, and `#refs`.
pub(crate) fn count_entity_tokens(query: &str) -> usize {
    let mut count = 0usize;
    // Quoted or backticked spans count as one entity reference each.
    for quote in ['"', '\'', '`'] {
        let mut rest = query;
        while let Some(start) = rest.find(quote) {
            if let Some(end_rel) = rest[start + 1..].find(quote) {
                let span = &rest[start + 1..start + 1 + end_rel];
                if span.chars().count() >= 2 {
                    count += 1;
                }
                rest = &rest[start + 1 + end_rel + 1..];
            } else {
                break;
            }
        }
    }
    for tok in
        query.split(|c: char| c.is_whitespace() || c == ',' || c == ';' || c == '(' || c == ')')
    {
        let tok = tok.trim();
        if tok.is_empty() {
            continue;
        }
        if tok.starts_with('#') && tok.len() > 1 {
            count += 1; // #176-style references
            continue;
        }
        let letters = tok.chars().filter(|c| c.is_alphabetic()).count();
        let digits = tok.chars().filter(|c| c.is_ascii_digit()).count();
        if letters == 0 {
            continue;
        }
        // Mixed letter+digit tokens ("PR176", "stripe-2026") are entity-ish.
        if digits > 0 && letters >= 2 {
            count += 1;
            continue;
        }
        let all_upper = tok
            .chars()
            .filter(|c| c.is_alphabetic())
            .all(|c| c.is_uppercase());
        if all_upper && letters >= 2 {
            count += 1; // acronyms: PR, AWS, LEDGER
            continue;
        }
        // Capitalized word not at the very start of the query (sentence
        // capitalization is noise; later capitals are usually names).
        if !query.trim_start().starts_with(tok) {
            let first = tok.chars().next().unwrap();
            if first.is_uppercase() && letters >= 2 {
                count += 1;
                continue;
            }
        }
        // Capitalized word at query start, but a SECOND capitalized word
        // later in the query: both are entity-ish ("Alpha relates to Beta").
        if first_cap_at_start_is_entity(query, tok) {
            count += 1;
        }
    }
    count
}

/// A CAPITALIZED token at the start of the query counts as an entity when a
/// relational connector follows it (the query names a hub first). Lowercase
/// sentence-start words never count (sentence capitalization is noise).
fn first_cap_at_start_is_entity(query: &str, tok: &str) -> bool {
    if !query.trim_start().starts_with(tok) {
        return false;
    }
    let first = tok.chars().next().unwrap();
    if !first.is_uppercase() {
        return false;
    }
    let lower = query.to_lowercase();
    STRONG_CONNECTORS.iter().any(|w| lower.contains(*w))
        || WEAK_CONNECTORS.iter().any(|w| lower.contains(*w))
}

/// Count non-stopword content words (lowercased input).
fn count_content_words(lower: &str) -> usize {
    lower
        .split(|c: char| !c.is_ascii_alphanumeric() && c != '_' && c != '-' && c != '#')
        .filter(|t| !t.is_empty())
        .filter(|t| !is_stopword(t))
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn route(q: &str) -> (f64, &'static str) {
        let r = classify_graph_utility(q);
        (r.utility, r.reason.as_str())
    }

    fn selected_default(q: &str) -> bool {
        classify_graph_utility(q).selected(DEFAULT_GRAPH_UTILITY_THRESHOLD)
    }

    #[test]
    fn multi_hop_dependency_questions_select_graph() {
        assert_eq!(
            route("what depends on the stripe_events replay job"),
            (0.55, "multi_hop")
        );
        assert!(selected_default(
            "what depends on the stripe_events replay job"
        ));
        assert_eq!(
            route("how is Alpha related to Beta through Gamma"),
            (1.0, "multi_hop")
        );
        assert!(selected_default(
            "how is Alpha related to Beta through Gamma"
        ));
        assert_eq!(route("what changed because of PR 176"), (1.0, "multi_hop"));
        assert!(selected_default("what changed because of PR 176"));
        assert!(selected_default(
            "what supports the claim that deploy windows drop webhooks"
        ));
        assert!(selected_default("lineage of the ledger receipt chain"));
        // One weak connector + two named entities is still a multi-hop shape.
        let r = classify_graph_utility("Alpha relates to Beta");
        assert_eq!(r.reason, GraphRouteReason::MultiHop);
        assert!(r.selected(DEFAULT_GRAPH_UTILITY_THRESHOLD));
    }

    #[test]
    fn global_overview_questions_select_graph() {
        assert!(selected_default(
            "overview of everything and how it connects"
        ));
        let r = classify_graph_utility("overview of everything and how it connects");
        assert_eq!(r.reason, GraphRouteReason::Global);
        assert!(selected_default("map of the whole service landscape"));
    }

    #[test]
    fn ordinary_factual_lookups_do_not_select_graph() {
        assert!(!selected_default("what is my rent"));
        assert!(!selected_default("user preferred dark mode"));
        assert!(!selected_default("the database is postgres"));
        assert!(!selected_default("cat"));
        let r = classify_graph_utility("what is my rent");
        assert_eq!(r.reason, GraphRouteReason::Ordinary);
    }

    #[test]
    fn temporal_questions_are_flagged_but_not_routed_to_graph() {
        let r = classify_graph_utility("what shipped on 2026-06-20");
        assert_eq!(r.reason, GraphRouteReason::Temporal);
        assert!(!r.selected(DEFAULT_GRAPH_UTILITY_THRESHOLD));
        assert_eq!(
            classify_graph_utility("deployed the worker tier on tuesday").reason,
            GraphRouteReason::Temporal
        );
    }

    #[test]
    fn single_weak_connector_without_anchors_stays_below_threshold() {
        let r = classify_graph_utility("related documents");
        assert_eq!(r.reason, GraphRouteReason::Relational);
        assert!(!r.selected(DEFAULT_GRAPH_UTILITY_THRESHOLD));
    }

    #[test]
    fn entity_centric_questions_select_graph_with_low_threshold_only() {
        let r = classify_graph_utility("tell me about Stripe");
        assert_eq!(r.reason, GraphRouteReason::EntityCentric);
        // 1 entity + 2 content words = 0.2 + 0.3 = 0.5 → engages at default.
        assert!(r.selected(DEFAULT_GRAPH_UTILITY_THRESHOLD));
        assert_eq!(
            classify_graph_utility("what about AWS us-east-1").reason,
            GraphRouteReason::EntityCentric
        );
    }

    #[test]
    fn empty_and_garbage_queries_are_no_signal() {
        assert_eq!(
            classify_graph_utility("").reason,
            GraphRouteReason::NoSignal
        );
        assert_eq!(
            classify_graph_utility("   ").reason,
            GraphRouteReason::NoSignal
        );
        assert!(!selected_default(""));
        assert_eq!(
            classify_graph_utility("a the of").reason,
            GraphRouteReason::NoSignal
        );
    }

    #[test]
    fn threshold_override_turns_the_gate_off_or_on() {
        let ordinary = classify_graph_utility("what is my rent");
        assert!(ordinary.selected(0.0));
        assert!(!ordinary.selected(1.0));
        let multi = classify_graph_utility("what depends on the stripe replay job");
        assert!(multi.selected(0.0));
        assert!(!multi.selected(1.0));
    }

    #[test]
    fn classification_is_deterministic() {
        let a = classify_graph_utility("what depends on the stripe_events replay job");
        let b = classify_graph_utility("what depends on the stripe_events replay job");
        assert_eq!(a, b);
    }
}
