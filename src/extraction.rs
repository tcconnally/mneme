//! Local, deterministic knowledge extraction (#234).
//!
//! Parses raw memory text into structured items — facts, preferences, temporal
//! events, episodes — using pure, dependency-free heuristics. No cloud LLM, no
//! embedding/API call, no network: this preserves Perseus Vault's air-gapped,
//! zero-dependency path (unlike GoodMem/Synap, which require a Gemini key).
//!
//! [`Extractor`] is the plugin point. [`NoopExtractor`] is the default (pure
//! storage, no extraction); [`RuleBasedExtractor`] is a concrete local
//! implementation. A future model-based extractor can slot in behind the same
//! trait without touching callers — keeping extraction strictly opt-in.

use serde::Serialize;

/// The structured kind of an extracted knowledge item.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ExtractKind {
    /// A declarative statement of fact ("X is Y", "the service uses Z").
    Fact,
    /// A first-person preference ("I prefer X", "my favorite is Y").
    Preference,
    /// A statement anchored to a date/time ("shipped on 2026-06-20", "met tuesday").
    TemporalEvent,
    /// A first-person experiential action ("we deployed the worker tier").
    Episode,
}

/// A single structured item extracted from raw memory text.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ExtractedItem {
    pub kind: ExtractKind,
    pub text: String,
}

/// The extraction plugin point. Implementors turn raw text into structured items.
pub trait Extractor {
    fn extract(&self, text: &str) -> Vec<ExtractedItem>;
}

/// The default: no extraction (pure storage). Keeps the zero-dependency path intact.
pub struct NoopExtractor;

impl Extractor for NoopExtractor {
    fn extract(&self, _text: &str) -> Vec<ExtractedItem> {
        Vec::new()
    }
}

/// A concrete, fully-local, deterministic rule-based extractor.
///
/// Splits text into sentences and classifies each with a fixed priority:
/// temporal marker → [`ExtractKind::TemporalEvent`]; first-person preference cue →
/// [`ExtractKind::Preference`]; first-person past action → [`ExtractKind::Episode`];
/// declarative copula → [`ExtractKind::Fact`]; otherwise the sentence is skipped
/// (precision over recall). Identical items are de-duplicated, order preserved.
pub struct RuleBasedExtractor;

const PREFERENCE_CUES: &[&str] = &[
    "i prefer",
    "i like",
    "i love",
    "i hate",
    "i dislike",
    "i want",
    "i'd rather",
    "i would rather",
    "my favorite",
    "my favourite",
    "we prefer",
    "prefer to use",
];

// First-person experiential actions (past or habitual) → episodes.
const EPISODE_CUES: &[&str] = &[
    "i did",
    "i went",
    "i met",
    "i built",
    "i wrote",
    "i fixed",
    "i shipped",
    "i deployed",
    "i decided",
    "i finished",
    "i completed",
    "i added",
    "i removed",
    "we did",
    "we met",
    "we built",
    "we shipped",
    "we deployed",
    "we decided",
    "we added",
    "we fixed",
    "we migrated",
    "we launched",
];

const MONTHS: &[&str] = &[
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
    "jan ",
    "feb ",
    "mar ",
    "apr ",
    "jun ",
    "jul ",
    "aug ",
    "sep ",
    "sept ",
    "oct ",
    "nov ",
    "dec ",
];

const WEEKDAYS: &[&str] = &[
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
];

const RELATIVE_TIME: &[&str] = &[
    "yesterday",
    "today",
    "tomorrow",
    "last week",
    "last month",
    "last year",
    "next week",
    "next month",
    "this morning",
    "this afternoon",
    "tonight",
];

// Declarative copulas / relations that mark a factual statement.
const FACT_MARKERS: &[&str] = &[
    " is ",
    " are ",
    " was ",
    " were ",
    " has ",
    " have ",
    " uses ",
    " runs on ",
    " consists of ",
    " supports ",
    " requires ",
    " depends on ",
    " stores ",
];

impl RuleBasedExtractor {
    /// True when the sentence carries an explicit date/time marker.
    /// `pub(crate)`: shared with the graph utility gate (#869), which uses
    /// temporal markers to classify query shape.
    pub(crate) fn has_temporal_marker(lower: &str) -> bool {
        if Self::contains_year(lower) {
            return true;
        }
        if Self::contains_clock_time(lower) {
            return true;
        }
        MONTHS.iter().any(|m| lower.contains(m))
            || WEEKDAYS.iter().any(|d| lower.contains(d))
            || RELATIVE_TIME.iter().any(|r| lower.contains(r))
    }

    /// Detects a 4-digit year in 1900–2099 (the dominant date signal).
    fn contains_year(lower: &str) -> bool {
        let bytes = lower.as_bytes();
        let mut i = 0;
        while i + 4 <= bytes.len() {
            // Must be a 4-digit run not bordered by other digits.
            let window = &bytes[i..i + 4];
            let all_digits = window.iter().all(|b| b.is_ascii_digit());
            let left_ok = i == 0 || !bytes[i - 1].is_ascii_digit();
            let right_ok = i + 4 == bytes.len() || !bytes[i + 4].is_ascii_digit();
            if all_digits && left_ok && right_ok {
                let yr = std::str::from_utf8(window).unwrap_or("0");
                if matches!(&yr[0..2], "19" | "20") {
                    return true;
                }
            }
            i += 1;
        }
        false
    }

    /// Detects a clock time like `14:30` or `9:05`.
    fn contains_clock_time(lower: &str) -> bool {
        let bytes = lower.as_bytes();
        for (i, &b) in bytes.iter().enumerate() {
            if b == b':' && i >= 1 && i + 2 < bytes.len() {
                let before = bytes[i - 1].is_ascii_digit();
                let after = bytes[i + 1].is_ascii_digit() && bytes[i + 2].is_ascii_digit();
                if before && after {
                    return true;
                }
            }
        }
        false
    }

    fn classify(sentence: &str) -> Option<ExtractKind> {
        let lower = sentence.to_lowercase();
        if Self::has_temporal_marker(&lower) {
            return Some(ExtractKind::TemporalEvent);
        }
        if PREFERENCE_CUES.iter().any(|c| lower.contains(c)) {
            return Some(ExtractKind::Preference);
        }
        if EPISODE_CUES.iter().any(|c| lower.contains(c)) {
            return Some(ExtractKind::Episode);
        }
        if FACT_MARKERS.iter().any(|m| lower.contains(m)) {
            return Some(ExtractKind::Fact);
        }
        None
    }
}

/// Split text into trimmed sentences on `.`, `!`, `?`, and newlines.
fn split_sentences(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut buf = String::new();
    for ch in text.chars() {
        match ch {
            '.' | '!' | '?' | '\n' | '\r' => {
                let s = buf.trim();
                if !s.is_empty() {
                    out.push(s.to_string());
                }
                buf.clear();
            }
            _ => buf.push(ch),
        }
    }
    let s = buf.trim();
    if !s.is_empty() {
        out.push(s.to_string());
    }
    out
}

impl Extractor for RuleBasedExtractor {
    fn extract(&self, text: &str) -> Vec<ExtractedItem> {
        let mut out: Vec<ExtractedItem> = Vec::new();
        for sentence in split_sentences(text) {
            if let Some(kind) = Self::classify(&sentence) {
                let item = ExtractedItem {
                    kind,
                    text: sentence,
                };
                if !out.contains(&item) {
                    out.push(item);
                }
            }
        }
        out
    }
}

/// Resolve a strategy name to an extractor. Unknown / "none" → [`NoopExtractor`].
pub fn extractor_for(strategy: &str) -> Box<dyn Extractor> {
    match strategy {
        "rule_based" => Box::new(RuleBasedExtractor),
        _ => Box::new(NoopExtractor),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kinds(items: &[ExtractedItem]) -> Vec<ExtractKind> {
        items.iter().map(|i| i.kind).collect()
    }

    #[test]
    fn noop_extracts_nothing() {
        let items = NoopExtractor.extract("I prefer dark mode. The db is Postgres.");
        assert!(items.is_empty());
    }

    #[test]
    fn classifies_preference_fact_temporal_episode() {
        let text = "I prefer dark mode. The database is PostgreSQL. \
                    We deployed the worker tier. We shipped v2 on 2026-06-20.";
        let items = RuleBasedExtractor.extract(text);
        assert_eq!(
            kinds(&items),
            vec![
                ExtractKind::Preference,
                ExtractKind::Fact,
                ExtractKind::Episode,
                ExtractKind::TemporalEvent, // dated → temporal wins over episode
            ]
        );
    }

    #[test]
    fn temporal_marker_takes_priority() {
        // A clock time and a weekday both mark temporal events.
        let items = RuleBasedExtractor.extract("The standup is at 09:30. We met on Tuesday.");
        assert_eq!(
            kinds(&items),
            vec![ExtractKind::TemporalEvent, ExtractKind::TemporalEvent]
        );
    }

    #[test]
    fn year_detection_is_bounded() {
        assert!(RuleBasedExtractor::contains_year("released in 2026"));
        assert!(RuleBasedExtractor::contains_year("back in 1998 it shipped"));
        assert!(!RuleBasedExtractor::contains_year("order 12345 failed")); // 5 digits, not a year
        assert!(!RuleBasedExtractor::contains_year("port 8080 is open")); // not 19/20xx
    }

    #[test]
    fn unclassifiable_sentences_are_skipped() {
        // No copula, no cue, no date → nothing (precision over recall).
        let items = RuleBasedExtractor.extract("Hello there. Wow!");
        assert!(items.is_empty());
    }

    #[test]
    fn deduplicates_identical_items() {
        let items = RuleBasedExtractor.extract("The db is Postgres. The db is Postgres.");
        assert_eq!(items.len(), 1);
    }

    #[test]
    fn extraction_is_deterministic() {
        let text = "I like Rust. We shipped on 2026-01-02. The cache is an LRU.";
        let a = RuleBasedExtractor.extract(text);
        let b = RuleBasedExtractor.extract(text);
        assert_eq!(a, b);
    }

    #[test]
    fn extractor_for_unknown_is_noop() {
        assert!(extractor_for("nope")
            .extract("The db is Postgres.")
            .is_empty());
        assert!(!extractor_for("rule_based")
            .extract("The db is Postgres.")
            .is_empty());
    }
}
